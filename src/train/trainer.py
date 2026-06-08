import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from tqdm import tqdm

from src.data.dataset import collate_fn


class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        train_cfg = config["training"]

        self.output_dir = Path(train_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = train_cfg["num_epochs"]
        self.grad_accum_steps = train_cfg["gradient_accumulation_steps"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.save_steps = train_cfg["save_steps"]
        self.eval_steps = train_cfg["eval_steps"]
        self.logging_steps = train_cfg["logging_steps"]

        self.accelerator = Accelerator(
            mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
            gradient_accumulation_steps=self.grad_accum_steps,
        )

        self._seed_everything(train_cfg.get("seed", 42))

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self, train_dataset, val_dataset=None):
        train_cfg = self.config["training"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg["dataloader_num_workers"],
            collate_fn=lambda batch: collate_fn(batch, self.model.processor, train_dataset.instruction),
            pin_memory=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
                num_workers=train_cfg["dataloader_num_workers"],
                collate_fn=lambda batch: collate_fn(batch, self.model.processor, val_dataset.instruction),
                pin_memory=True,
            )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        total_steps = len(train_loader) * self.num_epochs // self.grad_accum_steps
        warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )
        self.model = model

        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for step, batch in enumerate(pbar):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                    if loss is not None:
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item() if loss is not None else 0

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

                    if global_step % self.save_steps == 0:
                        self._save_checkpoint(global_step)

                    if val_loader is not None and global_step % self.eval_steps == 0:
                        val_loss = self.evaluate(val_loader)
                        self.accelerator.print(f"Step {global_step} | Val Loss: {val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self._save_checkpoint("best")

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        self._save_checkpoint("final")
        self.accelerator.print(f"Training done. Best val loss: {best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in val_loader:
            outputs = self.model(**batch)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
        self.model.train()
        return total_loss / len(val_loader) if len(val_loader) > 0 else float("inf")

    def _save_checkpoint(self, name: str | int):
        save_dir = self.output_dir / f"checkpoint-{name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(str(save_dir))
