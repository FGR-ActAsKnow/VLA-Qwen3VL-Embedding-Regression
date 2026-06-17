import json
import math
import os
import random
import csv
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
    def __init__(self, model: nn.Module, config: dict, resume_from: str = None):
        self.model = model
        self.config = config
        self.resume_from = Path(resume_from) if resume_from else None
        train_cfg = config["training"]

        self.output_dir = Path(train_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = train_cfg["num_epochs"]
        self.grad_accum_steps = train_cfg["gradient_accumulation_steps"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.save_steps = train_cfg["save_steps"]
        self.eval_steps = train_cfg["eval_steps"]
        self.logging_steps = train_cfg["logging_steps"]
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.accelerator = Accelerator(
            mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
            gradient_accumulation_steps=self.grad_accum_steps,
        )

        # Loss log file (append when resuming)
        self.log_file = self.output_dir / "loss_log.csv"
        if self.resume_from and self.log_file.exists():
            print(f"Appending to existing loss_log.csv")
        else:
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "val_loss", "lr"])

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
        self.optimizer = AdamW(
            trainable_params,
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

        total_steps = len(train_loader) * self.num_epochs // self.grad_accum_steps
        warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        model, self.optimizer, train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, self.scheduler
        )
        self.model = model

        # Resume: restore optimizer, scheduler, step counters
        if self.resume_from is not None:
            opt_path = self.resume_from / "optimizer.pt"
            sched_path = self.resume_from / "scheduler.pt"
            accel_dir = self.resume_from / "accel_state"

            if opt_path.exists():
                opt_state = torch.load(opt_path, map_location="cpu")
                self.optimizer.load_state_dict(opt_state)
                print(f"Loaded optimizer state from {opt_path}")
            elif accel_dir.exists():
                # Old format: full accelerator state (first save before code update)
                self.accelerator.load_state(str(accel_dir))
                print(f"Loaded accelerator state from {accel_dir}")

            if sched_path.exists() and not train_cfg.get("restart_lr", False):
                self.scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
                print(f"Loaded scheduler state from {sched_path}")

            state_path = self.resume_from / "training_state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.best_val_loss = state.get("best_val_loss", float("inf"))
                print(f"Resumed: global_step={self.global_step}, best_val_loss={self.best_val_loss:.4f}")

        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)

        global_step = self.global_step
        best_val_loss = self.best_val_loss

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
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    epoch_loss += loss.item() if loss is not None else 0

                if self.accelerator.sync_gradients:
                    global_step += 1
                    self.global_step = global_step

                    if global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        current_lr = self.scheduler.get_last_lr()[0]
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
                        with open(self.log_file, "a", newline="") as f:
                            csv.writer(f).writerow([global_step, f"{avg_loss:.4f}", "", f"{current_lr:.2e}"])

                    if global_step % self.save_steps == 0:
                        self._save_checkpoint(global_step)

                    if val_loader is not None and global_step % self.eval_steps == 0:
                        val_loss = self.evaluate(val_loader)
                        self.accelerator.print(f"Step {global_step} | Val Loss: {val_loss:.4f}")
                        with open(self.log_file, "a", newline="") as f:
                            csv.writer(f).writerow([global_step, "", f"{val_loss:.4f}", ""])
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.best_val_loss = best_val_loss
                            self._save_checkpoint("best")

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.accelerator.print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        self._save_checkpoint("final")
        self.accelerator.print(f"Training done. Best val loss: {best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 100) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            outputs = self.model(**batch)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                count += 1
        self.model.train()
        return total_loss / count if count > 0 else float("inf")

    def _save_checkpoint(self, name: str | int):
        save_dir = self.output_dir / f"checkpoint-{name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(str(save_dir))

        # Save optimizer + scheduler states (much smaller than full model)
        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), save_dir / "scheduler.pt")

        # Save training metadata (step counter, best_val_loss)
        state = {"global_step": self.global_step, "best_val_loss": self.best_val_loss}
        with open(save_dir / "training_state.json", "w") as f:
            json.dump(state, f)
