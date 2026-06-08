import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import Qwen3VLProcessor

from .processing import ActionNormalizer


class RobotActionDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        processor: Qwen3VLProcessor,
        normalizer: ActionNormalizer | None = None,
        history_frames: int = 4,
        image_size: tuple[int, int] = (224, 224),
        instruction: str = "Predict the next robot action as 7 continuous values: dx, dy, dz, droll, dpitch, dyaw, gripper.",
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        max_samples: int | None = None,
    ):
        self.processor = processor
        self.normalizer = normalizer
        self.history_frames = history_frames
        self.image_size = image_size
        self.instruction = instruction

        with open(metadata_path) as f:
            samples = json.load(f)

        samples.sort(key=lambda x: (x["episode_id"], x["frame_idx"]))

        episodes = {}
        for s in samples:
            ep = s["episode_id"]
            episodes.setdefault(ep, []).append(s)

        self.windows = []
        for ep_id, frames in episodes.items():
            n = len(frames)
            if n >= history_frames + 1:
                for i in range(history_frames, n):
                    window = frames[i - history_frames : i + 1]
                    self.windows.append(window)

        random.seed(seed)
        random.shuffle(self.windows)

        split_idx = int(len(self.windows) * train_ratio)
        if split == "train":
            self.windows = self.windows[:split_idx]
        else:
            self.windows = self.windows[split_idx:]

        if max_samples is not None and max_samples < len(self.windows):
            self.windows = self.windows[:max_samples]

        if self.normalizer is not None and self.normalizer.mean is None:
            all_actions = np.array([w[-1]["action"] for w in self.windows])
            self.normalizer.fit(all_actions)

    def __len__(self):
        return len(self.windows)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize(self.image_size, Image.BICUBIC)

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        images = [self._load_image(f["image_path"]) for f in window]
        action_raw = np.array(window[-1]["action"], dtype=np.float32)

        if self.normalizer is not None and self.normalizer.mean is not None:
            action = self.normalizer.normalize(action_raw)
        else:
            action = action_raw

        return {
            "images": images,
            "action_raw": torch.tensor(action_raw, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "episode_id": window[-1]["episode_id"],
            "frame_idx": window[-1]["frame_idx"],
        }


def collate_fn(batch: list[dict], processor: Qwen3VLProcessor, instruction: str) -> dict:
    all_input_ids = []
    all_attention_mask = []
    all_mm_token_type_ids = []
    all_pixel_values = []
    all_image_grid_thw = []

    for sample in batch:
        images = sample["images"]

        messages = [{
            "role": "user",
            "content": [
                *(dict(type="image", image=img) for img in images),
                dict(type="text", text=instruction),
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=images, return_tensors="pt")

        all_input_ids.append(inputs["input_ids"][0])
        all_attention_mask.append(inputs["attention_mask"][0])
        all_mm_token_type_ids.append(inputs["mm_token_type_ids"][0])
        all_pixel_values.append(inputs["pixel_values"])
        all_image_grid_thw.append(inputs["image_grid_thw"])

    max_len = max(ids.shape[0] for ids in all_input_ids)
    pad_token_id = processor.tokenizer.pad_token_id

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    mm_token_type_ids = torch.full((len(batch), max_len), -1, dtype=torch.long)

    for i, (ids, am, mm) in enumerate(zip(all_input_ids, all_attention_mask, all_mm_token_type_ids)):
        input_ids[i, :ids.shape[0]] = ids
        attention_mask[i, :am.shape[0]] = am
        mm_token_type_ids[i, :mm.shape[0]] = mm

    pixel_values = torch.cat(all_pixel_values, dim=0)
    image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
    actions = torch.stack([s["action"] for s in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "action_labels": actions,
    }
