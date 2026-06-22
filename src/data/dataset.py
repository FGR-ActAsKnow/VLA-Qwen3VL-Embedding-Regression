import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import Qwen3VLProcessor

from .processing import ActionNormalizer


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def format_prev_action(action: np.ndarray) -> str:
    parts = [f"{n}={v:.4f}" for n, v in zip(ACTION_NAMES, action)]
    return "Previous action: " + ", ".join(parts) + "."


class RobotActionDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        processor: Qwen3VLProcessor,
        normalizer: ActionNormalizer | None = None,
        history_frames: int = 4,
        action_horizon: int = 10,
        image_size: tuple[int, int] = (224, 224),
        instruction: str = "Based on the image sequence and previous action, predict the next robot actions.",
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        max_samples: int | None = None,
        gaussian_noise: float = 0.0,
    ):
        self.processor = processor
        self.normalizer = normalizer
        self.history_frames = history_frames
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.instruction = instruction
        self.gaussian_noise = gaussian_noise
        self.is_train = (split == "train")

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
            if n >= history_frames + action_horizon:
                for i in range(history_frames, n - action_horizon + 1):
                    window = frames[i - history_frames : i + action_horizon]
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
        images = [self._load_image(f["image_path"]) for f in window[:self.history_frames]]

        # Previous action (frame just before targets), denormalized
        prev_frame = window[self.history_frames - 1]
        prev_raw = np.array(prev_frame["action"], dtype=np.float32)
        if self.normalizer is not None and self.normalizer.mean is not None:
            prev_action_norm = self.normalizer.normalize(prev_raw)
        else:
            prev_action_norm = prev_raw

        # Target actions (action_horizon frames)
        pos_parts = []  # first 6 dims
        grip_parts = []  # 7th dim as binary
        for i in range(self.action_horizon):
            act_raw = np.array(window[self.history_frames + i]["action"], dtype=np.float32)
            if self.normalizer is not None and self.normalizer.mean is not None:
                act_norm = self.normalizer.normalize(act_raw)
            else:
                act_norm = act_raw.copy()

            # Gaussian noise on training data
            if self.is_train and self.gaussian_noise > 0:
                act_norm[:6] += np.random.normal(0, self.gaussian_noise, 6)

            pos_parts.append(torch.tensor(act_norm[:6], dtype=torch.float32))
            grip_parts.append(torch.tensor([1.0 if act_raw[6] > 0.5 else 0.0], dtype=torch.float32))

        pos_labels = torch.cat(pos_parts)   # (action_horizon * 6)
        grip_labels = torch.cat(grip_parts)  # (action_horizon * 1)

        # Format previous action as text for instruction
        prev_text = format_prev_action(prev_raw)

        return {
            "images": images,
            "pos_labels": pos_labels,
            "grip_labels": grip_labels,
            "prev_text": prev_text,
            "prev_action": torch.tensor(prev_action_norm, dtype=torch.float32),
            "episode_id": window[self.history_frames]["episode_id"],
            "frame_idx": window[self.history_frames]["frame_idx"],
        }


def collate_fn(batch: list[dict], processor: Qwen3VLProcessor, instruction: str) -> dict:
    all_input_ids = []
    all_attention_mask = []
    all_mm_token_type_ids = []
    all_pixel_values = []
    all_image_grid_thw = []

    for sample in batch:
        images = sample["images"]
        prev_text = sample.get("prev_text", "")
        full_inst = f"{prev_text}\n{instruction}"

        messages = [{
            "role": "user",
            "content": [
                *(dict(type="image", image=img) for img in images),
                dict(type="text", text=full_inst),
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
    pos_labels = torch.stack([s["pos_labels"] for s in batch])
    grip_labels = torch.stack([s["grip_labels"] for s in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "pos_labels": pos_labels,
        "grip_labels": grip_labels,
    }
