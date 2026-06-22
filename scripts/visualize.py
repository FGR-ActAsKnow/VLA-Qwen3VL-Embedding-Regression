#!/usr/bin/env python3
"""Visualize VLA model predictions vs ground truth on a trajectory episode.

Usage:
  python scripts/visualize.py --checkpoint checkpoints/checkpoint-best \
      --episode robomind_franka --frame 0

Output: viz/ directory with comparison plots, frames, and video.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
from src.data.dataset import RobotActionDataset, collate_fn
from src.data.processing import ActionNormalizer


ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
VIZ_DIR = Path("viz")


def load_episode_metadata(metadata_path: str, episode_id: str) -> list[dict]:
    """Extract a single episode from metadata."""
    with open(metadata_path) as f:
        all_samples = json.load(f)
    ep_samples = [s for s in all_samples if s["episode_id"] == episode_id]
    ep_samples.sort(key=lambda x: x["frame_idx"])
    return ep_samples


def draw_action_overlay(image: Image.Image, action: np.ndarray, label: str,
                        color: tuple = (0, 255, 0)) -> Image.Image:
    """Draw action vector as arrows on the image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = img.height, img.width
    cx, cy = w // 2, h // 2

    # Scale arrows assuming normalized actions
    dx, dy = action[0] * 50, -action[1] * 50  # flip y for image coords
    draw.line([(cx, cy), (cx + int(dx), cy + int(dy))], fill=color, width=3)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill="red")

    gripper = "OPEN" if action[6] > 0.5 else "CLOSE"
    draw.text((10, 10), f"{label} | Grip: {gripper}", fill="white")
    return img


def plot_action_comparison(all_preds: np.ndarray, all_gt: np.ndarray,
                           save_path: Path):
    """Plot all 7 action dimensions: pred vs ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = all_preds.shape[0]
    fig, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
    x = np.arange(n) / 5.0  # time in seconds (5 fps)

    for i in range(7):
        ax = axes[i]
        ax.plot(x, all_gt[:, i], 'k-', linewidth=2, label='Ground Truth')
        ax.plot(x, all_preds[:, i], 'r--', linewidth=1.5, label='Prediction',
                alpha=0.8)
        ax.set_ylabel(ACTION_LABELS[i])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Action Prediction vs Ground Truth", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--normalizer", type=str, default="checkpoints/normalizer.json")
    parser.add_argument("--episode", type=str, default=None,
                        help="Episode ID (default: first episode in metadata)")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--make-video", action="store_true",
                        help="Create MP4 video with ffmpeg")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if Path("model-embedding").exists():
        config["model"]["name"] = str(Path("model-embedding").resolve())

    print("Loading model...")
    model = Qwen3VLRobotPolicy.from_pretrained(args.checkpoint, config)
    model.action_head = model.action_head.to("cuda")
    model.eval()

    normalizer = ActionNormalizer()
    normalizer.load(args.normalizer)

    # Load episode data
    episodes = json.load(open(args.metadata))
    ep_groups = {}
    for s in episodes:
        ep_groups.setdefault(s["episode_id"], []).append(s)

    ep_id = args.episode or list(ep_groups.keys())[0]
    ep_samples = sorted(ep_groups[ep_id], key=lambda x: x["frame_idx"])
    print(f"Episode: {ep_id} ({len(ep_samples)} frames)")

    # Create output dirs
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = VIZ_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    all_preds, all_gt, all_actions_norm = [], [], []
    history = []

    print(f"Running inference on {min(args.max_frames, len(ep_samples))} frames...")
    for i in range(min(args.max_frames, len(ep_samples))):
        img_path = ep_samples[i]["image_path"]
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        # Read ground truth (denormalized)
        action_raw = np.array(ep_samples[i]["action"], dtype=np.float32)
        all_gt.append(action_raw)

        # Normalize for model
        action_norm = normalizer.normalize(action_raw)
        all_actions_norm.append(action_norm)

        # VLA inference
        history.append(img)
        while len(history) < 4:
            history.append(img)
        if len(history) > 4:
            history.pop(0)

        sample = {"images": list(history), "prev_text": "", "pos_labels": torch.zeros(60), "grip_labels": torch.zeros(10)}
        batch = collate_fn([sample], model.processor, "Execute.")
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch.pop("pos_labels", None)
        batch.pop("grip_labels", None)

        with torch.no_grad():
            outputs = model(**batch)

        # Take first step of chunk (pos + grip from separate heads)
        pos = outputs["pos_pred"][0, 0].cpu().numpy()     # (6,)
        grip = outputs["grip_logits"][0, 0].cpu().numpy()  # (1,)
        grip_val = 1.0 if torch.sigmoid(torch.tensor(float(grip))) > 0.5 else 0.0
        pred_combined = np.concatenate([pos, [grip_val]])
        all_preds.append(normalizer.denormalize(pred_combined))

        # Save overlay frame
        pred_img = draw_action_overlay(img, normalizer.denormalize(pred),
                                       f"Pred Frame {i}", color=(0, 200, 0))
        gt_img = draw_action_overlay(img, normalizer.denormalize(action_norm),
                                     f"GT Frame {i}", color=(200, 0, 0))
        combined = Image.new("RGB", (448, 224))
        combined.paste(pred_img, (0, 0))
        combined.paste(gt_img, (224, 0))
        combined.save(frames_dir / f"frame_{i:06d}.jpg")

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(args.max_frames, len(ep_samples))}")

    all_preds = np.array(all_preds)
    all_gt = np.array(all_gt)

    # Save comparison plot
    plot_action_comparison(all_preds, all_gt, VIZ_DIR / "comparison.png")

    # Save predicted vs ground truth trajectories
    print("\n=== Prediction vs Ground Truth ===")
    errors = np.sqrt(np.sum((all_preds - all_gt) ** 2, axis=-1))
    print(f"  Mean EU error: {np.mean(errors):.4f}")
    print(f"  Median EU error: {np.median(errors):.4f}")
    for i, label in enumerate(ACTION_LABELS):
        mae = np.mean(np.abs(all_preds[:, i] - all_gt[:, i]))
        print(f"  MAE {label}: {mae:.4f}")

    # Create video if requested
    if args.make_video:
        print("  Creating GIF animation (no ffmpeg needed)...")
        frame_files = sorted(frames_dir.glob("*.jpg"))
        images = []
        for i in range(0, len(frame_files), 2):
            img = Image.open(frame_files[i])
            images.append(img.resize((448, 224)))
        gif_path = VIZ_DIR / "trajectory.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=200, loop=0, optimize=False)
        print(f"  GIF: {gif_path}")

    print(f"\n=== Complete ===")
    print(f"  Comparison plot: {VIZ_DIR/'comparison.png'}")
    print(f"  Frames: {frames_dir}/")
    print(f"  Total frames: {len(all_preds)}")
    print(f"\n  To view: scp -r ld@server:~/VLM_Robotics/viz ./")


if __name__ == "__main__":
    main()
