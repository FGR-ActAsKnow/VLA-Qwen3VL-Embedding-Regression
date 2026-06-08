"""Convert RoboMIND 2.0 HDF5 to training metadata.

Usage:
  # Single file
  python scripts/convert_robomind.py --hdf5 data/raw/robomind/data/franka/.../trajectory.hdf5

  # Batch (recursively finds all trajectory.hdf5)
  python scripts/convert_robomind.py --data-dir data/raw/robomind --fps 5
"""
import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def decode_jpeg(data: bytes) -> Image.Image | None:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def process_one(hdf5_path: Path, camera: str, fps: int, output_dir: Path, max_frames: int = None) -> list[dict]:
    """Process one trajectory.hdf5 file, return samples."""
    ep_id = f"{hdf5_path.parent.parent.parent.name}_{hdf5_path.parent.parent.name}"
    img_key = f"camera_observations/color_images/{camera}"
    act_key = "puppet/end_effector_right_pose_align/data"

    try:
        h5 = h5py.File(hdf5_path, "r")
    except Exception as e:
        print(f"    [FAIL] can't open: {e}")
        return []

    if img_key not in h5:
        print(f"    [FAIL] no camera data")
        h5.close()
        return []

    n_frames = len(h5[img_key])
    actions = h5[act_key][:] if act_key in h5 else None
    step = max(1, n_frames // 30 // fps)

    ep_out_dir = output_dir / hdf5_path.parent.parent.parent.name / hdf5_path.parent.parent.name
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(0, n_frames, step):
        if max_frames and len(samples) >= max_frames:
            break

        jpeg = h5[img_key][i]
        if isinstance(jpeg, bytes):
            img = decode_jpeg(jpeg)
        elif isinstance(jpeg, np.ndarray) and jpeg.dtype == np.uint8:
            img = decode_jpeg(jpeg.tobytes())
        else:
            continue
        if img is None:
            continue

        frame_path = ep_out_dir / f"frame_{i:06d}.jpg"
        img.save(frame_path, quality=90)

        action = [0.0] * 7
        if actions is not None and i < len(actions):
            action = actions[i].tolist()[:7]

        samples.append({
            "episode_id": ep_id,
            "frame_idx": len(samples),
            "image_path": str(frame_path.resolve()),
            "action": action,
        })

    h5.close()
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=None, help="Single trajectory.hdf5")
    parser.add_argument("--data-dir", type=str, default=None, help="Dir scanned recursively for trajectory.hdf5")
    parser.add_argument("--output", type=str, default="data/processed/metadata.json")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frames-dir", type=str, default="data/raw/frames")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--camera", type=str, default="camera_front",
                        choices=["camera_front", "camera_left", "camera_right",
                                 "camera_top", "camera_wrist_left", "camera_wrist_right"])
    args = parser.parse_args()

    if args.data_dir:
        root = Path(args.data_dir)
        hdf5_files = sorted(root.rglob("trajectory.hdf5"))[:args.max_files]
    elif args.hdf5:
        hdf5_files = [Path(args.hdf5)]
    else:
        print("Specify --hdf5 or --data-dir")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} HDF5 files")
    frames_dir = Path(args.frames_dir) / "robomind"
    all_samples = []

    for path in hdf5_files:
        ep_dir = path.parent.parent.parent.name
        print(f"  [{ep_dir}] {path.name} ({path.stat().st_size / 1e6:.0f} MB)...")
        samples = process_one(path, args.camera, args.fps, frames_dir, args.max_frames)
        all_samples.extend(samples)
        print(f"    {len(samples)} samples")

    if not all_samples:
        print("No samples!")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    acts = np.array([s["action"] for s in all_samples])
    print(f"\n=== Done ===")
    print(f"Files: {len(hdf5_files)}, Samples: {len(all_samples)}")
    print("Action stats:")
    for i in range(7):
        print(f"  dim {i}: mean={acts[:,i].mean():.5f} std={acts[:,i].std():.5f}")
    print(f"\nTrain: python scripts/train.py --config configs/config.yaml --metadata {output_path}")


if __name__ == "__main__":
    main()
