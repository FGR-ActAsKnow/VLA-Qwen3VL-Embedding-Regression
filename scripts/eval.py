import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
from src.data.dataset import RobotActionDataset, collate_fn, ACTION_NAMES
from src.data.processing import ActionNormalizer
from torch.utils.data import DataLoader

DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]


def compute_metrics(pos_pred: np.ndarray, pos_gt: np.ndarray,
                    grip_pred: np.ndarray, grip_gt: np.ndarray,
                    normalizer: ActionNormalizer) -> dict:
    """Compute metrics with pos and grip split."""
    # Denormalize only position dims
    pred_full = np.concatenate([pos_pred, grip_pred[:, None]], axis=-1)
    gt_full = np.concatenate([pos_gt, grip_gt[:, None]], axis=-1)
    pred_denorm = normalizer.denormalize(pred_full)
    gt_denorm = normalizer.denormalize(gt_full)

    pos_denorm = pred_denorm[:, :6]
    pos_gt_denorm = gt_denorm[:, :6]

    euclidean_dist = np.sqrt(np.sum((pos_denorm - pos_gt_denorm) ** 2, axis=-1))
    dim_errors = np.abs(pos_denorm - pos_gt_denorm)

    metrics = {
        "mean_euclidean_error": float(np.mean(euclidean_dist)),
        "median_euclidean_error": float(np.median(euclidean_dist)),
    }
    for i, name in enumerate(DIM_NAMES):
        metrics[f"mae_{name}"] = float(np.mean(dim_errors[:, i]))

    # Gripper accuracy (binary classification)
    grip_pred_bin = (torch.sigmoid(torch.tensor(grip_pred)) > 0.5).numpy()
    grip_acc = float(np.mean(grip_pred_bin == grip_gt))
    metrics["gripper_accuracy"] = grip_acc

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--normalizer", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_model = Path("model-embedding")
    if local_model.exists() and local_model.is_dir():
        config["model"]["name"] = str(local_model.resolve())
        print(f"Using local model: {config['model']['name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLRobotPolicy.from_pretrained(args.checkpoint, config)
    model.action_head = model.action_head.to(device)
    model.eval()
    print("Model loaded from checkpoint.")

    normalizer = ActionNormalizer()
    normalizer.load(args.normalizer)

    data_cfg = config["data"]
    dataset = RobotActionDataset(
        metadata_path=args.metadata,
        processor=model.processor,
        normalizer=normalizer,
        history_frames=data_cfg["history_frames"],
        action_horizon=data_cfg["action_horizon"],
        image_size=tuple(data_cfg["image_size"]),
        split="val",
        train_ratio=data_cfg["train_split"],
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model.processor, dataset.instruction),
    )

    all_pos_pred = []
    all_pos_gt = []
    all_grip_pred = []
    all_grip_gt = []

    print(f"Evaluating {len(dataset)} samples...")
    for batch in tqdm(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pos_labels = batch.pop("pos_labels")
        grip_labels = batch.pop("grip_labels")

        with torch.no_grad():
            outputs = model(**batch)

        pos = outputs["pos_pred"][0, 0].cpu().numpy()    # (6,)
        grip = outputs["grip_logits"][0, 0].cpu().numpy()  # (1,)
        pgt = pos_labels[0, :6].cpu().numpy()             # (6,)
        ggt = grip_labels[0, 0].cpu().numpy()              # (1,)

        all_pos_pred.append(pos)
        all_pos_gt.append(pgt)
        all_grip_pred.append(grip.item())
        all_grip_gt.append(ggt.item())

    all_pos_pred = np.array(all_pos_pred)
    all_pos_gt = np.array(all_pos_gt)
    all_grip_pred = np.array(all_grip_pred)
    all_grip_gt = np.array(all_grip_gt)

    metrics = compute_metrics(all_pos_pred, all_pos_gt, all_grip_pred, all_grip_gt, normalizer)

    print("\n=== Evaluation Results ===")
    print(f"Samples: {len(all_pos_pred)}")
    print(f"Mean Euclidean Error: {metrics['mean_euclidean_error']:.4f}")
    print(f"Median Euclidean Error: {metrics['median_euclidean_error']:.4f}")
    print("Per-dimension MAE:")
    for dim in DIM_NAMES:
        key = f"mae_{dim}"
        if key in metrics:
            print(f"  {dim}: {metrics[key]:.4f}")
    if "gripper_accuracy" in metrics:
        print(f"Gripper Accuracy: {metrics['gripper_accuracy']:.2%}")


if __name__ == "__main__":
    main()
