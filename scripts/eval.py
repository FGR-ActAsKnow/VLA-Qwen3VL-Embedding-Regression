import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
from src.data.dataset import RobotActionDataset, collate_fn
from src.data.processing import ActionNormalizer
from torch.utils.data import DataLoader


def compute_metrics(pred: np.ndarray, gt: np.ndarray, normalizer: ActionNormalizer) -> dict:
    pred_denorm = normalizer.denormalize(pred)
    gt_denorm = normalizer.denormalize(gt)

    euclidean_dist = np.sqrt(np.sum((pred_denorm - gt_denorm) ** 2, axis=-1))
    dim_errors = np.abs(pred_denorm - gt_denorm)

    action_dim = pred.shape[-1]
    dim_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    metrics = {
        "mean_euclidean_error": float(np.mean(euclidean_dist)),
        "median_euclidean_error": float(np.median(euclidean_dist)),
    }
    for i in range(min(action_dim, len(dim_names))):
        metrics[f"mae_{dim_names[i]}"] = float(np.mean(dim_errors[:, i]))

    if action_dim >= 7:
        gripper_pred = (pred_denorm[:, 6] > 0.5).astype(np.float32)
        gripper_gt = (gt_denorm[:, 6] > 0.5).astype(np.float32)
        metrics["gripper_accuracy"] = float(np.mean(gripper_pred == gripper_gt))

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

    all_preds = []
    all_labels = []

    print(f"Evaluating {len(dataset)} samples...")
    for batch in tqdm(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop("action_labels")

        with torch.no_grad():
            outputs = model(**batch)

        pred = outputs["action_pred"].cpu().numpy()
        label = labels.cpu().numpy()
        all_preds.append(pred[0])
        all_labels.append(label[0])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_preds, all_labels, normalizer)

    print("\n=== Evaluation Results ===")
    print(f"Samples: {len(all_preds)}")
    print(f"Mean Euclidean Error: {metrics['mean_euclidean_error']:.4f}")
    print(f"Median Euclidean Error: {metrics['median_euclidean_error']:.4f}")
    print("Per-dimension MAE:")
    for dim in ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]:
        key = f"mae_{dim}"
        if key in metrics:
            print(f"  {dim}: {metrics[key]:.4f}")
    if "gripper_accuracy" in metrics:
        print(f"Gripper Accuracy: {metrics['gripper_accuracy']:.2%}")


if __name__ == "__main__":
    main()
