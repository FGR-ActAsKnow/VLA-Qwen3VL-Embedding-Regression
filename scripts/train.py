import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
from src.data.dataset import RobotActionDataset
from src.data.processing import ActionNormalizer
from src.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata JSON file")
    parser.add_argument("--normalizer", type=str, default=None, help="Path to saved normalizer JSON")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Auto-detect local model directory
    local_model = Path("model-embedding")
    if local_model.exists() and local_model.is_dir():
        config["model"]["name"] = str(local_model.resolve())
        print(f"Using local model: {config['model']['name']}")

    model = Qwen3VLRobotPolicy(config)
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    normalizer = ActionNormalizer()
    if args.normalizer:
        normalizer.load(args.normalizer)
        print(f"Loaded normalizer from {args.normalizer}")

    data_cfg = config["data"]
    train_dataset = RobotActionDataset(
        metadata_path=args.metadata,
        processor=model.processor,
        normalizer=normalizer,
        history_frames=data_cfg["history_frames"],
        action_horizon=data_cfg["action_horizon"],
        image_size=tuple(data_cfg["image_size"]),
        split="train",
        train_ratio=data_cfg["train_split"],
    )

    normalizer.save(str(Path(config["training"]["output_dir"]) / "normalizer.json"))
    print(f"Normalizer saved. Mean: {normalizer.mean.flatten()}, Std: {normalizer.std.flatten()}")

    val_dataset = RobotActionDataset(
        metadata_path=args.metadata,
        processor=model.processor,
        normalizer=normalizer,
        history_frames=data_cfg["history_frames"],
        action_horizon=data_cfg["action_horizon"],
        image_size=tuple(data_cfg["image_size"]),
        split="val",
        train_ratio=data_cfg["train_split"],
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    trainer = Trainer(model, config)
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
