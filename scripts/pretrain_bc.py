#!/usr/bin/env python3
"""用 VLA 演示数据预训练 SB3 PPO 的策略网络 (Behavior Cloning)。

先在 VLA 的监督信号下训练 Actor，然后保存整个 PPO 模型，后续用 PPO 微调。

用法:
    python scripts/pretrain_bc.py --data data/vla_demos.npz --output rl_logs/bc_pretrained.zip
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/vla_demos.npz")
    parser.add_argument("--output", type=str, default="rl_logs/bc_pretrained.zip")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    # Load demo data
    data = np.load(args.data)
    states = data["states"]
    actions = data["actions"]
    obs_dim = states.shape[1]
    act_dim = actions.shape[1]
    print(f"Loaded {len(states)} samples, obs_dim={obs_dim}, act_dim={act_dim}")

    # Split
    n_val = int(len(states) * args.val_split)
    indices = np.random.permutation(len(states))
    train_s = torch.from_numpy(states[indices[n_val:]]).float()
    train_a = torch.from_numpy(actions[indices[n_val:]]).float()
    val_s = torch.from_numpy(states[indices[:n_val]]).float()
    val_a = torch.from_numpy(actions[indices[:n_val]]).float()

    train_loader = DataLoader(TensorDataset(train_s, train_a), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_s, val_a), batch_size=args.batch_size)

    # Build SB3 PPO with a dummy env, then grab its policy for BC training
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.rl.envs.gym_wrapper import FrankaGymEnv

    def make_env():
        return FrankaGymEnv(task="reach")

    dummy_env = DummyVecEnv([make_env])

    # Use same net_arch as training
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO("MlpPolicy", dummy_env, verbose=0, policy_kwargs=policy_kwargs, device="cuda")

    # Extract the policy and its optimizer
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # BC training: minimize NLL on demo actions
    best_val_loss = float("inf")
    best_state = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(args.epochs):
        policy.train()
        train_loss = 0.0
        for s, a in train_loader:
            s, a = s.to(device), a.to(device)
            dist = policy.get_distribution(s)
            loss = -dist.log_prob(a).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * s.size(0)
        train_loss /= len(train_s)

        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, a in val_loader:
                s, a = s.to(device), a.to(device)
                dist = policy.get_distribution(s)
                val_loss += (-dist.log_prob(a)).sum().item()
        val_loss /= len(val_s)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # Restore best and save
    if best_state is not None:
        policy.load_state_dict(best_state)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    dummy_env.close()
    print(f"\nSaved BC-pretrained model to {args.output} (val_loss={best_val_loss:.6f})")


if __name__ == "__main__":
    main()
