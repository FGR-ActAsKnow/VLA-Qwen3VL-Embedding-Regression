#!/usr/bin/env python3
"""VLA 蒸馏数据采集：在 MuJoCo 中运行 BC 模型，(状态, 关节增量) 对。

用法:
    python scripts/collect_vla_demos.py --episodes 300 --output data/vla_demos.npz
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import yaml
from PIL import Image


def _ik_solve(model, data, target_pos, max_iter=100, tol=1e-3):
    import mujoco
    body_id = model.body("end_effector").id
    site_id = model.site("grasp").id
    jacp = np.zeros((3, model.nv))
    lam = 0.5
    for _ in range(max_iter):
        mujoco.mj_fwdPosition(model, data)
        err = target_pos - data.site(site_id).xpos.copy()
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        J = jacp[:, :7].copy()
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
        data.qpos[:7] += dq
        lam *= 0.95
    mujoco.mj_fwdPosition(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint-best")
    parser.add_argument("--normalizer", type=str, default="checkpoints/normalizer.json")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/vla_demos.npz")
    args = parser.parse_args()

    from src.rl.envs.reach import ReachEnv
    from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
    from src.data.dataset import collate_fn
    from src.data.processing import ActionNormalizer

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if Path("model-embedding").exists():
        config["model"]["name"] = str(Path("model-embedding").resolve())

    print("Loading VLA model...")
    model = Qwen3VLRobotPolicy.from_pretrained(args.checkpoint, config)
    model.action_head = model.action_head.to("cuda")
    model.eval()

    normalizer = ActionNormalizer()
    normalizer.load(args.normalizer)

    env = ReachEnv({"control_freq": 25, "max_episode_steps": args.steps_per_episode})
    instruction = "Execute the robot task."

    all_states = []
    all_actions = []

    print(f"Collecting {args.episodes} episodes x {args.steps_per_episode} steps...")
    for ep in range(args.episodes):
        obs = env.reset()
        history = []
        for step in range(args.steps_per_episode):
            img = Image.fromarray(env.render(224, 224))
            history.append(img)
            while len(history) < 4:
                history.append(img)
            if len(history) > 4:
                history.pop(0)

            sample = {"images": list(history), "action": torch.zeros(70),
                      "pos_labels": torch.zeros(60), "grip_labels": torch.zeros(10)}
            batch = collate_fn([sample], model.processor, instruction)
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch.pop("pos_labels", None)
            batch.pop("grip_labels", None)

            with torch.no_grad():
                outputs = model(**batch)

            pos_pred = outputs["pos_pred"][0, 0].cpu().numpy()
            grip_prob = float(torch.sigmoid(outputs["grip_logits"][0, 0]).item())
            vla_action = np.zeros(7, dtype=np.float32)
            vla_action[:6] = pos_pred
            vla_action[6] = 1.0 if grip_prob > 0.5 else 0.0
            vla_action = normalizer.denormalize(vla_action)

            current_joints = env.data.qpos[:7].copy()
            target_pos = env.data.site("grasp").xpos.copy() + vla_action[:3] * 0.05
            _ik_solve(env.model, env.data, target_pos)
            target_joints = env.data.qpos[:7].copy()
            env.data.qpos[:7] = current_joints

            joint_delta = np.clip(target_joints - current_joints, -0.1, 0.1)
            all_states.append(obs)
            all_actions.append(joint_delta)

            obs, _, _, _ = env.step(joint_delta)

        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep+1}/{args.episodes} | Samples: {len(all_states)}")

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, states=states, actions=actions)
    print(f"Saved {len(states)} samples to {args.output}")
    print(f"States: {states.shape}, Actions: {actions.shape}")
    for i, name in enumerate(["j1","j2","j3","j4","j5","j6","j7"]):
        print(f"  {name}: mean={actions[:,i].mean():.4f} std={actions[:,i].std():.4f} min={actions[:,i].min():.4f} max={actions[:,i].max():.4f}")


if __name__ == "__main__":
    main()
