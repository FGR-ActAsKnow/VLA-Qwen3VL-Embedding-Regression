#!/usr/bin/env python3
"""评估训练好的 PPO 模型，输出指标和 JSON 结果。

用法:
    python scripts/eval_ppo.py --task reach --model rl_logs/ppo_reach/best_model.zip --episodes 20
    python scripts/eval_ppo.py --task reach --model rl_logs/ppo_reach/best_model.zip --output eval_reach.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["MUJOCO_GL"] = "egl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--vec-normalize", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--record", type=str, default=None, help="录制轨迹 GIF 保存路径")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.rl.envs.gym_wrapper import FrankaGymEnv

    recorder = None
    record_frames = []
    if args.record:
        try:
            import imageio
            recorder = []
        except ImportError:
            print("imageio not installed, skipping recording")
            args.record = None

    def make_env():
        return FrankaGymEnv(task=args.task)

    raw_env = DummyVecEnv([make_env])
    vec_norm_path = args.vec_normalize or str(Path(args.model).parent / "vec_normalize.pkl")
    if Path(vec_norm_path).exists():
        env = VecNormalize.load(vec_norm_path, raw_env)
        env.training = False
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO.load(args.model, env=env)

    episodes = []
    total_reward = 0.0
    successes = 0
    all_best_dists = []
    all_final_dists = []

    for ep in range(args.episodes):
        obs = env.reset()
        ep_reward = 0.0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            step += 1
            if done[0]:
                break

        total_reward += ep_reward
        ep_info = info[0] if isinstance(info, (list, tuple)) else info
        success = bool(ep_info.get("success", False))
        best_dist = float(ep_info.get("best_dist", 1.0))
        final_dist = float(ep_info.get("dist", 1.0))

        if success:
            successes += 1
        all_best_dists.append(best_dist)
        all_final_dists.append(final_dist)

        print(f"Ep {ep+1:3d}/{args.episodes} | Reward: {ep_reward:8.3f} | "
              f"BestD: {best_dist:.3f}m | FinalD: {final_dist:.3f}m | "
              f"Steps: {step} | Success: {success}")

        episodes.append({
            "episode": ep + 1,
            "reward": round(ep_reward, 3),
            "best_dist": round(best_dist, 4),
            "final_dist": round(final_dist, 4),
            "steps": step,
            "success": success,
        })

    mean_rew = total_reward / args.episodes
    succ_rate = successes / args.episodes
    mean_best = sum(all_best_dists) / len(all_best_dists)
    mean_final = sum(all_final_dists) / len(all_final_dists)
    near_10 = sum(1 for d in all_best_dists if d < 0.10) / len(all_best_dists)
    near_5 = sum(1 for d in all_best_dists if d < 0.05) / len(all_best_dists)

    print(f"\n{'='*60}")
    print(f"Summary ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean Reward:        {mean_rew:.3f}")
    print(f"  Success Rate:        {successes}/{args.episodes} ({succ_rate:.1%})")
    print(f"  Mean Best Dist:      {mean_best:.4f}m")
    print(f"  Mean Final Dist:     {mean_final:.4f}m")
    print(f"  Fraction < 0.10m:    {near_10:.1%}")
    print(f"  Fraction < 0.05m:    {near_5:.1%}")

    result = {
        "task": args.task,
        "model": args.model,
        "episodes": args.episodes,
        "mean_reward": round(mean_rew, 3),
        "success_rate": round(succ_rate, 4),
        "mean_best_dist": round(mean_best, 4),
        "mean_final_dist": round(mean_final, 4),
        "frac_near_10cm": round(near_10, 4),
        "frac_near_5cm": round(near_5, 4),
        "episodes": episodes,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.output}")

    if args.record:
        try:
            import mujoco, imageio
            raw = env.venv.envs[0]._env
            obs = env.reset()
            frames = []
            for _ in range(200):
                a, _ = model.predict(obs, deterministic=True)
                obs, _, d, _ = env.step(a)
                r = mujoco.Renderer(raw.model, 224, 224)
                r.update_scene(raw.data, camera="head")
                frames.append(r.render().copy())
                r.close()
                if d[0]:
                    break
            imageio.mimsave(args.record, frames, fps=10)
            print(f"Saved GIF: {args.record} ({len(frames)} frames)")
        except Exception as e:
            print(f"GIF recording failed: {e}")


if __name__ == "__main__":
    main()
