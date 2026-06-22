#!/usr/bin/env python3
"""用 stable-baselines3 PPO 训练 Franka 操作任务。

BC→RL 管道：
  1. python scripts/collect_vla_demos.py --episodes 300 --output data/vla_demos.npz
  2. python scripts/pretrain_bc.py --data data/vla_demos.npz --output rl_logs/bc_pretrained.zip
  3. python scripts/train_ppo.py --task reach --bc-pretrain rl_logs/bc_pretrained.zip --timesteps 500000

从零训练：
  python scripts/train_ppo.py --task reach --timesteps 500000
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["MUJOCO_GL"] = "egl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="reach")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="./rl_logs")
    parser.add_argument("--bc-pretrain", type=str, default=None,
                        help="BC 蒸馏预训练的模型路径 (rl_logs/bc_pretrained.zip)")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.rl.envs.gym_wrapper import FrankaGymEnv

    def make_env():
        return FrankaGymEnv(task=args.task)

    vec_env = DummyVecEnv([make_env for _ in range(4)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = vec_env.obs_rms

    run_name = f"ppo_{args.task}"
    log_dir = str(Path(args.logdir) / run_name)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    # BC 微调时用低学习率保护 Actor 已有知识
    if args.bc_pretrain:
        print(f"Loading BC-pretrained model from {args.bc_pretrain}")
        model = PPO.load(args.bc_pretrain, env=vec_env, device="cuda")
        model.learning_rate = lambda _: 1e-5  # 低 lr 微调
        model.batch_size = 64
        model.n_epochs = 1  # 每轮只做 1 epoch，保守更新
        model.clip_range = lambda _: 0.1  # 更小clip
        model.ent_coef = 0.0
        model.verbose = 1
        model.tensorboard_log = log_dir
        model.seed = args.seed
        # 同步 eval env 的 obs_rms
        if hasattr(vec_env, "obs_rms"):
            eval_env.obs_rms = vec_env.obs_rms
        print("  BC→RL fine-tuning mode: lr=1e-5, epochs=1, clip=0.1")
    else:
        model = PPO(
            "MlpPolicy", vec_env, verbose=1, seed=args.seed,
            tensorboard_log=log_dir,
            n_steps=2048, batch_size=64, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="cuda",
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(str(Path(log_dir) / "final_model"))
    vec_env.save(str(Path(log_dir) / "vec_normalize.pkl"))
    print(f"Saved to {log_dir}")


if __name__ == "__main__":
    main()
