#!/usr/bin/env python3
"""生成 RL 完整评估报告图。在服务器上运行。"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent

with open(BASE / "rl_logs/eval_reach.json") as f:
    eval_data = json.load(f)
reach_npz = np.load(BASE / "rl_logs/ppo_reach/evaluations.npz")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("PPO on Franka Reach Task — Full Evaluation Report", fontsize=15, fontweight="bold")

# 1. Training Reward Curve
ax = fig.add_subplot(2, 3, 1)
ts = reach_npz["timesteps"]
results = reach_npz["results"]
mean_r = np.mean(results, axis=1)
std_r = np.std(results, axis=1)
ax.plot(ts, mean_r, "b-", linewidth=1.5, label="Eval Reward")
ax.fill_between(ts, mean_r - std_r, mean_r + std_r, alpha=0.15, color="blue")
ax.axhline(y=-200, color="gray", linestyle="--", alpha=0.5, label="Zero-action baseline")
ax.set_xlabel("Timesteps"); ax.set_ylabel("Episode Reward")
ax.set_title("Training Progress: Eval Reward")
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Reward Histogram
ax = fig.add_subplot(2, 3, 2)
rewards = np.array([ep["reward"] for ep in eval_data["episodes"]])
ax.hist(rewards, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(x=-200, color="gray", linestyle="--", alpha=0.7, label="Zero-action baseline")
ax.axvline(x=rewards.mean(), color="red", linestyle="-", alpha=0.7, label=f"Mean={rewards.mean():.0f}")
ax.set_xlabel("Episode Reward"); ax.set_ylabel("Count")
ax.set_title("Episode Reward Distribution (30 eval episodes)")
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Best Distance Distribution
ax = fig.add_subplot(2, 3, 3)
dists = np.array([ep["best_dist"] for ep in eval_data["episodes"]])
ax.hist(dists, bins=20, color="darkgreen", edgecolor="white", alpha=0.8)
ax.axvline(x=0.30, color="gray", linestyle="--", alpha=0.7, label="Zero-action (~0.30m)")
ax.axvline(x=0.03, color="red", linestyle="--", alpha=0.7, label="Success threshold (0.03m)")
ax.set_xlabel("Best Distance (m)"); ax.set_ylabel("Count")
ax.set_title("Best Episode Distance Distribution")
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Summary text
ax = fig.add_subplot(2, 3, 4)
ax.axis("off")
mbd = eval_data["mean_best_dist"]
mfd = eval_data["mean_final_dist"]
mr = eval_data["mean_reward"]
fn10 = eval_data["frac_near_10cm"]
fn5 = eval_data["frac_near_5cm"]
summary_lines = [
    "REACH TASK — EVALUATION RESULTS",
    "=" * 42,
    "",
    f"  Episodes evaluated:      30",
    f"  Success rate:             0/30 (0.0%)",
    f"  Closest approach:         0.031m (3.1cm)",
    "",
    f"  Mean Best Distance:       {mbd:.4f}m",
    f"  Mean Final Distance:      {mfd:.4f}m",
    f"  Mean Reward:              {mr:.1f}",
    "",
    f"  Fraction < 0.10m:         {fn10:.1%}",
    f"  Fraction < 0.05m:         {fn5:.1%}",
    "",
    "BASELINE COMPARISON",
    "-" * 30,
    "  Zero-action mean dist:    0.30m",
    f"  RL improvement:           {((0.30-mbd)/0.30*100):.0f}% (0.30->{mbd:.3f}m)",
    f"  Best single episode:      {((0.30-0.031)/0.30*100):.0f}% (0.30->0.031m)",
    "",
    "LIMITING FACTOR",
    "-" * 30,
    "  Position actuator steady-state error",
    "  prevents consistent <3cm convergence.",
    "  Policy reaches target area reliably",
    "  but overshoots due to arm dynamics.",
]
ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# 5. Reward vs Best Distance
ax = fig.add_subplot(2, 3, 5)
ax.scatter(dists, rewards, alpha=0.6, c="steelblue", edgecolors="white", s=60)
ax.axvline(x=0.03, color="red", linestyle="--", alpha=0.5, label="Success (0.03m)")
ax.set_xlabel("Best Distance (m)"); ax.set_ylabel("Episode Reward")
ax.set_title("Reward vs Best Distance")
ax.legend(); ax.grid(True, alpha=0.3)

# 6. Best vs Final Distance
ax = fig.add_subplot(2, 3, 6)
final_dists = np.array([ep["final_dist"] for ep in eval_data["episodes"]])
ax.scatter(dists, final_dists, alpha=0.6, c="darkgreen", edgecolors="white", s=60)
ax.plot([0, max(dists.max(), final_dists.max())], [0, max(dists.max(), final_dists.max())],
        "k--", alpha=0.3, label="Best = Final")
ax.axhline(y=0.03, color="red", linestyle="--", alpha=0.3)
ax.axvline(x=0.03, color="red", linestyle="--", alpha=0.3, label="Success zone")
ax.set_xlabel("Best Distance (m)"); ax.set_ylabel("Final Distance (m)")
ax.set_title("Best vs Final Distance (shows overshoot)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = BASE / "rl_eval_report.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
