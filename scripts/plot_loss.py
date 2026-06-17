"""Draw training loss curve from CSV log."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

steps_t, train_loss = [], []
steps_v, val_loss = [], []

with open("checkpoints/loss_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = int(row["step"])
        if row["train_loss"]:
            steps_t.append(s)
            train_loss.append(float(row["train_loss"]))
        if row["val_loss"]:
            steps_v.append(s)
            val_loss.append(float(row["val_loss"]))

# Downsample train for interpolation
step = max(1, len(steps_t) // 300)
st_ds = steps_t[::step]
tr_ds = train_loss[::step]

fine = np.linspace(steps_t[0], steps_t[-1], 500)
train_spline = make_interp_spline(st_ds, tr_ds, k=3)
train_smooth = np.maximum(train_spline(fine), 0)
val_spline = make_interp_spline(steps_v, val_loss, k=3)
val_smooth = np.maximum(val_spline(fine), 0)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(fine, train_smooth, 'b-', linewidth=2, label='Train Loss')
plt.plot(fine, val_smooth, 'r-', linewidth=2, label='Val Loss')
plt.scatter(steps_v, val_loss, color='red', s=20, alpha=0.4, zorder=3)
best_idx = np.argmin(val_loss)
plt.scatter(steps_v[best_idx], val_loss[best_idx], color='green', s=120, zorder=5, edgecolors='white')
plt.annotate(f'Best Val {val_loss[best_idx]:.4f}', (steps_v[best_idx], val_loss[best_idx]),
             xytext=(steps_v[best_idx]-1800, val_loss[best_idx]+0.15),
             arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('Iteration 4: 15x Data Scale (129K samples, batch=24)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Sliding window average (500-step window) to show trend
window = 500 // max(1, np.mean(np.diff(steps_t)))  # ~50 points per 500 steps
window = max(5, int(window))
train_rolling = np.convolve(train_loss, np.ones(window)/window, mode='valid')
roll_steps = steps_t[window-1:]
plt.plot(roll_steps, train_rolling, 'b-', linewidth=2, label='Train (rolling avg)')
plt.scatter(steps_v, val_loss, color='red', s=30, alpha=0.6, zorder=3, label='Val')
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('Trend (500-step Rolling Avg + Val)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=200)
print("Saved: training_curve.png")
