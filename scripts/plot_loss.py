"""Draw training loss curve from CSV log."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Read CSV
steps_t, train_loss = [], []
steps_v, val_loss = [], []
epoch_avg = {}

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

# Downsample train points for smoother interpolation (keep every 10th)
step = max(1, len(steps_t) // 100)
steps_t_ds = steps_t[::step]
train_ds = train_loss[::step]

# Spline interpolation
fine = np.linspace(steps_t[0], steps_t[-1], 500)
train_spline = make_interp_spline(steps_t_ds, train_ds, k=3)
train_smooth = np.maximum(train_spline(fine), 0)

val_spline = make_interp_spline(steps_v, val_loss, k=3)
val_smooth = np.maximum(val_spline(fine), 0)

plt.figure(figsize=(14, 5))

# Plot 1: Step-wise
plt.subplot(1, 2, 1)
plt.plot(fine, train_smooth, 'b-', linewidth=2, label='Train Loss')
plt.plot(fine, val_smooth, 'r-', linewidth=2, label='Val Loss')
plt.scatter(steps_v, val_loss, color='red', s=20, alpha=0.4, zorder=3)
best_idx = np.argmin(val_loss)
plt.scatter(steps_v[best_idx], val_loss[best_idx], color='green', s=120, zorder=5, edgecolors='white')
plt.annotate(f'Best Val {val_loss[best_idx]:.4f}', (steps_v[best_idx], val_loss[best_idx]),
             xytext=(steps_v[best_idx] - 1500, 0.07), arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss (Action Chunking)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 0.26)

# Plot 2: Epoch average
plt.subplot(1, 2, 2)
step_per_epoch = (steps_t[-1] - steps_t[0]) // 10
for ep in range(10):
    start_s = steps_t[0] + ep * step_per_epoch
    end_s = start_s + step_per_epoch
    ep_losses = [t for s, t in zip(steps_t, train_loss) if start_s <= s < end_s]
    if ep_losses:
        epoch_avg[ep + 1] = np.mean(ep_losses)

epochs = list(epoch_avg.keys())
losses = list(epoch_avg.values())
plt.plot(epochs, losses, 'b-o', linewidth=2.5, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Avg MSE Loss')
plt.title('Epoch Average Loss')
plt.grid(True, alpha=0.3)
for e, l in zip(epochs, losses):
    plt.annotate(f'{l:.4f}', (e, l), textcoords="offset points", xytext=(0, -18), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('training_curve.png', dpi=200)
print("Saved: training_curve.png")
