"""Draw training loss curve from logged data.
Usage: python scripts/plot_loss.py
"""
import matplotlib.pyplot as plt

# Manually entered from training output
steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
train_loss = [None, None, 0.2690, 0.1324, 0.1148, 0.0684, 0.0647, 0.0432, 0.0385]
val_loss = [None, 0.2100, 0.1600, 0.1021, 0.1001, 0.0855, 0.0588, 0.0450, 0.0487]
epoch_avg = {1: 0.2686, 2: 0.1130, 3: 0.0643, 4: 0.0380}

plt.figure(figsize=(12, 5))

# Plot 1: Step-wise loss
plt.subplot(1, 2, 1)
plt.plot([s for s, v in zip(steps, train_loss) if v],
         [v for v in train_loss if v], 'b-', label='Train Loss', linewidth=2)
plt.plot([s for s, v in zip(steps, val_loss) if v],
         [v for v in val_loss if v], 'r-', label='Val Loss', linewidth=2)
plt.scatter(3500, 0.0450, color='green', s=100, zorder=5)
plt.annotate('Best Val\n0.0450', (3500, 0.0450), xytext=(3200, 0.07),
             arrowprops=dict(arrowstyle='->'), fontsize=10)
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Epoch average
plt.subplot(1, 2, 2)
epochs = list(epoch_avg.keys())
losses = list(epoch_avg.values())
plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Avg MSE Loss')
plt.title('Epoch Average Loss')
plt.grid(True, alpha=0.3)
for e, l in zip(epochs, losses):
    plt.annotate(f'{l:.4f}', (e, l), textcoords="offset points", xytext=(0, -15), ha='center')

plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.savefig('training_curve.pdf')
print("Saved: training_curve.png, training_curve.pdf")
