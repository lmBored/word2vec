import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent.parent
datadir = base / "results/benchmarks"
outdir = base / "results/plots"

metrics = {}
with open(datadir / "sgns_epochs_metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs = int(row["epochs"])
        metrics[epochs] = {
            "wordsim353": float(row["wordsim353"]),
            "simlex999": float(row["simlex999"]),
            "analogy_pct": float(row["analogy_pct"]),
            "train_time": float(row["train_time"]),
        }

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

labels = ["10 epochs", "100 epochs"]
x = np.arange(len(labels))
width = 0.6

# WordSim-353 and SimLex-999
ax = axes[0]
ws_vals = [metrics[10]["wordsim353"], metrics[100]["wordsim353"]]
sl_vals = [metrics[10]["simlex999"], metrics[100]["simlex999"]]
ax.bar(x - 0.2, ws_vals, 0.35, label="WordSim-353", color="steelblue")
ax.bar(x + 0.2, sl_vals, 0.35, label="SimLex-999", color="coral")
ax.set_ylabel("Spearman rho")
ax.set_title("Similarity Benchmarks")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 1)
for i, (w, s) in enumerate(zip(ws_vals, sl_vals)):
    ax.text(i - 0.2, w + 0.02, f"{w:.3f}", ha="center", fontsize=9)
    ax.text(i + 0.2, s + 0.02, f"{s:.3f}", ha="center", fontsize=9)

# Analogy
ax = axes[1]
analogy_vals = [metrics[10]["analogy_pct"], metrics[100]["analogy_pct"]]
bars = ax.bar(x, analogy_vals, width, color=["#4a90d9", "#1a5fb4"])
ax.set_ylabel("Accuracy (%)")
ax.set_title("Analogy Task")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(analogy_vals) * 1.15)
for bar, v in zip(bars, analogy_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}%", ha="center", fontsize=10)

# Training time
ax = axes[2]
time_vals = [metrics[10]["train_time"] / 3600, metrics[100]["train_time"] / 3600]
bars = ax.bar(x, time_vals, width, color=["#4a90d9", "#1a5fb4"])
ax.set_ylabel("Time (hours)")
ax.set_title("Training Time")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(time_vals) * 1.15)
for bar, v in zip(bars, time_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}h", ha="center", fontsize=10)

plt.suptitle("SGNS: 10 vs 100 Epochs", fontsize=14, fontweight="bold")
plt.tight_layout()

plt.savefig(outdir / "sgns_epochs_comparison.png", dpi=150)
