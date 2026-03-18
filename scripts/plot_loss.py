import csv
from pathlib import Path

import matplotlib.pyplot as plt

base = Path(__file__).parent.parent
datadir = base / "results/benchmarks"
outdir = base / "results/plots"
outdir.mkdir(exist_ok=True)

sgns_epochs, sgns_losses = [], []
with open(datadir / "sgns_loss.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sgns_epochs.append(int(row["epoch"]))
        sgns_losses.append(float(row["loss"]))

cbow_epochs, cbow_losses = [], []
with open(datadir / "cbow_loss.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cbow_epochs.append(int(row["epoch"]))
        cbow_losses.append(float(row["loss"]))

plt.figure(figsize=(10, 6))
plt.plot(sgns_epochs, sgns_losses, "b-", linewidth=2, label=f"SGNS (final: {sgns_losses[-1]:.2f})")
plt.plot(cbow_epochs, cbow_losses, "r-", linewidth=2, label=f"CBOW (final: {cbow_losses[-1]:.2f})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(outdir / "loss_curves.png", dpi=150)
