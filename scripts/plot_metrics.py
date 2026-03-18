import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent.parent
datadir = base / "results/benchmarks"
outdir = base / "results/plots"
outdir.mkdir(exist_ok=True)

metrics = {}
with open(datadir / "metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metrics[row["model"]] = {
            "wordsim353": float(row["wordsim353"]),
            "simlex999": float(row["simlex999"]),
            "analogy_pct": float(row["analogy_pct"]),
        }

labels = ["WordSim-353", "SimLex-999", "Analogy (%)"]
sgns_vals = [metrics["sgns"]["wordsim353"], metrics["sgns"]["simlex999"], metrics["sgns"]["analogy_pct"]]
cbow_vals = [metrics["cbow"]["wordsim353"], metrics["cbow"]["simlex999"], metrics["cbow"]["analogy_pct"]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, sgns_vals, width, label="SGNS", color="steelblue")
bars2 = ax.bar(x + width / 2, cbow_vals, width, label="CBOW", color="coral")

ax.set_ylabel("Score")
ax.set_title("Evaluation Benchmarks")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(outdir / "metrics_comparison.png", dpi=150)
