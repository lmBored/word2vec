import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent.parent
datadir = base / "results/benchmarks"
outdir = base / "results/plots"
outdir.mkdir(exist_ok=True)

sgns_cats = {}
cbow_cats = {}
with open(datadir / "analogy_categories.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cat = row["category"]
        acc = float(row["accuracy_pct"])
        if row["model"] == "sgns":
            sgns_cats[cat] = acc
        else:
            cbow_cats[cat] = acc

# Sort by SGNS accuracy
categories = sorted(sgns_cats.keys(), key=lambda c: sgns_cats[c], reverse=True)

sgns_vals = [sgns_cats[c] for c in categories]
cbow_vals = [cbow_cats.get(c, 0) for c in categories]


# Clean up category names for display
def clean_name(name):
    if name.startswith("gram"):
        return name.split("-", 1)[1].replace("-", " ").title()
    return name.replace("-", " ").title()


display_names = [clean_name(c) for c in categories]

fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(categories))
height = 0.35

bars1 = ax.barh(y - height / 2, sgns_vals, height, label="SGNS", color="steelblue")
bars2 = ax.barh(y + height / 2, cbow_vals, height, label="CBOW", color="coral")

for i, c in enumerate(categories):
    color = (
        "#e8f4e8"
        if c in {"capital-common-countries", "capital-world", "currency", "city-in-state", "family"}
        else "#e8f0f4"
    )
    ax.axhspan(i - 0.5, i + 0.5, color=color, alpha=0.5, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels(display_names)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Analogy Accuracy by Category")
ax.legend(loc="lower right")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(outdir / "analogy_breakdown.png", dpi=150)
