import csv
from pathlib import Path

base = Path(__file__).parent.parent
outdir = base / "results/benchmarks"

content = (base / "results/sgns_10.out").read_text()


def parse_losses(text):
    losses = []
    for line in text.split("\n"):
        if line.startswith("Epoch ") and "Loss:" in line:
            epoch = int(line.split("/")[0].split()[-1])
            loss = float(line.split("Loss:")[1].strip())
            losses.append((epoch, loss))
    return losses


def parse_metrics(text):
    metrics = {}
    for line in text.split("\n"):
        if "WordSim-353:" in line:
            metrics["wordsim353"] = float(line.split("rho=")[1].split()[0])
        elif "SimLex-999:" in line:
            metrics["simlex999"] = float(line.split("rho=")[1].split()[0])
        elif line.startswith("Analogy:"):
            metrics["analogy_pct"] = float(line.split("%")[0].split()[-1])
        elif "Train time:" in line:
            metrics["train_time"] = float(line.split(":")[1].replace("s", "").strip())
    return metrics


def parse_categories(text):
    cats = []
    in_section = False
    for line in text.split("\n"):
        if "Analogy by category:" in line:
            in_section = True
            continue
        if in_section and line.strip() == "":
            break
        if in_section and ":" in line and "%" in line:
            name = line.split(":")[0].strip()
            pct = float(line.split("%")[0].split()[-1])
            nums = line.split("(")[1].split(")")[0]
            correct, total = nums.split("/")
            cats.append({"category": name, "accuracy_pct": pct, "correct": int(correct), "total": int(total)})
    return cats


losses = parse_losses(content)
metrics = parse_metrics(content)
categories = parse_categories(content)

with open(outdir / "sgns_10_loss.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss"])
    w.writerows(losses)

with open(outdir / "sgns_epochs_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epochs", "wordsim353", "simlex999", "analogy_pct", "train_time"])
    w.writerow([10, metrics["wordsim353"], metrics["simlex999"], metrics["analogy_pct"], metrics["train_time"]])
    w.writerow([100, 0.690, 0.223, 3.3, 13888.9])

with open(outdir / "sgns_10_categories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "accuracy_pct", "correct", "total"])
    for c in categories:
        w.writerow([c["category"], c["accuracy_pct"], c["correct"], c["total"]])
