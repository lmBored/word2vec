import csv
from pathlib import Path

base = Path(__file__).parent.parent
outfile = base / "results/sgns_vs_cbow.out"
content = outfile.read_text()

parts = content.split("Train command:")
sgns_text = parts[1]
cbow_text = parts[2]


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
            # gram8-plural: 20.9% (278/1332)
            name = line.split(":")[0].strip()
            pct = float(line.split("%")[0].split()[-1])
            nums = line.split("(")[1].split(")")[0]
            correct, total = nums.split("/")
            cats.append({"category": name, "accuracy_pct": pct, "correct": int(correct), "total": int(total)})
    return cats


sgns_losses = parse_losses(sgns_text)
cbow_losses = parse_losses(cbow_text)
sgns_metrics = parse_metrics(sgns_text)
cbow_metrics = parse_metrics(cbow_text)
sgns_cats = parse_categories(sgns_text)
cbow_cats = parse_categories(cbow_text)

outdir = base / "results/benchmarks"
outdir.mkdir(exist_ok=True)

with open(outdir / "sgns_loss.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss"])
    w.writerows(sgns_losses)

with open(outdir / "cbow_loss.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "loss"])
    w.writerows(cbow_losses)

with open(outdir / "metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "wordsim353", "simlex999", "analogy_pct"])
    w.writerow(["sgns", sgns_metrics["wordsim353"], sgns_metrics["simlex999"], sgns_metrics["analogy_pct"]])
    w.writerow(["cbow", cbow_metrics["wordsim353"], cbow_metrics["simlex999"], cbow_metrics["analogy_pct"]])

with open(outdir / "analogy_categories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "category", "accuracy_pct", "correct", "total"])
    for c in sgns_cats:
        w.writerow(["sgns", c["category"], c["accuracy_pct"], c["correct"], c["total"]])
    for c in cbow_cats:
        w.writerow(["cbow", c["category"], c["accuracy_pct"], c["correct"], c["total"]])
