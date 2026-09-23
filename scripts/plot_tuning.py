"""Render publication/export figures from saved tuning histories, without training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    runs = [r for r in json.loads((args.study / "ledger.json").read_text())
            if r["seed"] == 42 and r["instrumented"]]
    colors = ["#2563eb", "#0891b2", "#7c3aed", "#dc2626", "#ea580c", "#16a34a"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for axis, metric, title in zip(axes, ("macro", "original"), ("Category-macro validation loss", "Original-panel validation loss")):
        for run, color in zip(runs, colors):
            history = json.loads((args.study / run["run_id"] / "history.json").read_text())
            history = [h for h in history if h["step"] >= 250]
            axis.plot([h["step"] for h in history], [h[metric] for h in history],
                      color=color, linewidth=1.8, label=f"{run['sampler']}, LR {run['learning_rate']:g}")
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Cross-entropy loss", xlim=(250, 3000))
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controlled tuning: fixed data, six configurations", fontsize=16, fontweight="bold", y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .055))
    fig.text(.5, .015, "Seed 42. Curves start at step 250; complete histories include step 0. Lower is better.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .17, 1, .93))
    fig.savefig(args.study / "curves_matplotlib.png", dpi=180)
    fig.savefig(args.study / "curves_matplotlib.svg")
    plt.close(fig)
    print(args.study / "curves_matplotlib.png")


if __name__ == "__main__":
    main()
