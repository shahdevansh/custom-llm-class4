"""Export Matplotlib SVG/PNG figures from saved tuning-study JSON only.

Chart contract: compare the six seed-42 validation trajectories, preserving the
full history and a focused panel after initialization. Thirteen measurements per
configuration support line charts; points mark actual measurements. Sampler uses
two explicit colors and learning rate uses line/marker styles, so color is not
the only distinction. The original-panel chart includes the frozen eligibility
ceiling. Exports are standalone report figures; no model is loaded or evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot(study):
    study = study.resolve()
    ledger = read(study / "run_ledger.json")
    runs = [r for r in ledger if r["seed"] == 42]
    selection = read(study / "selection.json")
    assert len(runs) == 6
    histories = {r["run_id"]: read(study / "runs" / r["run_id"] / "history.json") for r in runs}
    colors = {"uniform": "#1768ac", "balanced": "#dd6b20"}
    styles = {0.0003: (":", "o"), 0.001: ("--", "s"), 0.003: ("-", "^")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#202b3a", "text.color": "#202b3a",
                         "xtick.color": "#394656", "ytick.color": "#394656",
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    for metric, name in [("macro", "Category-macro"), ("original", "Original-panel")]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9.5))
        fig.subplots_adjust(top=.865, bottom=.29, hspace=.43, left=.09, right=.965)
        fig.suptitle(f"Seed 42: {name.lower()} validation loss", x=.09, y=.97,
                     ha="left", fontsize=19, fontweight="bold")
        fig.text(.09, .931, "Mean next-token negative log likelihood (NLL); lower is better. All six settings.", fontsize=11)
        handles = []
        for axis, include_initial in zip(axes, (True, False)):
            for run in runs:
                points = [p for p in histories[run["run_id"]] if include_initial or p["step"] > 0]
                line_style, marker = styles[run["peak_lr"]]
                line, = axis.plot([p["step"] for p in points], [p[metric] for p in points],
                                  color=colors[run["sampling"]], linestyle=line_style, marker=marker,
                                  markersize=3.5, linewidth=1.9, markerfacecolor="white",
                                  markeredgewidth=.9,
                                  label=f"{run['sampling']}, LR {run['peak_lr']:g} (final {points[-1][metric]:.4f})")
                if include_initial:
                    handles.append(line)
            if metric == "original":
                axis.axhline(selection["original_panel_threshold"], color="#3f4750", linestyle=(0, (5, 3)), linewidth=1.4)
            axis.set_title("Complete history" if include_initial else "Later updates (step zero omitted)",
                           loc="left", fontsize=12, fontweight="bold", pad=10)
            axis.set_xlabel("Completed optimizer updates", labelpad=7)
            axis.set_ylabel("Validation NLL", labelpad=9)
            axis.grid(axis="y", color="#e4e9ef", linewidth=.8)
            axis.set_axisbelow(True)
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            axis.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color("#aab3bd")
            axis.set_xlim(left=0 if include_initial else 250, right=3000)
            axis.margins(y=.08)
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.084, .108),
                   ncol=2, frameon=False, fontsize=10, handlelength=4.2,
                   columnspacing=3, labelspacing=.8)
        if metric == "original":
            fig.text(.09, .082, f"Dashed gray line: 5% eligibility ceiling = {selection['original_panel_threshold']:.6f} "
                     "(1.05 × final matched baseline).", fontsize=10)
        fig.text(.09, .043, "Source: saved history.json files. Markers show measurements; lines connect recorded points.", fontsize=9)
        fig.text(.09, .022, "Color identifies sampler; line and marker style identify learning rate. The second panel changes the y-axis scale.", fontsize=9)
        for extension in ("svg", "png"):
            target = study / f"{metric}_validation_curves.{extension}"
            fig.savefig(target, dpi=180)
            print(target)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    plot(args.study)


if __name__ == "__main__":
    main()
