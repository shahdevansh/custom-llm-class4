"""Build a factual tuning report from completed, saved study artifacts.

Usage: python scripts/build_tuning_report.py tuning_runs/STUDY [--sensitivity-study PATH]
No training, sampling, or evaluation is performed. PyTorch is used only to read
the two recorded update checkpoints; SVG charts use the standard library.
"""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("starter", "grammar", "contrasts")
COLORS = ("#1768ac", "#dd6b20", "#28834f", "#8b4fc3", "#c43c53", "#157f83")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def number(value, digits=6):
    return f"{value:.{digits}f}"


def change(value, digits=6):
    return f"{value:+.{digits}f}"


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value):
    return "`" + cell(value) + "`"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|",
                      *("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)])


def link(path, label=None, *, from_dir=None):
    relative = Path(os.path.relpath(path, from_dir or ROOT / "docs")).as_posix()
    return f"[{label or path.name}]({relative})"


def label(run):
    return f"{run['sampling']}, LR {run['peak_lr']:g}"


def write_svg(path, runs, histories, metric, threshold=None):
    """Plot all six histories at full scale and with the initial point excluded."""
    width, height = 1100, 880
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           f'<title id="title">Seed 42 {escape(metric)} validation loss</title>',
           '<desc id="desc">All six settings. Upper panel includes initialization; lower panel excludes step zero to show later differences. Lower loss is better.</desc>',
           '<rect width="1100" height="880" fill="#fff"/>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202b3a} .tick{font-size:12px} .grid{stroke:#e4e9ef;stroke-width:1}</style>',
           f'<text x="65" y="34" font-size="24" font-weight="bold">Seed 42: {escape(metric)} validation loss</text>',
           '<text x="65" y="57" font-size="14">Mean next-token negative log likelihood (NLL); lower is better</text>']
    for panel, include_initial in enumerate((True, False)):
        left, top, chart_w, chart_h = 80, 102 + panel * 312, 950, 237
        series = [[p for p in histories[r["run_id"]] if include_initial or p["step"] > 0] for r in runs]
        values = [p[metric] for points in series for p in points]
        if threshold is not None:
            values.append(threshold)
        y_min, y_max = min(values), max(values)
        padding = max((y_max - y_min) * .07, .03)
        y_min, y_max = max(0, y_min - padding), y_max + padding
        x_min = min(p["step"] for points in series for p in points)
        x_max = max(p["step"] for points in series for p in points)
        x = lambda step: left + (step - x_min) / max(1, x_max - x_min) * chart_w
        y = lambda loss: top + chart_h - (loss - y_min) / (y_max - y_min) * chart_h
        svg.append(f'<text x="80" y="{top - 12}" font-size="14" font-weight="bold">{"Complete history" if include_initial else "Later updates (step zero omitted)"}</text>')
        for tick in range(6):
            value = y_min + tick / 5 * (y_max - y_min)
            yy = y(value)
            svg.append(f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{left + chart_w}" y2="{yy:.2f}"/>')
            svg.append(f'<text class="tick" x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end">{value:.3f}</text>')
        ticks = sorted({x_min, *range(500, x_max + 1, 500), x_max})
        for step in ticks:
            if step < x_min:
                continue
            xx = x(step)
            svg.append(f'<text class="tick" x="{xx:.2f}" y="{top + chart_h + 20}" text-anchor="middle">{step}</text>')
        svg.append(f'<text x="555" y="{top + chart_h + 41}" font-size="13" text-anchor="middle">Completed optimizer updates</text>')
        svg.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#657384"/>')
        if threshold is not None:
            yy = y(threshold)
            svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + chart_w}" y2="{yy:.2f}" stroke="#333" stroke-width="1.5" stroke-dasharray="6 5"/>')
        for index, (run, points) in enumerate(zip(runs, series)):
            pairs = " ".join(f"{x(p['step']):.2f},{y(p[metric]):.2f}" for p in points)
            svg.append(f'<polyline points="{pairs}" fill="none" stroke="{COLORS[index]}" stroke-width="2.3"><title>{escape(label(run))}</title></polyline>')
            end = points[-1]
            svg.append(f'<circle cx="{x(end["step"]):.2f}" cy="{y(end[metric]):.2f}" r="3.5" fill="{COLORS[index]}"/>')
    for index, run in enumerate(runs):
        xx, yy = 80 + (index % 3) * 335, 755 + (index // 3) * 27
        svg.append(f'<line x1="{xx}" y1="{yy}" x2="{xx + 28}" y2="{yy}" stroke="{COLORS[index]}" stroke-width="3"/>')
        svg.append(f'<text x="{xx + 37}" y="{yy + 4}" font-size="13">{escape(label(run))}</text>')
    if threshold is not None:
        svg.append('<line x1="80" y1="816" x2="108" y2="816" stroke="#333" stroke-width="1.5" stroke-dasharray="6 5"/>')
        svg.append(f'<text x="117" y="820" font-size="13">Eligibility ceiling: 1.05 × final matched baseline original-panel loss = {threshold:.6f}</text>')
    svg.append('<text x="80" y="855" font-size="12">Source: saved history.json files. Straight segments connect measurements; intermediate values were not measured.</text></svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def summarize_rows(rows):
    total = len(rows)
    correct = sum(r["score"] for r in rows)
    scorable = sum(r["status"] in {"scored", "tied"} for r in rows)
    return {"correct": correct, "total": total, "scorable": scorable,
            "success_rate_all_cases": correct / total,
            "accuracy_scorable_cases": correct / scorable if scorable else None,
            "coverage": scorable / total}


def build(study, sensitivity_study=None, fallback_svg=False):
    import torch

    study = study.resolve()
    if sensitivity_study is None and (study / "summary.json").is_file():
        previous_sensitivity = read(study / "summary.json").get("sensitivity")
        if previous_sensitivity:
            sensitivity_study = ROOT / previous_sensitivity["study"]
    required = ("preregistered_plan.json", "run_ledger.json", "selection.json",
                "paired_comparison.json", "audit.json", "instrumentation_control.json", "frozen_panels.json")
    missing = [name for name in required if not (study / name).is_file()]
    if missing:
        raise RuntimeError("Study is not complete; missing " + ", ".join(missing))
    plan, runs, selection, pairs, audit, control, panels = [read(study / name) for name in required]
    if audit["failed"]:
        raise RuntimeError("Study audit has failures; inspect audit.json before reporting.")
    if not all(control[key] for key in ("model_equal", "optimizer_equal", "torch_rng_equal")):
        raise RuntimeError("Instrumentation control did not match.")
    by_id = {r["run_id"]: r for r in runs}
    if len(by_id) != len(runs):
        raise ValueError("Duplicate run IDs")
    paths = {r["run_id"]: study / "runs" / r["run_id"] for r in runs}
    histories, summaries, results = {}, {}, {}
    for run in runs:
        rid, path = run["run_id"], paths[run["run_id"]]
        histories[rid] = read(path / "history.json")
        summaries[rid] = read(path / "evals/eval_summary.json")
        results[rid] = read(path / "evals/eval_results.json")
        assert run["completed_steps"] == plan["steps"] == histories[rid][-1]["step"]
        assert histories[rid][-1] == run["final"]
        assert summarize_rows(results[rid]) == summaries[rid]["overall"] == run["evaluation"]
        assert all(math.isfinite(p[g]) for p in histories[rid] for g in (*GROUPS, "macro", "original"))
    seed42 = [r for r in runs if r["seed"] == 42]
    assert len(seed42) == len(plan["configs"])
    assert {(r["sampling"], r["peak_lr"]) for r in seed42} == {tuple(c) for c in plan["configs"]}
    baseline = next(r for r in seed42 if r["sampling"] == "uniform" and r["peak_lr"] == .001)
    chosen = by_id[selection["selected_run"]]
    threshold = 1.05 * baseline["final"]["original"]
    eligible = [r for r in seed42 if r["final"]["original"] <= threshold]
    assert math.isclose(threshold, selection["original_panel_threshold"], abs_tol=1e-12)
    assert {r["run_id"] for r in eligible} == set(selection["eligible"])
    assert chosen == min(eligible, key=lambda r: (r["final"]["macro"], r["run_id"]))
    assert not selection["language_evaluation_used"]
    assert [chosen["sampling"], chosen["peak_lr"]] == selection["configuration"]
    assert sorted(p["seed"] for p in pairs) == sorted(plan["seeds"])
    for pair in pairs:
        base, selected = by_id[pair["baseline"]], by_id[pair["selected"]]
        assert base["seed"] == selected["seed"] == pair["seed"]
        assert (base["sampling"], base["peak_lr"]) == ("uniform", .001)
        assert [selected["sampling"], selected["peak_lr"]] == selection["configuration"]
        assert pair["original_guardrail_pass"] == (selected["final"]["original"] <= 1.05 * base["final"]["original"])
        a = {r["id"]: r for r in results[base["run_id"]]}
        b = {r["id"]: r for r in results[selected["run_id"]]}
        assert a.keys() == b.keys()
        assert pair["gained"] == [rid for rid in b if b[rid]["score"] > a[rid]["score"]]
        assert pair["regressed"] == [rid for rid in b if b[rid]["score"] < a[rid]["score"]]
        assert pair["baseline_correct"] == summaries[base["run_id"]]["overall"]["correct"]
        assert pair["selected_correct"] == summaries[selected["run_id"]]["overall"]["correct"]
        assert math.isclose(pair["macro_loss_change"], selected["final"]["macro"] - base["final"]["macro"], abs_tol=1e-12)
        assert math.isclose(pair["original_loss_change"], selected["final"]["original"] - base["final"]["original"], abs_tol=1e-12)
    all_cases = {r["evaluation"]["total"] for r in runs}
    all_coverage = {r["evaluation"]["scorable"] for r in runs}
    assert len(all_cases) == len(all_coverage) == 1
    total, covered = next(iter(all_cases)), next(iter(all_coverage))
    source_suite = results[baseline["run_id"]]
    for rows in results.values():
        assert [(r["id"], r["prompt"], r["choices"], r["expected"], r["status"] in {"scored", "tied"}) for r in rows] == [(r["id"], r["prompt"], r["choices"], r["expected"], r["status"] in {"scored", "tied"}) for r in source_suite]
    update_path = paths[chosen["run_id"]]
    update = read(update_path / "gradient_update.json")
    before = torch.load(update_path / "update_1000_before.pt", map_location="cpu", weights_only=True)
    after = torch.load(update_path / "update_1000_after.pt", map_location="cpu", weights_only=True)
    probe = update["probe"]
    token_id, coordinate = probe["token_id"], probe["coordinate"]
    assert before["vocabulary"][token_id] == after["vocabulary"][token_id] == probe["embedding_token"]
    assert before["completed_steps"] + 1 == after["completed_steps"] == update["update"] == plan["gradient_update"]
    before_vector = before["model"]["transformer.wte.weight"][token_id].tolist()
    after_vector = after["model"]["transformer.wte.weight"][token_id].tolist()
    assert len(before_vector) == len(after_vector) == before["model_args"]["n_embd"]
    assert before_vector[coordinate] == update["before"]["parameter"]
    assert after_vector[coordinate] == update["after"]["parameter"]
    assert abs(update["reconstruction_error"]) < 1e-7
    vector_artifact = {"run_id": chosen["run_id"], "update": update["update"], "probe": probe,
                       "embedding_dimensions": len(before_vector), "embedding_before": before_vector,
                       "embedding_after": after_vector,
                       "embedding_delta": [b - a for a, b in zip(before_vector, after_vector)],
                       "before_completed_steps": before["completed_steps"], "after_completed_steps": after["completed_steps"],
                       "source_before": f"runs/{chosen['run_id']}/update_1000_before.pt",
                       "source_after": f"runs/{chosen['run_id']}/update_1000_after.pt"}
    vector_file = study / "selected_update_vectors.json"
    dump(vector_file, vector_artifact)
    macro_svg, original_svg = study / "macro_validation_curves.svg", study / "original_validation_curves.svg"
    if fallback_svg or not (macro_svg.is_file() and original_svg.is_file()):
        write_svg(macro_svg, seed42, histories, "macro")
        write_svg(original_svg, seed42, histories, "original", threshold)
    base_eval, chosen_eval = summaries[baseline["run_id"]], summaries[chosen["run_id"]]
    seed42_pair = next(p for p in pairs if p["seed"] == 42)
    score_deltas = [p["selected_correct"] - p["baseline_correct"] for p in pairs]
    macro_deltas = [p["macro_loss_change"] for p in pairs]
    original_deltas = [p["original_loss_change"] for p in pairs]
    base_macro, chosen_macro = baseline["final"]["macro"], chosen["final"]["macro"]
    selected_mean = statistics.mean(p["selected_correct"] for p in pairs)
    baseline_mean = statistics.mean(p["baseline_correct"] for p in pairs)
    sensitivity = None
    if sensitivity_study is not None:
        sensitivity_study = sensitivity_study.resolve()
        other_plan = read(sensitivity_study / "protocol.json")
        other_pairs = read(sensitivity_study / "paired_results.json")
        other_runs = {r["run_id"]: r for r in read(sensitivity_study / "ledger.json")}
        other_audit = read(sensitivity_study / "audit.json")
        other_selection = read(sensitivity_study / "selection.json")
        assert all(c["passed"] for c in other_audit)
        assert other_plan["input_hashes"] == plan["input_sha256"]
        assert other_plan["panels"] == panels
        assert [other_selection["sampler"], other_selection["learning_rate"]] == selection["configuration"]
        for run in seed42:
            other = other_runs[run["run_id"]]
            assert other["final_model_sha256"] == run["model_sha256"]
            assert other["schedule_sha256"] == run["schedule_sha256"]
            assert other["final"] == run["final"] and other["evaluation"] == run["evaluation"]
        for pair in other_pairs:
            other_base, other_chosen = other_runs[pair["baseline"]], other_runs[pair["selected"]]
            assert other_base["seed"] == other_chosen["seed"] == pair["seed"]
            assert pair["baseline_correct"] == other_base["evaluation"]["correct"]
            assert pair["selected_correct"] == other_chosen["evaluation"]["correct"]
            for metric in ("macro", "original"):
                assert math.isclose(pair[f"{metric}_loss_change"], other_chosen["final"][metric] - other_base["final"][metric], abs_tol=1e-12)
            assert pair["guardrail_met"] == (other_chosen["final"]["original"] <= 1.05 * other_base["final"]["original"])
        for sampler in other_plan["samplers"]:
            assert len({r["schedule_sha256"] for r in other_runs.values() if r["sampler"] == sampler}) == 1
        sensitivity = {"study": str(sensitivity_study.relative_to(ROOT)), "data_rng_seed": other_plan["data_rng_seed"],
                       "protocol": "Vary initialization only; hold each sampler's seed-42 minibatch schedule fixed.",
                       "shared_seed42_sweep": True, "pooled": False, "paired_results": other_pairs,
                       "audit_passed": len(other_audit),
                       "interpretation": "Positive joint-seed findings did not replicate under fixed minibatch schedules. The selected setting remains exploratory; retain baseline pending crossed initialization and sampler seeds plus template-family holdouts."}
    output = ["# Controlled learning-rate and sampling study", "",
        f"Validation selected **{chosen['sampling']} sampling, peak learning rate {chosen['peak_lr']:g}**. "
        f"At seed 42, macro validation NLL changed from **{base_macro:.6f} to {chosen_macro:.6f}** "
        f"({100 * (chosen_macro / base_macro - 1):+.2f}%). Original-panel NLL changed from "
        f"**{baseline['final']['original']:.6f} to {chosen['final']['original']:.6f}** "
        f"({100 * (chosen['final']['original'] / baseline['final']['original'] - 1):+.2f}%), within the 5% guardrail. "
        "The public development benchmark changed from "
        f"**{base_eval['overall']['correct']}/{total} to {chosen_eval['overall']['correct']}/{total}**. "
        f"Across {len(pairs)} paired seeds, mean correct answers changed from **{baseline_mean:.2f} to {selected_mean:.2f}**; "
        f"the mean paired change was **{statistics.mean(score_deltas):+.2f} cases**.", "",
        ("The selected setting remains an **exploratory candidate**. Its positive confirmation results did not replicate when initialization changed while minibatch schedules stayed fixed; "
         "retain the baseline until crossed initialization and sampler seeds plus template-family holdouts establish robustness. " if sensitivity else "") +
        "Confidence is **high** in the saved measurements and replay checks, **low** in a robust advantage across seed protocols, "
        "and **low** in broader language generalization. These scores are experimental measurements; no official grade is inferred.", "",
        f"Evidence: {link(study / 'preregistered_plan.json', 'frozen plan')} · {link(study / 'selection.json', 'selection')} · "
        f"{link(study / 'run_ledger.json', 'run ledger')} · {link(study / 'paired_comparison.json', 'paired results')} · "
        f"{link(study / 'audit.json', 'audit')} · {link(study / 'summary.json', 'computed report summary')}.", "",
        "## Controls and selection", "",
        f"The expanded corpus, vocabulary, train/validation split, architecture, {plan['steps']:,}-update budget and "
        f"batch size {plan['batch_size']} stayed fixed. Each seed used identical initial weights across settings; "
        "within each sampling method, learning rates used identical saved minibatch schedules. "
        "In this study's confirmation runs, initialization and minibatch RNG seeds both vary. "
        "Uniform sampling draws from all training passages. Balanced sampling draws 16 starter, eight grammar and eight contrast passages per batch.", "",
        table(["Passage category", "Training passages", "Held-out passages", "Frozen category-panel passages"],
              [(g, plan["train_category_counts"][g], plan["validation_category_counts"][g], len(panels[g])) for g in GROUPS]), "",
        f"Each category NLL averages non-padding next-token targets. Macro NLL gives the {len(GROUPS)} category losses equal weight. "
        f"The separate original panel has {len(panels['original'])} held-out passages. The predeclared rule chose the lowest "
        f"final seed-42 macro NLL among settings with original-panel NLL at most **{threshold:.9f}** "
        f"(1.05 × the matched baseline's {baseline['final']['original']:.9f}); ties use run ID. "
        "The selection was saved before public-suite evaluation and remained fixed for the additional seeds.", "",
        "## Six-setting seed-42 sweep", "",
        "Language benchmark columns below were measured after validation selection. Scorable accuracy excludes cases without usable vocabulary/context; all-case success retains them as failures.", "",
        table(["Sampling", "Peak LR", "Macro val. NLL", "Original val. NLL", "Eligible", f"Correct/{total}", f"Correct/scorable ({covered})", "Selected"],
              [(r["sampling"], f"{r['peak_lr']:g}", number(r["final"]["macro"]), number(r["final"]["original"]),
                "yes" if r in eligible else "no", f"{r['evaluation']['correct']}/{total} ({percent(r['evaluation']['success_rate_all_cases'])})",
                f"{r['evaluation']['correct']}/{covered} ({percent(r['evaluation']['accuracy_scorable_cases'])})",
                "yes" if r == chosen else "") for r in seed42]), "",
        f"![All six macro validation curves]({Path(os.path.relpath(macro_svg, ROOT / 'docs')).as_posix()})", "",
        f"![All six original validation curves with eligibility ceiling]({Path(os.path.relpath(original_svg, ROOT / 'docs')).as_posix()})", "",
        (f"PNG exports: {link(study / 'macro_validation_curves.png', 'macro validation')} · "
         f"{link(study / 'original_validation_curves.png', 'original-panel validation')}." if (study / 'macro_validation_curves.png').is_file() and (study / 'original_validation_curves.png').is_file() else ""), "",
        "Both figures include initialization and a second panel excluding step zero. The dashed original-panel line is the final baseline's 5% eligibility ceiling. "
        "Lines connect recorded measurements; they do not supply unmeasured intermediate losses.", "",
        "## Paired confirmation", "",
        "Baseline is uniform sampling at peak LR 0.001. Selected settings remain fixed across seeds. Negative NLL changes mean lower loss.", "",
        table(["Seed", f"Untrained correct/{total}", f"Baseline correct/{total}", f"Selected correct/{total}", "Correct change", "Macro NLL change", "Original NLL change", "Original 5% guardrail", "Gains / regressions"],
              [(p["seed"], f"{read(study / 'untrained_evals' / f'seed{p["seed"]}' / 'eval_summary.json')['overall']['correct']}/{total}",
                f"{p['baseline_correct']}/{total}", f"{p['selected_correct']}/{total}", f"{p['selected_correct'] - p['baseline_correct']:+d}",
                change(p["macro_loss_change"]), change(p["original_loss_change"]), "pass" if p["original_guardrail_pass"] else "fail",
                f"{len(p['gained'])} / {len(p['regressed'])}") for p in pairs]), "",
        table(["Paired change, selected minus baseline", "Mean", "Sample SD", "Minimum", "Maximum"],
              [(name, change(statistics.mean(values)), number(statistics.stdev(values)) if len(values) > 1 else "N/A", change(min(values)), change(max(values)))
               for name, values in [("Correct cases", score_deltas), ("Macro validation NLL", macro_deltas), ("Original validation NLL", original_deltas)]]), "",
        f"The selected setting passed the original-panel guardrail in **{sum(p['original_guardrail_pass'] for p in pairs)}/{len(pairs)} seeds**. "
        "These are paired descriptive results from a small seed set. Seed 42 was also used to choose the setting; seeds 43 and 44 are confirmation runs on the same frozen data split. "
        "The standard deviation describes variation among these observed seed differences.", "",
        "## Coverage, groups and categories", "",
        f"Coverage stayed at **{covered}/{total} ({percent(covered / total)})** for every final run; the same cases remained scorable. "
        f"The {total - covered} unscorable cases stay in the all-case denominator. Learning-rate and sampler changes cannot add missing vocabulary. "
        "Table cells show correct/total (all-case success), followed by correct/scorable (scorable accuracy).", ""]
    if sensitivity:
        rows = []
        for protocol_label, protocol_pairs, guardrail_key in [
                ("Initialization + minibatch seeds vary", pairs, "original_guardrail_pass"),
                ("Initialization varies; minibatches fixed", other_pairs, "guardrail_met")]:
            for pair in protocol_pairs:
                if pair["seed"] == 42:
                    continue
                rows.append((protocol_label, pair["seed"], f"{pair['baseline_correct']}/{total} → {pair['selected_correct']}/{total}",
                             change(pair["macro_loss_change"], 9), change(pair["original_loss_change"], 9),
                             "pass" if pair[guardrail_key] else "fail"))
        sensitivity_section = ["## Sensitivity to seed protocol", "",
            f"A parallel {link(sensitivity_study / 'REPORT.md', 'study')} reran the same six-setting screen and selected the same configuration. "
            "Its seed-42 final weights, validation values and benchmark outcomes match this study exactly. "
            f"Its confirmation runs vary initialization while holding each sampler's minibatch schedule fixed (data RNG seed {other_plan['data_rng_seed']}); "
            "the present study changes initialization and minibatch RNG together. "
            f"The companion {link(sensitivity_study / 'protocol.json', 'protocol')}, {link(sensitivity_study / 'paired_results.json', 'paired results')} "
            f"and {link(sensitivity_study / 'audit.json', 'audit')} preserve the evidence.", "",
            table(["Confirmation protocol", "Initialization seed", "Correct: baseline → selected", "Macro NLL change", "Original NLL change", "5% guardrail"], rows), "",
            "The macro-loss advantage and benchmark gains did not replicate under the fixed-minibatch protocol. "
            "Both studies share the seed-42 screen; their results are shown separately and are not pooled as independent replications. "
            "Different confirmation outcomes indicate sensitivity to the combination of initialization and data order. "
            "The selected setting remains exploratory. Retain the baseline until a crossed experiment varies initialization and minibatch seeds independently, with template-family holdouts to assess transfer.", ""]
        index = output.index("## Coverage, groups and categories")
        output[index:index] = sensitivity_section
    for heading, key in (("Groups", "by_group"), ("Categories", "by_category")):
        def score_cell(values):
            return f"{values['correct']}/{values['total']} ({percent(values['success_rate_all_cases'])}); {values['correct']}/{values['scorable']} ({percent(values['accuracy_scorable_cases'])})"
        output.extend([f"### {heading}", "", table(["Group or category", "Baseline, seed 42", "Selected, seed 42", "Scorable / total in both"],
            [(name, score_cell(base_eval[key][name]), score_cell(chosen_eval[key][name]), f"{base_eval[key][name]['scorable']}/{base_eval[key][name]['total']}") for name in sorted(base_eval[key])]), ""])
    base_rows = {r["id"]: r for r in results[baseline["run_id"]]}
    selected_rows = {r["id"]: r for r in results[chosen["run_id"]]}
    changed_ids = sorted(set(seed42_pair["gained"]) | set(seed42_pair["regressed"]))
    additional_ids = [rid for rid in sorted(base_rows) if rid not in changed_ids][:2]
    output.extend(["## Individual cases and saved free continuations", "",
        f"Seed-42 gains: {', '.join(code(rid) for rid in seed42_pair['gained']) or 'none'}. "
        f"Seed-42 regressions: {', '.join(code(rid) for rid in seed42_pair['regressed']) or 'none'}.", "",
        "The examples include every seed-42 gain and regression, plus the first two remaining case IDs in sorted order. "
        "This deterministic display rule uses existing evaluation outputs and performs no new inference. "
        f"Generation settings were temperature {base_eval['settings']['temperature']}, at most {base_eval['settings']['max_tokens']} new tokens, "
        "and each case's saved sampling seed. Free continuations are separate from four-choice scoring.", ""])
    for rid in changed_ids + additional_ids:
        a, b = base_rows[rid], selected_rows[rid]
        outcome = "gain" if rid in seed42_pair["gained"] else "regression" if rid in seed42_pair["regressed"] else "additional fixed-order example"
        assert a["sample_seed"] == b["sample_seed"]
        output.extend([f"### {rid}: {outcome}", "",
            f"Prompt: {code(a['prompt'])}. Expected answer: {code(a['expected'])}. Sampling seed: {a['sample_seed']}.", "",
            table(["Model", "Choice", "Correct", "Status", "Actual free continuation"], [
                ("Baseline", code(a["predicted_choice"]), a["score"], a["status"], code(a["generated_text"])),
                ("Selected", code(b["predicted_choice"]), b["score"], b["status"], code(b["generated_text"]))]), ""])
    before_obs, after_obs = update["before"], update["after"]
    output.extend(["## One actual optimizer update", "",
        f"Update **{update['update']:,}** in {code(chosen['run_id'])} used the actual saved minibatch. "
        "The passage, prefix, target and coordinate were selected from its schedule before training.", "",
        f"Passage: {code(probe['passage'])}", "",
        f"Prefix: {code(probe['prefix'])}. Next-token target: {code(probe['target'])}. "
        f"Tracked embedding token: {code(probe['embedding_token'])}; token ID **{token_id}**, coordinate **{coordinate}**.", "",
        table(["Measurement", f"Before ({before['completed_steps']} completed updates)", f"After ({after['completed_steps']} completed updates)", "Change"],
              [(name, number(a, 12), number(b, 12), change(b - a, 12)) for name, a, b in [
                  ("Actual minibatch NLL", before_obs["batch_loss"], after_obs["batch_loss"]),
                  ("Probability of target after prefix", before_obs["target_probability"], after_obs["target_probability"]),
                  (f"Embedding coordinate {coordinate}", before_obs["parameter"], after_obs["parameter"]),
                  ("Macro validation NLL", before_obs["validation"]["macro"], after_obs["validation"]["macro"]),
                  ("Original-panel validation NLL", before_obs["validation"]["original"], after_obs["validation"]["original"]),
                  *((f"{g.title()} validation NLL", before_obs["validation"][g], after_obs["validation"][g]) for g in GROUPS)]]), "",
        table(["Optimizer measurement", "Recorded value"], [
            ("Effective learning rate", f"{update['learning_rate']:.17g}"),
            ("Raw coordinate gradient", f"{update['raw_gradient']:.17g}"),
            ("Global gradient norm before clipping", f"{update['gradient_norm_before_clipping']:.17g}"),
            ("Clip norm limit", plan["optimizer"]["clip_norm"]),
            ("Clip multiplier", f"{update['clip_multiplier']:.17g}"),
            ("Clipped coordinate gradient", f"{update['clipped_gradient']:.17g}"),
            ("AdamW first moment before", f"{update['moments_before']['m']:.17g}"),
            ("AdamW first moment after", f"{update['moments_after']['m']:.17g}"),
            ("AdamW second moment before", f"{update['moments_before']['v']:.17g}"),
            ("AdamW second moment after", f"{update['moments_after']['v']:.17g}"),
            ("Weight-decay contribution to coordinate change", f"{update['weight_decay_delta']:.17g}"),
            ("Adaptive contribution to coordinate change", f"{update['adaptive_delta']:.17g}"),
            ("Float64 reconstructed coordinate change", f"{update['predicted_delta_float64']:.17g}"),
            ("Actual saved float32 coordinate change", f"{update['actual_delta']:.17g}"),
            ("Actual minus reconstructed change", f"{update['reconstruction_error']:.17g}")]), "",
        f"AdamW used β₁={plan['optimizer']['betas'][0]}, β₂={plan['optimizer']['betas'][1]}, "
        f"ε={plan['optimizer']['eps']:g} and weight decay λ={plan['optimizer']['weight_decay']}. "
        "For clipped gradient g, mₜ=β₁mₜ₋₁+(1−β₁)g and vₜ=β₂vₜ₋₁+(1−β₂)g². "
        "The recorded coordinate change reconstructs as Δθ=−ηλθ−η[mₜ/(1−β₁ᵗ)]/[√(vₜ/(1−β₂ᵗ))+ε]. "
        "Float32 optimizer arithmetic and the displayed float64 reconstruction differ by the small residual above.", "",
        f"The complete **{len(before_vector)}-coordinate before and after embedding vectors**, their deltas and source-checkpoint paths are in "
        f"{link(vector_file)}. The full {link(update_path / 'gradient_update.json', 'update record')}, "
        f"{link(update_path / 'update_1000_before.pt', 'before checkpoint')} and "
        f"{link(update_path / 'update_1000_after.pt', 'after checkpoint')} retain the source evidence.", ""])
    worsened = [name for name in ("macro", "original", *GROUPS) if after_obs["validation"][name] > before_obs["validation"][name]]
    output.extend([
        ("This single update increased validation NLL on the " + " and ".join(worsened) + " panels. " if worsened else "Every recorded validation panel improved in this particular update. ") +
        "An update can reduce its minibatch loss while worsening held-out loss; a single update does not establish a generalization trend. "
        "AdamW changes the full network at once. The observed target-probability change cannot be assigned causally to this coordinate alone. "
        "The token embedding is also tied to the output weight matrix, so its gradient can include input and output uses.", "",
        "## Complete final-run evidence", "",
        f"The study contains **{len(runs)} final experimental runs**, plus one separate instrumentation control. "
        "Each experimental row links its checkpoint, loss history, raw case outputs, evaluation summary and update record.", "",
        table(["Run", "Model", "Loss history", "Cases / summary", "Measured update"],
              [(code(r["run_id"]), link(paths[r["run_id"]] / "model.pt", "weights"),
                link(paths[r["run_id"]] / "history.json", "history"),
                link(paths[r["run_id"]] / "evals/eval_results.json", "JSON") + " · " + link(paths[r["run_id"]] / "evals/eval_results.csv", "CSV") + " · " + link(paths[r["run_id"]] / "evals/eval_summary.json", "summary"),
                link(paths[r["run_id"]] / "gradient_update.json", "update 1,000")) for r in runs]), "",
        f"The uninstrumented control {code(control['control_run'])} had exactly equal final model weights, optimizer state and Torch RNG state "
        f"({link(study / 'instrumentation_control.json', 'comparison')}; {link(study / 'runs' / control['control_run'] / 'model.pt', 'control checkpoint')}). "
        f"The saved training audit reports **{audit['passed']} passed checks and {audit['failed']} failures**, including exact final-evaluation replays "
        "and reconstruction of every measured AdamW coordinate change. The report builder additionally checks saved summaries against case rows, "
        "recomputes selection eligibility and paired changes, and reads both embedding vectors directly from checkpoints.", "",
        "## Limits and ranked next experiments", "",
        "1. **Improve teaching coverage and example quality.** Add independently designed vocabulary and varied, correct grammatical and contrast examples, then freeze a new suite with separate development and untouched test cases. "
        f"The present vocabulary leaves {total - covered}/{total} cases unscorable, so this is a concrete ceiling on the current benchmark.",
        "2. **Use stronger held-out splits.** Hold out template families and source patterns, enlarge the validation panels and evaluate all held-out passages. "
        "Shared synthetic templates and a small frozen panel limit what the current losses establish.",
        "3. **Replicate the fixed selected setting.** Cross independent initialization and minibatch seeds and report paired differences, including regressions. "
        "Three observed seeds and a public development benchmark give limited evidence of transfer.",
        "4. **Test training budget and scheduling.** Compare longer runs and an explicitly predeclared stopping rule, keeping data and initialization paired. "
        "Use held-out curves to decide whether extra updates help; this study does not establish that longer training improves language quality.",
        "5. **Test tokenizer or architecture changes separately.** A subword tokenizer could remove the current whole-word vocabulary barrier, while capacity changes may alter learning. "
        "Each changes the experiment and requires fresh controlled baselines.", "",
        "This supplement holds the expanded data fixed and measures optimization changes. The corpus still uses a narrow synthetic distribution. "
        "The public benchmark previously informed corpus-category choices, and inspecting its outcomes can guide later development; it is not an untouched generalization test. "
        "Free continuation quality and highest-probability choice accuracy measure different behaviors.", "",
        "## Rebuild the report and exported figures", "",
        "These commands read saved artifacts and perform no training or inference. Matplotlib is an optional plotting dependency; the training requirements remain unchanged.", "",
        "```sh",
        f".venv/bin/python scripts/build_tuning_report.py {study.relative_to(ROOT)}" + (f" --sensitivity-study {sensitivity_study.relative_to(ROOT)}" if sensitivity else ""),
        f"uv run --with matplotlib scripts/plot_controlled_tuning.py {study.relative_to(ROOT)}",
        "```", "",
        "The report builder preserves existing curves. Its optional `--fallback-svg` flag regenerates standard-library SVGs when Matplotlib is unavailable; "
        "the plotting command replaces them with the standard Matplotlib SVG and PNG exports.", ""])
    summary = {"study": str(study.relative_to(ROOT)), "selected_run": chosen["run_id"],
               "configuration": selection["configuration"], "selection_metric": selection["selection_metric"],
               "experimental_runs": len(runs), "instrumentation_control_runs": 1,
               "seeds": plan["seeds"], "total_cases": total, "scorable_cases": covered,
               "coverage": covered / total, "original_panel_threshold": threshold,
               "seed42_baseline": {"run_id": baseline["run_id"], "final": baseline["final"], "evaluation": baseline["evaluation"]},
               "seed42_selected": {"run_id": chosen["run_id"], "final": chosen["final"], "evaluation": chosen["evaluation"]},
               "mean_baseline_correct": baseline_mean, "mean_selected_correct": selected_mean,
               "mean_correct_change": statistics.mean(score_deltas), "mean_macro_loss_change": statistics.mean(macro_deltas),
               "mean_original_loss_change": statistics.mean(original_deltas),
               "paired_correct_change_sample_sd": statistics.stdev(score_deltas) if len(score_deltas) > 1 else None,
               "seed42_gained": seed42_pair["gained"], "seed42_regressed": seed42_pair["regressed"],
               "displayed_case_ids": changed_ids + additional_ids, "update": update,
               "audit_passed": audit["passed"], "audit_failed": audit["failed"], "sensitivity": sensitivity,
               "recommendation": "Retain baseline; selected setting is exploratory pending crossed-seed and template-holdout confirmation." if sensitivity else "Treat the selected setting as a limited development result."}
    dump(study / "summary.json", summary)
    report = ROOT / "docs/tuning_results.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(output), encoding="utf-8")
    print(f"Report: {report}")
    print(f"Selected: {chosen['run_id']}; {len(runs)} runs; {audit['passed']} audit checks passed")
    print(f"SVG curves and complete embedding vectors: {study}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Completed study directory")
    parser.add_argument("--sensitivity-study", type=Path, help="Completed fixed-sampling companion study; retained on later rebuilds")
    parser.add_argument("--fallback-svg", action="store_true", help="Regenerate standard-library SVGs instead of preserving existing figures")
    args = parser.parse_args()
    build(args.study, args.sensitivity_study, args.fallback_svg)


if __name__ == "__main__":
    main()
