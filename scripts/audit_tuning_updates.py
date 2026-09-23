"""Independently replay saved controlled-study updates without retraining.

Usage: python scripts/audit_tuning_updates.py tuning_runs/<study>
Reads the completed tune_controlled.py study. The only output is
<study>/update_replay_audit.json; evaluation reproduction stays in memory.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanogpt_model import GPT, GPTConfig
from run_evals import generate_reply, load_suite, model_hash, suite_hash, word_tokens

GROUPS = ("starter", "grammar", "contrasts")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def restore_model(saved):
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(GPTConfig(**saved["model_args"]))
    model.load_state_dict(saved["model"])
    return model


def batch(docs, stoi):
    sequences = [[stoi["<BOS>"]] + [stoi.get(t, stoi["<UNK>"]) for t in word_tokens(d)] + [stoi["<EOS>"]] for d in docs]
    length = max(len(s) - 1 for s in sequences)
    x = torch.full((len(sequences), length), stoi["<EOS>"], dtype=torch.long)
    y = torch.full_like(x, -1)
    for row, seq in enumerate(sequences):
        x[row, :len(seq)-1] = torch.tensor(seq[:-1])
        y[row, :len(seq)-1] = torch.tensor(seq[1:])
    return x, y


@torch.no_grad()
def observe(model, minibatch, probe, panel_batches, stoi):
    model.eval()
    ids = torch.tensor([[stoi["<BOS>"]] + [stoi[w] for w in word_tokens(probe["prefix"])]])
    probability = torch.softmax(model(ids)[0][0, -1], -1)[stoi[probe["target"]]].item()
    loss = model(*minibatch)[1].item()
    validation = {g: model(*b)[1].item() for g, b in panel_batches.items()}
    validation["macro"] = sum(validation[g] for g in GROUPS) / 3
    return {"batch_loss": loss, "target_probability": probability, "validation": validation,
            "parameter": model.transformer.wte.weight[probe["token_id"], probe["coordinate"]].item()}


def moments(state, coordinate):
    return {"step": int(state["step"].item()), "m": state["exp_avg"][coordinate].item(),
            "v": state["exp_avg_sq"][coordinate].item()}


@torch.inference_mode()
def evaluate_rows(model, vocabulary, suite):
    """Recompute the saved runner's rows without writing evaluation files."""
    model.eval()
    stoi = {word: i for i, word in enumerate(vocabulary)}
    rows = []
    for index, case in enumerate(suite["cases"]):
        prompt_tokens = word_tokens(case["prompt"])
        unknown_prompt = sorted(set(t for t in prompt_tokens if t not in stoi))
        unknown_choices = [c for c in case["choices"] if word_tokens(c)[0] not in stoi]
        status, prediction, probabilities, score = "scored", None, {}, 0
        if len(prompt_tokens) + 1 > model.config.block_size:
            status = "context_too_long"
        elif unknown_prompt or unknown_choices:
            status = "out_of_vocabulary"
        else:
            ids = [stoi["<BOS>"]] + [stoi[t] for t in prompt_tokens]
            probabilities_all = torch.softmax(model(torch.tensor([ids]))[0][0, -1].float().cpu(), -1)
            probabilities = {c: probabilities_all[stoi[word_tokens(c)[0]]].item() for c in case["choices"]}
            ranked = sorted(probabilities, key=probabilities.get, reverse=True)
            if abs(probabilities[ranked[0]] - probabilities[ranked[1]]) <= 1e-10:
                status = "tied"
            else:
                prediction = ranked[0]
                score = int(prediction == case["answer"])
        sample = generate_reply(model, vocabulary, case["prompt"], seed=2026 + index, temperature=.8, max_tokens=24)
        rows.append({"id": case["id"], "group": case["group"], "category": case["category"],
                     "prompt": case["prompt"], "choices": case["choices"], "expected": case["answer"],
                     "predicted_choice": prediction, "score": score, "status": status,
                     "choice_probabilities": probabilities, "unknown_prompt_words": unknown_prompt,
                     "unknown_choices": unknown_choices, "generated_text": sample["response"],
                     "prompt_truncated": sample["prompt_truncated"], "reason": case["reason"],
                     "stage": "untrained", "sample_seed": 2026 + index})
    return rows


def summarize(rows):
    available = sum(r["status"] in {"scored", "tied"} for r in rows)
    correct = sum(r["score"] for r in rows)
    return {"correct": correct, "total": len(rows), "scorable": available,
            "success_rate_all_cases": correct / len(rows),
            "accuracy_scorable_cases": correct / available if available else None,
            "coverage": available / len(rows)}


def audit(study):
    torch.set_num_threads(4)
    plan = read_json(study / "preregistered_plan.json")
    selection = read_json(study / "selection.json")
    control = read_json(study / "instrumentation_control.json")
    runs = read_json(study / "run_ledger.json")
    expected_pairs = {tuple(c) for c in plan["configs"]}
    expected_runs = {(sampler, lr, 42) for sampler, lr in expected_pairs}
    expected_runs |= {(sampler, lr, seed) for seed in (43, 44)
                      for sampler, lr in {("uniform", .001), tuple(selection["configuration"])}}
    actual_runs = {(r["sampling"], r["peak_lr"], r["seed"]) for r in runs}
    if actual_runs != expected_runs or len(runs) != len(expected_runs):
        raise RuntimeError("The complete six-configuration sweep and seed confirmations are required.")
    required = [study / "untrained_evals" / f"seed{s}" / "eval_results.json" for s in (42, 43, 44)]
    required += [ROOT / r["path"] / name for r in runs
                 for name in ("model.pt", "update_1000_before.pt", "update_1000_after.pt", "gradient_update.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Study is incomplete; missing {missing}")

    checks = []
    def check(name, passed, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})
    split = read_json(ROOT / plan["base_run"] / "split.json")
    panels = read_json(study / "frozen_panels.json")
    check("input_files_match_frozen_hashes", all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in plan["input_sha256"].items()))
    check("all_frozen_panels_are_validation_only", all(set(docs) <= set(split["validation"]) and not set(docs) & set(split["train"]) for docs in panels.values()))
    for run in runs:
        name = run["run_id"]
        path = ROOT / run["path"]
        before = torch.load(path / "update_1000_before.pt", map_location="cpu", weights_only=True)
        after = torch.load(path / "update_1000_after.pt", map_location="cpu", weights_only=True)
        logged = read_json(path / "gradient_update.json")
        probe = plan["preselected_probes"][f"{run['seed']}_{run['sampling']}"]
        with gzip.open(study / f"batches_seed{run['seed']}_{run['sampling']}.json.gz", "rt") as stream:
            indices = json.load(stream)[999]
        check(name + "_frozen_probe_and_batch", logged["probe"] == probe and indices == probe["batch_document_indices"] and len(indices) == 32)
        check(name + "_checkpoint_steps", before["completed_steps"] == 999 and after["completed_steps"] == 1000)
        model = restore_model(before)
        stoi = {word: i for i, word in enumerate(before["vocabulary"])}
        minibatch = batch([split["train"][i] for i in indices], stoi)
        panel_batches = {g: batch(docs, stoi) for g, docs in panels.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=run["peak_lr"], betas=(.9, .95), weight_decay=.01)
        optimizer.load_state_dict(before["optimizer"])
        torch.set_rng_state(before["torch_rng_state"])
        observed_before = observe(model, minibatch, probe, panel_batches, stoi)
        check(name + "_before_observation_exact", same(observed_before, logged["before"]))
        parameter = model.transformer.wte.weight
        coordinate = (probe["token_id"], probe["coordinate"])
        m_before = moments(optimizer.state[parameter], coordinate)
        check(name + "_before_moments_exact", same(m_before, logged["moments_before"]))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(*minibatch)[1]
        loss.backward()
        raw = parameter.grad[coordinate].item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
        clipped = parameter.grad[coordinate].item()
        multiplier = min(1., 1. / (norm + 1e-6))
        check(name + "_gradient_and_clipping_exact", raw == logged["raw_gradient"] and norm == logged["gradient_norm_before_clipping"] and clipped == logged["clipped_gradient"] and multiplier == logged["clip_multiplier"])
        check(name + "_clipping_equation", math.isclose(clipped, raw * multiplier, rel_tol=2e-6, abs_tol=2e-10), clipped - raw * multiplier)
        optimizer.step()
        observed_after = observe(model, minibatch, probe, panel_batches, stoi)
        m_after = moments(optimizer.state[parameter], coordinate)
        check(name + "_after_observation_exact", same(observed_after, logged["after"]))
        check(name + "_after_moments_exact", same(m_after, logged["moments_after"]))
        check(name + "_entire_model_exact_update_replay", same(model.state_dict(), after["model"]))
        check(name + "_entire_optimizer_exact_update_replay", same(optimizer.state_dict(), after["optimizer"]))
        check(name + "_rng_exact_update_replay", same(torch.get_rng_state(), after["torch_rng_state"]))
        lr = optimizer.param_groups[0]["lr"]
        progress = (999 - 100) / (3000 - 100)
        expected_lr = run["peak_lr"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        check(name + "_update_learning_rate", lr == logged["learning_rate"] == expected_lr)
        expected_m = .9 * m_before["m"] + .1 * clipped
        expected_v = .95 * m_before["v"] + .05 * clipped * clipped
        check(name + "_moment_equations", m_before["step"] == 999 and m_after["step"] == 1000 and math.isclose(m_after["m"], expected_m, rel_tol=2e-6, abs_tol=2e-10) and math.isclose(m_after["v"], expected_v, rel_tol=2e-6, abs_tol=2e-12), {"m_error": m_after["m"] - expected_m, "v_error": m_after["v"] - expected_v})
        decay_delta = -lr * .01 * observed_before["parameter"]
        adaptive_delta = -lr * (m_after["m"] / (1 - .9 ** 1000)) / (math.sqrt(m_after["v"] / (1 - .95 ** 1000)) + 1e-8)
        actual_delta = observed_after["parameter"] - observed_before["parameter"]
        reconstruction_error = actual_delta - decay_delta - adaptive_delta
        check(name + "_adamw_equation", abs(reconstruction_error) < 1e-7, reconstruction_error)
        check(name + "_logged_adamw_components", logged["weight_decay_delta"] == decay_delta and logged["adaptive_delta"] == adaptive_delta and logged["actual_delta"] == actual_delta and logged["predicted_delta_float64"] == decay_delta + adaptive_delta)
        final = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        check(name + "_final_model_optimizer_finite", finite(final["model"]) and finite(final["optimizer"]))
        check(name + "_final_optimizer_steps", final["completed_steps"] == 3000 and all(int(s["step"].item()) == 3000 for s in final["optimizer"]["state"].values()))
        print(f"Replayed {name}: entire update model and optimizer checked", flush=True)

    winner_path = study / "runs" / control["instrumented_run"]
    control_path = study / "runs" / control["control_run"]
    instrumented_history, control_history = (read_json(p / "history.json") for p in (winner_path, control_path))
    check("selected_control_all_13_recorded_validation_points_exact", instrumented_history == control_history and [r["step"] for r in instrumented_history] == list(range(0, 3001, 250)))
    winner_final, control_final = (torch.load(p / "model.pt", map_location="cpu", weights_only=True) for p in (winner_path, control_path))
    for key in ("model", "optimizer", "torch_rng_state"):
        check("selected_control_final_" + key + "_exact", same(winner_final[key], control_final[key]))
    check("selected_control_optimizer_finite", finite(control_final["optimizer"]))

    suite = load_suite()
    for seed in (42, 43, 44):
        initial = torch.load(study / f"initial_seed{seed}.pt", map_location="cpu", weights_only=True)
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            seeded_model = GPT(GPTConfig(**initial["model_args"]))
        check(f"seed{seed}_saved_initial_matches_seed", same(seeded_model.state_dict(), initial["model"]))
        model = restore_model(initial)
        identity = model_hash(model)
        rows = evaluate_rows(model, initial["vocabulary"], suite)
        path = study / "untrained_evals" / f"seed{seed}"
        check(f"seed{seed}_untrained_rows_exact_replay", rows == read_json(path / "eval_results.json"))
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        check(f"seed{seed}_untrained_csv_exact_replay", csv_buffer.getvalue().encode() == (path / "eval_results.csv").read_bytes())
        summary = read_json(path / "eval_summary.json")
        expected_groups = {g: summarize([r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})}
        expected_categories = {g: summarize([r for r in rows if r["category"] == g]) for g in sorted({r["category"] for r in rows})}
        check(f"seed{seed}_untrained_summary_exact_metrics", summary["overall"] == summarize(rows) and summary["by_group"] == expected_groups and summary["by_category"] == expected_categories)
        check(f"seed{seed}_untrained_identity_and_settings", summary["model_sha256"] == identity == model_hash(model) and summary["suite_sha256"] == suite_hash(suite) and summary["stage"] == "untrained" and summary["settings"] == {"seed": 2026, "temperature": .8, "max_tokens": 24} and read_json(path / "eval_cases.json") == suite)
        print(f"Replayed seed {seed} untrained evaluations in memory", flush=True)

    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "study": str(study.relative_to(ROOT)),
              "scope": "Independent one-update replay for every instrumented run; exact selected-control final model, optimizer and RNG equality plus all 13 recorded validation points. Intermediate weights at every training step were not saved or compared.",
              "update_replays": len(runs), "untrained_evaluation_replays": 3,
              "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks), "checks": checks}
    (study / "update_replay_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Update replay audit: {result['passed']} passed, {result['failed']} failed.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = audit(args.study.resolve())
    if result["failed"]:
        raise SystemExit(1)
