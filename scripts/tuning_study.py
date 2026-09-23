"""Controlled supplemental study; original experiments are read-only inputs.

The notebook calls prepare, sweep, select, confirm, evaluate, verify, and report
in that order. Configuration selection never reads language-evaluation scores.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanogpt_model import GPT, GPTConfig
from run_evals import evaluate_suite, load_model, load_suite, model_hash, word_tokens

BASE = ROOT / "llm_runs/20260923T062639_403952Z"
GROUPS = ("starter", "grammar", "contrasts")
STEPS = 3000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def config_id(sampler, lr):
    return f"{sampler}_lr{lr:g}"


def lr_at(peak, zero_step):
    progress = max(0, zero_step - 100) / (STEPS - 100)
    return peak * min(1, (zero_step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


class Study:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.runs = []
        self.selection = None
        torch.set_num_threads(4)
        self.saved = torch.load(BASE / "model_untrained.pt", map_location="cpu", weights_only=True)
        self.vocab = self.saved["vocabulary"]
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.split = read_json(BASE / "split.json")
        self.train = self.split["train"]
        self.val = self.split["validation"]
        self.encoded = {d: [1] + [self.stoi.get(t, 0) for t in word_tokens(d)] + [2]
                        for d in self.train + self.val}

    def prepare(self):
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        self.originals = {p: digest(ROOT / p) for p in tracked if p and (ROOT / p).is_file()}
        write_json(self.output / "original_files.json", self.originals)
        self.source_sets = {}
        for group, filename in [("grammar", "grammar_practice.txt"), ("contrasts", "contextual_contrasts.txt")]:
            lines = (ROOT / "corpus/expanded" / filename).read_text().splitlines()
            self.source_sets[group] = {" ".join(word_tokens(line)) for line in lines if line.strip()}
        assert self.source_sets["grammar"].isdisjoint(self.source_sets["contrasts"])
        self.group = {d: next((g for g in ("grammar", "contrasts") if d in self.source_sets[g]), "starter")
                      for d in self.train + self.val}
        self.train_indices = {g: [i for i, d in enumerate(self.train) if self.group[d] == g] for g in GROUPS}
        rng = random.Random(20260923)
        self.panels = {g: rng.sample(sorted(d for d in self.val if self.group[d] == g), n)
                       for g, n in zip(GROUPS, (10, 5, 5))}
        assert set(self.train).isdisjoint(self.val)
        self.schedules, self.probes, schedule_records = {}, {}, {}
        for sampler in ("uniform", "balanced"):
            rng = random.Random(43)  # Fixed data RNG for every initialization seed.
            schedule = []
            for _ in range(STEPS):
                if sampler == "uniform":
                    indices = rng.choices(range(len(self.train)), k=32)
                else:
                    indices = sum((rng.choices(self.train_indices[g], k=n)
                                   for g, n in zip(GROUPS, (16, 8, 8))), [])
                    rng.shuffle(indices)
                schedule.append(indices)
            self.schedules[sampler] = schedule
            at_update = schedule[999]
            index = next(i for i in at_update if self.group[self.train[i]] == "grammar")
            passage = self.train[index]
            tokens = word_tokens(passage)
            probe = {"update": 1000, "train_index": index, "passage": passage,
                     "prefix": " ".join(tokens[:2]), "target": tokens[2],
                     "token_id": self.stoi[tokens[2]], "coordinate": 0,
                     "parameter": "transformer.wte.weight (tied to lm_head.weight)",
                     "rule": "first grammar passage in the predetermined update-1000 batch; third token; coordinate 0"}
            self.probes[sampler] = probe
            write_json(self.output / f"batches_{sampler}.json", schedule)
            schedule_records[sampler] = {"sha256": object_hash(schedule), "probe": probe}
        self.frozen = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_run": str(BASE.relative_to(ROOT)),
            "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [
                BASE / "corpus.txt", BASE / "split.json", BASE / "tokenization.json",
                BASE / "model_untrained.pt", ROOT / "nanogpt_model.py", ROOT / "run_evals.py",
                ROOT / "evals/language_evals.json", ROOT / "corpus/expanded/grammar_practice.txt",
                ROOT / "corpus/expanded/contextual_contrasts.txt"]},
            "training_script_sha256": digest(Path(__file__)),
            "vocabulary_sha256": object_hash(self.vocab), "model_args": self.saved["model_args"],
            "panels": {"original": self.split["evaluation_validation"], **self.panels},
            "training_group_counts": {g: len(self.train_indices[g]) for g in GROUPS},
            "validation_group_counts": {g: sum(self.group[d] == g for d in self.val) for g in GROUPS},
            "schedules": schedule_records, "steps": STEPS, "batch_size": 32,
            "learning_rates": [.0003, .001, .003], "samplers": ["uniform", "balanced"],
            "optimizer": {"name": "AdamW", "betas": [.9, .95], "weight_decay": .01, "eps": 1e-8,
                          "clip_norm": 1.0, "warmup_steps": 100, "minimum_lr_fraction": .1},
            "selection": "final category-macro validation NLL, original-panel NLL <= 1.05 times matched baseline; ties favor baseline then lower LR then uniform",
            "confirmation_seeds": [43, 44], "data_rng_seed": 43,
            "evaluation_policy": "language suite is loaded only after selection and confirmation",
            "hypothesis": "balanced exposure and learning-rate tuning may improve category loss; benchmark gains are uncertain",
        }
        write_json(self.output / "protocol.json", self.frozen)
        (self.output / "training_source.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({"training_groups": self.frozen["training_group_counts"],
                          "frozen_panels": {g: len(p) for g, p in self.frozen["panels"].items()},
                          "preselected_probes": self.probes}, indent=2))

    def batch(self, documents):
        sequences = [self.encoded[d] for d in documents]
        length = max(len(s) - 1 for s in sequences)
        x = torch.full((len(sequences), length), 2, dtype=torch.long)
        y = torch.full_like(x, -1)
        for i, seq in enumerate(sequences):
            x[i, :len(seq) - 1] = torch.tensor(seq[:-1])
            y[i, :len(seq) - 1] = torch.tensor(seq[1:])
        return x, y

    @torch.no_grad()
    def metrics(self, model):
        was_training = model.training
        model.eval()
        losses = {g: model(*self.batch(docs))[1].item() for g, docs in self.frozen["panels"].items()}
        losses["macro"] = statistics.mean(losses[g] for g in GROUPS)
        model.train(was_training)
        return losses

    @torch.no_grad()
    def probe_metrics(self, model, batch, probe):
        was_training = model.training
        model.eval()
        ids = [1] + [self.stoi[t] for t in word_tokens(probe["prefix"])]
        probability = model(torch.tensor([ids]))[0][0, -1].softmax(-1)[probe["token_id"]].item()
        result = {"batch_loss": model(*batch)[1].item(), "target_probability": probability,
                  "validation": self.metrics(model)}
        model.train(was_training)
        return result

    def checkpoint(self, model, **extra):
        return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "model_args": self.saved["model_args"], "vocabulary": self.vocab, **extra}

    def train_run(self, sampler, lr, seed, instrument=True, purpose="screen"):
        cid = config_id(sampler, lr)
        rid = f"{cid}_seed{seed}" + ("_no_instrumentation" if not instrument else "")
        directory = self.output / rid
        directory.mkdir(exist_ok=False)
        torch.manual_seed(seed)
        model = GPT(GPTConfig(**self.saved["model_args"]))
        if seed == 42:
            model.load_state_dict(self.saved["model"])
        initial_hash = model_hash(model)
        torch.save(self.checkpoint(model, completed_steps=0), directory / "model_untrained.pt")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=.01)
        history = [{"step": 0, **self.metrics(model)}]
        probe = self.probes[sampler]
        row = probe["token_id"]
        param = model.transformer.wte.weight
        peak_norm, evidence = 0., None
        started = time.perf_counter()
        for zero_step, indices in enumerate(self.schedules[sampler]):
            effective_lr = lr_at(lr, zero_step)
            for group in optimizer.param_groups:
                group["lr"] = effective_lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            xy = self.batch([self.train[i] for i in indices])
            _, loss = model(*xy)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{rid}: nonfinite loss at update {zero_step + 1}")
            loss.backward()
            if instrument and zero_step == 999:
                evidence = {"probe": probe, "batch_indices": indices, "batch_documents": [self.train[i] for i in indices],
                            "learning_rate": effective_lr, "before": self.probe_metrics(model, xy, probe),
                            "embedding_before": param[row].detach().tolist(),
                            "raw_gradient": param.grad[row].detach().tolist(),
                            "moments_before": {k: optimizer.state[param][k][row].tolist() for k in ("exp_avg", "exp_avg_sq")},
                            "optimizer_step_before": optimizer.state[param]["step"].item()}
                torch.save(self.checkpoint(model, completed_steps=999, optimizer=optimizer.state_dict()),
                           directory / "update_1000_before.pt")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
            peak_norm = max(peak_norm, norm)
            if instrument and zero_step == 999:
                evidence["gradient_norm_before_clipping"] = norm
                evidence["clipping_scale"] = min(1., 1. / (norm + 1e-6))
                evidence["clipped_gradient"] = param.grad[row].detach().tolist()
            optimizer.step()
            if instrument and zero_step == 999:
                evidence.update({"after": self.probe_metrics(model, xy, probe),
                                 "embedding_after": param[row].detach().tolist(),
                                 "moments_after": {k: optimizer.state[param][k][row].tolist() for k in ("exp_avg", "exp_avg_sq")},
                                 "optimizer_step_after": optimizer.state[param]["step"].item(),
                                 "model_hash_after_update": model_hash(model)})
                evidence["coordinate_change"] = evidence["embedding_after"][0] - evidence["embedding_before"][0]
                m = evidence["moments_after"]["exp_avg"][0] / (1 - .9 ** 1000)
                v = evidence["moments_after"]["exp_avg_sq"][0] / (1 - .95 ** 1000)
                evidence["adamw_formula_delta_float64"] = -effective_lr * .01 * evidence["embedding_before"][0] - effective_lr * m / (math.sqrt(v) + 1e-8)
                write_json(directory / "gradient_update.json", evidence)
            if (zero_step + 1) % 250 == 0:
                history.append({"step": zero_step + 1, **self.metrics(model)})
        elapsed = time.perf_counter() - started
        torch.save(self.checkpoint(model, completed_steps=STEPS), directory / "model.pt")
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        write_json(directory / "history.json", history)
        result = {"run_id": rid, "config_id": cid, "sampler": sampler, "learning_rate": lr,
                  "seed": seed, "purpose": purpose, "instrumented": instrument,
                  "completed_steps": STEPS, "elapsed_seconds": elapsed,
                  "initial_model_sha256": initial_hash, "final_model_sha256": model_hash(model),
                  "schedule_sha256": object_hash(self.schedules[sampler]),
                  "protocol_sha256": digest(self.output / "protocol.json"),
                  "peak_gradient_norm": peak_norm, "final": history[-1],
                  "all_loss_and_gradient_norm_checks_finite": True}
        write_json(directory / "run.json", result)
        self.runs.append(result)
        write_json(self.output / "ledger.json", self.runs)
        print(f"{rid}: macro={result['final']['macro']:.6f}, original={result['final']['original']:.6f}, {elapsed:.1f}s", flush=True)
        return result

    def sweep(self):
        for sampler in ("uniform", "balanced"):
            for lr in (.0003, .001, .003):
                self.train_run(sampler, lr, 42)

    def select(self):
        baseline = next(r for r in self.runs if r["config_id"] == "uniform_lr0.001")
        limit = 1.05 * baseline["final"]["original"]
        eligible = [r for r in self.runs if r["final"]["original"] <= limit]
        selected = min(eligible, key=lambda r: (r["final"]["macro"], r["config_id"] != baseline["config_id"],
                                               r["learning_rate"], r["sampler"] != "uniform"))
        self.selection = {"selected_config": selected["config_id"], "sampler": selected["sampler"],
                          "learning_rate": selected["learning_rate"], "baseline_config": baseline["config_id"],
                          "original_panel_limit": limit, "eligible": [r["config_id"] for r in eligible],
                          "selected_utc": datetime.now(timezone.utc).isoformat(),
                          "basis": "final held-out category-macro loss with original-panel guardrail; no suite scores read"}
        write_json(self.output / "selection.json", self.selection)
        print(json.dumps(self.selection, indent=2))
        return self.selection

    def confirm(self):
        assert self.selection
        for seed in (43, 44):
            self.train_run("uniform", .001, seed, purpose="confirmation")
            if self.selection["selected_config"] != "uniform_lr0.001":
                self.train_run(self.selection["sampler"], self.selection["learning_rate"], seed, purpose="confirmation")
        control = self.train_run(self.selection["sampler"], self.selection["learning_rate"], 42,
                                 instrument=False, purpose="instrumentation_control")
        selected = next(r for r in self.runs if r["config_id"] == self.selection["selected_config"] and r["seed"] == 42 and r["instrumented"])
        self.instrumentation = {"instrumented_run": selected["run_id"], "control_run": control["run_id"],
                                "identical_final_weights": control["final_model_sha256"] == selected["final_model_sha256"],
                                "identical_history": read_json(self.output / control["run_id"] / "history.json") == read_json(self.output / selected["run_id"] / "history.json")}
        write_json(self.output / "instrumentation_check.json", self.instrumentation)
        assert all(self.instrumentation[k] for k in ("identical_final_weights", "identical_history"))
        print(json.dumps(self.instrumentation, indent=2))

    def evaluate(self):
        assert (self.output / "selection.json").exists() and self.instrumentation["identical_final_weights"]
        suite = load_suite()
        write_json(self.output / "evaluation_start.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                                         "selection_sha256": digest(self.output / "selection.json")})
        for run in self.runs:
            if not run["instrumented"]:
                continue
            directory = self.output / run["run_id"]
            model, vocabulary, _ = load_model(directory / "model.pt")
            summary = evaluate_suite(model, vocabulary, suite, directory / "language_evals", stage="final")
            run["evaluation"] = summary["overall"]
            print(f"{run['run_id']}: {summary['overall']['correct']}/48; coverage {summary['overall']['scorable']}/48")
        self.pairs = []
        for seed in (42, 43, 44):
            baseline = next(r for r in self.runs if r["config_id"] == "uniform_lr0.001" and r["seed"] == seed and r["instrumented"])
            selected = next(r for r in self.runs if r["config_id"] == self.selection["selected_config"] and r["seed"] == seed and r["instrumented"])
            b_cases = read_json(self.output / baseline["run_id"] / "language_evals/eval_results.json")
            s_cases = read_json(self.output / selected["run_id"] / "language_evals/eval_results.json")
            self.pairs.append({"seed": seed, "baseline": baseline["run_id"], "selected": selected["run_id"],
                               "baseline_correct": baseline["evaluation"]["correct"], "selected_correct": selected["evaluation"]["correct"],
                               "macro_loss_change": selected["final"]["macro"] - baseline["final"]["macro"],
                               "original_loss_change": selected["final"]["original"] - baseline["final"]["original"],
                               "guardrail_met": selected["final"]["original"] <= baseline["final"]["original"] * 1.05,
                               "gains": [s["id"] for b, s in zip(b_cases, s_cases) if s["score"] > b["score"]],
                               "regressions": [s["id"] for b, s in zip(b_cases, s_cases) if s["score"] < b["score"]]})
        write_json(self.output / "paired_results.json", self.pairs)
        write_json(self.output / "ledger.json", self.runs)
        print(json.dumps(self.pairs, indent=2))

    def verify(self):
        checks = []
        def check(name, condition, detail=None):
            checks.append({"check": name, "passed": bool(condition), "detail": detail})
            if not condition:
                write_json(self.output / "audit.json", checks)
                raise AssertionError(name)
        check("all_original_tracked_files_unchanged_before_report", all(digest(ROOT / p) == h for p, h in self.originals.items()))
        check("fixed_input_hashes", all(digest(ROOT / p) == h for p, h in self.frozen["input_hashes"].items()))
        check("train_validation_disjoint", set(self.train).isdisjoint(self.val))
        check("balanced_panel_is_held_out", all(d in self.val for panel in self.panels.values() for d in panel))
        from collections import Counter
        counts = Counter(t for d in self.train for t in word_tokens(d))
        retained = sorted((t for t in counts if len(t) <= 128), key=lambda t: (-counts[t], t))[:509]
        check("vocabulary_reconstructed_from_training_only", ["<UNK>", "<BOS>", "<EOS>"] + sorted(retained) == self.vocab)
        for seed in (42, 43, 44):
            runs = [r for r in self.runs if r["seed"] == seed]
            check(f"seed_{seed}_identical_initial_weights", len({r["initial_model_sha256"] for r in runs}) == 1)
        check("matched_minibatches_within_sampler", all(r["schedule_sha256"] == object_hash(self.schedules[r["sampler"]]) for r in self.runs))
        check("sampling_schedules_use_training_indices", all(0 <= i < len(self.train) for schedule in self.schedules.values() for batch in schedule for i in batch))
        check("balanced_batches_exact_16_8_8", all([sum(self.group[self.train[i]] == g for i in batch) for g in GROUPS] == [16, 8, 8] for batch in self.schedules["balanced"]))
        original, _, _ = load_model(BASE / "model.pt")
        replicated = next(r for r in self.runs if r["config_id"] == "uniform_lr0.001" and r["seed"] == 42)
        check("original_expanded_baseline_reproduced_bit_for_bit", model_hash(original) == replicated["final_model_sha256"])
        check("instrumentation_preserves_weights_and_history", self.instrumentation["identical_final_weights"] and self.instrumentation["identical_history"])
        check("selection_precedes_suite_evaluation", self.selection["selected_utc"] < read_json(self.output / "evaluation_start.json")["started_utc"])
        suite = load_suite()
        for run in self.runs:
            directory = self.output / run["run_id"]
            model, vocabulary, saved = load_model(directory / "model.pt")
            check(f"{run['run_id']}.checkpoint", model_hash(model) == run["final_model_sha256"] and vocabulary == self.vocab and saved["completed_steps"] == STEPS)
            opt = torch.load(directory / "optimizer.pt", weights_only=True)
            check(f"{run['run_id']}.optimizer", all(v["step"].item() == STEPS and torch.isfinite(v["exp_avg"]).all() and torch.isfinite(v["exp_avg_sq"]).all() for v in opt["state"].values()))
            if not run["instrumented"]:
                continue
            with tempfile.TemporaryDirectory() as temp:
                evaluate_suite(model, vocabulary, suite, temp, stage="final")
                check(f"{run['run_id']}.eval_replay", read_json(Path(temp) / "eval_results.json") == read_json(directory / "language_evals/eval_results.json"))
            evidence = read_json(directory / "gradient_update.json")
            pre, _, saved = load_model(directory / "update_1000_before.pt")
            opt = torch.optim.AdamW(pre.parameters(), lr=run["learning_rate"], betas=(.9, .95), weight_decay=.01)
            opt.load_state_dict(saved["optimizer"])
            xy = self.batch(evidence["batch_documents"])
            check(f"{run['run_id']}.gradient_before_metrics", self.probe_metrics(pre, xy, evidence["probe"]) == evidence["before"])
            pre.train()
            opt.zero_grad(set_to_none=True)
            pre(*xy)[1].backward()
            row = evidence["probe"]["token_id"]
            check(f"{run['run_id']}.raw_gradient", pre.transformer.wte.weight.grad[row].tolist() == evidence["raw_gradient"])
            torch.nn.utils.clip_grad_norm_(pre.parameters(), 1., error_if_nonfinite=True)
            check(f"{run['run_id']}.clipped_gradient", pre.transformer.wte.weight.grad[row].tolist() == evidence["clipped_gradient"])
            opt.step()
            check(f"{run['run_id']}.update_replay", model_hash(pre) == evidence["model_hash_after_update"] and self.probe_metrics(pre, xy, evidence["probe"]) == evidence["after"])
            check(f"{run['run_id']}.adamw_equation", abs(evidence["coordinate_change"] - evidence["adamw_formula_delta_float64"]) < 2e-7,
                  {"actual": evidence["coordinate_change"], "formula": evidence["adamw_formula_delta_float64"], "tolerance": 2e-7})
        write_json(self.output / "audit.json", checks)
        print(f"{len(checks)} verification checks passed, including every checkpoint evaluation and every recorded optimizer update.")
        return checks

    def report(self):
        rows = []
        for r in self.runs:
            if r["instrumented"]:
                rows.append({"run": r["run_id"], "seed": r["seed"], "sampler": r["sampler"], "peak_lr": r["learning_rate"],
                             **{f"loss_{k}": r["final"][k] for k in ("original", *GROUPS, "macro")},
                             "correct": r["evaluation"]["correct"], "scorable": r["evaluation"]["scorable"]})
        with (self.output / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        self.draw_curves()
        selected = next(r for r in self.runs if r["config_id"] == self.selection["selected_config"] and r["seed"] == 42 and r["instrumented"])
        evidence = read_json(self.output / selected["run_id"] / "gradient_update.json")
        lines = ["# Controlled tuning study", "", "The expanded corpus, vocabulary, split, and architecture were held fixed. Six configurations were screened at seed 42. Selection used final category-macro validation loss with a 5% original-panel guardrail. The chosen setting and baseline were repeated at seeds 43 and 44 before loading the language suite.", "",
                 f"Selected configuration: **{self.selection['selected_config']}**. The three category losses each average non-padding token losses within their panel, then receive equal weight. The balanced training batches contain exactly 16 starter, eight grammar, and eight contrast passages. Token counts vary by passage length.", "",
                 "| Sampler | Peak LR | Original loss | Grammar loss | Contrast loss | Macro loss | Eligible | Correct / 48 |",
                 "|---|---:|---:|---:|---:|---:|---|---:|"]
        for r in self.runs:
            if r["seed"] == 42 and r["instrumented"]:
                f = r["final"]
                lines.append(f"| {r['sampler']} | {r['learning_rate']:g} | {f['original']:.6f} | {f['grammar']:.6f} | {f['contrasts']:.6f} | {f['macro']:.6f} | {'Yes' if r['config_id'] in self.selection['eligible'] else 'No'} | {r['evaluation']['correct']} |")
        lines += ["", "## Paired confirmation", "", "| Seed | Baseline correct | Selected correct | Macro loss change | Original loss change | Guardrail | Gains | Regressions |", "|---|---:|---:|---:|---:|---|---|---|"]
        for p in self.pairs:
            lines.append(f"| {p['seed']} | {p['baseline_correct']} | {p['selected_correct']} | {p['macro_loss_change']:+.6f} | {p['original_loss_change']:+.6f} | {'Pass' if p['guardrail_met'] else 'Fail'} | {', '.join(p['gains']) or 'None'} | {', '.join(p['regressions']) or 'None'} |")
        bmean = statistics.mean(p["baseline_correct"] for p in self.pairs)
        smean = statistics.mean(p["selected_correct"] for p in self.pairs)
        lines += ["", f"Mean correct answers: baseline **{bmean:.2f}/48**, selected **{smean:.2f}/48**. Mean paired macro-loss change: **{statistics.mean(p['macro_loss_change'] for p in self.pairs):+.6f}**. Coverage remains **27/48** in every run. Negative loss changes indicate improvement. Seeds 43 and 44 are confirmation runs; seed 42 was used for selection.", "",
                  "## A real update at step 1,000", "", f"Preselected training passage: `{evidence['probe']['passage']}`. Prefix: `{evidence['probe']['prefix']}`. Target: `{evidence['probe']['target']}`. Tracked parameter: embedding row {evidence['probe']['token_id']}, coordinate 0. Input/output embeddings share this weight.", "",
                  "| Measurement | Before | After |", "|---|---:|---:|"]
        for label, before, after in [("Batch loss", evidence["before"]["batch_loss"], evidence["after"]["batch_loss"]),
                                     ("Target probability", evidence["before"]["target_probability"], evidence["after"]["target_probability"]),
                                     ("Original validation loss", evidence["before"]["validation"]["original"], evidence["after"]["validation"]["original"]),
                                     ("Category-macro validation loss", evidence["before"]["validation"]["macro"], evidence["after"]["validation"]["macro"]),
                                     ("Embedding coordinate", evidence["embedding_before"][0], evidence["embedding_after"][0])]:
            lines.append(f"| {label} | {before:.10g} | {after:.10g} |")
        lines += ["", f"- Actual learning rate: `{evidence['learning_rate']:.12g}`; raw gradient: `{evidence['raw_gradient'][0]:.12g}`; clipped gradient: `{evidence['clipped_gradient'][0]:.12g}`.",
                  f"- Gradient norm: `{evidence['gradient_norm_before_clipping']:.12g}`; clipping multiplier: `{evidence['clipping_scale']:.12g}`; actual coordinate change: `{evidence['coordinate_change']:.12g}`.",
                  f"- Adam first moment: `{evidence['moments_before']['exp_avg'][0]:.12g}` → `{evidence['moments_after']['exp_avg'][0]:.12g}`; second moment: `{evidence['moments_before']['exp_avg_sq'][0]:.12g}` → `{evidence['moments_after']['exp_avg_sq'][0]:.12g}`.", "",
                  "AdamW carries moving averages of gradients and squared gradients. Bias correction and adaptive scaling determine the gradient contribution, while weight decay shrinks the existing weight. The float64 reconstruction below is compared with the actual float32 optimizer update; the complete checkpoint replay matches every resulting weight exactly. [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).", "",
                  "```text", "m_hat = m_after / (1 - 0.9^1000)", "v_hat = v_after / (1 - 0.95^1000)", "delta = -lr * 0.01 * weight_before - lr * m_hat / (sqrt(v_hat) + 1e-8)", f"formula delta = {evidence['adamw_formula_delta_float64']:.12g}", f"actual delta  = {evidence['coordinate_change']:.12g}", "```", "",
                  "All weights were updated together. The before/after measurements describe that whole optimizer step. A single coordinate cannot establish the cause of the prediction change. The passage, token and coordinate were fixed before training; every run retains its update evidence, including unfavorable changes.", "",
                  "## Learning curves", "", "![Validation curves](curves.svg)", "", "## Verification and limits", "",
                  f"- {len(read_json(self.output / 'audit.json'))} checks passed. The original expanded baseline was reproduced bit for bit. An uninstrumented selected run reproduced both final weights and the complete loss history.",
                  "- Every saved final checkpoint was reevaluated and each recorded update was reconstructed from saved weights, optimizer moments, and its actual minibatch.",
                  "- Confidence is high in the saved measurements. Three seeds and small, template-sharing validation panels support limited conclusions about this corpus. Confidence in broader language transfer remains low.",
                  "- The 48 public cases are a development benchmark. Their scores were read after selection. Vocabulary limits scorable cases to 27; all other cases receive zero credit.",
                  "- The original five article errors remain in the fixed corpus. Source grouping and passage-level balance do not equal token-level balance.", "", "## Next improvements", "",
                  "1. Broaden vocabulary using independent teaching material; measure coverage separately from accuracy.",
                  "2. Correct article errors and diversify grammar and contrast constructions in a new corpus version.",
                  "3. Hold out entire template families before further tuning to measure transfer beyond shared templates.",
                  "4. Extend training only where late validation curves continue to improve; preserve a separate selection set.",
                  "5. Consider model size or tokenization changes after these data and optimization checks, within assignment rules.", "", "## Reproduce", "",
                  "Run `custom_llm_tuning.ipynb` from the repository root. It creates a new timestamped study directory, freezes its protocol, executes the sweep and confirmations, evaluates all candidates, and verifies the saved evidence. The original experiment ledger and artifacts are retained.", ""]
        (self.output / "REPORT.md").write_text("\n".join(lines))
        print(f"Report: {self.output / 'REPORT.md'}")
        return {"selected": self.selection["selected_config"], "mean_baseline_correct": bmean,
                "mean_selected_correct": smean, "paired_results": self.pairs}

    def draw_curves(self):
        palette = ["#2563eb", "#0891b2", "#7c3aed", "#dc2626", "#ea580c", "#16a34a"]
        runs = [r for r in self.runs if r["seed"] == 42 and r["instrumented"]]
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" viewBox="0 0 1000 680">',
               '<rect width="1000" height="680" fill="white"/>', '<g font-family="Arial,sans-serif" fill="#172033">',
               '<text x="40" y="30" font-size="22">Fixed validation panels: six controlled configurations</text>']
        for panel, metric, label in [(0, "macro", "Category-macro loss"), (1, "original", "Original-panel loss")]:
            left, top, width, height = 70 + panel * 490, 85, 390, 390
            histories = [read_json(self.output / r["run_id"] / "history.json") for r in runs]
            # Show post-warmup region so differences remain readable. Step 0 is in JSON.
            values = [h[metric] for hist in histories for h in hist if h["step"] >= 250]
            lo, hi = min(values), max(values)
            pad = max((hi - lo) * .08, .01); lo = max(0, lo - pad); hi += pad
            svg.append(f'<text x="{left}" y="{top-20}" font-size="17">{label}</text>')
            for tick in range(5):
                value = lo + (hi - lo) * tick / 4
                y = top + height - height * tick / 4
                svg += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#e2e8f0"/>',
                        f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{value:.3f}</text>']
            for step in (250, 1000, 2000, 3000):
                x = left + (step - 250) / 2750 * width
                svg.append(f'<text x="{x}" y="{top+height+22}" text-anchor="middle" font-size="11">{step}</text>')
            for i, hist in enumerate(histories):
                points = " ".join(f"{left+(h['step']-250)/2750*width:.2f},{top+height-(h[metric]-lo)/(hi-lo)*height:.2f}" for h in hist if h["step"] >= 250)
                svg.append(f'<polyline points="{points}" fill="none" stroke="{palette[i]}" stroke-width="2"/>')
            svg.append(f'<text x="{left+width/2}" y="{top+height+45}" text-anchor="middle" font-size="13">Optimizer updates</text>')
        for i, run in enumerate(runs):
            x, y = 70 + (i % 3) * 310, 575 + (i // 3) * 30
            svg += [f'<line x1="{x}" y1="{y-4}" x2="{x+24}" y2="{y-4}" stroke="{palette[i]}" stroke-width="3"/>',
                    f'<text x="{x+32}" y="{y}" font-size="13">{run["sampler"]}, LR {run["learning_rate"]:g}</text>']
        svg += ['<text x="70" y="650" font-size="12">Seed 42. Curves start at step 250; complete histories include step 0. Lower loss indicates better fit.</text>', '</g></svg>']
        (self.output / "curves.svg").write_text("\n".join(svg))
