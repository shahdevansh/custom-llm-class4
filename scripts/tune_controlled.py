"""Controlled supplement: frozen expanded data, six settings, paired seeds.

Training never reads the language evaluation suite. Selection uses frozen corpus
validation panels. The original course experiments and code remain unchanged.
"""
from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanogpt_model import GPT, GPTConfig
from run_evals import word_tokens, model_hash, load_model, evaluate_suite, load_suite

BASE = ROOT / "llm_runs/20260923T062639_403952Z"
GROUPS = ("starter", "grammar", "contrasts")
STEPS = 3000


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def quiet_model(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(GPTConfig(**args))


def same_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


class Study:
    def __init__(self, output):
        self.out = Path(output).resolve()
        if self.out.exists():
            raise FileExistsError(f"Choose a fresh study directory: {self.out}")
        self.out.mkdir(parents=True)
        torch.set_num_threads(4)
        self.split = json.loads((BASE / "split.json").read_text())
        self.saved = torch.load(BASE / "model_untrained.pt", weights_only=True)
        self.vocab = self.saved["vocabulary"]
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.args = self.saved["model_args"]
        self.train = self.split["train"]
        self.val = self.split["validation"]
        sources = {}
        for group, filename in [("grammar", "grammar_practice.txt"), ("contrasts", "contextual_contrasts.txt")]:
            text = (ROOT / "corpus/expanded" / filename).read_text()
            sources[group] = {" ".join(word_tokens(unit)) for unit in re.split(r"(?<=[.!?])\s+|\n+", text) if unit.strip()}
        assert sources["grammar"].isdisjoint(sources["contrasts"])
        self.category = lambda doc: next((g for g in ("grammar", "contrasts") if doc in sources[g]), "starter")
        self.pools = {g: [i for i, d in enumerate(self.train) if self.category(d) == g] for g in GROUPS}
        val_groups = {g: sorted(d for d in self.val if self.category(d) == g) for g in GROUPS}
        rng = random.Random(20260923)
        self.panels = {g: rng.sample(val_groups[g], n) for g, n in zip(GROUPS, (10, 5, 5))}
        self.panels["original"] = self.split["evaluation_validation"]
        assert set(self.train).isdisjoint(self.val)
        assert all(set(docs).issubset(self.val) for docs in self.panels.values())
        self.panel_batches = {g: self.batch(docs) for g, docs in self.panels.items()}
        self.schedules = {}
        self.initials = {}
        self.runs = []
        self.configs = [("uniform", .001), ("uniform", .0003), ("uniform", .003),
                        ("balanced", .0003), ("balanced", .001), ("balanced", .003)]
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        self.originals = {p: sha(ROOT / p) for p in tracked if p and (ROOT / p).is_file()}
        dump(self.out / "original_files.json", self.originals)
        dump(self.out / "frozen_panels.json", self.panels)
        inputs = [BASE / f for f in ("corpus.txt", "split.json", "tokenization.json", "model_untrained.pt")]
        inputs += [ROOT / f for f in ("corpus/expanded/grammar_practice.txt", "corpus/expanded/contextual_contrasts.txt", "nanogpt_model.py", "run_evals.py", "evals/language_evals.json")]
        self.input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
        self.probes = {}
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            model = quiet_model(self.args)
            if seed == 42:
                assert same_state(model.state_dict(), self.saved["model"]), "Initialization must reproduce original expanded model"
            self.initials[seed] = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({"model": self.initials[seed], "model_args": self.args, "vocabulary": self.vocab,
                        "completed_steps": 0, "seed": seed}, self.out / f"initial_seed{seed}.pt")
            for sampling in ("uniform", "balanced"):
                rng = random.Random(seed + 1)
                schedule = []
                for _ in range(STEPS):
                    if sampling == "uniform":
                        indices = rng.choices(range(len(self.train)), k=32)
                    else:
                        indices = sum((rng.choices(self.pools[g], k=n) for g, n in zip(GROUPS, (16, 8, 8))), [])
                        rng.shuffle(indices)
                    schedule.append(indices)
                self.schedules[seed, sampling] = schedule
                with gzip.open(self.out / f"batches_seed{seed}_{sampling}.json.gz", "wt") as f:
                    json.dump(schedule, f)
                # The observation is chosen before any optimizer updates or evals.
                indices = schedule[999]
                doc_index = next((i for i in indices if self.category(self.train[i]) == "grammar"), indices[0])
                doc = self.train[doc_index]
                words = word_tokens(doc)
                position = min(3, len(words) - 2)
                probe_token = next(w for w in words[:position] if w not in {"a", "an", "the", "our", "this", "these", "those"})
                self.probes[f"{seed}_{sampling}"] = {"document_index": doc_index, "passage": doc,
                    "prefix": " ".join(words[:position]), "target": words[position], "embedding_token": probe_token,
                    "token_id": self.stoi[probe_token], "coordinate": 0, "batch_document_indices": indices}
        self.plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "base_run": str(BASE.relative_to(ROOT)),
            "steps": STEPS, "batch_size": 32, "configs": self.configs, "seeds": [42, 43, 44],
            "selection": "Lowest final macro category validation NLL among seed42 configurations whose original panel NLL <= 1.05 times matched baseline; ties by run_id. Public suite evaluated only after selection and confirmation.",
            "macro_reduction": "Each category uses mean non-padding target NLL; macro is equal mean across starter, grammar, contrasts.",
            "sampling": {"uniform": "32 independent draws from all training passages", "balanced": "16 starter, 8 grammar, 8 contrasts; shuffle minibatch"},
            "train_category_counts": {g: len(self.pools[g]) for g in GROUPS},
            "validation_category_counts": {g: len(val_groups[g]) for g in GROUPS},
            "input_sha256": self.input_hashes, "panel_sha256": identity(self.panels),
            "gradient_update": 1000, "preselected_probes": self.probes,
            "optimizer": {"type": "AdamW", "betas": [.9, .95], "eps": 1e-8, "weight_decay": .01,
                          "clip_norm": 1., "warmup_steps": 100, "final_lr_fraction": .1},
            "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "threads": 4, "device": "cpu"}}
        dump(self.out / "preregistered_plan.json", self.plan)
        print("Frozen inputs and panels; training categories:", self.plan["train_category_counts"])
        print("Six configurations; final checkpoint selection precedes language evaluation.")

    def batch(self, docs):
        sequences = [[1] + [self.stoi.get(w, 0) for w in word_tokens(d)] + [2] for d in docs]
        length = max(len(s) - 1 for s in sequences)
        x = torch.full((len(sequences), length), 2, dtype=torch.long)
        y = torch.full_like(x, -1)
        for i, seq in enumerate(sequences):
            x[i, :len(seq)-1] = torch.tensor(seq[:-1])
            y[i, :len(seq)-1] = torch.tensor(seq[1:])
        return x, y

    @torch.no_grad()
    def metrics(self, model):
        model.eval()
        values = {g: model(*batch)[1].item() for g, batch in self.panel_batches.items()}
        values["macro"] = sum(values[g] for g in GROUPS) / 3
        return values

    @torch.no_grad()
    def observation(self, model, batch, probe):
        model.eval()
        ids = torch.tensor([[1] + [self.stoi[w] for w in word_tokens(probe["prefix"])]])
        probability = torch.softmax(model(ids)[0][0, -1], -1)[self.stoi[probe["target"]]].item()
        return {"batch_loss": model(*batch)[1].item(), "target_probability": probability,
                "validation": self.metrics(model),
                "parameter": model.transformer.wte.weight[probe["token_id"], 0].item()}

    def checkpoint(self, model, optimizer, seed, step):
        return {"model": model.state_dict(), "model_args": self.args, "vocabulary": self.vocab,
                "optimizer": optimizer.state_dict(), "completed_steps": step, "seed": seed,
                "torch_rng_state": torch.get_rng_state(), "input_sha256": self.input_hashes,
                "panel_sha256": identity(self.panels)}

    def train_one(self, sampling, peak_lr, seed, instrument=True, control=False):
        run_id = f"{sampling}_lr{peak_lr:g}_seed{seed}" + ("_control" if control else "")
        destination = self.out / "runs" / run_id
        destination.mkdir(parents=True)
        torch.manual_seed(seed)
        model = quiet_model(self.args)
        model.load_state_dict(self.initials[seed])
        initial_hash = model_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(.9, .95), weight_decay=.01)
        schedule = self.schedules[seed, sampling]
        probe = self.probes[f"{seed}_{sampling}"]
        coordinate = (probe["token_id"], 0)
        parameter = model.transformer.wte.weight
        history = [{"step": 0, **self.metrics(model)}]
        update = None
        started = time.perf_counter()
        max_norm = 0.
        clipped_steps = 0
        for step, indices in enumerate(schedule):
            progress = max(0, step - 100) / (STEPS - 100)
            lr = peak_lr * min(1, (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            optimizer.param_groups[0]["lr"] = lr
            minibatch = self.batch([self.train[i] for i in indices])
            if instrument and step == 999:
                update = {"update": 1000, "probe": probe, "learning_rate": lr,
                          "before": self.observation(model, minibatch, probe)}
                torch.save(self.checkpoint(model, optimizer, seed, step), destination / "update_1000_before.pt")
                state = optimizer.state[parameter]
                update["moments_before"] = {"step": int(state["step"].item()), "m": state["exp_avg"][coordinate].item(), "v": state["exp_avg_sq"][coordinate].item()}
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(*minibatch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{run_id} step {step + 1}: nonfinite loss")
            loss.backward()
            if instrument and step == 999:
                update["raw_gradient"] = parameter.grad[coordinate].item()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
            max_norm = max(max_norm, norm)
            clipped_steps += norm > 1.
            if instrument and step == 999:
                update["gradient_norm_before_clipping"] = norm
                update["clip_multiplier"] = min(1., 1. / (norm + 1e-6))
                update["clipped_gradient"] = parameter.grad[coordinate].item()
            optimizer.step()
            if instrument and step == 999:
                update["after"] = self.observation(model, minibatch, probe)
                state = optimizer.state[parameter]
                update["moments_after"] = {"step": int(state["step"].item()), "m": state["exp_avg"][coordinate].item(), "v": state["exp_avg_sq"][coordinate].item()}
                m, v = update["moments_after"]["m"], update["moments_after"]["v"]
                update["weight_decay_delta"] = -lr * .01 * update["before"]["parameter"]
                update["adaptive_delta"] = -lr * (m / (1 - .9 ** 1000)) / (math.sqrt(v / (1 - .95 ** 1000)) + 1e-8)
                update["predicted_delta_float64"] = update["weight_decay_delta"] + update["adaptive_delta"]
                update["actual_delta"] = update["after"]["parameter"] - update["before"]["parameter"]
                update["reconstruction_error"] = update["actual_delta"] - update["predicted_delta_float64"]
                assert abs(update["reconstruction_error"]) < 1e-7
                torch.save(self.checkpoint(model, optimizer, seed, step + 1), destination / "update_1000_after.pt")
                dump(destination / "gradient_update.json", update)
            if (step + 1) % 250 == 0:
                history.append({"step": step + 1, **self.metrics(model)})
        elapsed = time.perf_counter() - started
        torch.save(self.checkpoint(model, optimizer, seed, STEPS), destination / "model.pt")
        dump(destination / "history.json", history)
        result = {"run_id": run_id, "sampling": sampling, "peak_lr": peak_lr, "seed": seed,
                  "initial_model_sha256": initial_hash, "model_sha256": model_hash(model),
                  "schedule_sha256": identity(schedule), "input_sha256": self.input_hashes,
                  "panel_sha256": identity(self.panels), "final": history[-1],
                  "elapsed_seconds": elapsed, "completed_steps": STEPS, "max_gradient_norm": max_norm,
                  "clipped_steps": clipped_steps, "instrumented": instrument,
                  "path": str(destination.relative_to(ROOT))}
        dump(destination / "summary.json", result)
        if not control:
            self.runs.append(result)
            dump(self.out / "run_ledger.json", self.runs)
        print(f"{run_id}: macro={result['final']['macro']:.6f}, original={result['final']['original']:.6f}, {elapsed:.1f}s", flush=True)
        return result

    def sweep(self):
        for sampling, lr in self.configs:
            self.train_one(sampling, lr, 42)
        baseline = self.runs[0]
        original = torch.load(BASE / "model.pt", weights_only=True)
        repeated = torch.load(ROOT / baseline["path"] / "model.pt", weights_only=True)
        assert same_state(original["model"], repeated["model"]), "Baseline reproduction differs from original expanded training"
        print("Baseline exactly reproduces original expanded model weights.")

    def select(self):
        threshold = self.runs[0]["final"]["original"] * 1.05
        eligible = [r for r in self.runs if r["final"]["original"] <= threshold]
        self.winner = min(eligible, key=lambda r: (r["final"]["macro"], r["run_id"]))
        self.selection = {"selected_utc": datetime.now(timezone.utc).isoformat(),
            "selected_run": self.winner["run_id"], "configuration": [self.winner["sampling"], self.winner["peak_lr"]],
            "original_panel_threshold": threshold, "eligible": [r["run_id"] for r in eligible],
            "language_evaluation_used": False, "selection_metric": "final macro category validation NLL"}
        dump(self.out / "selection.json", self.selection)
        print("Selected using validation only:", self.selection)
        return self.selection

    def confirm(self):
        chosen = (self.winner["sampling"], self.winner["peak_lr"])
        for seed in (43, 44):
            for sampling, lr in dict.fromkeys([("uniform", .001), chosen]):
                self.train_one(sampling, lr, seed)
        control = self.train_one(*chosen, 42, instrument=False, control=True)
        instrumented = torch.load(ROOT / self.winner["path"] / "model.pt", weights_only=True)
        uninstrumented = torch.load(ROOT / control["path"] / "model.pt", weights_only=True)
        self.control = {"instrumented_run": self.winner["run_id"], "control_run": control["run_id"],
                        "model_equal": same_state(instrumented["model"], uninstrumented["model"]),
                        "optimizer_equal": same_state(instrumented["optimizer"], uninstrumented["optimizer"]),
                        "torch_rng_equal": same_state(instrumented["torch_rng_state"], uninstrumented["torch_rng_state"])}
        assert all(self.control[k] for k in ("model_equal", "optimizer_equal", "torch_rng_equal"))
        dump(self.out / "instrumentation_control.json", self.control)
        print("Instrumented and uninstrumented final model, optimizer and RNG are exactly equal.")

    def evaluate(self):
        assert (self.out / "selection.json").exists() and (self.out / "instrumentation_control.json").exists()
        suite = load_suite()
        for seed in (42, 43, 44):
            model = quiet_model(self.args)
            model.load_state_dict(self.initials[seed])
            evaluate_suite(model, self.vocab, suite, self.out / "untrained_evals" / f"seed{seed}", stage="untrained")
        for run in self.runs:
            path = ROOT / run["path"]
            with contextlib.redirect_stdout(io.StringIO()):
                model, vocabulary, _ = load_model(path / "model.pt")
            summary = evaluate_suite(model, vocabulary, suite, path / "evals", stage="final")
            run["evaluation"] = summary["overall"]
        dump(self.out / "run_ledger.json", self.runs)
        pairs = []
        for seed in (42, 43, 44):
            baseline = next(r for r in self.runs if r["seed"] == seed and r["sampling"] == "uniform" and r["peak_lr"] == .001)
            chosen = next(r for r in self.runs if r["seed"] == seed and (r["sampling"], r["peak_lr"]) == tuple(self.selection["configuration"]))
            base_rows = json.loads((ROOT / baseline["path"] / "evals/eval_results.json").read_text())
            chosen_rows = json.loads((ROOT / chosen["path"] / "evals/eval_results.json").read_text())
            pairs.append({"seed": seed, "baseline": baseline["run_id"], "selected": chosen["run_id"],
                "macro_loss_change": chosen["final"]["macro"] - baseline["final"]["macro"],
                "original_loss_change": chosen["final"]["original"] - baseline["final"]["original"],
                "original_guardrail_pass": chosen["final"]["original"] <= 1.05 * baseline["final"]["original"],
                "baseline_correct": baseline["evaluation"]["correct"], "selected_correct": chosen["evaluation"]["correct"],
                "gained": [a["id"] for a, b in zip(chosen_rows, base_rows) if a["score"] > b["score"]],
                "regressed": [a["id"] for a, b in zip(chosen_rows, base_rows) if a["score"] < b["score"]]})
        dump(self.out / "paired_comparison.json", pairs)
        return pairs

    def audit(self):
        checks = []
        def check(name, passed, details=None):
            checks.append({"check": name, "passed": bool(passed), "details": details})
        check("original_files_unchanged_before_report", all(sha(ROOT / p) == h for p, h in self.originals.items()), len(self.originals))
        check("frozen_inputs_unchanged", all(sha(ROOT / p) == h for p, h in self.input_hashes.items()))
        check("all_validation_passages_excluded_from_training", set(self.train).isdisjoint(self.val))
        for seed in (42, 43, 44):
            rows = [r for r in self.runs if r["seed"] == seed]
            check(f"seed{seed}_identical_initial_weights", len({r["initial_model_sha256"] for r in rows}) == 1)
            for sampling in {r["sampling"] for r in rows}:
                check(f"seed{seed}_{sampling}_identical_minibatches", len({r["schedule_sha256"] for r in rows if r["sampling"] == sampling}) == 1)
        check("identical_frozen_panels", len({r["panel_sha256"] for r in self.runs}) == 1)
        check("instrumentation_unchanged_trajectory", all(self.control[k] for k in ("model_equal", "optimizer_equal", "torch_rng_equal")))
        suite = load_suite()
        for run in self.runs:
            path = ROOT / run["path"]
            with contextlib.redirect_stdout(io.StringIO()):
                model, vocabulary, saved = load_model(path / "model.pt")
            check(run["run_id"] + "_finite_complete", saved["completed_steps"] == STEPS and all(torch.isfinite(t).all() for t in saved["model"].values()))
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                replay = evaluate_suite(model, vocabulary, suite, temp, stage="final")
                check(run["run_id"] + "_48_case_exact_replay", all((Path(temp) / name).read_bytes() == (path / "evals" / name).read_bytes() for name in ("eval_results.json", "eval_results.csv", "eval_summary.json", "eval_cases.json")))
            update = json.loads((path / "gradient_update.json").read_text())
            check(run["run_id"] + "_adamw_reconstruction", abs(update["reconstruction_error"]) < 1e-7, update["reconstruction_error"])
        result = {"passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks), "checks": checks}
        dump(self.out / "audit.json", result)
        assert result["failed"] == 0, result
        print(f"Audit: {result['passed']} passed, {result['failed']} failed.")
        return result
