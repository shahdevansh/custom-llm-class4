"""Read-only consistency audit of the two executed Class 4 experiments.

Writes only the requested JSON report (default: qa/audit.json). It never trains,
edits evidence, or repairs a failing artifact. Optional eval replay writes to a
temporary directory that is removed afterward.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_HASHES = {
    "run_evals.py": "da87f28d128344807512e2bac1cfc662b37ac2c7e4a32b84c09f1950e92d67a0",
    "chat.py": "6152c8b7780f3b46fef5de38461adfc4b1a55df70ed106ca73ec5e9aded86d25",
    "evals/language_evals.json": "e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7",
    "nanogpt_model.py": "7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c",
}
UPSTREAM_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, path="value", tolerance=1e-7):
    """Recursive equality with a tolerance only for floating-point evidence."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{path}.{key}", tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list lengths differ")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{path}[{index}]", tolerance)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
                f"{path}: {actual!r} != {expected!r}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


class Audit:
    def __init__(self, root):
        self.root = root
        self.checks = []
        self.inputs = {}

    def relative(self, path):
        return str(Path(path).resolve().relative_to(self.root))

    def bytes(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.inputs[self.relative(path)] = hashlib.sha256(data).hexdigest()
        return data

    def json(self, path):
        return json.loads(self.bytes(path))

    def text(self, path):
        return self.bytes(path).decode("utf-8")

    def check(self, name, function):
        try:
            details = function()
            self.checks.append({"check": name, "status": "pass", "details": details})
            return details
        except Exception as error:
            self.checks.append({"check": name, "status": "fail", "error": f"{type(error).__name__}: {error}"})
            return None


def summary(rows):
    scorable = sum(row["status"] in {"scored", "tied"} for row in rows)
    correct = sum(row["score"] for row in rows)
    return {"correct": correct, "total": len(rows), "scorable": scorable,
            "success_rate_all_cases": correct / len(rows),
            "accuracy_scorable_cases": correct / scorable if scorable else None,
            "coverage": scorable / len(rows)}


def constants(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def audit_notebook(audit, path, baseline):
    notebook = audit.json(path)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    counters = [cell.get("execution_count") for cell in code]
    require(counters == list(range(1, len(code) + 1)), f"Execution counters are not complete, sequential: {counters}")
    errors = [output for cell in code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    require(not errors, f"Notebook contains {len(errors)} error outputs")
    require(sum(bool(cell.get("outputs")) for cell in code) >= 10, "Insufficient visible cell outputs")
    base_code = ["".join(cell["source"]) for cell in baseline["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code]
    # The first code cell contains experiment choices. All remaining original
    # source cells must occur unmodified and in order; extra chat cells may follow.
    position = 1
    for original in base_code[1:]:
        require(original in sources[position:], "An original code cell is absent, modified, or reordered")
        position = sources.index(original, position) + 1
    settings = constants(sources[0])
    return {"code_cells": len(code), "execution_counters": counters,
            "original_nonsettings_code_preserved": True, "settings": settings}


def audit_run(audit, name, entry, runner, suite, args, baseline):
    run = audit.root / entry["run_dir"]
    config = audit.json(run / "config.json")
    split = audit.json(run / "split.json")
    tokenization = audit.json(run / "tokenization.json")
    vocabulary = tokenization["vocabulary"]
    manifest = audit.json(run / "corpus_manifest.json")
    training = audit.json(run / "training_summary.json")
    notebook = audit.check(f"{name}.executed_notebook", lambda: audit_notebook(audit, audit.root / entry["notebook"], baseline))
    settings = notebook["settings"] if notebook else {}

    def corpus_check():
        raw = audit.text(run / "corpus.txt")
        raw_passages = raw.splitlines()
        train, val = split["train"], split["validation"]
        require(set(train).isdisjoint(val), "Training and validation overlap")
        require(len(train) == len(set(train)) and len(val) == len(set(val)), "Duplicates remain inside split")
        require(set(train) | set(val) == set(raw_passages), "Saved corpus and split do not contain the same passages")
        require(config["corpus_sha256"] == hashlib.sha256(raw.encode()).hexdigest(), "Corpus hash mismatch")
        docs = sorted(set(raw_passages))
        random.Random(config["seed"]).shuffle(docs)
        cut = int(0.9 * len(docs))
        require(train == docs[:cut] and val == docs[cut:], "Split differs from recorded deterministic 90/10 procedure")
        require(len(train) == config["train_documents"] and len(val) == config["validation_documents"], "Split count mismatch")
        require(len(docs) == manifest["unique_passages"], "Unique passage count mismatch")
        require(len(raw_passages) - len(docs) == manifest["duplicates_removed"], "Duplicate count mismatch")
        require(all(len(runner.word_tokens(p)) <= config["block_size"] - 1 for p in docs), "A passage exceeds the window")
        for key, documents, seed in [("train", train, 123), ("validation", val, 456)]:
            panel = split[f"evaluation_{key}"]
            require(panel == random.Random(seed).sample(documents, min(20, len(documents))), f"{key}: fixed panel differs")
            require(config["evaluation_panel_size"][key] == len(panel), f"{key}: panel count differs")
        folder = audit.root / settings["CORPUS_FOLDER"]
        runner.validate_corpus_location(folder)
        leaked = [(i, runner.matching_cases(p, suite)) for i, p in enumerate(docs) if runner.matching_cases(p, suite)]
        require(not leaked, f"Exact normalized eval prefixes in saved corpus: {leaked[:3]}")
        for record in manifest["files"]:
            source = folder / record["file"]
            require(hashlib.sha256(audit.bytes(source)).hexdigest() == record["sha256"], f"Imported file hash mismatch: {source}")
            if source.suffix.lower() in {".md", ".txt"}:
                runner.reject_eval_leakage(source.read_text(encoding="utf-8-sig"), suite, str(source))
        separation = audit.json(run / "eval_separation.json")
        require(separation["suite_sha256"] == runner.suite_hash(suite), "Separation suite hash mismatch")
        require(separation["excluded_passages"] == config["reserved_eval_passages"], "Excluded passage count mismatch")
        return {"train": len(train), "validation": len(val), "unique_passages": len(docs),
                "new_unique_passages": manifest["new_unique_passages"], "exact_prefix_matches": 0,
                "panels": config["evaluation_panel_size"], "imported_files": len(manifest["files"])}

    audit.check(f"{name}.corpus_split_panels_and_exact_prefix_separation", corpus_check)

    def vocabulary_check():
        counts = Counter(token for passage in split["train"] for token in runner.word_tokens(passage))
        retained = sorted((t for t in counts if len(t) <= 128), key=lambda t: (-counts[t], t))[:509]
        expected = ["<UNK>", "<BOS>", "<EOS>"] + sorted(retained)
        require(vocabulary == expected, "Vocabulary is not the exact training-only deterministic reconstruction")
        val_tokens = [t for passage in split["validation"] for t in runner.word_tokens(passage)]
        train_unknown = sum(counts[t] for t in counts if t not in vocabulary) / max(1, sum(counts.values()))
        val_unknown = sum(t not in vocabulary for t in val_tokens) / max(1, len(val_tokens))
        report = audit.json(run / "vocabulary_report.json")
        same(report, {"training_types": len(counts), "retained_types": len(retained),
                      "training_unknown_rate": train_unknown, "validation_unknown_rate": val_unknown,
                      "omitted_types": sorted(set(counts) - set(retained))})
        same(config["training_unknown_rate"], train_unknown)
        same(config["validation_unknown_rate"], val_unknown)
        require(config["vocabulary_size"] == len(vocabulary), "Vocabulary size mismatch")
        stoi = {token: index for index, token in enumerate(vocabulary)}
        ids = [1] + [stoi.get(t, 0) for t in runner.word_tokens(tokenization["example"])] + [2]
        require(tokenization["example"] == split["train"][0], "Token example is not first training passage")
        require(tokenization["ids"] == ids and tokenization["inputs"] == ids[:-1] and tokenization["targets"] == ids[1:], "Token IDs/targets mismatch")
        return {"vocabulary_size": len(vocabulary), "training_unknown_rate": train_unknown,
                "validation_unknown_rate": val_unknown, "exact_training_only_reconstruction": True}

    audit.check(f"{name}.training_only_vocabulary", vocabulary_check)

    models, model_identities, summaries = {}, {}, {}
    for stage, filename in [("untrained", "model_untrained.pt"), ("final", "model.pt")]:
        def checkpoint_check(stage=stage, filename=filename):
            audit.bytes(run / filename)
            with contextlib.redirect_stdout(io.StringIO()):
                model, vocab, saved = runner.load_model(run / filename)
            identity = runner.model_hash(model)
            require(vocab == vocabulary, "Saved model vocabulary differs")
            expected_steps = 0 if stage == "untrained" else training["completed_steps"]
            require(saved["completed_steps"] == expected_steps, "Checkpoint completed_steps mismatch")
            for key in ["n_embd", "n_head", "n_layer", "block_size"]:
                require(getattr(model.config, key) == config[key], f"Model/config {key} mismatch")
            require(sum(p.numel() for p in model.parameters()) == config["parameters"], "Parameter count mismatch")
            models[stage] = model
            model_identities[stage] = identity
            return {"model_sha256": identity, "completed_steps": saved["completed_steps"]}
        audit.check(f"{name}.{stage}.saved_model", checkpoint_check)

        def evaluation_check(stage=stage):
            folder = run / "language_evals" / stage
            cases = audit.json(folder / "eval_cases.json")
            rows = audit.json(folder / "eval_results.json")
            result_summary = audit.json(folder / "eval_summary.json")
            require(cases == suite, "Saved evaluation suite changed")
            require(len(rows) == 48 and [r["id"] for r in rows] == [c["id"] for c in suite["cases"]], "Eval cases omitted, added, repeated, or reordered")
            require(result_summary["suite_sha256"] == runner.suite_hash(suite) == config["language_eval_suite_sha256"], "Evaluation suite hash mismatch")
            require(result_summary["model_sha256"] == model_identities[stage], "Evaluation model hash mismatch")
            require(result_summary["stage"] == stage, "Evaluation stage mismatch")
            for index, (row, case) in enumerate(zip(rows, suite["cases"])):
                for key in ["id", "group", "category", "prompt", "choices", "reason"]:
                    require(row[key] == case[key], f"{case['id']}: changed {key}")
                require(row["expected"] == case["answer"] and row["stage"] == stage, "Answer key or stage changed")
                require(isinstance(row["generated_text"], str), "Missing free continuation")
                require(row["sample_seed"] == result_summary["settings"]["seed"] + index, "Continuation seed mismatch")
                tokens = runner.word_tokens(case["prompt"])
                unknown = sorted(set(t for t in tokens if t not in vocabulary))
                unknown_choices = [c for c in case["choices"] if runner.word_tokens(c)[0] not in vocabulary]
                require(row["unknown_prompt_words"] == unknown and row["unknown_choices"] == unknown_choices, "Unknown-word accounting mismatch")
                context_long = len(tokens) + 1 > config["block_size"]
                require(row["prompt_truncated"] == context_long, "Context flag mismatch")
                if context_long or unknown or unknown_choices:
                    require(row["status"] == ("context_too_long" if context_long else "out_of_vocabulary"), "Unscorable status mismatch")
                    require(row["score"] == 0 and row["predicted_choice"] is None and row["choice_probabilities"] == {}, "Unscorable case received credit/prediction")
                else:
                    probs = row["choice_probabilities"]
                    require(set(probs) == set(case["choices"]) and all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values()), "Invalid choice probabilities")
                    ranked = sorted(probs, key=probs.get, reverse=True)
                    tied = abs(probs[ranked[0]] - probs[ranked[1]]) <= 1e-10
                    require(row["status"] == ("tied" if tied else "scored"), "Scored/tied status mismatch")
                    require(row["predicted_choice"] == (None if tied else ranked[0]), "Prediction not highest-probability choice")
                    require(row["score"] == (0 if tied else int(ranked[0] == case["answer"])), "Case score arithmetic mismatch")
            same(result_summary["overall"], summary(rows))
            for field, key in [("by_group", "group"), ("by_category", "category")]:
                expected = {group: summary([r for r in rows if r[key] == group]) for group in sorted({r[key] for r in rows})}
                same(result_summary[field], expected)
            csv_rows = list(csv.DictReader(io.StringIO(audit.text(folder / "eval_results.csv"))))
            require(len(csv_rows) == len(rows), "CSV row count mismatch")
            for actual, expected in zip(csv_rows, rows):
                rendered = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else "" if v is None else str(v) for k, v in expected.items()}
                require(actual == rendered, "CSV differs from JSON result")
            summaries[stage] = result_summary
            return {"cases": len(rows), "suite_sha256": result_summary["suite_sha256"], "overall": result_summary["overall"]}
        audit.check(f"{name}.{stage}.complete_evaluation_and_arithmetic", evaluation_check)

        if args.replay_evals:
            def replay(stage=stage):
                expected_rows = audit.json(run / "language_evals" / stage / "eval_results.json")
                settings_for_eval = summaries[stage]["settings"]
                with tempfile.TemporaryDirectory(prefix="custom-llm-audit-") as temporary:
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner.evaluate_suite(models[stage], vocabulary, suite, temporary, stage=stage, **settings_for_eval)
                    actual_rows = json.loads((Path(temporary) / "eval_results.json").read_text())
                    same(actual_rows, expected_rows, tolerance=1e-6)
                require(runner.model_hash(models[stage]) == model_identities[stage], "Replay mutated weights")
                return {"replayed_cases": len(actual_rows), "weights_unchanged": True, "numeric_tolerance": 1e-6}
            audit.check(f"{name}.{stage}.saved_model_eval_replay", replay)

    def paired_checks():
        require(model_identities["untrained"] != model_identities["final"], "Training did not change model hash")
        require(summaries["untrained"]["settings"] == summaries["final"]["settings"], "Before/after eval settings changed")
        same(audit.json(run / "language_eval_comparison.json"), summaries)
        require(config["upstream_commit"] == UPSTREAM_COMMIT, "NanoGPT source pin mismatch")
        require(training["completed_steps"] == config["training_steps"] and not training["interrupted"], "Training incomplete or interrupted")
        require(settings["TRAINING_STEPS"] == config["training_steps"] and settings["LEARNING_RATE"] == config["learning_rate"], "Notebook/config settings differ")
        require(math.isfinite(training["elapsed_seconds"]) and training["elapsed_seconds"] > 0, "Invalid elapsed time")
        return {"completed_steps": training["completed_steps"], "elapsed_seconds": training["elapsed_seconds"]}
    audit.check(f"{name}.before_after_identity_and_settings", paired_checks)

    def learning_artifacts():
        history = audit.json(run / "history.json")
        expected_steps = sorted({0, config["training_steps"] // 2, config["training_steps"]})
        require([row["step"] for row in history] == expected_steps, "Initial/halfway/final history steps differ")
        for row in history:
            require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ["training_loss", "validation_loss"]), "Invalid loss")
            sample_path = run / "samples" / f"step_{row['step']:04d}.txt"
            # Splitting on the literal separator preserves an empty fourth sample.
            require(len(audit.text(sample_path).split("\n")) == 4, "Sample file does not preserve all four samples")
        csv_history = [{"step": int(r["step"]), "training_loss": float(r["training_loss"]), "validation_loss": float(r["validation_loss"])}
                       for r in csv.DictReader(io.StringIO(audit.text(run / "training.csv")))]
        same(csv_history, history)
        require("<svg" in audit.text(run / "training_curves.svg"), "Missing SVG plot")
        temperatures = audit.json(run / "temperature_comparison.json")
        require(set(temperatures) == {"0.3", "0.8", "1.2"} and all(len(samples) == 4 for samples in temperatures.values()), "Incomplete temperature samples")
        require(temperatures["0.8"] == audit.text(run / "samples" / f"step_{training['completed_steps']:04d}.txt").split("\n"), "Final samples differ from same-temperature samples")
        inspection = audit.json(run / "inspection.json")
        checkpoint = audit.json(run / "checkpoint.json")
        token_id = inspection["token_id"]
        require(vocabulary[token_id] == inspection["token"], "Inspection token ID mismatch")
        require(len(inspection["embedding_before"]) == len(inspection["embedding_after"]) == config["n_embd"] == 64, "Embedding dimensions differ")
        same(inspection["embedding_before"], models["untrained"].transformer.wte.weight[token_id].detach().tolist())
        same(inspection["embedding_after"], models["final"].transformer.wte.weight[token_id].detach().tolist())
        same(checkpoint["initial_embeddings"][token_id], inspection["embedding_before"])
        same(checkpoint["weights"]["wte"][token_id], inspection["embedding_after"])
        same(checkpoint["config"], config)
        require(checkpoint["vocabulary"] == vocabulary and checkpoint["completed_steps"] == training["completed_steps"], "Viewer checkpoint identity differs")
        update = inspection["first_update"]
        require(update["token"] == inspection["token"] and update["coordinate"] == 0, "First-update coordinate differs")
        same(update["before"], inspection["embedding_before"][0])
        require(all(math.isfinite(update[k]) for k in ["before", "gradient", "learning_rate", "after"]), "Nonfinite gradient/update")
        require(update["learning_rate"] > 0 and update["before"] != update["after"], "No real first parameter change")
        for stage, key in [("untrained", "probabilities_before"), ("final", "probabilities_after")]:
            import torch
            ids = [1] + [vocabulary.index(t) if t in vocabulary else 0 for t in runner.word_tokens(inspection["prefix"])]
            with torch.inference_mode():
                probs = torch.softmax(models[stage](torch.tensor([ids[-config["block_size"]:]]))[0][0, -1], -1).tolist()
            same(inspection[key], probs, tolerance=1e-6)
            require(math.isclose(sum(inspection[key]), 1, abs_tol=1e-6), "Probabilities do not sum to one")
        return {"history_rows": len(history), "sample_stages": expected_steps,
                "inspection_token": inspection["token"], "embedding_dimensions": 64,
                "first_update_delta": update["after"] - update["before"]}
    audit.check(f"{name}.loss_samples_embeddings_probabilities_and_update", learning_artifacts)

    def chat_check():
        transcript = audit.json(run / "chat_transcript.json")
        require(transcript["model_sha256"] == model_identities["final"], "Chat used another model")
        require(transcript["completed_steps"] == training["completed_steps"], "Chat step identity differs")
        require(transcript["fresh_context_per_prompt"] is True and len(transcript["turns"]) >= 3, "Missing three fresh-context turns")
        for turn in transcript["turns"]:
            reply = runner.generate_reply(models["final"], vocabulary, turn["prompt"], seed=turn["seed"],
                                          temperature=transcript["temperature"], max_tokens=transcript["max_tokens"])
            require({key: turn[key] for key in reply} == reply, "Saved chat reply does not replay from saved weights")
        require(runner.model_hash(models["final"]) == model_identities["final"], "Chat replay mutated weights")
        return {"turns": len(transcript["turns"]), "model_sha256": model_identities["final"], "all_replies_replayed": True,
                "turns_with_unknown_words": sum(bool(t["unknown_prompt_words"]) for t in transcript["turns"])}
    audit.check(f"{name}.chat_identity_and_actual_reply_replay", chat_check)

    def archive_check():
        archive = run.with_suffix(".zip")
        audit.bytes(archive)
        with zipfile.ZipFile(archive) as zipped:
            require(zipped.testzip() is None, "ZIP CRC failure")
            names = [i.filename for i in zipped.infolist() if not i.is_dir()]
            require(len(names) == len(set(names)), "Duplicate ZIP members")
            expected = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
            require(set(names) == expected, "ZIP file inventory differs from run directory")
            for name_in_zip in names:
                require(zipped.read(name_in_zip) == audit.bytes(run / name_in_zip), f"ZIP stale/different file: {name_in_zip}")
        return {"archive": audit.relative(archive), "files": len(names), "crc_and_byte_parity": True}
    audit.check(f"{name}.archive_integrity_and_freshness", archive_check)
    return {"run_dir": entry["run_dir"], "notebook": entry["notebook"], "summaries": summaries,
            "model_hashes": model_identities, "config": config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit available runs; explicitly report missing runs as pending")
    parser.add_argument("--replay-evals", action="store_true", help="Reproduce all saved 48-case results from each saved checkpoint")
    parser.add_argument("--check-readme", action="store_true", help="Check local Markdown links in README after the report is written")
    parser.add_argument("--output", type=Path, default=ROOT / "qa/audit.json")
    args = parser.parse_args()
    audit = Audit(ROOT)
    def sources():
        for relative, expected in SUPPORT_HASHES.items():
            require(hashlib.sha256(audit.bytes(ROOT / relative)).hexdigest() == expected, f"Fixed source changed: {relative}")
        return {"upstream_commit": UPSTREAM_COMMIT, "verified_sha256": SUPPORT_HASHES}
    verified = audit.check("fixed_runner_suite_chat_and_nanogpt_source_hashes", sources)
    runs = {}
    missing = []
    if verified:
        sys.path.insert(0, str(ROOT))
        import run_evals as runner
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        suite = runner.load_suite(ROOT / "evals/language_evals.json")
        ledger = audit.json(ROOT / "experiment_runs.json")
        baseline = audit.json(ROOT / "source/course_starter.ipynb")
        missing = [name for name in ["starter", "expanded"] if name not in ledger]
        audit.check("fixed_suite_has_48_unique_cases", lambda: require(len(suite["cases"]) == len({c["id"] for c in suite["cases"]}) == 48, "Suite is not 48 distinct cases") or {"cases": 48, "suite_sha256": runner.suite_hash(suite)})
        if missing and not args.allow_incomplete:
            audit.check("both_experiments_present", lambda: require(False, f"Missing experiments: {missing}"))
        for name in ["starter", "expanded"]:
            if name in ledger:
                result = audit.check(f"{name}.audit_completed", lambda name=name: audit_run(audit, name, ledger[name], runner, suite, args, baseline))
                if result:
                    runs[name] = result
        if len(runs) == 2:
            def cross_run():
                require(runs["starter"]["run_dir"] != runs["expanded"]["run_dir"], "Experiments share a run directory")
                require(runs["starter"]["model_hashes"]["untrained"] != runs["expanded"]["model_hashes"]["untrained"], "Unexpected identical initial model hashes despite corpus extension")
                for key in ["training_steps", "learning_rate", "seed", "n_embd", "n_head", "n_layer", "block_size", "batch_size"]:
                    require(runs["starter"]["config"][key] == runs["expanded"]["config"][key], f"Corpus comparison changed {key}")
                require(runs["starter"]["config"]["corpus_files"] == 0, "Starter contains added files")
                require(runs["expanded"]["config"]["corpus_files"] >= 2, "Expected at least two focused extension files")
                require(sum(s["overall"]["total"] for r in runs.values() for s in r["summaries"].values()) == 192, "Four result sets do not total 192 cases")
                return {"experiments": 2, "stages": 4, "case_results": 192, "same_training_settings": True}
            audit.check("two_fresh_experiments_four_complete_result_sets", cross_run)
    if args.check_readme:
        def links_check():
            readme = audit.text(ROOT / "README.md")
            checked = []
            for raw in re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", readme):
                target = raw.strip().strip("<>").split(' "', 1)[0]
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                local = (ROOT / unquote(parsed.path)).resolve()
                require(local.is_relative_to(ROOT), f"README link points outside repository: {target}")
                require(local.exists() or local == args.output.resolve(), f"Broken README link: {target}")
                checked.append(target)
            require(checked, "README has no local evidence links")
            return {"local_links_checked": len(checked), "note": "Existence only; hosted access and fragment anchors are not checked."}
        audit.check("readme_local_evidence_links", links_check)
    failed = [check["check"] for check in audit.checks if check["status"] == "fail"]
    status = "fail" if failed else "incomplete" if missing else "pass"
    report = {"schema": "class4-evidence-audit-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": status, "missing_experiments": missing,
              "check_counts": {"passed": sum(c["status"] == "pass" for c in audit.checks), "failed": len(failed)},
              "replayed_all_evaluations": args.replay_evals, "checks": audit.checks,
              "input_file_sha256": dict(sorted(audit.inputs.items())),
              "limitations": [
                  "Exact normalized prefix detection does not prove absence of semantic leakage, paraphrased tests, or answer-list contamination; inspect teaching material independently.",
                  "Vocabulary is reconstructed from saved training passages; file consistency cannot prove the historical process independently of the saved code and evidence.",
                  "The recorded gradient and first update are checked for finite values and initial/final checkpoint consistency; the optimizer trajectory is not retrained by this audit.",
                  "No claim of general language understanding, unseen-source generalization, or statistically reliable improvement follows from these public development tests.",
                  "This local audit does not verify signed-out GitHub accessibility, repository rendering, course-portal submission, screenshot authenticity, or student understanding.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed": report["check_counts"]["passed"], "failed": failed,
                      "missing_experiments": missing, "report": str(args.output)}, indent=2))
    return 1 if failed or (missing and not args.allow_incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
