"""Reconstruct corpus provenance and optionally replay every saved training trajectory.

Only qa/training_separation.json is written. --retrain-all reproduces all original
and supplemental checkpoints without evaluation prompts, answers or outputs in
the optimizer loop. Output overlap is reported separately from input provenance.
"""
import argparse
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from nanogpt_model import GPT, GPTConfig
from run_evals import load_suite, model_hash, normalized, reject_eval_leakage, reserve_classroom_passages, word_tokens


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def new_model(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(GPTConfig(**args))


def signatures(text, cases):
    content, hits = normalized(text), []
    for case in cases:
        patterns = {"prompt": case["prompt"], "prompt_plus_answer": case["prompt"] + " " + case["answer"],
                    "ordered_four_choices": " ".join(case["choices"]), "keyed_answer": case["id"] + " " + case["answer"]}
        for kind, pattern in patterns.items():
            if normalized(pattern) in content:
                hits.append({"kind": kind, "case_id": case["id"]})
    if re.search(r"\blang_\d{2}\b|[\"'](?:expected|predicted_choice|choice_probabilities|generated_text|suite_sha256)[\"']\s*:", text):
        hits.append({"kind": "serialized_eval_record"})
    answers = [c["answer"] for c in cases]
    for i in range(len(answers) - 7):
        if normalized(" ".join(answers[i:i + 8])) in content:
            hits.append({"kind": "ordered_answer_key_window", "start_case": cases[i]["id"]})
    return hits


def replay(spec, train, vocab):
    """Optimizer inputs: reconstructed training documents, IDs, initialization, schedule."""
    torch.manual_seed(spec["seed"])
    model = new_model(spec["model_args"])
    initial = model_hash(model)
    assert initial == spec["initial_hash"], "Fresh initialization differs"
    stoi = {t: i for i, t in enumerate(vocab)}
    sequences = [[1] + [stoi.get(t, 0) for t in word_tokens(d)] + [2] for d in train]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], betas=(.9, .95), weight_decay=.01)
    opened, active = [], [True]
    def trace(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, bytes)):
            opened.append(str(args[0]))
    sys.addaudithook(trace)
    started = time.perf_counter()
    try:
        model.train()
        for step, indices in enumerate(spec["schedule"]):
            seqs = [sequences[i] for i in indices]
            length = max(len(s) - 1 for s in seqs)
            x = torch.full((32, length), 2, dtype=torch.long)
            y = torch.full_like(x, -1)
            for i, s in enumerate(seqs):
                x[i, :len(s) - 1] = torch.tensor(s[:-1])
                y[i, :len(s) - 1] = torch.tensor(s[1:])
            progress = max(0, step - 100) / 2900
            optimizer.param_groups[0]["lr"] = spec["lr"] * min(1, (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    finally:
        active[0] = False
    forbidden = [p for p in opened if any(part in Path(p).parts for part in ("evals", "llm_runs", "tuning_runs", "evidence", "qa"))]
    assert not forbidden, forbidden
    return {"fresh_initial_hash": initial, "final_hash": model_hash(model), "completed_steps": 3000,
            "seconds": time.perf_counter() - started, "file_opens_during_updates": sorted(set(opened)),
            "forbidden_file_reads": forbidden}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain-all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / "qa/training_separation.json"
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "corpora": {},
              "models": [], "replays": [], "script_sha256": sha(__file__),
              "scope": "Two required experiments and both supplemental tuning studies present at execution time",
              "limits": ["Signature scans cannot rule out every paraphrase or disguised answer key.",
                         "Common words and underlying knowledge may overlap, as allowed by the assignment.",
                         "Exact checkpoint reproduction establishes a complete clean-input route to the saved weights; hashes alone do not authenticate history.",
                         "Generated outputs may reproduce original teaching sentences; overlap alone does not establish output ingestion."]}
    def save():
        report["passed"] = sum(c["passed"] for c in report["checks"])
        report["failed"] = sum(not c["passed"] for c in report["checks"])
        out.write_text(json.dumps(report, indent=2) + "\n")
    def check(name, passed, details=None):
        report["checks"].append({"check": name, "passed": bool(passed), "details": details})
        if not passed:
            save()
            raise AssertionError(name)
    suite = load_suite()
    cases = suite["cases"]
    check("unchanged_48_case_suite", sha(ROOT / "evals/language_evals.json") == "e8affcd72841e3ed7da5c0b6b116327fe9f69c9abd66a1180d1d88ceaa3e17f7" and len(cases) == 48)
    generator_path = ROOT / "scripts/build_extension_corpus.py"
    ms = importlib.util.spec_from_file_location("teaching_generator", generator_path)
    generator = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(generator)
    opened, active = [], [True]
    def trace_generation(event, args):
        if active[0] and event == "open":
            opened.append(str(args[0]))
            raise AssertionError("Pure teaching generator attempted file access")
    sys.addaudithook(trace_generation)
    try:
        material = {"grammar_practice.txt": "\n".join(generator.build_grammar()) + "\n",
                    "contextual_contrasts.txt": "\n".join(generator.build_contrasts()) + "\n"}
    finally:
        active[0] = False
    check("extension_regenerated_without_reading_any_files", not opened,
          {"generator_sha256": sha(generator_path), "generated_passages": 912})
    for filename, text in material.items():
        path = ROOT / "corpus/expanded" / filename
        check(filename + ".exact_generator_bytes", text.encode() == path.read_bytes(), {"sha256": sha(path)})
    source = ast.parse((ROOT / "custom_llm.py").read_text())
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in {"classroom_corpus", "chunk_text", "word_tokens"}]
    assert len(nodes) == 3
    functions = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pure course functions>", "exec"), functions)
    base = functions["chunk_text"](functions["classroom_corpus"]())
    retained, separation = reserve_classroom_passages(base, suite)
    check("160_reserved_prefix_passages_removed", separation["excluded_passages"] == 160)
    altered = json.loads(json.dumps(suite))
    for case in altered["cases"]:
        case["answer"], case["choices"] = "REFERENCE_ANSWER_CANARY", ["ANSWER_KEY_CANARY"] * 4
    check("answer_keys_cannot_change_training_exclusions", reserve_classroom_passages(base, altered)[0] == retained)
    specs, data = [], {}
    for name, run_id in [("starter", "20260923T043154_967139Z"), ("expanded", "20260923T062639_403952Z")]:
        run = ROOT / "llm_runs" / run_id
        extra = [] if name == "starter" else [c for f in sorted(material) for c in functions["chunk_text"](material[f])]
        corpus = "\n".join(retained + extra)
        check(name + ".entire_saved_corpus_reconstructed", corpus.encode() == (run / "corpus.txt").read_bytes())
        manifest = read(run / "corpus_manifest.json")
        expected = [] if name == "starter" else sorted(material)
        check(name + ".manifest_only_teaching_sources", [f["file"] for f in manifest["files"]] == expected)
        folder = ROOT / "corpus" / name
        actual = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in (".txt", ".md", ".pdf") and p.name != "README.md")
        check(name + ".current_training_folder_allowlist", actual == expected, actual)
        for entry in manifest["files"]:
            check(name + ".source_hash." + entry["file"], sha(folder / entry["file"]) == entry["sha256"])
        docs = sorted(set(retained + extra))
        random.Random(42).shuffle(docs)
        cut = int(.9 * len(docs))
        train, val = docs[:cut], docs[cut:]
        split = read(run / "split.json")
        check(name + ".split_reconstructed", train == split["train"] and val == split["validation"] and set(train).isdisjoint(val))
        counts = Counter(t for d in train for t in word_tokens(d))
        kept = sorted((t for t in counts if len(t) <= 128), key=lambda t: (-counts[t], t))[:509]
        vocab = ["<UNK>", "<BOS>", "<EOS>"] + sorted(kept)
        check(name + ".vocabulary_from_training_only", vocab == read(run / "tokenization.json")["vocabulary"])
        hits = signatures(corpus, cases)
        check(name + ".no_prompt_pair_choices_key_or_result_signatures", not hits, hits)
        report["corpora"][name] = {"run": str(run.relative_to(ROOT)), "corpus_sha256": sha(run / "corpus.txt"),
                                   "split_sha256": sha(run / "split.json"), "unique_passages": len(docs),
                                   "train_passages": len(train), "validation_passages": len(val), "signature_hits": hits,
                                   "fully_reconstructed_from_declared_generators": True}
        data[name] = train, vocab
        initial = torch.load(run / "model_untrained.pt", weights_only=True)
        model = new_model(initial["model_args"])
        model.load_state_dict(initial["model"])
        rng = random.Random(43)
        specs.append({"name": name, "data": name, "seed": 42, "lr": .001, "model_args": initial["model_args"],
                      "initial_hash": model_hash(model), "schedule": [rng.choices(range(len(train)), k=32) for _ in range(3000)],
                      "checkpoint": str((run / "model.pt").relative_to(ROOT))})
    injected = [cases[0]["prompt"], cases[0]["prompt"] + " " + cases[0]["answer"],
                " ".join(cases[0]["choices"]), json.dumps({cases[0]["id"]: cases[0]["answer"]}),
                json.dumps({"generated_text": "synthetic audit canary"}), " ".join(c["answer"] for c in cases)]
    for i, text in enumerate(injected):
        check(f"in_memory_contamination_positive_control_{i}", bool(signatures(text, cases)))
    rejected = False
    try:
        reject_eval_leakage(cases[0]["prompt"], suite, "in-memory audit injection")
    except ValueError:
        rejected = True
    check("course_guard_rejects_injected_prompt", rejected)
    expanded_spec = specs[-1]
    for study in sorted((ROOT / "tuning_runs").iterdir()):
        if not study.is_dir():
            continue
        if (study / "protocol.json").exists():
            hashes = read(study / "protocol.json")["input_hashes"]
            for row in read(study / "ledger.json"):
                schedule = read(study / f"batches_{row['sampler']}.json")
                check(study.name + "/" + row["run_id"] + ".schedule_hash", identity(schedule) == row["schedule_sha256"])
                specs.append({"name": study.name + "/" + row["run_id"], "data": "expanded", "seed": row["seed"],
                              "lr": row["learning_rate"], "model_args": expanded_spec["model_args"],
                              "initial_hash": row["initial_model_sha256"], "schedule": schedule,
                              "checkpoint": str((study / row["run_id"] / "model.pt").relative_to(ROOT))})
        elif (study / "preregistered_plan.json").exists():
            hashes = read(study / "preregistered_plan.json")["input_sha256"]
            for row in read(study / "run_ledger.json"):
                schedule = json.loads(gzip.decompress((study / f"batches_seed{row['seed']}_{row['sampling']}.json.gz").read_bytes()))
                check(study.name + "/" + row["run_id"] + ".schedule_hash", identity(schedule) == row["schedule_sha256"])
                specs.append({"name": study.name + "/" + row["run_id"], "data": "expanded", "seed": row["seed"],
                              "lr": row["peak_lr"], "model_args": expanded_spec["model_args"],
                              "initial_hash": row["initial_model_sha256"], "schedule": schedule,
                              "checkpoint": str(Path(row["path"]) / "model.pt")})
        else:
            continue
        check(study.name + ".unchanged_frozen_inputs", all(sha(ROOT / p) == h for p, h in hashes.items()))
        check(study.name + ".same_expanded_corpus_and_split", all(hashes[f"llm_runs/20260923T062639_403952Z/{f}"] == sha(ROOT / "llm_runs/20260923T062639_403952Z" / f) for f in ("corpus.txt", "split.json", "tokenization.json")))
    for spec in specs:
        check(spec["name"] + ".3000_training_only_batches", len(spec["schedule"]) == 3000 and all(len(b) == 32 and all(isinstance(i, int) and 0 <= i < len(data[spec["data"]][0]) for i in b) for b in spec["schedule"]))
        saved = torch.load(ROOT / spec["checkpoint"], weights_only=True)
        model = new_model(saved["model_args"])
        model.load_state_dict(saved["model"])
        spec["expected_hash"] = model_hash(model)
        report["models"].append({k: v for k, v in spec.items() if k not in ("schedule", "model_args")})
    output_files = sorted(set((ROOT / "llm_runs").glob("**/eval_results.json")) | set((ROOT / "tuning_runs").glob("**/eval_results.json")))
    overlaps = []
    expanded_docs = {" ".join(word_tokens(d)) for d in data["expanded"][0]}
    for path in output_files:
        for row in read(path):
            output = " ".join(word_tokens(row.get("generated_text", "")))
            if output and output in expanded_docs:
                overlaps.append({"file": str(path.relative_to(ROOT)), "case_id": row["id"], "output": output,
                                 "source": "original teaching generator reconstructed byte for byte; present in frozen training split"})
    report["output_overlap_review"] = {"files_scanned": len(output_files), "complete_output_equals_training_passage": overlaps,
                                       "interpretation": "Generation may reproduce teaching sentences. Source reconstruction and exact retraining test direction of use; overlap alone cannot establish leakage."}
    save()
    if args.retrain_all:
        cache = {}
        for spec in specs:
            fingerprint = identity({"seed": spec["seed"], "lr": spec["lr"], "data": spec["data"],
                                    "initial_hash": spec["initial_hash"], "schedule": spec["schedule"]})
            if fingerprint not in cache:
                print("Replaying", spec["name"], flush=True)
                cache[fingerprint] = replay(spec, *data[spec["data"]])
                report["replays"].append({"trajectory_sha256": fingerprint, "representative": spec["name"], **cache[fingerprint]})
            actual = cache[fingerprint]
            check(spec["name"] + ".clean_input_retraining_matches_every_weight", actual["final_hash"] == spec["expected_hash"],
                  {"actual": actual["final_hash"], "expected": spec["expected_hash"], "trajectory_sha256": fingerprint})
            save()
            print("PASS", spec["name"], flush=True)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_models_retrained"] = args.retrain_all
    save()
    print(json.dumps({"passed": report["passed"], "failed": report["failed"], "saved_models": len(specs),
                      "distinct_retrained_trajectories": len(report["replays"]), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
