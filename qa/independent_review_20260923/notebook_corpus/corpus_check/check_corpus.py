"""Independent reconstruction from tracked sources; never imports prior QA results."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import random
import re
import subprocess
import types
import zipfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TARGET = "56f610f7e4ae183db78bf9201cf9d543dfde654a"
BASELINE = "9e04ddb6aacb8efcb790e70c62550ca55e0f2a75"
RUNS = {"starter": "20260923T043154_967139Z", "expanded": "20260923T062639_403952Z"}
result = {"target_commit": TARGET, "checks": {}, "sources": {}, "runs": {}}

def check(label, condition):
    result["checks"][label] = bool(condition)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git_blob(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)

def tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)

def passages(text):
    answer = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        words = tokens(sentence)
        for offset in range(0, len(words), 47):
            answer.append(" ".join(words[offset:offset + 47]))
    return answer

def padded(text):
    return " " + " ".join(tokens(text)) + " "

def isolate_classroom(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "classroom_corpus")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<isolated corpus>", "exec"), namespace)
    return namespace["classroom_corpus"](), ast.dump(function, include_attributes=False)

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
check("head_matches_requested_commit", head == TARGET)
upstream_code = git_blob(BASELINE, "custom_llm.py")
current_code = (ROOT / "custom_llm.py").read_bytes()
check("course_python_unchanged_from_baseline", current_code == upstream_code)
check("preserved_course_notebook_byte_matches_baseline", (ROOT / "source/course_starter.ipynb").read_bytes() == git_blob(BASELINE, "custom_llm.ipynb"))
check("eval_suite_unchanged_from_baseline", (ROOT / "evals/language_evals.json").read_bytes() == git_blob(BASELINE, "evals/language_evals.json"))
base_text, base_ast = isolate_classroom(current_code.decode())
original_base_text, original_base_ast = isolate_classroom(upstream_code.decode())
check("classroom_generator_matches_original", base_ast == original_base_ast and base_text == original_base_text)

suite = json.loads((ROOT / "evals/language_evals.json").read_text())
suite_hash = digest(json.dumps(suite, sort_keys=True, ensure_ascii=False).encode())
raw_base = passages(base_text)
excluded = [(p, [c["id"] for c in suite["cases"] if padded(c["prompt"]) in padded(p)]) for p in raw_base]
base = [p for p, matches in excluded if not matches]
excluded = [(p, matches) for p, matches in excluded if matches]
result["base"] = {"raw_passages": len(raw_base), "excluded_passages": len(excluded), "excluded_unique_passages": len(set(p for p, _ in excluded)), "retained_passages": len(base), "retained_unique_passages": len(set(base)), "excluded_case_counts": dict(Counter(i for _, matches in excluded for i in matches))}

generator = types.ModuleType("independent_generator_import")
generator.__file__ = str(ROOT / "scripts/build_extension_corpus.py")
exec(compile(Path(generator.__file__).read_text(), generator.__file__, "exec"), generator.__dict__)
generated = {"grammar_practice.txt": generator.build_grammar(), "contextual_contrasts.txt": generator.build_contrasts()}
extra = []
records = []
source_sets = {}
for filename in sorted(generated):
    path = ROOT / "corpus/expanded" / filename
    data = path.read_bytes()
    text = data.decode("utf-8-sig")
    chunks = passages(text)
    check(f"generator_exact_bytes_{filename}", data == ("\n".join(generated[filename]) + "\n").encode())
    check(f"one_complete_sentence_per_line_{filename}", len(chunks) == len(text.splitlines()) and all(line.endswith(".") for line in text.splitlines()))
    check(f"unique_within_file_{filename}", len(chunks) == len(set(chunks)))
    source_sets[filename] = set(chunks)
    extra.extend(chunks)
    record = {"file": filename, "bytes": len(data), "sha256": digest(data), "warnings": [], "characters": len(text), "passages": len(chunks), "unique_passages": len(set(chunks)), "preview": text[:300]}
    records.append(record)
    result["sources"][filename] = {**record, "min_tokens": min(map(lambda x: len(tokens(x)), chunks)), "max_tokens": max(map(lambda x: len(tokens(x)), chunks)), "token_types": len(set(tokens(text)))}
check("912_unique_additions", len(extra) == len(set(extra)) == len(set(extra) - set(base)) == 912)
check("no_full_eval_prefix_in_extensions", not any(padded(c["prompt"]) in padded(p) for p in extra for c in suite["cases"]))
check("no_concatenated_eval_choice_lists_in_extensions", not any(padded(" ".join(c["choices"])) in padded(p) for p in extra for c in suite["cases"]))
result["extension"] = {"unique_passages": len(set(extra)), "overlap_with_base": len(set(extra) & set(base)), "token_types": len(set(tokens("\n".join(extra)))), "new_token_types": len(set(tokens("\n".join(extra))) - set(tokens(base_text))), "combined_token_types": len(set(tokens("\n".join(base + extra))))}

for kind, run_name in RUNS.items():
    run = ROOT / "llm_runs" / run_name
    chunks = base + (extra if kind == "expanded" else [])
    unique = sorted(set(chunks))
    random.Random(42).shuffle(unique)
    cut = int(.9 * len(unique))
    train, val = unique[:cut], unique[cut:]
    expected_split = {"train": train, "validation": val, "evaluation_train": random.Random(123).sample(train, min(20, len(train))), "evaluation_validation": random.Random(456).sample(val, min(20, len(val)))}
    split = json.loads((run / "split.json").read_text())
    check(f"{kind}_exact_ordered_split_and_panels", split == expected_split)
    check(f"{kind}_train_validation_disjoint", not (set(split["train"]) & set(split["validation"])))
    check(f"{kind}_no_duplicates_in_either_split", all(len(split[k]) == len(set(split[k])) for k in ("train", "validation")))
    check(f"{kind}_union_equals_all_deduplicated_sources", set(split["train"] + split["validation"]) == set(chunks))
    check(f"{kind}_all_documents_complete_and_within_limit", all(p.endswith(" .") and len(tokens(p)) <= 47 for p in split["train"] + split["validation"]))
    check(f"{kind}_no_full_eval_prefix_in_any_split", not any(padded(c["prompt"]) in padded(p) for p in split["train"] + split["validation"] for c in suite["cases"]))
    expected_corpus = "\n".join(chunks).encode()
    check(f"{kind}_corpus_exact_bytes", (run / "corpus.txt").read_bytes() == expected_corpus)
    manifest = json.loads((run / "corpus_manifest.json").read_text())
    expected_manifest = {"files": records if kind == "expanded" else [], "ignored": [], "external_passages": len(extra) if kind == "expanded" else 0, "mode": "classroom", "base_passages": len(base), "unique_passages": len(unique), "new_unique_passages": len(set(extra)-set(base)) if kind == "expanded" else 0, "duplicates_removed": len(chunks)-len(unique), "max_passage_tokens": 47, "split_unit": "deduplicated passage, not source file"}
    check(f"{kind}_manifest_all_fields", manifest == expected_manifest)
    config = json.loads((run / "config.json").read_text())
    check(f"{kind}_config_corpus_hash", config["corpus_sha256"] == digest(expected_corpus))
    expected_sep = {"excluded_passages": len(excluded), "case_ids": sorted({i for _, ids in excluded for i in ids}), "suite_sha256": suite_hash, "method": "normalized contiguous prompt match; not a semantic leakage detector"}
    check(f"{kind}_separation_exact", json.loads((run / "eval_separation.json").read_text()) == expected_sep)
    counts = Counter(t for p in train for t in tokens(p))
    retained = sorted((t for t in counts if len(t) <= 128), key=lambda t: (-counts[t], t))[:509]
    vocabulary = ["<UNK>", "<BOS>", "<EOS>"] + sorted(retained)
    vocab_set = set(vocabulary)
    val_tokens = [t for p in val for t in tokens(p)]
    expected_vocab_report = {"training_types": len(counts), "retained_types": len(retained), "training_unknown_rate": sum(counts[t] for t in counts if t not in vocab_set)/sum(counts.values()), "validation_unknown_rate": sum(t not in vocab_set for t in val_tokens)/len(val_tokens), "omitted_types": sorted(set(counts)-set(retained))}
    check(f"{kind}_vocabulary_training_only", json.loads((run / "tokenization.json").read_text())["vocabulary"] == vocabulary)
    check(f"{kind}_vocabulary_report", json.loads((run / "vocabulary_report.json").read_text()) == expected_vocab_report)
    check(f"{kind}_config_counts_and_rates", config["vocabulary_size"] == len(vocabulary) and config["train_documents"] == len(train) and config["validation_documents"] == len(val) and all(config[k] == expected_vocab_report[k] for k in ("training_unknown_rate", "validation_unknown_rate")))
    with zipfile.ZipFile(run.with_suffix(".zip")) as z:
        exact_zip_files = {}
        for name in ("corpus.txt", "corpus_manifest.json", "split.json", "config.json", "tokenization.json", "vocabulary_report.json", "eval_separation.json"):
            matches = [n for n in z.namelist() if n == name or n.endswith("/" + name)]
            exact_zip_files[name] = len(matches) == 1 and z.read(matches[0]) == (run / name).read_bytes()
        check(f"{kind}_zip_corpus_evidence_matches_directory", all(exact_zip_files.values()))
    grammar_errors = [{"file": filename, "line": index, "sentence": line, "split": "train" if passages(line)[0] in train else "validation"} for filename in generated for index, line in enumerate((ROOT / "corpus/expanded" / filename).read_text().splitlines(), 1) if "a artist" in line] if kind == "expanded" else []
    relative_pronoun_errors = [{"file": "contextual_contrasts.txt", "line": index, "sentence": line, "split": "train" if passages(line)[0] in train else "validation"} for index, line in enumerate((ROOT / "corpus/expanded/contextual_contrasts.txt").read_text().splitlines(), 1) if index in {154, 156, 172, 174}] if kind == "expanded" else []
    target_coverage = [{"id": c["id"], "category": c["category"], "prompt": c["prompt"], "answer": c["answer"], "unknown_prompt": sorted(set(tokens(c["prompt"]))-vocab_set), "unknown_choices": [x for x in c["choices"] if tokens(x)[0] not in vocab_set]} for c in suite["cases"] if c["category"] in {"grammar", "opposites"}]
    result["runs"][kind] = {"directory": str(run.relative_to(ROOT)), "raw_passages": len(chunks), "unique_passages": len(unique), "duplicates_removed": len(chunks)-len(unique), "train": len(train), "validation": len(val), "corpus_sha256": digest(expected_corpus), "vocabulary": expected_vocab_report, "zip_corpus_files": exact_zip_files, "source_split_counts": {filename: {"train": len(source_set & set(train)), "validation": len(source_set & set(val))} for filename, source_set in source_sets.items()}, "target_category_lexical_coverage": target_coverage, "article_errors": grammar_errors, "relative_pronoun_errors": relative_pronoun_errors}

result["cross_run"] = {"starter_train_to_expanded_validation": len(set(json.loads((ROOT / "llm_runs" / RUNS["starter"] / "split.json").read_text())["train"]) & set(json.loads((ROOT / "llm_runs" / RUNS["expanded"] / "split.json").read_text())["validation"])), "starter_validation_to_expanded_train": len(set(json.loads((ROOT / "llm_runs" / RUNS["starter"] / "split.json").read_text())["validation"]) & set(json.loads((ROOT / "llm_runs" / RUNS["expanded"] / "split.json").read_text())["train"]))}
result["all_checks_passed"] = all(result["checks"].values())
(OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"checks": len(result["checks"]), "all_checks_passed": result["all_checks_passed"], "failures": [k for k,v in result["checks"].items() if not v], "sources": result["sources"], "base": result["base"], "extension": result["extension"], "cross_run": result["cross_run"]}, indent=2))
