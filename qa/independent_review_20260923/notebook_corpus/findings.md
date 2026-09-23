# Independent notebook, corpus, and learning-evidence review

Scope: original commit `56f610f7e4ae183db78bf9201cf9d543dfde654a`, only the starter run `20260923T043154_967139Z` and expanded run `20260923T062639_403952Z`. No existing QA assertions were used as proof. No tracked files were edited. No memory or earlier conversation was consulted.

## Results

**High confidence: the notebook and numerical learning evidence are authentic and internally consistent.** `verify_notebooks_learning.py` reconstructed both full 3,000-step trainings in memory and passed all 116 checks. This did not execute notebook cells that write new run directories. All initial tensors match fresh seed-42 initialization, and every final tensor exactly matches the corresponding submitted `model.pt`.

- Both submitted notebooks validate as notebook format 4, each with 28 cells and 15 code cells. Code execution counts are exactly 1–15; no errors are saved; all code parses. All original course cells except title, settings, and prediction are unchanged. The only added code consists of three instances of the original chat cell with different prompts. Human-readable dumps and per-cell inventories are saved beside this report.
- Both first real gradient/AdamW updates, the complete 64-coordinate probe vectors, entire embedding tables, vocabulary IDs, shifted tokenization inputs/targets, full next-token probability arrays, and inspected first-head attention matrices match reconstructed values exactly.
- Both training/validation losses at all three milestones (0, 1,500, 3,000), all 24 timeline generations, and all 24 temperature generations match exactly. The cross-entropy values also match an independent mean of observed-target negative log probabilities, excluding padding, within floating-point tolerance.
- Each panel contains 20 documents. The numbers of scored next-token targets are starter 253 train / 243 validation and expanded 239 train / 245 validation. Thus the documented means are over tokens, not equal-weight document means. The code and explanation accurately disclose the small panels and shared templates.
- Saved SVG plots are valid XML with finite coordinates, and the notebook's stored SVG output equals the run's plot file. This is structural verification; no additional visual browser rendering was performed in this subtask.
- Corpus reconstruction is independently recorded in `corpus_check/`: 528 grammar passages plus 384 contrast passages, 912 unique additions, no overlap with the retained starter set, matching generator/source hashes, manifests, corpus text, split membership, fixed panels, training-only vocabulary, and zero corpus UNK rates.

## Concrete deficiencies and limits

1. **Minor teaching-data quality defect, already disclosed:** `corpus/expanded/grammar_practice.txt:127,135,143,151,159` says `a artist ...`; these should use `an`. Three entered training and two validation. README lines 249–257 acknowledge this and propose correction in the next experiment. Preserving the original source now is correct provenance, but the sentences remain poor grammar teaching examples.
2. **Minor additional wording defect:** `corpus/expanded/contextual_contrasts.txt:154,156,172,174` uses `which` for human runners/visitors; standard person-relative wording here is `who`. For example line 172 says `the first visitor, which was early`. These result from applying an object-relative-clause template to people. This is a small corpus proofreading issue, not evidence of test leakage or invalid training.
3. **Student understanding remains unverified:** section 11 of both notebooks and `STUDENT_README.md` retain course prompts. Root README lines 157–195 and `docs/learning_evidence.md` do provide substantial actual-value explanations, so this is not a missing-explanation deduction by itself. `docs/reproduction_notes.md:15` expressly says the student learning check remains unanswered; generated prose cannot establish personal understanding.
4. **Category coverage remains weak, accurately disclosed:** half of the six targeted grammar/opposites cases remain unscorable from lexical gaps; targeted trained results do not improve. This is an experimental outcome and teaching-design limitation, not an automatic grading deduction. The explanation separates vocabulary coverage from learning and does not claim success.
5. **Embedded notebook download links are machine-local absolute paths.** They will not work directly for a remote repository reader, although the root README's relative run ZIP links do provide working downloadable artifacts. This is a minor inherited notebook presentation issue.

## Deliverable-quality assessment

The requested executed experiments, settings/reasons, corpus rationale/provenance, actual learning explanations, full vectors, samples, loss evidence, and stated limitations are present. No technical notebook or numerical-evidence deduction is supported. A modest discretionary polish deduction for the defective teaching examples is supportable (approximately 0–0.2 of the 4 deliverable-quality points, depending on grading strictness). Do not deduct merely for low extension scores or for leaving an unused course-template README unchanged when the root README supplies the required explanation. Evaluate personal understanding separately rather than assuming it from an assistant-written report.

## Machine evidence

- `verify_notebooks_learning.py`: independent replay/check source.
- `notebook_learning_checks.json`: 116 checks, all passed; model/settings, loss target counts, and source hashes.
- `notebook_learning_checks.log`: milestone replay results.
- `plot_checks.json`: stored plot/output structure consistency.
- `custom_llm.ipynb.txt`, `custom_llm_expanded.ipynb.txt`: every source cell and output dumped for inspection.
- `custom_llm.ipynb.cells.json`, `custom_llm_expanded.ipynb.cells.json`: every cell's execution/output inventory.
- `corpus_check/`: independent corpus script, results, and detailed findings.
