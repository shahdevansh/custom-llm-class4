# Independent review of the original delivered experiments

**Provisional score: 9.6/10 — deliverable quality 3.8/4, testing and evaluation 2.9/3, working result 2.9/3.** This is an evidence-based reviewer judgment under the repository's 4/3/3 assignment rubric, not an official grade. Confidence is high in the saved-model replay and artifact checks, moderate in grading judgment, and low in causal attribution of differences between the two independently initialized models.

The artifact is much more complete than the model is capable. The original expanded model scores 24/48 on the four-choice development benchmark, but none of its 24 extension cases is correct. That weak model quality is not an automatic assignment deduction: the supplied rubric explicitly allows full evaluation credit for a complete, valid and honestly explained unsuccessful experiment.

## Rubric judgment

| Criterion | Score | Evidence supporting credit | Concrete deduction |
|---|---:|---|---|
| Deliverable quality | **3.8/4** | Both complete executed experiments; readable supplied code; source hashes and deterministic corpus generator; detailed category rationale; actual token/embedding/gradient/attention/probability/loss evidence; clear README and evidence links. | **−0.2:** nine teaching passages need proofreading: five `a artist` errors and four relative clauses applying `which` to people. The latter are an unreported template-fit problem. Embedded notebook ZIP links are also machine-local, although README relative links work. |
| Testing and evaluation | **2.9/3** | Four complete unchanged 48-case result sets, all exactly replayed; CSV/JSON/summary/separation artifacts; all-case, scorable, coverage, group/category and free-generation comparison; transparent failure and confound analysis. | **−0.1:** the documented untrained replay commands produce misleading `final` stage metadata. The original saved untrained results themselves are correctly labeled and valid. |
| Working result | **2.9/3** | Actual trained nanoGPT saved weights; eight new interactive PTY replies; run/model identity; saved-model replays; recording/transcript; successful isolated clean-environment remedy. | **−0.1:** original README setup stops at a missing `python3.12` launcher on this shell and does not provide the now-verified alternative. This is a small setup-documentation deduction, not a nonworking-model finding. |
| **Total** | **9.6/10** | All substantive deliverable, evaluation and working-model requirements are evidenced. | Remaining deductions are limited to corpus and reproducibility polish. |

The fractional deductions are reviewer judgment, not an instructor-provided subrubric. I would not deduct for 0/24 extension success itself, for retaining failed samples, or for accurately disclosing confounded causal attribution. The clean `uv` remedy below strengthens confidence in the working result but was not present in the README snapshot being scored.

## Scope and independent method

- Snapshot: commit `56f610f7e4ae183db78bf9201cf9d543dfde654a`; review began with a clean tracked worktree and froze SHA-256 hashes for all 174 tracked files in `snapshot.json`.
- Original runs only: starter `20260923T043154_967139Z` and expanded `20260923T062639_403952Z`. Later tuning is outside this review.
- Entry point was the README. It already displayed existing chat/eval examples, so this is not a fully blinded prompt test. Eight new prompts were frozen in `own_prompts.json` before opening raw case/result/transcript files. This boundary is recorded explicitly rather than claiming blindness.
- Replayed each original untrained and trained checkpoint through the README's `run_evals.py` command, preserving outputs in `replay/` and logs. Compared all 192 rows including probabilities, continuations and coverage, all four summaries, all saved CSV rows, and all four case suites. Comparison ignores only the replay stage-label defect described below.
- Launched the actual README CLI in a pseudo-terminal with the original expanded checkpoint, entered eight prompts, and captured actual terminal bytes/timestamps plus the interface's JSON transcript. No canned answers or external model API were used.
- Reviewed the complete executed notebooks and original corpus sources; separate detailed evidence is in `notebook_corpus/`.
- No tracked files or original artifacts were edited by this review. Durable review files are under this QA directory; an expressly requested temporary clean environment was created outside the repository and removed afterward. Browser opening of the local replay page was blocked by browser security policy; no workaround was attempted. The actual `.cast`, terminal log and self-contained replay HTML are available, but browser playback and a new screenshot were not visually verified.

## Complete notebooks, training and corpus audit

All **158 notebook/learning/corpus checks passed**: 116 learning/notebook checks and 42 corpus checks, plus structural SVG checks. Each notebook has 28 cells, 15 sequentially executed code cells and no stored errors. All original course code is preserved except experiment settings and explanatory title/prediction text, with three copied original chat cells added. Complete source/output dumps were inspected and saved.

Independently reconstructed seed-42 initialization exactly matches each saved untrained model. An independently coded 3,000-step training loop reproduces **every tensor** of both final saved models. Both real first gradients/AdamW updates, all 64-coordinate probe vectors, full embedding tables, token IDs, next-token probability arrays and attention rows match. All 24 timeline samples and 24 temperature samples match. This is strong evidence that the reported learning values correspond to these actual experiments.

| Run | Step | Training panel loss | Validation panel loss |
|---|---:|---:|---:|
| Starter | 0 | 4.926252365112305 | 4.927547931671143 |
| Starter | 1500 | 0.6821381449699402 | 0.7182430624961853 |
| Starter | 3000 | 0.6783124208450317 | 0.7061358690261841 |
| Expanded | 0 | 5.781627655029297 | 5.803586006164551 |
| Expanded | 1500 | 0.7778765559196472 | 0.7909225821495056 |
| Expanded | 3000 | 0.7322326302528381 | 0.7467306852340698 |

Every panel contains 20 passages. Means exclude padding and weight observed next-token targets equally: 253/243 training/validation targets for starter, 239/245 for expanded. Loss trends are meaningful within each run; cross-run absolute loss comparisons confound corpus and vocabulary changes.

Corpus reconstruction confirms **528 grammar + 384 contrast = 912 unique additions**, with no overlap with the retained starter passages. Generator output, source file hashes, full manifests, stored corpus, split membership, panels and training-only vocabulary match. Grammar material targets agreement and tense; contrast material puts opposing attributes into varied local contexts. The rationale genuinely addresses two categories, but limited template variety and missing vocabulary prevent broad test coverage. Exact-prefix checks show no suite prompt in the retained corpus; this is not proof against every semantic overlap with a public benchmark.

Nine added passages (0.99%) need proofreading: five article errors (`a artist`) and four human relative clauses using `which` where `who` is the natural grammatical choice. Seven of the nine entered training and two entered validation. Their numerical effect on outcomes has not been isolated.

## Verified four-way evaluation

| Experiment/stage | Correct/all | All-case success | Scorable | Scorable accuracy | Coverage |
|---|---:|---:|---:|---:|---:|
| starter-untrained | 9/48 | 18.75% | 24 | 37.50% | 50.00% |
| starter-final | 20/48 | 41.67% | 24 | 83.33% | 50.00% |
| expanded-untrained | 9/48 | 18.75% | 27 | 33.33% | 56.25% |
| expanded-final | 24/48 | 50.00% | 27 | 88.89% | 56.25% |

All 192 replayed rows match the original saved results exactly, including floating-point choice probabilities and actual free continuations. All summaries agree. Each CSV has 48 rows and agrees with its JSON. The cases and scoring runner are byte-identical to pinned upstream course commit `9e04ddb6aacb8efcb790e70c62550ca55e0f2a75`. Both original ZIPs contain 33 entries, pass archive integrity checks, and match the corresponding run directories byte for byte.

| Group: correct/total | Starter untrained | Starter final | Expanded untrained | Expanded final |
|---|---:|---:|---:|---:|
| Starter patterns | 6/16 | 16/16 | 4/16 | 16/16 |
| Familiar-vocabulary new wording | 3/8 | 4/8 | 3/8 | 8/8 |
| Extension challenges | 0/24 | 0/24 | 2/24 | 0/24 |

The expanded model's four net gains over the starter are all in new wording. The targeted grammar/opposites subset has only 3/6 scorable cases and 0/6 correct after training; the other 18 extension cases remain unscorable. Across all extension challenges, 21 failures are coverage failures and three are scored pattern failures. This is why the 88.89% accuracy on scorable cases must never be presented without 56.25% coverage and 50.00% all-case success.

The three newly scorable cases are `lang_27` (past tense: chooses `walk` for `yesterday she`, expected `walked`), `lang_28` (chooses `heavy` for opposite of hot, expected `cold`), and `lang_29` (chooses `soft` for opposite of empty, expected `full`). The expanded untrained checkpoint happened to get two of those three correct; training gets zero. This is direct evidence against claiming successful transfer of the targeted extension relationships.

Choice ranking and generation are distinct. The fixed free continuations begin with the expected answer on 14/48 starter-final cases and 11/48 expanded-final cases; untrained runs do so on 0/48. This additional descriptive check is not the rubric's metric or a reliable estimate of fluent response quality: it is one fixed stochastic sample per case. It nevertheless shows why the higher multiple-choice score should not be equated with better unconstrained generation. The starter-final outputs also include 10 empty continuations, all retained in the saved evidence. See `free_continuation_diagnostics.json`.

## Actual new-user chat

Exact command tested after activating the existing environment:

```sh
source .venv/bin/activate
python chat.py --model llm_runs/20260923T062639_403952Z/model.pt --transcript qa/independent_review_20260923/own_chat_transcript.json
```

The process exited successfully after eight real interactions. The model state hash in the new transcript exactly equals the independently replayed expanded-final checkpoint: `759c8b1e9c073f74b820d3c962fa108b5793a5b9c67ee263328e01ee6fe9cbd8`. Checkpoint metadata reports 3,000 completed steps, 319 vocabulary entries and 123,584 parameters. The CLI uses fresh context per prompt, temperature 0.8, a 24-token continuation limit, and seeds 2026–2033.

| Prompt | Actual continuation | Unknown prompt words |
|---|---|---|
| `the investor reviewed the` | `wall in the sun , which is bright , which is dark .` | investor |
| `the doctor works at the` | `hospital is short , but the hospital is long .` | None |
| `yesterday the artist` | `is empty , while the visitor is soft .` | None |
| `the teachers` | `the box in the hall , which is empty , with the office is full .` | teachers |
| `the stone is heavy but the feather is` | `light .` | feather |
| `the opposite of cold is` | `loud and the new instructor was loud and the lesson at the school was short .` | None |
| `write a story about a dragon` | `the local customer .` | dragon, story, write |
| `two plus two equals` | `the course and the local instructor .` | equals, plus, two |

These responses establish a working interface and narrow corpus pattern generation. They do not establish general instruction following, reliable grammar or factual competence. Even the apparently correct `light .` continuation has an unknown input noun (`feather`), so it cannot establish understanding of that noun. Familiar in-vocabulary prompts can fail too: `yesterday the artist` continues in present tense, and the explicit opposite query produces `loud` rather than `hot`.

The following is a **post hoc qualitative review**, not a preregistered scoring rule or an additional benchmark percentage. The prompts were frozen; these judgment categories were assigned after seeing the responses. Grammar judgments concern the prompt plus continuation where the prompt is a sentence prefix. Relevance asks whether the continuation follows the requested context or relation; lexical coverage is reported separately.

| Prompt ID | Lexical coverage | Grammar / sentence form | Relevance and sensible continuation |
|---|---|---|---|
| review_01: investor | `investor` unknown | Has a complete clause structure, but chains incompatible bright/dark descriptions of the same wall. | Weak: wall description does not demonstrate the intended finance association; overall coherence fails. |
| review_02: doctor | Fully covered | Malformed full sentence: `the doctor works at the hospital is short ...`. | Partial local relevance (`hospital`), followed by an incoherent short/long contrast. |
| review_03: yesterday artist | Fully covered | Present `is` conflicts with the tested ordinary past-time context; the person/property descriptions are unnatural. | Fails intended tense continuation and yields unrelated attributes. |
| review_04: teachers | `teachers` unknown | Malformed: the opening noun phrase runs into `the box` with no connecting verb. | Does not demonstrate plural agreement or a coherent continuation. |
| review_05: stone/feather | `feather` unknown | Grammatical completed sentence: `the stone is heavy but the feather is light .`. | Sensible contrast in this sample. Unknown `feather` prevents attributing it to knowledge of that word. |
| review_06: opposite cold | Fully covered | Surface clause structure is acceptable, although repetitive. | Fails the requested relation: `loud` is not the opposite of `cold`; later school-related clauses are irrelevant. |
| review_07: dragon story | `write`, `story`, `dragon` unknown | Response is only a noun phrase, `the local customer .`. | No story or dragon-related response; illustrates the instruction and vocabulary boundary. |
| review_08: arithmetic | `two`, `plus`, `equals` unknown | The noun phrase can occupy the completion position, but it supplies no mathematical answer. | Fails arithmetic relevance; this test is outside the trained vocabulary and task distribution. |

Evidence: `own_chat_terminal.txt`, `own_chat_transcript.json`, `own_chat.cast`, and `own_chat_recording.html`. The original submitted four-turn chat transcript was also independently replayed and all four responses and unknown-word lists matched exactly.

## Reproducibility defects and boundaries

1. **Stage metadata is wrong when following the README literally for untrained models.** Both documented untrained commands omit `--stage untrained`; the runner's default is `final`. They produce the correct 9/48 numerical results but every row and summary is labeled `final`. The original notebook-saved untrained results are correctly labeled. Fix the two README commands, or infer the default stage from checkpoint metadata while preserving an explicit override.
2. **The original literal setup path did not work on this shell; a separate clean remedy did.** `python3.12 -m venv ...` failed because `python3.12` is not on PATH. The existing `.venv/bin/python` runs chat/evals and its installed package versions match the complete lock file, but it has no `pip` module, so `python -m pip check` fails. This does not prove installation would fail on a machine with the stated prerequisites; it identifies a prerequisite/documentation gap. A separately requested `uv` route then installed the exact lock file in a fresh temporary environment and reproduced all 48 expanded-final results plus three chat responses exactly. Add that verified route or explicit Python/pip prerequisite checks to the README.
3. **Corpus defects are preserved honestly but remain defects.** The grammar teaching file includes five `a artist seems ...` sentences; four contrast sentences also apply `which` to human runners/visitors. Preserving the original corpus is correct for provenance; fix the generator only in a separately identified future experiment and retain these originals.
4. **Causal/generalization limits are handled correctly.** Across the two runs, the corpus, vocabulary/parameter count, token IDs, initialization and split/panels change. The repository acknowledges these confounds. The public suite guided extension-category selection, so it is a development benchmark. Exact-prefix separation cannot certify semantic independence, although the original source review and training-input checks provide evidence against copied test items.
5. **Student understanding is not directly established by artifact completeness.** The repository openly attributes assistance and does not fabricate a student reflection. Its detailed explanation supports assessment of the deliverable; it cannot substitute for an oral or independently written understanding check.

## Priority recommendations

1. Correct the two untrained replay commands with `--stage untrained`; retain the original evidence unchanged.
2. Add the independently verified clean `uv` setup route below, with a short dependency check and an actual saved-model chat/eval invocation. Put prerequisites before launch instructions.
3. For a future experiment, repair the nine proofreading issues and broaden naturally authored teaching material across subjects, tense contexts and contrast forms. Inspect coverage at category level. Do not solve vocabulary gaps by copying the held-out prompts or answer lists.
4. Freeze independently authored, unseen wording before corpus revision, and compare several seeds. Keep existing original runs and explicitly separate any new benchmark-informed tuning from an untouched test.
5. Add direct README links to the fixed suite, scoring runner and full inspection JSON; use a short student-authored explanation to show understanding if that is expected by the instructor.

## Separately verified clean-environment remedy

This addendum records a proposed repair; it does not silently change the original review snapshot. With the available `/opt/homebrew/bin/uv`, these commands succeeded in an isolated temporary environment:

```sh
uv venv --python 3.12 --seed <temporary-directory>/venv
uv pip install --python <temporary-directory>/venv/bin/python -r requirements-lock.txt
<temporary-directory>/venv/bin/python -m pip check
```

The resolver installed all 59 locked packages; `pip check` found no broken requirements. The interpreter was Python **3.12.12** (the original experiment used 3.12.14), with PyTorch 2.14.0, NumPy 2.5.3 and pypdf 6.19.0. The saved expanded-final checkpoint then reproduced every 48-case row exactly. Three additional frozen prompts gave identical seeded transcripts across the clean and original environments:

| Prompt | Actual continuation |
|---|---|
| `the gardener feels` | `tired before work .` |
| `the client discussed the` | `brand after checking the price .` |
| `last week the musicians` | `walked through the park .` |

All three responses were nonempty; both transcripts identify the original expanded model hash. The temporary environment was removed after the check. Exact commands, outputs, versions and equality checks are in `clean_uv_setup_commands.json` and `clean_uv_setup_checks.json`; the replay is in `clean_setup_eval/`. This supports clean-environment saved-model reproducibility using the proposed route, without claiming that the original README contained it.

## Evidence files

- `snapshot.json`: starting commit, clean status, all tracked SHA-256 hashes.
- `checks.json`: machine-readable four-replay equality, model identity and own-chat checks.
- `additional_checks.json`: upstream byte identity, ZIP consistency, original chat replay, local link resolution and environment boundary.
- `command_log.json` / `additional_command_log.json`: exact executed commands and outputs.
- `own_prompts.json`: frozen prompt plan and explicit independence boundary.
- `replay/*`: four complete independent 48-case result sets.
- `notebook_corpus/*`: complete notebook/corpus inspection and independent training replay evidence.
