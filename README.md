# Class 4: two executed custom-LLM experiments

Two fresh nanoGPT models completed 3,000 training updates each. The same 48-case suite ran before and after each experiment, producing **192 recorded case results**. Executed notebooks: [starter](custom_llm.ipynb) and [expanded corpus](custom_llm_expanded.ipynb).

## Evaluation result sets

| Experiment / stage | Correct / all | All-case success | Scorable | Scorable accuracy | Coverage | Complete evidence |
|---|---:|---:|---:|---:|---:|---|
| Starter / untrained | 9/48 | 18.75% | 24 | 37.50% | 50.00% | [JSON](llm_runs/20260923T043154_967139Z/language_evals/untrained/eval_results.json) · [CSV](llm_runs/20260923T043154_967139Z/language_evals/untrained/eval_results.csv) · [summary](llm_runs/20260923T043154_967139Z/language_evals/untrained/eval_summary.json) · [cases](llm_runs/20260923T043154_967139Z/language_evals/untrained/eval_cases.json) |
| Starter / trained | 20/48 | 41.67% | 24 | 83.33% | 50.00% | [JSON](llm_runs/20260923T043154_967139Z/language_evals/final/eval_results.json) · [CSV](llm_runs/20260923T043154_967139Z/language_evals/final/eval_results.csv) · [summary](llm_runs/20260923T043154_967139Z/language_evals/final/eval_summary.json) · [cases](llm_runs/20260923T043154_967139Z/language_evals/final/eval_cases.json) |
| Expanded / untrained | 9/48 | 18.75% | 27 | 33.33% | 56.25% | [JSON](llm_runs/20260923T062639_403952Z/language_evals/untrained/eval_results.json) · [CSV](llm_runs/20260923T062639_403952Z/language_evals/untrained/eval_results.csv) · [summary](llm_runs/20260923T062639_403952Z/language_evals/untrained/eval_summary.json) · [cases](llm_runs/20260923T062639_403952Z/language_evals/untrained/eval_cases.json) |
| Expanded / trained | 24/48 | 50.00% | 27 | 88.89% | 56.25% | [JSON](llm_runs/20260923T062639_403952Z/language_evals/final/eval_results.json) · [CSV](llm_runs/20260923T062639_403952Z/language_evals/final/eval_results.csv) · [summary](llm_runs/20260923T062639_403952Z/language_evals/final/eval_summary.json) · [cases](llm_runs/20260923T062639_403952Z/language_evals/final/eval_cases.json) |

- **Extension categories and added data:** **grammar** and **opposites**. Added [528 grammar passages](corpus/expanded/grammar_practice.txt) and [384 contextual-contrast passages](corpus/expanded/contextual_contrasts.txt), for **912 unique additions**. Total corpus size grew from 4,592 to 5,504 unique passages. [Design and sources](docs/corpus_design.md).
- **Finding and concrete failure:** trained scores rose from **20/48 to 24/48**, with all four gains in familiar-vocabulary rephrasings. The expanded model still failed **all 24 extension cases**. For `yesterday she`, it selected `walk` instead of `walked` even though every required word was known. [Case evidence and limitations](#what-improved-and-what-failed).
- **Working chat evidence:** **four actual interactions** with the saved expanded model, including failures. [Recording](evidence/chat_recording.html) · [transcript](evidence/chat_transcript.json) · [screenshot](evidence/chat_screenshot.png) · [interface source](chat.py) · [launch instructions](#working-chat).
- **Training/evaluation separation:** [98 verification checks](qa/training_separation.json), source reconstruction, and exact retraining of all 23 original and supplemental final checkpoints. [Controls and recheck command](#proof-of-trainingevaluation-separation).

## What was chosen and why

**Corpus:** first the supplied synthetic classroom corpus with no added files, then the same classroom generator plus focused **grammar** and **opposites** teaching material. The starter repeats domain associations; the expansion was intended to add subject/verb agreement, tense, and contextual contrasts. The [deterministic generator](scripts/build_extension_corpus.py) created 528 grammar passages and 384 contextual contrast passages, and the executed manifest confirms all **912 unique additions**. They use different roles, situations, and wording from the test stories.

**Training budget:** 3,000 optimizer steps per fresh model, batch size 32, the suggested course starting budget. Each step updates weights using a minibatch of 32 passages. **Learning rate:** 0.001 as the configured peak, with the supplied 100-step warmup and cosine decay. The first effective rate is 0.00001. Excessively large updates could destabilize the loss; very small ones could make too little progress within 3,000 steps. These choices were proposed with Codex and recorded before execution.

The model has 2 transformer blocks, 4 attention heads, 64-dimensional embeddings, a 48-token context, and seed 42. Training ran locally on CPU; saved hardware is `macOS-26.6.2-arm64-arm-64bit`, Python 3.12.14 and PyTorch 2.14.0. The locally trained model generated every reply. The unchanged nanoGPT source is pinned to [Karpathy commit 3adf61e](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py), with its [MIT license](NANOGPT_LICENSE). Whole-word tokenization and teaching/evaluation helpers come from the [course project](https://github.com/pepealonso95/custom-llm).

## Corpus, permissions, and experimental controls

The starter text is generated by the supplied course code. The two added UTF-8 TXT files contain original synthetic examples created with Codex assistance and are shared with this project. Source files: [grammar](corpus/expanded/grammar_practice.txt) and [contextual contrasts](corpus/expanded/contextual_contrasts.txt). Source hashes, previews, and passage counts are saved in the manifests. Both TXT files loaded with zero extraction warnings or ignored files.

| Measurement | Starter | Expanded |
|---|---:|---:|
| Unique passages | 4592 | 5504 |
| Added unique passages | 0 | 912 |
| Training / validation passages | 4132 / 460 | 4953 / 551 |
| Vocabulary including UNK/BOS/EOS | 136 | 319 |
| Retained ordinary token types | 133 | 316 |
| Training / held-out token UNK rate | 0.00% / 0.00% | 0.00% / 0.00% |
| Parameters | 111,872 | 123,584 |
| Completed optimizer steps | 3000 | 3000 |
| Training-loop elapsed seconds | 9.063363 | 13.797743 |
| Full notebook wall seconds | 20.248840 | 18.271616 |
| Interrupted | No | No |

Both runs remove 160 generated passages containing reserved test prefixes and remove 1,608 duplicate passages, then use a 90/10 split of unique passages. The vocabulary is built only from the training split (maximum 509 ordinary types plus three special tokens). Both runs retained every training token type. Held-out corpus UNK rates were 0%; separate test-suite coverage was 50.00% and 56.25%.

Within each run, the vocabulary, train/validation split, fixed loss panels, seed and baseline generation settings are unchanged before/after training. Across the two fresh runs, adding passages changes split membership and panels; changing vocabulary changes token IDs, embedding/output dimensions, parameter counts, and random initialization even with seed 42. These simultaneous changes limit causal attribution. Compare loss trends within each run because the corpora and vocabularies differ.

**Starter audit links:** [config.json](llm_runs/20260923T043154_967139Z/config.json) · [corpus_manifest.json](llm_runs/20260923T043154_967139Z/corpus_manifest.json) · [vocabulary_report.json](llm_runs/20260923T043154_967139Z/vocabulary_report.json) · [split.json](llm_runs/20260923T043154_967139Z/split.json) · [corpus.txt](llm_runs/20260923T043154_967139Z/corpus.txt) · [tokenization.json](llm_runs/20260923T043154_967139Z/tokenization.json) · [training_summary.json](llm_runs/20260923T043154_967139Z/training_summary.json) · [eval_separation.json](llm_runs/20260923T043154_967139Z/eval_separation.json)

**Expanded audit links:** [config.json](llm_runs/20260923T062639_403952Z/config.json) · [corpus_manifest.json](llm_runs/20260923T062639_403952Z/corpus_manifest.json) · [vocabulary_report.json](llm_runs/20260923T062639_403952Z/vocabulary_report.json) · [split.json](llm_runs/20260923T062639_403952Z/split.json) · [corpus.txt](llm_runs/20260923T062639_403952Z/corpus.txt) · [tokenization.json](llm_runs/20260923T062639_403952Z/tokenization.json) · [training_summary.json](llm_runs/20260923T062639_403952Z/training_summary.json) · [eval_separation.json](llm_runs/20260923T062639_403952Z/eval_separation.json)

## Proof of training/evaluation separation

- **Enforced before training:** the course loader excludes 160 reserved-prefix passages before splitting or building vocabulary, rejects exact eval prompts in imported text, and rejects training folders containing the project root or eval directory. Evaluation runs in inference mode and checks that weights stay unchanged.
- **Inputs reconstructed:** the [separation audit](qa/training_separation.json) regenerated both teaching files and both saved corpora byte for byte, then reconstructed the training splits and vocabularies. The manifests contain only the classroom generator and the two declared teaching files. Scans found **zero matches** for normalized eval prompts, prompt-answer sequences, ordered choice lists, or serialized answer-key/result signatures. Deliberately injected examples were detected.
- **Weights reproduced:** **98 checks passed, zero failed.** Fifteen fresh 3,000-step runs reproduced the exact model-state hashes of **all 23 final checkpoints**, including both tuning studies. Only reconstructed teaching passages and verified training indices entered these replays; tracing recorded **zero file opens during weight updates**.

The audit records one generated response matching a teaching sentence already present in the pre-tuning source commit. Its source and clean replay account for that overlap. These checks establish reproducible input separation; exhaustive paraphrase detection remains a source-review judgment. [Independent corpus review](qa/independent_review_20260923/notebook_corpus/corpus_check/findings.md).

Reverify with the [audit script](scripts/audit_training_separation.py):

~~~sh
.venv/bin/python scripts/audit_training_separation.py --retrain-all
~~~

## Before training: prediction

The [pretraining plan](docs/pretraining_plan.md) set three expectations:

- **Moderate confidence:** training and held-out loss would fall, and samples would resemble the teaching sentences more closely.
- **Low to moderate confidence:** added grammar and contrast examples would improve the corresponding tests.
- **Exploratory:** `customer` would move closer to buyer-related words in embedding space.

## After training: observation

- Loss fell in both runs, and samples became more corpus-like.
- The expanded model gained coverage on three extension cases and missed all three. Two had been correct at random initialization.
- In the expanded run, `customer`'s closest vectors changed from `traffic`, `chair`, and `waiting` to `subscriber`, `client`, and `shopper`.
- Plausible templates coexisted with malformed sentences and failed transfer.

## Evaluation method and detailed results

- **Suite:** 48 fixed cases: 16 starter patterns, 8 new phrasings, and 24 extension challenges.
- **Scoring:** the correct word must have the highest probability among four choices. Ties earn zero; free continuations are saved separately.
- **Coverage:** every prompt word and answer choice must be in the vocabulary, and the prompt must fit the context window. Unscorable cases count as zero in the overall result.
- **Metrics:** all-case success = correct/48; scorable accuracy = correct/scorable; coverage = scorable/48. `N/A` indicates zero scorable cases.
- **Separation:** only the prefix enters the model; weights stay fixed during evaluation. Each corpus excludes 160 reserved-prefix passages. Exact-prefix checks are supplemented by source review; semantic overlap remains a review limitation.
- **Interpretation:** these public tests guided category selection and serve as a development benchmark.


### Group results

*Cell format: correct/total (success); scorable/total (coverage); scorable accuracy.*

| Group or category | Starter untrained | Starter trained | Expanded untrained | Expanded trained |
|---|---|---|---|---|
| extend_corpus | 0/24 (0.00%); 0/24 (0.00%); N/A | 0/24 (0.00%); 0/24 (0.00%); N/A | 2/24 (8.33%); 3/24 (12.50%); 66.67% | 0/24 (0.00%); 3/24 (12.50%); 0.00% |
| starter_patterns | 6/16 (37.50%); 16/16 (100.00%); 37.50% | 16/16 (100.00%); 16/16 (100.00%); 100.00% | 4/16 (25.00%); 16/16 (100.00%); 25.00% | 16/16 (100.00%); 16/16 (100.00%); 100.00% |
| starter_transfer | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 4/8 (50.00%); 8/8 (100.00%); 50.00% | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 8/8 (100.00%); 8/8 (100.00%); 100.00% |

### Category results

| Group or category | Starter untrained | Starter trained | Expanded untrained | Expanded trained |
|---|---|---|---|---|
| categories_and_analogies | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |
| domain_context | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 8/8 (100.00%); 8/8 (100.00%); 100.00% | 2/8 (25.00%); 8/8 (100.00%); 25.00% | 8/8 (100.00%); 8/8 (100.00%); 100.00% |
| domain_place | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 8/8 (100.00%); 8/8 (100.00%); 100.00% | 2/8 (25.00%); 8/8 (100.00%); 25.00% | 8/8 (100.00%); 8/8 (100.00%); 100.00% |
| everyday_knowledge | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |
| grammar | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 1/3 (33.33%); 1/3 (33.33%); 100.00% | 0/3 (0.00%); 1/3 (33.33%); 0.00% |
| negation | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |
| new_wording | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 4/8 (50.00%); 8/8 (100.00%); 50.00% | 3/8 (37.50%); 8/8 (100.00%); 37.50% | 8/8 (100.00%); 8/8 (100.00%); 100.00% |
| opposites | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 1/3 (33.33%); 2/3 (66.67%); 50.00% | 0/3 (0.00%); 2/3 (66.67%); 0.00% |
| reference | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |
| sequence | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |
| spatial_relations | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A | 0/3 (0.00%); 0/3 (0.00%); N/A |

### What improved and what failed

- **Gains:** `lang_18`, `lang_19`, `lang_22`, and `lang_23` became correct; every previously correct case stayed correct. All four gains occurred among cases already scorable in the starter.
- **Targeted skills:** extension results stayed at 0/24 across the trained models, while coverage rose from 0 to 3 cases. Within the expanded run, performance on those three cases fell from 2/3 before training to 0/3 afterward.
- **Attribution:** the changed corpus, initialization, and split could each contribute to the observed differences.

| Concrete case | Expanded trained outcome | Interpretation |
|---|---|---|
| `lang_27`: `yesterday she`; expected `walked` | Choice: `walk`; status `scored`; unknown words: none. Continuation: `happy before work .` | All required words were known; the tense pattern failed. |
| `lang_28`: `the opposite of hot is`; expected `cold` | Choice: `heavy`; status `scored`; unknown words: none. Continuation: `river and the orange near the fruit was heavy .` | Contrast vocabulary was present, but transfer to this question form failed. |
| `lang_29`: `the opposite of empty is`; expected `full` | Choice: `soft`; status `scored`; unknown words: none. Continuation: `is late , but the sound of the window was voice was soft .` | The model knew the words and chose the wrong opposite. |
| `lang_25`: `one bird`; expected `is` | Choice: `None`; status `out_of_vocabulary`; unknown words: bird, one. Continuation: `the different consumer yesterday .` | Missing prompt vocabulary prevents a valid four-choice score. |
| `lang_30`: `the opposite of noisy is`; expected `quiet` | Choice: `None`; status `out_of_vocabulary`; unknown words: round. Continuation: `were calm after checking the small , but the room by the garden was quiet .` | A missing distractor makes this case unscorable; it remains in the denominator. |

Choice ranking and free generation can diverge: starter-trained `lang_03` ranked `payment` highest among the choices (0.257614 probability), then sampled `return in detail .`. Expanded-trained `lang_19` selected `return` among the choices and generated `interest .`.

## Full fixed-panel loss measurements

Each row reports mean cross-entropy over non-padding next-token targets in fixed panels of **20 training passages and 20 validation passages**. Lower loss means more probability assigned to the observed next tokens. The small panels provide an estimate of performance within each run.

| Experiment | Step | Training loss | Validation loss |
|---|---:|---:|---:|
| Starter | 0 | 4.926252365112305 | 4.927547931671143 |
| Starter | 1500 | 0.6821381449699402 | 0.7182430624961853 |
| Starter | 3000 | 0.6783124208450317 | 0.7061358690261841 |
| Expanded | 0 | 5.781627655029297 | 5.803586006164551 |
| Expanded | 1500 | 0.7778765559196472 | 0.7909225821495056 |
| Expanded | 3000 | 0.7322326302528381 | 0.7467306852340698 |

**Starter:** [full CSV](llm_runs/20260923T043154_967139Z/training.csv) · [raw values](llm_runs/20260923T043154_967139Z/history.json)

![Starter fixed-panel losses](llm_runs/20260923T043154_967139Z/training_curves.svg)

**Expanded:** [full CSV](llm_runs/20260923T062639_403952Z/training.csv) · [raw values](llm_runs/20260923T062639_403952Z/history.json)

![Expanded fixed-panel losses](llm_runs/20260923T062639_403952Z/training_curves.svg)

Both held-out panel losses fell alongside training loss. Related templates and source files occur on both sides of the split, limiting the result to this narrow distribution. Sample quality could still deteriorate: the expanded model produced `last week the musicians walked through the park .` halfway through training and `last week the musicians low .` at the end. Four samples per checkpoint provide only a small view of fluency.

All four samples at each of steps 0, 1,500 and 3,000 are shown unedited in [learning evidence](docs/learning_evidence.md). Baseline settings are BOS start, temperature 0.8, seed 2026 and at most 32 generated tokens, with EOS stopping. All samples, including garbled outputs, are retained. Each run also preserves 12 temperature samples. Every temperature call resets the same seed and BOS start; differing output lengths can shift the random draws used by later samples.

**Starter sample links:** [samples/step_0000.txt](llm_runs/20260923T043154_967139Z/samples/step_0000.txt) · [samples/step_1500.txt](llm_runs/20260923T043154_967139Z/samples/step_1500.txt) · [samples/step_3000.txt](llm_runs/20260923T043154_967139Z/samples/step_3000.txt) · [temperature_comparison.json](llm_runs/20260923T043154_967139Z/temperature_comparison.json)

**Expanded sample links:** [samples/step_0000.txt](llm_runs/20260923T062639_403952Z/samples/step_0000.txt) · [samples/step_1500.txt](llm_runs/20260923T062639_403952Z/samples/step_1500.txt) · [samples/step_3000.txt](llm_runs/20260923T062639_403952Z/samples/step_3000.txt) · [temperature_comparison.json](llm_runs/20260923T062639_403952Z/temperature_comparison.json)

## How the model learned

The model began with random weights. Each training passage supplied a sequence of next-word prediction problems:

    text → tokens → IDs → embeddings → context → prediction → loss → gradient → update

1. **Read the text.** The corpus is the collection of teaching passages. The tokenizer splits them into lowercase words and punctuation, adding BOS/EOS boundaries and using UNK for missing words.
2. **Look up a vector.** `customer` maps to ID **28** in the starter and **62** in the expanded vocabulary. Each ID selects a learned vector of **64 numbers**. Position embeddings add word-order information.
3. **Use the context.** Attention weights the current and earlier tokens; a causal mask blocks future tokens. For `[BOS, the, customer]`, the starter's first head assigned **[0.485155, 0.423004, 0.091841]** to those positions. Attention and feed-forward layers transform the vectors using learned weights, GELU, normalization, and residual connections.
4. **Learn from the prediction.** Cross-entropy measures how little probability the model gave the observed next token. Backpropagation computes each weight's influence on that loss. AdamW applies an update, and the cycle repeats.

One coordinate of the starter's `customer` embedding shows that process:

| Measurement | Recorded value |
|---|---:|
| Initial coordinate 0 | −0.057591915130615234 |
| First gradient, before clipping | +0.000692586530931294 |
| First effective learning rate | 0.00001 |
| Coordinate after first update | −0.05760190635919571 |
| Coordinate after 3,000 updates | +0.0366341732442379 |

The positive first gradient led to a small downward update. AdamW uses gradient history, adaptive scaling, and weight decay; the run also clips the global gradient norm at 1.0. Repeated updates changed predictions: after `the customer`, the probability of `reviewed` rose from **0.007111448 to 0.178246647**. [Full vectors, updates, and probability tables](docs/learning_evidence.md) trace both runs.

**Then the model generated text.** It sampled a token from `softmax(logits / T)`, appended it to the context, and repeated. Temperature changed sampling while weights stayed fixed:

| Temperature | Sampling effect | Observation |
|---|---|---|
| 0.3 | Concentrates probability on likely tokens | Starter's first sample matched its samples at 0.8 and 1.2 |
| 0.8 | Baseline setting | Expanded output included `last week the musicians low .` |
| 1.2 | Spreads probability more widely | Produced different corpus-style samples; fluency varied |

All comparisons used the same BOS start and sampling seed. The small sample supports inspecting individual changes; a reliable fluency comparison would require more generations.

## Working chat

The next step was to load the expanded model into [chat.py](chat.py) and try four prompts.

- **Model:** expanded run `20260923T062639_403952Z`, trained for 3,000 steps; loads `model.pt` and its saved vocabulary.
- **Behavior:** each prompt starts fresh; the model continues text with fixed weights.
- **Limits:** 48-token context, up to 24 new tokens, temperature 0.8. The interface flags unknown words and truncates long prompts.
- **Identity:** model hash `759c8b1e9c073f74b820d3c962fa108b5793a5b9c67ee263328e01ee6fe9cbd8`.

| Prompt | Actual continuation | Unknown words |
|---|---|---|
| `the customer` | `reviewed the package after checking the price .` | None |
| `the nurse` | `was tired after work .` | None |
| `the small cups` | `the bag in the kitchen , which is big .` | cups |
| `explain quantum teleportation` | `the box of books , which is heavy , while the visitor is light .` | explain, quantum, teleportation |

The familiar customer prompt produced a coherent sentence. The quantum prompt mapped every word to UNK and produced an unrelated continuation. The [recording](evidence/chat_recording.html) and [transcript](evidence/chat_transcript.json) preserve all four interactions.

An independent reviewer launched this interface using only the repository instructions and recorded [eight new interactions](qa/independent_review_20260923/own_chat_recording.html) with [frozen prompts](qa/independent_review_20260923/own_prompts.json) and a [model-identified transcript](qa/independent_review_20260923/own_chat_transcript.json). A [fresh-environment check](qa/independent_review_20260923/clean_uv_setup_checks.json) reproduced three further replies and the complete expanded evaluation.

The same reviewer also recorded [eight interactions with the tuned candidate](qa/independent_review_20260923/selected_chat_recording.html). The [paired comparison](qa/independent_review_20260923/tuning_chat_followup.md) was mixed: the past-tense and opposite-word starts improved, while the doctor/workplace and stone/feather responses regressed. Complete replies still contained malformed clauses and topic drift.

**Launch from the repository root after setup:**

```sh
python chat.py --model llm_runs/20260923T062639_403952Z/model.pt --transcript results/new-chat.json
```

To try the tuned seed-42 candidate through the same interface:

```sh
python chat.py --model tuning_runs/20260923T065357_257422Z/runs/balanced_lr0.003_seed42/model.pt --transcript results/tuned-chat.json
```

Its state hash is `f9adaae02d3d29d3d0a70b572ef6d8a8f7eeff300e87657a0e6a9afeefc68de5`. Both commands save actual replies and model identity. Use a fresh transcript filename for each session.

<details>
<summary>Environment setup, notebook execution, and evaluation commands</summary>

Use Python 3.12 and the recorded dependency versions:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m ipykernel install --prefix .venv --name python3 --display-name "Custom LLM (Python 3.12)"
```

For a fresh checkout with `uv` installed, this alternative supplies Python and seeds `pip`. An independent reviewer verified it in an isolated environment:

```sh
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
source .venv/bin/activate
python -m pip check
python -m ipykernel install --prefix .venv --name python3 --display-name "Custom LLM (Python 3.12)"
```

Open either executed notebook in VS Code/Jupyter, select that environment, and use Run All. Alternatively, run fresh experiments with:

```sh
python scripts/execute_experiment.py starter
python scripts/build_extension_corpus.py --check
python scripts/build_extension_corpus.py
python scripts/execute_experiment.py expanded
```

Keep a copy of the submitted notebooks before rerunning: these commands replace notebook outputs and update the ledger, while retaining timestamped run folders.

The saved models are included in the repository and ZIPs. Replay all four evaluations using fresh output directories:

```sh
python run_evals.py --model llm_runs/20260923T043154_967139Z/model_untrained.pt --stage untrained --output results/starter-untrained
python run_evals.py --model llm_runs/20260923T043154_967139Z/model.pt --output results/starter-final
python run_evals.py --model llm_runs/20260923T062639_403952Z/model_untrained.pt --stage untrained --output results/expanded-untrained
python run_evals.py --model llm_runs/20260923T062639_403952Z/model.pt --output results/expanded-final
```

Rerun either supplemental tuning protocol with its own launcher. Each creates a fresh timestamped study directory and replaces its corresponding supplemental notebook outputs:

```sh
python scripts/execute_tuning.py
python scripts/execute_controlled_tuning.py
```

</details>

![Actual terminal recording replay](evidence/chat_screenshot.png)

*Screenshot of the recorded terminal session. [Reproduction notes](docs/reproduction_notes.md) document its preparation.*

## Limitations

| What happened | What it tells us |
|---|---|
| All 24 extension cases failed after training; only 3 were scorable | Vocabulary coverage and learning the tested relationship are separate hurdles |
| Loss fell while some sentences became malformed | Average next-token fit can improve alongside individual failures |
| Training and validation share templates and source files | The evidence covers a narrow distribution; transfer needs separate testing |
| Nine teaching examples need proofreading: five `a artist` forms and four uses of `which` for people | Correct these in a new corpus version; the saved corpus preserves the text actually trained |

The contrast examples may have transferred poorly to the test's tense and “opposite of” forms. Confidence in that explanation is **low** because a single run cannot isolate the cause.

## Next experiment

1. Correct the five article errors and four human relative-pronoun examples in a new corpus version; review agreement and tense.
2. Freeze a new test suite with unseen situations and phrasing before revising the corpus further.
3. Compare grammar-only and contrasts-only additions across multiple initialization and minibatch-sampling seeds. Report vocabulary coverage, case scores, and free continuations separately; the tuning supplement showed that the seed protocol changes the conclusion.

**Assistance:** Codex helped design the corpus, execute the notebooks and chat demonstration, audit the evidence, and write the report.

## Evidence entry points

This repository contains executed notebooks, saved trained and untrained weights, all 192 original case results, raw samples, and recorded terminal sessions. The original two experiments are preserved; supplemental tuning and independent review are linked separately. Confidence is **high** in the recorded results, **moderate** in the interpretation of learned corpus patterns, and **low** in attributing the original gains solely to the added examples because initialization and split membership also changed.

- [Machine-readable verification report](qa/audit.json) and [audit source](scripts/audit_artifacts.py): reconstruct training-only vocabularies, check splits and source hashes, replay every saved evaluation, verify notebook execution and ZIP contents. The [14 upstream tests](qa/course_tests.log) also passed, and the [documented evaluation CLI](qa/cli_eval_replay.log) reproduced the expanded score.

- **Starter**: [executed notebook](custom_llm.ipynb), [complete run folder](llm_runs/20260923T043154_967139Z/), [results ZIP](llm_runs/20260923T043154_967139Z.zip). Run ID: `20260923T043154_967139Z`.

- **Expanded**: [executed notebook](custom_llm_expanded.ipynb), [complete run folder](llm_runs/20260923T062639_403952Z/), [results ZIP](llm_runs/20260923T062639_403952Z.zip). Run ID: `20260923T062639_403952Z`.

- [Detailed learning evidence](docs/learning_evidence.md): complete 64-coordinate before/after vectors, gradients, probabilities, all timeline and temperature samples, and case-by-case score changes.

- [Actual terminal recording](evidence/chat_recording.html), [recording source](evidence/chat.cast), [terminal text](evidence/chat_terminal.txt), and [chat transcript](evidence/chat_transcript.json). Download/open the HTML locally to play it if GitHub shows its source.

- [Prediction recorded before training](docs/pretraining_plan.md), [corpus rationale and provenance](docs/corpus_design.md), [execution ledger](experiment_runs.json), and [original assignment](ASSIGNMENT.md).

- **Independent review:** [findings](qa/independent_review_20260923/review.md), [machine-readable checks](qa/independent_review_20260923/review_summary.json), and [eight actual new chat interactions](qa/independent_review_20260923/own_chat_recording.html). The reviewer reproduced all 192 cases and both complete training runs.

- **Fixed sources:** [48-case suite](evals/language_evals.json), [evaluation runner](run_evals.py), [starter token/embedding/gradient inspection](llm_runs/20260923T043154_967139Z/inspection.json), and [expanded inspection](llm_runs/20260923T062639_403952Z/inspection.json).

## Actual embedding viewers

Open the [starter viewer](evidence/embeddings_starter.html) or [expanded viewer](evidence/embeddings_expanded.html) locally; each includes the saved checkpoint. The expanded 3D projection retains **22.2%** of pooled variance, while neighbor rankings use all **64 dimensions**. `customer` moved from neighbors `traffic`, `chair`, and `waiting` to `subscriber`, `client`, and `shopper`, reflecting the shared contexts in its training text.

![Expanded customer before training](evidence/embeddings_before.png)

![Expanded customer after training](evidence/embeddings_after.png)

## Controlled tuning supplement

- Six settings compared peak learning rates **0.0003, 0.001, 0.003** with uniform sampling or batches of **16 starter, eight grammar, eight contrast passages**. Corpus, vocabulary, split and architecture stayed fixed.
- Validation selected balanced sampling at **0.003**. At seed 42, category-average loss fell **0.906704 → 0.834300** and the benchmark improved **24/48 → 26/48**, with two opposite-word gains and zero regressions. Coverage stayed **27/48**.
- Confirmation depended on the seed protocol. The candidate remains exploratory; the original baseline remains the default.

| Confirmation protocol | Seed 43: baseline → candidate | Seed 44: baseline → candidate | Category-average validation loss |
|---|---:|---:|---|
| Change initialization; retain seed-42 minibatches | 25 → 25 | 26 → 25 | Increased on both confirmation seeds |
| Change initialization and minibatch sampling | 26 → 26 | 25 → 27 | Decreased on both confirmation seeds |

Both protocols share the seed-42 comparison. Their results are reported separately: [initialization-only findings and recommendations](docs/tuning_findings.md), [full study](tuning_runs/20260923T065432_482471Z/REPORT.md), [executed notebook](custom_llm_tuning.ipynb), [initialization-and-sampling study](docs/tuning_results.md), and [executed notebook](output/jupyter-notebook/custom_llm_tuning.ipynb). Each retains all trials, checkpoints, optimizer states, case results and validation curves.

**One actual update, step 1,000:** the preselected training passage was `a gardener seems calm today .`. AdamW used learning rate `0.00240881256`; the gradient of embedding token `gardener` (ID 101), coordinate 0 was `0.00666711060`.

| Measurement | Before | After |
|---|---:|---:|
| Minibatch loss | 0.652824819 | 0.640562475 |
| P(`calm` after `a gardener seems`) | 21.3535% | 23.4117% |
| Embedding coordinate | -0.124654554 | -0.124877147 |
| Category-average validation loss | 0.853395243 | 0.852127512 |
| Original-panel validation loss | 0.780086398 | 0.781157315 |

The whole network changed during this step. Batch fit and the selected target probability improved; original-panel loss increased. [Saved gradients and AdamW moments](tuning_runs/20260923T065357_257422Z/runs/balanced_lr0.003_seed42/gradient_update.json) reconstruct the coordinate change within `1.5e-9`. An [independent replay audit](tuning_runs/20260923T065357_257422Z/update_replay_audit.json) reproduced all ten recorded updates exactly, including every model weight and optimizer state.
