# Selected-checkpoint chat follow-up

**Generation is mixed on these eight fixed prompts.** The selected checkpoint improves two useful local starts—past-tense `was` and the opposite-word answer `hot`—but its complete responses remain malformed or drift away, and two other prompts clearly regress. These interactions do not establish a broad improvement in coherent generation. Confidence is **high** in the observed outputs and identity checks, **low** in generalizing from eight fixed-seed samples.

This is a separate follow-up to the original review. **No new grade is assigned**, and the original `review.md`, frozen prompts and original chat transcript were checked byte-for-byte unchanged after this run. The original 9.6/10 judgment concerns the original snapshot's deliverable completeness, not this selected model's generation quality.

## Exact model and test conditions

- Selected checkpoint: `tuning_runs/20260923T065357_257422Z/runs/balanced_lr0.003_seed42/model.pt`.
- Checkpoint reports **3,000 completed steps**, **319 vocabulary entries**, 2 layers, 4 heads, 64-dimensional embeddings and a 48-token context.
- State hash: `f9adaae02d3d29d3d0a70b572ef6d8a8f7eeff300e87657a0e6a9afeefc68de5`.
- Checkpoint file SHA-256: `092bf04c266d0985491707b6ea9cc3aeafa7d505c58829d351aeacc622fd5169`.
- Saved transcript SHA-256: `3299335f840e7be7e8a9388a9a01913d72b0890517f50f2dd91661cd8d2b1018`.
- The transcript's model hash equals the independently loaded selected checkpoint. The checkpoint file hash is unchanged after chat.
- Eight actual interactions ran through `chat.py` in a pseudo-terminal. The process exited normally. Prompts are exactly the existing frozen `own_prompts.json` prompts; seeds are exactly **2026–2033**, matching the original run, with temperature **0.8**, at most **24 generated tokens**, and fresh context per prompt. No prompt was truncated.
- Unknown-word lists are unchanged on all eight prompts. This follows from actual interface outputs; additional lexical coverage is not responsible for the observed differences.

Command executed:

```sh
source .venv/bin/activate
python chat.py --model tuning_runs/20260923T065357_257422Z/runs/balanced_lr0.003_seed42/model.pt --transcript qa/independent_review_20260923/selected_chat_transcript.json
```

## Paired actual responses

All responses below are unedited. The same prompt and seed appear in each pair.

| ID / prompt | Original expanded model | Selected tuned checkpoint | Unknown words in both |
|---|---|---|---|
| review_01: `the investor reviewed the` | `wall in the sun , which is bright , which is dark .` | `bag of rice , while the kitchen , which is bag in the kitchen .` | investor |
| review_02: `the doctor works at the` | `hospital is short , but the hospital is long .` | `garden is short .` | None |
| review_03: `yesterday the artist` | `is empty , while the visitor is soft .` | `was empty , which is empty , the rice bowl on the table is full .` | None |
| review_04: `the teachers` | `the box in the hall , which is empty , with the office is full .` | `the box in the hall is empty , traffic , peach near the tree is full .` | teachers |
| review_05: `the stone is heavy but the feather is` | `light .` | `heavy , but the stone near the farmer is light .` | feather |
| review_06: `the opposite of cold is` | `loud and the new instructor was loud and the lesson at the school was short .` | `hot , but the soup in the pot in the new pot .` | None |
| review_07: `write a story about a dragon` | `the local customer .` | `our local customer and service .` | dragon, story, write |
| review_08: `two plus two equals` | `the course and the local instructor .` | `the important peach was mentioned in the fruit report yesterday .` | equals, plus, two |

## Qualitative comparison

A second reviewer defined comparison criteria after reading only the frozen prompts and original responses, before inspecting selected outputs: evaluate combined prefix+continuation grammar, coherence, local relevance, and the intended tense/agreement/relation/instruction separately; ignore tokenization spaces; distinguish a correct first word from later degradation; report unknown vocabulary separately. These are qualitative judgments, **not a preregistered benchmark or a new aggregate success percentage**. A full long story is not required within 24 tokens, but an intelligible story opening would count as evidence of following that instruction.

| ID | Direction | Grammar/coherence | Relevance or intended skill |
|---|---|---|---|
| review_01 | **Still fails; grammar worse** | Original has a clause structure but contradictory brightness attributes; selected has a broken while-clause and missing determiner in which is bag. | Neither demonstrates finance context; investor is unknown in both. |
| review_02 | **Worse local relevance; grammar still fails** | Both append a second finite predicate to the doctor works at the ... prefix. | Original at least selects hospital; selected selects garden, losing the intended medical workplace association. |
| review_03 | **Partial local improvement; full response still fails** | was improves the past-tense start compared with is; selected then shifts to is, comma-splices clauses and drifts. | Describing an artist as empty and introducing a rice bowl remains unnatural and unrelated. |
| review_04 | **Still fails** | Selected contains an internal the box ... is empty clause, but it does not connect to the initial the teachers; later traffic, peach fragment is malformed. | Neither produces a coherent continuation or demonstrates plural agreement; teachers is unknown in both. |
| review_05 | **Worse** | Original completes one clean sentence. Selected repeats but and changes the stone-related contrast. | Original heavy stone/light feather becomes heavy feather/light stone; sensible familiar contrast is lost. Feather remains unknown, limiting any knowledge claim in either direction. |
| review_06 | **Partial local improvement; full response still fails** | Selected starts with the correct relation but ends in the incomplete clause but the soup in the pot in the new pot. | hot correctly contrasts cold whereas original loud does not; the remaining text is repetitive and incomplete. |
| review_07 | **Still fails** | Both return noun phrases rather than a story opening with an event or action. | No dragon-related narrative or instruction response; write, story, dragon remain unknown. |
| review_08 | **Still fails** | Selected response alone is a sentence, but combined with two plus two equals it is malformed/semantically incompatible. | Neither supplies a numeric answer; two, plus, equals remain unknown. |

An independent second reader applied the prospectively defined criteria after seeing the selected outputs and reached the same conclusion: **mixed, with no clear overall improvement in these eight continuations**.

The strongest gains occur on fully covered prompts: `yesterday the artist` changes from `is` to `was`, and `the opposite of cold is` changes from `loud` to `hot`. But a correct beginning does not repair the remainder. The cold/hot continuation finishes with no verb after `but`, and the artist response changes tense and switches to a rice bowl.

The regressions matter: `the doctor works at the` changes from a locally appropriate `hospital` to `garden`, while the one clean stone/feather contrast changes from `light .` to `heavy ...`. The feather noun is unknown in both, so neither outcome proves the model represents feather knowledge; the response-quality regression is still directly visible. The story and arithmetic probes remain outside the known prompt vocabulary and continue to fail. The selected run should therefore be described as improving some local learned patterns while leaving broad generation quality unresolved.

The frozen prompts provide a useful side-by-side check, not an unseen final benchmark or a guarantee of future behavior. This follow-up did not inspect or grade the new tuning evaluation score, and it does not isolate which tuning change caused any difference.

## Recorded evidence

- [Actual terminal recording](selected_chat_recording.html)
- [Asciinema recording source](selected_chat.cast)
- [Raw terminal output](selected_chat_terminal.txt)
- [Actual interface transcript](selected_chat_transcript.json)
- [Commands and outputs](selected_chat_command_log.json)
- [Identity/settings/preservation checks](selected_chat_checks.json)
- [Machine-readable qualitative comparison](selected_chat_qualitative_comparison.json)
- [Reproduction script](run_selected_chat_followup.py)

The HTML replays actual captured PTY bytes; it is not a live model or fabricated terminal. No browser playback/screenshot verification was attempted because the prior local-file browser action was policy-blocked. The recording and transcript themselves establish eight actual selected-model interactions.
