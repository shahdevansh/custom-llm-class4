# Original extension corpus: grammar and contextual contrasts

This design adds **912 original synthetic passages**: 528 grammar examples and
384 contextual contrast examples. These are planned generator counts; the run's
`corpus_manifest.json` is the authoritative record of what training actually loaded.
The generator is [build_extension_corpus.py](../scripts/build_extension_corpus.py).
Its complete content is visible and deterministic. No external text was copied.
The examples were authored with AI assistance for this assignment, rather than
collected from people, private records, websites, or copyrighted source documents.
They may be included with the project; this statement describes their provenance
and does not claim exclusive ownership of commonplace sentences.

## Why these categories

The starter largely repeats associations between professions, products, places,
and topics. Its frames do not provide much direct practice with subject agreement
or with contrasting qualities of objects and situations. Grammar and opposites
therefore make a focused extension with interpretable learning goals. Selection
of these categories was informed by the published assignment and development
benchmark, so this is a development experiment, not an untouched final test.

The hypothesis is that additional examples can teach useful local grammatical
patterns and contextual contrast vocabulary. The hypothesis does **not** require
an increase in the benchmark score. In particular, a model that sees contrasts in
ordinary descriptions may fail to transfer them to an unfamiliar question form.

## Grammar: 528 passages

- **320 agreement passages:** eight human roles, five states, and eight frames.
  Singular and plural forms have equal representation within this section.
  Frames include `is/are`, `was/were`, `feels/feel`, and `seems/seem`.
- **160 action passages:** eight roles, five everyday actions, and four frames.
  These provide singular present, plural present, past, and progressive forms.
  Actions cover walking, working, waiting, singing, and reading; the latter two
  include irregular past forms or unchanged spelling.
- **48 pronoun passages:** six pronouns, four states, and present/past forms of
  the verb *be*. These include first-, second-, and third-person subjects.

The roles are baker, gardener, artist, runner, helper, visitor, musician, and
farmer. All can plausibly perform the selected ordinary actions. These are short
imagined situations, not factual assertions about particular real people.

An illustrative new teaching sentence is:

> several gardeners seem calm today.

## Contextual contrasts: 384 passages

Sixteen contrast families each have four authored situations and six sentence
frames, for **64 situations and 384 passages**. Families span temperature, amount,
sound, weight, speed, timing, size, length, texture, moisture, hardness, brightness,
thickness, and height. Every passage contains both sides of a contrast, while one
frame presents the scene in reverse order. Sentences use comparisons, `while`,
`but`, relative clauses, and `unlike`; one frame explicitly describes opposite
qualities. They are not question/answer items or lists of benchmark answers.

An illustrative new teaching sentence is:

> unlike the coat near the door, which is wet, the coat near the fire is dry.

Descriptions such as an early bus and a late bus represent a particular imagined
comparison. They do not assert that a given bus route is always early or late.
Words can have more than one sense: *soft* describes sound as well as touch, and
*light* describes weight as well as serving as a noun. The small model may mix
these uses. Some contrast families are relative or context dependent; the corpus
does not imply that every pair is a strict binary in all settings.

## Passage and vocabulary design

The assignment loader splits at sentence endings and newlines, and limits each
passage to 47 word/punctuation tokens. The generator therefore writes exactly one
complete sentence per line, with no multi-sentence stories, headings, metadata,
citations, or instructions inside its training files. It rejects duplicates and
overlong sentences before writing anything. A UTF-8 text source avoids PDF
extraction and reading-order problems.

The script audits the union of extension types and the starter generator's types
against the 509-type limit. It reads only the isolated starter-corpus function for
that check; it does not run the notebook. This is a conservative content audit,
**not** a custom vocabulary construction step. The actual learned vocabulary
must still come only from the notebook's training split. Training and validation
unknown-token rates, retained types, and their actual counts must be taken from
the completed run's `vocabulary_report.json`. Do not substitute these planned
counts for execution evidence.

Run from the repository root after preserving the starter experiment:

```sh
python scripts/build_extension_corpus.py --check
python scripts/build_extension_corpus.py
```

The two generated files are `corpus/expanded/grammar_practice.txt` and
`corpus/expanded/contextual_contrasts.txt`. Use `CORPUS = "classroom"` and
`CORPUS_FOLDER = "corpus/expanded"` for the extension experiment so that only
these intended extra files augment the generated starter corpus. Keep this
design document outside the corpus directory. Save the command output alongside
the experiment evidence to retain the generated file hashes and lexical audit.

## Separation and interpretation

All 48 public cases and the guide were reviewed to avoid copying test stories or
prefixes. The generator never opens the eval JSON, extracts answer words, copies
choices, reads results, or changes the evaluation suite. The vocabulary is not
padded with a missing-word list from the tests. Ordinary vocabulary and underlying
grammatical or semantic knowledge may overlap, as expected for teaching material.
The new passages use human-role activities and independently authored physical
comparisons. They do not reproduce the tests as definitions or paraphrased
questions with an answer attached.

The notebook's separation check remains required after generation. It detects
normalized exact prefixes, not all semantic leakage. Source review adds a useful
check but does not prove the absence of every possible relationship to a public
benchmark. Keep the unchanged suite, outputs, chat logs, this document, and all
analysis files outside every training directory.

These examples are varied within a deliberately limited template system. The
912 passages are **not 912 independent underlying situations**. The random
passage split will place closely related templates and contexts in both training
and validation, so a low held-out loss is evidence about this narrow distribution.
It does not establish transfer to new templates, source files, tasks, or general
language use. Repeated frames could dominate learning, and additional material may
also reduce performance on starter patterns at a fixed training-step budget.

Some development cases can remain unscorable because their complete prompt and
all four choices must be known. Teaching a category is not equivalent to ensuring
that every word in every test appears in the corpus. Report those cases, all-case
success, accuracy on scorable cases, coverage, and free continuations separately.
Do not describe a vocabulary-coverage gain as improved reasoning. Different
vocabularies also change the output layer size and initialization, so identical
seeds do not make two independently initialized models an exact controlled pair.

Confidence is **high** in the transparency and intended provenance of this
construction, **moderate** in its usefulness as focused teaching data, and
**unknown before training** regarding score improvement. The completed
experiments, rather than these expectations, determine the outcome.
