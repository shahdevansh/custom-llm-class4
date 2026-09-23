#!/usr/bin/env python3
"""Build original, AI-assisted teaching sentences for grammar and contrasts.

This deterministic generator does not open the evaluation suite, saved results,
chat transcripts, or model files. Its hand-authored content is fully visible here.
The starter generator is read in isolation only for a conservative lexical audit;
the training notebook still builds the actual vocabulary from its training split.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = r"\w+(?:['’]\w+)*|[^\w\s]"

# Human roles keep the same actions plausible across singular and plural subjects.
SUBJECTS = (
    ("baker", "bakers"),
    ("gardener", "gardeners"),
    ("artist", "artists"),
    ("runner", "runners"),
    ("helper", "helpers"),
    ("visitor", "visitors"),
    ("musician", "musicians"),
    ("farmer", "farmers"),
)
STATES = ("ready", "tired", "happy", "calm", "busy")
AGREEMENT_FRAMES = (
    "the {singular} is {state} before work.",
    "the {plural} are {state} before work.",
    "this {singular} was {state} after lunch.",
    "these {plural} were {state} after lunch.",
    "our {singular} feels {state} this morning.",
    "our {plural} feel {state} this morning.",
    "a {singular} seems {state} today.",
    "several {plural} seem {state} today.",
)

# (third-person present, base form, past, progressive, ordinary circumstance)
ACTIONS = (
    ("walks", "walk", "walked", "walking", "through the park"),
    ("works", "work", "worked", "working", "in the garden"),
    ("waits", "wait", "waited", "waiting", "near the gate"),
    ("sings", "sing", "sang", "singing", "after lunch"),
    ("reads", "read", "read", "reading", "before lunch"),
)
PRONOUNS = (
    ("i", "am", "was"),
    ("we", "are", "were"),
    ("you", "are", "were"),
    ("he", "is", "was"),
    ("she", "is", "was"),
    ("they", "are", "were"),
)

# Each family has four separately authored situations. These are observations
# within a described situation, not universal claims about the named objects.
# All noun phrases are singular, so every frame below has correct agreement.
CONTRASTS = (
    ("hot", "cold", (
        ("the soup in the pot", "the soup in the bowl"),
        ("the tea by the fire", "the tea by the window"),
        ("the water in the pan", "the water in the jug"),
        ("the sand in the sun", "the sand in the shade"),
    )),
    ("warm", "cool", (
        ("the room near the kitchen", "the room near the garden"),
        ("the cloth on the chair", "the cloth on the floor"),
        ("the air near the fire", "the air near the door"),
        ("the milk in the cup", "the milk in the bottle"),
    )),
    ("empty", "full", (
        ("the soup bowl", "the rice bowl"),
        ("the basket near the gate", "the basket near the tree"),
        ("the cup on the shelf", "the cup on the table"),
        ("the box in the hall", "the box in the office"),
    )),
    ("noisy", "quiet", (
        ("the busy kitchen", "the small garden"),
        ("the street near the school", "the path near the pond"),
        ("the room near the market", "the room near the field"),
        ("the large bus", "the small car"),
    )),
    ("loud", "soft", (
        ("the sound of the bell", "the sound of the rain"),
        ("the music in the hall", "the music in the room"),
        ("the voice of the runner", "the voice of the visitor"),
        ("the song near the stage", "the song near the gate"),
    )),
    ("heavy", "light", (
        ("the bag of rice", "the bag of cloth"),
        ("the large stone", "the small stone"),
        ("the chair in the hall", "the chair in the garden"),
        ("the box of books", "the box of paper"),
    )),
    ("fast", "slow", (
        ("the new train", "the old bus"),
        ("the runner near the gate", "the runner near the pond"),
        ("the car on the road", "the car on the hill"),
        ("the water in the river", "the water in the small stream"),
    )),
    ("early", "late", (
        ("the first visitor", "the last visitor"),
        ("the bus from the market", "the bus from the station"),
        ("the morning delivery", "the evening delivery"),
        ("the first payment", "the last payment"),
    )),
    ("big", "small", (
        ("the house near the river", "the house near the pond"),
        ("the bowl on the table", "the bowl on the shelf"),
        ("the tree near the school", "the tree near the house"),
        ("the bag in the hall", "the bag in the kitchen"),
    )),
    ("long", "short", (
        ("the path through the field", "the path through the garden"),
        ("the rope by the tree", "the rope by the gate"),
        ("the song at the start", "the song at the end"),
        ("the coat on the chair", "the coat on the shelf"),
    )),
    ("rough", "smooth", (
        ("the stone near the path", "the stone near the river"),
        ("the wood near the gate", "the wood near the table"),
        ("the paper in the box", "the paper in the book"),
        ("the road through the field", "the road through the town"),
    )),
    ("wet", "dry", (
        ("the coat near the door", "the coat near the fire"),
        ("the grass in the shade", "the grass in the sun"),
        ("the cloth in the sink", "the cloth on the shelf"),
        ("the sand near the water", "the sand near the road"),
    )),
    ("hard", "soft", (
        ("the seat by the window", "the seat by the door"),
        ("the bread on the shelf", "the bread on the table"),
        ("the ground near the gate", "the ground near the pond"),
        ("the old brush", "the new brush"),
    )),
    ("bright", "dark", (
        ("the room by the window", "the room by the hall"),
        ("the sky in the morning", "the sky at night"),
        ("the path by the light", "the path by the tree"),
        ("the wall in the sun", "the wall in the shade"),
    )),
    ("thick", "thin", (
        ("the book on the shelf", "the book on the table"),
        ("the rope by the pond", "the rope by the house"),
        ("the soup in the bowl", "the soup in the pan"),
        ("the cloth on the floor", "the cloth on the chair"),
    )),
    ("high", "low", (
        ("the shelf by the window", "the shelf by the door"),
        ("the branch near the house", "the branch near the pond"),
        ("the price at the store", "the price at the market"),
        ("the wall near the road", "the wall near the garden"),
    )),
)
CONTRAST_FRAMES = (
    "{left} is {first}, while {right} is {second}.",
    "{right} is {second}, but {left} is {first}.",
    "we described {left} as {first} and {right} as {second}.",
    "we compared {left}, which was {first}, with {right}, which was {second}.",
    "we noticed opposite qualities because {left} was {first} and {right} was {second}.",
    "unlike {left}, which is {first}, {right} is {second}.",
)


def word_tokens(text: str) -> list[str]:
    return re.findall(TOKEN_PATTERN, text.lower(), flags=re.UNICODE)


def build_grammar() -> list[str]:
    sentences = []
    for singular, plural in SUBJECTS:
        for state in STATES:
            for frame in AGREEMENT_FRAMES:
                sentences.append(frame.format(singular=singular, plural=plural, state=state))
        for present, base, past, progressive, circumstance in ACTIONS:
            sentences.extend((
                f"each {singular} {present} {circumstance} every day.",
                f"many {plural} {base} {circumstance} every day.",
                f"last week the {plural} {past} {circumstance}.",
                f"our {singular} is {progressive} {circumstance} today.",
            ))
    for pronoun, present, past in PRONOUNS:
        for state in ("ready", "tired", "hungry", "happy"):
            sentences.extend((
                f"{pronoun} {present} {state} before work.",
                f"{pronoun} {past} {state} after work.",
            ))
    return sentences


def build_contrasts() -> list[str]:
    return [
        frame.format(left=left, right=right, first=first, second=second)
        for first, second, scenes in CONTRASTS
        for left, right in scenes
        for frame in CONTRAST_FRAMES
    ]


def starter_text_for_lexical_audit() -> str:
    """Read only the pure classroom generator, never execute the notebook module."""
    source = ast.parse((ROOT / "custom_llm.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == "classroom_corpus")
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(isolated), "<isolated classroom generator>", "exec"), namespace)
    return namespace["classroom_corpus"]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate and print statistics without writing files")
    args = parser.parse_args()
    materials = {
        "grammar_practice.txt": build_grammar(),
        "contextual_contrasts.txt": build_contrasts(),
    }
    assert len(materials["grammar_practice.txt"]) == 528
    assert len(materials["contextual_contrasts.txt"]) == 384
    all_sentences = [sentence for sentences in materials.values() for sentence in sentences]
    normalized = [" ".join(word_tokens(sentence)) for sentence in all_sentences]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate normalized teaching passages found.")
    if any(len(word_tokens(sentence)) > 47 for sentence in all_sentences):
        raise ValueError("A teaching sentence would exceed one classroom passage.")
    if any(len(re.split(r"(?<=[.!?])\s+|\n+", sentence)) != 1 for sentence in all_sentences):
        raise ValueError("A teaching sentence would split into more than one passage.")

    extension_types = set(word_tokens("\n".join(all_sentences)))
    combined_types = extension_types | set(word_tokens(starter_text_for_lexical_audit()))
    if len(combined_types) > 509:
        raise ValueError(f"Combined teaching vocabulary has {len(combined_types)} types; limit is 509.")

    output_dir = ROOT / "corpus" / "expanded"
    statistics = {
        "mode": "check_only" if args.check else "write",
        "provenance": "original, AI-assisted, deterministic synthetic teaching text",
        "unique_extension_passages": len(normalized),
        "extension_token_types": len(extension_types),
        "starter_plus_extension_token_types_before_split": len(combined_types),
        "max_passage_tokens": max(map(lambda sentence: len(word_tokens(sentence)), all_sentences)),
        "files": [],
        "note": "Lexical counts are audits, not the learned vocabulary; the notebook uses its training split only.",
    }
    for filename, sentences in materials.items():
        text = "\n".join(sentences) + "\n"
        statistics["files"].append({
            "path": str((output_dir / filename).relative_to(ROOT)),
            "passages": len(sentences),
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        if not args.check:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / filename).write_text(text, encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
