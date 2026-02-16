"""Download HF dataset and fix punctuation labels.

Punctuation tokens (., ?, !, ;, :, etc.) should never carry bias labels.
In the original dataset, punctuation is often attached to words (e.g. "week?",
"starve.", "aid,") and inherits the word's bias tags.

This script:
  1. Downloads ethical-spectacle/gus-dataset-v1 from Hugging Face
  2. Splits trailing/leading punctuation into separate word entries
  3. Sets all punctuation-only entries to ["O"]
  4. Repairs BIO consistency: when an O-labeled punctuation breaks a span,
     the next token's I-tags are promoted to B-tags
  5. Saves the cleaned dataset as JSON

Usage:
    python fix_punctuation_labels.py
"""

import ast
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────────

OUTPUT_PATH = Path(__file__).parent.parent.parent / "dataset" / "gus_dataset_clean.json"

# Characters that count as punctuation (never bias carriers)
_TRAILING_RE = re.compile(r'^(.+?)([.,!?;:"\')…]+)$')
_LEADING_RE = re.compile(r'^(["\'\(]+)(.+)$')
_PURE_PUNCT_RE = re.compile(r'^[^\w]+$')


def is_punctuation(word: str) -> bool:
    """True if word contains no alphanumeric characters."""
    return bool(_PURE_PUNCT_RE.match(word))


def split_punctuation(word: str, tags: list) -> list:
    """Split a word into (leading_punct, core, trailing_punct) entries.

    Returns a list of (word, tags) tuples.  Punctuation parts always
    get ["O"], the core part keeps original tags.
    """
    # Already pure punctuation
    if is_punctuation(word):
        return [(word, ["O"])]

    results = []

    # Leading punctuation (quotes, parentheses)
    lead_match = _LEADING_RE.match(word)
    if lead_match:
        lead_punct, word = lead_match.group(1), lead_match.group(2)
        results.append((lead_punct, ["O"]))

    # Trailing punctuation
    trail_match = _TRAILING_RE.match(word)
    if trail_match:
        core, trail_punct = trail_match.group(1), trail_match.group(2)
        # Only split if core still has alphanumeric content
        if re.search(r'\w', core):
            results.append((core, tags))
            results.append((trail_punct, ["O"]))
        else:
            # Edge case: core is also punctuation
            results.append((core, ["O"]))
            results.append((trail_punct, ["O"]))
    else:
        results.append((word, tags))

    return results


def repair_bio_continuity(tags: list) -> tuple:
    """Fix BIO violations after punctuation gets O labels.

    When a punctuation token (now ["O"]) breaks a span, the next token
    that still has I-X tags becomes invalid — it needs B-X instead.

    For each bias type (STEREO, GEN, UNFAIR) we track whether a span is
    "open".  If we hit an O token the span closes.  If the next non-O
    token has I-X without an open span, we promote I-X → B-X.

    Returns (repaired_tags, num_promotions).
    """
    # All bias types in the dataset
    BIAS_TYPES = ("STEREO", "GEN", "UNFAIR")

    # Track which spans are currently open
    span_open = {bt: False for bt in BIAS_TYPES}
    promotions = 0

    repaired = []
    for tag_list in tags:
        new_tags = list(tag_list)

        # Check each bias type
        for bt in BIAS_TYPES:
            b_tag = f"B-{bt}"
            i_tag = f"I-{bt}"

            has_b = b_tag in new_tags
            has_i = i_tag in new_tags

            if has_b:
                # B-tag opens/resets the span
                span_open[bt] = True
            elif has_i:
                if not span_open[bt]:
                    # I-tag without open span → promote to B-tag
                    new_tags = [b_tag if t == i_tag else t for t in new_tags]
                    span_open[bt] = True
                    promotions += 1
                # else: span is open, I-tag is valid
            else:
                # This token has neither B nor I for this type → span closes
                span_open[bt] = False

        repaired.append(new_tags)

    return repaired, promotions


def process_dataset():
    """Download, clean, and save the dataset."""
    print("Downloading ethical-spectacle/gus-dataset-v1 from Hugging Face...")
    ds = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")
    print(f"Loaded {len(ds)} examples")

    cleaned = []
    stats = {
        "total_examples": len(ds),
        "punct_split": 0,        # words where punctuation was separated
        "punct_standalone": 0,   # pure punctuation tokens fixed to O
        "labels_changed": 0,     # punctuation tokens that had non-O labels
        "bio_promotions": 0,     # I-tags promoted to B-tags after span break
    }

    for entry in ds:
        text = entry["text_str"]
        try:
            tags = ast.literal_eval(entry["ner_tags"])
        except (ValueError, SyntaxError):
            continue

        words = text.split()
        if len(words) != len(tags):
            min_len = min(len(words), len(tags))
            words = words[:min_len]
            tags = tags[:min_len]

        new_words = []
        new_tags = []

        for word, tag in zip(words, tags):
            parts = split_punctuation(word, tag)

            if len(parts) > 1:
                stats["punct_split"] += 1

            for part_word, part_tag in parts:
                if is_punctuation(part_word):
                    if tag != ["O"]:
                        stats["labels_changed"] += 1
                    stats["punct_standalone"] += 1
                new_words.append(part_word)
                new_tags.append(part_tag)

        # Repair BIO: promote orphaned I-tags to B-tags after span breaks
        repaired_tags, promos = repair_bio_continuity(new_tags)
        stats["bio_promotions"] += promos

        cleaned.append({
            "text_str": " ".join(new_words),
            "ner_tags": repaired_tags,
            "id": entry.get("id", -1),
        })

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(cleaned)} examples to {OUTPUT_PATH}")
    print(f"\nStats:")
    print(f"  Words split (punct separated):  {stats['punct_split']}")
    print(f"  Punctuation tokens (total):     {stats['punct_standalone']}")
    print(f"  Labels changed (non-O -> O):    {stats['labels_changed']}")
    print(f"  BIO promotions (I -> B):        {stats['bio_promotions']}")


if __name__ == "__main__":
    process_dataset()
