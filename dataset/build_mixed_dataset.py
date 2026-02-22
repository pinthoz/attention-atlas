"""
Build a mixed bias dataset combining:
  1. HuggingFace ethical-spectacle/biased-corpus (biased sentences)
  2. HuggingFace ethical-spectacle/gus-dataset-v1 (biased + neutral, token NER)
  3. HuggingFace ethical-spectacle/babe-gus-labels (biased + neutral, token NER)
  4. Local bias_sentences.json (existing dataset, biased + neutral)

Output: a balanced (50/50) JSON file with the same schema as bias_sentences.json
"""

import json
import os
import ast
import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "bias_sentences_mixed.json")
EXISTING_PATH = os.path.join(os.path.dirname(__file__), "bias_sentences.json")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load existing dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading existing dataset...")
with open(EXISTING_PATH, "r", encoding="utf-8") as f:
    existing_data = json.load(f)

existing_entries = existing_data["entries"]
df_existing = pd.DataFrame(existing_entries)
df_existing["source"] = "local"

print(f"   Existing: {len(df_existing)} entries")
print(f"   Biased:   {(df_existing['has_bias'] == True).sum()}")
print(f"   Neutral:  {(df_existing['has_bias'] == False).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load HuggingFace biased-corpus (all biased)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Loading ethical-spectacle/biased-corpus...")
ds_corpus = load_dataset("ethical-spectacle/biased-corpus", split="train")
df_corpus = ds_corpus.to_pandas()

# Map bias type columns to our bias_type categories
bias_cols = [
    "racial", "religious", "gender", "age", "nationality",
    "sexuality", "socioeconomic", "educational", "disability",
    "political", "physical"
]

def get_bias_type_corpus(row):
    """Get the primary bias type from multi-label columns."""
    for col in bias_cols:
        if col in row and row[col] == 1:
            return col
    return "other"

def get_bias_description_corpus(row):
    """Build description from statement_type and target_group."""
    parts = []
    if "statement_type" in row and pd.notna(row["statement_type"]):
        parts.append(str(row["statement_type"]).capitalize())
    if "target_group" in row and pd.notna(row["target_group"]):
        parts.append(f"targeting {row['target_group']}")
    return " ".join(parts) if parts else "Biased statement"

df_corpus_clean = pd.DataFrame({
    "text": df_corpus["biased_text"],
    "has_bias": True,
    "bias_type": df_corpus.apply(get_bias_type_corpus, axis=1),
    "bias_description": df_corpus.apply(get_bias_description_corpus, axis=1),
    "source": "biased-corpus"
})

# Drop empty/null texts
df_corpus_clean = df_corpus_clean.dropna(subset=["text"])
df_corpus_clean = df_corpus_clean[df_corpus_clean["text"].str.strip().str.len() > 0]

print(f"   Loaded: {len(df_corpus_clean)} biased sentences")
print(f"   Bias types: {df_corpus_clean['bias_type'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load HuggingFace gus-dataset-v1 (token NER → sentence-level)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Loading ethical-spectacle/gus-dataset-v1...")
ds_gus = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")
df_gus = ds_gus.to_pandas()

def parse_ner_tags(ner_str):
    """Parse NER tags string to determine bias presence and type."""
    try:
        tags = ast.literal_eval(ner_str) if isinstance(ner_str, str) else ner_str
    except (ValueError, SyntaxError):
        return False, None, None

    has_gen = False
    has_unfair = False
    has_stereo = False

    for word_tags in tags:
        if isinstance(word_tags, list):
            for tag in word_tags:
                if "GEN" in str(tag):
                    has_gen = True
                if "UNFAIR" in str(tag):
                    has_unfair = True
                if "STEREO" in str(tag):
                    has_stereo = True
        elif isinstance(word_tags, str):
            if "GEN" in word_tags:
                has_gen = True
            if "UNFAIR" in word_tags:
                has_unfair = True
            if "STEREO" in word_tags:
                has_stereo = True

    has_bias = has_gen or has_unfair or has_stereo

    if has_bias:
        types = []
        if has_gen:
            types.append("generalization")
        if has_unfair:
            types.append("unfairness")
        if has_stereo:
            types.append("stereotype")
        bias_type = "+".join(types)
        desc = f"GUS-Net: {', '.join(types)}"
    else:
        bias_type = None
        desc = None

    return has_bias, bias_type, desc

gus_rows = []
for _, row in df_gus.iterrows():
    text = row.get("text_str", "")
    if not text or not str(text).strip():
        continue
    has_bias, bias_type, desc = parse_ner_tags(row.get("ner_tags", ""))
    gus_rows.append({
        "text": str(text).strip(),
        "has_bias": has_bias,
        "bias_type": bias_type,
        "bias_description": desc,
        "source": "gus-dataset-v1"
    })

df_gus_clean = pd.DataFrame(gus_rows)
n_biased_gus = (df_gus_clean["has_bias"] == True).sum()
n_neutral_gus = (df_gus_clean["has_bias"] == False).sum()
print(f"   Loaded: {len(df_gus_clean)} sentences")
print(f"   Biased:  {n_biased_gus}")
print(f"   Neutral: {n_neutral_gus}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load HuggingFace babe-gus-labels (token NER → sentence-level)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Loading ethical-spectacle/babe-gus-labels...")
ds_babe = load_dataset("ethical-spectacle/babe-gus-labels", split="train")
df_babe = ds_babe.to_pandas()

babe_rows = []
for _, row in df_babe.iterrows():
    text = row.get("text", "")
    if not text or not str(text).strip():
        continue
    has_bias, bias_type, desc = parse_ner_tags(row.get("ner_tags", ""))
    babe_rows.append({
        "text": str(text).strip(),
        "has_bias": has_bias,
        "bias_type": bias_type,
        "bias_description": desc,
        "source": "babe-gus-labels"
    })

df_babe_clean = pd.DataFrame(babe_rows)
n_biased_babe = (df_babe_clean["has_bias"] == True).sum()
n_neutral_babe = (df_babe_clean["has_bias"] == False).sum()
print(f"   Loaded: {len(df_babe_clean)} sentences")
print(f"   Biased:  {n_biased_babe}")
print(f"   Neutral: {n_neutral_babe}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Combine all sources
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Combining all sources...")

# Standardize existing dataset columns
df_existing_std = pd.DataFrame({
    "text": df_existing["text"],
    "has_bias": df_existing["has_bias"],
    "bias_type": df_existing["bias_type"],
    "bias_description": df_existing["bias_description"],
    "source": df_existing["source"]
})

df_all = pd.concat([
    df_corpus_clean,      # HF biased-corpus (all biased)
    df_gus_clean,         # HF gus-dataset-v1 (mixed)
    df_babe_clean,        # HF babe-gus-labels (mixed)
    df_existing_std       # Local bias_sentences.json (mixed)
], ignore_index=True)

print(f"   Total before dedup: {len(df_all)}")
print(f"   Biased:  {(df_all['has_bias'] == True).sum()}")
print(f"   Neutral: {(df_all['has_bias'] == False).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Deduplicate by text (keep first occurrence)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Deduplicating by text...")

# Normalize text for dedup (lowercase strip)
df_all["text_norm"] = df_all["text"].str.strip().str.lower()
df_all = df_all.drop_duplicates(subset=["text_norm"], keep="first")
df_all = df_all.drop(columns=["text_norm"])

print(f"   After dedup: {len(df_all)}")
n_biased = (df_all["has_bias"] == True).sum()
n_neutral = (df_all["has_bias"] == False).sum()
print(f"   Biased:  {n_biased}")
print(f"   Neutral: {n_neutral}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Balance to 50/50 - USE ALL biased-corpus + neutral from others
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Balancing to 50/50...")

df_biased = df_all[df_all["has_bias"] == True].copy()
df_neutral = df_all[df_all["has_bias"] == False].copy()

# Strategy: Use ALL biased from biased-corpus + biased from other sources
# Then match with neutral sentences from gus, babe, and local
print("\n   Strategy: ALL biased-corpus + neutral from other sources")

# Separate biased by source
df_biased_corpus = df_biased[df_biased["source"] == "biased-corpus"].copy()
df_biased_others = df_biased[df_biased["source"] != "biased-corpus"].copy()

print(f"   Biased from biased-corpus: {len(df_biased_corpus)}")
print(f"   Biased from other sources: {len(df_biased_others)}")

# Use ALL biased sentences (corpus + others)
df_biased_final = df_biased.copy()
target_count = len(df_biased_final)

print(f"\n   Total biased (target): {target_count}")
print(f"   Available neutral: {len(df_neutral)}")

# Sample neutral to match biased count
if len(df_neutral) >= target_count:
    # Prioritize diversity: mix local + HF neutral
    df_neutral_local = df_neutral[df_neutral["source"] == "local"]
    df_neutral_gus = df_neutral[df_neutral["source"] == "gus-dataset-v1"]
    df_neutral_babe = df_neutral[df_neutral["source"] == "babe-gus-labels"]

    print(f"   Neutral sources:")
    print(f"     - local: {len(df_neutral_local)}")
    print(f"     - gus-dataset-v1: {len(df_neutral_gus)}")
    print(f"     - babe-gus-labels: {len(df_neutral_babe)}")

    # Strategy: proportional sampling from all neutral sources
    total_neutral = len(df_neutral)
    samples_local = int(target_count * len(df_neutral_local) / total_neutral)
    samples_gus = int(target_count * len(df_neutral_gus) / total_neutral)
    samples_babe = target_count - samples_local - samples_gus  # remaining

    print(f"\n   Sampling strategy (proportional):")
    print(f"     - local: {samples_local}")
    print(f"     - gus-dataset-v1: {samples_gus}")
    print(f"     - babe-gus-labels: {samples_babe}")

    neutral_parts = []
    if samples_local > 0 and len(df_neutral_local) > 0:
        neutral_parts.append(df_neutral_local.sample(
            n=min(samples_local, len(df_neutral_local)), random_state=42
        ))
    if samples_gus > 0 and len(df_neutral_gus) > 0:
        neutral_parts.append(df_neutral_gus.sample(
            n=min(samples_gus, len(df_neutral_gus)), random_state=42
        ))
    if samples_babe > 0 and len(df_neutral_babe) > 0:
        neutral_parts.append(df_neutral_babe.sample(
            n=min(samples_babe, len(df_neutral_babe)), random_state=42
        ))

    df_neutral_final = pd.concat(neutral_parts, ignore_index=True)

    # If we're short, fill from remaining neutral
    if len(df_neutral_final) < target_count:
        remaining_needed = target_count - len(df_neutral_final)
        used_indices = df_neutral_final.index
        df_neutral_remaining = df_neutral[~df_neutral.index.isin(used_indices)]
        if len(df_neutral_remaining) > 0:
            df_neutral_extra = df_neutral_remaining.sample(
                n=min(remaining_needed, len(df_neutral_remaining)), random_state=42
            )
            df_neutral_final = pd.concat([df_neutral_final, df_neutral_extra], ignore_index=True)
else:
    # Not enough neutral - use all available
    print(f"   WARNING: Not enough neutral ({len(df_neutral)}) to match biased ({target_count})")
    print(f"   Using all available neutral sentences")
    df_neutral_final = df_neutral.copy()

df_final = pd.concat([df_biased_final, df_neutral_final], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n   Final dataset: {len(df_final)}")
print(f"   Biased:  {(df_final['has_bias'] == True).sum()}")
print(f"   Neutral: {(df_final['has_bias'] == False).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Build output JSON with same schema as bias_sentences.json
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Building output JSON...")

entries = []
for idx, row in df_final.iterrows():
    entry = {
        "id": idx + 1,
        "type": "single",
        "text": str(row["text"]).strip(),
        "has_bias": bool(row["has_bias"]),
        "bias_type": row["bias_type"] if pd.notna(row.get("bias_type")) else None,
        "bias_description": row["bias_description"] if pd.notna(row.get("bias_description")) else None,
        "source": row["source"] if pd.notna(row.get("source")) else "unknown",
    }
    entries.append(entry)

output = {"entries": entries}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"   Saved to: {OUTPUT_PATH}")
print(f"   Total entries: {len(entries)}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Summary")
print("=" * 60)

# Source breakdown
print("\nSource breakdown:")
for source, group in df_final.groupby("source"):
    n = len(group)
    n_b = (group["has_bias"] == True).sum()
    n_n = (group["has_bias"] == False).sum()
    print(f"  {source:<20} {n:>5} total  ({n_b:>5} biased, {n_n:>5} neutral)")

# Bias type breakdown (for biased entries)
print("\nBias type breakdown (biased entries only):")
biased_entries = df_final[df_final["has_bias"] == True]
print(biased_entries["bias_type"].value_counts().to_string())

print(f"\nUnique texts: {df_final['text'].nunique()} / {len(df_final)}")
print("\nDone!")
