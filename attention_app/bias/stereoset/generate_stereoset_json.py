"""One-time script to generate pre-computed StereoSet evaluation data.

Loads the StereoSet intersentence benchmark from HuggingFace, computes
bias scores for BERT (PLL) and/or GPT-2 (autoregressive log-likelihood),
extracts head sensitivity metrics, and saves to model-specific JSON files.

Usage:
    python -m attention_app.bias.stereoset.generate_stereoset_json              # both models
    python -m attention_app.bias.stereoset.generate_stereoset_json --model bert  # BERT only
    python -m attention_app.bias.stereoset.generate_stereoset_json --model gpt2  # GPT-2 only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from scipy.stats import kruskal

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from attention_app.models import ModelManager
from attention_app.bias.feature_extraction_notebooks import extract_features_for_sentence

CATEGORIES = ["gender", "race", "religion", "profession"]
OUTPUT_DIR = Path(__file__).parent / "results"

# Model configurations
MODEL_CONFIGS = {
    "bert": {
        "model_name": "bert-base-uncased",
        "scoring": "pll",
        "n_layers": 12,
        "n_heads": 12,
        "output_file": "stereoset_precomputed_bert.json",
    },
    "bert_large": {
        "model_name": "bert-large-uncased",
        "scoring": "pll",
        "n_layers": 24,
        "n_heads": 16,
        "output_file": "stereoset_precomputed_bert_large.json",
    },
    "gpt2": {
        "model_name": "gpt2",
        "scoring": "autoregressive",
        "n_layers": 12,
        "n_heads": 12,
        "output_file": "stereoset_precomputed_gpt2.json",
    },
    "gpt2_medium": {
        "model_name": "gpt2-medium",
        "scoring": "autoregressive",
        "n_layers": 24,
        "n_heads": 16,
        "output_file": "stereoset_precomputed_gpt2_medium.json",
    },
}


# ── Scoring functions ─────────────────────────────────────────────────


def score_sentence_pll(text, model, tokenizer):
    """Compute Pseudo-Log-Likelihood (PLL) for BERT.

    Masks each token in turn and averages the log-probability of the
    correct token.  Higher = more "natural" to the model.
    """
    tokenize_input = tokenizer(text, return_tensors="pt")
    if tokenize_input["input_ids"].shape[1] > 512:
        return -9999.0

    input_ids = tokenize_input["input_ids"].to(model.device)
    token_type_ids = tokenize_input.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(model.device)
    attention_mask = tokenize_input["attention_mask"].to(model.device)

    with torch.no_grad():
        loss = 0.0
        seq_len = input_ids.shape[1]
        for i in range(1, seq_len - 1):
            target_id = input_ids[0, i].clone()
            input_ids[0, i] = tokenizer.mask_token_id
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            log_prob = F.log_softmax(outputs.logits[0, i], dim=0)[target_id]
            loss += log_prob.item()
            input_ids[0, i] = target_id
    return loss / max(seq_len - 2, 1)


def score_sentence_autoregressive(text, model, tokenizer):
    """Compute autoregressive log-likelihood for GPT-2.

    Feeds the full sequence and computes the average log-probability of
    each token given its preceding context.  Higher = more "natural".
    """
    inputs = tokenizer(text, return_tensors="pt")
    if inputs["input_ids"].shape[1] > 1024:
        return -9999.0

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab]

    # Shift: predict token t from logits at position t-1
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Average across all predicted tokens
    avg_ll = token_log_probs.mean().item()
    return avg_ll


def _load_scorer(model_key, device):
    """Load the appropriate scoring model + tokenizer."""
    from transformers import AutoTokenizer

    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["model_name"]

    if cfg["scoring"] == "pll":
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(model_name).to(device)
    else:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _score_sentence(text, model, tokenizer, scoring_method):
    """Dispatch to the correct scoring function."""
    if scoring_method == "pll":
        return score_sentence_pll(text, model, tokenizer)
    return score_sentence_autoregressive(text, model, tokenizer)


# ── Aggregation helpers ───────────────────────────────────────────────


def _compute_aggregate_scores(examples):
    """Compute SS, LMS, ICAT overall and per-category."""
    n = len(examples)
    if n == 0:
        return {"overall": {}, "by_category": {}}

    stereo_preferred = sum(1 for e in examples if e["stereo_pll"] > e["anti_pll"])
    ss = (stereo_preferred / n * 100)

    n_unrel = sum(1 for e in examples if e["unrelated_pll"] > -9999.0)
    meaningful = sum(
        1 for e in examples
        if max(e["stereo_pll"], e["anti_pll"]) > e["unrelated_pll"]
        and e["unrelated_pll"] > -9999.0
    )
    lms = (meaningful / n_unrel * 100) if n_unrel > 0 else 100.0
    icat = lms * min(ss, 100 - ss) / 50.0
    mean_bias = float(np.mean([e["bias_score"] for e in examples]))

    overall = {
        "ss": round(ss, 2), "lms": round(lms, 2), "icat": round(icat, 2),
        "n": n, "mean_bias_score": round(mean_bias, 6),
    }

    by_category = {}
    for cat in CATEGORIES:
        cat_ex = [e for e in examples if e["category"] == cat]
        cat_n = len(cat_ex)
        if cat_n == 0:
            continue
        cat_stereo = sum(1 for e in cat_ex if e["stereo_pll"] > e["anti_pll"])
        cat_ss = cat_stereo / cat_n * 100
        cat_n_unrel = sum(1 for e in cat_ex if e["unrelated_pll"] > -9999.0)
        cat_meaningful = sum(
            1 for e in cat_ex
            if max(e["stereo_pll"], e["anti_pll"]) > e["unrelated_pll"]
            and e["unrelated_pll"] > -9999.0
        )
        cat_lms = (cat_meaningful / cat_n_unrel * 100) if cat_n_unrel > 0 else 100.0
        cat_icat = cat_lms * min(cat_ss, 100 - cat_ss) / 50.0
        cat_mean_bias = float(np.mean([e["bias_score"] for e in cat_ex]))
        by_category[cat] = {
            "ss": round(cat_ss, 2), "lms": round(cat_lms, 2), "icat": round(cat_icat, 2),
            "n": cat_n, "mean_bias_score": round(cat_mean_bias, 6),
        }

    return {"overall": overall, "by_category": by_category}


def _compute_head_sensitivity(df, num_layers=12, num_heads=12):
    """Compute 12×12 head sensitivity matrix and top heads.

    Uses ALL features per head (GAM, AttMap, Spec) to compute
    aggregate sensitivity, not just AttMap_mean.
    """
    import re

    # Find all head-level columns and group by (layer, head)
    head_pattern = re.compile(r"_L(\d+)_H(\d+)_")
    head_features = {}  # (layer, head) -> list of column names
    for col in df.columns:
        m = head_pattern.search(col)
        if m:
            layer, head = int(m.group(1)), int(m.group(2))
            if layer < num_layers and head < num_heads:
                head_features.setdefault((layer, head), []).append(col)

    matrix = [[0.0] * num_heads for _ in range(num_layers)]
    records = []

    for (layer, head), cols in head_features.items():
        # For each feature of this head, compute variance of category means
        variances = []
        correlations = []
        for col in cols:
            vals = df[col]
            if vals.var() == 0:
                continue
            cat_means = df.groupby("category")[col].mean()
            variances.append(float(cat_means.var()))
            if "bias_score" in df.columns:
                corr = df[[col, "bias_score"]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if not variances:
            continue

        # Aggregate: mean variance across all features for this head
        agg_variance = float(np.mean(variances))
        agg_correlation = float(np.mean(correlations)) if correlations else 0.0

        matrix[layer][head] = agg_variance

        # Category means for the top-variance feature of this head
        best_col = cols[np.argmax(variances)] if variances else cols[0]
        cat_means = df.groupby("category")[best_col].mean()
        cat_means_dict = {cat: round(float(cat_means.get(cat, 0)), 8) for cat in CATEGORIES}

        records.append({
            "layer": layer, "head": head,
            "variance": round(agg_variance, 10),
            "correlation": round(agg_correlation, 6),
            "n_features": len(cols),
            "best_feature": best_col,
            "category_means": cat_means_dict,
        })

    records.sort(key=lambda h: h["variance"], reverse=True)
    return matrix, records[:10]


def _compute_top_features(df):
    """Kruskal-Wallis test on all numeric features."""
    df_numeric = df.select_dtypes(include=[np.number]).fillna(0)
    df_numeric = df_numeric.loc[:, df_numeric.var() > 0]
    feature_cols = [c for c in df_numeric.columns if c not in ("bias_score", "stereo_prob", "anti_prob")]
    y = df["category"]

    kw_results = {}
    for col in feature_cols:
        groups = [df_numeric[y == cat][col].values for cat in CATEGORIES]
        try:
            _, p_val = kruskal(*groups)
            kw_results[col] = p_val
        except Exception:
            kw_results[col] = 1.0

    sorted_feats = sorted(kw_results.items(), key=lambda x: x[1])
    top_features = [{"name": name, "p_value": float(pval)} for name, pval in sorted_feats[:20]]
    sig_count = sum(1 for _, pval in kw_results.items() if pval < 0.001)
    return top_features, sig_count, len(kw_results)


def _build_head_profile_stats(sensitive_heads, df):
    """Compute population statistics for each sensitive head's best feature.

    Returns dict keyed by 'L{l}_H{h}' with mean, std, min, max,
    and per-category means for the best feature of that head.
    """
    stats = {}
    for rec in sensitive_heads:
        feat = rec.get("best_feature")
        if not feat or feat not in df.columns:
            continue
        key = f"L{rec['layer']}_H{rec['head']}"
        vals = df[feat].dropna()
        cat_means = df.groupby("category")[feat].mean()
        stats[key] = {
            "feature": feat,
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "min": round(float(vals.min()), 6),
            "max": round(float(vals.max()), 6),
            "cat_means": {cat: round(float(cat_means.get(cat, 0)), 6) for cat in CATEGORIES if cat in cat_means.index},
        }
    return stats


def _extract_head_profile(feats, sensitive_heads):
    """Extract the best-feature value for each sensitive head from a feature dict.

    Returns dict like {'L1_H3': 0.452, 'L8_H7': 0.231, ...}.
    """
    profile = {}
    for rec in sensitive_heads:
        feat = rec.get("best_feature")
        if not feat:
            continue
        key = f"L{rec['layer']}_H{rec['head']}"
        val = feats.get(feat)
        if val is not None:
            profile[key] = round(float(val), 6)
    return profile


# ── Main pipeline ─────────────────────────────────────────────────────


def run_for_model(model_key, stereoset):
    """Run the full pipeline for a single model and save JSON."""
    import pandas as pd

    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["model_name"]
    scoring_method = cfg["scoring"]
    output_path = OUTPUT_DIR / cfg["output_file"]

    scoring_label = "PLL (masked)" if scoring_method == "pll" else "Autoregressive LL"
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}  |  Scoring: {scoring_label}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load scorer
    print(f"\n  Loading scoring model ({model_name})...")
    scorer_model, scorer_tokenizer = _load_scorer(model_key, device)

    # Load attention model via ModelManager
    print(f"  Loading attention model ({model_name})...")
    manager = ModelManager()
    manager.get_model(model_name)
    print(f"  Device: {device}")

    # Process examples
    print(f"\n  Computing scores and extracting features...")
    examples = []
    features_list = []
    example_feats_pairs = []

    for example in tqdm(stereoset, desc=f"  {model_key}"):
        try:
            context = example["context"]
            category = example["bias_type"]
            target = example.get("target", "")
            sentences = example["sentences"]
            gold_labels = sentences["gold_label"]

            # StereoSet gold_label encoding:
            #   0 = anti-stereotype, 1 = stereotype, 2 = unrelated
            stereo_idx = anti_idx = unrelated_idx = -1
            for idx, lbl in enumerate(gold_labels):
                if lbl == 1:
                    stereo_idx = idx
                elif lbl == 0:
                    anti_idx = idx
                elif lbl == 2:
                    unrelated_idx = idx

            if stereo_idx == -1 or anti_idx == -1:
                continue

            stereo_sentence = sentences["sentence"][stereo_idx]
            anti_sentence = sentences["sentence"][anti_idx]
            unrelated_sentence = sentences["sentence"][unrelated_idx] if unrelated_idx != -1 else ""

            stereo_text = context + " " + stereo_sentence
            anti_text = context + " " + anti_sentence
            unrelated_text = (context + " " + unrelated_sentence) if unrelated_sentence else ""

            stereo_score = _score_sentence(stereo_text, scorer_model, scorer_tokenizer, scoring_method)
            anti_score = _score_sentence(anti_text, scorer_model, scorer_tokenizer, scoring_method)
            unrelated_score = (
                _score_sentence(unrelated_text, scorer_model, scorer_tokenizer, scoring_method)
                if unrelated_text else -9999.0
            )
            bias_score = stereo_score - anti_score

            examples.append({
                "category": category,
                "target": target,
                "context": context,
                "stereo_sentence": stereo_sentence,
                "anti_sentence": anti_sentence,
                "unrelated_sentence": unrelated_sentence,
                "stereo_pll": round(stereo_score, 6),
                "anti_pll": round(anti_score, 6),
                "unrelated_pll": round(unrelated_score, 6),
                "bias_score": round(bias_score, 6),
            })

            feats = extract_features_for_sentence(stereo_text, model_name, manager)
            feats["category"] = category
            feats["bias_score"] = bias_score
            features_list.append(feats)

            # Extract anti-sentence features for head profiles
            try:
                anti_feats = extract_features_for_sentence(anti_text, model_name, manager)
            except Exception:
                anti_feats = None
            example_feats_pairs.append((feats, anti_feats))

        except Exception:
            continue

    print(f"  Processed {len(examples)} examples")

    # Free scorer model to save memory before aggregation
    del scorer_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate
    print("  Computing aggregate scores...")
    scores = _compute_aggregate_scores(examples)

    print("  Computing head sensitivity...")
    df = pd.DataFrame(features_list)
    n_layers = cfg.get("n_layers", 12)
    n_heads = cfg.get("n_heads", 12)
    matrix, sensitive_heads = _compute_head_sensitivity(df, n_layers, n_heads)

    print("  Building per-example head profiles...")
    head_profile_stats = _build_head_profile_stats(sensitive_heads, df)
    for i, (stereo_feats, anti_feats) in enumerate(example_feats_pairs):
        stereo_profile = _extract_head_profile(stereo_feats, sensitive_heads)
        anti_profile = _extract_head_profile(anti_feats, sensitive_heads) if anti_feats else {}
        examples[i]["head_profile"] = {"stereo": stereo_profile, "anti": anti_profile}

    print("  Running Kruskal-Wallis tests...")
    top_features, sig_count, total_tested = _compute_top_features(df)

    # Save
    result = {
        "metadata": {
            "model": model_name,
            "scoring_method": scoring_method,
            "total_examples": len(examples),
            "significant_features": sig_count,
            "total_features_tested": total_tested,
            "date": datetime.now().isoformat(),
        },
        "scores": scores,
        "sensitive_heads": sensitive_heads,
        "top_features": top_features,
        "head_sensitivity_matrix": matrix,
        "head_profile_stats": head_profile_stats,
        "examples": examples,
    }

    print(f"\n  Saving to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    overall = scores["overall"]
    print(f"  {file_size_mb:.1f} MB  |  {len(examples)} examples")
    print(f"  SS: {overall['ss']:.1f}%  LMS: {overall['lms']:.1f}%  ICAT: {overall['icat']:.1f}")
    print(f"  Significant features (p<0.001): {sig_count}/{total_tested}")
    if sensitive_heads:
        print(f"  Top sensitive head: L{sensitive_heads[0]['layer']}H{sensitive_heads[0]['head']}")


def main():
    parser = argparse.ArgumentParser(description="Generate StereoSet pre-computed JSON")
    choices = list(MODEL_CONFIGS.keys()) + ["all"]
    parser.add_argument(
        "--model", choices=choices, default="all",
        help=f"Which model to generate data for (choices: {choices}, default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("StereoSet Pre-computation Script")
    print("=" * 60)

    print("\nLoading StereoSet dataset...")
    stereoset = load_dataset("stereoset", "intersentence", split="validation")
    print(f"Total examples: {len(stereoset)}")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_for_model(model_key, stereoset)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
