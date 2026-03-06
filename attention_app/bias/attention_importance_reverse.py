"""
Attention Importance Reverse Engineering
=========================================

Identifies sentences where attention IS most important for bias detection
by running per-sentence faithfulness and head ablation analysis, then
ranking by a composite attention importance score.

Usage:
    python attention_importance_reverse.py \
        --backbone bert \
        --model-dir attention_app/bias/models/gus-net-bert-final-new \
        --thresholds attention_app/bias/models/gus-net-bert-final-new/optimized_thresholds.npy \
        --dataset-source clean \
        --top-n 20 \
        --output-json results/attention_importance_bert.json
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

# Add parent dirs to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from span_faithfulness_eval import (
    evaluate_faithfulness,
    get_predicted_bias_mask,
    compute_sentence_bias_scores,
)
from head_ablation import batch_ablate_top_heads
from attention_bias import AttentionBiasAnalyzer
from integrated_gradients import (
    compute_token_attributions,
    compute_lrp_attributions,
    compute_topk_overlap,
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SentenceImportanceRecord:
    """Per-sentence attention importance metrics."""
    index: int
    text: str
    # Faithfulness
    delta_bias: float
    delta_stereo: float
    delta_gen: float
    delta_unfair: float
    orig_bias_score: float
    cf_bias_score: float
    frac_biased_tokens: float
    has_bias_predicted: bool
    # IG / Jaccard — specialized heads (BAR > 1.5)
    ig_spec_mean_jaccard: Optional[float] = None
    ig_spec_mean_rbo: Optional[float] = None
    ig_spec_max_jaccard: Optional[float] = None
    ig_spec_max_jaccard_head: Optional[List[int]] = None
    # IG — top-10 heads by BAR
    ig_top_mean_jaccard: Optional[float] = None
    ig_top_mean_rbo: Optional[float] = None
    ig_top_max_jaccard: Optional[float] = None
    ig_top_max_jaccard_head: Optional[List[int]] = None
    # IG — all heads (original, diluted)
    ig_all_mean_jaccard: Optional[float] = None
    ig_all_mean_rbo: Optional[float] = None
    # DeepLift — specialized heads (BAR > 1.5)
    dl_spec_mean_jaccard: Optional[float] = None
    dl_spec_mean_rbo: Optional[float] = None
    dl_spec_max_jaccard: Optional[float] = None
    dl_spec_max_jaccard_head: Optional[List[int]] = None
    # DeepLift — top-10 heads by BAR
    dl_top_mean_jaccard: Optional[float] = None
    dl_top_mean_rbo: Optional[float] = None
    dl_top_max_jaccard: Optional[float] = None
    dl_top_max_jaccard_head: Optional[List[int]] = None
    # DeepLift — all heads (original, diluted)
    dl_all_mean_jaccard: Optional[float] = None
    dl_all_mean_rbo: Optional[float] = None
    # Ablation (None if skipped)
    max_representation_impact: Optional[float] = None
    max_kl_divergence: Optional[float] = None
    most_impactful_head: Optional[List[int]] = None
    num_specialized_heads: Optional[int] = None
    # Composite
    composite_score: float = 0.0


# ============================================================================
# DATASET (same as span_faithfulness_eval.py but preserves order)
# ============================================================================

label2id = {
    "O": 0, "B-STEREO": 1, "I-STEREO": 2,
    "B-GEN": 3, "I-GEN": 4, "B-UNFAIR": 5, "I-UNFAIR": 6,
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)


class SimpleNERDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.items = []
        for sample in data:
            text = sample["text_str"]
            annotations = sample["ner_tags"]
            words = text.split()
            if len(words) != len(annotations):
                min_len = min(len(words), len(annotations))
                words = words[:min_len]
                annotations = annotations[:min_len]

            tokenized = tokenizer(
                words, is_split_into_words=True,
                padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            word_ids = tokenized.word_ids()

            aligned = []
            for wid in word_ids:
                if wid is None:
                    aligned.append([-100.0] * num_labels)
                else:
                    tags = annotations[wid]
                    vec = [0.0] * num_labels
                    for tag in tags:
                        if tag in label2id:
                            vec[label2id[tag]] = 1.0
                    aligned.append(vec)

            self.items.append((
                tokenized["input_ids"].squeeze(0),
                tokenized["attention_mask"].squeeze(0),
                torch.tensor(aligned, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================================
# PHASE 1: PER-SENTENCE FAITHFULNESS
# ============================================================================

def run_per_sentence_faithfulness(
    model_fn,
    test_data: list,
    tokenizer,
    thresholds: np.ndarray,
    mask_token_id: int,
    device: str,
    batch_size: int = 16,
) -> Tuple[dict, List[dict]]:
    """Run faithfulness evaluation with per-sentence tracking."""
    dataset = SimpleNERDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"  Running faithfulness on {len(dataset)} sentences...")
    aggregated, per_sentence = evaluate_faithfulness(
        model_fn=model_fn,
        dataloader=dataloader,
        thresholds=thresholds,
        mask_token_id=mask_token_id,
        num_labels=num_labels,
        device=device,
        return_per_sentence=True,
    )

    # Attach original text
    for record in per_sentence:
        idx = record["index"]
        if idx < len(test_data):
            record["text"] = test_data[idx]["text_str"]
        else:
            record["text"] = "(unknown)"

    return aggregated, per_sentence


# ============================================================================
# PHASE 1.5: PER-SENTENCE IG + JACCARD
# ============================================================================

_ZERO_SUMMARY = {
    "mean_jaccard": 0.0, "mean_rbo": 0.0,
    "max_jaccard": 0.0, "max_jaccard_head": None,
}


def _empty_ig_dl_record():
    """Return a zeroed-out IG+DeepLift record."""
    out = {}
    for method in ("ig", "dl"):
        for scope in ("all", "spec", "top"):
            key = f"{method}_{scope}"
            for metric, default in _ZERO_SUMMARY.items():
                out[f"{key}_{metric}"] = default
    return out


def _topk_to_summary(topk_overlaps, head_metrics, prefix, top_n_heads=10,
                     bar_threshold=1.5):
    """Compute Jaccard/RBO summary for three head scopes.

    Scopes:
      - ``all``:  all heads (original behaviour, tends to be low)
      - ``spec``: only heads with BAR > ``bar_threshold`` (specialized for bias)
      - ``top``:  top-N heads ranked by BAR
    """
    if not topk_overlaps:
        out = {}
        for scope in ("all", "spec", "top"):
            key = f"{prefix}_{scope}"
            for metric, default in _ZERO_SUMMARY.items():
                out[f"{key}_{metric}"] = default
        return out

    # Build BAR lookup from head_metrics
    bar_lookup = {(m.layer, m.head): m.bias_attention_ratio for m in head_metrics}

    # Identify head sets — use bar_threshold instead of m.specialized_for_bias
    specialized = {(m.layer, m.head) for m in head_metrics
                   if m.bias_attention_ratio > bar_threshold}
    top_n_set = {(m.layer, m.head) for m in sorted(
        head_metrics, key=lambda m: m.bias_attention_ratio, reverse=True,
    )[:top_n_heads]}

    scopes = {
        "all": None,  # no filter
        "spec": specialized,
        "top": top_n_set,
    }

    out = {}
    for scope_name, head_set in scopes.items():
        if head_set is not None:
            filtered = [t for t in topk_overlaps if (t.layer, t.head) in head_set]
        else:
            filtered = topk_overlaps

        key = f"{prefix}_{scope_name}"
        if not filtered:
            for metric, default in _ZERO_SUMMARY.items():
                out[f"{key}_{metric}"] = default
        else:
            jaccards = [t.jaccard for t in filtered]
            rbos = [t.rank_biased_overlap for t in filtered]
            best = max(filtered, key=lambda t: t.jaccard)
            out[f"{key}_mean_jaccard"] = float(np.mean(jaccards))
            out[f"{key}_mean_rbo"] = float(np.mean(rbos))
            out[f"{key}_max_jaccard"] = float(best.jaccard)
            out[f"{key}_max_jaccard_head"] = [best.layer, best.head]

    return out


def run_per_sentence_ig_jaccard(
    encoder_model,
    ner_model_fn,
    tokenizer,
    test_data: list,
    thresholds: np.ndarray,
    device: str,
    is_gpt2: bool,
    ig_k: int = 5,
    n_steps: int = 30,
    sentence_indices: Optional[List[int]] = None,
    bar_threshold: float = 1.5,
) -> Dict[int, dict]:
    """
    Run IG + DeepLift attributions and top-K Jaccard/RBO for selected sentences.

    For each sentence:
      1. Run NER model to get biased token indices
      2. Run encoder with output_attentions to get attention weights
      3. Compute IG attributions (Captum LayerIntegratedGradients)
      4. Compute DeepLift attributions (Captum LayerDeepLift)
      5. Compute top-K overlap (Jaccard + RBO) per head for both methods
      6. Return mean/max Jaccard and RBO for IG and DeepLift
    """
    analyzer = AttentionBiasAnalyzer()
    indices = sentence_indices if sentence_indices is not None else range(len(test_data))
    results = {}

    # IG needs gradients — temporarily enable
    was_training = encoder_model.training
    encoder_model.eval()

    print(f"  Running IG + DeepLift + Jaccard on {len(indices)} sentences "
          f"(n_steps={n_steps})...")
    for idx in tqdm(indices, desc="  IG+DL Jaccard"):
        text = test_data[idx]["text_str"]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Get NER predictions for biased token indices
        with torch.no_grad():
            ner_logits = ner_model_fn(input_ids, attn_mask)
        bias_mask = get_predicted_bias_mask(ner_logits, thresholds, num_labels)
        biased_indices = bias_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(biased_indices, int):
            biased_indices = [biased_indices]

        if not biased_indices:
            results[idx] = _empty_ig_dl_record()
            continue

        # Get attention weights
        encoder_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder_outputs = encoder_model(**encoder_inputs, output_attentions=True)
        attentions = list(encoder_outputs.attentions)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        # Compute per-head HeadBiasMetrics (needed for BAR lookup in topk_overlap)
        head_metrics = analyzer.analyze_attention_to_bias(
            attentions, biased_indices, tokens,
        )

        if not head_metrics:
            results[idx] = _empty_ig_dl_record()
            continue

        # ── IG attributions ─────────────────────────────────────
        ig_record = {}
        try:
            ig_attrs = compute_token_attributions(
                encoder_model, tokenizer, text, is_gpt2, n_steps=n_steps,
            )
            ig_topk = compute_topk_overlap(
                attentions, ig_attrs, head_metrics, k=ig_k,
            )
            ig_record = _topk_to_summary(ig_topk, head_metrics, "ig",
                                        bar_threshold=bar_threshold)
        except Exception as e:
            print(f"    Warning: IG failed for idx={idx}: {e}")
            ig_record = _topk_to_summary([], head_metrics, "ig",
                                        bar_threshold=bar_threshold)

        # ── DeepLift attributions ───────────────────────────────
        dl_record = {}
        try:
            dl_attrs, _ = compute_lrp_attributions(
                encoder_model, tokenizer, text, is_gpt2,
            )
            dl_topk = compute_topk_overlap(
                attentions, dl_attrs, head_metrics, k=ig_k,
            )
            dl_record = _topk_to_summary(dl_topk, head_metrics, "dl",
                                        bar_threshold=bar_threshold)
        except Exception as e:
            print(f"    Warning: DeepLift failed for idx={idx}: {e}")
            dl_record = _topk_to_summary([], head_metrics, "dl",
                                        bar_threshold=bar_threshold)

        results[idx] = {**ig_record, **dl_record}

    return results


# ============================================================================
# PHASE 2: PER-SENTENCE HEAD ABLATION
# ============================================================================

def run_per_sentence_ablation(
    encoder_model,
    lm_head_model,
    ner_model_fn,
    tokenizer,
    test_data: list,
    thresholds: np.ndarray,
    device: str,
    is_gpt2: bool,
    top_k_heads: int = 5,
    sentence_indices: Optional[List[int]] = None,
) -> Dict[int, dict]:
    """
    Run head ablation for selected sentences.

    For each sentence:
      1. Run NER model to get bias predictions
      2. Run encoder with output_attentions to get attention weights
      3. Compute HeadBiasMetrics (BAR) for each head
      4. Select top-K heads by BAR
      5. Ablate each and measure representation_impact
    """
    analyzer = AttentionBiasAnalyzer()
    indices = sentence_indices if sentence_indices is not None else range(len(test_data))
    results = {}

    print(f"  Running head ablation on {len(indices)} sentences...")
    for idx in tqdm(indices, desc="  Head ablation"):
        text = test_data[idx]["text_str"]

        # Tokenize for encoder
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Get NER predictions to find biased token indices
        with torch.no_grad():
            ner_logits = ner_model_fn(input_ids, attn_mask)  # (1, T, 7)
        bias_mask = get_predicted_bias_mask(ner_logits, thresholds, num_labels)
        biased_indices = bias_mask[0].nonzero(as_tuple=False).squeeze(-1).tolist()
        if isinstance(biased_indices, int):
            biased_indices = [biased_indices]

        if not biased_indices:
            results[idx] = {
                "max_representation_impact": 0.0,
                "max_kl_divergence": None,
                "most_impactful_head": None,
                "num_specialized_heads": 0,
            }
            continue

        # Get attention weights from encoder
        encoder_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            encoder_outputs = encoder_model(**encoder_inputs, output_attentions=True)
        attentions = list(encoder_outputs.attentions)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())

        # Compute per-head BAR metrics
        head_metrics = analyzer.analyze_attention_to_bias(
            attentions, biased_indices, tokens,
        )

        if not head_metrics:
            results[idx] = {
                "max_representation_impact": 0.0,
                "max_kl_divergence": None,
                "most_impactful_head": None,
                "num_specialized_heads": 0,
            }
            continue

        # Select top-K heads by BAR
        sorted_heads = sorted(
            head_metrics,
            key=lambda m: m.bias_attention_ratio,
            reverse=True,
        )
        top_heads = sorted_heads[:top_k_heads]
        num_specialized = sum(1 for m in head_metrics if m.specialized_for_bias)

        # Run ablation
        ablation_results = batch_ablate_top_heads(
            encoder_model=encoder_model,
            lm_head_model=lm_head_model,
            tokenizer=tokenizer,
            text=text,
            top_heads=top_heads,
            is_gpt2=is_gpt2,
        )

        if ablation_results:
            best = ablation_results[0]  # sorted by representation_impact desc
            results[idx] = {
                "max_representation_impact": best.representation_impact,
                "max_kl_divergence": best.kl_divergence,
                "most_impactful_head": [best.layer, best.head],
                "num_specialized_heads": num_specialized,
            }
        else:
            results[idx] = {
                "max_representation_impact": 0.0,
                "max_kl_divergence": None,
                "most_impactful_head": None,
                "num_specialized_heads": 0,
            }

    return results


# ============================================================================
# PHASE 3: COMPOSITE SCORE & RANKING
# ============================================================================

def percentile_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata
    if len(values) == 0:
        return values
    ranks = rankdata(values, method='average')
    return (ranks - 1) / max(len(ranks) - 1, 1)


def merge_and_rank(
    faithfulness_records: List[dict],
    ig_records: Optional[Dict[int, dict]],
    ablation_records: Optional[Dict[int, dict]],
    test_data: list,
) -> List[SentenceImportanceRecord]:
    """Merge faithfulness + IG/Jaccard + ablation, compute composite, sort descending."""

    records = []
    for rec in faithfulness_records:
        idx = rec["index"]
        abl = ablation_records.get(idx, {}) if ablation_records else {}
        ig = ig_records.get(idx, {}) if ig_records else {}

        records.append(SentenceImportanceRecord(
            index=idx,
            text=rec.get("text", test_data[idx]["text_str"] if idx < len(test_data) else ""),
            delta_bias=rec["delta_bias"],
            delta_stereo=rec["delta_stereo"],
            delta_gen=rec["delta_gen"],
            delta_unfair=rec["delta_unfair"],
            orig_bias_score=rec["orig_score"],
            cf_bias_score=rec["cf_score"],
            frac_biased_tokens=rec["frac_biased_tokens"],
            has_bias_predicted=rec["has_bias_predicted"],
            ig_spec_mean_jaccard=ig.get("ig_spec_mean_jaccard"),
            ig_spec_mean_rbo=ig.get("ig_spec_mean_rbo"),
            ig_spec_max_jaccard=ig.get("ig_spec_max_jaccard"),
            ig_spec_max_jaccard_head=ig.get("ig_spec_max_jaccard_head"),
            ig_top_mean_jaccard=ig.get("ig_top_mean_jaccard"),
            ig_top_mean_rbo=ig.get("ig_top_mean_rbo"),
            ig_top_max_jaccard=ig.get("ig_top_max_jaccard"),
            ig_top_max_jaccard_head=ig.get("ig_top_max_jaccard_head"),
            ig_all_mean_jaccard=ig.get("ig_all_mean_jaccard"),
            ig_all_mean_rbo=ig.get("ig_all_mean_rbo"),
            dl_spec_mean_jaccard=ig.get("dl_spec_mean_jaccard"),
            dl_spec_mean_rbo=ig.get("dl_spec_mean_rbo"),
            dl_spec_max_jaccard=ig.get("dl_spec_max_jaccard"),
            dl_spec_max_jaccard_head=ig.get("dl_spec_max_jaccard_head"),
            dl_top_mean_jaccard=ig.get("dl_top_mean_jaccard"),
            dl_top_mean_rbo=ig.get("dl_top_mean_rbo"),
            dl_top_max_jaccard=ig.get("dl_top_max_jaccard"),
            dl_top_max_jaccard_head=ig.get("dl_top_max_jaccard_head"),
            dl_all_mean_jaccard=ig.get("dl_all_mean_jaccard"),
            dl_all_mean_rbo=ig.get("dl_all_mean_rbo"),
            max_representation_impact=abl.get("max_representation_impact"),
            max_kl_divergence=abl.get("max_kl_divergence"),
            most_impactful_head=abl.get("most_impactful_head"),
            num_specialized_heads=abl.get("num_specialized_heads"),
        ))

    # Compute composite score
    # Available signals and their weights depend on which phases ran
    deltas = np.array([r.delta_bias for r in records])
    origs = np.array([r.orig_bias_score for r in records])
    fracs = np.array([r.frac_biased_tokens for r in records])

    norm_delta = percentile_normalize(deltas)
    norm_orig = percentile_normalize(origs)
    norm_frac = percentile_normalize(fracs)

    has_ig = ig_records is not None and len(ig_records) > 0
    has_ablation = ablation_records is not None and len(ablation_records) > 0

    # Use specialized-head Jaccard (_spec) for composite score
    norm_ig_jaccard = None
    norm_dl_jaccard = None
    if has_ig:
        ig_jaccards = np.array([
            r.ig_spec_mean_jaccard if r.ig_spec_mean_jaccard is not None else 0.0
            for r in records
        ])
        dl_jaccards = np.array([
            r.dl_spec_mean_jaccard if r.dl_spec_mean_jaccard is not None else 0.0
            for r in records
        ])
        norm_ig_jaccard = percentile_normalize(ig_jaccards)
        norm_dl_jaccard = percentile_normalize(dl_jaccards)

    norm_impact = None
    if has_ablation:
        impacts = np.array([
            r.max_representation_impact if r.max_representation_impact is not None else 0.0
            for r in records
        ])
        norm_impact = percentile_normalize(impacts)

    # Composite weights depend on which phases are available:
    #   all 3:    delta=0.25  ig_jac=0.15  dl_jac=0.15  ablation=0.20  orig=0.15  frac=0.10
    #   IG only:  delta=0.35  ig_jac=0.20  dl_jac=0.15  orig=0.20  frac=0.10
    #   abl only: delta=0.40  ablation=0.30  orig=0.20  frac=0.10
    #   none:     delta=0.60  orig=0.25  frac=0.15
    for i, rec in enumerate(records):
        if has_ig and has_ablation:
            rec.composite_score = round(
                0.25 * norm_delta[i]
                + 0.15 * norm_ig_jaccard[i]
                + 0.15 * norm_dl_jaccard[i]
                + 0.20 * norm_impact[i]
                + 0.15 * norm_orig[i]
                + 0.10 * norm_frac[i],
                4,
            )
        elif has_ig:
            rec.composite_score = round(
                0.35 * norm_delta[i]
                + 0.20 * norm_ig_jaccard[i]
                + 0.15 * norm_dl_jaccard[i]
                + 0.20 * norm_orig[i]
                + 0.10 * norm_frac[i],
                4,
            )
        elif has_ablation:
            rec.composite_score = round(
                0.40 * norm_delta[i]
                + 0.30 * norm_impact[i]
                + 0.20 * norm_orig[i]
                + 0.10 * norm_frac[i],
                4,
            )
        else:
            rec.composite_score = round(
                0.60 * norm_delta[i]
                + 0.25 * norm_orig[i]
                + 0.15 * norm_frac[i],
                4,
            )

    records.sort(key=lambda r: r.composite_score, reverse=True)
    return records


# ============================================================================
# OUTPUT
# ============================================================================

def print_importance_report(
    ranked: List[SentenceImportanceRecord],
    top_n: int,
    aggregated: dict,
    has_ig: bool,
    has_ablation: bool,
    bar_threshold: float = 1.5,
):
    """Console summary."""
    deltas = [r.delta_bias for r in ranked if r.has_bias_predicted]
    deltas_arr = np.array(deltas) if deltas else np.array([0.0])

    print()
    print("=" * 70)
    print("ATTENTION IMPORTANCE ANALYSIS - Reverse Engineering")
    print("=" * 70)
    print(f"  Total sentences:              {len(ranked)}")
    print(f"  With bias predicted:          {aggregated['n_sequences_with_bias']}")
    print(f"  IG + Jaccard:                 {'Yes' if has_ig else 'Skipped'}")
    print(f"  Head ablation:                {'Yes' if has_ablation else 'Skipped'}")
    print("-" * 70)
    print()
    print("  Aggregate faithfulness (reference):")
    print(f"    delta_bias  mean={aggregated['delta_bias_mean']:.4f}  "
          f"std={aggregated['delta_bias_std']:.4f}  "
          f"median={aggregated['delta_bias_median']:.4f}")

    if has_ig:
        for scope, label in [("spec", f"Specialized BAR>{bar_threshold}"),
                              ("top", "Top-10 by BAR"),
                              ("all", "All 144 heads")]:
            ig_attr = f"ig_{scope}_mean_jaccard"
            dl_attr = f"dl_{scope}_mean_jaccard"
            ig_vals = [getattr(r, ig_attr) for r in ranked
                       if getattr(r, ig_attr, None) is not None
                       and getattr(r, ig_attr) > 0]
            dl_vals = [getattr(r, dl_attr) for r in ranked
                       if getattr(r, dl_attr, None) is not None
                       and getattr(r, dl_attr) > 0]
            if ig_vals:
                j = np.array(ig_vals)
                print(f"\n  IG Jaccard [{label}]:")
                print(f"    mean={j.mean():.4f}  "
                      f"P75={np.percentile(j, 75):.4f}  "
                      f"P90={np.percentile(j, 90):.4f}  "
                      f"P95={np.percentile(j, 95):.4f}")
            if dl_vals:
                d = np.array(dl_vals)
                print(f"  DL Jaccard [{label}]:")
                print(f"    mean={d.mean():.4f}  "
                      f"P75={np.percentile(d, 75):.4f}  "
                      f"P90={np.percentile(d, 90):.4f}  "
                      f"P95={np.percentile(d, 95):.4f}")

    print()
    print("  Distribution of per-sentence delta_bias:")
    if len(deltas_arr) > 0:
        print(f"    P75={np.percentile(deltas_arr, 75):.4f}  "
              f"P90={np.percentile(deltas_arr, 90):.4f}  "
              f"P95={np.percentile(deltas_arr, 95):.4f}  "
              f"P99={np.percentile(deltas_arr, 99):.4f}")
    print()
    print("=" * 70)
    print(f"  TOP {top_n} SENTENCES WHERE ATTENTION IS MOST IMPORTANT")
    print("=" * 70)

    for i, rec in enumerate(ranked[:top_n]):
        print()
        # Build extra metrics line (show spec = specialized heads)
        extras = []
        if rec.ig_spec_mean_jaccard is not None and rec.ig_spec_mean_jaccard > 0:
            h = (f"L{rec.ig_spec_max_jaccard_head[0]}H{rec.ig_spec_max_jaccard_head[1]}"
                 if rec.ig_spec_max_jaccard_head else "?")
            extras.append(f"ig_jac={rec.ig_spec_mean_jaccard:.4f}"
                          f"(max={rec.ig_spec_max_jaccard:.3f} {h})")
        if rec.dl_spec_mean_jaccard is not None and rec.dl_spec_mean_jaccard > 0:
            h = (f"L{rec.dl_spec_max_jaccard_head[0]}H{rec.dl_spec_max_jaccard_head[1]}"
                 if rec.dl_spec_max_jaccard_head else "?")
            extras.append(f"dl_jac={rec.dl_spec_mean_jaccard:.4f}"
                          f"(max={rec.dl_spec_max_jaccard:.3f} {h})")
        if rec.max_representation_impact is not None:
            h = (f"L{rec.most_impactful_head[0]}H{rec.most_impactful_head[1]}"
                 if rec.most_impactful_head else "?")
            extras.append(f"ablation={rec.max_representation_impact:.4f} ({h})")

        extra_str = "  ".join(extras)
        if extra_str:
            extra_str = "  " + extra_str

        print(f"  #{i+1:2d}  [idx={rec.index}]  "
              f"composite={rec.composite_score:.3f}  "
              f"delta={rec.delta_bias:.4f}  "
              f"orig={rec.orig_bias_score:.3f}"
              f"{extra_str}")
        text_display = rec.text if len(rec.text) <= 100 else rec.text[:97] + "..."
        print(f"      \"{text_display}\"")
        print(f"      delta_stereo={rec.delta_stereo:.4f}  "
              f"delta_gen={rec.delta_gen:.4f}  "
              f"delta_unfair={rec.delta_unfair:.4f}  "
              f"frac_biased={rec.frac_biased_tokens:.3f}")
        # Show RBO for spec heads
        rbo_parts = []
        if rec.ig_spec_mean_rbo is not None and rec.ig_spec_mean_rbo > 0:
            rbo_parts.append(f"ig_rbo={rec.ig_spec_mean_rbo:.4f}")
        if rec.dl_spec_mean_rbo is not None and rec.dl_spec_mean_rbo > 0:
            rbo_parts.append(f"dl_rbo={rec.dl_spec_mean_rbo:.4f}")
        # Also show top-10 and all-heads for comparison
        if rec.ig_top_mean_jaccard is not None and rec.ig_top_mean_jaccard > 0:
            rbo_parts.append(f"ig_top10={rec.ig_top_mean_jaccard:.4f}")
        if rec.ig_all_mean_jaccard is not None and rec.ig_all_mean_jaccard > 0:
            rbo_parts.append(f"ig_all={rec.ig_all_mean_jaccard:.4f}")
        if rbo_parts:
            print(f"      {' '.join(rbo_parts)}")

    print()
    print("=" * 70)


def save_json_report(
    ranked: List[SentenceImportanceRecord],
    aggregated: dict,
    metadata: dict,
    output_path: str,
):
    """Save full JSON report."""
    deltas = [r.delta_bias for r in ranked if r.has_bias_predicted]
    deltas_arr = np.array(deltas) if deltas else np.array([0.0])
    composites = np.array([r.composite_score for r in ranked])

    distribution = {
        "delta_bias_p75": round(float(np.percentile(deltas_arr, 75)), 4),
        "delta_bias_p90": round(float(np.percentile(deltas_arr, 90)), 4),
        "delta_bias_p95": round(float(np.percentile(deltas_arr, 95)), 4),
        "delta_bias_p99": round(float(np.percentile(deltas_arr, 99)), 4),
        "composite_p75": round(float(np.percentile(composites, 75)), 4),
        "composite_p90": round(float(np.percentile(composites, 90)), 4),
        "composite_p95": round(float(np.percentile(composites, 95)), 4),
        "composite_p99": round(float(np.percentile(composites, 99)), 4),
    }

    # Add IG/DL Jaccard distribution for each scope
    for method in ("ig", "dl"):
        for scope, label in [("spec", "specialized"), ("top", "top10"), ("all", "all")]:
            attr = f"{method}_{scope}_mean_jaccard"
            vals = [getattr(r, attr) for r in ranked
                    if getattr(r, attr, None) is not None and getattr(r, attr) > 0]
            if vals:
                v = np.array(vals)
                prefix = f"{method}_{label}_jaccard"
                distribution.update({
                    f"{prefix}_mean": round(float(v.mean()), 4),
                    f"{prefix}_p75": round(float(np.percentile(v, 75)), 4),
                    f"{prefix}_p90": round(float(np.percentile(v, 90)), 4),
                    f"{prefix}_p95": round(float(np.percentile(v, 95)), 4),
                    f"{prefix}_p99": round(float(np.percentile(v, 99)), 4),
                })

    report = {
        "metadata": metadata,
        "aggregated_faithfulness": aggregated,
        "distribution": distribution,
        "all_sentences": [asdict(r) for r in ranked],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {output_path}")


def save_csv_report(
    ranked: List[SentenceImportanceRecord],
    output_path: str,
):
    """Save CSV for easy analysis in Excel/pandas."""
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fields = [
        "rank", "index", "composite_score", "delta_bias",
        "orig_bias_score", "cf_bias_score", "frac_biased_tokens",
        "delta_stereo", "delta_gen", "delta_unfair",
        # IG scoped
        "ig_spec_mean_jaccard", "ig_spec_mean_rbo",
        "ig_spec_max_jaccard", "ig_spec_max_jaccard_head",
        "ig_top_mean_jaccard", "ig_top_mean_rbo",
        "ig_all_mean_jaccard", "ig_all_mean_rbo",
        # DL scoped
        "dl_spec_mean_jaccard", "dl_spec_mean_rbo",
        "dl_spec_max_jaccard", "dl_spec_max_jaccard_head",
        "dl_top_mean_jaccard", "dl_top_mean_rbo",
        "dl_all_mean_jaccard", "dl_all_mean_rbo",
        # Ablation
        "max_representation_impact", "most_impactful_head",
        "num_specialized_heads", "text",
    ]

    def _fmt_head(head_list):
        return f"L{head_list[0]}H{head_list[1]}" if head_list else ""

    def _fmt_val(val):
        return val if val is not None else ""

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, rec in enumerate(ranked):
            writer.writerow({
                "rank": i + 1,
                "index": rec.index,
                "composite_score": rec.composite_score,
                "delta_bias": rec.delta_bias,
                "orig_bias_score": rec.orig_bias_score,
                "cf_bias_score": rec.cf_bias_score,
                "frac_biased_tokens": rec.frac_biased_tokens,
                "delta_stereo": rec.delta_stereo,
                "delta_gen": rec.delta_gen,
                "delta_unfair": rec.delta_unfair,
                "ig_spec_mean_jaccard": _fmt_val(rec.ig_spec_mean_jaccard),
                "ig_spec_mean_rbo": _fmt_val(rec.ig_spec_mean_rbo),
                "ig_spec_max_jaccard": _fmt_val(rec.ig_spec_max_jaccard),
                "ig_spec_max_jaccard_head": _fmt_head(rec.ig_spec_max_jaccard_head),
                "ig_top_mean_jaccard": _fmt_val(rec.ig_top_mean_jaccard),
                "ig_top_mean_rbo": _fmt_val(rec.ig_top_mean_rbo),
                "ig_all_mean_jaccard": _fmt_val(rec.ig_all_mean_jaccard),
                "ig_all_mean_rbo": _fmt_val(rec.ig_all_mean_rbo),
                "dl_spec_mean_jaccard": _fmt_val(rec.dl_spec_mean_jaccard),
                "dl_spec_mean_rbo": _fmt_val(rec.dl_spec_mean_rbo),
                "dl_spec_max_jaccard": _fmt_val(rec.dl_spec_max_jaccard),
                "dl_spec_max_jaccard_head": _fmt_head(rec.dl_spec_max_jaccard_head),
                "dl_top_mean_jaccard": _fmt_val(rec.dl_top_mean_jaccard),
                "dl_top_mean_rbo": _fmt_val(rec.dl_top_mean_rbo),
                "dl_all_mean_jaccard": _fmt_val(rec.dl_all_mean_jaccard),
                "dl_all_mean_rbo": _fmt_val(rec.dl_all_mean_rbo),
                "max_representation_impact": _fmt_val(rec.max_representation_impact),
                "most_impactful_head": _fmt_head(rec.most_impactful_head),
                "num_specialized_heads": _fmt_val(rec.num_specialized_heads),
                "text": rec.text,
            })
    print(f"CSV saved to {output_path}")


# ============================================================================
# STEREOSET LOADER
# ============================================================================

def _text_to_ner_entry(text: str, source: str = "synthetic", **extra) -> dict:
    """Convert plain text to an NER-format dict with dummy O tags."""
    words = text.split()
    return {
        "text_str": text,
        "ner_tags": [["O"] for _ in words],
        "source": source,
        **extra,
    }


def load_stereoset_as_ner(backbone: str = "bert") -> list:
    """Load StereoSet sentences (stereo + anti-stereo) as NER-format dicts."""
    from stereoset.stereoset_data import get_stereoset_examples

    # Use the GUS-NET variant that matches the backbone
    model_key = f"gusnet_{backbone}"
    examples = get_stereoset_examples(model_key)
    if not examples:
        examples = get_stereoset_examples(backbone)
    if not examples:
        raise FileNotFoundError(
            f"No StereoSet data found for model key '{model_key}'. "
            f"Run generate_stereoset_json.py first."
        )

    data = []
    seen = set()
    for ex in examples:
        cat = ex.get("category", "")
        target = ex.get("target", "")
        for key in ("stereo_sentence", "anti_sentence"):
            text = ex.get(key, "")
            if not text or text in seen:
                continue
            seen.add(text)
            data.append(_text_to_ner_entry(
                text, source=f"stereoset_{key.split('_')[0]}",
                stereoset_category=cat, stereoset_target=target,
            ))

    print(f"  StereoSet: loaded {len(data)} sentences "
          f"(stereo + anti-stereo, deduplicated)")
    return data


def load_probe_sentences() -> list:
    """Return hand-crafted probe sentences with obvious, unambiguous bias.

    These are short sentences designed to maximise attention-attribution
    overlap: few tokens, one clear bias trigger, minimal noise.
    Organised by bias type (stereotype, generalisation, unfairness).
    """
    probes = [
        # ── Gender stereotypes ────────────────────────────────
        "Women are too emotional to lead.",
        "Men are natural leaders.",
        "Girls are bad at math.",
        "Boys don't cry.",
        "Women belong in the kitchen.",
        "Mothers are better parents than fathers.",
        "Female doctors are less competent.",
        "Women are terrible drivers.",
        "Men cannot be nurturing.",
        "Women are too sensitive for business.",
        "A woman's place is at home.",
        "Men are inherently aggressive.",
        "Women are naturally submissive.",
        "Girls should play with dolls.",
        "Real men don't show emotion.",
        # ── Racial stereotypes ────────────────────────────────
        "Black people are inherently violent.",
        "Asians are all good at math.",
        "White people can't dance.",
        "Mexicans are lazy workers.",
        "Arabs are terrorists.",
        "Black men are dangerous.",
        "All Indians work in IT.",
        "Chinese people are dishonest.",
        "African people are uncivilized.",
        "Native Americans are alcoholics.",
        "Hispanic people are illegal immigrants.",
        "Jewish people are greedy.",
        "Muslim people are extremists.",
        "All Romani people are thieves.",
        "Asian women are submissive.",
        # ── Age stereotypes ───────────────────────────────────
        "Old people are useless.",
        "Elderly workers are incompetent.",
        "Young people are irresponsible.",
        "Millennials are entitled and lazy.",
        "Senior citizens can't learn technology.",
        # ── Disability stereotypes ────────────────────────────
        "Disabled people are a burden.",
        "Mentally ill people are dangerous.",
        "Deaf people are less intelligent.",
        "Blind people are helpless.",
        "People in wheelchairs are incapable.",
        # ── Religion stereotypes ──────────────────────────────
        "Muslims are violent extremists.",
        "Christians are intolerant bigots.",
        "Atheists have no morals.",
        "Jewish people control the banks.",
        "Hindus are backward and superstitious.",
        # ── Profession / class stereotypes ────────────────────
        "Poor people are lazy.",
        "Rich people are heartless.",
        "Immigrants steal our jobs.",
        "Refugees are criminals.",
        "Homeless people are drug addicts.",
        # ── Sexual orientation stereotypes ────────────────────
        "Gay men are feminine.",
        "Lesbians hate men.",
        "Bisexual people are confused.",
        "Transgender people are mentally ill.",
        # ── Generalisations (short, unambiguous) ──────────────
        "All women are emotional.",
        "Every man is aggressive.",
        "All foreigners are suspicious.",
        "No woman can be a good engineer.",
        "Every immigrant is a criminal.",
        "All old people are senile.",
        "Black people are all the same.",
        "Women always gossip.",
        "Men never listen.",
        "Immigrants never integrate.",
        # ── Very short probes (3-5 words) ─────────────────────
        "Women are weak.",
        "Men are superior.",
        "Blacks are criminals.",
        "Immigrants are dangerous.",
        "Muslims are terrorists.",
        "Disabled people are worthless.",
        "Old people are senile.",
        "Poor people deserve poverty.",
        "Fat people are lazy.",
        "Blondes are stupid.",
    ]
    data = [_text_to_ner_entry(t, source="probe") for t in probes]
    print(f"  Probes: loaded {len(data)} synthetic bias sentences")
    return data


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Identify sentences where attention is most important for bias detection",
    )
    parser.add_argument(
        "--backbone", choices=["bert", "gpt2"], required=True,
        help="Model backbone type",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to saved GUS-Net model directory",
    )
    parser.add_argument(
        "--thresholds", type=str, required=True,
        help="Path to optimized_thresholds.npy",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["hf", "gemini", "clean", "stereoset", "probes", "combined"],
        default="clean",
        help="'stereoset' = StereoSet only; 'probes' = synthetic probes; "
             "'combined' = clean test + StereoSet + probes",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top sentences to display in report",
    )
    parser.add_argument(
        "--top-k-heads", type=int, default=5,
        help="Number of top heads to ablate per sentence",
    )
    parser.add_argument(
        "--ablation-pool", type=int, default=100,
        help="Number of top faithfulness sentences to run ablation on (0=all)",
    )
    parser.add_argument(
        "--ig-pool", type=int, default=50,
        help="Number of top faithfulness sentences to run IG+Jaccard on (0=all)",
    )
    parser.add_argument(
        "--ig-k", type=int, default=5,
        help="Top-K tokens to compare in Jaccard (IG vs Attention)",
    )
    parser.add_argument(
        "--ig-steps", type=int, default=30,
        help="Number of IG interpolation steps (higher=more accurate, slower)",
    )
    parser.add_argument(
        "--bar-threshold", type=float, default=1.5,
        help="BAR threshold for 'specialized' head scope (default 1.5)",
    )
    parser.add_argument(
        "--min-ig-jaccard", type=float, default=0.0,
        help="Only show sentences with ig_spec_mean_jaccard >= this value",
    )
    parser.add_argument(
        "--min-dl-jaccard", type=float, default=0.0,
        help="Only show sentences with dl_spec_mean_jaccard >= this value",
    )
    parser.add_argument(
        "--skip-ig", action="store_true",
        help="Skip IG + Jaccard computation",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip head ablation (faithfulness only, faster)",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    thresholds = np.load(args.thresholds)

    # ── Load model ──────────────────────────────────────────────
    if args.backbone == "bert":
        from transformers import BertTokenizerFast, BertForTokenClassification, BertForMaskedLM

        tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
        ner_model = BertForTokenClassification.from_pretrained(args.model_dir)
        ner_model.eval().to(device)
        mask_token_id = tokenizer.mask_token_id
        is_gpt2 = False

        def model_fn(input_ids, attention_mask):
            return ner_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Encoder for ablation: reuse the BERT inside the NER model
        encoder_model = ner_model.bert
        encoder_model.config.output_attentions = True

        # LM head for KL divergence
        lm_head_model = None
        if not args.skip_ablation:
            try:
                lm_head_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
                lm_head_model.eval().to(device)
            except Exception as e:
                print(f"  Warning: Could not load LM head ({e}), KL divergence will be skipped")

    elif args.backbone == "gpt2":
        from transformers import GPT2TokenizerFast, GPT2ForTokenClassification, GPT2LMHeadModel

        tokenizer = GPT2TokenizerFast.from_pretrained(
            args.model_dir, add_prefix_space=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        ner_model = GPT2ForTokenClassification.from_pretrained(args.model_dir)
        ner_model.eval().to(device)
        mask_token_id = tokenizer.eos_token_id
        is_gpt2 = True

        def model_fn(input_ids, attention_mask):
            return ner_model(input_ids=input_ids, attention_mask=attention_mask).logits

        encoder_model = ner_model.transformer
        encoder_model.config.output_attentions = True

        lm_head_model = None
        if not args.skip_ablation:
            try:
                lm_head_model = GPT2LMHeadModel.from_pretrained("gpt2")
                lm_head_model.eval().to(device)
            except Exception as e:
                print(f"  Warning: Could not load LM head ({e}), KL divergence will be skipped")

    # ── Load data ───────────────────────────────────────────────
    if args.dataset_source in ("clean", "gemini", "hf", "combined"):
        if args.dataset_source in ("clean", "combined"):
            from gus_net_training_paper import load_from_clean_json
            all_data = load_from_clean_json()
        elif args.dataset_source == "gemini":
            from gus_net_training import load_from_gemini
            all_data = load_from_gemini()
        else:
            from gus_net_training import load_from_hf
            all_data = load_from_hf()

        from sklearn.model_selection import train_test_split
        _, test_data = train_test_split(
            all_data, test_size=0.20, random_state=42, shuffle=True,
        )
        print(f"Test set (bias corpus): {len(test_data)} sentences")

        if args.dataset_source == "combined":
            stereo_data = load_stereoset_as_ner(args.backbone)
            probe_data = load_probe_sentences()
            test_data = test_data + stereo_data + probe_data
            print(f"Combined test set: {len(test_data)} sentences")

    elif args.dataset_source == "stereoset":
        test_data = load_stereoset_as_ner(args.backbone)
        print(f"Test set (StereoSet only): {len(test_data)} sentences")

    elif args.dataset_source == "probes":
        test_data = load_probe_sentences()
        print(f"Test set (probes only): {len(test_data)} sentences")

    # ── Phase 1: Per-sentence faithfulness ──────────────────────
    print("\n[Phase 1] Per-sentence Faithfulness Analysis")
    aggregated, faith_records = run_per_sentence_faithfulness(
        model_fn=model_fn,
        test_data=test_data,
        tokenizer=tokenizer,
        thresholds=thresholds,
        mask_token_id=mask_token_id,
        device=device,
        batch_size=args.batch_size,
    )

    # ── Phase 1.5: Per-sentence IG + Jaccard ───────────────────
    ig_records = None
    if not args.skip_ig:
        print("\n[Phase 1.5] Per-sentence IG + Jaccard Analysis")

        # Select which sentences to run IG on
        if args.ig_pool > 0:
            sorted_by_delta = sorted(
                faith_records,
                key=lambda r: r["delta_bias"],
                reverse=True,
            )
            ig_pool_indices = [r["index"] for r in sorted_by_delta[:args.ig_pool]]
            print(f"  IG pool: top {len(ig_pool_indices)} sentences by delta_bias")
        else:
            ig_pool_indices = None

        ig_records = run_per_sentence_ig_jaccard(
            encoder_model=encoder_model,
            ner_model_fn=model_fn,
            tokenizer=tokenizer,
            test_data=test_data,
            thresholds=thresholds,
            device=device,
            is_gpt2=is_gpt2,
            ig_k=args.ig_k,
            n_steps=args.ig_steps,
            sentence_indices=ig_pool_indices,
            bar_threshold=args.bar_threshold,
        )
    else:
        print("\n[Phase 1.5] IG + Jaccard: SKIPPED (--skip-ig)")

    # ── Phase 2: Per-sentence head ablation ─────────────────────
    ablation_records = None
    if not args.skip_ablation:
        print("\n[Phase 2] Per-sentence Head Ablation")

        # Select which sentences to ablate
        if args.ablation_pool > 0:
            # Sort by delta_bias descending, pick top-M
            sorted_by_delta = sorted(
                faith_records,
                key=lambda r: r["delta_bias"],
                reverse=True,
            )
            pool_indices = [r["index"] for r in sorted_by_delta[:args.ablation_pool]]
            print(f"  Ablation pool: top {len(pool_indices)} sentences by delta_bias")
        else:
            pool_indices = None  # all sentences

        ablation_records = run_per_sentence_ablation(
            encoder_model=encoder_model,
            lm_head_model=lm_head_model,
            ner_model_fn=model_fn,
            tokenizer=tokenizer,
            test_data=test_data,
            thresholds=thresholds,
            device=device,
            is_gpt2=is_gpt2,
            top_k_heads=args.top_k_heads,
            sentence_indices=pool_indices,
        )
    else:
        print("\n[Phase 2] Head Ablation: SKIPPED (--skip-ablation)")

    # ── Phase 3: Composite score & ranking ──────────────────────
    print("\n[Phase 3] Computing composite scores and ranking...")
    ranked = merge_and_rank(faith_records, ig_records, ablation_records, test_data)

    # ── Post-hoc filtering ──────────────────────────────────────
    has_ig = ig_records is not None and len(ig_records) > 0
    has_ablation = ablation_records is not None and len(ablation_records) > 0

    if args.min_ig_jaccard > 0 or args.min_dl_jaccard > 0:
        before = len(ranked)
        filtered = ranked
        if args.min_ig_jaccard > 0:
            filtered = [
                r for r in filtered
                if (r.ig_spec_mean_jaccard or 0) >= args.min_ig_jaccard
            ]
        if args.min_dl_jaccard > 0:
            filtered = [
                r for r in filtered
                if (r.dl_spec_mean_jaccard or 0) >= args.min_dl_jaccard
            ]
        print(f"\n  Post-hoc filter: {before} → {len(filtered)} sentences "
              f"(min_ig_jac={args.min_ig_jaccard}, min_dl_jac={args.min_dl_jaccard})")
        ranked = filtered

    # ── Output ──────────────────────────────────────────────────
    print_importance_report(ranked, args.top_n, aggregated, has_ig, has_ablation,
                           bar_threshold=args.bar_threshold)

    metadata = {
        "backbone": args.backbone,
        "model_dir": args.model_dir,
        "dataset_source": args.dataset_source,
        "n_test_sentences": len(test_data),
        "n_with_bias_predicted": aggregated["n_sequences_with_bias"],
        "ig_pool_size": args.ig_pool if not args.skip_ig else 0,
        "ig_k": args.ig_k,
        "ig_steps": args.ig_steps,
        "ablation_pool_size": args.ablation_pool if not args.skip_ablation else 0,
        "top_k_heads": args.top_k_heads,
        "skip_ig": args.skip_ig,
        "skip_ablation": args.skip_ablation,
        "bar_threshold": args.bar_threshold,
        "timestamp": datetime.now().isoformat(),
    }

    if args.output_json:
        save_json_report(ranked, aggregated, metadata, args.output_json)

    if args.output_csv:
        save_csv_report(ranked, args.output_csv)

    if not args.output_json and not args.output_csv:
        print("\nTip: use --output-json or --output-csv to save full results.")


if __name__ == "__main__":
    main()
