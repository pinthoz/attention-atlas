"""
Span Faithfulness Evaluation — Minimality Metric
=================================================

Given a trained GUS-Net model, measures whether predicted bias spans are
"necessary" for the bias prediction by masking them out and observing
the drop in sentence-level bias score.

Metric:
    delta_bias = orig_bias_score - counterfactual_bias_score

A higher delta_bias means the predicted spans are more faithful:
removing exactly those tokens causes the largest decrease in the model's
bias confidence.

Usage:
    python span_faithfulness_eval.py \\
        --backbone bert \\
        --model-dir attention_app/bias/models/gus-net-bert-final-new \\
        --thresholds attention_app/bias/models/gus-net-bert-final-new/optimized_thresholds.npy

Or import and call directly:
    from span_faithfulness_eval import evaluate_faithfulness
    results = evaluate_faithfulness(model_fn, dataloader, thresholds, mask_token_id, num_labels, device)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader


# ============================================================================
# CORE METRIC
# ============================================================================

def compute_sentence_bias_scores(logits):
    """
    Compute per-sequence bias scores from raw logits.

    For each token, collapses BIO → 3 bias categories via sigmoid + max,
    then takes the maximum bias probability across all tokens in the
    sequence as the sentence-level score.

    Args:
        logits: (batch, seq_len, 7) raw logits

    Returns:
        sentence_scores: (batch,) max per-token bias probability per sequence
        per_cat_scores:  (batch, 3) max per-token probability per category
                         [STEREO, GEN, UNFAIR]
    """
    probs = torch.sigmoid(logits.float())  # (B, T, 7)

    # Collapse BIO → categories
    stereo = torch.max(probs[..., 1], probs[..., 2])   # (B, T)
    gen    = torch.max(probs[..., 3], probs[..., 4])    # (B, T)
    unfair = torch.max(probs[..., 5], probs[..., 6])    # (B, T)

    cat_probs = torch.stack([stereo, gen, unfair], dim=-1)  # (B, T, 3)

    # Sentence-level: max over tokens
    per_cat_scores = cat_probs.max(dim=1).values       # (B, 3)
    sentence_scores = per_cat_scores.max(dim=-1).values  # (B,)

    return sentence_scores, per_cat_scores


def get_predicted_bias_mask(logits, thresholds, num_labels=7):
    """
    Identify which tokens are predicted as biased (any non-O label above
    threshold).

    Args:
        logits:     (batch, seq_len, 7) raw logits
        thresholds: np.ndarray (7,) per-class thresholds
        num_labels: number of BIO labels (default 7)

    Returns:
        bias_mask: (batch, seq_len) bool — True where token is predicted biased
    """
    probs = torch.sigmoid(logits.float()).cpu().numpy()
    thr = thresholds.reshape(1, 1, num_labels)
    preds = (probs >= thr).astype(int)

    # Token is "biased" if ANY non-O class is predicted
    # Non-O classes are indices 1..6
    bias_mask = preds[:, :, 1:].any(axis=-1)  # (B, T)
    return torch.tensor(bias_mask, dtype=torch.bool)


@torch.no_grad()
def evaluate_faithfulness(
    model_fn,
    dataloader,
    thresholds,
    mask_token_id,
    num_labels,
    device,
    attention_mask_key="attention_mask",
    return_per_sentence=False,
):
    """
    Evaluate faithfulness of predicted bias spans via counterfactual masking.

    Protocol:
        1. Run model on original input → orig_bias_score (max bias prob)
        2. Identify predicted biased tokens (thresholded)
        3. Replace biased tokens with mask_token_id
        4. Re-run model on masked input → cf_bias_score
        5. delta_bias = orig - cf (per sequence)
        6. Aggregate over dataset

    Args:
        model_fn:       callable(input_ids, attention_mask) → logits (B, T, 7)
                        Should be model.eval() and on `device`.
        dataloader:     yields batches of (input_ids, attention_mask, labels)
                        OR dicts with "input_ids", "attention_mask", "labels".
        thresholds:     np.ndarray (7,) optimised per-class thresholds
        mask_token_id:  int — token id for masking (e.g. tokenizer.mask_token_id
                        for BERT, tokenizer.eos_token_id for GPT-2)
        num_labels:     int (7)
        device:         torch.device or str

    Returns:
        dict with keys:
            "delta_bias_mean":    float — average delta_bias over all sequences
            "delta_bias_std":     float — std of delta_bias
            "delta_bias_median":  float — median delta_bias
            "delta_stereo_mean":  float — per-category delta for STEREO
            "delta_gen_mean":     float — per-category delta for GEN
            "delta_unfair_mean":  float — per-category delta for UNFAIR
            "frac_biased_tokens": float — average fraction of tokens predicted as biased
            "n_sequences":        int   — total sequences evaluated
            "n_sequences_with_bias": int — sequences that had at least one biased token

        If return_per_sentence=True, returns (dict, list) where the list
        contains one record per sentence with per-sentence metrics.
    """
    all_delta = []
    all_delta_cat = []  # (N, 3) per-category deltas
    all_frac_biased = []
    per_sentence_records = [] if return_per_sentence else None
    n_total = 0
    n_with_bias = 0

    for batch in dataloader:
        # Handle both tuple and dict batch formats
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
        else:
            input_ids = batch[0].to(device)
            attn_mask = batch[1].to(device)
            labels = batch[2]

        batch_size, seq_len = input_ids.shape

        # --- Original forward pass ---
        logits_orig = model_fn(input_ids, attn_mask)  # (B, T, 7)
        orig_scores, orig_cat_scores = compute_sentence_bias_scores(logits_orig)
        # orig_scores: (B,), orig_cat_scores: (B, 3)

        # --- Identify predicted biased tokens ---
        bias_mask = get_predicted_bias_mask(
            logits_orig, thresholds, num_labels,
        ).to(device)  # (B, T) bool

        # --- Build counterfactual input ---
        input_ids_cf = input_ids.clone()
        input_ids_cf[bias_mask] = mask_token_id

        # --- Counterfactual forward pass ---
        logits_cf = model_fn(input_ids_cf, attn_mask)
        cf_scores, cf_cat_scores = compute_sentence_bias_scores(logits_cf)

        # --- Compute deltas ---
        delta = (orig_scores - cf_scores).cpu().numpy()           # (B,)
        delta_cat = (orig_cat_scores - cf_cat_scores).cpu().numpy()  # (B, 3)

        # --- Fraction of biased tokens per sequence ---
        # Use valid-token mask (attention_mask == 1 AND label != -100)
        if isinstance(labels, torch.Tensor):
            valid_mask = (labels[..., 0] >= 0).to(device) & (attn_mask == 1)
        else:
            valid_mask = attn_mask == 1
        n_valid = valid_mask.float().sum(dim=1).clamp(min=1.0)
        n_biased = (bias_mask & valid_mask).float().sum(dim=1)
        frac_biased = (n_biased / n_valid).cpu().numpy()

        # --- Accumulate ---
        for i in range(batch_size):
            has_bias = bias_mask[i].any().item()
            n_total += 1
            all_frac_biased.append(frac_biased[i])
            if has_bias:
                n_with_bias += 1
                all_delta.append(delta[i])
                all_delta_cat.append(delta_cat[i])

            if return_per_sentence:
                per_sentence_records.append({
                    "index": n_total - 1,
                    "delta_bias": float(delta[i]) if has_bias else 0.0,
                    "delta_stereo": float(delta_cat[i, 0]) if has_bias else 0.0,
                    "delta_gen": float(delta_cat[i, 1]) if has_bias else 0.0,
                    "delta_unfair": float(delta_cat[i, 2]) if has_bias else 0.0,
                    "frac_biased_tokens": float(frac_biased[i]),
                    "has_bias_predicted": has_bias,
                    "orig_score": float(orig_scores[i].cpu()),
                    "cf_score": float(cf_scores[i].cpu()),
                })

    # --- Aggregate ---
    if len(all_delta) == 0:
        result = {
            "delta_bias_mean": 0.0,
            "delta_bias_std": 0.0,
            "delta_bias_median": 0.0,
            "delta_stereo_mean": 0.0,
            "delta_gen_mean": 0.0,
            "delta_unfair_mean": 0.0,
            "frac_biased_tokens": 0.0,
            "n_sequences": n_total,
            "n_sequences_with_bias": 0,
        }
        if return_per_sentence:
            return result, per_sentence_records
        return result

    all_delta = np.array(all_delta)
    all_delta_cat = np.array(all_delta_cat)  # (N_with_bias, 3)

    result = {
        "delta_bias_mean":   round(float(all_delta.mean()), 4),
        "delta_bias_std":    round(float(all_delta.std()), 4),
        "delta_bias_median": round(float(np.median(all_delta)), 4),
        "delta_stereo_mean": round(float(all_delta_cat[:, 0].mean()), 4),
        "delta_gen_mean":    round(float(all_delta_cat[:, 1].mean()), 4),
        "delta_unfair_mean": round(float(all_delta_cat[:, 2].mean()), 4),
        "frac_biased_tokens": round(float(np.mean(all_frac_biased)), 4),
        "n_sequences":       n_total,
        "n_sequences_with_bias": n_with_bias,
    }
    if return_per_sentence:
        return result, per_sentence_records
    return result


def print_faithfulness_report(results):
    """Pretty-print the faithfulness evaluation results."""
    print("\n" + "=" * 60)
    print("FAITHFULNESS / MINIMALITY EVALUATION")
    print("=" * 60)
    print(f"  Sequences evaluated:          {results['n_sequences']}")
    print(f"  Sequences with bias predicted: {results['n_sequences_with_bias']}")
    print(f"  Avg fraction biased tokens:   {results['frac_biased_tokens']:.4f}")
    print()
    print("  delta_bias (higher = more faithful):")
    print(f"    Mean:   {results['delta_bias_mean']:.4f}")
    print(f"    Std:    {results['delta_bias_std']:.4f}")
    print(f"    Median: {results['delta_bias_median']:.4f}")
    print()
    print("  Per-category delta (mean):")
    print(f"    STEREO: {results['delta_stereo_mean']:.4f}")
    print(f"    GEN:    {results['delta_gen_mean']:.4f}")
    print(f"    UNFAIR: {results['delta_unfair_mean']:.4f}")
    print("=" * 60)


# ============================================================================
# STANDALONE CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of predicted bias spans",
    )
    parser.add_argument(
        "--backbone", choices=["bert", "gpt2"], required=True,
        help="Model backbone type",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--thresholds", type=str, required=True,
        help="Path to optimized_thresholds.npy",
    )
    parser.add_argument(
        "--dataset-source", choices=["hf", "gemini", "clean"], default="hf",
        help="Dataset source (default: hf)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    thresholds = np.load(args.thresholds)
    num_labels = 7

    label2id = {
        "O": 0, "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4, "B-UNFAIR": 5, "I-UNFAIR": 6,
    }
    id2label = {v: k for k, v in label2id.items()}

    if args.backbone == "bert":
        from transformers import BertTokenizerFast, BertForTokenClassification

        tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
        model = BertForTokenClassification.from_pretrained(args.model_dir)
        model.eval().to(device)
        mask_token_id = tokenizer.mask_token_id

        def model_fn(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

    elif args.backbone == "gpt2":
        from transformers import GPT2TokenizerFast, GPT2ForTokenClassification

        tokenizer = GPT2TokenizerFast.from_pretrained(
            args.model_dir, add_prefix_space=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForTokenClassification.from_pretrained(args.model_dir)
        model.eval().to(device)
        # GPT-2 has no [MASK] token; use pad/eos as neutral replacement
        mask_token_id = tokenizer.eos_token_id

        def model_fn(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Load and tokenize data
    from gus_net_training import load_from_hf, load_from_gemini, bio_postprocess

    if args.dataset_source == "clean":
        from gus_net_training_paper import load_from_clean_json
        all_data = load_from_clean_json()
    elif args.dataset_source == "gemini":
        all_data = load_from_gemini()
    else:
        all_data = load_from_hf()

    # Use last 20% as test set (same split as training script)
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(
        all_data, test_size=0.20, random_state=42, shuffle=True,
    )

    # Tokenize
    import ast
    from torch.utils.data import Dataset as TorchDataset

    class SimpleNERDataset(TorchDataset):
        def __init__(self, data, tokenizer, label2id, max_length=128):
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

    dataset = SimpleNERDataset(test_data, tokenizer, label2id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Evaluating faithfulness on {len(dataset)} test sequences...")
    results = evaluate_faithfulness(
        model_fn=model_fn,
        dataloader=dataloader,
        thresholds=thresholds,
        mask_token_id=mask_token_id,
        num_labels=num_labels,
        device=device,
    )

    print_faithfulness_report(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
