"""
GUS-Net Training (BERT / GPT-2)
=====================================

Unified training script for GUS-Net using either:
  - BERT backbone (`bert-base-uncased`)
  - GPT-2 backbone (`gpt2`)

Select the backbone at runtime via user input.

Uses the `ethical-spectacle/gus-dataset-v1` from Hugging Face.

Label scheme (7 classes, shared by BERT and GPT-2):
    0: O  |  1: B-STEREO  |  2: I-STEREO  |  3: B-GEN  |  4: I-GEN  |  5: B-UNFAIR  |  6: I-UNFAIR
"""

import os
import gc
import glob
import json
import ast
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    GPT2TokenizerFast,
    GPT2ForTokenClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from scipy.optimize import minimize_scalar


# ============================================================================
# SHARED LOSS & POST-PROCESSING UTILITIES
# ============================================================================

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label token classification.

    Key properties vs standard focal loss:
    - gamma_pos: focal factor for positives (lower → preserve recall on rare bias tags)
    - gamma_neg: focal factor for negatives (higher → down-weight confident O predictions)
    - clip: shifts negative probs down by `clip` before computing loss.
            Tokens where p_bias < clip contribute 0 loss → drastically reduces false positives
    - per-class alpha tensor from training-data frequencies (clipped to max 10x ratio)

    Based on: Ridnik et al. 2021 "Asymmetric Loss For Multi-Label Classification"
    """

    def __init__(self, alpha, gamma_pos=1.0, gamma_neg=3.0, clip=0.05,
                 label_smoothing=0.0, reduction="mean"):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha.float())
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weight=None):
        # Cast to float32 to avoid fp16 underflow (log(0) = -inf → NaN gradients)
        inputs = inputs.float()
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(inputs)
        # Shift negative probs down: p < clip → 0, reduces false-positive gradient
        # clip=0.01 is conservative — avoids killing gradients for weakly-predicted bias tokens
        probs_neg = torch.clamp(probs - self.clip, min=0.0)

        # Separate BCE components (clamp to fp32-safe floor 1e-4 instead of 1e-8)
        loss_pos = -targets * torch.log(probs.clamp(min=1e-4))
        loss_neg = -(1.0 - targets) * torch.log((1.0 - probs_neg).clamp(min=1e-4))

        # Asymmetric focal modulation
        focal_pos = (1.0 - probs).clamp(min=0.0) ** self.gamma_pos
        focal_neg = probs_neg ** self.gamma_neg
        focal = torch.where(targets > 0.5, focal_pos, focal_neg)

        loss = focal * (loss_pos + loss_neg)
        loss = self.alpha.to(inputs.device) * loss

        # Per-token span-position weights (progressive I-tag penalisation)
        if sample_weight is not None:
            loss = loss * sample_weight

        return loss.mean() if self.reduction == "mean" else loss.sum()


def compute_alpha_from_data(samples, label2id, max_ratio=10.0):
    """
    Compute per-class alpha weights from label frequencies in training samples.
    Clips the max/min weight ratio to `max_ratio` to avoid extreme class imbalance.

    Args:
        samples:   list of dicts with keys "text_str" and "ner_tags"
        label2id:  dict mapping label string to index
        max_ratio: maximum allowed ratio between largest and smallest alpha

    Returns:
        alpha: torch.FloatTensor of shape (num_labels,)
        label_counts: np.ndarray with token-level positive counts per class
    """
    num_labels = len(label2id)
    label_counts = np.zeros(num_labels, dtype=np.int64)
    total_tokens = 0

    for sample in samples:
        annotations = sample["ner_tags"]
        for word_tags in annotations:
            for tag in word_tags:
                if tag in label2id:
                    label_counts[label2id[tag]] += 1
            total_tokens += 1

    label_counts = np.maximum(label_counts, 1)
    freq = label_counts / float(max(total_tokens, 1))
    inv_freq = 1.0 / freq
    # Clip ratio: prevent extreme down-weighting of majority class
    inv_freq = np.clip(inv_freq, inv_freq.min(), inv_freq.min() * max_ratio)
    alpha = inv_freq / inv_freq.sum()
    return torch.tensor(alpha, dtype=torch.float32), label_counts


def save_training_log(entry, log_path=None):
    """
    Append a training run entry to a persistent JSON log file.
    Creates the file if it doesn't exist; appends otherwise.

    Args:
        entry:    dict with training metadata and results
        log_path: Path to the JSON file (default: <script_dir>/training_log.json)
    """
    if log_path is None:
        log_path = _SCRIPT_DIR / "training_log.json"
    log_path = Path(log_path)

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\nTraining log updated → {log_path}  ({len(log)} run(s) total)")


def _report_to_dict(report_dict, keys):
    """Extract precision/recall/f1-score from a classification_report dict."""
    out = {}
    for k in keys:
        if k in report_dict:
            v = report_dict[k]
            out[k] = {
                "precision": round(float(v["precision"]), 4),
                "recall":    round(float(v["recall"]), 4),
                "f1":        round(float(v["f1-score"]), 4),
                "support":   int(v["support"]),
            }
    return out


def bio_postprocess(seq_preds, id2label):
    """
    Enforce BIO validity on a single token sequence.

    Rules (applied in order):
    1. If any bias tag is predicted, suppress O
    2. Orphan I-X (no preceding B-X or I-X) → convert to B-X
    3. Token with no label → assign O

    Args:
        seq_preds: np.ndarray (seq_len, num_labels), binary int
        id2label:  dict {int: str}
    Returns:
        corrected: np.ndarray same shape
    """
    label_names = [id2label[i] for i in sorted(id2label.keys())]
    O_idx = next(i for i, n in enumerate(label_names) if n == "O")
    corrected = seq_preds.copy().astype(int)
    active_types: set = set()

    for t in range(len(corrected)):
        # 1) Suppress O when any bias tag is active
        if any(corrected[t, i] == 1 for i, n in enumerate(label_names) if n != "O"):
            corrected[t, O_idx] = 0

        # 2) Fix orphan I-X → B-X
        new_active: set = set()
        for i, name in enumerate(label_names):
            if corrected[t, i] != 1:
                continue
            if name.startswith("B-"):
                new_active.add(name[2:])
            elif name.startswith("I-"):
                bio_type = name[2:]
                if bio_type not in active_types:
                    b_idx = next(
                        (j for j, n in enumerate(label_names) if n == f"B-{bio_type}"),
                        None,
                    )
                    if b_idx is not None:
                        corrected[t, i] = 0
                        corrected[t, b_idx] = 1
                new_active.add(bio_type)
        active_types = new_active

        # 3) No label → assign O
        if corrected[t].sum() == 0:
            corrected[t, O_idx] = 1

    return corrected


def compute_itag_span_weights(labels, id2label,
                              decay=0.18, min_weight=0.30,
                              label_decay=0.08, label_floor=0.55):
    """
    Compute per-token loss weights AND smoothed labels for I-tags deeper
    in BIO spans.

    Two complementary mechanisms:
      1) Weight decay: reduces loss gradient for deep I-tags so the model
         cares less about getting them "right"
      2) Label softening: lowers the target itself so the model learns to
         predict lower probabilities for deep I-tag positions

    Weight schedule (per I-tag column only):
        B-X  → weight 1.0, label 1.0  (unaffected)
        I-X position 1 → weight (1 - decay*1), label (1 - label_decay*1)
        I-X position N → weight max(min_weight, 1 - decay*N),
                         label  max(label_floor, 1 - label_decay*N)

    All non-I-tag columns always keep weight 1.0 and label unchanged.

    Args:
        labels:      np.ndarray (batch, seq_len, num_labels) or (seq_len, num_labels)
        id2label:    dict {int: str}
        decay:       loss weight reduction per I-tag position (default 0.18)
        min_weight:  weight floor (default 0.30)
        label_decay: label target reduction per I-tag position (default 0.08)
        label_floor: minimum label target — must stay > 0.5 so focal loss
                     treats it as a positive (default 0.55)

    Returns:
        weights:         np.ndarray same shape as labels
        smoothed_labels: np.ndarray same shape as labels (I-tag targets decayed,
                         padding tokens preserved as -100)
    """
    squeezed = False
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
        squeezed = True

    batch_size, seq_len, n_labels = labels.shape
    weights = np.ones_like(labels, dtype=np.float32)
    smoothed = labels.copy().astype(np.float32)

    # Map bias types → (B-col, I-col)
    bias_types = {}
    for idx, name in id2label.items():
        if name.startswith("B-"):
            bt = name[2:]
            bias_types.setdefault(bt, {})["b"] = idx
        elif name.startswith("I-"):
            bt = name[2:]
            bias_types.setdefault(bt, {})["i"] = idx

    # Only process types that have both B and I columns
    bi_pairs = [(v["b"], v["i"]) for v in bias_types.values()
                if "b" in v and "i" in v]

    for b in range(batch_size):
        for b_col, i_col in bi_pairs:
            span_pos = -1  # -1 = not in a span
            for t in range(seq_len):
                # Skip padding tokens (label = -100)
                if labels[b, t, 0] < -99.0:
                    span_pos = -1
                    continue

                if labels[b, t, b_col] > 0.5:
                    # B-tag: start new span, position 0 (B itself keeps weight 1.0)
                    span_pos = 0
                elif labels[b, t, i_col] > 0.5 and span_pos >= 0:
                    # I-tag continuing a span
                    span_pos += 1
                    weights[b, t, i_col] = max(min_weight,
                                               1.0 - decay * span_pos)
                    smoothed[b, t, i_col] = max(label_floor,
                                                1.0 - label_decay * span_pos)
                else:
                    # Not in a span for this bias type
                    span_pos = -1

    if squeezed:
        return weights[0], smoothed[0]
    return weights, smoothed


# ============================================================================
# PATHS
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
SEED = 42


# ============================================================================
# SHARED HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_EPOCHS = 20
THRESHOLD = 0.5
PATIENCE = 3
MAX_LENGTH = 128

# Per-class minimum threshold floors applied after grid/scalar optimisation.
# I-tags use higher floors than B-tags to prevent span bleed:
# once a span starts (B-), the model tends to assign high I- scores to all
# subsequent tokens including function words — forcing a higher floor on I-tags
# keeps only the most confident in-span tokens.
MIN_THR_PER_CLASS = {
    "O":        0.45,
    "B-STEREO": 0.50,
    "I-STEREO": 0.80,   # high — prevent stereotype bleed to function words
    "B-GEN":    0.40,
    "I-GEN":    0.65,
    "B-UNFAIR": 0.40,
    "I-UNFAIR": 0.65,
}

# Progressive I-tag span penalisation: reduce loss weight AND soften labels
# for I-tags deeper in BIO spans so the model learns lower confidence for
# later in-span tokens (typically function words like "are", "and", "for").
#
# Two complementary mechanisms:
#   1) Weight decay  — model cares LESS about deep I-tags (lower gradient)
#   2) Label softening — target itself drops (model learns lower probabilities)
SPAN_DECAY = 0.18           # loss weight reduction per I-tag position
SPAN_MIN_WEIGHT = 0.30      # weight floor (deep I-tags → 30 % of normal loss)
SPAN_LABEL_DECAY = 0.08     # label target reduction per I-tag position
SPAN_LABEL_FLOOR = 0.55     # label floor (stays > 0.5 so focal treats as positive)


# ============================================================================
# BERT-SPECIFIC CONFIG
# ============================================================================

BERT_CONFIG = {
    "backbone": "bert-base-uncased",
    "alpha": 0.75,
    "gamma": 2.0,
    "save_dir": _SCRIPT_DIR / "gus-net-bert-final-new",
    "label2id": {
        "O": 0,
        "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4,
        "B-UNFAIR": 5, "I-UNFAIR": 6,
    },
}
BERT_CONFIG["num_labels"] = len(BERT_CONFIG["label2id"])
BERT_CONFIG["id2label"] = {v: k for k, v in BERT_CONFIG["label2id"].items()}


# ============================================================================
# GPT-2-SPECIFIC CONFIG
# ============================================================================

GPT2_CONFIG = {
    "backbone": "gpt2",
    "gamma": 1.5,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-final-new",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-new"),
    "label2id": {
        "O": 0,
        "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4,
        "B-UNFAIR": 5, "I-UNFAIR": 6,
    },
}
GPT2_CONFIG["num_labels"] = len(GPT2_CONFIG["label2id"])
GPT2_CONFIG["id2label"] = {v: k for k, v in GPT2_CONFIG["label2id"].items()}


# ============================================================================
# BERT-LARGE CONFIG
# ============================================================================

BERT_LARGE_CONFIG = {
    "backbone": "bert-large-uncased",
    "alpha": 0.75,
    "gamma": 2.0,
    "save_dir": _SCRIPT_DIR / "gus-net-bert-large-final-new",
    "label2id": {
        "O": 0,
        "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4,
        "B-UNFAIR": 5, "I-UNFAIR": 6,
    },
}
BERT_LARGE_CONFIG["num_labels"] = len(BERT_LARGE_CONFIG["label2id"])
BERT_LARGE_CONFIG["id2label"] = {v: k for k, v in BERT_LARGE_CONFIG["label2id"].items()}


# ============================================================================
# GPT-2 MEDIUM CONFIG
# ============================================================================

GPT2_MEDIUM_CONFIG = {
    "backbone": "gpt2-medium",
    "gamma": 1.5,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-medium-final-new",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-medium-new"),
    "label2id": {
        "O": 0,
        "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4,
        "B-UNFAIR": 5, "I-UNFAIR": 6,
    },
}
GPT2_MEDIUM_CONFIG["num_labels"] = len(GPT2_MEDIUM_CONFIG["label2id"])
GPT2_MEDIUM_CONFIG["id2label"] = {v: k for k, v in GPT2_MEDIUM_CONFIG["label2id"].items()}


# ============================================================================
# GPT-2 LARGE CONFIG
# ============================================================================

GPT2_LARGE_CONFIG = {
    "backbone": "gpt2-large",
    "gamma": 1.5,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-large-final-new",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-large-new"),
    "label2id": {
        "O": 0,
        "B-STEREO": 1, "I-STEREO": 2,
        "B-GEN": 3, "I-GEN": 4,
        "B-UNFAIR": 5, "I-UNFAIR": 6,
    },
}
GPT2_LARGE_CONFIG["num_labels"] = len(GPT2_LARGE_CONFIG["label2id"])
GPT2_LARGE_CONFIG["id2label"] = {v: k for k, v in GPT2_LARGE_CONFIG["label2id"].items()}


# ============================================================================
# DATA LOADING (SHARED)
# ============================================================================

def load_from_hf():
    """Load dataset from Hugging Face and transform."""
    print("Loading dataset from ethical-spectacle/gus-dataset-v1...")
    dataset = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")

    transformed = []
    for entry in dataset:
        text_str = entry["text_str"]
        try:
            combined_tags = ast.literal_eval(entry["ner_tags"])
        except (ValueError, SyntaxError):
            print(f"Skipping entry {entry.get('id', '?')} due to tag parse error.")
            continue
        transformed.append({
            "text_str": text_str,
            "ner_tags": combined_tags,
            "id": entry.get("id", -1),
        })
    return transformed


def load_from_gemini():
    """Load dataset from gemini_annotations.json."""
    gemini_path = _PROJECT_ROOT / "notebooks" / "gemini_annotations.json"
    print(f"Loading dataset from {gemini_path}...")
    with open(gemini_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    transformed = []
    for i, entry in enumerate(raw):
        text = entry["text"]
        tags = entry["gemini_annotations"]
        # Wrap each tag in a list to match multi-label format from HF dataset
        ner_tags = [[tag] for tag in tags]
        transformed.append({
            "text_str": text,
            "ner_tags": ner_tags,
            "id": i,
        })
    print(f"Loaded {len(transformed)} samples from Gemini annotations")
    return transformed


# ============================================================================
# BERT TRAINING
# ============================================================================

def train_bert(config=None, dataset_source="hf"):
    """Train GUS-Net with BERT backbone using PyTorch Lightning."""
    if config is None:
        config = BERT_CONFIG

    dataset_name = "ethical-spectacle/gus-dataset-v1" if dataset_source == "hf" else "gemini_annotations.json"

    print("=" * 60)
    print("GUS-Net BERT Training")
    print("=" * 60)
    print(f"Backbone:       {config['backbone']}")
    print(f"Dataset:        {dataset_name}")
    print(f"Labels:         {config['num_labels']} ({', '.join(config['label2id'].keys())})")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Focal loss:     alpha={config['alpha']}, gamma={config['gamma']}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Max epochs:     {MAX_EPOCHS}")
    print(f"Patience:       {PATIENCE}")
    print(f"Device:         {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config["backbone"])
    label2id = config["label2id"]
    num_labels = config["num_labels"]

    def tokenize_and_align_labels(text, annotations, max_length=MAX_LENGTH):
        """Tokenize pre-split words and align multi-label vectors."""
        original_words = text.split()
        if len(original_words) != len(annotations):
            min_len = min(len(original_words), len(annotations))
            original_words = original_words[:min_len]
            annotations = annotations[:min_len]

        tokenized = tokenizer(
            original_words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = tokenized.word_ids()

        aligned_labels = []
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append([-100] * num_labels)
            else:
                tags = annotations[word_idx]
                label_vector = [0] * num_labels
                for tag in tags:
                    if tag in label2id:
                        label_vector[label2id[tag]] = 1
                aligned_labels.append(label_vector)
        return tokenized, aligned_labels

    def preprocess_data(df):
        """Tokenize and align all rows in a DataFrame."""
        tokenized_list, labels_list = [], []
        for _, row in df.iterrows():
            inputs, labels = tokenize_and_align_labels(row["text_str"], row["ner_tags"])
            tokenized_list.append(inputs)
            labels_list.append(labels)
        return tokenized_list, labels_list

    # Dataset class
    class NERDataset(Dataset):
        def __init__(self, tokenized_texts, labels):
            self.input_ids = [t["input_ids"].squeeze() for t in tokenized_texts]
            self.attention_mask = [t["attention_mask"].squeeze() for t in tokenized_texts]
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return (
                self.input_ids[idx],
                self.attention_mask[idx],
                torch.tensor(self.labels[idx]),
            )

    # DataModule
    class NERDataModule(pl.LightningDataModule):
        def __init__(self, tokenized_texts, labels, batch_size=BATCH_SIZE, val_split=0.15):
            super().__init__()
            self.dataset = NERDataset(tokenized_texts, labels)
            self.batch_size = batch_size
            self.val_split = val_split

        def setup(self, stage=None):
            val_size = int(len(self.dataset) * self.val_split)
            train_size = len(self.dataset) - val_size
            self.train_ds, self.val_ds = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(SEED),
            )

        def train_dataloader(self):
            return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        def val_dataloader(self):
            return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)

    # Lightning Model
    class GUSNetBERT(pl.LightningModule):
        def __init__(self, learning_rate=LEARNING_RATE, threshold=THRESHOLD,
                     alpha_per_class=None, gamma_pos=1.0, gamma_neg=None):
            super().__init__()
            if gamma_neg is None:
                gamma_neg = config["gamma"]
            self.save_hyperparameters(ignore=["alpha_per_class"])
            self.bert = BertForTokenClassification.from_pretrained(
                config["backbone"], num_labels=num_labels,
            )
            self.learning_rate = learning_rate
            self.threshold = threshold
            self.gamma_pos = gamma_pos
            self.gamma_neg = gamma_neg

            if alpha_per_class is None:
                alpha_per_class = torch.ones(num_labels, dtype=torch.float32) / num_labels
            self.loss_fn = AsymmetricFocalLoss(
                alpha=alpha_per_class,
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                clip=0.01,
            )

        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

        def focal_loss(self, logits, labels):
            # Compute span-position weights + smoothed labels BEFORE flattening
            sw_np, sl_np = compute_itag_span_weights(
                labels.cpu().numpy(), config["id2label"],
                decay=SPAN_DECAY, min_weight=SPAN_MIN_WEIGHT,
                label_decay=SPAN_LABEL_DECAY, label_floor=SPAN_LABEL_FLOOR,
            )
            span_w = torch.tensor(sw_np, dtype=torch.float32, device=logits.device)
            smooth = torch.tensor(sl_np, dtype=torch.float32, device=logits.device)

            logits = logits.view(-1, num_labels)
            orig    = labels.view(-1, num_labels).float()   # for valid mask
            smooth  = smooth.view(-1, num_labels)
            span_w  = span_w.view(-1, num_labels)
            valid_mask = (orig >= 0).all(dim=1)
            logits = logits[valid_mask]
            smooth = smooth[valid_mask]
            span_w = span_w[valid_mask]
            if logits.shape[0] == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            return self.loss_fn(logits, smooth, sample_weight=span_w)

        def training_step(self, batch, batch_idx):
            logits = self(batch[0], batch[1])
            loss = self.focal_loss(logits, batch[2])
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            logits = self(batch[0], batch[1])
            loss = self.focal_loss(logits, batch[2])
            self.log("val_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            no_decay = ["bias", "LayerNorm.weight"]
            n_layers = self.bert.config.num_hidden_layers
            llrd_decay = 0.9
            param_groups = []

            # Classifier head — 10× base LR, no weight decay
            param_groups.append({
                "params": [p for n, p in self.named_parameters() if "classifier" in n],
                "lr": self.learning_rate * 10,
                "weight_decay": 0.0,
            })

            # Transformer layers: top → bottom, decaying LR
            for layer_idx in range(n_layers - 1, -1, -1):
                lr = self.learning_rate * (llrd_decay ** (n_layers - 1 - layer_idx))
                prefix = f"bert.bert.encoder.layer.{layer_idx}."
                d  = [p for n, p in self.named_parameters()
                      if prefix in n and not any(k in n for k in no_decay)]
                nd = [p for n, p in self.named_parameters()
                      if prefix in n and any(k in n for k in no_decay)]
                if d:  param_groups.append({"params": d,  "lr": lr, "weight_decay": 0.01})
                if nd: param_groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

            # Embeddings — lowest LR
            emb_lr = self.learning_rate * (llrd_decay ** n_layers)
            d  = [p for n, p in self.named_parameters()
                  if "bert.bert.embeddings" in n and not any(k in n for k in no_decay)]
            nd = [p for n, p in self.named_parameters()
                  if "bert.bert.embeddings" in n and any(k in n for k in no_decay)]
            if d:  param_groups.append({"params": d,  "lr": emb_lr, "weight_decay": 0.01})
            if nd: param_groups.append({"params": nd, "lr": emb_lr, "weight_decay": 0.0})

            print(f"LLRD (BERT): classifier_lr={self.learning_rate * 10:.2e}, "
                  f"top_layer_lr={self.learning_rate:.2e}, emb_lr={emb_lr:.2e}")

            optimizer = AdamW(param_groups, lr=self.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * self.trainer.estimated_stepping_batches),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [scheduler]

    # Load data
    print("\n" + "=" * 60)
    print(f"LOADING DATA ({dataset_name})")
    print("=" * 60)
    all_data = load_from_gemini() if dataset_source == "gemini" else load_from_hf()
    print(f"Loaded {len(all_data)} samples")

    train_val_data, test_data = train_test_split(
        all_data, test_size=0.20, random_state=SEED, shuffle=True,
    )
    print(f"Train/Val: {len(train_val_data)}")
    print(f"Test:      {len(test_data)}")

    # Compute per-class alpha from training data
    print("\nComputing per-class alpha weights...")
    alpha_per_class, label_counts = compute_alpha_from_data(train_val_data, label2id)
    print("Per-class alpha (clipped 10× ratio):")
    for i, name in enumerate(config["id2label"].values()):
        print(f"  {name:10s}: count={label_counts[i]:>6}, alpha={alpha_per_class[i]:.4f}")

    # Tokenize
    print("\nTokenizing...")
    df = pd.DataFrame(train_val_data)
    tokenized_texts, labels = preprocess_data(df)
    print(f"Processed {len(tokenized_texts)} training samples")

    # DataModule
    data_module = NERDataModule(tokenized_texts, labels, batch_size=BATCH_SIZE)

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    shutil.rmtree("lightning_logs", ignore_errors=True)
    shutil.rmtree("checkpoints", ignore_errors=True)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="gusnet-bert-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=False,
        save_weights_only=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        verbose=True,
        mode="min",
    )

    model = GUSNetBERT(alpha_per_class=alpha_per_class)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )
    trainer.fit(model, data_module)

    # Evaluate
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
        print("No best model saved. Using current model.")
        best_model = model
    else:
        print(f"Loading best model from {best_model_path}")
        best_model = GUSNetBERT.load_from_checkpoint(best_model_path)

    best_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)

    # Threshold optimization
    def optimize_thresholds(model, val_loader):
        print("\nOptimizing thresholds on validation set...")
        model.eval()
        device = model.device
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                mask = batch[1].to(device)
                labels = batch[2].cpu().numpy()
                
                logits = model(input_ids, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels)
                
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        
        # Flatten valid tokens
        valid = labels[:, :, 0] != -100.0
        # Check if any valid tokens
        if not valid.any():
            print("Warning: No valid tokens found for optimization. Defaulting to 0.5")
            return np.full(num_labels, 0.5), 0.0

        pf = probs[valid]
        lf = labels[valid].astype(int)
        
        best_thr = np.zeros(num_labels, dtype=np.float32)
        
        print("Pass 1 - grid search:")
        grid = np.arange(0.05, 0.96, 0.025)
        
        for c in range(num_labels):
            best_f1, best_t = 0, 0.5
            # Vectorized grid search for speed if possible, but loop is fine for 7 labels
            for t in grid:
                f1 = f1_score(lf[:, c], (pf[:, c] >= t).astype(int), average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            best_thr[c] = best_t
            print(f"  {config['id2label'][c]:10s}: thr={best_t:.3f}, F1={best_f1:.4f}")
            
        print("\nPass 2 - refinement:")
        for c in range(num_labels):
            lo = max(0.01, best_thr[c] - 0.05)
            hi = min(0.99, best_thr[c] + 0.05)

            def neg_f1(t):
                pred = (pf[:, c] >= t).astype(int)
                return -f1_score(lf[:, c], pred, average="binary", zero_division=0)

            res = minimize_scalar(neg_f1, bounds=(lo, hi), method="bounded")
            if -res.fun >= f1_score(lf[:, c], (pf[:, c] >= best_thr[c]).astype(int), average="binary", zero_division=0):
                best_thr[c] = res.x
            print(f"  {config['id2label'][c]:10s}: thr={best_thr[c]:.4f}, F1={-res.fun:.4f}")

        # Per-class floor: I-tags use higher floor than B-tags to prevent span bleed
        print("\nApplying per-class minimum threshold floors:")
        for c in range(num_labels):
            label = config["id2label"][c]
            floor = MIN_THR_PER_CLASS.get(label, 0.35)
            if best_thr[c] < floor:
                print(f"  {label:10s}: {best_thr[c]:.4f} → {floor:.4f}")
                best_thr[c] = floor

        # Calc macro F1
        preds = (pf >= best_thr.reshape(1, -1)).astype(int)
        macro = f1_score(lf, preds, average="macro", zero_division=0)
        return best_thr, macro

    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)
    # Use validation dataloader from data_module
    val_loader = data_module.val_dataloader()
    best_thr, best_f1_val = optimize_thresholds(best_model, val_loader)
    print(f"\nOptimized thresholds: {best_thr}")
    print(f"Val macro-F1: {best_f1_val:.4f}")
    
    # Use optimized thresholds for testing
    thresholds = best_thr

    # Evaluate
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    # ... (rest of evaluation using 'thresholds') ...
    best_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)

    test_df = pd.DataFrame(test_data)
    test_tok, test_lbl = preprocess_data(test_df)
    test_ds = NERDataset(test_tok, test_lbl)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch[0].to(device)
            mask = batch[1].to(device)
            labels_np = batch[2].cpu().numpy()
            logits = best_model(input_ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            # Use optimized thresholds
            preds = (probs >= thresholds.reshape(1, 1, num_labels)).astype(int)
            for i in range(len(labels_np)):
                valid = np.where((labels_np[i] >= 0).all(axis=1))[0]
                if len(valid) > 0:
                    seq_preds = bio_postprocess(preds[i][valid], config["id2label"])
                    all_preds.extend(seq_preds)
                    all_trues.extend(labels_np[i][valid])

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    def collapse_bio(arr):
        """Collapse 7-col BIO array -> 4-col category array [O, STEREO, GEN, UNFAIR]."""
        return np.column_stack([
            arr[:, 0],
            np.maximum(arr[:, 1], arr[:, 2]),
            np.maximum(arr[:, 3], arr[:, 4]),
            np.maximum(arr[:, 5], arr[:, 6]),
        ])

    cat_preds = collapse_bio(all_preds)
    cat_trues = collapse_bio(all_trues)
    cat_names = ["O", "STEREO", "GEN", "UNFAIR"]

    print("\n--- Test Set Results (Category-Level) ---")
    report_cat = classification_report(
        cat_trues, cat_preds, target_names=cat_names, zero_division=0, output_dict=True,
    )
    print(classification_report(cat_trues, cat_preds, target_names=cat_names, zero_division=0))
    exact_match = float(accuracy_score(cat_trues, cat_preds))
    print(f"Exact Match: {exact_match:.4f}")

    print("\n--- Test Set Results (BIO Detail) ---")
    report_bio = classification_report(
        all_trues, all_preds, target_names=list(label2id.keys()), zero_division=0, output_dict=True,
    )
    print(classification_report(all_trues, all_preds, target_names=list(label2id.keys()), zero_division=0))

    # Training log
    print("\n" + "=" * 60)
    print("TRAINING LOG")
    print("=" * 60)
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "backbone": config["backbone"],
        "dataset": dataset_name,
        "epochs_trained": trainer.current_epoch + 1,
        "val_macro_f1": round(float(best_f1_val), 4),
        "thresholds": {
            config["id2label"][i]: round(float(thresholds[i]), 4) for i in range(num_labels)
        },
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "gamma_neg": config["gamma"],
            "gamma_pos": 1.0,
            "asl_clip": 0.01,
            "llrd_decay": 0.9,
            "span_decay": SPAN_DECAY,
            "span_min_weight": SPAN_MIN_WEIGHT,
            "span_label_decay": SPAN_LABEL_DECAY,
            "span_label_floor": SPAN_LABEL_FLOOR,
        },
        "test": {
            "exact_match": round(exact_match, 4),
            "category_level": _report_to_dict(
                report_cat, cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                report_bio, list(label2id.keys()) + ["macro avg", "weighted avg"],
            ),
        },
    }
    save_training_log(log_entry)

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    save_dir = config["save_dir"]
    os.makedirs(str(save_dir), exist_ok=True)
    best_model.bert.config.id2label = config["id2label"]
    best_model.bert.config.label2id = config["label2id"]
    best_model.bert.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    
    # Save optimized thresholds
    np.save(f"{save_dir}/optimized_thresholds.npy", thresholds)
    print(f"Saved optimized thresholds to {save_dir}/optimized_thresholds.npy")
    
    print(f"Model saved to {save_dir}")
    print("Done.")


# ============================================================================
# GPT-2 TRAINING
# ============================================================================

def train_gpt2(config=None, dataset_source="hf"):
    """Train GUS-Net with GPT-2 backbone using HuggingFace Trainer (7 labels)."""
    if config is None:
        config = GPT2_CONFIG

    label2id = config["label2id"]
    num_labels = config["num_labels"]
    label_names = list(label2id.keys())

    dataset_name = "ethical-spectacle/gus-dataset-v1" if dataset_source == "hf" else "gemini_annotations.json"

    print("=" * 60)
    print("GUS-Net GPT-2 Training")
    print("=" * 60)
    print(f"Backbone:       {config['backbone']}")
    print(f"Dataset:        {dataset_name}")
    print(f"Labels:         {num_labels} ({', '.join(label_names)})")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Focal loss:     gamma={config['gamma']}, smoothing={config['label_smoothing']}")
    print(f"LLRD:           decay={config['llrd_decay_factor']}, classifier_lr={config['classifier_lr']}")
    batch_size = config.get("batch_size", BATCH_SIZE)
    grad_accum = config.get("gradient_accumulation_steps", 2)
    use_grad_ckpt = config.get("gradient_checkpointing", False)

    print(f"Batch size:     {batch_size} (accum={grad_accum}, effective={batch_size * grad_accum})")
    print(f"Grad checkpoint:{use_grad_ckpt}")
    print(f"Max epochs:     {MAX_EPOCHS}")
    print(f"Device:         {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(config["backbone"], add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"\nVocab size:    {tokenizer.vocab_size}")
    print(f"Pad token:     {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # Load dataset
    print("\n" + "=" * 60)
    print(f"LOADING DATA ({dataset_name})")
    print("=" * 60)

    if dataset_source == "gemini":
        gemini_data = load_from_gemini()
        dataset = HFDataset.from_dict({
            "text_str": [d["text_str"] for d in gemini_data],
            "ner_tags": [str(d["ner_tags"]) for d in gemini_data],
        })
    else:
        dataset = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")
    print(f"Loaded {len(dataset)} examples")

    def parse_annotations(example):
        return ast.literal_eval(example["ner_tags"])

    def prepare_example(example):
        text = example["text_str"]
        word_tags = parse_annotations(example)
        words = text.split()

        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

        word_ids = tokenized.word_ids()
        seq_len = len(word_ids)
        labels_multi = np.zeros((seq_len, num_labels), dtype=np.float32)

        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                prev_word_id = None
                continue
            if word_id >= len(word_tags):
                # Default to O for out-of-range words
                labels_multi[idx, label2id["O"]] = 1.0
                prev_word_id = word_id
                continue
            tags = word_tags[word_id]
            has_bias_tag = False
            for tag in tags:
                if tag == "O":
                    continue
                has_bias_tag = True
                if word_id == prev_word_id:
                    # Continuation sub-token: B- -> I-
                    if tag.startswith("B-"):
                        i_tag = "I-" + tag[2:]
                        if i_tag in label2id:
                            labels_multi[idx, label2id[i_tag]] = 1.0
                    elif tag in label2id:
                        labels_multi[idx, label2id[tag]] = 1.0
                else:
                    if tag in label2id:
                        labels_multi[idx, label2id[tag]] = 1.0
            if not has_bias_tag:
                labels_multi[idx, label2id["O"]] = 1.0
            prev_word_id = word_id

        final_labels = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                final_labels.append([-100.0] * num_labels)
            else:
                final_labels.append(labels_multi[idx].tolist())

        tokenized["labels"] = final_labels
        return tokenized

    print("Tokenizing dataset with GPT-2 BPE...")
    tokenized_dataset = DatasetDict({
        "train": dataset.map(
            prepare_example,
            batched=False,
            remove_columns=dataset.column_names,
        )
    })
    print("Tokenization complete!")

    # Split
    train_devtest = tokenized_dataset["train"].train_test_split(test_size=0.30, seed=SEED)
    train_split = train_devtest["train"]
    dev_test = train_devtest["test"].train_test_split(test_size=0.5, seed=SEED)
    dev_split = dev_test["train"]
    test_split = dev_test["test"]

    print(f"\nTrain: {len(train_split)}")
    print(f"Dev:   {len(dev_split)}")
    print(f"Test:  {len(test_split)}")

    # Model
    model_config = AutoConfig.from_pretrained(config["backbone"])
    model_config.num_labels = num_labels
    model_config.problem_type = "multi_label_classification"
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.classifier_dropout = 0.3
    model_config.resid_pdrop = 0.15
    model_config.embd_pdrop = 0.15
    model_config.attn_pdrop = 0.15

    model = GPT2ForTokenClassification.from_pretrained(config["backbone"], config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    print(f"\nModel:       {model_config.model_type}")
    print(f"Parameters:  {model.num_parameters():,}")

    # Label statistics for focal loss
    def estimate_label_frequencies(dataset_split):
        positives = np.zeros(num_labels, dtype=np.int64)
        total = 0
        for ex in dataset_split:
            labels = np.array(ex["labels"])
            valid = labels[labels[:, 0] != -100.0]
            if valid.size == 0:
                continue
            positives += valid.sum(axis=0).astype(np.int64)
            total += valid.shape[0]
        return positives, total

    label_pos, total_tokens = estimate_label_frequencies(train_split)
    label_pos = np.maximum(label_pos, 1)
    freq = label_pos / float(total_tokens)
    inv_freq = 1.0 / freq
    # Clip weight ratio to max 10× to prevent extreme class imbalance
    inv_freq = np.clip(inv_freq, inv_freq.min(), inv_freq.min() * 10.0)
    alpha_labels = inv_freq / inv_freq.sum()
    alpha_labels = torch.tensor(alpha_labels, dtype=torch.float32)

    print("\nLabel statistics:")
    for i, name in enumerate(label_names):
        print(f"  {name:10s}: {label_pos[i]:>6} positives, alpha={alpha_labels[i]:.4f}")

    # Custom Trainer with LLRD + Asymmetric Focal Loss
    class FocalLossTrainerGPT2(Trainer):
        def __init__(self, *args, alpha_channel, gamma=1.5, label_smoothing=0.0,
                     llrd_decay_factor=0.85, classifier_lr=2e-4, **kwargs):
            super().__init__(*args, **kwargs)
            # gamma is gamma_neg; gamma_pos is fixed at 1.0 to preserve recall
            self.focal_loss = AsymmetricFocalLoss(
                alpha=alpha_channel,
                gamma_pos=1.0,
                gamma_neg=gamma,
                clip=0.01,
                label_smoothing=label_smoothing,
            )
            self.llrd_decay_factor = llrd_decay_factor
            self.classifier_lr = classifier_lr

        def create_optimizer(self):
            base_lr = self.args.learning_rate
            decay = self.llrd_decay_factor
            no_decay_keys = ["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]
            n_layers = self.model.config.n_layer

            opt_params = []
            opt_params.append({
                "params": [p for n, p in self.model.named_parameters()
                           if "classifier" in n or "score" in n],
                "lr": self.classifier_lr,
                "weight_decay": 0.0,
            })

            for layer_idx in range(n_layers - 1, -1, -1):
                layer_lr = base_lr * (decay ** (n_layers - 1 - layer_idx))
                layer_prefix = f"transformer.h.{layer_idx}."
                d, nd = [], []
                for n, p in self.model.named_parameters():
                    if layer_prefix in n:
                        (nd if any(k in n for k in no_decay_keys) else d).append(p)
                if d:
                    opt_params.append({"params": d, "lr": layer_lr, "weight_decay": self.args.weight_decay})
                if nd:
                    opt_params.append({"params": nd, "lr": layer_lr, "weight_decay": 0.0})

            emb_lr = base_lr * (decay ** n_layers)
            emb_names = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]
            d, nd = [], []
            for n, p in self.model.named_parameters():
                if any(n.startswith(prefix) for prefix in emb_names):
                    (nd if any(k in n for k in no_decay_keys) else d).append(p)
            if d:
                opt_params.append({"params": d, "lr": emb_lr, "weight_decay": self.args.weight_decay})
            if nd:
                opt_params.append({"params": nd, "lr": emb_lr, "weight_decay": 0.0})

            self.optimizer = torch.optim.AdamW(opt_params, lr=base_lr, eps=1e-8)
            print(f"LLRD optimizer (GPT-2): classifier_lr={self.classifier_lr}, base_lr={base_lr}")
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            if optimizer is None:
                optimizer = self.optimizer
            warmup = int(num_training_steps * self.args.warmup_ratio)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup, num_training_steps=num_training_steps,
            )
            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute span-position weights + smoothed labels BEFORE flattening
            sw_np, sl_np = compute_itag_span_weights(
                labels.cpu().numpy(), config["id2label"],
                decay=SPAN_DECAY, min_weight=SPAN_MIN_WEIGHT,
                label_decay=SPAN_LABEL_DECAY, label_floor=SPAN_LABEL_FLOOR,
            )
            span_w = torch.tensor(sw_np, dtype=torch.float32, device=logits.device)
            smooth = torch.tensor(sl_np, dtype=torch.float32, device=logits.device)

            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1, num_labels)  # for valid mask
            smooth_flat = smooth.view(-1, num_labels)
            sw_flat = span_w.view(-1, num_labels)
            valid = labels_flat[:, 0] != -100.0
            loss = self.focal_loss(
                logits_flat[valid], smooth_flat[valid],
                sample_weight=sw_flat[valid],
            )
            return (loss, outputs) if return_outputs else loss

    # Metrics
    thresholds = np.array([0.5] * num_labels, dtype=np.float32)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        probs = 1 / (1 + np.exp(-predictions))
        valid = labels[:, :, 0] != -100.0
        probs_flat = probs[valid]
        labels_flat = labels[valid]
        thr = thresholds.reshape(1, num_labels)
        preds_bin = (probs_flat >= thr).astype(int)
        labels_bin = labels_flat.astype(int)

        if labels_bin.sum() == 0:
            return {"f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "hamming_loss": 0.0}

        label_f1s = [f1_score(labels_bin[:, c], preds_bin[:, c], average="binary", zero_division=0)
                     for c in range(num_labels)]
        return {
            "f1_macro": np.mean(label_f1s),
            "precision_macro": precision_score(labels_bin, preds_bin, average="macro", zero_division=0),
            "recall_macro": recall_score(labels_bin, preds_bin, average="macro", zero_division=0),
            "hamming_loss": np.mean(preds_bin != labels_bin),
        }

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir=config["training_output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = FocalLossTrainerGPT2(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=dev_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        alpha_channel=alpha_labels,
        gamma=config["gamma"],
        label_smoothing=config["label_smoothing"],
        llrd_decay_factor=config["llrd_decay_factor"],
        classifier_lr=config["classifier_lr"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("Starting training...")
    train_result = trainer.train()
    print(f"\nTraining loss: {train_result.training_loss:.4f}")

    # SWA
    def apply_swa(trainer, checkpoint_dir, last_n=5):
        checkpoints = sorted(
            glob.glob(f"{checkpoint_dir}/checkpoint-*"),
            key=lambda x: int(x.split("-")[-1]),
        )
        if len(checkpoints) < 2:
            print(f"Only {len(checkpoints)} checkpoint(s), skipping SWA.")
            return

        last = checkpoints[-last_n:]
        print(f"SWA: averaging {len(last)} checkpoints")

        trainer.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        avg = None
        n = 0
        for cp in last:
            sf = os.path.join(cp, "model.safetensors")
            bf = os.path.join(cp, "pytorch_model.bin")
            if os.path.exists(sf):
                from safetensors.torch import load_file
                state = load_file(sf, device="cpu")
            elif os.path.exists(bf):
                state = torch.load(bf, map_location="cpu", weights_only=True)
            else:
                continue
            n += 1
            if avg is None:
                avg = {k: v.float() for k, v in state.items()}
            else:
                for k in avg:
                    avg[k] += (state[k].float() - avg[k]) / n
            del state
            gc.collect()

        if avg and n > 0:
            trainer.model.load_state_dict(avg)
            del avg
            gc.collect()
            trainer.model.to(trainer.args.device)
            print(f"SWA applied ({n} checkpoints).")
        else:
            trainer.model.to(trainer.args.device)

    apply_swa(trainer, config["training_output_dir"], last_n=5)

    # Threshold optimization
    def optimize_thresholds(trainer, dev_dataset):
        model = trainer.model
        model.eval()
        grid = np.arange(0.05, 0.96, 0.025).tolist()

        all_probs, all_labels = [], []
        for batch in trainer.get_eval_dataloader(dev_dataset):
            with torch.no_grad():
                labels = batch["labels"].cpu().numpy()
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                logits = model(**inputs).logits.cpu().numpy()
            all_probs.append(1 / (1 + np.exp(-logits)))
            all_labels.append(labels)

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        valid = labels[:, :, 0] != -100.0
        pf = probs[valid]
        lf = labels[valid].astype(int)

        best_thr = np.zeros(num_labels, dtype=np.float32)

        print("Pass 1 - grid search:")
        for c in range(num_labels):
            best_f1, best_t = 0, 0.5
            for t in grid:
                f1 = f1_score(lf[:, c], (pf[:, c] >= t).astype(int), average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            best_thr[c] = best_t
            print(f"  {label_names[c]:10s}: thr={best_t:.3f}, F1={best_f1:.4f}")

        print("\nPass 2 - refinement:")
        for c in range(num_labels):
            lo = max(0.01, best_thr[c] - 0.05)
            hi = min(0.99, best_thr[c] + 0.05)
            res = minimize_scalar(
                lambda t: -f1_score(lf[:, c], (pf[:, c] >= t).astype(int), average="binary", zero_division=0),
                bounds=(lo, hi), method="bounded",
            )
            if -res.fun >= f1_score(lf[:, c], (pf[:, c] >= best_thr[c]).astype(int), average="binary", zero_division=0):
                best_thr[c] = res.x
            print(f"  {label_names[c]:10s}: thr={best_thr[c]:.4f}, F1={-res.fun:.4f}")

        # Per-class floor: I-tags use higher floor than B-tags to prevent span bleed
        print("\nApplying per-class minimum threshold floors:")
        for c in range(num_labels):
            label = label_names[c]
            floor = MIN_THR_PER_CLASS.get(label, 0.35)
            if best_thr[c] < floor:
                print(f"  {label:10s}: {best_thr[c]:.4f} → {floor:.4f}")
                best_thr[c] = floor

        preds = (pf >= best_thr.reshape(1, -1)).astype(int)
        macro = np.mean([f1_score(lf[:, c], preds[:, c], average="binary", zero_division=0)
                         for c in range(num_labels)])
        return best_thr, macro

    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)
    best_thr, best_f1_dev = optimize_thresholds(trainer, dev_split)
    print(f"\nOptimized thresholds: {best_thr}")
    print(f"Dev macro-F1: {best_f1_dev:.4f}")
    thresholds = best_thr

    # Evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_split)
    print(f"\nTest results:")
    print(f"  Macro F1:      {test_metrics['eval_f1_macro']:.4f}")
    print(f"  Precision:     {test_metrics['eval_precision_macro']:.4f}")
    print(f"  Recall:        {test_metrics['eval_recall_macro']:.4f}")
    print(f"  Hamming Loss:  {test_metrics['eval_hamming_loss']:.4f}")

    # Detailed BIO + category report (same as BERT)
    print("\n--- Detailed BIO Report ---")
    model_eval = trainer.model
    model_eval.eval()
    device = model_eval.device

    all_preds, all_trues = [], []
    for batch in trainer.get_eval_dataloader(test_split):
        with torch.no_grad():
            labels_np = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model_eval(**inputs).logits.cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= thresholds.reshape(1, 1, num_labels)).astype(int)
        for i in range(len(labels_np)):
            valid = np.where((labels_np[i] >= 0).all(axis=1))[0]
            if len(valid) > 0:
                seq_preds = bio_postprocess(preds[i][valid], config["id2label"])
                all_preds.extend(seq_preds)
                all_trues.extend(labels_np[i][valid])

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    print(classification_report(all_trues, all_preds, target_names=label_names, zero_division=0))

    def collapse_bio(arr):
        """Collapse 7-col BIO array -> 4-col category array [O, STEREO, GEN, UNFAIR]."""
        return np.column_stack([
            arr[:, 0],
            np.maximum(arr[:, 1], arr[:, 2]),
            np.maximum(arr[:, 3], arr[:, 4]),
            np.maximum(arr[:, 5], arr[:, 6]),
        ])

    cat_preds = collapse_bio(all_preds)
    cat_trues = collapse_bio(all_trues)
    cat_names = ["O", "STEREO", "GEN", "UNFAIR"]
    print("--- Category-Level Report ---")
    report_cat = classification_report(
        cat_trues, cat_preds, target_names=cat_names, zero_division=0, output_dict=True,
    )
    report_bio = classification_report(
        all_trues, all_preds, target_names=label_names, zero_division=0, output_dict=True,
    )
    print(classification_report(cat_trues, cat_preds, target_names=cat_names, zero_division=0))
    exact_match = float(accuracy_score(cat_trues, cat_preds))
    print(f"Exact Match: {exact_match:.4f}")

    # Training log
    print("\n" + "=" * 60)
    print("TRAINING LOG")
    print("=" * 60)
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "backbone": config["backbone"],
        "dataset": dataset_name,
        "epochs_trained": round(float(train_result.metrics.get("epoch", 0)), 1),
        "train_loss": round(float(train_result.training_loss), 4),
        "dev_macro_f1": round(float(best_f1_dev), 4),
        "thresholds": {
            config["id2label"][i]: round(float(thresholds[i]), 4) for i in range(num_labels)
        },
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "gamma_neg": config["gamma"],
            "gamma_pos": 1.0,
            "asl_clip": 0.01,
            "label_smoothing": config["label_smoothing"],
            "llrd_decay_factor": config["llrd_decay_factor"],
            "classifier_lr": config["classifier_lr"],
            "gradient_checkpointing": use_grad_ckpt,
            "span_decay": SPAN_DECAY,
            "span_min_weight": SPAN_MIN_WEIGHT,
            "span_label_decay": SPAN_LABEL_DECAY,
            "span_label_floor": SPAN_LABEL_FLOOR,
        },
        "test": {
            "exact_match": round(exact_match, 4),
            "macro_f1":    round(float(test_metrics["eval_f1_macro"]), 4),
            "precision":   round(float(test_metrics["eval_precision_macro"]), 4),
            "recall":      round(float(test_metrics["eval_recall_macro"]), 4),
            "hamming_loss": round(float(test_metrics["eval_hamming_loss"]), 4),
            "category_level": _report_to_dict(
                report_cat, cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                report_bio, label_names + ["macro avg", "weighted avg"],
            ),
        },
    }
    save_training_log(log_entry)

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    save_dir = config["save_dir"]
    os.makedirs(str(save_dir), exist_ok=True)
    model_eval.config.id2label = config["id2label"]
    model_eval.config.label2id = config["label2id"]
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    np.save(f"{save_dir}/optimized_thresholds.npy", thresholds)
    print(f"Model saved to {save_dir}")
    print("Done.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Dataset selection
    print("=" * 60)
    print("GUS-Net Training - Dataset Selection")
    print("=" * 60)
    print("\nSelect dataset:")
    print("  1. Hugging Face (ethical-spectacle/gus-dataset-v1)")
    print("  2. Gemini Annotations (gemini_annotations.json)")
    print()

    while True:
        ds_choice = input("Enter your choice (1-2): ").strip()
        if ds_choice == "1":
            dataset_source = "hf"
            break
        elif ds_choice == "2":
            dataset_source = "gemini"
            break
        else:
            print("Invalid choice. Please enter 1-2.")

    # Model selection
    print()
    print("=" * 60)
    print("GUS-Net Training - Model Selection")
    print("=" * 60)
    print("\nSelect backbone to train:")
    print("  1. BERT Base (bert-base-uncased)")
    print("  2. BERT Large (bert-large-uncased)")
    print("  3. GPT-2 (gpt2)")
    print("  4. GPT-2 Medium (gpt2-medium)")
    print("  5. GPT-2 Large (gpt2-large)")
    print()

    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            print("\nStarting BERT Base training...\n")
            train_bert(BERT_CONFIG, dataset_source=dataset_source)
            break
        elif choice == "2":
            print("\nStarting BERT Large training...\n")
            train_bert(BERT_LARGE_CONFIG, dataset_source=dataset_source)
            break
        elif choice == "3":
            print("\nStarting GPT-2 training...\n")
            train_gpt2(GPT2_CONFIG, dataset_source=dataset_source)
            break
        elif choice == "4":
            print("\nStarting GPT-2 Medium training...\n")
            train_gpt2(GPT2_MEDIUM_CONFIG, dataset_source=dataset_source)
            break
        elif choice == "5":
            print("\nStarting GPT-2 Large training...\n")
            train_gpt2(GPT2_LARGE_CONFIG, dataset_source=dataset_source)
            break
        else:
            print("Invalid choice. Please enter 1-5.")
