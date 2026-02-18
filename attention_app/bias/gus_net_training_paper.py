"""
GUS-Net Training — Paper-Faithful Version
==========================================

Reproduces the ORIGINAL GUS-Net training pipeline from:
    "Responsible AI in NLP: GUS-Net Span-Level Bias Detection"
    (arXiv:2410.08388)

Key differences vs gus_net_training.py (our modified version):
    - Standard symmetric focal loss (NOT asymmetric)
    - Scalar alpha=0.65 per-SAMPLE (NOT per-class from frequencies)
    - gamma=3.5 (from paper's code, not 2.0 from paper's text)
    - No probability clipping (no ASL clip mechanism)
    - No LLRD — single learning rate for all parameters
    - No span decay or label smoothing
    - No early stopping — trains full MAX_EPOCHS
    - Linear scheduler with warmup (not cosine)

Purpose: ablation experiment to compare paper's simpler approach
against our modified loss / optimizer stack.
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
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
)
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    GPT2TokenizerFast,
    GPT2ForTokenClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from scipy.optimize import minimize_scalar

# Import shared utilities from main training script
from gus_net_training import (
    load_from_hf,
    load_from_gemini,
    bio_postprocess,
    save_training_log,
    _report_to_dict,
)


# ============================================================================
# PATHS & SEED
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_CLEAN_DATASET = _SCRIPT_DIR.parent.parent / "dataset" / "gus_dataset_clean.json"
SEED = 42


def load_from_clean_json():
    """Load the cleaned dataset (punctuation labels fixed, BIO repaired)."""
    print(f"Loading cleaned dataset from {_CLEAN_DATASET}...")
    with open(_CLEAN_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


# ============================================================================
# HYPERPARAMETERS — matching the paper's code exactly
# ============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_EPOCHS = 40          # extended from paper's 20; early stopping prevents overfit
PATIENCE = 5             # stop if no improvement for 5 epochs
THRESHOLD = 0.5          # paper uses fixed 0.5
MAX_LENGTH = 128

# Focal loss — paper's values (from their published code)
ALPHA = 0.65             # scalar, applied per-sample (not per-class)
GAMMA = 3.5              # from code; paper text says 2 but code uses 3.5

label2id = {
    "O": 0,
    "B-STEREO": 1, "I-STEREO": 2,
    "B-GEN": 3, "I-GEN": 4,
    "B-UNFAIR": 5, "I-UNFAIR": 6,
}
num_labels = len(label2id)
id2label = {v: k for k, v in label2id.items()}


# ============================================================================
# STANDARD FOCAL LOSS (paper's implementation)
# ============================================================================

def focal_loss_paper(logits, labels, alpha=ALPHA, gamma=GAMMA):
    """
    Exact focal loss from the GUS-Net paper's code.

    Key properties:
    - Uses F.binary_cross_entropy_with_logits (numerically stable)
    - alpha is PER-SAMPLE: alpha for positives, (1-alpha) for negatives
    - Single gamma for both positives and negatives (symmetric)
    - p_t clamped to [0.01, 0.99] for numerical stability
    - Masked to ignore padding tokens (label == -100)
    - Reduction: sum / num_valid_elements
    """
    mask = (labels != -100).float()
    labels = labels.float()

    bce_loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none",
    )
    bce_loss = bce_loss * mask

    probs = torch.sigmoid(logits)
    p_t = labels * probs + (1 - labels) * (1 - probs)
    p_t = torch.clamp(p_t, 0.01, 0.99)

    # Per-sample alpha: alpha for positive class, (1-alpha) for negative
    alpha_t = labels * alpha + (1 - labels) * (1 - alpha)

    focal_modulation = (1 - p_t) ** gamma
    focal = alpha_t * focal_modulation * bce_loss
    focal = focal * mask

    return focal.sum() / mask.sum().clamp(min=1.0)


# ============================================================================
# BERT TRAINING — paper-faithful
# ============================================================================

def train_bert_paper(dataset_source="hf"):
    """Train GUS-Net BERT with the paper's original settings."""

    save_dir = (_MODELS_DIR / "gus-net-bert-paper-clean-2"
                if dataset_source == "clean"
                else _MODELS_DIR / "gus-net-bert-paper")

    print("=" * 60)
    print("GUS-Net BERT Training (PAPER-FAITHFUL)")
    print("=" * 60)
    print(f"Backbone:       bert-base-uncased")
    print(f"Loss:           Standard Focal Loss (alpha={ALPHA}, gamma={GAMMA})")
    print(f"Optimizer:      AdamW (single LR, NO LLRD)")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Max epochs:     {MAX_EPOCHS} (early stopping, patience={PATIENCE})")
    print(f"Threshold:      {THRESHOLD} (fixed)")
    print(f"Device:         {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_and_align_labels(text, annotations, max_length=MAX_LENGTH):
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
        prev_word_id = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append([-100] * num_labels)
            elif word_idx == prev_word_id:
                # Mask secondary subtokens — only first subtoken is labelled
                aligned_labels.append([-100] * num_labels)
            else:
                tags = annotations[word_idx]
                label_vector = [0] * num_labels
                for tag in tags:
                    if tag in label2id:
                        label_vector[label2id[tag]] = 1
                aligned_labels.append(label_vector)
            prev_word_id = word_idx
        return tokenized, aligned_labels

    def preprocess_data(df):
        tokenized_list, labels_list = [], []
        for _, row in df.iterrows():
            inputs, labels = tokenize_and_align_labels(
                row["text_str"], row["ner_tags"],
            )
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

    # DataModule — 70/15/15 split, stratified by bias presence
    class NERDataModule(pl.LightningDataModule):
        def __init__(self, tokenized_texts, labels, batch_size=BATCH_SIZE,
                     val_split=0.15, test_split=0.15):
            super().__init__()
            self.tokenized_texts = tokenized_texts
            self.all_labels = labels
            self.batch_size = batch_size
            self.val_split = val_split
            self.test_split = test_split

        @staticmethod
        def _sample_has_bias(label_seq):
            """1 if any valid token has a non-O class, else 0."""
            for vec in label_seq:
                if vec[0] != -100 and any(v == 1 for v in vec[1:]):
                    return 1
            return 0

        def setup(self, stage=None):
            strat = [self._sample_has_bias(lbl) for lbl in self.all_labels]
            indices = list(range(len(self.all_labels)))

            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_split,
                stratify=strat,
                random_state=SEED,
            )
            strat_tv = [strat[i] for i in train_val_idx]
            rel_val = self.val_split / (1 - self.test_split)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=rel_val,
                stratify=strat_tv,
                random_state=SEED,
            )

            full_ds = NERDataset(self.tokenized_texts, self.all_labels)
            self.train_ds = torch.utils.data.Subset(full_ds, train_idx)
            self.val_ds = torch.utils.data.Subset(full_ds, val_idx)
            self.test_ds = torch.utils.data.Subset(full_ds, test_idx)

        def train_dataloader(self):
            return DataLoader(
                self.train_ds, batch_size=self.batch_size,
                shuffle=True, num_workers=0,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_ds, batch_size=self.batch_size, num_workers=0,
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_ds, batch_size=self.batch_size, num_workers=0,
            )

    # Lightning Model — paper's architecture (no LLRD, simple AdamW)
    class GUSNetBERTPaper(pl.LightningModule):
        def __init__(self, learning_rate=LEARNING_RATE, threshold=THRESHOLD):
            super().__init__()
            self.save_hyperparameters()
            self.bert = BertForTokenClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels,
            )
            self.learning_rate = learning_rate
            self.threshold = threshold

        def forward(self, input_ids, attention_mask):
            return self.bert(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits

        def _compute_loss(self, logits, labels):
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1, num_labels)
            return focal_loss_paper(logits, labels)

        def training_step(self, batch, batch_idx):
            logits = self(batch[0], batch[1])
            loss = self._compute_loss(logits, batch[2])
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            logits = self(batch[0], batch[1])
            loss = self._compute_loss(logits, batch[2])
            self.log("val_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            # Paper: simple AdamW, single LR, no LLRD
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    0.1 * self.trainer.estimated_stepping_batches,
                ),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            print(f"Optimizer: AdamW (single LR={self.learning_rate})")
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # Load data
    dataset_names = {
        "hf": "ethical-spectacle/gus-dataset-v1",
        "gemini": "gemini_annotations.json",
        "clean": "gus_dataset_clean.json (punct-fixed)",
    }
    dataset_name = dataset_names.get(dataset_source, dataset_source)
    print(f"\nLoading data ({dataset_name})...")
    if dataset_source == "clean":
        all_data = load_from_clean_json()
    elif dataset_source == "gemini":
        all_data = load_from_gemini()
    else:
        all_data = load_from_hf()
    print(f"Loaded {len(all_data)} samples")

    # Tokenize ALL data (paper doesn't do a separate test split before tokenizing)
    df = pd.DataFrame(all_data)
    print("Tokenizing...")
    tokenized_texts, labels = preprocess_data(df)
    print(f"Processed {len(tokenized_texts)} samples")

    # DataModule (70/15/15 split inside)
    data_module = NERDataModule(tokenized_texts, labels)
    data_module.setup()

    # Train — early stopping with patience
    print("\n" + "=" * 60)
    print("TRAINING (no early stopping)")
    print("=" * 60)

    shutil.rmtree("lightning_logs", ignore_errors=True)
    shutil.rmtree("checkpoints", ignore_errors=True)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="gusnet-bert-paper-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )

    model = GUSNetBERTPaper()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="16-mixed",
    )
    trainer.fit(model, data_module)

    # Load best checkpoint
    best_path = checkpoint_cb.best_model_path
    if best_path:
        print(f"\nLoading best model from {best_path}")
        best_model = GUSNetBERTPaper.load_from_checkpoint(best_path)
    else:
        best_model = model

    best_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)

    # ----------------------------------------------------------------
    # Threshold optimisation (NOT in the paper, but useful for comparison)
    # We report results with BOTH fixed 0.5 and optimised thresholds.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION (for comparison — paper uses 0.5)")
    print("=" * 60)

    val_loader = data_module.val_dataloader()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            mask = batch[1].to(device)
            labels_np = batch[2].cpu().numpy()
            logits = best_model(input_ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels_np)

    probs_cat = np.concatenate(all_probs)
    labels_cat = np.concatenate(all_labels)
    valid = labels_cat[:, :, 0] != -100.0
    pf = probs_cat[valid]
    lf = labels_cat[valid].astype(int)

    # Grid search
    opt_thr = np.zeros(num_labels, dtype=np.float32)
    grid = np.arange(0.05, 0.96, 0.025)
    print("\nGrid search:")
    for c in range(num_labels):
        best_f1, best_t = 0, 0.5
        for t in grid:
            f1 = f1_score(
                lf[:, c], (pf[:, c] >= t).astype(int),
                average="binary", zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        opt_thr[c] = best_t
        print(f"  {id2label[c]:10s}: thr={best_t:.3f}, F1={best_f1:.4f}")

    # Refinement
    print("\nRefinement:")
    for c in range(num_labels):
        lo = max(0.01, opt_thr[c] - 0.05)
        hi = min(0.99, opt_thr[c] + 0.05)
        res = minimize_scalar(
            lambda t, c=c: -f1_score(
                lf[:, c], (pf[:, c] >= t).astype(int),
                average="binary", zero_division=0,
            ),
            bounds=(lo, hi), method="bounded",
        )
        if -res.fun >= f1_score(
            lf[:, c], (pf[:, c] >= opt_thr[c]).astype(int),
            average="binary", zero_division=0,
        ):
            opt_thr[c] = res.x
        print(f"  {id2label[c]:10s}: thr={opt_thr[c]:.4f}, F1={-res.fun:.4f}")

    print(f"\nOptimised thresholds: {opt_thr}")

    # ----------------------------------------------------------------
    # TEST SET EVALUATION — run twice: fixed 0.5 and optimised
    # ----------------------------------------------------------------
    test_loader = data_module.test_dataloader()

    def evaluate_with_thresholds(model, loader, thresholds, label=""):
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch[0].to(device)
                mask_t = batch[1].to(device)
                labels_np = batch[2].cpu().numpy()
                logits = model(input_ids, mask_t)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= thresholds.reshape(1, 1, num_labels)).astype(int)
                for i in range(len(labels_np)):
                    v = np.where((labels_np[i] >= 0).all(axis=1))[0]
                    if len(v) > 0:
                        seq_preds = bio_postprocess(preds[i][v], id2label)
                        all_preds.extend(seq_preds)
                        all_trues.extend(labels_np[i][v])

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        def collapse_bio(arr):
            return np.column_stack([
                arr[:, 0],
                np.maximum(arr[:, 1], arr[:, 2]),
                np.maximum(arr[:, 3], arr[:, 4]),
                np.maximum(arr[:, 5], arr[:, 6]),
            ])

        cat_preds = collapse_bio(all_preds)
        cat_trues = collapse_bio(all_trues)
        cat_names = ["O", "STEREO", "GEN", "UNFAIR"]

        print(f"\n--- {label} (Category-Level) ---")
        report_cat = classification_report(
            cat_trues, cat_preds, target_names=cat_names,
            zero_division=0, output_dict=True,
        )
        print(classification_report(
            cat_trues, cat_preds, target_names=cat_names, zero_division=0,
        ))
        exact_match = float(accuracy_score(cat_trues, cat_preds))
        print(f"Exact Match: {exact_match:.4f}")

        print(f"\n--- {label} (BIO Detail) ---")
        label_names = list(label2id.keys())
        report_bio = classification_report(
            all_trues, all_preds, target_names=label_names,
            zero_division=0, output_dict=True,
        )
        print(classification_report(
            all_trues, all_preds, target_names=label_names, zero_division=0,
        ))

        return report_cat, report_bio, exact_match

    print("\n" + "=" * 60)
    print("TEST SET — FIXED THRESHOLD 0.5 (paper's approach)")
    print("=" * 60)
    fixed_thr = np.full(num_labels, 0.5, dtype=np.float32)
    rep_cat_fixed, rep_bio_fixed, em_fixed = evaluate_with_thresholds(
        best_model, test_loader, fixed_thr, "Fixed thr=0.5",
    )

    print("\n" + "=" * 60)
    print("TEST SET — OPTIMISED THRESHOLDS (for comparison)")
    print("=" * 60)
    rep_cat_opt, rep_bio_opt, em_opt = evaluate_with_thresholds(
        best_model, test_loader, opt_thr, "Optimised thresholds",
    )

    # Training log
    print("\n" + "=" * 60)
    print("TRAINING LOG")
    print("=" * 60)
    label_names = list(label2id.keys())
    cat_names = ["O", "STEREO", "GEN", "UNFAIR"]

    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "backbone": "bert-base-uncased",
        "variant": "paper-faithful",
        "dataset": dataset_name,
        "epochs_trained": trainer.current_epoch + 1,
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": False,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "loss": "standard_focal_loss",
            "llrd": False,
            "span_decay": False,
            "label_smoothing": False,
            "optimizer": "AdamW (single LR)",
            "scheduler": "linear_warmup (0.1)",
        },
        "test_fixed_thr": {
            "threshold": 0.5,
            "exact_match": round(em_fixed, 4),
            "category_level": _report_to_dict(
                rep_cat_fixed,
                cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                rep_bio_fixed,
                label_names + ["macro avg", "weighted avg"],
            ),
        },
        "test_optimised_thr": {
            "thresholds": {
                id2label[i]: round(float(opt_thr[i]), 4)
                for i in range(num_labels)
            },
            "exact_match": round(em_opt, 4),
            "category_level": _report_to_dict(
                rep_cat_opt,
                cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                rep_bio_opt,
                label_names + ["macro avg", "weighted avg"],
            ),
        },
    }
    save_training_log(log_entry)

    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    os.makedirs(str(save_dir), exist_ok=True)
    best_model.bert.config.id2label = id2label
    best_model.bert.config.label2id = label2id
    best_model.bert.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    np.save(f"{save_dir}/optimized_thresholds.npy", opt_thr)
    print(f"Model saved to {save_dir}")
    print("Done.")


# ============================================================================
# GPT-2 TRAINING — paper-faithful
# ============================================================================

def train_gpt2_paper(dataset_source="hf"):
    """Train GUS-Net GPT-2 with the paper's original focal loss settings."""

    save_dir = (_MODELS_DIR / "gus-net-gpt2-paper-clean-2"
                if dataset_source == "clean"
                else _MODELS_DIR / "gus-net-gpt2-paper")
    training_output_dir = str(save_dir) + "-output"
    label_names = list(label2id.keys())

    dataset_names = {
        "hf": "ethical-spectacle/gus-dataset-v1",
        "gemini": "gemini_annotations.json",
        "clean": "gus_dataset_clean.json (punct-fixed)",
    }
    dataset_name = dataset_names.get(dataset_source, dataset_source)

    print("=" * 60)
    print("GUS-Net GPT-2 Training (PAPER-FAITHFUL)")
    print("=" * 60)
    print(f"Backbone:       gpt2")
    print(f"Loss:           Standard Focal Loss (alpha={ALPHA}, gamma={GAMMA})")
    print(f"Optimizer:      AdamW (single LR, NO LLRD)")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Batch size:     {BATCH_SIZE} (accum=2, effective=32)")
    print(f"Max epochs:     {MAX_EPOCHS} (early stopping, patience={PATIENCE})")
    print(f"Threshold:      {THRESHOLD} (fixed)")
    print(f"Device:         {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    print(f"\nLoading data ({dataset_name})...")
    if dataset_source == "clean":
        clean_data = load_from_clean_json()
        dataset = HFDataset.from_dict({
            "text_str": [d["text_str"] for d in clean_data],
            "ner_tags": [str(d["ner_tags"]) for d in clean_data],
        })
    elif dataset_source == "gemini":
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

    print("Tokenizing...")
    tokenized_dataset = DatasetDict({
        "train": dataset.map(
            prepare_example,
            batched=False,
            remove_columns=dataset.column_names,
        )
    })

    # 70/15/15 split
    train_devtest = tokenized_dataset["train"].train_test_split(
        test_size=0.30, seed=SEED,
    )
    train_split = train_devtest["train"]
    dev_test = train_devtest["test"].train_test_split(test_size=0.5, seed=SEED)
    dev_split = dev_test["train"]
    test_split = dev_test["test"]

    print(f"\nTrain: {len(train_split)}")
    print(f"Dev:   {len(dev_split)}")
    print(f"Test:  {len(test_split)}")

    # Model
    model_config = AutoConfig.from_pretrained("gpt2")
    model_config.num_labels = num_labels
    model_config.problem_type = "multi_label_classification"
    model_config.pad_token_id = tokenizer.pad_token_id

    model = GPT2ForTokenClassification.from_pretrained("gpt2", config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    print(f"\nParameters: {model.num_parameters():,}")

    # Custom Trainer with paper's focal loss — NO LLRD
    class FocalLossTrainerPaper(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1, num_labels)

            loss = focal_loss_paper(logits_flat, labels_flat)
            return (loss, outputs) if return_outputs else loss

    # Metrics
    thresholds = np.array([THRESHOLD] * num_labels, dtype=np.float32)

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
            return {"f1_macro": 0.0}

        label_f1s = [
            f1_score(
                labels_bin[:, c], preds_bin[:, c],
                average="binary", zero_division=0,
            )
            for c in range(num_labels)
        ]
        return {
            "f1_macro": np.mean(label_f1s),
            "precision_macro": precision_score(
                labels_bin, preds_bin, average="macro", zero_division=0,
            ),
            "recall_macro": recall_score(
                labels_bin, preds_bin, average="macro", zero_division=0,
            ),
        }

    # Training args — early stopping with patience
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
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

    trainer = FocalLossTrainerPaper(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=dev_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("\nStarting training...")
    train_result = trainer.train()
    print(f"\nTraining loss: {train_result.training_loss:.4f}")

    # Threshold optimisation (for comparison)
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION (for comparison)")
    print("=" * 60)

    model_eval = trainer.model
    model_eval.eval()

    all_probs, all_labels_list = [], []
    for batch in trainer.get_eval_dataloader(dev_split):
        with torch.no_grad():
            labels_np = batch["labels"].cpu().numpy()
            inputs = {
                k: v.to(model_eval.device)
                for k, v in batch.items() if k != "labels"
            }
            logits = model_eval(**inputs).logits.cpu().numpy()
        all_probs.append(1 / (1 + np.exp(-logits)))
        all_labels_list.append(labels_np)

    probs_cat = np.concatenate(all_probs)
    labels_cat = np.concatenate(all_labels_list)
    valid = labels_cat[:, :, 0] != -100.0
    pf = probs_cat[valid]
    lf = labels_cat[valid].astype(int)

    opt_thr = np.zeros(num_labels, dtype=np.float32)
    grid = np.arange(0.05, 0.96, 0.025)
    for c in range(num_labels):
        best_f1, best_t = 0, 0.5
        for t in grid:
            f1 = f1_score(
                lf[:, c], (pf[:, c] >= t).astype(int),
                average="binary", zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_t = f1, t
        opt_thr[c] = best_t
        print(f"  {label_names[c]:10s}: thr={best_t:.3f}, F1={best_f1:.4f}")

    for c in range(num_labels):
        lo = max(0.01, opt_thr[c] - 0.05)
        hi = min(0.99, opt_thr[c] + 0.05)
        res = minimize_scalar(
            lambda t, c=c: -f1_score(
                lf[:, c], (pf[:, c] >= t).astype(int),
                average="binary", zero_division=0,
            ),
            bounds=(lo, hi), method="bounded",
        )
        if -res.fun >= f1_score(
            lf[:, c], (pf[:, c] >= opt_thr[c]).astype(int),
            average="binary", zero_division=0,
        ):
            opt_thr[c] = res.x

    print(f"\nOptimised thresholds: {opt_thr}")

    # Evaluation — both fixed and optimised
    def evaluate_gpt2(model, test_dataset, thresholds, label=""):
        device = model.device
        all_preds, all_trues = [], []
        for batch in trainer.get_eval_dataloader(test_dataset):
            with torch.no_grad():
                labels_np = batch["labels"].cpu().numpy()
                inputs = {
                    k: v.to(device)
                    for k, v in batch.items() if k != "labels"
                }
                logits = model(**inputs).logits.cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs >= thresholds.reshape(1, 1, num_labels)).astype(int)
            for i in range(len(labels_np)):
                v = np.where((labels_np[i] >= 0).all(axis=1))[0]
                if len(v) > 0:
                    seq_preds = bio_postprocess(preds[i][v], id2label)
                    all_preds.extend(seq_preds)
                    all_trues.extend(labels_np[i][v])

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        def collapse_bio(arr):
            return np.column_stack([
                arr[:, 0],
                np.maximum(arr[:, 1], arr[:, 2]),
                np.maximum(arr[:, 3], arr[:, 4]),
                np.maximum(arr[:, 5], arr[:, 6]),
            ])

        cat_preds = collapse_bio(all_preds)
        cat_trues = collapse_bio(all_trues)
        cat_names = ["O", "STEREO", "GEN", "UNFAIR"]

        print(f"\n--- {label} (Category-Level) ---")
        report_cat = classification_report(
            cat_trues, cat_preds, target_names=cat_names,
            zero_division=0, output_dict=True,
        )
        print(classification_report(
            cat_trues, cat_preds, target_names=cat_names, zero_division=0,
        ))
        exact_match = float(accuracy_score(cat_trues, cat_preds))
        print(f"Exact Match: {exact_match:.4f}")

        print(f"\n--- {label} (BIO Detail) ---")
        report_bio = classification_report(
            all_trues, all_preds, target_names=label_names,
            zero_division=0, output_dict=True,
        )
        print(classification_report(
            all_trues, all_preds, target_names=label_names, zero_division=0,
        ))

        return report_cat, report_bio, exact_match

    print("\n" + "=" * 60)
    print("TEST SET — FIXED THRESHOLD 0.5")
    print("=" * 60)
    fixed_thr = np.full(num_labels, 0.5, dtype=np.float32)
    rep_cat_fixed, rep_bio_fixed, em_fixed = evaluate_gpt2(
        model_eval, test_split, fixed_thr, "Fixed thr=0.5",
    )

    print("\n" + "=" * 60)
    print("TEST SET — OPTIMISED THRESHOLDS")
    print("=" * 60)
    rep_cat_opt, rep_bio_opt, em_opt = evaluate_gpt2(
        model_eval, test_split, opt_thr, "Optimised thresholds",
    )

    # Training log
    cat_names = ["O", "STEREO", "GEN", "UNFAIR"]
    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "backbone": "gpt2",
        "variant": "paper-faithful",
        "dataset": dataset_name,
        "epochs_trained": round(
            float(train_result.metrics.get("epoch", 0)), 1,
        ),
        "train_loss": round(float(train_result.training_loss), 4),
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 32,
            "max_epochs": MAX_EPOCHS,
            "early_stopping": False,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "loss": "standard_focal_loss",
            "llrd": False,
            "span_decay": False,
            "label_smoothing": False,
            "optimizer": "AdamW (single LR)",
            "scheduler": "linear_warmup (0.1)",
        },
        "test_fixed_thr": {
            "threshold": 0.5,
            "exact_match": round(em_fixed, 4),
            "category_level": _report_to_dict(
                rep_cat_fixed,
                cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                rep_bio_fixed,
                label_names + ["macro avg", "weighted avg"],
            ),
        },
        "test_optimised_thr": {
            "thresholds": {
                id2label[i]: round(float(opt_thr[i]), 4)
                for i in range(num_labels)
            },
            "exact_match": round(em_opt, 4),
            "category_level": _report_to_dict(
                rep_cat_opt,
                cat_names + ["macro avg", "weighted avg"],
            ),
            "bio_level": _report_to_dict(
                rep_bio_opt,
                label_names + ["macro avg", "weighted avg"],
            ),
        },
    }
    save_training_log(log_entry)

    # Save
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    os.makedirs(str(save_dir), exist_ok=True)
    model_eval.config.id2label = id2label
    model_eval.config.label2id = label2id
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    np.save(f"{save_dir}/optimized_thresholds.npy", opt_thr)
    print(f"Model saved to {save_dir}")
    print("Done.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)

    # Dataset selection
    print("=" * 60)
    print("GUS-Net Training (Paper-Faithful) - Dataset Selection")
    print("=" * 60)
    print("\nSelect dataset:")
    print("  1. Cleaned dataset (gus_dataset_clean.json — punct-fixed)")
    print("  2. Hugging Face (ethical-spectacle/gus-dataset-v1)")
    print("  3. Gemini Annotations (gemini_annotations.json)")
    print()

    while True:
        ds_choice = input("Enter your choice (1-3): ").strip()
        if ds_choice == "1":
            dataset_source = "clean"
            break
        elif ds_choice == "2":
            dataset_source = "hf"
            break
        elif ds_choice == "3":
            dataset_source = "gemini"
            break
        else:
            print("Invalid choice.")

    # Model selection
    print()
    print("=" * 60)
    print("Select backbone:")
    print("=" * 60)
    print("  1. BERT (bert-base-uncased)")
    print("  2. GPT-2 (gpt2)")
    print()

    while True:
        choice = input("Enter your choice (1-2): ").strip()
        if choice == "1":
            print("\nStarting BERT training (paper-faithful)...\n")
            train_bert_paper(dataset_source=dataset_source)
            break
        elif choice == "2":
            print("\nStarting GPT-2 training (paper-faithful)...\n")
            train_gpt2_paper(dataset_source=dataset_source)
            break
        else:
            print("Invalid choice.")
