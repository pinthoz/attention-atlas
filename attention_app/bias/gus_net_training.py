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


# ============================================================================
# BERT-SPECIFIC CONFIG
# ============================================================================

BERT_CONFIG = {
    "backbone": "bert-base-uncased",
    "alpha": 0.75,
    "gamma": 3.0,
    "save_dir": _SCRIPT_DIR / "gus-net-bert-final-my",
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
    "gamma": 2.0,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-final-my",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-2nd"),
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
    "gamma": 3.0,
    "save_dir": _SCRIPT_DIR / "gus-net-bert-large-final-my",
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
    "gamma": 2.0,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-medium-final-my",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-medium"),
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
    "gamma": 2.0,
    "label_smoothing": 0.05,
    "llrd_decay_factor": 0.85,
    "classifier_lr": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "save_dir": _SCRIPT_DIR / "gus-net-gpt2-large-final",
    "training_output_dir": str(_SCRIPT_DIR / "gus-net-gpt2-large"),
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
                     alpha=None, gamma=None):
            super().__init__()
            if alpha is None:
                alpha = config["alpha"]
            if gamma is None:
                gamma = config["gamma"]
            self.save_hyperparameters()
            self.bert = BertForTokenClassification.from_pretrained(
                config["backbone"], num_labels=num_labels,
            )
            self.learning_rate = learning_rate
            self.threshold = threshold
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, input_ids, attention_mask):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

        def focal_loss(self, logits, labels):
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1, num_labels).float()
            valid_mask = (labels >= 0).all(dim=1)
            logits = logits[valid_mask]
            labels = labels[valid_mask]
            if logits.shape[0] == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            pt = torch.exp(-bce)
            loss = self.alpha * (1 - pt) ** self.gamma * bce
            return loss.mean()

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
            optimizer = AdamW(self.parameters(), lr=self.learning_rate)
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

    model = GUSNetBERT()
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
            preds = (probs > THRESHOLD).astype(int)
            for i in range(len(labels_np)):
                valid = np.where((labels_np[i] >= 0).all(axis=1))[0]
                if len(valid) > 0:
                    all_preds.extend(preds[i][valid])
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
    print(classification_report(cat_trues, cat_preds, target_names=cat_names, zero_division=0))
    print(f"Exact Match: {accuracy_score(cat_trues, cat_preds):.4f}")

    print("\n--- Test Set Results (BIO Detail) ---")
    print(classification_report(all_trues, all_preds, target_names=list(label2id.keys()), zero_division=0))

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
    alpha_labels = inv_freq / inv_freq.sum()
    alpha_labels = torch.tensor(alpha_labels, dtype=torch.float32)

    print("\nLabel statistics:")
    for i, name in enumerate(label_names):
        print(f"  {name:10s}: {label_pos[i]:>6} positives, alpha={alpha_labels[i]:.4f}")

    # Focal Loss
    class FocalLossMultiLabel(nn.Module):
        def __init__(self, alpha, gamma=2.0, reduction="mean", label_smoothing=0.0):
            super().__init__()
            self.register_buffer("alpha", alpha)
            self.gamma = gamma
            self.reduction = reduction
            self.label_smoothing = label_smoothing

        def forward(self, inputs, targets):
            if self.label_smoothing > 0:
                targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            bce = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
            pt = torch.exp(-bce)
            focal = self.alpha.to(inputs.device) * (1 - pt) ** self.gamma * bce
            return focal.mean() if self.reduction == "mean" else focal.sum()

    # Custom Trainer with LLRD
    class FocalLossTrainerGPT2(Trainer):
        def __init__(self, *args, alpha_channel, gamma=2.0, label_smoothing=0.0,
                     llrd_decay_factor=0.85, classifier_lr=2e-4, **kwargs):
            super().__init__(*args, **kwargs)
            self.focal_loss = FocalLossMultiLabel(
                alpha=alpha_channel, gamma=gamma, label_smoothing=label_smoothing,
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
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1, num_labels)
            valid = labels_flat[:, 0] != -100.0
            loss = self.focal_loss(logits_flat[valid], labels_flat[valid])
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
                all_preds.extend(preds[i][valid])
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
    print(classification_report(cat_trues, cat_preds, target_names=cat_names, zero_division=0))
    print(f"Exact Match: {accuracy_score(cat_trues, cat_preds):.4f}")

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
