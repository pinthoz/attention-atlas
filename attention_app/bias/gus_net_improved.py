"""
GUS-Net BERT Training
=====================================

Training script for GUS-Net using a BERT backbone (`bert-base-uncased`).
Matches the architecture of `ner_bert_training.ipynb`.
Uses the `ethical-spectacle/gus-dataset-v1` from Hugging Face.

Architecture:
  1. BERT Backbone (`bert-base-uncased`)
  2. Focal loss
  3. AdamW + linear warmup scheduler
  4. EarlyStopping on val_loss (patience=3)
  5. sklearn classification_report evaluation

Label scheme (7 classes):
    0: O  |  1: B-STEREO  |  2: I-STEREO
    3: B-GEN  |  4: I-GEN  |  5: B-UNFAIR  |  6: I-UNFAIR
"""

import json
import os
import shutil
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 5e-5        # Matches NER BERT notebook
MAX_EPOCHS = 20
THRESHOLD = 0.5
PATIENCE = 3
ALPHA = 0.75                # focal loss alpha
GAMMA = 3.0                 # focal loss gamma
MAX_LENGTH = 128
SEED = 42
BACKBONE = "bert-base-uncased"


# ============================================================================
# PATHS
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
SAVE_DIR = _SCRIPT_DIR / "gus-net-bert-final"


# ============================================================================
# LABEL SCHEME
# ============================================================================

label2id = {
    "O": 0,
    "B-STEREO": 1, "I-STEREO": 2,
    "B-GEN": 3,    "I-GEN": 4,
    "B-UNFAIR": 5, "I-UNFAIR": 6,
}
num_labels = len(label2id)
id2label = {v: k for k, v in label2id.items()}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_from_hf():
    """Load dataset from Hugging Face and transform."""
    print("Loading dataset from ethical-spectacle/gus-dataset-v1...")
    dataset = load_dataset("ethical-spectacle/gus-dataset-v1", split="train")
    
    transformed = []
    
    # Iterate over the HF dataset
    for entry in dataset:
        text_str = entry["text_str"]
        
        # 'ner_tags' is a stringified list of lists: "[['O'], ['B-GEN', 'I-STEREO'], ...]"
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


# ============================================================================
# TOKENIZATION
# ============================================================================

tokenizer = BertTokenizerFast.from_pretrained(BACKBONE)

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


# ============================================================================
# LIGHTNING DATASET & DATAMODULE
# ============================================================================

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
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=0,
        )


# ============================================================================
# LIGHTNING MODEL  — BERT
# ============================================================================

class GUSNetBERT(pl.LightningModule):
    def __init__(self, learning_rate=LEARNING_RATE, threshold=THRESHOLD,
                 alpha=ALPHA, gamma=GAMMA):
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertForTokenClassification.from_pretrained(
            BACKBONE, num_labels=num_labels,
        )
        
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input_ids, attention_mask):
        return self.bert(
            input_ids=input_ids, attention_mask=attention_mask,
        ).logits

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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("GUS-Net BERT Training")
    print("=" * 60)
    print(f"Backbone:       {BACKBONE}")
    print(f"Dataset:        ethical-spectacle/gus-dataset-v1")
    print(f"Labels:         {num_labels} ({', '.join(label2id.keys())})")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Focal loss:     alpha={ALPHA}, gamma={GAMMA}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Max epochs:     {MAX_EPOCHS}")
    print(f"Patience:       {PATIENCE}")
    print(f"Device:         {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    # ── Load data ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING DATA (Hugging Face)")
    print("=" * 60)

    all_data = load_from_hf()
    print(f"Loaded {len(all_data)} samples")

    # Deterministic split: 80% train/val, 20% test
    train_val_data, test_data = train_test_split(
        all_data, test_size=0.20, random_state=SEED, shuffle=True,
    )
    print(f"Train/Val: {len(train_val_data)}")
    print(f"Test:      {len(test_data)}")

    # ── Tokenize ───────────────────────────────────────────────────────
    print("\nTokenizing...")
    df = pd.DataFrame(train_val_data)
    tokenized_texts, labels = preprocess_data(df)
    print(f"Processed {len(tokenized_texts)} training samples")

    # ── DataModule ─────────────────────────────────────────────────────
    data_module = NERDataModule(tokenized_texts, labels, batch_size=BATCH_SIZE)

    # ── Train ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Clean up logs and checkpoints from previous runs
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

    # ── Evaluate on held-out test set ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
         print("No best model saved (likely training failed or stopped early). Using current model.")
         best_model = model
    else:
        print(f"Loading best model from {best_model_path}")
        best_model = GUSNetBERT.load_from_checkpoint(best_model_path)
    
    best_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)

    # Tokenize test data
    test_df = pd.DataFrame(test_data)
    test_tok, test_lbl = preprocess_data(test_df)
    test_ds = NERDataset(test_tok, test_lbl)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    all_preds = []
    all_trues = []

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

    # ── Category-level evaluation ──────────────────────────────────
    def collapse_bio(arr):
        """Collapse 7-col BIO array → 4-col category array [O, STEREO, GEN, UNFAIR]."""
        return np.column_stack([
            arr[:, 0],                              # O
            np.maximum(arr[:, 1], arr[:, 2]),        # STEREO
            np.maximum(arr[:, 3], arr[:, 4]),        # GEN
            np.maximum(arr[:, 5], arr[:, 6]),        # UNFAIR
        ])

    cat_preds = collapse_bio(all_preds)
    cat_trues = collapse_bio(all_trues)
    cat_names = ["O", "STEREO", "GEN", "UNFAIR"]

    print("\n--- Test Set Results (Category-Level) ---")
    print(classification_report(
        cat_trues, cat_preds,
        target_names=cat_names,
        zero_division=0,
    ))
    print(f"Exact Match: {accuracy_score(cat_trues, cat_preds):.4f}")

    # Also print per-token BIO detail for reference
    print("\n--- Test Set Results (BIO Detail) ---")
    target_names_bio = list(label2id.keys())
    print(classification_report(
        all_trues, all_preds,
        target_names=target_names_bio,
        zero_division=0,
    ))

    # ── Save HuggingFace-compatible model ──────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    os.makedirs(str(SAVE_DIR), exist_ok=True)
    best_model.bert.config.id2label = id2label
    best_model.bert.config.label2id = label2id
    best_model.bert.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    print(f"Model saved to {SAVE_DIR}")
    print("Done.")
