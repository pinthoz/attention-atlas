"""
GUS-Net Fine-tuning Script (Matching Original GUS-Net Training)
================================================================
Fine-tunes the pre-trained ethical-spectacle/social-bias-ner model
on custom annotated data using the EXACT same training setup as
the original GUS-Net notebook.

Key matching parameters:
- Focal Loss with alpha=0.65, gamma=3.5
- Learning rate: 5e-5
- Batch size: 16
- Max epochs: 20
- Threshold: 0.5
- PyTorch Lightning training loop
- Same label mapping and metrics

Usage:
    python finetune_gusnet.py --annotations dataset/bias_annotations.json
"""

import json
import os
import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import (
    MultilabelPrecision, 
    MultilabelRecall, 
    MultilabelF1Score,
    MultilabelConfusionMatrix
)
from torchmetrics import HammingDistance
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION - Matching Original GUS-Net Notebook
# ============================================================================

CONFIG = {
    "pretrained_model": "ethical-spectacle/social-bias-ner",
    "base_model": "bert-base-uncased",  # Original used this
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 5e-5,  # Original: 5e-5
    "num_epochs": 20,       # Original: 20
    "threshold": 0.5,       # Original: 0.5
    "alpha": 0.65,          # Original focal loss alpha
    "gamma": 3.5,           # Original focal loss gamma
    "val_split": 0.15,
    "test_split": 0.15,
}

# Label mapping - EXACT same as original notebook
LABEL2ID = {
    'O': 0,
    'B-STEREO': 1,
    'I-STEREO': 2,
    'B-GEN': 3,
    'I-GEN': 4,
    'B-UNFAIR': 5,
    'I-UNFAIR': 6
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


# ============================================================================
# DATA LOADING AND PREPROCESSING - Matching Original
# ============================================================================

def load_annotations(filepath: str) -> list:
    """Load annotations from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} annotations")
    return data


def tokenize_and_align_labels(text, annotations, tokenizer, label2id, max_length=128):
    """
    Tokenize text and align labels - EXACT same logic as original notebook.
    """
    # Tokenize with word alignment
    tokenized_inputs = tokenizer(
        text.split(), 
        is_split_into_words=True, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    word_ids = tokenized_inputs.word_ids()
    
    # Align labels
    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:  # Padding or special tokens
            aligned_labels.append([-100] * NUM_LABELS)
        else:
            try:
                labels = annotations[word_idx]
            except IndexError:
                labels = []
            
            # Convert text labels to fixed-size label vector
            label_vector = [0] * NUM_LABELS
            for label in labels:
                if label in label2id:
                    label_vector[label2id[label]] = 1
            aligned_labels.append(label_vector)
    
    assert len(aligned_labels) == len(tokenized_inputs['input_ids'][0])
    
    return tokenized_inputs, aligned_labels


def preprocess_data(annotations_list, tokenizer, label2id, max_length=128):
    """Preprocess all data - matching original notebook workflow."""
    tokenized_inputs_list = []
    aligned_labels_list = []
    
    for entry in annotations_list:
        text = entry['text_str']
        # Use individual_annotations to build ner_tags format
        ner_tags = entry.get('ner_tags', [])
        
        tokenized_inputs, aligned_labels = tokenize_and_align_labels(
            text, ner_tags, tokenizer, label2id, max_length
        )
        tokenized_inputs_list.append(tokenized_inputs)
        aligned_labels_list.append(aligned_labels)
    
    return tokenized_inputs_list, aligned_labels_list


# ============================================================================
# DATASET - Matching Original
# ============================================================================

class NERDataset(Dataset):
    """NER Dataset - exact same as original notebook."""
    
    def __init__(self, tokenized_texts, labels):
        assert len(tokenized_texts) == len(labels)
        
        self.input_ids = [inputs['input_ids'].squeeze() for inputs in tokenized_texts]
        self.attention_mask = [inputs['attention_mask'].squeeze() for inputs in tokenized_texts]
        self.labels = labels
        
        # Check for length mismatches and trim if necessary
        for i in range(len(self.input_ids)):
            if len(self.input_ids[i]) != len(self.labels[i]):
                min_len = min(len(self.input_ids[i]), len(self.labels[i]))
                self.input_ids[i] = self.input_ids[i][:min_len]
                self.attention_mask[i] = self.attention_mask[i][:min_len]
                self.labels[i] = self.labels[i][:min_len]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            torch.tensor(self.labels[idx])
        )


class NERDataModule(pl.LightningDataModule):
    """Data module - exact same as original notebook."""
    
    def __init__(self, tokenized_texts, labels, batch_size=16, val_split=0.15, test_split=0.15):
        super().__init__()
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
    
    def setup(self, stage=None):
        dataset = NERDataset(self.tokenized_texts, self.labels)
        val_size = int(self.val_split * len(dataset))
        test_size = int(self.test_split * len(dataset))
        train_size = len(dataset) - val_size - test_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# ============================================================================
# MODEL - Matching Original GUS-Net Architecture
# ============================================================================

class NERModel(pl.LightningModule):
    """
    NER Model - EXACT same as original GUS-Net notebook.
    Using Focal Loss with same alpha/gamma parameters.
    """
    
    def __init__(
        self, 
        pretrained_model: str = None,
        learning_rate: float = 5e-5, 
        threshold: float = 0.5, 
        alpha: float = 0.65, 
        gamma: float = 3.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model or base BERT
        if pretrained_model:
            print(f"Loading pre-trained: {pretrained_model}")
            self.bert = BertForTokenClassification.from_pretrained(
                pretrained_model, 
                num_labels=NUM_LABELS,
                ignore_mismatched_sizes=True
            )
        else:
            print("Loading base bert-base-uncased")
            self.bert = BertForTokenClassification.from_pretrained(
                'bert-base-uncased', 
                num_labels=NUM_LABELS
            )
        
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.num_labels = NUM_LABELS
        self.alpha = alpha
        self.gamma = gamma
        
        # Metrics - same as original (4 merged classes: O, GEN, UNFAIR, STEREO)
        self.precision = MultilabelPrecision(num_labels=4, average=None)
        self.recall = MultilabelRecall(num_labels=4, average=None)
        self.f1 = MultilabelF1Score(num_labels=4, average=None)
        self.hamming_loss = HammingDistance(task='multilabel', num_labels=4)
        self.confmat = MultilabelConfusionMatrix(num_labels=4)
        
        self.all_preds = []
        self.all_labels = []
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return outputs
    
    def focal_loss_with_logits(self, outputs, labels, reduction='mean'):
        """
        Focal Loss - EXACT same implementation as original notebook.
        """
        # Mask out -100s
        mask = (labels != -100)
        labels = labels.float()
        
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
        bce_loss = bce_loss * mask
        
        # Sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        
        # Calculate p_t
        p_t = labels * probabilities + (1 - labels) * (1 - probabilities)
        p_t = torch.clamp(p_t, 0.01, 0.99)
        
        # Modulation factors
        alpha_t = labels * self.alpha + (1 - labels) * (1 - self.alpha)
        focal_modulation = (1 - p_t) ** self.gamma
        
        # Apply modulation
        focal_loss = alpha_t * focal_modulation * bce_loss
        focal_loss = focal_loss * mask
        
        if reduction == 'mean':
            return focal_loss.sum() / mask.sum()
        elif reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
    def loss_fn(self, outputs, labels):
        """Loss function - matching original."""
        outputs = outputs.view(-1, self.num_labels)
        labels = labels.view(-1, self.num_labels)
        return self.focal_loss_with_logits(outputs, labels.float(), reduction='mean')
    
    def merge_labels(self, labels):
        """
        Merge B/I labels into entity-level labels.
        Output: [O, GEN, UNFAIR, STEREO] - same as original.
        """
        merged = torch.zeros(labels.size(0), 4).to(labels.device)
        
        merged[:, 0] = labels[:, 0]  # O
        merged[:, 1] = torch.max(labels[:, 3], labels[:, 4])  # GEN (B-GEN or I-GEN)
        merged[:, 2] = torch.max(labels[:, 5], labels[:, 6])  # UNFAIR
        merged[:, 3] = torch.max(labels[:, 1], labels[:, 2])  # STEREO
        
        return merged.long()
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss)
        
        # Store predictions
        self.store_predictions_and_labels(outputs, labels)
        return loss
    
    def store_predictions_and_labels(self, outputs, labels):
        """Store predictions for metrics - same as original."""
        probabilities = torch.sigmoid(outputs)
        batch_predictions = (probabilities > self.threshold).long()
        
        preds_flat = batch_predictions.view(-1, self.num_labels)
        labels_flat = labels.view(-1, self.num_labels)
        
        # Filter out padding (-100)
        # Verify if any label in the multi-label vector is -100
        # Since padding is usually applied to all labels for a token, checking index 0 is sufficient
        active_mask = labels_flat[:, 0] != -100
        
        preds_flat = preds_flat[active_mask]
        labels_flat = labels_flat[active_mask]
        
        if len(labels_flat) > 0:
            merged_preds = self.merge_labels(preds_flat)
            merged_labels = self.merge_labels(labels_flat)
            
            self.all_preds.append(merged_preds.cpu())
            self.all_labels.append(merged_labels.cpu())
    
    def on_test_epoch_end(self):
        """Calculate and log metrics - same as original."""
        all_preds = torch.cat(self.all_preds, dim=0).to(self.device)
        all_labels = torch.cat(self.all_labels, dim=0).to(self.device)
        
        precision = self.precision(all_preds, all_labels)
        recall = self.recall(all_preds, all_labels)
        f1 = self.f1(all_preds, all_labels)
        hamming = self.hamming_loss(all_preds, all_labels)
        
        labels_names = ['O', 'GEN', 'UNFAIR', 'STEREO']
        
        print("\n" + "=" * 50)
        print("Test Results")
        print("=" * 50)
        
        for i, name in enumerate(labels_names):
            print(f"{name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")
            self.log(f'{name}_precision', precision[i])
            self.log(f'{name}_recall', recall[i])
            self.log(f'{name}_f1', f1[i])
        
        print(f"\nMacro Avg: P={precision.mean():.4f}, R={recall.mean():.4f}, F1={f1.mean():.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        
        self.log('precision_avg', precision.mean())
        self.log('recall_avg', recall.mean())
        self.log('f1_avg', f1.mean())
        self.log('hamming_loss', hamming)
        
        # Clear for next test
        self.all_preds.clear()
        self.all_labels.clear()
    
    def configure_optimizers(self):
        """Optimizer setup - same as original with warmup scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.trainer.estimated_stepping_batches // 10,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def save_model(self, save_path):
        """Save the underlying BERT model."""
        self.bert.save_pretrained(save_path, safe_serialization=False)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def finetune(
    annotations_path: str,
    output_dir: str = "models/gusnet-finetuned",
    use_pretrained: bool = True,
    **kwargs
):
    """
    Fine-tune GUS-Net using the original training setup.
    """
    print("=" * 60)
    print("GUS-Net Fine-tuning (Original Setup)")
    print("=" * 60)
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
    
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    annotations = load_annotations(annotations_path)
    tokenized_texts, labels = preprocess_data(annotations, tokenizer, LABEL2ID, CONFIG["max_length"])
    
    print(f"   Tokenized {len(tokenized_texts)} samples")
    
    # Create data module
    print("\n3. Creating data module...")
    data_module = NERDataModule(
        tokenized_texts, 
        labels,
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"],
        test_split=CONFIG["test_split"]
    )
    
    # Create model
    print("\n4. Initializing model...")
    model = NERModel(
        pretrained_model=CONFIG["pretrained_model"] if use_pretrained else None,
        learning_rate=CONFIG["learning_rate"],
        threshold=CONFIG["threshold"],
        alpha=CONFIG["alpha"],
        gamma=CONFIG["gamma"]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        ModelCheckpoint(
            dirpath=output_dir,
            filename='gusnet-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        )
    ]
    
    # Trainer
    print("\n5. Starting training...")
    trainer = Trainer(
        max_epochs=CONFIG["num_epochs"],
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    print("\n6. Running test evaluation...")
    trainer.test(model, data_module)
    
    # Save final model
    print(f"\n7. Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    info = {
        "base_model": CONFIG["pretrained_model"] if use_pretrained else "bert-base-uncased",
        "config": CONFIG,
        "num_samples": len(tokenized_texts),
    }
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)
    
    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GUS-Net (Original Setup)")
    parser.add_argument(
        "--annotations", 
        default="dataset/bias_annotations.json",
        help="Path to annotations file"
    )
    parser.add_argument(
        "--output", 
        default="models/gusnet-finetuned",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from bert-base-uncased instead of pre-trained GUS-Net"
    )
    
    args = parser.parse_args()
    
    finetune(
        annotations_path=args.annotations,
        output_dir=args.output,
        use_pretrained=not args.from_scratch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
