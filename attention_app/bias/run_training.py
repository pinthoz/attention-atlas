
# Multi-label NER Training (Scientific Method)
#
# Fine-tunes the pre-trained GUS-Net (ethical-spectacle/social-bias-ner) on
# custom annotated data. Supports combining multiple datasets and ablation
# via --from-scratch flag.
#
# Usage:
#   python run_training.py                           # fine-tune GUS-Net on new_dataset.json
#   python run_training.py --combine-datasets        # fine-tune on new_dataset + bias_annotations
#   python run_training.py --from-scratch            # train from bert-base-uncased (ablation)
#   python run_training.py --dataset path/to/data.json

import json
import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.functional as F
from torch.optim import AdamW
import shutil
from sklearn.metrics import classification_report, accuracy_score

# --- Configuration ---
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_EPOCHS = 20
THRESHOLD = 0.5
PATIENCE = 3
ALPHA = 0.65
GAMMA = 3.5
PRETRAINED_MODEL = 'ethical-spectacle/social-bias-ner'

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATASET_PATH = PROJECT_ROOT / 'dataset' / 'new_dataset.json'
BIAS_ANNOTATIONS_PATH = PROJECT_ROOT / 'dataset' / 'bias_annotations.json'
TEST_SET_OUTPUT_PATH = PROJECT_ROOT / 'dataset' / 'held_out_test_set.json'

# --- Label Mapping ---
label2id = {
    'O': 0, 'B-STEREO': 1, 'I-STEREO': 2, 'B-GEN': 3, 'I-GEN': 4, 'B-UNFAIR': 5, 'I-UNFAIR': 6
}
num_labels = len(label2id)

# --- Data Loading ---

def load_new_dataset_format(path):
    """Load data from new_dataset.json format (separate GEN/STEREO/UNFAIR annotation channels)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'bias_dataset' in data:
        data = data['bias_dataset']

    transformed_data = []
    for entry in data:
        text_str = entry['text_str']
        annotations = entry['annotations']
        seq_len = len(annotations['GEN'])

        combined_tags = []
        for i in range(seq_len):
            tags_for_token = []
            if annotations['GEN'][i] != 'O': tags_for_token.append(annotations['GEN'][i])
            if annotations['STEREO'][i] != 'O': tags_for_token.append(annotations['STEREO'][i])
            if annotations['UNFAIR'][i] != 'O': tags_for_token.append(annotations['UNFAIR'][i])

            if not tags_for_token:
                tags_for_token.append('O')

            combined_tags.append(tags_for_token)

        transformed_data.append({
            'text_str': text_str,
            'ner_tags': combined_tags,
            'id': entry.get('id', len(transformed_data))
        })

    return transformed_data


def load_bias_annotations_format(path):
    """Load data from bias_annotations.json format (ner_tags as list-of-lists)."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    transformed_data = []
    for i, entry in enumerate(data):
        transformed_data.append({
            'text_str': entry['text_str'],
            'ner_tags': entry['ner_tags'],
            'id': entry.get('original_id', i)
        })

    return transformed_data


def load_data(dataset_path, combine_datasets=False):
    """Load dataset(s) with optional combination."""
    dataset_path = Path(dataset_path)

    # Detect format by examining the file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'bias_dataset' in raw:
        all_data = load_new_dataset_format(dataset_path)
        print(f"Loaded {len(all_data)} samples from {dataset_path.name} (new_dataset format)")
    elif isinstance(raw, list) and len(raw) > 0 and 'ner_tags' in raw[0]:
        all_data = load_bias_annotations_format(dataset_path)
        print(f"Loaded {len(all_data)} samples from {dataset_path.name} (bias_annotations format)")
    else:
        raise ValueError(f"Unknown dataset format in {dataset_path}")

    if combine_datasets:
        # Combine with bias_annotations.json if primary is new_dataset.json, or vice versa
        if dataset_path.name == 'new_dataset.json' and BIAS_ANNOTATIONS_PATH.exists():
            extra = load_bias_annotations_format(BIAS_ANNOTATIONS_PATH)
            # Deduplicate by text_str
            existing_texts = {d['text_str'] for d in all_data}
            new_samples = [d for d in extra if d['text_str'] not in existing_texts]
            print(f"Adding {len(new_samples)} samples from {BIAS_ANNOTATIONS_PATH.name} (deduplicated)")
            all_data.extend(new_samples)
        elif dataset_path.name == 'bias_annotations.json' and DATASET_PATH.exists():
            extra = load_new_dataset_format(DATASET_PATH)
            existing_texts = {d['text_str'] for d in all_data}
            new_samples = [d for d in extra if d['text_str'] not in existing_texts]
            print(f"Adding {len(new_samples)} samples from {DATASET_PATH.name} (deduplicated)")
            all_data.extend(new_samples)

    print(f"Total samples: {len(all_data)}")
    return all_data


# --- Tokenization ---

def tokenize_and_align_labels(text, annotations, tokenizer, max_length=128):
    original_words = text.split()
    if len(original_words) != len(annotations):
        min_len = min(len(original_words), len(annotations))
        original_words = original_words[:min_len]
        annotations = annotations[:min_len]

    tokenized_inputs = tokenizer(original_words, is_split_into_words=True, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    word_ids = tokenized_inputs.word_ids()

    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append([-100] * num_labels)
        else:
            current_tags = annotations[word_idx]
            label_vector = [0] * num_labels
            for tag in current_tags:
                if tag in label2id:
                    label_vector[label2id[tag]] = 1
            aligned_labels.append(label_vector)

    return tokenized_inputs, aligned_labels

def preprocess_data(df, tokenizer, max_length=128):
    tokenized_inputs_list = []
    aligned_labels_list = []
    for _, row in df.iterrows():
        text = row['text_str']
        annotations = row['ner_tags']
        inputs, labels = tokenize_and_align_labels(text, annotations, tokenizer, max_length)
        tokenized_inputs_list.append(inputs)
        aligned_labels_list.append(labels)
    return tokenized_inputs_list, aligned_labels_list

# --- DataModule ---
class NERDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.input_ids = [t['input_ids'].squeeze() for t in tokenized_texts]
        self.attention_mask = [t['attention_mask'].squeeze() for t in tokenized_texts]
        self.labels = labels
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.attention_mask[idx], torch.tensor(self.labels[idx])

class NERDataModule(pl.LightningDataModule):
    def __init__(self, tokenized_texts, labels, batch_size=16, val_split=0.15):
        super().__init__()
        self.dataset = NERDataset(tokenized_texts, labels)
        self.batch_size = batch_size
        self.val_split = val_split
    def setup(self, stage=None):
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        self.train_ds, self.val_ds = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)

# --- Model ---
class NERModel(LightningModule):
    def __init__(self, pretrained_model=PRETRAINED_MODEL, learning_rate=5e-5, threshold=0.5, alpha=0.75, gamma=3):
        super().__init__()
        self.save_hyperparameters()
        if pretrained_model and pretrained_model != 'bert-base-uncased':
            self.bert = BertForTokenClassification.from_pretrained(
                pretrained_model, num_labels=7, ignore_mismatched_sizes=True
            )
        else:
            self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=7)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
    def focal_loss(self, logits, labels):
        logits = logits.view(-1, 7)
        labels = labels.view(-1, 7).float()
        valid_mask = (labels >= 0).all(dim=1)
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        if logits.shape[0] == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        p_t = torch.exp(-bce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean()
    def training_step(self, batch, batch_idx):
        loss = self.focal_loss(self(batch[0], batch[1]), batch[2])
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.focal_loss(self(batch[0], batch[1]), batch[2])
        self.log('val_loss', loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * self.trainer.estimated_stepping_batches), num_training_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='GUS-Net NER Fine-tuning')
    parser.add_argument('--dataset', type=str, default=str(DATASET_PATH),
                        help='Path to training dataset (default: dataset/new_dataset.json)')
    parser.add_argument('--combine-datasets', action='store_true',
                        help='Combine new_dataset.json and bias_annotations.json for training')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train from bert-base-uncased instead of pre-trained GUS-Net (ablation)')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS, help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Early stopping patience')
    args = parser.parse_args()

    base_model = 'bert-base-uncased' if args.from_scratch else PRETRAINED_MODEL
    print(f"Working Directory: {os.getcwd()}")
    print(f"Base model: {base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Combine datasets: {args.combine_datasets}")

    # --- Load Data ---
    all_data = load_data(Path(args.dataset), combine_datasets=args.combine_datasets)

    # --- Split ---
    train_val_data, test_data = train_test_split(all_data, test_size=0.20, random_state=42, shuffle=True)
    print(f"Train/Val Size: {len(train_val_data)}")
    print(f"Test Size: {len(test_data)}")

    with open(TEST_SET_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved held-out test set to {TEST_SET_OUTPUT_PATH}")

    ner_annotations_df = pd.DataFrame(train_val_data)

    # --- Tokenize ---
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_texts, labels = preprocess_data(ner_annotations_df, tokenizer)

    # --- DataModule ---
    data_module = NERDataModule(tokenized_texts, labels, batch_size=args.batch_size)

    # --- Training ---
    shutil.rmtree('lightning_logs', ignore_errors=True)
    shutil.rmtree('checkpoints', ignore_errors=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', dirpath='checkpoints',
        filename='gusnet-best-{epoch:02d}-{val_loss:.2f}_fn',
        save_top_k=1, mode='min', save_last=False,
        save_weights_only=True
    )
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=True, mode='min')

    model = NERModel(pretrained_model=base_model, learning_rate=args.lr, alpha=ALPHA, gamma=GAMMA)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto', devices=1, enable_progress_bar=False, accumulate_grad_batches=2
    )
    trainer.fit(model, data_module)

    # --- Evaluation ---
    print(f"Loading best model from {checkpoint_callback.best_model_path}")
    best_model = NERModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    best_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    test_df = pd.DataFrame(test_data)
    test_in, test_lbl = preprocess_data(test_df, tokenizer)
    test_ds = NERDataset(test_in, test_lbl)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    all_preds = []
    all_trues = []
    device = best_model.device
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch[0].to(device)
            mask = batch[1].to(device)
            batch_labels = batch[2].cpu().numpy()
            logits = best_model(input_ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)
            for i in range(len(batch_labels)):
                valid_indices = np.where((batch_labels[i] >= 0).all(axis=1))[0]
                if len(valid_indices) > 0:
                    p = preds[i][valid_indices]
                    l = batch_labels[i][valid_indices]
                    all_preds.extend(p)
                    all_trues.extend(l)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    print("\n--- Test Set Results ---")
    target_names = ['O', 'B-STEREO', 'I-STEREO', 'B-GEN', 'I-GEN', 'B-UNFAIR', 'I-UNFAIR']
    print(classification_report(all_trues, all_preds, target_names=target_names, zero_division=0))
    print(f"Exact Match Pct: {accuracy_score(all_trues, all_preds):.4f}")


if __name__ == '__main__':
    main()
