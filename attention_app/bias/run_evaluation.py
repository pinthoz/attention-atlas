
import json
import os
import getpass
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from pytorch_lightning import LightningModule
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from torchmetrics import HammingDistance
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F

# --- Configuration ---
BATCH_SIZE = 16
THRESHOLD = 0.5
BASE_DIR = Path('../../') 
TEST_SET_OUTPUT_PATH = BASE_DIR / 'dataset' / 'held_out_test_set.json'
CHECKPOINT_DIR = Path('checkpoints')

# Find best checkpoint
checkpoints = list(CHECKPOINT_DIR.glob('gusnet-best-*.ckpt'))
if not checkpoints:
    raise FileNotFoundError("No checkpoints found!")
# Sort by modification time or name? Name has loss.
# gusnet-best-epoch=00-val_loss=0.08.ckpt
# Let's pick the one with lowest loss in name just to be safe, or just the last one.
best_checkpoint = sorted(checkpoints, key=lambda p: str(p))[-1] # Simple sort
print(f"Loading checkpoint: {best_checkpoint}")

# --- Model Definition (Must verify it matches training) ---
class NERModel(LightningModule):
    def __init__(self, pretrained_model='ethical-spectacle/social-bias-ner', learning_rate=5e-5, threshold=0.5, alpha=0.75, gamma=3):
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

# --- Data Loading ---
with open(TEST_SET_OUTPUT_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
label2id = {
    'O': 0, 'B-STEREO': 1, 'I-STEREO': 2, 'B-GEN': 3, 'I-GEN': 4, 'B-UNFAIR': 5, 'I-UNFAIR': 6
}
num_labels = len(label2id)

def tokenize_and_align_labels(text, annotations, tokenizer, label2id, max_length=128):
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

def preprocess_data(df, tokenizer, label2id, max_length=128):
    tokenized_inputs_list = []
    aligned_labels_list = []
    for _, row in df.iterrows():
        text = row['text_str']
        annotations = row['ner_tags']
        inputs, labels = tokenize_and_align_labels(text, annotations, tokenizer, label2id, max_length)
        tokenized_inputs_list.append(inputs)
        aligned_labels_list.append(labels)
    return tokenized_inputs_list, aligned_labels_list

class NERDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.input_ids = [t['input_ids'].squeeze() for t in tokenized_texts]
        self.attention_mask = [t['attention_mask'].squeeze() for t in tokenized_texts]
        self.labels = labels
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.attention_mask[idx], torch.tensor(self.labels[idx])

test_df = pd.DataFrame(test_data)
test_in, test_lbl = preprocess_data(test_df, tokenizer, label2id)
test_ds = NERDataset(test_in, test_lbl)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- Inference ---
model = NERModel.load_from_checkpoint(best_checkpoint)
model.eval()
device = 'cpu'
model.to(device)

all_preds = []
all_trues = []

print(f"Evaluating on {len(test_data)} test samples...")

with torch.no_grad():
    for batch in test_dl:
        input_ids = batch[0].to(device)
        mask = batch[1].to(device)
        labels = batch[2].cpu().numpy()
        
        logits = model(input_ids, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)
        
        for i in range(len(labels)):
            # Filter where labels are NOT -100 (which is represented as [0,0,0..] or something? 
            # In tokenization I put [-100] * num_labels
            # So if any element is -100, skip.
            # My logic in training script was: valid_indices = np.where((labels[i] >= 0).all(axis=1))[0]
            
            valid_indices = np.where((labels[i] >= 0).all(axis=1))[0]
            if len(valid_indices) > 0:
                p = preds[i][valid_indices]
                l = labels[i][valid_indices]
                all_preds.extend(p)
                all_trues.extend(l)

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

print("\n--- Test Set Results ---")
target_names = ['O', 'B-STEREO', 'I-STEREO', 'B-GEN', 'I-GEN', 'B-UNFAIR', 'I-UNFAIR']
report = classification_report(all_trues, all_preds, target_names=target_names, zero_division=0)
print(report)
acc = accuracy_score(all_trues, all_preds)
print(f"Exact Match Pct: {acc:.4f}")

with open("evaluation_results.txt", "w") as f:
    f.write(report)
    f.write(f"\nExact Match Pct: {acc:.4f}")
