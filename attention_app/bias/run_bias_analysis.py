
import sys
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add root directory to path
current_dir = os.getcwd()
root_dir = current_dir
if 'attention_app' in current_dir:
     while 'attention_app' in current_dir and os.path.basename(current_dir) != 'attention-atlas':
        current_dir = os.path.dirname(current_dir)
        root_dir = current_dir

if root_dir not in sys.path:
    sys.path.append(root_dir)

print(f"Project root set to: {root_dir}")

from attention_app.models import ModelManager
import attention_app.bias.feature_extraction as fe
import importlib
importlib.reload(fe) # Ensure we get the latest code
from attention_app.bias.feature_extraction import extract_features_for_sentence

# Set plotting style
sns.set_theme(style="whitegrid")

# 1. Load Dataset
dataset_path = os.path.join(root_dir, 'dataset', 'bias_sentences.json')
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}")
    sys.exit(1)

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df_sentences = pd.DataFrame(data['entries'])
print(f"Loaded {len(df_sentences)} sentences.")

# 2. Init Model
model_name = "bert-base-uncased"
print(f"Loading Model: {model_name}...")
manager = ModelManager()

# 3. Extract Features
feature_rows = []
print("Starting feature extraction (including RAW ATTENTION)...")

for index, row in tqdm(df_sentences.iterrows(), total=df_sentences.shape[0]):
    text = row['text']
    label = 1 if row['has_bias'] else 0
    
    try:
        features = extract_features_for_sentence(text, model_name, manager)
        features['label'] = label
        features['original_id'] = row['id']
        feature_rows.append(features)
    except Exception as e:
        print(f"Error processing sentence id {row.get('id', '?')}: {e}")

print("Extraction complete.")

# 4. Train Model
df_features = pd.DataFrame(feature_rows)
feature_cols = [c for c in df_features.columns if c not in ['label', 'original_id']]
print(f"Feature Matrix Shape: {df_features.shape}")

X = df_features[feature_cols].fillna(0)
y = df_features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training RandomForest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) 
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Analysis
importances = clf.feature_importances_
fi_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values(by='importance', ascending=False)

def get_viz_group(feature_name):
    if feature_name.startswith('GAM_'): return 'Global Attention Metrics'
    if feature_name.startswith('Spec_'): return 'Head Specialization'
    if feature_name.startswith('ISA_'): return 'Inter-Sentence Attention'
    if feature_name.startswith('Tree_'): return 'Dependency Tree'
    if feature_name.startswith('AttMap_'): return 'Attention Map / Flow'
    return 'Other'

fi_df['Visualization'] = fi_df['feature'].apply(get_viz_group)

# Top 20 Features
print("\nTop 20 Features:")
print(fi_df.head(20)[['feature', 'importance', 'Visualization']])

# Grouped Analysis
group_importance = fi_df.groupby('Visualization')['importance'].sum().sort_values(ascending=False)
print("\nTotal Importance by Visualization Type:")
print(group_importance)

# Save report
report_path = os.path.join(root_dir, 'BIAS_ANALYSIS_FULL_ATTENTION.md')
with open(report_path, 'w') as f:
    f.write("# Bias Classification Analysis (Full Attention Features)\n\n")
    f.write(f"**Total Features:** {len(feature_cols)}\n")
    f.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}\n\n")
    
    f.write("## Importance by Visualization Type\n")
    for viz, imp in group_importance.items():
        f.write(f"- **{viz}**: {imp:.4f}\n")
        
    f.write("\n## Top 20 Features\n")
    f.write(fi_df.head(20).to_markdown(index=False))

print(f"Report saved to {report_path}")
