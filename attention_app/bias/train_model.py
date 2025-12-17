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

# Ensure we can import attention_app
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

from attention_app.models import ModelManager
from attention_app.bias.feature_extraction import extract_features_for_sentence

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['entries']

def main():
    print("Initializing Bias Classification Pipeline...")
    
    # 1. Load Dataset
    dataset_path = os.path.join(root_dir, 'dataset', 'bias_sentences.json')
    print(f"Loading dataset from {dataset_path}...")
    entries = load_dataset(dataset_path)
    print(f"Loaded {len(entries)} sentences.")
    
    # 2. Setup Model
    model_name = "bert-base-uncased"
    print(f"Loading NLP Model: {model_name}...")
    manager = ModelManager()
    
    # 3. Extract Features
    print("Extracting features (this may take a while)...")
    feature_rows = []
    labels = []
    
    for entry in tqdm(entries):
        text = entry['text']
        label = 1 if entry['has_bias'] else 0
        
        try:
            features = extract_features_for_sentence(text, model_name, manager)
            features['label'] = label
            feature_rows.append(features)
        except Exception as e:
            print(f"Error processing sentence id {entry.get('id', '?')}: {e}")
            
    # 4. Prepare DataFrame
    df = pd.DataFrame(feature_rows)
    print(f"Feature Matrix Shape: {df.shape}")
    
    # Fill NaNs if any (shouldn't be, but safety first)
    df = df.fillna(0)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 5. Train Classifier
    print("Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = clf.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # 7. Feature Importance Analysis
    importances = clf.feature_importances_
    feature_names = X.columns
    
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    
    # Group by Visualization
    def get_viz_group(feature_name):
        if feature_name.startswith('GAM_'): return 'Global Attention Metrics'
        if feature_name.startswith('Spec_'): return 'Head Specialization'
        if feature_name.startswith('ISA_'): return 'Inter-Sentence Attention'
        if feature_name.startswith('Tree_'): return 'Dependency Tree'
        return 'Other'
        
    fi_df['pviz'] = fi_df['feature'].apply(get_viz_group)
    
    viz_importance = fi_df.groupby('pviz')['importance'].sum().sort_values(ascending=False)
    viz_avg_importance = fi_df.groupby('pviz')['importance'].mean().sort_values(ascending=False)
    
    # 8. Generate Report
    report_path = os.path.join(root_dir, 'BIAS_CLASSIFICATION_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Bias Classification Report: Visualization Analysis\n\n")
        f.write("This report analyzes which attention visualizations provide the best features for detecting bias.\n\n")
        
        f.write("## 1. Overall Ranking (Total Importance)\n")
        f.write("Which visualization contributes the most *total* signal?\n\n")
        f.write(viz_importance.to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Efficiency Ranking (Average Importance per Feature)\n")
        f.write("Which visualization has the most 'valuable' individual features on average?\n\n")
        f.write(viz_avg_importance.to_markdown())
        f.write("\n\n")
    
        f.write("## 3. Top 20 Most Predictive Features\n")
        f.write(fi_df.head(20)[['feature', 'pviz', 'importance']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. Model Details\n")
        f.write(f"- **Accuracy**: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"- **Total Sentences**: {len(df)}\n")
        f.write(f"- **Feature Count**: {len(feature_names)}\n")

    print(f"\nReport generated at: {report_path}")

if __name__ == "__main__":
    main()
