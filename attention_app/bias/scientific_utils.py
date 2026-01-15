
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import  Pipeline

def bootstrap_confidence_intervals(y_true, y_pred, y_prob=None, n_iterations=1000, confidence=0.95):
    """
    Calculate confidence intervals for performance metrics using bootstrapping.
    Critical for scientific rigor to show stability of results.
    """
    stats_dict = {
        "accuracy": [],
        "f1": [],
        "roc_auc": [] if y_prob is not None else None
    }
    
    # Run bootstrap
    for i in range(n_iterations):
        # Resample with replacement
        indices = resample(np.arange(len(y_true)), replace=True)
        if y_prob is not None:
            y_p_iter = y_prob[indices]
        y_true_iter = np.array(y_true)[indices]
        y_pred_iter = np.array(y_pred)[indices]
        
        # Calculate metrics
        try:
            stats_dict["accuracy"].append(accuracy_score(y_true_iter, y_pred_iter))
            stats_dict["f1"].append(f1_score(y_true_iter, y_pred_iter))
            if y_prob is not None:
                # Handle case where only one class is present in bootstrap sample
                if len(np.unique(y_true_iter)) > 1:
                    stats_dict["roc_auc"].append(roc_auc_score(y_true_iter, y_p_iter))
        except Exception:
            continue
    
    # Compute intervals
    alpha = (1.0 - confidence) / 2.0 * 100
    lower = alpha
    upper = 100 - alpha
    
    results = {}
    for metric, values in stats_dict.items():
        if values is None or len(values) == 0: continue
        mean_val = np.mean(values)
        std_val = np.std(values)
        low_val = np.percentile(values, lower)
        high_val = np.percentile(values, upper)
        results[metric] = {
            "mean": mean_val,
            "std": std_val,
            "95_ci_lower": low_val,
            "95_ci_upper": high_val,
            "formatted": f"{mean_val:.4f} ± {std_val*1.96:.4f} (95% CI: [{low_val:.4f}, {high_val:.4f}])"
        }
    
    return results

def plot_model_calibration(y_true, y_prob, model_name="Model"):
    """
    Plot calibration curve to check if predicted probabilities match observed frequencies.
    Essential for asserting the model is not 'overconfident'.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier={brier:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Observed)')
    plt.title(f'Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return brier

def analyze_feature_stability(X, y, model_class, param_grid=None, n_splits=5):
    """
    Analyze which features are consistently selected across different Cross-Validation folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_counts = pd.Series(0, index=X.columns)
    
    print(f"Running Feature Stability Analysis ({n_splits} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Simple RFE or Model-based selection
        try:
            model = model_class()
            model.fit(X_train, y_train)
            
            # Get importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                continue
                
            # Select top 50 features for this fold
            top_indices = np.argsort(importances)[-50:]
            top_features = X.columns[top_indices]
            feature_counts[top_features] += 1
        except Exception as e:
            print(f"Fold {fold} failed: {e}")
            continue
    
    # Calculate stability score
    stable_features = feature_counts[feature_counts == n_splits]
    
    print(f"\nFeature Stability Report:")
    print(f"Total features selected at least once: {len(feature_counts[feature_counts > 0])}")
    print(f"Features robust across all {n_splits} folds: {len(stable_features)}")
    
    plt.figure(figsize=(10, 5))
    feature_counts[feature_counts > 0].sort_values(ascending=False).head(20).plot(kind='bar')
    plt.title("Top Most Stable Features (Selected across CV folds)")
    plt.ylabel("Number of Folds Selected")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return stable_features.index.tolist()

def compare_with_baseline(X_test, y_test, y_pred_model, model_name="Attention-Model"):
    """
    Statistical difference test between your model and a random/dummy baseline.
    """
    # 1. Random Baseline (50/50 for balanced)
    random_acc = 0.5
    model_acc = accuracy_score(y_test, y_pred_model)
    
    # Binomial test
    n = len(y_test)
    k = int(model_acc * n)
    try:
        res = stats.binomtest(k, n, p=random_acc, alternative='greater')
        p_value = res.pvalue
    except AttributeError:
        p_value = stats.binom_test(k, n, p=random_acc, alternative='greater')
    
    print(f"\nStatistical Significance Test (vs Random Baseline):")
    print(f"Model Accuracy: {model_acc:.4f}")
    print(f"Baseline Accuracy: {random_acc:.4f}")
    print(f"p-value: {p_value:.2e}")
    if p_value < 0.05:
        print("Result: Statistically Significant Improvement (p < 0.05)")
    else:
        print("Result: Not Statistically Significant")
        
    return p_value

def run_tfidf_baseline(df_train=None, df_test=None, df=None, text_col='text', label_col='label', n_splits=5):
    """
    Runs a strong baseline (TF-IDF + Logistic Regression) to compare against the complex model.

    Args:
        df_train: Training dataframe (optional, for train/test split approach)
        df_test: Test dataframe (optional, for train/test split approach)
        df: Full dataframe (optional, for cross-validation approach)
        text_col: Column name containing text
        label_col: Column name containing labels
        n_splits: Number of cross-validation folds (only used if df is provided)

    Returns:
        Accuracy score (test accuracy if df_train/df_test provided, CV mean if df provided)
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train/Test split approach
    if df_train is not None and df_test is not None:
        print("\nRunning Strong Baseline (TF-IDF + Logistic Regression) on Train/Test Split...")
        X_train = df_train[text_col]
        y_train = df_train[label_col]
        X_test = df_test[text_col]
        y_test = df_test[label_col]

        pipeline.fit(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)

        print(f"Baseline Test Accuracy: {test_acc:.4f}")
        return test_acc

    # Cross-validation approach
    elif df is not None:
        print(f"\nRunning Strong Baseline (TF-IDF + Logistic Regression) with {n_splits}-fold CV...")
        X = df[text_col]
        y = df[label_col]

        scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring='accuracy')

        print(f"Baseline Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores.mean()

    else:
        raise ValueError("Must provide either (df_train and df_test) or (df)")

def analyze_error_types(model, X_test, y_test, df_sentences, text_col='text'):
    """
    Identifies and prints false positives and false negatives with their original text.
    Relies on X_test indices matching df_sentences indices.
    """
    print("\n=== Qualitative Error Analysis ===")
    
    y_pred = model.predict(X_test)
    
    # Find indices
    X_test_reset = X_test.copy()
    X_test_reset['predicted'] = y_pred
    X_test_reset['actual'] = y_test.values
    
    # False Positives (Predicted 1, Actual 0)
    fp_indices = X_test_reset[(X_test_reset['predicted'] == 1) & (X_test_reset['actual'] == 0)].index
    
    # False Negatives (Predicted 0, Actual 1)
    fn_indices = X_test_reset[(X_test_reset['predicted'] == 0) & (X_test_reset['actual'] == 1)].index
    
    print(f"Found {len(fp_indices)} False Positives and {len(fn_indices)} False Negatives.")
    
    print("\n--- False Positives (Labeled Neutral, Predicted Biased) ---")
    for idx in fp_indices[:5]:
        if idx in df_sentences.index:
            print(f"[{idx}] {df_sentences.loc[idx, text_col]}")
            
    print("\n--- False Negatives (Labeled Biased, Predicted Neutral) ---")
    for idx in fn_indices[:5]:
        if idx in df_sentences.index:
            print(f"[{idx}] {df_sentences.loc[idx, text_col]}")

def analyze_bias_threshold(y_true, y_prob):
    """
    Analyzes prediction stability across different thresholds.
    """
    thresholds = np.arange(0.3, 0.8, 0.05)
    accuracies = []
    
    print("\nThreshold Sensitivity Analysis:")
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        accuracies.append(acc)
        print(f"Threshold {t:.2f} -> Accuracy: {acc:.4f}")
    
    plt.figure(figsize=(8,4))
    plt.plot(thresholds, accuracies, marker='o')
    plt.xlabel("Decision Threshold")
    plt.ylabel("Accuracy")
    plt.title("Impact of Decision Threshold on Accuracy")
    plt.grid(True)
    plt.show()

