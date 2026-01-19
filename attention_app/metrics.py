import numpy as np
from scipy.special import rel_entr
from typing import Dict, List, Tuple, Optional, Union


def calculate_confidence(attention_matrix):
    """
    Attention Confidence (Eq. 5-6 from paper)
    
    MaxA = max(aᵢⱼ) - Maximum attention weight in matrix
    AvgMaxA = (1/dₖ) × Σᵢ max_j(aᵢⱼ) - Average of row maxes
    
    Higher values = more confident attention (head focuses strongly on specific tokens)
    """
    max_weight = float(np.max(attention_matrix))
    avg_max_per_row = float(np.mean(np.max(attention_matrix, axis=1)))
    return max_weight, avg_max_per_row


def calculate_focus(attention_matrix):
    """
    Attention Focus / Entropy (Eq. 8 from paper)
    
    E = -Σᵢⱼ aᵢⱼ × log(aᵢⱼ)
    
    Returns raw entropy. Higher values = more dispersed attention.
    Note: Should be normalized by caller using log(n²) for 0-1 range.
    """
    epsilon = 1e-10  # avoid log(0)
    attn_flat = attention_matrix.flatten()
    attn_flat = attn_flat[attn_flat > epsilon]  # filter near-zero values
    entropy = -np.sum(attn_flat * np.log(attn_flat))
    return float(entropy)


def calculate_sparsity(attention_matrix, threshold=None):
    """
    Attention Sparsity (Eq. 11 from paper)
    
    S = Σᵢⱼ 𝟙(aᵢⱼ < τ) / (n²)
    
    Uses ADAPTIVE threshold: τ = 1/seq_len (instead of fixed 0.01)
    This makes sparsity comparable across different sequence lengths.
    
    Higher values = most tokens are ignored (more selective attention)
    """
    seq_len = attention_matrix.shape[0]
    
    # Adaptive threshold based on sequence length
    # For uniform distribution, each cell would have weight 1/n²
    # We use 1/n as threshold (slightly above uniform per-row)
    if threshold is None:
        threshold = 1.0 / seq_len
    
    return float(np.mean(attention_matrix < threshold))


def calculate_distribution_attributes(attention_matrix):
    """
    Attention Distribution Attributes (Eq. 12-13 from paper)
    
    Calculates quantiles of attention weights:
    Q_p = quantile(A, p) for p in {0, 0.25, 0.5, 0.75, 1.0}
    """
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    distribution = np.quantile(attention_matrix, quantiles)
    return {
        "min": float(distribution[0]),
        "q25": float(distribution[1]),
        "median": float(distribution[2]),
        "q75": float(distribution[3]),
        "max": float(distribution[4]),
    }


def calculate_uniformity(attention_matrix):
    """
    Attention Uniformity (Eq. 15 from paper)
    
    U = std(A) = √[(1/(n²)) × Σᵢⱼ(aᵢⱼ - μ)²]
    
    Lower values = more uniform attention distribution
    Higher values = more variable/concentrated attention
    """
    return float(np.std(attention_matrix))


def calculate_flow_change(all_layer_attentions):
    """
    Attention Flow Change (Eq. 9 from paper)
    
    Measures how attention patterns change between first and last layer
    using Jensen-Shannon Divergence.
    
    JSD(P, Q) = √[ ½ × KL(P || M) + ½ × KL(Q || M) ]
    where M = (P + Q) / 2
    
    Args:
        all_layer_attentions: list of arrays [num_heads, seq_len, seq_len] per layer
    
    Returns:
        float: JSD between 0 (identical) and 1 (completely different)
    """
    from scipy.spatial.distance import jensenshannon
    
    if len(all_layer_attentions) < 2:
        return 0.0
    
    def get_distribution(layer_attn):
        # Average across heads, flatten, normalize to probability distribution
        if len(layer_attn.shape) == 3:
            mean_attn = np.mean(layer_attn, axis=0)
        else:
            mean_attn = layer_attn
        dist = mean_attn.flatten()
        dist = np.clip(dist, 1e-10, None)
        return dist / dist.sum()
    
    first_dist = get_distribution(all_layer_attentions[0])
    last_dist = get_distribution(all_layer_attentions[-1])
    
    return float(jensenshannon(first_dist, last_dist))


def calculate_balance(attention_matrix, cls_index=0):
    """
    Attention Balance - Normalized (0-1)
    
    Measures proportion of attention directed to [CLS] vs total.
    
    Balance = attn_to_CLS / attn_total
    
    Args:
        attention_matrix: array [seq_len, seq_len]
        cls_index: index of [CLS] token (default: 0)
    
    Returns:
        float: 0 = all attention to content, 0.5 = balanced, 1 = all to [CLS]
    """
    seq_len = attention_matrix.shape[0]
    
    if seq_len < 2:
        return 0.5
    
    attn_to_cls = attention_matrix[:, cls_index].sum()
    attn_total = attention_matrix.sum()
    
    return float(attn_to_cls / attn_total) if attn_total > 0 else 0.5


def compute_all_attention_metrics(attention_matrix):
    """
    Convenience wrapper for computing all attention metrics at once.
    
    Based on paper: "From Attention to Assurance" (Golshanrad & Faghih)
    
    Returns dict with:
    - confidence_max: Max attention weight (Eq. 5)
    - confidence_avg: Average of row maxes (Eq. 6)
    - focus_entropy: Raw entropy (Eq. 8) - normalize with log(n²) for 0-1
    - sparsity: Proportion below adaptive threshold (Eq. 11)
    - distribution_median: Median attention weight (Eq. 12)
    - uniformity: Standard deviation of weights (Eq. 15)
    """
    max_conf, avg_conf = calculate_confidence(attention_matrix)
    focus = calculate_focus(attention_matrix)
    sparsity = calculate_sparsity(attention_matrix)  # Now uses adaptive threshold
    distribution = calculate_distribution_attributes(attention_matrix)
    uniformity = calculate_uniformity(attention_matrix)
    balance = calculate_balance(attention_matrix)

    return {
        "confidence_max": max_conf,
        "confidence_avg": avg_conf,
        "focus_entropy": focus,
        "sparsity": sparsity,
        "distribution_median": distribution["median"],
        "distribution_q25": distribution["q25"],
        "distribution_q75": distribution["q75"],
        "uniformity": uniformity,
        "balance": balance,
    }


__all__ = [
    "calculate_confidence",
    "calculate_focus",
    "calculate_sparsity",
    "calculate_distribution_attributes",
    "calculate_uniformity",
    "calculate_flow_change",
    "calculate_balance",
    "compute_all_attention_metrics",
]
