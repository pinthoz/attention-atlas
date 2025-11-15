"""Attention metric helpers extracted from the monolithic script."""

import numpy as np


def calculate_confidence(attention_matrix):
    """
    1. Attention Confidence (Confianca)
    C_max = max(A_ij)
    C_avg = (1/n) * sum(max_j(A_ij))
    """
    max_weight = np.max(attention_matrix)
    avg_max_per_row = np.mean(np.max(attention_matrix, axis=1))
    return max_weight, avg_max_per_row


def calculate_focus(attention_matrix):
    """
    3. Attention Focus (Foco)
    E = -sum(A_ij * log(A_ij))
    Entropy: valores mais altos = attention mais dispersa
    """
    epsilon = 1e-10  # evitar log(0)
    entropy = -np.sum(attention_matrix * np.log(attention_matrix + epsilon))
    return entropy


def calculate_sparsity(attention_matrix, threshold=0.01):
    """
    5. Attention Sparsity (Esparsidade)
    S = (1/n^2) * sum(1(A_ij < epsilon))
    Proporcao de pesos abaixo do threshold
    """
    return np.mean(attention_matrix < threshold)


def calculate_distribution_attributes(attention_matrix):
    """
    6. Attention Distribution Attributes (Atributos de Distribuicao)
    Q_p = quantile(A, p) para p in {0, 0.25, 0.5, 0.75, 1.0}
    """
    quantiles = [0, 0.25, 0.5, 0.75, 1.0]
    distribution = np.quantile(attention_matrix, quantiles)
    return {
        "min": distribution[0],
        "q25": distribution[1],
        "median": distribution[2],
        "q75": distribution[3],
        "max": distribution[4],
    }


def calculate_uniformity(attention_matrix):
    """
    8. Attention Uniformity (Uniformidade)
    U = std(A)
    Desvio padrao dos pesos
    """
    return np.std(attention_matrix)


def compute_all_attention_metrics(attention_matrix):
    """Convenience wrapper for computing all attention metrics at once."""
    max_conf, avg_conf = calculate_confidence(attention_matrix)
    focus = calculate_focus(attention_matrix)
    sparsity = calculate_sparsity(attention_matrix)
    distribution = calculate_distribution_attributes(attention_matrix)
    uniformity = calculate_uniformity(attention_matrix)

    return {
        "confidence_max": max_conf,
        "confidence_avg": avg_conf,
        "focus_entropy": focus,
        "sparsity": sparsity,
        "distribution_median": distribution["median"],
        "distribution_q25": distribution["q25"],
        "distribution_q75": distribution["q75"],
        "uniformity": uniformity,
    }


__all__ = [
    "calculate_confidence",
    "calculate_focus",
    "calculate_sparsity",
    "calculate_distribution_attributes",
    "calculate_uniformity",
    "compute_all_attention_metrics",
]
