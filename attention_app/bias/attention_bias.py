"""Attention × Bias Interaction Analysis.

Quantifies how each attention head interacts with tokens flagged as
socially biased (by GUS-Net NER), using two complementary metrics:

  BAR  — Bias Attention Ratio
         Measures whether a head *over-attends* to biased tokens
         relative to a uniform baseline.

  BSR  — Bias Self-Reinforcement
         Measures whether biased tokens disproportionately attend
         to *each other*, forming attention "echo-chambers".

Both are expressed as observed / expected ratios centred at 1.0.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HeadBiasMetrics:
    """Per-head bias interaction metrics.

    Attributes
    ----------
    bias_attention_ratio : float
        BAR(l,h) = μ̂_B / μ₀   where
            μ̂_B = (1/N) Σ_i Σ_{j∈B} α_ij   (observed)
            μ₀   = |B| / N                   (expected under uniform)
        Centred at 1.0; values > 1.5 indicate specialisation.

    amplification_score : float
        BSR(l,h) = (1/|B|) Σ_{i∈B} Σ_{j∈B} α_ij  /  (|B|/N)
        Measures bias self-reinforcement (biased→biased attention).
        Centred at 1.0; high values indicate echo-chamber patterns.
    """
    layer: int
    head: int
    bias_attention_ratio: float   # BAR — observed / expected attention to biased tokens
    amplification_score: float    # BSR — biased-to-biased self-reinforcement
    max_bias_attention: float     # max α_ij where j ∈ B
    specialized_for_bias: bool    # BAR > 1.5


class AttentionBiasAnalyzer:
    """Analyzes how attention patterns interact with bias."""

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze_attention_to_bias(
        self,
        attention_weights: List[torch.Tensor],
        biased_token_indices: List[int],
        tokens: List[str]
    ) -> List[HeadBiasMetrics]:
        """Analyze how much each attention head focuses on biased tokens.

        Args:
            attention_weights: List of attention tensors, one per layer
                              Shape: [batch, num_heads, seq_len, seq_len]
            biased_token_indices: Indices of tokens identified as biased
            tokens: List of token strings

        Returns:
            List of HeadBiasMetrics for each (layer, head) combination
        """
        if not biased_token_indices:
            return []

        results = []
        biased_indices_set = set(biased_token_indices)

        for layer_idx, layer_attention in enumerate(attention_weights):
            # layer_attention shape: [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = layer_attention.shape

            for head_idx in range(num_heads):
                # Get attention matrix for this head: [seq_len, seq_len]
                head_attn = layer_attention[0, head_idx].cpu().numpy()

                # Calculate metrics
                metrics = self._compute_head_metrics(
                    head_attn,
                    biased_indices_set,
                    layer_idx,
                    head_idx
                )

                results.append(metrics)

        return results

    def _compute_head_metrics(
        self,
        attention_matrix: np.ndarray,
        biased_indices: set,
        layer_idx: int,
        head_idx: int
    ) -> HeadBiasMetrics:
        """Compute BAR and BSR for a single attention head.

        Let A ∈ ℝ^{N×N} be the attention matrix (rows = queries,
        cols = keys) and B ⊆ {1,…,N} the biased-token indices.

        BAR (Bias Attention Ratio)
        ──────────────────────────
        μ̂_B  = (1/N) Σ_i Σ_{j∈B} α_{ij}     (observed)
        μ₀   = |B| / N                        (expected, uniform)
        BAR  = μ̂_B / μ₀

        BSR (Bias Self-Reinforcement)
        ─────────────────────────────
        ν̂   = (1/|B|) Σ_{i∈B} Σ_{j∈B} α_{ij}   (observed)
        ν₀   = |B| / N                            (expected, uniform)
        BSR  = ν̂ / ν₀
        """
        seq_len = attention_matrix.shape[0]
        biased_mask = np.array([i in biased_indices for i in range(seq_len)])
        n_biased = int(biased_mask.sum())

        if n_biased == 0:
            return HeadBiasMetrics(
                layer=layer_idx, head=head_idx,
                bias_attention_ratio=0.0, amplification_score=0.0,
                max_bias_attention=0.0, specialized_for_bias=False,
            )

        # ── BAR: Bias Attention Ratio ──────────────────────────
        # μ̂_B = (1/N) Σ_i Σ_{j∈B} α_ij   (mean attention mass → biased keys)
        mu_observed = attention_matrix[:, biased_mask].sum() / seq_len
        # μ₀  = |B| / N                   (expected under uniform)
        mu_expected = n_biased / seq_len
        bias_attention_ratio = mu_observed / mu_expected if mu_expected > 0 else 0.0

        # ── BSR: Bias Self-Reinforcement ───────────────────────
        # ν̂ = (1/|B|) Σ_{i∈B} Σ_{j∈B} α_ij
        #    → average attention from a biased query to all biased keys
        nu_observed = attention_matrix[biased_mask][:, biased_mask].sum() / n_biased
        # ν₀ = |B| / N  (same expected baseline)
        amplification_score = nu_observed / mu_expected if mu_expected > 0 else 0.0

        # Maximum single attention weight to any biased key
        max_bias_attention = float(attention_matrix[:, biased_mask].max())

        # Specialisation flag: BAR > 1.5
        specialized_for_bias = bias_attention_ratio > 1.5

        return HeadBiasMetrics(
            layer=layer_idx,
            head=head_idx,
            bias_attention_ratio=float(bias_attention_ratio),
            amplification_score=float(amplification_score),
            max_bias_attention=float(max_bias_attention),
            specialized_for_bias=bool(specialized_for_bias),
        )

    def get_bias_focused_heads(
        self,
        metrics: List[HeadBiasMetrics],
        threshold: float = 1.5
    ) -> List[HeadBiasMetrics]:
        """Get heads that focus significantly on biased tokens.

        Args:
            metrics: List of all head metrics
            threshold: Minimum bias_attention_ratio to be considered focused

        Returns:
            Filtered list of heads with high bias attention
        """
        return [m for m in metrics if m.bias_attention_ratio >= threshold]

    def analyze_bias_propagation(
        self,
        attention_weights: List[torch.Tensor],
        biased_token_indices: List[int],
        tokens: List[str]
    ) -> Dict:
        """Analyze how bias propagates through attention layers.

        Args:
            attention_weights: List of attention tensors
            biased_token_indices: Indices of biased tokens
            tokens: List of tokens

        Returns:
            Dictionary with propagation analysis:
                - layer_propagation: Bias attention ratio per layer
                - peak_layer: Layer with maximum bias focus
                - propagation_pattern: "increasing", "decreasing", or "mixed"
        """
        if not biased_token_indices:
            return {
                "layer_propagation": [],
                "peak_layer": None,
                "propagation_pattern": "none"
            }

        layer_ratios = []

        for layer_idx, layer_attention in enumerate(attention_weights):
            # Average across all heads in this layer
            layer_avg_ratio = []

            num_heads = layer_attention.shape[1]
            for head_idx in range(num_heads):
                head_attn = layer_attention[0, head_idx].cpu().numpy()
                metrics = self._compute_head_metrics(
                    head_attn,
                    set(biased_token_indices),
                    layer_idx,
                    head_idx
                )
                layer_avg_ratio.append(metrics.bias_attention_ratio)

            layer_ratios.append(np.mean(layer_avg_ratio))

        # Determine propagation pattern
        if len(layer_ratios) < 2:
            pattern = "single_layer"
        else:
            # Check if generally increasing or decreasing
            diffs = np.diff(layer_ratios)
            if np.mean(diffs) > 0.1:
                pattern = "increasing"  # Bias gets more attention in later layers
            elif np.mean(diffs) < -0.1:
                pattern = "decreasing"  # Bias gets less attention in later layers
            else:
                pattern = "stable"

        peak_layer = int(np.argmax(layer_ratios)) if layer_ratios else None

        return {
            "layer_propagation": [float(x) for x in layer_ratios],
            "peak_layer": peak_layer,
            "propagation_pattern": pattern,
            "avg_bias_ratio": float(np.mean(layer_ratios)) if layer_ratios else 0.0
        }

    def create_attention_bias_matrix(
        self,
        attention_weights: List[torch.Tensor],
        biased_token_indices: List[int]
    ) -> np.ndarray:
        """Create a matrix showing bias attention for each (layer, head).

        Args:
            attention_weights: List of attention tensors
            biased_token_indices: Indices of biased tokens

        Returns:
            Matrix of shape [num_layers, num_heads] with bias attention ratios
        """
        if not biased_token_indices:
            num_layers = len(attention_weights)
            num_heads = attention_weights[0].shape[1] if num_layers > 0 else 0
            return np.zeros((num_layers, num_heads))

        num_layers = len(attention_weights)
        num_heads = attention_weights[0].shape[1]

        matrix = np.zeros((num_layers, num_heads))

        for layer_idx, layer_attention in enumerate(attention_weights):
            for head_idx in range(num_heads):
                head_attn = layer_attention[0, head_idx].cpu().numpy()
                metrics = self._compute_head_metrics(
                    head_attn,
                    set(biased_token_indices),
                    layer_idx,
                    head_idx
                )
                matrix[layer_idx, head_idx] = metrics.bias_attention_ratio

        return matrix

    def get_bias_influence_on_token(
        self,
        attention_weights: torch.Tensor,
        token_idx: int,
        biased_token_indices: List[int],
        layer_idx: int,
        head_idx: int
    ) -> Dict:
        """Analyze how much bias influences a specific token's representation.

        Args:
            attention_weights: Attention tensor for one layer
            token_idx: Index of token to analyze
            biased_token_indices: Indices of biased tokens
            layer_idx: Layer index
            head_idx: Head index

        Returns:
            Dictionary with influence metrics
        """
        head_attn = attention_weights[0, head_idx].cpu().numpy()

        # How much does this token attend to biased tokens?
        attention_to_bias = sum(
            head_attn[token_idx, bias_idx]
            for bias_idx in biased_token_indices
            if bias_idx < len(head_attn)
        )

        # Total attention this token gives
        total_attention = head_attn[token_idx].sum()

        # Percentage of attention going to biased tokens
        bias_percentage = (attention_to_bias / total_attention * 100) if total_attention > 0 else 0

        return {
            "token_index": token_idx,
            "layer": layer_idx,
            "head": head_idx,
            "attention_to_bias": float(attention_to_bias),
            "total_attention": float(total_attention),
            "bias_percentage": float(bias_percentage),
            "is_bias_influenced": bias_percentage > 30.0  # Threshold: 30% of attention
        }


__all__ = ["AttentionBiasAnalyzer", "HeadBiasMetrics"]
