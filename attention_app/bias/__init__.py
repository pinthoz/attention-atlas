"""Bias Detection Module for Attention Atlas.

This module provides comprehensive bias analysis for transformer models:
- Token-level bias detection (GUS-Net approach)
- Attention × Bias interaction analysis
"""

from .token_detector import TokenBiasDetector, BiasSpan
from .attention_bias import AttentionBiasAnalyzer, HeadBiasMetrics
from .visualizations import (
    create_token_bias_heatmap,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization
)

__all__ = [
    # Analyzers
    "TokenBiasDetector",
    "AttentionBiasAnalyzer",
    # Data classes
    "BiasSpan",
    "HeadBiasMetrics",
    # Visualizations
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
]
