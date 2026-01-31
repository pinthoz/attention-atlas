"""Bias Detection Module for Attention Atlas.

This module provides comprehensive bias analysis for transformer models:
- Neural bias detection via GUS-Net (ethical-spectacle/social-bias-ner)
- Rule-based token-level bias detection (lexicon)
- Attention x Bias interaction analysis
"""

from .token_detector import TokenBiasDetector, BiasSpan
from .gusnet_detector import GusNetDetector
from .attention_bias import AttentionBiasAnalyzer, HeadBiasMetrics
from .visualizations import (
    create_token_bias_heatmap,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_bias_criteria_html,
)

__all__ = [
    # Detectors
    "TokenBiasDetector",
    "GusNetDetector",
    "AttentionBiasAnalyzer",
    # Data classes
    "BiasSpan",
    "HeadBiasMetrics",
    # Visualizations
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_bias_criteria_html",
]
