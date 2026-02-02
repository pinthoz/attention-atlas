"""Bias Detection Module for Attention Atlas.

Neural bias detection via GUS-Net (ethical-spectacle/social-bias-ner)
with attention x bias interaction analysis.
"""

from .gusnet_detector import GusNetDetector
from .attention_bias import AttentionBiasAnalyzer, HeadBiasMetrics
from .visualizations import (
    create_token_bias_heatmap,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_ratio_formula_html,
    create_bias_criteria_html,
    create_bias_sentence_preview,
    create_token_bias_strip,
)

__all__ = [
    # Detectors
    "GusNetDetector",
    "AttentionBiasAnalyzer",
    # Data classes
    "HeadBiasMetrics",
    # Visualizations
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_ratio_formula_html",
    "create_bias_criteria_html",
    "create_bias_sentence_preview",
    "create_token_bias_strip",
]
