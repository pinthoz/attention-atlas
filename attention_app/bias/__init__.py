"""Bias Detection Module for Attention Atlas.

Neural bias detection via GUS-Net (ethical-spectacle/social-bias-ner)
with attention x bias interaction analysis.
"""

from .gusnet_detector import GusNetDetector, EnsembleGusNetDetector, MODEL_REGISTRY
from .attention_bias import AttentionBiasAnalyzer, HeadBiasMetrics
from .head_ablation import HeadAblationResult, batch_ablate_top_heads
from .integrated_gradients import IGCorrelationResult, IGAnalysisBundle, batch_compute_ig_correlation
from .visualizations import (
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_ratio_formula_html,
    create_bias_criteria_html,
    create_bias_sentence_preview,
    create_token_bias_strip,
    create_confidence_breakdown,
    create_ablation_impact_chart,
    create_ig_correlation_chart,
    create_ig_token_comparison_chart,
    create_ig_distribution_chart,
    create_ig_layer_summary_chart,
)

__all__ = [
    # Detectors
    "GusNetDetector",
    "EnsembleGusNetDetector",
    "MODEL_REGISTRY",
    "AttentionBiasAnalyzer",
    # Data classes
    "HeadBiasMetrics",
    "HeadAblationResult",
    "IGCorrelationResult",
    "IGAnalysisBundle",
    # Ablation
    "batch_ablate_top_heads",
    # Integrated Gradients
    "batch_compute_ig_correlation",
    # Visualizations
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_ratio_formula_html",
    "create_bias_criteria_html",
    "create_bias_sentence_preview",
    "create_token_bias_strip",
    "create_confidence_breakdown",
    "create_ablation_impact_chart",
    "create_ig_correlation_chart",
    "create_ig_token_comparison_chart",
    "create_ig_distribution_chart",
    "create_ig_layer_summary_chart",
]
