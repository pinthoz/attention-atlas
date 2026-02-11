"""Bias Detection Module for Attention Atlas.

Neural bias detection via GUS-Net (pinthoz/gus-net-*)
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
    create_stereoset_overview_html,
    create_stereoset_category_chart,
    create_stereoset_head_sensitivity_heatmap,
    create_stereoset_bias_distribution,
    create_stereoset_demographic_chart,
    create_stereoset_example_html,
)
from .stereoset import (
    load_stereoset_data,
    get_stereoset_scores,
    get_stereoset_examples,
    get_head_sensitivity_matrix,
    get_sensitive_heads,
    get_top_features,
    get_metadata,
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
    "create_stereoset_overview_html",
    "create_stereoset_category_chart",
    "create_stereoset_head_sensitivity_heatmap",
    "create_stereoset_bias_distribution",
    "create_stereoset_demographic_chart",
    "create_stereoset_example_html",
    # StereoSet data access
    "load_stereoset_data",
    "get_stereoset_scores",
    "get_stereoset_examples",
    "get_head_sensitivity_matrix",
    "get_sensitive_heads",
    "get_top_features",
    "get_metadata",
]
