"""StereoSet evaluation subpackage.

Pre-computed benchmark data and loader utilities for measuring
stereotypical bias in language models via the StereoSet intersentence task.
"""

from .stereoset_data import (
    load_stereoset_data,
    get_stereoset_scores,
    get_stereoset_examples,
    get_head_sensitivity_matrix,
    get_sensitive_heads,
    get_top_features,
    get_head_profile_stats,
    get_metadata,
)

__all__ = [
    "load_stereoset_data",
    "get_stereoset_scores",
    "get_stereoset_examples",
    "get_head_sensitivity_matrix",
    "get_sensitive_heads",
    "get_top_features",
    "get_head_profile_stats",
    "get_metadata",
]
