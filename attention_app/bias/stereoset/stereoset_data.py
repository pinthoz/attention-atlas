"""Data loader for pre-computed StereoSet evaluation results.

Reads model-specific JSON files and caches them so the dashboard can
access scores, examples, and head sensitivity data without reloading.

Files:
    stereoset_precomputed_bert.json  — BERT PLL scoring
    stereoset_precomputed_gpt2.json  — GPT-2 autoregressive scoring
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

_DATA_DIR = Path(__file__).parent

# Per-model cache
_cache: Dict[str, dict] = {}

# Map base model names and GUS-Net keys to our short model key
_MODEL_KEY_MAP = {
    # Short keys
    "bert": "bert",
    "bert_large": "bert_large",
    "gpt2": "gpt2",
    "gpt2_medium": "gpt2_medium",
    # Full HuggingFace model names
    "bert-base-uncased": "bert",
    "bert-large-uncased": "bert_large",
    "gpt2-medium": "gpt2_medium",
    # GUS-Net model keys
    "gusnet-bert": "bert",
    "gusnet-bert-large": "bert_large",
    "gusnet-bert-custom": "bert",
    "gusnet-ensemble": "bert",
    "gusnet-gpt2": "gpt2",
    "gusnet-gpt2-medium": "gpt2_medium",
}

_FILE_MAP = {
    "bert": "stereoset_precomputed_bert.json",
    "bert_large": "stereoset_precomputed_bert_large.json",
    "gpt2": "stereoset_precomputed_gpt2.json",
    "gpt2_medium": "stereoset_precomputed_gpt2_medium.json",
}


def _resolve_key(model: Optional[str]) -> str:
    """Resolve any model identifier to 'bert' or 'gpt2'."""
    if model is None:
        return "bert"
    return _MODEL_KEY_MAP.get(model, "bert")


def load_stereoset_data(model: Optional[str] = None) -> Optional[dict]:
    """Load and cache the pre-computed StereoSet JSON for a model.

    Parameters
    ----------
    model : str or None
        Any model identifier: 'bert', 'gpt2', a HuggingFace model name,
        or a GUS-Net key.  Defaults to 'bert'.

    Returns None if the file does not exist (script hasn't been run yet).
    """
    key = _resolve_key(model)

    if key in _cache:
        return _cache[key]

    filepath = _DATA_DIR / _FILE_MAP.get(key, _FILE_MAP["bert"])
    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        _cache[key] = json.load(f)
    return _cache[key]


def get_stereoset_scores(model: Optional[str] = None) -> Optional[Dict]:
    """Return overall + per-category SS/LMS/ICAT scores."""
    data = load_stereoset_data(model)
    if data is None:
        return None
    return data.get("scores")


def get_stereoset_examples(
    model: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict]:
    """Return StereoSet examples, optionally filtered by category."""
    data = load_stereoset_data(model)
    if data is None:
        return []
    examples = data.get("examples", [])
    if category:
        examples = [e for e in examples if e["category"] == category]
    return examples


def get_head_sensitivity_matrix(model: Optional[str] = None) -> Optional[List[List[float]]]:
    """Return the 12×12 head sensitivity variance matrix."""
    data = load_stereoset_data(model)
    if data is None:
        return None
    return data.get("head_sensitivity_matrix")


def get_sensitive_heads(model: Optional[str] = None) -> List[Dict]:
    """Return the top-10 most sensitive heads."""
    data = load_stereoset_data(model)
    if data is None:
        return []
    return data.get("sensitive_heads", [])


def get_top_features(model: Optional[str] = None) -> List[Dict]:
    """Return top-20 most discriminative features by Kruskal-Wallis p-value."""
    data = load_stereoset_data(model)
    if data is None:
        return []
    return data.get("top_features", [])


def get_metadata(model: Optional[str] = None) -> Optional[Dict]:
    """Return metadata dict (model, date, counts)."""
    data = load_stereoset_data(model)
    if data is None:
        return None
    return data.get("metadata")
