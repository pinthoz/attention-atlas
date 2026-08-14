"""JSON-safe serialisation helpers for the HTTP API.

Pure functions: they take *pieces* of a
:class:`~attention_app.server.logic.ComputeResult` and return primitives that
``json.dumps`` accepts. Nothing here touches Shiny, and nothing here computes
metrics - the numbers come from :mod:`attention_app.metrics`.

Never serialise a ComputeResult wholesale (``asdict``/``dict(result)``): the
dataclass exposes ``tokenizer`` / ``encoder_model`` / ``mlm_model`` as
properties that resolve real model instances through ModelManager, so a
whole-object dump would drag the model weights into the response. Serialise
field by field with the helpers below.

Two things every helper handles, because the pipeline produces both:

* numpy scalars (``np.float32``, ``np.int64``, ``np.bool_``) - not JSON
  serialisable, and not instances of the Python builtins either;
* torch tensors - moved to CPU and converted before any conversion.

Floats are rounded to :data:`ROUND_DP` decimals to keep payloads small, and
non-finite values (NaN / ±inf) become ``None`` rather than the bare ``NaN``
token that ``json.dumps`` would otherwise emit, which is not valid JSON.
"""

import math

import numpy as np
import torch

#: Decimal places kept for every float in an API payload.
ROUND_DP = 5


def to_float(value, ndigits: int = ROUND_DP):
    """Convert a numeric scalar to a rounded Python float.

    Accepts Python numbers, numpy scalars and 0-d tensors/arrays. Returns
    ``None`` for ``None`` and for non-finite values (NaN, ±inf), which are
    not representable in strict JSON.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim != 0:
            raise ValueError("to_float expects a scalar, got a tensor of "
                             f"shape {tuple(value.shape)}")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("to_float expects a scalar, got an array of "
                             f"shape {value.shape}")
        value = value.item()

    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, ndigits)


def to_builtin(value, ndigits: int = ROUND_DP):
    """Recursively convert a value into JSON-safe builtins.

    Tensors and arrays become nested lists, numpy scalars become Python
    scalars, floats are rounded, dict keys are stringified (JSON object keys
    must be strings; the pipeline uses int layer/head keys). Anything not
    recognised falls back to ``str(value)`` so a stray object can never break
    a response.
    """
    if value is None:
        return None
    # bool before int: bool is a subclass of int, and np.bool_ is np.generic.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, torch.Tensor):
        return to_builtin(value.detach().cpu().numpy(), ndigits)
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist(), ndigits)
    if isinstance(value, np.generic):
        return to_builtin(value.item(), ndigits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return to_float(value, ndigits)
    if isinstance(value, dict):
        return {str(k): to_builtin(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v, ndigits) for v in value]
    return str(value)


def serialize_tokens(tokens):
    """Token strings as a plain list of ``str``."""
    if tokens is None:
        return []
    return [str(t) for t in tokens]


def serialize_matrix(matrix, ndigits: int = ROUND_DP):
    """A 2-D attention matrix as a list of lists of rounded floats.

    Accepts a torch tensor or a numpy array. Non-finite cells become ``None``.
    """
    if matrix is None:
        return None
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    rounded = np.round(matrix.astype(np.float64), ndigits)
    finite = np.isfinite(rounded)
    return [
        [float(v) if ok else None for v, ok in zip(row, row_ok)]
        for row, row_ok in zip(rounded.tolist(), finite.tolist())
    ]


def serialize_metrics(metrics, ndigits: int = ROUND_DP):
    """One metric dict from ``compute_all_attention_metrics``.

    Values are already Python floats there, but ``balance`` is ``None`` for
    models without a ``[CLS]`` token - that ``None`` is preserved so the
    frontend can render "N/A" instead of a misleading zero.
    """
    if metrics is None:
        return None
    return {str(k): to_float(v, ndigits) for k, v in metrics.items()}


def serialize_metrics_grid(grid, ndigits: int = ROUND_DP):
    """A ``[layer][head]`` nested list of metric dicts."""
    return [[serialize_metrics(m, ndigits) for m in layer] for layer in grid]


def serialize_segments(inputs):
    """``token_type_ids`` as a flat list of ints, or ``None``.

    BERT-style tokenizers emit segment ids (0 for sentence A, 1 for sentence
    B); GPT-2 does not produce the key at all, in which case ``None`` is
    returned and the caller should omit the field.
    """
    if not inputs:
        return None
    segments = inputs.get("token_type_ids")
    if segments is None:
        return None
    if isinstance(segments, torch.Tensor):
        segments = segments.detach().cpu().numpy()
    segments = np.asarray(segments)
    if segments.ndim > 1:  # (batch, seq) -> first (only) sequence
        segments = segments[0]
    return [int(v) for v in segments.tolist()]


def serialize_head_specialization(head_specialization, ndigits: int = ROUND_DP):
    """``{layer: {head: metrics}}`` with stringified keys and rounded values."""
    if not head_specialization:
        return None
    return {
        str(layer_idx): {
            str(head_idx): {str(k): to_float(v, ndigits) for k, v in metrics.items()}
            for head_idx, metrics in heads.items()
        }
        for layer_idx, heads in head_specialization.items()
    }


def serialize_clusters(head_clusters, ndigits: int = ROUND_DP):
    """The t-SNE + K-Means cluster list from ``compute_head_clusters``.

    Each entry is ``{'layer', 'head', 'x', 'y', 'cluster', 'metrics', ...}``;
    the values are a mix of numpy scalars and Python primitives, so the whole
    list goes through :func:`to_builtin`.
    """
    if not head_clusters:
        return []
    return [to_builtin(entry, ndigits) for entry in head_clusters]


def serialize_isa(isa_data, ndigits: int = ROUND_DP):
    """The ISA dict, with its sentence-attention matrix as nested lists."""
    if not isa_data:
        return None
    payload = {
        "sentence_texts": [str(s) for s in isa_data.get("sentence_texts", [])],
        "sentence_boundaries_ids": [
            int(b) for b in isa_data.get("sentence_boundaries_ids", [])
        ],
        "model_type": isa_data.get("model_type"),
        "is_causal": bool(isa_data.get("is_causal", False)),
        "aggregation_method": isa_data.get("aggregation_method"),
    }
    matrix = isa_data.get("sentence_attention_matrix")
    payload["sentence_attention_matrix"] = (
        serialize_matrix(matrix, ndigits) if matrix is not None else None
    )
    return payload


__all__ = [
    "ROUND_DP",
    "to_float",
    "to_builtin",
    "serialize_tokens",
    "serialize_matrix",
    "serialize_metrics",
    "serialize_metrics_grid",
    "serialize_segments",
    "serialize_head_specialization",
    "serialize_clusters",
    "serialize_isa",
]
