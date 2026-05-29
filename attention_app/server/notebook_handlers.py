"""Server-side reactive logic for the Auditor Notebook (hybrid construct).

The Auditor Notebook in this app is a **hybrid** construct in the sense
of Chapter 6.5 of the thesis:

- Three free-text fields capture analyst judgment that the system cannot
  observe: ``hypothesis``, ``uncertainty`` acknowledged, and ``next
  steps``. An optional ``title`` is also kept.

- The remaining audit context (which model was running, which prompt was
  being inspected, which layer/head was selected, which thresholds were
  active, which tokens were highlighted, etc.) is **captured
  automatically** from the dashboard's reactive inputs at the moment
  ``Add entry`` is clicked. Auditors do not type this; the system records
  it as structured JSON.

- Each saved entry can be **restored**: clicking the restore icon on an
  entry calls ``ui.update_*`` for every captured input, snapping the
  dashboard back to the exact state the analyst was inspecting. This is
  the reconstructability requirement that ISO/IEC 42001 control
  evidence and Articles 12/17 of the EU AI Act expect.

The module also handles:

- Persistence to ``downloads/sessions/auditor_notebook.json`` so that
  entries survive an app restart.
- Markdown and JSON exports that include the captured context.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from shiny import reactive, render, ui


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
_NOTEBOOK_PATH = Path("downloads") / "sessions" / "auditor_notebook.json"

# Three free-text fields plus an optional title.
_REQUIRED_FIELDS = ("hypothesis", "uncertainty", "next_steps")
_FIELD_LABELS = {
    "hypothesis": "Hypothesis",
    "uncertainty": "Uncertainty acknowledged",
    "next_steps": "Next steps",
}

# Inputs whose value we try to capture in the structured context block.
# Each tuple is (input_id, json_key, human_label).
_CONTEXT_INPUTS = [
    # Navigation
    ("main_navbar",              "active_tab",             "Active tab"),
    # Attention tab
    ("model_family",             "att_model_family_a",     "Model family A"),
    ("model_name",               "att_model_name_a",       "Model A"),
    ("model_family_B",           "att_model_family_b",     "Model family B"),
    ("model_name_B",             "att_model_name_b",       "Model B"),
    ("text_input",               "att_prompt_a",           "Prompt A"),
    ("text_input_B",             "att_prompt_b",           "Prompt B"),
    ("compare_mode",             "att_compare_models",     "Compare models"),
    ("compare_prompts_mode",     "att_compare_prompts",    "Compare prompts"),
    ("view_mode",                "att_view_mode",          "Attention view"),
    ("global_layer",             "att_layer",              "Layer"),
    ("global_head",              "att_head",               "Head"),
    ("global_focus_token",       "att_focus_token",        "Focus token"),
    # Bias tab
    ("bias_model_key",           "bias_model_a",           "Bias model A"),
    ("bias_model_key_B",         "bias_model_b",           "Bias model B"),
    ("bias_input_text",          "bias_prompt_a",          "Bias prompt A"),
    ("bias_input_text_b",        "bias_prompt_b",          "Bias prompt B"),
    ("bias_compare_mode",        "bias_compare_models",    "Bias compare models"),
    ("bias_compare_prompts_mode","bias_compare_prompts",   "Bias compare prompts"),
    ("bias_active_prompt_tab",   "bias_active_prompt",     "Bias active prompt"),
    ("bias_thresh_unfair",       "bias_thresh_unfair_a",   "Threshold (unfair, A)"),
    ("bias_thresh_gen",          "bias_thresh_gen_a",      "Threshold (generalisation, A)"),
    ("bias_thresh_stereo",       "bias_thresh_stereo_a",   "Threshold (stereotype, A)"),
    ("bias_thresh_unfair_b",     "bias_thresh_unfair_b",   "Threshold (unfair, B)"),
    ("bias_thresh_gen_b",        "bias_thresh_gen_b",      "Threshold (generalisation, B)"),
    ("bias_thresh_stereo_b",     "bias_thresh_stereo_b",   "Threshold (stereotype, B)"),
    ("bias_selected_tokens_A",   "bias_selected_tokens_a", "Selected tokens (A)"),
    ("bias_selected_tokens_B",   "bias_selected_tokens_b", "Selected tokens (B)"),
]

# Reverse map: json key -> input id (used by the restore handler).
_RESTORE_MAP = {jk: iid for iid, jk, _ in _CONTEXT_INPUTS}

# json_key -> human label for the *input* state. Metric labels are
# appended below once the metric registry is defined.
_INPUT_LABELS = {jk: lbl for _, jk, lbl in _CONTEXT_INPUTS}

# Inputs that are switches/booleans rather than text.
_SWITCH_INPUTS = {
    "att_compare_models",
    "att_compare_prompts",
    "bias_compare_models",
    "bias_compare_prompts",
}
# Inputs that are numeric (sliders/numeric).
_NUMERIC_INPUTS = {
    "bias_thresh_unfair_a", "bias_thresh_gen_a", "bias_thresh_stereo_a",
    "bias_thresh_unfair_b", "bias_thresh_gen_b", "bias_thresh_stereo_b",
}
_RADIO_INPUTS = {"att_view_mode"}
_SELECT_INPUTS = {
    "att_model_family_a", "att_model_name_a",
    "att_model_family_b", "att_model_name_b",
    "bias_model_a", "bias_model_b",
    "bias_active_prompt",
}
_TEXT_AREA_INPUTS = {
    "att_prompt_a", "att_prompt_b",
    "bias_prompt_a", "bias_prompt_b",
}
_NAVSET_INPUTS = {"active_tab"}

# These inputs are maintained by bespoke JavaScript controls. They can be
# restored by pushing a Shiny input event back to the browser, but they do not
# have a native Shiny update_* widget to target.
_CLIENT_EVENT_INPUTS = {
    "att_layer", "att_head", "att_focus_token",
    "bias_selected_tokens_a", "bias_selected_tokens_b",
}

# Computed-metric keys, grouped by side. These are *outputs* of the
# analysis (not user inputs), so they are not restorable — only readable
# evidence.
_METRIC_LABELS: Dict[str, str]
_METRIC_LABELS = {
    # ── Attention structural & per-head ──────────────────────────────
    "att_metric_n_tokens":           "Tokens",
    "att_metric_n_layers":           "Layers",
    "att_metric_n_heads":            "Heads per layer",
    "att_metric_head_entropy":       "Selected-head entropy",
    "att_metric_head_max_attn":      "Selected-head max attention",
    "att_metric_head_top_pair":      "Selected-head peak src → tgt",
    "att_metric_head_top_received":  "Top-5 tokens receiving attention (sel. head)",
    "att_metric_head_top_attending": "Top-5 tokens distributing attention (sel. head)",
    "att_metric_head_special_mass":  "Attention mass on [CLS]/[SEP]/[PAD] (sel. head)",
    "att_metric_head_specialization": "Selected-head specialisation",
    # ── Attention global (across all layers/heads) ───────────────────
    "att_metric_global_top_attended": "Top-5 tokens receiving attention (global)",
    "att_metric_global_concentrated": "Top-5 most-concentrated heads (lowest entropy)",
    "att_metric_layer_entropy":       "Mean entropy per layer",
    # ── Bias detection (side A) ──────────────────────────────────────
    "bias_metric_n_tokens":          "Bias tokens analysed (A)",
    "bias_metric_n_biased":          "Biased tokens (A)",
    "bias_metric_biased_preview":    "Biased tokens preview (A)",
    "bias_metric_biased_scored":     "Top biased tokens with scores (A)",
    "bias_metric_type_counts":       "Bias type counts (A)",
    "bias_metric_strongest_token":   "Strongest biased token (A)",
    "bias_metric_mean_confidence":   "Mean confidence over biased tokens (A)",
    "bias_metric_bias_spans":        "Biased spans (A)",
    "bias_metric_peak_layer":        "Peak propagation layer (A)",
    "bias_metric_propagation":       "Propagation pattern (A)",
    "bias_metric_matrix_peak_head":  "Peak head in bias-attention matrix (A)",
    # ── Integrated Gradients (A) ─────────────────────────────────────
    "bias_metric_ig_top_tokens":          "IG top-5 tokens (A)",
    "bias_metric_ig_top_corr_heads":      "Top-3 heads by IG ↔ attention ρ (A)",
    "bias_metric_ig_mean_spearman":       "Mean IG ↔ attention Spearman ρ (A)",
    # ── Head ablation (A) ─────────────────────────────────────────────
    "bias_metric_ablation_top_heads":     "Top-3 heads by ablation impact (A)",
    "bias_metric_ablation_max_kl":        "Max KL divergence after ablation (A)",
    "bias_metric_ablation_mean_impact":   "Mean representation impact (A)",
    # ── Perturbation (A) ─────────────────────────────────────────────
    "bias_metric_perturb_top_tokens":     "Perturbation top-5 tokens (A)",
    "bias_metric_perturb_vs_ig":          "Perturbation ↔ IG Spearman ρ (A)",
    # ── LRP (A) ──────────────────────────────────────────────────────
    "bias_metric_lrp_top_tokens":         "LRP top-5 tokens (A)",
    "bias_metric_lrp_vs_ig":              "LRP ↔ IG Spearman ρ (A)",
    # ── Cross-method consistency ─────────────────────────────────────
    "bias_metric_methods_top1_agree":     "Top-1 token agreement across methods",
    # ── B side (only when bias compare is active) ────────────────────
    "bias_metric_n_biased_b":             "Biased tokens (B)",
    "bias_metric_biased_scored_b":        "Top biased tokens with scores (B)",
    "bias_metric_type_counts_b":          "Bias type counts (B)",
    "bias_metric_peak_layer_b":           "Peak propagation layer (B)",
    "bias_metric_strongest_token_b":      "Strongest biased token (B)",
    # ── A ↔ B deltas (counterfactual / cross-model contrast) ─────────
    "bias_metric_delta_n_biased":          "Δ Biased tokens (A → B)",
    "bias_metric_delta_strongest_score":   "Δ Strongest-token score (A → B)",
    "bias_metric_delta_mean_confidence":   "Δ Mean confidence (A → B)",
    "bias_metric_delta_type_counts":       "Δ Bias type counts (A → B)",
    "bias_metric_delta_peak_layer":        "Δ Peak propagation layer (A → B)",
    "bias_metric_delta_token_overlap":     "Biased-token overlap (A ∩ B)",
    "bias_metric_delta_unique_to_a":       "Biased only in A",
    "bias_metric_delta_unique_to_b":       "Biased only in B",
    # ── Model identity (ISO 42001 traceability) ──────────────────────
    "att_metric_model_id":                 "Model checkpoint (A)",
    "att_metric_model_id_b":               "Model checkpoint (B)",
    "att_metric_transformers_version":     "transformers library version",
    "att_metric_model_param_hash":         "Model parameter hash (A, sha256[:16])",
    "att_metric_model_param_hash_b":       "Model parameter hash (B, sha256[:16])",
    "bias_metric_model_id":                "Bias model checkpoint (A)",
    "bias_metric_model_id_b":              "Bias model checkpoint (B)",
}

# Combined label map used by the renderer/markdown: inputs first (so
# they appear first in the rendered block), then computed metrics.
_CONTEXT_LABELS: Dict[str, str] = {**_INPUT_LABELS, **_METRIC_LABELS}

_PROVENANCE_KEYS = (
    "att_metric_model_id",
    "att_metric_model_id_b",
    "att_metric_transformers_version",
    "att_metric_model_param_hash",
    "att_metric_model_param_hash_b",
    "bias_metric_model_id",
    "bias_metric_model_id_b",
)


def _disconfirming_keys(ctx: Dict[str, Any]) -> set:
    """Identify keys whose value implies *disconfirming* evidence.

    DR7 in the thesis requires the tool to actively surface signals
    that contradict the analyst's working hypothesis rather than only
    confirmatory ones. This function returns the set of context keys
    that the renderer should highlight (e.g. red row) so the auditor
    cannot miss them while filling in the Notebook.

    Triggers:
    - Cross-method top-1 disagreement (IG/LRP/Perturbation pick
      different tokens as the most salient).
    - Mean IG ↔ attention Spearman ρ below ~0.30 (attention is poorly
      aligned with gradient attribution).
    - Perturbation ↔ IG Spearman ρ below ~0.30.
    - LRP ↔ IG Spearman ρ below ~0.30.
    - The selected head sends > 60 % of its attention to special
      tokens ([CLS]/[SEP]/[PAD]) — a known "attention sink" failure
      mode that undermines any claim that the head encodes linguistic
      content.
    """
    flagged: set = set()
    agree = ctx.get("bias_metric_methods_top1_agree")
    if isinstance(agree, str) and not agree.lower().startswith("all"):
        flagged.add("bias_metric_methods_top1_agree")
    for key in (
        "bias_metric_ig_mean_spearman",
        "bias_metric_perturb_vs_ig",
        "bias_metric_lrp_vs_ig",
    ):
        v = ctx.get(key)
        try:
            if v is not None and abs(float(v)) < 0.3:
                flagged.add(key)
        except Exception:
            pass
    sm = ctx.get("att_metric_head_special_mass")
    try:
        if sm is not None and float(sm) > 0.6:
            flagged.add("att_metric_head_special_mass")
    except Exception:
        pass
    return flagged


def _load_entries() -> List[Dict[str, Any]]:
    try:
        if _NOTEBOOK_PATH.is_file():
            with _NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
    except Exception:
        _logger.exception("Could not load Auditor Notebook entries from disk")
    return []


def _save_entries(entries: List[Dict[str, Any]]) -> None:
    try:
        _NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception:
        _logger.exception("Could not save Auditor Notebook entries to disk")


# ---------------------------------------------------------------------------
# Capture / restore helpers
# ---------------------------------------------------------------------------

def _safe_read(input, input_id: str) -> Any:
    """Read ``input.<input_id>()`` without crashing on missing/silent inputs."""
    try:
        accessor = getattr(input, input_id, None)
        if accessor is None:
            return None
        value = accessor()
        # Some Shiny inputs return empty strings instead of None.
        if value == "":
            return None
        return value
    except Exception:
        return None


def _capture_context(input, *, cached_result=None, cached_result_B=None,
                     bias_state=None) -> Dict[str, Any]:
    """Snapshot the current dashboard state into a flat dict.

    Captures only what is *relevant* to the current run:

    - Attention ``B`` fields (model B, prompt B) are dropped if neither
      ``compare_mode`` nor ``compare_prompts_mode`` is active.
    - Bias ``B`` fields (model B, prompt B, thresholds B, selected
      tokens B) are dropped if neither ``bias_compare_mode`` nor
      ``bias_compare_prompts_mode`` is active.
    - The flag fields themselves (``compare_models``, ``compare_prompts``)
      are kept only when they are actually ``True``; a flag set to
      ``False`` is noise, not signal, and would clutter the audit trail.
    - ``bias_active_prompt`` is only meaningful in compare-prompts mode.
    """
    raw: Dict[str, Any] = {}
    for input_id, json_key, _label in _CONTEXT_INPUTS:
        value = _safe_read(input, input_id)
        if value is None:
            continue
        if isinstance(value, (tuple, set)):
            value = list(value)
        raw[json_key] = value

    att_compare_active = bool(
        raw.get("att_compare_models") or raw.get("att_compare_prompts")
    )
    bias_compare_active = bool(
        raw.get("bias_compare_models") or raw.get("bias_compare_prompts")
    )

    # Fields to drop entirely when their compare mode is off.
    att_b_only = {
        "att_model_family_b", "att_model_name_b", "att_prompt_b",
    }
    bias_b_only = {
        "bias_model_b", "bias_prompt_b",
        "bias_thresh_unfair_b", "bias_thresh_gen_b", "bias_thresh_stereo_b",
        "bias_selected_tokens_b",
    }

    context: Dict[str, Any] = {}
    for json_key, value in raw.items():
        # Hide compare flags when not active.
        if json_key in ("att_compare_models", "att_compare_prompts",
                        "bias_compare_models", "bias_compare_prompts"):
            if value:
                context[json_key] = value
            continue
        # Hide B-fields when their compare context is inactive.
        if json_key in att_b_only and not att_compare_active:
            continue
        if json_key in bias_b_only and not bias_compare_active:
            continue
        # bias_active_prompt only makes sense in compare-prompts mode.
        if json_key == "bias_active_prompt" and not raw.get("bias_compare_prompts"):
            continue
        context[json_key] = value

    # ── Computed-metric block (post-run evidence) ─────────────────
    cr = _safe_get(cached_result) if cached_result is not None else None
    att_metrics = _extract_attention_metrics(
        cr, raw.get("att_layer"), raw.get("att_head")
    )
    context.update(att_metrics)

    # Model B identity (only when compare-models is active for attention).
    if att_compare_active and cached_result_B is not None:
        cr_b = _safe_get(cached_result_B)
        ident_b = _model_identity(cr_b)
        if "name_or_path" in ident_b:
            context["att_metric_model_id_b"] = ident_b["name_or_path"]
        if "param_hash" in ident_b:
            context["att_metric_model_param_hash_b"] = ident_b["param_hash"]

    bias_metrics = _extract_bias_metrics(bias_state, bias_compare_active)
    context.update(bias_metrics)

    # Bias model identity (sources A and B). The bias_results dict stores
    # the model key; the HF checkpoint id is the same as the key for
    # gusnet-* models. Capture both sides when compare is active.
    if bias_state is not None:
        br_a = _safe_get(bias_state.get("bias_results"))
        if isinstance(br_a, dict) and br_a.get("bias_model_key"):
            context["bias_metric_model_id"] = br_a["bias_model_key"]
        if bias_compare_active:
            br_b = _safe_get(bias_state.get("bias_results_B"))
            if isinstance(br_b, dict) and br_b.get("bias_model_key"):
                context["bias_metric_model_id_b"] = br_b["bias_model_key"]

    return context


def _safe_get(reactive_value):
    """Read a ``reactive.value`` (or anything ``.get``-able) defensively."""
    if reactive_value is None:
        return None
    try:
        return reactive_value.get()
    except Exception:
        return None


def _model_identity(cached_result) -> Dict[str, Any]:
    """Pull a stable model identity from a ``ComputeResult``.

    Captures:
    - ``_name_or_path``: the Hugging Face id (e.g. ``bert-base-uncased``)
    - ``transformers_version`` if the config recorded it at load time
    - a short sha256 over a small slice of weights as a fingerprint

    The fingerprint is deterministic per checkpoint and lets a reviewer
    detect, after the fact, if the audit was re-run against a different
    weight file even with the same nominal model name.
    """
    out: Dict[str, Any] = {}
    if cached_result is None:
        return out
    model = getattr(cached_result, "encoder_model", None)
    if model is None:
        return out
    # _name_or_path + transformers_version from config.
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            nm = getattr(cfg, "_name_or_path", None)
            if nm:
                out["name_or_path"] = str(nm)
            tv = getattr(cfg, "transformers_version", None)
            if tv:
                out["transformers_version"] = str(tv)
    except Exception:
        pass
    # Fingerprint: sha256 over a small deterministic slice of weights.
    try:
        import hashlib
        h = hashlib.sha256()
        # Take the first few tensors in parameter order, flatten the
        # first 4096 floats of each to keep the cost bounded.
        for i, (_, p) in enumerate(model.named_parameters()):
            if i >= 4:
                break
            try:
                arr = p.detach().cpu().numpy().ravel()[:4096].tobytes()
                h.update(arr)
            except Exception:
                continue
        digest = h.hexdigest()
        if digest:
            out["param_hash"] = digest[:16]
    except Exception:
        pass
    return out


def _top_k_indices(scores, k=5, descending=True):
    """Return the indices of the top-K scores. Defensive against shape issues."""
    try:
        import numpy as _np
        arr = _np.asarray(scores)
        if arr.size == 0:
            return []
        if descending:
            order = _np.argsort(-arr)
        else:
            order = _np.argsort(arr)
        return [int(i) for i in order[:k]]
    except Exception:
        return []


def _extract_attention_metrics(cached_result, layer_idx, head_idx) -> Dict[str, Any]:
    """Pull a comprehensive set of attention numbers from a ``ComputeResult``.

    Covers: structural facts (#tokens/layers/heads), selected-head numbers
    (entropy, max, top src→tgt, top-K received/distributed, special-token
    mass), global facts (top-K attended across all heads, top-K most
    concentrated heads, per-layer mean entropy), and the linguistic
    specialisation of the selected head when available.
    """
    out: Dict[str, Any] = {}
    if cached_result is None:
        return out
    # Model identity is independent of whether attention tensors are
    # present, so extract it first.
    ident = _model_identity(cached_result)
    if "name_or_path" in ident:
        out["att_metric_model_id"] = ident["name_or_path"]
    if "transformers_version" in ident:
        out["att_metric_transformers_version"] = ident["transformers_version"]
    if "param_hash" in ident:
        out["att_metric_model_param_hash"] = ident["param_hash"]
    try:
        import numpy as _np
    except Exception:
        return out
    try:
        tokens = list(getattr(cached_result, "tokens", []) or [])
        n_tok = len(tokens)
        out["att_metric_n_tokens"] = n_tok
        attentions = getattr(cached_result, "attentions", None)
        if attentions is None or len(attentions) == 0 or n_tok == 0:
            return out
        n_layers = len(attentions)
        out["att_metric_n_layers"] = n_layers
        try:
            n_heads = int(attentions[0].shape[1])
            out["att_metric_n_heads"] = n_heads
        except Exception:
            n_heads = None

        # ── Per-layer mean entropy (used for global summary too) ────
        layer_entropies = []
        # Accumulators for global attended/concentrated stats.
        global_col_sums = _np.zeros(n_tok, dtype=float)
        head_entropies: list = []  # (mean_entropy, layer, head)
        for li, lt in enumerate(attentions):
            try:
                # shape (batch, heads, seq, seq) → batch 0
                arr = lt[0].detach().cpu().numpy()
            except Exception:
                continue
            # arr shape: (heads, seq, seq) — trim to first n_tok rows/cols.
            arr = arr[:, :n_tok, :n_tok]
            if arr.size == 0:
                continue
            # Per-head entropy (mean over queries).
            p = arr + 1e-12
            ent = -(p * _np.log(p)).sum(axis=-1).mean(axis=-1)  # (heads,)
            layer_entropies.append(round(float(ent.mean()), 4))
            for hi_ in range(arr.shape[0]):
                head_entropies.append(
                    (round(float(ent[hi_]), 4), li, hi_)
                )
            # Aggregate column sums for global "top attended" tokens.
            global_col_sums += arr.sum(axis=(0, 1))

        if layer_entropies:
            # Report only top 5 (lowest entropy = most concentrated layers).
            ranked_layers = sorted(
                enumerate(layer_entropies), key=lambda x: x[1]
            )[:5]
            out["att_metric_layer_entropy"] = {
                f"L{li}": e for li, e in ranked_layers
            }

        if head_entropies:
            head_entropies.sort(key=lambda x: x[0])  # ascending = most concentrated
            top_concentrated = [
                f"L{li}H{hi_} ρ={ent:.3f}" for ent, li, hi_ in head_entropies[:5]
            ]
            out["att_metric_global_concentrated"] = top_concentrated

        if global_col_sums.sum() > 0:
            top_idx = _top_k_indices(global_col_sums, k=5)
            out["att_metric_global_top_attended"] = [
                f"{tokens[i]} ({global_col_sums[i]:.2f})"
                for i in top_idx if 0 <= i < n_tok
            ]

        # ── Selected head: skip rest if no selection ────────────────
        try:
            li = int(layer_idx) if layer_idx is not None else None
            hi = int(head_idx) if head_idx is not None else None
        except Exception:
            return out
        if li is None or hi is None or not (0 <= li < n_layers):
            return out
        try:
            head_attn = attentions[li][0, hi].detach().cpu().numpy()
            head_attn = head_attn[:n_tok, :n_tok]
        except Exception:
            return out

        # Entropy + max + peak pair.
        try:
            p = head_attn + 1e-12
            row_entropy = -(p * _np.log(p)).sum(axis=-1)
            out["att_metric_head_entropy"] = round(float(row_entropy.mean()), 4)
            out["att_metric_head_max_attn"] = round(float(head_attn.max()), 4)
            am = int(head_attn.argmax())
            src = am // head_attn.shape[1]
            tgt = am % head_attn.shape[1]
            if 0 <= src < n_tok and 0 <= tgt < n_tok:
                out["att_metric_head_top_pair"] = (
                    f"{tokens[src]} → {tokens[tgt]} "
                    f"({head_attn[src, tgt]:.3f})"
                )
        except Exception:
            pass

        # Top-K tokens RECEIVING attention (column sums).
        try:
            col_sums = head_attn.sum(axis=0)
            top_recv = _top_k_indices(col_sums, k=5)
            out["att_metric_head_top_received"] = [
                f"{tokens[i]} ({col_sums[i]:.3f})"
                for i in top_recv if 0 <= i < n_tok
            ]
        except Exception:
            pass

        # Top-K tokens DISTRIBUTING attention (lowest entropy rows = most
        # concentrated queries → most "decisive" tokens).
        try:
            sorted_rows = list(_np.argsort(row_entropy))
            top_distr = [int(i) for i in sorted_rows[:5]]
            out["att_metric_head_top_attending"] = [
                f"{tokens[i]} (H={row_entropy[i]:.3f})"
                for i in top_distr if 0 <= i < n_tok
            ]
        except Exception:
            pass

        # Attention mass on special tokens.
        try:
            special = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>", "<s>", "</s>"}
            spec_idx = [i for i, t in enumerate(tokens) if t in special]
            if spec_idx:
                spec_mass = float(head_attn[:, spec_idx].sum(axis=1).mean())
                out["att_metric_head_special_mass"] = round(spec_mass, 4)
        except Exception:
            pass

        # Head specialisation (if cached_result has it).
        try:
            spec = getattr(cached_result, "head_specialization", None)
            if isinstance(spec, dict):
                layer_spec = spec.get(li) or spec.get(str(li))
                if isinstance(layer_spec, dict):
                    head_spec = layer_spec.get(hi) or layer_spec.get(str(hi))
                    if isinstance(head_spec, dict):
                        # Pick a small, human-readable subset.
                        family = head_spec.get("family") or head_spec.get(
                            "cluster_name"
                        )
                        if family:
                            out["att_metric_head_specialization"] = str(family)
        except Exception:
            pass
    except Exception:
        pass
    return out


def _bias_token_top_score(token_label) -> tuple:
    """Return (score, category) for the strongest non-O bias score in a token."""
    scores = token_label.get("scores") or {}
    best = (0.0, None)
    for cat in ("STEREO", "UNFAIR", "GEN"):
        v = scores.get(cat)
        if v is not None and v > best[0]:
            best = (float(v), cat)
    return best


def _summarise_bias_result(bias_result) -> Dict[str, Any]:
    """Turn the rich bias-processing dict into a structured summary."""
    out: Dict[str, Any] = {}
    if not isinstance(bias_result, dict):
        return out
    token_labels = bias_result.get("token_labels") or []
    out["n_tokens"] = len(token_labels)
    biased = [t for t in token_labels if t.get("is_biased")]
    out["n_biased"] = len(biased)
    if biased:
        # Plain preview (first few tokens) — for back-compat.
        out["biased_preview"] = [str(t.get("token", "?")) for t in biased[:10]]
        # Top-5 scored: include the actual score and category.
        scored = []
        for t in biased:
            score, cat = _bias_token_top_score(t)
            if cat:
                scored.append((score, cat, str(t.get("token", "?"))))
        scored.sort(reverse=True)
        if scored:
            out["biased_scored"] = [
                f"{tok}={score:.2f} {cat}" for score, cat, tok in scored[:5]
            ]
            out["strongest_token"] = f"{scored[0][2]} ({scored[0][1]}, {scored[0][0]:.2f})"
            out["mean_confidence"] = round(
                sum(s for s, _c, _t in scored) / len(scored), 4
            )
    # Counts by category.
    type_counts: Dict[str, int] = {"GEN": 0, "UNFAIR": 0, "STEREO": 0}
    for t in biased:
        for bt in (t.get("bias_types") or []):
            if bt in type_counts:
                type_counts[bt] += 1
    if any(type_counts.values()):
        out["type_counts"] = type_counts
    # Biased spans.
    spans = bias_result.get("bias_spans") or []
    if isinstance(spans, list) and spans:
        # spans can be dicts {start, end, text, ...} or tuples.
        rendered = []
        for s in spans[:5]:
            if isinstance(s, dict):
                text = s.get("text") or s.get("span")
                if text:
                    rendered.append(str(text))
            else:
                try:
                    rendered.append(" ".join(str(x) for x in s))
                except Exception:
                    pass
        if rendered:
            out["bias_spans"] = rendered
    # Propagation analysis.
    prop = bias_result.get("propagation_analysis") or {}
    if isinstance(prop, dict):
        if prop.get("peak_layer") is not None:
            out["peak_layer"] = prop.get("peak_layer")
        pattern = prop.get("propagation_pattern")
        if pattern and pattern != "none":
            out["propagation"] = pattern
    # Peak head in bias matrix.
    bm = bias_result.get("bias_matrix")
    if bm is not None:
        try:
            import numpy as _np
            arr = _np.asarray(bm)
            if arr.size > 0 and arr.ndim == 2:
                am = int(arr.argmax())
                li = am // arr.shape[1]
                hi = am % arr.shape[1]
                out["matrix_peak_head"] = f"L{li}H{hi} ({float(arr.max()):.3f})"
        except Exception:
            pass
    return out


def _extract_ig_metrics(ig_bundle, tokens_hint=None) -> Dict[str, Any]:
    """Summarise an ``IGAnalysisBundle`` (top tokens + head correlations)."""
    out: Dict[str, Any] = {}
    if ig_bundle is None:
        return out
    try:
        token_attrs = getattr(ig_bundle, "token_attributions", None)
        tokens = getattr(ig_bundle, "tokens", None) or tokens_hint or []
        if token_attrs is not None and len(tokens) > 0:
            top_idx = _top_k_indices(token_attrs, k=5)
            out["bias_metric_ig_top_tokens"] = [
                f"{tokens[i]} ({float(token_attrs[i]):.3f})"
                for i in top_idx if 0 <= i < len(tokens)
            ]
        # Head correlations: sorted by absolute Spearman desc upstream;
        # take top 3 and a global mean for the audit trail.
        corr = getattr(ig_bundle, "correlations", None) or []
        if corr:
            top3 = corr[:3]
            out["bias_metric_ig_top_corr_heads"] = [
                f"L{c.layer}H{c.head} ρ={c.spearman_rho:+.3f}"
                for c in top3
            ]
            mean_rho = sum(abs(c.spearman_rho) for c in corr) / len(corr)
            out["bias_metric_ig_mean_spearman"] = round(float(mean_rho), 4)
    except Exception:
        pass
    return out


def _extract_ablation_metrics(ablation_list) -> Dict[str, Any]:
    """Summarise a list of ``HeadAblationResult``."""
    out: Dict[str, Any] = {}
    if not ablation_list:
        return out
    try:
        # Sort by representation_impact descending.
        ranked = sorted(
            ablation_list,
            key=lambda r: getattr(r, "representation_impact", 0.0) or 0.0,
            reverse=True,
        )
        out["bias_metric_ablation_top_heads"] = [
            f"L{r.layer}H{r.head} Δrep={r.representation_impact:+.3f}"
            for r in ranked[:3]
        ]
        kls = [
            r.kl_divergence for r in ablation_list
            if getattr(r, "kl_divergence", None) is not None
        ]
        if kls:
            out["bias_metric_ablation_max_kl"] = round(float(max(kls)), 4)
        impacts = [
            r.representation_impact for r in ablation_list
            if getattr(r, "representation_impact", None) is not None
        ]
        if impacts:
            out["bias_metric_ablation_mean_impact"] = round(
                float(sum(impacts) / len(impacts)), 4
            )
    except Exception:
        pass
    return out


def _extract_perturbation_metrics(perturb_bundle) -> Dict[str, Any]:
    """Summarise a ``PerturbationAnalysisBundle``."""
    out: Dict[str, Any] = {}
    if perturb_bundle is None:
        return out
    try:
        token_results = getattr(perturb_bundle, "token_results", None) or []
        if token_results:
            ranked = sorted(
                token_results,
                key=lambda r: getattr(r, "importance", 0.0) or 0.0,
                reverse=True,
            )
            out["bias_metric_perturb_top_tokens"] = [
                f"{r.token} ({r.importance:.3f})" for r in ranked[:5]
            ]
        rho = getattr(perturb_bundle, "perturb_vs_ig_spearman", None)
        if rho is not None:
            out["bias_metric_perturb_vs_ig"] = round(float(rho), 4)
    except Exception:
        pass
    return out


def _extract_lrp_metrics(lrp_bundle, tokens_hint=None) -> Dict[str, Any]:
    """Summarise an ``LRPAnalysisBundle``."""
    out: Dict[str, Any] = {}
    if lrp_bundle is None:
        return out
    try:
        token_attrs = getattr(lrp_bundle, "token_attributions", None)
        tokens = getattr(lrp_bundle, "tokens", None) or tokens_hint or []
        if token_attrs is not None and len(tokens) > 0:
            top_idx = _top_k_indices(token_attrs, k=5)
            out["bias_metric_lrp_top_tokens"] = [
                f"{tokens[i]} ({float(token_attrs[i]):.3f})"
                for i in top_idx if 0 <= i < len(tokens)
            ]
        rho = getattr(lrp_bundle, "lrp_vs_ig_spearman", None)
        if rho is not None:
            out["bias_metric_lrp_vs_ig"] = round(float(rho), 4)
    except Exception:
        pass
    return out


def _compute_compare_deltas(summary_a: Dict[str, Any],
                            summary_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compute A→B differences for the central bias signals.

    In a counterfactual or cross-model audit, the *difference* is the
    point of the experiment: a non-zero ``Δ`` means the edit moved the
    model's bias signal. The values here are signed (positive = B has
    more than A).
    """
    out: Dict[str, Any] = {}
    if not summary_a or not summary_b:
        return out
    # Counts.
    if "n_biased" in summary_a and "n_biased" in summary_b:
        out["bias_metric_delta_n_biased"] = (
            summary_b["n_biased"] - summary_a["n_biased"]
        )
    # Strongest token score difference.
    sa = summary_a.get("strongest_token") or ""
    sb = summary_b.get("strongest_token") or ""
    if sa and sb:
        try:
            score_a = float(sa.rsplit(", ", 1)[-1].rstrip(")"))
            score_b = float(sb.rsplit(", ", 1)[-1].rstrip(")"))
            out["bias_metric_delta_strongest_score"] = round(score_b - score_a, 4)
        except Exception:
            pass
    # Mean confidence.
    mca = summary_a.get("mean_confidence")
    mcb = summary_b.get("mean_confidence")
    if mca is not None and mcb is not None:
        out["bias_metric_delta_mean_confidence"] = round(float(mcb) - float(mca), 4)
    # Type-count differences as a dict.
    tca = summary_a.get("type_counts") or {}
    tcb = summary_b.get("type_counts") or {}
    if tca or tcb:
        all_cats = set(tca) | set(tcb)
        diff = {c: int(tcb.get(c, 0) - tca.get(c, 0)) for c in sorted(all_cats)}
        if any(diff.values()):
            out["bias_metric_delta_type_counts"] = diff
    # Peak propagation layer shift.
    pla = summary_a.get("peak_layer")
    plb = summary_b.get("peak_layer")
    if pla is not None and plb is not None:
        out["bias_metric_delta_peak_layer"] = int(plb) - int(pla)
    # Biased-token set overlap (use the plain preview list).
    set_a = set(summary_a.get("biased_preview") or [])
    set_b = set(summary_b.get("biased_preview") or [])
    if set_a or set_b:
        overlap = sorted(set_a & set_b)
        only_a = sorted(set_a - set_b)
        only_b = sorted(set_b - set_a)
        if overlap:
            out["bias_metric_delta_token_overlap"] = overlap
        if only_a:
            out["bias_metric_delta_unique_to_a"] = only_a
        if only_b:
            out["bias_metric_delta_unique_to_b"] = only_b
    return out


def _cross_method_top1_agreement(ig, lrp, perturb) -> Optional[str]:
    """Check whether IG, LRP, and perturbation agree on the top-1 token.

    Returns a short string describing the agreement, or ``None`` if not
    enough methods are available to compute it.
    """
    try:
        candidates: Dict[str, str] = {}
        if ig is not None:
            ta = getattr(ig, "token_attributions", None)
            tks = getattr(ig, "tokens", None) or []
            if ta is not None and len(tks) > 0:
                idx = int(_top_k_indices(ta, k=1)[0])
                if 0 <= idx < len(tks):
                    candidates["IG"] = tks[idx]
        if lrp is not None:
            ta = getattr(lrp, "token_attributions", None)
            tks = getattr(lrp, "tokens", None) or []
            if ta is not None and len(tks) > 0:
                idx = int(_top_k_indices(ta, k=1)[0])
                if 0 <= idx < len(tks):
                    candidates["LRP"] = tks[idx]
        if perturb is not None:
            trs = getattr(perturb, "token_results", None) or []
            if trs:
                ranked = sorted(
                    trs,
                    key=lambda r: getattr(r, "importance", 0.0) or 0.0,
                    reverse=True,
                )
                candidates["perturb"] = str(ranked[0].token)
        if len(candidates) < 2:
            return None
        unique = set(candidates.values())
        if len(unique) == 1:
            tok = next(iter(unique))
            return f"all {len(candidates)} methods agree on '{tok}'"
        return ", ".join(f"{m}={t}" for m, t in candidates.items())
    except Exception:
        return None


def _extract_bias_metrics(bias_state, bias_compare_active: bool) -> Dict[str, Any]:
    """Pull bias-summary numbers + Faithfulness panel results from the
    cross-handler reactives. This is the full evidence block surfaced in
    each Notebook entry.
    """
    out: Dict[str, Any] = {}
    if not bias_state:
        return out

    # ── Side A bias summary ──────────────────────────────────────
    bias_res_a = _safe_get(bias_state.get("bias_results"))
    summary_a = _summarise_bias_result(bias_res_a)
    if summary_a.get("n_tokens"):
        out["bias_metric_n_tokens"] = summary_a["n_tokens"]
    if "n_biased" in summary_a:
        out["bias_metric_n_biased"] = summary_a["n_biased"]
    if summary_a.get("biased_preview"):
        out["bias_metric_biased_preview"] = summary_a["biased_preview"]
    if summary_a.get("biased_scored"):
        out["bias_metric_biased_scored"] = summary_a["biased_scored"]
    if summary_a.get("type_counts"):
        out["bias_metric_type_counts"] = summary_a["type_counts"]
    if summary_a.get("strongest_token"):
        out["bias_metric_strongest_token"] = summary_a["strongest_token"]
    if summary_a.get("mean_confidence") is not None:
        out["bias_metric_mean_confidence"] = summary_a["mean_confidence"]
    if summary_a.get("bias_spans"):
        out["bias_metric_bias_spans"] = summary_a["bias_spans"]
    if summary_a.get("peak_layer") is not None:
        out["bias_metric_peak_layer"] = summary_a["peak_layer"]
    if summary_a.get("propagation"):
        out["bias_metric_propagation"] = summary_a["propagation"]
    if summary_a.get("matrix_peak_head"):
        out["bias_metric_matrix_peak_head"] = summary_a["matrix_peak_head"]

    # Tokens hint for IG/LRP (fall back to bias_results tokens if needed).
    tokens_hint = (bias_res_a or {}).get("tokens") if isinstance(bias_res_a, dict) else None

    # ── XAI panels (Faithfulness Validation) ─────────────────────
    ig_bundle = _safe_get(bias_state.get("ig_results"))
    out.update(_extract_ig_metrics(ig_bundle, tokens_hint=tokens_hint))

    abl_list = _safe_get(bias_state.get("ablation_results"))
    out.update(_extract_ablation_metrics(abl_list))

    pert_bundle = _safe_get(bias_state.get("perturbation_results"))
    out.update(_extract_perturbation_metrics(pert_bundle))

    lrp_bundle = _safe_get(bias_state.get("lrp_results"))
    out.update(_extract_lrp_metrics(lrp_bundle, tokens_hint=tokens_hint))

    # Cross-method top-1 token agreement (DR3: causal humility).
    agreement = _cross_method_top1_agreement(ig_bundle, lrp_bundle, pert_bundle)
    if agreement is not None:
        out["bias_metric_methods_top1_agree"] = agreement

    # ── Side B (compare mode) + Δ deltas ─────────────────────────
    if bias_compare_active:
        summary_b = _summarise_bias_result(_safe_get(bias_state.get("bias_results_B")))
        if "n_biased" in summary_b:
            out["bias_metric_n_biased_b"] = summary_b["n_biased"]
        if summary_b.get("biased_scored"):
            out["bias_metric_biased_scored_b"] = summary_b["biased_scored"]
        if summary_b.get("type_counts"):
            out["bias_metric_type_counts_b"] = summary_b["type_counts"]
        if summary_b.get("peak_layer") is not None:
            out["bias_metric_peak_layer_b"] = summary_b["peak_layer"]
        if summary_b.get("strongest_token"):
            out["bias_metric_strongest_token_b"] = summary_b["strongest_token"]
        # Counterfactual / cross-model deltas.
        out.update(_compute_compare_deltas(summary_a, summary_b))

    return out


def _format_context_value(value: Any) -> str:
    """Render a captured value as a short string for the UI/Markdown."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        if not value:
            return "—"
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, list):
        if not value:
            return "—"
        return ", ".join(str(v) for v in value[:8]) + (
            f" (+{len(value) - 8} more)" if len(value) > 8 else ""
        )
    if isinstance(value, float):
        return f"{value:.4f}"
    s = str(value)
    if len(s) > 220:
        s = s[:217] + "…"
    return s


# ---------------------------------------------------------------------------
# Formatting helpers (exports + rendering)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slugify(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return "untitled"
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:40] or "untitled"


def _context_to_markdown(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "_(no context captured)_"
    lines = _context_lines(context, _CONTEXT_LABELS)
    if not lines:
        return "_(no context captured)_"
    return "\n".join(lines)


def _context_lines(context: Dict[str, Any], labels: Dict[str, str]) -> List[str]:
    lines = []
    for json_key, label in labels.items():
        if json_key in context:
            lines.append(f"- **{label}:** {_format_context_value(context[json_key])}")
    return lines


def _context_subset(context: Optional[Dict[str, Any]],
                    labels: Dict[str, str]) -> Dict[str, Any]:
    if not context:
        return {}
    return {
        key: context[key]
        for key in labels
        if key in context
    }


def _condition_labels() -> Dict[str, str]:
    """Fields that correspond to the thesis' 'conditions tested' element."""
    labels = dict(_INPUT_LABELS)
    labels.update({k: _METRIC_LABELS[k] for k in _PROVENANCE_KEYS if k in _METRIC_LABELS})
    return labels


def _signal_labels() -> Dict[str, str]:
    """Fields that correspond to the thesis' 'signals observed' element."""
    return {
        k: v for k, v in _METRIC_LABELS.items()
        if k not in _PROVENANCE_KEYS
    }


def _context_section_to_markdown(context: Optional[Dict[str, Any]],
                                 labels: Dict[str, str]) -> str:
    if not context:
        return "_(none captured)_"
    lines = _context_lines(context, labels)
    return "\n".join(lines) if lines else "_(none captured)_"


def _entry_conditions(entry: Dict[str, Any]) -> Dict[str, Any]:
    saved = entry.get("conditions_tested")
    if isinstance(saved, dict):
        return saved
    return _context_subset(entry.get("context"), _condition_labels())


def _entry_signals(entry: Dict[str, Any]) -> Dict[str, Any]:
    saved = entry.get("signals_observed")
    if isinstance(saved, dict):
        return saved
    return _context_subset(entry.get("context"), _signal_labels())


def _context_rows_html(ctx: Dict[str, Any], labels: Dict[str, str],
                       flagged: set) -> str:
    rows = []
    for json_key, label in labels.items():
        if json_key not in ctx:
            continue
        row_cls = "nb-ctx-row"
        if json_key in flagged:
            row_cls += " nb-ctx-row-warning"
        rows.append(
            f'<div class="{row_cls}">'
            f'<span class="nb-ctx-key">{_html.escape(label)}</span>'
            f'<span class="nb-ctx-val">{_html.escape(_format_context_value(ctx[json_key]))}</span>'
            f"</div>"
        )
    return "".join(rows)


def _context_sections_html(ctx: Dict[str, Any], flagged: set,
                           *, entry: bool = False) -> str:
    block_cls = "nb-ctx-block"
    if entry:
        block_cls += " nb-ctx-block-entry"
    sections = []
    for title, labels in (
        ("Conditions tested", _condition_labels()),
        ("Signals observed", _signal_labels()),
    ):
        body = _context_rows_html(ctx, labels, flagged)
        if not body:
            body = '<div class="nb-ctx-empty-inline">(none captured)</div>'
        sections.append(
            '<div class="nb-ctx-subsection">'
            f'<div class="nb-ctx-subtitle">{_html.escape(title)}</div>'
            f'<div class="{block_cls}">{body}</div>'
            "</div>"
        )
    return "".join(sections)


def _entry_to_markdown(entry: Dict[str, Any]) -> str:
    title = entry.get("title") or "Untitled entry"
    ts = entry.get("timestamp", "")
    lines = [f"## {title}", "", f"*Recorded: {ts}*", ""]
    context = entry.get("context")

    lines.append("### Hypothesis")
    lines.append("")
    lines.append((entry.get("hypothesis") or "").strip() or "_(empty)_")
    lines.append("")

    # These two thesis elements are captured automatically from the dashboard
    # state and computed evidence at save time.
    lines.append("### Conditions tested")
    lines.append("")
    lines.append(_context_section_to_markdown(_entry_conditions(entry), _condition_labels()))
    lines.append("")

    lines.append("### Signals observed")
    lines.append("")
    lines.append(_context_section_to_markdown(_entry_signals(entry), _signal_labels()))
    lines.append("")

    lines.append("### Uncertainty acknowledged")
    lines.append("")
    lines.append((entry.get("uncertainty") or "").strip() or "_(empty)_")
    lines.append("")

    lines.append("### Next steps")
    lines.append("")
    lines.append((entry.get("next_steps") or "").strip() or "_(empty)_")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _entries_to_markdown(entries: List[Dict[str, Any]]) -> str:
    head = [
        "# Auditor Notebook",
        "",
        f"_Exported: {_now_iso()}_",
        "",
        f"**Total entries:** {len(entries)}",
        "",
        "---",
        "",
    ]
    if not entries:
        return "\n".join(head + ["_No entries recorded yet._", ""])
    body = "\n---\n\n".join(_entry_to_markdown(e) for e in entries)
    return "\n".join(head) + body


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------

def notebook_server_handlers(input, output, session, *,
                              cached_result=None, cached_result_B=None,
                              bias_state=None):
    """Register all reactive components of the Auditor Notebook drawer.

    Parameters
    ----------
    cached_result, cached_result_B
        ``reactive.value`` holding the Attention tab's compute results
        for model A and B respectively. When available, the captured
        context will include attention metrics derived from them.
    bias_state
        dict returned by ``bias_server_handlers`` exposing the bias
        analysis reactives (``bias_results``, ``ig_results``, etc.).
        Used to summarise bias metrics in the captured context.
    """

    entries = reactive.value(_load_entries())
    last_status = reactive.value(("", "ok"))  # (message, kind)

    def _capture() -> Dict[str, Any]:
        return _capture_context(
            input,
            cached_result=cached_result,
            cached_result_B=cached_result_B,
            bias_state=bias_state,
        )

    # ---- Add entry ------------------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_add)
    def _add_entry():
        record = {
            "case_id": (input.nb_case_id() or "").strip(),
            "title": (input.nb_title() or "").strip(),
            "hypothesis": (input.nb_hypothesis() or "").strip(),
            "uncertainty": (input.nb_uncertainty() or "").strip(),
            "next_steps": (input.nb_next_steps() or "").strip(),
        }
        missing = [k for k in _REQUIRED_FIELDS if not record[k]]
        if missing:
            human = ", ".join(_FIELD_LABELS[k] for k in missing)
            last_status.set((f"Missing required field(s): {human}.", "error"))
            return
        record["schema_version"] = 2
        record["construct"] = "auditor_notebook_five_element_hybrid"
        record["timestamp"] = _now_iso()
        record["context"] = _capture()
        record["conditions_tested"] = _context_subset(
            record["context"],
            _condition_labels(),
        )
        record["signals_observed"] = _context_subset(
            record["context"],
            _signal_labels(),
        )
        current = list(entries.get())
        current.append(record)
        entries.set(current)
        _save_entries(current)
        last_status.set(
            (
                f"Entry saved with {len(record['context'])} context field(s) captured.",
                "ok",
            )
        )
        # Reset form (free-text only — context is captured fresh next time).
        # ``nb_case_id`` is intentionally preserved so that consecutive
        # entries belonging to the same investigation share the same ID
        # without re-typing.
        ui.update_text("nb_title", value="")
        ui.update_text_area("nb_hypothesis", value="")
        ui.update_text_area("nb_uncertainty", value="")
        ui.update_text_area("nb_next_steps", value="")

    # ---- Clear form -----------------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_clear)
    def _clear_form():
        # Explicit clear *does* wipe the case_id (full reset).
        ui.update_text("nb_case_id", value="")
        ui.update_text("nb_title", value="")
        ui.update_text_area("nb_hypothesis", value="")
        ui.update_text_area("nb_uncertainty", value="")
        ui.update_text_area("nb_next_steps", value="")
        last_status.set(("Form cleared.", "ok"))

    # ---- Dismiss transient status when leaving the drawer ----------------
    @reactive.effect
    @reactive.event(input.nb_dismiss_status, ignore_init=True)
    def _dismiss_status():
        last_status.set(("", "ok"))

    # ---- Clear all entries ---------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_clear_all)
    def _clear_all():
        if not entries.get():
            return
        entries.set([])
        _save_entries([])
        last_status.set(("All entries cleared.", "ok"))

    # ---- Delete individual entry ---------------------------------------
    @reactive.effect
    @reactive.event(input.nb_delete_idx, ignore_init=True)
    def _delete_one():
        idx = input.nb_delete_idx()
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            return
        current = list(entries.get())
        if 0 <= idx_int < len(current):
            del current[idx_int]
            entries.set(current)
            _save_entries(current)
            last_status.set(("Entry removed.", "ok"))

    # ---- Restore state from an entry -----------------------------------
    @reactive.effect
    @reactive.event(input.nb_restore_idx, ignore_init=True)
    def _restore_state():
        raw = input.nb_restore_idx()
        try:
            idx_int = int(raw)
        except (TypeError, ValueError):
            return
        current = entries.get()
        if not (0 <= idx_int < len(current)):
            return
        ctx = current[idx_int].get("context") or {}
        if not ctx:
            last_status.set(("This entry has no captured context to restore.", "error"))
            return

        n_applied = 0
        client_event_values: Dict[str, Any] = {}
        for json_key, value in ctx.items():
            input_id = _RESTORE_MAP.get(json_key)
            if input_id is None:
                continue
            try:
                if json_key in _NAVSET_INPUTS:
                    ui.update_navset(input_id, selected=value)
                elif json_key in _SWITCH_INPUTS:
                    ui.update_switch(input_id, value=bool(value))
                elif json_key in _NUMERIC_INPUTS:
                    try:
                        ui.update_slider(input_id, value=value)
                    except Exception:
                        ui.update_numeric(input_id, value=value)
                elif json_key in _RADIO_INPUTS:
                    ui.update_radio_buttons(input_id, selected=value)
                elif json_key in _SELECT_INPUTS:
                    ui.update_select(input_id, selected=value)
                elif json_key in _TEXT_AREA_INPUTS:
                    ui.update_text_area(input_id, value=str(value))
                elif json_key in _CLIENT_EVENT_INPUTS:
                    client_event_values[input_id] = value
                else:
                    ui.update_text(input_id, value=str(value))
                n_applied += 1
            except Exception:
                # Updates for unknown inputs are silent failures; keep going.
                continue
        if client_event_values:
            try:
                asyncio.create_task(
                    session.send_custom_message(
                        "nb_restore_client_inputs",
                        {"values": client_event_values},
                    )
                )
            except Exception:
                _logger.exception("Could not restore custom client inputs")
        last_status.set(
            (f"Restored {n_applied} field(s) from the saved entry.", "ok")
        )

    # ---- FAB visibility: only reveal after a run in Attention or Bias --
    @reactive.effect
    def _toggle_fab_visibility():
        """Push the FAB visibility state to the browser via custom message.

        Triggers whenever ``Generate All`` (Attention) or ``Analyze Bias``
        (Bias) is clicked. The matching JS handler is registered in
        ``NOTEBOOK_JS`` and adds/removes the ``nb-fab-visible`` class.
        """
        try:
            gen_clicks = int(input.generate_all() or 0)
        except Exception:
            gen_clicks = 0
        try:
            bias_clicks = int(input.analyze_bias_btn() or 0)
        except Exception:
            bias_clicks = 0
        visible = gen_clicks > 0 or bias_clicks > 0
        try:
            asyncio.create_task(
                session.send_custom_message("nb_fab_toggle", {"visible": visible})
            )
        except Exception:
            _logger.exception("Could not toggle FAB visibility")

    # ---- Status banner -------------------------------------------------
    @output
    @render.ui
    def nb_status():
        msg, kind = last_status.get()
        if not msg:
            return ui.HTML("")
        klass = "nb-status nb-status-error" if kind == "error" else "nb-status"
        return ui.tags.div(msg, class_=klass)

    # ---- Entry counter badge -------------------------------------------
    @output
    @render.ui
    def nb_count():
        n = len(entries.get())
        label = "entry" if n == 1 else "entries"
        return ui.HTML(f'<span class="nb-count">{n} {label}</span>')

    # ---- Live preview of what would be captured ------------------------
    @output
    @render.ui
    def nb_context_preview():
        """Show the analyst the structured fields the system is about to record."""
        ctx = _capture()
        if not ctx:
            return ui.tags.div(
                "No dashboard state to capture yet. Pick a model and a prompt "
                "in the Attention or Bias tabs to populate this preview.",
                class_="nb-ctx-empty",
            )
        flagged = _disconfirming_keys(ctx)
        warning_banner = ""
        if flagged:
            warning_banner = (
                f'<div class="nb-ctx-warning-banner">'
                f"⚠ {len(flagged)} disconfirming signal(s) detected. "
                f"Consider tempering or revising the hypothesis before saving."
                f"</div>"
            )
        return ui.HTML(
            warning_banner
            + _context_sections_html(ctx, flagged)
        )

    # ---- Entries list --------------------------------------------------
    @output
    @render.ui
    def nb_entries():
        items = entries.get()
        if not items:
            return ui.tags.div(
                "No entries yet. Pick a prompt + model in Attention or Bias, "
                "fill the three judgment fields, and click \"Add entry\". The "
                "rest is captured automatically.",
                class_="nb-empty",
            )
        pieces = []
        for i, e in enumerate(items):
            title = _html.escape(e.get("title") or f"Entry {i + 1}")
            ts = _html.escape(e.get("timestamp", ""))
            case_id = (e.get("case_id") or "").strip()
            case_chip = (
                f'<span class="nb-case-chip">{_html.escape(case_id)}</span>'
                if case_id else ""
            )
            # Automatically captured conditions/signals (structured evidence)
            ctx = e.get("context") or {}
            if ctx:
                flagged = _disconfirming_keys(ctx)
                banner = (
                    f'<div class="nb-ctx-warning-banner">'
                    f"⚠ {len(flagged)} disconfirming signal(s) recorded with this entry."
                    f"</div>"
                ) if flagged else ""
                ctx_html = (
                    '<div class="nb-entry-field">'
                    '<span class="nb-entry-field-label">Automatically captured thesis elements</span>'
                    + banner
                    + _context_sections_html(ctx, flagged, entry=True)
                    + "</div>"
                )
            else:
                ctx_html = (
                    '<div class="nb-entry-field">'
                    '<span class="nb-entry-field-label">Automatically captured thesis elements</span>'
                    '<div class="nb-entry-field-value">(no context captured for '
                    'this entry — created before context-capture was enabled)'
                    "</div></div>"
                )
            # Free-text fields
            fields_html = [ctx_html]
            for k in _REQUIRED_FIELDS:
                label = _FIELD_LABELS[k]
                body = _html.escape(e.get(k, "") or "")
                fields_html.append(
                    f'<div class="nb-entry-field">'
                    f'<span class="nb-entry-field-label">{label}</span>'
                    f'<div class="nb-entry-field-value">{body}</div>'
                    f"</div>"
                )
            # Per-entry actions
            actions_html = (
                f'<div class="nb-entry-actions">'
                f'<button class="nb-entry-restore" '
                f'onclick="Shiny.setInputValue(\'nb_restore_idx\', {i}, '
                f"{{priority: 'event'}}); return false;\">"
                f"⤺ Restore state</button>"
                f'<button class="nb-entry-delete" '
                f'onclick="Shiny.setInputValue(\'nb_delete_idx\', {i}, '
                f"{{priority: 'event'}}); return false;\">"
                f"× delete</button>"
                f"</div>"
            )
            pieces.append(
                f'<div class="nb-entry">'
                f'<div class="nb-entry-header">'
                f'<span class="nb-entry-title">{title}{case_chip}</span>'
                f'<span class="nb-entry-meta">{ts}</span>'
                f"</div>"
                f"{''.join(fields_html)}"
                f"{actions_html}"
                f"</div>"
            )
        return ui.HTML("".join(pieces))

    # ---- Downloads -----------------------------------------------------
    @render.download(filename=lambda: f"auditor_notebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    def nb_download_md():
        yield _entries_to_markdown(list(entries.get()))

    @render.download(filename=lambda: f"auditor_notebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def nb_download_json():
        payload = {
            "exported_at": _now_iso(),
            "n_entries": len(entries.get()),
            "entries": list(entries.get()),
        }
        yield json.dumps(payload, ensure_ascii=False, indent=2)
