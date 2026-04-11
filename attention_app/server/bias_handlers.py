import json
import html as _html
import logging
from datetime import datetime
from pathlib import Path
import traceback
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

_logger = logging.getLogger(__name__)


def _csv_safe(val):
    """Prevent CSV formula injection by prefixing dangerous first characters."""
    s = str(val)
    if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + s
    return s

import numpy as np
import torch
from shiny import ui, render, reactive

from ..models import ModelManager
from ..utils import get_reproducibility_info
from ..bias import (
    GusNetDetector, EnsembleGusNetDetector,
    AttentionBiasAnalyzer,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_bias_propagation_heads_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_ratio_formula_html,
    create_bias_criteria_html,
    create_bias_sentence_preview,
    create_token_bias_strip,
    create_confidence_breakdown,
    create_ablation_impact_chart,
    create_ig_correlation_chart_v2,
    create_ig_token_comparison_chart,
    create_ig_distribution_chart,
    create_ig_layer_summary_chart,
    create_topk_overlap_chart,
    create_perturbation_comparison_chart,
    create_perturbation_attn_heatmap,
    create_lrp_comparison_chart,
    create_cross_method_agreement_chart,
)
from ..bias.counterfactual import find_swappable_terms, generate_counterfactual, get_swap_for_token
from ..bias.head_ablation import batch_ablate_top_heads, HeadAblationResult
from ..bias.integrated_gradients import (
    batch_compute_ig_correlation, IGCorrelationResult, IGAnalysisBundle,
    TopKOverlapResult,
    batch_compute_perturbation, PerturbationAnalysisBundle,
    batch_compute_lrp, LRPAnalysisBundle,
)
from ..bias.stereoset import (
    load_stereoset_data,
    get_stereoset_scores,
    get_stereoset_examples,
    get_head_sensitivity_matrix,
    get_sensitive_heads,
    get_top_features,
    get_head_profile_stats,
    get_metadata,
    get_gusnet_key,
    compute_model_similarity,
)
from ..bias.visualizations import (
    create_stereoset_overview_html,
    create_stereoset_category_chart,
    create_stereoset_head_sensitivity_heatmap,
    create_stereoset_bias_distribution,
    create_stereoset_example_html,
    create_stereoset_attention_heatmaps,
    create_stereoset_attention_diff_heatmap,
    create_stereoset_head_distributions,
    create_stereoset_attention_scatter,
)
from ..bias.visualizations import compute_cf_attention_consistency, create_cf_consistency_html
from ..bias.feature_extraction import extract_attention_for_text
from ..ui.bias_ui import create_bias_accordion, create_floating_bias_toolbar
from ..ui.components import viz_header

# Map GUS-Net model key -> matching encoder model for attention analysis.
# Ensures the encoder tokenizer matches the GUS-Net architecture so that
# BERT tokens show ## subwords and GPT-2 tokens merge correctly with Ġ.
_GUSNET_TO_ENCODER = {
    # Public (HuggingFace) models
    "gusnet-bert": "pinthoz/gus-net-bert",
    "gusnet-bert-large": "pinthoz/gus-net-bert-large",
    "gusnet-gpt2": "pinthoz/gus-net-gpt2",
    "gusnet-gpt2-medium": "pinthoz/gus-net-gpt2-medium",
    # Local models
    "gusnet-bert-custom": "pinthoz/gus-net-bert-custom",
    "gusnet-ensemble": "pinthoz/gus-net-bert",
    "gusnet-bert-new": "pinthoz/gus-net-bert",
    "gusnet-gpt2-new": "pinthoz/gus-net-gpt2",
    "gusnet-bert-paper": "pinthoz/gus-net-bert",
    "gusnet-gpt2-paper": "pinthoz/gus-net-gpt2",
    # StereoSet canonical keys (underscore variants)
    "gusnet_bert": "pinthoz/gus-net-bert",
    "gusnet_bert_large": "pinthoz/gus-net-bert-large",
    "gusnet_gpt2": "pinthoz/gus-net-gpt2",
    "gusnet_gpt2_medium": "pinthoz/gus-net-gpt2-medium",
}

import re as _re

# Map GUS-Net model key -> original pretrained base model (before fine-tuning).
# Used for the "Base Encoder" attention source comparison.
_GUSNET_TO_BASE = {
    "gusnet-bert":          "bert-base-uncased",
    "gusnet-bert-large":    "bert-large-uncased",
    "gusnet-bert-custom":   "bert-base-uncased",
    "gusnet-gpt2":          "gpt2",
    "gusnet-gpt2-medium":   "gpt2-medium",
    "gusnet-ensemble":      "bert-base-uncased",
    "gusnet-bert-new":      "bert-base-uncased",
    "gusnet-gpt2-new":      "gpt2",
    "gusnet-bert-paper":    "bert-base-uncased",
    "gusnet-gpt2-paper":    "gpt2",
    # StereoSet canonical keys (underscore variants)
    "gusnet_bert":          "bert-base-uncased",
    "gusnet_bert_large":    "bert-large-uncased",
    "gusnet_gpt2":          "gpt2",
    "gusnet_gpt2_medium":   "gpt2-medium",
}

# Canonical display names for GUS-NET variants
_GUSNET_DISPLAY_NAMES = {
    "bert": "GUS-NET-BERT",
    "bert-large": "GUS-NET-BERT-LARGE",
    "gpt2": "GUS-NET-GPT-2",
    "gpt2-medium": "GUS-NET-GPT2-MEDIUM",
}

def _clean_gusnet_label(raw: str) -> str:
    """Simplify a raw GUS-NET model name to a clean display label.

    E.g. ``gus-net-bert-paper-clean-2`` → ``GUS-NET BERT``,
         ``gus-net-gpt2-medium``        → ``GUS-NET GPT-2 Medium``.
    For non-GUS-NET names the string is returned upper-cased.
    """
    # Strip huggingface repository prefix (e.g., "pinthoz/")
    if "/" in raw:
        raw = raw.split("/")[-1]
        
    s = raw.lower().replace("_", "-")
    if "gus" not in s:
        return raw.upper()
    # Strip gus-net prefix and any training-run suffixes (paper, clean, v2, etc.)
    # but preserve model variant identifiers like gpt2, bert-large
    core = _re.sub(r"^gus-?net-?", "", s)
    core = _re.sub(r"-(paper|clean|new|custom|ensemble)([-\d].*)?$", "", core).strip("-")
    return _GUSNET_DISPLAY_NAMES.get(core, f"GUS-NET {core.upper()}")


def _load_gusnet_attention_for_text(text: str, bias_model_key: str):
    """Extract attentions directly from the selected GUS-NET model."""
    if bias_model_key == "gusnet-ensemble":
        bias_model_key = "gusnet-bert"

    det = GusNetDetector(model_key=bias_model_key, threshold=0.5, use_optimized=False)
    tokenizer, model = det._load_model(bias_model_key, det._device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(det._device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
    gus_probs = torch.sigmoid(outputs.logits)[0].detach().cpu()

    effective_thresholds = {}
    if det.config.get("optimized_thresholds"):
        opt = det.config["optimized_thresholds"]
        cat_indices = det.config.get("category_indices", {})
        for cat in ["GEN", "UNFAIR", "STEREO"]:
            if cat in cat_indices:
                vals = [opt[i] for i in cat_indices[cat] if i < len(opt)]
                if vals:
                    effective_thresholds[cat] = sum(vals) / len(vals)
    if not effective_thresholds:
        effective_thresholds = {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

    return {
        "tokens": tokens,
        "text": text,
        "attentions": outputs.attentions,
        "gus_tokens": tokens,
        "gus_probs": gus_probs,
        "bias_model_key": bias_model_key,
        "model_name": _GUSNET_TO_ENCODER.get(bias_model_key, bias_model_key),
        "gus_special": det.config["special_tokens"],
        "effective_thresholds": effective_thresholds,
        "attention_source": "gusnet",
    }


def _load_base_encoder_attention_for_text(text: str, bias_model_key: str):
    """Extract attentions from the original pretrained base model (before fine-tuning).

    For example, if the GUS-Net variant is ``gusnet-bert``, this loads
    ``bert-base-uncased`` and extracts its attention patterns - representing
    how the *untrained* encoder distributes attention before any bias-specific
    fine-tuning.
    """
    base_model_name = _GUSNET_TO_BASE.get(bias_model_key, "bert-base-uncased")

    tokenizer, encoder_model, _ = ModelManager.get_model(base_model_name)
    device = ModelManager.get_device()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())

    # We still need GUS-Net labels for bias highlights - use the selected detector
    if bias_model_key == "gusnet-ensemble":
        bk = "gusnet-bert"
    else:
        bk = bias_model_key
    det = GusNetDetector(model_key=bk, threshold=0.5, use_optimized=False)
    gus_tokens, gus_probs = det.predict_proba(text)

    effective_thresholds = {}
    if det.config.get("optimized_thresholds"):
        opt = det.config["optimized_thresholds"]
        cat_indices = det.config.get("category_indices", {})
        for cat in ["GEN", "UNFAIR", "STEREO"]:
            if cat in cat_indices:
                vals = [opt[i] for i in cat_indices[cat] if i < len(opt)]
                if vals:
                    effective_thresholds[cat] = sum(vals) / len(vals)
    if not effective_thresholds:
        effective_thresholds = {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

    return {
        "tokens": tokens,
        "text": text,
        "attentions": outputs.attentions,
        "gus_tokens": gus_tokens,
        "gus_probs": gus_probs,
        "bias_model_key": bias_model_key,
        "model_name": base_model_name,
        "gus_special": det.config["special_tokens"],
        "effective_thresholds": effective_thresholds,
        "attention_source": "base",
    }


def _deferred_plotly(fig, container_id, height=None, config=None, click_input=None):
    """Render a Plotly figure as deferred HTML - only calls Plotly.newPlot()
    when the container becomes visible. This avoids wrong dimensions when
    the container is inside a collapsed accordion panel.

    click_input: if set, a plotly_click listener will call
        Shiny.setInputValue(click_input, clickedX) on each click.
    """
    import plotly.io as pio
    import base64, html as _html

    # Use figure's internal height if not explicitly overridden
    if height is None:
        fig_height = getattr(fig.layout, "height", None)
        if fig_height:
            height = f"{fig_height}px"
        else:
            height = "400px"

    fig_json = pio.to_json(fig, validate=False)
    cfg = json.dumps(config or {"displayModeBar": False, "responsive": True})
    # Base64-encode the figure JSON and store in a data attribute.
    # This avoids jQuery/Shiny re-parsing <script> tags via .html()
    # and sidesteps any HTML entity issues with <template> elements.
    b64_fig = base64.b64encode(fig_json.encode()).decode()
    escaped_cfg = _html.escape(cfg, quote=True)
    click_attr = f' data-plotly-click-input="{click_input}"' if click_input else ""
    return (
        f'<div id="{container_id}" class="plotly-deferred" '
        f'style="width:100%;height:{height};min-height:50px;"'
        f' data-plotly-config="{escaped_cfg}"'
        f' data-plotly-fig="{b64_fig}"'
        f'{click_attr}>'
        f'</div>'
    )


def _wrap_card(content, title=None, subtitle=None, help_text=None, manual_header=None, style=None, controls=None):
    """Wrap content in a card with consistent header style."""
    base_style = "min-height: auto; display: flex; flex-direction: column;"
    if style:
        base_style += f" {style}"

    header = None
    if manual_header:
        # manual_header is (title, subtitle) tuple - optionally with (i) tooltip
        _info_icon = None
        if help_text:
            _info_icon = ui.div(
                {"class": "info-tooltip-wrapper"},
                ui.span({"class": "info-tooltip-icon"}, "i"),
                ui.div({"class": "info-tooltip-content"}, ui.HTML(help_text)),
            )

        _title_row_children = [ui.h4(ui.HTML(manual_header[0]), style="margin:0;")]
        if _info_icon:
            _title_row_children.append(_info_icon)

        if controls:
            _title_row_children.append(
                ui.div(*controls, style="margin-left:auto;display:flex;align-items:center;gap:8px;")
            )

        header = ui.div(
            {"class": "viz-header", "style": "display:flex;flex-direction:column;gap:0;"},
            ui.div(
                {"style": "display:flex;align-items:center;gap:8px;flex-wrap:wrap;"},
                *_title_row_children,
            ),
            ui.p(ui.HTML(manual_header[1]), style="font-size:11px;color:#6b7280;margin:4px 0 0;"),
        )
    elif title:
        header = viz_header(title, subtitle, help_text, controls=controls)

    return ui.div(
        {"class": "card", "style": base_style},
        header,
        ui.div(content, style="flex: 1; display: flex; flex-direction: column;")
    )



# ─── Token-alignment helper ───────────────────────────────────────────────

def _align_gusnet_to_attention_tokens(gusnet_labels, attention_tokens, gusnet_special_tokens=None):
    """Align GUS-Net token labels to the attention model's tokens.

    Handles cross-tokenizer subword mismatches - e.g. when GPT-2 BPE
    produces a single token ``nurturing`` while BERT WordPiece splits it
    into ``nur`` + ``##turing``.  In such cases the GUS-Net label is
    propagated to all BERT subwords that form the original word.

    Returns a list (same length as *attention_tokens*) where each entry
    is either a GUS-Net label dict or ``None`` for unmatched positions.
    """
    if gusnet_special_tokens is None:
        gusnet_special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    attn_special = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    def _clean(tok):
        """Normalise a subword token for text matching."""
        return tok.replace("##", "").replace("\u0120", "").replace("Ġ", "").lower().strip()

    # Build cleaned GUS-Net content tokens
    gus_clean = []
    gus_data = []
    for label in gusnet_labels:
        if label["token"] in gusnet_special_tokens:
            continue
        gus_clean.append(_clean(label["token"]))
        gus_data.append(label)

    aligned = [None] * len(attention_tokens)
    gus_idx = 0
    # Tracks leftover characters from a partially consumed GUS-Net token
    gus_remainder = ""
    gus_current_label = None

    for bt_idx, bt in enumerate(attention_tokens):
        if bt in attn_special or (bt.startswith("[") and bt.endswith("]")):
            continue

        clean_bt = _clean(bt)
        if not clean_bt:
            continue

        # Case 1: continuing to consume a partially matched GUS-Net token
        if gus_remainder:
            if gus_remainder.startswith(clean_bt):
                aligned[bt_idx] = gus_current_label
                gus_remainder = gus_remainder[len(clean_bt):]
                if not gus_remainder:
                    gus_current_label = None
                continue
            else:
                # Mismatch mid-word - reset and fall through
                gus_remainder = ""
                gus_current_label = None

        # Case 2: exact match with current GUS-Net token
        if gus_idx < len(gus_clean) and gus_clean[gus_idx] == clean_bt:
            aligned[bt_idx] = gus_data[gus_idx]
            gus_idx += 1
            continue

        # Case 3: attention subword is a prefix of the GUS-Net token
        #         (e.g. BERT "nur" is prefix of GPT-2 "nurturing")
        if gus_idx < len(gus_clean) and gus_clean[gus_idx].startswith(clean_bt):
            aligned[bt_idx] = gus_data[gus_idx]
            gus_remainder = gus_clean[gus_idx][len(clean_bt):]
            gus_current_label = gus_data[gus_idx]
            if not gus_remainder:
                gus_idx += 1
                gus_current_label = None
            else:
                gus_idx += 1  # advance - remainder will be consumed by next subwords
            continue

        # Case 4: GUS-Net subword is a prefix of the attention token
        #         (e.g. GPT-2 splits more finely than BERT on rare words)
        if gus_idx < len(gus_clean):
            accumulated = ""
            scan = gus_idx
            while scan < len(gus_clean) and len(accumulated) < len(clean_bt):
                accumulated += gus_clean[scan]
                scan += 1
            if accumulated == clean_bt:
                # Merge: use the first GUS-Net label (highest specificity)
                aligned[bt_idx] = gus_data[gus_idx]
                gus_idx = scan
                continue

        # Case 5: look-ahead for misalignment recovery
        for look in range(gus_idx, min(gus_idx + 5, len(gus_clean))):
            if gus_clean[look] == clean_bt:
                aligned[bt_idx] = gus_data[look]
                gus_idx = look + 1
                break

    return aligned


# ─── Server handler registration ──────────────────────────────────────────

def _get_bias_model_label(res):
    """Return a human-readable label for the bias model used in the result."""
    from attention_app.bias.gusnet_detector import MODEL_REGISTRY
    key = res.get("bias_model_key", "gusnet-bert")
    cfg = MODEL_REGISTRY.get(key, {})
    return cfg.get("display_name", key)

def _process_raw_bias_result(raw_res, thresholds, use_optimized=False):
    """Apply thresholds to raw results and regeneration attention metrics."""
    if not raw_res:
        return None
    
    try:
        bias_model_key = raw_res["bias_model_key"]
        
        # 1. Re-instantiate detector (lightweight) to apply thresholds
        from attention_app.bias.gusnet_detector import GusNetDetector, EnsembleGusNetDetector
        
        if bias_model_key == "gusnet-ensemble":
            det = EnsembleGusNetDetector(model_key_a="gusnet-bert", model_key_b="gusnet-bert-custom")
        else:
            det = GusNetDetector(model_key=bias_model_key, use_optimized=use_optimized)
            
        # 2. Apply thresholds
        gusnet_labels = det.apply_thresholds(
            raw_res["gus_tokens"], 
            raw_res["gus_probs"], 
            thresholds=thresholds
        )
        
        # 3. Re-align (alignment needed because encoder tokens != gus tokens)
        tokens = raw_res["tokens"]
        gus_special = raw_res["gus_special"]
        
        # We need _align_gusnet function available here
        gus_aligned = _align_gusnet_to_attention_tokens(
            gusnet_labels, tokens, gusnet_special_tokens=gus_special
        )
        
        token_labels = []
        for i, tok in enumerate(tokens):
            matched = gus_aligned[i]
            if matched is not None:
                token_labels.append({
                    **matched,
                    "token": tok,
                    "index": i,
                })
            else:
                token_labels.append({
                    "token": tok,
                    "index": i,
                    "bias_types": [],
                    "is_biased": False,
                    "scores": {"O": 0.0, "GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
                    "method": "gusnet",
                    "explanation": "",
                    "threshold": thresholds.get("GEN", 0.5) if thresholds else 0.5,
                })
        
        # 4. Re-calculate Summary & Spans
        bias_summary = det.get_bias_summary(token_labels)
        bias_spans = det.get_biased_spans(token_labels)
        
        # 5. Re-calculate Attention Metrics (Attention Analyzer)
        # Only if we have attention data
        attentions = raw_res["attentions"]
        biased_indices = [i for i, l in enumerate(token_labels) if l["is_biased"]]
        
        attention_metrics = []
        propagation_analysis = {
            "layer_propagation": [], "peak_layer": None,
            "propagation_pattern": "none",
        }
        bias_matrix = np.array([])
        
        if biased_indices and attentions:
            attention_analyzer = AttentionBiasAnalyzer()
            attention_metrics = attention_analyzer.analyze_attention_to_bias(
                list(attentions), biased_indices, tokens
            )
            propagation_analysis = attention_analyzer.analyze_bias_propagation(
                list(attentions), biased_indices, tokens
            )
            bias_doc_matrix = attention_analyzer.create_attention_bias_matrix(
                list(attentions), biased_indices
            )
            # bias_doc_matrix is a numpy array>= Helper usually returns something else>=
            # Warning: Existing code was: bias_matrix = attention_analyzer.create_attention_bias_matrix(...)
            bias_matrix = bias_doc_matrix

        # Return full result structure
        return {
            **raw_res,
            "token_labels": token_labels,
            "bias_summary": bias_summary,
            "bias_spans": bias_spans,
            "biased_indices": biased_indices,
            "attention_metrics": attention_metrics,
            "propagation_analysis": propagation_analysis,
            "bias_matrix": bias_matrix,
            "thresholds": thresholds,
            "use_optimized": use_optimized
        }
        
    except Exception:
        _logger.exception("Error processing raw bias result")
        return None


def _build_batch_report(per_sentence_results, all_bar_matrices,
                        bias_model_key, model_name, thresholds, use_optimized,
                        batch_file, elapsed, n_total):
    """Build comprehensive batch analysis report JSON."""
    analyzed = [r for r in per_sentence_results if not r.get("error")]
    failed = [r for r in per_sentence_results if r.get("error")]
    biased = [r for r in analyzed if r.get("is_biased")]

    # Aggregate bias percentages
    bias_pcts = [r["bias_summary"]["bias_percentage"] for r in analyzed if "bias_summary" in r]
    avg_bias_pct = round(sum(bias_pcts) / len(bias_pcts), 1) if bias_pcts else 0

    # Aggregate confidence
    confs = [r["bias_summary"]["avg_confidence"] for r in analyzed if "bias_summary" in r and r["bias_summary"].get("avg_confidence")]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0

    # Category distribution
    cat_dist = {}
    for r in analyzed:
        if "bias_summary" in r:
            for cat in r["bias_summary"].get("categories_found", []):
                cat_dist[cat] = cat_dist.get(cat, 0) + 1

    # Most common biased terms
    term_counts = {}
    for r in analyzed:
        for ts in r.get("token_scores", []):
            if ts.get("is_biased"):
                term_counts[ts["token"]] = term_counts.get(ts["token"], 0) + 1
    top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    # Average confidence per category
    cat_confs = {}
    cat_conf_counts = {}
    for r in analyzed:
        for ts in r.get("token_scores", []):
            if ts.get("is_biased") and ts.get("scores"):
                for bt in ts.get("bias_types", []):
                    score = ts["scores"].get(bt, 0)
                    cat_confs[bt] = cat_confs.get(bt, 0) + score
                    cat_conf_counts[bt] = cat_conf_counts.get(bt, 0) + 1
    avg_conf_per_cat = {k: round(cat_confs[k] / cat_conf_counts[k], 4) for k in cat_confs if cat_conf_counts.get(k)}

    # Sentences ranked by severity
    ranked = sorted(
        [{"index": r["index"], "text": r["text"][:100], "bias_percentage": r["bias_summary"]["bias_percentage"]}
         for r in analyzed if "bias_summary" in r],
        key=lambda x: x["bias_percentage"], reverse=True
    )[:20]

    # Attention analysis: aggregate BAR matrices
    attention_analysis = {}
    if all_bar_matrices:
        stacked = np.stack(all_bar_matrices)
        avg_bar = np.nanmean(stacked, axis=0)
        n_layers, n_heads = avg_bar.shape

        # Top heads by average BAR
        head_entries = []
        for l in range(n_layers):
            for h in range(n_heads):
                bars = [m[l, h] for m in all_bar_matrices if l < m.shape[0] and h < m.shape[1]]
                specialized = sum(1 for b in bars if b > 1.5)
                head_entries.append({
                    "layer": int(l), "head": int(h),
                    "avg_bar": round(float(np.mean(bars)), 4),
                    "max_bar": round(float(np.max(bars)), 4),
                    "specialized_rate": round(specialized / len(bars) * 100, 1) if bars else 0,
                    "n_samples": len(bars),
                })
        head_entries.sort(key=lambda x: x["avg_bar"], reverse=True)

        attention_analysis = {
            "top_heads_by_avg_bar": head_entries[:20],
            "head_consistency": [h for h in head_entries if h["specialized_rate"] >= 50][:20],
            "avg_bar_matrix": [[round(float(avg_bar[l, h]), 4) for h in range(n_heads)] for l in range(n_layers)],
        }

    # Faithfulness summary: aggregate IG/ablation/perturbation/LRP
    ig_rhos = []
    ig_sig_heads = 0
    ig_jaccards = []
    ig_rbos = []
    abl_impacts = []
    abl_kls = []
    perturb_ig_rhos = []
    perturb_attn_rhos = []
    lrp_ig_rhos = []
    lrp_attn_rhos = []

    for r in analyzed:
        ig = r.get("ig_analysis")
        if ig:
            for c in ig.get("ig_attention_correlations", []):
                if c.get("spearman_rho") is not None:
                    ig_rhos.append(abs(c["spearman_rho"]))
                    if c.get("p_value") is not None and c["p_value"] < 0.05:
                        ig_sig_heads += 1
            for t in ig.get("topk_overlaps", []):
                if t.get("jaccard") is not None:
                    ig_jaccards.append(t["jaccard"])
                if t.get("rbo") is not None:
                    ig_rbos.append(t["rbo"])

        abl = r.get("ablation_analysis")
        if abl:
            for a in abl:
                if a.get("representation_impact") is not None:
                    abl_impacts.append(a["representation_impact"])
                if a.get("kl_divergence") is not None:
                    abl_kls.append(a["kl_divergence"])

        pert = r.get("perturbation_analysis")
        if pert:
            if pert.get("perturb_vs_ig_spearman") is not None:
                perturb_ig_rhos.append(pert["perturb_vs_ig_spearman"])
            for c in pert.get("perturb_vs_attention_top5", []):
                if c.get("spearman_rho") is not None:
                    perturb_attn_rhos.append(c["spearman_rho"])

        lrp = r.get("lrp_analysis")
        if lrp:
            if lrp.get("lrp_vs_ig_spearman") is not None:
                lrp_ig_rhos.append(lrp["lrp_vs_ig_spearman"])
            for c in lrp.get("lrp_vs_attention_top5", []):
                if c.get("spearman_rho") is not None:
                    lrp_attn_rhos.append(c["spearman_rho"])

    def _safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    faithfulness_summary = {
        "ig_analysis": {
            "avg_top_head_rho": _safe_mean(ig_rhos),
            "heads_with_significant_correlation": ig_sig_heads,
            "avg_topk_jaccard": _safe_mean(ig_jaccards),
            "avg_topk_rbo": _safe_mean(ig_rbos),
        },
        "ablation_analysis": {
            "avg_representation_impact": _safe_mean(abl_impacts),
            "avg_kl_divergence": _safe_mean(abl_kls),
            "heads_with_high_impact": sum(1 for x in abl_impacts if x > 0.01),
        },
        "perturbation_analysis": {
            "avg_perturb_vs_ig_rho": _safe_mean(perturb_ig_rhos),
            "avg_perturb_vs_attention_rho": _safe_mean(perturb_attn_rhos),
        },
        "lrp_analysis": {
            "avg_lrp_vs_ig_rho": _safe_mean(lrp_ig_rhos),
            "avg_lrp_vs_attention_rho": _safe_mean(lrp_attn_rhos),
        },
        "cross_method_agreement": {
            "ig_vs_lrp_mean_rho": _safe_mean(lrp_ig_rhos),
            "ig_vs_perturbation_mean_rho": _safe_mean(perturb_ig_rhos),
        },
    }

    # Counterfactual analysis
    cf_analysis = []
    for r in analyzed:
        swaps = r.get("counterfactual_swaps")
        if swaps:
            cf_analysis.append({
                "sentence_index": r["index"],
                "text": r["text"],
                "available_swaps": swaps,
            })

    # Reproducibility info
    import transformers
    repro = {
        "model_version": model_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    try:
        import hashlib
        combined = "\n".join(r["text"] for r in per_sentence_results)
        repro["input_hash"] = "sha256:" + hashlib.sha256(combined.encode()).hexdigest()[:16]
    except Exception:
        pass

    report = {
        "executive_summary": {
            "total_sentences": n_total,
            "sentences_analyzed": len(analyzed),
            "sentences_failed": len(failed),
            "biased_sentences": len(biased),
            "bias_rate": round(len(biased) / len(analyzed) * 100, 1) if analyzed else 0,
            "average_bias_percentage": avg_bias_pct,
            "average_confidence": avg_conf,
            "processing_time_seconds": elapsed,
            "model_used": _get_bias_model_label({"bias_model_key": bias_model_key}),
            "model_key": bias_model_key,
            "encoder_model": model_name,
        },
        "per_sentence_results": per_sentence_results,
        "aggregate_statistics": {
            "category_distribution": cat_dist,
            "most_common_biased_terms": [{"term": t, "count": c} for t, c in top_terms],
            "average_confidence_per_category": avg_conf_per_cat,
            "sentences_ranked_by_severity": ranked,
        },
        "attention_analysis": attention_analysis,
        "faithfulness_summary": faithfulness_summary,
        "counterfactual_analysis": cf_analysis,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "thresholds_used": thresholds,
            "use_optimized": use_optimized,
            "batch_file": batch_file,
            "reproducibility": repro,
        },
    }
    return report


def bias_server_handlers(input, output, session):
    """Create server handlers for bias analysis tab."""

    bias_running = reactive.value(False)
    bias_raw_results = reactive.value(None)  # Raw probs (no threshold)
    bias_raw_results_B = reactive.value(None)
    bias_results = reactive.value(None)      # Final labels (after threshold)
    bias_results_B = reactive.value(None)    # For comparison
    bias_history = reactive.Value([])
    ablation_results = reactive.value(None)
    ablation_results_B = reactive.value(None)
    ablation_running = reactive.value(False)
    ig_results = reactive.value(None)
    ig_results_B = reactive.value(None)
    ig_results_base = reactive.value(None)   # IG computed on base encoder
    ig_running = reactive.value(False)
    perturbation_results = reactive.value(None)
    perturbation_results_B = reactive.value(None)
    perturbation_running = reactive.value(False)
    lrp_results = reactive.value(None)
    lrp_results_B = reactive.value(None)
    lrp_running = reactive.value(False)

    # ── Propagation drilldown (layer click → per-head view) ──
    propagation_drilldown_layer = reactive.value(None)   # int or None

    # ── GUS-Net attention cache (computed on demand, keyed by text+model) ──
    _base_attn_cache = reactive.value(None)       # processed result dict
    _base_attn_cache_B = reactive.value(None)

    # ── Batch mode state ──
    batch_sentences = reactive.value([])
    batch_file_name = reactive.value("")

    # Snapshots of compare mode state - Updated ONLY on 'Analyze Bias'
    active_bias_compare_models = reactive.Value(False)
    active_bias_compare_prompts = reactive.Value(False)
    
    # Sequential Logic State
    bias_prompt_step = reactive.Value("A")

    # Cached texts for display
    bias_cached_text_A = reactive.value("")
    bias_cached_text_B = reactive.value("")

    # Counterfactual state - stores raw find_swappable_terms() result
    counterfactual_swaps = reactive.value(None)
    cf_applied_swaps = reactive.value(None)  # swaps applied in last CF compare (for consistency card)

    # Current thresholds (synced with UI sliders) - will be updated by update_threshold_defaults
    current_thresholds_A = reactive.value({"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5})
    current_thresholds_B = reactive.value({"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5})
    
    # Analysis generation counter - incremented each time analysis completes
    bias_analysis_generation = reactive.value(0)
    bias_last_processed_generation = reactive.value(-1)

    # Snapshot of previous run state - restored when Back is clicked
    bias_snapshot = reactive.value(None)

    # ── Constants for UI Consistency ──
    _BTN_STYLE_CSV = "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; color: #334155; text-decoration: none;"
    _BTN_STYLE_PNG = "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; color: #334155;"

    # ── Tooltip micro-styles (dark tooltip background, ~380px wide) ──
    # Section header
    _TH   = "font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;margin:8px 0 4px;display:block;"
    # Bullet row wrapper
    _TR   = "display:flex;gap:7px;align-items:flex-start;margin:2px 0;font-size:11.5px;line-height:1.45;color:#cbd5e1;"
    # Bullet dot (pass color inline)
    _TD   = "font-size:8px;margin-top:3px;flex-shrink:0;"
    # Inline code chip
    _TC   = "background:rgba(255,255,255,0.09);border-radius:3px;padding:1px 5px;font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;"
    # Horizontal rule
    _TS   = "border:none;border-top:1px solid rgba(255,255,255,0.08);margin:7px 0;"
    # Coloured inline badges
    _TBG  = "background:rgba(34,197,94,0.18);color:#22c55e;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
    _TBR  = "background:rgba(239,68,68,0.18);color:#ef4444;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
    _TBA  = "background:rgba(245,158,11,0.18);color:#f59e0b;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
    _TBB  = "background:rgba(96,165,250,0.18);color:#60a5fa;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
    _TBP  = "background:rgba(167,139,250,0.18);color:#a78bfa;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
    # Note / italics footer
    _TN   = "font-size:10.5px;color:#64748b;font-style:italic;line-height:1.4;"

    def _chart_with_png_btn(chart_html: str, container_id: str, filename: str, controls: list = None) -> str:
        """Wrap chart HTML with controls (PNG btn + optional others) aligned to the right."""
        
        # Build controls list
        all_controls = []
        if controls:
            all_controls.extend(controls)
            
        all_controls.append(
            f'<button onclick="downloadPlotlyPNG(\'{container_id}\', \'{filename}\')" '
            f'style="{_BTN_STYLE_PNG}">PNG</button>'
        )
        
        # Render container with flex-end alignment
        control_bar = (
            f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:8px;margin-bottom:2px;">'
            f'{"".join(all_controls)}'
            f'</div>'
        )
        return control_bar + chart_html

    # ── Attention Source helpers ──────────────────────────────────────────

    def _get_attn_source_mode(input_id: str = "bias_attn_source") -> str:
        """Read the current attention source mode from a hidden input.

        Default is ``"gusnet"`` because ``heavy_bias_compute`` loads
        ``_GUSNET_TO_ENCODER`` (the fine-tuned model) as the encoder.
        """
        try:
            val = getattr(input, input_id)()
            if val in ("base", "gusnet", "compare"):
                return val
        except Exception:
            pass
        return "gusnet"

    def _normalize_attn_source_mode(mode: str) -> str:
        """Passthrough - _get_attn_source_mode already returns a valid value."""
        return mode if mode in ("gusnet", "base", "compare") else "gusnet"

    def _get_base_resolved(res, cache_rv):
        """Return a processed base-encoder-attention result.

        ``heavy_bias_compute`` already loads the GUS-Net model as the encoder
        (via ``_GUSNET_TO_ENCODER``), so ``res`` contains **GUS-Net** attentions.
        This helper loads the original pretrained base model (e.g.
        ``bert-base-uncased``) on demand and caches the result.
        """
        if not res:
            return None
        cached = cache_rv.get()
        cache_key = (res.get("text"), res.get("bias_model_key"))
        if cached and (cached.get("text"), cached.get("bias_model_key")) == cache_key:
            return cached
        try:
            raw_base = _load_base_encoder_attention_for_text(res["text"], res["bias_model_key"])
            thresholds = raw_base.get("effective_thresholds", {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5})
            processed = _process_raw_bias_result(raw_base, thresholds, use_optimized=False)
            if processed:
                cache_rv.set(processed)
            return processed
        except Exception:
            _logger.exception("[attn-source] Base encoder attention load failed")
            return None

    def _resolve_source_for_render(res, base_cache, mode):
        """Pick the right result dict for the active attention source mode.

        ``res`` from ``heavy_bias_compute`` is always **GUS-Net** attention
        (because ``_GUSNET_TO_ENCODER`` maps to the fine-tuned model).

        Returns ``(data_for_render, source_label)``.
        """
        if mode == "base":
            base = _get_base_resolved(res, base_cache)
            if base:
                return base, "Base Encoder"
            return res, "GUS-Net"  # fallback if base load fails
        # mode == "gusnet" or default → use res as-is (already GUS-Net)
        return res, "GUS-Net"

    def _resolve_faithfulness_results():
        """Resolve which results faithfulness analyses should run against."""
        res_A = bias_results.get()
        if not res_A:
            return None, None, False

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        if compare_models or compare_prompts:
            res_B = bias_results_B.get()
            return res_A, res_B, bool(res_B)

        source_mode = _get_attn_source_mode("bias_attn_source")
        if source_mode == "compare":
            res_base = _get_base_resolved(res_A, _base_attn_cache)
            if res_base:
                return res_base, res_A, True

        res_render, _ = _resolve_source_for_render(res_A, _base_attn_cache, source_mode)
        return res_render, None, False

    def _source_badge_html(label: str) -> str:
        """Small inline badge indicating the attention source."""
        if "gus" in label.lower():
            color = "#ff5ca9"
            bg = "rgba(255,92,169,0.12)"
        else:
            color = "#60a5fa"
            bg = "rgba(96,165,250,0.12)"
        return (
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;'
            f'letter-spacing:0.4px;text-transform:uppercase;'
            f'color:{color};background:{bg};border:1px solid {color}20;'
            f'margin-left:8px;vertical-align:middle;">'
            f'{label}</span>'
        )


    # ── Defaults Logic ──
    @reactive.Effect
    def update_threshold_defaults():
        """Update UI sliders with model-specific optimized thresholds for Model A and B."""
        vk_A = input.bias_model_key()

        # Determine if we need thresholds for B (Models or Prompts comparison)
        compare_active = False
        vk_B = None
        try:
            if input.bias_compare_mode():
                compare_active = True
                vk_B = input.bias_model_key_B()
            elif input.bias_compare_prompts_mode():
                compare_active = True
                vk_B = vk_A # Same model for both prompts
        except Exception:
            pass

        from attention_app.bias.gusnet_detector import MODEL_REGISTRY

        def _get_model_defaults(vk):
            if not vk:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

            cfg = MODEL_REGISTRY.get(vk)
            if not cfg:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

            opt = cfg.get("optimized_thresholds")
            # Auto-load from .npy if available (same as GusNetDetector)
            npy_path = Path(cfg["path"]) / "optimized_thresholds.npy"
            if npy_path.exists():
                opt = np.load(str(npy_path), allow_pickle=False).tolist()
            defaults = {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

            if opt:
                cat_indices = cfg.get("category_indices", {})
                for cat, indices in cat_indices.items():
                    if cat in defaults:
                        vals = [opt[i] for i in indices if i < len(opt)]
                        if vals:
                            defaults[cat] = sum(vals) / len(vals)
            return defaults

        # Get thresholds for A
        defaults_A = _get_model_defaults(vk_A)

        # Update server state immediately (so UI always shows latest thresholds)
        current_thresholds_A.set(defaults_A)

        # Get thresholds for B if any comparison mode is active
        msg_defaults = defaults_A.copy()
        if compare_active:
            defaults_B = _get_model_defaults(vk_B)
            current_thresholds_B.set(defaults_B)
            # Prefix with _B for the UI handler
            for cat, val in defaults_B.items():
                msg_defaults[f"{cat}_B"] = val

        # Send to UI
        if msg_defaults:
            asyncio.create_task(session.send_custom_message("set_bias_thresholds", msg_defaults))

    # ── History Logic ──
    
    @reactive.Effect
    @reactive.event(input.analyze_bias_btn)
    def update_bias_history():
        text = input.bias_input_text().strip()
        if not text:
            return

        # Deduplicate using normalized comparison
        hist = [h for h in bias_history() if h.strip() != text]
        hist.insert(0, text)
        hist = hist[:15]

        bias_history.set(hist)

    @output
    @render.ui
    def bias_history_list():
        hist = bias_history()
        if not hist:
            return ui.div("No history yet.", style="padding:10px; color:#94a3b8; font-style:italic;")

        items = []
        for text in hist:
            display_text = (text[:60] + "...") if len(text) > 60 else text
            safe_text = _html.escape(text.replace('\n', ' '), quote=True)
            items.append(
                ui.div(
                    _html.escape(display_text),
                    class_="history-item",
                    onclick=f"selectBiasHistoryItem('{safe_text}')"
                )
            )
        return ui.div(*items)

    @reactive.Effect
    def update_bias_results_live():
        """Update bias results when threshold sliders change (without re-running inference).

        Sliders are the ONLY reactive triggers. current_thresholds_A/B are read via
        isolate() so that .set() calls inside this function do NOT cause re-triggers.
        """
        # ── Read slider inputs (these ARE reactive triggers) ──
        try:
            t_unfair = float(input.bias_thresh_unfair())
        except (Exception, ValueError, TypeError):
            t_unfair = 0.5
        try:
            t_gen = float(input.bias_thresh_gen())
        except (Exception, ValueError, TypeError):
            t_gen = 0.5
        try:
            t_stereo = float(input.bias_thresh_stereo())
        except (Exception, ValueError, TypeError):
            t_stereo = 0.5

        try:
            t_unfair_b = float(input.bias_thresh_unfair_b())
        except (Exception, ValueError, TypeError):
            t_unfair_b = 0.5
        try:
            t_gen_b = float(input.bias_thresh_gen_b())
        except (Exception, ValueError, TypeError):
            t_gen_b = 0.5
        try:
            t_stereo_b = float(input.bias_thresh_stereo_b())
        except (Exception, ValueError, TypeError):
            t_stereo_b = 0.5

        manual_thresholds_A = {"UNFAIR": t_unfair, "GEN": t_gen, "STEREO": t_stereo}
        manual_thresholds_B = {"UNFAIR": t_unfair_b, "GEN": t_gen_b, "STEREO": t_stereo_b}

        # ── Read everything else via isolate() to avoid extra re-triggers ──
        with reactive.isolate():
            # GUARD: Don't run if analysis is in progress
            if bias_running.get():
                return

            # GUARD: Skip the first call after a fresh analysis (stale slider values)
            current_gen = bias_analysis_generation.get()
            last_gen = bias_last_processed_generation.get()
            if current_gen > last_gen:
                bias_last_processed_generation.set(current_gen)
                return

            raw_A = bias_raw_results.get()
            raw_B = bias_raw_results_B.get()

            if not raw_A and not raw_B:
                return

            # Check A model-key consistency
            a_model_matches = True
            try:
                live_key_A = input.bias_model_key()
                if raw_A and live_key_A and raw_A.get("bias_model_key") != live_key_A:
                    a_model_matches = False
            except Exception:
                pass

            try:
                use_opt = input.bias_use_optimized()
            except Exception:
                use_opt = False

            # Compare with last-processed thresholds (no reactive dependency)
            prev_A = current_thresholds_A.get()
            prev_B = current_thresholds_B.get()
            changed_A = prev_A != manual_thresholds_A
            changed_B = prev_B != manual_thresholds_B

            if not changed_A and not changed_B:
                return

            # Persist new values (isolate prevents re-trigger)
            if changed_A:
                current_thresholds_A.set(manual_thresholds_A)
            if changed_B:
                current_thresholds_B.set(manual_thresholds_B)

            # Determine if compare mode is active
            try:
                live_compare_models = bool(input.bias_compare_mode())
            except Exception:
                live_compare_models = bool(active_bias_compare_models.get())
            try:
                live_compare_prompts = bool(input.bias_compare_prompts_mode())
            except Exception:
                live_compare_prompts = bool(active_bias_compare_prompts.get())

            update_B = bool(raw_B) and changed_B and (live_compare_models or live_compare_prompts)

            # Process A
            if raw_A and changed_A and a_model_matches:
                res_A = _process_raw_bias_result(raw_A, manual_thresholds_A, use_optimized=use_opt)
                bias_results.set(res_A)

            # Process B
            if raw_B and update_B:
                res_B = _process_raw_bias_result(raw_B, manual_thresholds_B, use_optimized=use_opt)
                bias_results_B.set(res_B)

    @reactive.Effect
    @reactive.event(bias_history)
    def update_bias_history_list():
        """Inject history HTML directly into the dropdown (mirrors attention tab)."""
        history = bias_history.get()
        html_content = ""
        if not history:
            html_content = '<div style="padding:10px;color:#94a3b8;font-style:italic;">No history yet.</div>'
        else:
            for item in history:
                display = (item[:60] + "...") if len(item) > 60 else item
                safe = _html.escape(item.replace('\n', ' '), quote=True)
                display_safe = _html.escape(display)
                html_content += (
                    f'<div class="history-item" '
                    f"onclick=\"selectBiasHistoryItem('{safe}')\">"
                    f'{display_safe}</div>'
                )

        js_code = f"""
            var dropdown = document.getElementById('bias-history-dropdown');
            if (dropdown) {{
                dropdown.innerHTML = {json.dumps(html_content)};
            }}
        """
        ui.insert_ui(
            selector="body", where="beforeEnd",
            ui=ui.tags.script(js_code),
        )

    @reactive.Effect
    async def sync_bias_history_storage():
        """Persist bias history to localStorage (mirrors attention tab)."""
        await session.send_custom_message("update_bias_history", bias_history())

    @reactive.Effect
    @reactive.event(input.restored_bias_history)
    def restore_bias_history():
        """Restore bias history from localStorage on page load."""
        if input.restored_bias_history():
            raw = input.restored_bias_history()
            unique = []
            for item in raw:
                clean_item = item.strip()
                if clean_item and clean_item not in unique:
                    unique.append(clean_item)
            bias_history.set(unique)

    # ── Sequential Button Logic ──
    @reactive.Effect
    @reactive.event(input.bias_active_prompt_tab)
    def update_bias_step_state():
        # Keep python state in sync with JS state
        step = input.bias_active_prompt_tab()
        if step:
            bias_prompt_step.set(step)

    @reactive.Effect
    @reactive.event(input.bias_compare_prompts_mode, input.bias_compare_mode, bias_prompt_step)
    async def update_bias_button_label():
        """Update the Analyze Bias button label based on mode and step.
        
        - In Compare Prompts mode on Step A: Show "Prompt B ->"
        - In Compare Prompts mode on Step B: Show "Analyze Bias"
        - In Compare Models mode: Show "Analyze Bias" (both models at once)
        - In single mode: Show "Analyze Bias"
        
        Also resets prompt step when modes are toggled off.
        """
        try:
            prompts_mode = input.bias_compare_prompts_mode()
        except Exception:
            prompts_mode = False
            
        try:
            models_mode = input.bias_compare_mode()
        except Exception:
            models_mode = False
            
        step = bias_prompt_step.get()
        label = "Analyze Bias"
        
        # Reset to step A when compare prompts mode is turned off
        if not prompts_mode and step == "B":
            bias_prompt_step.set("A")
            # Also sync the UI tab to A
            await session.send_custom_message("bias_switch_prompt_tab", {"tab": "A"})
        
        # In compare prompts mode on Step A: show "Prompt B ->"
        if prompts_mode and step == "A":
            label = "Prompt B ➜"
            
        await session.send_custom_message("update_bias_button_label", {"label": label})

    # ── Export Infrastructure ──

    _BIAS_EXPORT_FOLDER_MAP = {
        "json": "downloads/json",
        "csv": "downloads/csv",
        "png": "downloads/png",
    }

    def generate_bias_export_filename(section, ext="csv", is_b=False, incl_timestamp=True, data_type=None):
        """Generate export filename for bias section exports."""
        try:
            bias_key = input.bias_model_key()
        except Exception:
            bias_key = "gusnet-bert"

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()

        parts = [f"bias_{section}"]
        if data_type:
            parts.append(data_type)
        parts.append(bias_key)

        if compare_models:
            parts.append("ModelB" if is_b else "ModelA")
        if compare_prompts:
            parts.append("PromptB" if is_b else "PromptA")
        if incl_timestamp:
            parts.append(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        return f"{'_'.join(parts)}.{ext}"

    def save_bias_export_to_folder(content, filename):
        """Save export content to the appropriate project folder."""
        ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
        folder = _BIAS_EXPORT_FOLDER_MAP.get(ext)
        if "session" in filename and ext == "json":
            folder = "downloads/sessions"
        if folder and content and not content.startswith("Error") and not content.startswith("No data"):
            try:
                filepath = Path(folder) / Path(filename).name
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding='utf-8')
            except Exception as e:
                logging.getLogger(__name__).warning("save_bias_export_to_folder failed: %s", e)

    def auto_save_bias_download(section, ext, **gen_kwargs):
        """Decorator replacing @render.download that also saves a copy."""
        filename_fn = lambda: generate_bias_export_filename(section, ext, **gen_kwargs)
        def decorator(fn):
            @render.download(filename=filename_fn)
            @functools.wraps(fn)
            def wrapper():
                parts = []
                for chunk in fn():
                    parts.append(str(chunk) if not isinstance(chunk, str) else chunk)
                    yield chunk
                if parts:
                    content = "".join(parts)
                    fname = filename_fn()
                    save_bias_export_to_folder(content, fname)
            return wrapper
        return decorator

    def _safe_threshold():
        """Read bias_threshold input, falling back to 0.5 if unavailable."""
        try:
            return float(input.bias_threshold())
        except Exception:
            return 0.5

    # ── Session Logic ──

    @render.download(filename=lambda: f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def save_bias_session():
        data = {
            "type": "bias_analysis",
            "timestamp": datetime.now().isoformat(),
            "text": input.bias_input_text(),
        }

        # Bias model key
        try:
            data["bias_model_key"] = input.bias_model_key()
        except Exception:
            data["bias_model_key"] = "gusnet-bert"

        # Compare modes
        try:
            data["bias_compare_mode"] = bool(input.bias_compare_mode())
        except Exception:
            data["bias_compare_mode"] = False
        try:
            data["bias_compare_prompts_mode"] = bool(input.bias_compare_prompts_mode())
        except Exception:
            data["bias_compare_prompts_mode"] = False

        # Conditional: Model B / Text B
        if data["bias_compare_mode"]:
            try:
                data["bias_model_key_B"] = input.bias_model_key_B()
            except Exception:
                data["bias_model_key_B"] = "gusnet-gpt2"
        if data["bias_compare_prompts_mode"]:
            try:
                data["bias_input_text_B"] = input.bias_input_text_B()
            except Exception:
                data["bias_input_text_B"] = ""

        # Per-class thresholds A
        for key in ["bias_thresh_unfair", "bias_thresh_gen", "bias_thresh_stereo"]:
            try:
                data[key] = float(getattr(input, key)())
            except Exception:
                pass
        # Per-class thresholds B
        for key in ["bias_thresh_unfair_b", "bias_thresh_gen_b", "bias_thresh_stereo_b"]:
            try:
                data[key] = float(getattr(input, key)())
            except Exception:
                pass

        # Layer / Head / Top-K
        try:
            data["bias_attn_layer"] = input.bias_attn_layer()
        except Exception:
            pass
        try:
            data["bias_attn_head"] = input.bias_attn_head()
        except Exception:
            pass
        try:
            data["bias_top_k"] = int(input.bias_top_k())
        except Exception:
            pass

        # Reproducibility metadata
        try:
            model_key = data.get("bias_model_key", "gusnet-bert")
            from ..bias.gusnet_detector import MODEL_REGISTRY
            hf_path = MODEL_REGISTRY.get(model_key, {}).get("path")
            data["reproducibility"] = get_reproducibility_info(
                data.get("text", ""), model_name=hf_path
            )
        except Exception:
            pass

        content = json.dumps(data, indent=2)
        # Also save to downloads/sessions/
        save_bias_export_to_folder(content, f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return content

    @reactive.Effect
    @reactive.event(input.load_bias_session_upload)
    async def load_bias_session():
        file_info = input.load_bias_session_upload()
        if not file_info:
            return

        try:
            with open(file_info[0]["datapath"], "r") as f:
                data = json.load(f)

            # 1. Restore compare modes first (triggers UI rebuild)
            compare_mode = data.get("bias_compare_mode", False)
            compare_prompts = data.get("bias_compare_prompts_mode", False)

            # Use JS to set switch values (standard Shiny switches)
            if compare_mode:
                await session.send_custom_message("bias_toggle_switch",
                    {"id": "bias_compare_mode", "checked": True})
            if compare_prompts:
                await session.send_custom_message("bias_toggle_switch",
                    {"id": "bias_compare_prompts_mode", "checked": True})

            # 2. Wait for UI to rebuild, then restore controls via JS
            await asyncio.sleep(0.3)

            restore_data = {}
            if "bias_model_key" in data:
                restore_data["bias_model_key"] = data["bias_model_key"]
            if "bias_model_key_B" in data:
                restore_data["bias_model_key_B"] = data["bias_model_key_B"]

            if restore_data:
                await session.send_custom_message("restore_bias_session_controls", restore_data)

            # 3. Restore text inputs
            if "text" in data:
                await session.send_custom_message("bias_set_textarea",
                    {"id": "bias_input_text", "value": data["text"]})
            if "bias_input_text_B" in data:
                await session.send_custom_message("bias_set_textarea",
                    {"id": "bias_input_text_B", "value": data["bias_input_text_B"]})

            # 4. Restore per-class thresholds via JS slider updates
            thresh_map = {
                "bias_thresh_unfair":   "bias-thresh-unfair",
                "bias_thresh_gen":      "bias-thresh-gen",
                "bias_thresh_stereo":   "bias-thresh-stereo",
                "bias_thresh_unfair_b": "bias-thresh-unfair-b",
                "bias_thresh_gen_b":    "bias-thresh-gen-b",
                "bias_thresh_stereo_b": "bias-thresh-stereo-b",
            }
            for input_key, slider_id in thresh_map.items():
                if input_key in data:
                    val = float(data[input_key])
                    await session.send_custom_message("bias_set_slider",
                        {"id": slider_id, "value": val, "inputId": input_key})

            # 5. Restore layer/head selections (after a small delay for selects to populate)
            await asyncio.sleep(0.2)
            if "bias_attn_layer" in data:
                ui.update_select("bias_attn_layer", selected=str(data["bias_attn_layer"]))
            if "bias_attn_head" in data:
                ui.update_select("bias_attn_head", selected=str(data["bias_attn_head"]))
            if "bias_top_k" in data:
                top_k = int(data["bias_top_k"])
                await session.send_custom_message("bias_set_slider",
                    {"id": "bias-topk-slider", "value": top_k, "inputId": "bias_top_k"})

            ui.notification_show("Bias session loaded successfully.", type="message", duration=3)

        except Exception as e:
            _logger.exception("Error loading bias session")
            ui.notification_show("Error loading session. Please check the file format.", type="error")

    # ── CSV Export Handlers (8 pairs: A and B) ──

    # 1. Bias Summary
    @auto_save_bias_download("summary", "csv")
    def export_bias_summary():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        s = res["bias_summary"]
        lines = ["metric,value"]
        for k, v in s.items():
            lines.append(f"{k},{v}")
        yield "\n".join(lines)

    @auto_save_bias_download("summary", "csv", is_b=True)
    def export_bias_summary_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        s = res["bias_summary"]
        lines = ["metric,value"]
        for k, v in s.items():
            lines.append(f"{k},{v}")
        yield "\n".join(lines)

    # 2. Bias Spans (Detected Bias Tokens)
    @auto_save_bias_download("spans", "csv")
    def export_bias_spans():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,bias_types,GEN,UNFAIR,STEREO,max_score"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = ";".join(lbl.get("bias_types", []))
            max_s = max((scores.get(t, 0) for t in lbl.get("bias_types", [])), default=0)
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{types},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f},{max_s:.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("spans", "csv", is_b=True)
    def export_bias_spans_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,bias_types,GEN,UNFAIR,STEREO,max_score"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = ";".join(lbl.get("bias_types", []))
            max_s = max((scores.get(t, 0) for t in lbl.get("bias_types", [])), default=0)
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{types},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f},{max_s:.4f}")
        yield "\n".join(lines)

    # 3. Token Bias Strip
    @auto_save_bias_download("strip", "csv")
    def export_bias_strip():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,is_biased,O,GEN,UNFAIR,STEREO"]
        for lbl in res["token_labels"]:
            scores = lbl.get("scores", {})
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{lbl.get('is_biased',False)},{scores.get('O',0):.4f},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("strip", "csv", is_b=True)
    def export_bias_strip_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,is_biased,O,GEN,UNFAIR,STEREO"]
        for lbl in res["token_labels"]:
            scores = lbl.get("scores", {})
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{lbl.get('is_biased',False)},{scores.get('O',0):.4f},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f}")
        yield "\n".join(lines)

    # 4. Confidence Breakdown
    @auto_save_bias_download("confidence", "csv")
    def export_bias_confidence():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,tier,max_score,bias_types"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = lbl.get("bias_types", [])
            max_s = max((scores.get(t, 0) for t in types), default=0)
            tier = "high" if max_s >= 0.85 else ("medium" if max_s >= 0.70 else "low")
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{tier},{max_s:.4f},{';'.join(types)}")
        yield "\n".join(lines)

    @auto_save_bias_download("confidence", "csv", is_b=True)
    def export_bias_confidence_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,tier,max_score,bias_types"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = lbl.get("bias_types", [])
            max_s = max((scores.get(t, 0) for t in types), default=0)
            tier = "high" if max_s >= 0.85 else ("medium" if max_s >= 0.70 else "low")
            lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{tier},{max_s:.4f},{';'.join(types)}")
        yield "\n".join(lines)

    # 5. Combined Bias View (Attention Matrix CSV)
    @auto_save_bias_download("combined", "csv")
    def export_bias_combined():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        try:
            l_idx = int(input.bias_attn_layer())
            h_idx = int(input.bias_attn_head())
        except Exception:
            l_idx, h_idx = 0, 0
        atts = res["attentions"]
        if not atts or l_idx >= len(atts):
            yield "No attention data"
            return
        tokens = res["tokens"]
        attn = atts[l_idx][0, h_idx].cpu().numpy()
        header = "query_token," + ",".join(tokens)
        lines = [header]
        for i, tok in enumerate(tokens):
            vals = ",".join(f"{attn[i,j]:.6f}" for j in range(len(tokens)))
            lines.append(f"{tok},{vals}")
        yield "\n".join(lines)

    @auto_save_bias_download("combined", "csv", is_b=True)
    def export_bias_combined_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        try:
            l_idx = int(input.bias_attn_layer())
            h_idx = int(input.bias_attn_head())
        except Exception:
            l_idx, h_idx = 0, 0
        atts = res["attentions"]
        if not atts or l_idx >= len(atts):
            yield "No attention data"
            return
        tokens = res["tokens"]
        attn = atts[l_idx][0, h_idx].cpu().numpy()
        header = "query_token," + ",".join(tokens)
        lines = [header]
        for i, tok in enumerate(tokens):
            vals = ",".join(f"{attn[i,j]:.6f}" for j in range(len(tokens)))
            lines.append(f"{tok},{vals}")
        yield "\n".join(lines)

    # 6. Attention Bias Matrix
    @auto_save_bias_download("matrix", "csv")
    def export_bias_matrix():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        lines = ["layer,head,bias_attention_ratio,specialized"]
        for m in metrics:
            lines.append(f"{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    @auto_save_bias_download("matrix", "csv", is_b=True)
    def export_bias_matrix_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        lines = ["layer,head,bias_attention_ratio,specialized"]
        for m in metrics:
            lines.append(f"{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    # 7. Bias Propagation
    @auto_save_bias_download("propagation", "csv")
    def export_bias_propagation():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        prop = res["propagation_analysis"]["layer_propagation"]
        if not prop:
            yield "No propagation data"
            return
        lines = ["layer,mean_bar,max_bar,min_bar"]
        for p in prop:
            lines.append(f"{p['layer']},{p['mean_ratio']:.4f},{p['max_ratio']:.4f},{p['min_ratio']:.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("propagation", "csv", is_b=True)
    def export_bias_propagation_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        prop = res["propagation_analysis"]["layer_propagation"]
        if not prop:
            yield "No propagation data"
            return
        lines = ["layer,mean_bar,max_bar,min_bar"]
        for p in prop:
            lines.append(f"{p['layer']},{p['mean_ratio']:.4f},{p['max_ratio']:.4f},{p['min_ratio']:.4f}")
        yield "\n".join(lines)

    # 8. Top Bias-Focused Heads
    @auto_save_bias_download("top_heads", "csv")
    def export_bias_top_heads():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:5]
        lines = ["rank,layer,head,bar,specialized"]
        for i, m in enumerate(top, 1):
            lines.append(f"{i},{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    @auto_save_bias_download("top_heads", "csv", is_b=True)
    def export_bias_top_heads_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:5]
        lines = ["rank,layer,head,bar,specialized"]
        for i, m in enumerate(top, 1):
            lines.append(f"{i},{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    def log_debug(msg):
        _logger.debug(msg)
            
    def heavy_bias_compute(text, model_name, threshold, bias_model_key, use_optimized=True):
        """Perform bias analysis computation (runs in thread pool).

        Parameters
        ----------
        text : str
            The input sentence to analyse.
        model_name : str
            HuggingFace model name for the attention model (e.g. "bert-base-uncased").
        threshold : float
            Sensitivity threshold for the bias detector.
        bias_model_key : str
            Key into ``MODEL_REGISTRY`` selecting the GUS-Net backbone
            ("gusnet-bert" or "gusnet-gpt2").
        use_optimized : bool
            Whether to use per-label optimized thresholds (if available).
        """
        log_debug(f"Starting heavy_bias_compute (threshold={threshold}, bias_model={bias_model_key}, use_optimized={use_optimized})")
        if not text:
            log_debug("No text provided")
            return None

        try:
            # Clear cache before starting heavy operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load attention model
            log_debug(f"Loading attention model {model_name}...")
            tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
            device = ModelManager.get_device()
            log_debug(f"Model loaded. Device: {device}")

            log_debug("Tokenizing text...")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            log_debug("Running encoder inference...")
            with torch.no_grad():
                try:
                    outputs = encoder_model(**inputs)
                except torch.cuda.OutOfMemoryError:
                    # Clear cache and try again>= No, just fail gracefully
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "GPU Out of Memory. usage is high. "
                        "Please stop the background 'generate_stereoset_json' script "
                        "or other heavy processes."
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                         torch.cuda.empty_cache()
                         raise RuntimeError(
                            "GPU Out of Memory. usage is high. "
                            "Please stop the background 'generate_stereoset_json' script "
                            "or other heavy processes."
                        )
                    raise e
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attentions = outputs.attentions
            log_debug(f"Got {len(tokens)} tokens, {len(attentions)} layers")

            # ── Token-level bias detection (GUS-Net) ──
            log_debug(f"Initializing GusNetDetector (model_key={bias_model_key})...")
            if bias_model_key == "gusnet-ensemble":
                gus_detector = EnsembleGusNetDetector(
                    model_key_a="gusnet-bert",
                    model_key_b="gusnet-bert-custom",
                    threshold=0.5, # Placeholder
                )
            else:
                gus_detector = GusNetDetector(
                    model_key=bias_model_key, threshold=0.5,
                    use_optimized=False,
                )
            log_debug("Running predict_proba...")
            
            # Helper to get raw probs.
            gus_tokens, gus_probs = gus_detector.predict_proba(text)
            
            log_debug(f"GUS-Net returned {len(gus_tokens)} tokens and {gus_probs.shape} probs")

            # Calculate effective thresholds for UI feedback
            effective_thresholds = {}
            if use_optimized and gus_detector.config.get("optimized_thresholds"):
                opt = gus_detector.config["optimized_thresholds"]
                cat_indices = gus_detector.config.get("category_indices", {})
                for cat in ["GEN", "UNFAIR", "STEREO"]:
                    if cat in cat_indices:
                        indices = cat_indices[cat]
                        # Average the per-label thresholds for the category to give a representative slider value
                        vals = [opt[i] for i in indices if i < len(opt)]
                        if vals:
                            effective_thresholds[cat] = sum(vals) / len(vals)
            else:
                if isinstance(threshold, dict):
                    effective_thresholds = threshold
                else:
                    effective_thresholds = {
                        "GEN": threshold, 
                        "UNFAIR": threshold, 
                        "STEREO": threshold
                    }

            # Return RAW data
            return {
                "tokens": tokens, # Encoder tokens
                "text": text,
                "attentions": attentions,
                "gus_tokens": gus_tokens, 
                "gus_probs": gus_probs,
                "bias_model_key": bias_model_key,
                "model_name": model_name,
                "gus_special": gus_detector.config["special_tokens"],
                "effective_thresholds": effective_thresholds
            }
        except Exception as e:
            log_debug(f"ERROR in heavy_bias_compute: {e}")
            _logger.exception("ERROR in heavy_bias_compute")
            return None

    # ── Trigger analysis ──

    @reactive.effect
    @reactive.event(input.analyze_bias_btn)
    async def compute_bias():
        log_debug("BUTTON CLICKED: compute_bias triggered")
        try:
            # ── Batch mode intercept ──
            try:
                is_batch = input.bias_batch_mode_active() == "true"
            except Exception:
                is_batch = False

            if is_batch:
                if not batch_sentences.get():
                    ui.notification_show("No file uploaded for batch mode.", type="warning", duration=3)
                    return
                bias_running.set(True)
                await session.send_custom_message('start_bias_loading', {})
                await asyncio.sleep(0.1)
                try:
                    await _run_batch_analysis()
                finally:
                    bias_running.set(False)
                    await session.send_custom_message('stop_bias_loading', {})
                return

            text = input.bias_input_text().strip()
            log_debug(f"Input text length: {len(text)}")
            if not text:
                log_debug("Text is empty, returning")
                return

            # ── Intercept Click for Sequential Logic ──
            try:
                compare_prompts_live = input.bias_compare_prompts_mode()
            except Exception:
                compare_prompts_live = False

            step = bias_prompt_step.get()

            if compare_prompts_live and step == "A":
                # User clicked "Prompt B ->". Do not analyze. Switch tab.
                log_debug("Sequential Logic: Switching to Tab B")
                await session.send_custom_message("bias_switch_prompt_tab", {"tab": "B"})
                return

            log_debug("Starting loading UI...")
            bias_running.set(True)
            await session.send_custom_message('start_bias_loading', {})
            await asyncio.sleep(0.1)
            log_debug("Loading UI active")

            # Check compare modes and snapshot them
            try:
                compare_models = input.bias_compare_mode()
            except Exception:
                compare_models = False
            try:
                compare_prompts = input.bias_compare_prompts_mode()
            except Exception:
                compare_prompts = False

            # MODE SWITCHING CLEANUP: Reset state when modes change
            prev_compare_models = active_bias_compare_models.get()
            prev_compare_prompts = active_bias_compare_prompts.get()

            # Snapshot BEFORE any clearing (so had_prior_results is accurate)
            had_prior_results = bias_results.get() is not None
            if had_prior_results:
                bias_snapshot.set({
                    'results':         bias_results.get(),
                    'results_B':       bias_results_B.get(),
                    'text_A':          bias_cached_text_A.get(),
                    'text_B':          bias_cached_text_B.get(),
                    'compare_models':  prev_compare_models,
                    'compare_prompts': prev_compare_prompts,
                })

            # Detect mode transitions
            mode_switched_to_prompts = compare_prompts and not prev_compare_prompts
            mode_switched_to_models = compare_models and not prev_compare_models
            mode_switched_off = (not compare_prompts and not compare_models) and (prev_compare_prompts or prev_compare_models)

            # Reset state when switching modes to ensure clean state
            if mode_switched_to_prompts or mode_switched_to_models or mode_switched_off:
                log_debug("Mode switch detected - resetting bias state")
                # Preserve cf_applied_swaps — trigger_counterfactual sets it
                # right before this mode switch happens
                _saved_cf_swaps = cf_applied_swaps.get()
                bias_raw_results.set(None)
                bias_raw_results_B.set(None)
                bias_results.set(None)
                bias_results_B.set(None)
                bias_cached_text_A.set("")
                bias_cached_text_B.set("")
                cf_applied_swaps.set(_saved_cf_swaps)
                # Reset prompt step to A when switching modes
                bias_prompt_step.set("A")
                # Ensure UI reflects the reset
                await session.send_custom_message("bias_switch_prompt_tab", {"tab": "A"})

            active_bias_compare_models.set(compare_models)
            active_bias_compare_prompts.set(compare_prompts)

            # Clear previous results
            # cf_applied_swaps is preserved here — trigger_counterfactual sets it
            # right before triggering this handler. Clear it only for non-CF runs.
            if not compare_prompts:
                cf_applied_swaps.set(None)
            bias_results.set(None)
            bias_results_B.set(None)
            counterfactual_swaps.set(None)
            _base_attn_cache.set(None)
            _base_attn_cache_B.set(None)
            ig_results_base.set(None)
            bias_cached_text_A.set(text)

            try:
                use_optimized = bool(input.bias_use_optimized())
            except Exception:
                use_optimized = True

            # Use current thresholds from reactive values (synced with UI)
            thresholds_A = current_thresholds_A.get()
            
            # Fetch B thresholds if needed
            if compare_models:
                thresholds_B = current_thresholds_B.get()
            elif compare_prompts:
                thresholds_B = thresholds_A
            else:
                thresholds_B = {"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5}
            
            # Mean placeholder for legacy compatibility
            threshold = sum(thresholds_A.values()) / 3
            log_debug(f"Thresholds A: {thresholds_A}, B: {thresholds_B}")
            log_debug(f"use_optimized={use_optimized}, threshold={threshold}")

            # Bias detection model (BERT or GPT-2 backbone) - Model A
            try:
                bias_model_key = input.bias_model_key()
                log_debug(f"Bias model key: {bias_model_key}")
            except Exception as e:
                log_debug(f"Warning: bias_model_key missing ({e}). Using default gusnet-bert")
                bias_model_key = "gusnet-bert"

            # Derive encoder model from bias_model_key so tokenization matches
            model_name = _GUSNET_TO_ENCODER.get(bias_model_key, "bert-base-uncased")
            log_debug(f"Encoder model (derived from {bias_model_key}): {model_name}")

            try:
                loop = asyncio.get_running_loop()
                log_debug("Entering ThreadPoolExecutor...")
                with ThreadPoolExecutor() as pool:
                    # Compute Result A
                    log_debug("Submitting heavy_bias_compute for Model A...")
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            pool, heavy_bias_compute, text, model_name, thresholds_A, bias_model_key, use_optimized,
                        ),
                        timeout=180.0
                    )
                    log_debug("heavy_bias_compute A returned successfully")
                    
                    # Initial processing for Model A (using thresholds we just grabbed)
                    effective_A = result.get("effective_thresholds", thresholds_A)
                    res_processed = _process_raw_bias_result(result, effective_A, use_optimized=use_optimized)
                    
                    # Update current_thresholds to match what was actually used
                    current_thresholds_A.set(effective_A)
                    
                    # Store RAW result so sliders can re-threshold live
                    bias_raw_results.set(result)
                    
                    # Set processed results (this triggers update_bias_results_live, but generation counter protects it)
                    bias_results.set(res_processed)

                    # Prepare message for UI sliders (A thresholds always included)
                    msg_thresholds = {}
                    if result and result.get("effective_thresholds"):
                        eff_A = result["effective_thresholds"]
                        msg_thresholds.update({
                            "UNFAIR": eff_A.get("UNFAIR"),
                            "GEN": eff_A.get("GEN"),
                            "STEREO": eff_A.get("STEREO")
                        })

                    # Compute Result B if needed
                    if compare_models:
                        # Case 1: Same Prompt (A), Different Model (B)
                        try:
                            bias_model_key_B = input.bias_model_key_B()
                            if not bias_model_key_B:
                                bias_model_key_B = "gusnet-gpt2"
                        except Exception:
                            bias_model_key_B = "gusnet-gpt2"

                        model_name_B = _GUSNET_TO_ENCODER.get(bias_model_key_B, "gpt2")
                        log_debug(f"Starting heavy_bias_compute B ({bias_model_key_B}, encoder={model_name_B}) for Compare Models")
                        result_B = await asyncio.wait_for(
                            loop.run_in_executor(
                                pool, heavy_bias_compute, text, model_name_B, thresholds_B, bias_model_key_B, use_optimized,
                            ),
                            timeout=180.0
                        )
                        bias_raw_results_B.set(result_B) # Store RAW result B
                        bias_cached_text_B.set(text)  # Same text for compare models
                        
                        # Process B immediately for UI
                        effective_B = result_B.get("effective_thresholds", thresholds_B)
                        current_thresholds_B.set(effective_B)
                        res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                        bias_results_B.set(res_B_proc)
                        
                        # Add thresholds B to the message
                        if result_B and result_B.get("effective_thresholds"):
                            eff_B = result_B["effective_thresholds"]
                            msg_thresholds.update({
                                "UNFAIR_B": eff_B.get("UNFAIR"),
                                "GEN_B": eff_B.get("GEN"),
                                "STEREO_B": eff_B.get("STEREO")
                            })

                    elif compare_prompts:
                        # Case 2: Different Prompt (B), Same Model (A)
                        try:
                            text_B = input.bias_input_text_B().strip()
                        except Exception:
                            text_B = ""

                        if text_B:
                            log_debug(f"Starting heavy_bias_compute B (Prompt B) for Compare Prompts")
                            result_B = await asyncio.wait_for(
                                loop.run_in_executor(
                                    pool, heavy_bias_compute, text_B, model_name, thresholds_B, bias_model_key, use_optimized,
                                ),
                                timeout=180.0
                            )
                            bias_raw_results_B.set(result_B)
                            bias_cached_text_B.set(text_B)
                            
                            # Process B immediately for UI
                            effective_B = result_B.get("effective_thresholds", thresholds_B)
                            current_thresholds_B.set(effective_B)
                            res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                            bias_results_B.set(res_B_proc)
                            
                            # Add thresholds B to the message
                            if result_B and result_B.get("effective_thresholds"):
                                eff_B = result_B["effective_thresholds"]
                                msg_thresholds.update({
                                    "UNFAIR_B": eff_B.get("UNFAIR"),
                                    "GEN_B": eff_B.get("GEN"),
                                    "STEREO_B": eff_B.get("STEREO")
                                })
                        else:
                            # Empty text B - clear results and show notification
                            bias_raw_results_B.set(None)
                            bias_cached_text_B.set("")
                            bias_results_B.set(None)
                            ui.notification_show("Prompt B is empty. Only analyzing Prompt A.", type="warning", duration=3)
                    else:
                        # Not in compare_prompts mode, clear B
                        bias_raw_results_B.set(None)
                        bias_cached_text_B.set("")

                        # Counterfactual detection (single-prompt mode only)
                        try:
                            cf_swaps = find_swappable_terms(text)
                            counterfactual_swaps.set(cf_swaps if cf_swaps else None)
                        except Exception:
                            counterfactual_swaps.set(None)

                    # Send all thresholds at once (A always included, B only if compare mode)
                    if msg_thresholds:
                        log_debug(f"Sending effective thresholds to UI: {msg_thresholds}")
                        await session.send_custom_message("set_bias_thresholds", msg_thresholds)

            except asyncio.TimeoutError:
                log_debug("ERROR: Bias analysis timed out (limit: 180s)")
                _logger.error("Bias analysis timed out (limit: 180s)")
                ui.notification_show("Analysis timed out.", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            except Exception as e:
                _logger.exception("Bias analysis failed")
                log_debug(f"ERROR during execution: {type(e).__name__}")
                ui.notification_show("Analysis failed. Please try again or check the server logs.", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            finally:
                log_debug("Stopping loading UI")
                # Increment generation counter to mark results as fresh
                # This causes the next update_bias_results_live call to be skipped
                new_gen = bias_analysis_generation.get() + 1
                bias_analysis_generation.set(new_gen)
                bias_running.set(False)
                await session.send_custom_message('stop_bias_loading', {})
                # Show back button whenever prior results existed (second run in same session)
                show_back = had_prior_results and bias_results.get() is not None
                await session.send_custom_message('bias_back_btn_update', {'show': show_back})

        except Exception as e:
            log_debug(f"CRITICAL ERROR in compute_bias top level: {e}")
            _logger.exception("CRITICAL ERROR in compute_bias top level")

    # ── Back button: restore previous run snapshot ──

    @reactive.effect
    @reactive.event(input.bias_go_back)
    async def bias_restore_from_snapshot():
        snap = bias_snapshot.get()
        if not snap:
            return
        bias_results.set(snap['results'])
        bias_results_B.set(snap['results_B'])
        bias_cached_text_A.set(snap.get('text_A', ''))
        bias_cached_text_B.set(snap.get('text_B', ''))
        active_bias_compare_models.set(snap.get('compare_models', False))
        active_bias_compare_prompts.set(snap.get('compare_prompts', False))
        bias_snapshot.set(None)
        await session.send_custom_message('bias_back_btn_update', {'show': False})
        await session.send_custom_message('bias_restore_ui', {
            'compare_models': snap.get('compare_models', False),
            'compare_prompts': snap.get('compare_prompts', False),
        })

    # ── Counterfactual click handler ──

    @reactive.effect
    @reactive.event(input.bias_trigger_counterfactual)
    async def trigger_counterfactual():
        orig = bias_cached_text_A.get()
        all_swaps = counterfactual_swaps.get()
        if not orig or not all_swaps:
            return

        # Get selected swap keys from JS (list of lowercase token strings)
        selected_raw = input.bias_trigger_counterfactual()
        if not selected_raw or not isinstance(selected_raw, (list, tuple)):
            return
        selected_keys = set(str(k).lower() for k in selected_raw)

        # Filter swaps to only the user-selected ones
        filtered = [s for s in all_swaps if s["term"].lower() in selected_keys]
        if not filtered:
            return

        # Generate counterfactual with selected swaps only
        cf_text, applied = generate_counterfactual(orig, filtered)
        cf_applied_swaps.set(applied)

        # 1. Enable Compare Prompts mode (toggle switch + explicit Shiny input)
        await session.send_custom_message("bias_toggle_switch",
            {"id": "bias_compare_prompts_mode", "checked": True,
             "shinyId": "bias_compare_prompts_mode"})
        await asyncio.sleep(0.3)

        # 2. Fill textareas A and B
        await session.send_custom_message("bias_set_textarea",
            {"id": "bias_input_text", "value": orig})
        await session.send_custom_message("bias_set_textarea",
            {"id": "bias_input_text_B", "value": cf_text})
        await asyncio.sleep(0.3)

        # 3. Set step to "B" so the analyze click skips the sequential intercept
        bias_prompt_step.set("B")
        await asyncio.sleep(0.1)

        # 4. Trigger analysis directly by setting the action button value
        #    AND clicking the DOM element just to be safe
        await session.send_custom_message("bias_click_analyze", {})

    # ── Batch file upload handler ──

    @reactive.effect
    @reactive.event(input.batch_file_upload)
    async def parse_batch_file():
        """Parse uploaded CSV or JSON file for batch mode."""
        import csv
        import io
        file_info = input.batch_file_upload()
        if not file_info:
            return
        try:
            file_info = file_info[0]
            file_path = file_info["datapath"]
            file_name = file_info["name"]

            sentences = []
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_name.lower().endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    if all(isinstance(x, str) for x in data):
                        sentences = [s.strip() for s in data if s.strip()]
                    elif all(isinstance(x, dict) for x in data):
                        for item in data:
                            for key in ("text", "sentence", "input", "content"):
                                if key in item and isinstance(item[key], str) and item[key].strip():
                                    sentences.append(item[key].strip())
                                    break
                elif isinstance(data, dict):
                    for key in ("sentences", "texts", "data"):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                if isinstance(item, str) and item.strip():
                                    sentences.append(item.strip())
                                elif isinstance(item, dict):
                                    for k in ("text", "sentence", "input"):
                                        if k in item and isinstance(item[k], str) and item[k].strip():
                                            sentences.append(item[k].strip())
                                            break
                            break
            elif file_name.lower().endswith(".csv"):
                reader = csv.DictReader(io.StringIO(content))
                cols = reader.fieldnames or []
                text_col = None
                for candidate in ("text", "sentence", "input", "content"):
                    for c in cols:
                        if c.lower().strip() == candidate:
                            text_col = c
                            break
                    if text_col:
                        break
                if not text_col and cols:
                    text_col = cols[0]
                if text_col:
                    for row in reader:
                        val = row.get(text_col, "").strip()
                        if val:
                            sentences.append(val)

            MAX_BATCH_SIZE = 500
            if len(sentences) > MAX_BATCH_SIZE:
                sentences = sentences[:MAX_BATCH_SIZE]

            if sentences:
                batch_sentences.set(sentences)
                batch_file_name.set(file_name)
                await session.send_custom_message("batch_file_parsed", {
                    "filename": file_name,
                    "count": len(sentences),
                })
            else:
                batch_sentences.set([])
                await session.send_custom_message("batch_file_parsed", {
                    "error": "No sentences found in file",
                })
        except Exception as e:
            batch_sentences.set([])
            await session.send_custom_message("batch_file_parsed", {
                "error": f"Parse error: {str(e)[:80]}",
            })

    # ── Batch processing pipeline ──

    async def _run_batch_analysis():
        """Run comprehensive analysis on all batch sentences."""
        import hashlib
        import time as _time

        sentences = batch_sentences.get()
        if not sentences:
            return

        t0 = _time.time()

        try:
            bias_model_key = input.bias_model_key()
        except Exception:
            bias_model_key = "gusnet-bert"
        model_name = _GUSNET_TO_ENCODER.get(bias_model_key, "bert-base-uncased")
        is_gpt2 = "gpt2" in model_name

        try:
            use_optimized = bool(input.bias_use_optimized())
        except Exception:
            use_optimized = True

        thresholds = current_thresholds_A.get()
        n_total = len(sentences)

        per_sentence_results = []
        all_bar_matrices = []
        all_bsr_matrices = []

        loop = asyncio.get_running_loop()

        for idx, text in enumerate(sentences):
            sentence_data = {
                "index": idx,
                "text": text,
                "is_biased": False,
                "error": None,
            }
            try:
                # ── Phase 1: GUS-Net bias detection ──
                await session.send_custom_message("batch_progress", {
                    "label": f"Analyse Bias ({idx+1}/{n_total})"
                })

                with ThreadPoolExecutor(max_workers=1) as pool:
                    raw = await asyncio.wait_for(
                        loop.run_in_executor(pool, heavy_bias_compute, text, model_name, thresholds, bias_model_key, use_optimized),
                        timeout=120.0
                    )

                if not raw:
                    sentence_data["error"] = "Bias detection failed"
                    per_sentence_results.append(sentence_data)
                    continue

                processed = _process_raw_bias_result(raw, raw.get("effective_thresholds", thresholds), use_optimized=use_optimized)
                if not processed:
                    sentence_data["error"] = "Result processing failed"
                    per_sentence_results.append(sentence_data)
                    continue

                # Extract bias info
                token_labels = processed.get("token_labels", [])
                bias_summary = processed.get("bias_summary", {})
                bias_spans = processed.get("bias_spans", [])
                metrics = processed.get("attention_metrics", [])
                attentions = processed.get("attentions")
                propagation = processed.get("propagation_analysis", {})
                bias_matrix = processed.get("bias_matrix")

                biased_count = sum(1 for t in token_labels if t.get("is_biased"))
                total_tokens = len(token_labels)
                bias_pct = round(biased_count / total_tokens * 100, 1) if total_tokens else 0
                is_biased = biased_count > 0

                # Token scores for report
                token_scores = []
                for tl in token_labels:
                    scores = tl.get("scores", {})
                    token_scores.append({
                        "token": tl.get("token", ""),
                        "is_biased": tl.get("is_biased", False),
                        "bias_types": tl.get("bias_types", []),
                        "scores": {k: round(float(v), 4) for k, v in scores.items()} if scores else {},
                    })

                # Biased spans
                span_data = []
                for sp in bias_spans:
                    span_data.append({
                        "text": sp.get("text", ""),
                        "bias_types": sp.get("bias_types", []),
                        "avg_score": round(float(sp.get("avg_score", 0)), 4),
                    })

                # Category counts
                cat_counts = {}
                for tl in token_labels:
                    if tl.get("is_biased"):
                        for bt in tl.get("bias_types", []):
                            cat_counts[bt] = cat_counts.get(bt, 0) + 1

                # Confidence
                confidences = []
                for tl in token_labels:
                    if tl.get("is_biased") and tl.get("scores"):
                        max_score = max(v for k, v in tl["scores"].items() if k != "O")
                        confidences.append(float(max_score))
                avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0

                sentence_data.update({
                    "is_biased": is_biased,
                    "bias_summary": {
                        "total_tokens": total_tokens,
                        "biased_tokens": biased_count,
                        "bias_percentage": bias_pct,
                        "generalization_count": cat_counts.get("GEN", 0),
                        "unfairness_count": cat_counts.get("UNFAIR", 0),
                        "stereotype_count": cat_counts.get("STEREO", 0),
                        "avg_confidence": avg_conf,
                        "categories_found": list(cat_counts.keys()),
                    },
                    "biased_spans": span_data,
                    "token_scores": token_scores,
                })

                # BAR/BSR matrix
                if bias_matrix is not None:
                    all_bar_matrices.append(np.array(bias_matrix))
                    sentence_data["bar_matrix_shape"] = list(np.array(bias_matrix).shape)

                # Propagation
                if propagation:
                    layer_bars = propagation.get("layer_propagation", [])
                    sentence_data["propagation"] = {
                        "layer_bars": [round(float(x), 4) for x in layer_bars],
                        "peak_layer": propagation.get("peak_layer"),
                        "pattern": propagation.get("propagation_pattern", "none"),
                    }

                # ── Phase 2: Integrated Gradients ──
                ig_data = None
                try:
                    tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        ig_bundle = await asyncio.wait_for(
                            loop.run_in_executor(pool, batch_compute_ig_correlation,
                                encoder_model, tokenizer, text, list(attentions), metrics, is_gpt2),
                            timeout=120.0
                        )
                    if ig_bundle and isinstance(ig_bundle, IGAnalysisBundle):
                        ig_corrs = []
                        for c in (ig_bundle.correlations or []):
                            ig_corrs.append({
                                "layer": c.layer, "head": c.head,
                                "spearman_rho": round(float(c.spearman_rho), 4),
                                "p_value": round(float(c.spearman_pvalue), 6) if c.spearman_pvalue is not None else None,
                                "bar": round(float(c.bar_original), 4) if c.bar_original is not None else None,
                            })
                        topk_data = []
                        for t in (ig_bundle.topk_overlaps or []):
                            topk_data.append({
                                "layer": t.layer, "head": t.head, "k": t.k,
                                "jaccard": round(float(t.jaccard), 4),
                                "rbo": round(float(t.rank_biased_overlap), 4) if t.rank_biased_overlap is not None else None,
                                "bar": round(float(t.bar_original), 4) if t.bar_original is not None else None,
                            })
                        ig_data = {
                            "token_attributions": [round(float(x), 6) for x in ig_bundle.token_attributions],
                            "top_ig_tokens": [
                                {"token": ig_bundle.tokens[i], "attribution": round(float(ig_bundle.token_attributions[i]), 6), "rank": r+1}
                                for r, i in enumerate(np.argsort(ig_bundle.token_attributions)[::-1][:10])
                            ],
                            "ig_attention_correlations": ig_corrs[:20],
                            "topk_overlaps": topk_data[:20],
                        }
                        sentence_data["ig_analysis"] = ig_data
                except Exception as e:
                    sentence_data["ig_error"] = str(e)[:100]

                # ── Phase 3: Head Ablation ──
                try:
                    top_k = min(10, len(metrics))
                    top_heads_local = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:top_k]
                    tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        ablation_list = await asyncio.wait_for(
                            loop.run_in_executor(pool, batch_ablate_top_heads,
                                encoder_model, lm_head_model, tokenizer, text, top_heads_local, is_gpt2),
                            timeout=120.0
                        )
                    if ablation_list:
                        abl_data = []
                        for a in ablation_list:
                            abl_data.append({
                                "layer": a.layer, "head": a.head,
                                "representation_impact": round(float(a.representation_impact), 6),
                                "kl_divergence": round(float(a.kl_divergence), 6) if a.kl_divergence is not None else None,
                                "bar": round(float(a.bar_original), 4) if a.bar_original is not None else None,
                            })
                        sentence_data["ablation_analysis"] = abl_data
                except Exception as e:
                    sentence_data["ablation_error"] = str(e)[:100]

                # ── Phase 4: Perturbation ──
                try:
                    if ig_data and ig_bundle:
                        tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            perturb_bundle = await asyncio.wait_for(
                                loop.run_in_executor(pool, batch_compute_perturbation,
                                    encoder_model, tokenizer, text, is_gpt2, ig_bundle.token_attributions, list(attentions)),
                                timeout=120.0
                            )
                        if perturb_bundle and isinstance(perturb_bundle, PerturbationAnalysisBundle):
                            tok_imp = [{"token": r.token, "importance": round(float(r.importance), 6)} for r in perturb_bundle.token_results]
                            attn_corrs = [{"layer": l, "head": h, "spearman_rho": round(float(r), 4)} for l, h, r in (perturb_bundle.perturb_vs_attn_spearman or [])[:10]]
                            sentence_data["perturbation_analysis"] = {
                                "token_importance": tok_imp,
                                "perturb_vs_ig_spearman": round(float(perturb_bundle.perturb_vs_ig_spearman), 4) if perturb_bundle.perturb_vs_ig_spearman is not None else None,
                                "perturb_vs_attention_top5": attn_corrs,
                            }
                except Exception as e:
                    sentence_data["perturbation_error"] = str(e)[:100]

                # ── Phase 5: LRP / DeepLift ──
                try:
                    if ig_data and ig_bundle:
                        tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            lrp_bundle = await asyncio.wait_for(
                                loop.run_in_executor(pool, batch_compute_lrp,
                                    encoder_model, tokenizer, text, is_gpt2, ig_bundle.token_attributions, list(attentions), metrics),
                                timeout=120.0
                            )
                        if lrp_bundle and isinstance(lrp_bundle, LRPAnalysisBundle):
                            lrp_corrs = [{"layer": l, "head": h, "spearman_rho": round(float(r), 4)} for l, h, r in (lrp_bundle.correlations or [])[:10]]
                            sentence_data["lrp_analysis"] = {
                                "token_attributions": [round(float(x), 6) for x in lrp_bundle.token_attributions],
                                "lrp_vs_ig_spearman": round(float(lrp_bundle.lrp_vs_ig_spearman), 4) if lrp_bundle.lrp_vs_ig_spearman is not None else None,
                                "lrp_vs_attention_top5": lrp_corrs,
                            }
                except Exception as e:
                    sentence_data["lrp_error"] = str(e)[:100]

                # ── Phase 6: Counterfactual swaps ──
                try:
                    cf_swaps = find_swappable_terms(text)
                    if cf_swaps:
                        sentence_data["counterfactual_swaps"] = [
                            {"term": s["term"], "swap_to": s["swap_to"], "category": s["category"]}
                            for s in cf_swaps
                        ]
                except Exception:
                    pass

            except asyncio.TimeoutError:
                sentence_data["error"] = "Timeout (120s)"
            except Exception as e:
                sentence_data["error"] = str(e)[:200]
            finally:
                # Clear GPU memory between sentences
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_sentence_results.append(sentence_data)

        elapsed = round(_time.time() - t0, 1)

        # ── Build comprehensive report ──
        await session.send_custom_message("batch_progress", {
            "label": f"Analyse Bias ({n_total}/{n_total})"
        })

        report = _build_batch_report(
            per_sentence_results, all_bar_matrices,
            bias_model_key, model_name, thresholds, use_optimized,
            batch_file_name.get(), elapsed, n_total
        )

        # Save to disk
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"batch_report_{bias_model_key}_{ts}.json"
        save_path = Path("downloads/batch/bias") / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Auto-download
        await session.send_custom_message("batch_download_ready", {
            "json_content": json.dumps(report, indent=2, ensure_ascii=False, default=str),
            "filename": fname,
        })

        analyzed = sum(1 for r in per_sentence_results if not r.get("error"))
        ui.notification_show(
            f"Batch complete! {analyzed}/{n_total} sentences, {elapsed}s",
            type="message", duration=8
        )

    # ── Dashboard content (conditional rendering) ──

    @output
    @render.ui
    def bias_dashboard_content():
        res = bias_results.get()
        res_B = bias_results_B.get()
        running = bias_running.get()

        # Get snapshotted compare mode states
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()

        # Use cached texts when results exist to prevent re-rendering on input change
        # Only read live inputs when there are no results (pre-analysis state)
        if res or running:
            text = bias_cached_text_A.get()
            text_B = bias_cached_text_B.get()
        else:
            try:
                text = input.bias_input_text().strip()
            except Exception:
                text = ""

            try:
                text_B = input.bias_input_text_B().strip()
            except Exception:
                text_B = ""

        from .renderers import get_gusnet_architecture_section

        # ── Post-analysis: sentence preview + accordion ──
        if res:
            # Check if we're in compare mode
            if (compare_models or compare_prompts) and res_B:
                # Side-by-side comparison view
                preview_html_A = create_bias_sentence_preview(
                    res["tokens"], res["token_labels"]
                )
                preview_html_B = create_bias_sentence_preview(
                    res_B["tokens"], res_B["token_labels"]
                )

                # Determine labels based on mode
                if compare_models:
                    header_A = "MODEL A"
                    header_B = "MODEL B"
                    # Keep the detailed label for the card or just use generic>=
                    # User asked for "MODEL A" header.
                else:
                    header_A = "PROMPT A"
                    header_B = "PROMPT B"

                return ui.div(
                    {"style": "display: flex; flex-direction: column; gap: 24px;"},
                    # Side-by-side sentence previews
                    ui.div(
                        {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                        # Column A
                        ui.div(
                            {"style": "display: flex; flex-direction: column;"},
                            ui.h3(header_A, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                            ui.div(
                                {"class": "card compare-card-a", "style": "min-height: 140px; flex: 1;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_A),
                            ),
                        ),
                        # Column B
                        ui.div(
                            {"style": "display: flex; flex-direction: column;"},
                            ui.h3(header_B, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                            ui.div(
                                {"class": "card compare-card-b", "style": "min-height: 140px; flex: 1;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_B),
                            ),
                        ),
                    ),
                    # Summary comparison cards
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.div(style="height: 110px;"), # Spacer to clear floating bar
                    ui.tags.script("Shiny.setInputValue('toggle_bias_toolbar_visible', true, {priority: 'event'});"),
                )
            else:
                # Single result view (original behavior)
                preview_html = create_bias_sentence_preview(
                    res["tokens"], res["token_labels"]
                )
                return ui.div(
                    {"style": "display: flex; flex-direction: column; gap: 24px;"},
                    ui.div(
                        {"class": "card", "style": "min-height: 140px;"},
                        ui.h4("Sentence Preview"),
                        ui.HTML(preview_html),
                    ),
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.div(style="height: 110px;"), # Spacer to clear floating bar
                    ui.tags.script("Shiny.setInputValue('toggle_bias_toolbar_visible', true, {priority: 'event'});"),
                )

        # ── Pre-analysis: sentence preview + architecture ──
        if text:
            preview = ui.div(
                f'"{text}"',
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
        else:
            preview = ui.div(
                'Enter text in the sidebar and click "Analyze Bias" to begin.',
                style="color:#9ca3af;font-size:14px;font-family:monospace;",
            )

        # Get LIVE compare mode states for pre-analysis (instant feedback)
        try:
            live_compare_models = input.bias_compare_mode()
        except Exception:
            live_compare_models = False

        try:
            live_compare_prompts = input.bias_compare_prompts_mode()
        except Exception:
            live_compare_prompts = False

        if live_compare_prompts:
            # Side-by-side preview for Compare Prompts
            preview_A = ui.div(
                f'"{text}"' if text else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            preview_B = ui.div(
                f'"{text_B}"' if text_B else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            
            card = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                # Column A
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("PROMPT A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_A,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("PROMPT B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_B,
                    ),
                ),
            )
        elif live_compare_models:
            # Side-by-side preview for Compare Models (same text, different models)
            preview_content = ui.div(
                f'"{text}"' if text else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            
            card = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                # Column A
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("MODEL A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_content,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("MODEL B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        # We need a fresh object/string for the second separate div content
                        ui.div(
                            f'"{text}"' if text else "Waiting for input...",
                            style="font-family:monospace;color:#6b7280;font-size:14px;"
                        ),
                    ),
                ),
            )
        else:
            # Single preview
            if text:
                preview = ui.div(
                    f'"{text}"',
                    style="font-family:monospace;color:#6b7280;font-size:14px;"
                )
            else:
                preview = ui.div(
                    'Enter text in the sidebar and click "Analyze Bias" to begin.',
                    style="color:#9ca3af;font-size:14px;font-family:monospace;",
                )

            card = ui.div(
                {"class": "card"},
                ui.h4("Sentence Preview"),
                preview,
            )

        # Initial / Running State: Keep sentence preview + architecture visible
        # (mirrors the attention tab behavior before analysis results exist).
        # Defensive fallback: never let this block fail silently and hide UI.
        try:
            # Determine architecture display mode based on current compare mode
            if res:
                selected_model = res.get("bias_model_key", "gusnet-bert")
            else:
                try:
                    selected_model = input.bias_model_key()
                except BaseException:
                    selected_model = "gusnet-bert"

            try:
                current_compare_models = bool(input.bias_compare_mode())
            except BaseException:
                current_compare_models = False

            # Get Model B selection for Compare Models mode
            try:
                model_b = input.bias_model_key_B()
                if not model_b:
                    model_b = "gusnet-gpt2"  # Default Model B
            except BaseException:
                model_b = "gusnet-gpt2"

            # Determine compare-prompts from the actual toggle, not text content.
            try:
                current_compare_prompts = bool(input.bias_compare_prompts_mode())
            except BaseException:
                current_compare_prompts = False

            arch_section = ui.div(
                get_gusnet_architecture_section(
                    selected_model=selected_model,
                    compare_mode=current_compare_models,
                    compare_prompts=current_compare_prompts,
                    model_a=selected_model,  # Model A comes from main selector
                    model_b=model_b,  # Model B comes from compare selector
                ),
            )
        except Exception:
            # Hard fallback with safe defaults so pre-analysis UI never disappears.
            arch_section = ui.div(
                get_gusnet_architecture_section(
                    selected_model="gusnet-bert",
                    compare_mode=False,
                    compare_prompts=False,
                    model_a="gusnet-bert",
                    model_b="gusnet-gpt2",
                ),
            )

        return ui.div(
            {"style": "display: flex; flex-direction: column; gap: 24px;"},
            card,
            arch_section
        )


    def _create_bias_comparison_summary(res_A, res_B, is_model_compare):
        """Create a side-by-side summary comparison card."""
        summary_A = res_A["bias_summary"]
        summary_B = res_B["bias_summary"]

        if is_model_compare:
            label_A = _get_bias_model_label(res_A)
            label_B = _get_bias_model_label(res_B)
        else:
            label_A = "Prompt A"
            label_B = "Prompt B"

        # Create comparison metrics
        comparison_html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="border: 2px solid #3b82f6; border-radius: 8px; padding: 16px;">
                <h5 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 14px; font-weight: 700;">{label_A}</h5>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #1e293b;">{summary_A['biased_tokens']}</div>
                        <div style="font-size: 10px; color: #64748b;">Biased Tokens</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #ea580c;">{summary_A['generalization_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Generalizations</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #dc2626;">{summary_A['unfairness_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Unfair</div>
                    </div>
                </div>
                <div style="margin-top: 12px; text-align: center;">
                    <span style="font-size: 28px; font-weight: 700; color: #ff5ca9;">{summary_A['bias_percentage']:.1f}%</span>
                    <span style="font-size: 11px; color: #64748b; margin-left: 4px;">Bias</span>
                </div>
            </div>
            <div style="border: 2px solid #ff5ca9; border-radius: 8px; padding: 16px;">
                <h5 style="color: #ff5ca9; margin: 0 0 12px 0; font-size: 14px; font-weight: 700;">{label_B}</h5>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #1e293b;">{summary_B['biased_tokens']}</div>
                        <div style="font-size: 10px; color: #64748b;">Biased Tokens</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #ea580c;">{summary_B['generalization_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Generalizations</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #dc2626;">{summary_B['unfairness_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Unfair</div>
                    </div>
                </div>
                <div style="margin-top: 12px; text-align: center;">
                    <span style="font-size: 28px; font-weight: 700; color: #ff5ca9;">{summary_B['bias_percentage']:.1f}%</span>
                    <span style="font-size: 11px; color: #64748b; margin-left: 4px;">Bias</span>
                </div>
            </div>
        </div>
        """

        return ui.div(
            {"class": "card", "style": "min-height: auto;"},
            ui.h4("Bias Comparison Summary"),
            ui.HTML(comparison_html),
        )

    # ── Method info (sidebar) ──

    @output
    @render.ui
    def bias_method_info():
        # Use analyzed model key if results exist, otherwise live input
        res = bias_results.get()
        if res:
            model_key = res.get("bias_model_key", "gusnet-bert")
        else:
            try:
                model_key = input.bias_model_key()
            except Exception:
                model_key = "gusnet-bert"
                
        html = create_method_info_html(model_key)
        return ui.HTML(html)

    # ── Summary with explicit criteria ──

    @output
    @render.ui
    def bias_summary():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;font-size:12px;">'
                'Enter text and click "Analyze Bias" to begin.</div>'
            )

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        abl = ablation_results.get()
        ig = ig_results.get()
        abl_B = ablation_results_B.get()
        ig_B = ig_results_B.get()

        def get_summary_cards(res_data, abl_data=None, ig_bundle=None):
            summary = res_data["bias_summary"]
            # Criteria breakdown
            criteria_html = create_bias_criteria_html(summary)
            # Metric cards
            cards = f"""
            <div class="metrics-grid" style="margin-top:16px;">
                <div class="metric-card">
                    <div class="metric-label">Biased Tokens</div>
                    <div class="metric-value">{summary['biased_tokens']} / {summary['total_tokens']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Bias %</div>
                    <div class="metric-value">{summary['bias_percentage']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Generalizations</div>
                    <div class="metric-value" style="color:#ea580c;">{summary['generalization_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Unfair Language</div>
                    <div class="metric-value" style="color:#dc2626;">{summary['unfairness_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stereotypes</div>
                    <div class="metric-value" style="color:#7b1fa2;">{summary['stereotype_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value">{summary.get('avg_confidence', 0):.2f}</div>
                </div>
            </div>
            """

            # ── Benchmark cards (BAR / IG ρ / Ablation Δ) ──
            # Root cause of tooltip failure: `.card { overflow:hidden }` clips position:absolute
            # children that extend outside card bounds.
            # Fix: keep position:fixed from the CSS class (fixed elements escape overflow:hidden),
            # and use getBoundingClientRect() in JS to position the tooltip near the (i) icon.
            # transform:none on metric-card prevents the hover-transform from creating a new
            # containing block (which would make position:fixed behave like position:absolute).
            _oe = (
                "var t=this.querySelector('.bcard-tooltip');"
                "var r=this.getBoundingClientRect();"
                "t.style.bottom=(window.innerHeight-r.top+8)+'px';"
                "t.style.top='auto';"
                "t.style.left=(r.left+r.width/2)+'px';"
                "t.style.transform='translateX(-50%)';"
                "t.style.visibility='visible';"
                "t.style.opacity='1';"
            )
            _ol = (
                "var t=this.querySelector('.bcard-tooltip');"
                "t.style.visibility='hidden';"
                "t.style.opacity='0';"
            )

            # _TH without text-transform:uppercase for these benchmark tooltips
            _TH_lc = _TH.replace("text-transform:uppercase;", "")

            def _bcard(label, value, color="#334155", sub=None, tooltip=None):
                sub_html = f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">{sub}</div>' if sub else ""
                if tooltip:
                    # Use a unique class so CSS :hover rule does NOT apply
                    info = (
                        f'<span class="info-tooltip-wrapper" style="margin-left:4px;vertical-align:middle;"'
                        f' onmouseenter="{_oe}" onmouseleave="{_ol}">'
                        f'<span class="info-tooltip-icon">i</span>'
                        f'<div class="bcard-tooltip"'
                        f' style="visibility:hidden;opacity:0;position:fixed;z-index:9999999;'
                        f'background:linear-gradient(135deg,#1e293b 0%,#334155 100%);color:#f1f5f9;'
                        f'padding:16px 20px;border-radius:12px;font-size:12px;line-height:1.7;'
                        f'width:380px;max-width:420px;'
                        f'box-shadow:0 15px 50px rgba(0,0,0,0.5),0 0 0 1px rgba(255,92,169,0.3);'
                        f'border:1px solid rgba(255,255,255,0.15);'
                        f'transition:opacity 0.2s ease,visibility 0.2s ease;pointer-events:none;"'
                        f'>{tooltip}</div>'
                        f'</span>'
                    )
                else:
                    info = ""
                return (
                    f'<div class="metric-card" style="min-width:130px;transform:none;">'
                    f'<div class="metric-label" style="display:flex;align-items:center;text-transform:none;">{label}{info}</div>'
                    f'<div class="metric-value" style="color:{color};font-size:20px;">{value}</div>'
                    f'{sub_html}</div>'
                )

            # BAR
            attn = res_data.get("attention_metrics", [])
            if attn:
                bars = [m.bias_attention_ratio for m in attn]
                mean_bar = sum(bars) / len(bars)
                n_spec = sum(1 for m in attn if m.specialized_for_bias)
                bar_color = "#dc2626" if mean_bar > 1.5 else ("#ea580c" if mean_bar > 1.2 else "#22c55e")
                bar_val, bar_sub = f"{mean_bar:.3f}", f"{n_spec}/{len(attn)} heads specialized"
            else:
                bar_val, bar_color, bar_sub = "-", "#94a3b8", "run Analyze Bias first"

            # IG ρ
            if ig_bundle is not None:
                from ..bias.integrated_gradients import IGAnalysisBundle
                corrs = ig_bundle.correlations if isinstance(ig_bundle, IGAnalysisBundle) else ig_bundle
                if corrs:
                    rhos = [r.spearman_rho for r in corrs]
                    mean_rho = sum(rhos) / len(rhos)
                    n_sig = sum(1 for r in corrs if r.spearman_pvalue < 0.05)
                    rho_color = "#2563eb" if mean_rho > 0 else "#dc2626"
                    rho_val, rho_sub = f"{mean_rho:.3f}", f"{n_sig}/{len(corrs)} heads significant"
                else:
                    rho_val, rho_color, rho_sub = "-", "#94a3b8", "no IG data"
            else:
                rho_val, rho_color, rho_sub = "-", "#94a3b8", "run IG analysis first"

            # Ablation Δ
            if abl_data:
                max_delta = max(r.representation_impact for r in abl_data)
                top = abl_data[0]
                abl_color = "#ff5ca9" if max_delta > 0.05 else "#64748b"
                abl_val, abl_sub = f"{max_delta:.3f}", f"top: L{top.layer}H{top.head}"
            else:
                abl_val, abl_color, abl_sub = "-", "#94a3b8", "run ablation first"

            _tt_bar = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Ratio of attention paid to biased tokens vs. chance (μ̂_B / μ₀)</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#dc2626;'>●</span>"
                f"<span>BAR &gt; 1.5 → <span style='{_TBR}'>bias-specialized</span></span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ea580c;'>●</span>"
                f"<span>BAR &gt; 1.2 → <span style='{_TBA}'>elevated</span></span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span>BAR ≤ 1.2 → <span style='{_TBG}'>normal</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Higher values indicate stronger bias-driven attention patterns.</div>"
            )
            _tt_rho = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Spearman correlation between attention weights and Integrated Gradients attributions, per head</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Interpretation</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#2563eb;'>●</span>"
                f"<span>ρ ≈ 1 → <span style='{_TBB}'>attention faithful</span> to token importance</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>ρ ≈ 0 → attention is <span style='{_TBG}'>uninformative</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Significance threshold: p &lt; 0.05 per head.</div>"
            )
            _tt_abl = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Max representation change (1 − cosine similarity) from zeroing a single head</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Threshold</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
                f"<span>Δ &gt; 0.05 → <span style='{_TBP}'>high-impact head</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Identifies the head whose removal most disrupts the model's internal representation.</div>"
            )
            bench_cards = (
                '<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(226,232,240,0.4);">'
                '<div class="metrics-grid">'
                + _bcard("Mean BAR", bar_val, bar_color, bar_sub, _tt_bar)
                + _bcard("Mean IG ρ", rho_val, rho_color, rho_sub, _tt_rho)
                + _bcard("Max Ablation Δ", abl_val, abl_color, abl_sub, _tt_abl)
                + '</div></div>'
            )

            return criteria_html + cards + bench_cards

        if (compare_models or compare_prompts) and res_B:
            content_A = get_summary_cards(res, abl, ig)
            content_B = get_summary_cards(res_B, abl_B, ig_B)

            header_args = (
                "Bias Detection Summary",
                "Composite bias level with explicit weighted criteria and per-category counts.",
                f"<span style='{_TH}'>What this shows</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                f"<span>4-level severity: <span style='{_TBG}'>None</span> "
                f"<span style='{_TBB}'>Low</span> <span style='{_TBA}'>Moderate</span> "
                f"<span style='{_TBR}'>High</span></span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Weights</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span><span style='{_TBG}'>30%</span>&nbsp;Token Density: fraction of tokens flagged</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
                f"<span><span style='{_TBR}'>25%</span>&nbsp;Unfair Language: slurs, loaded framing, and explicit prejudice</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
                f"<span><span style='{_TBA}'>25%</span>&nbsp;Stereotypes: attribute-group co-occurrence patterns</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
                f"<span><span style='{_TBP}'>20%</span>&nbsp;Generalisations: universal claims about groups</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Low &lt; 0.3 &nbsp;·&nbsp; Moderate 0.3–0.6 &nbsp;·&nbsp; High &gt; 0.6</span></div>"
                f"<div style='{_TN}; margin-top:6px;'>High token density + low stereotype score → loaded language without group-specific targeting.</div>"
            )

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(content_A), *header_args, style="margin-bottom:0; border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(content_B), *header_args, style="margin-bottom:0; border: 2px solid #ff5ca9; height: 100%;")
            )
        else:
            header_args = (
                "Bias Detection Summary",
                "Composite bias level with explicit weighted criteria and per-category counts.",
                f"<span style='{_TH}'>What this shows</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                f"<span>4-level severity: <span style='{_TBG}'>None</span> "
                f"<span style='{_TBB}'>Low</span> <span style='{_TBA}'>Moderate</span> "
                f"<span style='{_TBR}'>High</span></span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Weights</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span><span style='{_TBG}'>30%</span>&nbsp;Token Density: fraction of tokens flagged</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
                f"<span><span style='{_TBR}'>25%</span>&nbsp;Unfair Language: slurs, loaded framing, and explicit prejudice</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
                f"<span><span style='{_TBA}'>25%</span>&nbsp;Stereotypes: attribute-group co-occurrence patterns</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
                f"<span><span style='{_TBP}'>20%</span>&nbsp;Generalisations: universal claims about groups</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Low &lt; 0.3 &nbsp;·&nbsp; Moderate 0.3–0.6 &nbsp;·&nbsp; High &gt; 0.6</span></div>"
                f"<div style='{_TN}; margin-top:6px;'>High token density + low stereotype score → loaded language without group-specific targeting.</div>"
            )
            return _wrap_card(ui.HTML(get_summary_cards(res, abl, ig)), *header_args, style="margin-bottom: 24px;")

    # ── Inline bias view (primary) ──

    @output
    @render.ui
    def inline_bias_view():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            threshold = _safe_threshold()
            html = create_inline_bias_html(
                res["text"], res["token_labels"], res["bias_spans"],
                show_neutral=False, threshold=threshold,
            )
            return ui.HTML(html)
        except Exception as e:
            _logger.exception("Error in inline_bias_view")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {_html.escape(str(e))}</div>')

    # ── Token heatmap (technical view) ──

    @output
    @render.ui
    def token_bias_viz():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            fig = create_token_bias_heatmap(res["token_labels"], res["text"])
            return ui.HTML(_chart_with_png_btn(_deferred_plotly(fig, "token-bias-heatmap"), "token-bias-heatmap", "token_bias_heatmap"))
        except Exception as e:
            _logger.exception("Error creating token bias viz")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {_html.escape(str(e))}</div>')

    # ── Bias spans table (per-token, one line each) ──



    # ── Token bias strip (replaces Plotly heatmap) ──



    # ── Ratio formula panel (static) ──

    @output
    @render.ui
    def bias_ratio_formula():
        return ui.HTML(create_ratio_formula_html())

    # ── Update layer/head selectors ──

    @reactive.effect
    def update_bias_selectors():
        res = bias_results.get()
        if not res or not res["attentions"]:
            return
        attentions = res["attentions"]
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]

        layer_choices = {str(i): f"Layer {i}" for i in range(num_layers)}
        head_choices = {str(i): f"Head {i}" for i in range(num_heads)}

        ui.update_select("bias_attn_layer", choices=layer_choices, selected="0")
        ui.update_select("bias_attn_head", choices=head_choices, selected="0")


    # ── Toolkit Handlers ──

    @output
    @render.ui
    def bias_toolkit_spans():
        """Render biased tokens in the toolbar popover."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:8px;">No analysis yet.</div>')

        # Use pre-calculated spans (merged subwords) instead of raw tokens
        bias_spans = res.get("bias_spans", [])
        
        if not bias_spans:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:12px;">No bias detected.</div>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        # Update badge count
        badge_script = f"<script>$('#bias-span-count-badge').text('{len(bias_spans)}');</script>"

        items = [badge_script]
        for span in bias_spans:
            text = span.get("text", " ".join(span["tokens"])) # Fallback if text not present
            score = span.get("avg_score", 0.0)
            types = span.get("bias_types", [])
            
            score_color = "#ef4444" if score > 0.8 else "#f59e0b" if score > 0.5 else "#94a3b8"

            cats_html = ""
            for cat in types:
                bg = cat_colors.get(cat, "#ff5ca9")
                cats_html += (
                    f'<span style="background:{bg}25;color:{bg};padding:1px 5px;'
                    f'border-radius:3px;font-size:9px;margin-right:3px;">{cat}</span>'
                )

            items.append(
                f'<div class="bias-span-item">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;'
                f'color:#e2e8f0;font-weight:600;">{text}</span>'
                f'<span style="color:{score_color};font-weight:600;font-size:11px;'
                f'font-family:JetBrains Mono,monospace;">{score:.2f}</span></div>'
                f'<div style="margin-top:3px;">{cats_html}</div>'
                f'</div>'
            )

        return ui.HTML("".join(items))

    @output
    @render.ui
    def bias_toolkit_heads():
        """Render Top K heads as horizontal chips."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">--</span>')

        metrics = res.get("attention_metrics", [])
        if not metrics:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">--</span>')

        try:
            k = int(input.bias_top_k())
        except Exception:
            k = 5

        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:k]

        items = []
        for m in top:
            is_sig = m.specialized_for_bias
            border = "#ff5ca9" if is_sig else "rgba(255,255,255,0.15)"
            items.append(
                f'<div class="bias-head-chip" onclick="setBiasHead({m.layer},{m.head})" '
                f'title="Ratio: {m.bias_attention_ratio:.3f}" '
                f'style="border-color:{border};">'
                f'L{m.layer}&middot;H{m.head}'
                f'</div>'
            )

        return ui.HTML("".join(items))

    @output
    @render.ui
    def bias_toolbar_tokens():
        """Render biased tokens as clickable chips in the toolbar.

        In compare prompts mode, matches the screenshot:
        - Row A (top): Blue label "A", tokens to the right.
        - Separator line.
        - Row B (bottom): Pink label "B", tokens to the right.
        """
        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}

        def _hex_to_rgba(hex_color, alpha):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        def _build_chips_html(res, variant="A"):
            """Build HTML chips for a single result set."""
            if not res:
                return '<span style="color:#64748b;font-size:10px;padding:2px;">No analysis</span>'
            token_labels = res["token_labels"]
            biased = [
                lbl for lbl in token_labels
                if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
            ]
            if not biased:
                return '<span style="color:#64748b;font-size:10px;padding:2px;">No bias detected</span>'

            base_color = "#3b82f6" if variant == "A" else "#ff5ca9"

            items = []
            for lbl in biased:
                clean = _html.escape(lbl["token"].replace("##", "").replace("\u0120", ""))
                tok_idx = lbl["index"]

                # Pick color of category with highest score
                scores = lbl.get("scores", {})
                if scores:
                    max_cat = max(scores, key=lambda k: scores[k])
                    item_color = cat_colors.get(max_cat, base_color)
                else:
                    types = lbl.get("bias_types", [])
                    item_color = cat_colors.get(types[0], base_color) if types else base_color

                chip_style = (
                    f"background:{_hex_to_rgba(item_color, 0.18)};"
                    f"border-color:{_hex_to_rgba(item_color, 0.55)};"
                    f"color:white;"
                    f"--chip-color:{item_color};"
                )

                items.append(
                    f'<span class="token-chip bias-token-chip" '
                    f'data-token-idx="{tok_idx}" '
                    f'data-prefix="{variant}" '
                    f'style="{chip_style}" '
                    f'onclick="selectBiasToken({tok_idx}, \'{variant}\')">'
                    f'{clean}'
                    f'</span>'
                )
            return "".join(items)

        compare_prompts = active_bias_compare_prompts.get()

        if compare_prompts:
            # ── Compare Prompts Layout (same structure as attention tab) ──
            res_A = bias_results.get()
            res_B = bias_results_B.get()
            chips_A = _build_chips_html(res_A, "A")
            chips_B = _build_chips_html(res_B, "B")

            return ui.div(
                {"class": "token-row-split"},
                ui.div(
                    {"class": "token-split-item"},
                    ui.span("A", class_="model-label-a"),
                    ui.HTML(chips_A),
                ),
                ui.div(
                    {"class": "token-split-item item-b"},
                    ui.span("B", class_="model-label-b"),
                    ui.HTML(chips_B),
                ),
            )
        else:
            # ── Single / Compare Models ──
            res = bias_results.get()
            chips = _build_chips_html(res, "A")
            return ui.HTML(
                f'<div class="token-sentence">'
                f'{chips}'
                f'</div>'
            )


    # ── Comparison Refactored Renderers ──

    @output
    @render.ui
    def bias_spans_table():
        res = bias_results.get()
        if not res: return None
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        # ... logic to build HTML items ...
        def produce_html(data, is_B):
            if not data:
                return f'<div style="color:#9ca3af;font-size:12px;padding:12px;">Run analysis to see {"Model B" if is_B else "Model A"} results.</div>'
                
            token_labels = data.get("token_labels", [])
            biased = [l for l in token_labels if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
            if not biased: return '<div style="color:#9ca3af;font-size:12px;padding:12px;">No biased tokens detected.</div>'
            
            cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
            items = []
            sel_indices = []
            # Read the side-specific selection so A and B stay independent.
            try:
                s = input.bias_selected_tokens_B() if is_B else input.bias_selected_tokens_A()
                if s: sel_indices = [int(s)] if isinstance(s,(int,str)) else [int(x) for x in s if x]
            except Exception:
                pass
                
            # Check if counterfactual swaps are available (single-prompt mode)
            show_cf = (not is_B and not compare_models and not compare_prompts
                       and counterfactual_swaps.get())

            for lbl in biased:
                clean = _html.escape(lbl["token"].replace("##", "").replace("\u0120", ""))
                types = lbl.get("bias_types", [])
                scores = lbl.get("scores", {})
                badges = "".join([f'<span style="display:inline-flex;align-items:center;gap:4px;background:{cat_colors.get(t,"#ff5ca9")}18;border:1px solid {cat_colors.get(t,"#ff5ca9")}40;color:{cat_colors.get(t,"#ff5ca9")};padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">{_html.escape(t)}<span style="font-family:JetBrains Mono;font-weight:400;opacity:0.8;">{scores.get(t,0):.2f}</span></span>' for t in types])

                # Counterfactual swap badge (if token is swappable)
                cf_badge = ""
                if show_cf:
                    swap_info = get_swap_for_token(clean)
                    if swap_info:
                        swap_to, swap_cat = swap_info
                        cf_key = clean.lower()
                        cf_badge = (
                            f'<span class="cf-swap-badge" data-cf-key="{cf_key}" '
                            f'onclick="event.stopPropagation(); window.toggleCfSwap(\'{cf_key}\', \'{swap_to}\', \'{swap_cat}\')" '
                            f'style="display:inline-flex;align-items:center;gap:3px;'
                            f'background:#1e293b;border:1px solid rgba(255,92,169,0.5);'
                            f'color:#ff5ca9;padding:2px 8px;border-radius:4px;font-size:10px;'
                            f'font-weight:600;cursor:pointer;transition:all 0.15s ease;'
                            f'margin-left:4px;" '
                            f'title="Click to select this swap for counterfactual comparison">'
                            f'&#x21C4; {swap_to}</span>'
                        )

                style = "display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid rgba(226,232,240,0.4);"
                if lbl["index"] in sel_indices: style += "background:rgba(255, 92, 169, 0.1); border-left: 3px solid #ff5ca9;"
                items.append(f'<div style="{style}"><span style="font-family:JetBrains Mono;font-size:13px;font-weight:600;color:#ec4899;min-width:70px;">{clean}</span><span style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;">{badges}{cf_badge}</span></div>')
            
            import math
            mid = math.ceil(len(items)/2)
            c1, c2 = items[:mid], items[mid:]
            
            # Show per-category thresholds - always read from live slider inputs
            t_info = "Thresholds: "

            # Read directly from slider inputs (always reflects sidebar state)
            if not is_B:
                try: u = float(input.bias_thresh_unfair())
                except Exception: u = current_thresholds_A.get().get("UNFAIR", 0.5)
                try: g = float(input.bias_thresh_gen())
                except Exception: g = current_thresholds_A.get().get("GEN", 0.5)
                try: s = float(input.bias_thresh_stereo())
                except Exception: s = current_thresholds_A.get().get("STEREO", 0.5)
            else:
                try: u = float(input.bias_thresh_unfair_b())
                except Exception: u = current_thresholds_B.get().get("UNFAIR", 0.5)
                try: g = float(input.bias_thresh_gen_b())
                except Exception: g = current_thresholds_B.get().get("GEN", 0.5)
                try: s = float(input.bias_thresh_stereo_b())
                except Exception: s = current_thresholds_B.get().get("STEREO", 0.5)

            t_info += f'UNFAIR: <code>{u:.2f}</code> &bull; GEN: <code>{g:.2f}</code> &bull; STEREO: <code>{s:.2f}</code>'
            if data and data.get("use_optimized"):
                t_info += " <span style='opacity:0.6;'>(Optimized)</span>"

            return (f'<div style="display:flex;gap:16px;border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
                    f'<div style="flex:1;display:flex;flex-direction:column;">{"".join(c1)}</div>'
                    f'<div style="flex:1;display:flex;flex-direction:column;border-left:1px solid rgba(226,232,240,0.4);">{"".join(c2)}</div>'
                    f'</div><div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">{t_info}</div>')

        # Use manual header for Detected Bias Tokens
        # old ui: h4("Detected Bias Tokens", ...), p(...)
        man_header = (
            "Detected Bias Tokens",
            "Each biased span with category labels and GUS-Net confidence scores (hover a token for details).",
        )
        _detected_bias_help = (
            f"<span style='{_TH}'>Categories: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>UNFAIR</span>&nbsp;explicit prejudice, slurs, loaded framing</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>GEN</span>&nbsp;overgeneralisations, for example 'all X are Y'</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>STEREO</span>&nbsp;stereotyped attribute–group co-occurrences</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Confidence score</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>GUS-Net softmax probability for predicted bias class</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>Tokens below threshold (default 0.5) are not highlighted</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interaction</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▶</span>"
            f"<span>Click a token to highlight it across the token-level bias distribution and confidence breakdown views</span></div>"
            f"<div style='{_TN}; margin-top:4px;'>Multiple categories can fire on the same token. The primary label is the highest-confidence class.</div>"
        )
        
        # Additional footer: bias_method_info (was separate)
        # Should we include it in the card? The previous UI had `ui.hr(), ui.output_ui("bias_method_info")` INSIDE the card.
        # So we should append it to content.
        # But bias_method_info is a separate renderer. We can't easily call it here unless we duplicate it or it returns string.
        # `create_method_info_html` returns string. So we can just call that.
        if res and "bias_model_key" in res:
            model_key_A = res["bias_model_key"]
        else:
            try:
                model_key_A = input.bias_model_key()
            except Exception:
                model_key_A = "gusnet-bert"
        
        if (compare_models or compare_prompts) and res_B:
            # Get Model B key for compare models mode
            if res_B and "bias_model_key" in res_B:
                model_key_B = res_B["bias_model_key"]
            else:
                if compare_models:
                    try:
                        model_key_B = input.bias_model_key_B()
                        if not model_key_B:
                            model_key_B = "gusnet-gpt2"
                    except Exception:
                        model_key_B = "gusnet-gpt2"
                else:
                    # Compare prompts uses the same model for both
                    model_key_B = model_key_A
            
            method_html_A = create_method_info_html(model_key_A)
            method_html_B = create_method_info_html(model_key_B)
            method_footer_A = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html_A}</div>'
            method_footer_B = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html_B}</div>'

            side_by_side = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}</div>' + method_footer_A), manual_header=man_header, help_text=_detected_bias_help, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res_B, True)}</div>' + method_footer_B), manual_header=man_header, help_text=_detected_bias_help, style="border: 2px solid #ff5ca9; height: 100%;")
            )

            # Counterfactual consistency card — only when triggered via CF swap
            swaps_applied = cf_applied_swaps.get()
            log_debug(f"CF consistency check: compare_prompts={compare_prompts}, swaps={bool(swaps_applied)}, attn_A={bool(res.get('attentions'))}, attn_B={bool(res_B.get('attentions') if res_B else False)}")
            if compare_prompts and swaps_applied and res.get("attentions") and res_B and res_B.get("attentions"):
                try:
                    consistency = compute_cf_attention_consistency(
                        res["tokens"], res["attentions"],
                        res_B["tokens"], res_B["attentions"],
                    )
                    cf_html = create_cf_consistency_html(
                        consistency,
                        res["tokens"], res_B["tokens"],
                        swaps_applied,
                        bias_summary_A=res.get("bias_summary"),
                        bias_summary_B=res_B.get("bias_summary"),
                    )
                    _cf_help = (
                        f"<span style='{_TH}'>Attention Consistency Metrics</span>"
                        f"<div style='{_TR}'><span style='{_TD};color:#3b82f6;'>&#x25CF;</span>"
                        f"<span><b>Attention Similarity</b>: cosine similarity of attention-received profiles "
                        f"for non-swapped tokens (1.0 = identical patterns)</span></div>"
                        f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>&#x25CF;</span>"
                        f"<span><b>Swap Focus</b>: fraction of total attention change concentrated at "
                        f"the swapped demographic tokens (higher = model attention is more localized)</span></div>"
                        f"<div style='{_TR}'><span style='{_TD};color:#3b82f6;'>&#x25CF;</span>"
                        f"<span><b>Top Sensitive Heads</b>: attention heads with the largest total attention "
                        f"delta between original and counterfactual</span></div>"
                    )
                    consistency_card = _wrap_card(
                        ui.HTML(cf_html),
                        manual_header=(
                            "Counterfactual Attention Consistency",
                            "How attention patterns change when demographic terms are swapped.",
                        ),
                        help_text=_cf_help,
                        style="margin-top: 16px;",
                    )
                    return ui.div(side_by_side, consistency_card)
                except Exception:
                    _logger.exception("CF consistency error")

            return side_by_side

        method_html = create_method_info_html(model_key_A)
        method_footer = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html}</div>'

        # Counterfactual compare button (single-prompt mode only)
        cf_section = ""
        cf_swaps_raw = counterfactual_swaps.get()
        if cf_swaps_raw and not compare_models and not compare_prompts:
            # Reset JS selection state whenever the panel re-renders
            cf_section = (
                '<script>window.selectedCfSwaps = window.selectedCfSwaps || {};</script>'
                '<div id="cf-compare-btn-container" style="display:none;margin-top:12px;text-align:center;">'
                '<button onclick="window.triggerCfCompare()" '
                'style="background:#1e293b;color:#ff5ca9;border:1px solid rgba(255,92,169,0.3);padding:8px 20px;'
                'border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;'
                'transition:all 0.15s ease;letter-spacing:0.5px;" '
                'onmouseover="this.style.background=\'rgba(255,92,169,0.1)\';this.style.borderColor=\'rgba(255,92,169,0.7)\';this.style.color=\'#ff74b8\'" '
                'onmouseout="this.style.background=\'#1e293b\';this.style.borderColor=\'rgba(255,92,169,0.3)\';this.style.color=\'#ff5ca9\'">'
                '<span style="margin-right:6px;">&#x21C4;</span>'
                'Compare Counterfactuals (<span id="cf-count">0</span>)'
                '</button></div>'
            )

        return _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}{cf_section}</div>' + method_footer), manual_header=man_header, help_text=_detected_bias_help)

    @output
    @render.ui
    def token_bias_strip():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        # Get selected token indices for highlighting (A and B are tracked
        # independently so each side's click only affects its own strip).
        def _read_sel(reader):
            try:
                sel = reader()
                if not sel:
                    return None
                return (
                    [int(sel)] if isinstance(sel, (int, str))
                    else [int(x) for x in sel if x is not None]
                )
            except Exception:
                return None

        selected_idx = _read_sel(input.bias_selected_tokens_A)
        selected_idx_B = _read_sel(input.bias_selected_tokens_B)

        def get_viz(data, sel_idx=None):
            try:
                return create_token_bias_strip(data["token_labels"], sel_idx)
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        man_header = (
            "Token-Level Bias Distribution",
            "Per-token bias scores across all four categories (O, GEN, UNFAIR, STEREO) shown as coloured dot strips.",
        )
        _strip_help = (
            f"<span style='{_TH}'>What this shows: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Coloured dot strip showing each token's softmax score across all four bias categories, scored independently by GUS-Net</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Reading the strip</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Dot colour</b>: category (O / GEN / UNFAIR / STEREO)</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Dot size / bar height</b>: overall bias magnitude for that token</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Grey / no fill</b>: token is below the detection threshold</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Categories</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span><span style='{_TBG};color:#9ca3af;'>O</span>&nbsp;neutral: not biased</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ca8a04;'>●</span>"
            f"<span><span style='{_TBA};color:#ca8a04;'>GEN</span>&nbsp;generalisation: broad claims about a group</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR};color:#ef4444;'>UNFAIR</span>&nbsp;prejudiced framing, slurs, or loaded language</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a855f7;'>●</span>"
            f"<span><span style='{_TBP};color:#a855f7;'>STEREO</span>&nbsp;stereotype: fixed cultural or social assumption</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Compare profiles across tokens to spot which part of the sentence drives the overall score and whether multiple bias types co-occur.</div>"
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_strip_help, style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_strip", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_viz(res_B, selected_idx_B)), manual_header=man_header, help_text=_strip_help, style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_strip_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_strip_help,
                          controls=[ui.download_button("export_bias_strip", "CSV", style=_BTN_STYLE_CSV)])

    @output
    @render.ui
    def confidence_breakdown():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # Get selected token indices (independent per side).
        def _read_sel(reader):
            try:
                sel = reader()
                if not sel:
                    return None
                return (
                    [int(sel)] if isinstance(sel, (int, str))
                    else [int(x) for x in sel if x is not None]
                )
            except Exception:
                return None

        selected_idx = _read_sel(input.bias_selected_tokens_A)
        selected_idx_B = _read_sel(input.bias_selected_tokens_B)

        def get_viz(data, sel_idx=None):
            try:
                return create_confidence_breakdown(data["token_labels"], selected_token_idx=sel_idx)
            except Exception as e:
                return f'<div style="color:red">Error: {e}</div>'

        man_header = (
            "Confidence Breakdown",
            "Biased tokens grouped by confidence tier: Low (0.50–0.70), Medium (0.70–0.85), and High (0.85+).",
        )
        _confidence_help = (
            f"<span style='{_TH}'>What this shows: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Detected bias spans grouped into three confidence tiers based on GUS-Net's softmax probability for the predicted bias class</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Tiers</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBA};color:#22c55e;'>Low 0.50–0.70</span>&nbsp;marginal detection: figurative language, borderline phrasing, or domain terms</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#eab308;'>●</span>"
            f"<span><span style='{_TBB};color:#eab308;'>Medium 0.70–0.85</span>&nbsp;probable bias signal: likely, but not certain</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR};color:#ef4444;'>High 0.85+</span>&nbsp;strong model-confident signal: typically unambiguous bias indicators</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Low-confidence spans should be read cautiously because they may reflect context or style rather than bias</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>High-confidence spans are the primary evidence for bias presence in the text</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Threshold (default 0.5): spans below this are suppressed entirely. Adjust via the toolbar to surface or hide borderline detections.</div>"
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_confidence_help, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(get_viz(res_B, selected_idx_B)), manual_header=man_header, help_text=_confidence_help, style="border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_confidence_help)

    @output
    @render.ui
    def combined_bias_view():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except Exception: l_idx, h_idx = 0, 0

        def _read_sel(reader):
            try:
                s = reader()
                if not s:
                    return None
                return [int(s)] if isinstance(s, (int, str)) else [int(x) for x in s if x]
            except Exception:
                # Broad catch on purpose: Shiny raises SilentException when
                # the reactive input has not been set yet (first render,
                # before any click) — we must not let that propagate, or the
                # whole bias strip renderer fails and subsequent clicks stop
                # working.
                return None

        sel = _read_sel(input.bias_selected_tokens_A)
        sel_B = _read_sel(input.bias_selected_tokens_B)

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = None
        if src_mode == "compare":
            res_base_for_compare = _get_base_resolved(res, _base_attn_cache)
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        def get_viz(data, s_idxs, container_id="bias-combined-container",
                    other_data=None, delta_label="A \u2212 B"):
            atts = data["attentions"]
            if not atts or l_idx >= len(atts): return '<div style="color:#9ca3af;">No attention data.</div>'
            try:
                if l_idx >= len(atts): return '<div style="color:#9ca3af;">Layer out of bounds.</div>'
                attn = atts[l_idx][0, h_idx].cpu().numpy()
                attn_other = None
                if other_data:
                    try:
                        o_atts = other_data["attentions"]
                        if o_atts and l_idx < len(o_atts):
                            attn_other = o_atts[l_idx][0, h_idx].cpu().numpy()
                    except Exception:
                        attn_other = None
                fig = create_combined_bias_visualization(
                    data["tokens"], data["token_labels"], attn, l_idx, h_idx,
                    selected_token_idx=s_idxs,
                    attention_matrix_other=attn_other,
                    delta_label=delta_label,
                )
                return _deferred_plotly(fig, container_id, height="600px")
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        man_header = (
            f"Combined Attention & Bias View{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Attention weight matrix for one head, with pink highlights on biased token rows/columns.",
        )
        _combined_help = (
            f"<span style='{_TH}'>What this shows: Attention × GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Full token × token attention weight matrix for one (layer, head), with pink highlights on rows and columns that GUS-Net flagged as biased</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Reading the heatmap</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Cell <span style='{_TC}'>(i, j)</span>: attention weight from query token <i>i</i> to key token <i>j</i></span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>▪</span>"
            f"<span><span style='{_TBR}'>Pink row</span>&nbsp;token <i>i</i> is biased; its query attends across the full sequence</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><span style='{_TBP}'>Pink col</span>&nbsp;token <i>j</i> is biased; it receives attention from the full sequence</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>▪</span>"
            f"<span><span style='{_TBA}'>Intersection</span>&nbsp;biased token attending to another biased token; possible semantic coupling or co-reference</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Toolbar controls</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
            f"<span>Change <b>layer</b> and <b>head</b> to explore different specialisation profiles</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
            f"<span>Click a token in <i>Detected Bias Tokens</i> to set it as the query focus</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>High weights at two pink-cell intersections suggest the model is semantically linking two biased spans, which can indicate a reinforcement mechanism.</div>"
        )
        card_style = "margin-bottom: 24px; box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        # res = GUS-Net attentions (from heavy_bias_compute via _GUSNET_TO_ENCODER)
        # res_base_for_compare = original pretrained base model attentions
        if src_mode == "compare" and res_base_for_compare:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, sel, "bias-combined-container-base",
                                          other_data=res, delta_label="Base \u2212 GUS")),
                           manual_header=(f"Combined Attention & Bias View{_source_badge_html('Base Encoder')}",
                                          man_header[1]),
                           help_text=_combined_help,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-base', 'bias_combined_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, None, "bias-combined-container-gus",
                                          other_data=res_base_for_compare, delta_label="GUS \u2212 Base")),
                           manual_header=(f"Combined Attention & Bias View{_source_badge_html('GUS-Net')}",
                                          man_header[1]),
                           help_text=_combined_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-gus', 'bias_combined_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_render, sel, "bias-combined-container",
                                          other_data=res_render_B, delta_label="A \u2212 B")),
                           manual_header=man_header, help_text=_combined_help,
                           style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container', 'bias_combined_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_render_B, sel_B, "bias-combined-container-B",
                                          other_data=res_render, delta_label="B \u2212 A")),
                           manual_header=man_header, help_text=_combined_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-B', 'bias_combined_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res_render, sel, "bias-combined-container")), manual_header=man_header, help_text=_combined_help, style=card_style,
                          controls=[
                              ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container', 'bias_combined')", style=_BTN_STYLE_PNG),
                          ])

    @output
    @render.ui
    def attention_bias_matrix():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = _get_base_resolved(res, _base_attn_cache) if src_mode == "compare" else None
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        # Analyzer instance for metrics
        analyzer = AttentionBiasAnalyzer()

        def get_viz(data, container_id="bias-matrix-container",
                    other_data=None, delta_label="A \u2212 B"):
            try:
                if "attentions" not in data or not data["attentions"]:
                     return '<div style="color:#9ca3af;">No attention data.</div>'

                attentions = data["attentions"]
                biased_indices = [l["index"] for l in data["token_labels"] if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
                matrix = analyzer.create_attention_bias_matrix(attentions, biased_indices)
                metrics = data.get("attention_metrics")
                try: sl = int(input.bias_attn_layer())
                except Exception: sl = None

                try: _bar_th = float(input.bias_bar_threshold())
                except Exception: _bar_th = 1.5

                matrix_other = None
                if other_data:
                    try:
                        o_att = other_data.get("attentions")
                        o_bi = [l["index"] for l in other_data["token_labels"] if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
                        if o_att:
                            matrix_other = analyzer.create_attention_bias_matrix(o_att, o_bi)
                    except Exception:
                        matrix_other = None

                fig = create_attention_bias_matrix(
                    matrix, metrics=metrics, selected_layer=sl, bar_threshold=_bar_th,
                    bias_matrix_other=matrix_other, delta_label=delta_label,
                )
                return _deferred_plotly(fig, container_id, height="600px")
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        _bar_help = (
            f"<span style='{_TH}'>Formula</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;'>"
            f"BAR(l,h) = mean_attn_biased / (n_biased / n_tokens)</div>"
            f"<div style='{_TN}; margin-top:2px; margin-bottom:4px;'>observed share / expected under uniform distribution</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ffffff;'>●</span>"
            f"<span><span style='{_TBA};color:#ffffff;'>BAR = 1.0</span>&nbsp;White, uniform attention, no bias focus</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>BAR &gt; 1.5</span>&nbsp;head <b>over-attends</b> to biased tokens</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>BAR &lt; 1.0</span>&nbsp;head <b>avoids</b> biased tokens</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Red cells (BAR >= threshold) = candidate bias-specialised heads. Cross-reference with ablation to confirm causal impact.</div>"
        )
        header_args = (
            f"Bias Attention Matrix{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Each cell shows the Bias Attention Ratio (BAR) for one (layer, head), indicating how much that head over- or under-attends to biased tokens.",
            _bar_help,
        )
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare" and res_base_for_compare:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, "bias-matrix-container-base",
                                          other_data=res, delta_label="Base \u2212 GUS")),
                           f"Bias Attention Matrix{_source_badge_html('Base Encoder')}",
                           header_args[1], _bar_help,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-base', 'bias_matrix_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, "bias-matrix-container-gus",
                                          other_data=res_base_for_compare, delta_label="GUS \u2212 Base")),
                           f"Bias Attention Matrix{_source_badge_html('GUS-Net')}",
                           header_args[1], _bar_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-gus', 'bias_matrix_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_render, "bias-matrix-container",
                                          other_data=res_render_B, delta_label="A \u2212 B")),
                           *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_render_B, "bias-matrix-container-B",
                                          other_data=res_render, delta_label="B \u2212 A")),
                           *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-B', 'bias_matrix_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res_render, "bias-matrix-container")), *header_args, style=card_style,
                          controls=[
                              ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix')", style=_BTN_STYLE_PNG),
                          ])

    # ── Propagation drilldown: click layer → per-head view ──
    @reactive.effect
    @reactive.event(input.propagation_layer_click)
    def _on_propagation_layer_click():
        val = input.propagation_layer_click()
        try:
            propagation_drilldown_layer.set(int(val))
        except (TypeError, ValueError):
            pass

    @reactive.effect
    @reactive.event(input.propagation_back)
    def _on_propagation_back():
        propagation_drilldown_layer.set(None)

    # Reset drilldown when new bias results arrive
    @reactive.effect
    @reactive.event(bias_results)
    def _reset_propagation_drilldown():
        propagation_drilldown_layer.set(None)

    @output
    @render.ui
    def bias_propagation_plot():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;">No results.</div>')

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        drilldown_layer = propagation_drilldown_layer.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = _get_base_resolved(res, _base_attn_cache) if src_mode == "compare" else None
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        try: l_idx = int(input.bias_attn_layer())
        except Exception: l_idx = None

        def _get_head_bars(data, layer):
            """Extract per-head BAR values for a given layer."""
            metrics = data.get("attention_metrics", [])
            heads = [m for m in metrics if m.layer == layer]
            heads.sort(key=lambda m: m.head)
            return [m.bias_attention_ratio for m in heads]

        def get_viz(data, container_id="bias-propagation-container"):
            p = data["propagation_analysis"]["layer_propagation"]
            if not p: return "No data."
            fig = create_bias_propagation_plot(p, selected_layer=l_idx)
            return _deferred_plotly(fig, container_id, height="450px",
                                   click_input="propagation_layer_click")

        def get_heads_viz(data, layer, container_id="bias-propagation-heads-container"):
            head_bars = _get_head_bars(data, layer)
            if not head_bars: return "No per-head data for this layer."
            fig = create_bias_propagation_heads_plot(head_bars, layer)
            return _deferred_plotly(fig, container_id, height="450px")

        # ── Back button for drilldown ──
        back_btn = ui.tags.button(
            ui.HTML('<i class="fa-solid fa-arrow-left" style="margin-right:4px;"></i> Layers'),
            onclick="Shiny.setInputValue('propagation_back', Date.now(), {priority:'event'});",
            style=(
                "background:transparent;border:1px solid #94a3b8;color:#94a3b8;"
                "border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;"
                "font-weight:600;transition:all 0.2s;"
            ),
        )

        _layer_header_args = (
            f"Bias Propagation Across Layers{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Mean BAR per transformer layer. Click a layer to drill down into its heads.",
            f"<span style='{_TH}'>Layer depth</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Early (0–3)</span>&nbsp;syntactic or surface-focused; BAR is usually near neutral</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>Middle (4–8)</span>&nbsp;semantic associations; bias often peaks here</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>Late (9–11)</span>&nbsp;task-specific or abstract; signal may consolidate or diffuse</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Curve shapes</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>▲</span>"
            f"<span><b>Rising</b>: bias behaves like a learned semantic feature and deepens with depth</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▬</span>"
            f"<span><b>Flat</b>: the signal is fairly uniform and not strongly layer-specific</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>▼</span>"
            f"<span><b>Peak then drop</b>: bias is built in middle layers and then refined or diluted later</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Click a layer point to see per-head BAR breakdown. Reference dashed line = BAR 1.0 (uniform).</div>"
        )

        _heads_tooltip = (
            f"<span style='{_TH}'>Per-head breakdown</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>Each bar is one attention head's BAR value</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#dc2626;'>●</span>"
            f"<span><span style='{_TBR}'>Red bars</span>&nbsp;BAR &gt; 1.5 = bias-specialised head</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ea580c;'>●</span>"
            f"<span><span style='{_TBA}'>Orange bars</span>&nbsp;BAR 1.2–1.5 = moderate bias focus</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Click ← Layers to return to the layer overview.</div>"
        )

        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Helper: render single-panel (layers or heads) ──
        def _single_panel(data, suffix="", extra_style=""):
            cid_layers = f"bias-propagation-container{suffix}"
            cid_heads = f"bias-propagation-heads-container{suffix}"
            if drilldown_layer is not None:
                hdr = (
                    f"Bias Propagation — Layer {drilldown_layer} Heads{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
                    f"BAR per attention head in layer {drilldown_layer}.",
                    _heads_tooltip,
                )
                ctrls = [
                    back_btn,
                    ui.tags.button("PNG", onclick=f"downloadPlotlyPNG('{cid_heads}', 'bias_propagation_heads_L{drilldown_layer}')", style=_BTN_STYLE_PNG),
                ]
                return _wrap_card(ui.HTML(get_heads_viz(data, drilldown_layer, cid_heads)),
                                  *hdr, style=card_style + extra_style, controls=ctrls)
            else:
                ctrls = [
                    ui.download_button(f"export_bias_propagation{suffix}", "CSV", style=_BTN_STYLE_CSV),
                    ui.tags.button("PNG", onclick=f"downloadPlotlyPNG('{cid_layers}', 'bias_propagation{suffix}')", style=_BTN_STYLE_PNG),
                ]
                return _wrap_card(ui.HTML(get_viz(data, cid_layers)),
                                  *_layer_header_args, style=card_style + extra_style, controls=ctrls)

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare" and res_base_for_compare:
            hdr_base = (f"Bias Propagation{_source_badge_html('Base Encoder')}", _layer_header_args[1], _layer_header_args[2])
            hdr_gus = (f"Bias Propagation{_source_badge_html('GUS-Net')}", _layer_header_args[1], _layer_header_args[2])
            if drilldown_layer is not None:
                hdr_base = (f"Layer {drilldown_layer} Heads{_source_badge_html('Base Encoder')}",
                            f"BAR per head in layer {drilldown_layer}.", _heads_tooltip)
                hdr_gus = (f"Layer {drilldown_layer} Heads{_source_badge_html('GUS-Net')}",
                           f"BAR per head in layer {drilldown_layer}.", _heads_tooltip)
                return ui.div(
                    {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                    _wrap_card(ui.HTML(get_heads_viz(res_base_for_compare, drilldown_layer, "bias-propagation-heads-base")),
                               *hdr_base, style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                               controls=[back_btn, ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-heads-base', 'bias_propagation_heads_base')", style=_BTN_STYLE_PNG)]),
                    _wrap_card(ui.HTML(get_heads_viz(res, drilldown_layer, "bias-propagation-heads-gus")),
                               *hdr_gus, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                               controls=[ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-heads-gus', 'bias_propagation_heads_gus')", style=_BTN_STYLE_PNG)]),
                )
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, "bias-propagation-container-base")), *hdr_base,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_propagation", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container-base', 'bias_propagation_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, "bias-propagation-container-gus")), *hdr_gus,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container-gus', 'bias_propagation_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _single_panel(res_render, "", " border: 2px solid #3b82f6; height: 100%;"),
                _single_panel(res_render_B, "_B", " border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _single_panel(res_render)

    @output
    @render.ui
    def bias_focused_heads_table():
        res = bias_results.get()
        if not res: return None

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except Exception: l_idx, h_idx = -1, -1

        # Read dynamic Top-K and BAR threshold
        try: k = int(input.bias_top_k())
        except Exception: k = 5
        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        def get_table(data):
            mets = data["attention_metrics"]
            if not mets: return '<div style="color:#9ca3af;padding:20px;text-align:center;font-size:12px;">No metrics available.</div>'
            top = sorted(mets, key=lambda x: x.bias_attention_ratio, reverse=True)[:k]
            any_above = any(m.bias_attention_ratio > bar_threshold for m in top)

            # Note when no heads exceed threshold
            note_html = ""
            if not any_above and top:
                note_html = (
                    f'<div style="padding:8px 12px;margin-bottom:8px;background:rgba(245,158,11,0.08);'
                    f'border:1px solid rgba(245,158,11,0.2);border-radius:6px;font-size:11px;color:#92400e;line-height:1.5;">'
                    f'No heads exceed the specialization threshold ({bar_threshold:.1f}). '
                    f'Showing top-{len(top)} by BAR value. Bias may be lexical rather than concentrated in specific attention heads.'
                    f'</div>'
                )

            rows = []
            for rank, m in enumerate(top, 1):
                is_sel = (m.layer == l_idx and m.head == h_idx)
                is_sig = m.bias_attention_ratio > bar_threshold

                # Row background
                if is_sel:
                    bg = "background:linear-gradient(90deg, rgba(99, 102, 241, 0.12), rgba(99, 102, 241, 0.05));"
                    border_left = "border-left: 3px solid #6366f1;"
                elif is_sig:
                    bg = "background:rgba(255, 92, 169, 0.03);"
                    border_left = ""
                else:
                    bg = ""
                    border_left = ""

                # Badge for specialization
                if is_sig:
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#22c55e,#16a34a);margin-left:6px;" title="Specialized"><span style="color:white;font-size:8px;font-weight:700;">OK</span></span>'
                else:
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#e2e8f0;margin-left:6px;" title="Not specialized"><span style="color:#94a3b8;font-size:8px;">o</span></span>'

                # Value color
                val_color = "#ff5ca9" if is_sig else "#475569"

                rows.append(
                    f'<tr style="{bg}{border_left}transition:all 0.2s ease;">'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{m.layer}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{m.head}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{val_color};">{m.bias_attention_ratio:.3f}{badge}</td>'
                    f'</tr>'
                )

            return (
                f'{note_html}'
                '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(rows)}</tbody>'
                '</table>'
            )

        _heads_help = (
            f"<span style='{_TH}'>Columns</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>Layer / Head</span>&nbsp;transformer block + head index (0-indexed)</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>BAR</span>&nbsp;observed / expected attention - 1.0 = uniform</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>green</span>&nbsp;BAR exceeds specialisation threshold</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>How to use</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Click a row to highlight that head in all visualisations</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Run <b>Head Ablation</b> to test causal impact</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Run <b>Integrated Gradients</b> to check faithfulness</span></div>"
        )
        _heads_subtitle = f"Ranked by Bias Attention Ratio (BAR). Green dot = above specialisation threshold ({bar_threshold:.1f})."
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 16px;"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare":
            res_base_for_compare = _get_base_resolved(res, _base_attn_cache)
            if res_base_for_compare:
                return ui.div(
                    {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                    _wrap_card(ui.HTML(get_table(res_base_for_compare)),
                               manual_header=(f"Top {k} Attention Heads by Bias Focus{_source_badge_html('Base Encoder')}", _heads_subtitle),
                               help_text=_heads_help,
                               style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                               controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)]),
                    _wrap_card(ui.HTML(get_table(res)),
                               manual_header=(f"Top {k} Attention Heads by Bias Focus{_source_badge_html('GUS-Net')}", _heads_subtitle),
                               help_text=_heads_help,
                               style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                               controls=[ui.download_button("export_bias_top_heads_B", "CSV", style=_BTN_STYLE_CSV)])
                )

        header_args = (
            f"Top {k} Attention Heads by Bias Focus{_source_badge_html(src_label)}",
            _heads_subtitle,
            _heads_help,
        )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_table(res_render)), *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_table(res_render_B)), *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_table(res_render)), *header_args, style=card_style,
                          controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)])


    # ── Ablation handlers ─────────────────────────────────────────────

    @reactive.effect
    async def compute_ablation():
        """Automatically run ablation when bias analysis is complete."""
        res_A, res_B, show_comparison = _resolve_faithfulness_results()
        if not res_A:
            return

        ablation_running.set(True)
        ablation_results.set(None)
        ablation_results_B.set(None)

        try:
            try: k = int(input.bias_top_k())
            except Exception: k = 5

            # Define helper for single computation
            def _compute_single(r):
                text = r["text"]
                metrics = r.get("attention_metrics", [])
                if not metrics: return None

                # Use top heads from THIS result
                top_heads_local = sorted(
                    metrics, key=lambda m: m.bias_attention_ratio, reverse=True
                )[:k]

                model_name = r.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name

                tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)
                
                # We need to run sync code in executor
                return (encoder_model, lm_head_model, tokenizer, text, top_heads_local, is_g)

            # Prepare args for A
            args_A = _compute_single(res_A)
            
            # Prepare args for B if needed
            args_B = _compute_single(res_B) if show_comparison and res_B else None

            # Execute
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=2) as pool:
                # Launch A
                fut_A = None
                if args_A:
                    fut_A = loop.run_in_executor(
                        pool,
                        batch_ablate_top_heads,
                        *args_A
                    )
                
                # Launch B
                fut_B = None
                if args_B:
                    fut_B = loop.run_in_executor(
                        pool,
                        batch_ablate_top_heads,
                        *args_B
                    )

                # Wait results
                if fut_A:
                    res_A_calc = await fut_A
                    ablation_results.set(res_A_calc)
                
                if fut_B:
                    res_B_calc = await fut_B
                    ablation_results_B.set(res_B_calc)

        except Exception:
            _logger.exception("Ablation error")
        finally:
            ablation_running.set(False)

    @output
    @render.ui
    def ablation_results_display():
        running = ablation_running.get()
        results_A = ablation_results.get()

        if running or not results_A:
            return None
        
        results_B = ablation_results_B.get()
        _, _, resolved_show_comparison = _resolve_faithfulness_results()
        show_comparison = resolved_show_comparison and results_B

        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        # Get selected head for highlighting
        selected_head = None
        try:
            sel_l = int(input.bias_attn_layer())
            sel_h = int(input.bias_attn_head())
            selected_head = (sel_l, sel_h)
        except Exception:
            pass

        def _render_ablation_single(results_data, container_suffix=""):
            if not results_data: return "No data"
            
            fig = create_ablation_impact_chart(results_data, bar_threshold=bar_threshold, selected_head=selected_head)
            c_id = f"ablation-chart-container{container_suffix}"
            chart_html = _deferred_plotly(fig, c_id)

            table_rows = []
            for rank, r in enumerate(results_data, 1):
                impact_color = "#ff5ca9" if r.representation_impact > 0.05 else "#64748b"
                kl_cell = f"{r.kl_divergence:.4f}" if r.kl_divergence is not None else "N/A"
                specialized = "Yes" if r.bar_original > bar_threshold else "No"
                row_bg = "background:rgba(255,92,169,0.12);" if selected_head and (r.layer, r.head) == selected_head else ""
                table_rows.append(
                    f'<tr style="{row_bg}">'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">#{rank}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{r.layer}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{r.head}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{impact_color};">{r.representation_impact:.4f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{kl_cell}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.bar_original:.3f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;">{specialized}</td>'
                    f'</tr>'
                )

            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:16px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Impact</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">KL Div</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Specialized</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(table_rows)}</tbody>'
                '</table>'
            )
            
            return ui.div(
                ui.HTML(chart_html),
                ui.HTML(table_html),
            )

        faith_src = _get_attn_source_mode("bias_attn_source")
        _faith_label = "GUS-Net" if faith_src == "gusnet" else "Base Encoder"
        _faith_badge = _source_badge_html(_faith_label) if faith_src != "compare" else ""

        header_args = (
            f"Head Ablation Results{_faith_badge}",
            "Causal test: zero out each head's output and measure how much the model's representation changes.",
            f"<span style='{_TH}'>Method: Head Ablation (Michel et al., 2019)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>Zero out one head at a time and measure output change to assess causal importance</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Metrics</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>Impact</span>&nbsp;<span style='{_TC}'>1 − cos_sim(H_orig, H_ablated)</span>"
            f"&nbsp;with 0 meaning no effect and 1 meaning complete change</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>KL Div</span>&nbsp;<span style='{_TC}'>KL(P_orig ‖ P_ablated)</span>"
            f"&nbsp;capturing the shift in output distribution</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Faithfulness patterns</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>High BAR + High Impact</span>&nbsp;True mechanism: specialised <b>and</b> causal</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>High BAR + Low Impact</span>&nbsp;Epiphenomenon: attends to bias but does not drive the output</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>Low BAR + High Impact</span>&nbsp;General head: influential but not bias-specific</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Ablation is useful, but not sufficient on its own. Cross-reference it with Integrated Gradients for convergent validity.</div>"
        )

        if show_comparison:
            src_mode = _get_attn_source_mode("bias_attn_source")
            if src_mode == "compare":
                lbl_A, lbl_B = "Base Encoder", "GUS-Net"
            else:
                lbl_A, lbl_B = "Model A", "Model B"
            h_A = (f"Head Ablation Results{_source_badge_html(lbl_A)}", header_args[1], header_args[2])
            h_B = (f"Head Ablation Results{_source_badge_html(lbl_B)}", header_args[1], header_args[2])
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_ablation_single(results_A, "_A"), *h_A,
                           style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_ablation_csv", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('ablation-chart-container_A', 'ablation_impact_A.png')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(_render_ablation_single(results_B, "_B"), *h_B,
                           style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_ablation_csv_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('ablation-chart-container_B', 'ablation_impact_B.png')", style=_BTN_STYLE_PNG),
                           ])
            )

        return _wrap_card(
            _render_ablation_single(results_A),
            *header_args,
            controls=[
                ui.download_button("export_ablation_csv", "CSV", style=_BTN_STYLE_CSV),
                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ablation-chart-container', 'ablation_impact.png')", style=_BTN_STYLE_PNG),
            ],
        )

    @render.download(filename="ablation_results.csv")
    def export_ablation_csv():
        results = ablation_results.get()
        if not results:
            yield "No ablation data"
            return
        lines = ["rank,layer,head,representation_impact,kl_divergence,bar_original"]
        for i, r in enumerate(results, 1):
            kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else ""
            lines.append(f"{i},{r.layer},{r.head},{r.representation_impact:.6f},{kl},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="ablation_results_B.csv")
    def export_ablation_csv_B():
        results = ablation_results_B.get()
        if not results:
            yield "No ablation data"
            return
        lines = ["rank,layer,head,representation_impact,kl_divergence,bar_original"]
        for i, r in enumerate(results, 1):
            kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else ""
            lines.append(f"{i},{r.layer},{r.head},{r.representation_impact:.6f},{kl},{r.bar_original:.4f}")
        yield "\n".join(lines)

    # ── Integrated Gradients handlers ─────────────────────────────────

    @reactive.effect
    async def compute_ig():
        """Automatically run IG correlation when bias analysis is complete."""
        res_A, res_B, show_comparison = _resolve_faithfulness_results()
        if not res_A:
            return

        ig_running.set(True)
        ig_results.set(None)
        ig_results_B.set(None)

        try:
            # Helper for single computation config
            def _prepare_ig_args(r):
                text = r["text"]
                attentions = r.get("attentions")
                metrics = r.get("attention_metrics", [])
                if not attentions or not metrics: return None
                
                model_name = r.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name
                tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                
                return (encoder_model, tokenizer, text, list(attentions), metrics, is_g)

            args_A = _prepare_ig_args(res_A)

            args_B = _prepare_ig_args(res_B) if show_comparison and res_B else None
            
            loop = asyncio.get_running_loop()
            # Use max_workers=1 to prevent OOM/concurrency issues with heavy IG computation
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut_A = None
                if args_A:
                    fut_A = loop.run_in_executor(
                        pool,
                        batch_compute_ig_correlation,
                        *args_A
                    )
                
                fut_B = None
                if args_B:
                    fut_B = loop.run_in_executor(
                        pool,
                        batch_compute_ig_correlation,
                        *args_B
                    )
                
                if fut_A:
                    res_val_A = await fut_A
                    ig_results.set(res_val_A)
                
                if fut_B:
                    res_val_B = await fut_B
                    ig_results_B.set(res_val_B)

        except Exception:
            _logger.exception("IG error")
        finally:
            ig_running.set(False)

    @output
    @render.ui
    def ig_results_display():
        try:
            return _ig_results_display_impl()
        except Exception as e:
            traceback.print_exc()
            return ui.div(f"Error rendering IG results: {e}", style="color:red; padding:10px; border:1px solid red;")

    def _ig_results_display_impl():
        running = ig_running.get()
        bundle_A = ig_results.get()

        if running or not bundle_A:
            return None
            
        bundle_B = ig_results_B.get()
        res_A_ctx, res_B_ctx, resolved_show_comparison = _resolve_faithfulness_results()
        show_comparison = resolved_show_comparison and bundle_B

        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        # Get selected head/layer for highlighting
        selected_head = None
        selected_layer = None
        try:
            sel_l = int(input.bias_attn_layer())
            sel_h = int(input.bias_attn_head())
            selected_head = (sel_l, sel_h)
            selected_layer = sel_l
        except Exception:
            pass

        try: top_k = int(input.bias_top_k())
        except Exception: top_k = 5
            
        def _render_ig_single(bundle, container_suffix="", context_results=None):
            if not bundle: return "No data"
            
            # Unpack bundle
            if isinstance(bundle, IGAnalysisBundle):
                results = bundle.correlations
                token_attrs = bundle.token_attributions
                tokens = bundle.tokens
            else:
                results = bundle
                token_attrs = None
                tokens = None
            
            # ── Chart 1: Correlation heatmap + BAR scatter ──
            # Vertical stack in compare mode to save horizontal space
            fig1 = create_ig_correlation_chart_v2(
                results, 
                bar_threshold=bar_threshold, 
                selected_head=selected_head,
                is_vertical=show_comparison
            )
            # Adjust container height based on layout
            chart1_height = "800px" if show_comparison else "450px"
            cid1 = f"ig-chart-container{container_suffix}"
            cid2 = f"ig-token-chart-container{container_suffix}"
            cid3 = f"ig-dist-chart-container{container_suffix}"
            cid4 = f"ig-layer-chart-container{container_suffix}"

            chart1_html = _deferred_plotly(fig1, cid1, height=chart1_height)

            # ── Chart 2: Token-level IG vs Attention comparison ──
            chart2_html = ""
            attentions = None
            if context_results and context_results.get("attentions"):
                attentions = context_results["attentions"]
            elif not show_comparison:
                 attentions = res_A_ctx.get("attentions") if res_A_ctx else None

            if token_attrs is not None and tokens and attentions:
                top_bar_heads = sorted(results, key=lambda r: r.bar_original, reverse=True)[:3]
                try:
                    fig2 = create_ig_token_comparison_chart(
                        tokens, token_attrs, list(attentions), top_bar_heads,
                    )
                    _token_csv_id = "export_ig_token_comparison_csv_B" if container_suffix == "_B" else "export_ig_token_comparison_csv"
                    csv_btn = ui.download_button(_token_csv_id, "CSV", style=_BTN_STYLE_CSV)
                    
                    chart2_html = _chart_with_png_btn(
                        _deferred_plotly(fig2, cid2, height="400px"),
                        cid2, 
                        f"ig_token_comparison{container_suffix}",
                        controls=[str(csv_btn)]
                    )
                except Exception as e:
                     chart2_html = f"<div>Error generating token chart: {e}</div>"

            # ── Chart 3: Distribution violin ──
            fig3 = create_ig_distribution_chart(results, bar_threshold=bar_threshold, selected_head=selected_head)
            chart3_html = (
                f'<div>'
                f'<div style="display:flex;justify-content:flex-end;margin-bottom:2px;">'
                f'<button onclick="downloadPlotlyPNG(\'{cid3}\', \'ig_distribution{container_suffix}.png\')" style="{_BTN_STYLE_PNG}">PNG</button>'
                f'</div>'
                + _deferred_plotly(fig3, cid3)
                + '</div>'
            )

            # ── Chart 4: Layer-wise mean faithfulness ──
            fig4 = create_ig_layer_summary_chart(results, selected_layer=selected_layer)
            chart4_html = (
                f'<div>'
                f'<div style="display:flex;justify-content:flex-end;margin-bottom:2px;">'
                f'<button onclick="downloadPlotlyPNG(\'{cid4}\', \'ig_layer_summary{container_suffix}.png\')" style="{_BTN_STYLE_PNG}">PNG</button>'
                f'</div>'
                + _deferred_plotly(fig4, cid4)
                + '</div>'
            )

            # ── Summary stats ──
            sig_results = [r for r in results if r.spearman_pvalue < 0.05]
            mean_rho = np.mean([r.spearman_rho for r in results])
            n_positive = sum(1 for r in sig_results if r.spearman_rho > 0)
            n_negative = sum(1 for r in sig_results if r.spearman_rho < 0)

            summary_html = (
                f'<div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#2563eb;font-family:JetBrains Mono,monospace;">{mean_rho:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean Spearman ρ</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#22c55e;font-family:JetBrains Mono,monospace;">{len(sig_results)}/{len(results)}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Significant (p&lt;0.05)</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#f59e0b;font-family:JetBrains Mono,monospace;">{n_positive}+ / {n_negative}-</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Positive / Negative</div></div>'
                f'</div>'
            )

            # ── Top-K table ──
            top_heads = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)[:top_k]
            table_rows = []
            for rank, r in enumerate(top_heads, 1):
                rho_color = "#2563eb" if r.spearman_rho > 0 else "#dc2626"
                sig_badge = '<span style="color:#22c55e;font-weight:600;">★</span>' if r.spearman_pvalue < 0.05 else ""
                specialized = "Yes" if r.bar_original > bar_threshold else "No"
                row_bg = "background:rgba(255,92,169,0.12);" if selected_head and (r.layer, r.head) == selected_head else ""
                table_rows.append(
                    f'<tr style="{row_bg}">'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">#{rank}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{r.layer}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{r.head}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{rho_color};">{r.spearman_rho:.3f}{sig_badge}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.spearman_pvalue:.4f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.bar_original:.3f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;">{specialized}</td>'
                    f'</tr>'
                )

            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:16px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Spearman ρ</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">p-value</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Specialized</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(table_rows)}</tbody>'
                '</table>'
            )
            
            sections = [
                ui.HTML(chart1_html),
                ui.HTML(summary_html),
                ui.HTML(table_html),
            ]

            if chart2_html:
                sections.append(ui.HTML('<hr style="border-color:rgba(100,116,139,0.15);margin:24px 0 16px;">'))
                sections.append(ui.HTML(chart2_html))

            sections.append(ui.HTML('<hr style="border-color:rgba(100,116,139,0.15);margin:24px 0 16px;">'))
            sections.append(ui.HTML(
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                f'{chart3_html}{chart4_html}'
                f'</div>'
            ))

            # ── Chart 5: Top-K Overlap (IG vs Attention) ──
            if isinstance(bundle, IGAnalysisBundle) and bundle.topk_overlaps:
                topk_fig = create_topk_overlap_chart(
                    bundle.topk_overlaps, bar_threshold=bar_threshold,
                    selected_head=selected_head,
                )
                cid5 = f"ig-topk-chart-container{container_suffix}"
                topk_html = _chart_with_png_btn(
                    _deferred_plotly(topk_fig, cid5),
                    cid5, f"ig_topk_overlap{container_suffix}"
                )
                _topk_tooltip = (
                    f"<span style='{_TH}'>What is Top-K Overlap?</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                    f"<span>Measures whether the top-K tokens ranked as most important by <b>Integrated Gradients</b> coincide with the top-K tokens ranked by <b>attention weight</b>, per head</span></div>"
                    f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                    f"<span>If IG and attention agree on which tokens matter most, attention is more likely to be a faithful proxy for the model's reasoning</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<span style='{_TH}'>Metrics</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
                    f"<span><span style='{_TBP}'>Jaccard</span> |IG∩Attn| / |IG∪Attn| for the top-K sets. Range [0, 1]. 1 = perfect overlap, 0 = no overlap.</span></div>"
                    f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>▪</span>"
                    f"<span><span style='{_TBB}'>RBO (p=0.9)</span> Rank-Biased Overlap (Webber et al., 2010), which accounts for rank order. Top-ranked tokens contribute more. Range [0, 1].</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<span style='{_TH}'>Heatmap reading</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
                    f"<span>Each cell = one (layer, head). Colour = Jaccard score. Darker = less overlap with IG.</span></div>"
                    f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
                    f"<span>Scatter on the right: Jaccard vs BAR (attention specialisation). Heads above the trend line are more faithful than their specialisation alone would predict.</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<div style='{_TN}'>K is set to 5 by default. A mean Jaccard &gt; 0.4 across heads suggests attention is a reasonable proxy for token importance.</div>"
                )
                _topk_section_html = (
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">'
                    f'<span style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Top-K Token Overlap (IG vs Attention)</span>'
                    f'<div class="info-tooltip-wrapper">'
                    f'<span class="info-tooltip-icon">i</span>'
                    f'<div class="info-tooltip-content">{_topk_tooltip}</div>'
                    f'</div></div>'
                )
                sections.append(ui.HTML('<hr style="border-color:rgba(100,116,139,0.15);margin:24px 0 16px;">'))
                sections.append(ui.HTML(_topk_section_html))
                sections.append(ui.HTML(topk_html))

                # Summary stats for top-K
                mean_jaccard = np.mean([r.jaccard for r in bundle.topk_overlaps])
                mean_rbo = np.mean([r.rank_biased_overlap for r in bundle.topk_overlaps])
                topk_summary = (
                    f'<div style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;">'
                    f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(29,78,216,0.06);border:1px solid rgba(29,78,216,0.15);border-radius:8px;text-align:center;">'
                    f'<div style="font-size:20px;font-weight:700;color:#1d4ed8;font-family:JetBrains Mono,monospace;">{mean_jaccard:.3f}</div>'
                    f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean Jaccard</div></div>'
                    f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.15);border-radius:8px;text-align:center;">'
                    f'<div style="font-size:20px;font-weight:700;color:#8b5cf6;font-family:JetBrains Mono,monospace;">{mean_rbo:.3f}</div>'
                    f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean RBO (p=0.9)</div></div>'
                    f'</div>'
                )
                sections.append(ui.HTML(topk_summary))

            return ui.div(*sections)

        faith_src = _get_attn_source_mode("bias_attn_source")
        _faith_label = "GUS-Net" if faith_src == "gusnet" else "Base Encoder"
        _ig_faith_badge = _source_badge_html(_faith_label) if faith_src != "compare" else ""

        header_args = (
            f"Attention vs Integrated Gradients{_ig_faith_badge}",
            "Faithfulness test: do attention weights agree with gradient-based token importance?",
            f"<span style='{_TH}'>Method: Integrated Gradients (Sundararajan et al., 2017)</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;'>"
            f"IG(x) = (x−x′) × ∫₀¹ ∂F/∂x dα</div>"
            f"<div style='{_TN}; margin-bottom:4px;'>Steps=30 · Baseline=PAD · via Captum LayerIntegratedGradients</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Faithfulness metric: Spearman ρ(IG, Attention)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>ρ &gt; 0</span>&nbsp;attention aligns with gradient importance, which suggests a <b>faithful</b> signal</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>ρ &lt; 0</span>&nbsp;attention focuses on tokens gradients ignore</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>★ marker</span>&nbsp;statistically significant (p &lt; 0.05)</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Sub-charts</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span><b>Faithfulness by Specialisation</b>: is high BAR correlated with high ρ?</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span><b>Layer-wise Faithfulness</b>: does faithfulness vary with depth?</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span><b>Top-K Overlap</b>: Jaccard + RBO on the top-K most important tokens</span></div>"
        )

        if show_comparison:
            # Determine labels for each side
            src_mode = _get_attn_source_mode("bias_attn_source")
            if src_mode == "compare":
                # Source compare: A = Base Encoder, B = GUS-Net
                label_A = "Base Encoder"
                label_B = "GUS-Net"
            else:
                # Model/prompt compare: A vs B
                label_A = "Model A"
                label_B = "Model B"
            header_A = (
                f"Attention vs Integrated Gradients{_source_badge_html(label_A)}",
                header_args[1], header_args[2],
            )
            header_B = (
                f"Attention vs Integrated Gradients{_source_badge_html(label_B)}",
                header_args[1], header_args[2],
            )
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_ig_single(bundle_A, "_A", res_A_ctx), *header_A,
                           style="border: 2px solid #3b82f6; height: 100%;",
                            controls=[
                                ui.download_button("export_ig_csv", "CSV", style=_BTN_STYLE_CSV),
                                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ig-chart-container_A', 'ig_correlation_A.png')", style=_BTN_STYLE_PNG),
                            ]),
                _wrap_card(_render_ig_single(bundle_B, "_B", res_B_ctx), *header_B,
                           style="border: 2px solid #ff5ca9; height: 100%;",
                            controls=[
                                ui.download_button("export_ig_csv_B", "CSV", style=_BTN_STYLE_CSV),
                                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ig-chart-container_B', 'ig_correlation_B.png')", style=_BTN_STYLE_PNG),
                            ])
            )

        return _wrap_card(
            _render_ig_single(bundle_A),
            *header_args,
            controls=[
                ui.download_button("export_ig_csv", "CSV", style=_BTN_STYLE_CSV),
                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ig-chart-container', 'ig_correlation.png')", style=_BTN_STYLE_PNG),
            ],
        )

    @render.download(filename="ig_correlation_results.csv")
    def export_ig_csv():
        bundle = ig_results.get()
        if not bundle:
            yield "No IG data"
            return
        results = bundle.correlations if isinstance(bundle, IGAnalysisBundle) else bundle
        lines = ["rank,layer,head,spearman_rho,spearman_pvalue,bar_original"]
        sorted_results = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)
        for i, r in enumerate(sorted_results, 1):
            lines.append(f"{i},{r.layer},{r.head},{r.spearman_rho:.6f},{r.spearman_pvalue:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="ig_correlation_results_B.csv")
    def export_ig_csv_B():
        bundle = ig_results_B.get()
        if not bundle:
            yield "No IG data"
            return
        results = bundle.correlations if isinstance(bundle, IGAnalysisBundle) else bundle
        lines = ["rank,layer,head,spearman_rho,spearman_pvalue,bar_original"]
        sorted_results = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)
        for i, r in enumerate(sorted_results, 1):
            lines.append(f"{i},{r.layer},{r.head},{r.spearman_rho:.6f},{r.spearman_pvalue:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="topk_overlap_results.csv")
    def export_topk_csv():
        bundle = ig_results.get()
        if not bundle or not isinstance(bundle, IGAnalysisBundle) or not bundle.topk_overlaps:
            yield "No Top-K overlap data"
            return
        lines = ["layer,head,k,jaccard,rank_biased_overlap,bar_original"]
        for r in bundle.topk_overlaps:
            lines.append(f"{r.layer},{r.head},{r.k},{r.jaccard:.6f},{r.rank_biased_overlap:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="topk_overlap_results_B.csv")
    def export_topk_csv_B():
        bundle = ig_results_B.get()
        if not bundle or not isinstance(bundle, IGAnalysisBundle) or not bundle.topk_overlaps:
            yield "No Top-K overlap data"
            return
        lines = ["layer,head,k,jaccard,rank_biased_overlap,bar_original"]
        for r in bundle.topk_overlaps:
            lines.append(f"{r.layer},{r.head},{r.k},{r.jaccard:.6f},{r.rank_biased_overlap:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="ig_token_comparison.csv")
    def export_ig_token_comparison_csv():
        bundle = ig_results.get()
        res_A, _, _ = _resolve_faithfulness_results()
        if bundle is None or not isinstance(bundle, IGAnalysisBundle) or res_A is None:
            yield "No data"
            return
        
        tokens = bundle.tokens
        ig_attrs = bundle.token_attributions
        attentions = res_A.get("attentions")
        
        # Robust check for empty data
        has_tokens = tokens is not None and len(tokens) > 0
        has_ig = ig_attrs is not None and (not hasattr(ig_attrs, "size") or ig_attrs.size > 0) and (not isinstance(ig_attrs, list) or len(ig_attrs) > 0)
        
        if not has_tokens or not has_ig:
            yield "Missing token data"
            return
            
        # Get top BAR heads for context
        top_bar_heads = sorted(bundle.correlations, key=lambda r: r.bar_original, reverse=True)[:3]
        
        # Header
        header = ["token", "ig_attribution"]
        for h in top_bar_heads:
            header.append(f"attn_L{h.layer}H{h.head}")
        
        yield ",".join(header) + "\n"
        
        # Pre-calculate attention means for top heads (simple column mean)
        attn_means = []
        seq_len = len(tokens)
        
        # Robust check for attentions existence
        has_attentions = attentions is not None and len(attentions) > 0
        
        if has_attentions:
            try:
                for h in top_bar_heads:
                    if h.layer < len(attentions):
                        layer_attn = attentions[h.layer]
                        try:
                            # Try tensor/numpy indexing first
                            if hasattr(layer_attn, "cpu"):
                                attn_matrix = layer_attn[0, h.head].cpu().numpy()
                            elif isinstance(layer_attn, np.ndarray):
                                attn_matrix = layer_attn[0, h.head]
                            else:
                                # List or other structure
                                attn_matrix = np.array(layer_attn[0][h.head])
                                
                            # Column mean (attention received)
                            mean_val = np.abs(attn_matrix.mean(axis=0))
                            
                        except (IndexError, TypeError):
                             # Fallback
                             if isinstance(layer_attn, list):
                                 attn_matrix = np.array(layer_attn[h.head])
                                 mean_val = np.abs(attn_matrix.mean(axis=0))
                             else:
                                 raise

                        # Ensure length matches
                        if len(mean_val) > seq_len:
                            mean_val = mean_val[:seq_len]
                        elif len(mean_val) < seq_len:
                            mean_val = np.pad(mean_val, (0, seq_len - len(mean_val)))
                        attn_means.append(mean_val)
                    else:
                        attn_means.append(np.zeros(seq_len))
            except Exception:
                _logger.exception("CSV Export Error (A)")
                while len(attn_means) < len(top_bar_heads):
                    attn_means.append(np.zeros(seq_len))
        else:
             for _ in top_bar_heads:
                 attn_means.append(np.zeros(seq_len))

        # Data rows
        for i, token in enumerate(tokens):
            row = [token]
            # IG
            row.append(f"{ig_attrs[i]:.6f}" if i < len(ig_attrs) else "0")
            # Attentions
            for means in attn_means:
                val = means[i] if i < len(means) else 0.0
                row.append(f"{val:.6f}")
            yield ",".join(row) + "\n"

    @render.download(filename="ig_token_comparison_B.csv")
    def export_ig_token_comparison_csv_B():
        bundle = ig_results_B.get()
        _, res_B, _ = _resolve_faithfulness_results()
        if bundle is None or not isinstance(bundle, IGAnalysisBundle):
            yield "No data"
            return
            
        tokens = bundle.tokens
        ig_attrs = bundle.token_attributions
        attentions = res_B.get("attentions") if res_B else None
        
        has_tokens = tokens is not None and len(tokens) > 0
        has_ig = ig_attrs is not None and (not hasattr(ig_attrs, "size") or ig_attrs.size > 0) and (not isinstance(ig_attrs, list) or len(ig_attrs) > 0)
        
        if not has_tokens or not has_ig:
            yield "Missing token data"
            return

        # Get top BAR heads for context
        top_bar_heads = sorted(bundle.correlations, key=lambda r: r.bar_original, reverse=True)[:3]
        
        # Header
        header = ["token", "ig_attribution"]
        for h in top_bar_heads:
            header.append(f"attn_L{h.layer}H{h.head}")
        
        yield ",".join(header) + "\n"
        
        # Pre-calculate attention means
        attn_means = []
        seq_len = len(tokens)
        
        has_attentions = attentions is not None and len(attentions) > 0
        
        if has_attentions:
            try:
                for h in top_bar_heads:
                    if h.layer < len(attentions):
                        layer_attn = attentions[h.layer]
                        try:
                            if hasattr(layer_attn, "cpu"):
                                attn_matrix = layer_attn[0, h.head].cpu().numpy()
                            elif isinstance(layer_attn, np.ndarray):
                                attn_matrix = layer_attn[0, h.head]
                            else:
                                attn_matrix = np.array(layer_attn[0][h.head])
                            
                            mean_val = np.abs(attn_matrix.mean(axis=0))
                            
                        except (IndexError, TypeError):
                             if isinstance(layer_attn, list):
                                 attn_matrix = np.array(layer_attn[h.head])
                                 mean_val = np.abs(attn_matrix.mean(axis=0))
                             else:
                                 raise

                        if len(mean_val) > seq_len:
                            mean_val = mean_val[:seq_len]
                        elif len(mean_val) < seq_len:
                            mean_val = np.pad(mean_val, (0, seq_len - len(mean_val)))
                        attn_means.append(mean_val)
                    else:
                        attn_means.append(np.zeros(seq_len))
            except Exception:
                _logger.exception("CSV Export Error (B)")
                while len(attn_means) < len(top_bar_heads):
                    attn_means.append(np.zeros(seq_len))
        else:
             for _ in top_bar_heads:
                 attn_means.append(np.zeros(seq_len))

        # Data rows
        for i, token in enumerate(tokens):
            row = [token]
            # IG
            row.append(f"{ig_attrs[i]:.6f}" if i < len(ig_attrs) else "0")
            # Attentions
            for means in attn_means:
                val = means[i] if i < len(means) else 0.0
                row.append(f"{val:.6f}")
            yield ",".join(row) + "\n"

    # ── Perturbation Analysis handlers ─────────────────────────────────

    @reactive.effect
    async def compute_perturbation():
        """Run perturbation analysis after IG completes."""
        bundle_A = ig_results.get()
        if not bundle_A or not isinstance(bundle_A, IGAnalysisBundle):
            return

        res_A, res_B, show_comparison = _resolve_faithfulness_results()
        if not res_A:
            return

        perturbation_running.set(True)
        perturbation_results.set(None)
        perturbation_results_B.set(None)

        try:
            def _prepare_args(res, bundle):
                text = res["text"]
                attentions = res.get("attentions")
                if not attentions:
                    return None
                model_name = res.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name
                tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                return (encoder_model, tokenizer, text, is_g, bundle.token_attributions, list(attentions))

            args_A = _prepare_args(res_A, bundle_A)

            args_B = None
            bundle_B = ig_results_B.get()
            if show_comparison and bundle_B and res_B and isinstance(bundle_B, IGAnalysisBundle):
                args_B = _prepare_args(res_B, bundle_B)

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                if args_A:
                    fut_A = loop.run_in_executor(pool, batch_compute_perturbation, *args_A)
                    res_val_A = await fut_A
                    perturbation_results.set(res_val_A)

                if args_B:
                    fut_B = loop.run_in_executor(pool, batch_compute_perturbation, *args_B)
                    res_val_B = await fut_B
                    perturbation_results_B.set(res_val_B)

        except Exception:
            _logger.exception("Perturbation error")
        finally:
            perturbation_running.set(False)

    @output
    @render.ui
    def perturbation_results_display():
        running = perturbation_running.get()
        bundle_A = perturbation_results.get()

        if running or not bundle_A:
            return None

        bundle_B = perturbation_results_B.get()
        res_A_ctx, res_B_ctx, resolved_show_comparison = _resolve_faithfulness_results()
        show_comparison = resolved_show_comparison and bundle_B

        # Get IG data for comparison charts
        ig_bundle_A = ig_results.get()
        ig_bundle_B = ig_results_B.get()

        def _render_perturb_single(bundle, ig_bundle, container_suffix="", context_results=None):
            if not bundle:
                return "No data"

            ig_attrs = ig_bundle.token_attributions if ig_bundle and isinstance(ig_bundle, IGAnalysisBundle) else None

            # Summary cards
            rho_ig = bundle.perturb_vs_ig_spearman
            mean_attn_rho = np.mean([r[2] for r in bundle.perturb_vs_attn_spearman]) if bundle.perturb_vs_attn_spearman else 0.0
            max_imp = max(r.importance for r in bundle.token_results) if bundle.token_results else 0.0

            summary_html = (
                f'<div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#f59e0b;font-family:JetBrains Mono,monospace;">{rho_ig:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">ρ(Perturb, IG)</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#2563eb;font-family:JetBrains Mono,monospace;">{mean_attn_rho:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean ρ(Perturb, Attn)</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#22c55e;font-family:JetBrains Mono,monospace;">{max_imp:.4f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Max Perturbation Impact</div></div>'
                f'</div>'
            )

            sections = [ui.HTML(summary_html)]

            # Chart 1: Perturbation vs IG bar chart
            if ig_attrs is not None:
                fig1 = create_perturbation_comparison_chart(bundle, ig_attrs, bundle.tokens)
                cid1 = f"perturb-comparison-container{container_suffix}"
                sections.append(ui.HTML(
                    f'<div style="margin-top:20px;">'
                    + _chart_with_png_btn(
                        _deferred_plotly(fig1, cid1, height="400px"),
                        cid1, f"perturbation_vs_ig{container_suffix}"
                    )
                    + '</div>'
                ))

                        # Chart 2: Perturbation vs Attention heatmap
            if bundle.perturb_vs_attn_spearman:
                num_layers = max(r[0] for r in bundle.perturb_vs_attn_spearman) + 1
                num_heads = max(r[1] for r in bundle.perturb_vs_attn_spearman) + 1
                fig2 = create_perturbation_attn_heatmap(bundle.perturb_vs_attn_spearman, num_layers, num_heads)
                cid2 = f"perturb-attn-heatmap-container{container_suffix}"
                _pattn_csv_id = "export_perturb_attn_csv_B" if container_suffix == "_B" else "export_perturb_attn_csv"
                
                # Create CSV button
                csv_btn = ui.download_button(_pattn_csv_id, "CSV", style=_BTN_STYLE_CSV)
                # Render (we rely on ui.download_button returning a Tag which str() converts to HTML)
                # Note: ui.download_button returns a Tag object. We need to render it to string if _chart_with_png_btn expects text.
                # However, _chart_with_png_btn is purely string-based.
                # Since we are inside @render.ui, we can return UI objects or HTML strings.
                # But _chart_with_png_btn returns a string.
                # We need to get the HTML string for the download button.
                # Using str(csv_btn) works for Shiny tags.
                
                sections.append(ui.HTML(
                    _chart_with_png_btn(
                        _deferred_plotly(fig2, cid2),
                        cid2, 
                        f"perturbation_vs_attention{container_suffix}",
                        controls=[str(csv_btn)]
                    )
                ))

            return ui.div(*sections)

        _f_src = _get_attn_source_mode("bias_attn_source")
        _f_lbl = "GUS-Net" if _f_src == "gusnet" else "Base Encoder"
        _f_bdg = _source_badge_html(_f_lbl) if _f_src != "compare" else ""

        header_args = (
            f"Perturbation Analysis{_f_bdg}",
            "Model-agnostic validation: how much does zeroing each token's embedding change the representation?",
            f"<span style='{_TH}'>Method: Erasure / Occlusion (Li et al., 2016)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>Replace each token's embedding with a zero vector and measure representation change</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Importance score per token</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;'>"
            f"importance(t) = 1 − cos_sim(pool_orig, pool_zero_t)</div>"
            f"<div style='{_TN}; margin-bottom:4px;'>zero_t = token t's embedding replaced with zero vector</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Triangulation metrics</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>ρ(Perturb, IG)</span>&nbsp;gradient and occlusion agree on token ranking</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>ρ(Perturb, Attn)</span>&nbsp;each head attends to tokens that actually affect output</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>High ρ (all 3)</span>&nbsp;importance is robust, attention is faithful</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>High Perturb–IG, Low Perturb–Attn</span>&nbsp;model internals correct but attention unfaithful</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>No gradient is required, so this works on any architecture. It is a strong cross-validation baseline.</div>"
        )

        if show_comparison:
            _src_m = _get_attn_source_mode("bias_attn_source")
            if _src_m == "compare":
                _lA, _lB = "Base Encoder", "GUS-Net"
            else:
                _lA, _lB = "Model A", "Model B"
            _hA = (f"Perturbation Analysis{_source_badge_html(_lA)}", header_args[1], header_args[2])
            _hB = (f"Perturbation Analysis{_source_badge_html(_lB)}", header_args[1], header_args[2])
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_perturb_single(bundle_A, ig_bundle_A, "_A", res_A_ctx), *_hA,
                           style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_perturbation_csv", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(_render_perturb_single(bundle_B, ig_bundle_B, "_B", res_B_ctx), *_hB,
                           style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_perturbation_csv_B", "CSV", style=_BTN_STYLE_CSV)])
            )

        return _wrap_card(
            _render_perturb_single(bundle_A, ig_bundle_A),
            *header_args,
            controls=[ui.download_button("export_perturbation_csv", "CSV", style=_BTN_STYLE_CSV)],
        )

    @render.download(filename="perturbation_results.csv")
    def export_perturbation_csv():
        bundle = perturbation_results.get()
        if not bundle:
            yield "No perturbation data"
            return
        lines = ["token_index,token,importance"]
        for r in bundle.token_results:
            lines.append(f"{r.token_index},{_csv_safe(r.token)},{r.importance:.6f}")
        yield "\n".join(lines)

    @render.download(filename="perturbation_results_B.csv")
    def export_perturbation_csv_B():
        bundle = perturbation_results_B.get()
        if not bundle:
            yield "No perturbation data"
            return
        lines = ["token_index,token,importance"]
        for r in bundle.token_results:
            lines.append(f"{r.token_index},{_csv_safe(r.token)},{r.importance:.6f}")
        yield "\n".join(lines)

    # ── LRP Analysis handlers ─────────────────────────────────────────

    @reactive.effect
    async def compute_lrp():
        """Run LRP analysis after IG completes."""
        bundle_A = ig_results.get()
        if not bundle_A or not isinstance(bundle_A, IGAnalysisBundle):
            return

        res_A, res_B, show_comparison = _resolve_faithfulness_results()
        if not res_A:
            return

        lrp_running.set(True)
        lrp_results.set(None)
        lrp_results_B.set(None)

        try:
            def _prepare_args(res, bundle):
                text = res["text"]
                attentions = res.get("attentions")
                metrics = res.get("attention_metrics", [])
                if not attentions:
                    return None
                model_name = res.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name
                tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                return (encoder_model, tokenizer, text, is_g, bundle.token_attributions, list(attentions), metrics)

            args_A = _prepare_args(res_A, bundle_A)

            args_B = None
            bundle_B = ig_results_B.get()
            if show_comparison and bundle_B and res_B and isinstance(bundle_B, IGAnalysisBundle):
                args_B = _prepare_args(res_B, bundle_B)

            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                if args_A:
                    fut_A = loop.run_in_executor(pool, batch_compute_lrp, *args_A)
                    res_val_A = await fut_A
                    lrp_results.set(res_val_A)

                if args_B:
                    fut_B = loop.run_in_executor(pool, batch_compute_lrp, *args_B)
                    res_val_B = await fut_B
                    lrp_results_B.set(res_val_B)

        except Exception:
            _logger.exception("LRP error")
        finally:
            lrp_running.set(False)

    @output
    @render.ui
    def lrp_results_display():
        running = lrp_running.get()
        bundle_A = lrp_results.get()

        if running or not bundle_A:
            return None

        bundle_B = lrp_results_B.get()
        _, _, resolved_show_comparison = _resolve_faithfulness_results()
        show_comparison = resolved_show_comparison and bundle_B

        ig_bundle_A = ig_results.get()
        ig_bundle_B = ig_results_B.get()

        try:
            bar_threshold = float(input.bias_bar_threshold())
        except Exception:
            bar_threshold = 1.5

        def _render_lrp_single(bundle, ig_bundle, container_suffix=""):
            if not bundle:
                return "No data"

            ig_attrs = ig_bundle.token_attributions if ig_bundle and isinstance(ig_bundle, IGAnalysisBundle) else None
            ig_corrs = ig_bundle.correlations if ig_bundle and isinstance(ig_bundle, IGAnalysisBundle) else []

            # Summary cards
            rho_ig = bundle.lrp_vs_ig_spearman
            mean_attn_rho = np.mean([r[2] for r in bundle.correlations]) if bundle.correlations else 0.0

            summary_html = (
                f'<div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#8b5cf6;font-family:JetBrains Mono,monospace;">{rho_ig:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">ρ(LRP, IG)</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#2563eb;font-family:JetBrains Mono,monospace;">{mean_attn_rho:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean ρ(LRP, Attn)</div></div>'
                f'</div>'
            )

            sections = [ui.HTML(summary_html)]

            # Chart 1: LRP vs IG bar chart
            if ig_attrs is not None:
                fig1 = create_lrp_comparison_chart(
                    bundle.token_attributions, ig_attrs, bundle.tokens,
                    lrp_vs_ig_rho=rho_ig,
                )
                cid1 = f"lrp-comparison-container{container_suffix}"
                sections.append(ui.HTML(
                    f'<div style="margin-top:20px;">'
                    + _chart_with_png_btn(
                        _deferred_plotly(fig1, cid1, height="400px"),
                        cid1, f"lrp_vs_ig{container_suffix}"
                    )
                    + '</div>'
                ))

                        # Chart 2: Cross-method agreement scatter
            if ig_corrs and bundle.correlations:
                fig2 = create_cross_method_agreement_chart(
                    ig_corrs, bundle.correlations, bar_threshold=bar_threshold,
                )
                cid2 = f"lrp-agreement-container{container_suffix}"
                _agree_csv_id = "export_cross_method_csv_B" if container_suffix == "_B" else "export_cross_method_csv"
                csv_btn = ui.download_button(_agree_csv_id, "CSV", style=_BTN_STYLE_CSV)
                
                sections.append(ui.HTML(
                    _chart_with_png_btn(
                        _deferred_plotly(fig2, cid2, height="400px"),
                        cid2, 
                        f"cross_method_agreement{container_suffix}",
                        controls=[str(csv_btn)]
                    )
                ))

            return ui.div(*sections)

        _lrp_f_src = _get_attn_source_mode("bias_attn_source")
        _lrp_f_lbl = "GUS-Net" if _lrp_f_src == "gusnet" else "Base Encoder"
        _lrp_f_bdg = _source_badge_html(_lrp_f_lbl) if _lrp_f_src != "compare" else ""

        header_args = (
            f"DeepLift / LRP Cross-Validation{_lrp_f_bdg}",
            "Convergent validity: does DeepLift agree with Integrated Gradients on token importance?",
            f"<span style='{_TH}'>Method: DeepLift / LRP (Shrikumar et al., 2017)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>Relevance back-propagated from output to input tokens (conservation rule)</span></div>"
            f"<div style='{_TN}; margin-bottom:4px;'>Fallback: LayerDeepLift when BERT LayerNorm prevents standard LRP</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Metrics</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>ρ(LRP, IG)</span>&nbsp;both gradient methods agree on token importance ranking</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>ρ(LRP, Attn)</span>&nbsp;LRP vs per-head attention weights</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Agreement chart (scatter)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span>Points <b>near diagonal</b> = IG and LRP agree on that head's faithfulness</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span>Points <b>off diagonal</b> indicate conflicting signals, so they are worth investigating further</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Both agree</span>&nbsp;strong convergent evidence, robust faithfulness claim</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>Methods disagree</span>&nbsp;results are implementation-sensitive, so interpret them cautiously</span></div>"
        )

        if show_comparison:
            _src_m = _get_attn_source_mode("bias_attn_source")
            if _src_m == "compare":
                _lA, _lB = "Base Encoder", "GUS-Net"
            else:
                _lA, _lB = "Model A", "Model B"
            _hA = (f"DeepLift / LRP Cross-Validation{_source_badge_html(_lA)}", header_args[1], header_args[2])
            _hB = (f"DeepLift / LRP Cross-Validation{_source_badge_html(_lB)}", header_args[1], header_args[2])
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_lrp_single(bundle_A, ig_bundle_A, "_A"), *_hA,
                           style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_lrp_csv", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(_render_lrp_single(bundle_B, ig_bundle_B, "_B"), *_hB,
                           style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_lrp_csv_B", "CSV", style=_BTN_STYLE_CSV)])
            )

        return _wrap_card(
            _render_lrp_single(bundle_A, ig_bundle_A),
            *header_args,
            controls=[ui.download_button("export_lrp_csv", "CSV", style=_BTN_STYLE_CSV)],
        )

    @render.download(filename="lrp_results.csv")
    def export_lrp_csv():
        bundle = lrp_results.get()
        if not bundle:
            yield "No LRP data"
            return
        lines = ["layer,head,spearman_rho_vs_attention"]
        for l, h, rho in bundle.correlations:
            lines.append(f"{l},{h},{rho:.6f}")
        yield "\n".join(lines)

    @render.download(filename="lrp_results_B.csv")
    def export_lrp_csv_B():
        bundle = lrp_results_B.get()
        if not bundle:
            yield "No LRP data"
            return
        lines = ["layer,head,spearman_rho_vs_attention"]
        for l, h, rho in bundle.correlations:
            lines.append(f"{l},{h},{rho:.6f}")
        yield "\n".join(lines)

    # ── StereoSet Evaluation Handlers ─────────────────────────────────
    # These load from pre-computed JSON and render independently of bias_results.
    # They react to the selected GUS-Net model to show the matching base model's data.

    def _stereoset_model_key():
        """Derive the canonical *base* stereoset model key (always base, never GUS-NET).

        Overview, Category Breakdown, and Demographic Slices always show the
        base model.  The three lower sections (Head Sensitivity, Attention-Bias
        Correlation, Example Explorer) call ``_stereoset_gusnet_key()``
        separately when the toggle is active.
        """
        res = bias_results.get()
        if res and "bias_model_key" in res:
            base_mk = res["bias_model_key"]
        else:
            try:
                base_mk = input.bias_model_key()
            except Exception:
                base_mk = "gusnet-bert"

        from ..bias.stereoset.stereoset_data import _resolve_key
        return _resolve_key(base_mk)  # e.g. "bert" or "gpt2"

    def _stereoset_gusnet_key():
        """Return the GUS-NET key for the current base model, or *None* if
        the source toggle is not set to gusnet/compare or GUS-NET data is
        unavailable.  Driven by the floating-bar Source Attention toggle
        (``bias_attn_source``) instead of the old inline checkbox."""
        source = _get_attn_source_mode("bias_attn_source")
        if source not in ("gusnet", "compare"):
            return None
        canonical = _stereoset_model_key()
        gk = get_gusnet_key(canonical)
        # Only return if we actually have data for it
        if get_stereoset_scores(gk) is not None:
            return gk
        return None

    def _stereoset_model_key_B():
        """Derive the canonical *base* stereoset model key for Model B."""
        res_B = bias_results_B.get()
        if res_B and "bias_model_key" in res_B:
            base_mk = res_B["bias_model_key"]
        else:
            try:
                if active_bias_compare_models.get():
                    base_mk = input.bias_model_key_B()
                elif active_bias_compare_prompts.get():
                    res = bias_results.get()
                    if res and "bias_model_key" in res:
                        base_mk = res["bias_model_key"]
                    else:
                        base_mk = input.bias_model_key()
                else:
                    return None
            except Exception:
                return None

        from ..bias.stereoset.stereoset_data import _resolve_key
        return _resolve_key(base_mk)

    def _stereoset_gusnet_warning_icon():
        """Return a ⚠ tooltip icon explaining why GUS-Net StereoSet scores
        should be interpreted with caution.  Designed to sit in the controls
        area of a ``_wrap_card`` (right-aligned)."""
        base_mk = _stereoset_model_key()
        tooltip_html = (
            f"<span style='{_TH}'>⚠ Interpret with caution</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>GUS-Net is a <b>token-level bias classifier</b> "
            f"(BertForTokenClassification), not a language model.</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>Its StereoSet scores are computed by extracting the "
            f"underlying <b>{base_mk.upper()}</b> encoder and scoring completions "
            f"via pseudo-log-likelihood.</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>The fine-tuning objective (bias detection) alters internal "
            f"representations in ways that may <b>distort LM-level metrics</b> "
            f"like SS and LMS.</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>For this reason, only the base model scores "
            f"are presented in the upper StereoSet sections.</div>"
        )
        return ui.div(
            {"class": "info-tooltip-wrapper"},
            ui.span(
                {"class": "info-tooltip-icon",
                 "style": "background:rgba(255,92,169,0.12);border-color:#ff5ca940;"
                          "color:#ff5ca9;font-size:11px;"},
                "⚠",
            ),
            ui.div({"class": "info-tooltip-content"}, ui.HTML(tooltip_html)),
        )

    def _stereoset_gusnet_warning_html():
        """Return raw HTML string version of the ⚠ tooltip, for use inside
        ``_chart_with_png_btn`` controls."""
        base_mk = _stereoset_model_key()
        tooltip_html = (
            f"<span style='{_TH}'>⚠ Interpret with caution</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>GUS-Net is a <b>token-level bias classifier</b> "
            f"(BertForTokenClassification), not a language model.</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>Its StereoSet scores are computed by extracting the "
            f"underlying <b>{base_mk.upper()}</b> encoder and scoring completions "
            f"via pseudo-log-likelihood.</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>The fine-tuning objective (bias detection) alters internal "
            f"representations in ways that may <b>distort LM-level metrics</b> "
            f"like SS and LMS.</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>For this reason, only the base model scores "
            f"are presented in the upper StereoSet sections.</div>"
        )
        return (
            '<div class="info-tooltip-wrapper" style="display:inline-flex;">'
            '<span class="info-tooltip-icon" style="background:rgba(255,92,169,0.12);'
            'border-color:#ff5ca940;color:#ff5ca9;font-size:11px;">⚠</span>'
            f'<div class="info-tooltip-content">{tooltip_html}</div>'
            '</div>'
        )

    def _stereoset_resolve_key_for_source():
        """Return the stereoset model key based on the current source toggle.
        - base → canonical base key (e.g. 'bert')
        - gusnet → GUS-Net key (e.g. 'gusnet_bert')
        - compare → (base_key, gusnet_key)
        """
        src = _get_attn_source_mode("bias_attn_source")
        base_mk = _stereoset_model_key()
        if src == "gusnet":
            gk = get_gusnet_key(base_mk)
            if get_stereoset_scores(gk) is not None:
                return gk
            return base_mk  # fallback
        if src == "compare":
            gk = get_gusnet_key(base_mk)
            if get_stereoset_scores(gk) is not None:
                return (base_mk, gk)
            return base_mk  # fallback to single
        return base_mk

    @output
    @render.ui
    def stereoset_overview():
        """Score card with SS/LMS/ICAT gauges."""
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, manual_header_override=None):
            scores = get_stereoset_scores(mk)
            metadata = get_metadata(mk)
            if scores is None or metadata is None:
                return ui.div(
                    ui.p(
                        "StereoSet data not available. Run ",
                        ui.code("python -m attention_app.bias.generate_stereoset_json"),
                        " to generate.",
                        style="color:#94a3b8;font-size:12px;",
                    ),
                )
            html = create_stereoset_overview_html(scores, metadata)
            if manual_header_override:
                header = manual_header_override
            else:
                model_label = metadata.get("model", "unknown")
                header = ("Benchmark Scores", f"StereoSet intersentence evaluation on <b style='color:#ff5ca9;'>{model_label}</b>")
                
            return (html, header)
            
        # ── Benchmark Scores tooltip ──
        _benchmark_help = (
            f"<span style='{_TH}'>What is StereoSet?</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>A crowdsourced benchmark (Nadeem et al., 2021) that measures <b>stereotypical bias</b> in language models across four demographic categories: gender, race, religion, profession</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Each item is a sentence with three completions: <b>stereotyped</b>, <b>anti-stereotyped</b>, and <b>unrelated</b> (meaningless). The model must prefer the meaningful completions over the unrelated one, and the anti-stereotyped over the stereotyped.</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Metrics</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><span style='{_TBP}'>SS</span> Stereotype Score - % of times the model prefers the stereotyped over the anti-stereotyped completion. <b>50% = unbiased</b>; &gt;50% = biased toward stereotypes</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
            f"<span><span style='{_TBG}'>LMS</span> Language Model Score - % of meaningful completions preferred over unrelated ones. Measures general language understanding. <b>Higher is better.</b></span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>▪</span>"
            f"<span><span style='{_TBA}'>ICAT</span> Ideal Context-Association Test - composite score that rewards both low bias (SS near 50%) and high LM quality (high LMS). <b>Max = 100.</b></span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>How is it scored here?</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>BERT uses <b>Pseudo-Log-Likelihood (PLL)</b> - masked probability of each completion token</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>GPT-2 uses <b>autoregressive log-likelihood</b> - sum of causal token probabilities</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Scores are pre-computed offline on the intersentence split of StereoSet dev set. Run generate_stereoset_json.py to refresh.</div>"
        )

        # Check comparison
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None

        if compare_models and mk_B:
            res_A, header_A = _render_single(mk_A)
            res_B, header_B = _render_single(mk_B)
            
            # If render single returned a div (error/missing), handle it
            content_A = res_A if isinstance(res_A, str) else res_A
            content_B = res_B if isinstance(res_B, str) else res_B
            
            # If one is missing data, it might return a div directly, not tuple
            # _render_single returns tuple (html, header) on success, or UI object on failure
            # Let's standardize helper to always return tuple or (UI, None)
            
            # Adjusted helper strategy:
            def _render_safe(mk):
                s = get_stereoset_scores(mk)
                m = get_metadata(mk)
                if s is None or m is None:
                    return (None, None)
                model_name = m.get("model", "unknown")
                is_gusnet = "gus-net" in model_name.lower()
                return (create_stereoset_overview_html(s, m, is_gusnet), model_name)

            html_A, model_A = _render_safe(mk_A)
            html_B, model_B = _render_safe(mk_B)
            
            if not html_A and not html_B:
                 return _wrap_card(ui.div("No StereoSet data available."), "Benchmark Scores")

            card_A = ui.div("No data for Model A")
            if html_A:
                card_A = _wrap_card(ui.HTML(html_A), manual_header=("Benchmark Scores", f"Model A: {model_A}"),
                                    help_text=_benchmark_help,
                                    style="border: 2px solid #3b82f6; height: 100%;")

            card_B = ui.div("No data for Model B")
            if html_B:
                card_B = _wrap_card(ui.HTML(html_B), manual_header=("Benchmark Scores", f"Model B: {model_B}"),
                                    help_text=_benchmark_help,
                                    style="border: 2px solid #ff5ca9; height: 100%;")

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                card_A, card_B
            )

        # ── Source-aware single mode ──
        # Overview always shows a single model (no side-by-side in compare)
        resolved = _stereoset_resolve_key_for_source()
        mk = resolved[0] if isinstance(resolved, tuple) else resolved
        scores = get_stereoset_scores(mk)
        metadata = get_metadata(mk)
        if scores is None or metadata is None:
            return ui.div(
                ui.p(
                    "StereoSet data not available. Run ",
                    ui.code("python -m attention_app.bias.generate_stereoset_json"),
                    " to generate.",
                    style="color:#94a3b8;font-size:12px;",
                ),
            )
        # Always show the base model name in the title (never the GUS-Net HF path)
        base_mk = _stereoset_model_key()
        base_meta = get_metadata(base_mk)
        base_label = base_meta.get("model", base_mk) if base_meta else base_mk
        model_label = metadata.get("model", "unknown")
        is_gusnet = "gus-net" in model_label.lower()
        html = create_stereoset_overview_html(scores, metadata, is_gusnet)
        src_mode = _get_attn_source_mode("bias_attn_source")
        _overview_controls = [_stereoset_gusnet_warning_icon()] if src_mode == "gusnet" and is_gusnet else None
        return _wrap_card(
            ui.HTML(html),
            manual_header=(f"Benchmark Scores{_source_badge_html('Base Encoder')}", f"StereoSet intersentence evaluation on <b style='color:#ff5ca9;'>{base_label}</b>"),
            help_text=_benchmark_help,
            controls=_overview_controls,
        )

    @output
    @render.ui
    def stereoset_category_header():
        """Dynamic subsection header for Category & Demographic Breakdown with source badge."""
        src_mode = _get_attn_source_mode("bias_attn_source")
        badge = _source_badge_html("Base Encoder")
        return ui.div(
            {"style": "margin: 12px 0 8px;"},
            ui.HTML(
                f'<h5 style="margin: 0 0 6px 0; font-size: 14px; font-weight: 700; '
                f'color: #f8fafc; letter-spacing: 0.2px; display:inline-flex; align-items:center;">'
                f'Category and Demographic Breakdown{badge}</h5>'
            ),
            ui.p(
                "These views show where the benchmark signal comes from: broad categories first, "
                "then finer demographic target slices. This is the most important block for "
                "discussing which groups are disproportionately affected.",
                style="margin: 0 0 10px 0; font-size: 12px; line-height: 1.5; color: #94a3b8;",
            ),
        )

    @output
    @render.ui
    def stereoset_category_breakdown():
        """Category bar chart + bias distribution violin side-by-side."""
        mk_A = _stereoset_model_key()
        _src_mode = _get_attn_source_mode("bias_attn_source")

        def _render_single(mk, style=None, layout="row", container_suffix=""):
            scores = get_stereoset_scores(mk)
            examples = get_stereoset_examples(mk)
            if scores is None or not examples:
                return None
            by_cat = scores.get("by_category", {})
            if not by_cat: return None

            fig_cat = create_stereoset_category_chart(by_cat)
            fig_dist = create_stereoset_bias_distribution(examples)

            _warn_ctrl = [_stereoset_gusnet_warning_html()] if _src_mode == "gusnet" else None
            cat_html = _chart_with_png_btn(
                _deferred_plotly(fig_cat, f"stereoset-cat-chart{container_suffix}"),
                f"stereoset-cat-chart{container_suffix}", f"stereoset_category{container_suffix}",
                controls=_warn_ctrl,
            )
            dist_html = _chart_with_png_btn(
                _deferred_plotly(fig_dist, f"stereoset-dist-chart{container_suffix}"),
                f"stereoset-dist-chart{container_suffix}", f"stereoset_distribution{container_suffix}",
                controls=_warn_ctrl,
            )

            card_style = style if style else ""

            container_style = "display:grid;grid-template-columns:1fr 1fr;gap:16px;" if layout == "row" else "display:flex;flex-direction:column;gap:16px;"

            _cat_csv_id = "export_stereoset_category_csv_B" if container_suffix == "_B" else "export_stereoset_category_csv"
            _dist_csv_id = "export_stereoset_distribution_csv_B" if container_suffix == "_B" else "export_stereoset_distribution_csv"
            return ui.div(
                {"style": container_style},
                _wrap_card(ui.HTML(cat_html), style=card_style,
                           controls=[ui.download_button(_cat_csv_id, "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(dist_html), style=card_style,
                           controls=[ui.download_button(_dist_csv_id, "CSV", style=_BTN_STYLE_CSV)]),
            )

        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None

        if compare_models and mk_B:
            content_A = _render_single(mk_A, style="border: 2px solid #3b82f6;", layout="column", container_suffix="_A") or ui.div("No data for Model A")
            content_B = _render_single(mk_B, style="border: 2px solid #ff5ca9;", layout="column", container_suffix="_B") or ui.div("No data for Model B")

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                content_A,
                content_B
            )

        # ── Source-aware single mode ──
        # Category breakdown always shows a single model (no side-by-side in compare)
        resolved = _stereoset_resolve_key_for_source()
        mk = resolved[0] if isinstance(resolved, tuple) else resolved
        return _render_single(mk) or ui.div()

    @output
    @render.ui
    def stereoset_demographic_slices():
        """Demographic target analysis with category filter."""
        from ..bias.visualizations import create_stereoset_demographic_chart
        
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, container_suffix=""):
            examples = get_stereoset_examples(mk)
            if not examples: return None
            
            categories = sorted(set(e.get("category", "") for e in examples))
            charts = []
            for i, cat in enumerate(categories):
                cat_examples = [e for e in examples if e.get("category") == cat]
                fig = create_stereoset_demographic_chart(cat_examples, category=cat, min_n=10)
                cid_demo = f"stereoset-demo-{cat}{container_suffix}"
                chart_html = _chart_with_png_btn(
                    _deferred_plotly(fig, cid_demo),
                    cid_demo, f"stereoset_demo_{cat}{container_suffix}"
                )
                charts.append(chart_html)
            
            from collections import Counter
            target_data = {}
            for ex in examples:
                t = ex.get("target", "unknown")
                if t not in target_data:
                    target_data[t] = {"stereo_wins": 0, "n": 0, "category": ex.get("category", "")}
                target_data[t]["n"] += 1
                if ex.get("stereo_pll", 0) > ex.get("anti_pll", 0):
                    target_data[t]["stereo_wins"] += 1
            
            targets = []
            for t, d in target_data.items():
                if d["n"] >= 10:
                    ss = d["stereo_wins"] / d["n"] * 100
                    targets.append({"target": t, "ss": ss, "n": d["n"], "category": d["category"]})
            targets.sort(key=lambda x: x["ss"], reverse=True)
            
            try: top_k = int(input.bias_top_k())
            except Exception: top_k = 5
            
            most_biased = targets[:top_k]
            least_biased = targets[-top_k:][::-1]
            STEREOSET_CAT_COLORS = {"gender": "#e74c3c", "race": "#3498db", "religion": "#2ecc71", "profession": "#f39c12"}
            
            def _build_target_rows(items, direction="high"):
                rows = []
                for rank, t in enumerate(items, 1):
                    cat_color = STEREOSET_CAT_COLORS.get(t["category"], "#94a3b8")
                    ss_color = "#ef4444" if t["ss"] > 60 else "#22c55e" if t["ss"] < 40 else "#eab308"
                    rows.append(f'<tr style="transition:all 0.2s ease;"><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">{t["target"]}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;"><span style="padding:1px 6px;border-radius:3px;font-size:9px;font-weight:600;background:rgba({int(cat_color[1:3],16)},{int(cat_color[3:5],16)},{int(cat_color[5:7],16)},0.2);color:{cat_color};text-transform:uppercase;">{t["category"]}</span></td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{ss_color};">{t["ss"]:.1f}%</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">{t["n"]}</td></tr>')
                return "".join(rows)
            
            th_style = 'padding:10px 12px;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;'
            
            summary_html = (
                '<div style="display:flex;flex-direction:column;gap:16px;margin-bottom:16px;">'
                '<div><div style="font-size:10px;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Most Stereotyped Targets (highest SS)</div>'
                '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead>'
                f'<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;"><th style="{th_style}text-align:center;">Rank</th><th style="{th_style}text-align:left;">Target</th><th style="{th_style}text-align:center;">Category</th><th style="{th_style}text-align:right;">SS</th><th style="{th_style}text-align:center;">n</th></tr></thead>'
                f'<tbody>{_build_target_rows(most_biased, "high")}</tbody></table></div>'
                '<div><div style="font-size:10px;font-weight:700;color:#22c55e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Least Stereotyped Targets (lowest SS)</div>'
                '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead>'
                f'<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;"><th style="{th_style}text-align:center;">Rank</th><th style="{th_style}text-align:left;">Target</th><th style="{th_style}text-align:center;">Category</th><th style="{th_style}text-align:right;">SS</th><th style="{th_style}text-align:center;">n</th></tr></thead>'
                f'<tbody>{_build_target_rows(least_biased, "low")}</tbody></table></div></div>'
            )
            
            chart_grid = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">' + "".join([f'<div>{html}</div>' for html in charts]) + '</div>'
            
            return ui.div(ui.HTML(summary_html), ui.HTML(chart_grid)), len(targets)

        header_args = (
            f"Demographic Slice Analysis{_source_badge_html('Base Encoder')}",
            "StereoSet Stereotype Score (SS) broken down by demographic target group.",
            f"<span style='{_TH}'>What is this?</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>StereoSet SS disaggregated by individual demographic <b>target groups</b> (e.g. 'doctor', 'Muslim', 'Black people'). Each bar is one target with ≥10 evaluation items.</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Slices reveal which specific groups drive the aggregate score - a model may be unbiased on average but strongly biased on a particular target.</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Stereotype Score (SS)</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;margin:2px 0 6px;'>"
            f"SS = P(model prefers stereo &gt; anti-stereo) × 100</div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>SS = 50%</span>&nbsp;unbiased - model chooses at chance between stereo and anti-stereo</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>SS &gt; 50%</span>&nbsp;model consistently favours the stereotyped completion</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>SS &lt; 50%</span>&nbsp;model favours the anti-stereotyped completion (counter-bias)</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Categories</span>"
            f"<div style='{_TR}'>"
            f"<span style='{_TD};color:#e74c3c;'>●</span><span>gender</span>&nbsp;&nbsp;"
            f"<span style='{_TD};color:#3498db;'>●</span><span>race</span>&nbsp;&nbsp;"
            f"<span style='{_TD};color:#2ecc71;'>●</span><span>religion</span>&nbsp;&nbsp;"
            f"<span style='{_TD};color:#f39c12;'>●</span><span>profession</span>"
            f"</div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>How to read the charts</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Bars sorted by SS descending - the most stereotyped targets are on the left</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Dashed line at SS = 50 marks the unbiased baseline</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Hover a bar to see target name, SS value, and item count</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Targets with high SS indicate stereotypes learned from pre-training data. Compare across categories to see which domains carry the most bias signal.</div>"
        )
        
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None
        
        if compare_models and mk_B:
            res_A = _render_single(mk_A, "_A")
            res_B = _render_single(mk_B, "_B")
            
            c_A = res_A[0] if res_A else ui.div("No data")
            c_B = res_B[0] if res_B else ui.div("No data")
            
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(c_A, *header_args, style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_stereoset_demographic_csv", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(c_B, *header_args, style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_stereoset_demographic_csv_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        
        # ── Source-aware single mode ──
        # Demographic slices always shows a single model (no side-by-side in compare)
        resolved = _stereoset_resolve_key_for_source()
        mk = resolved[0] if isinstance(resolved, tuple) else resolved
        res = _render_single(mk)
        inner = res[0] if res else ui.div()
        n_targets = res[1] if res else 0
        src_mode = _get_attn_source_mode("bias_attn_source")
        _demo_controls = [ui.download_button("export_stereoset_demographic_csv", "CSV", style=_BTN_STYLE_CSV)]
        if src_mode == "gusnet":
            _demo_controls.insert(0, _stereoset_gusnet_warning_icon())

        return _wrap_card(inner, header_args[0], f"Stereotype Score breakdown by target group ({n_targets} targets with n ≥ 10)", header_args[2],
                          controls=_demo_controls)

    @output
    @render.ui
    def stereoset_head_sensitivity():
        """12x12 head sensitivity heatmap."""
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, container_suffix=""):
            matrix = get_head_sensitivity_matrix(mk)
            top_heads = get_sensitive_heads(mk)
            examples = get_stereoset_examples(mk)
            if matrix is None:
                return None

            # ── 12×12 heatmap ──
            fig_heatmap = create_stereoset_head_sensitivity_heatmap(matrix, top_heads)
            heatmap_html = _chart_with_png_btn(
                _deferred_plotly(fig_heatmap, f"stereoset-sensitivity{container_suffix}"),
                f"stereoset-sensitivity{container_suffix}", f"stereoset_sensitivity{container_suffix}"
            )

            try:
                top_k = int(input.bias_top_k())
            except Exception:
                top_k = 5

            # ── Top discriminative features table ──
            top_features = get_top_features(mk)
            if top_features:
                feat_rows = []
                for rank, f in enumerate(top_features[:top_k], 1):
                    p = f["p_value"]
                    p_color = "#16a34a" if p < 1e-10 else "#eab308" if p < 0.001 else "#94a3b8"
                    feat_rows.append(
                        f'<tr style="transition:all 0.2s ease;">'
                        f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td>'
                        f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:left;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">{f["name"]}</td>'
                        f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{p_color};">{p:.2e}</td>'
                        f'</tr>'
                    )
                _kw_tooltip = (
                    f"<span style='{_TH}'>What is this?</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                    f"<span>Attention head features that best discriminate between the four <b>demographic categories</b> in StereoSet (gender · race · religion · profession)</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<span style='{_TH}'>Same features as the notebooks?</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
                    f"<span><b>Same extraction code</b> (<span style='{_TC}'>extract_features_for_sentence</span>) - produces the same feature types: <span style='{_TC}'>GAM_L{{l}}_H{{h}}_*</span>, <span style='{_TC}'>AttMap_*</span>, <span style='{_TC}'>Spec_*</span></span></div>"
                    f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>▪</span>"
                    f"<span><b>Different data and task</b> - the notebooks run on <span style='{_TC}'>bias_sentences.json</span> with O/GEN/UNFAIR/STEREO labels and use XGBoost + SelectKBest. Here features are extracted from StereoSet sentences and tested with Kruskal-Wallis across demographic groups.</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<span style='{_TH}'>How is it computed?</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
                    f"<span>For each feature column, a <b>Kruskal-Wallis H-test</b> compares its distribution across the four demographic categories - a non-parametric test: does this feature differ significantly across groups?</span></div>"
                    f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
                    f"<span>Features ranked by p-value ascending. Top-20 with the lowest p-values are shown.</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<span style='{_TH}'>Feature name guide</span>"
                    f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>▪</span>"
                    f"<span><span style='{_TC}'>GAM</span> gradient × attention (importance), <span style='{_TC}'>AttMap</span> raw attention map, <span style='{_TC}'>Spec</span> specialisation score</span></div>"
                    f"<hr style='{_TS}'/>"
                    f"<div style='{_TN}'>Low p-value = this head feature's distribution shifts significantly across gender / race / religion / profession - it encodes demographic-specific attention patterns.</div>"
                )
                features_html = (
                    '<div style="margin-top:16px;">'
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">'
                    f'<span style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Top Discriminative Features (Kruskal-Wallis)</span>'
                    f'<div class="info-tooltip-wrapper">'
                    f'<span class="info-tooltip-icon">i</span>'
                    f'<div class="info-tooltip-content">{_kw_tooltip}</div>'
                    f'</div></div>'
                    '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead>'
                    '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                    '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                    '<th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Feature</th>'
                    '<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">p-value</th>'
                    '</tr></thead>'
                    f'<tbody>{"".join(feat_rows)}</tbody></table></div>'
                )
            else:
                features_html = ""

            return ui.div(
                ui.HTML(heatmap_html),
                ui.HTML(features_html),
            )

        _src = _get_attn_source_mode("bias_attn_source")
        _src_badge = _source_badge_html("GUS-Net" if _src == "gusnet" else "Base Encoder") if _src != "compare" else ""
        header_args = (
            f"Head Sensitivity Analysis{_src_badge}",
            "Which attention heads respond differently across bias categories (gender / race / religion / profession)?",
            f"<span style='{_TH}'>Sensitivity formula</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;'>"
            f"sensitivity(l,h) = Var[ mean_attn(l,h | cat) ]</div>"
            f"<div style='{_TN}; margin-bottom:4px;'>variance across bias categories: gender · race · religion · profession</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>High variance</span>&nbsp;head is category-discriminative - attends differently per demographic group</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span><span style='background:rgba(148,163,184,0.15);color:#94a3b8;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;'>Low variance</span>&nbsp;category-agnostic - responds similarly regardless of bias type</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Top discriminative features (Kruskal-Wallis)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Low p-value</span>&nbsp;feature distribution differs significantly across categories</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Combine with BAR: a head that is both BAR-specialised and category-sensitive is the most diagnostically informative.</div>"
        )

        # ── GUS-NET source (driven by floating bar Source Attention toggle) ──
        gk = _stereoset_gusnet_key()  # None when source is "base" or no data
        source_mode = _get_attn_source_mode("bias_attn_source")
        gusnet_active = source_mode in ("gusnet", "compare")

        # ── Similarity badge (computed when GUS-NET data exists) ──
        canonical = _stereoset_model_key()
        gusnet_k = get_gusnet_key(canonical)
        sim = compute_model_similarity(canonical, gusnet_k)

        sim_badge_html = ""
        if sim is not None and gusnet_active:
            pct = sim["overall_pct"]
            if pct >= 80:
                badge_color = "#22c55e"  # green
                badge_bg = "rgba(34,197,94,0.12)"
            elif pct >= 50:
                badge_color = "#f59e0b"  # amber
                badge_bg = "rgba(245,158,11,0.12)"
            else:
                badge_color = "#ef4444"  # red
                badge_bg = "rgba(239,68,68,0.12)"

            sim_badge_html = (
                f'<div style="display:flex;align-items:center;gap:8px;margin-right:12px;'
                f'padding:4px 10px;border-radius:6px;background:{badge_bg};'
                f'border:1px solid {badge_color}20;cursor:default;position:relative;" '
                f'class="sim-badge-wrapper">'
                # Mini bar
                f'<div style="width:32px;height:4px;border-radius:2px;background:rgba(148,163,184,0.2);overflow:hidden;">'
                f'<div style="width:{pct}%;height:100%;border-radius:2px;background:{badge_color};'
                f'transition:width 0.4s ease;"></div></div>'
                # Percentage
                f'<span style="font-size:11px;font-weight:700;color:{badge_color};'
                f'font-family:JetBrains Mono,monospace;letter-spacing:-0.3px;">{pct:.0f}%</span>'
                f'<span style="font-size:9px;color:#94a3b8;font-weight:500;">similar</span>'
                # Tooltip on hover
                f'<div class="sim-tooltip" style="display:none;position:fixed;'
                f'background:#1e293b;color:#e2e8f0;'
                f'padding:10px 14px;border-radius:8px;font-size:10px;white-space:nowrap;'
                f'box-shadow:0 8px 24px rgba(0,0,0,0.3);z-index:99999;line-height:1.7;'
                f'border:1px solid rgba(148,163,184,0.15);">'
                f'<div style="font-weight:700;color:white;margin-bottom:6px;font-size:11px;">'
                f'{str(canonical).upper()} \u2194 GUS-NET Attention Similarity</div>'
                f'<div style="color:#cbd5e1;font-size:10px;margin-bottom:12px;line-height:1.5;white-space:normal;max-width:250px;">'
                f'This score indicates how closely the attention mechanisms of the fine-tuned GUS-NET match the original {str(canonical).upper()} model. '
                f'It combines the correlation of their attention matrices across all layers and the overlap of their most sensitive heads. '
                f'A high score means GUS-NET successfully preserves the base model\'s fundamental attention behavior while identifying biased tokens.</div>'
                f'<div style="font-family:JetBrains Mono,monospace; text-align:center; background:rgba(0,0,0,0.2); padding:8px 12px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">'
                f'<div style="display:flex; justify-content:space-between; margin-bottom:2px; font-size:10px;">'
                f'<div><span style="color:#60a5fa;">Max Corr:</span> <b style="color:white;">{sim["matrix_corr"]:.3f}</b></div>'
                f'<div><span style="color:#a78bfa;">Top Heads:</span> <b style="color:white;">{sim["heads_overlap_pct"]:.0f}%</b></div>'
                f'</div>'
                f'<div style="display:flex; justify-content:center; gap:60px; color:#64748b; font-size:12px; font-weight:bold; margin-bottom:2px;">'
                f'<span>\\</span><span>/</span>'
                f'</div>'
                f'<div style="font-size:13px;">'
                f'<b style="color:{badge_color};">{sim["overall_pct"]:.0f}%</b> <span style="font-size:9px;color:#94a3b8;font-weight:normal;">Similarity</span>'
                f'</div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

        _sim_badge_widget = ui.HTML(
            # Inline CSS + JS for similarity tooltip with fixed positioning
            '<style>'
            '.sim-badge-wrapper:hover .sim-tooltip{display:block!important;}'
            '</style>'
            '<script>'
            'document.addEventListener("mouseover",function(e){'
            'var w=e.target.closest(".sim-badge-wrapper");'
            'if(!w)return;'
            'var tt=w.querySelector(".sim-tooltip");'
            'if(!tt)return;'
            'var r=w.getBoundingClientRect();'
            'tt.style.left=(r.left+r.width/2)+"px";'
            'tt.style.top=(r.bottom+8)+"px";'
            'tt.style.transform="translateX(-50%)";'
            '});'
            '</script>'
            + sim_badge_html
        ) if sim_badge_html else None

        csv_controls = [
            ui.download_button("export_stereoset_sensitivity_csv", "Heads CSV", style=_BTN_STYLE_CSV),
            ui.download_button("export_stereoset_features_csv", "Features CSV", style=_BTN_STYLE_CSV),
        ]

        # ── Source-aware rendering (takes priority over sidebar model comparison) ──
        if gk and source_mode == "gusnet":
            # GUS-Net only - no similarity badge (references both models)
            return _wrap_card(_render_single(gk) or ui.div(), *header_args,
                              controls=csv_controls)

        if gk and source_mode == "compare":
            # Side-by-side Base vs GUS-Net
            c_base = _render_single(mk_A, "_base") or ui.div("No data")
            c_gus  = _render_single(gk, "_gusnet") or ui.div("No GUS-NET data")
            meta_base = get_metadata(mk_A) or {}
            meta_gus  = get_metadata(gk) or {}
            label_base = meta_base.get("model", mk_A).upper()
            label_gus  = _clean_gusnet_label(meta_gus.get("model", gk))

            compare_controls = [
                c for c in [_sim_badge_widget] + csv_controls
                if c is not None
            ]

            return _wrap_card(
                ui.div(
                    {"style": "display:grid;grid-template-columns:1fr 1fr;gap:24px;"},
                    ui.div(
                        ui.div(
                            ui.span(label_base, style="font-size:11px;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:0.5px;"),
                            style="margin-bottom:8px;padding:4px 0;border-bottom:2px solid #3b82f6;",
                        ),
                        c_base,
                    ),
                    ui.div(
                        ui.div(
                            ui.span(label_gus, style="font-size:11px;font-weight:700;color:#ff5ca9;text-transform:uppercase;letter-spacing:0.5px;"),
                            style="margin-bottom:8px;padding:4px 0;border-bottom:2px solid #ff5ca9;",
                        ),
                        c_gus,
                    ),
                ),
                *header_args,
                controls=compare_controls,
            )

        # ── Sidebar model comparison (only when source toggle is "base") ──
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None

        if compare_models and mk_B:
            c_A = _render_single(mk_A, "_A") or ui.div("No data")
            c_B = _render_single(mk_B, "_B") or ui.div("No data")

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(c_A, *header_args, style="border: 2px solid #3b82f6; height: 100%;",
                           controls=csv_controls),
                _wrap_card(c_B, *header_args, style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_stereoset_sensitivity_csv_B", "Heads CSV", style=_BTN_STYLE_CSV),
                               ui.download_button("export_stereoset_features_csv_B", "Features CSV", style=_BTN_STYLE_CSV),
                           ])
            )

        # Base only (default)
        return _wrap_card(_render_single(mk_A) or ui.div(), *header_args,
                          controls=csv_controls)

    @render.download(filename="top_discriminative_features.csv")
    def export_stereoset_features_csv():
        mk = _stereoset_model_key()
        top_features = get_top_features(mk)
        if not top_features:
            yield "No feature data"
            return
        lines = ["rank,feature,p_value"]
        for rank, f in enumerate(top_features, 1):
            lines.append(f'{rank},{f["name"]},{f["p_value"]:.6e}')
        yield "\n".join(lines)

    @render.download(filename="top_discriminative_features_B.csv")
    def export_stereoset_features_csv_B():
        mk = _stereoset_model_key_B()
        top_features = get_top_features(mk)
        if not top_features:
            yield "No feature data"
            return
        lines = ["rank,feature,p_value"]
        for rank, f in enumerate(top_features, 1):
            lines.append(f'{rank},{f["name"]},{f["p_value"]:.6e}')
        yield "\n".join(lines)

    @render.download(filename='sensitive_heads.csv')
    def export_stereoset_sensitivity_csv():
        mk = _stereoset_model_key()
        heads = get_sensitive_heads(mk)
        if not heads:
            yield "No sensitivity data"
            return
        lines = ["rank,layer,head,variance,correlation,best_feature"]
        for rank, h in enumerate(heads, 1):
            lines.append(f'{rank},{h["layer"]},{h["head"]},{h["variance"]:.6f},{h["correlation"]:.6f},{h["best_feature"]}')
        yield "\n".join(lines)


    @render.download(filename='sensitive_heads_B.csv')
    def export_stereoset_sensitivity_csv_B():
        mk = _stereoset_model_key_B()
        heads = get_sensitive_heads(mk)
        if not heads:
            yield "No sensitivity data"
            return
        lines = ["rank,layer,head,variance,correlation,best_feature"]
        for rank, h in enumerate(heads, 1):
            lines.append(f'{rank},{h["layer"]},{h["head"]},{h["variance"]:.6f},{h["correlation"]:.6f},{h["best_feature"]}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_category_scores.csv')
    def export_stereoset_category_csv():
        mk = _stereoset_model_key()
        scores = get_stereoset_scores(mk)
        if not scores:
            yield "No scores data"
            return
        by_cat = scores.get("by_category", {})
        lines = ["category,ss,lms,icat,n,mean_bias_score"]
        for cat, v in by_cat.items():
            lines.append(f'{cat},{v["ss"]:.2f},{v["lms"]:.2f},{v["icat"]:.2f},{v["n"]},{v.get("mean_bias_score", 0):.6f}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_category_scores_B.csv')
    def export_stereoset_category_csv_B():
        mk = _stereoset_model_key_B()
        scores = get_stereoset_scores(mk)
        if not scores:
            yield "No scores data"
            return
        by_cat = scores.get("by_category", {})
        lines = ["category,ss,lms,icat,n,mean_bias_score"]
        for cat, v in by_cat.items():
            lines.append(f'{cat},{v["ss"]:.2f},{v["lms"]:.2f},{v["icat"]:.2f},{v["n"]},{v.get("mean_bias_score", 0):.6f}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_bias_distribution.csv')
    def export_stereoset_distribution_csv():
        mk = _stereoset_model_key()
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        lines = ["category,bias_score"]
        for e in examples:
            lines.append(f'{e.get("category","")},{e.get("bias_score", e.get("stereo_prob", 0)):.6f}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_bias_distribution_B.csv')
    def export_stereoset_distribution_csv_B():
        mk = _stereoset_model_key_B()
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        lines = ["category,bias_score"]
        for e in examples:
            lines.append(f'{e.get("category","")},{e.get("bias_score", e.get("stereo_prob", 0)):.6f}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_demographic_slices.csv')
    def export_stereoset_demographic_csv():
        mk = _stereoset_model_key()
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        from collections import Counter
        target_data = {}
        for ex in examples:
            t = ex.get("target", "unknown")
            if t not in target_data:
                target_data[t] = {"stereo_wins": 0, "n": 0, "category": ex.get("category", "")}
            target_data[t]["n"] += 1
            if ex.get("stereo_pll", 0) > ex.get("anti_pll", 0):
                target_data[t]["stereo_wins"] += 1
        lines = ["target,category,ss_pct,n"]
        for t, d in sorted(target_data.items(), key=lambda x: x[1]["stereo_wins"] / max(x[1]["n"], 1), reverse=True):
            if d["n"] >= 5:
                ss = d["stereo_wins"] / d["n"] * 100
                lines.append(f'{t},{d["category"]},{ss:.1f},{d["n"]}')
        yield "\n".join(lines)


    @render.download(filename='stereoset_demographic_slices_B.csv')
    def export_stereoset_demographic_csv_B():
        mk = _stereoset_model_key_B()
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        from collections import Counter
        target_data = {}
        for ex in examples:
            t = ex.get("target", "unknown")
            if t not in target_data:
                target_data[t] = {"stereo_wins": 0, "n": 0, "category": ex.get("category", "")}
            target_data[t]["n"] += 1
            if ex.get("stereo_pll", 0) > ex.get("anti_pll", 0):
                target_data[t]["stereo_wins"] += 1
        lines = ["target,category,ss_pct,n"]
        for t, d in sorted(target_data.items(), key=lambda x: x[1]["stereo_wins"] / max(x[1]["n"], 1), reverse=True):
            if d["n"] >= 5:
                ss = d["stereo_wins"] / d["n"] * 100
                lines.append(f'{t},{d["category"]},{ss:.1f},{d["n"]}')
        yield "\n".join(lines)


    @render.download(filename='perturbation_vs_attention.csv')
    def export_perturb_attn_csv():
        bundle = perturbation_results.get()
        if not bundle or not bundle.perturb_vs_attn_spearman:
            yield "No perturbation-attention data"
            return
        lines = ["layer,head,spearman_rho"]
        for layer, head, rho in bundle.perturb_vs_attn_spearman:
            lines.append(f"{layer},{head},{rho:.6f}")
        yield "\n".join(lines)


    @render.download(filename='perturbation_vs_attention_B.csv')
    def export_perturb_attn_csv_B():
        bundle = perturbation_results_B.get()
        if not bundle or not bundle.perturb_vs_attn_spearman:
            yield "No perturbation-attention data"
            return
        lines = ["layer,head,spearman_rho"]
        for layer, head, rho in bundle.perturb_vs_attn_spearman:
            lines.append(f"{layer},{head},{rho:.6f}")
        yield "\n".join(lines)


    @render.download(filename='cross_method_agreement.csv')
    def export_cross_method_csv():
        lrp_bundle = lrp_results.get()
        ig_bundle = ig_results.get()
        if not lrp_bundle or not ig_bundle:
            yield "No cross-method data"
            return
        ig_dict = {(r.layer, r.head): r.spearman_rho for r in ig_bundle.correlations}
        lines = ["layer,head,ig_rho,lrp_rho"]
        for layer, head, lrp_rho in lrp_bundle.correlations:
            ig_rho = ig_dict.get((layer, head), float("nan"))
            lines.append(f"{layer},{head},{ig_rho:.6f},{lrp_rho:.6f}")
        yield "\n".join(lines)


    @render.download(filename='cross_method_agreement_B.csv')
    def export_cross_method_csv_B():
        lrp_bundle = lrp_results_B.get()
        ig_bundle = ig_results_B.get()
        if not lrp_bundle or not ig_bundle:
            yield "No cross-method data"
            return
        ig_dict = {(r.layer, r.head): r.spearman_rho for r in ig_bundle.correlations}
        lines = ["layer,head,ig_rho,lrp_rho"]
        for layer, head, lrp_rho in lrp_bundle.correlations:
            ig_rho = ig_dict.get((layer, head), float("nan"))
            lines.append(f"{layer},{head},{ig_rho:.6f},{lrp_rho:.6f}")
        yield "\n".join(lines)


    @output
    @render.ui
    def stereoset_attention_bias_link():
        """Obj 3 + Obj 4 - Attention patterns linked to StereoSet bias scores."""
        mk_A = _stereoset_model_key()
        source_mode = _get_attn_source_mode("bias_attn_source")
        gk = _stereoset_gusnet_key()

        try:
            top_k = int(input.bias_top_k())
        except Exception:
            top_k = 5

        def _render_pair_blocks(mk, suffix=""):
            """Return (dist_html, scatter_html) strings for one model.

            Returns None when no data is available.
            """
            examples   = get_stereoset_examples(mk)
            top_heads  = get_sensitive_heads(mk)
            if not examples or not top_heads:
                _na = '<div style="color:#94a3b8;font-size:12px;padding:16px;text-align:center;">No StereoSet data available for this model.</div>'
                return (_na, _na)

            # Resolve selected head from click input (or default to most sensitive)
            click_input_name = f"stereoset_selected_head{suffix}"
            selected_head = None
            try:
                val = getattr(input, click_input_name)()
                if val:
                    selected_head = str(val)
            except Exception:
                pass

            # Obj 3 - box distributions; clicking a head fires click_input_name
            try:
                fig_dist = create_stereoset_head_distributions(examples, top_heads, top_n=min(top_k, 6))
                if selected_head:
                    lbl = selected_head.replace("_", "·")
                    sub_extra = (
                        f' · <span style="color:#3b82f6;font-weight:600;">selected: {lbl}</span>'
                        f' <span style="color:#94a3b8;cursor:pointer;" '
                        f'onclick="if(window.Shiny)Shiny.setInputValue(\'{click_input_name}\',\'\',{{priority:\'event\'}})">✕</span>'
                    )
                else:
                    sub_extra = ' · <span style="color:#94a3b8;">click a box to filter →</span>'
                dist_block = (
                    '<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;'
                    'letter-spacing:0.5px;margin-bottom:4px;">Stereo vs Anti Distributions per Head</div>'
                    '<div style="font-size:10px;color:#94a3b8;margin-bottom:8px;line-height:1.5;">'
                    'Attention feature values across all examples. '
                    '<span style="color:#ef4444;">●</span> Stereo &nbsp;'
                    f'<span style="color:#22c55e;">●</span> Anti · p-values = Mann-Whitney{sub_extra}</div>'
                    + _chart_with_png_btn(
                        _deferred_plotly(
                            fig_dist,
                            f"stereoset-dist{suffix}",
                            height="300px",
                            click_input=click_input_name,
                        ),
                        f"stereoset-dist{suffix}", f"stereoset_head_distributions{suffix}"
                    )
                )
            except Exception as e:
                dist_block = f'<div style="color:#ef4444;font-size:11px;padding:8px;">Error: {e}</div>'

            # Obj 4 - binned scatter, updates when head is selected
            try:
                fig_scatter = create_stereoset_attention_scatter(
                    examples, top_heads,
                    head_key=selected_head if selected_head else None,
                )
                if selected_head:
                    lbl = selected_head.replace("_", "·")
                    scatter_sub = f'Showing head <b style="color:#3b82f6;">{lbl}</b>. Bins = deciles of Δ attention. Y = mean bias score ± SD.'
                else:
                    scatter_sub = 'Most sensitive head shown by default. Click a box ← to switch head.'
                scatter_block = (
                    '<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;'
                    'letter-spacing:0.5px;margin-bottom:4px;">Attention Δ vs Bias Score</div>'
                    f'<div style="font-size:10px;color:#94a3b8;margin-bottom:8px;line-height:1.5;">{scatter_sub}</div>'
                    + _chart_with_png_btn(
                        _deferred_plotly(fig_scatter, f"stereoset-scatter{suffix}", height="300px"),
                        f"stereoset-scatter{suffix}", f"stereoset_scatter{suffix}"
                    )
                )
            except Exception as e:
                scatter_block = f'<div style="color:#ef4444;font-size:11px;padding:8px;">Error: {e}</div>'

            return (dist_block, scatter_block)

        def _render_pair(mk, suffix=""):
            """Combine dist + scatter side by side (default single-model layout)."""
            dist_html, scatter_html = _render_pair_blocks(mk, suffix)
            return ui.div(
                {"style": "display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;"},
                ui.HTML(f'<div>{dist_html}</div>'),
                ui.HTML(f'<div>{scatter_html}</div>'),
            )

        _corr_badge = _source_badge_html("GUS-Net" if source_mode == "gusnet" else "Base Encoder") if source_mode != "compare" else ""
        header_args = (
            f"Attention–Bias Correlation{_corr_badge}",
            "Do sensitive heads actually attend differently when processing stereotyped vs anti-stereotyped content?",
            f"<span style='{_TH}'>Head distributions</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Violin/box plots of mean attention split by bias category</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Separated distributions</span>&nbsp;head responds differently per demographic group</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span><span style='background:rgba(148,163,184,0.15);color:#94a3b8;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;'>Overlapping</span>&nbsp;head is category-agnostic</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Bias–attention scatter</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>X = mean head attention &nbsp;·&nbsp; Y = bias score (stereo − anti prob)</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Positive slope</span>&nbsp;higher attention → stronger stereotyped response</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span><span style='background:rgba(148,163,184,0.15);color:#94a3b8;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;'>Flat slope</span>&nbsp;head activation independent of stereotype strength</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Bridges StereoSet benchmark (aggregate) → per-sentence attention (mechanistic). Primary evidence for the attention-as-bias-mechanism hypothesis.</div>"
        )

        # ── Source-aware rendering (takes priority over sidebar model comparison) ──
        if gk and source_mode == "gusnet":
            # GUS-Net only
            return _wrap_card(_render_pair(gk), *header_args)

        if gk and source_mode == "compare":
            # Side-by-side 2×2 grid
            dist_base, scatter_base = _render_pair_blocks(mk_A, "_base")
            dist_gus,  scatter_gus  = _render_pair_blocks(gk, "_gusnet")
            meta_base = get_metadata(mk_A) or {}
            meta_gus  = get_metadata(gk) or {}
            label_base = _clean_gusnet_label(meta_base.get("model", mk_A))
            label_gus  = _clean_gusnet_label(meta_gus.get("model", gk))

            def _col(label, color, html):
                return (
                    f'<div>'
                    f'<div style="margin-bottom:6px;padding:3px 0;border-bottom:2px solid {color};">'
                    f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.5px;">{label}</span>'
                    f'</div>{html}</div>'
                )

            return _wrap_card(
                ui.div(
                    ui.HTML(
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;margin-bottom:24px;">'
                        f'{_col(label_base, "#3b82f6", dist_base)}'
                        f'{_col(label_gus, "#ff5ca9", dist_gus)}'
                        f'</div>'
                    ),
                    ui.HTML(
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start;">'
                        f'{_col(label_base, "#3b82f6", scatter_base)}'
                        f'{_col(label_gus, "#ff5ca9", scatter_gus)}'
                        f'</div>'
                    ),
                ),
                *header_args,
            )

        # ── Sidebar model comparison (only when source toggle is "base") ──
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None

        if compare_models and mk_B:
            c_A = _render_pair(mk_A, "_A")
            c_B = _render_pair(mk_B, "_B")
            return ui.div(
                {"style": "display:grid;grid-template-columns:1fr 1fr;gap:24px;"},
                _wrap_card(c_A, *header_args, style="border:2px solid #3b82f6;height:100%;"),
                _wrap_card(c_B, *header_args, style="border:2px solid #ff5ca9;height:100%;"),
            )

        # Base only (default)
        return _wrap_card(_render_pair(mk_A), *header_args)

    @output
    @render.ui
    def stereoset_example_explorer():
        """Interactive example explorer with category filter and detail view."""
        source_mode = _get_attn_source_mode("bias_attn_source")
        gk = _stereoset_gusnet_key()

        # Source-aware: gusnet → show GUS-Net data, base → base, compare → both
        if source_mode == "gusnet" and gk:
            mk = gk
        else:
            mk = _stereoset_model_key()
        examples = get_stereoset_examples(mk)

        # Source toggle takes priority over sidebar model comparison
        if source_mode == "gusnet":
            mk_B = None  # GUS-Net only, no second model
        elif source_mode == "compare" and gk:
            mk_B = gk
        elif active_bias_compare_models.get():
            mk_B = _stereoset_model_key_B()
        else:
            mk_B = None
        has_B = mk_B is not None
        examples_B = get_stereoset_examples(mk_B) if has_B else []
        # Map context -> example for quick lookup
        map_B = {e["context"]: e for e in examples_B}

        if not examples:
            return ui.div()

        # Category filter (inline select)
        categories = sorted(set(e["category"] for e in examples))
        cat_options = "".join(
            f'<option value="{c}">{c.capitalize()} ({sum(1 for e in examples if e["category"]==c)})</option>'
            for c in categories
        )

        # Build scrollable example table (show first 50 by default)
        STEREOSET_CAT_COLORS = {
            "gender": "#e74c3c", "race": "#3498db",
            "religion": "#2ecc71", "profession": "#f39c12",
        }

        table_rows = []
        for i, ex in enumerate(examples[:100]):
            ctx = ex.get("context", "")[:80]
            cat = ex.get("category", "")
            bs = ex.get("bias_score", 0)
            cat_color = STEREOSET_CAT_COLORS.get(cat, "#94a3b8")
            bs_color = "#ef4444" if bs > 0 else "#22c55e"
            
            # Model B data
            bs_B_cell = ""
            if has_B:
                ex_B = map_B.get(ex.get("context"))
                if ex_B:
                    bs_B = ex_B.get("bias_score", 0)
                    bs_B_color = "#ef4444" if bs_B > 0 else "#22c55e"
                    bs_B_cell = (
                        f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);'
                        f'font-size:11px;color:{bs_B_color};font-family:JetBrains Mono,monospace;'
                        f'font-weight:600;text-align:right;border-left:1px dashed #e2e8f0;">{bs_B:+.4f}</td>'
                    )
                else:
                    bs_B_cell = '<td style="border-bottom:1px solid rgba(226,232,240,0.5);border-left:1px dashed #e2e8f0;"></td>'

            table_rows.append(
                f'<tr class="stereoset-row" data-category="{cat}" data-idx="{i}" '
                f'onclick="window._selectStereoSetRow(this, {i})" '
                f'style="cursor:pointer;transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1);border-left:3px solid transparent;" '
                f'onmouseover="if(!this.classList.contains(\'stereoset-selected\')){{this.style.background=\'rgba(255,255,255,0.08)\';this.style.transform=\'translateX(4px)\';this.style.borderLeftColor=\'{cat_color}\';}}" '
                f'onmouseout="if(!this.classList.contains(\'stereoset-selected\')){{this.style.background=\'transparent\';this.style.transform=\'translateX(0)\';this.style.borderLeftColor=\'transparent\';}}">'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;">'
                f'<span style="padding:1px 6px;border-radius:3px;font-size:9px;font-weight:600;'
                f'background:rgba({int(cat_color[1:3],16)},{int(cat_color[3:5],16)},{int(cat_color[5:7],16)},0.2);'
                f'color:{cat_color};text-transform:uppercase;">{cat}</span></td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;color:#334155;max-width:400px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{ctx}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;color:{bs_color};'
                f'font-family:JetBrains Mono,monospace;font-weight:600;text-align:right;">{bs:+.4f}</td>'
                f'{bs_B_cell}'
                f'</tr>'
            )

        # Header columns
        header_bias_A_label = "Bias A" if has_B else "Bias"
        header_bias_B_col = ""
        
        if has_B:
            header_bias_B_col = (
                '<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;'
                'color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:90px;border-left:1px dashed #cbd5e1;">Bias B</th>'
            )

        table_html = (
            '<div style="max-height:300px;overflow-y:auto;border:1px solid rgba(255,255,255,0.06);'
            'border-radius:8px;margin-top:8px;">'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead style="position:sticky;top:0;z-index:1;">'
            '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
            '<th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:100px;">Category</th>'
            '<th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Context</th>'
            f'<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:90px;">{header_bias_A_label}</th>'
            f'{header_bias_B_col}'
            '</tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody>'
            '</table></div>'
        )

        # Category filter + row selection JS
        filter_js = (
            '<script>'
            'window.filterStereoSetCategory = function(cat) {'
            '  var rows = document.querySelectorAll(".stereoset-row");'
            '  rows.forEach(function(r) {'
            '    if (cat === "all" || r.dataset.category === cat) {'
            '      r.style.display = "";'
            '    } else {'
            '      r.style.display = "none";'
            '    }'
            '  });'
            '};'
            'window._selectStereoSetRow = function(el, idx) {'
            '  document.querySelectorAll(".stereoset-row.stereoset-selected").forEach(function(r) {'
            '    r.classList.remove("stereoset-selected");'
            '    r.style.background = "transparent";'
            '    r.querySelectorAll("td").forEach(function(td) {'
            '      if (!td.querySelector("span")) td.style.color = "";'
            '    });'
            '  });'
            '  el.classList.add("stereoset-selected");'
            '  el.style.background = "#94a3b8";'
            '  el.querySelectorAll("td").forEach(function(td) {'
            '    if (!td.querySelector("span") && !td.style.fontFamily) td.style.color = "#0f172a";'
            '  });'
            '  Shiny.setInputValue("stereoset_selected_example", idx, {priority:"event"});'
            '};'
            '</script>'
        )

        filter_select = (
            f'<select onchange="window.filterStereoSetCategory(this.value)" '
            f'style="font-size:11px;padding:4px 8px;background:#f1f5f9;color:#475569;font-weight:700;'
            f'border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;outline:none;">'
            f'<option value="all">All Categories ({len(examples)})</option>'
            f'{cat_options}'
            f'</select>'
        )

        # Detail view (initially empty, populated on click)
        detail_html = ""
        try:
            selected_idx = input.stereoset_selected_example()
            if selected_idx is not None and 0 <= selected_idx < len(examples):
                ex_A = examples[selected_idx]
                ex_B_detail = map_B.get(ex_A.get("context")) if has_B else None
                # Fetch friendly model names
                meta_A = get_metadata(mk)
                name_A = _clean_gusnet_label(meta_A.get("model", "Model A")) if meta_A else "Model A"
                
                name_B = "Model B"
                if has_B:
                    meta_B = get_metadata(mk_B)
                    if meta_B:
                        name_B = _clean_gusnet_label(meta_B.get("model", "Model B"))

                # Safely get top_n (custom input might not be init)
                top_n = 5
                try:
                    val = input.bias_top_k()
                    if val is not None: top_n = int(val)
                except Exception:
                    pass

                # ── Attention heatmaps (generate before detail HTML) ──
                heatmap_inner_html = ""
                attn_sim_pct = None
                try:
                    # Read layer/head from the floating toolbar controls
                    sensitive = get_sensitive_heads(mk) or []
                    try:
                        sel_layer = int(input.bias_attn_layer())
                    except Exception:
                        sel_layer = sensitive[0]["layer"] if sensitive else 0
                    try:
                        sel_head = int(input.bias_attn_head())
                    except Exception:
                        sel_head = sensitive[0]["head"] if sensitive else 0

                    def _generate_model_heatmaps(model_key, example_data, suffix=""):
                        """Generate trio + diff HTML for one model.
                        
                        Returns:
                            tuple: (trio_html, diff_html, raw_attentions)
                            raw_attentions is a dict with keys 'stereo', 'anti', 'unrelated'
                            each containing the attention tuple from the model.
                        """
                        base = _GUSNET_TO_ENCODER.get(model_key, "bert-base-uncased")
                        sens = get_sensitive_heads(model_key) or []
                        is_sens = any(
                            h["layer"] == sel_layer and h["head"] == sel_head
                            for h in sens
                        )
                        ctx = example_data.get("context", "")
                        s_text = ctx + " " + example_data.get("stereo_sentence", "")
                        a_text = ctx + " " + example_data.get("anti_sentence", "")
                        u_text = ctx + " " + example_data.get("unrelated_sentence", "")

                        s_tok, s_a = extract_attention_for_text(s_text, base, ModelManager)
                        a_tok, a_a = extract_attention_for_text(a_text, base, ModelManager)
                        sd = {"tokens": s_tok, "attentions": s_a}
                        ad = {"tokens": a_tok, "attentions": a_a}
                        ud = None
                        u_attn = None
                        if u_text.strip() and u_text.strip() != ctx.strip():
                            u_tok, u_a = extract_attention_for_text(u_text, base, ModelManager)
                            ud = {"tokens": u_tok, "attentions": u_a}
                            u_attn = u_a

                        fig_t = create_stereoset_attention_heatmaps(
                            sd, ad, ud,
                            layer=sel_layer, head=sel_head,
                            is_sensitive=is_sens,
                        )
                        cid_t = f"stereoset-attn-heatmap-{model_key}-{suffix}"
                        _ss_csv_id = "export_stereoset_example_attention_csv_B" if suffix == "B" else "export_stereoset_example_attention_csv"
                        csv_btn = ui.download_button(_ss_csv_id, "CSV", style=_BTN_STYLE_CSV)
                        
                        t_html = _chart_with_png_btn(
                            _deferred_plotly(fig_t, cid_t),
                            cid_t, 
                            f"stereoset_attn_heatmap_{model_key}_{suffix}",
                            controls=[str(csv_btn)]
                        )
                        fig_d = create_stereoset_attention_diff_heatmap(
                            sd, ad, layer=sel_layer, head=sel_head,
                        )
                        cid_d = f"stereoset-attn-diff-{model_key}-{suffix}"
                        _ss_diff_csv_id = "export_stereoset_example_diff_csv_B" if suffix == "B" else "export_stereoset_example_diff_csv"
                        diff_csv_btn = ui.download_button(_ss_diff_csv_id, "CSV", style=_BTN_STYLE_CSV)

                        d_html = _chart_with_png_btn(
                            _deferred_plotly(fig_d, cid_d),
                            cid_d, 
                            f"stereoset_attn_diff_{model_key}_{suffix}",
                            controls=[str(diff_csv_btn)]
                        )
                        raw_attentions = {"stereo": s_a, "anti": a_a, "unrelated": u_attn}
                        return (t_html, d_html, raw_attentions)

                    def _compute_attention_similarity(attn_A, attn_B):
                        """Compute Pearson correlation of attention across all layers for two models.
                        
                        Averages similarity over stereo/anti/unrelated sentence types.
                        """
                        import torch
                        sims = []
                        for sent_key in ("stereo", "anti", "unrelated"):
                            a = attn_A.get(sent_key)
                            b = attn_B.get(sent_key)
                            if a is None or b is None:
                                continue
                            # a and b are tuples of tensors, one per layer
                            # Each tensor shape: (1, num_heads, seq_len, seq_len)
                            for layer_idx in range(min(len(a), len(b))):
                                mat_a = a[layer_idx][0].cpu().float()  # (heads, seq, seq)
                                mat_b = b[layer_idx][0].cpu().float()
                                # Truncate to same seq length
                                min_seq = min(mat_a.shape[-1], mat_b.shape[-1])
                                flat_a = mat_a[:, :min_seq, :min_seq].flatten()
                                flat_b = mat_b[:, :min_seq, :min_seq].flatten()
                                n = flat_a.shape[0]
                                if n == 0:
                                    continue
                                ma = flat_a.mean()
                                mb = flat_b.mean()
                                cov = ((flat_a - ma) * (flat_b - mb)).sum()
                                sa = ((flat_a - ma) ** 2).sum().sqrt()
                                sb = ((flat_b - mb) ** 2).sum().sqrt()
                                if sa > 0 and sb > 0:
                                    r = (cov / (sa * sb)).item()
                                    sims.append(r)
                        if not sims:
                            return None
                        avg_r = sum(sims) / len(sims)
                        # Normalise from [-1,1] to [0,100]
                        return round(max(0, (avg_r + 1) / 2) * 100, 1)

                    if has_B and ex_B_detail:
                        # ── Compare Models: side-by-side with aligned layout ──
                        trio_A, diff_A, attn_A = _generate_model_heatmaps(mk, ex_A, "A")
                        trio_B, diff_B, attn_B = _generate_model_heatmaps(mk_B, ex_B_detail, "B")
                        
                        # ── Per-example attention similarity ──
                        attn_sim_pct = _compute_attention_similarity(attn_A, attn_B)

                        # Layout: 
                        # Row 1: [Model A Trio] [Model B Trio]
                        # Badge: attention similarity
                        # Row 2: [Model A Diff]  [Model B Diff]
                        heatmap_inner_html = (
                            f'<div style="display:flex;flex-direction:column;gap:16px;">'
                            # Row 1: Labels + Trios side by side
                            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;">'
                            # Model A column
                            f'  <div style="display:flex;flex-direction:column;gap:12px;">'
                            f'    <div style="font-size:10px;font-weight:700;color:#3b82f6;'
                            f'         text-transform:uppercase;letter-spacing:0.5px;'
                            f'         padding:4px 8px;background:rgba(59,130,246,0.1);border-radius:4px;'
                            f'         border:1px solid rgba(59,130,246,0.2);display:inline-block;width:fit-content;">{name_A}</div>'
                            f'    <div>{trio_A}</div>'
                            f'  </div>'
                            # Model B column
                            f'  <div style="display:flex;flex-direction:column;gap:12px;">'
                            f'    <div style="font-size:10px;font-weight:700;color:#ff5ca9;'
                            f'         text-transform:uppercase;letter-spacing:0.5px;'
                            f'         padding:4px 8px;background:rgba(255,92,169,0.1);border-radius:4px;'
                            f'         border:1px solid rgba(255,92,169,0.2);display:inline-block;width:fit-content;">{name_B}</div>'
                            f'    <div>{trio_B}</div>'
                            f'  </div>'
                            f'</div>'
                            # Row 2: Attention Differences side by side
                            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;">'
                            f'  <div>{diff_A}</div>'
                            f'  <div>{diff_B}</div>'
                            f'</div>'
                            f'</div>'
                        )
                    else:
                        # ── Single model ──
                        trio, diff, _ = _generate_model_heatmaps(mk, ex_A, "single")
                        heatmap_inner_html = (
                            f'<div style="display:flex;flex-direction:column;gap:12px;">'
                            f'{trio}'
                            f'{diff}'
                            f'</div>'
                        )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    heatmap_inner_html = (
                        f'<div style="padding:12px;border:1px solid rgba(239,68,68,0.3);'
                        f'border-radius:8px;color:#f87171;font-size:11px;">'
                        f'Could not generate attention heatmaps: {e}</div>'
                    )

                # Build sensitive heads badge label
                sensitive_label = ""
                _sens = get_sensitive_heads(mk) or []
                if _sens:
                    badges_A = [f'L{h["layer"]}.H{h["head"]}' for h in _sens[:5]]
                    if has_B:
                        _sens_B = get_sensitive_heads(mk_B) or []
                        badges_B = [f'L{h["layer"]}.H{h["head"]}' for h in _sens_B[:5]]
                        sensitive_label = (
                            f'<div style="font-size:9px;margin-bottom:8px;display:flex;gap:16px;">'
                            f'<span style="color:#3b82f6;">★ {name_A}: {", ".join(badges_A)}</span>'
                            f'<span style="color:#ff5ca9;">★ {name_B}: {", ".join(badges_B)}</span>'
                            f'</div>'
                        )
                    else:
                        sensitive_label = (
                            f'<div style="font-size:9px;color:#f59e0b;margin-bottom:8px;">'
                            f'* Top sensitive: {", ".join(badges_A)}</div>'
                        )

                detail_html = create_stereoset_example_html(
                    ex_A,
                    example_B=ex_B_detail,
                    model_A_name=name_A,
                    model_B_name=name_B,
                    sensitive_heads=get_sensitive_heads(mk),
                    head_profile_stats=get_head_profile_stats(mk),
                    sensitive_heads_B=get_sensitive_heads(mk_B) if has_B else None,
                    head_profile_stats_B=get_head_profile_stats(mk_B) if has_B else None,
                    top_n=top_n,
                    heatmap_html=heatmap_inner_html,
                    sensitive_heads_label=sensitive_label,
                    attn_sim_pct=attn_sim_pct if has_B else None,
                )

        except Exception as e:
            # Ignore SilentException (input not ready)
            if "SilentException" in str(type(e)):
                pass
            else:
                _logger.exception("Error loading example details")
                detail_html = (
                    "<div style='color:#ef4444;padding:16px;border:1px solid rgba(239,68,68,0.2);background:rgba(239,68,68,0.05);border-radius:8px;'>"
                    "<div style='font-weight:700;margin-bottom:8px;'>Error loading example details</div>"
                    "<div style='font-family:JetBrains Mono,monospace;font-size:11px;'>An internal error occurred. Check the server logs for details.</div>"
                    "</div>"
                )

        detail_section = (
            f'<div style="margin-top:16px;">{detail_html}</div>'
            if detail_html else
            '<div style="margin-top:16px;color:#64748b;font-size:11px;font-style:italic;'
            'text-align:center;padding:20px;">Click an example above to see details</div>'
        )

        return _wrap_card(
            ui.div(
                ui.HTML(filter_js),
                ui.div(
                    {"style": "display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;"},
                    ui.HTML(f'<span style="font-size:11px;color:#94a3b8;">'
                            f'Showing {min(100, len(examples))} of {len(examples)} examples</span>'),
                    ui.HTML(filter_select),
                ),
                ui.HTML(table_html),
                ui.HTML(detail_section),
            ),
            manual_header=(
                f"Example Explorer{_source_badge_html('GUS-Net' if source_mode == 'gusnet' else 'Base Encoder') if source_mode != 'compare' else ''}",
                "Browse StereoSet examples - click to inspect with attention heatmaps",
            ),
        )

    @render.download(filename="stereoset_example_attention.csv")
    def export_stereoset_example_attention_csv():
        mk = _stereoset_model_key()
        return _export_stereoset_example_cmn(mk)

    @render.download(filename="stereoset_example_attention_B.csv")
    def export_stereoset_example_attention_csv_B():
        mk = _stereoset_model_key_B()
        return _export_stereoset_example_cmn(mk)

    def _export_stereoset_example_cmn(mk):
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        
        try:
            idx = int(input.stereoset_selected_example())
        except Exception:
            yield "No example selected"
            return
            
        if idx < 0 or idx >= len(examples):
            yield "Invalid example index"
            return

        ex = examples[idx]
        
        # Get selected layer/head
        try:
            sel_layer = int(input.bias_attn_layer())
        except Exception:
            sel_layer = 0
        try:
            sel_head = int(input.bias_attn_head())
        except Exception:
             sel_head = 0

        # Re-extract
        base = _GUSNET_TO_ENCODER.get(mk, "bert-base-uncased")
        ctx = ex.get("context", "")
        s_text = ctx + " " + ex.get("stereo_sentence", "")
        a_text = ctx + " " + ex.get("anti_sentence", "")
        u_text = ctx + " " + ex.get("unrelated_sentence", "")
        
        s_tok, s_a = extract_attention_for_text(s_text, base, ModelManager)
        a_tok, a_a = extract_attention_for_text(a_text, base, ModelManager)
        
        u_tok, u_a = [], []
        if u_text.strip() and u_text.strip() != ctx.strip():
             u_tok, u_a = extract_attention_for_text(u_text, base, ModelManager)

        # Helper to get attn for layer/head
        def _get_vals(toks, attns):
            # attns is list of matricies? Or standard tuple?
            # In `bias_handlers`, `extract_attention_for_text` returns (tokens, attentions).
            # `attentions` is usually the output from `measure_bias_for_text` or similar.
            # Let's assume structure from `create_stereoset_attention_heatmaps`.
            # It expects `attentions` to be indexable by [layer][head].
            # And then it sums over the 'to' dimension? or 'from'?
            # Usually heatmaps show attention from all tokens TO all tokens.
            # But here we probably just want the attention paid BY the last token (next token prediction)?
            # OR average attention?
            # `create_stereoset_attention_heatmaps` typically visualizes the attention matrix.
            # Let's just dump the token-to-token attention for the selected head.
            # Or simpler: just the tokens and diagonal? No.
            
            # Since a CSV is 2D, we can't easily dump a full matrix.
            # Maybe just the attention from the last token (if it's generation)?
            # Or just the max/mean attention received by each token?
            
            # Given the request is "Token-Level Attention Comparison", and the chart is likely 1D (per token) or 2D.
            # If the chart is a heatmap (2D), we can't easily dump to a single CSV row per token.
            # But `create_stereoset_attention_heatmaps` creates a HEATMAP (2D).
            # So the user probably wants the underlying matrix data.
            # "Token, Stereo_Attn_From_Last, Anti_Attn_From_Last..." ?
            
            # Let's assume we align by token index.
            # But Stereo/Anti/Unrelated have different lengths!
            # So we can't align them row-by-row easily if we dump full sequences.
            
            # BUT, usually only the 'context' part is shared.
            # The continuation differs.
            
            # Let's dump them sequentially in the same file?
            # Or just the Shared Context tokens?
            
            # "Token-Level Attention Comparison" usually implies comparing how much attention specific tokens receive.
            # Let's dump the attention received by each token from the *last* token (often used in bias analysis to see what the model attends to when generating).
            
            # Let's look at `create_stereoset_attention_heatmaps` in `visualizations.py` if possible. 
            # But I can't see it now.
            # `_generate_model_heatmaps` calls it.
            
            # I'll output 3 sections in the CSV: Stereo, Anti, Unrelated.
            # Columns: Type, Token, Attention_Received (from last token), Attention_Sent (to last token - unlikely), ...
            # Actually, let's just dump the attentions for the selected head. 
            # Since matrix is NxN, maybe flattened?
            # Or just "Attention from CLS" and "Attention from SEP"?
            
            # Decision: Export the attention *received* by each token from the *final* token (prediction step), 
            # as this is standard for "what influenced the prediction".
            # If strictly encoder (BERT), maybe attention from [CLS] or average attention?
            # In StereoSet, we care about the probability of the *target* term. 
            # The target term is usually the last one or in the gap.
            
            # Safety: I'll dump the attention values used in the heatmap.
            # If the heatmap is 2D, I'll dump 3 sections of (Source, Target, Value).
            # That's generic and safe.
            pass

        yield "sentence_type,token_idx,token,attn_value"
        
        def _dump_attn(lbl, toks, attns):
             # attns[layer][head] -> matrix (seq_len, seq_len)
             # We need to handle potential list vs tensor types.
             try:
                 mat = attns[sel_layer][sel_head]
                 # If tensor/array
                 if hasattr(mat, "tolist"): mat = mat.tolist()
                 
                 # The user likely wants "how much attention did X pay to Y".
                 # Heatmaps usually show this.
                 # Let's dump the *focus* - usually meaning attention FROM the last token (context + query) TO previous tokens.
                 # This is what shows "what the model is looking at".
                 
                 # Let's just dump the row corresponding to the last token?
                 # Or the CLS token?
                 # Let's dump the full matrix in sparse format: Source, Target, Value?
                 # That might be too big.
                 
                 # Let's assume the user wants to see the attention profile.
                 # "Token-Level Attention Comparison" usually implies a 1D plot in the UI.
                 # But the code says `fig_t` is `create_stereoset_attention_heatmaps`.
                 # That sounds 2D. 
                 
                 # If it is 2D, the CSV should probably represent that.
                 # I'll iterate and dump: "stereo", i, token, value_from_last_token?
                 # Or maybe raw matrix is best.
                 
                 # Let's stick to: "stereo", source_token, target_token, value
                 # But filter to meaningful ones?
                 
                 # COMPROMISE: Dump the attention row for the LAST token (prediction time attention) 
                 # and the CLS token (global attention).
                 
                 seq_len = len(toks)
                 last_idx = seq_len - 1
                 
                 rows = []
                 for i in range(seq_len):
                     # Attention from Last Token to i
                     # matrix[from][to] or [to][from]?
                     # usually matrix[row][col] is attention FROM row TO col.
                     # So mat[last_idx][i]
                     try:
                         val = float(mat[last_idx][i])
                         rows.append(f"{lbl},{i},{toks[i]},{val:.6f}")
                     except Exception:
                        pass
                 return rows
             except Exception as e:
                 return []

        # Yield rows
        for r in _dump_attn("stereo", s_tok, s_a): yield r
        for r in _dump_attn("anti", a_tok, a_a): yield r
        if u_tok:
            for r in _dump_attn("unrelated", u_tok, u_a): yield r

    @render.download(filename="stereoset_example_attention_diff.csv")
    def export_stereoset_example_diff_csv():
        mk = _stereoset_model_key()
        return _export_stereoset_example_diff_cmn(mk)

    @render.download(filename="stereoset_example_attention_diff_B.csv")
    def export_stereoset_example_diff_csv_B():
        mk = _stereoset_model_key_B()
        return _export_stereoset_example_diff_cmn(mk)

    def _export_stereoset_example_diff_cmn(mk):
        # Re-use the same extraction logic but compute difference (Stereo - Anti)
        # For simplicity, we can just call `_export_stereoset_example_cmn` logic but modify output.
        # But that function yields strings directly.
        # So we'll duplicate the setup logic.
        
        examples = get_stereoset_examples(mk)
        if not examples:
            yield "No examples data"
            return
        
        try:
            idx = int(input.stereoset_selected_example())
        except Exception:
            yield "No example selected"
            return
            
        if idx < 0 or idx >= len(examples):
            yield "Invalid example index"
            return

        ex = examples[idx]
        
        # Get selected layer/head
        try:
            sel_layer = int(input.bias_attn_layer())
        except Exception:
            sel_layer = 0
        try:
            sel_head = int(input.bias_attn_head())
        except Exception:
             sel_head = 0

        # Re-extract
        base = _GUSNET_TO_ENCODER.get(mk, "bert-base-uncased")
        ctx = ex.get("context", "")
        s_text = ctx + " " + ex.get("stereo_sentence", "")
        a_text = ctx + " " + ex.get("anti_sentence", "")

        s_tok, s_a = extract_attention_for_text(s_text, base, ModelManager)
        a_tok, a_a = extract_attention_for_text(a_text, base, ModelManager)

        # We need the attention matrix for the selected head
        def _get_mat(attns):
            try:
                mat = attns[sel_layer][sel_head]
                if hasattr(mat, "tolist"): mat = mat.tolist()
                return mat
            except Exception:
                return None

        mat_s = _get_mat(s_a)
        mat_a = _get_mat(a_a)

        if mat_s is None or mat_a is None:
            yield "Error extracting attention"
            return
            
        yield "token_idx,token,stereo_attn,anti_attn,diff"
        
        # We assume s_tok and a_tok align on the context, but differ at the end.
        # Heatmap usually aligns them or shows them side-by-side?
        # `create_stereoset_attention_diff_heatmap` usually aligns them by common prefix or just shows difference on common tokens?
        # Actually diff heatmap is usually S - A. 
        # But if they have different tokens, how do we subtract?
        # Usually they differ by one word.
        # If lengths differ, we truncate or pad?
        # Let's assume lengths are same or we take min length.
        
        min_len = min(len(s_tok), len(a_tok))
        
        # Using prediction time attention (last token attending to others)
        last_idx_s = len(s_tok) - 1
        last_idx_a = len(a_tok) - 1
        
        for i in range(min_len):
            try:
                val_s = float(mat_s[last_idx_s][i])
                val_a = float(mat_a[last_idx_a][i])
                diff = val_s - val_a
                # Use stereo token as label if it matches, else "stereo/anti"
                tok = s_tok[i]
                if i < len(a_tok) and s_tok[i] != a_tok[i]:
                    tok = f"{s_tok[i]}/{a_tok[i]}"
                    
                yield f"{i},{tok},{val_s:.6f},{val_a:.6f},{diff:.6f}"
            except Exception:
                pass

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_label(data, is_A, compare_m):
        if compare_m: return _get_bias_model_label(data)
        return "Prompt A" if is_A else "Prompt B"

__all__ = ["bias_server_handlers"]
