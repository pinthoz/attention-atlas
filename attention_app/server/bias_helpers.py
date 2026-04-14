"""Pure helper functions and constants for bias analysis.

These functions have **no reactive dependencies** and can be imported and
tested independently of the Shiny server closure.
"""

import json
import logging
import re as _re
from datetime import datetime

import numpy as np
import torch
from shiny import ui

from ..bias import GusNetDetector, EnsembleGusNetDetector, AttentionBiasAnalyzer
from ..models import ModelManager
from ..ui.components import viz_header

_logger = logging.getLogger(__name__)


# ── GUS-Net model mapping constants ──────────────────────────────────────

# Map GUS-Net model key -> matching encoder model for attention analysis.
_GUSNET_TO_ENCODER = {
    "gusnet-bert": "pinthoz/gus-net-bert",
    "gusnet-bert-large": "pinthoz/gus-net-bert-large",
    "gusnet-gpt2": "pinthoz/gus-net-gpt2",
    "gusnet-gpt2-medium": "pinthoz/gus-net-gpt2-medium",
    "gusnet-bert-custom": "pinthoz/gus-net-bert-custom",
    "gusnet-ensemble": "pinthoz/gus-net-bert",
    "gusnet-bert-new": "pinthoz/gus-net-bert",
    "gusnet-gpt2-new": "pinthoz/gus-net-gpt2",
    "gusnet-bert-paper": "pinthoz/gus-net-bert",
    "gusnet-gpt2-paper": "pinthoz/gus-net-gpt2",
    "gusnet_bert": "pinthoz/gus-net-bert",
    "gusnet_bert_large": "pinthoz/gus-net-bert-large",
    "gusnet_gpt2": "pinthoz/gus-net-gpt2",
    "gusnet_gpt2_medium": "pinthoz/gus-net-gpt2-medium",
}

# Map GUS-Net model key -> original pretrained base model (before fine-tuning).
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


# ── Helper functions ─────────────────────────────────────────────────────

def _clean_gusnet_label(raw: str) -> str:
    """Simplify a raw GUS-NET model name to a clean display label.

    E.g. ``gus-net-bert-paper-clean-2`` -> ``GUS-NET BERT``,
         ``gus-net-gpt2-medium``        -> ``GUS-NET GPT-2 Medium``.
    For non-GUS-NET names the string is returned upper-cased.
    """
    if "/" in raw:
        raw = raw.split("/")[-1]
    s = raw.lower().replace("_", "-")
    if "gus" not in s:
        return raw.upper()
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
    """Extract attentions from the original pretrained base model (before fine-tuning)."""
    base_model_name = _GUSNET_TO_BASE.get(bias_model_key, "bert-base-uncased")

    tokenizer, encoder_model, _ = ModelManager.get_model(base_model_name)
    device = ModelManager.get_device()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())

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


def _source_badge_html(label: str) -> str:
    if label == "GUS-Net":
        return " <span style='font-size:10px;padding:2px 6px;border-radius:4px;background:#fce7f3;color:#be185d;border:1px solid #fbcfe8;vertical-align:middle;margin-left:8px;font-weight:600;'>GUS-Net</span>"
    elif label == "Base Encoder":
        return " <span style='font-size:10px;padding:2px 6px;border-radius:4px;background:#e0f2fe;color:#1d4ed8;border:1px solid #bfdbfe;vertical-align:middle;margin-left:8px;font-weight:600;'>Base Encoder</span>"
    return f" <span style='font-size:10px;padding:2px 6px;border-radius:4px;background:#f1f5f9;color:#475569;border:1px solid #e2e8f0;vertical-align:middle;margin-left:8px;font-weight:600;'>{label}</span>"

def _deferred_plotly(fig, container_id, height=None, config=None, click_input=None):
    """Render a Plotly figure as deferred HTML - only calls Plotly.newPlot()
    when the container becomes visible."""
    import plotly.io as pio
    import base64
    import html as _html

    if height is None:
        fig_height = getattr(fig.layout, "height", None)
        if fig_height:
            height = f"{fig_height}px"
        else:
            height = "400px"

    fig_json = pio.to_json(fig, validate=False)
    cfg = json.dumps(config or {"displayModeBar": False, "responsive": True})
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

def _chart_with_png_btn(chart_html: str, container_id: str, filename: str, controls: list = None) -> str:
    """Wrap chart HTML with controls (PNG btn + optional others) aligned to the right."""
    from .bias_styles import BTN_STYLE_PNG as _BTN_STYLE_PNG
    
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



def _wrap_card(content, title=None, subtitle=None, help_text=None, manual_header=None, style=None, controls=None):
    """Wrap content in a card with consistent header style."""
    base_style = "min-height: auto; display: flex; flex-direction: column;"
    if style:
        base_style += f" {style}"

    header = None
    if manual_header:
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


# ── Token-alignment helper ───────────────────────────────────────────────

def _align_gusnet_to_attention_tokens(gusnet_labels, attention_tokens, gusnet_special_tokens=None):
    """Align GUS-Net token labels to the attention model's tokens."""
    if gusnet_special_tokens is None:
        gusnet_special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    attn_special = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    def _clean(tok):
        return tok.replace("##", "").replace("\u0120", "").replace("\u0120", "").lower().strip()

    gus_clean = []
    gus_data = []
    for label in gusnet_labels:
        if label["token"] in gusnet_special_tokens:
            continue
        gus_clean.append(_clean(label["token"]))
        gus_data.append(label)

    aligned = [None] * len(attention_tokens)
    gus_idx = 0
    gus_remainder = ""
    gus_current_label = None

    for bt_idx, bt in enumerate(attention_tokens):
        if bt in attn_special or (bt.startswith("[") and bt.endswith("]")):
            continue

        clean_bt = _clean(bt)
        if not clean_bt:
            continue

        if gus_remainder:
            if gus_remainder.startswith(clean_bt):
                aligned[bt_idx] = gus_current_label
                gus_remainder = gus_remainder[len(clean_bt):]
                if not gus_remainder:
                    gus_current_label = None
                continue
            else:
                gus_remainder = ""
                gus_current_label = None

        if gus_idx < len(gus_clean) and gus_clean[gus_idx] == clean_bt:
            aligned[bt_idx] = gus_data[gus_idx]
            gus_idx += 1
            continue

        if gus_idx < len(gus_clean) and gus_clean[gus_idx].startswith(clean_bt):
            aligned[bt_idx] = gus_data[gus_idx]
            gus_remainder = gus_clean[gus_idx][len(clean_bt):]
            gus_current_label = gus_data[gus_idx]
            if not gus_remainder:
                gus_idx += 1
                gus_current_label = None
            else:
                gus_idx += 1
            continue

        if gus_idx < len(gus_clean):
            accumulated = ""
            scan = gus_idx
            while scan < len(gus_clean) and len(accumulated) < len(clean_bt):
                accumulated += gus_clean[scan]
                scan += 1
            if accumulated == clean_bt:
                aligned[bt_idx] = gus_data[gus_idx]
                gus_idx = scan
                continue

        for look in range(gus_idx, min(gus_idx + 5, len(gus_clean))):
            if gus_clean[look] == clean_bt:
                aligned[bt_idx] = gus_data[look]
                gus_idx = look + 1
                break

    return aligned


def _get_bias_model_label(res):
    """Return a human-readable label for the bias model used in the result."""
    from attention_app.bias.gusnet_detector import MODEL_REGISTRY
    key = res.get("bias_model_key", "gusnet-bert")
    cfg = MODEL_REGISTRY.get(key, {})
    return cfg.get("display_name", key)


def _process_raw_bias_result(raw_res, thresholds, use_optimized=False):
    """Apply thresholds to raw results and regenerate attention metrics."""
    if not raw_res:
        return None

    try:
        bias_model_key = raw_res["bias_model_key"]

        from attention_app.bias.gusnet_detector import GusNetDetector, EnsembleGusNetDetector

        if bias_model_key == "gusnet-ensemble":
            det = EnsembleGusNetDetector(model_key_a="gusnet-bert", model_key_b="gusnet-bert-custom")
        else:
            det = GusNetDetector(model_key=bias_model_key, use_optimized=use_optimized)

        gusnet_labels = det.apply_thresholds(
            raw_res["gus_tokens"],
            raw_res["gus_probs"],
            thresholds=thresholds
        )

        tokens = raw_res["tokens"]
        gus_special = raw_res["gus_special"]

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

        bias_summary = det.get_bias_summary(token_labels)
        bias_spans = det.get_biased_spans(token_labels)

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
            bias_matrix = attention_analyzer.create_attention_bias_matrix(
                list(attentions), biased_indices
            )

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

    bias_pcts = [r["bias_summary"]["bias_percentage"] for r in analyzed if "bias_summary" in r]
    avg_bias_pct = round(sum(bias_pcts) / len(bias_pcts), 1) if bias_pcts else 0

    confs = [r["bias_summary"]["avg_confidence"] for r in analyzed if "bias_summary" in r and r["bias_summary"].get("avg_confidence")]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0

    cat_dist = {}
    for r in analyzed:
        if "bias_summary" in r:
            for cat in r["bias_summary"].get("categories_found", []):
                cat_dist[cat] = cat_dist.get(cat, 0) + 1

    term_counts = {}
    for r in analyzed:
        for ts in r.get("token_scores", []):
            if ts.get("is_biased"):
                term_counts[ts["token"]] = term_counts.get(ts["token"], 0) + 1
    top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:20]

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

    ranked = sorted(
        [{"index": r["index"], "text": r["text"][:100], "bias_percentage": r["bias_summary"]["bias_percentage"]}
         for r in analyzed if "bias_summary" in r],
        key=lambda x: x["bias_percentage"], reverse=True
    )[:20]

    attention_analysis = {}
    if all_bar_matrices:
        stacked = np.stack(all_bar_matrices)
        avg_bar = np.nanmean(stacked, axis=0)
        n_layers, n_heads = avg_bar.shape

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

    ig_rhos, ig_sig_heads, ig_jaccards, ig_rbos = [], 0, [], []
    abl_impacts, abl_kls = [], []
    perturb_ig_rhos, perturb_attn_rhos = [], []
    lrp_ig_rhos, lrp_attn_rhos = [], []

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

    cf_analysis = []
    for r in analyzed:
        swaps = r.get("counterfactual_swaps")
        if swaps:
            cf_analysis.append({
                "sentence_index": r["index"],
                "text": r["text"],
                "available_swaps": swaps,
            })

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
        _logger.debug("Suppressed exception", exc_info=True)
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
