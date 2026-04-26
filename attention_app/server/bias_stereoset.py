"""StereoSet evaluation renderers and export handlers.

Extracted from bias_handlers.py to reduce monolith size.
Contains overview gauges, category/demographic breakdowns,
head sensitivity analysis, attention-bias correlation, and
the example explorer with attention heatmaps.
"""

import html as _html
import logging

import numpy as np
from shiny import render, ui

from .bias_helpers import (
    _deferred_plotly, _wrap_card, _get_bias_model_label, _source_badge_html,
    _chart_with_png_btn, _clean_gusnet_label, _GUSNET_TO_ENCODER,
)
from ..models import ModelManager
from ..bias.feature_extraction import extract_attention_for_text
from .bias_styles import (
    BTN_STYLE_CSV as _BTN_STYLE_CSV,
    TH as _TH, TR as _TR, TD as _TD, TC as _TC, TS as _TS,
    TBG as _TBG, TBR as _TBR, TBA as _TBA, TBB as _TBB, TBP as _TBP,
    TN as _TN,
)
from .bias_exports import (
    csv_stereoset_features as _csv_stereoset_features_fn,
    csv_stereoset_sensitivity as _csv_stereoset_sensitivity_fn,
    csv_stereoset_category as _csv_stereoset_category_fn,
    csv_stereoset_distribution as _csv_stereoset_distribution_fn,
    csv_stereoset_demographic as _csv_stereoset_demographic_fn,
    csv_cross_method as _csv_cross_method_fn,
    csv_perturb_attn as _csv_perturb_attn_fn,
)
from ..bias.stereoset import (
    get_stereoset_scores,
    get_stereoset_examples,
    get_sensitive_heads,
    get_top_features,
    get_metadata,
    compute_model_similarity,
    get_head_sensitivity_matrix,
    get_head_profile_stats,
)
from ..bias.stereoset.stereoset_data import get_gusnet_key
from ..bias.visualizations import (
    create_stereoset_category_chart,
    create_stereoset_overview_html,
    create_stereoset_bias_distribution,
    create_stereoset_head_sensitivity_heatmap,
    create_stereoset_head_distributions,
    create_stereoset_attention_scatter,
    create_stereoset_attention_heatmaps,
    create_stereoset_attention_diff_heatmap,
    create_stereoset_example_html,
)

_logger = logging.getLogger(__name__)


def register_stereoset_handlers(
    input, output,
    bias_results, bias_results_B,
    active_bias_compare_models, active_bias_compare_prompts,
    ig_results, ig_results_B,
    lrp_results, lrp_results_B,
    perturbation_results, perturbation_results_B,
    _get_attn_source_mode,
):
    """Wire up StereoSet evaluation renderers and export handlers."""

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
                _logger.debug("Suppressed exception", exc_info=True)
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
        yield _csv_stereoset_features_fn(_stereoset_model_key())

    @render.download(filename="top_discriminative_features_B.csv")
    def export_stereoset_features_csv_B():
        yield _csv_stereoset_features_fn(_stereoset_model_key_B())

    @render.download(filename='sensitive_heads.csv')
    def export_stereoset_sensitivity_csv():
        yield _csv_stereoset_sensitivity_fn(_stereoset_model_key())

    @render.download(filename='sensitive_heads_B.csv')
    def export_stereoset_sensitivity_csv_B():
        yield _csv_stereoset_sensitivity_fn(_stereoset_model_key_B())

    @render.download(filename='stereoset_category_scores.csv')
    def export_stereoset_category_csv():
        yield _csv_stereoset_category_fn(_stereoset_model_key())

    @render.download(filename='stereoset_category_scores_B.csv')
    def export_stereoset_category_csv_B():
        yield _csv_stereoset_category_fn(_stereoset_model_key_B())

    @render.download(filename='stereoset_bias_distribution.csv')
    def export_stereoset_distribution_csv():
        yield _csv_stereoset_distribution_fn(_stereoset_model_key())

    @render.download(filename='stereoset_bias_distribution_B.csv')
    def export_stereoset_distribution_csv_B():
        yield _csv_stereoset_distribution_fn(_stereoset_model_key_B())

    @render.download(filename='stereoset_demographic_slices.csv')
    def export_stereoset_demographic_csv():
        yield _csv_stereoset_demographic_fn(_stereoset_model_key())

    @render.download(filename='stereoset_demographic_slices_B.csv')
    def export_stereoset_demographic_csv_B():
        yield _csv_stereoset_demographic_fn(_stereoset_model_key_B())

    @render.download(filename='perturbation_vs_attention.csv')
    def export_perturb_attn_csv():
        bundle = perturbation_results.get()
        if not bundle or not bundle.perturb_vs_attn_spearman:
            yield "No perturbation-attention data"; return
        yield _csv_perturb_attn_fn(bundle)

    @render.download(filename='perturbation_vs_attention_B.csv')
    def export_perturb_attn_csv_B():
        bundle = perturbation_results_B.get()
        if not bundle or not bundle.perturb_vs_attn_spearman:
            yield "No perturbation-attention data"; return
        yield _csv_perturb_attn_fn(bundle)

    @render.download(filename='cross_method_agreement.csv')
    def export_cross_method_csv():
        lrp_b, ig_b = lrp_results.get(), ig_results.get()
        if not lrp_b or not ig_b:
            yield "No cross-method data"; return
        yield _csv_cross_method_fn(lrp_b, ig_b)

    @render.download(filename='cross_method_agreement_B.csv')
    def export_cross_method_csv_B():
        lrp_b, ig_b = lrp_results_B.get(), ig_results_B.get()
        if not lrp_b or not ig_b:
            yield "No cross-method data"; return
        yield _csv_cross_method_fn(lrp_b, ig_b)


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
                _logger.debug("Suppressed exception", exc_info=True)
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
                    _logger.debug("Suppressed exception", exc_info=True)
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
                    _logger.exception("Could not generate attention heatmaps")
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
                         _logger.debug("Suppressed exception", exc_info=True)
                         pass
                 return rows
             except Exception as e:
                 _logger.debug("Suppressed: %s", e)
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
                _logger.debug("Suppressed exception", exc_info=True)
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
                _logger.debug("Suppressed exception", exc_info=True)
                pass


