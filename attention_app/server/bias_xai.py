"""XAI analysis renderers: ablation, IG, perturbation, LRP.

Extracted from bias_handlers.py to reduce monolith size.
Each renderer displays results from a specific explainability method.
"""

import logging

from shiny import render, ui, reactive

from .bias_helpers import _deferred_plotly, _wrap_card
from .bias_styles import (
    BTN_STYLE_CSV as _BTN_STYLE_CSV, BTN_STYLE_PNG as _BTN_STYLE_PNG,
    TH as _TH, TR as _TR, TD as _TD, TC as _TC, TS as _TS,
    TBG as _TBG, TBR as _TBR, TBA as _TBA, TBB as _TBB, TBP as _TBP,
    TN as _TN,
)
from .bias_exports import (
    csv_ablation as _csv_ablation_fn,
    csv_ig_correlation as _csv_ig_correlation_fn,
    csv_topk_overlap as _csv_topk_overlap_fn,
    csv_perturbation as _csv_perturbation_fn,
    csv_lrp as _csv_lrp_fn,
    csv_perturb_attn as _csv_perturb_attn_fn,
)

_logger = logging.getLogger(__name__)


def register_xai_handlers(
    input, output,
    ablation_results, ablation_results_B, ablation_running,
    ig_results, ig_results_B, ig_running,
    perturbation_results, perturbation_results_B, perturbation_running,
    lrp_results, lrp_results_B, lrp_running,
    active_bias_compare_models, active_bias_compare_prompts,
    bias_results_B_rv,
    _get_attn_source_mode,
    _get_bias_model_label,
):
    """Wire up ablation / IG / perturbation / LRP renderers and exports."""

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
            _logger.debug("Suppressed exception", exc_info=True)
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
            yield "No ablation data"; return
        yield _csv_ablation_fn(results)

    @render.download(filename="ablation_results_B.csv")
    def export_ablation_csv_B():
        results = ablation_results_B.get()
        if not results:
            yield "No ablation data"; return
        yield _csv_ablation_fn(results)

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
            _logger.exception("Error rendering IG results")
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
            _logger.debug("Suppressed exception", exc_info=True)
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
            yield "No IG data"; return
        yield _csv_ig_correlation_fn(bundle)

    @render.download(filename="ig_correlation_results_B.csv")
    def export_ig_csv_B():
        bundle = ig_results_B.get()
        if not bundle:
            yield "No IG data"; return
        yield _csv_ig_correlation_fn(bundle)

    @render.download(filename="topk_overlap_results.csv")
    def export_topk_csv():
        bundle = ig_results.get()
        if not bundle or not isinstance(bundle, IGAnalysisBundle) or not bundle.topk_overlaps:
            yield "No Top-K overlap data"; return
        yield _csv_topk_overlap_fn(bundle)

    @render.download(filename="topk_overlap_results_B.csv")
    def export_topk_csv_B():
        bundle = ig_results_B.get()
        if not bundle or not isinstance(bundle, IGAnalysisBundle) or not bundle.topk_overlaps:
            yield "No Top-K overlap data"; return
        yield _csv_topk_overlap_fn(bundle)

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
            yield "No perturbation data"; return
        yield _csv_perturbation_fn(bundle)

    @render.download(filename="perturbation_results_B.csv")
    def export_perturbation_csv_B():
        bundle = perturbation_results_B.get()
        if not bundle:
            yield "No perturbation data"; return
        yield _csv_perturbation_fn(bundle)

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
            yield "No LRP data"; return
        yield _csv_lrp_fn(bundle)

    @render.download(filename="lrp_results_B.csv")
    def export_lrp_csv_B():
        bundle = lrp_results_B.get()
        if not bundle:
            yield "No LRP data"; return
        yield _csv_lrp_fn(bundle)


