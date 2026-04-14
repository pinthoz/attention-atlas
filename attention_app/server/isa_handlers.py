"""ISA (Inter-Sentence Attention) reactive handlers.

Extracted from ``main.py`` to reduce monolith size.  Contains scatter
renderers, token-to-token heatmaps, detail info cards, and click-event
handlers for the ISA section.
"""

import json
import logging

import plotly.graph_objects as go
from shiny import reactive, render, ui

from ..isa import get_sentence_token_attention
from .renderers import get_isa_scatter_view

_logger = logging.getLogger(__name__)


def register_isa_handlers(
    input,
    output,
    get_active_result,
    generate_export_filename,
    isa_selected_pair,
    isa_selected_pair_B,
):
    """Wire up all ISA-section output renderers and event handlers.

    Parameters
    ----------
    input : shiny Input
    output : shiny Output (used via ``@output`` decorator)
    get_active_result : callable
        ``get_active_result(suffix="")`` → ``ComputeResult | None``
    generate_export_filename : callable
        Passed through to ``get_isa_scatter_view`` as ``export_filename_fn``.
    isa_selected_pair, isa_selected_pair_B : reactive.Value
        Stores the ``(target_idx, source_idx)`` pair clicked on the scatter.
    """

    # ── Scatter renderers ───────────────────────────────────────────────

    @output(id="isa_scatter")
    @render.ui
    def isa_scatter_renderer():
        res = get_active_result()
        return get_isa_scatter_view(
            res, suffix="", plot_only=False,
            export_filename_fn=generate_export_filename,
        )

    @output(id="isa_scatter_A_compare")
    @render.ui
    def isa_scatter_renderer_A_compare():
        res = get_active_result()
        return get_isa_scatter_view(
            res, suffix="", vertical_layout=True,
            export_filename_fn=generate_export_filename,
        )

    @output(id="isa_scatter_B_compare")
    @render.ui
    def isa_scatter_renderer_B_compare():
        res = get_active_result("_B")
        return get_isa_scatter_view(
            res, suffix="_B", vertical_layout=True,
            export_filename_fn=generate_export_filename,
        )

    @output(id="isa_scatter_B")
    @render.ui
    def isa_scatter_renderer_B():
        res = get_active_result("_B")
        return get_isa_scatter_view(
            res, suffix="_B", vertical_layout=True,
            export_filename_fn=generate_export_filename,
        )

    @output
    @render.ui
    def isa_row_dynamic():
        """Dynamic ISA Layout for Single Mode."""
        return ui.output_ui("isa_scatter")

    # ── Token-to-token heatmap (shared logic for A / B) ─────────────────

    def _isa_token_view(pair, res, selected_tokens_reader, model_label,
                        div_id):
        """Render a token-to-token ISA heatmap for one side (A or B)."""
        if res is None or pair is None:
            return ui.div(
                ui.p(
                    "Select a point on the scatter plot to view "
                    "token-to-token attention.",
                    style="color:#94a3b8;font-size:13px;font-style:italic;",
                ),
                style=(
                    "height:350px;display:flex;align-items:center;"
                    "justify-content:center;border:1px dashed #e2e8f0;"
                    "border-radius:8px;background:#f8fafc;"
                ),
            )

        target_idx, source_idx = pair
        tokens, attentions = res.tokens, res.attentions
        isa_data = res[-2]
        boundaries = isa_data["sentence_boundaries_ids"]

        sub_att, tokens_combined, src_start = get_sentence_token_attention(
            attentions, tokens, target_idx, source_idx, boundaries,
        )

        toks_target = [t.replace("Ġ", "").replace("##", "")
                       for t in tokens_combined[:src_start]]
        toks_source = [t.replace("Ġ", "").replace("##", "")
                       for t in tokens_combined[src_start:]]

        # --- Highlight selected tokens ---
        selected_indices = []
        try:
            val = selected_tokens_reader()
            if val:
                selected_indices = json.loads(val)
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
        if not selected_indices:
            try:
                idx = int(input.global_focus_token())
            except Exception:
                idx = -1
            if idx != -1:
                selected_indices = [idx]

        def get_range(idx):
            start = boundaries[idx]
            end = boundaries[idx + 1] if idx < len(boundaries) - 1 else len(tokens)
            return start, end

        t_start, t_end = get_range(target_idx)
        s_start, s_end = get_range(source_idx)

        target_hl = [s - t_start for s in selected_indices
                     if t_start <= s < t_end]
        source_hl = [s - s_start for s in selected_indices
                     if s_start <= s < s_end]

        def style_ticks(toks, hl_set):
            out = []
            for i, tok in enumerate(toks):
                if i in hl_set:
                    out.append(
                        f"<span style='color:#ec4899;font-weight:bold;"
                        f"font-size:12px'>{tok}</span>"
                    )
                else:
                    out.append(tok)
            return out

        styled_target = style_ticks(toks_target, target_hl)
        styled_source = style_ticks(toks_source, source_hl)

        colorscale = [
            [0.0, "#f8fafc"], [0.2, "#e0f2fe"], [0.4, "#bae6fd"],
            [0.6, "#60a5fa"], [0.8, "#3b82f6"], [1.0, "#4f46e5"],
        ]

        fig = go.Figure(data=go.Heatmap(
            z=sub_att, x=toks_source, y=toks_target,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text="Attention", side="right",
                           font=dict(color="#64748b", size=11)),
                tickfont=dict(color="#64748b", size=10),
            ),
            hovertemplate=(
                "Target: %{y}<br>Source: %{x}<br>"
                "Weight: %{z:.4f}<extra></extra>"
            ),
        ))

        for idx in target_hl:
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=len(toks_source) - 0.5,
                y0=idx - 0.5, y1=idx + 0.5,
                fillcolor="rgba(236,72,153,0.15)",
                line=dict(color="#ec4899", width=1), layer="above",
            )
        for idx in source_hl:
            fig.add_shape(
                type="rect",
                x0=idx - 0.5, x1=idx + 0.5,
                y0=-0.5, y1=len(toks_target) - 0.5,
                fillcolor="rgba(236,72,153,0.15)",
                line=dict(color="#ec4899", width=1), layer="above",
            )

        fig.update_layout(
            title=dict(
                text=f"Token-to-Token - S{target_idx} ← S{source_idx} ({model_label})",
                font=dict(size=14, color="#1e293b",
                          family="Inter, system-ui, sans-serif"),
            ),
            xaxis=dict(
                title=dict(text="Source tokens",
                           font=dict(color="#475569", size=11)),
                tickmode="array",
                tickvals=list(range(len(toks_source))),
                ticktext=styled_source,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9",
            ),
            yaxis=dict(
                title=dict(text="Target tokens",
                           font=dict(color="#475569", size=11)),
                tickmode="array",
                tickvals=list(range(len(toks_target))),
                ticktext=styled_target,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9",
                autorange="reversed",
            ),
            height=420, width=440, autosize=True,
            margin=dict(l=60, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif"),
        )
        return ui.HTML(fig.to_html(
            include_plotlyjs="cdn", full_html=False,
            div_id=div_id, config={"displayModeBar": False},
        ))

    @output
    @render.ui
    def isa_token_view():
        return _isa_token_view(
            isa_selected_pair(), get_active_result(),
            input.global_selected_tokens, "Model A",
            "isa_token_view_plot",
        )

    @output(id="isa_token_view_B")
    @render.ui
    def isa_token_view_B():
        return _isa_token_view(
            isa_selected_pair_B(), get_active_result("_B"),
            input.global_selected_tokens_B, "Model B",
            "isa_token_view_plot_B",
        )

    # ── Detail info cards ───────────────────────────────────────────────

    def _isa_detail_info(pair, res):
        if pair is None:
            return ui.HTML(
                "<em style='color:#94a3b8;'>Click a dot on the ISA chart.</em>"
            )
        tx, sy = pair
        score = 0.0
        if res and res[-2]:
            score = res[-2]["sentence_attention_matrix"][tx, sy]
        return ui.HTML(
            f"Sentence {tx} (target) ← Sentence {sy} (source) · "
            f"ISA: <strong>{score:.4f}</strong>"
        )

    @output
    @render.ui
    def isa_detail_info():
        return _isa_detail_info(isa_selected_pair(), get_active_result())

    @output(id="isa_detail_info_B")
    @render.ui
    def isa_detail_info_B():
        return _isa_detail_info(isa_selected_pair_B(),
                                get_active_result("_B"))

    # ── Click event handlers ────────────────────────────────────────────

    @reactive.effect
    @reactive.event(input.isa_scatter_click)
    def _handle_isa_click():
        click = input.isa_scatter_click()
        if not click or "x" not in click or "y" not in click:
            return
        isa_selected_pair.set((int(click["y"]), int(click["x"])))

    @reactive.effect
    @reactive.event(input.isa_scatter_click_B)
    def _handle_isa_click_B():
        click = input.isa_scatter_click_B()
        if not click or "x" not in click or "y" not in click:
            return
        isa_selected_pair_B.set((int(click["y"]), int(click["x"])))

    @reactive.effect
    @reactive.event(input.isa_click)
    def handle_isa_overlay():
        trigger_data = input.isa_click()
        _logger.debug("handle_isa_overlay triggered with: %s", trigger_data)
        if not trigger_data:
            return
        sent_x_idx = trigger_data.get("y")
        sent_y_idx = trigger_data.get("x")
        if sent_x_idx is None or sent_y_idx is None:
            return
        _logger.debug("Setting isa_selected_pair to (%s, %s)",
                      sent_x_idx, sent_y_idx)
        isa_selected_pair.set((sent_x_idx, sent_y_idx))

    @reactive.effect
    @reactive.event(input.isa_overlay_trigger)
    def _handle_isa_overlay_trigger():
        data = input.isa_overlay_trigger()
        _logger.debug("isa_overlay_trigger received: %s", data)
        if data:
            try:
                x = int(data["sentXIdx"])
                y = int(data["sentYIdx"])
                _logger.debug("Setting isa_selected_pair to (%s, %s)", x, y)
                isa_selected_pair.set((x, y))
            except Exception as e:
                _logger.debug("Error parsing trigger data: %s", e)
