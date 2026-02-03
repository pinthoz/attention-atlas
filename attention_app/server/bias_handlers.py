import json
from datetime import datetime
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from shiny import ui, render, reactive

from ..models import ModelManager
from ..bias import (
    GusNetDetector,
    AttentionBiasAnalyzer,
    create_token_bias_heatmap,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_ratio_formula_html,
    create_bias_criteria_html,
    create_bias_sentence_preview,
    create_token_bias_strip,
)
from ..ui.bias_ui import create_bias_accordion, create_floating_bias_toolbar


# ─── Token-alignment helper ───────────────────────────────────────────────

def _align_gusnet_to_bert(gusnet_labels, bert_tokens):
    """Align GUS-Net results (own tokenizer) to BERT tokens by text matching.

    Returns a list of dicts (same length as bert_tokens) with GUS-Net scores
    merged in.  Unmatched BERT tokens get zeroed scores.
    """
    aligned = []
    gus_idx = 0
    gus_clean = [
        l["token"].replace("##", "").lower()
        for l in gusnet_labels
        if l["token"] not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    gus_data = [
        l for l in gusnet_labels
        if l["token"] not in ("[CLS]", "[SEP]", "[PAD]")
    ]

    for bt in bert_tokens:
        clean_bt = bt.replace("##", "").replace("Ġ", "").lower()
        if bt.startswith("[") and bt.endswith("]"):
            aligned.append(None)
            continue

        # Try to find a matching GUS-Net token
        matched = None
        if gus_idx < len(gus_clean) and gus_clean[gus_idx] == clean_bt:
            matched = gus_data[gus_idx]
            gus_idx += 1
        else:
            # Search ahead a few positions
            for look in range(gus_idx, min(gus_idx + 3, len(gus_clean))):
                if gus_clean[look] == clean_bt:
                    matched = gus_data[look]
                    gus_idx = look + 1
                    break

        aligned.append(matched)

    return aligned


# ─── Server handler registration ──────────────────────────────────────────

def bias_server_handlers(input, output, session):
    """Create server handlers for bias analysis tab."""

    bias_running = reactive.value(False)
    bias_results = reactive.value(None)
    bias_history = reactive.Value([])

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
            safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '&quot;').replace('\n', ' ')
            items.append(
                ui.div(
                    display_text,
                    class_="history-item",
                    onclick=f"selectBiasHistoryItem('{safe_text}')"
                )
            )
        return ui.div(*items)

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
                safe = item.replace("\\", "\\\\").replace("'", "\\'").replace('"', '&quot;').replace('\n', ' ')
                display_safe = display.replace("`", "\\`").replace("${", "\\${")
                html_content += (
                    f'<div class="history-item" '
                    f"onclick=\"selectBiasHistoryItem('{safe}')\">"
                    f'{display_safe}</div>'
                )

        js_code = f"""
            var dropdown = document.getElementById('bias-history-dropdown');
            if (dropdown) {{
                dropdown.innerHTML = `{html_content}`;
            }}
        """
        ui.insert_ui(
            selector="body", where="beforeEnd",
            ui=ui.tags.script(js_code),
        )

    # ── Session Logic ──

    @render.download(filename=lambda: f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def save_bias_session():
        data = {
            "text": input.bias_input_text(),
            "threshold": input.bias_threshold(),
            "timestamp": datetime.now().isoformat(),
            "type": "bias_analysis"
        }
        return json.dumps(data, indent=2)

    @reactive.Effect
    @reactive.event(input.load_bias_session_upload)
    def load_bias_session():
        file_info = input.load_bias_session_upload()
        if not file_info:
            return
            
        try:
            with open(file_info[0]["datapath"], "r") as f:
                data = json.load(f)
                
            if "text" in data:
                ui.update_text_area("bias_input_text", value=data["text"])
            if "threshold" in data:
                ui.update_slider("bias_threshold", value=data["threshold"])
                
            # Trigger analysis automatically as sessions usually imply a saved result state
            # but we just restore parameters here for now to be safe
        except Exception as e:
            print(f"Error loading bias session: {e}")
            traceback.print_exc()

    def heavy_bias_compute(text, model_name, threshold):
        """Perform bias analysis computation (runs in thread pool)."""
        print(f"DEBUG: Starting heavy_bias_compute (threshold={threshold})")
        if not text:
            return None

        try:
            # Load attention model
            tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
            device = ModelManager.get_device()

            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = encoder_model(**inputs)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attentions = outputs.attentions
            print(f"DEBUG: Got {len(tokens)} tokens, {len(attentions)} layers")

            # ── Token-level bias detection (GUS-Net only) ──
            gus_detector = GusNetDetector(threshold=threshold)
            gusnet_labels = gus_detector.detect_bias(text)
            print(f"DEBUG: GUS-Net returned {len(gusnet_labels)} labels")

            # Align GUS-Net labels to BERT attention tokens
            gus_aligned = _align_gusnet_to_bert(gusnet_labels, tokens)

            # Build unified token_labels aligned to BERT tokens
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
                        "threshold": threshold,
                    })

            bias_summary = gus_detector.get_bias_summary(token_labels)
            bias_spans = gus_detector.get_biased_spans(token_labels)
            print(f"DEBUG: {len(bias_spans)} spans detected")

            # ── Attention x Bias ──
            attention_analyzer = AttentionBiasAnalyzer()
            biased_indices = [i for i, l in enumerate(token_labels) if l["is_biased"]]

            if biased_indices and attentions:
                attention_metrics = attention_analyzer.analyze_attention_to_bias(
                    list(attentions), biased_indices, tokens
                )
                propagation_analysis = attention_analyzer.analyze_bias_propagation(
                    list(attentions), biased_indices, tokens
                )
                bias_matrix = attention_analyzer.create_attention_bias_matrix(
                    list(attentions), biased_indices
                )
            else:
                attention_metrics = []
                propagation_analysis = {
                    "layer_propagation": [], "peak_layer": None,
                    "propagation_pattern": "none",
                }
                bias_matrix = np.array([])

            return {
                "tokens": tokens,
                "text": text,
                "attentions": attentions,
                "token_labels": token_labels,
                "bias_summary": bias_summary,
                "bias_spans": bias_spans,
                "biased_indices": biased_indices,
                "attention_metrics": attention_metrics,
                "propagation_analysis": propagation_analysis,
                "bias_matrix": bias_matrix,
            }
        except Exception as e:
            print(f"ERROR in heavy_bias_compute: {e}")
            traceback.print_exc()
            return None

    # ── Trigger analysis ──

    @reactive.effect
    @reactive.event(input.analyze_bias_btn)
    async def compute_bias():
        text = input.bias_input_text().strip()
        if not text:
            return
        bias_running.set(True)
        await session.send_custom_message('start_bias_loading', {})
        await asyncio.sleep(0.1)

        try:
            model_name = input.model_name()
        except Exception:
            model_name = "bert-base-uncased"

        threshold = input.bias_threshold()

        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool, heavy_bias_compute, text, model_name, threshold,
                )
            bias_results.set(result)
        except Exception as e:
            print(f"ERROR in compute_bias: {e}")
            traceback.print_exc()
            bias_results.set(None)
        finally:
            bias_running.set(False)
            await session.send_custom_message('stop_bias_loading', {})

    # ── Dashboard content (conditional rendering) ──

    @output
    @render.ui
    def bias_dashboard_content():
        res = bias_results.get()
        running = bias_running.get()

        try:
            text = input.bias_input_text().strip()
        except Exception:
            text = ""

        # ── Post-analysis: sentence preview + accordion ──
        if res:
            preview_html = create_bias_sentence_preview(
                res["tokens"], res["token_labels"]
            )
            return ui.div(
                ui.div(
                    {"class": "card sentence-preview-card", "style": "margin-bottom: 24px;"},
                    ui.div(
                        {"class": "viz-header"},
                        ui.h4("Sentence Preview", style="margin:0;"),
                        ui.p(
                            "Tokens coloured by bias type. Hover for category and confidence.",
                            style="font-size:11px;color:#6b7280;margin:4px 0 0;",
                        ),
                    ),
                    ui.HTML(preview_html),
                ),
                create_bias_accordion(),
                create_floating_bias_toolbar(),
                ui.tags.script("Shiny.setInputValue('toggle_bias_toolbar_visible', true, {priority: 'event'});"), 
            )

        # ── Pre-analysis: sentence preview card only ──
        if text:
            preview = ui.div(
                text,
                style=(
                    "font-family:'JetBrains Mono',monospace;color:#94a3b8;"
                    "font-size:14px;line-height:1.8;padding:8px 0;"
                    "min-height:48px;display:flex;align-items:center;"
                ),
            )
        else:
            preview = ui.div(
                'Enter text in the sidebar and click "Analyze Bias" to begin.',
                style="color:#9ca3af;font-size:12px;min-height:48px;display:flex;align-items:center;",
            )

        card = ui.div(
            {"class": "card"},
            ui.div(
                {"class": "viz-header"},
                ui.h4("Sentence Preview", style="margin:0;"),
                ui.p(
                    "Analyze text to see token-level bias detection.",
                    style="font-size:11px;color:#6b7280;margin:4px 0 0;",
                ),
            ),
            preview,
        )

        if running:
            loading = ui.div(
                {"style": "padding:40px;text-align:center;color:#9ca3af;"},
                ui.p(
                    "Analyzing bias...",
                    style="font-size:14px;color:#ff5ca9;animation:pulse 1.5s infinite;",
                ),
            )
            return ui.div(card, loading)

        return card

    # ── Method info (sidebar) ──

    @output
    @render.ui
    def bias_method_info():
        html = create_method_info_html()
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

        summary = res["bias_summary"]

        # Criteria breakdown
        criteria_html = create_bias_criteria_html(summary)

        # Metric cards using existing .metric-card design
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

        return ui.HTML(criteria_html + cards)

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
            threshold = input.bias_threshold()
            html = create_inline_bias_html(
                res["text"], res["token_labels"], res["bias_spans"],
                show_neutral=False, threshold=threshold,
            )
            return ui.HTML(html)
        except Exception as e:
            print(f"Error in inline_bias_view: {e}")
            traceback.print_exc()
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

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
            return ui.HTML(
                fig.to_html(include_plotlyjs='cdn', full_html=False,
                            config={'displayModeBar': False})
            )
        except Exception as e:
            print(f"Error creating token bias viz: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Bias spans table (per-token, one line each) ──

    @output
    @render.ui
    def bias_spans_table():
        res = bias_results.get()
        if not res:
            return None

        token_labels = res["token_labels"]
        biased = [
            lbl for lbl in token_labels
            if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if not biased:
            return ui.HTML(
                '<div style="color:#9ca3af;font-size:12px;padding:12px;">'
                'No biased tokens detected.</div>'
            )

        threshold = input.bias_threshold()
        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        items = []
        for lbl in biased:
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            types = lbl.get("bias_types", [])
            scores = lbl.get("scores", {})

            # Category badges with individual scores
            badge_parts = []
            for bt in types:
                bg = cat_colors.get(bt, "#ff5ca9")
                sc = scores.get(bt, 0)
                badge_parts.append(
                    f'<span style="display:inline-flex;align-items:center;gap:4px;'
                    f'background:{bg}18;border:1px solid {bg}40;color:{bg};'
                    f'padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">'
                    f'{bt}'
                    f'<span style="font-family:JetBrains Mono,monospace;font-weight:400;'
                    f'opacity:0.8;">{sc:.2f}</span>'
                    f'</span>'
                )
            badges_html = "".join(badge_parts)

            items.append(
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:8px 10px;border-bottom:1px solid rgba(226,232,240,0.4);">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:13px;'
                f'font-weight:600;color:#ec4899;min-width:70px;">{clean}</span>'
                f'<span style="display:flex;gap:4px;flex-wrap:wrap;">{badges_html}</span>'
                f'</div>'
            )

        html = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0 16px;'
            f'border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
            f'{"".join(items)}</div>'
            f'<div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">'
            f'Threshold: <code style="font-family:JetBrains Mono,monospace;">{threshold:.2f}</code>'
            f' &middot; Method: GUS-Net (multi-label token NER)'
            f' &middot; {len(biased)} biased tokens</div>'
        )
        return ui.HTML(html)

    # ── Token bias strip (replaces Plotly heatmap) ──

    @output
    @render.ui
    def token_bias_strip():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            html = create_token_bias_strip(res["token_labels"])
            return ui.HTML(html)
        except Exception as e:
            print(f"Error creating token bias strip: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Attention bias matrix ──

    @output
    @render.ui
    def attention_bias_matrix():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        bm = res["bias_matrix"]
        if bm.size == 0:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;">No biased tokens detected.</div>')
        try:
            fig = create_attention_bias_matrix(bm, res["attention_metrics"])
            return ui.HTML(
                fig.to_html(include_plotlyjs='cdn', full_html=False,
                            config={'displayModeBar': False})
            )
        except Exception as e:
            print(f"Error creating attention bias matrix: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Propagation plot ──

    @output
    @render.ui
    def bias_propagation_plot():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        prop = res["propagation_analysis"]["layer_propagation"]
        if not prop:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;">No propagation data.</div>')
        try:
            fig = create_bias_propagation_plot(prop)
            return ui.HTML(
                fig.to_html(include_plotlyjs='cdn', full_html=False,
                            config={'displayModeBar': False})
            )
        except Exception as e:
            print(f"Error creating propagation plot: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Combined attention + bias view ──

    @output
    @render.ui
    def combined_bias_view():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            layer_idx = int(input.bias_attn_layer())
            head_idx = int(input.bias_attn_head())
        except Exception:
            layer_idx, head_idx = 0, 0

        attentions = res["attentions"]
        if not attentions or layer_idx >= len(attentions):
            return ui.HTML('<div style="color:#9ca3af;">No attention data available.</div>')

        try:
            attn_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
            fig = create_combined_bias_visualization(
                res["tokens"], res["token_labels"], attn_matrix, layer_idx, head_idx,
            )
            return ui.HTML(
                fig.to_html(include_plotlyjs='cdn', full_html=False,
                            config={'displayModeBar': False})
            )
        except Exception as e:
            print(f"Error creating combined view: {e}")
            traceback.print_exc()
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Bias-focused heads table ──

    @output
    @render.ui
    def bias_focused_heads_table():
        res = bias_results.get()
        if not res:
            return None

        metrics = res["attention_metrics"]
        if not metrics:
            return ui.HTML(
                '<div style="color:#9ca3af;font-size:12px;padding:12px;">'
                'No attention metrics available.</div>'
            )

        # Always show top 5 heads by ratio, even if below threshold
        top_heads = sorted(
            metrics,
            key=lambda x: x.bias_attention_ratio,
            reverse=True,
        )[:5]

        n_specialized = sum(1 for m in top_heads if m.specialized_for_bias)

        rows = []
        for m in top_heads:
            is_sig = m.specialized_for_bias
            row_bg = "background:rgba(255,92,169,0.04);" if is_sig else ""
            ratio_color = "#ff5ca9" if is_sig else "#64748b"
            sig_dot = (
                '<span style="color:#22c55e;font-size:8px;margin-left:4px;" '
                'title="Above threshold (1.5)">●</span>'
                if is_sig else
                '<span style="color:#94a3b8;font-size:8px;margin-left:4px;" '
                'title="Below threshold (1.5)">○</span>'
            )

            rows.append(f"""
                <tr style="{row_bg}">
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:center;">
                        Layer {m.layer}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:center;">
                        Head {m.head}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;
                        font-family:monospace;font-weight:600;color:{ratio_color};">
                        {m.bias_attention_ratio:.3f}{sig_dot}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;
                        font-family:monospace;">{m.amplification_score:.3f}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;
                        font-family:monospace;">{m.max_bias_attention:.3f}</td>
                </tr>
            """)

        # Contextual explanation
        if n_specialized == 0:
            note_html = (
                '<div style="margin-top:12px;padding:12px;background:#f0f9ff;'
                'border:1px solid #bae6fd;border-radius:8px;font-size:11px;color:#0369a1;">'
                '<b>No heads exceed the specialization threshold (1.5).</b> '
                'This does not contradict token-level bias detection — '
                'lexical bias detected by GUS-Net may be present without being '
                'concentrated in any single attention head. The bias is distributed '
                'across the network rather than localised.'
                '</div>'
            )
        else:
            note_html = (
                f'<div style="margin-top:12px;font-size:11px;color:#64748b;">'
                f'<span style="color:#22c55e;">●</span> = above threshold (1.5) · '
                f'<span style="color:#94a3b8;">○</span> = below threshold · '
                f'Showing top 5 heads by ratio.'
                f'</div>'
            )

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f8fafc;">
                    <th style="padding:12px;text-align:center;font-size:11px;color:#64748b;
                        border-bottom:2px solid #e2e8f0;">Layer</th>
                    <th style="padding:12px;text-align:center;font-size:11px;color:#64748b;
                        border-bottom:2px solid #e2e8f0;">Head</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;
                        border-bottom:2px solid #e2e8f0;">Bias Attention Ratio</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;
                        border-bottom:2px solid #e2e8f0;">Amplification</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;
                        border-bottom:2px solid #e2e8f0;">Max Attention</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        {note_html}
        """
        return ui.HTML(table_html)

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

        token_labels = res["token_labels"]
        biased = [
            lbl for lbl in token_labels
            if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if not biased:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:12px;">No bias detected.</div>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        badge_script = f"<script>$('#bias-span-count-badge').text('{len(biased)}');</script>"

        items = [badge_script]
        for lbl in biased:
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            types = lbl.get("bias_types", [])
            scores = lbl.get("scores", {})
            max_score = max((scores.get(t, 0) for t in types), default=0)
            score_color = "#ef4444" if max_score > 0.8 else "#f59e0b" if max_score > 0.5 else "#94a3b8"

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
                f'color:#e2e8f0;font-weight:600;">{clean}</span>'
                f'<span style="color:{score_color};font-weight:600;font-size:11px;'
                f'font-family:JetBrains Mono,monospace;">{max_score:.2f}</span></div>'
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
            return ui.HTML('<span style="color:#64748b;font-size:9px;">—</span>')

        metrics = res.get("attention_metrics", [])
        if not metrics:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">—</span>')

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
        """Render biased tokens as horizontal chips in the toolbar."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<span style="color:#64748b;font-size:10px;">No analysis yet</span>')

        token_labels = res["token_labels"]
        biased = [
            lbl for lbl in token_labels
            if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if not biased:
            return ui.HTML('<span style="color:#64748b;font-size:10px;">No bias detected</span>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        items = []
        for lbl in biased:
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            types = lbl.get("bias_types", [])
            scores = lbl.get("scores", {})
            # Use the primary category color (first type) or pink as fallback
            primary_color = cat_colors.get(types[0], "#ff5ca9") if types else "#ff5ca9"
            max_score = max((scores.get(t, 0) for t in types), default=0)
            
            # Build category abbreviations
            cat_abbrevs = "·".join(types) if types else ""
            
            items.append(
                f'<span class="bias-token-chip" '
                f'style="display:inline-flex;align-items:center;gap:3px;'
                f'font-size:9px;font-family:JetBrains Mono,monospace;'
                f'padding:1px 6px;border-radius:4px;flex-shrink:0;'
                f'background:{primary_color}20;border:1px solid {primary_color}50;'
                f'color:{primary_color};white-space:nowrap;cursor:default;" '
                f'title="{cat_abbrevs} ({max_score:.2f})">'
                f'{clean}'
                f'</span>'
            )

        return ui.HTML("".join(items))


__all__ = ["bias_server_handlers"]
