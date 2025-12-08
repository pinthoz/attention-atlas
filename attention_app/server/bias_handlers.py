"""Server-side handlers for bias analysis functionality."""

import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from shiny import ui, render, reactive

from ..models import ModelManager
from ..bias import (
    TokenBiasDetector,
    AttentionBiasAnalyzer,
    create_token_bias_heatmap,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization
)


def bias_server_handlers(input, output, session):
    """Create server handlers for bias analysis tab.

    Args:
        input: Shiny input object
        output: Shiny output object
        session: Shiny session object
    """
    # Reactive values for bias analysis
    bias_running = reactive.value(False)
    bias_results = reactive.value(None)

    def heavy_bias_compute(text, model_name):
        """Perform bias analysis computation (runs in thread pool)."""
        print("DEBUG: Starting heavy_bias_compute")
        if not text:
            return None

        try:
            # Load model
            tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
            device = ModelManager.get_device()
            print("DEBUG: Model loaded")

            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model outputs
            with torch.no_grad():
                outputs = encoder_model(**inputs)

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attentions = outputs.attentions
            print(f"DEBUG: Got {len(tokens)} tokens and {len(attentions)} attention layers")

            # Initialize analyzers
            # Initialize analyzers
            token_detector = TokenBiasDetector()
            attention_analyzer = AttentionBiasAnalyzer()

            # 1. Token-level bias detection
            print("DEBUG: Detecting token-level bias...")
            token_labels = token_detector.detect_bias(text, tokens)
            bias_summary = token_detector.get_bias_summary(token_labels)
            bias_spans = token_detector.get_biased_spans(token_labels)
            print(f"DEBUG: Found {len(bias_spans)} bias spans")

            # 3. Attention × Bias analysis
            print("DEBUG: Analyzing attention-bias interaction...")
            biased_indices = [i for i, label in enumerate(token_labels) if label["is_biased"]]

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
                print(f"DEBUG: Analyzed {len(attention_metrics)} attention heads")
            else:
                attention_metrics = []
                propagation_analysis = {"layer_propagation": [], "peak_layer": None, "propagation_pattern": "none"}
                bias_matrix = np.array([])

            print("DEBUG: Bias analysis complete")
            return {
                "tokens": tokens,
                "text": text,
                "attentions": attentions,
                "tokenizer": tokenizer,
                "encoder_model": encoder_model,
                "tokenizer": tokenizer,
                "encoder_model": encoder_model,
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

    @reactive.effect
    @reactive.event(input.analyze_bias_btn)
    async def compute_bias():
        """Trigger bias analysis computation."""
        print("DEBUG: compute_bias triggered")
        text = input.bias_input_text().strip()
        if not text:
            print("DEBUG: No text input")
            return

        bias_running.set(True)
        await session.send_custom_message('start_bias_loading', {})
        await asyncio.sleep(0.1)

        # Use the same model as attention analysis or default to BERT
        try:
            model_name = input.model_name()
        except:
            model_name = "bert-base-uncased"

        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                print("DEBUG: Starting bias computation in executor")
                result = await loop.run_in_executor(pool, heavy_bias_compute, text, model_name)
                print("DEBUG: Bias computation returned")
            bias_results.set(result)
        except Exception as e:
            print(f"ERROR in compute_bias: {e}")
            traceback.print_exc()
            bias_results.set(None)
        finally:
            bias_running.set(False)
            await session.send_custom_message('stop_bias_loading', {})

    @output
    @render.ui
    def bias_summary():
        """Render bias detection summary."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">Enter text and click "Analyze Bias" to begin.</div>')

        summary = res["bias_summary"]
        biased_pct = summary["bias_percentage"]

        # Color code based on bias level
        if biased_pct < 10:
            color = "#10b981"  # Green
            level = "Low"
        elif biased_pct < 30:
            color = "#f59e0b"  # Orange
            level = "Moderate"
        else:
            color = "#ef4444"  # Red
            level = "High"

        html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 12px;">
            <div class="metric-card" style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); border-color: {color};">
                <div class="metric-label">Bias Level</div>
                <div class="metric-value" style="color: {color};">{level}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Biased Tokens</div>
                <div class="metric-value">{summary['biased_tokens']} / {summary['total_tokens']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bias Percentage</div>
                <div class="metric-value">{biased_pct:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Generalizations</div>
                <div class="metric-value">{summary['generalization_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Unfair Language</div>
                <div class="metric-value">{summary['unfairness_count']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Stereotypes</div>
                <div class="metric-value">{summary['stereotype_count']}</div>
            </div>
        </div>
        """
        return ui.HTML(html)

    @output
    @render.ui
    def token_bias_viz():
        """Render token-level bias heatmap."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        try:
            fig = create_token_bias_heatmap(res["token_labels"], res["text"])
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}))
        except Exception as e:
            print(f"Error creating token bias viz: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {str(e)}</div>')

    @output
    @render.ui
    def bias_spans_table():
        """Render table of detected bias spans."""
        res = bias_results.get()
        if not res:
            return None

        spans = res["bias_spans"]
        if not spans:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;padding:12px;">No bias spans detected.</div>')

        rows = []
        for i, span in enumerate(spans):
            tokens_text = " ".join(span.tokens)
            bias_badges = " ".join([
                f'<span style="background:#ff5ca9;color:white;padding:2px 6px;border-radius:4px;font-size:10px;margin-right:4px;">{bt}</span>'
                for bt in span.bias_types
            ])
            rows.append(f"""
                <tr>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;">{i+1}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;font-family:monospace;color:#ff5ca9;">{tokens_text}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;">{bias_badges}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;font-size:11px;color:#64748b;">{span.explanation}</td>
                </tr>
            """)

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f8fafc;">
                    <th style="padding:12px;text-align:left;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">#</th>
                    <th style="padding:12px;text-align:left;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Span</th>
                    <th style="padding:12px;text-align:left;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Types</th>
                    <th style="padding:12px;text-align:left;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Explanation</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """
        return ui.HTML(table_html)



    @output
    @render.ui
    def attention_bias_matrix():
        """Render attention×bias matrix."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        bias_matrix = res["bias_matrix"]
        if bias_matrix.size == 0:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;">No biased tokens detected.</div>')

        try:
            fig = create_attention_bias_matrix(bias_matrix, res["attention_metrics"])
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}))
        except Exception as e:
            print(f"Error creating attention bias matrix: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {str(e)}</div>')

    @output
    @render.ui
    def bias_propagation_plot():
        """Render bias propagation across layers."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        propagation = res["propagation_analysis"]["layer_propagation"]
        if not propagation:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;">No propagation data available.</div>')

        try:
            fig = create_bias_propagation_plot(propagation)
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}))
        except Exception as e:
            print(f"Error creating propagation plot: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {str(e)}</div>')

    @output
    @render.ui
    def combined_bias_view():
        """Render combined attention & bias visualization."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        try:
            layer_idx = int(input.bias_attn_layer())
            head_idx = int(input.bias_attn_head())
        except:
            layer_idx = 0
            head_idx = 0

        attentions = res["attentions"]
        if not attentions or layer_idx >= len(attentions):
            return ui.HTML('<div style="color:#9ca3af;">No attention data available.</div>')

        try:
            attention_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
            fig = create_combined_bias_visualization(
                res["tokens"],
                res["token_labels"],
                attention_matrix,
                layer_idx,
                head_idx
            )
            return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False}))
        except Exception as e:
            print(f"Error creating combined view: {e}")
            traceback.print_exc()
            return ui.HTML(f'<div style="color:#ef4444;">Error: {str(e)}</div>')

    @output
    @render.ui
    def bias_focused_heads_table():
        """Render table of bias-focused attention heads."""
        res = bias_results.get()
        if not res:
            return None

        metrics = res["attention_metrics"]
        focused = [m for m in metrics if m.specialized_for_bias]

        if not focused:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;padding:12px;">No heads with significant bias focus detected.</div>')

        # Sort by bias attention ratio
        focused.sort(key=lambda x: x.bias_attention_ratio, reverse=True)

        rows = []
        for m in focused:
            rows.append(f"""
                <tr>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:center;">Layer {m.layer}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:center;">Head {m.head}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;font-family:monospace;font-weight:600;color:#ff5ca9;">{m.bias_attention_ratio:.3f}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;font-family:monospace;">{m.amplification_score:.3f}</td>
                    <td style="padding:8px;border-bottom:1px solid #e2e8f0;text-align:right;font-family:monospace;">{m.max_bias_attention:.3f}</td>
                </tr>
            """)

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#f8fafc;">
                    <th style="padding:12px;text-align:center;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Layer</th>
                    <th style="padding:12px;text-align:center;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Head</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Bias Attention Ratio</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Amplification</th>
                    <th style="padding:12px;text-align:right;font-size:11px;color:#64748b;border-bottom:2px solid #e2e8f0;">Max Attention</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        <div style="margin-top:12px;font-size:11px;color:#64748b;">
            <strong>Note:</strong> Heads shown have bias attention ratio > 1.5 (focus significantly more on biased tokens than average).
        </div>
        """
        return ui.HTML(table_html)

    # Update layer/head selectors when results change
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


__all__ = ["bias_server_handlers"]
