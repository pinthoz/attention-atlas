import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from shiny import ui, render, reactive
from shinywidgets import render_plotly

from .helpers import positional_encoding, array_to_base64_img
from .metrics import compute_all_attention_metrics
from .models import tokenizer, encoder_model, mlm_model


def server(input, output, session):
    running = reactive.value(False)
    cached_result = reactive.value(None)
    focused_token = reactive.value(None)

    def tokenize_with_segments(text: str):
        pattern = re.search(r"([.!?])\s+([A-Za-z])", text)
        if pattern:
            split_idx = pattern.end(1)
            sentence_a = text[:split_idx].strip()
            sentence_b = text[split_idx:].strip()
            if sentence_a and sentence_b:
                return tokenizer(sentence_a, sentence_b, return_tensors="pt")
        return tokenizer(text, return_tensors="pt")

    def heavy_compute(text):
        if not text:
            return None

        inputs = tokenize_with_segments(text)
        with torch.no_grad():
            outputs = encoder_model(**inputs)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        embeddings = outputs.last_hidden_state[0].cpu().numpy()
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        pos_enc = positional_encoding(len(tokens), embeddings.shape[1])

        return (tokens, embeddings, pos_enc, attentions, hidden_states, inputs)

    # === COMPUTE ALL ===
    @reactive.effect
    @reactive.event(input.generate_all)
    async def compute_all():
        running.set(True)
        try:
            await session.send_custom_message('start_loading', {})
        except Exception:
            pass

        await asyncio.sleep(0.2)  # Give UI time to update
        text = input.text_input().strip()

        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, heavy_compute, text)
            cached_result.set(result)

        except Exception:
            cached_result.set(None)
        finally:
            try:
                await session.send_custom_message('stop_loading', {})
            except Exception:
                pass
            running.set(False)

    # === Sentence preview ===
    @output
    @render.ui
    def preview_text():
        res = cached_result.get()
        if not res:
            t = input.text_input().strip()
            return ui.HTML(f'<div style="font-family:monospace;color:#6b7280;font-size:14px;">"{t}"</div>' if t else '<div style="color:#9ca3af;font-size:12px;">Type a sentence above and click Generate All.</div>')

        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        if attentions is None or len(attentions) == 0:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">No attention data available.</div>')

        # Calculate per-token attention metrics
        # Average attention across all layers and heads
        # Shape: (num_layers, num_heads, seq_len, seq_len)
        att_layers = [layer[0].cpu().numpy() for layer in attentions]

        # Average across layers and heads: (seq_len, seq_len)
        att_avg = np.mean(att_layers, axis=(0, 1))

        # Calculate attention received by each token (sum of incoming attention)
        # att_avg[i, j] = attention from token i to token j
        # Sum over rows (axis=0) to get how much attention each token receives
        attention_received = att_avg.sum(axis=0)

        # Normalize values for coloring (0-1 range)
        att_received_norm = (attention_received - attention_received.min()) / (attention_received.max() - attention_received.min() + 1e-10)

        # Build HTML with colored tokens
        token_html = []
        for i, (tok, att_recv, recv_norm) in enumerate(zip(tokens, attention_received, att_received_norm)):
            # Use attention received for background color (darker blue gradient)
            # Map normalized value from [0,1] to [0.2, 0.8] for better visibility
            opacity = 0.2 + (recv_norm * 0.6)
            bg_color = f"rgba(59, 130, 246, {opacity})"  # Blue-500 with variable opacity

            # Remove \n from tooltip - use HTML entity for line break
            tooltip = f"Token: {tok}&#10;Attention Received: {att_recv:.3f}"

            token_html.append(
                f'<span class="token-viz" style="background:{bg_color};" title="{tooltip}">{tok}</span>'
            )

        html = '<div class="token-viz-container">' + ''.join(token_html) + '</div>'

        # Add legend with updated colors
        legend_html = '''
        <div style="display:flex;gap:12px;margin-top:8px;font-size:9px;color:#6b7280;">
            <div style="display:flex;align-items:center;gap:4px;">
                <div style="width:10px;height:10px;background:rgba(59,130,246,0.8);border-radius:2px;"></div>
                <span>High Attention</span>
            </div>
            <div style="display:flex;align-items:center;gap:4px;">
                <div style="width:10px;height:10px;background:rgba(59,130,246,0.2);border-radius:2px;"></div>
                <span>Low Attention</span>
            </div>
        </div>
        '''

        return ui.HTML(html + legend_html)

    # === Metrics display ===
    @output
    @render.ui
    def metrics_display():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        
        if attentions is None or len(attentions) == 0:
            return ui.HTML("")
        
        att_layers = [layer[0].cpu().numpy() for layer in attentions]
        att_avg = np.mean(att_layers, axis=(0, 1))
        metrics_dict = compute_all_attention_metrics(att_avg)
        
        metrics = [
            ("Confidence (Max)", f"{metrics_dict['confidence_max']:.4f}", "", "Confidence Max", "'Global'", "'Avg'"),
            ("Confidence (Avg)", f"{metrics_dict['confidence_avg']:.4f}", "", "Confidence Avg", "'Global'", "'Avg'"),
            ("Focus (Entropy)", f"{metrics_dict['focus_entropy']:.2f}", "", "Focus", "'Global'", "'Avg'"),
            ("Sparsity", f"{metrics_dict['sparsity']:.2%}", "", "Sparsity", "'Global'", "'Avg'"),
            ("Distribution (Median)", f"{metrics_dict['distribution_median']:.4f}", "", "Distribution", "'Global'", "'Avg'"),
            ("Uniformity", f"{metrics_dict['uniformity']:.4f}", "", "Uniformity", "'Global'", "'Avg'"),
        ]
        
        gradients = [
            "#fdf5f8",
            "#fef7fa",
            "#fdf6f9",
            "#fef8fb",
            "#fcf5f7",
            "#fef6f9",
        ]
        
        cards_html = '<div class="metrics-grid">'
        for idx, (label, value, icon, metric_name, layer, head) in enumerate(metrics):
            gradient = gradients[idx % len(gradients)]
            cards_html += f'''
                <div class="metric-card" onclick="showMetricModal('{metric_name}', {layer}, {head})" 
                     style="background: {gradient};">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
            '''
        cards_html += '</div>'
        return ui.HTML(cards_html)

    # === Embedding table ===
    @output
    @render.ui
    def embedding_table():
        res = cached_result.get()
        if not res:
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>No data yet</p>")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        rows = []
        for i, tok in enumerate(tokens):
            vec = embeddings[i]
            strip = array_to_base64_img(vec[:64], cmap="Blues", height=0.18)
            tip = "Embedding (first 32 dims): " + ", ".join(f"{v:.3f}" for v in vec[:32])
            rows.append(
                f"<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
                f"</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Embedding Vector</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Positional encodings ===
    @output
    @render.ui
    def posenc_table():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        rows = []
        for i, tok in enumerate(tokens):
            pe = pos_enc[i]
            strip = array_to_base64_img(pe[:64], cmap="RdBu", height=0.18)
            tip = f"Position {i} encoding: " + ", ".join(f"{v:.3f}" for v in pe[:32])
            rows.append(
                f"<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
                f"</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Position Encoding</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Q/K/V table ===
    @output
    @render.ui
    def qkv_table():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        layer_idx = int(input.qkv_layer())
        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]

        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()
            V = layer.value(hs_in)[0].cpu().numpy()

        rows = []
        for i, tok in enumerate(tokens):
            q_strip = array_to_base64_img(Q[i][:48], "Greens", 0.12)
            k_strip = array_to_base64_img(K[i][:48], "Oranges", 0.12)
            v_strip = array_to_base64_img(V[i][:48], "Purples", 0.12)
            q_tip = "Query: " + ", ".join(f"{x:.3f}" for x in Q[i][:24])
            k_tip = "Key: " + ", ".join(f"{x:.3f}" for x in K[i][:24])
            v_tip = "Value: " + ", ".join(f"{x:.3f}" for x in V[i][:24])
            cell = (
                f"<div style='font-size:9px;color:#065f46;font-weight:600;'>Q</div>"
                f"<img class='heatmap' src='data:image/png;base64,{q_strip}' title='{q_tip}'>"
                f"<div style='font-size:9px;color:#92400e;margin-top:2px;font-weight:600;'>K</div>"
                f"<img class='heatmap' src='data:image/png;base64,{k_strip}' title='{k_tip}'>"
                f"<div style='font-size:9px;color:#4c1d95;margin-top:2px;font-weight:600;'>V</div>"
                f"<img class='heatmap' src='data:image/png;base64,{v_strip}' title='{v_tip}'>"
            )
            rows.append(f"<tr><td class='token-name'>{tok}</td><td>{cell}</td></tr>")

        html = (
            "<div class='card-scroll'>"
            "<table class='token-table-compact'><tr><th>Token</th><th>Vectors</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Segment embeddings ===
    @output
    @render.ui
    def segment_embedding_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        segment_ids = inputs.get("token_type_ids")
        if segment_ids is None:
            return ui.HTML("<p style='font-size:10px;color:#6b7280;'>No segment information available.</p>")
        ids = segment_ids[0].cpu().numpy().tolist()
        rows = []
        counts = {}
        colors = ["#6366f1", "#a855f7", "#ec4899", "#f97316"]

        for tok, seg in zip(tokens, ids):
            counts[seg] = counts.get(seg, 0) + 1
            color = colors[seg % len(colors)]
            badge = f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{color};color:white;font-size:9px;font-weight:600;'>Seg {seg}</span>"
            rows.append(
                f"<tr><td class='token-name'>{tok}</td><td>{badge}</td></tr>"
            )

        summary_badges = []
        for k, v in sorted(counts.items()):
            color = colors[k % len(colors)]
            summary_badges.append(
                f"<span style='display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:999px;background:{color}15;border:1px solid {color}40;font-size:10px;font-weight:600;color:{color};'>"
                f"<span style='width:8px;height:8px;border-radius:50%;background:{color};'></span>"
                f"Segment {k}: <strong>{v} tokens</strong></span>"
            )

        html = (
            "<div class='card-scroll'>"
            "<table class='token-table-segment'>"
            "<tr><th>Token</th><th>Segment</th></tr>"
            + "".join(rows)
            + "</table></div>"
            "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;padding-top:12px;border-top:1px solid #e5e7eb;'>"
            + "".join(summary_badges)
            + "</div>"
        )
        return ui.HTML(html)

    # === Sum + LayerNorm ===
    @output
    @render.ui
    def sum_layernorm_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        input_ids = inputs["input_ids"]
        segment_ids = inputs.get("token_type_ids")
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        seq_len = input_ids.shape[1]
        device = input_ids.device
        with torch.no_grad():
            word_embed = encoder_model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embed = encoder_model.embeddings.position_embeddings(position_ids)
            seg_embed = encoder_model.embeddings.token_type_embeddings(segment_ids)
            summed = word_embed + pos_embed + seg_embed
            normalized = encoder_model.embeddings.LayerNorm(summed)
        summed_np = summed[0].cpu().numpy()
        norm_np = normalized[0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            sum_strip = array_to_base64_img(summed_np[i][:96], "cividis", 0.15)
            norm_strip = array_to_base64_img(norm_np[i][:96], "viridis", 0.15)
            rows.append(
                "<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{sum_strip}' title='Sum of embeddings'></td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{norm_strip}' title='LayerNorm output'></td>"
                "</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Sum</th><th>LayerNorm</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Attention map (Plotly) ===
    @render_plotly
    def attention_map():
        res = cached_result.get()
        if not res:
            return None
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        if attentions is None or len(attentions) == 0:
            return None

        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())

        att = attentions[layer_idx][0, head_idx].cpu().numpy()

        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]
        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()

        L = len(tokens)
        d_k = Q.shape[-1] // 12
        custom = np.empty((L, L, 5), dtype=object)
        for i in range(L):
            for j in range(L):
                dot_product = np.dot(Q[i], K[j])
                scaled = dot_product / np.sqrt(d_k)
                custom[i, j, 0] = np.array2string(Q[i][:5], precision=3, separator=", ")
                custom[i, j, 1] = np.array2string(K[j][:5], precision=3, separator=", ")
                custom[i, j, 2] = f"{dot_product:.4f}"
                custom[i, j, 3] = f"{scaled:.4f}"
                custom[i, j, 4] = f"{att[i, j]:.4f}"

        hover = (
            "<b>Query Token:</b> %{y}<br>"
            "<b>Key Token:</b> %{x}<br>"
            "<br><b>Calculation:</b><br>"
            "1. Dot Product: Q·K = %{customdata[2]}<br>"
            "2. Scaled: (Q·K)/√d_k = %{customdata[3]}<br>"
            "3. Softmax Result: <b>%{customdata[4]}</b><br>"
            "<br><b>Vectors (first 5 dims):</b><br>"
            "Q = %{customdata[0]}<br>"
            "K = %{customdata[1]}<br>"
            "<extra></extra>"
        )

        fig = px.imshow(
            att,
            x=tokens,
            y=tokens,
            color_continuous_scale="Blues",
            aspect="auto",
            title="Attention Map",
        )
        fig.update_traces(customdata=custom, hovertemplate=hover)
        fig.update_layout(
            xaxis_title="Key (attending to)",
            yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40),
            coloraxis_colorbar=dict(title="Attention"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#111827"),
        )
        return fig

    @output
    @render.ui
    def scaled_attention_selector():
        res = cached_result.get()
        if not res:
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>No data yet</p>")
        tokens, *_ = res
        options = {str(i): f"{i}: {tok}" for i, tok in enumerate(tokens)}
        try:
            selected = input.scaled_attention_token()
        except Exception:
            selected = "0"
        if selected not in options:
            selected = "0"
        return ui.div(
            ui.input_select(
                "scaled_attention_token",
                "Focus token",
                choices=options,
                selected=selected,
            ),
            class_="focus-select",
        )

    @output
    @render.ui
    def scaled_attention_selector_inline():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, *_ = res
        options = {str(i): str(i) for i in range(len(tokens))}
        try:
            selected = input.scaled_attention_token()
        except Exception:
            selected = "0"
        if selected not in options:
            selected = "0"

        return ui.HTML(f"""
            <div class="select-mini">
                <select id="scaled_attention_token_mini" onchange="Shiny.setInputValue('scaled_attention_token', this.value)">
                    {''.join(f'<option value="{val}" {"selected" if val == selected else ""}>{label}</option>' for val, label in options.items())}
                </select>
            </div>
        """)

    # === Scaled attention formula ===
    @output
    @render.ui
    def scaled_attention_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        if attentions is None or len(attentions) == 0:
            return ui.HTML("")
        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        try:
            focus_idx = int(input.scaled_attention_token())
        except Exception:
            focus_idx = 0
        focus_idx = max(0, min(focus_idx, len(tokens) - 1))
        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]
        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()
        d_k = Q.shape[-1] // layer.num_attention_heads if hasattr(layer, "num_attention_heads") else Q.shape[-1]
        top_idx = np.argsort(att[focus_idx])[::-1][:3]
        rows = ""
        for j in top_idx:
            dot = float(np.dot(Q[focus_idx], K[j]))
            scaled = dot / np.sqrt(d_k)
            prob = att[focus_idx, j]
            rows += (
                "<div class='scaled-attention-row'>"
                f"<span>{tokens[focus_idx]} → {tokens[j]}</span>"
                f"<span>dot={dot:.2f}, scaled={scaled:.2f}, softmax={prob:.3f}</span>"
                "</div>"
            )
        html = (
            "<div class='scaled-attention-box'>"
            f"<div style='font-size:11px;margin-bottom:6px;'>Focus token: <b>{tokens[focus_idx]}</b></div>"
            "<div style='font-size:10px;color:#6b7280;margin-bottom:6px;'>softmax(Q·K<sup>T</sup>/√d<sub>k</sub>) for top connections:</div>"
            + rows
            + "</div>"
        )
        return ui.HTML(html)

    # === Reset focus ===
    @reactive.effect
    @reactive.event(input.generate_all)
    def reset_focus():
        focused_token.set(None)

    # === Attention flow ===
    @render_plotly
    def attention_flow():
        res = cached_result.get()
        if not res:
            return None
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())
        focus_idx = focused_token()

        if attentions is None or len(attentions) == 0:
            return None

        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        n_tokens = len(tokens)
        
        color_palette = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa07a', '#98d8c8',
            '#f7dc6f', '#bb8fce', '#85c1e2', '#f8b739', '#52be80',
            '#ec7063', '#af7ac5', '#5dade2', '#f39c12', '#27ae60'
        ]
        
        fig = go.Figure()
        block_width = 0.8 / n_tokens
        for i, tok in enumerate(tokens):
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2
            show_focus = focus_idx is not None
            is_selected = focus_idx == i if show_focus else True
            font_size = 13 if is_selected else 10
            text_color = color if (show_focus and is_selected) else "#111827"
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[1.05], mode='text', text=tok,
                textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[-0.05], mode='text', text=tok,
                textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'),
                showlegend=False, hoverinfo='skip'
            ))
        
        threshold = 0.04
        for i in range(n_tokens):
            for j in range(n_tokens):
                weight = att[i, j]
                if weight > threshold:
                    is_line_focused = (focus_idx is None) or (i == focus_idx)
                    x_source = i / n_tokens + block_width / 2
                    x_target = j / n_tokens + block_width / 2
                    x_vals = [x_source, (x_source + x_target) / 2, x_target]
                    y_vals = [1, 0.5, 0]
                    if is_line_focused:
                        line_color = color_palette[i % len(color_palette)]
                        line_opacity = min(0.95, weight * 3)
                        line_width = max(2, weight * 15)
                    else:
                        line_color = '#2a2a2a'
                        line_opacity = 0.003
                        line_width = 0.1
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines',
                        line=dict(color=line_color, width=line_width),
                        opacity=line_opacity, showlegend=False,
                        hoverinfo='text' if is_line_focused else 'skip',
                        hovertext=f"<b>{tokens[i]} to {tokens[j]}</b><br>Attention: {weight:.4f}",
                    ))
        
        title_text = "Attention Flow"
        if focus_idx is not None:
            focus_color = color_palette[focus_idx % len(color_palette)]
            title_text += f" · <b style='color:{focus_color}'>Focused: '{tokens[focus_idx]}'</b>"
        
        fig.update_layout(
            title=title_text,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.05, 1.05]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.25, 1.25]),
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#111827'),
            height=500, margin=dict(l=20, r=20, t=60, b=40),
            clickmode='event+select', hovermode='closest', dragmode=False,
        )
        return fig

    # === Token buttons ===
    @output
    @render.ui
    def token_selector_buttons():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, _, _, _, _, _ = res
        focus_idx = focused_token()
        color_palette = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa07a', '#98d8c8',
                         '#f7dc6f', '#bb8fce', '#85c1e2', '#f8b739', '#52be80',
                         '#ec7063', '#af7ac5', '#5dade2', '#f39c12', '#27ae60']
        buttons = []
        for i, tok in enumerate(tokens):
            color = color_palette[i % len(color_palette)]
            active_class = "active" if focus_idx == i else ""
            buttons.append(
                ui.input_action_button(
                    f"token_btn_{i}", tok,
                    class_=f"token-btn {active_class}",
                    style=f"background:{color};color:white;"
                )
            )
        buttons.append(
            ui.input_action_button("token_btn_reset", "Show All", class_="token-btn-reset")
        )
        return ui.div(*buttons)

    # === Handle token clicks ===
    @reactive.effect
    def handle_token_buttons():
        res = cached_result.get()
        if not res:
            return
        tokens, _, _, _, _, _ = res
        for i in range(len(tokens)):
            @reactive.effect
            @reactive.event(input[f"token_btn_{i}"])
            def _handle_click(idx=i):
                current = focused_token()
                focused_token.set(None if current == idx else idx)
        @reactive.effect
        @reactive.event(input.token_btn_reset)
        def _handle_reset():
            focused_token.set(None)

    # === Add & Norm ===
    @output
    @render.ui
    def add_norm_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())
        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")
        hs_in = hidden_states[layer_idx][0].cpu().numpy()
        hs_out = hidden_states[layer_idx + 1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            diff = np.linalg.norm(hs_out[i] - hs_in[i])
            norm = np.linalg.norm(hs_in[i]) + 1e-6
            ratio = diff / norm
            width = max(4, min(100, int(ratio * 80)))
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;'>"
                f"<div style='width:{width}%;height:10px;border-radius:999px;"
                f"background:linear-gradient(90deg,#22c55e,#22d3ee);'></div></div></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Change Magnitude</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === FFN ===
    @output
    @render.ui
    def ffn_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())
        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")
        layer = encoder_model.encoder.layer[layer_idx]
        hs_in = hidden_states[layer_idx][0]
        with torch.no_grad():
            inter = layer.intermediate.dense(hs_in)
            inter_act = layer.intermediate.intermediate_act_fn(inter)
            proj = layer.output.dense(inter_act)
        inter_np = inter_act.cpu().numpy()
        proj_np = proj.cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            inter_strip = array_to_base64_img(inter_np[i][:96], "plasma", 0.15)
            proj_strip = array_to_base64_img(proj_np[i][:96], "magma", 0.15)
            rows.append(
                "<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{inter_strip}' title='Intermediate 3072 dims'></td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{proj_strip}' title='Projection back to 768 dims'></td>"
                "</tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>GELU Activation</th><th>Projection</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Add & Norm after FFN ===
    @output
    @render.ui
    def add_norm_post_ffn_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())
        if layer_idx + 2 >= len(hidden_states):
            return ui.HTML("<p style='font-size:10px;color:#6b7280;'>Select a lower layer to inspect residual output.</p>")
        hs_mid = hidden_states[layer_idx + 1][0].cpu().numpy()
        hs_out = hidden_states[layer_idx + 2][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            diff = np.linalg.norm(hs_out[i] - hs_mid[i])
            norm = np.linalg.norm(hs_mid[i]) + 1e-6
            ratio = diff / norm
            width = max(4, min(100, int(ratio * 80)))
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;'>"
                f"<div style='width:{width}%;height:10px;border-radius:999px;"
                f"background:linear-gradient(90deg,#14b8a6,#0ea5e9);'></div></div></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Residual Change (FFN)</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Layer stats ===
    @output
    @render.ui
    def layer_output_view():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())
        if layer_idx + 1 >= len(hidden_states):
            return ui.HTML("")
        hs = hidden_states[layer_idx + 1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            mean_val = float(hs[i].mean())
            std_val = float(hs[i].std())
            max_val = float(hs[i].max())
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td style='font-size:9px;color:#374151;'>"
                f"μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}</td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Statistics</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === Final hidden state ===
    @output
    @render.ui
    def linear_projection_viz():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        final_h = hidden_states[-1][0].cpu().numpy()

        rows = []
        for i, tok in enumerate(tokens):
            hidden_strip = array_to_base64_img(final_h[i][:96], "magma", 0.16)
            rows.append(
                "<tr>"
                f"<td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{hidden_strip}' title='Hidden state (768 dims)'></td>"
                "</tr>"
            )

        html = (
            f"<div class='projection-card'>"
            f"<div class='sub-label'>Hidden representations for {len(tokens)} tokens</div>"
            "<div class='card-scroll' style='max-height:360px;'>"
            "<table class='token-table'><tr><th>Token</th><th>Hidden Pattern</th></tr>"
            + "".join(rows)
            + "</table></div>"
            "<div style='font-size:10px;color:#6b7280;text-align:center;margin-top:10px;'>"
            "These hidden states flow into the vocabulary projection before Softmax."
            "</div></div>"
        )
        return ui.HTML(html)

    # === MLM Top-5 ===
    @output
    @render.ui
    def output_probabilities():
        res = cached_result.get()
        if not res:
            return ui.HTML("<div class='prediction-panel'><p style='font-size:11px;color:#9ca3af;'>No data yet</p></div>")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        if not input.use_mlm():
            return ui.HTML(
                "<div class='prediction-panel'>"
                "<h5>Predictions</h5>"
                "<p style='font-size:11px;color:#6b7280;'>Enable <b>Use MLM head for predictions</b> to render top-k token probabilities.</p>"
                "</div>"
            )

        text = input.text_input().strip()
        if not text:
            return ui.HTML(
                "<div class='prediction-panel'>"
                "<p style='font-size:11px;color:#9ca3af;'>Type a sentence to see predictions.</p>"
                "</div>"
            )
        mlm_inputs = tokenize_with_segments(text)
        with torch.no_grad():
            mlm_outputs = mlm_model(**mlm_inputs)
            probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]
        logits_tensor = mlm_outputs.logits[0]
        logsumexp_vals = torch.logsumexp(logits_tensor, dim=-1)

        mlm_tokens = tokenizer.convert_ids_to_tokens(mlm_inputs["input_ids"][0])
        rows = []
        top_k = 5
        for i, tok in enumerate(mlm_tokens):
            token_probs = probs[i]
            top_vals, top_idx = torch.topk(token_probs, top_k)
            inner = "<div style='font-size:10px;'>"
            for rank, (p, idx) in enumerate(zip(top_vals, top_idx)):
                ptok = tokenizer.decode([idx.item()]) or "[UNK]"
                pval = float(p)
                width = max(4, int(pval * 100))
                logit_val = float(logits_tensor[i, idx])
                lse_val = float(logsumexp_vals[i])
                detail_id = f"prob_detail_{i}_{rank}"
                detail_text = (
                    f"<div id='{detail_id}' class='prob-detail-box'>"
                    f"<div><b>Softmax</b> = exp({logit_val:.2f} - {lse_val:.2f}) = {pval:.4%}</div>"
                    "<div style='color:#6b7280;'>exp(logit) / Σ exp(logits)</div>"
                    "</div>"
                )
                inner += (
                    "<div style='margin:3px 0;'>"
                    "<div class='prob-line'>"
                    f"<span style='font-family:monospace;font-size:10px;min-width:70px;font-weight:600;'>{ptok}</span>"
                    f"<div style='flex:1;background:#e5e7eb;border-radius:999px;height:12px;'>"
                    f"<div style='width:{width}%;height:12px;border-radius:999px;"
                    f"background:linear-gradient(90deg,#3b82f6,#8b5cf6);'></div></div>"
                    f"<button type='button' class='prob-detail-btn' onclick=\"toggleProbDetail('{detail_id}')\">{pval:.1%}</button>"
                    "</div>"
                    f"{detail_text}"
                    "</div>"
                )
            inner += "</div>"
            rows.append(f"<tr><td class='token-name'>{tok}</td><td>{inner}</td></tr>")

        html = (
            "<div class='prediction-panel'>"
            "<h5>MLM Top-k Predictions</h5>"
            "<div class='card-scroll' style='max-height:340px;'>"
            "<table class='token-table'><tr><th>Token</th><th>Top-5 Probabilities</th></tr>"
            + "".join(rows)
            + "</table></div></div>"
        )
        return ui.HTML(html)
