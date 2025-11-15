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

    def heavy_compute(text):
        if not text:
            return None

        inputs = tokenizer(text, return_tensors="pt")
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
    @render.text
    def preview_text():
        t = input.text_input().strip()
        return f'"{t}"' if t else "Type a sentence and click Generate All."

    # === Metrics display ===
    @output
    @render.ui
    def metrics_display():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res
        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())
        
        if attentions is None or len(attentions) == 0:
            return ui.HTML("")
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        metrics_dict = compute_all_attention_metrics(att)
        
        metrics = [
            ("Confidence (Max)", f"{metrics_dict['confidence_max']:.4f}", "", "Confidence Max", layer_idx, head_idx),
            ("Confidence (Avg)", f"{metrics_dict['confidence_avg']:.4f}", "", "Confidence Avg", layer_idx, head_idx),
            ("Focus (Entropy)", f"{metrics_dict['focus_entropy']:.2f}", "", "Focus", layer_idx, head_idx),
            ("Sparsity", f"{metrics_dict['sparsity']:.2%}", "", "Sparsity", layer_idx, head_idx),
            ("Distribution (Median)", f"{metrics_dict['distribution_median']:.4f}", "", "Distribution", layer_idx, head_idx),
            ("Uniformity", f"{metrics_dict['uniformity']:.4f}", "", "Uniformity", layer_idx, head_idx),
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
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>Click <b>Generate All</b> to start</p>")
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

    # === Attention map (Plotly) ===
    @render_plotly
    def attention_map():
        res = cached_result.get()
        if not res:
            return None
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        layer_idx = int(input.att_layer())
        head_idx = int(input.att_head())

        if attentions is None or len(attentions) == 0:
            return None

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
            title=f"Layer {layer_idx} · Head {head_idx}",
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
            is_focused = focus_idx is None or focus_idx == i
            font_size = 12 if is_focused else 9
            opacity = 1.0 if is_focused else 0.3
            fig.add_trace(go.Scatter(x=[x_pos], y=[1.05], mode='text', text=tok,
                                    textfont=dict(size=font_size, color=color, family='monospace', weight='bold'),
                                    opacity=opacity, showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.05], mode='text', text=tok,
                                    textfont=dict(size=font_size, color=color, family='monospace', weight='bold'),
                                    opacity=opacity, showlegend=False, hoverinfo='skip'))
        
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
        
        title_text = f"Layer {layer_idx} · Head {head_idx}"
        if focus_idx is not None:
            title_text += f" · <b style='color:#ff5ca9'>Focused: '{tokens[focus_idx]}'</b>"
        
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
        return ui.div({"style": "display:flex;flex-wrap:wrap;gap:4px;align-items:center;"}, *buttons)

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
        hs = hidden_states[layer_idx + 1][0].cpu().numpy()
        rows = []
        for i, tok in enumerate(tokens):
            strip = array_to_base64_img(hs[i][:96], "plasma", 0.18)
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' "
                f"title='Layer {layer_idx} output pattern'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Activation Pattern</th></tr>"
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
            strip = array_to_base64_img(final_h[i][:64], "magma", 0.18)
            tip = "Final encoder hidden state (768 dims)"
            rows.append(
                f"<tr><td class='token-name'>{tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td></tr>"
            )
        html = (
            "<div class='card-scroll'>"
            "<table class='token-table'><tr><th>Token</th><th>Hidden State</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)

    # === MLM Top-5 ===
    @output
    @render.ui
    def output_probabilities():
        res = cached_result.get()
        if not res:
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>Run <b>Generate All</b> first</p>")
        tokens, embeddings, pos_enc, attentions, hidden_states, inputs = res

        if not input.use_mlm():
            return ui.HTML("<p style='font-size:10px;color:#9ca3af;'>Enable <b>Use MLM head</b> to see token probabilities.</p>")

        text = input.text_input().strip()
        if not text:
            return ui.HTML("")
        mlm_inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            mlm_outputs = mlm_model(**mlm_inputs)
            probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]

        mlm_tokens = tokenizer.convert_ids_to_tokens(mlm_inputs["input_ids"][0])
        rows = []
        top_k = 5
        for i, tok in enumerate(mlm_tokens):
            token_probs = probs[i]
            top_vals, top_idx = torch.topk(token_probs, top_k)
            inner = "<div style='font-size:10px;'>"
            for p, idx in zip(top_vals, top_idx):
                ptok = tokenizer.decode([idx.item()]) or "[UNK]"
                pval = float(p)
                width = max(4, int(pval * 100))
                inner += (
                    "<div style='margin:3px 0;'>"
                    "<div style='display:flex;align-items:center;gap:6px;'>"
                    f"<span style='font-family:monospace;font-size:10px;min-width:70px;font-weight:600;'>{ptok}</span>"
                    f"<div style='flex:1;background:#e5e7eb;border-radius:999px;height:12px;'>"
                    f"<div style='width:{width}%;height:12px;border-radius:999px;"
                    f"background:linear-gradient(90deg,#3b82f6,#8b5cf6);'></div></div>"
                    f"<span style='font-size:10px;color:#6b7280;width:50px;text-align:right;font-weight:500;'>{pval:.1%}</span>"
                    "</div></div>"
                )
            inner += "</div>"
            rows.append(f"<tr><td class='token-name'>{tok}</td><td>{inner}</td></tr>")

        html = (
            "<div class='card-scroll' style='max-height:350px;'>"
            "<table class='token-table'><tr><th>Position</th><th>Top-5 Predictions</th></tr>"
            + "".join(rows)
            + "</table></div>"
        )
        return ui.HTML(html)