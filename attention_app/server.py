import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

from shiny import ui, render, reactive
from shinywidgets import render_plotly, output_widget, render_widget

from .helpers import positional_encoding, array_to_base64_img, compute_influence_tree
from .metrics import compute_all_attention_metrics
from .models import ModelManager
from .head_specialization import compute_all_heads_specialization
from .isa import compute_isa

# HELPER FUNCTIONS FOR HTML GENERATION 

def get_embedding_table(res):
    tokens, embeddings, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>Embedding Vector</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_segment_embedding_view(res):
    tokens, _, _, _, _, inputs, *_ = res
    segment_ids = inputs.get("token_type_ids")
    if segment_ids is None:
        return ui.HTML("<p style='font-size:10px;color:#6b7280;'>No segment information available.</p>")
    ids = segment_ids[0].cpu().numpy().tolist()
    
    rows = ""
    for i, (tok, seg) in enumerate(zip(tokens, ids)):
        row_class = f"seg-row-{seg}" if seg in [0, 1] else ""
        rows += f"""
        <tr class='{row_class}'>
            <td class='token-cell'>{tok}</td>
            <td class='segment-cell'>{seg}</td>
        </tr>
        """
    
    return ui.HTML(
        f"""
        <div class='card-scroll'>
            <table class='segment-table-clean'>
                <thead>
                    <tr>
                        <th>Token</th>
                        <th>Segment</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    )

def get_posenc_table(res):
    tokens, _, pos_enc, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>Position Encoding</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_sum_layernorm_view(res, encoder_model):
    tokens, _, _, _, _, inputs, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>Sum</th><th>LayerNorm</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_qkv_table(res, layer_idx):
    tokens, _, _, _, hidden_states, _, _, encoder_model, *_ = res
    layer = encoder_model.encoder.layer[layer_idx].attention.self
    hs_in = hidden_states[layer_idx]

    with torch.no_grad():
        Q = layer.query(hs_in)[0].cpu().numpy()
        K = layer.key(hs_in)[0].cpu().numpy()
        V = layer.value(hs_in)[0].cpu().numpy()

    cards = []
    for i, tok in enumerate(tokens):
        # Clean token for display
        display_tok = tok.replace("##", "") if tok.startswith("##") else tok
        
        q_strip = array_to_base64_img(Q[i][:48], "Greens", 0.12)
        k_strip = array_to_base64_img(K[i][:48], "Oranges", 0.12)
        v_strip = array_to_base64_img(V[i][:48], "Purples", 0.12)
        q_tip = "Query: " + ", ".join(f"{x:.3f}" for x in Q[i][:24])
        k_tip = "Key: " + ", ".join(f"{x:.3f}" for x in K[i][:24])
        v_tip = "Value: " + ", ".join(f"{x:.3f}" for x in V[i][:24])

        card = f"""
        <div class='qkv-item'>
            <div class='qkv-token-header'>{display_tok}</div>
            <div class='qkv-row-item'>
                <span class='qkv-label'>Q</span>
                <img class='heatmap' src='data:image/png;base64,{q_strip}' title='{q_tip}'>
            </div>
            <div class='qkv-row-item'>
                <span class='qkv-label'>K</span>
                <img class='heatmap' src='data:image/png;base64,{k_strip}' title='{k_tip}'>
            </div>
            <div class='qkv-row-item'>
                <span class='qkv-label'>V</span>
                <img class='heatmap' src='data:image/png;base64,{v_strip}' title='{v_tip}'>
            </div>
        </div>
        """
        cards.append(card)

    return ui.HTML(
        "<div class='card-scroll'>"
        "<div class='qkv-container'>"
        + "".join(cards)
        + "</div></div>"
    )

def get_scaled_attention_view(res, layer_idx, head_idx, focus_idx):
    tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
    if attentions is None or len(attentions) == 0:
        return ui.HTML("")
    
    att = attentions[layer_idx][0, head_idx].cpu().numpy()
    focus_idx = max(0, min(focus_idx, len(tokens) - 1))
    layer = encoder_model.encoder.layer[layer_idx].attention.self
    hs_in = hidden_states[layer_idx]
    with torch.no_grad():
        Q = layer.query(hs_in)[0].cpu().numpy()
        K = layer.key(hs_in)[0].cpu().numpy()
    d_k = Q.shape[-1] // layer.num_attention_heads if hasattr(layer, "num_attention_heads") else Q.shape[-1]

    # Get top 3 connections
    top_idx = np.argsort(att[focus_idx])[::-1][:3]

    # Build computation display
    computations = ""
    for rank, j in enumerate(top_idx, 1):
        dot = float(np.dot(Q[focus_idx], K[j]))
        scaled = dot / np.sqrt(d_k)
        prob = att[focus_idx, j]

        computations += f"""
        <div class='scaled-computation-row'>
            <div class='scaled-rank'>#{rank}</div>
            <div class='scaled-details'>
                <div class='scaled-connection'>
                    <span class='token-name' style='color:#ff5ca9;'>{tokens[focus_idx]}</span>
                    <span style='color:#94a3b8;margin:0 4px;'>→</span>
                    <span class='token-name' style='color:#3b82f6;'>{tokens[j]}</span>
                </div>
                <div class='scaled-values'>
                    <span class='scaled-step'>Q·K = <b>{dot:.2f}</b></span>
                    <span class='scaled-step'>÷√d<sub>k</sub> = <b>{scaled:.2f}</b></span>
                    <span class='scaled-step'>softmax = <b>{prob:.3f}</b></span>
                </div>
            </div>
        </div>
        """

    html = f"""
    <div class='scaled-attention-box'>
        <div class='scaled-header'>
            <span class='scaled-label'>Focus Token</span>
            <span class='token-name' style='font-size:12px;'>{tokens[focus_idx]}</span>
        </div>
        <div class='scaled-formula'>softmax(Q·K<sup>T</sup>/√d<sub>k</sub>)</div>
        <div class='scaled-computations'>
            {computations}
        </div>
    </div>
    """
    return ui.HTML(html)

def get_add_norm_view(res, layer_idx):
    tokens, _, _, _, hidden_states, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>Change Magnitude</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_ffn_view(res, layer_idx):
    tokens, _, _, _, hidden_states, _, _, encoder_model, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>GELU Activation</th><th>Projection</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_add_norm_post_ffn_view(res, layer_idx):
    tokens, _, _, _, hidden_states, *_ = res
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
    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'><tr><th>Token</th><th>Residual Change (FFN)</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_layer_output_view(res, layer_idx):
    tokens, _, _, _, hidden_states, *_ = res
    if layer_idx + 1 >= len(hidden_states):
        return ui.HTML("")
    hs = hidden_states[layer_idx + 1][0].cpu().numpy()

    rows = []
    for i, tok in enumerate(tokens):
        vec_strip = array_to_base64_img(hs[i][:64], "viridis", 0.15)
        vec_tip = "Hidden state (first 32 dims): " + ", ".join(f"{v:.3f}" for v in hs[i][:32])
        mean_val = float(hs[i].mean())
        std_val = float(hs[i].std())
        max_val = float(hs[i].max())

        rows.append(f"""
            <tr>
                <td class='token-name'>{tok}</td>
                <td><img class='heatmap' src='data:image/png;base64,{vec_strip}' title='{vec_tip}'></td>
                <td style='font-size:9px;color:#374151;white-space:nowrap;'>
                    μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}
                </td>
            </tr>
        """)

    return ui.HTML(
        "<div class='card-scroll'>"
        "<table class='token-table'>"
        "<tr><th>Token</th><th>Vector (64 dims)</th><th>Statistics</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

def get_output_probabilities(res, use_mlm, text):
    if not use_mlm:
        return ui.HTML(
            "<div class='prediction-panel'>"
            "<p style='font-size:11px;color:#6b7280;'>Enable <b>Use MLM head for predictions</b> to render top-k token probabilities.</p>"
            "</div>"
        )
    
    if not text:
        return ui.HTML(
            "<div class='prediction-panel'>"
            "<p style='font-size:11px;color:#9ca3af;'>Type a sentence to see predictions.</p>"
            "</div>"
        )

    _, _, _, _, _, inputs, tokenizer, _, mlm_model, *_ = res
    device = ModelManager.get_device()
    
    # We need to re-tokenize to be sure, but we can reuse inputs if they match
    # For safety, let's just use the inputs we have
    
    with torch.no_grad():
        mlm_outputs = mlm_model(**inputs)
        probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]
    logits_tensor = mlm_outputs.logits[0]

    mlm_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cards = ""
    top_k = 5
    
    for i, tok in enumerate(mlm_tokens):
        token_probs = probs[i]
        top_vals, top_idx = torch.topk(token_probs, top_k)
        
        pred_rows = ""
        for rank, (p, idx) in enumerate(zip(top_vals, top_idx)):
            ptok = tokenizer.decode([idx.item()]) or "[UNK]"
            pval = float(p)
            width = max(4, int(pval * 100))
            logit_val = float(logits_tensor[i, idx])
            exp_logit = float(torch.exp(logits_tensor[i, idx]))
            sum_exp = float(torch.sum(torch.exp(logits_tensor[i])))
            
            unique_id = f"mlm-detail-{i}-{rank}"

            pred_rows += f"""
            <div class='mlm-pred-row'>
                <span class='mlm-pred-token' onclick="toggleMlmDetails('{unique_id}')">
                    {ptok}
                </span>
                <div class='mlm-bar-bg'>
                    <div class='mlm-bar-fill' style='width:{width}%;'></div>
                </div>
                <span class='mlm-prob-text'>{pval:.1%}</span>
            </div>
            <div id='{unique_id}' class='mlm-details-panel'>
                <div class='mlm-math'>softmax(logit<sub>i</sub>) = exp(logit<sub>i</sub>) / Σ<sub>j</sub> exp(logit<sub>j</sub>)</div>
                <div class='mlm-step'>
                    <span>logit<sub>i</sub></span>
                    <b>{logit_val:.4f}</b>
                </div>
                <div class='mlm-step'>
                    <span>exp(logit<sub>i</sub>)</span>
                    <b>{exp_logit:.4f}</b>
                </div>
                <div class='mlm-step'>
                    <span>Σ exp(logit<sub>j</sub>)</span>
                    <b>{sum_exp:.4f}</b>
                </div>
                <div class='mlm-step' style='margin-top:4px;padding-top:4px;border-top:1px dashed #cbd5e1;'>
                    <span>Probability</span>
                    <b style='color:var(--primary-color);'>{pval:.6f}</b>
                </div>
            </div>
            """
        
        cards += f"""
        <div class='mlm-card'>
            <div class='mlm-token-header'>{tok}</div>
            <div style='display:flex;flex-direction:column;gap:4px;'>
                {pred_rows}
            </div>
        </div>
        """

    return ui.HTML(
        f"<div class='prediction-panel'><div class='card-scroll' style='max-height:340px;'><div class='mlm-grid'>{cards}</div></div></div>"
    )

def get_metrics_display(res):
    _, _, _, attentions, *_ = res
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
    
    gradients = ["#fdf5f8", "#fef7fa", "#fdf6f9", "#fef8fb", "#fcf5f7", "#fef6f9"]
    
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

def get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, max_depth):
    """Generate JSON tree data for D3.js visualization."""
    tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
    if attentions is None or len(attentions) == 0:
        return None
    
    # Get attention matrix for selected layer and head
    att = attentions[layer_idx][0, head_idx].cpu().numpy()
    
    # Get Q and K for computing dot products
    layer = encoder_model.encoder.layer[layer_idx].attention.self
    hs_in = hidden_states[layer_idx]
    with torch.no_grad():
        Q = layer.query(hs_in)[0].cpu().numpy()
        K = layer.key(hs_in)[0].cpu().numpy()
    
    num_heads = layer.num_attention_heads if hasattr(layer, "num_attention_heads") else 12
    d_k = Q.shape[-1] // num_heads
    
    # Compute the tree structure with proper JSON format
    tree = compute_influence_tree(att, tokens, Q, K, d_k, root_idx, top_k, max_depth)
    
    return tree





def server(input, output, session):
    running = reactive.value(False)
    cached_result = reactive.value(None)
    isa_selected_pair = reactive.Value(None)
    

    def tokenize_with_segments(text: str, tokenizer):
        pattern = re.search(r"([.!?])\s+([A-Za-z])", text)
        if pattern:
            split_idx = pattern.end(1)
            sentence_a = text[:split_idx].strip()
            sentence_b = text[split_idx:].strip()
            if sentence_a and sentence_b:
                return tokenizer(sentence_a, sentence_b, return_tensors="pt")
        return tokenizer(text, return_tensors="pt")

    def heavy_compute(text, model_name):
        print("DEBUG: Starting heavy_compute")
        if not text: return None
        tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
        print("DEBUG: Models loaded")
        device = ModelManager.get_device()
        inputs = tokenize_with_segments(text, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = encoder_model(**inputs)
        print("DEBUG: Model inference complete")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        embeddings = outputs.last_hidden_state[0].cpu().numpy()
        attentions = outputs.attentions
        hidden_states = outputs.hidden_states
        pos_enc = positional_encoding(len(tokens), embeddings.shape[1])
        
        # Compute head specialization metrics
        head_specialization = None
        try:
            print("DEBUG: Computing head specialization")
            head_specialization = compute_all_heads_specialization(attentions, tokens, text)
            print("DEBUG: Head specialization complete")
        except Exception as e:
            print(f"Warning: Could not compute head specialization: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute ISA
        isa_data = None
        try:
            print("DEBUG: Computing ISA")
            isa_data = compute_isa(attentions, tokens, text, tokenizer, inputs)
            print("DEBUG: ISA complete")
        except Exception as e:
            print(f"Warning: Could not compute ISA: {e}")
            traceback.print_exc()

        print("DEBUG: heavy_compute finished")
        return (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, mlm_model, head_specialization, isa_data)

    @reactive.effect
    @reactive.event(input.generate_all)
    async def compute_all():
        text = input.text_input().strip()
        if not text:
            return
        
        running.set(True)
        await session.send_custom_message('start_loading', {})
        await asyncio.sleep(0.1)
        model_name = input.model_name()
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, heavy_compute, text, model_name)
            cached_result.set(result)
        except Exception:
            cached_result.set(None)
            await session.send_custom_message('stop_loading', {}) # Only stop if error
            running.set(False)
        # Note: We do NOT stop loading here on success. The dashboard render will do it.

    @output
    @render.ui
    def preview_text():
        res = cached_result.get()
        if not res:
            t = input.text_input().strip()
            return ui.HTML(f'<div style="font-family:monospace;color:#6b7280;font-size:14px;">"{t}"</div>' if t else '<div style="color:#9ca3af;font-size:12px;">Type a sentence above and click Generate All.</div>')
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">No attention data available.</div>')
        att_layers = [layer[0].cpu().numpy() for layer in attentions]
        att_avg = np.mean(att_layers, axis=(0, 1))
        attention_received = att_avg.sum(axis=0)
        att_received_norm = (attention_received - attention_received.min()) / (attention_received.max() - attention_received.min() + 1e-10)
        token_html = []
        for i, (tok, att_recv, recv_norm) in enumerate(zip(tokens, attention_received, att_received_norm)):
            opacity = 0.2 + (recv_norm * 0.6)
            bg_color = f"rgba(59, 130, 246, {opacity})" # Keep blue for attention
            tooltip = f"Token: {tok}&#10;Attention Received: {att_recv:.3f}"
            token_html.append(f'<span class="token-viz" style="background:{bg_color};" title="{tooltip}">{tok}</span>')
        html = '<div class="token-viz-container">' + ''.join(token_html) + '</div>'
        legend_html = '''
        <div style="display:flex;gap:12px;margin-top:8px;font-size:9px;color:#6b7280;">
            <div style="display:flex;align-items:center;gap:4px;">
                <div style="width:10px;height:10px;background:rgba(59,130,246,0.8);border-radius:2px;"></div><span>High Attention</span>
            </div>
            <div style="display:flex;align-items:center;gap:4px;">
                <div style="width:10px;height:10px;background:rgba(59,130,246,0.2);border-radius:2px;"></div><span>Low Attention</span>
            </div>
        </div>
        '''
        return ui.HTML(html + legend_html)

    @output
    @render.ui
    def dashboard_content():
        try:
            res = cached_result.get()
            if not res:
                return ui.HTML("<script>$('#loading_spinner').hide(); $('#generate_all').prop('disabled', false).css('opacity', '1');</script>")
            
            tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
            num_layers = len(encoder_model.encoder.layer)
            num_heads = encoder_model.encoder.layer[0].attention.self.num_attention_heads

            # Get current selections or defaults
            try: qkv_layer = int(input.qkv_layer())
            except: qkv_layer = 0
            
            try: att_layer = int(input.att_layer())
            except: att_layer = 0
            
            try: att_head = int(input.att_head())
            except: att_head = 0
            
            try: focus_token_idx = int(input.scaled_attention_token())
            except: focus_token_idx = 0

            try: flow_select = input.flow_token_select()
            except: flow_select = "all"

            try: use_mlm_val = input.use_mlm()
            except: use_mlm_val = False

            try: text_val = input.text_input()
            except: text_val = ""
            
            # Get inputs for new selectors
            try: radar_mode = input.radar_mode()
            except: radar_mode = "single"
            
            try: radar_layer = int(input.radar_layer())
            except: radar_layer = 0
            
            try: radar_head = int(input.radar_head())
            except: radar_head = 0
            
            try: tree_root_idx = int(input.tree_root_token())
            except: tree_root_idx = 0
            
            # Clean tokens for selectors
            clean_tokens = [t.replace("##", "") if t.startswith("##") else t for t in tokens]

            # Construct UI
            layout = ui.div(
                {"class": "dashboard-stack"},
                # Row 1
                ui.layout_columns(
                    ui.div({"class": "card"}, ui.h4("Token Embeddings"), get_embedding_table(res)),
                    ui.div({"class": "card"}, ui.h4("Segment Embeddings"), get_segment_embedding_view(res)),
                    ui.div({"class": "card"}, ui.h4("Positional Embeddings"), get_posenc_table(res)),
                    col_widths=[4, 4, 4]
                ),
                # Row 2
                ui.layout_columns(
                    ui.div({"class": "card"}, ui.h4("Sum & Layer Normalization"), get_sum_layernorm_view(res, encoder_model)),
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls"},
                            ui.h4("Q/K/V Projections", title="Query / Key / Value Projections"),
                            ui.div(
                                {"class": "select-compact"},
                                ui.input_select("qkv_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(qkv_layer))
                            )
                        ),
                        get_qkv_table(res, qkv_layer)
                    ),
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls"},
                            ui.h4("Scaled Dot-Product Attention"),
                            ui.div(
                                {"class": "header-right"},
                                ui.tags.span("Focus:", style="font-size:10px; font-weight:600; color:#64748b;"),
                                ui.div(
                                    {"class": "select-compact"},
                                    ui.input_select("scaled_attention_token", None, choices={str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}, selected=str(focus_token_idx))
                                )
                            )
                        ),
                        get_scaled_attention_view(res, att_layer, att_head, focus_token_idx)
                    ),
                    col_widths=[4, 4, 4]
                ),
                # Global Metrics
                ui.div({"class": "card"}, ui.h4("Global Attention Metrics"), get_metrics_display(res)),
                # Row 3
                ui.layout_columns(
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls"},
                            ui.h4("Multi-Head Attention"),
                            ui.div(
                                {"class": "header-right"},
                                ui.div({"class": "select-compact"}, ui.input_select("att_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(att_layer))),
                                ui.div({"class": "select-compact"}, ui.input_select("att_head", None, choices={str(i): f"Head {i}" for i in range(num_heads)}, selected=str(att_head))),
                            )
                        ),
                        output_widget("attention_map")
                    ),
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls"},
                            ui.h4("Attention Flow"),
                            ui.div(
                                {"class": "header-right"},
                                ui.tags.span("Filter:", style="font-size:12px; font-weight:600; color:#64748b;"),
                                ui.div(
                                    {"class": "select-compact"},
                                    ui.input_select("flow_token_select", None, choices={"all": "All tokens", **{str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}}, selected=flow_select)
                                )
                            )
                        ),
                        output_widget("attention_flow")
                    ),
                    col_widths=[6, 6]
                ),
                # Head Specialization Radar

                # Row 4 - Radar and Tree (Side by Side)
                ui.layout_columns(
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls-stacked"},
                            ui.div(
                                {"class": "header-row-top"},
                                ui.h4("Attention Head Specialization"),
                                ui.div(
                                    {"class": "header-right"},
                                    ui.div({"class": "select-compact", "id": "radar_head_selector"}, ui.input_select("radar_head", None, choices={str(i): f"Head {i}" for i in range(num_heads)}, selected=str(radar_head))),
                                    ui.div({"class": "select-compact"}, ui.input_select("radar_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(radar_layer))),
                                )
                            ),
                            ui.div(
                                {"class": "header-row-bottom"},
                                ui.span("Attention Mode:", class_="toggle-label"),
                                ui.input_radio_buttons("radar_mode", None, {"single": "Single Head", "all": "All Heads"}, selected=radar_mode, inline=True)
                            )
                        ),
                        head_specialization_radar(res, radar_layer, radar_head, radar_mode),
                        ui.HTML(f"""
                            <style>
                                .metric-tag.specialization {{
                                    color: #ff5ca9 !important;
                                    font-weight: 700 !important;
                                    font-size: 13px !important;
                                    padding: 6px 12px;
                                    border-radius: 20px;
                                    transition: all 0.2s ease;
                                    display: inline-block;
                                    cursor: pointer;
                                }}
                                .metric-tag.specialization:hover {{
                                    transform: scale(1.05);
                                    color: #ff78bc !important;
                                    background-color: rgba(255, 92, 169, 0.1);
                                }}
                            </style>
                            <div class="radar-explanation" style="margin-top: 10px;">
                                <p style="margin: 10px 0 12px 0; font-size: 13px; color: #1e293b; text-align: center; font-weight: 600; line-height: 1.8;">
                                    <strong style="color: #ff5ca9;">Attention Specialization Dimensions</strong> — click any to see detailed explanation:<br>
                                </p>
                                <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; padding: 12px; background: linear-gradient(135deg, #fff5f9 0%, #ffe5f3 100%); border-radius: 12px; border: 2px solid #ffcce5;">
                                    <span class="metric-tag specialization" onclick="showMetricModal('Syntax', {radar_layer}, {radar_head})">Syntax</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('Semantics', {radar_layer}, {radar_head})">Semantics</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('CLS Focus', {radar_layer}, {radar_head})">CLS Focus</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('Punctuation', {radar_layer}, {radar_head})">Punctuation</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('Entities', {radar_layer}, {radar_head})">Entities</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('Long-range', {radar_layer}, {radar_head})">Long-range</span>
                                    <span class="metric-tag specialization" onclick="showMetricModal('Self-attention', {radar_layer}, {radar_head})">Self-attention</span>
                                </div>
                            </div>
                        """)
                    ),
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-controls-stacked"},
                            ui.div(
                                {"class": "header-row-top"},
                                ui.h4("Attention Dependency Tree"),
                                ui.div(
                                    {"class": "header-right"},
                                    ui.tags.span("Root:", style="font-size:11px; font-weight:600; color:#64748b; margin-right: 4px;"),
                                    ui.div(
                                        {"class": "select-compact"},
                                        ui.input_select(
                                            "tree_root_token",
                                            None,
                                            choices={str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)},
                                            selected=str(tree_root_idx)
                                        )
                                    )
                                )
                            )
                        ),
                        get_influence_tree_ui(res, tree_root_idx, radar_layer, radar_head)
                    ),
                    col_widths=[5, 7]
                ),
                # Row 4.5 - Inter-Sentence Attention (ISA)
                ui.div(
                    {"class": "card"},
                    ui.h4("Inter-Sentence Attention (ISA)"),
                    ui.layout_columns(
                        ui.div(
                            {"style": "overflow: auto; height: 520px;"},
                            output_widget("isa_scatter")
                        ),
                        ui.div(
                            ui.output_ui("isa_detail_info"),
                            ui.div(output_widget("isa_token_view")),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.div(
                        {"class": "isa-explanation-block"},
                        ui.tags.p(
                            ui.tags.strong("Inter-Sentence Attention (ISA):", style="color: #ff5ca9;"), 
                            " visualizes the relationship between two sentences, focusing on how the tokens in Sentence X attend to the tokens in Sentence Y. The ", 
                            ui.tags.strong("ISA score"), 
                            " quantifies this relationship, with higher values indicating a stronger connection between the tokens in Sentence X and Sentence Y.",
                            ui.br(), ui.br(),
                            "In the ", 
                            ui.tags.strong("Token-to-Token Attention", style="color: #ff5ca9;"), 
                            " plot, each square represents the attention strength between a token from Sentence X (left) and a token from Sentence Y (top). Thicker squares indicate stronger attention, meaning those tokens are more related in terms of the model's attention mechanism.",
                            style = "margin: 0; font-size: 11px; color: #64748b; line-height: 1.6;"
                        )
                    )
                ),
                
                # Row 5 - Residuals & FFN
                ui.layout_columns(
                    ui.div({"class": "card"}, ui.h4("Add & Norm"), get_add_norm_view(res, att_layer)),
                    ui.div({"class": "card"}, ui.h4("Feed-Forward Network"), get_ffn_view(res, att_layer)),
                    ui.div({"class": "card"}, ui.h4("Add & Norm (post-FFN)"), get_add_norm_post_ffn_view(res, att_layer)),
                    col_widths=[4, 4, 4]
                ),
                # Row 6
                ui.layout_columns(
                    ui.div({"class": "card"}, ui.h4("Hidden States"), get_layer_output_view(res, att_layer)),
                    ui.div({"class": "card"}, ui.h4("Token Output Predictions (MLM)"), get_output_probabilities(res, use_mlm_val, text_val)),
                    col_widths=[6, 6]
                ),
                # Script to hide spinner when this UI is mounted
                ui.tags.script("$('#loading_spinner').hide(); $('#generate_all').prop('disabled', false).css('opacity', '1');")
            )
            return layout
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(err_msg)
            return ui.HTML(f"""
                <div style='color:red; padding:20px; border:1px solid red; border-radius:8px; background:#fff0f0;'>
                    <h3>Error Rendering Dashboard</h3>
                    <pre style='white-space:pre-wrap; font-size:11px;'>{err_msg}</pre>
                </div>
                <script>$('#loading_spinner').hide(); $('#generate_all').prop('disabled', false).css('opacity', '1');</script>
            """)

    @output
    @output
    @render_widget
    def isa_scatter():
        res = cached_result.get()
        if not res or not res[-1]:
            return None

        isa_data = res[-1]
        matrix = isa_data["sentence_attention_matrix"]
        sentences = isa_data["sentence_texts"]
        n = len(sentences)

        x, y = np.meshgrid(np.arange(n), np.arange(n))
        x_flat, y_flat = x.flatten(), y.flatten()
        scores = np.nan_to_num(matrix.flatten(), nan=0.0)

        hover_texts = [
            f"Target ← {sentences[int(r)][:60]}...<br>Source → {sentences[int(c)][:60]}...<br>ISA = {s:.4f}"
            for r, c, s in zip(y_flat, x_flat, scores)
        ]

        customdata = list(zip(y_flat.tolist(), x_flat.tolist()))

        fig = go.FigureWidget(data=go.Scatter(
            x=x_flat, y=y_flat,
            mode="markers",
            marker=dict(
                size=np.clip(scores * 40 + 12, 12, 80),
                color=scores,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="ISA Score"),
                line=dict(width=1, color="white")
            ),
            text=hover_texts,
            hoverinfo="text",
            customdata=customdata
        ))

        labels = [s[:30] + "..." if len(s) > 30 else s for s in sentences]

        fig.update_layout(
            title="Click a dot",
            xaxis=dict(title="Source (Sentence Y)", tickmode="array", tickvals=np.arange(n), ticktext=labels),
            yaxis=dict(title="Target (Sentence X)", tickmode="array", tickvals=np.arange(n), ticktext=labels, autorange="reversed"),
            height=500,
            autosize=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            clickmode="event+select",
            margin=dict(l=60, r=60, t=60, b=60),
        )

        def on_click(trace, points, state):
            if points.point_inds:
                target_idx = int(points.ys[0])
                source_idx = int(points.xs[0])
                isa_selected_pair.set((target_idx, source_idx))

        fig.data[0].on_click(on_click)
        return fig


    @output
    @render_plotly
    def isa_token_view():
        pair = isa_selected_pair()
        res = cached_result.get()

        if res is None or pair is None:
            fig = go.Figure()
            fig.update_layout(
                title="Click a dot on the left chart",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=500,
                autosize=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        target_idx, source_idx = pair
        tokens, _, _, attentions, *_ = res
        isa_data = res[-1]
        boundaries = isa_data["sentence_boundaries_ids"]

        from .isa import get_sentence_token_attention
        sub_att, tokens_combined, src_start = get_sentence_token_attention(
            attentions, tokens, target_idx, source_idx, boundaries
        )

        toks_target = tokens_combined[:src_start]
        toks_source = tokens_combined[src_start:]

        fig = go.Figure(data=go.Heatmap(
            z=sub_att,
            x=toks_source,
            y=toks_target,
            colorscale="Viridis",
            colorbar=dict(title="Attention"),
            hovertemplate="Target: %{y}<br>Source: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=f"Token-to-Token — S{target_idx} ← S{source_idx}",
            xaxis_title="Source tokens",
            yaxis_title="Target tokens",
            height=500,
            autosize=True,
            margin=dict(l=80, r=60, t=60, b=60),
        )
        return fig


    @output
    @render.ui
    def isa_detail_info():
        pair = isa_selected_pair()
        if pair is None:
            return ui.HTML("<em style='color:#94a3b8;'>Click a dot on the ISA chart.</em>")
        tx, sy = pair
        res = cached_result.get()
        score = 0.0
        if res and res[-1]:
            score = res[-1]["sentence_attention_matrix"][tx, sy]
        return ui.HTML(f"Sentence {tx} (target) ← Sentence {sy} (source) · ISA: <strong>{score:.4f}</strong>")


    @reactive.effect
    @reactive.event(input.isa_click)
    def _handle_isa_click():
        click = input.isa_click()
        if not click or "x" not in click or "y" not in click:
            return
        # Plotly coordinates: x = source (B), y = target (A)
        source_idx = click["x"]
        target_idx = click["y"]
        isa_selected_pair.set((int(target_idx), int(source_idx)))



    # New ISA overlay trigger handler
    @reactive.effect
    @reactive.event(input.isa_click)
    def handle_isa_overlay():
        trigger_data = input.isa_click()
        print(f"DEBUG: handle_isa_overlay triggered with: {trigger_data}")
        if not trigger_data: return
        
        # Map coordinates from Plotly click
        # input x is Source (Sentence B) -> sent_y_idx
        # input y is Target (Sentence A) -> sent_x_idx
        sent_x_idx = trigger_data.get('y')
        sent_y_idx = trigger_data.get('x')
        
        if sent_x_idx is None or sent_y_idx is None: return
        
        # Store the selected pair for the drilldown renderer
        print(f"DEBUG: Setting isa_selected_pair to ({sent_x_idx}, {sent_y_idx})")
        isa_selected_pair.set((sent_x_idx, sent_y_idx))

    @reactive.effect
    @reactive.event(input.isa_overlay_trigger)
    def _handle_isa_overlay_trigger():
        data = input.isa_overlay_trigger()
        print(f"DEBUG: isa_overlay_trigger received: {data}")
        if data:
            try:
                x = int(data["sentXIdx"])
                y = int(data["sentYIdx"])
                print(f"DEBUG: Setting isa_selected_pair to ({x}, {y})")
                isa_selected_pair.set((x, y))
            except Exception as e:
                print(f"DEBUG: Error parsing trigger data: {e}")



    @render_plotly
    def attention_map():
        res = cached_result.get()
        if not res: return None
        tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0: return None
        try: layer_idx = int(input.att_layer())
        except: layer_idx = 0
        try: head_idx = int(input.att_head())
        except: head_idx = 0
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        layer = encoder_model.encoder.layer[layer_idx].attention.self
        hs_in = hidden_states[layer_idx]
        with torch.no_grad():
            Q = layer.query(hs_in)[0].cpu().numpy()
            K = layer.key(hs_in)[0].cpu().numpy()
        L = len(tokens)
        num_heads = layer.num_attention_heads if hasattr(layer, "num_attention_heads") else 12
        d_k = Q.shape[-1] // num_heads
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
            "<b>Query Token:</b> %{y}<br><b>Key Token:</b> %{x}<br><br><b>Calculation:</b><br>"
            "1. Dot Product: Q·K = %{customdata[2]}<br>2. Scaled: (Q·K)/√d_k = %{customdata[3]}<br>"
            "3. Softmax Result: <b>%{customdata[4]}</b><br><br><b>Vectors (first 5 dims):</b><br>"
            "Q = %{customdata[0]}<br>K = %{customdata[1]}<extra></extra>"
        )
        fig = px.imshow(att, x=tokens, y=tokens, color_continuous_scale="Blues", aspect="auto")
        fig.update_traces(customdata=custom, hovertemplate=hover)
        fig.update_layout(
            xaxis_title="Key (attending to)", yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40), coloraxis_colorbar=dict(title="Attention"),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#111827"),
        )
        return fig

    @render_plotly
    def attention_flow():
        res = cached_result.get()
        if not res: return None
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0: return None
        try: layer_idx = int(input.att_layer())
        except: layer_idx = 0
        try: head_idx = int(input.att_head())
        except: head_idx = 0
        try: selected = input.flow_token_select()
        except: selected = "all"
        focus_idx = None if selected == "all" else int(selected)
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        n_tokens = len(tokens)
        color_palette = ['#ff5ca9', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#0ea5e9']
        fig = go.Figure()
        block_width = 0.8 / n_tokens
        for i, tok in enumerate(tokens):
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2
            show_focus = focus_idx is not None
            is_selected = focus_idx == i if show_focus else True
            font_size = 13 if is_selected else 10
            text_color = color if (show_focus and is_selected) else "#111827"
            fig.add_trace(go.Scatter(x=[x_pos], y=[1.05], mode='text', text=tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.05], mode='text', text=tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'), showlegend=False, hoverinfo='skip'))
        
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
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=line_color, width=line_width), opacity=line_opacity, showlegend=False, hoverinfo='text' if is_line_focused else 'skip', hovertext=f"<b>{tokens[i]} to {tokens[j]}</b><br>Attention: {weight:.4f}"))
        
        title_text = ""
        if focus_idx is not None:
            focus_color = color_palette[focus_idx % len(color_palette)]
            title_text += f" · <b style='color:{focus_color}'>Focused: '{tokens[focus_idx]}'</b>"
        
        fig.update_layout(title=title_text, xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.05, 1.05]), yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.25, 1.25]), plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font=dict(color='#111827'), height=500, margin=dict(l=20, r=20, t=60, b=40), clickmode='event+select', hovermode='closest', dragmode=False)
        return fig

    # This function is now called directly from dashboard_content
    def head_specialization_radar(res, layer_idx, head_idx, mode):
        if not res: return None
        
        tokens, _, _, attentions, _, _, _, _, _, head_specialization, *_ = res
        if attentions is None or len(attentions) == 0 or head_specialization is None:
            return None
        
        # Get metrics for the selected layer
        if layer_idx not in head_specialization:
            return None
        
        layer_metrics = head_specialization[layer_idx]
        
        # Dimension names for radar chart
        dimensions = ["Syntax", "Semantics", "CLS Focus", "Punctuation", "Entities", "Long-range", "Self-attention"]
        dimension_keys = ["syntax", "semantics", "cls", "punct", "entities", "long_range", "self"]
        
        # Color palette - Attention Atlas colors (Blue/Pink theme)
        colors = ['#ff5ca9', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#6366f1', '#f43f5e', 
                  '#a855f7', '#0ea5e9', '#d946ef', '#2dd4bf', '#f59e0b']
        
        fig = go.Figure()
        
        if mode == "single":
            # Single head mode
            if head_idx not in layer_metrics:
                return None
            
            metrics = layer_metrics[head_idx]
            values = [metrics[key] for key in dimension_keys]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions + [dimensions[0]],
                fill='toself',
                fillcolor=f'rgba(255, 92, 169, 0.3)',
                line=dict(color='#ff5ca9', width=2),
                name=f'Head {head_idx}',
                hovertemplate='<b>%{theta}</b><br>Value: %{r:.4f}<extra></extra>'
            ))
            
            title_text = f'Head Specialization Radar — Layer {layer_idx}, Head {head_idx}'
        else:
            # All heads mode
            num_heads = len(layer_metrics)
            for h_idx in range(num_heads):
                if h_idx not in layer_metrics:
                    continue
                
                metrics = layer_metrics[h_idx]
                values = [metrics[key] for key in dimension_keys]
                values.append(values[0])  # Close the polygon
                
                color = colors[h_idx % len(colors)]
                # Convert hex to rgba with transparency
                if color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    fill_color = f'rgba({r}, {g}, {b}, 0.15)'
                    line_color = color
                else:
                    fill_color = color
                    line_color = color
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=dimensions + [dimensions[0]],
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1.5),
                    name=f'Head {h_idx}',
                    hovertemplate=f'<b>Head {h_idx}</b><br>%{{theta}}: %{{r:.4f}}<extra></extra>'
                ))
            
            title_text = f'Head Specialization Radar — Layer {layer_idx} (All Heads)'
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=True,
                    ticks='outside',
                    tickfont=dict(size=10),
                    gridcolor='#e2e8f0'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color='#475569'),
                    gridcolor='#e2e8f0'
                ),
                bgcolor='#ffffff'
            ),
            showlegend=(mode == "all"),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            ),
            title=dict(
                text=title_text,
                font=dict(size=14, color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff',
            height=500,
            margin=dict(l=80, r=150, t=60, b=80)
        )
        
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))

    # This function replaces the previous @output @render.ui def influence_tree():
    def get_influence_tree_ui(res, root_idx=0, layer_idx=0, head_idx=0):
        if not res:
            return ui.HTML("""
                <div style='padding: 20px; text-align: center;'>
                    <p style='font-size:11px;color:#9ca3af;'>Generate attention data to view the influence tree.</p>
                </div>
            """)
        
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0:
            return ui.HTML("<p style='font-size:11px;color:#6b7280;'>No attention data available.</p>")
        
        # Use passed layer/head indices
        depth = 2     # Maximum depth 
        top_k = 3     # Default top-k
        
        # Ensure valid indices
        root_idx = max(0, min(root_idx, len(tokens) - 1))
        
        # Placeholder for actual tree data generation logic
        # This function would typically call a backend utility to compute the tree
        # For this example, we'll simulate a simple tree structure.
        
        # Example: A simple tree structure for demonstration
        # In a real application, this would be computed from 'attentions'
        # based on 'layer_idx', 'head_idx', 'root_idx', 'depth', 'top_k'.
        
        # For the purpose of this edit, we'll assume a function `_generate_tree_data` exists
        # that takes these parameters and returns a dict suitable for D3.
        # Since the actual `_generate_tree_data` is not provided, we'll create a dummy one.
        
        def _generate_tree_data(tokens, root_idx, layer_idx, head_idx, max_depth, top_k):
            # Get attention matrix for the specific layer and head
            # attentions[layer_idx] is (batch, num_heads, seq_len, seq_len)
            try:
                att_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
            except:
                return None

            def build_node(current_idx, current_depth, current_value):
                token = tokens[current_idx]
                node = {
                    "name": f"{current_idx}: {token}",
                    "att": current_value, 
                    "children": []
                }

                if current_depth < max_depth:
                    # Get attention weights for this token (what it attends to)
                    row = att_matrix[current_idx]
                    
                    # Get top-k indices
                    top_indices = np.argsort(row)[-top_k:][::-1]
                    
                    for child_idx in top_indices:
                        child_idx = int(child_idx) # Ensure native int
                        raw_att = float(row[child_idx])
                        
                        # Handle NaN/Inf for JSON safety
                        if np.isnan(raw_att) or np.isinf(raw_att):
                            raw_att = 0.0
                            
                        # Cumulative influence: parent_value * current_attention
                        child_value = current_value * raw_att if current_depth > 0 else raw_att
                        
                        child_node = build_node(child_idx, current_depth + 1, child_value)
                        # We store the raw attention too if needed, but 'att' is now cumulative influence
                        child_node["raw_att"] = raw_att 
                        child_node["qk_sim"] = 0.0 # Placeholder for D3 compatibility
                        
                        node["children"].append(child_node)
                
                return node

            # Root starts with influence 1.0
            return build_node(root_idx, 0, 1.0)

        try:
            tree_data = _generate_tree_data(tokens, root_idx, layer_idx, head_idx, depth, top_k)
            
            if tree_data is None:
                return ui.HTML("<p style='font-size:11px;color:#6b7280;'>Unable to generate tree.</p>")
            
            # Convert to JSON
            import json
            tree_json = json.dumps(tree_data)
        except Exception as e:
            return ui.HTML(f"<p style='font-size:11px;color:#ef4444;'>Error generating tree: {str(e)}</p>")
        
        # Explanation text - minimalist and after tree
        explanation = """
        <div class="tree-explanation">
            <p style="margin: 0; font-size: 11px; color: #64748b; line-height: 1.6;">
                <strong>Attention Dependency Tree:</strong> Visualizes how the root token attends to other tokens (Depth 1), and how those tokens attend to others (Depth 2).
                <span style="color: #94a3b8;">Click nodes to expand further. Thicker edges = stronger influence.</span>
            </p>
        </div>
        """
        
        html = f"""
        <div class="influence-tree-wrapper" style="height: 100%; min-height: 600px; display: flex; flex-direction: column; justify-content: space-between;">
        <div class="influence-tree-wrapper" style="height: 100%; min-height: 600px; display: flex; flex-direction: column; justify-content: space-between;">
            <div id="tree-viz-container" class="tree-viz-container" style="flex: 1; width: 100%; overflow-x: auto; overflow-y: hidden; text-align: center; position: relative;"></div>
            {explanation}
        </div>
        <script>
                (function() {{
                    function tryRender() {{
                        if (typeof d3 !== 'undefined' && typeof renderInfluenceTree !== 'undefined') {{
                            try {{
                                renderInfluenceTree({tree_json}, 'tree-viz-container');
                            }} catch(e) {{
                                console.error('Error rendering tree:', e);
                                document.getElementById('tree-viz-container').innerHTML = 
                                    '<p style="color:#ef4444;padding:20px;font-size:12px;">Error rendering tree. Check console for details.</p>';
                            }}
                        }} else {{
                            // Retry after a short delay
                            setTimeout(tryRender, 100);
                        }}
                    }}
                    tryRender();
                }})();
            </script>
        </div>
        """
        
        return ui.HTML(html)