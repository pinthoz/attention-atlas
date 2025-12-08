import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from shiny import ui

from ..utils import array_to_base64_img, compute_influence_tree
from ..metrics import compute_all_attention_metrics
from ..models import ModelManager

def get_layer_block(model, layer_idx):
    """Get the layer block for BERT or GPT-2."""
    if hasattr(model, "encoder"): # BERT
        return model.encoder.layer[layer_idx]
    else: # GPT-2
        return model.h[layer_idx]


def extract_qkv(layer_block, hidden_states):
    """Extract Q, K, V from a layer block given hidden states."""
    with torch.no_grad():
        if hasattr(layer_block, "attention"): # BERT
            # layer_block is BertLayer
            self_attn = layer_block.attention.self
            Q = self_attn.query(hidden_states)[0].cpu().numpy()
            K = self_attn.key(hidden_states)[0].cpu().numpy()
            V = self_attn.value(hidden_states)[0].cpu().numpy()
        elif hasattr(layer_block, "attn"): # GPT-2
            # layer_block is GPT2Block
            attn = layer_block.attn
            # c_attn projects to 3 * hidden_size
            # shape: (batch, seq_len, 3 * hidden_size)
            c_attn_out = attn.c_attn(hidden_states)

            # Split
            # c_attn_out is (batch, seq_len, 3*hidden)
            # We take [0] to get (seq_len, 3*hidden)
            c_attn_out = c_attn_out[0]

            hidden_size = c_attn_out.shape[-1] // 3
            Q = c_attn_out[:, :hidden_size].cpu().numpy()
            K = c_attn_out[:, hidden_size:2*hidden_size].cpu().numpy()
            V = c_attn_out[:, 2*hidden_size:].cpu().numpy()
        else:
            raise ValueError("Unknown layer type")
    return Q, K, V


def arrow(from_section, to_section, direction="horizontal", **kwargs):
    """
    Uniform arrow component - centered positioning
    direction: "horizontal" | "vertical" | "initial"
    """
    arrow_id = f"arrow_{from_section.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}_{to_section.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}"

    # Use the same "↓" glyph for both to ensure identical design (thickness/style)
    # Rotate it -90deg for horizontal to point right
    if direction == "horizontal":
        icon = ui.tags.span({"style": "display: inline-block; transform: rotate(-90deg);"}, "↓")
    else:
        icon = "↓"

    attrs = {
        "class": f"transition-arrow arrow-{direction}",
        "onclick": f"showTransitionModal('{from_section}', '{to_section}')",
        "id": arrow_id,
        "title": f"Click: {from_section} → {to_section}"
    }
    attrs.update(kwargs)

    return ui.tags.div(attrs, icon)


def get_choices(tokens):
    if not tokens: return {}
    return {str(i): f"{i}: {t}" for i, t in enumerate(tokens)}


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
        seg_label = "A" if seg == 0 else "B" if seg == 1 else str(seg)
        rows += f"""
        <tr class='{row_class}'>
            <td class='token-cell'>{tok}</td>
            <td class='segment-cell'>{seg_label}</td>
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
        if hasattr(encoder_model, "embeddings"): # BERT
            word_embed = encoder_model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embed = encoder_model.embeddings.position_embeddings(position_ids)
            seg_embed = encoder_model.embeddings.token_type_embeddings(segment_ids)
            summed = word_embed + pos_embed + seg_embed
            normalized = encoder_model.embeddings.LayerNorm(summed)
        else: # GPT-2
            # GPT-2 uses wte (token) and wpe (position)
            word_embed = encoder_model.wte(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embed = encoder_model.wpe(position_ids)
            summed = word_embed + pos_embed
            normalized = encoder_model.ln_f(summed) # Use final layernorm as proxy or first block's ln_1?
            # Actually GPT-2 has LN inside blocks. The initial embedding is just sum.
            # But for visualization consistency, we can show sum vs normalized (if applicable).
            # Standard GPT-2 doesn't have a LayerNorm immediately after embedding, it's pre-norm inside blocks.
            # So "normalized" here might be misleading for GPT-2.
            # Let's just use summed for both columns or skip normalization viz for GPT-2?
            # Better: Show summed and "N/A" or just summed.
            # Or use the first layer's LN?
            normalized = summed # Placeholder since GPT-2 is pre-norm

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
    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx]

    Q, K, V = extract_qkv(layer_block, hs_in)

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

    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx]

    Q, K, V = extract_qkv(layer_block, hs_in)

    # Determine d_k
    if hasattr(layer_block, "attention"): # BERT
        num_heads = layer_block.attention.self.num_attention_heads
    else: # GPT-2
        num_heads = layer_block.attn.num_heads

    d_k = Q.shape[-1] // num_heads

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
    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx][0]
    with torch.no_grad():
        if hasattr(layer_block, "intermediate"): # BERT
            inter = layer_block.intermediate.dense(hs_in)
            inter_act = layer_block.intermediate.intermediate_act_fn(inter)
            proj = layer_block.output.dense(inter_act)
        else: # GPT-2
            # GPT-2: mlp.c_fc -> act -> mlp.c_proj
            inter = layer_block.mlp.c_fc(hs_in)
            inter_act = layer_block.mlp.act(inter)
            proj = layer_block.mlp.c_proj(inter_act)
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
        f"<div class='prediction-panel'><div class='card-scroll'><div class='mlm-grid'>{cards}</div></div></div>"
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
            <div class="metric-card"
                 data-metric-name="{metric_name}"
                 data-layer="{layer}"
                 data-head="{head}"
                 onclick="showMetricModal('{metric_name}', {layer}, {head})"
                 style="background: {gradient}; cursor: pointer;">
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
    # Get Q and K for computing dot products
    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx]

    Q, K, V = extract_qkv(layer_block, hs_in)

    if hasattr(layer_block, "attention"): # BERT
        num_heads = layer_block.attention.self.num_attention_heads
    else: # GPT-2
        num_heads = layer_block.attn.num_heads

    d_k = Q.shape[-1] // num_heads

    # Compute the tree structure with proper JSON format
    tree = compute_influence_tree(att, tokens, Q, K, d_k, root_idx, top_k, max_depth)

    return tree


__all__ = [
    "get_layer_block",
    "extract_qkv",
    "arrow",
    "get_choices",
    "get_embedding_table",
    "get_segment_embedding_view",
    "get_posenc_table",
    "get_sum_layernorm_view",
    "get_qkv_table",
    "get_scaled_attention_view",
    "get_add_norm_view",
    "get_ffn_view",
    "get_add_norm_post_ffn_view",
    "get_layer_output_view",
    "get_output_probabilities",
    "get_metrics_display",
    "get_influence_tree_data",
]
