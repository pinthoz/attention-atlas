import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from shiny import ui

from ..utils import array_to_base64_img, compute_influence_tree
from ..metrics import compute_all_attention_metrics, calculate_flow_change, calculate_balance
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


def arrow(from_section, to_section, direction="horizontal", suffix="", **kwargs):
    """
    Uniform arrow component - centered positioning
    direction: "horizontal" | "vertical" | "initial"
    """
    arrow_id = f"arrow_{from_section.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}_{to_section.replace(' ', '_').replace('&', '').replace('(', '').replace(')', '')}{suffix}"

    # Use the same "↓" glyph for both to ensure identical design (thickness/style)
    # Rotate it -90deg for horizontal to point right
    if direction == "horizontal":
        icon = ui.tags.span({"style": "display: inline-block; transform: rotate(-90deg);"}, "↓")
    else:
        icon = "↓"

    classes = f"transition-arrow arrow-{direction}"
    if "extra_class" in kwargs:
        classes += f" {kwargs.pop('extra_class')}"

    attrs = {
        "class": classes,
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        rows.append(
            f"<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
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
    
    if len(ids) != len(tokens):
         return ui.HTML("<p style='font-size:10px;color:#6b7280;'>Segment breakdown not available in Word-Level mode.</p>")

    rows = ""
    for i, (tok, seg) in enumerate(zip(tokens, ids)):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        row_class = f"seg-row-{seg}" if seg in [0, 1] else ""
        seg_label = "A" if seg == 0 else "B" if seg == 1 else str(seg)
        rows += f"""
        <tr class='{row_class}'>
            <td class='token-cell'>{clean_tok}</td>
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        strip = array_to_base64_img(pe[:64], cmap="Blues", height=0.18)
        tip = f"Position {i} encoding: " + ", ".join(f"{v:.3f}" for v in pe[:32])
        rows.append(
            f"<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
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
            # Check for aggregation mismatch
            if seq_len != len(tokens):
                # Fallback: Use aggregated embeddings from res[1] (which is Sum+Norm usually or close to it)
                # We can't separate Sum vs Norm easily aggregated.
                # Just show the same vector for both or just one column.
                return ui.HTML(f"<div class='card-scroll'><p style='font-size:11px;color:#6b7280;padding:8px;'>Detailed Sum/Norm breakdown not available in Word-Level mode (aggregated).</p></div>")
            
            word_embed = encoder_model.embeddings.word_embeddings(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embed = encoder_model.embeddings.position_embeddings(position_ids)
            seg_embed = encoder_model.embeddings.token_type_embeddings(segment_ids)
            summed = word_embed + pos_embed + seg_embed
            normalized = encoder_model.embeddings.LayerNorm(summed)
        else: # GPT-2
            if seq_len != len(tokens):
                 return ui.HTML(f"<div class='card-scroll'><p style='font-size:11px;color:#6b7280;padding:8px;'>Detailed Sum/Norm breakdown not available in Word-Level mode (aggregated).</p></div>")

            # GPT-2 uses wte (token) and wpe (position)
            word_embed = encoder_model.wte(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_embed = encoder_model.wpe(position_ids)
            summed = word_embed + pos_embed
            normalized = summed # Placeholder since GPT-2 is pre-norm

    summed_np = summed[0].cpu().numpy()
    norm_np = normalized[0].cpu().numpy()
    rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        sum_strip = array_to_base64_img(summed_np[i][:96], "Blues", 0.15)
        norm_strip = array_to_base64_img(norm_np[i][:96], "Blues", 0.15)
        rows.append(
            "<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
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
        display_tok = tok.replace("##", "").replace("Ġ", "")

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


def get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=3):
    tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
    if attentions is None or len(attentions) == 0:
        return ui.HTML("")

    # Support both single int and list of ints
    if isinstance(focus_indices, int):
        focus_indices = [focus_indices]
    elif not focus_indices:
        # Default to first token if empty list provided
        focus_indices = [0]
    
    # Limit number of tokens to display to avoid UI explosion (max 5)
    if len(focus_indices) > 5:
        focus_indices = focus_indices[:5]

    att_head = attentions[layer_idx][0, head_idx].cpu().numpy()
    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx]
    Q, K, V = extract_qkv(layer_block, hs_in)

    if hasattr(layer_block, "attention"): # BERT
        num_heads = layer_block.attention.self.num_attention_heads
    else: # GPT-2
        num_heads = layer_block.attn.num_heads
    d_k = Q.shape[-1] // num_heads

    all_blocks = ""

    for f_idx in focus_indices:
        f_idx = max(0, min(f_idx, len(tokens) - 1))
        
        # Get top k for this token
        if hasattr(layer_block, "attention"): # BERT
            top_idx = np.argsort(att_head[f_idx])[::-1][:top_k]
            # Add invisible spacer to match GPT-2's causal note height
            causal_note = "<div style='font-size:10px;margin-bottom:4px;visibility:hidden;'>Causal: Future tokens are masked</div>"
        else: # GPT-2 (Causal)
            valid_scores = [(j, att_head[f_idx, j]) for j in range(len(tokens)) if j <= f_idx]
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            top_idx = [x[0] for x in valid_scores[:top_k]]
            causal_note = "<div style='font-size:10px;color:#888;margin-bottom:4px;font-style:italic;'>Causal: Future tokens are masked</div>"

        computations = causal_note
        for rank, j in enumerate(top_idx, 1):
            dot = float(np.dot(Q[f_idx], K[j]))
            scaled = dot / np.sqrt(d_k)
            prob = att_head[f_idx, j]

            computations += f"""
            <div class='scaled-computation-row'>
                <div class='scaled-rank'>#{rank}</div>
                <div class='scaled-details'>
                    <div class='scaled-connection'>
                        <span class='token-name' style='color:#ff5ca9;'>{tokens[f_idx].replace("##", "").replace("Ġ", "")}</span>
                        <span style='color:#94a3b8;margin:0 4px;'>→</span>
                        <span class='token-name' style='color:#3b82f6;'>{tokens[j].replace("##", "").replace("Ġ", "")}</span>
                    </div>
                    <div class='scaled-values'>
                        <span class='scaled-step'>Q·K = <b>{dot:.2f}</b></span>
                        <span class='scaled-step'>÷√d<sub>k</sub> = <b>{scaled:.2f}</b></span>
                        <span class='scaled-step'>softmax = <b>{prob:.3f}</b></span>
                    </div>
                </div>
            </div>
            """
        
        # Wrap each token's block
        all_blocks += f"""
        <div class='scaled-attention-box' style='margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 16px;'>
            <div class='scaled-formula' style='margin-bottom:8px;'>
                <span style='color:#ff5ca9;font-weight:bold;'>{tokens[f_idx].replace("##", "").replace("Ġ", "")}</span>: softmax(Q·K<sup>T</sup>/√d<sub>k</sub>)
            </div>
            <div class='scaled-computations'>
                {computations}
            </div>
        </div>
        """

    html = f"""
    <div class='card-scroll'>
        {all_blocks}
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        diff = np.linalg.norm(hs_out[i] - hs_in[i])
        norm = np.linalg.norm(hs_in[i]) + 1e-6
        ratio = diff / norm
        width = max(4, min(100, int(ratio * 80)))
        rows.append(
            f"<tr><td class='token-name'>{clean_tok}</td>"
            f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;' title='Change: {ratio:.1%}'>"
            f"<div style='width:{width}%;height:10px;border-radius:999px;"
            f"background:linear-gradient(90deg,#ff5ca9,#3b82f6);'></div></div></td></tr>"
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        inter_strip = array_to_base64_img(inter_np[i][:96], "Blues", 0.15)
        proj_strip = array_to_base64_img(proj_np[i][:96], "Blues", 0.15)
        rows.append(
            "<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        diff = np.linalg.norm(hs_out[i] - hs_mid[i])
        norm = np.linalg.norm(hs_mid[i]) + 1e-6
        ratio = diff / norm
        width = max(4, min(100, int(ratio * 80)))
        rows.append(
            f"<tr><td class='token-name'>{clean_tok}</td>"
            f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;' title='Change: {ratio:.1%}'>"
            f"<div style='width:{width}%;height:10px;border-radius:999px;"
            f"background:linear-gradient(90deg,#ff5ca9,#3b82f6);'></div></div></td></tr>"
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
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        vec_strip = array_to_base64_img(hs[i][:64], "Blues", 0.15)
        vec_tip = "Hidden state (first 32 dims): " + ", ".join(f"{v:.3f}" for v in hs[i][:32])
        mean_val = float(hs[i].mean())
        std_val = float(hs[i].std())
        max_val = float(hs[i].max())

        rows.append(f"""
            <tr>
                <td class='token-name'>{clean_tok}</td>
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


def get_output_probabilities(res, use_mlm, text, suffix="", top_k=5):
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

    _, _, _, _, _, inputs, tokenizer, encoder_model, mlm_model, *_ = res
    device = ModelManager.get_device()

    is_gpt2 = not hasattr(encoder_model, "encoder")
    
    # Check for aggregation
    input_seq_len = inputs["input_ids"].shape[1]
    # We don't have tokens passed explicitly? 
    # Wait, get_output_probabilities definition does NOT take 'tokens'.
    # It takes check logic or we can use mlm_tokens length.
    
    # Let's verify compatibility
    # If we are in word level, inputs["input_ids"] is original length.
    # But this view blindly regenerates tokens from inputs. 
    # To detect word level, we need to know if 'res' is aggregated.
    # 'res' tuple has 'tokens' at index 0.
    tokens_in_res = res[0]
    if len(tokens_in_res) != input_seq_len:
         return ui.HTML(
            "<div class='prediction-panel'>"
            "<p style='font-size:11px;color:#ef4444;'>MLM Predictions are not compatible with Word-Level Aggregation.</p>"
            "<p style='font-size:10px;color:#6b7280;'>Switch off 'Word Lvl' to see predictions.</p>"
            "</div>"
        )
    
    with torch.no_grad():
        if is_gpt2:
            # GPT-2: Standard Causal Prediction (Next Token)
            mlm_outputs = mlm_model(**inputs)
            probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]
            logits_tensor = mlm_outputs.logits[0]
        else:
            # BERT: Iterative Masking (Pseudo-Likelihood)
            # We create a batch where each token is masked individually
            input_ids = inputs["input_ids"][0]
            seq_len = len(input_ids)
            mask_token_id = tokenizer.mask_token_id
            
            # Create a batch of (seq_len, seq_len)
            # Be careful with max sequence length - BERT usually handles up to 512
            # but batching 512x512 might be memory intensive on small GPUs.
            # Assuming typical shiny usage (short sentences < 50 tokens), this is fine.
            # For longer, we should chunk, but let's implement basic version first.
            
            batch_input_ids = input_ids.repeat(seq_len, 1)
            batch_input_ids.fill_diagonal_(mask_token_id)
            
            # Repeat other inputs if present
            attention_mask = inputs["attention_mask"].repeat(seq_len, 1) if "attention_mask" in inputs else None
            token_type_ids = inputs["token_type_ids"].repeat(seq_len, 1) if "token_type_ids" in inputs else None
            
            batch_inputs = {"input_ids": batch_input_ids}
            if attention_mask is not None: batch_inputs["attention_mask"] = attention_mask
            if token_type_ids is not None: batch_inputs["token_type_ids"] = token_type_ids
            
            # Run inference on the batch
            outputs = mlm_model(**batch_inputs)
            # outputs.logits shape: (seq_len, seq_len, vocab_size)
            full_logits = outputs.logits
            
            # We want the prediction for the MASKED position at each row
            # Row i has mask at index i. We want logits[i, i, :]
            diagonal_logits = full_logits[torch.arange(seq_len), torch.arange(seq_len), :]
            
            probs = torch.softmax(diagonal_logits, dim=-1)
            logits_tensor = diagonal_logits

    mlm_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cards = ""
    # top_k passed as argument

    for i, tok in enumerate(mlm_tokens):
        # Clean token header
        tok = tok.replace("##", "").replace("Ġ", "")
        if not tok: tok = "&nbsp;"
        
        # Calculate context string for display (PLL)
        context_html = ""
        if not is_gpt2:
            try:
                masked_copy = list(mlm_tokens)
                masked_copy[i] = "[MASK]"
                if hasattr(tokenizer, "convert_tokens_to_string"):
                    context_str = tokenizer.convert_tokens_to_string(masked_copy)
                else:
                    context_str = " ".join(masked_copy).replace(" ##", "").replace(" Ġ", "")
                context_html = f"<div class='mlm-context' style='margin-bottom:8px;font-size:11px;color:#475569;background:#f1f5f9;padding:4px;border-radius:4px;border:1px solid #e2e8f0;'>Context: <b>{context_str}</b></div>"
            except:
                context_html = ""

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

            unique_id = f"mlm-detail-{i}-{rank}{suffix}"

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

        # Header context expansion logic
        header_id = f"mlm-header-{i}{suffix}"
        header_class = "mlm-token-header clickable" if context_html else "mlm-token-header"
        onclick_attr = f"onclick=\"toggleMlmDetails('{header_id}')\"" if context_html else ""
        header_title = "Click to see masked context" if context_html else ""
        
        cards += f"""
        <div class='mlm-card'>
            <div class='{header_class}' {onclick_attr} title='{header_title}'>
                {tok}
                {'<span style="font-size:10px;opacity:0.6;margin-left:4px;">▼</span>' if context_html else ''}
            </div>
            <div id='{header_id}' class='mlm-details-panel' style='margin-bottom:8px;'>
                {context_html}
            </div>
            <div style='display:flex;flex-direction:column;gap:4px;'>
                {pred_rows}
            </div>
        </div>
        """

    return ui.HTML(
        f"<div class='prediction-panel'><div class='card-scroll'><div class='mlm-grid'>{cards}</div></div></div>"
    )


def get_metrics_display(res, layer_idx=None, head_idx=None):
    tokens, _, _, attentions, *_ = res
    if attentions is None or len(attentions) == 0:
        return ui.HTML("")

    att_layers = [layer[0].cpu().numpy() for layer in attentions]
    
    # If specific layer/head selected, use that; otherwise average all
    if layer_idx is not None and head_idx is not None:
        # Single layer, single head
        att_matrix = att_layers[layer_idx][head_idx]
    else:
        # Average across all layers and heads
        att_matrix = np.mean(att_layers, axis=(0, 1))
    
    metrics_dict = compute_all_attention_metrics(att_matrix)
    
    # Calculate Flow Change (JSD between first and last layer)
    flow_change = calculate_flow_change(att_layers)
    
    # Balance is now in metrics_dict (from compute_all_attention_metrics)
    balance = metrics_dict.get('balance', 0.5)
    
    # Get token count for normalization
    num_tokens = len(tokens) if tokens else att_avg.shape[0]
    
    # Normalize focus entropy by max possible entropy
    # For attention matrix: each row sums to 1, max entropy per row = log(n)
    # With n rows, max total entropy = n × log(n)
    # This gives focus_normalized in range [0, 1]: 0=focused, 1=diffuse
    max_entropy = num_tokens * np.log(num_tokens) if num_tokens > 1 else 1
    focus_normalized = metrics_dict['focus_entropy'] / max_entropy if max_entropy > 0 else 0

    # Thresholds based on paper "From Attention to Assurance" (Golshanrad & Faghih)
    # Format: (low_max, high_min, min_range, max_range, reverse)
    # reverse=True means lower values are "better" (like focus - lower = more focused)
    interpretations = {
        # Confidence Max (Eq. 5): max attention weight
        # Higher = more confident = head focuses strongly on specific token
        'confidence_max': (0.20, 0.50, 0.0, 1.0, False),
        # Confidence Avg (Eq. 6): average of row maxes
        # Higher = queries consistently find confident targets
        'confidence_avg': (0.15, 0.40, 0.0, 0.8, False),
        # Focus Normalized (Eq. 8): entropy / log(n²)
        # 0 = fully focused, 1 = fully uniform
        # LOWER = more focused = better (reverse=True)
        'focus_normalized': (0.30, 0.70, 0.0, 1.0, True),
        # Sparsity (Eq. 11): % below adaptive threshold (1/seq_len)
        # Higher = most tokens ignored = very selective
        'sparsity': (0.30, 0.60, 0.0, 1.0, False),
        # Distribution Median (Eq. 12): median attention weight
        'distribution_median': (0.005, 0.02, 0.0, 0.05, False),
        # Uniformity (Eq. 15): std dev of attention weights
        # Lower = more uniform, Higher = more variable
        'uniformity': (0.03, 0.10, 0.0, 0.2, False),
        # Flow Change (Eq. 9): JSD between first and last layer
        # Higher = more transformation = better feature extraction
        'flow_change': (0.10, 0.25, 0.0, 0.5, False),
        # Balance: proportion of attention to [CLS] (0-1)
        # Lower = content focus, Higher = CLS focus (potential shortcut)
        'balance': (0.15, 0.40, 0.0, 1.0, False),
    }

    def get_interpretation(key, value):
        """Return (level, color, gauge_percent, low_pct, high_pct)"""
        low_max, high_min, min_r, max_r, reverse = interpretations.get(key, (0.3, 0.7, 0, 1, False))
        
        # Calculate gauge percentage for value position
        gauge_pct = min(100, max(0, ((value - min_r) / (max_r - min_r)) * 100))
        
        # Calculate fixed threshold positions on the gauge
        low_pct = ((low_max - min_r) / (max_r - min_r)) * 100
        high_pct = ((high_min - min_r) / (max_r - min_r)) * 100
        
        # Determine level - Color scheme: Low=Green, Medium=Yellow, High=Red
        if reverse:
            # For entropy: low values = focused (good), high values = diffuse (bad)
            if value <= low_max:
                return ("Focused", "#22c55e", gauge_pct, low_pct, high_pct)  # Green
            elif value >= high_min:
                return ("Diffuse", "#ef4444", gauge_pct, low_pct, high_pct)  # Red
            else:
                return ("Moderate", "#f59e0b", gauge_pct, low_pct, high_pct)  # Yellow/Amber
        else:
            # Normal metrics: Low=Green, Medium=Yellow, High=Red
            if value <= low_max:
                return ("Low", "#22c55e", gauge_pct, low_pct, high_pct)  # Green
            elif value >= high_min:
                return ("High", "#ef4444", gauge_pct, low_pct, high_pct)  # Red
            else:
                return ("Medium", "#f59e0b", gauge_pct, low_pct, high_pct)  # Yellow/Amber

    # Build metrics with enhanced info - use normalized focus
    # Format: (label, value, value_fmt, key, modal_name, scale_max_label)
    metrics = [
        ("Confidence (Max)", metrics_dict['confidence_max'], "{:.2f}", "confidence_max", "Confidence Max", "1.0", False),
        ("Confidence (Avg)", metrics_dict['confidence_avg'], "{:.2f}", "confidence_avg", "Confidence Avg", "1.0", False),
        ("Focus (Normalized)", focus_normalized, "{:.2f}", "focus_normalized", "Focus", "1.0", False),
        ("Sparsity", metrics_dict['sparsity'], "{:.0%}", "sparsity", "Sparsity", "100%", False),
        ("Distribution", metrics_dict['distribution_median'], "{:.3f}", "distribution_median", "Distribution", "0.05", False),
        ("Uniformity", metrics_dict['uniformity'], "{:.3f}", "uniformity", "Uniformity", "0.2", False),
        ("Balance", balance, "{:.2f}", "balance", "Balance", "1.0", False),
        ("Flow Change", flow_change, "{:.2f}", "flow_change", "Flow Change", "∞", True),  # Global metric - always uses all layers
    ]

    cards_html = '<div class="metrics-grid">'
    for idx, (label, raw_value, fmt, key, modal_name, scale_max, is_global) in enumerate(metrics):
        value_str = fmt.format(raw_value)
        interp_label, interp_color, gauge_pct, low_pct, high_pct = get_interpretation(key, raw_value)
        
        # Fixed scale gauge: Low zone | Medium zone | High zone
        # Zone colors: Green (Low) | Yellow (Medium) | Red (High)
        zone1_color = "#22c55e"  # Green (Low)
        zone2_color = "#f59e0b"  # Yellow/Amber (Medium)
        zone3_color = "#ef4444"  # Red (High)
        
        # Add global indicator for metrics that always use all layers
        global_indicator = ''
        if is_global:
            global_indicator = '''
                <span class="global-info-icon info-tooltip-icon" 
                      onmouseenter="showGlobalMetricInfo(this);"
                      onmouseleave="hideGlobalMetricInfo();"
                      onclick="event.stopPropagation();"
                      style="font-size: 8px; width: 14px; height: 14px; line-height: 14px; margin-left: 4px; vertical-align: middle; font-family: 'PT Serif', serif;">i</span>
            '''
        
        cards_html += f'''
            <div class="metric-card"
                 data-metric-name="{modal_name}"
                 onclick="showMetricModal('{modal_name}', 'Global', 'Avg')">
                <div class="metric-label">{label}{global_indicator}</div>
                <div class="metric-value">{value_str}</div>
                <div class="metric-gauge-wrapper">
                    <span class="gauge-scale-label">0</span>
                    <div class="metric-gauge-fixed">
                        <div class="gauge-zone" style="width: {low_pct}%; background: {zone1_color}30;"></div>
                        <div class="gauge-zone" style="width: {high_pct - low_pct}%; background: {zone2_color}30;"></div>
                        <div class="gauge-zone" style="width: {100 - high_pct}%; background: {zone3_color}30;"></div>
                        <div class="gauge-marker" style="left: {gauge_pct}%; background: {interp_color};"></div>
                    </div>
                    <span class="gauge-scale-label">{scale_max}</span>
                </div>
                <div class="metric-badge-container">
                    <div class="metric-badge" style="background: {interp_color}20; color: {interp_color};">{interp_label}</div>
                </div>
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
