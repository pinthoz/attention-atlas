import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

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


def _compute_cosine_similarity_matrix(vectors):
    """Compute pairwise cosine similarity matrix for a set of vectors."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = vectors / norms
    # Compute cosine similarity matrix
    return np.dot(normalized, normalized.T)


def _render_cosine_sim_mini(tokens, sim_matrix, top_k=3):
    """Render a compact cosine similarity view showing top-k neighbors per token."""
    rows = []
    n = len(tokens)
    for i in range(n):
        clean_tok = tokens[i].replace("##", "").replace("Ġ", "")
        # Get top-k similar tokens (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self
        top_indices = np.argsort(sims)[::-1][:top_k]

        neighbors = []
        for j in top_indices:
            other_tok = tokens[j].replace("##", "").replace("Ġ", "")
            sim_val = sim_matrix[i, j]
            neighbors.append(f"<span class='sim-neighbor' title='Similarity: {sim_val:.3f}'>{other_tok} <small>({sim_val:.2f})</small></span>")

        rows.append(f"<tr><td class='token-name'>{clean_tok}</td><td class='sim-neighbors'>{' '.join(neighbors)}</td></tr>")

    return (
        "<table class='sim-table'>"
        "<tr><th>Token</th><th>Most Similar Tokens</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _render_pca_scatter(tokens, vectors, color_class="embedding"):
    """Render a PCA 2D scatter plot of token embeddings as an inline SVG."""
    n = len(tokens)
    if n < 2:
        return "<p class='pca-note'>Need at least 2 tokens for PCA visualization.</p>"

    # Compute PCA (2 components)
    n_components = min(2, n, vectors.shape[1])
    pca = PCA(n_components=n_components)
    try:
        coords = pca.fit_transform(vectors)
    except Exception:
        return "<p class='pca-note'>Could not compute PCA for these vectors.</p>"

    # Get explained variance
    var_explained = pca.explained_variance_ratio_
    var_total = sum(var_explained) * 100

    # Normalize coordinates to SVG space (with padding)
    svg_width, svg_height = 280, 180
    padding = 30

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min() if n_components > 1 else 0, coords[:, 1].max() if n_components > 1 else 1

    # Avoid division by zero
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    # Map coordinates to SVG space
    points = []
    for i in range(n):
        x = padding + ((coords[i, 0] - x_min) / x_range) * (svg_width - 2 * padding)
        y = padding + ((coords[i, 1] - y_min) / y_range) * (svg_height - 2 * padding) if n_components > 1 else svg_height / 2
        # Flip y for SVG coordinate system
        y = svg_height - y
        clean_tok = tokens[i].replace("##", "").replace("Ġ", "").replace("<", "&lt;").replace(">", "&gt;")
        points.append((x, y, clean_tok, i))

    # Generate SVG elements
    circles = []
    labels = []
    for x, y, tok, idx in points:
        # Circle
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' class='pca-point pca-{color_class}' data-idx='{idx}'/>"
        )
        # Label (offset slightly)
        labels.append(
            f"<text x='{x + 7:.1f}' y='{y + 3:.1f}' class='pca-label'>{tok}</text>"
        )

    # Variance info
    var_text = f"PC1: {var_explained[0]*100:.1f}%"
    if n_components > 1:
        var_text += f", PC2: {var_explained[1]*100:.1f}%"

    svg = f"""
    <div class='pca-container'>
        <svg viewBox='0 0 {svg_width} {svg_height}' class='pca-svg'>
            <!-- Axes -->
            <line x1='{padding}' y1='{svg_height - padding}' x2='{svg_width - padding}' y2='{svg_height - padding}' class='pca-axis'/>
            <line x1='{padding}' y1='{padding}' x2='{padding}' y2='{svg_height - padding}' class='pca-axis'/>
            <!-- Axis labels -->
            <text x='{svg_width/2}' y='{svg_height - 5}' class='pca-axis-label'>PC1</text>
            <text x='10' y='{svg_height/2}' class='pca-axis-label' transform='rotate(-90 10 {svg_height/2})'>PC2</text>
            <!-- Points -->
            {"".join(circles)}
            <!-- Labels -->
            {"".join(labels)}
        </svg>
        <div class='pca-variance'>Explained variance: {var_total:.1f}% ({var_text})</div>
    </div>
    """
    return svg


def _render_qkv_pca_scatter(tokens, Q, K, V):
    """Render a combined PCA plot showing Q, K, V vectors for each token."""
    n = len(tokens)
    if n < 2:
        return "<p class='pca-note'>Need at least 2 tokens for PCA visualization.</p>"

    # Combine all vectors for joint PCA
    all_vectors = np.vstack([Q, K, V])

    # Compute PCA (2 components)
    n_components = min(2, all_vectors.shape[0], all_vectors.shape[1])
    pca = PCA(n_components=n_components)
    try:
        all_coords = pca.fit_transform(all_vectors)
    except Exception:
        return "<p class='pca-note'>Could not compute PCA for QKV vectors.</p>"

    # Split back into Q, K, V coordinates
    q_coords = all_coords[:n]
    k_coords = all_coords[n:2*n]
    v_coords = all_coords[2*n:]

    # Get explained variance
    var_explained = pca.explained_variance_ratio_
    var_total = sum(var_explained) * 100

    # Normalize coordinates to SVG space
    svg_width, svg_height = 320, 200
    padding = 35

    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min() if n_components > 1 else 0, all_coords[:, 1].max() if n_components > 1 else 1

    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    def map_coords(coords, idx):
        x = padding + ((coords[idx, 0] - x_min) / x_range) * (svg_width - 2 * padding)
        y = padding + ((coords[idx, 1] - y_min) / y_range) * (svg_height - 2 * padding) if n_components > 1 else svg_height / 2
        return x, svg_height - y  # Flip y

    # Generate SVG elements
    elements = []

    # Draw connecting lines between Q, K, V for each token (subtle)
    for i in range(n):
        qx, qy = map_coords(q_coords, i)
        kx, ky = map_coords(k_coords, i)
        vx, vy = map_coords(v_coords, i)
        elements.append(
            f"<path d='M{qx:.1f},{qy:.1f} L{kx:.1f},{ky:.1f} L{vx:.1f},{vy:.1f}' class='qkv-connector' fill='none'/>"
        )

    # Draw points (Q=green, K=orange, V=purple)
    for i in range(n):
        clean_tok = tokens[i].replace("##", "").replace("Ġ", "").replace("<", "&lt;").replace(">", "&gt;")

        # Query point
        qx, qy = map_coords(q_coords, i)
        elements.append(f"<circle cx='{qx:.1f}' cy='{qy:.1f}' r='4' class='pca-point pca-query' title='Q: {clean_tok}'/>")

        # Key point
        kx, ky = map_coords(k_coords, i)
        elements.append(f"<circle cx='{kx:.1f}' cy='{ky:.1f}' r='4' class='pca-point pca-key' title='K: {clean_tok}'/>")

        # Value point
        vx, vy = map_coords(v_coords, i)
        elements.append(f"<circle cx='{vx:.1f}' cy='{vy:.1f}' r='4' class='pca-point pca-value' title='V: {clean_tok}'/>")

        # Label near Query point
        elements.append(f"<text x='{qx + 6:.1f}' y='{qy - 4:.1f}' class='pca-label pca-label-small'>{clean_tok}</text>")

    # Variance info
    var_text = f"PC1: {var_explained[0]*100:.1f}%"
    if n_components > 1:
        var_text += f", PC2: {var_explained[1]*100:.1f}%"

    svg = f"""
    <div class='pca-container'>
        <div class='qkv-pca-legend'>
            <span class='legend-item'><span class='legend-dot pca-query'></span>Query</span>
            <span class='legend-item'><span class='legend-dot pca-key'></span>Key</span>
            <span class='legend-item'><span class='legend-dot pca-value'></span>Value</span>
        </div>
        <svg viewBox='0 0 {svg_width} {svg_height}' class='pca-svg'>
            <!-- Axes -->
            <line x1='{padding}' y1='{svg_height - padding}' x2='{svg_width - padding}' y2='{svg_height - padding}' class='pca-axis'/>
            <line x1='{padding}' y1='{padding}' x2='{padding}' y2='{svg_height - padding}' class='pca-axis'/>
            <!-- Axis labels -->
            <text x='{svg_width/2}' y='{svg_height - 5}' class='pca-axis-label'>PC1</text>
            <text x='10' y='{svg_height/2}' class='pca-axis-label' transform='rotate(-90 10 {svg_height/2})'>PC2</text>
            <!-- Elements -->
            {"".join(elements)}
        </svg>
        <div class='pca-variance'>Explained variance: {var_total:.1f}% ({var_text})</div>
    </div>
    """
    return svg


def get_embedding_table(res, top_k=3):
    tokens, embeddings, *_ = res
    n = len(tokens)
    unique_id = "embed_tab"

    # Compute norms and cosine similarity
    norms = [np.linalg.norm(embeddings[i]) for i in range(n)]
    emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
    sim_matrix = _compute_cosine_similarity_matrix(emb_array)

    # 1. Norm View Rows
    norm_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        norm_val = norms[i]
        norm_rows.append(
            f"<tr><td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td><td class='norm-value' style='text-align:center;'>{norm_val:.2f}</td></tr>"
        )
    
    html_norm = f"""
    <table class='combined-summary-table'>
        <tr>
            <th style='text-align:left;padding-left:8px;'>Token</th>
            <th style='text-align:center;'>L2 Norm (Magnitude)</th>
        </tr>
        {''.join(norm_rows)}
    </table>
    """

    # 2. Similarity View Rows
    sim_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        # Get top-k similar tokens
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top_indices = np.argsort(sims)[::-1][:top_k]
        neighbors = []
        for j in top_indices:
            other_tok = tokens[j].replace("##", "").replace("Ġ", "")
            sim_val = sim_matrix[i, j]
            neighbors.append(f"<span class='sim-neighbor' title='{sim_val:.3f}'>{other_tok}</span>")
        
        sim_rows.append(
            f"<tr><td class='token-name'>{clean_tok}</td><td class='sim-neighbors'>{' '.join(neighbors)}</td></tr>"
        )

    html_sim = f"""
    <table class='combined-summary-table'>
        <tr><th>Token</th><th>Cosine Similarity (Top-{top_k})</th></tr>
        {''.join(sim_rows)}
    </table>
    """

    # 3. PCA View
    html_pca = _render_pca_scatter(tokens, emb_array, color_class="embedding")

    # 4. Raw Vector View
    vector_rows = []
    for i, tok in enumerate(tokens):
        vec = embeddings[i]
        strip = array_to_base64_img(vec[:64], cmap="Blues", height=0.18)
        tip = "Embedding (first 32 dims): " + ", ".join(f"{v:.3f}" for v in vec[:32])
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        vector_rows.append(
            f"<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
            f"</tr>"
        )
    html_vec = f"""
    <table class='combined-summary-table'>
        <tr><th>Token</th><th>Embedding Vector (64 dims)</th></tr>
        {''.join(vector_rows)}
    </table>
    """

    # Assemble Tabbed Interface
    html = f"""
    <div id='{unique_id}'>
        <div class='view-controls'>
            <button class='view-btn active' data-tab='norm' onclick="switchView('{unique_id}', 'norm')" title="View vector magnitude (L2 Norm)">Norm</button>
            <button class='view-btn' data-tab='sim' onclick="switchView('{unique_id}', 'sim')" title="Cosine similarity between tokens (Query-Key interaction)">Similarity</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}', 'pca')" title="2D Principal Component Analysis projection">PCA</button>
            <button class='view-btn' data-tab='vec' onclick="switchView('{unique_id}', 'vec')" title="Visual heatmap of raw Q/K/V vectors">Raw Heatmap</button>
        </div>

        <div class='card-scroll vector-summary-container'>
            <div id='{unique_id}_norm' class='view-pane' style='display:block;'>
                {html_norm}
            </div>
            <div id='{unique_id}_sim' class='view-pane'>
                {html_sim}
            </div>
            <div id='{unique_id}_pca' class='view-pane'>
                <div class='pca-container-simple'>
                    <p style='font-size:11px; color:var(--text-muted); margin-bottom:8px; margin-top:0;'>
                        2D projection of the embedding space using PCA. Points closer together are more similar in vector space.
                    </p>
                    {html_pca}
                </div>
            </div>
            <div id='{unique_id}_vec' class='view-pane'>
                {html_vec}
            </div>
        </div>
    </div>
    """
    return ui.HTML(html)


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
            <td class='token-cell' style='text-align:left;padding-left:8px;'>{clean_tok}</td>
            <td class='segment-cell' style='text-align:center;'>{seg_label}</td>
        </tr>
        """

    return ui.HTML(
        f"""
        <div style='height: 10px;'></div>
        <div class='card-scroll vector-summary-container'>
            <table class='combined-summary-table'>
                <thead>
                    <tr>
                        <th style='width:auto;text-align:left;padding-left:8px;'>Token</th>
                        <th style='width:65px;text-align:center;'>Segment</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    )


def get_posenc_table(res, top_k=3):
    tokens, _, pos_enc, *_ = res
    n = len(tokens)
    unique_id = "pos_tab"

    # Compute norms and cosine similarity
    norms = [np.linalg.norm(pos_enc[i]) for i in range(n)]
    pe_array = np.array(pos_enc) if not isinstance(pos_enc, np.ndarray) else pos_enc
    sim_matrix = _compute_cosine_similarity_matrix(pe_array)

    # 1. Norm View
    norm_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        norm_val = norms[i]
        norm_rows.append(
            f"<tr><td class='pos-index' style='text-align:center;'>{i}</td><td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td><td class='norm-value' style='text-align:center;'>{norm_val:.2f}</td></tr>"
        )
    html_norm = f"""
    <table class='combined-summary-table'>
        <tr>
            <th style='text-align:center;'>Pos</th>
            <th style='text-align:left;padding-left:8px;'>Token</th>
            <th style='text-align:center;'>L2 Norm</th>
        </tr>
        {''.join(norm_rows)}
    </table>
    """

    # 2. Similarity View
    sim_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        # Get top-k similar positions
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top_indices = np.argsort(sims)[::-1][:top_k]
        neighbors = []
        for j in top_indices:
            other_tok = tokens[j].replace("##", "").replace("Ġ", "")
            sim_val = sim_matrix[i, j]
            neighbors.append(f"<span class='sim-neighbor' title='{sim_val:.3f}'>{other_tok}</span>")
        
        sim_rows.append(
            f"<tr><td class='pos-index'>{i}</td><td class='token-name'>{clean_tok}</td><td class='sim-neighbors'>{' '.join(neighbors)}</td></tr>"
        )
    html_sim = f"""
    <table class='combined-summary-table'>
        <tr><th>Pos</th><th>Token</th><th>Cosine Similarity (Top-{top_k})</th></tr>
        {''.join(sim_rows)}
    </table>
    """

    # 3. PCA View
    html_pca = _render_pca_scatter(tokens, pe_array, color_class="position")

    # 4. Raw Vector View
    vector_rows = []
    for i, tok in enumerate(tokens):
        pe = pos_enc[i]
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        strip = array_to_base64_img(pe[:64], cmap="Blues", height=0.18)
        tip = f"Position {i} encoding: " + ", ".join(f"{v:.3f}" for v in pe[:32])
        vector_rows.append(
            f"<tr>"
            f"<td class='pos-index'>{i}</td>"
            f"<td class='token-name'>{clean_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{strip}' title='{tip}'></td>"
            f"</tr>"
        )
    html_vec = f"""
    <table class='combined-summary-table'>
        <tr><th>Pos</th><th>Token</th><th>Position Encoding (64 dims)</th></tr>
        {''.join(vector_rows)}
    </table>
    """

    # Assemble Tabbed Interface
    html = f"""
    <div id='{unique_id}'>
        <div class='view-controls'>
            <button class='view-btn active' data-tab='norm' onclick="switchView('{unique_id}', 'norm')" title="View vector magnitude (L2 Norm)">Norm</button>
            <button class='view-btn' data-tab='sim' onclick="switchView('{unique_id}', 'sim')" title="Cosine similarity between positions">Similarity</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}', 'pca')" title="2D Principal Component Analysis projection">PCA</button>
            <button class='view-btn' data-tab='vec' onclick="switchView('{unique_id}', 'vec')" title="Visual heatmap of raw vector values">Raw Heatmap</button>
        </div>

        <div class='card-scroll vector-summary-container'>
            <div id='{unique_id}_norm' class='view-pane' style='display:block;'>
                {html_norm}
            </div>
            <div id='{unique_id}_sim' class='view-pane'>
                {html_sim}
            </div>
            <div id='{unique_id}_pca' class='view-pane'>
                <div class='pca-container-simple'>
                    {html_pca}
                </div>
            </div>
            <div id='{unique_id}_vec' class='view-pane'>
                {html_vec}
            </div>
        </div>
    </div>
    """
    return ui.HTML(html)


def _render_dual_tab_view(unique_id, html_heatmap, tokens, vectors_for_pca, html_change=None):
    """Helper to render standardized Raw Vectors | PCA tabs."""
    
    # Generate PCA view
    html_pca = _render_pca_scatter(tokens, vectors_for_pca, color_class="embedding")

    # Change tab logic
    change_btn = ""
    change_pane = ""
    
    # Defaults (Heatmap is active unless Change is present)
    heat_active_class = "active"
    heat_display = "block"
    change_display = "none"
    
    if html_change:
        # If Change exists, it becomes the default active tab
        change_btn = f'<button class=\'view-btn active\' data-tab=\'change\' onclick="switchView(\'{unique_id}\', \'change\')" title="Magnitude of residual update" style="flex: 0 1 auto; width: 25%;">Change</button>'
        change_pane = f'<div id=\'{unique_id}_change\' class=\'view-pane\' style=\'display:block;\'>{html_change}</div>'
        
        # Deactivate Heatmap default
        heat_active_class = ""
        heat_display = "none"

    return ui.HTML(f"""
    <div id='{unique_id}'>
        <div class='view-controls' style='justify-content: center;'>
            {change_btn}
            <button class='view-btn {heat_active_class}' data-tab='heat' onclick="switchView('{unique_id}\', \'heat\')" title="Visual heatmap of vector values" style="flex: 0 1 auto; width: 25%;">Raw Vectors</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}\', \'pca\')" title="2D Principal Component Analysis projection" style="flex: 0 1 auto; width: 25%;">PCA</button>
        </div>

        <div class='card-scroll vector-summary-container'>
            {change_pane}
            <div id='{unique_id}_heat' class='view-pane' style='display:{heat_display};'>{html_heatmap}</div>
            <div id='{unique_id}_pca' class='view-pane'>
                <div class='pca-container-simple'>
                    <p style='font-size:11px; color:var(--text-muted); margin-bottom:8px; margin-top:0;'>
                        2D projection of vectors.
                    </p>
                    {html_pca}
                </div>
            </div>
        </div>
    </div>
    """)

def get_sum_layernorm_view(res, encoder_model):
    tokens, _, _, _, hidden_states, inputs, *_ = res
    unique_id = "sumnorm_tab"
    
    # Text aggregation check
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    is_aggregated = False
    
    if seq_len != len(tokens):
        is_aggregated = True

    # Compute Sum/Norm separated if possible
    summed_np = None
    norm_np = None
    
    if not is_aggregated:
        try:
            device = input_ids.device
            with torch.no_grad():
                if hasattr(encoder_model, "embeddings"): # BERT
                    segment_ids = inputs.get("token_type_ids")
                    if segment_ids is None: segment_ids = torch.zeros_like(input_ids)
                    
                    word_embed = encoder_model.embeddings.word_embeddings(input_ids)
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_embed = encoder_model.embeddings.position_embeddings(position_ids)
                    seg_embed = encoder_model.embeddings.token_type_embeddings(segment_ids)
                    summed = word_embed + pos_embed + seg_embed
                    normalized = encoder_model.embeddings.LayerNorm(summed)
                else: # GPT-2
                    word_embed = encoder_model.wte(input_ids)
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    pos_embed = encoder_model.wpe(position_ids)
                    summed = word_embed + pos_embed
                    normalized = summed

            summed_np = summed[0].cpu().numpy()
            norm_np = normalized[0].cpu().numpy()
        except:
            is_aggregated = True

    # Combined vector for PCA
    combined_vectors = hidden_states[0][0].cpu().numpy() if hidden_states else None
    
    if combined_vectors is None:
        return ui.HTML("<p>No data available</p>")

    # Generate Heatmap HTML
    rows = []
    if not is_aggregated and summed_np is not None:
        header = "<tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Sum</th><th>LayerNorm</th></tr>"
        for i, tok in enumerate(tokens):
            clean_tok = tok.replace("##", "").replace("Ġ", "")
            sum_strip = array_to_base64_img(summed_np[i][:96], "Blues", 0.15)
            norm_strip = array_to_base64_img(norm_np[i][:96], "Blues", 0.15)
            rows.append(
                f"<tr>"
                f"<td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{sum_strip}' title='Sum'></td>"
                f"<td><img class='heatmap' src='data:image/png;base64,{norm_strip}' title='LayerNorm'></td>"
                f"</tr>"
            )
    else:
        header = "<tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Combined Vector</th></tr>"
        for i, tok in enumerate(tokens):
            clean_tok = tok.replace("##", "").replace("Ġ", "")
            if i < len(combined_vectors):
                vec_strip = array_to_base64_img(combined_vectors[i][:96], "Blues", 0.15)
                rows.append(
                    f"<tr>"
                    f"<td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
                    f"<td><img class='heatmap' src='data:image/png;base64,{vec_strip}' title='Combined Sum+Norm'></td>"
                    f"</tr>"
                )
        if is_aggregated:
            rows.insert(0, "<tr><td colspan='2' style='color:#6b7280;font-size:10px;text-align:center;'>Aggregated view (Word Level). Showing combined vector.</td></tr>")

    html_heatmap = f"<table class='combined-summary-table distribute-cols'>{header}{''.join(rows)}</table>"

    return _render_dual_tab_view(unique_id, html_heatmap, tokens, combined_vectors)


def get_qkv_table(res, layer_idx, top_k=3):
    tokens, _, _, _, hidden_states, _, _, encoder_model, *_ = res
    layer_block = get_layer_block(encoder_model, layer_idx)
    hs_in = hidden_states[layer_idx]
    unique_id = "qkv_tab"

    Q, K, V = extract_qkv(layer_block, hs_in)
    n = len(tokens)

    # Compute L2 norms for Q, K, V
    q_norms = [np.linalg.norm(Q[i]) for i in range(n)]
    k_norms = [np.linalg.norm(K[i]) for i in range(n)]
    v_norms = [np.linalg.norm(V[i]) for i in range(n)]

    # Compute Q-K cosine similarity (which tokens align in attention space)
    qk_sim = _compute_cosine_similarity_matrix(Q) @ _compute_cosine_similarity_matrix(K).T
    # Normalize to [0,1] range approximately
    qk_sim = (qk_sim + 1) / 2

    # 1. Norm View
    norm_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        norm_rows.append(
            f"<tr>"
            f"<td class='token-name'>{clean_tok}</td>"
            f"<td class='norm-value q-norm'>{q_norms[i]:.1f}</td>"
            f"<td class='norm-value k-norm'>{k_norms[i]:.1f}</td>"
            f"<td class='norm-value v-norm'>{v_norms[i]:.1f}</td>"
            f"</tr>"
        )
    html_norm = f"""
    <table class='combined-summary-table'>
        <tr><th>Token</th><th>Q Norm</th><th>K Norm</th><th>V Norm</th></tr>
        {''.join(norm_rows)}
    </table>
    """

    # 2. Alignment View (Sim)
    sim_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        # Get top-k keys this query attends to
        sims = qk_sim[i].copy()
        top_indices = np.argsort(sims)[::-1][:top_k]
        neighbors = []
        for j in top_indices:
            other_tok = tokens[j].replace("##", "").replace("Ġ", "")
            sim_val = sims[j]
            neighbors.append(f"<span class='sim-neighbor qk-neighbor' title='{sim_val:.3f}'>{other_tok}</span>")
        
        sim_rows.append(
             f"<tr><td class='token-name'>{clean_tok}</td><td class='sim-neighbors'>{' '.join(neighbors)}</td></tr>"
        )
    
    html_sim = f"""
    <table class='combined-summary-table'>
        <tr><th>Token</th><th>Q·K Alignment (Potential Attention)</th></tr>
        {''.join(sim_rows)}
    </table>
    """

    # 3. PCA View
    html_pca = _render_qkv_pca_scatter(tokens, Q, K, V)

    # 4. Raw Vector View (Table based for consistency)
    vec_rows = []
    for i, tok in enumerate(tokens):
        display_tok = tok.replace("##", "").replace("Ġ", "")
        q_strip = array_to_base64_img(Q[i][:48], "Greens", 0.12)
        k_strip = array_to_base64_img(K[i][:48], "Oranges", 0.12)
        v_strip = array_to_base64_img(V[i][:48], "Purples", 0.12)
        q_tip = "Query: " + ", ".join(f"{x:.3f}" for x in Q[i][:24])
        k_tip = "Key: " + ", ".join(f"{x:.3f}" for x in K[i][:24])
        v_tip = "Value: " + ", ".join(f"{x:.3f}" for x in V[i][:24])

        vec_rows.append(
            f"<tr>"
            f"<td class='token-name'>{display_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{q_strip}' title='{q_tip}'></td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{k_strip}' title='{k_tip}'></td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{v_strip}' title='{v_tip}'></td>"
            f"</tr>"
        )
    
    html_vec = f"""
    <table class='combined-summary-table'>
        <tr><th>Token</th><th>Q Vector</th><th>K Vector</th><th>V Vector</th></tr>
        {''.join(vec_rows)}
    </table>
    """

    # Assemble Tabbed Interface
    html = f"""
    <div id='{unique_id}'>
        <div class='view-controls'>
            <button class='view-btn active' data-tab='norm' onclick="switchView('{unique_id}', 'norm')" title="View vector magnitude (L2 Norm)">Norms</button>
            <button class='view-btn' data-tab='sim' onclick="switchView('{unique_id}', 'sim')" title="Cosine similarity between tokens">Alignment</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}', 'pca')" title="2D Principal Component Analysis projection">PCA</button>
            <button class='view-btn' data-tab='vec' onclick="switchView('{unique_id}', 'vec')" title="Visual heatmap of raw vector values">Raw Vectors</button>
        </div>

        <div class='card-scroll vector-summary-container'>
            <div id='{unique_id}_norm' class='view-pane' style='display:block;'>
                {html_norm}
            </div>
            <div id='{unique_id}_sim' class='view-pane'>
                {html_sim}
            </div>
            <div id='{unique_id}_pca' class='view-pane'>
                <div class='pca-container-simple'>
                    {html_pca}
                </div>
            </div>
            <div id='{unique_id}_vec' class='view-pane'>
                {html_vec}
            </div>
        </div>
    </div>
    """
    return ui.HTML(html)


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
    unique_id = f"addnorm_{layer_idx}"
    
    # 1. Change View (Bars)
    change_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        diff = np.linalg.norm(hs_out[i] - hs_in[i])
        norm = np.linalg.norm(hs_in[i]) + 1e-6
        ratio = diff / norm
        width = max(4, min(100, int(ratio * 80)))
        change_rows.append(
            f"<tr><td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
            f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;' title='Change: {ratio:.1%}'>"
            f"<div style='width:{width}%;height:10px;border-radius:999px;"
            f"background:linear-gradient(90deg,#ff5ca9,#3b82f6);'></div></div></td></tr>"
        )
    html_change = f"<table class='combined-summary-table'><tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Change Magnitude</th></tr>{''.join(change_rows)}</table>"
    
    # 2. Raw Vectors View (Heatmap of Output)
    rows = []
    header = "<tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Sub-Layer Output (Heatmap)</th></tr>"
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        vec_strip = array_to_base64_img(hs_out[i][:96], "Blues", 0.15)
        rows.append(
            f"<tr>"
            f"<td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{vec_strip}' title='Output vector after Add & Norm'></td>"
            f"</tr>"
        )
    html_heatmap = f"<table class='combined-summary-table distribute-cols'>{header}{''.join(rows)}</table>"
    
    return _render_dual_tab_view(unique_id, html_heatmap, tokens, hs_out, html_change=html_change)


def get_ffn_view(res, layer_idx):
    tokens, _, _, _, hidden_states, _, _, encoder_model, *_ = res
    if layer_idx + 1 >= len(hidden_states):
        return ui.HTML("")
    unique_id = f"ffn_{layer_idx}"
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
    header = "<tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Activation</th><th>Projection</th></tr>"
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        inter_strip = array_to_base64_img(inter_np[i][:96], "Blues", 0.15)
        proj_strip = array_to_base64_img(proj_np[i][:96], "Blues", 0.15)
        rows.append(
            f"<tr>"
            f"<td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{inter_strip}' title='Intermediate Activation'></td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{proj_strip}' title='Projection (FFN Output)'></td>"
            f"</tr>"
        )
        
    html_heatmap = f"<table class='combined-summary-table distribute-cols'>{header}{''.join(rows)}</table>"
    
    return _render_dual_tab_view(unique_id, html_heatmap, tokens, proj_np)


def get_add_norm_post_ffn_view(res, layer_idx):
    tokens, _, _, _, hidden_states, *_ = res
    if layer_idx + 2 >= len(hidden_states):
        return ui.HTML("")
    hs_mid = hidden_states[layer_idx + 1][0].cpu().numpy()
    hs_out = hidden_states[layer_idx + 2][0].cpu().numpy()
    unique_id = f"addnormpost_{layer_idx}"
    
    # 1. Change View (Bars)
    change_rows = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        diff = np.linalg.norm(hs_out[i] - hs_mid[i])
        norm = np.linalg.norm(hs_mid[i]) + 1e-6
        ratio = diff / norm
        width = max(4, min(100, int(ratio * 80)))
        change_rows.append(
            f"<tr><td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
            f"<td><div style='background:#e5e7eb;border-radius:999px;height:10px;' title='Change: {ratio:.1%}'>"
            f"<div style='width:{width}%;height:10px;border-radius:999px;"
            f"background:linear-gradient(90deg,#ff5ca9,#3b82f6);'></div></div></td></tr>"
        )
    html_change = f"<table class='combined-summary-table'><tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Residual Change (FFN)</th></tr>{''.join(change_rows)}</table>"
    
    # 2. Raw Vectors View (Heatmap)
    rows = []
    header = "<tr><th style='text-align:left;padding-left:8px;'>Token</th><th>Sub-Layer Output (Heatmap)</th></tr>"
    for i, tok in enumerate(tokens):
        clean_tok = tok.replace("##", "").replace("Ġ", "")
        vec_strip = array_to_base64_img(hs_out[i][:96], "Blues", 0.15)
        rows.append(
            f"<tr>"
            f"<td class='token-name' style='text-align:left;padding-left:8px;'>{clean_tok}</td>"
            f"<td><img class='heatmap' src='data:image/png;base64,{vec_strip}' title='Output vector after Add & Norm'></td>"
            f"</tr>"
        )
    
    html_heatmap = f"<table class='combined-summary-table distribute-cols'>{header}{''.join(rows)}</table>"
    
    return _render_dual_tab_view(unique_id, html_heatmap, tokens, hs_out, html_change=html_change)


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
                <td style='font-size:9px;color:#374151;white-space:nowrap;padding-left:12px;'>
                    μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}
                </td>
            </tr>
        """)

    return ui.HTML(
        "<div class='card-scroll vector-summary-container'>"
        "<table class='combined-summary-table distribute-cols'>"
        "<tr><th>Token</th><th>Vector (64 dims)</th><th style='padding-left:12px;'>Statistics</th></tr>"
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
