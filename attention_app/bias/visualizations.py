"""Visualization components for bias analysis.

Visualizations for GUS-Net bias detection results:
- Inline text highlighting with bias annotations
- Token-level bias heatmaps
- Attention x Bias matrices
- Method info panel
"""

import html as html_lib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Union
from .attention_bias import HeadBiasMetrics


# ── Colour mapping shared across views ──
BIAS_COLORS = {
    "GEN": {"bg": "rgba(249, 115, 22, 0.18)", "border": "#f97316", "text": "#ea580c", "label": "Generalization"},
    "UNFAIR": {"bg": "rgba(239, 68, 68, 0.18)", "border": "#ef4444", "text": "#dc2626", "label": "Unfair Language"},
    "STEREO": {"bg": "rgba(156, 39, 176, 0.18)", "border": "#9c27b0", "text": "#7b1fa2", "label": "Stereotype"},
}









def create_attention_bias_matrix(
    bias_matrix: np.ndarray,
    metrics: Optional[List[HeadBiasMetrics]] = None,
    selected_layer: Optional[int] = None,
    bar_threshold: float = 1.5,
) -> go.Figure:
    """Create heatmap showing attention to bias for each (layer, head).

    Args:
        bias_matrix: Matrix of shape [num_layers, num_heads]
        metrics: Optional list of HeadBiasMetrics for enhanced tooltips
        selected_layer: Optional integer index of the layer to highlight

    Returns:
        Plotly Figure object
    """
    num_layers, num_heads = bias_matrix.shape

    # Handle all-zeros or empty (precaution)
    if bias_matrix.size == 0 or np.all(bias_matrix == 0):
        # Return a "blank" but valid figure to avoid white-out
        fig = go.Figure()
        fig.update_layout(
             title=dict(text="No bias attention detected (values are 0)", font=dict(size=14, color="#64748b")),
             xaxis=dict(visible=False),
             yaxis=dict(visible=False),
             plot_bgcolor="rgba(0,0,0,0)",
             paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Create hover text
    if metrics:
        # Create lookup dict
        metrics_dict = {(m.layer, m.head): m for m in metrics}
        hover_text = []
        for layer in range(num_layers):
            row = []
            for head in range(num_heads):
                m = metrics_dict.get((layer, head))
                if m:
                    text = (
                        f"<b>Layer {layer}, Head {head}</b><br>"
                        f"BAR (μ̂/μ₀): {m.bias_attention_ratio:.3f}<br>"
                        f"BSR (self-reinforcement): {m.amplification_score:.3f}<br>"
                        f"Max α→biased: {m.max_bias_attention:.3f}<br>"
                        f"Specialised: {'Yes' if m.specialized_for_bias else 'No'}"
                    )
                else:
                    text = f"<b>Layer {layer}, Head {head}</b><br>BAR: {bias_matrix[layer, head]:.3f}"
                row.append(text)
            hover_text.append(row)
    else:
        hover_text = [[f"<b>L{l}, H{h}</b><br>Ratio: {bias_matrix[l,h]:.3f}"
                       for h in range(num_heads)] for l in range(num_layers)]

    # Dynamic range: ensure 1.0 is always central
    # Range is [0, max(2.5, data_max)]
    # This ensures 1.0 (neutral) aligns with White in the colorscale below
    max_val = float(bias_matrix.max())
    z_max = max(2.5, max_val)
    z_min = 0.0

    # Custom colorscale (matches z_max alignment)
    # We want:
    # 0.0 (Blue) -> 1.0 (White) -> >1.5 (Red)
    # Since we map [0, z_max], we need to find where 1.0 falls.
    # mid_point = 1.0 / z_max
    
    mid_point = 1.0 / z_max if z_max > 0 else 0.5
    
    # Construct scale: 0->Blue, mid->White, 1->Red
    # Plotly colorscales are [normalized_val, color]
    colorscale = [
        [0.0, '#dbeafe'],        # 0.0: Light Blue
        [mid_point, '#ffffff'],  # 1.0: White (Neutral)
        [min(1.0, mid_point + (0.5/z_max)), '#fecaca'], # ~1.5: Light Red
        [1.0, '#dc2626']         # Max: Dark Red
    ]

    fig = go.Figure(data=go.Heatmap(
        z=bias_matrix.tolist(),  # Convert to list for reliable serialization
        x=[f"H{h}" for h in range(num_heads)],
        y=[f"L{l}" for l in range(num_layers)],
        colorscale=colorscale,
        zmin=z_min,
        zmax=z_max,
        zauto=False,  # Force explicit colorscale application
        showscale=True,
        colorbar=dict(
            title=dict(
                text="BAR<br>(μ̂/μ₀)",
                side="right",
                font=dict(size=11, color="#64748b")
            ),
            tickfont=dict(size=10, color="#64748b")
        ),
        hovertemplate="%{text}<extra></extra>",
        text=hover_text
    ))

    # Add shape for selected layer
    if selected_layer is not None and 0 <= selected_layer < num_layers:
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=num_heads - 0.5,
            y0=selected_layer - 0.5,
            y1=selected_layer + 0.5,
            line=dict(
                color="#ec4899",  # Pink highlight
                width=3,
            ),
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        title=dict(
            text="Attention Head Bias Specialization<br><sub>Which heads focus on biased tokens?</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Attention Head",
            tickfont=dict(size=10, color="#475569"),
            side="bottom"
        ),
        yaxis=dict(
            title="Layer",
            tickfont=dict(size=10, color="#475569"),
            autorange="reversed"  # Layer 0 at top
        ),
        autosize=True,
        height=max(300, num_layers * 40),  # Explicit height for reliable rendering
        margin=dict(l=60, r=150, t=100, b=60, autoexpand=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )

    # Add reference annotation with formula and interpretation
    fig.add_annotation(
        x=1.25,
        y=0.0,
        xref="paper",
        yref="paper",
        text=(
            "<b>BAR(l, h) = μ̂<sub>B</sub> / μ₀</b><br>"
            "<span style='font-size:9px'>observed / expected attention</span><br><br>"
            f"<span style='color:#dc2626'>■</span> <b>≥ {bar_threshold:.1f}</b> Specialised<br>"
            "<span style='color:#93c5fd'>■</span> <b>= 1.0</b> Uniform<br>"
            "<span style='color:#3b82f6'>■</span> <b>&lt; 1.0</b> Under-attends"
        ),
        showarrow=False,
        font=dict(size=10, color="#64748b"),
        align="left",
        xanchor="left"
    )

    return fig


def create_bias_propagation_plot(
    layer_propagation: List[float],
    selected_layer: Optional[int] = None
) -> go.Figure:
    """Create line plot showing how bias attention changes across layers.

    Args:
        layer_propagation: List of average bias ratios per layer
        selected_layer: Optional layer index to highlight
    """
    layers = list(range(len(layer_propagation)))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers,
        y=layer_propagation,
        mode='lines+markers',
        line=dict(color='#ff5ca9', width=3),
        marker=dict(size=8, color='#ff5ca9', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(255, 92, 169, 0.1)',
        name='BAR',
        hovertemplate="<b>Layer %{x}</b><br>Avg BAR: %{y:.3f}<extra></extra>"
    ))

    # Add reference line at 1.0 (neutral)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Neutral (1.0)",
        annotation_position="right",
        annotation_font=dict(size=10, color="#64748b")
    )

    # Add arrow annotation for selected layer if provided
    if selected_layer is not None and 0 <= selected_layer < len(layers):
        fig.add_annotation(
            x=selected_layer,
            y=layer_propagation[selected_layer],
            text="Selected",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#64748b",
            ax=0,
            ay=-40,
            font=dict(size=10, color="#64748b"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            borderpad=4,
            opacity=0.9
        )

    fig.update_layout(
        title=dict(
            text="Bias Attention Propagation Across Layers<br><sub>How bias focus evolves through the network</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Layer",
            tickmode="linear",
            tick0=0,
            dtick=1,
            gridcolor='#e2e8f0',
            tickfont=dict(size=11, color="#64748b")
        ),
        yaxis=dict(
            title="Avg BAR (μ̂ / μ₀)",
            gridcolor='#e2e8f0',
            tickfont=dict(size=11, color="#64748b"),
            rangemode="tozero"
        ),
        autosize=True,
        height=400,  # Explicit height for reliable rendering
        margin=dict(l=80, r=40, t=100, b=60, autoexpand=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        hovermode='x unified'
    )

    return fig


def create_combined_bias_visualization(
    tokens: List[str],
    token_labels: List[Dict],
    attention_matrix: np.ndarray,
    layer_idx: int,
    head_idx: int,
    selected_token_idx: Optional[Union[int, List[int]]] = None,
) -> go.Figure:
    """Create attention heatmap matching the Multi-Head Attention style.

    By default no tokens are highlighted.  When *selected_token_idx* is
    provided, that token's row and column are highlighted using its
    primary bias-category colour (GEN → orange, STEREO → purple,
    UNFAIR → red).
    """
    n = len(tokens)
    
    # Normalize selection to a set of indices
    selected_indices = set()
    if isinstance(selected_token_idx, list):
        selected_indices = set(selected_token_idx)
    elif isinstance(selected_token_idx, int) and selected_token_idx >= 0:
        selected_indices = {selected_token_idx}
        
    biased_indices = set(
        i for i, label in enumerate(token_labels) if label.get("is_biased")
    )

    # Category-specific highlight palettes
    _CAT_HIGHLIGHT = {
        "GEN":    {"fill": "rgba(249, 115, 22, 0.14)", "border": "#f97316"},
        "UNFAIR": {"fill": "rgba(239, 68, 68, 0.14)",  "border": "#ef4444"},
        "STEREO": {"fill": "rgba(156, 39, 176, 0.14)", "border": "#9c27b0"},
    }
    _DEFAULT_HL = {"fill": "rgba(236, 72, 153, 0.12)", "border": "#ec4899"}

    # ── Clean token labels (remove subword markers, deduplicate) ──
    from collections import Counter
    base = [t.replace("##", "").replace("\u0120", "") for t in tokens]
    counts = Counter(base)
    occ: Dict[str, int] = {}
    cleaned = []
    for t in base:
        if counts[t] > 1:
            occ[t] = occ.get(t, 0) + 1
            cleaned.append(f"{t}_{occ[t]}")
        else:
            cleaned.append(t)

    sel_hl_map = {} # map idx -> color info
    for idx in selected_indices:
        if 0 <= idx < n:
            lbl = token_labels[idx]
            if lbl.get("is_biased") and lbl.get("bias_types"):
                primary = lbl["bias_types"][0]
                sel_hl_map[idx] = _CAT_HIGHLIGHT.get(primary, _DEFAULT_HL)
            else:
                sel_hl_map[idx] = _DEFAULT_HL

    # ── HTML-styled tick labels ──
    styled = []
    for i, tok in enumerate(cleaned):
        if i in selected_indices:
            color = sel_hl_map[i]["border"]
            styled.append(
                f"<span style='color:{color};font-weight:bold'>{tok}</span>"
            )
        else:
            styled.append(tok)

    # ── Attention colorscale (matches attention tab) ──
    att_colorscale = [
        [0.0, '#ffffff'],
        [0.1, '#f0f9ff'],
        [0.3, '#bae6fd'],
        [0.6, '#3b82f6'],
        [1.0, '#1e3a8a'],
    ]

    # ── Compute totals for hover ──
    attn_received = attention_matrix.sum(axis=0)
    attn_sent = attention_matrix.sum(axis=1)

    hover_text = []
    for i in range(n):
        row = []
        for j in range(n):
            parts = [
                f"<b>Query:</b> {cleaned[i]}",
                f"<b>Key:</b> {cleaned[j]}",
                f"<b>Attention:</b> {attention_matrix[i, j]:.4f}",
            ]
            if j in biased_indices:
                lbl = token_labels[j]
                types = ", ".join(lbl.get("bias_types", []))
                parts.append(f"<b style='color:#ec4899'>Key bias: {types}</b>")
                parts.append(f"Attn received: {attn_received[j]:.3f}")
            if i in biased_indices:
                lbl = token_labels[i]
                types = ", ".join(lbl.get("bias_types", []))
                parts.append(f"<b style='color:#f97316'>Query bias: {types}</b>")
            row.append("<br>".join(parts))
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix.tolist(),
        x=list(range(n)),
        y=list(range(n)),
        colorscale=att_colorscale,
        zmin=0, zmax=float(attention_matrix.max()) if attention_matrix.max() > 0 else 1,
        colorbar=dict(
            title=dict(text="Attention", font=dict(color="#64748b", size=11)),
            tickfont=dict(color="#64748b", size=10),
        ),
        hovertemplate="%{text}<extra></extra>",
        text=hover_text,
    ))

    # ── Row/column highlight for the SELECTED token only ──
    for idx in selected_indices:
        if 0 <= idx < n:
            hl = sel_hl_map.get(idx, _DEFAULT_HL)
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=n - 0.5,
                y0=idx - 0.5, y1=idx + 0.5,
                fillcolor=hl["fill"],
                line=dict(color=hl["border"], width=1),
                layer="above",
            )
            fig.add_shape(
                type="rect",
                x0=idx - 0.5, x1=idx + 0.5,
                y0=-0.5, y1=n - 0.5,
                fillcolor=hl["fill"],
                line=dict(color=hl["border"], width=1),
                layer="above",
            )

    fig.update_layout(
        title=dict(
            text=f"Attention × Bias — Layer {layer_idx}, Head {head_idx}",
            x=0.5, y=0.98, xanchor="center", yanchor="top",
            font=dict(size=14, color="#334155", family="Inter, sans-serif"),
        ),
        xaxis=dict(
            title=dict(text="Key (attending to)", font=dict(size=11)),
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=styled,
            tickfont=dict(size=10),
            side="bottom",
        ),
        yaxis=dict(
            title=dict(text="Query (attending from)", font=dict(size=11)),
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=styled,
            tickfont=dict(size=10),
            autorange="reversed",
        ),
        autosize=True,
        height=max(500, n * 30),  # Explicit height for reliable rendering
        margin=dict(l=40, r=10, t=40, b=40, autoexpand=True),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", family="Inter, system-ui, sans-serif"),
    )

    return fig


def create_inline_bias_html(
    text: str,
    token_labels: List[Dict],
    bias_spans: list,
    show_neutral: bool = False,
    threshold: float = 0.5,
) -> str:
    """Render original text with inline highlighted bias spans.

    Returns an HTML string where biased spans are wrapped in coloured
    ``<span>`` elements with hover tooltips showing category, score,
    method and explanation.
    """
    # Filter spans if show_neutral is False
    display_spans = bias_spans
    if not show_neutral:
        display_spans = [s for s in bias_spans if s.get("avg_score", 0) >= 0.35]

    if not display_spans:
        return (
            '<div class="bias-inline-text">'
            f'<p style="font-size:15px;line-height:1.8;">{html_lib.escape(text)}</p>'
            '<div style="color:#10b981;font-size:13px;margin-top:12px;">'
            'No bias patterns detected' + (' above threshold' if not show_neutral and bias_spans else '') + '.</div></div>'
        )

    # Reconstruct text from tokens, mapping token indices → char positions
    tokens = [l["token"] for l in token_labels]
    char_positions = _align_tokens_to_text(text, tokens)

    # Build a set of char ranges that are biased
    highlighted_ranges = []  # (start_char, end_char, primary_type, tooltip_html)
    for span in display_spans:
        s_start, s_end = span["start_idx"], span["end_idx"]
        s_types = span["bias_types"]
        s_score = span.get("avg_score", 1.0)
        s_explanation = span.get("explanation", "")

        # Find char range for this span
        if s_start >= len(char_positions) or s_end >= len(char_positions):
            continue
        char_start = char_positions[s_start][0]
        char_end = char_positions[s_end][1]
        if char_start is None or char_end is None:
            continue

        primary_type = s_types[0] if s_types else "GEN"
        type_badges = " ".join(
            f'<span class="bias-method-badge" style="background:{BIAS_COLORS.get(t, BIAS_COLORS["GEN"])["border"]};'
            f'color:white;">{t}</span>'
            for t in s_types
        )

        tooltip = (
            f'<div class="bias-tooltip">'
            f'<div style="margin-bottom:8px;">'
            f'{type_badges}'
            f'</div>'
            f'<div class="bias-score-row">'
            f'<span>Confidence:</span>'
            f'<div class="bias-score-bar-container">'
            f'<div class="bias-score-bar" style="width:{s_score*100:.0f}%;'
            f'background:{BIAS_COLORS.get(primary_type, BIAS_COLORS["GEN"])["border"]};"></div>'
            f'</div>'
            f'<span style="font-weight:700; color:#fff;">{s_score:.2f}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:#cbd5e1;margin-top:6px; font-style:italic;">"{html_lib.escape(s_explanation)}"</div>'
            f'<div style="font-size:9px;color:#64748b;margin-top:6px;border-top:1px solid rgba(255,255,255,0.08);padding-top:5px;">'
            f'Threshold: {threshold:.2f} &middot; Method: GUS-Net</div>'
            f'</div>'
        )
        highlighted_ranges.append((char_start, char_end, primary_type, tooltip))

    # Sort by start position
    highlighted_ranges.sort(key=lambda x: x[0])

    # Build HTML
    parts = []
    cursor = 0
    for start, end, ptype, tooltip in highlighted_ranges:
        if start < cursor:
            continue  # overlapping span, skip
        if start > cursor:
            parts.append(html_lib.escape(text[cursor:start]))
        color_info = BIAS_COLORS.get(ptype, BIAS_COLORS["GEN"])
        parts.append(
            f'<span class="bias-highlight" style="background:{color_info["bg"]};'
            f'border-bottom:2px solid {color_info["border"]};">'
            f'{html_lib.escape(text[start:end])}'
            f'{tooltip}</span>'
        )
        cursor = end
    if cursor < len(text):
        parts.append(html_lib.escape(text[cursor:]))

    # Legend
    legend_items = []
    for cat, info in BIAS_COLORS.items():
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:16px;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
            f'background:{info["bg"]};border-bottom:2px solid {info["border"]};"></span>'
            f'<span style="font-size:11px;color:#64748b;">{info["label"]}</span></span>'
        )

    return (
        '<div class="bias-inline-text">'
        f'<p style="font-size:15px;line-height:2.4;font-family:Inter,system-ui,sans-serif;'
        f'color:#1e293b;">{"".join(parts)}</p>'
        f'<div style="margin-top:16px;padding-top:12px;border-top:1px solid #e2e8f0;'
        f'display:flex;flex-wrap:wrap;gap:8px;">{"".join(legend_items)}</div>'
        '</div>'
    )


def _align_tokens_to_text(text: str, tokens: List[str]) -> List[tuple]:
    """Align sub-word tokens back to character positions in original text.

    Returns a list of (start_char, end_char) tuples, one per token.
    Special tokens ([CLS], [SEP], etc.) get (None, None).
    """
    positions = []
    cursor = 0
    text_lower = text.lower()

    for token in tokens:
        if token.startswith("[") and token.endswith("]"):
            positions.append((None, None))
            continue

        # Handle BERT sub-word prefix
        clean = token.replace("##", "").replace("Ġ", "")  # BERT ## and GPT-2 Ġ
        clean_lower = clean.lower()

        if not clean_lower:
            positions.append((None, None))
            continue

        # Search forward from cursor
        idx = text_lower.find(clean_lower, cursor)
        if idx == -1:
            # Fallback: try from beginning (shouldn't normally happen)
            idx = text_lower.find(clean_lower)
        if idx == -1:
            positions.append((None, None))
            continue

        positions.append((idx, idx + len(clean)))
        cursor = idx + len(clean)

    return positions


def create_method_info_html(model_key: str = "gusnet-bert") -> str:
    """Render a compact, visually rich info panel for GUS-Net.
    
    Shows different information depending on the selected model:
    - gusnet-bert: Original ethical-spectacle/social-bias-ner
    - gusnet-bert-large, gusnet-gpt2, gusnet-gpt2-medium: Custom-trained
    """
    # Category badges
    cat_badges = []
    for cat, info in BIAS_COLORS.items():
        cat_badges.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;'
            f'background:{info["bg"]};color:{info["text"]};'
            f'border:1px solid {info["border"]}30;">'
            f'{cat}</span>'
        )
    badges_html = " ".join(cat_badges)

    def bullet(title, text):
        return (
            f'<div style="display:flex;gap:6px;align-items:flex-start;margin-bottom:4px;">'
            f'<span style="color:#ff5ca9;font-size:7px;margin-top:4px;flex-shrink:0;">'
            f'&#9679;</span>'
            f'<span style="color:#64748b;font-size:10px;line-height:1.5;">'
            f'<b style="color:#475569;">{title}:</b> {text}</span></div>'
        )

    # Model-specific information
    MODEL_INFO = {
        "gusnet-bert": {
            "display_name": "GUS-Net (BERT)",
            "model_path": "ethical-spectacle/social-bias-ner",
            "base_model": "bert-base-uncased",
            "is_original": True,
            "paper": "arxiv.org/pdf/2410.08388",
            "huggingface": "ethical-spectacle/gus-net",
        },
        "gusnet-bert-large": {
            "display_name": "GUS-Net (BERT Large)",
            "model_path": "models/gusnet-bert-large",
            "base_model": "bert-large-uncased",
            "is_original": False,
            "training_script": "gus_net_training.py",
        },
        "gusnet-gpt2": {
            "display_name": "GUS-Net (GPT-2)",
            "model_path": "models/gusnet-gpt2",
            "base_model": "gpt2",
            "is_original": False,
            "training_script": "gus_net_training.py",
        },
        "gusnet-gpt2-medium": {
            "display_name": "GUS-Net (GPT-2 Medium)",
            "model_path": "models/gusnet-gpt2-medium",
            "base_model": "gpt2-medium",
            "is_original": False,
            "training_script": "gus_net_training.py",
        },
    }

    info = MODEL_INFO.get(model_key, MODEL_INFO["gusnet-bert"])
    
    # Build different specs based on whether it's the original model or custom-trained
    if info["is_original"]:
        specs_html = (
            '<span style="color:#94a3b8;font-weight:600;">Model</span>'
            '<code style="font-size:10px;padding:1px 6px;background:rgba(148,163,184,0.1);'
            'border-radius:3px;color:#475569;font-family:JetBrains Mono,monospace;">'
            f'{info["model_path"]}</code>'

            '<span style="color:#94a3b8;font-weight:600;">Paper</span>'
            f'<a href="https://arxiv.org/pdf/2410.08388" target="_blank" '
            'style="color:#3b82f6;text-decoration:none;font-size:10px;">'
            f'{info["paper"]}</a>'

            '<span style="color:#94a3b8;font-weight:600;">HuggingFace</span>'
            f'<a href="https://huggingface.co/collections/ethical-spectacle/gus-net" target="_blank" '
            'style="color:#3b82f6;text-decoration:none;font-size:10px;">'
            f'{info["huggingface"]}</a>'

            '<span style="color:#94a3b8;font-weight:600;">Technique</span>'
            '<span style="color:#475569;">Multi-label token classification (sigmoid)</span>'

            '<span style="color:#94a3b8;font-weight:600;">Categories</span>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{badges_html}</div>'
        )
        limitations_html = (
            bullet('Sub-word split',
                   'BERT tokenization may fragment entities, lowering scores on pieces')
            + bullet('Ambiguity',
                     'Some adjectives (e.g. <em>emotional</em>) score high regardless of context')
            + bullet('Fixed scope',
                     'Trained on specific datasets &mdash; may miss niche or subtle bias')
        )
    else:
        # Custom-trained models
        base_model = info["base_model"]
        tokenization_type = "BERT" if "bert" in base_model else "GPT-2"
        specs_html = (
            '<span style="color:#94a3b8;font-weight:600;">Base Model</span>'
            '<code style="font-size:10px;padding:1px 6px;background:rgba(148,163,184,0.1);'
            f'border-radius:3px;color:#475569;font-family:JetBrains Mono,monospace;">'
            f'{base_model}</code>'

            '<span style="color:#94a3b8;font-weight:600;">Training</span>'
            '<code style="font-size:10px;padding:1px 6px;background:rgba(148,163,184,0.1);'
            f'border-radius:3px;color:#475569;font-family:JetBrains Mono,monospace;">'
            f'{info["training_script"]}</code>'

            '<span style="color:#94a3b8;font-weight:600;">Dataset</span>'
            '<span style="color:#475569;">ethical-spectacle/gus-dataset-v1</span>'

            '<span style="color:#94a3b8;font-weight:600;">Technique</span>'
            '<span style="color:#475569;">Multi-label token classification (sigmoid)</span>'

            '<span style="color:#94a3b8;font-weight:600;">Categories</span>'
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{badges_html}</div>'
        )
        limitations_html = (
            bullet('Sub-word split',
                   f'{tokenization_type} tokenization may fragment entities, lowering scores on pieces')
            + bullet('Ambiguity',
                     'Some adjectives (e.g. <em>emotional</em>) score high regardless of context')
            + bullet('Custom training',
                     'Fine-tuned on specific dataset &mdash; may differ from original GUS-Net')
        )

    return (
        '<div style="margin-top:20px;padding:16px 20px;'
        'background:linear-gradient(135deg,rgba(248,250,252,0.8),rgba(241,245,249,0.4));'
        'border:1px solid rgba(226,232,240,0.6);border-radius:12px;">'

        # ── Header ──
        '<div style="margin-bottom:14px;">'
        '<div style="font-size:14px;font-weight:700;color:#1e293b;line-height:1.2;display:flex;align-items:center;gap:6px;">'
        f'{info["display_name"]}'
        '<span style="display:inline-flex;align-items:center;justify-content:center;'
        'width:14px;height:14px;border-radius:50%;background:rgba(59,130,246,0.1);'
        'color:#3b82f6;font-size:9px;font-weight:600;cursor:help;" '
        'title="GUS-Net = Generalization, Unfairness, and Stereotype Network">ℹ</span>'
        '</div>'
        '<div style="font-size:11px;color:#64748b;">'
        'Social Bias Named Entity Recognition</div>'
        '</div>'

        # ── Specs grid ──
        '<div style="display:grid;grid-template-columns:90px 1fr;gap:6px 12px;'
        'font-size:11px;margin-bottom:14px;padding-bottom:14px;'
        'border-bottom:1px dashed rgba(203,213,225,0.5);">'
        + specs_html +
        '</div>'

        # ── Limitations ──
        '<div>'
        '<div style="font-size:9px;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
        'Limitations</div>'
        + limitations_html
        + '</div></div>'
    )


def create_ratio_formula_html() -> str:
    """Render the Bias Attention Ratio formula as an academic-style panel."""

    # ── CSS helper: inline fraction ──
    def frac(num: str, den: str, color: str = "#e2e8f0") -> str:
        return (
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'vertical-align:middle;margin:0 3px;">'
            f'<span style="padding:0 5px 2px;line-height:1.2;">{num}</span>'
            f'<span style="border-top:1.5px solid {color};padding:2px 5px 0;'
            f'line-height:1.2;">{den}</span></span>'
        )

    # Variable colour tokens
    mu_hat = '<span style="color:#60a5fa;">μ&#770;<sub style="font-size:80%;">𝐵</sub></span>'
    mu_hat_lh = f'{mu_hat}<sup style="font-size:75%;color:#94a3b8;">(<i>l,h</i>)</sup>'
    mu_0 = '<span style="color:#f59e0b;">μ<sub style="font-size:80%;">0</sub></span>'
    alpha = '<span style="color:#a5b4fc;">α<sub style="font-size:75%;"><i>ij</i></sub></span>'
    alpha_lh = f'{alpha}<sup style="font-size:70%;color:#94a3b8;">(<i>l,h</i>)</sup>'
    B_set = '<span style="color:#f472b6;">𝐵</span>'

    return (
        '<div style="background:linear-gradient(135deg,#f8fafc,#f0f4f8);'
        'border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-bottom:16px;">'

        # ── Title ──
        '<h4 style="margin:0 0 14px 0;font-size:18px;font-weight:600;color:#0f172a;text-align:center;">Bias Attention Ratio - Definition</h4>'

        # ── Main formula block ──
        '<div style="background:#1e293b;border-radius:8px;padding:14px 18px;margin-bottom:14px;'
        'font-family:JetBrains Mono,monospace;font-size:13px;color:#e2e8f0;'
        'line-height:1.3;text-align:center;">'
        # BAR(l,h) = fraction
        '<div style="margin-bottom:10px;">'
        '<span style="color:#cbd5e1;font-weight:600;">BAR</span>'
        '<span style="color:#94a3b8;font-size:11px;">(<i>l, h</i>)</span>'
        '<span style="margin:0 8px;color:#64748b;">=</span>'
        + frac(mu_hat_lh, mu_0) +
        '<span style="margin:0 8px;color:#64748b;">=</span>'
        + frac(
            f'<span style="color:#94a3b8;font-size:10px;">observed</span>',
            f'<span style="color:#94a3b8;font-size:10px;">expected</span>',
        ) +
        '</div>'

        # ── where clause ──
        '<div style="border-top:1px solid rgba(148,163,184,0.2);'
        'padding-top:10px;margin-top:6px;text-align:left;font-size:11px;line-height:1.8;">'

        # μ̂_B definition
        f'<div>{mu_hat_lh}'
        f'<span style="color:#64748b;margin:0 6px;">=</span>'
        + frac('<span style="color:#e2e8f0;">1</span>',
               '<span style="color:#e2e8f0;"><i>N</i></span>') +
        f'<span style="color:#94a3b8;margin:0 2px;">·</span>'
        f'<span style="color:#94a3b8;">&#8721;<sub style="font-size:70%;"><i>i</i>=1</sub>'
        f'<sup style="font-size:70%;"><i>N</i></sup></span>'
        f'<span style="color:#94a3b8;margin:0 1px;">&#8721;<sub style="font-size:70%;"><i>j</i>∈{B_set}</sub></span>'
        f'<span style="margin-left:3px;">{alpha_lh}</span>'
        f'<span style="color:#475569;font-size:9px;margin-left:10px;font-style:italic;">'
        f'observed mean attention → biased tokens</span></div>'

        # μ₀ definition
        f'<div>{mu_0}'
        f'<span style="color:#64748b;margin:0 6px;">=</span>'
        + frac(f'<span style="color:#e2e8f0;">|{B_set}|</span>',
               '<span style="color:#e2e8f0;"><i>N</i></span>') +
        f'<span style="color:#475569;font-size:9px;margin-left:10px;font-style:italic;">'
        f'expected under uniform distribution</span></div>'

        '</div></div>'

        # ── Variable definitions ──
        '<div style="font-size:10px;color:#94a3b8;line-height:1.7;margin-bottom:12px;'
        'padding:8px 12px;background:rgba(241,245,249,0.5);border-radius:6px;'
        'font-family:JetBrains Mono,monospace;">'
        f'<div>{alpha_lh} '
        '<span style="color:#64748b;">— attention weight from token <i>i</i> '
        'to token <i>j</i> at layer <i>l</i>, head <i>h</i></span></div>'
        f'<div>{B_set} ⊆ &#123;1, …, <i>N</i>&#125; '
        '<span style="color:#64748b;">— indices flagged by GUS-Net</span></div>'
        '<div><span style="color:#e2e8f0;"><i>N</i></span> '
        '<span style="color:#64748b;">— sequence length</span></div>'
        '</div>'

        # ── Interpretation ──
        '<div style="font-size:11px;color:#64748b;line-height:1.7;">'
        '<code style="background:#dbeafe;padding:1px 6px;border-radius:3px;'
        'font-size:10px;color:#1e40af;">= 1.0</code> '
        'Uniform baseline (head treats biased and non-biased tokens equally) '
        '<br>'
        '<code style="background:#fee2e2;padding:1px 6px;border-radius:3px;'
        'font-size:10px;color:#991b1b;">&gt; 1.5</code> '
        '<b>Specialised</b> — head disproportionately attends to biased tokens '
        '<br>'
        '<code style="background:#dbeafe;padding:1px 6px;border-radius:3px;'
        'font-size:10px;color:#1e40af;">&lt; 1.0</code> '
        'Head <b>under-attends</b> biased tokens relative to baseline'
        '</div>'
        '</div>'
    )


def create_token_bias_strip(
    token_labels: List[Dict],
    selected_token_idx: Optional[Union[int, List[int]]] = None
) -> str:
    """Render a compact HTML strip showing per-token bias categories.
    
    Logic:
    - BERT (detected by "##"): Keep split and show "##" (e.g. "nur", "##turing").
    - GPT-2 (detected by "Ġ" absence): Merge subwords.
    """
    # Normalize selection
    selected_indices = set()
    if isinstance(selected_token_idx, list):
        selected_indices = set(selected_token_idx)
    elif isinstance(selected_token_idx, int) and selected_token_idx >= 0:
        selected_indices = {selected_token_idx}

    cats = ["O", "GEN", "UNFAIR", "STEREO"]
    cat_colors = {"O": "#64748b", "GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}

    # Detect global tokenizer mode for safe merging defaults
    has_gpt2_tokens = any("\u0120" in lbl["token"] for lbl in token_labels)
    # If we see Ġ, assume GPT-2 merging logic is active. 
    # But if we see ##, that takes precedence for blocking merge on that specific token.
    
    merged_tokens = []
    current_token = None

    for i, lbl in enumerate(token_labels):
        tok = lbl["token"]
        if tok in ("[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"):
            continue
            
        # Determine if this token is a subword that should be merged
        # It is a merge candidate if:
        # 1. We are in GPT-2 mode (have seen Ġ in sequence)
        # 2. It does NOT have a Ġ (continuation in GPT-2)
        # 3. It does NOT start with ## (BERT subword - keep split)
        # 4. It is not the first token (current_token exists)
        
        is_gpt2_subword = ("\u0120" not in tok) and has_gpt2_tokens
        is_bert_subword = tok.startswith("##")

        # Prepare text for display (remove markers)
        # For BERT ##, we want to KEEP ## in the visual text per user request.
        # For GPT-2 Ġ, we remove it.
        clean_text = tok.replace("\u0120", "").replace("##", "")

        # Standalone punctuation (no alphanumeric chars) should NOT be merged
        is_standalone_punct = clean_text and not any(c.isalnum() for c in clean_text)
        should_merge_this = is_gpt2_subword and not is_bert_subword and (current_token is not None) and not is_standalone_punct
        
        if not clean_text:
            continue

        if should_merge_this:
            # Merge with previous
            current_token["text"] += clean_text
            current_token["is_biased"] = current_token["is_biased"] or lbl.get("is_biased", False)
            current_token["bias_types"] = list(set(current_token["bias_types"] + lbl.get("bias_types", [])))
            for cat, score in lbl.get("scores", {}).items():
                current_token["scores"][cat] = max(current_token["scores"].get(cat, 0), score)
            current_token["indices"].append(lbl.get("index", -1))
        else:
            # New token (or preserved split BERT token)
            current_token = {
                "text": clean_text,
                "is_biased": lbl.get("is_biased", False),
                "bias_types": lbl.get("bias_types", []),
                "scores": lbl.get("scores", {}).copy(),
                "indices": [lbl.get("index", -1)]
            }
            merged_tokens.append(current_token)

    # 2. Render HTML
    cells = []
    for item in merged_tokens:
        text = item["text"]
        is_selected = any(idx in selected_indices for idx in item["indices"])
        is_biased = item["is_biased"]
        
        bg_style = "background:rgba(236,72,153,0.06);" if is_biased else ""
        if is_selected:
            bg_style = "background:rgba(255, 92, 169, 0.2); box-shadow: 0 0 0 2px #ff5ca9; transform: translateY(-1px);"

        types = item["bias_types"]
        scores = item["scores"]

        score_items = []
        for cat in cats:
            active = cat in types
            score = scores.get(cat, 0)
            color = cat_colors[cat] if active else "#64748b"
            opacity = "1" if active else "0.4"
            score_items.append(
                f'<span style="display:inline-flex;align-items:center;gap:2px;opacity:{opacity};">'
                f'<span style="width:6px;height:6px;border-radius:50%;background:{color};'
                f'display:inline-block;"></span>'
                f'<span style="font-size:9px;color:{color};font-family:JetBrains Mono,monospace;">'
                f'{score:.2f}</span></span>'
            )
        scores_row = (
            f'<span style="display:flex;gap:5px;justify-content:center;margin-top:3px;">'
            f'{"".join(score_items)}</span>'
        )

        tooltip = html_lib.escape(text)
        for cat in cats:
            sc = scores.get(cat, 0)
            marker = " *" if cat in types else ""
            tooltip += f"&#10;{cat}: {sc:.3f}{marker}"

        cells.append(
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'padding:4px 6px;border-radius:6px;{bg_style}'
            f'font-family:JetBrains Mono,monospace;font-size:12px;cursor:help;transition:all 0.2s ease;" '
            f'title="{tooltip}">'
            f'<span style="line-height:1.2;color:#0f172a;">{html_lib.escape(text)}</span>'
            f'{scores_row}'
            f'</span>'
        )

    if not cells:
        return '<div style="color:#9ca3af;font-size:12px;">No tokens to display.</div>'

    # Legend
    cat_labels = {"O": "Outside \u2014 neutral, no bias detected", "GEN": "Generalization", "UNFAIR": "Unfair Language", "STEREO": "Stereotype"}
    legend_items = []
    for cat in cats:
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{cat_colors[cat]};display:inline-block;"></span>'
            f'<span style="font-size:9px;color:#6b7280;">{cat_labels[cat]}</span></span>'
        )

    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:3px;padding:12px 0;'
        f'align-items:flex-start;justify-content:center;">{"".join(cells)}</div>'
        f'<div style="display:flex;gap:14px;margin-top:4px;padding-top:8px;'
        f'border-top:1px solid #f1f5f9;justify-content:center;">{"".join(legend_items)}</div>'
    )


def create_bias_sentence_preview(tokens: List[str], token_labels: List[Dict]) -> str:
    """Create a token-viz style sentence preview with bias coloring.
    
    Logic matches create_token_bias_strip:
    - BERT (##): Split and show ##
    - GPT-2 (Ġ): Merge subwords
    """
    token_html = []
    
    # Detect global mode
    has_gpt2_tokens = any("\u0120" in tok for tok in tokens)

    merged_tokens = []
    current_token = None

    for tok, label in zip(tokens, token_labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"):
            continue

        is_gpt2_subword = ("\u0120" not in tok) and has_gpt2_tokens
        is_bert_subword = tok.startswith("##")

        clean_text = tok.replace("\u0120", "").replace("##", "")
        if not clean_text:
            continue

        is_standalone_punct = not any(c.isalnum() for c in clean_text)
        should_merge_this = is_gpt2_subword and not is_bert_subword and (current_token is not None) and not is_standalone_punct

        if should_merge_this:
            # Merge
            current_token["text"] += clean_text
            current_token["is_biased"] = current_token["is_biased"] or label["is_biased"]
            current_token["bias_types"] = list(set(current_token["bias_types"] + label.get("bias_types", [])))
            for cat, score in label.get("scores", {}).items():
                current_token["scores"][cat] = max(current_token["scores"].get(cat, 0), score)
        else:
            current_token = {
                "text": clean_text,
                "is_biased": label.get("is_biased", False),
                "bias_types": label.get("bias_types", []),
                "scores": label.get("scores", {}).copy()
            }
            merged_tokens.append(current_token)

    # 2. Generate HTML
    for item in merged_tokens:
        text = item["text"]
        is_biased = item["is_biased"]
        types = item["bias_types"]
        scores = item["scores"]

        if is_biased and types:
            primary = types[0]
            color_info = BIAS_COLORS.get(primary, BIAS_COLORS["GEN"])
            max_score = max(scores.get(t, 0) for t in types)
            types_str = ", ".join(types)
            tooltip = f"{html_lib.escape(text)}&#10;{types_str} (score: {max_score:.2f})"

            # Build a colour bar with one segment per detected category
            bar_segments = "".join(
                f'<span style="flex:1;background:'
                f'{BIAS_COLORS.get(t, BIAS_COLORS["GEN"])["border"]};"></span>'
                for t in types
            )
            bar_html = (
                f'<span style="position:absolute;bottom:0;left:0;right:0;'
                f'height:3px;display:flex;border-radius:0 0 4px 4px;">'
                f'{bar_segments}</span>'
            )

            style = (
                f"background:{color_info['bg']};"
                f"position:relative;padding-bottom:5px;overflow:hidden;"
            )
            inner = f'{html_lib.escape(text)}{bar_html}'
        else:
            tooltip = f"{html_lib.escape(text)}&#10;No bias detected"
            style = "background:rgba(241,245,249,0.6);"
            inner = html_lib.escape(text)

        token_html.append(
            f'<span class="token-viz" style="{style}" '
            f'title="{tooltip}">{inner}</span>'
        )

    if not token_html:
        return '<div style="color:#9ca3af;font-size:12px;">No tokens to display.</div>'

    legend = []
    for cat, info in BIAS_COLORS.items():
        legend.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
            f'background:{info["bg"]};border-bottom:2px solid {info["border"]};"></span>'
            f'<span style="font-size:9px;color:#6b7280;">{info["label"]}</span></span>'
        )

    legend_html = (
        f'<div style="display:flex;gap:12px;margin-top:8px;font-size:9px;'
        f'color:#6b7280;align-items:center;">{"".join(legend)}</div>'
    )

    return (
        f'<div class="token-viz-container">{"".join(token_html)}</div>'
        f'{legend_html}'
    )


def create_bias_criteria_html(summary: Dict, weights: Optional[Dict] = None) -> str:
    """Render explicit bias-level criteria breakdown.

    Shows *why* the bias level is what it is, with formula and component scores.
    """
    if not weights:
        weights = {"pct": 0.30, "gen": 0.20, "unfair": 0.25, "stereo": 0.25}

    total = max(summary.get("total_tokens", 1), 1)
    pct = summary.get("bias_percentage", 0) / 100
    gen_r = summary.get("generalization_count", 0) / total
    unfair_r = summary.get("unfairness_count", 0) / total
    stereo_r = summary.get("stereotype_count", 0) / total

    # Components (normalized to 1.0)
    c_pct = pct
    c_gen = min(gen_r * 5, 1.0)
    c_unfair = min(unfair_r * 5, 1.0)
    c_stereo = min(stereo_r * 5, 1.0)

    score = (
        weights["pct"] * c_pct
        + weights["gen"] * c_gen
        + weights["unfair"] * c_unfair
        + weights["stereo"] * c_stereo
    )
    score = min(score, 1.0)

    if score < 0.15:
        level, color = "Low", "#10b981"
    elif score < 0.40:
        level, color = "Moderate", "#f59e0b"
    else:
        level, color = "High", "#ef4444"

    # Component bars
    def bar(label, value, weight, bar_color="#ff5ca9"):
        pct_width = min(value * 100, 100)
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
            f'<span style="width:100px;font-size:11px;color:#64748b;text-align:right;">{label}</span>'
            f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden;">'
            f'<div style="width:{pct_width:.0f}%;height:100%;background:{bar_color};border-radius:4px;'
            f'transition:width 0.4s;"></div></div>'
            f'<span style="width:75px;font-size:11px;color:#475569;font-family:JetBrains Mono,monospace;">'
            f'{value:.2f} <small style="color:#94a3b8;">x{weight:.2f}</small></span></div>'
        )

    # Breakdown reasons
    reasons = []
    if summary.get("bias_percentage", 0) > 0:
        reasons.append(f'<b>{summary["bias_percentage"]:.1f}%</b> tokens flagged')
    if summary.get("generalization_count", 0) > 0:
        reasons.append(f'<b>{summary["generalization_count"]}</b> generalizations')
    if summary.get("unfairness_count", 0) > 0:
        reasons.append(f'<b>{summary["unfairness_count"]}</b> unfair words')
    if summary.get("stereotype_count", 0) > 0:
        reasons.append(f'<b>{summary["stereotype_count"]}</b> stereotypes')
    
    reason_text = " + ".join(reasons) if reasons else "No significant bias detected"

    return (
        f'<div class="bias-criteria-breakdown">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
        f'<div>'
        f'<div style="font-size:36px;font-weight:800;color:{color};'
        f'font-family:Space Grotesk,Inter,sans-serif;letter-spacing:-1px; line-height:1;">{level}</div>'
        f'<div style="font-size:13px;color:#64748b;font-family:Inter,sans-serif; margin-top:8px;">'
        f'Criterion: {reason_text}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<div style="font-size:10px; color:#94a3b8; font-weight:600; text-transform:uppercase;">Composite Score</div>'
        f'<div style="font-size:20px; font-weight:700; color:#1e293b; font-family:JetBrains Mono;">{score:.3f}</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:20px; padding-top:12px; border-top:1px dashed #cbd5e1;">'
        f'<div style="font-size:10px; color:#94a3b8; margin-bottom:8px; font-weight:600; text-transform:uppercase;">Weighted Components</div>'
        f'{bar("Token density", c_pct, weights["pct"], bar_color="#3b82f6")}'
        f'{bar("Generalization", c_gen, weights["gen"], bar_color="#f97316")}'
        f'{bar("Unfair language", c_unfair, weights["unfair"], bar_color="#ef4444")}'
        f'{bar("Stereotypes", c_stereo, weights["stereo"], bar_color="#9c27b0")}'
        f'</div>'
        f'<div style="font-size:10px; color:#94a3b8; margin-top:12px; font-style:italic;">'
        f'Thresholds: Low &lt; 0.15 | Moderate &lt; 0.40 | High &ge; 0.40'
        f'</div>'
        f'</div>'
    )





def create_confidence_breakdown(token_labels: List[Dict]) -> str:
    """Render biased tokens grouped by confidence tier (Low / Medium / High).

    For each biased token the confidence value is
    ``max(scores[t] for t in bias_types)``.

    Tiers:
        Low    — [0.50, 0.70)
        Medium — [0.70, 0.85)
        High   — [0.85, 1.00]

    Returns an HTML string with:
    * A summary bar section (one horizontal bar per tier with count + %)
    * A token list per tier (badges with bias type(s) and max score)
    """
    # Collect biased tokens with their confidence
    biased = []
    for lbl in token_labels:
        if not lbl.get("is_biased"):
            continue
        tok = lbl.get("token", "")
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        types = lbl.get("bias_types", [])
        scores = lbl.get("scores", {})
        if not types:
            continue
        conf = max(scores.get(t, 0) for t in types)
        biased.append({"label": lbl, "conf": conf, "types": types, "scores": scores})

    if not biased:
        return (
            '<div style="color:#9ca3af;font-size:12px;padding:12px;text-align:center;">'
            'No biased tokens to break down.</div>'
        )

    # Define tiers
    tiers = [
        {"name": "High",   "min": 0.85, "max": 1.01, "color": "#ef4444", "bg": "rgba(239,68,68,0.15)"},
        {"name": "Medium", "min": 0.70, "max": 0.85, "color": "#f59e0b", "bg": "rgba(245,158,11,0.15)"},
        {"name": "Low",    "min": 0.50, "max": 0.70, "color": "#22c55e", "bg": "rgba(34,197,94,0.15)"},
    ]

    # Bucket tokens into tiers
    tier_buckets: Dict[str, list] = {t["name"]: [] for t in tiers}
    for item in biased:
        for t in tiers:
            if t["min"] <= item["conf"] < t["max"]:
                tier_buckets[t["name"]].append(item)
                break
        else:
            # Below 0.50 — still place in Low
            tier_buckets["Low"].append(item)

    total_biased = len(biased)

    # ── Summary bars ──
    bars_html = []
    for t in tiers:
        count = len(tier_buckets[t["name"]])
        pct = (count / total_biased * 100) if total_biased else 0
        bars_html.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="min-width:56px;font-size:11px;font-weight:600;color:{t["color"]};'
            f'font-family:JetBrains Mono,monospace;text-align:right;">{t["name"]}</span>'
            f'<div style="flex:1;background:#1e293b;border-radius:4px;height:10px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{t["color"]};border-radius:4px;'
            f'transition:width 0.4s;"></div></div>'
            f'<span style="min-width:60px;font-size:10px;color:#94a3b8;'
            f'font-family:JetBrains Mono,monospace;">{count} ({pct:.0f}%)</span>'
            f'</div>'
        )

    # ── Token badges per tier ──
    sections_html = []
    for t in tiers:
        items = tier_buckets[t["name"]]
        if not items:
            continue

        badges = []
        for item in sorted(items, key=lambda x: x["conf"], reverse=True):
            lbl = item["label"]
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            if not clean:
                continue

            # Build category dots
            type_dots = ""
            for bt in item["types"]:
                c_info = BIAS_COLORS.get(bt, BIAS_COLORS["GEN"])
                type_dots += (
                    f'<span style="display:inline-flex;align-items:center;gap:2px;">'
                    f'<span style="width:6px;height:6px;border-radius:50%;'
                    f'background:{c_info["border"]};display:inline-block;"></span>'
                    f'<span style="font-size:9px;color:{c_info["text"]};">{bt}</span>'
                    f'</span>'
                )

            badges.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;'
                f'padding:3px 8px;border-radius:6px;background:{t["bg"]};'
                f'border:1px solid {t["color"]}30;font-family:JetBrains Mono,monospace;">'
                f'<span style="font-size:11px;font-weight:600;color:#0f172a;">'
                f'{html_lib.escape(clean)}</span>'
                f'<span style="display:inline-flex;gap:4px;">{type_dots}</span>'
                f'<span style="font-size:10px;font-weight:700;color:{t["color"]};">'
                f'{item["conf"]:.2f}</span>'
                f'</span>'
            )

        sections_html.append(
            f'<div style="margin-top:10px;">'
            f'<div style="font-size:10px;font-weight:700;color:{t["color"]};'
            f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
            f'{t["name"]} Confidence</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{"".join(badges)}</div>'
            f'</div>'
        )

    return (
        f'<div>'
        f'<div style="margin-bottom:14px;">{"".join(bars_html)}</div>'
        f'{"".join(sections_html)}'
        f'</div>'
    )


def create_ablation_impact_chart(
    ablation_results: list,
    bar_threshold: float = 1.5,
) -> go.Figure:
    """Create a bar chart showing ablation impact per head.

    Parameters
    ----------
    ablation_results : list[HeadAblationResult]
        Results from batch_ablate_top_heads().
    bar_threshold : float
        BAR specialization threshold for color coding.
    """
    if not ablation_results:
        fig = go.Figure()
        fig.update_layout(
            title="No ablation results",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    labels = [f"L{r.layer}H{r.head}" for r in ablation_results]
    impacts = [r.representation_impact for r in ablation_results]
    bars_original = [r.bar_original for r in ablation_results]
    kl_divs = [r.kl_divergence for r in ablation_results]

    colors = [
        "#ff5ca9" if bar > bar_threshold else "#94a3b8"
        for bar in bars_original
    ]

    hover_text = []
    for r in ablation_results:
        parts = [
            f"<b>L{r.layer} H{r.head}</b>",
            f"Representation Impact: {r.representation_impact:.4f}",
            f"BAR (original): {r.bar_original:.3f}",
        ]
        if r.kl_divergence is not None:
            parts.append(f"KL Divergence: {r.kl_divergence:.4f}")
        hover_text.append("<br>".join(parts))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=impacts,
        marker_color=colors,
        hovertemplate="%{text}<extra></extra>",
        text=hover_text,
        name="Rep. Impact",
    ))

    has_kl = any(kl is not None for kl in kl_divs)
    if has_kl:
        kl_vals = [kl if kl is not None else 0 for kl in kl_divs]
        fig.add_trace(go.Scatter(
            x=labels,
            y=kl_vals,
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=6),
            name="KL Divergence",
            yaxis="y2",
        ))

    layout_kwargs = dict(
        title=dict(
            text="Head Ablation Impact<br><sub>Causal effect of zeroing each head on model representation</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif"),
        ),
        xaxis=dict(title="Attention Head", tickfont=dict(size=10, color="#475569")),
        yaxis=dict(title="1 − cos_sim (higher = more impact)", tickfont=dict(size=10, color="#475569")),
        autosize=True,
        height=400,
        margin=dict(l=80, r=80, t=100, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        showlegend=has_kl,
        legend=dict(x=0.5, y=1.12, xanchor="center", orientation="h"),
    )
    if has_kl:
        layout_kwargs["yaxis2"] = dict(
            title="KL Divergence",
            overlaying="y",
            side="right",
            tickfont=dict(size=10, color="#f59e0b"),
        )

    fig.update_layout(**layout_kwargs)
    return fig


def create_ig_correlation_chart(
    ig_results: list,
    bar_threshold: float = 1.5,
) -> go.Figure:
    """Create a combined heatmap + scatter showing IG vs attention correlation.

    Parameters
    ----------
    ig_results : list[IGCorrelationResult]
        Results from batch_compute_ig_correlation().
    bar_threshold : float
        BAR specialization threshold for annotations.

    Returns
    -------
    Plotly Figure with two subplots:
        Left: layer × head heatmap of Spearman ρ
        Right: scatter of BAR vs Spearman ρ (one dot per head)
    """
    if not ig_results:
        fig = go.Figure()
        fig.update_layout(
            title="No IG correlation results",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Determine grid dimensions
    max_layer = max(r.layer for r in ig_results)
    max_head = max(r.head for r in ig_results)
    num_layers = max_layer + 1
    num_heads = max_head + 1

    # Build correlation matrix
    rho_matrix = np.zeros((num_layers, num_heads))
    lookup = {}
    for r in ig_results:
        rho_matrix[r.layer, r.head] = r.spearman_rho
        lookup[(r.layer, r.head)] = r

    # Hover text for heatmap
    hover_text = []
    for layer in range(num_layers):
        row = []
        for head in range(num_heads):
            r = lookup.get((layer, head))
            if r:
                sig = "Yes" if r.spearman_pvalue < 0.05 else "No"
                row.append(
                    f"<b>L{layer} H{head}</b><br>"
                    f"Spearman ρ: {r.spearman_rho:.3f}<br>"
                    f"p-value: {r.spearman_pvalue:.4f}<br>"
                    f"Significant (p<0.05): {sig}<br>"
                    f"BAR: {r.bar_original:.3f}"
                )
            else:
                row.append(f"<b>L{layer} H{head}</b><br>No data")
            row_text = row  # keep reference
        hover_text.append(row)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Attention–IG Correlation per Head",
            "BAR vs Faithfulness (Spearman ρ)",
        ),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
    )

    # ── Left: Correlation heatmap ──
    # Divergent colorscale centered at 0: red (negative) → white (0) → blue (positive)
    rho_abs_max = max(abs(rho_matrix.min()), abs(rho_matrix.max()), 0.3)
    colorscale = [
        [0.0, "#dc2626"],     # Negative: red
        [0.5, "#ffffff"],     # Zero: white
        [1.0, "#2563eb"],     # Positive: blue
    ]

    fig.add_trace(
        go.Heatmap(
            z=rho_matrix.tolist(),
            x=[f"H{h}" for h in range(num_heads)],
            y=[f"L{l}" for l in range(num_layers)],
            colorscale=colorscale,
            zmin=-rho_abs_max,
            zmax=rho_abs_max,
            zauto=False,
            showscale=True,
            colorbar=dict(
                title=dict(text="Spearman ρ", font=dict(size=11, color="#64748b")),
                tickfont=dict(size=10, color="#64748b"),
                x=0.48,
                len=0.9,
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        ),
        row=1, col=1,
    )

    # ── Right: Scatter (BAR vs Spearman ρ) ──
    bars = [r.bar_original for r in ig_results]
    rhos = [r.spearman_rho for r in ig_results]
    significant = [r.spearman_pvalue < 0.05 for r in ig_results]
    labels = [f"L{r.layer}H{r.head}" for r in ig_results]

    colors = [
        "#2563eb" if sig else "#cbd5e1"
        for sig in significant
    ]

    scatter_hover = [
        f"<b>{lbl}</b><br>BAR: {b:.3f}<br>ρ: {rh:.3f}<br>"
        f"p: {r.spearman_pvalue:.4f}<br>Significant: {'Yes' if s else 'No'}"
        for lbl, b, rh, r, s in zip(labels, bars, rhos, ig_results, significant)
    ]

    fig.add_trace(
        go.Scatter(
            x=bars,
            y=rhos,
            mode="markers",
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color="#475569"),
                opacity=0.8,
            ),
            hovertemplate="%{text}<extra></extra>",
            text=scatter_hover,
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Reference lines on scatter
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1, row=1, col=2)
    fig.add_vline(x=bar_threshold, line_dash="dash", line_color="#ff5ca9",
                  line_width=1, row=1, col=2,
                  annotation_text=f"BAR={bar_threshold}", annotation_position="top right",
                  annotation_font_size=9, annotation_font_color="#ff5ca9")

    fig.update_layout(
        height=450,
        autosize=True,
        margin=dict(l=60, r=40, t=80, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        title=dict(
            text="Integrated Gradients Faithfulness Analysis<br>"
                 "<sub>Does attention correlate with gradient-based token importance?</sub>",
            font=dict(size=16, color="#1e293b"),
        ),
    )

    # Axis labels
    fig.update_xaxes(title_text="Head", row=1, col=1, tickfont=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="Layer", row=1, col=1, tickfont=dict(size=10, color="#475569"),
                     autorange="reversed")
    fig.update_xaxes(title_text="BAR (bias attention ratio)", row=1, col=2,
                     tickfont=dict(size=10, color="#475569"))
    fig.update_yaxes(title_text="Spearman ρ (attention vs IG)", row=1, col=2,
                     tickfont=dict(size=10, color="#475569"))

    return fig


def create_ig_token_comparison_chart(
    tokens: list,
    token_attributions: "np.ndarray",
    attentions: list,
    top_heads: list,
    max_heads: int = 3,
) -> go.Figure:
    """Grouped bar chart: IG attribution vs attention column-mean per token.

    Shows side-by-side comparison for the top-N heads (by BAR), so you can
    visually check whether attention and IG agree on which tokens matter.

    Parameters
    ----------
    tokens : list[str]
        Sub-word tokens (from tokenizer).
    token_attributions : np.ndarray, shape [seq_len]
        Absolute IG attribution per token.
    attentions : list of torch.Tensor
        Attention weights per layer [batch, heads, seq, seq].
    top_heads : list of IGCorrelationResult
        Heads to display (will use first *max_heads*).
    max_heads : int
        Maximum number of heads to overlay (default 3).
    """
    import torch

    if not tokens or token_attributions is None or len(token_attributions) == 0:
        fig = go.Figure()
        fig.update_layout(title="No token data", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    seq_len = len(tokens)
    heads_to_show = top_heads[:max_heads]

    # Normalize IG to [0, 1] for visual comparison
    ig = token_attributions[:seq_len].copy()
    ig_max = ig.max() if ig.max() > 0 else 1.0
    ig_norm = ig / ig_max

    head_colors = ["#2563eb", "#f59e0b", "#22c55e"]

    fig = go.Figure()

    # IG attribution bars
    fig.add_trace(go.Bar(
        x=list(range(seq_len)),
        y=ig_norm,
        name="IG Attribution",
        marker_color="#dc2626",
        opacity=0.7,
        hovertemplate="<b>%{customdata}</b><br>IG (norm): %{y:.3f}<extra>IG</extra>",
        customdata=tokens[:seq_len],
    ))

    # Attention column-mean for each top head
    for i, head in enumerate(heads_to_show):
        layer_attn = attentions[head.layer]
        if isinstance(layer_attn, torch.Tensor):
            attn_matrix = layer_attn[0, head.head].cpu().numpy()
        else:
            attn_matrix = np.array(layer_attn[0, head.head])
        attn_imp = attn_matrix.mean(axis=0)[:seq_len]
        # Normalize to [0, 1]
        attn_max = attn_imp.max() if attn_imp.max() > 0 else 1.0
        attn_norm = attn_imp / attn_max

        fig.add_trace(go.Bar(
            x=list(range(seq_len)),
            y=attn_norm,
            name=f"Attn L{head.layer}H{head.head} (ρ={head.spearman_rho:.2f})",
            marker_color=head_colors[i % len(head_colors)],
            opacity=0.6,
            hovertemplate=(
                f"<b>%{{customdata}}</b><br>"
                f"Attn L{head.layer}H{head.head} (norm): %{{y:.3f}}"
                f"<extra>L{head.layer}H{head.head}</extra>"
            ),
            customdata=tokens[:seq_len],
        ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Token-Level: IG Attribution vs Attention<br>"
                 "<sub>Normalized comparison — do attention and gradients agree on important tokens?</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif"),
        ),
        xaxis=dict(
            tickvals=list(range(seq_len)),
            ticktext=tokens[:seq_len],
            tickangle=45,
            tickfont=dict(size=9, color="#475569", family="JetBrains Mono, monospace"),
            title="Token",
        ),
        yaxis=dict(title="Normalized importance", tickfont=dict(size=10, color="#475569")),
        height=400,
        autosize=True,
        margin=dict(l=60, r=40, t=80, b=100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.5, y=1.15, xanchor="center", orientation="h",
                    font=dict(size=10)),
    )
    return fig


def create_ig_distribution_chart(
    ig_results: list,
    bar_threshold: float = 1.5,
) -> go.Figure:
    """Violin plot of Spearman ρ, split by specialized vs non-specialized heads.

    Directly answers: "Are heads that focus on biased tokens faithful?"

    Parameters
    ----------
    ig_results : list[IGCorrelationResult]
    bar_threshold : float
        BAR threshold to split specialized vs non-specialized.
    """
    if not ig_results:
        fig = go.Figure()
        fig.update_layout(title="No IG data", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    specialized_rhos = [r.spearman_rho for r in ig_results if r.bar_original > bar_threshold]
    non_specialized_rhos = [r.spearman_rho for r in ig_results if r.bar_original <= bar_threshold]

    fig = go.Figure()

    if non_specialized_rhos:
        fig.add_trace(go.Violin(
            y=non_specialized_rhos,
            name=f"Non-specialized (BAR ≤ {bar_threshold})",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(148,163,184,0.3)",
            line_color="#64748b",
            marker_color="#64748b",
            points="all",
            jitter=0.3,
            pointpos=-0.5,
            hovertemplate="ρ = %{y:.3f}<extra>Non-specialized</extra>",
        ))

    if specialized_rhos:
        fig.add_trace(go.Violin(
            y=specialized_rhos,
            name=f"Specialized (BAR > {bar_threshold})",
            box_visible=True,
            meanline_visible=True,
            fillcolor="rgba(255,92,169,0.25)",
            line_color="#ff5ca9",
            marker_color="#ff5ca9",
            points="all",
            jitter=0.3,
            pointpos=-0.5,
            hovertemplate="ρ = %{y:.3f}<extra>Specialized</extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        title=dict(
            text="Faithfulness by Specialization<br>"
                 "<sub>Do bias-specialized heads have faithful attention patterns?</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif"),
        ),
        yaxis=dict(title="Spearman ρ (attention vs IG)", tickfont=dict(size=10, color="#475569")),
        height=380,
        autosize=True,
        margin=dict(l=60, r=40, t=80, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.5, y=1.12, xanchor="center", orientation="h",
                    font=dict(size=10)),
        violinmode="group",
    )
    return fig


def create_ig_layer_summary_chart(
    ig_results: list,
) -> go.Figure:
    """Bar chart of mean Spearman ρ per layer with std error bars.

    Shows which layers have more faithful attention patterns.

    Parameters
    ----------
    ig_results : list[IGCorrelationResult]
    """
    if not ig_results:
        fig = go.Figure()
        fig.update_layout(title="No IG data", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Aggregate by layer
    from collections import defaultdict
    layer_rhos = defaultdict(list)
    for r in ig_results:
        layer_rhos[r.layer].append(r.spearman_rho)

    layers = sorted(layer_rhos.keys())
    means = [np.mean(layer_rhos[l]) for l in layers]
    stds = [np.std(layer_rhos[l]) for l in layers]
    n_heads = [len(layer_rhos[l]) for l in layers]

    # Color gradient: blue for positive mean, red for negative
    colors = ["#2563eb" if m >= 0 else "#dc2626" for m in means]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"L{l}" for l in layers],
        y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#94a3b8", thickness=1.5),
        marker_color=colors,
        opacity=0.85,
        hovertemplate=(
            "<b>Layer %{x}</b><br>"
            "Mean ρ: %{y:.3f}<br>"
            "Std: %{customdata[0]:.3f}<br>"
            "Heads: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=list(zip(stds, n_heads)),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        title=dict(
            text="Layer-wise Faithfulness<br>"
                 "<sub>Mean Spearman ρ per layer (error bars = ±1 std across heads)</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif"),
        ),
        xaxis=dict(title="Layer", tickfont=dict(size=10, color="#475569")),
        yaxis=dict(title="Mean Spearman ρ", tickfont=dict(size=10, color="#475569")),
        height=350,
        autosize=True,
        margin=dict(l=60, r=40, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
    )
    return fig


__all__ = [
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_ratio_formula_html",
    "create_bias_criteria_html",
    "create_bias_sentence_preview",
    "create_token_bias_strip",
    "create_confidence_breakdown",
    "create_ablation_impact_chart",
    "create_ig_correlation_chart",
    "create_ig_token_comparison_chart",
    "create_ig_distribution_chart",
    "create_ig_layer_summary_chart",
]
