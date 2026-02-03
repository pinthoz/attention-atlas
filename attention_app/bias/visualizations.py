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
from typing import List, Dict, Optional
from .attention_bias import HeadBiasMetrics


# ── Colour mapping shared across views ──
BIAS_COLORS = {
    "GEN": {"bg": "rgba(249, 115, 22, 0.18)", "border": "#f97316", "text": "#ea580c", "label": "Generalization"},
    "UNFAIR": {"bg": "rgba(239, 68, 68, 0.18)", "border": "#ef4444", "text": "#dc2626", "label": "Unfair Language"},
    "STEREO": {"bg": "rgba(156, 39, 176, 0.18)", "border": "#9c27b0", "text": "#7b1fa2", "label": "Stereotype"},
}





def create_token_bias_heatmap(token_labels: List[Dict], text: str) -> go.Figure:
    """Create heatmap showing bias categories for each token.

    Args:
        token_labels: Output from GusNetDetector.detect_bias()
        text: Original text

    Returns:
        Plotly Figure object
    """
    tokens = [label["token"] for label in token_labels]

    # Create binary matrix for each bias type
    categories = ["GEN", "UNFAIR", "STEREO"]
    matrix = []

    for category in categories:
        row = [1 if category in label["bias_types"] else 0 for label in token_labels]
        matrix.append(row)

    # Custom colorscale
    colorscale = [
        [0.0, '#f8fafc'],  # Light background for no bias
        [1.0, '#ff5ca9']   # Pink for detected bias
    ]

    # Create annotations for hover
    hover_text = []
    for i, category in enumerate(categories):
        row_hover = []
        for j, label in enumerate(token_labels):
            score = label.get("scores", {}).get(category, 0.0)
            if category in label["bias_types"]:
                row_hover.append(f"<b>{label['token']}</b><br>{category}: <b>Detected</b> (Score: {score:.2f})<br>{label['explanation']}")
            else:
                row_hover.append(f"<b>{label['token']}</b><br>{category}: Not detected (Score: {score:.2f})")
        hover_text.append(row_hover)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=tokens,
        y=categories,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="%{text}<extra></extra>",
        text=hover_text
    ))

    fig.update_layout(
        title=dict(
            text="Token-Level Bias Distribution Across Categories",
            font=dict(size=14, color="#1e293b", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Tokens",
            tickangle=-30,
            tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace"),
            side="bottom",
            automargin=True
        ),
        yaxis=dict(
            title="Category",
            tickfont=dict(size=11, color="#475569")
        ),
        height=280,
        margin=dict(l=80, r=40, t=60, b=100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_attention_bias_matrix(
    bias_matrix: np.ndarray,
    metrics: Optional[List[HeadBiasMetrics]] = None
) -> go.Figure:
    """Create heatmap showing attention to bias for each (layer, head).

    Args:
        bias_matrix: Matrix of shape [num_layers, num_heads]
        metrics: Optional list of HeadBiasMetrics for enhanced tooltips

    Returns:
        Plotly Figure object
    """
    num_layers, num_heads = bias_matrix.shape

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
                        f"Bias Attention Ratio: {m.bias_attention_ratio:.3f}<br>"
                        f"Amplification Score: {m.amplification_score:.3f}<br>"
                        f"Max Bias Attention: {m.max_bias_attention:.3f}<br>"
                        f"Specialized: {'Yes' if m.specialized_for_bias else 'No'}"
                    )
                else:
                    text = f"<b>Layer {layer}, Head {head}</b><br>Ratio: {bias_matrix[layer, head]:.3f}"
                row.append(text)
            hover_text.append(row)
    else:
        hover_text = [[f"<b>L{l}, H{h}</b><br>Ratio: {bias_matrix[l,h]:.3f}"
                       for h in range(num_heads)] for l in range(num_layers)]

    # Custom colorscale (blue for low, red for high bias attention)
    colorscale = [
        [0.0, '#dbeafe'],   # Light blue
        [0.5, '#ffffff'],   # White (neutral)
        [0.75, '#fecaca'],  # Light red
        [1.0, '#dc2626']    # Dark red (high bias)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=bias_matrix,
        x=[f"H{h}" for h in range(num_heads)],
        y=[f"L{l}" for l in range(num_layers)],
        colorscale=colorscale,
        zmid=1.0,
        colorbar=dict(
            title=dict(
                text="Bias Attention<br>Ratio",
                side="right",
                font=dict(size=11, color="#64748b")
            ),
            tickfont=dict(size=10, color="#64748b")
        ),
        hovertemplate="%{text}<extra></extra>",
        text=hover_text
    ))

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
        height=max(300, num_layers * 40),
        margin=dict(l=60, r=150, t=100, b=60),
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
            "<b>Bias Attention Ratio</b><br>"
            "<span style='font-size:9px'>attn(→biased) / E[attn]</span><br><br>"
            "<span style='color:#dc2626'>■</span> <b>≥ 1.5</b> Specialized<br>"
            "<span style='color:#93c5fd'>■</span> <b>= 1.0</b> Neutral<br>"
            "<span style='color:#3b82f6'>■</span> <b>&lt; 1.0</b> Avoids bias"
        ),
        showarrow=False,
        font=dict(size=10, color="#64748b"),
        align="left",
        xanchor="left"
    )

    return fig


def create_bias_propagation_plot(layer_propagation: List[float]) -> go.Figure:
    """Create line plot showing how bias attention changes across layers.

    Args:
        layer_propagation: List of average bias ratios per layer

    Returns:
        Plotly Figure object
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
        name='Bias Attention',
        hovertemplate="<b>Layer %{x}</b><br>Avg Bias Ratio: %{y:.3f}<extra></extra>"
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
            title="Average Bias Attention Ratio",
            gridcolor='#e2e8f0',
            tickfont=dict(size=11, color="#64748b")
        ),
        height=400,
        margin=dict(l=80, r=40, t=100, b=60),
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
    head_idx: int
) -> go.Figure:
    """Create attention heatmap matching the Multi-Head Attention style.

    Biased tokens are highlighted with pink row/column overlays,
    exactly like the attention tab highlights a selected token.
    """
    n = len(tokens)
    biased_indices = set(
        i for i, label in enumerate(token_labels) if label.get("is_biased")
    )

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

    # ── HTML-styled tick labels (pink + bold for biased tokens) ──
    styled = []
    for i, tok in enumerate(cleaned):
        if i in biased_indices:
            styled.append(
                f"<span style='color:#ec4899;font-weight:bold'>{tok}</span>"
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

    # ── Pink row/column highlights for biased tokens ──
    for idx in biased_indices:
        if idx < 0 or idx >= n:
            continue
        # Row highlight
        fig.add_shape(
            type="rect",
            x0=-0.5, x1=n - 0.5,
            y0=idx - 0.5, y1=idx + 0.5,
            fillcolor="rgba(236, 72, 153, 0.12)",
            line=dict(color="#ec4899", width=1),
            layer="above",
        )
        # Column highlight
        fig.add_shape(
            type="rect",
            x0=idx - 0.5, x1=idx + 0.5,
            y0=-0.5, y1=n - 0.5,
            fillcolor="rgba(236, 72, 153, 0.12)",
            line=dict(color="#ec4899", width=1),
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
        height=max(500, n * 30),
        margin=dict(l=40, r=10, t=40, b=40),
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


def create_method_info_html() -> str:
    """Render a compact, visually rich info panel for GUS-Net."""
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

    return (
        '<div style="margin-top:20px;padding:16px 20px;'
        'background:linear-gradient(135deg,rgba(248,250,252,0.8),rgba(241,245,249,0.4));'
        'border:1px solid rgba(226,232,240,0.6);border-radius:12px;">'

        # ── Header ──
        '<div style="margin-bottom:14px;">'
        '<div style="font-size:14px;font-weight:700;color:#1e293b;line-height:1.2;">'
        'GUS-Net</div>'
        '<div style="font-size:11px;color:#64748b;">'
        'Social Bias Named Entity Recognition</div>'
        '</div>'

        # ── Specs grid ──
        '<div style="display:grid;grid-template-columns:90px 1fr;gap:6px 12px;'
        'font-size:11px;margin-bottom:14px;padding-bottom:14px;'
        'border-bottom:1px dashed rgba(203,213,225,0.5);">'

        '<span style="color:#94a3b8;font-weight:600;">Model</span>'
        '<code style="font-size:10px;padding:1px 6px;background:rgba(148,163,184,0.1);'
        'border-radius:3px;color:#475569;font-family:JetBrains Mono,monospace;">'
        'ethical-spectacle/social-bias-ner</code>'

        '<span style="color:#94a3b8;font-weight:600;">Paper</span>'
        '<a href="https://arxiv.org/pdf/2410.08388" target="_blank" '
        'style="color:#3b82f6;text-decoration:none;font-size:10px;">'
        'arxiv.org/pdf/2410.08388</a>'

        '<span style="color:#94a3b8;font-weight:600;">HuggingFace</span>'
        '<a href="https://huggingface.co/collections/ethical-spectacle/gus-net" target="_blank" '
        'style="color:#3b82f6;text-decoration:none;font-size:10px;">'
        'ethical-spectacle/gus-net</a>'

        '<span style="color:#94a3b8;font-weight:600;">Technique</span>'
        '<span style="color:#475569;">Multi-label token classification (sigmoid)</span>'

        '<span style="color:#94a3b8;font-weight:600;">Categories</span>'
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{badges_html}</div>'
        '</div>'

        # ── Limitations ──
        '<div>'
        '<div style="font-size:9px;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">'
        'Limitations</div>'
        + bullet('Sub-word split',
                 'BERT tokenization may fragment entities, lowering scores on pieces')
        + bullet('Ambiguity',
                 'Some adjectives (e.g. <em>emotional</em>) score high regardless of context')
        + bullet('Fixed scope',
                 'Trained on specific datasets &mdash; may miss niche or subtle bias')
        + '</div></div>'
    )


def create_ratio_formula_html() -> str:
    """Render the Bias Attention Ratio formula explanation panel."""
    return (
        '<div style="background:linear-gradient(135deg,#f8fafc,#f0f4f8);'
        'border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-bottom:16px;">'

        '<div style="margin-bottom:12px;">'
        '<span style="font-size:13px;font-weight:700;color:#1e293b;">Bias Attention Ratio — Definition</span>'
        '</div>'

        '<div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px;'
        'font-family:JetBrains Mono,monospace;font-size:12px;color:#e2e8f0;line-height:1.6;">'
        'ratio(<span style="color:#94a3b8;">l, h</span>) = '
        '<span style="color:#60a5fa;">mean_attn(all → biased_tokens)</span>'
        ' / '
        '<span style="color:#f59e0b;">E[attn | uniform]</span>'
        '</div>'

        '<div style="font-size:11px;color:#64748b;line-height:1.6;">'
        'A ratio of '
        '<code style="background:#dbeafe;padding:1px 5px;border-radius:3px;font-size:10px;color:#1e40af;">1.0</code>'
        ' means the head distributes attention uniformly (neutral baseline). '
        'Ratio '
        '<code style="background:#fee2e2;padding:1px 5px;border-radius:3px;font-size:10px;color:#991b1b;">&gt; 1.5</code>'
        ' indicates the head <b>specializes</b> on biased tokens. '
        'Ratio '
        '<code style="background:#dbeafe;padding:1px 5px;border-radius:3px;font-size:10px;color:#1e40af;">&lt; 1.0</code>'
        ' means the head <b>avoids</b> biased tokens.'
        '</div>'
        '</div>'
    )


def create_token_bias_strip(token_labels: List[Dict]) -> str:
    """Render a compact HTML strip showing per-token bias categories.

    Each token is displayed with small coloured dots beneath it indicating
    which categories (GEN / UNFAIR / STEREO) are active.  This replaces
    the Plotly heatmap with a lighter, card-free visualisation.
    """
    cats = ["O", "GEN", "UNFAIR", "STEREO"]
    cat_colors = {"O": "#64748b", "GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}

    cells = []
    for lbl in token_labels:
        tok = lbl["token"]
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        clean = tok.replace("##", "").replace("\u0120", "")
        if not clean:
            continue

        is_biased = lbl.get("is_biased", False)
        tok_bg = "background:rgba(236,72,153,0.06);" if is_biased else ""
        types = lbl.get("bias_types", [])
        scores = lbl.get("scores", {})

        # Category score indicators (always show numeric value)
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

        tooltip = html_lib.escape(clean)
        for cat in cats:
            sc = scores.get(cat, 0)
            marker = " *" if cat in types else ""
            tooltip += f"&#10;{cat}: {sc:.3f}{marker}"

        cells.append(
            f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
            f'padding:4px 6px;border-radius:6px;{tok_bg}'
            f'font-family:JetBrains Mono,monospace;font-size:12px;cursor:help;" '
            f'title="{tooltip}">'
            f'<span style="line-height:1.2;color:#0f172a;">{html_lib.escape(clean)}</span>'
            f'{scores_row}'
            f'</span>'
        )

    if not cells:
        return '<div style="color:#9ca3af;font-size:12px;">No tokens to display.</div>'

    # Legend
    cat_labels = {"O": "Outside — neutral, no bias detected", "GEN": "Generalization", "UNFAIR": "Unfair Language", "STEREO": "Stereotype"}
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
        f'align-items:flex-start;">{"".join(cells)}</div>'
        f'<div style="display:flex;gap:14px;margin-top:4px;padding-top:8px;'
        f'border-top:1px solid #f1f5f9;">{"".join(legend_items)}</div>'
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


def create_bias_sentence_preview(tokens: List[str], token_labels: List[Dict]) -> str:
    """Create a token-viz style sentence preview with bias coloring.

    Mirrors the attention tab's ``get_preview_text_view`` but colours each
    token by its bias type.  Tokens with multiple categories show a
    multi-segment colour bar at the bottom.
    """
    token_html = []
    for tok, label in zip(tokens, token_labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        clean = tok.replace("##", "").replace("\u0120", "")
        if not clean:
            continue

        if label["is_biased"] and label["bias_types"]:
            types = label["bias_types"]
            primary = types[0]
            color_info = BIAS_COLORS.get(primary, BIAS_COLORS["GEN"])
            max_score = max(label["scores"].get(t, 0) for t in types)
            types_str = ", ".join(types)
            tooltip = f"{html_lib.escape(clean)}&#10;{types_str} (score: {max_score:.2f})"

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
            inner = f'{html_lib.escape(clean)}{bar_html}'
        else:
            tooltip = f"{html_lib.escape(clean)}&#10;No bias detected"
            style = "background:rgba(241,245,249,0.6);"
            inner = html_lib.escape(clean)

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


__all__ = [
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_ratio_formula_html",
    "create_bias_criteria_html",
    "create_bias_sentence_preview",
    "create_token_bias_strip",
]
