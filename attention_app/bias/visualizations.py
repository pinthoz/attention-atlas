"""Visualization components for bias analysis.

This module creates visualizations for:
- Inline text highlighting with bias annotations
- Token-level bias heatmaps
- Attention x Bias matrices
- Method info panels
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
METHOD_LABELS = {
    "gusnet": "GUS-Net",
    "lexicon": "Lexicon",
    "combined": "Combined",
}





def create_token_bias_heatmap(token_labels: List[Dict], text: str) -> go.Figure:
    """Create heatmap showing bias categories for each token.

    Args:
        token_labels: Output from TokenBiasDetector.detect_bias()
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

    # Add reference line at ratio = 1.5 (specialization threshold)
    fig.add_annotation(
        x=1.25,
        y=0.0,
        xref="paper",
        yref="paper",
        text="<b>Red ≥ 1.5:</b> Head<br>specializes in bias",
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
    """Create attention heatmap with bias highlighting overlay.

    Args:
        tokens: List of tokens
        token_labels: Bias detection results
        attention_matrix: Attention weights [seq_len, seq_len]
        layer_idx: Layer index
        head_idx: Head index

    Returns:
        Plotly Figure object combining attention and bias
    """
    # Base attention heatmap
    fig = go.Figure()

    # Attention heatmap
    fig.add_trace(go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title="Attention",
            x=1.15,
            len=0.5,
            y=0.75
        ),
        name="Attention",
        hovertemplate="<b>%{y} → %{x}</b><br>Attention: %{z:.4f}<extra></extra>"
    ))

    # Add bias annotations
    biased_indices = [i for i, label in enumerate(token_labels) if label["is_biased"]]

    # Add colored rectangles around biased tokens
    shapes = []
    for idx in biased_indices:
        label = token_labels[idx]
        # Determine color by bias type
        if "STEREO" in label["bias_types"]:
            color = "#9c27b0"  # Purple for stereotypes
        elif "UNFAIR" in label["bias_types"]:
            color = "#ef4444"  # Red for unfair
        elif "GEN" in label["bias_types"]:
            color = "#f97316"  # Orange for generalizations
        else:
            color = "#ff5ca9"  # Pink default

        # Add rectangle highlight
        shapes.append(
            dict(
                type="rect",
                x0=idx - 0.5,
                x1=idx + 0.5,
                y0=-0.5,
                y1=len(tokens) - 0.5,
                line=dict(color=color, width=3),
                fillcolor="rgba(0,0,0,0)"
            )
        )

    fig.update_layout(
        shapes=shapes,
        title=dict(
            text=f"Attention × Bias — Layer {layer_idx}, Head {head_idx}<br><sub>Colored borders indicate biased tokens</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Key (attending to)",
            tickangle=-45,
            tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace"),
            side="bottom"
        ),
        yaxis=dict(
            title="Query (attending from)",
            tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace")
        ),
        height=max(500, len(tokens) * 30),
        margin=dict(l=100, r=160, t=100, b=120),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )

    # Add legend for bias colors
    fig.add_annotation(
        x=1.16,
        y=0.35,
        xref="paper",
        yref="paper",
        text="<b>Bias Types:</b><br>" +
             "<span style='color:#9c27b0'>■</span> Stereotype<br>" +
             "<span style='color:#ef4444'>■</span> Unfair<br>" +
             "<span style='color:#f97316'>■</span> Generalization",
        showarrow=False,
        font=dict(size=10, color="#64748b"),
        align="left",
        xanchor="left"
    )

    return fig


def create_inline_bias_html(
    text: str,
    token_labels: List[Dict],
    bias_spans: list,
    show_neutral: bool = False,
) -> str:
    """Render original text with inline highlighted bias spans.

    Returns an HTML string where biased spans are wrapped in coloured
    ``<span>`` elements with hover tooltips showing category, score,
    method and explanation.
    """
    # Filter spans if show_neutral is False
    # Neutral/controversial are detections with score < median or specific low threshold
    display_spans = bias_spans
    if not show_neutral:
        # Only show spans where average score is >= 0.35 (adjustable)
        # Rule-based (lexicon) always 1.0, so they stay.
        display_spans = [s for s in bias_spans if (getattr(s, "avg_score", 0) >= 0.35 if hasattr(s, "avg_score") else s.get("avg_score", 0) >= 0.35)]

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
        if hasattr(span, "start_idx"):
            s_start, s_end = span.start_idx, span.end_idx
            s_types = span.bias_types
            s_score = getattr(span, "avg_score", span.confidence)
            s_method = getattr(span, "method", "lexicon")
            s_explanation = span.explanation
            s_threshold = 0.5 # Default
        else:
            s_start, s_end = span["start_idx"], span["end_idx"]
            s_types = span["bias_types"]
            s_score = span.get("avg_score", 1.0)
            s_method = span.get("method", "gusnet")
            s_explanation = span.get("explanation", "")
            s_threshold = span.get("threshold", 0.5)

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
        method_label = METHOD_LABELS.get(s_method, s_method)
        
        # Determine origin icon
        origin_icon = "🧠" if "gusnet" in s_method else ("📜" if "lexicon" in s_method else "🔗")
        
        tooltip = (
            f'<div class="bias-tooltip">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
            f'<div>{type_badges}</div>'
            f'<div style="font-size:14px;">{origin_icon}</div>'
            f'</div>'
            f'<div class="bias-score-row">'
            f'<span>Confidence:</span>'
            f'<div class="bias-score-bar-container">'
            f'<div class="bias-score-bar" style="width:{s_score*100:.0f}%;'
            f'background:{BIAS_COLORS.get(primary_type, BIAS_COLORS["GEN"])["border"]};"></div>'
            f'</div>'
            f'<span style="font-weight:700; color:#fff;">{s_score:.2f}</span>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between; margin-top:6px; font-size:9px; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">'
            f'<span>Method: <b>{method_label}</b></span>'
            f'<span>Threshold: <b>{s_threshold}</b></span>'
            f'</div>'
            f'<div style="font-size:10px;color:#cbd5e1;margin-top:6px; font-style:italic;">"{html_lib.escape(s_explanation)}"</div>'
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


def create_method_info_html(method: str) -> str:
    """Render an info panel explaining how bias is detected.

    Args:
        method: One of "gusnet", "lexicon", "combined"
    """
    if method == "gusnet":
        return (
            '<div class="bias-method-info">'
            '<h5 style="color:#ff5ca9;margin:0 0 8px; font-weight:700;">🧠 How it works: GUS-Net (Neural NER)</h5>'
            '<p style="font-size:11px; color:#64748b; margin-bottom:10px;">A specialized BERT-based model fine-tuned for Social Bias Named Entity Recognition.</p>'
            '<table class="bias-info-table">'
            '<tr><td>Origin</td><td><code>ethical-spectacle/social-bias-ner</code></td></tr>'
            '<tr><td>Technique</td><td>Multi-label token classification with probability sigmoid</td></tr>'
            '<tr><td>Capabilities</td><td>Identifies complex generalizations and stereotypes through learned context</td></tr>'
            '</table>'
            '<div class="bias-limitations">'
            '<strong>Model Limitations:</strong>'
            '<ul>'
            '<li><b>Sub-word split:</b> BERT tokenization may split names/entities, causing lower scores on fragments</li>'
            '<li><b>Ambiguity:</b> Some adjectives (e.g. "emotional") have high probability regardless of context</li>'
            '<li><b>Fixed Scope:</b> Trained on specific bias datasets; may miss niche or extremely subtle forms of bias</li>'
            '</ul></div></div>'
        )
    elif method == "lexicon":
        return (
            '<div class="bias-method-info">'
            '<h5 style="color:#ff5ca9;margin:0 0 8px; font-weight:700;">📜 How it works: Lexicon (Rule-Based)</h5>'
            '<p style="font-size:11px; color:#64748b; margin-bottom:10px;">Pattern matching using a curated database of biased terms and linguistic markers.</p>'
            '<table class="bias-info-table">'
            '<tr><td>Data</td><td><code>bias_lexicon.json</code> (Group nouns, traits, stereotypic verbs)</td></tr>'
            '<tr><td>Technique</td><td>Regex and context-window matching (window: 5 tokens)</td></tr>'
            '<tr><td>Confidence</td><td>Always 1.0 (binary match)</td></tr>'
            '</table>'
            '<div class="bias-limitations">'
            '<strong>Logic Limitations:</strong>'
            '<ul>'
            '<li><b>No nuances:</b> Flags words based on presence, regardless of intent (sarcasm, negation, etc.)</li>'
            '<li><b>Limited coverage:</b> Only detects bias explicitly listed in the dictionary</li>'
            '<li><b>Rigidity:</b> Cannot adapt to spelling variations or novel biased slangs</li>'
            '</ul></div></div>'
        )
    else:
        return (
            '<div class="bias-method-info">'
            '<h5 style="color:#ff5ca9;margin:0 0 8px; font-weight:700;">🔗 How it works: Combined Mode</h5>'
            '<p style="font-size:11px; color:#64748b; margin-bottom:10px;">Aggregates signals from both Neural and Rule-based systems for maximum recall.</p>'
            '<table class="bias-info-table">'
            '<tr><td>Strategy</td><td><b>Union</b> (Flags token if EITHER system detects bias)</td></tr>'
            '<tr><td>Scoring</td><td>Takes the <b>Max Score</b> from both systems</td></tr>'
            '<tr><td>Reliability</td><td>Detections flagged by both (origin icon 🔗) are highly relevant</td></tr>'
            '</table>'
            '<div class="bias-limitations">'
            '<strong>Operational Notes:</strong>'
            '<ul>'
            '<li>Best for exploring potential issues across different layers of abstraction</li>'
            '<li>Total bias score is a weighted average of categorical probabilities</li>'
            '</ul></div></div>'
        )


def create_bias_criteria_html(summary: Dict, method: str, weights: Optional[Dict] = None) -> str:
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


__all__ = [
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization",
    "create_inline_bias_html",
    "create_method_info_html",
    "create_bias_criteria_html",
]
