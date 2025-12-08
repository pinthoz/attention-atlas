"""Visualization components for bias analysis.

This module creates Plotly visualizations for:
- Token-level bias heatmaps (GUS-Net style)
- Attention×Bias matrices
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
from .attention_bias import HeadBiasMetrics





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
            if category in label["bias_types"]:
                row_hover.append(f"<b>{label['token']}</b><br>{category}: Detected<br>{label['explanation']}")
            else:
                row_hover.append(f"<b>{label['token']}</b><br>{category}: Not detected")
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
            text="Token-Level Bias Detection<br><sub>GEN: Generalization | UNFAIR: Unfair Language | STEREO: Stereotype</sub>",
            font=dict(size=16, color="#1e293b", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Tokens",
            tickangle=-45,
            tickfont=dict(size=10, color="#475569", family="JetBrains Mono, monospace"),
            side="bottom"
        ),
        yaxis=dict(
            title="Bias Category",
            tickfont=dict(size=11, color="#475569")
        ),
        height=300,
        margin=dict(l=80, r=40, t=100, b=120),
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


__all__ = [
    "create_token_bias_heatmap",
    "create_attention_bias_matrix",
    "create_bias_propagation_plot",
    "create_combined_bias_visualization"
]
