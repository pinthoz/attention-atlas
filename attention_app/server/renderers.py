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


def arrow(from_section, to_section, direction="horizontal", suffix="", model_type=None, **kwargs):
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

    onclick_js = f"showTransitionModal('{from_section}', '{to_section}')"
    if model_type:
        onclick_js = f"showTransitionModal('{from_section}', '{to_section}', '{model_type}')"

    attrs = {
        "class": classes,
        "onclick": onclick_js,
        "id": arrow_id,
        "title": f"Click: {from_section} → {to_section}"
    }
    attrs.update(kwargs)

    return ui.tags.div(attrs, icon)


# ──────────────────────────────────────────────────────────────
# Architecture Diagram helpers
# ──────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color):
    """Convert '#3b82f6' → (59, 130, 246) for rgba() usage."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_architecture_diagram(model_type, accent_color, title_prefix=None, layer_count_val=12):
    """Return raw HTML string for a single architecture column diagram.

    Matches the design in architecture.html — dark blocks on page background.
    model_type: 'bert' | 'gpt2'
    accent_color: hex colour string e.g. '#3b82f6'
    title_prefix: Optional string like "Model A" or "Prompt B"
    layer_count_val: Number of layers to display (e.g. 12, 24)
    """
    r, g, b = _hex_to_rgb(accent_color)
    accent_rgba = f"{r},{g},{b}"

    if model_type == "bert":
        model_name = "BERT (Encoder)"
        has_segment_embed = True
        attn_label = "Multi-Head Attn"
        output_label = "MLM Predictions"
        output_example = (
            '<div style="font-size:8px;font-family:monospace;color:#64748b;'
            'border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:3px;margin-bottom:3px;line-height:1.5;">'
            '&ldquo;The capital of <span style="color:rgba(255,92,169,0.8);">[MASK]</span> is Paris.&rdquo;</div>'
            '<div style="display:flex;justify-content:space-between;width:100%;font-size:9px;font-family:monospace;color:#cbd5e1;">'
            '<span style="color:#f9a8d4;font-weight:700;">France</span><span style="opacity:0.75;">99.1%</span></div>'
            '<div style="display:flex;justify-content:space-between;width:100%;font-size:9px;font-family:monospace;color:#cbd5e1;opacity:0.4;">'
            '<span>Europe</span><span>0.4%</span></div>'
        )
    else:
        model_name = "GPT-2 (Decoder)"
        has_segment_embed = False
        attn_label = "Masked Attn"
        output_label = "Next Token Pred"
        output_example = (
            '<div style="font-size:8px;font-family:monospace;color:#64748b;'
            'border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:3px;margin-bottom:3px;line-height:1.5;">'
            '&ldquo;Deep Learning is really <span style="color:rgba(255,92,169,0.8);">&hellip;</span>&rdquo;</div>'
            '<div style="display:flex;justify-content:space-between;width:100%;font-size:9px;font-family:monospace;color:#cbd5e1;">'
            '<span style="color:#f9a8d4;font-weight:700;">powerful</span><span style="opacity:0.75;">65.4%</span></div>'
            '<div style="display:flex;justify-content:space-between;width:100%;font-size:9px;font-family:monospace;color:#cbd5e1;opacity:0.4;">'
            '<span>important</span><span>12.2%</span></div>'
        )

    # Title assembly
    model_label = f'<span style="opacity:0.5;font-weight:400;">{title_prefix}:</span> {model_name}' if title_prefix else model_name

    # Shared inline styles — sidebar-blue blocks on page bg ------------------
    block = (
        "position:relative;z-index:10;display:flex;align-items:center;justify-content:center;"
        "text-align:center;height:34px;font-size:10px;padding:0 10px;border-radius:6px;"
        "transition:all 0.3s;border:1px solid rgba(255,255,255,0.15);"
        "user-select:none;background-color:#0f172a;width:110px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.2);"
    )
    label = "font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#e2e8f0;"
    badge = (
        "position:absolute;top:-6px;right:-6px;font-family:'JetBrains Mono',monospace;"
        "font-size:7px;padding:1px 4px;border-radius:3px;z-index:20;"
        "background:rgba(30,41,59,0.9);color:#94a3b8;border:1px solid rgba(255,255,255,0.12);"
    )
    connector = (
        "position:relative;width:1px;background-color:#cbd5e1;margin:0 auto;overflow:hidden;"
    )
    pulse_base = "position:absolute;top:0;width:100%;height:60%;animation:archMovePulse 2.5s linear infinite;"
    add_norm_style = block + "height:22px;width:90px;font-size:8px;"
    layer_box = (
        "position:relative;padding:14px;padding-bottom:24px;border-radius:8px;"
        "border:1px solid rgba(255,92,169,0.3);background-color:rgba(255,92,169,0.06);"
        "display:flex;flex-direction:column;align-items:center;"
        "box-shadow:inset 0 0 10px rgba(255,92,169,0.05);"
    )
    layer_count = (
        "position:absolute;right:-14px;top:50%;transform:translateY(-50%) rotate(90deg);"
        "font-size:9px;color:#64748b;font-family:monospace;letter-spacing:0.25em;font-weight:700;"
    )
    output_block = (
        "position:relative;z-index:10;display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;padding:12px 10px;border-radius:6px;"
        "transition:all 0.3s;user-select:none;width:150px;"
        "border:1.5px solid rgba(255,92,169,0.5);background-color:#0f172a;"
        "box-shadow:0 4px 6px -1px rgba(0,0,0,0.2), 0 2px 4px -1px rgba(255,92,169,0.1);"
    )


    # Residual SVG -----------------------------------------------------------
    residual_svg = (
        '<svg style="position:absolute;width:140px;height:100%;pointer-events:none;'
        'top:0;left:50%;transform:translateX(-50%);opacity:0.4;">'
        '<path d="M -45 28 C -65 28 -65 110 -45 110" fill="none" stroke="#10b981" stroke-width="1" transform="translate(70,0)"/>'
        '<path d="M -45 148 C -65 148 -65 186 -45 186" fill="none" stroke="#10b981" stroke-width="1" transform="translate(70,0)"/>'
        '</svg>'
    )

    # Helper closures --------------------------------------------------------
    def _block(lbl, extra_style="", badge_text="DD", badge_extra="", label_extra=""):
        return (
            f'<div style="{block}{extra_style}">'
            f'<div style="{badge}{badge_extra}">{badge_text}</div>'
            f'<span style="{label}{label_extra}">{lbl}</span>'
            f'</div>'
        )

    def _connector(h, pulse_color=""):
        inner = ""
        if pulse_color:
            inner = f'<div style="{pulse_base}background:linear-gradient(transparent,{pulse_color},transparent);"></div>'
        return f'<div style="{connector}height:{h}px;">{inner}</div>'

    def _add_norm():
        return (
            f'<div style="{add_norm_style}">'
            f'<div style="{badge}">DD</div>'
            f'<span style="{label}font-size:7px;">Add &amp; Norm</span>'
            f'</div>'
        )

    # Build HTML -------------------------------------------------------------
    parts = []

    # Embedded keyframes
    parts.append(
        '<style>'
        '@keyframes archMovePulse { 0% { top: -50%; } 100% { top: 100%; } }'
        '</style>'
    )

    parts.append('<div class="arch-diagram-wrapper" style="display:flex;flex-direction:column;align-items:center;">')

    # Title
    parts.append(
        f'<div style="margin-bottom:14px;font-size:12px;font-weight:800;color:#334155;'
        f'text-transform:uppercase;letter-spacing:0.25em;">{model_label}</div>'
    )


    # Sentence Preview
    parts.append(_block("Sentence Preview",
        extra_style=f"border-color:rgba({accent_rgba},0.3);",
        badge_text="R0",
        label_extra=f"color:rgba({accent_rgba},0.85);"))
    parts.append(_connector(20, accent_color))

    # Token Embeddings
    parts.append(_block("Token Embeddings"))
    parts.append(_connector(16))

    # Segment Embed (BERT only)
    if has_segment_embed:
        parts.append(_block("Segment Embed",
            extra_style="border-color:rgba(139,92,246,0.3);",
            label_extra="color:#a78bfa;"))
        parts.append(_connector(16))

    # Pos Encoding
    parts.append(_block("Pos Encoding"))
    parts.append(_connector(20, accent_color))

    # Layer Box
    parts.append(f'<div style="{layer_box}">')
    parts.append(f'<div style="{layer_count}">&times;{layer_count_val}</div>')

    # Q/K/V
    parts.append(_block("Q/K/V Projections",
        extra_style=f"border-color:rgba({accent_rgba},0.2);font-size:8px;"))
    parts.append(_connector(12))

    # Attention
    parts.append(_block(attn_label,
        extra_style=f"background-color:rgba({accent_rgba},0.05);border-color:rgba({accent_rgba},0.4);",
        badge_text="VIS",
        badge_extra=f"color:rgba({accent_rgba},0.85);",
        label_extra=f"color:rgba({accent_rgba},0.85);"))

    # Residual SVG
    parts.append(residual_svg)

    parts.append(_connector(12, "#10b981"))

    # Add & Norm 1
    parts.append(_add_norm())
    parts.append(_connector(12))

    # FFN
    parts.append(_block("Feed Forward", badge_text="TECH"))
    parts.append(_connector(12, "#10b981"))

    # Add & Norm 2
    parts.append(_add_norm())

    # Close layer box
    parts.append('</div>')

    # Connector to output
    parts.append(_connector(24, "#ff5ca9"))

    # Output block
    parts.append(
        f'<div style="{output_block}">'
        f'<div style="{badge}color:#ff5ca9;">ESS</div>'
        f'<span style="{label}color:#f9a8d4;margin-bottom:4px;font-size:10px;">{output_label}</span>'
        f'<div style="display:flex;flex-direction:column;gap:2px;align-items:flex-start;width:100%;padding:0 6px;">'
        f'{output_example}'
        f'</div></div>'
    )

    # Close wrapper
    parts.append('</div>')

    return "\n".join(parts)


def get_architecture_section(model_type="bert", accent_color="#3b82f6", title_prefix=None):
    """Single-mode architecture diagram → ui.HTML."""
    diagram_html = get_architecture_diagram(model_type, accent_color, title_prefix=title_prefix)
    return ui.HTML(
        f'<div class="arch-section">'
        f'<div class="arch-body">{diagram_html}</div>'
        f'</div>'
    )


def get_paired_architecture_section(model_type_a="bert", model_type_b="gpt2",
                                     color_a="#3b82f6", color_b="#ff5ca9",
                                     label_a="Model A", label_b="Model B",
                                     active_model="both", dual_color=False,
                                     layers_a=12, layers_b=12):
    """
    Compare-mode — always show both diagrams side-by-side.
    active_model: "both" (default), "A", or "B".
                  If "A", dim B. If "B", dim A.
    dual_color: If True (compare models), A gets blue border, B gets pink.
                If False (single/compare prompts), selected model gets pink border.
    layers_a: Number of layers for Model A
    layers_b: Number of layers for Model B
    """
    diagram_a = get_architecture_diagram(model_type_a, color_a, title_prefix=label_a, layer_count_val=layers_a)
    diagram_b = get_architecture_diagram(model_type_b, color_b, title_prefix=label_b, layer_count_val=layers_b)
    
    # Styles for active/inactive states
    base_style = "transition: all 0.3s; border-radius: 12px; padding: 10px;"
    
    # Pink is default color, blue only used in dual_color mode for Model A
    pink = "#ff5ca9"
    blue = "#3b82f6"
    
    style_pink = f"{base_style} border: 2px solid {pink}; background: rgba(255, 92, 169, 0.05);"
    style_blue = f"{base_style} border: 2px solid {blue}; background: rgba(59, 130, 246, 0.05);"
    style_inactive = f"{base_style} border: 2px solid transparent;"
    
    if dual_color and active_model == "both":
        # Compare Models mode: A=blue, B=pink
        style_a = style_blue
        style_b = style_pink
    elif active_model == "A":
        # Single mode or Compare Prompts: selected model gets pink
        style_a = style_pink
        style_b = style_inactive
    elif active_model == "B":
        # Single mode or Compare Prompts: selected model gets pink
        style_a = style_inactive
        style_b = style_pink
    else:
        # Default fallback: both inactive
        style_a = style_inactive
        style_b = style_inactive

    return ui.HTML(
        f'<div style="background-color: #ffffff; border-radius: 16px; padding: 30px; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);">'
        f'<div style="text-align: center; margin-bottom: 30px;">'
        f'<h4 style="color: #ff5ca9; font-weight: 800; letter-spacing: 2px; margin: 0; font-size: 18px; text-transform: uppercase;">Architectures Available</h4>'
        f'</div>'
        f'<div style="display: flex; align-items: center; justify-content: center; gap: 40px; width: 100%;">'
        f'<div style="{style_a}">{diagram_a}</div>'
        f'<div style="{style_b}">{diagram_b}</div>'
        f'</div>'
        f'</div>'
    )


def get_gusnet_architecture_section(selected_model="gusnet-bert", compare_mode=False, compare_prompts=False, model_a=None, model_b=None):
    """
    GUS-Net architecture diagrams for bias detection.
    Shows two dynamic diagrams (BERT and GPT-2) that adapt layers/head
    based on the selected model variant.

    selected_model: active model key (used in single/compare_prompts mode)
    compare_mode: If True, compare models mode (model_a=blue, model_b=pink)
    compare_prompts: If True, compare prompts mode (only selected model has pink)
    model_a: Model selected as A in compare mode
    model_b: Model selected as B in compare mode
    """
    # Colors
    blue = "#3b82f6"
    green = "#10b981"
    pink = "#ff5ca9"

    # Model specs lookup
    BERT_FAMILY = {"gusnet-bert": (12, 768), "gusnet-bert-large": (24, 1024), "gusnet-bert-new": (12, 768), "gusnet-bert-paper": (12, 768), "gusnet-bert-custom": (12, 768)}
    GPT2_FAMILY = {"gusnet-gpt2": (12, 768), "gusnet-gpt2-medium": (24, 1024), "gusnet-gpt2-new": (12, 768), "gusnet-gpt2-paper": (12, 768)}

    def _is_bert(key):
        return key in BERT_FAMILY

    def _is_gpt2(key):
        return key in GPT2_FAMILY

    # Determine dynamic specs for each diagram
    if compare_mode:
        # BERT diagram: use whichever BERT variant is in model_a or model_b
        bert_key = next((k for k in (model_a, model_b) if k and _is_bert(k)), "gusnet-bert")
        gpt2_key = next((k for k in (model_a, model_b) if k and _is_gpt2(k)), "gusnet-gpt2")
    else:
        bert_key = selected_model if _is_bert(selected_model) else "gusnet-bert"
        gpt2_key = selected_model if _is_gpt2(selected_model) else "gusnet-gpt2"

    bert_layers, bert_head = BERT_FAMILY[bert_key]
    gpt2_layers, gpt2_head = GPT2_FAMILY[gpt2_key]

    # Determine highlighting
    bert_is_selected = _is_bert(selected_model)
    gpt2_is_selected = _is_gpt2(selected_model)

    if compare_mode:
        bert_is_model_a = _is_bert(model_a) if model_a else False
        bert_is_model_b = _is_bert(model_b) if model_b else False
        gpt2_is_model_a = _is_gpt2(model_a) if model_a else False
        gpt2_is_model_b = _is_gpt2(model_b) if model_b else False
    else:
        bert_is_model_a = False
        bert_is_model_b = False
        gpt2_is_model_a = False
        gpt2_is_model_b = False

    # Styles
    base_section = "position:relative;padding:16px 8px 12px 8px;border-radius:12px;border:2px solid;display:flex;flex-direction:column;align-items:center;width:120px;transition:all 0.3s ease;"

    block_style = "position:relative;z-index:10;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;height:33px;font-size:7px;padding:0 4px;border-radius:6px;border:1px solid rgba(255,255,255,0.15);background-color:#0f172a;width:100px;transition:transform 0.2s;margin-bottom:0;box-shadow:0 1px 3px rgba(0,0,0,0.2);"
    block_label = "font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:#e2e8f0;margin-bottom:1px;"
    block_sub = "font-family:'JetBrains Mono',monospace;font-size:6px;color:#94a3b8;"
    badge_style = "position:absolute;top:-5px;right:6px;font-family:'JetBrains Mono',monospace;font-size:5px;padding:1px 3px;border-radius:3px;background:rgba(30,41,59,0.9);color:#94a3b8;border:1px solid rgba(255,255,255,0.12);z-index:20;"
    layer_box = "position:relative;padding:8px;border-radius:8px;border:1px solid rgba(255,92,169,0.3);background-color:rgba(255,92,169,0.06);display:flex;flex-direction:column;align-items:center;width:100%;margin:6px 0;"
    connector = "width:1px;height:12px;background-color:#cbd5e1;margin:3px auto;"

    # LLRD visualization (green bars) for GPT-2
    llrd_html = """
        <div style="width:60%;margin:3px auto 0;">
            <div style="height:2px;background:#10b981;margin-bottom:1px;border-radius:1px;opacity:1.0;width:100%;"></div>
            <div style="height:2px;background:#10b981;margin-bottom:1px;border-radius:1px;opacity:0.85;width:90%;"></div>
            <div style="height:2px;background:#10b981;margin-bottom:1px;border-radius:1px;opacity:0.5;width:70%;"></div>
        </div>
    """

    # --- Helper to resolve diagram colors ---
    def _resolve_colors(is_model_a, is_model_b, is_selected):
        """Return (border, bg, accent, accent_rgb, text_dark, text_light, text_accent)."""
        if compare_mode:
            if is_model_a:
                return (f"border-color:{blue};box-shadow:0 0 20px rgba(59,130,246,0.2);",
                        "background:rgba(59,130,246,0.08);",
                        blue, "59,130,246", "#2563eb", "#93c5fd", "#60a5fa")
            elif is_model_b:
                return (f"border-color:{pink};box-shadow:0 0 20px rgba(255,92,169,0.2);",
                        "background:rgba(255,92,169,0.08);",
                        pink, "255,92,169", "#db2777", "#fbcfe8", "#f9a8d4")
            else:
                return ("border-color:rgba(255,92,169,0.2);opacity:0.4;",
                        "background:rgba(255,92,169,0.02);",
                        pink, "255,92,169", "#db2777", "#fbcfe8", "#f9a8d4")
        elif is_selected:
            return (f"border-color:{pink};box-shadow:0 0 20px rgba(255,92,169,0.2);",
                    "background:rgba(255,92,169,0.08);",
                    pink, "255,92,169", "#db2777", "#fbcfe8", "#f9a8d4")
        else:
            return ("border-color:rgba(255,92,169,0.2);opacity:0.4;",
                    "background:rgba(255,92,169,0.02);",
                    pink, "255,92,169", "#db2777", "#fbcfe8", "#f9a8d4")

    # --- BERT Diagram (dynamic: adapts to base/large) ---
    (bert_border, bert_bg, bert_accent, bert_accent_rgb,
     bert_text_dark, bert_text_light, bert_text_accent) = _resolve_colors(
        bert_is_model_a, bert_is_model_b, bert_is_selected)

    bert_html = f"""
        <div style="{base_section}{bert_border}{bert_bg}">
            <div style="{block_style}border-color:rgba({bert_accent_rgb},0.4);">
                <span style="{block_label}color:{bert_accent};">BertTokenizerFast</span>
                <span style="{block_sub}">Pad: Max Length (128) | WordPiece</span>
            </div>

            <div style="{connector}"></div>

            <div style="{layer_box}border-color:rgba({bert_accent_rgb},0.3);background:rgba({bert_accent_rgb},0.06);">
                <div style="{badge_style}color:{bert_accent};">ENC</div>
                <div style="{block_style}width:100%;background:rgba({bert_accent_rgb},0.1);border-color:rgba({bert_accent_rgb},0.3);margin-bottom:4px;">
                    <span style="{block_label}color:{bert_text_dark};">BertForTokenClas</span>
                    <span style="{block_sub}color:{bert_text_dark};">{bert_layers} Layers (Encoder)</span>
                </div>
                <div style="{connector}height:4px;"></div>
                <div style="{block_style}width:100%;margin-bottom:4px;">
                    <span style="{block_label}">Linear Head</span>
                    <span style="{block_sub}">{bert_head} → 7 Labels</span>
                </div>
                <div style="{connector}height:4px;"></div>
                <div style="{block_style}width:100%;border-color:rgba({bert_accent_rgb},0.5);">
                    <span style="{block_label}color:{bert_text_light};">Sigmoid+Thresh</span>
                    <span style="{block_sub}font-weight:700;color:{bert_text_accent};">Dynamic Opt.</span>
                </div>
            </div>

            <div style="{connector}"></div>

            <div style="{layer_box}border-color:rgba(255,255,255,0.1);background:rgba(255,255,255,0.02);">
                <div style="{badge_style}color:#64748b;">OPT</div>
                <div style="{block_style}width:100%;padding:4px 2px;height:auto;min-height:0;margin-bottom:3px;">
                    <span style="{block_label}font-size:7px;color:#ffffff;">Focal Loss</span>
                    <span style="{block_sub}color:#cbd5e1;">α=0.75, γ=3.0</span>
                </div>
                <div style="{block_style}width:100%;padding:4px 2px;height:auto;min-height:0;">
                    <span style="{block_label}font-size:7px;color:#ffffff;">Optimizer</span>
                    <span style="{block_sub}color:#cbd5e1;">AdamW + Warmup</span>
                </div>
            </div>
        </div>
    """

    # --- GPT-2 Diagram (dynamic: adapts to base/medium) ---
    (gpt2_border, gpt2_bg, gpt2_accent, gpt2_accent_rgb,
     gpt2_text_dark, gpt2_text_light, gpt2_text_accent) = _resolve_colors(
        gpt2_is_model_a, gpt2_is_model_b, gpt2_is_selected)

    gpt2_html = f"""
        <div style="{base_section}{gpt2_border}{gpt2_bg}">
            <div style="{block_style}border-color:rgba({gpt2_accent_rgb},0.4);">
                <span style="{block_label}color:{gpt2_accent};">GPT2TokenizerFast</span>
                <span style="{block_sub}">Pad: Right | BPE</span>
            </div>

            <div style="{connector}"></div>

            <div style="{layer_box}border-color:rgba({gpt2_accent_rgb},0.3);background:rgba({gpt2_accent_rgb},0.06);">
                <div style="{badge_style}color:{gpt2_accent};">DEC</div>
                <div style="{block_style}width:100%;background:rgba({gpt2_accent_rgb},0.1);border-color:rgba({gpt2_accent_rgb},0.3);margin-bottom:4px;">
                    <span style="{block_label}color:{gpt2_text_dark};">GPT2ForTokenClas</span>
                    <span style="{block_sub}color:{gpt2_text_dark};">{gpt2_layers} Layers (Decoder)</span>
                </div>
                <div style="{connector}height:4px;"></div>
                <div style="{block_style}width:100%;margin-bottom:4px;">
                    <span style="{block_label}">Linear Head</span>
                    <span style="{block_sub}">{gpt2_head} → 7 Labels</span>
                </div>
                <div style="{connector}height:4px;"></div>
                <div style="{block_style}width:100%;border-color:rgba({gpt2_accent_rgb},0.5);">
                    <span style="{block_label}color:{gpt2_text_light};">Sigmoid+Thresh</span>
                    <span style="{block_sub}font-weight:700;color:{gpt2_text_accent};">Dynamic Opt.</span>
                </div>
            </div>

            <div style="{connector}"></div>

            <div style="{layer_box}border-color:rgba(255,255,255,0.1);background:rgba(255,255,255,0.02);">
                <div style="{badge_style}color:#64748b;">OPT</div>
                <div style="{block_style}width:100%;padding:4px 2px;height:auto;min-height:0;margin-bottom:3px;">
                    <span style="{block_label}font-size:7px;color:#ffffff;">Focal Loss</span>
                    <span style="{block_sub}color:#cbd5e1;">γ=2.0, Sm=0.05</span>
                </div>
                <div style="{block_style}width:100%;padding:4px 2px;height:auto;min-height:0;">
                    <span style="{block_label}font-size:7px;color:#ffffff;">LLRD Optimizer</span>
                    <span style="{block_sub}color:#cbd5e1;">Decay: 0.85</span>
                    {llrd_html}
                </div>
            </div>
        </div>
    """

    # Output example (NER inference) - dark theme
    output_html = f"""
        <div style="position:relative;padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);background:#1e293b;display:flex;flex-direction:column;align-items:center;width:150px;margin-top:12px;">
            <div style="{badge_style}color:#94a3b8;">OUTPUT</div>
            <span style="{block_label}font-size:6px;color:#94a3b8;margin-bottom:4px;">Inference Example</span>

            <div style="display:flex;flex-direction:row;justify-content:center;align-items:flex-start;width:100%;gap:2px;">
                <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:6px;color:#e2e8f0;padding:1px 0;background:#0f172a;border-radius:3px;border:1px solid rgba(255,255,255,0.1);margin-bottom:2px;width:100%;text-align:center;">Women</div>
                    <div style="font-size:4px;font-weight:bold;padding:1px 0;border-radius:2px;text-transform:uppercase;width:100%;text-align:center;background:rgba(139,92,246,0.3);color:#a78bfa;border:1px solid rgba(139,92,246,0.5);">B-GEN</div>
                </div>
                <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:6px;color:#e2e8f0;padding:1px 0;background:#0f172a;border-radius:3px;border:1px solid rgba(255,255,255,0.1);margin-bottom:2px;width:100%;text-align:center;">cannot</div>
                    <div style="height:6px;"></div>
                </div>
                <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:6px;color:#e2e8f0;padding:1px 0;background:#0f172a;border-radius:3px;border:1px solid rgba(255,255,255,0.1);margin-bottom:2px;width:100%;text-align:center;">drive</div>
                    <div style="font-size:4px;font-weight:bold;padding:1px 0;border-radius:2px;text-transform:uppercase;width:100%;text-align:center;background:rgba(249,115,22,0.3);color:#fb923c;border:1px solid rgba(249,115,22,0.5);">B-STER</div>
                    <div style="font-size:4px;font-weight:bold;padding:1px 0;border-radius:2px;text-transform:uppercase;width:100%;text-align:center;background:rgba(239,68,68,0.3);color:#f87171;border:1px solid rgba(239,68,68,0.5);margin-top:1px;">B-UNF</div>
                </div>
                <div style="display:flex;flex-direction:column;align-items:center;width:30px;">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:6px;color:#e2e8f0;padding:1px 0;background:#0f172a;border-radius:3px;border:1px solid rgba(255,255,255,0.1);margin-bottom:2px;width:100%;text-align:center;">well</div>
                    <div style="font-size:4px;font-weight:bold;padding:1px 0;border-radius:2px;text-transform:uppercase;width:100%;text-align:center;background:rgba(249,115,22,0.3);color:#fb923c;border:1px solid rgba(249,115,22,0.5);">I-STER</div>
                    <div style="font-size:4px;font-weight:bold;padding:1px 0;border-radius:2px;text-transform:uppercase;width:100%;text-align:center;background:rgba(239,68,68,0.3);color:#f87171;border:1px solid rgba(239,68,68,0.5);margin-top:1px;">I-UNF</div>
                </div>
            </div>

            <div style="margin-top:6px;display:flex;gap:6px;justify-content:center;width:100%;border-top:1px solid rgba(255,255,255,0.08);padding-top:4px;">
                <div style="display:flex;align-items:center;gap:2px;"><div style="width:3px;height:3px;background:#a78bfa;border-radius:50%;"></div><span style="font-size:4px;color:#94a3b8;font-family:monospace;letter-spacing:0.05em;">GEN</span></div>
                <div style="display:flex;align-items:center;gap:2px;"><div style="width:3px;height:3px;background:#fb923c;border-radius:50%;"></div><span style="font-size:4px;color:#94a3b8;font-family:monospace;letter-spacing:0.05em;">STEREO</span></div>
                <div style="display:flex;align-items:center;gap:2px;"><div style="width:3px;height:3px;background:#f87171;border-radius:50%;"></div><span style="font-size:4px;color:#94a3b8;font-family:monospace;letter-spacing:0.05em;">UNFAIR</span></div>
            </div>
        </div>
    """

    # Merge connector (2 inputs → 1 output)
    merge_connector = """
        <div style="width:140px;height:16px;position:relative;margin-top:6px;">
            <div style="position:absolute;left:25%;top:0;width:1px;height:8px;background:#cbd5e1;"></div>
            <div style="position:absolute;right:25%;top:0;width:1px;height:8px;background:#cbd5e1;"></div>
            <div style="position:absolute;top:8px;left:25%;right:25%;height:1px;background:#cbd5e1;"></div>
            <div style="position:absolute;top:8px;left:50%;height:8px;width:1px;background:#cbd5e1;"></div>
        </div>
    """

    # Model labels for compare mode - REMOVED per user request
    if compare_mode:
        label_A = ""
        label_B = ""
    else:
        label_A = ""
        label_B = ""

    # Main container (2 diagrams)
    return ui.HTML(
        f'<div style="background-color:#ffffff;border-radius:16px;padding:20px;display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;border:1px solid #e2e8f0;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1),0 2px 4px -1px rgba(0,0,0,0.06);">'
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<h4 style="color:#ff5ca9;font-weight:800;letter-spacing:2px;margin:0;font-size:13px;text-transform:uppercase;">GUS-Net Training Pipeline</h4>'
        f'</div>'
        f'<div style="display:flex;align-items:flex-start;justify-content:center;gap:12px;width:100%;">'
        f'<div>{label_A}{bert_html}</div>'
        f'<div>{label_B}{gpt2_html}</div>'
        f'</div>'
        f'{merge_connector}'
        f'{output_html}'
        f'</div>'
    )


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


# Assuming array_to_base64_img is defined here or imported from elsewhere
# For the purpose of this edit, we'll place matrix_to_base64_img after the assumed location of array_to_base64_img.
# The provided context implies array_to_base64_img ends with plt.close(fig) and buf.seek(0) and returns base64.
# Since array_to_base64_img is not in the provided content, we'll insert matrix_to_base64_img before get_embedding_table.

# Placeholder for array_to_base64_img if it were in the file:
# def array_to_base64_img(arr, cmap="Blues", height=0.18):
#     import matplotlib.pyplot as plt
#     import io
#     import base64
#     fig, ax = plt.subplots(figsize=(len(arr) * 0.1, height))
#     ax.imshow(arr[np.newaxis, :], cmap=cmap, aspect='auto')
#     ax.axis('off')
#     plt.tight_layout(pad=0)
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     return base64.b64encode(buf.getvalue()).decode('utf-8')


def matrix_to_base64_img(matrix, cmap="Blues", figsize=(5, 5)):
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np
    fig, ax = plt.subplots(figsize=figsize)
    # Use generic heatmap
    ax.imshow(matrix, cmap=cmap, aspect='equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def get_embedding_table(res, top_k=3, suffix=""):
    tokens, embeddings, *_ = res
    n = len(tokens)
    unique_id = f"embed_tab{suffix}"
    print(f"DEBUG: get_embedding_table called with suffix='{suffix}', unique_id='{unique_id}'")

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
            <th style='text-align:left;padding-left:8px; width:55%;'>Token</th>
            <th style='text-align:center; width:45%;'>L2 Norm (Magnitude)</th>
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


def get_posenc_table(res, top_k=3, suffix=""):
    tokens, _, pos_enc, *_ = res
    n = len(tokens)
    unique_id = f"pos_tab{suffix}"

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


def _render_dual_tab_view(unique_id, html_heatmap, tokens, vectors_for_pca, html_change=None, controls_style="justify-content: center;"):
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
        <div class='view-controls' style='{controls_style}'>
            {change_btn}
            <button class='view-btn {heat_active_class}' data-tab='heat' onclick="switchView('{unique_id}', 'heat')" title="Visual heatmap of vector values" style="flex: 0 1 auto; width: 25%;">Raw Vectors</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}', 'pca')" title="2D Principal Component Analysis projection" style="flex: 0 1 auto; width: 25%;">PCA</button>
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

def get_sum_layernorm_view(res, encoder_model, suffix=""):
    tokens, _, _, _, hidden_states, inputs, *_ = res
    unique_id = f"sumnorm_tab{suffix}"
    
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

    return _render_dual_tab_view(unique_id, html_heatmap, tokens, combined_vectors, controls_style="justify-content: center; padding: 8px 0;")


def get_qkv_table(res, layer_idx, top_k=3, suffix="", norm_mode="raw", use_global=False):
    """
    Generate Q/K/V projection table with multiple views.

    Args:
    layer_idx: Layer index to visualize
        top_k: Number of top neighbors to show
        suffix: Suffix for unique IDs (for comparison mode)
        norm_mode: Normalization mode for attention weights ("raw", "col", "rollout")
        use_global: Whether to compute metrics across all layers (averaged)
    """
    tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
    unique_id = f"qkv_tab{suffix}"
    
    # Determine if causal (GPT-2) - check first layer
    layer_block_0 = get_layer_block(encoder_model, 0)
    is_causal = not hasattr(layer_block_0, "attention")

    if is_causal:
        num_layers = len(encoder_model.h)
    else:
        num_layers = len(encoder_model.encoder.layer)

    # --- Global Mode Logic ---
    if use_global:
        # Iterate over ALL layers and average the metrics
        # Note: We cannot average Q/K/V vectors themselves (different spaces)
        # But we CAN average their Norms and Similarity matrices.
        
        avg_q_norms = np.zeros(len(tokens))
        avg_k_norms = np.zeros(len(tokens))
        avg_v_norms = np.zeros(len(tokens))
        avg_dir_alignment = np.zeros((len(tokens), len(tokens)))
        avg_qk_sim = np.zeros((len(tokens), len(tokens))) # For alignment view
        
        valid_layers = 0
        
        for l_idx in range(num_layers):
            try:
                lb = get_layer_block(encoder_model, l_idx)
                hs = hidden_states[l_idx]
                Q_l, K_l, V_l = extract_qkv(lb, hs)
                
                # Norms
                avg_q_norms += np.linalg.norm(Q_l, axis=1)
                avg_k_norms += np.linalg.norm(K_l, axis=1)
                avg_v_norms += np.linalg.norm(V_l, axis=1)
                
                # Directional Alignment
                q_normed = Q_l / (np.linalg.norm(Q_l, axis=1, keepdims=True) + 1e-8)
                k_normed = K_l / (np.linalg.norm(K_l, axis=1, keepdims=True) + 1e-8)
                avg_dir_alignment += np.dot(q_normed, k_normed.T)
                
                # QK Sim (for table)
                avg_qk_sim += _compute_cosine_similarity_matrix(Q_l) @ _compute_cosine_similarity_matrix(K_l).T
                
                valid_layers += 1
            except:
                continue
                
        if valid_layers > 0:
            q_norms = (avg_q_norms / valid_layers).tolist()
            k_norms = (avg_k_norms / valid_layers).tolist()
            v_norms = (avg_v_norms / valid_layers).tolist()
            dir_alignment = avg_dir_alignment / valid_layers
            qk_sim = (avg_qk_sim / valid_layers)
            qk_sim = (qk_sim + 1) / 2 # Normalize to 0-1 range roughly
        else:
             return ui.HTML("Error computing global metrics.")
             
        # Average Attention Weights across ALL layers/heads
        att_layers = [layer[0].cpu().numpy() for layer in attentions]
        att_avg = np.mean(att_layers, axis=(0, 1))

        # Title adjustments
        att_label = "Average Attention (all layers)"
        layer_desc = "Averaged across all layers"
        
        # Disable Vector views for Global mode (meaningless to show one)
        # We'll hide the buttons/tabs logic below
        show_vectors = False

    else:
        # --- Single Layer Logic (Existing) ---
        layer_block = get_layer_block(encoder_model, layer_idx)
        hs_in = hidden_states[layer_idx]
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

        # Compute TRUE directional alignment: cosine similarity between Q[i] and K[j]
        # This shows how aligned Q and K vectors are, independent of magnitude
        q_normalized = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        k_normalized = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-8)
        dir_alignment = np.dot(q_normalized, k_normalized.T)  # Range: [-1, 1]

        # Get attention weights for this layer (averaged across heads for comparison)
        att_layer = attentions[layer_idx][0].cpu().numpy()  # Shape: (num_heads, seq_len, seq_len)
        att_avg = np.mean(att_layer, axis=0)  # Average across heads
        
        att_label = "Attention Weights (query perspective)"
        layer_desc = f"Layer {layer_idx}"
        show_vectors = True

    # Common Logic continues...
    tokens_len = len(tokens) # used later for loops
    
    # -------------------------------------------------------------
    # (Rest of normalization/display logic, reusing variables)
    # -------------------------------------------------------------


    # Apply normalization to attention weights based on mode
    # Note: In global mode, att_avg is already the average across all layers
    if norm_mode == "col":
        # Column normalization: each column sums to 1
        col_sums = att_avg.sum(axis=0, keepdims=True)
        att_avg = att_avg / (col_sums + 1e-8)
    elif norm_mode == "rollout" and not use_global:
        # Attention rollout: accumulated attention flow through layers
        # (Only applies in single-layer mode - in global mode we use the average)
        att_per_layer = []
        for l_idx in range(layer_idx + 1):
            layer_att = attentions[l_idx][0].cpu().numpy()
            layer_att_avg = np.mean(layer_att, axis=0)
            att_per_layer.append(layer_att_avg)

        seq_len = att_per_layer[0].shape[0]
        rollout = np.eye(seq_len)

        for l_idx in range(layer_idx + 1):
            attention = att_per_layer[l_idx]
            # Add residual connection
            attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
            # Re-normalize rows
            row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
            attention_with_residual = attention_with_residual / (row_sums + 1e-8)
            # Accumulate
            rollout = np.matmul(attention_with_residual, rollout)

        att_avg = rollout
    elif norm_mode == "rollout" and use_global:
        # In global mode with rollout, compute rollout across ALL layers
        att_per_layer = []
        for l_idx in range(num_layers):
            layer_att = attentions[l_idx][0].cpu().numpy()
            layer_att_avg = np.mean(layer_att, axis=0)
            att_per_layer.append(layer_att_avg)

        seq_len = att_per_layer[0].shape[0]
        rollout = np.eye(seq_len)

        for l_idx in range(num_layers):
            attention = att_per_layer[l_idx]
            attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
            row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
            attention_with_residual = attention_with_residual / (row_sums + 1e-8)
            rollout = np.matmul(attention_with_residual, rollout)

        att_avg = rollout

    # Get normalization mode label for display
    if use_global:
        if norm_mode == "raw":
            att_label = "Average Attention (all layers)"
        elif norm_mode == "col":
            att_label = "Average Attention - Key Perspective (all layers)"
        else:
            att_label = f"Accumulated Attention Flow (0→{num_layers-1})"
    else:
        if norm_mode == "raw":
            att_label = "Attention Weights (query perspective)"
        elif norm_mode == "col":
            att_label = "Attention Weights (key perspective)"
        else:
            att_label = f"Accumulated Attention Flow (0→{layer_idx})"

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

    # 3. PCA View (Only if show_vectors is True)
    if show_vectors:
        html_pca = _render_qkv_pca_scatter(tokens, Q, K, V)
    else:
        html_pca = "<div style='text-align:center;padding:20px;color:#94a3b8;'>PCA not available in Global Mode (vectors differ per layer).</div>"

    # 4. Raw Vector View (Only if show_vectors is True)
    if show_vectors:
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
        html_vec = f"<table class='combined-summary-table'><tr><th>Token</th><th>Q Vector</th><th>K Vector</th><th>V Vector</th></tr>{''.join(vec_rows)}</table>"
    else:
        html_vec = "<div style='text-align:center;padding:20px;color:#94a3b8;'>Raw vectors not available in Global Mode.</div>"

    # 5. Directional Alignment View (Cosine Similarity Q vs K)
    # Generate heatmap images for side-by-side comparison
    dir_align_img = matrix_to_base64_img(dir_alignment, cmap="RdBu_r", figsize=(4, 4))
    att_weights_img = matrix_to_base64_img(att_avg, cmap="Blues", figsize=(4, 4))

    # Clean token labels for axis display
    clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
    token_labels_html = "".join([f"<span class='axis-label'>{t}</span>" for t in clean_tokens])

    html_dir = f"""
    <div class='directional-alignment-container'>
        <div style='margin-bottom: 10px; display: flex; align-items: flex-start; gap: 6px;'>
            <div class='info-tooltip-wrapper'>
                <span class='info-tooltip-icon' style='font-size:8px; width:14px; height:14px; line-height:14px;'>i</span>
                <div class='info-tooltip-content' style='width: 360px;'>
                    <strong>Directional Alignment (Cosine Similarity)</strong>
                    <p>Measures how aligned Q and K vectors are in direction, independent of magnitude.</p>
                    <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 8px 0;'>
                        <code style='font-size: 10px;'>cos(Q, K) = (Q · K) / (||Q|| × ||K||)</code>
                    </div>
                    <p><strong style='color:#ef4444;'>Red:</strong> Similar directions (high alignment)</p>
                    <p><strong style='color:#3b82f6;'>Blue:</strong> Opposite directions (negative)</p>
                    <p><strong style='color:#9ca3af;'>White:</strong> Orthogonal (no relationship)</p>

                    <hr style='border-color: rgba(255,255,255,0.2); margin: 10px 0;'>
                    <strong>Why compare with Attention Weights?</strong>
                    <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 8px 0;'>
                        <code style='font-size: 10px;'>Attention = softmax(Q·K / √dk)</code>
                    </div>
                    <p style='font-size: 10px;'>The dot product Q·K combines both direction AND magnitude. Cosine isolates just the direction.</p>
                    <p style='font-size: 10px;'><strong>→</strong> If attention is high where cosine is low: the model may be relying on vector magnitude rather than semantic similarity.</p>
                    <p style='font-size: 10px;'><strong>→</strong> If attention is low where cosine is high: other tokens with larger magnitudes may be "stealing" attention.</p>

                    <hr style='border-color: rgba(255,255,255,0.2); margin: 10px 0;'>
                    <strong>Why does this change per layer?</strong>
                    <p style='font-size: 10px;'>Each layer has its own learned Wq and Wk matrices. Early layers typically capture positional and syntactic patterns, while deeper layers encode semantic and contextual relationships.</p>
                </div>
            </div>
            <span style='font-size: 10px; color: #6b7280; line-height: 1.4;'>Compares Q·K directional similarity (cosine) with actual attention weights. Helps identify if attention is driven by genuine semantic alignment or by vector magnitude effects.</span>
        </div>
        <div class='heatmap-comparison'>
            <div class='heatmap-panel'>
                <div class='heatmap-title'>Directional Alignment (Cosine)</div>
                <div class='heatmap-subtitle'>Q → K similarity by direction</div>
                <div class='heatmap-wrapper'>
                    <img class='comparison-heatmap' src='data:image/png;base64,{dir_align_img}' alt='Directional Alignment Heatmap'>
                    <div class='heatmap-colorbar cosine-colorbar'>
                        <span>-1</span>
                        <div class='colorbar-gradient cosine-gradient'></div>
                        <span>+1</span>
                    </div>
                </div>
            </div>
            <div class='heatmap-panel'>
                <div class='heatmap-title'>{att_label}</div>
                <div class='heatmap-subtitle'>{"Raw attention (avg across heads)" if norm_mode == "raw" else "Column-normalized" if norm_mode == "col" else f"Rollout through layers 0→{layer_idx}"}</div>
                <div class='heatmap-wrapper'>
                    <img class='comparison-heatmap' src='data:image/png;base64,{att_weights_img}' alt='Attention Weights Heatmap'>
                    <div class='heatmap-colorbar attention-colorbar'>
                        <span>0</span>
                        <div class='colorbar-gradient attention-gradient'></div>
                        <span>1</span>
                    </div>
                </div>
            </div>
        </div>
        <div class='axis-labels-container'>
            <div class='axis-label-row'>
                <span style='font-size: 9px; color: #94a3b8; margin-right: 8px;'>Tokens:</span>
                {token_labels_html}
            </div>
        </div>
    </div>
    """

    # Assemble Tabbed Interface
    html = f"""
    <div id='{unique_id}'>
        <div class='view-controls qkv-controls'>
            <button class='view-btn active' data-tab='norm' onclick="switchView('{unique_id}', 'norm')" title="View vector magnitude (L2 Norm)">Norms</button>
            <button class='view-btn' data-tab='dir' onclick="switchView('{unique_id}', 'dir')" title="Cosine similarity between Q and K vectors (direction only, ignores magnitude)">Directional</button>
            <button class='view-btn' data-tab='sim' onclick="switchView('{unique_id}', 'sim')" title="Q·K alignment showing potential attention targets">Alignment</button>
            <button class='view-btn' data-tab='pca' onclick="switchView('{unique_id}', 'pca')" title="2D Principal Component Analysis projection">PCA</button>
            <button class='view-btn' data-tab='vec' onclick="switchView('{unique_id}', 'vec')" title="Visual heatmap of raw vector values">Raw Vectors</button>
        </div>

        <div class='card-scroll vector-summary-container'>
            <div id='{unique_id}_norm' class='view-pane' style='display:block;'>
                {html_norm}
            </div>
            <div id='{unique_id}_dir' class='view-pane'>
                {html_dir}
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


def get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=3, norm_mode="raw", use_global=False):
    """
    Generate scaled attention view showing attention computation details.

        focus_indices: Token indices to show attention for
        top_k: Number of top attention targets to show
        norm_mode: Normalization mode ("raw", "col", "rollout") - affects ranking
        use_global: Whether to average across all heads/layers (hides Q/K details)
    """
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

    if use_global:
        # Global Mode: Average attention across ALL layers/heads
        att_layers = [layer[0].cpu().numpy() for layer in attentions]
        # att_layers shape: list of (heads, seq, seq)
        # mean over layers -> (heads, seq, seq)
        # mean over heads -> (seq, seq)
        att_head = np.mean([np.mean(l, axis=0) for l in att_layers], axis=0)
        
        # Determine number of heads from first layer just for d_k calc (unused but needed for var init)
        layer_block_0 = get_layer_block(encoder_model, 0)
        if hasattr(layer_block_0, "attention"): # BERT
            num_heads = layer_block_0.attention.self.num_attention_heads
            dim = hidden_states[0].shape[-1]
            is_causal = False
        else: # GPT-2
            num_heads = layer_block_0.attn.num_heads
            dim = hidden_states[0].shape[-1]
            is_causal = True
        
        # d_k, Q, K are not valid for global average
        d_k = 1
        Q, K = None, None
        
        note_html = "<div style='font-size:10px;color:#6b7280;margin-bottom:8px;font-style:italic;'>Values averaged across all layers and heads. Calculation details hidden.</div>"
    
    else:
        # Standard Single Head Mode
        att_head = attentions[layer_idx][0, head_idx].cpu().numpy()
        layer_block = get_layer_block(encoder_model, layer_idx)
        hs_in = hidden_states[layer_idx]
        Q, K, V = extract_qkv(layer_block, hs_in)

        is_causal = not hasattr(layer_block, "attention")

        if hasattr(layer_block, "attention"): # BERT
            num_heads = layer_block.attention.self.num_attention_heads
        else: # GPT-2
            num_heads = layer_block.attn.num_heads
        d_k = Q.shape[-1] // num_heads
        note_html = ""

    # Compute normalized attention if needed
    if norm_mode == "col":
        # Column normalization
        col_sums = att_head.sum(axis=0, keepdims=True)
        att_normalized = att_head / (col_sums + 1e-8)
        norm_label = "col-norm"
    elif norm_mode == "rollout":
        # Attention rollout
        att_per_layer = []
        for l_idx in range(layer_idx + 1):
            layer_att = attentions[l_idx][0].cpu().numpy()
            layer_att_avg = np.mean(layer_att, axis=0)
            att_per_layer.append(layer_att_avg)

        seq_len = att_per_layer[0].shape[0]
        rollout = np.eye(seq_len)

        for l_idx in range(layer_idx + 1):
            attention = att_per_layer[l_idx]
            attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
            row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
            attention_with_residual = attention_with_residual / (row_sums + 1e-8)
            rollout = np.matmul(attention_with_residual, rollout)

        att_normalized = rollout
        norm_label = "rollout"
    else:
        att_normalized = att_head
        norm_label = None

    all_blocks = ""

    for f_idx in focus_indices:
        f_idx = max(0, min(f_idx, len(tokens) - 1))

        # Get top k for this token using normalized values for ranking
        if not is_causal: # BERT
            top_idx = np.argsort(att_normalized[f_idx])[::-1][:top_k]
            causal_note = "<div style='font-size:10px;margin-bottom:4px;visibility:hidden;'>Causal: Future tokens are masked</div>"
        else: # GPT-2 (Causal)
            valid_scores = [(j, att_normalized[f_idx, j]) for j in range(len(tokens)) if j <= f_idx]
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            top_idx = [x[0] for x in valid_scores[:top_k]]
            causal_note = "<div style='font-size:10px;color:#888;margin-bottom:4px;font-style:italic;'>Causal: Future tokens are masked</div>"

        computations = causal_note
        for rank, j in enumerate(top_idx, 1):
            prob = att_head[f_idx, j]  # Raw softmax value

            # Build the values row
            if use_global:
                 # Simplified view for global
                 values_html = f"""
                        <span class='scaled-step'>Avg Attention = <b>{prob:.3f}</b></span>
                 """
            else:
                dot = float(np.dot(Q[f_idx], K[j]))
                scaled = dot / np.sqrt(d_k)
                values_html = f"""
                        <span class='scaled-step'>Q·K = <b>{dot:.2f}</b></span>
                        <span class='scaled-step'>÷√d<sub>k</sub> = <b>{scaled:.2f}</b></span>
                        <span class='scaled-step'>softmax = <b>{prob:.3f}</b></span>
                """

            # Add normalized value if not raw mode
            if norm_label:
                norm_val = att_normalized[f_idx, j]
                values_html += f"""<span class='scaled-step' style='color:#8b5cf6;'>{norm_label} = <b>{norm_val:.3f}</b></span>"""

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
                        {values_html}
                    </div>
                </div>
            </div>
            """

        # Add normalization indicator to the formula
        formula_suffix = ""
        if norm_label:
            formula_suffix = f" <span style='color:#8b5cf6;font-size:10px;'>[ranked by {norm_label}]</span>"

        # Wrap each token's block
        all_blocks += f"""
        <div class='scaled-attention-box' style='margin-bottom: 16px; border-bottom: 1px solid #f1f5f9; padding-bottom: 16px;'>
            {note_html}
            <div class='scaled-formula' style='margin-bottom:8px;'>
                <span style='color:#ff5ca9;font-weight:bold;'>{tokens[f_idx].replace("##", "").replace("Ġ", "")}</span>: { "softmax(Q·K<sup>T</sup>/√d<sub>k</sub>)" if not use_global else "Top tokens by average attention"} {formula_suffix}
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


def get_add_norm_view(res, layer_idx, suffix=""):
    tokens, _, _, _, hidden_states, *_ = res
    if layer_idx + 1 >= len(hidden_states):
        return ui.HTML("")
    hs_in = hidden_states[layer_idx][0].cpu().numpy()
    hs_out = hidden_states[layer_idx + 1][0].cpu().numpy()
    unique_id = f"addnorm_{layer_idx}{suffix}"
    
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
    html_change = f"<table class='combined-summary-table'><tr><th style='text-align:left;padding-left:8px; width:55%;'>Token</th><th style='width:45%;'>Change Magnitude</th></tr>{''.join(change_rows)}</table>"
    
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


def get_ffn_view(res, layer_idx, suffix=""):
    tokens, _, _, _, hidden_states, _, _, encoder_model, *_ = res
    if layer_idx + 1 >= len(hidden_states):
        return ui.HTML("")
    unique_id = f"ffn_{layer_idx}{suffix}"
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


def get_add_norm_post_ffn_view(res, layer_idx, suffix=""):
    tokens, _, _, _, hidden_states, *_ = res
    if layer_idx + 2 >= len(hidden_states):
        return ui.HTML("")
    hs_mid = hidden_states[layer_idx + 1][0].cpu().numpy()
    hs_out = hidden_states[layer_idx + 2][0].cpu().numpy()
    unique_id = f"addnormpost_{layer_idx}{suffix}"
    
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
    html_change = f"<table class='combined-summary-table'><tr><th style='text-align:left;padding-left:8px; width:55%;'>Token</th><th style='width:45%;'>Residual Change (FFN)</th></tr>{''.join(change_rows)}</table>"
    
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


def get_layer_output_view(res, layer_idx, suffix=""):
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
        "<table class='combined-summary-table' style='width:100%; table-layout:fixed;'>"
        "<tr>"
        "<th style='width:30%;'>Token</th>"
        "<th style='width:40%;'>Vector (64 dims)</th>"
        "<th style='width:30%; padding-left:12px;'>Statistics</th>"
        "</tr>"
        + "".join(rows)
        + "</table></div>"
    )


def get_output_probabilities(res, use_mlm, text, suffix="", top_k=5, manual_mode=False, custom_mask_indices=None):
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
    tokens_in_res = res[0]
    if len(tokens_in_res) != input_seq_len:
         return ui.HTML(
            "<div class='prediction-panel'>"
            "<p style='font-size:11px;color:#ef4444;'>MLM Predictions are not compatible with Word-Level Aggregation.</p>"
            "<p style='font-size:10px;color:#6b7280;'>Switch off 'Word Lvl' to see predictions.</p>"
            "</div>"
        )
    
    mlm_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Model prefix to distinguish A vs B tokens
    model_prefix = "B" if suffix == "_B" else "A"

    # --- Interactive Token Selector HTML ---
    # We generate this regardless, or only in manual mode?
    # In manual mode, we need it to be clickable.
    selector_html = ""
    if manual_mode:
        selector_buttons = []
        current_masks = set(custom_mask_indices) if custom_mask_indices else set()
        
        for i, tok in enumerate(mlm_tokens):
            clean_tok = tok.replace("##", "").replace("Ġ", "")
            is_masked = i in current_masks
            active_class = "masked-active" if is_masked else ""
            
            # Using span instead of button to avoid form submission, handled by JS
            # Add data-model attribute to distinguish A vs B
            btn = f"""
            <div class='maskable-token {active_class}' data-index='{i}' data-model='{model_prefix}' onclick='toggleMask({i}, "{model_prefix}")'>
                <div class='token-text'>{clean_tok}</div>
            </div>
            """
            selector_buttons.append(btn)
            
        button_id = "run_custom_mask_B" if model_prefix == "B" else "run_custom_mask"
        selector_html = f"""
        <div class='mask-selector-container'>
            <div class='mask-selector-label'>Click tokens to mask:</div>
            <div class='mask-token-list'>
                {''.join(selector_buttons)}
            </div>
            <div style='margin-top:8px;display:flex;justify-content:flex-end;'>
                <button id='{button_id}' class='metric-tag'>Predict Masked</button>
            </div>
        </div>
        """

    # --- Inference Logic ---
    logits_tensor = None
    probs = None
    target_indices = [] # Indices to visualize

    with torch.no_grad():
        if is_gpt2:
            # GPT-2: Standard Causal Prediction (Next Token)
            mlm_outputs = mlm_model(**inputs)
            probs = torch.softmax(mlm_outputs.logits, dim=-1)[0]
            logits_tensor = mlm_outputs.logits[0]
            target_indices = range(len(mlm_tokens)) # Show all
        elif manual_mode:
            # BERT Manual Multi-Masking
            # Create ONE input with specific masks
            input_ids = inputs["input_ids"].clone() # (1, seq_len)
            mask_token_id = tokenizer.mask_token_id
            
            # Apply masks
            if custom_mask_indices:
                for idx in custom_mask_indices:
                    input_ids[0, idx] = mask_token_id
            
            # Prepare inputs
            custom_inputs = {"input_ids": input_ids}
            if "attention_mask" in inputs: custom_inputs["attention_mask"] = inputs["attention_mask"]
            if "token_type_ids" in inputs: custom_inputs["token_type_ids"] = inputs["token_type_ids"]
            
            outputs = mlm_model(**custom_inputs)
            full_logits = outputs.logits # (1, seq_len, vocab)
            
            probs = torch.softmax(full_logits, dim=-1)[0] # (seq_len, vocab)
            logits_tensor = full_logits[0]
            
            # Show predictions only for MASKED indices, or all? 
            # Usually only masked are interesting.
            target_indices = sorted(list(current_masks)) if custom_mask_indices else []
        else:
            # BERT: Iterative Masking (Pseudo-Likelihood) - ORIGINAL LOGIC
            input_ids = inputs["input_ids"][0]
            seq_len = len(input_ids)
            mask_token_id = tokenizer.mask_token_id
            
            batch_input_ids = input_ids.repeat(seq_len, 1)
            batch_input_ids.fill_diagonal_(mask_token_id)
            
            attention_mask = inputs["attention_mask"].repeat(seq_len, 1) if "attention_mask" in inputs else None
            token_type_ids = inputs["token_type_ids"].repeat(seq_len, 1) if "token_type_ids" in inputs else None
            
            batch_inputs = {"input_ids": batch_input_ids}
            if attention_mask is not None: batch_inputs["attention_mask"] = attention_mask
            if token_type_ids is not None: batch_inputs["token_type_ids"] = token_type_ids
            
            outputs = mlm_model(**batch_inputs)
            full_logits = outputs.logits
            
            # Extract diagonal (masked positions)
            diagonal_logits = full_logits[torch.arange(seq_len), torch.arange(seq_len), :]
            
            probs = torch.softmax(diagonal_logits, dim=-1)
            logits_tensor = diagonal_logits
            target_indices = range(seq_len)

    cards = ""
    # Process Results
    
    predicted_sentence_html = ""
    if manual_mode:
        predicted_tokens = []
        
        # We need to reconstruct the sentence with interactive spans
        for i, tok in enumerate(mlm_tokens):
            clean_tok = tok.replace("##", "").replace("Ġ", "")
            if not clean_tok: clean_tok = "&nbsp;"
            
            # Determine state
            is_masked = i in target_indices # currently masked/predicted
            
            if is_masked:
                # Get the prediction
                token_probs = probs[i]
                top_idx = torch.argmax(token_probs).item()
                pred_tok = tokenizer.decode([top_idx]) or "[UNK]"
                clean_pred = pred_tok.replace("##", "").replace("Ġ", "")

                # Render as active predicted word
                # Class 'predicted-word' + 'masked-active' (meaning mask is ON, showing prediction)
                # Add data-model attribute to distinguish A vs B
                span = f"<span class='predicted-word masked-active' data-index='{i}' data-model='{model_prefix}' onclick='toggleMask({i}, \"{model_prefix}\")'>{clean_pred}</span>"
                predicted_tokens.append(span)
            else:
                # Original word
                # Class 'interactive-token'. Not active.
                # Add data-model attribute to distinguish A vs B
                span = f"<span class='interactive-token' data-index='{i}' data-model='{model_prefix}' onclick='toggleMask({i}, \"{model_prefix}\")'>{clean_tok}</span>"
                predicted_tokens.append(span)
        
        # Reconstruct sentence with spaces
        display_sentence = ""
        for i, html_tok in enumerate(predicted_tokens):
            # Space logic based on ORIGINAL tokens (to respect subwords)
            if i > 0:
                raw_tok = mlm_tokens[i]
                if not raw_tok.startswith("##"):
                    display_sentence += " "
            display_sentence += html_tok
            
        predicted_sentence_html = f"""
        <div class='predicted-sentence-card'>
            <div class='predicted-label'>MODEL PREDICTION</div>
            <div class='predicted-text'>
                {display_sentence}
            </div>
        </div>
        """

    # If manual mode and no masks, show prompt
    if manual_mode and not target_indices:
        cards = ""
    
    # (Removed duplicate 'cards += predicted_sentence_html')

    for i in target_indices:
        # For iterative mode, i is simple index.
        # For manual mode, i is a masked index.
        tok = mlm_tokens[i]
        
        # Clean token header
        tok_display = tok.replace("##", "").replace("Ġ", "")
        if not tok_display: tok_display = "&nbsp;"
        
        # Calculate context string for display (PLL) - Only for Iterative Mode
        context_html = ""
        if not is_gpt2 and not manual_mode:
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
        
        # In manual mode, maybe show what the original token was vs prediction?
        header_extra = ""
        if manual_mode:
            header_extra = f"<span style='font-size:10px;color:#ef4444;margin-left:6px;'>(Masked)</span>"

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
                {tok_display}
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

    results_html = ""
    if cards:
         results_html = predicted_sentence_html + f"<div class='prediction-panel'><div class='card-scroll'><div class='mlm-grid'>{cards}</div></div></div>"

    return ui.HTML(selector_html + results_html)


def get_metrics_display(res, layer_idx=None, head_idx=None, use_full_scale=False, baseline_stats=None, norm_mode="raw"):
    """
    Generate metrics display for attention patterns.

    Args:
        res: Result tuple from model computation
        layer_idx: Layer index (None for global average)
        head_idx: Head index (None for global average)
        use_full_scale: Whether to use full 0-1 scale for gauges
        baseline_stats: Pre-computed baseline statistics
        norm_mode: Normalization mode ("raw", "col", "rollout")
    """
    tokens, _, _, attentions, *_ = res
    if attentions is None or len(attentions) == 0:
        return ui.HTML("")

    att_layers = [layer[0].cpu().numpy() for layer in attentions]

    def apply_normalization(att_matrix, layer_for_rollout=None):
        """Apply normalization based on mode."""
        if norm_mode == "col":
            col_sums = att_matrix.sum(axis=0, keepdims=True)
            return att_matrix / (col_sums + 1e-8)
        elif norm_mode == "rollout" and layer_for_rollout is not None:
            # Attention rollout
            att_per_layer = []
            for l_idx in range(layer_for_rollout + 1):
                layer_att = att_layers[l_idx]
                layer_att_avg = np.mean(layer_att, axis=0)
                att_per_layer.append(layer_att_avg)

            seq_len = att_per_layer[0].shape[0]
            rollout = np.eye(seq_len)

            for l_idx in range(layer_for_rollout + 1):
                attention = att_per_layer[l_idx]
                attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
                row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
                attention_with_residual = attention_with_residual / (row_sums + 1e-8)
                rollout = np.matmul(attention_with_residual, rollout)

            return rollout
        else:
            return att_matrix

    # 1. Compute Metrics for the Selected View (Specific Head or Global Avg)
    if layer_idx is not None and head_idx is not None:
        # Single layer, single head
        att_matrix = att_layers[layer_idx][head_idx]
        att_matrix = apply_normalization(att_matrix, layer_idx)
        current_metrics = compute_all_attention_metrics(att_matrix)
    else:
        # Average across all layers and heads
        att_matrix = np.mean(att_layers, axis=(0, 1))
        att_matrix = apply_normalization(att_matrix, len(att_layers) - 1)
        current_metrics = compute_all_attention_metrics(att_matrix)

    # 2. Compute Context (Layer Stats) if in Specific Head Mode
    context_stats = {}
    if layer_idx is not None and head_idx is not None:
        # We need to compute metrics for ALL heads in this layer to get percentiles
        layer_heads = att_layers[layer_idx] # shape (num_heads, seq, seq)
        num_heads = layer_heads.shape[0]

        # Collect metrics for all heads (with normalization applied)
        layer_metrics_list = []
        for h in range(num_heads):
            head_att = apply_normalization(layer_heads[h], layer_idx)
            m = compute_all_attention_metrics(head_att)
            # Normalize focus entropy locally for comparison
            num_tokens = head_att.shape[0]
            max_ent = num_tokens * np.log(num_tokens) if num_tokens > 1 else 1
            m['focus_normalized'] = m['focus_entropy'] / max_ent if max_ent > 0 else 0
            layer_metrics_list.append(m)
        
        # Compute Percentiles and Averages for each key
        keys_to_context = ['confidence_max', 'confidence_avg', 'focus_normalized', 'sparsity', 'distribution_median', 'uniformity', 'balance']
        
        for key in keys_to_context:
            values = [stats[key] for stats in layer_metrics_list]
            current_val = current_metrics.get(key)
            if key == 'focus_normalized':
                # Re-calc current normalized for consistency
                num_tokens = att_matrix.shape[0]
                max_ent = num_tokens * np.log(num_tokens) if num_tokens > 1 else 1
                current_val = current_metrics['focus_entropy'] / max_ent if max_ent > 0 else 0
            
            # Percentile (rank)
            # strictly less count
            smaller_count = sum(v < current_val for v in values)
            percentile = (smaller_count / len(values)) * 100
            
            # Average
            avg_val = float(np.mean(values))
            
            context_stats[key] = {
                "percentile": percentile,
                "avg": avg_val,
                "is_available": True
            }
            
    # CSS for markers
    marker_styles = """
    <style>
        .gauge-marker-avg {
            position: absolute;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: #64748b;
            z-index: 10;
        }
        .gauge-marker-baseline {
            position: absolute;
            top: -4px;
            bottom: -4px;
            width: 2px;
            background: #8b5cf6; /* Purple for baseline */
            z-index: 11;
        }
        .gauge-marker-baseline::before {
            content: '';
            position: absolute;
            top: 0;
            left: -2px;
            width: 6px;
            height: 6px;
            background: #8b5cf6;
            transform: rotate(45deg);
        }
    </style>
    """
    
    # Global metrics like Flow Change don't have "layer context" in the same way
    flow_change = calculate_flow_change(att_layers)
    
    # Balance is in metrics_dict
    balance = current_metrics.get('balance', 0.5)
    
    # Normalize current focus
    num_tokens = att_matrix.shape[0]
    max_entropy = num_tokens * np.log(num_tokens) if num_tokens > 1 else 1
    focus_normalized = current_metrics['focus_entropy'] / max_entropy if max_entropy > 0 else 0

    # Thresholds / Interpretations
    # If use_full_scale is True, we overwrite max_range to 1.0 (or theoretical max)
    interpretations = {
        'confidence_max': (0.20, 0.50, 0.0, 1.0, False),
        'confidence_avg': (0.15, 0.40, 0.0, 0.8, False), # Max 0.8 reasonable
        'focus_normalized': (0.30, 0.70, 0.0, 1.0, True),
        'sparsity': (0.30, 0.60, 0.0, 1.0, False),
        'distribution_median': (0.005, 0.02, 0.0, 0.05, False), # Very small usually
        'uniformity': (0.03, 0.10, 0.0, 0.2, False),
        'flow_change': (0.10, 0.25, 0.0, 0.5, False),
        'balance': (0.15, 0.40, 0.0, 1.0, False),
    }

    def get_interpretation(key, value):
        low_max, high_min, min_r, max_r, reverse = interpretations.get(key, (0.3, 0.7, 0, 1, False))
        
        # Override scale if Full Scale requested
        # For distribution_median/uniformity/flow_change, 1.0 is technically possible but rare.
        # We will use 1.0 for most, but maybe 0.5 for flow/uniformity if 1.0 is impossible.
        # Ideally, user wants 0-1 for "Absolute".
        scale_max_display = str(max_r)
        
        if use_full_scale:
            max_r = 1.0
            scale_max_display = "1.0"
            # Special cases where 1.0 is absurdly high? 
            # Uniformity max is roughly 0.5 (if half 0 half 1). 
            # sticking to 1.0 for consistency of "Full Scale".

        gauge_pct = min(100, max(0, ((value - min_r) / (max_r - min_r)) * 100))
        
        # Recalculate zone positions relative to the NEW max_r
        low_pct = ((low_max - min_r) / (max_r - min_r)) * 100
        high_pct = ((high_min - min_r) / (max_r - min_r)) * 100
        
        # Color logic (Unchanged)
        if reverse: # Lower is better/focused
            if value <= low_max: interp = ("Focused", "#22c55e")
            elif value >= high_min: interp = ("Diffuse", "#ef4444")
            else: interp = ("Moderate", "#f59e0b")
        else: # Higher is "High"
            if value <= low_max: interp = ("Low", "#22c55e") # Green=Low (often "Good" or just "Low")
            elif value >= high_min: interp = ("High", "#ef4444")
            else: interp = ("Medium", "#f59e0b")
            
        return interp[0], interp[1], gauge_pct, low_pct, high_pct, scale_max_display, max_r, min_r

    # Metrics List
    metrics = [
        ("Confidence (Max)", current_metrics['confidence_max'], "{:.2f}", "confidence_max", "Confidence Max"),
        ("Confidence (Avg)", current_metrics['confidence_avg'], "{:.2f}", "confidence_avg", "Confidence Avg"),
        ("Focus (Normalized)", focus_normalized, "{:.2f}", "focus_normalized", "Focus"),
        ("Sparsity", current_metrics['sparsity'], "{:.0%}", "sparsity", "Sparsity"),
        ("Distribution", current_metrics['distribution_median'], "{:.3f}", "distribution_median", "Distribution"),
        ("Uniformity", current_metrics['uniformity'], "{:.3f}", "uniformity", "Uniformity"),
        ("Balance", balance, "{:.2f}", "balance", "Balance"),
        ("Flow Change", flow_change, "{:.2f}", "flow_change", "Flow Change"),
    ]

    cards_html = '<div class="metrics-grid">'
    for label, raw_value, fmt, key, modal_name in metrics:
        value_str = fmt.format(raw_value)
        label_text, color, gauge_pct, low_pct, high_pct, scale_lbl, max_r, min_r = get_interpretation(key, raw_value)
        
        # Context Info Generation
        context_html = ""
        context_marker = ""
        baseline_marker = ""
        
        # 1. Baseline Marker Computation
        b_val = None
        if baseline_stats:
            if key == "flow_change":
                # Global metric lookup
                b_val = baseline_stats.get("global", {}).get("flow_change")
            elif layer_idx is not None and head_idx is not None:
                # Head metric lookup
                b_key = (layer_idx, head_idx)
                if b_key in baseline_stats:
                    b_val = baseline_stats[b_key].get(key)
            
            if b_val is not None:
                # Calculate position for baseline
                b_pct = min(100, max(0, ((b_val - min_r) / (max_r - min_r)) * 100))
                baseline_marker = f'<div class="gauge-marker-baseline" style="left: {b_pct}%;" title="Global Baseline: {fmt.format(b_val)}"></div>'

        # 2. Context HTML Generation
        ref_html = ""
        if b_val is not None:
             ref_html = f'<div title="Average value from baseline sentences (Global Reference)" style="color:#8b5cf6;">Ref: <b>{fmt.format(b_val)}</b></div>'

        if key in context_stats and context_stats[key]["is_available"]:
            stats = context_stats[key]
            pctile = stats["percentile"]
            avg = stats["avg"]
            
            # Add marker for avg
            avg_pct = min(100, max(0, ((avg - min_r) / (max_r - min_r)) * 100))
            context_marker = f'<div class="gauge-marker-avg" style="left: {avg_pct}%;" title="Layer Avg: {fmt.format(avg)}"></div>'
            
            # Use grid for alignment: Rank | Avg | Ref
            context_html = f'''
            <div class="metric-context" style="display:flex; flex-direction:column; gap:2px; margin-top:4px;">
                <div style="display:flex; justify-content:space-between; width:100%;">
                    <span title="Percentile within this layer">Rank: <b>{pctile:.0f}%</b></span>
                    <span title="Average value for all heads in this layer">Avg: <b>{fmt.format(avg)}</b></span>
                </div>
                {ref_html}
            </div>
            '''
        elif key == "flow_change":
             # Global metric - just show Ref if available
             if ref_html:
                 context_html = f'<div class="metric-context" style="margin-top:4px;">{ref_html}</div>'
             else:
                 context_html = '<div class="metric-context"><span style="color:#9ca3af; font-style:italic;">Global Metric</span></div>'

        # Zones
        zone1_color, zone2_color, zone3_color = "#22c55e", "#f59e0b", "#ef4444"

        cards_html += f'''
            <div class="metric-card" onclick="showMetricModal('{modal_name}', 'Global', 'Avg')">
                <div class="metric-header-row">
                    <div class="metric-label">{label}</div>
                    <div class="metric-badge" style="background: {color}20; color: {color};">{label_text}</div>
                </div>
                
                <div class="metric-value-row">
                    <div class="metric-value">{value_str}</div>
                    {context_html}
                </div>

                <div class="metric-gauge-wrapper">
                    <span class="gauge-scale-label">0</span>
                    <div class="metric-gauge-fixed">
                        <div class="gauge-zone" style="width: {low_pct}%; background: {zone1_color}30;"></div>
                        <div class="gauge-zone" style="width: {high_pct - low_pct}%; background: {zone2_color}30;"></div>
                        <div class="gauge-zone" style="width: {100 - high_pct}%; background: {zone3_color}30;"></div>
                        {context_marker}
                        {baseline_marker}
                        <div class="gauge-marker" style="left: {gauge_pct}%; background: {color}; z-index:2;"></div>
                    </div>
                    <span class="gauge-scale-label">{scale_lbl}</span>
                </div>
            </div>
        '''
    cards_html += '</div>'
    return ui.HTML(cards_html)


def get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, max_depth, norm_mode="raw", att_matrix_override=None):
    """
    Generate JSON tree data for D3.js visualization.

    Args:
        res: Result tuple from model computation
        layer_idx: Layer index
        head_idx: Head index
        root_idx: Root token index for the tree
        top_k: Number of top children to show per node
        max_depth: Maximum depth of the tree
        norm_mode: Normalization mode ("raw", "col", "rollout")
        att_matrix_override: Optional pre-computed attention matrix (e.g. for global view)
    """
    tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
    if attentions is None or len(attentions) == 0:
        return None

    if att_matrix_override is not None:
        # Use provided matrix (Global Mode)
        raw_att = att_matrix_override
        # Q, K not valid for global average
        Q, K = None, None
        d_k = 1.0 
        is_causal = False # or check model?
        
        # Basic check for causal logic if needed
        layer_block = get_layer_block(encoder_model, 0)
        is_causal = not hasattr(layer_block, "attention")
        
    else:
        # Get raw attention matrix for selected layer and head
        raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

        # Get Q and K for computing dot products
        layer_block = get_layer_block(encoder_model, layer_idx)
        hs_in = hidden_states[layer_idx]

        Q, K, V = extract_qkv(layer_block, hs_in)

        if hasattr(layer_block, "attention"): # BERT
            num_heads = layer_block.attention.self.num_attention_heads
            is_causal = False
        else: # GPT-2
            num_heads = layer_block.attn.num_heads
            is_causal = True

        d_k = Q.shape[-1] // num_heads

    # Apply normalization based on mode
    if norm_mode == "col":
        # Column normalization
        col_sums = raw_att.sum(axis=0, keepdims=True)
        att = raw_att / (col_sums + 1e-8)
    elif norm_mode == "rollout":
        # Attention rollout
        att_per_layer = []
        for l_idx in range(layer_idx + 1):
            layer_att = attentions[l_idx][0].cpu().numpy()
            layer_att_avg = np.mean(layer_att, axis=0)
            att_per_layer.append(layer_att_avg)

        seq_len = att_per_layer[0].shape[0]
        rollout = np.eye(seq_len)

        for l_idx in range(layer_idx + 1):
            attention = att_per_layer[l_idx]
            attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
            row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
            attention_with_residual = attention_with_residual / (row_sums + 1e-8)
            rollout = np.matmul(attention_with_residual, rollout)

        att = rollout
    else:
        att = raw_att

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
    "get_architecture_diagram",
    "get_architecture_section",
    "get_paired_architecture_section",
    "get_gusnet_architecture_section",
]
def compute_attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    Compute attention rollout (Abnar & Zuidema, 2020).
    Assumes attentions is a list of (batch_size, num_heads, seq_len, seq_len) tensors.
    """
    result = None
    
    # Process layer by layer
    for layer_att in attentions:
        # layer_att shape: (batch_size, num_heads, seq_len, seq_len)
        # Taking batch 0 for visualization
        att = layer_att[0] 
        
        # Fuse heads
        if head_fusion == "mean":
            att = att.mean(dim=0) # (seq_len, seq_len)
        elif head_fusion == "max":
            att = att.max(dim=0)[0]
        elif head_fusion == "min":
            att = att.min(dim=0)[0]
            
        # To numpy
        att = att.cpu().numpy()
        
        # Add residual connection (identity matrix)
        # "To account for residual connections... we add the identity matrix to the attention matrix"
        eye = np.eye(att.shape[0])
        att = 0.5 * att + 0.5 * eye
        
        # Normalize rows to sum to 1
        att = att / att.sum(axis=1, keepdims=True)
        
        if result is None:
            result = att
        else:
            # Recursive multiplication: joint_attention(l) = attention(l) * joint_attention(l-1)
            result = np.matmul(att, result)
            
    return result
