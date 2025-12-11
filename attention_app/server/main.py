import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import re
import json

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from shiny import ui, render, reactive
from shinywidgets import render_plotly, output_widget, render_widget

from .logic import tokenize_with_segments, heavy_compute
from .renderers import *
from .bias_handlers import bias_server_handlers

# Additional imports needed for server function
from ..models import ModelManager
from ..utils import positional_encoding, array_to_base64_img, compute_influence_tree
from ..metrics import compute_all_attention_metrics
from ..head_specialization import compute_all_heads_specialization
from ..isa import compute_isa
from ..isa import get_sentence_token_attention

import traceback

def server(input, output, session):
    # Register bias analysis handlers
    bias_server_handlers(input, output, session)
    running = reactive.value(False)
    cached_result = reactive.value(None)
    isa_selected_pair = reactive.Value(None)
    
    # Synchronize text inputs between tabs
    @reactive.Effect
    @reactive.event(input.text_input)
    def sync_attention_to_bias():
        val = input.text_input()
        current_bias = input.bias_input_text()
        if val != current_bias:
            ui.update_text_area("bias_input_text", value=val)

    @reactive.Effect
    @reactive.event(input.bias_input_text)
    def sync_bias_to_attention():
        val = input.bias_input_text()
        current_attn = input.text_input()
        if val != current_attn:
            ui.update_text_area("text_input", value=val)
    
    @reactive.Effect
    @reactive.event(input.model_family)
    def update_model_choices():
        family = input.model_family()
        if family == "bert":
            choices = {
                "bert-base-uncased": "BERT Base (Uncased)",
                "bert-large-uncased": "BERT Large (Uncased)",
                "bert-base-multilingual-uncased": "BERT Multilingual",
            }
            selected = "bert-base-uncased"
        else: # gpt2
            choices = {
                "gpt2": "GPT-2 Small",
                "gpt2-medium": "GPT-2 Medium",
                "gpt2-large": "GPT-2 Large",
                "openai-community/gpt2-xl": "GPT-2 XL",
                "EleutherAI/gpt-neo-125M": "GPT-Neo 125M",
            }
            selected = "gpt2"
            
        ui.update_select("model_name", choices=choices, selected=selected)

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
        print("DEBUG: compute_all triggered")
        text = input.text_input().strip()
        print(f"DEBUG: Input text: '{text}'")
        if not text:
            print("DEBUG: No text input, returning")
            return
        
        running.set(True)
        await session.send_custom_message('start_loading', {})
        await asyncio.sleep(0.1)
        model_name = input.model_name()
        print(f"DEBUG: Model name: {model_name}")
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                print("DEBUG: Starting heavy_compute in executor")
                result = await loop.run_in_executor(pool, heavy_compute, text, model_name)
                print("DEBUG: heavy_compute returned")
            cached_result.set(result)
        except Exception as e:
            print(f"ERROR in compute_all: {e}")
            traceback.print_exc()
            cached_result.set(None)
            await session.send_custom_message('stop_loading', {})
        finally:
            running.set(False)



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



    def get_gpt2_dashboard_ui(res, input, output, session):
        tokens, _, _, _, _, _, _, encoder_model, *_ = res
        num_layers = len(encoder_model.h)
        num_heads = encoder_model.h[0].attn.num_heads
        
        # Get current selections
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
        
        try: radar_mode = input.radar_mode()
        except: radar_mode = "single"
        try: radar_layer = int(input.radar_layer())
        except: radar_layer = 0
        try: radar_head = int(input.radar_head())
        except: radar_head = 0
        try: tree_root_idx = int(input.tree_root_token())
        except: tree_root_idx = 0
        
        clean_tokens = [t.replace("##", "") if t.startswith("##") else t for t in tokens]

        return ui.div(
            {"class": "dashboard-stack gpt2-layout"},
            
            # Row 1: Embeddings
            ui.layout_columns(
                ui.div(
                    {"class": "card"}, 
                    ui.h4("Token Embeddings"), 
                    ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_embedding_table(res)
                ),
                ui.div(
                    {"class": "card"}, 
                    ui.h4("Positional Embeddings"), 
                    ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_posenc_table(res)
                ),
                ui.div(
                    {"class": "card"},
                    ui.h4("Sum & Layer Normalization"),
                    ui.p("Sum of embeddings + Pre-Norm", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_sum_layernorm_view(res, encoder_model)
                ),
                col_widths=[4, 4, 4]
            ),

            # Row 2: Transformer Block Details
            ui.layout_columns(
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-with-selectors"},
                        ui.h4("Q/K/V Projections"),
                        ui.div(
                            {"class": "selection-boxes-container"},
                            ui.div(
                                {"class": "selection-box"},
                                ui.div(
                                    {"class": "select-compact"},
                                    ui.input_select("qkv_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(qkv_layer))
                                )
                            )
                        )
                    ),
                    ui.p("Projects input to Query, Key, Value vectors.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_qkv_table(res, qkv_layer)
                ),
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-with-selectors"},
                        ui.h4("Scaled Dot-Product Attention"),
                        ui.div(
                            {"class": "selection-boxes-container"},
                            ui.tags.span("Focus:", style="font-size:10px; font-weight:600; color:#64748b; margin-right: 4px;"),
                            ui.div(
                                {"class": "selection-box"},
                                ui.div(
                                    {"class": "select-compact"},
                                    ui.input_select("scaled_attention_token", None, choices={str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}, selected=str(focus_token_idx))
                                )
                            )
                        )
                    ),
                    ui.p("Calculates attention scores between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_scaled_attention_view(res, att_layer, att_head, focus_token_idx)
                ),
                ui.div(
                    {"class": "card"}, 
                    ui.h4("Feed-Forward Network"), 
                    ui.p("Expansion -> Activation -> Projection", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_ffn_view(res, att_layer)
                ),
                col_widths=[4, 4, 4]
            ),

            # Row 3: Global Metrics & Attention Map
            ui.div({"class": "card"}, ui.h4("Global Attention Metrics"), get_metrics_display(res)),
            
            ui.layout_columns(
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-with-selectors"},
                        ui.h4("Multi-Head Attention"),
                        ui.div(
                            {"class": "selection-boxes-container"},
                            ui.div(
                                {"class": "selection-box"},
                                ui.div({"class": "select-compact"}, ui.input_select("att_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(att_layer)))
                            ),
                            ui.div(
                                {"class": "selection-box"},
                                ui.div({"class": "select-compact"}, ui.input_select("att_head", None, choices={str(i): f"Head {i}" for i in range(num_heads)}, selected=str(att_head)))
                            )
                        )
                    ),
                    ui.p("Visualizes attention weights. Lower triangular due to causal masking.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    output_widget("attention_map")
                ),
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-with-selectors"},
                        ui.h4("Attention Flow"),
                        ui.div(
                            {"class": "selection-boxes-container"},
                            ui.tags.span("Filter:", style="font-size:12px; font-weight:600; color:#64748b; margin-right: 4px;"),
                            ui.div(
                                {"class": "selection-box"},
                                ui.div(
                                    {"class": "select-compact"},
                                    ui.input_select("flow_token_select", None, choices={"all": "All tokens", **{str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}}, selected=flow_select)
                                )
                            )
                        )
                    ),
                    ui.div(
                        {"style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                        ui.output_ui("attention_flow")
                    )
                ),
                col_widths=[6, 6]
            ),

            # Row 4: Radar & Tree
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
                                color: white !important;
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
                            <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; padding: 12px; background: linear-gradient(135deg, #fff5f9 0%, #ffe5f3 100%); border-radius: 12px; border: 2px solid #ffcce5; color: #ffffff;">
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

            # Row 5: ISA
            ui.div(
                {"class": "card"},
                ui.h4("Inter-Sentence Attention (ISA)"),
                ui.layout_columns(
                    ui.div(
                        {"style": "height: 500px; max-height: 60vh; width: 100%; display: flex; justify-content: center; align-items: center;"},
                        ui.output_ui("isa_scatter")
                    ),
                    ui.div(
                        ui.output_ui("isa_detail_info"),
                        ui.div(ui.output_ui("isa_token_view")),
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

            # Row 6: Unembedding & Predictions
            ui.layout_columns(
                ui.div({"class": "card"}, ui.h4("Hidden States"), ui.p("Final vector representation before projection.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_layer_output_view(res, num_layers - 1)),
                ui.div({"class": "card"}, ui.h4("Next Token Predictions"), ui.p("Probabilities for the next token (Softmax output).", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_output_probabilities(res, use_mlm_val, text_val)),
                col_widths=[6, 6]
            ),
            
            # Autoregressive Loop Note
            ui.div(
                {"class": "card"},
                ui.h4("Autoregressive Loop"),
                ui.p("In generation, the predicted token is added to the input, and the entire process repeats.", style="font-size:13px; color:#4b5563;")
            )
        )

    @output
    @render.ui
    def visualization_options_container():
        # Only show Visualization Options for BERT (encoder-only models)
        # For GPT-2, we hide the entire section
        try:
            model_family = input.model_family()
        except:
            model_family = "bert"
            
        if model_family == "gpt2":
            return None
            
        return ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("Visualization Options", class_="sidebar-label"),
            ui.input_switch("use_mlm", ui.span("Show MLM Predictions", style="font-size: 14px; color: #64748b; font-weight: 500;"), value=False)
        )

    @output
    @render.ui
    def render_embedding_table():
        res = cached_result.get()
        if not res: return None
        return get_embedding_table(res)

    @output
    @render.ui
    def render_segment_table():
        res = cached_result.get()
        if not res: return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Segment Embeddings"), 
            ui.p("Segment ID (Sentence A/B)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_segment_embedding_view(res)
        )

    @output
    @render.ui
    def render_posenc_table():
        res = cached_result.get()
        if not res: return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Positional Embeddings"), 
            ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_posenc_table(res)
        )

    @output
    @render.ui
    def render_sum_layernorm():
        res = cached_result.get()
        if not res: return None
        _, _, _, _, _, _, _, encoder_model, *_ = res
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Sum & Layer Normalization"),
            ui.p("Sum of embeddings + Pre-Norm", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_sum_layernorm_view(res, encoder_model)
        )

    @output
    @render.ui
    def render_qkv_table():
        res = cached_result.get()
        if not res: return None
        try: layer_idx = int(input.qkv_layer())
        except: layer_idx = 0
        
        # Get num_layers for selector
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        if is_gpt2: num_layers = len(encoder_model.h)
        else: num_layers = len(encoder_model.encoder.layer)

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-with-selectors"},
                ui.h4("Q/K/V Projections"),
                ui.div(
                    {"class": "selection-boxes-container"},
                    ui.div(
                        {"class": "selection-box"},
                        ui.div(
                            {"class": "select-compact"},
                            ui.input_select("qkv_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(layer_idx))
                        )
                    )
                )
            ),
            ui.p("Projects input to Query, Key, Value vectors.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_qkv_table(res, layer_idx)
        )

    @output
    @render.ui
    def render_scaled_attention():
        res = cached_result.get()
        if not res: return None
        try: selected_token = input.scaled_attention_token()
        except: selected_token = "0"
        
        try: layer_idx = int(input.att_layer())
        except: layer_idx = 0
        try: head_idx = int(input.att_head())
        except: head_idx = 0
        
        focus_idx = int(selected_token) if selected_token else 0
        
        # Get tokens for selector
        clean_tokens = tokens_data()
        choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-with-selectors"},
                ui.h4("Scaled Dot-Product Attention"),
                ui.div(
                    {"class": "selection-boxes-container"},
                    ui.tags.span("Focus:", style="font-size:10px; font-weight:600; color:#64748b; margin-right: 4px;"),
                    ui.div(
                        {"class": "selection-box"},
                        ui.div(
                            {"class": "select-compact"},
                            ui.input_select("scaled_attention_token", None, choices=choices, selected=selected_token)
                        )
                    )
                )
            ),
            ui.p("Calculates attention scores between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_idx)
        )

    @output
    @render.ui
    def render_ffn():
        res = cached_result.get()
        if not res: return None
        try: layer = int(input.att_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Feed-Forward Network"), 
            ui.p("Expansion -> Activation -> Projection", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_ffn_view(res, layer)
        )

    @output
    @render.ui
    def render_add_norm():
        res = cached_result.get()
        if not res: return None
        try: layer = int(input.att_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Add & Norm"),
            ui.p("Residual Connection + Layer Normalization", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_add_norm_view(res, layer)
        )

    @output
    @render.ui
    def render_add_norm_post_ffn():
        res = cached_result.get()
        if not res: return None
        try: layer = int(input.att_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Add & Norm (Post-FFN)"),
            ui.p("Residual Connection + Layer Normalization", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_add_norm_post_ffn_view(res, layer)
        )

    @output
    @render.ui
    def render_layer_output():
        res = cached_result.get()
        if not res: return None
        _, _, _, _, _, _, _, encoder_model, *_ = res
        if hasattr(encoder_model, "encoder"): # BERT
            num_layers = len(encoder_model.encoder.layer)
        else: # GPT-2
            num_layers = len(encoder_model.h)
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Hidden States"),
            ui.p("Final vector representation before projection.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_layer_output_view(res, num_layers - 1)
        )

    @output
    @render.ui
    def render_mlm_predictions():
        res = cached_result.get()
        if not res: return None
        
        # Determine if we should show predictions
        # GPT-2: Always show
        # BERT: Show only if switch is on
        try:
            model_family = input.model_family()
        except:
            model_family = "bert"
            
        if model_family == "gpt2":
            use_mlm = True
        else:
            try: use_mlm = input.use_mlm()
            except: use_mlm = False
            
        try: text = input.text_input()
        except: text = ""
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Next Token Predictions"),
            ui.p("Probabilities for the next token (Softmax output).", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_output_probabilities(res, use_mlm, text)
        )

    @output
    @render.ui
    def render_radar_view():
        res = cached_result.get()
        if not res: return None
        try: layer_idx = int(input.radar_layer())
        except: layer_idx = 0
        try: head_idx = int(input.radar_head())
        except: head_idx = 0
        try: mode = input.radar_mode()
        except: mode = "single"
        
        # Get num_layers/heads for selector
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        if is_gpt2: 
            num_layers = len(encoder_model.h)
            num_heads = encoder_model.h[0].attn.num_heads
        else: 
            num_layers = len(encoder_model.encoder.layer)
            num_heads = encoder_model.encoder.layer[0].attention.self.num_attention_heads

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            ui.div(
                {"class": "header-controls-stacked"},
                ui.div(
                    {"class": "header-row-top"},
                    ui.h4("Attention Head Specialization"),
                    ui.div(
                        {"class": "header-right"},
                        ui.div({"class": "select-compact", "id": "radar_head_selector"}, ui.input_select("radar_head", None, choices={str(i): f"Head {i}" for i in range(num_heads)}, selected=str(head_idx))),
                        ui.div({"class": "select-compact"}, ui.input_select("radar_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(layer_idx))),
                    )
                ),
                ui.div(
                    {"class": "header-row-bottom", "style": "display: flex; flex-direction: column; gap: 8px; margin-top: 4px;"},
                    ui.p("Analyzes the linguistic roles (syntax, semantics, etc.) performed by each attention head.", style="font-size:11px; color:#6b7280; margin: 0; width: 100%;"),
                    ui.div(
                        {"style": "display: flex; align-items: center; gap: 12px; align-self: flex-end;"},
                        ui.span("ATTENTION MODE:", style="font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 0.5px;"),
                        ui.input_radio_buttons("radar_mode", None, {"single": "Single Head", "all": "All Heads"}, inline=True, selected=mode)
                    )
                )
            ),
            head_specialization_radar(res, layer_idx, head_idx, mode),
            ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; margin-top: auto; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center;">
                    <strong style="color: #ff5ca9;">Attention Specialization Dimensions</strong> — click any to see detailed explanation:<br>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; justify-content: center;">
                        <span class="metric-tag" onclick="showMetricModal('Syntax', 0, 0)">Syntax</span>
                        <span class="metric-tag" onclick="showMetricModal('Semantics', 0, 0)">Semantics</span>
                        <span class="metric-tag" onclick="showMetricModal('CLS Focus', 0, 0)">CLS Focus</span>
                        <span class="metric-tag" onclick="showMetricModal('Punctuation', 0, 0)">Punctuation</span>
                        <span class="metric-tag" onclick="showMetricModal('Entities', 0, 0)">Entities</span>
                        <span class="metric-tag" onclick="showMetricModal('Long-range', 0, 0)">Long-range</span>
                        <span class="metric-tag" onclick="showMetricModal('Self-attention', 0, 0)">Self-attention</span>
                    </div>
                </div>
            """)
        )

    @output
    @render.ui
    def render_tree_view():
        res = cached_result.get()
        if not res: return None
        try: root_idx = int(input.tree_root_token())
        except: root_idx = 0
        try: layer_idx = int(input.radar_layer())
        except: layer_idx = 0
        try: head_idx = int(input.radar_head())
        except: head_idx = 0
        
        # Get tokens for selector
        clean_tokens = tokens_data()
        choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
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
                                choices=choices,
                                selected=str(root_idx)
                            )
                        )
                    )
                )
            ),
            ui.p("Visualizes the hierarchical influence of tokens on the selected root token.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_influence_tree_ui(res, root_idx, layer_idx, head_idx)
        )

    @output
    @render.ui
    def render_global_metrics():
        res = cached_result.get()
        if not res: return None
        return ui.div(
            {"class": "card"}, 
            ui.h4("Global Attention Metrics"), 
            get_metrics_display(res),
            ui.tags.script("$('#loading_spinner').hide(); $('#generate_all').prop('disabled', false).css('opacity', '1'); $('#dashboard-container').removeClass('content-hidden').addClass('content-visible');")
        )

    def dashboard_layout_helper(is_gpt2, num_layers, num_heads, clean_tokens):
        # Helper to generate choices dict
        def get_choices(items):
            return {str(i): f"{i}: {t}" for i, t in enumerate(items)}

        if is_gpt2:
            # GPT-2 Layout
            return ui.div(
                {"id": "dashboard-container", "class": "dashboard-stack gpt2-layout content-hidden"},
                
                # Row 1: Embeddings
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Sentence Preview", "Token Embeddings", "vertical", style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), ui.output_ui("render_embedding_table", style="height: 100%;"))
                    ),
                    arrow("Token Embeddings", "Positional Embeddings", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_posenc_table", style="height: 100%;"),
                        arrow("Sum & Layer Normalization", "Q/K/V Projections", "vertical", style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Positional Embeddings", "Sum & Layer Normalization", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_sum_layernorm", style="height: 100%;")),
                ),

                # Row 2: Transformer Block Details (Attention)
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_qkv_table", style="height: 100%;")
                    ),
                    arrow("Q/K/V Projections", "Scaled Dot-Product Attention", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_scaled_attention", style="height: 100%;")
                    ),
                ),
                
                # Row 3: Global Metrics & Attention Map
                ui.output_ui("render_global_metrics"),
                
                ui.layout_columns(
                    ui.output_ui("attention_map"),
                    ui.output_ui("attention_flow"),
                    col_widths=[6, 6]
                ),

                # Row 4: Radar & Tree
                ui.layout_columns(
                    ui.output_ui("render_radar_view"),
                    ui.output_ui("render_tree_view"),
                    col_widths=[5, 7]
                ),

                # Row 5: ISA
                ui.output_ui("isa_scatter"),

                # Row 6: Transformer Block Details (FFN)
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Inter-Sentence Attention", "Add & Norm", "vertical", style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.output_ui("render_add_norm", style="height: 100%;")
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_ffn", style="height: 100%;"),
                        arrow("Add & Norm (post-FFN)", "Hidden States", "vertical", style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_add_norm_post_ffn", style="height: 100%;")
                    ),
                ),

                # Row 7: Unembedding & Predictions
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div({"class": "flex-card"}, ui.output_ui("render_layer_output", style="height: 100%;")),
                    arrow("Hidden States", "Next Token Predictions", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_mlm_predictions", style="height: 100%;")),
                )
            )



        # Construct UI (BERT)
        return ui.div(
            {"id": "dashboard-container", "class": "dashboard-stack content-hidden"}, # Initially hidden
            ui.div(
                {"class": "dashboard-stack"},
                # Row 1: Initial Embeddings 
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Sentence Preview", "Token Embeddings", "vertical", style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), ui.output_ui("render_embedding_table"))
                    ),
                    arrow("Token Embeddings", "Segment Embeddings", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_segment_table", style="height: 100%;")),
                    arrow("Segment Embeddings", "Positional Embeddings", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_posenc_table", style="height: 100%;")),
                ),
                
                # Row 2: Processing
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_sum_layernorm")
                    ),
                    arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Positional Embeddings", "Sum & Layer Normalization", "vertical", style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;"),
                        ui.output_ui("render_qkv_table")
                    ),
                    arrow("Q/K/V Projections", "Scaled Dot-Product Attention", "horizontal"),
                    ui.div(
                        {"class": "flex-card"},
                        ui.output_ui("render_scaled_attention")
                    ),
                ),
                
                # Row 3: Global Metrics
                ui.output_ui("render_global_metrics"),
                
                # Row 4: Attention Visualizations 
                ui.layout_columns(
                    ui.output_ui("attention_map"),
                    ui.output_ui("attention_flow"),
                    col_widths=[6, 6]
                ),
                
                
                # Row 5: Specialization Analysis
                ui.layout_columns(
                    ui.output_ui("render_radar_view"),
                    ui.output_ui("render_tree_view"),
                    col_widths=[5, 7]
                ),
                
                
                # Row 6: Inter-Sentence Attention (full width)
                ui.output_ui("isa_scatter"),
                
                
                # Row 7: Residual Connections
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Inter-Sentence Attention", "Add & Norm", "vertical", style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.output_ui("render_add_norm")
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_ffn"),
                        arrow("Add & Norm (post-FFN)", "Hidden States", "vertical", style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_add_norm_post_ffn")),
                ),
                
                # Row 8: Final Outputs 
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui("render_layer_output")
                    ),
                    arrow("Hidden States", "Token Output Predictions", "horizontal"),
                    ui.div({"class": "flex-card"}, ui.output_ui("render_mlm_predictions")),
                ),
            ),
        )

    # Deduplicating reactive value for layout configuration
    # This ensures dashboard_content only re-renders when the model structure actually changes
    current_layout_config = reactive.Value(None)

    @reactive.Effect
    def _update_layout_config():
        res = cached_result.get()
        if not res: return
        
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        
        if is_gpt2:
             num_layers = len(encoder_model.h)
             num_heads = encoder_model.h[0].attn.num_heads
        else:
             num_layers = len(encoder_model.encoder.layer)
             num_heads = encoder_model.encoder.layer[0].attention.self.num_attention_heads
        
        new_config = (is_gpt2, num_layers, num_heads)
        
        # Only update if changed
        if current_layout_config.get() != new_config:
            print(f"DEBUG: Layout config changed to {new_config}")
            current_layout_config.set(new_config)

    @reactive.calc
    def tokens_data():
        res = cached_result.get()
        if not res: return []
        tokens = res[0]
        return [t.replace("##", "") if t.startswith("##") else t for t in tokens]

    @reactive.effect
    def update_selectors():
        clean_tokens = tokens_data()
        if not clean_tokens: return
        
        choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}
        
        # Update all token-based selectors
        ui.update_select("scaled_attention_token", choices=choices)
        ui.update_select("flow_token_select", choices={"all": "All tokens", **choices})
        ui.update_select("tree_root_token", choices=choices)

    @output
    @render.ui
    def dashboard_content():
        config = current_layout_config.get()
        if not config:
            return ui.HTML("<script>$('#loading_spinner').hide(); $('#generate_all').prop('disabled', false).css('opacity', '1');</script>")
        
        is_gpt2, num_layers, num_heads = config
        
        print("DEBUG: Rendering dashboard_content (Layout Re-build)")
        # Pass empty tokens list to avoid re-rendering layout when tokens change
        # The selectors will be populated by update_selectors
        return dashboard_layout_helper(is_gpt2, num_layers, num_heads, [])

    @output(id="isa_scatter")
    @render.ui
    def isa_scatter_renderer():
        res = cached_result.get()
        if not res:
            return None

        isa_data = res[-1]

        if not isa_data or "sentence_attention_matrix" not in isa_data or "sentence_texts" not in isa_data:
             return ui.div(
                {"class": "card"},
                ui.h4("Inter-Sentence Attention (ISA)"),
                ui.HTML("<div style='color:#9ca3af;font-size:12px;padding:20px;'>ISA data not available. Ensure input has multiple sentences.</div>")
             )
             
        matrix = isa_data["sentence_attention_matrix"]
        sentences = isa_data["sentence_texts"]
        n = len(sentences)

        x, y = np.meshgrid(np.arange(n), np.arange(n))
        x_flat = x.flatten().tolist()
        y_flat = y.flatten().tolist()
        scores = np.nan_to_num(matrix.flatten(), nan=0.0).tolist()

        hover_texts = [
            f"Target ← {sentences[int(r)][:60]}...<br>Source → {sentences[int(c)][:60]}...<br>ISA = {s:.4f}"
            for r, c, s in zip(y_flat, x_flat, scores)
        ]

        customdata = list(zip(y_flat, x_flat))
        
        sizes = np.clip(np.array(scores) * 40 + 12, 12, 80).tolist()

        # Custom colorscale matching the app's theme (Pink/Blue/Purple)
        # Using a custom sequential scale from light blue/slate to vibrant pink/purple
        custom_colorscale = [
            [0.0, '#f1f5f9'],   # Slate-100 (Lightest)
            [0.2, '#cbd5e1'],   # Slate-300
            [0.4, '#94a3b8'],   # Slate-400
            [0.6, '#60a5fa'],   # Blue-400
            [0.8, '#818cf8'],   # Indigo-400
            [1.0, '#ec4899']    # Pink-500 (Strongest)
        ]

        fig = go.Figure(data=go.Scatter(
            x=x_flat, y=y_flat,
            mode="markers",
            marker=dict(
                size=sizes,
                color=scores,
                colorscale=custom_colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="ISA Score",
                        side="right",
                        font=dict(color="#64748b", size=11)
                    ),
                    tickfont=dict(color="#64748b", size=10)
                ),
                line=dict(width=1, color="rgba(255,255,255,0.5)") # Subtle white border
            ),
            text=hover_texts,
            hoverinfo="text",
            customdata=customdata
        ))

        labels = [s[:30] + "..." if len(s) > 30 else s for s in sentences]

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Source (Sentence Y)",
                    font=dict(color="#475569", size=12, family="Inter, system-ui, sans-serif")
                ),
                tickmode="array", 
                tickvals=list(range(n)), 
                ticktext=labels,
                showgrid=False,
                zeroline=False,
                tickfont=dict(color="#64748b", size=11),
            ),
            yaxis=dict(
                title=dict(
                    text="Target (Sentence X)",
                    font=dict(color="#475569", size=12, family="Inter, system-ui, sans-serif")
                ),
                tickmode="array", 
                tickvals=list(range(n)), 
                ticktext=labels, 
                autorange="reversed",
                showgrid=False,
                zeroline=False,
                tickfont=dict(color="#64748b", size=11),
            ),
            height=500,
            width=500,
            autosize=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            clickmode="event+select",
            margin=dict(l=40, r=40, t=20, b=20),
            font=dict(family="Inter, system-ui, sans-serif")
        )

        # Generate HTML with unique ID
        plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="isa_scatter_plot", config={'displayModeBar': False})
        
        # Custom JS to handle clicks and send to Shiny
        js = """
        <script>
        (function() {
            console.log("DEBUG: Initializing ISA Plot Script");
            function initPlot() {
                var plot = document.getElementById('isa_scatter_plot');
                if (plot) {
                    console.log("DEBUG: ISA Plot found, attaching listener");
                    plot.on('plotly_click', function(data){
                        var pt = data.points[0];
                        var x = pt.x; // source index
                        var y = pt.y; // target index
                        // Send to Shiny input 'isa_scatter_click'
                        Shiny.setInputValue('isa_scatter_click', {x: x, y: y}, {priority: 'event'});
                    });
                } else {
                    console.log("DEBUG: ISA Plot not found yet, retrying...");
                    setTimeout(initPlot, 100);
                }
            }
            initPlot();
        })();
        </script>
        """
        
        # Determine model type for explanation
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        model_type = "GPT-2" if is_gpt2 else "BERT"

        return ui.div(
            {"class": "card"},
            ui.div(
                {"style": "display: flex; align-items: center; gap: 8px; margin-bottom: 8px;"},
                ui.h4(
                    "Inter-Sentence Attention (ISA)", 
                    style="margin: 0; cursor: pointer; border-bottom: 1px dashed #cbd5e1; display: inline-block;",
                    onclick=f"showISACalcExplanation('{model_type}')",
                    title="Click to see how this is calculated"
                ),
            ),
            ui.p("Visualizes the relationship between two sentences, focusing on how the tokens in Sentence X attend to the tokens in Sentence Y. The ISA score quantifies this relationship, with higher values indicating a stronger connection between the tokens in Sentence X and Sentence Y.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            ui.layout_columns(
                ui.div(
                    {"style": "height: 500px; width: 100%; display: flex; justify-content: center; align-items: center;"},
                    ui.HTML(plot_html + js)
                ),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; height: 100%;"},
                    ui.div(ui.output_ui("isa_detail_info"), style="flex: 0 0 auto; margin-bottom: 10px;"),
                    ui.div(ui.output_ui("isa_token_view"), style="flex: 1; display: flex; flex-direction: column;"),
                ),
                col_widths=[6, 6],
            ),
        )


    @output
    @render.ui
    def isa_token_view():
        pair = isa_selected_pair()
        res = cached_result.get()

        if res is None or pair is None:
            return ui.div(
                ui.p("Select a point on the scatter plot to view token-to-token attention.", 
                     style="color: #94a3b8; font-size: 13px; font-style: italic;"),
                style="height: 350px; display: flex; align-items: center; justify-content: center; border: 1px dashed #e2e8f0; border-radius: 8px; background: #f8fafc;"
            )

        target_idx, source_idx = pair
        tokens, _, _, attentions, *_ = res
        isa_data = res[-1]
        boundaries = isa_data["sentence_boundaries_ids"]

        sub_att, tokens_combined, src_start = get_sentence_token_attention(
            attentions, tokens, target_idx, source_idx, boundaries
        )

        toks_target = tokens_combined[:src_start]
        toks_source = tokens_combined[src_start:]

        # Custom colorscale for heatmap (Light Blue -> Deep Blue/Purple)
        heatmap_colorscale = [
            [0.0, '#f8fafc'],   # Slate-50
            [0.2, '#e0f2fe'],   # Sky-100
            [0.4, '#bae6fd'],   # Sky-200
            [0.6, '#60a5fa'],   # Blue-400
            [0.8, '#3b82f6'],   # Blue-500
            [1.0, '#4f46e5']    # Indigo-600
        ]

        fig = go.Figure(data=go.Heatmap(
            z=sub_att,
            x=toks_source,
            y=toks_target,
            colorscale=heatmap_colorscale,
            colorbar=dict(
                title=dict(
                    text="Attention",
                    side="right",
                    font=dict(color="#64748b", size=11)
                ),
                tickfont=dict(color="#64748b", size=10)
            ),
            hovertemplate="Target: %{y}<br>Source: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(
                text=f"Token-to-Token — S{target_idx} ← S{source_idx}",
                font=dict(size=14, color="#1e293b", family="Inter, system-ui, sans-serif")
            ),
            xaxis=dict(
                title=dict(
                    text="Source tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9"
            ),
            yaxis=dict(
                title=dict(
                    text="Target tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9",
                autorange="reversed" 
            ),
            height=350,
            autosize=True,
            margin=dict(l=60, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif")
        )
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="isa_token_view_plot", config={'displayModeBar': False}))


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
    @reactive.event(input.isa_scatter_click)
    def _handle_isa_click():
        click = input.isa_scatter_click()
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



    @output
    @render.ui
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
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        layer_block = get_layer_block(encoder_model, layer_idx)
        
        if hasattr(layer_block, "attention"): # BERT
            num_heads = layer_block.attention.self.num_attention_heads
            num_layers = len(encoder_model.encoder.layer)
        else: # GPT-2
            num_heads = layer_block.attn.num_heads
            num_layers = len(encoder_model.h)
            
        hs_in = hidden_states[layer_idx]
        
        Q, K, V = extract_qkv(layer_block, hs_in)
        
        L = len(tokens)
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
        # Custom colorscale for attention map (White -> Blue -> Dark Blue)
        att_colorscale = [
            [0.0, '#ffffff'],
            [0.1, '#f0f9ff'],
            [0.3, '#bae6fd'],
            [0.6, '#3b82f6'],
            [1.0, '#1e3a8a']
        ]

        fig = px.imshow(att, x=tokens, y=tokens, color_continuous_scale=att_colorscale, aspect="auto")
        fig.update_traces(customdata=custom, hovertemplate=hover)
        fig.update_layout(
            xaxis_title="Key (attending to)", 
            yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40), 
            coloraxis_colorbar=dict(
                title=dict(
                    text="Attention",
                    font=dict(color="#64748b", size=11)
                ),
                tickfont=dict(color="#64748b", size=10)
            ),
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#64748b", family="Inter, system-ui, sans-serif"),
            xaxis=dict(
                tickfont=dict(size=10), 
                title=dict(font=dict(size=11))
            ),
            yaxis=dict(
                tickfont=dict(size=10), 
                title=dict(font=dict(size=11))
            )
        )
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-with-selectors"},
                ui.h4("Multi-Head Attention"),
                ui.div(
                    {"class": "selection-boxes-container"},
                    ui.div(
                        {"class": "selection-box"},
                        ui.div({"class": "select-compact"}, ui.input_select("att_layer", None, choices={str(i): f"Layer {i}" for i in range(num_layers)}, selected=str(layer_idx)))
                    ),
                    ui.div(
                        {"class": "selection-box"},
                        ui.div({"class": "select-compact"}, ui.input_select("att_head", None, choices={str(i): f"Head {i}" for i in range(num_heads)}, selected=str(head_idx)))
                    )
                )
            ),
            ui.p("Visualizes attention weights.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_map_plot"))
        )

    @output
    @render.ui
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

        clean_tokens = tokens_data()
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        n_tokens = len(tokens)
        color_palette = ['#ff5ca9', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#0ea5e9']

        fig = go.Figure()

        # Calculate dynamic width based on token count to ensure enough horizontal space
        # Increased to 50 pixels per token for proper spacing even with long tokens
        min_pixels_per_token = 50
        calculated_width = max(1000, n_tokens * min_pixels_per_token)

        # Adjust block spacing to prevent horizontal overlap
        block_width = 0.95 / n_tokens  # Maximum spacing

        for i, tok in enumerate(tokens):
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2
            show_focus = focus_idx is not None
            is_selected = focus_idx == i if show_focus else True

            # Dynamically adjust font size for many tokens
            if n_tokens > 30:
                font_size = 9 if is_selected else 8
            elif n_tokens > 20:
                font_size = 11 if is_selected else 10
            else:
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

        fig.update_layout(
            title=title_text,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.05, 1.05]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.25, 1.25]),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#111827'),
            height=500,
            width=calculated_width,  # Dynamic width for horizontal spacing
            margin=dict(l=20, r=20, t=60, b=40),
            clickmode='event+select',
            hovermode='closest',
            dragmode=False,
            autosize=False  # Disable autosize to respect fixed width
        )

        # Configure the figure to prevent responsive resizing
        fig.update_layout(
            modebar=dict(orientation='v')
        )

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-with-selectors"},
                ui.h4("Attention Flow"),
                ui.div(
                    {"class": "selection-boxes-container"},
                    ui.tags.span("Filter:", style="font-size:12px; font-weight:600; color:#64748b; margin-right: 4px;"),
                    ui.div(
                        {"class": "selection-box"},
                        ui.div(
                            {"class": "select-compact"},
                            ui.input_select("flow_token_select", None, choices={"all": "All tokens", **{str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}}, selected=selected)
                        )
                    )
                )
            ),
            ui.p("Traces how information flows from one token to another through attention layers.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            ui.div(
                {"style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot"))
            )
        )

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
                    tickfont=dict(size=10, color="#94a3b8"),
                    gridcolor='#e2e8f0',
                    linecolor='#e2e8f0'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color='#475569', family="Inter, system-ui, sans-serif"),
                    gridcolor='#e2e8f0',
                    linecolor='#e2e8f0'
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=(mode == "all"),
            legend=dict(font=dict(size=10, color="#64748b")),
            title=dict(
                text=title_text,
                font=dict(size=14, color="#1e293b", family="Inter, system-ui, sans-serif"),
                y=0.95
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif"),
            height=300,
            width=350,
            autosize=False
        )
        
        return ui.HTML(ui.div(
            {"style": "display: flex; justify-content: center; width: 100%;"},
            ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="radar_plot", config={'displayModeBar': False}))
        ))


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
        depth = 3     # Maximum depth 
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
            tree_json = json.dumps(tree_data)
        except Exception as e:
            return ui.HTML(f"<p style='font-size:11px;color:#ef4444;'>Error generating tree: {str(e)}</p>")
        
        # Explanation text - minimalist and after tree
        explanation = """
        <div class="tree-explanation" style="background: white; margin-top: 10px; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
            <p style="margin: 0; font-size: 11px; color: #64748b; line-height: 1.6;">
                <strong>Attention Dependency Tree:</strong> Visualizes how the root token attends to other tokens (Depth 1), and how those tokens attend to others (Depth 2).
                <span style="color: #94a3b8;">Click nodes to collapse/expand. Thicker edges = stronger influence.</span>
            </p>
        </div>
        """
        
        html = f"""
    <div class="influence-tree-wrapper" style="height: 100%; display: flex; flex-direction: column; position: relative;">
        <div id="tree-viz-container" class="tree-viz-container" style="height: 600px; width: 100%; overflow-x: auto; overflow-y: hidden; text-align: center; display: block;"></div>
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
        """
        
        return ui.HTML(html)

__all__ = ["server"]
