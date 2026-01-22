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
from ..ui.components import viz_header

# Additional imports needed for server function
from ..models import ModelManager
from ..utils import positional_encoding, array_to_base64_img, compute_influence_tree
from ..metrics import compute_all_attention_metrics
from ..head_specialization import compute_all_heads_specialization
from ..isa import compute_isa
from ..isa import get_sentence_token_attention

import traceback

# Helper function to generate loading placeholder cards for Model B sections
def loading_placeholder(title, description="Loading...", card_class="card"):
    """Generate a loading placeholder card with consistent styling."""
    return ui.div(
        {"class": card_class, "style": "height: 100%;"},
        ui.h4(title),
        ui.p(description, style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
        ui.div(
            {"style": "padding: 40px; text-align: center;"},
            ui.HTML('<div class="loading-container"><div class="spinner"></div> <span style="color:#9ca3af;">Loading...</span></div>')
        )
    )

def server(input, output, session):

    # Register bias analysis handlers
    bias_server_handlers(input, output, session)
    running = reactive.value(False)
    cached_result = reactive.value(None)
    cached_result_B = reactive.value(None) # For comparison
    isa_selected_pair = reactive.Value(None)
    isa_selected_pair = reactive.Value(None)
    isa_selected_pair_B = reactive.Value(None) # For comparison
    
    # --- History Logic ---
    input_history = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.restored_history)
    def restore_history():
        if input.restored_history():
            # Dedup restored data immediately with normalization
            raw = input.restored_history()
            unique = []
            for item in raw:
                clean_item = item.strip()
                if clean_item and clean_item not in unique:
                    unique.append(clean_item)
            input_history.set(unique)

    @reactive.Effect
    @reactive.event(input.generate_all)
    def update_history():
        text = input.text_input()
        if not text:
            return
        clean_text = text.strip()
        if not clean_text:
            return
        
        # Remove ALL instances of this text checking normalized version
        hist = [h for h in input_history() if h.strip() != clean_text]
        hist.insert(0, clean_text)
        hist = hist[:20]
        input_history.set(hist)

    @reactive.Effect
    async def sync_history_storage():
        await session.send_custom_message("update_history", input_history())

    @output
    @render.ui
    def history_list():
        hist = input_history()
        if not hist:
            return ui.div("No history yet.", style="padding:10px; color:#94a3b8; font-style:italic;")
        
        items = []
        for text in hist:
            display_text = (text[:60] + "...") if len(text) > 60 else text
            safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '&quot;').replace('\n', ' ')
            items.append(
                ui.div(
                    display_text, 
                    class_="history-item", 
                    onclick=f"selectHistoryItem('{safe_text}')"
                )
            )
        return ui.div(*items)

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
            }
            selected = "gpt2"
            
        ui.update_select("model_name", choices=choices, selected=selected)

    # Prompt Wizard Step State: 'A' or 'B' or 'DONE'
    prompt_entry_step = reactive.Value("A")

    # Mutual Exclusivity for Compare Modes
    @reactive.Effect
    @reactive.event(input.compare_mode)
    def handle_compare_mode():
        if input.compare_mode():
            ui.update_switch("compare_prompts_mode", value=False)

    @reactive.Effect
    @reactive.event(input.compare_prompts_mode)
    def handle_compare_prompts_mode():
        if input.compare_prompts_mode():
            ui.update_switch("compare_mode", value=False)
            # Reset wizard
            prompt_entry_step.set("A")
            # Force UI to A (handled by JS listener normally, but ensuring state)

    # Dynamic Button Label Updater
    @reactive.Effect
    @reactive.event(input.compare_prompts_mode, prompt_entry_step) # Explicitly depend on both
    async def update_generate_button_label():
        mode = input.compare_prompts_mode()
        step = prompt_entry_step.get()
        
        label = "Generate All"
        print(f"DEBUG: update_button_label triggered. mode={mode}, step={step}")
        
        if mode and step == "A":
            label = "Prompt B ➜"
            
        print(f"DEBUG: sending update_button_label: {label}")
        await session.send_custom_message("update_button_label", {"label": label})


    @reactive.Effect
    @reactive.event(input.model_family_B)
    def update_model_choices_B():
        family = input.model_family_B()
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
            }
            selected = "gpt2"
            
        ui.update_select("model_name_B", choices=choices, selected=selected)






    
    # MLM Trigger Logic
    show_mlm_A = reactive.Value(False)
    show_mlm_B = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.trigger_mlm_A)
    def trigger_mlm_A():
        show_mlm_A.set(True)

    @reactive.Effect
    @reactive.event(input.trigger_mlm_B)
    def trigger_mlm_B():
        show_mlm_B.set(True)

    # Input History Logic
    input_history = reactive.Value([
        "All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.",
        "The quick brown fox jumps over the lazy dog.",
        "The doctor said he would be back soon.",
        "The nurse said she was tired."
    ])

    @reactive.effect
    @reactive.event(input.generate_all)
    async def compute_all():
        # Reset MLM triggers on new computation
        show_mlm_A.set(False)
        show_mlm_B.set(False)

        # Wizard Logic
        try: mode = input.compare_prompts_mode()
        except: mode = False
        
        if mode:
            step = prompt_entry_step.get()
            if step == "A":
                # Transition to B, DO NOT COMPUTE
                prompt_entry_step.set("B")
                await session.send_custom_message("switch_prompt_tab", "B")
                return

            # If we are here, we are in Step B (or DONE), so we compute
            prompt_entry_step.set("DONE")

        print("DEBUG: compute_all triggered")
        text = input.text_input().strip()
        print(f"DEBUG: Input text: '{text}'")
        if not text:
            print("DEBUG: No text input, returning")
            return

        # Update History
        current_history = input_history.get()
        # Only add if unique and non-empty
        if text and (not current_history or text != current_history[0]):
            if text in current_history:
                current_history.remove(text) # Move to top
            updated_history = [text] + current_history
            input_history.set(updated_history[:20]) # Limit to 20 items

        # Reset MLM states on new generation
        show_mlm_A.set(False)
        show_mlm_B.set(False)

        running.set(True)
        await session.send_custom_message('start_loading', {})
        await asyncio.sleep(0.1)

        model_name = input.model_name()
        print(f"DEBUG: Model name A: {model_name}")
        
        # Check compare modes
        try: compare_models = input.compare_mode()
        except: compare_models = False
        
        try: compare_prompts = input.compare_prompts_mode()
        except: compare_prompts = False
        
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                # Compute Model A (Prompt A)
                print("DEBUG: Starting heavy_compute A")
                result_A = await loop.run_in_executor(pool, heavy_compute, text, model_name)
                cached_result.set(result_A)
                
                # Compute Second Result if needed
                if compare_models:
                    # Case 1: Same Prompt (A), Different Model (B)
                    model_name_B = input.model_name_B()
                    print(f"DEBUG: Starting heavy_compute B ({model_name_B}) for Compare Models")
                    result_B = await loop.run_in_executor(pool, heavy_compute, text, model_name_B)
                    cached_result_B.set(result_B)
                
                elif compare_prompts:
                    # Case 2: Different Prompt (B), Same Model (A)
                    try: text_B = input.text_input_B().strip()
                    except: text_B = ""
                    
                    if text_B:
                        print(f"DEBUG: Starting heavy_compute B (Prompt B) for Compare Prompts")
                        # Use Model A for Prompt B
                        result_B = await loop.run_in_executor(pool, heavy_compute, text_B, model_name)
                        cached_result_B.set(result_B)
                    else:
                        cached_result_B.set(None)
                        
                else:
                    cached_result_B.set(None)
                    
        except Exception as e:
            print(f"ERROR in compute_all: {e}")
            traceback.print_exc()
            cached_result.set(None)
            cached_result_B.set(None)
        finally:
            running.set(False)

    # Sync History UI
    @reactive.Effect
    @reactive.event(input_history)
    def update_history_list():
        history = input_history.get()
        # Create HTML string
        html_content = ""
        for item in history:
             safe_item = item.replace("'", "\\'").replace('"', '&quot;')
             html_content += f"""<div class="history-item" onclick="selectHistoryItem('{safe_item}')">{item}</div>"""
        
        # Inject JS to update the dropdown content
        js_code = f"""
            var dropdown = document.getElementById('history-dropdown');
            if (dropdown) {{
                dropdown.innerHTML = `{html_content}`;
            }}
        """
        ui.insert_ui(selector="body", where="beforeEnd", ui=ui.tags.script(js_code))

    # -------------------------------------------------------------------------
    # SYNCHRONIZATION LOGIC (Cross-Model Control)
    # -------------------------------------------------------------------------
    
    # helper to sync inputs if values differ preventing infinite loops
    def sync_inputs(src, dest):
        val_src = input[src]()
        val_dest = input[dest]() if dest in input else None
        if val_src != val_dest:
            # We need to determine type of input to update correctly
            # Assumption: all synced inputs are selects or texts that act like strings
            # For most selectors in this app, ui.update_select works.
            ui.update_select(dest, selected=val_src)

    # A-B Sync logic removed as inputs are global
    @reactive.Effect
    @reactive.event(input.att_head)
    def sync_att_head_A_to_B():
        if input.compare_mode(): ui.update_select("att_head_B", selected=input.att_head())

    @reactive.Effect
    @reactive.event(input.att_head_B)
    def sync_att_head_B_to_A():
        if input.compare_mode(): ui.update_select("att_head", selected=input.att_head_B())

    # Focus Token Sync
    @reactive.Effect
    @reactive.event(input.scaled_attention_token)
    def sync_focus_A_to_B():
        if input.compare_mode(): ui.update_select("scaled_attention_token_B", selected=input.scaled_attention_token())

    @reactive.Effect
    @reactive.event(input.scaled_attention_token_B)
    def sync_focus_B_to_A():
        if input.compare_mode(): ui.update_select("scaled_attention_token", selected=input.scaled_attention_token_B())

    # Radar Layer/Head Sync
    @reactive.Effect
    @reactive.event(input.radar_layer)
    def sync_radar_layer_A_to_B():
        if input.compare_mode(): ui.update_select("radar_layer_B", selected=input.radar_layer())

    @reactive.Effect
    @reactive.event(input.radar_layer_B)
    def sync_radar_layer_B_to_A():
        if input.compare_mode(): ui.update_select("radar_layer", selected=input.radar_layer_B())

    @reactive.Effect
    @reactive.event(input.radar_head)
    def sync_radar_head_A_to_B():
        if input.compare_mode(): ui.update_select("radar_head_B", selected=input.radar_head())

    @reactive.Effect
    @reactive.event(input.radar_head_B)
    def sync_radar_head_B_to_A():
        if input.compare_mode(): ui.update_select("radar_head", selected=input.radar_head_B())

    # -------------------------------------------------------------------------
    # GLOBAL CONTROL BAR SYNC
    # -------------------------------------------------------------------------
    # Observers removed as inputs are now global directly

    # Set global view (All Heads, All Tokens) - TOGGLE
    @reactive.Effect
    @reactive.event(input.trigger_global_view)
    def set_global_view():
        # Toggle global metrics mode
        if global_metrics_mode.get() == "all":
            # Deselect: go back to specific layer/head
            global_metrics_mode.set("specific")
            ui.update_radio_buttons("radar_mode", selected="single")
            if input.compare_mode(): ui.update_radio_buttons("radar_mode_B", selected="single")
        else:
            # Select: set to global (all layers/heads)
            global_metrics_mode.set("all")
            ui.update_radio_buttons("radar_mode", selected="all")
            if input.compare_mode(): ui.update_radio_buttons("radar_mode_B", selected="all")
    
    # Reactive value to track if global metrics should use all layers/heads
    global_metrics_mode = reactive.Value("specific")  # "all" or "specific"
    
    # Reset to specific when sliders change
    @reactive.Effect
    @reactive.event(input.global_layer, input.global_head)
    def reset_metrics_mode():
        global_metrics_mode.set("specific")

    # -------------------------------------------------------------------------
    # FLOATING CONTROL BAR RENDERER
    # -------------------------------------------------------------------------
    @output
    @render.ui
    def floating_control_bar():
        """Render the floating fixed control bar with range sliders and clickable token sentence."""
        res = get_active_result()
        if not res:
            return None  # Don't show control bar until data is ready
        
        # Determine mode
        try: compare_prompts = input.compare_prompts_mode()
        except: compare_prompts = False

        # Get defaults
        try: current_layer = int(input.global_layer())
        except: current_layer = 0
        try: current_head = int(input.global_head())
        except: current_head = 0

        # --- MODEL A DATA ---
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        if is_gpt2:
            num_layers = len(encoder_model.h)
            num_heads = encoder_model.h[0].attn.num_heads
        else:
            num_layers = len(encoder_model.encoder.layer)
            num_heads = encoder_model.encoder.layer[0].attention.self.num_attention_heads
            
        tokens_A = res[0]
        clean_tokens_A = [t.replace("##", "").replace("Ġ", "") for t in tokens_A]
        
        # Get selected tokens A
        try:
            val_A = input.global_selected_tokens()
            selected_tokens_A = json.loads(val_A) if val_A else []
        except: selected_tokens_A = []
        if not selected_tokens_A: 
             try: 
                 t = int(input.global_focus_token())
                 if t >= 0: selected_tokens_A = [t]
             except: pass


        # --- MODEL B DATA (Only if Compare Prompts) ---
        tokens_B = []
        clean_tokens_B = []
        selected_tokens_B = []
        
        if compare_prompts:
            res_B = get_active_result("_B")
            if res_B:
                tokens_B = res_B[0]
                clean_tokens_B = [t.replace("##", "").replace("Ġ", "") for t in tokens_B]
                
                # Get selected tokens B
                try:
                    val_B = input.global_selected_tokens_B()
                    selected_tokens_B = json.loads(val_B) if val_B else []
                except: selected_tokens_B = []

        # --- BUILD CHIPS ---
        def build_chips(tokens, selected, prefix="A"):
            chips = []
            for i, token in enumerate(tokens):
                is_active = "active" if i in selected else ""
                chips.append(
                    ui.tags.span(
                        token,
                        class_=f"token-chip {is_active}",
                        **{"data-idx": str(i), "data-prefix": prefix, "onclick": f"handleTokenClick(this, '{prefix}')"}
                    )
                )
            return chips

        chips_A = build_chips(clean_tokens_A, selected_tokens_A, "A")
        chips_B = build_chips(clean_tokens_B, selected_tokens_B, "B") if compare_prompts else []
        
        
        # --- JAVASCRIPT ---
        # Updated to handle prefix (A/B) for independent selection
        slider_js = f"""
        (function() {{
            // State for span selections: {{ "A": {{anchor, selected}}, "B": ... }}
            window._spanState = {{
                "A": {{ anchor: null, selected: {json.dumps(selected_tokens_A)} }},
                "B": {{ anchor: null, selected: {json.dumps(selected_tokens_B)} }}
            }};

            // Expose handler globally so inline onclick can find it
            window.handleTokenClick = function(el, prefix) {{
                const idx = parseInt(el.getAttribute('data-idx'));
                const state = window._spanState[prefix];
                const event = window.event;
                
                if (event.shiftKey && state.anchor !== null) {{
                    // Range selection
                    const start = Math.min(state.anchor, idx);
                    const end = Math.max(state.anchor, idx);
                    const span = [];
                    for (let i = start; i <= end; i++) span.push(i);
                    updateSelection(prefix, span);
                }} else {{
                    // Toggle
                    const current = state.selected;
                    const idxInList = current.indexOf(idx);
                    let newSel;
                    
                    if (idxInList > -1) {{
                        newSel = current.filter(i => i !== idx);
                        if (state.anchor === idx) state.anchor = null;
                    }} else {{
                        newSel = current.concat([idx]);
                        state.anchor = idx; 
                    }}
                    updateSelection(prefix, newSel);
                }}
            }};

            function updateSelection(prefix, selectedArray) {{
                window._spanState[prefix].selected = selectedArray;
                
                // Update UI for this prefix
                const chips = document.querySelectorAll(`.token-chip[data-prefix='${{prefix}}']`);
                chips.forEach(chip => {{
                    const i = parseInt(chip.getAttribute('data-idx'));
                    if (selectedArray.includes(i)) chip.classList.add('active');
                    else chip.classList.remove('active');
                }});
                
                // Send to Shiny
                const inputName = prefix === 'A' ? 'global_selected_tokens' : 'global_selected_tokens_B';
                Shiny.setInputValue(inputName, JSON.stringify(selectedArray), {{priority: 'event'}});
                
                // Legacy fallback for A
                if (prefix === 'A') {{
                     Shiny.setInputValue('global_focus_token', selectedArray.length > 0 ? selectedArray[0] : -1, {{priority: 'event'}});
                }}
            }}

            // Debouncers for Sliders (Shared)
            function debounce(func, wait) {{
                let timeout;
                return function(...args) {{
                    const context = this;
                    clearTimeout(timeout);
                    timeout = setTimeout(() => func.apply(context, args), wait);
                }};
            }}
            
            const setLayer = debounce(val => Shiny.setInputValue('global_layer', val, {{priority: 'event'}}), 200);
            const setHead = debounce(val => Shiny.setInputValue('global_head', val, {{priority: 'event'}}), 200);
            const setTopK = debounce(val => Shiny.setInputValue('global_topk', val, {{priority: 'event'}}), 200);

            // Bind Sliders
            const bindSlider = (id, setter, valId) => {{
                const el = document.getElementById(id);
                if (el) el.oninput = function() {{ 
                    document.getElementById(valId).textContent = this.value; 
                    setter(this.value); 
                }};
            }};
            
            bindSlider('layer-slider', setLayer, 'layer-value');
            bindSlider('head-slider', setHead, 'head-value');
            bindSlider('topk-slider', setTopK, 'topk-value');
            
            // Global View Toggle
             const globalBtn = document.getElementById('trigger_global_view');
             if (globalBtn) {{
                globalBtn.addEventListener('click', function() {{
                    this.classList.toggle('active');
                }});
             }}
             
             // Norm Radio
            const radioGroup = document.getElementById('norm-radio-group');
            if (radioGroup) {{
                radioGroup.querySelectorAll('.radio-option').forEach(btn => {{
                    btn.onclick = function() {{
                        radioGroup.querySelectorAll('.radio-option').forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        Shiny.setInputValue('global_norm', this.getAttribute('data-value'), {{priority: 'event'}});
                    }};
                }});
            }}

            document.querySelector('.content')?.classList.add('has-control-bar');
        }})();
        """
        
        # --- UI LAYOUT ---
        
        token_area_content = []
        
        if compare_prompts:
            # Dual Row Layout
            token_area_content = [
                ui.div(
                    {"class": "token-row-split"},
                    ui.div(
                        {"class": "token-split-item"},
                        ui.span("A", class_="model-label-a"),
                        *chips_A
                    ),
                    ui.div(
                        {"class": "token-split-item item-b"},
                        ui.span("B", class_="model-label-b"),
                        *chips_B
                    )
                )
            ]
        else:
            # Single Row Layout
            token_area_content = [
                ui.div(
                    {"class": "token-sentence"},
                    *chips_A
                )
            ]

        return ui.div(
            {"class": "floating-control-bar"},
            
            # Title
            ui.span("Configurations", class_="bar-title"),
            
            # Controls
            ui.div(
                {"class": "controls-row"},
                
                # Global View
                ui.div(
                    {"class": "control-group"},
                    ui.span("View", class_="control-label"),
                    ui.input_action_button("trigger_global_view", "Global", class_=f"btn-global{' active' if global_metrics_mode.get() == 'all' else ''}")
                ),

                # Layer
                ui.div(
                    {"class": "control-group"},
                    ui.span("Layer", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span(str(current_layer), id="layer-value", class_="slider-value"),
                        ui.tags.input(type="range", id="layer-slider", min="0", max=str(num_layers - 1), value=str(current_layer), step="1")
                    )
                ),
                
                # Head
                ui.div(
                    {"class": "control-group"},
                    ui.span("Head", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span(str(current_head), id="head-value", class_="slider-value"),
                        ui.tags.input(type="range", id="head-slider", min="0", max=str(num_heads - 1), value=str(current_head), step="1")
                    )
                ),
                
                # Divider
                ui.div({"class": "control-divider"}),
                
                # Tokens (Dynamic)
                *token_area_content,
                
                # Divider
                ui.div({"class": "control-divider"}),

                # Norm
                ui.div(
                    {"class": "control-group"},
                    ui.span("Norm", class_="control-label"),
                    ui.div(
                        {"class": "radio-group", "id": "norm-radio-group"},
                        ui.span("Raw", class_="radio-option active", **{"data-value": "raw"}),
                        ui.span("Row", class_="radio-option", **{"data-value": "row"}),
                        ui.span("Rollout", class_="radio-option", **{"data-value": "rollout"}),
                    )
                ),
                
                # Top-K
                ui.div(
                    {"class": "control-group"},
                    ui.span("Top-K", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span("3", id="topk-value", class_="slider-value"),
                        ui.tags.input(type="range", id="topk-slider", min="1", max="20", value="3", step="1")
                    )
                ),
            ),
            
            # Script
            ui.tags.script(slider_js)
        )



    @output
    @render.ui
    def static_preview_text():
        """Shows just the input text in quotes before generation. Hidden after."""
        # Hide if results exist (dashboard will show its own preview)
        if cached_result.get():
            return None
        t = input.text_input().strip()
        if t:
            return ui.HTML(f'<div style="font-family:monospace;color:#6b7280;font-size:14px;">"{t}"</div>')
        else:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">Type a sentence above and click Generate All.</div>')

    @output
    @render.ui
    def static_preview_text_compare_a():
        """Static preview for Model A in compare mode. Hidden after generation."""
        # Only show if not generated AND we are in step A or just starting
        if cached_result.get():
             return None
             
        step = prompt_entry_step.get()
        # If we are in Step B, hide A's preview (user requested only show the current one)
        if step == "B":
            return None
            
        t = input.text_input().strip()
        if t:
            return ui.HTML(f'<div style="font-family:monospace;color:#3b82f6;font-size:14px;">"{t}"</div>')
        else:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">Type a sentence and click Go to Prompt B.</div>')

    @output
    @render.ui
    def static_preview_text_compare_b():
        """Static preview for Model B in compare mode. Hidden after generation."""
        # If we are in STEP B (Editing), we MUST show the preview, regardless of cached_result
        step = prompt_entry_step.get()
        
        if step != "B":
             return None

        # Check cached_result only if we were NOT editing (but we are, so skip check)
        # if cached_result.get(): return None 
        
        t = input.text_input_B().strip() # Use text_input_B here!
        print(f"DEBUG: static_preview_text_compare_b triggered. Step={step}, Text='{t}'")
        if t:
            return ui.HTML(f'<div style="font-family:monospace;color:#ff5ca9;font-size:14px;">"{t}"</div>')
        else:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">Type a sentence and click Generate All.</div>')

    def get_preview_text_view(res, text_input, model_suffix="", footer_html=""):
        t = text_input.strip() if text_input else ""
        
        # Determine colors based on model suffix
        color_rgb = "59, 130, 246"  # Blue (Default/Model A)
        color_hex = "#3b82f6"
        if model_suffix == "_B":
            color_rgb = "255, 92, 169" # Pink (Model B)
            color_hex = "#ff5ca9"

        if not res:
            return ui.HTML(f'<div style="font-family:monospace;color:#6b7280;font-size:14px; min-height: 48px; display: flex; align-items: center;">"{t}"</div>' if t else f'<div style="color:#9ca3af;font-size:12px; min-height: 48px; display: flex; align-items: center;">Model {model_suffix.replace("_", "") or "A"} not loaded.</div>')
            
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0:
            return ui.HTML('<div style="color:#9ca3af;font-size:12px;">No attention data available.</div>')
            
        att_layers = [layer[0].cpu().numpy() for layer in attentions]
        att_avg = np.mean(att_layers, axis=(0, 1))
        attention_received = att_avg.sum(axis=0)
        att_received_norm = (attention_received - attention_received.min()) / (attention_received.max() - attention_received.min() + 1e-10)
        
        token_html = []
        for i, (tok, att_recv, recv_norm) in enumerate(zip(tokens, attention_received, att_received_norm)):
            clean_tok = tok.replace("##", "").replace("Ġ", "")
            # GPT-2 lowercase for display if aggregated? No, visualization handles it.
            # Just keep clean logic.
            
            opacity = 0.2 + (recv_norm * 0.6)
            bg_color = f"rgba({color_rgb}, {opacity})"
            tooltip = f"Token: {clean_tok}&#10;Attention Received: {att_recv:.3f}"
            token_html.append(f'<span class="token-viz" style="background:{bg_color};" title="{tooltip}">{clean_tok}</span>')
            
        html = '<div class="token-viz-container">' + ''.join(token_html) + '</div>'
        
        legend_html = f'''
        <div style="display:flex; justify-content:space-between; margin-top:8px; align-items:center;">
            <div style="display:flex;gap:12px;font-size:9px;color:#6b7280;align-items:center;">
                <div style="display:flex;align-items:center;gap:4px;">
                    <div style="width:10px;height:10px;background:rgba({color_rgb},0.8);border-radius:2px;"></div><span>High Attention</span>
                </div>
                <div style="display:flex;align-items:center;gap:4px;">
                    <div style="width:10px;height:10px;background:rgba({color_rgb},0.2);border-radius:2px;"></div><span>Low Attention</span>
                </div>
            </div>
            <div style="font-size:9px;">
                {footer_html}
            </div>
        </div>
        '''
        return ui.HTML(html + legend_html)

    @output
    @render.ui
    def preview_text():
        res = get_active_result()
        return get_preview_text_view(res, input.text_input(), "")

    @output
    @render.ui
    def preview_text_B():
        res = get_active_result("_B")
        return get_preview_text_view(res, input.text_input(), "_B")


    def get_gpt2_dashboard_ui(res, input, output, session):
        tokens, _, _, _, _, _, _, encoder_model, *_ = res
        num_layers = len(encoder_model.h)
        num_heads = encoder_model.h[0].attn.num_heads
        
        # Get current selections

        
        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]

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
                        {"class": "header-simple"},
                        ui.h4("Q/K/V Projections")
                    ),
                    ui.p("Projects input to Query, Key, Value vectors.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_qkv_table(res, qkv_layer)
                ),
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-simple"},
                        ui.h4("Scaled Dot-Product Attention")
                    ),
                    ui.p("Calculates attention scores between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_scaled_attention_view(res, att_layer, att_head, focus_token_idx, top_k=top_k)
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
            ui.div({
                "class": "card"
            }, 
                ui.div(
                    {"style": "display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;"},
                    ui.h4("Global Attention Metrics", style="margin: 0;"),
                    ui.span("All Layers · All Heads", style="font-size: 11px; color: #94a3b8; font-weight: 500;")
                ),
                get_metrics_display(res)
            ),
            
            # Row 3: Attention Visualizations 
            ui.layout_columns(
                ui.output_ui("attention_map"),
                ui.output_ui("attention_flow"),
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
                            ui.h4("Head Specialization")
                        ),
                        ui.div(
                            {"class": "header-row-bottom"},
                            ui.span("Attention Mode:", class_="toggle-label"),
                            ui.output_ui("render_radar_view")
                        )
                    ),
                ),

                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-controls-stacked"},
                            ui.div(
                                {"class": "header-simple"},
                                ui.h4("Attention Dependency Tree")
                            )
                    ),
                    ui.output_ui("render_tree_view")
                ),
                col_widths=[5, 7]
            ),

            # Row 5: ISA
            ui.output_ui("isa_row_dynamic"),

            # Row 6: Unembedding & Predictions
            ui.layout_columns(
                ui.div({"class": "card"}, ui.h4("Hidden States"), ui.p("Final vector representation before projection.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_layer_output_view(res, num_layers - 1)),
                ui.div({"class": "card"}, ui.h4("Next Token Predictions"), ui.p("Probabilities for the next token (Softmax output).", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_output_probabilities(res, use_mlm_val, text_val, top_k=top_k)),
                col_widths=[6, 6]
            ),
            
            # Autoregressive Loop Note
            ui.div(
                {"class": "card"},
                ui.h4("Autoregressive Loop"),
                ui.p("In generation, the predicted token is added to the input, and the entire process repeats.", style="font-size:13px; color:#4b5563;")
            )
        )


    # --- Helper for Renderers to get Aggregated Data ---
    def get_active_result(suffix=""):
        res = cached_result.get() if suffix == "" else cached_result_B.get()
        if not res: return None
        
        use_word_level = False
        try:
            cm = input.compare_mode()
            wl = input.word_level_toggle()
            # Allow word level in single mode too if enabled
            if wl:
                use_word_level = True
        except Exception as e:
            # print(f"DEBUG: get_active_result error reading inputs: {e}")
            pass
            
        if use_word_level:
            from ..utils import aggregate_data_to_words
            res = aggregate_data_to_words(res, filter_special=True)
            
        return res

    @output
    @render.ui
    def visualization_options_container():
        return None

    @output
    @render.ui
    def render_embedding_table():
        res = get_active_result()
        if not res: return None
        return get_embedding_table(res)

    @output
    @render.ui
    def render_segment_table():
        res = get_active_result()
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
        res = get_active_result()
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
        res = get_active_result()
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
        res = get_active_result()
        if not res: return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Q/K/V Projections"),
            ui.p("Projects input to Query, Key, Value vectors.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_qkv_table(res, layer_idx)
        )

    @output
    @render.ui
    def render_scaled_attention():
        res = get_active_result()
        if not res: return None
        
        # Use global_selected_tokens for span support
        selected_indices = []
        try:
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass
        
        focus_indices = selected_indices if selected_indices else [0]
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        try: top_k = int(input.global_topk())
        except: top_k = 3

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Scaled Dot-Product Attention"),
            ui.p("Calculates attention scores between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=top_k)
        )

    @output
    @render.ui
    def render_ffn():
        res = get_active_result()
        if not res: return None
        try: layer = int(input.global_layer())
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
        res = get_active_result()
        if not res: return None
        try: layer = int(input.global_layer())
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
        res = get_active_result()
        if not res: return None
        try: layer = int(input.global_layer())
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
        res = get_active_result()
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
        res = get_active_result()
        if not res: return None
        
        # Determine if we should show predictions
        # GPT-2: Always show
        # BERT: Show only if switch is on
        try:
            model_family = input.model_family()
        except:
            model_family = "bert"
            
        try: text = input.text_input()
        except: text = ""
        try: top_k = int(input.global_topk())
        except: top_k = 3

        if model_family == "gpt2":
            use_mlm = True
            title = "Next Token Predictions (Causal)"
            desc = "Predicting the probability of the next token appearing after the sequence."
        else:
            # BERT logic: Check local state
            use_mlm = show_mlm_A.get()
            
            title = "Masked Token Predictions (MLM)"
            desc = "Pseudo-Likelihood: Each token is individually masked and predicted using the bidirectional context."

            if not use_mlm:
                 return ui.div(
                    {"class": "card", "style": "height: 100%; display: flex; flex-direction: column; justify-content: space-between;"},
                    ui.div(
                        ui.h4(title),
                        ui.p(desc, style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    ),
                    ui.div(
                         {"style": "flex-grow: 1; display: flex; align-items: center; justify-content: center; padding: 20px;"},
                         ui.input_action_button("trigger_mlm_A", "Generate Predictions", class_="btn-primary")
                    )
                )

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4(title),
            ui.p(desc, style="font-size:11px; color:#6b7280; margin-bottom:8px; min-height: 32px;"),
            get_output_probabilities(res, use_mlm, text, top_k=top_k)
        )

    @output
    @render.ui
    def render_radar_view():
        res = get_active_result()
        if not res: return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        try: mode = input.radar_mode()
        except: mode = "single"

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            ui.div(
                {"class": "header-controls-stacked"},
                ui.div(
                    {"class": "header-row-top"},
                    ui.h4("Head Specialization"),
                    ui.div(
                        {"style": "display: flex; align-items: center; gap: 12px;"},
                        ui.span("MODE:", style="font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 0.5px;"),
                        ui.input_radio_buttons("radar_mode", None, {"single": "Single Head", "all": "All Heads"}, inline=True, selected=mode)
                    )
                ),
                ui.p("Analyzes the linguistic roles (syntax, semantics, etc.) performed by each attention head.", style="font-size:11px; color:#6b7280; margin: 4px 0 0 0;"),
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
        res = get_active_result()
        if not res: return None
        try: root_idx = int(input.global_focus_token())
        except: root_idx = 0
        if root_idx == -1: root_idx = 0
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            ui.h4("Attention Dependency Tree"),
            ui.p("Visualizes the hierarchical influence of tokens on the selected focus token.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_influence_tree_ui(res, root_idx, layer_idx, head_idx)
        )

    @output
    @render.ui
    def render_global_metrics():
        res = get_active_result()
        if not res: return None
        
        # Check if we should use all layers/heads or specific selection
        use_all = global_metrics_mode.get() == "all"
        
        if use_all:
            layer_idx = None
            head_idx = None
            subtitle = "All Layers · All Heads"
        else:
            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0
            subtitle = f"Layer {layer_idx} · Head {head_idx}"
        
        return ui.div(
            {"class": "card"}, 
            ui.div(
                {"style": "display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;"},
                ui.h4("Global Attention Metrics", style="margin: 0;"),
                ui.span(subtitle, style="font-size: 11px; color: #94a3b8; font-weight: 500;")
            ),
            get_metrics_display(res, layer_idx=layer_idx, head_idx=head_idx)
        )

    def dashboard_layout_helper(is_gpt2, num_layers, num_heads, clean_tokens, suffix=""):
        # Helper to generate choices dict
        def get_choices(items):
            return {str(i): f"{i}: {t}" for i, t in enumerate(items)}
        
        # --- Word-Level Aggregation Logic ---
        res = get_active_result() if suffix == "" else get_active_result("_B")
        
        # Check if we should aggregate
        use_word_level = False
        try:
            cm = input.compare_mode()
            wl = input.word_level_toggle()
            if wl:
                use_word_level = True
        except:
            pass
            
        if use_word_level and res:
            # Apply aggregation
            from ..utils import aggregate_data_to_words
            res = aggregate_data_to_words(res, filter_special=True)
            
            # Update local tokens var for this render pass
            if res:
                 tokens = res[0]
                 # Update clean tokens for display
                 clean_tokens = tokens # They are already cleaned in aggregation
        
        # Legend for Tokenization Mode
        tokenization_legend = ""
        if use_word_level:
            tokenization_legend = ui.div(
                {"class": "tokenization-legend", "style": "font-size: 10px; color: #6b7280; margin-bottom: 8px; font-style: italic; background: #f3f4f6; padding: 4px 8px; border-radius: 4px; display: inline-block;"},
                "Tokenization: Word-Level (Aggregated)"
            )

        if is_gpt2:
            # GPT-2 Layout
            return ui.div(
                {"id": f"dashboard-container{suffix}", "class": "dashboard-stack gpt2-layout content-hidden"},
                
                # Tokenization Legend
                tokenization_legend,

                # Row 1: Embeddings
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Sentence Preview", "Token Embeddings", "vertical", suffix=suffix, style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), ui.output_ui(f"render_embedding_table{suffix}", style="height: 100%;"))
                    ),
                    arrow("Token Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_posenc_table{suffix}", style="height: 100%;"),
                        arrow("Sum & Layer Normalization", "Q/K/V Projections", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Positional Embeddings", "Sum & Layer Normalization", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_sum_layernorm{suffix}", style="height: 100%;")),
                ),

                # Row 2: Transformer Block Details (Attention)
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_qkv_table{suffix}", style="height: 100%;")
                    ),
                    arrow("Q/K/V Projections", "Scaled Dot-Product Attention", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_scaled_attention{suffix}", style="height: 100%;")
                    ),
                ),
                
                # Row 3: Global Metrics & Attention Map
                ui.output_ui(f"render_global_metrics{suffix}"),
                
                ui.layout_columns(
                    ui.output_ui(f"attention_map{suffix}"),
                    ui.output_ui(f"attention_flow{suffix}"),
                    col_widths=[6, 6]
                ),

                # Row 4: Radar & Tree
                ui.layout_columns(
                    ui.output_ui(f"render_radar_view{suffix}"),
                    ui.output_ui(f"render_tree_view{suffix}"),
                    col_widths=[5, 7]
                ),

                # Row 5: ISA
                ui.output_ui(f"isa_scatter{suffix}"),

                # Row 6: Transformer Block Details (FFN)
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Inter-Sentence Attention", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.output_ui(f"render_add_norm{suffix}", style="height: 100%;")
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_ffn{suffix}", style="height: 100%;"),
                        arrow("Add & Norm (post-FFN)", "Hidden States", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_add_norm_post_ffn{suffix}", style="height: 100%;")
                    ),
                ),

                # Row 7: Unembedding & Predictions
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_layer_output{suffix}", style="height: 100%;")),
                    arrow("Hidden States", "Next Token Predictions", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_mlm_predictions{suffix}", style="height: 100%;")),
                )
            )



        # Construct UI (BERT)
        return ui.div(
            {"id": f"dashboard-container{suffix}", "class": "dashboard-stack content-hidden"}, # Initially hidden
            ui.div(
                {"class": "dashboard-stack"},
                
                # Tokenization Legend
                tokenization_legend,

                # Row 1: Initial Embeddings 
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Sentence Preview", "Token Embeddings", "vertical", suffix=suffix, style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), ui.output_ui(f"render_embedding_table{suffix}"))
                    ),
                    arrow("Token Embeddings", "Segment Embeddings", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_segment_table{suffix}", style="height: 100%;")),
                    arrow("Segment Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_posenc_table{suffix}", style="height: 100%;")),
                ),
                
                # Row 2: Processing
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_sum_layernorm{suffix}")
                    ),
                    arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Positional Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;"),
                        ui.output_ui(f"render_qkv_table{suffix}")
                    ),
                    arrow("Q/K/V Projections", "Scaled Dot-Product Attention", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card"},
                        ui.output_ui(f"render_scaled_attention{suffix}")
                    ),
                ),
                
                # Row 3: Global Metrics
                ui.output_ui(f"render_global_metrics{suffix}"),
                
                # Row 4: Attention Visualizations 
                ui.layout_columns(
                    ui.output_ui(f"attention_map{suffix}"),
                    ui.output_ui(f"attention_flow{suffix}"),
                    col_widths=[6, 6]
                ),
                
                
                # Row 5: Specialization Analysis
                ui.layout_columns(
                    ui.output_ui(f"render_radar_view{suffix}"),
                    ui.output_ui(f"render_tree_view{suffix}"),
                    col_widths=[5, 7]
                ),
                
                
                # Row 6: Inter-Sentence Attention (full width)
                ui.output_ui("isa_row_dynamic") if suffix == "" else ui.output_ui(f"isa_scatter{suffix}"),
                
                
                # Row 7: Residual Connections
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Inter-Sentence Attention", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -27px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.output_ui(f"render_add_norm{suffix}")
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_ffn{suffix}"),
                        arrow("Add & Norm (post-FFN)", "Hidden States", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0;")
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_add_norm_post_ffn{suffix}")),
                ),
                
                # Row 8: Final Outputs 
                ui.div(
                    {"class": "flex-row-container"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.output_ui(f"render_layer_output{suffix}")
                    ),
                    arrow("Hidden States", "Token Output Predictions", "horizontal", suffix=suffix),
                    ui.div({"class": "flex-card"}, ui.output_ui(f"render_mlm_predictions{suffix}")),
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
        res = get_active_result()
        if not res: return []
        tokens = res[0]
        return [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]

    @reactive.effect
    def update_selectors():
        # Update A selectors
        clean_tokens = tokens_data()
        if clean_tokens:
            choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}
            ui.update_select("scaled_attention_token", choices=choices)
            ui.update_select("flow_token_select", choices={"all": "All tokens", **choices})
            ui.update_select("tree_root_token", choices=choices)

        # Update B selectors (if they exist in the UI)
        res_B = get_active_result("_B")
        if res_B:
            tokens_B = res_B[0]
            clean_tokens_B = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens_B]
            choices_B = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens_B)}
            
            # These selectors might not exist yet if B is not rendered, but Shiny handles that gracefully usually
            ui.update_select("scaled_attention_token_B", choices=choices_B)
            ui.update_select("flow_token_select_B", choices={"all": "All tokens", **choices_B})
            ui.update_select("tree_root_token_B", choices=choices_B)

    @output
    @render.ui
    def dashboard_content():
        config = current_layout_config.get()
        
        # Check comparison mode and fetch model details early
        res_A = get_active_result()
        res_B = get_active_result("_B")
        
        try: compare_models = input.compare_mode()
        except: compare_models = False
        try: compare_prompts = input.compare_prompts_mode()
        except: compare_prompts = False
        
        is_family_diff = False
        tokenization_info = ""
        
        if compare_models and res_A and res_B:
            _, _, _, _, _, _, _, encoder_model_A, *_ = res_A
            _, _, _, _, _, _, _, encoder_model_B, *_ = res_B
            is_gpt2_A = not hasattr(encoder_model_A, "encoder")
            is_gpt2_B = not hasattr(encoder_model_B, "encoder")
            
            if is_gpt2_A != is_gpt2_B:
                is_family_diff = True
            
            type_A = "Byte-Pair" if is_gpt2_A else "WordPiece"
            type_B = "Byte-Pair" if is_gpt2_B else "WordPiece"
            # tokenization_info constructed below
            
            # Condense legend with tooltips
            def get_tok_badge(variant, color):
                name = "Byte Pair Encoding" if variant == "Byte-Pair" else "WordPiece"
                short_name = "BPE" if variant == "Byte-Pair" else "WordPiece"
                desc = "GPT-2's subword algorithm. Merges frequent byte pairs. Treats spaces as part of tokens (Ġ prefix)." if variant == "Byte-Pair" else "BERT's subword algorithm. Splits words into stem + suffix (## prefix)."
                return f"""
                <span class='info-tooltip-wrapper' style='cursor:help;'>
                    <span style='display:inline-flex; align-items:center; gap:4px; background:{color}15; border:1px solid {color}40; padding:2px 8px; border-radius:4px;'>
                        <strong style='color:{color}; font-size:10px;'>{short_name}</strong>
                    </span>
                    <div class='info-tooltip-content'>
                        <strong>{name}</strong>
                        <p>{desc}</p>
                    </div>
                </span>
                """

            badge_A = get_tok_badge(type_A, "#3b82f6")
            badge_B = get_tok_badge(type_B, "#ff5ca9")
            
            tok_info_A = f"<span style='font-size:9px; color:#9ca3af;'>Tokenization: {badge_A}</span>"
            tok_info_B = f"<span style='font-size:9px; color:#9ca3af;'>Tokenization: {badge_B}</span>"

        # Helper to create the toggle UI with updated tooltip and theming
        def create_word_level_toggle(color_theme="blue"):
            # Define colors based on theme
            if color_theme == "pink":
                # Pink Theme (Model B)
                bg_gradient = "linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%)" # Rose-50 to Rose-100
                border_color = "#fecdd3" # Rose-200
                text_color = "#be185d" # Pink-700
                icon_bg = "#ec4899" # Pink-500
                highlight_bg = "rgba(236,72,153,0.15)"
                highlight_border = "rgba(236,72,153,0.3)"
                highlight_text = "#ec4899"
            else:
                # Blue Theme (Model A - Default)
                bg_gradient = "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)" # Sky-50 to Sky-100
                border_color = "#bae6fd" # Sky-200
                text_color = "#0369a1" # Sky-700
                icon_bg = "#0ea5e9" # Sky-500
                highlight_bg = "rgba(59,130,246,0.1)" # Blue-ish
                highlight_border = "rgba(59,130,246,0.3)"
                highlight_text = "#3b82f6"

            return ui.div(
                {"class": "info-tooltip-wrapper word-level-toggle", "style": f"display: inline-flex; align-items: center; justify-content: center; gap: 4px; background: {bg_gradient}; border: 1px solid {border_color}; padding: 0 6px; border-radius: 5px; height: 24px;"},
                ui.input_switch("word_level_toggle", None, value=input.word_level_toggle() if "word_level_toggle" in input else False, width="auto"),
                ui.span("Activate Word Level", style=f"font-size: 9px; font-weight: 600; color: {text_color}; white-space: nowrap; line-height: 1;"),
                ui.span({"class": "info-tooltip-icon", "style": f"font-size:6px; width:10px; height:10px; line-height:10px; font-family:'PT Serif', serif; background: {icon_bg}; color: white;"}, "i"),
                ui.div(
                    {"class": "info-tooltip-content", "style": "width: 320px; max-width: 320px;"},
                    ui.HTML(f"""
                        <strong style='font-size: 11px;'>Word Level Aggregation</strong>
                        <p style='margin-bottom: 8px;'>Aligns BERT (WordPiece) with GPT-2 (BPE) by merging sub-tokens and re-distributing attention.</p>

                        <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 4px 0;'>
                            <strong style='color:{highlight_text}; font-size: 10px;'>1. Filtering & Re-normalization step-by-step:</strong>
                            <p style='font-size:9px; margin:4px 0;'>BERT often attends heavily to special tokens like [sep]. Because GPT-2 doesn't have these, we remove them to compare "content-to-content" attention.</p>
                            
                            <div style='background:{highlight_bg}; padding:6px; border-radius:3px; margin-top:4px; border:1px dashed {highlight_border};'>
                                <strong style='font-size:9px; color:{highlight_text};'>Example:</strong>
                                <div style='display: grid; grid-template-columns: 1fr auto 1fr; gap: 4px; align-items: center; font-size: 9px; margin-top: 4px;'>
                                    <div style='text-align:center;'>
                                        <div style='color:#94a3b8; margin-bottom:2px;'>Raw Attention</div>
                                        <div>[CLS]: 20%</div>
                                        <div>[SEP]: 10%</div>
                                        <div style='font-weight:bold; color:white;'>Content: 70%</div>
                                    </div>
                                    <div style='color:{highlight_text}; font-weight:bold;'>→</div>
                                    <div style='text-align:center;'>
                                        <div style='color:#94a3b8; margin-bottom:2px;'>Re-normalized</div>
                                        <div style='text-decoration:line-through; opacity:0.5;'>[CLS]: 0%</div>
                                        <div style='text-decoration:line-through; opacity:0.5;'>[SEP]: 0%</div>
                                        <div style='font-weight:bold; color:{highlight_text};'>Content: 100%</div>
                                    </div>
                                </div>
                                <div style='font-size: 8px; color: #94a3b8; margin-top: 4px; text-align: center;'>
                                    (0.70 / 0.70 = 1.0)
                                </div>
                            </div>
                        </div>

                        <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 8px 0;'>
                            <strong style='color:{highlight_text}; font-size: 10px;'>2. Sub-token Merging:</strong>
                             
                            <div style='background:{highlight_bg}; padding:6px; border-radius:3px; margin:4px 0; border:1px dashed {highlight_border};'>
                                <strong style='font-size:9px; color:{highlight_text};'>Merge Example:</strong>
                                <div style='display: flex; gap: 4px; align-items: center; justify-content: center; font-size: 9px; margin-top: 4px;'>
                                    <div style='border: 1px solid {highlight_text}60; padding: 1px 4px; border-radius: 3px; background: {highlight_text}20;'>play</div>
                                    <div style='color: #94a3b8;'>+</div>
                                    <div style='border: 1px solid {highlight_text}60; padding: 1px 4px; border-radius: 3px; background: {highlight_text}20;'>##ing</div>
                                    <div style='color:{highlight_text}; font-weight:bold;'>→</div>
                                    <div style='border: 1px solid {highlight_text}; padding: 1px 6px; border-radius: 3px; background: {highlight_text}40; font-weight: bold;'>playing*</div>
                                </div>
                            </div>

                             <ul style='font-size:9px; margin:4px 0 0 0; padding-left:14px; color:#e2e8f0;'>
                                <li><b>Vectors:</b> Average of sub-token vectors (mean pool).</li>
                                <li><b>Attention:</b> Average input attention (receiving), Sum output attention (sending).</li>
                            </ul>
                        </div>
                        
                        <div style='margin-top: 6px; font-size: 9px; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 4px;'>
                            Tokens marked with <b style="color:{highlight_text};">*</b> are merged.
                        </div>
                    """)
                )
            )

        # Helper to create the header
        def create_preview_header(has_toggle=False, color_theme="blue"):
            right_content = ui.div(class_="toggle-placeholder", style="width: 1px;") # Placeholder to keep height
            if has_toggle:
                right_content = ui.div(
                    {"style": "display: flex; align-items: center; gap: 8px; margin-left: 12px;"},
                    create_word_level_toggle(color_theme)
                )
            
            return ui.div(
                {"class": "viz-header-with-info", "style": "margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; min-height: 24px;"},
                ui.div(
                    {"style": "display: flex; align-items: center;"},
                    ui.h4("Sentence Preview", style="margin: 0; margin-right: 8px;"),
                    ui.div(
                        {"class": "info-tooltip-wrapper", "style": "margin-left: 6px;"},
                        ui.span({"class": "info-tooltip-icon", "style": "font-size:8px; width:14px; height:14px; line-height:14px; font-family:'PT Serif', serif;"}, "i"),
                        ui.div(
                            {"class": "info-tooltip-content"},
                            ui.HTML("""
                                <strong>Attention Received</strong>
                                <p>Background opacity highlights tokens that the model focuses on most.</p>
                                <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 8px 0;'>
                                    <strong style='color:#3b82f6;'>Calculation:</strong>
                                    <code style='display:block; margin-top:4px;'>AVG(All Layers, All Heads) → Sum(Columns)</code>
                                </div>
                                <p style='font-size:10px; color:#fff; margin-bottom: 8px;'>Sum of attention weights received from all other tokens, averaged across all layers and heads.</p>
                                <p style='font-style:italic; color:#fff;'>Hover tokens for attention received</p>
                            """)
                        )
                    ),
                ),
                right_content
            )

        # Logic to place toggle on BERT when families differ
        toggle_on_A = False
        toggle_on_B = False

        if is_family_diff:
            # Place on whichever is NOT GPT-2 (i.e. is BERT)
            if is_gpt2_A:
                toggle_on_B = True # A is GPT2, so B must be BERT
            else:
                toggle_on_A = True # A is BERT
        
        # Create headers with appropriate themes
        # Header A is always Blue-themed naturally, Header B is Pink-themed
        preview_title = create_preview_header(toggle_on_A, "blue") 
        preview_title_B = create_preview_header(toggle_on_B, "pink")

        compare = compare_models or compare_prompts
        
        # Determine labels
        header_a = "Model A"
        header_b = "Model B"
        if compare_prompts:
            header_a = "Prompt A"
            header_b = "Prompt B"
        
        if not config:
            # BEFORE GENERATION - Show static preview
            t = input.text_input().strip()
            preview_html = f'<div style="font-family:monospace;color:#6b7280;font-size:14px;">"{t}"</div>' if t else '<div style="color:#9ca3af;font-size:12px;">Type a sentence above and click Generate All.</div>'
            
            if not compare:
                # Single mode static preview
                return ui.div(
                    ui.div(
                        {"class": "card"},
                        ui.h4("Sentence Preview"),
                        ui.HTML(preview_html),
                    ),
                    ui.HTML("<script>$('#generate_all').html('Generate All').prop('disabled', false).css('opacity', '1');</script>")
                )
            else:
                # Compare mode - show headers + paired static previews
                # Use separate variables for A and B inputs
                if compare_prompts:
                    t_b = input.text_input_B().strip()
                else:
                    t_b = input.text_input().strip()
                
                preview_a = f'<div style="font-family:monospace;color:#3b82f6;font-size:14px;">"{t}"</div>' if t else '<div style="color:#9ca3af;font-size:12px;">Type a sentence and click Generate All.</div>'
                preview_b = f'<div style="font-family:monospace;color:#ff5ca9;font-size:14px;">"{t_b}"</div>' if t_b else '<div style="color:#9ca3af;font-size:12px;">Type a sentence and click Generate All.</div>'
                return ui.div(
                    {"id": "dashboard-container-compare", "class": "dashboard-stack"},
                    # Header: MODEL A | MODEL B
                    ui.layout_columns(
                        ui.h3(header_a, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                        ui.h3(header_b, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                        col_widths=[6, 6]
                    ),
                    # Paired previews
                    ui.layout_columns(
                        ui.div({"class": "card", "style": "border: 2px solid #3b82f6;"}, ui.h4("Sentence Preview"), ui.HTML(preview_a)),
                        ui.div({"class": "card", "style": "border: 2px solid #ff5ca9;"}, ui.h4("Sentence Preview"), ui.HTML(preview_b)),
                        col_widths=[6, 6]
                    ),
                    ui.HTML("<script>$('#generate_all').html('Generate All').prop('disabled', false).css('opacity', '1');</script>")
                )
        
        is_gpt2, num_layers, num_heads = config
        
        print("DEBUG: Rendering dashboard_content (Layout Re-build)")
        
        if not compare:
            # ORIGINAL MODE - Always use output_ui for preview to avoid layout flash
            res = get_active_result()
            
            if not res:
                # Only show preview card + loading message when waiting for data
                return ui.div(
                    # Always use output_ui for preview - it handles grey/colored states internally
                    ui.div(
                        {"class": "card", "style": "margin-bottom: 32px;"},
                        preview_title,
                        get_preview_text_view(res, input.text_input(), ""),
                    ),
                    ui.div(
                        {"style": "padding: 40px; text-align: center; color: #9ca3af;"},
                        ui.p("Generating data...", style="font-size: 14px; animation: pulse 1.5s infinite;")
                    )
                )
            
            # Data is ready - show preview + full dashboard
            return ui.div(
                ui.div(
                    {"class": "card", "style": "margin-bottom: 32px;"},
                    preview_title,
                    get_preview_text_view(res, input.text_input(), ""),
                ),
                # Dashboard layout
                dashboard_layout_helper(is_gpt2, num_layers, num_heads, [], suffix="")
            )
        else:
            # SIDE-BY-SIDE MODE - Paired Sections (no arrows)
            # Each section rendered as: Section A | Section B
            # Get Model B config
            res_B = cached_result_B.get()
            is_gpt2_B = False
            num_layers_B = 12
            num_heads_B = 12
            if res_B:
                 _, _, _, _, _, _, _, encoder_model_B, *_ = res_B
                 is_gpt2_B = not hasattr(encoder_model_B, "encoder")
                 if is_gpt2_B:
                    num_layers_B = len(encoder_model_B.h)
                    num_heads_B = encoder_model_B.h[0].attn.num_heads
                 else:
                    num_layers_B = len(encoder_model_B.encoder.layer)
                    num_heads_B = encoder_model_B.encoder.layer[0].attention.self.num_attention_heads

            # Simple pairing helper - adds colored border around existing cards
            def paired(output_a, output_b):
                return ui.layout_columns(
                    ui.div({"class": "compare-wrapper-a"}, output_a),
                    ui.div({"class": "compare-wrapper-b"}, output_b),
                    col_widths=[6, 6]
                )

            # For outputs that need card wrapper (preview_text doesn't have one)
            def paired_with_card(title, output_a, output_b):
                header = ui.h4(title) if isinstance(title, str) else title
                return ui.layout_columns(
                    ui.div({"class": "card compare-card-a"}, header, output_a),
                    ui.div({"class": "card compare-card-b"}, header, output_b),
                    col_widths=[6, 6]
                )

            # Paired arrows - vertical arrows for both models
            def paired_arrows(from_section, to_section):
                return ui.layout_columns(
                    ui.div(
                        {"style": "display: flex; justify-content: center; padding: 0; margin: 0;"},
                        arrow(from_section, to_section, "vertical", suffix="_A", extra_class="arrow-blue")
                    ),
                    ui.div(
                        {"style": "display: flex; justify-content: center; padding: 0; margin: 0;"},
                        arrow(from_section, to_section, "vertical", suffix="_B", extra_class="arrow-pink")
                    ),
                    col_widths=[6, 6]
                )

            # Dynamic Arrow Renderers for Compare Mode to prevent "pop-in"
            # These renderers wait for data to be available before showing the arrow
            arrow_defs = [
                ("arrow_pos_qkv", "Positional Embeddings", "Q/K/V Projections"),
                ("arrow_qkv_scaled", "Q/K/V Projections", "Scaled Attention"),
                ("arrow_scaled_global", "Scaled Attention", "Global Metrics"),
                ("arrow_global_map", "Global Metrics", "Attention Map"),
                ("arrow_map_flow", "Attention Map", "Attention Flow"),
                ("arrow_flow_radar", "Attention Flow", "Radar View"),
                ("arrow_radar_tree", "Radar View", "Tree View"),
                ("arrow_tree_isa", "Tree View", "ISA"),
                ("arrow_isa_addnorm", "ISA", "Add & Norm"),
                ("arrow_addnorm_ffn", "Add & Norm", "Feed-Forward Network"),
                ("arrow_ffn_post", "Feed-Forward Network", "Add & Norm (Post-FFN)"),
                ("arrow_post_hidden", "Add & Norm (Post-FFN)", "Hidden States"),
                ("arrow_hidden_next", "Hidden States", "Next Token Predictions"),
            ]

            for arrow_id, from_s, to_s in arrow_defs:
                def register_arrow_renderer(aid, f, t):
                    @output(id=aid)
                    @render.ui
                    def _render_arrow():
                        res = get_active_result()
                        res_B = get_active_result("_B")
                        if not res or not res_B: 
                            return None
                        return paired_arrows(f, t)
                register_arrow_renderer(arrow_id, from_s, to_s)

            # Load results
            res_A = get_active_result()
            res_B = get_active_result("_B")

            # If strictly waiting for data, show only preview row + loading message
            # But ALWAYS use output_ui (not static HTML) to avoid blank flash on transition
            if not res_A or not res_B:
                 return ui.div(
                     {"id": "dashboard-container-compare", "class": "dashboard-stack"},
                      ui.layout_columns(
                          ui.h3(header_a, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                          ui.h3(header_b, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                          col_widths=[6, 6]
                     ),
                    # Use output_ui - the renderers handle grey/colored states internally
                    paired_with_card("Sentence Preview", ui.output_ui("preview_text"), ui.output_ui("preview_text_B")),
                    ui.div(
                        {"style": "padding: 40px; text-align: center; color: #9ca3af;"},
                        ui.p("Generate comparison data...", style="font-size: 14px; animation: pulse 1.5s infinite;")
                    )
                 )

            # Render Paired Layout - simple approach like single mode
            return ui.div(
                {"id": "dashboard-container-compare", "class": "dashboard-stack"},
                
                # Top Header: MODEL A | MODEL B
                ui.layout_columns(
                    ui.h3(header_a, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.h3(header_b, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    col_widths=[6, 6]
                ),
                
                # Row 0: Sentence Preview
                # Use manual layout to put Toggle only on Card A (avoid duplicate ID)
                ui.layout_columns(
                    ui.div({"class": "card compare-card-a"}, preview_title, get_preview_text_view(res_A, input.text_input(), "", footer_html=tok_info_A if is_family_diff else "")),

                    ui.div(
                        {"class": "card compare-card-b"},
                        # Header for B 
                        preview_title_B,
                        get_preview_text_view(res_B, input.text_input(), "_B", footer_html=tok_info_B if is_family_diff else "")
                    ),
                    col_widths=[6, 6]
                ),
                
                # Hidden container for Simultaneous Reveal
                ui.div(
                    {"id": "compare-content-body", "class": "content-hidden"},
                    
                    # Row 0.5: First Arrow (Inserted here so it reveals with the content)
                    ui.div(paired_arrows("Sentence Preview", "Token Embeddings"), class_="arrow-row"),

                    # Row 1: Token Embeddings
                    paired_with_card("Token Embeddings", get_embedding_table(res_A), get_embedding_table(res_B)),
                    ui.div(paired_arrows("Token Embeddings", "Positional Embeddings"), class_="arrow-row"),

                    # Row 2: Positional Embeddings
                    paired_with_card("Positional Embeddings", get_posenc_table(res_A), get_posenc_table(res_B)),
                    ui.output_ui("arrow_pos_qkv", class_="arrow-row"),

                    # Row 3: Q/K/V Projections
                    paired(ui.output_ui("render_qkv_table"), ui.output_ui("render_qkv_table_B")),
                    ui.output_ui("arrow_qkv_scaled", class_="arrow-row"),

                    # Row 4: Scaled Attention
                    paired(ui.output_ui("render_scaled_attention"), ui.output_ui("render_scaled_attention_B")),
                    ui.output_ui("arrow_scaled_global", class_="arrow-row"),

                    # Row 5: Global Metrics
                    paired(ui.output_ui("render_global_metrics"), ui.output_ui("render_global_metrics_B")),
                    ui.output_ui("arrow_global_map", class_="arrow-row"),

                    # Row 6: Attention Map
                    paired(ui.output_ui("attention_map"), ui.output_ui("attention_map_B")),
                    ui.output_ui("arrow_map_flow", class_="arrow-row"),

                    # Row 7: Attention Flow
                    paired(ui.output_ui("attention_flow"), ui.output_ui("attention_flow_B")),
                    ui.output_ui("arrow_flow_radar", class_="arrow-row"),

                    # Row 8: Radar View
                    paired(ui.output_ui("render_radar_view"), ui.output_ui("render_radar_view_B")),
                    ui.output_ui("arrow_radar_tree", class_="arrow-row"),

                    # Row 9: Tree View
                    paired(ui.output_ui("render_tree_view"), ui.output_ui("render_tree_view_B")),
                    ui.output_ui("arrow_tree_isa", class_="arrow-row"),

                    # Row 10: ISA
                    paired(ui.output_ui("isa_scatter_A_compare"), ui.output_ui("isa_scatter_B_compare")),
                    ui.output_ui("arrow_isa_addnorm", class_="arrow-row"),

                    # Row 11: Add & Norm
                    paired(ui.output_ui("render_add_norm"), ui.output_ui("render_add_norm_B")),
                    ui.output_ui("arrow_addnorm_ffn", class_="arrow-row"),

                    # Row 12: Feed-Forward Network
                    paired(ui.output_ui("render_ffn"), ui.output_ui("render_ffn_B")),
                    ui.output_ui("arrow_ffn_post", class_="arrow-row"),

                    # Row 13: Add & Norm (Post-FFN)
                    paired(ui.output_ui("render_add_norm_post_ffn"), ui.output_ui("render_add_norm_post_ffn_B")),
                    ui.output_ui("arrow_post_hidden", class_="arrow-row"),

                    # Row 14: Hidden States
                    paired(ui.output_ui("render_layer_output"), ui.output_ui("render_layer_output_B")),
                    ui.output_ui("arrow_hidden_next", class_="arrow-row"),

                    # Row 15: Next Token Predictions
                    paired(ui.output_ui("render_mlm_predictions"), ui.output_ui("render_mlm_predictions_B")),
                )
            )

    # This function replaces the previous @output @render.ui def influence_tree():
    def get_isa_scatter_view(res, suffix="", vertical_layout=False, plot_only=False):
        print(f"DEBUG: get_isa_scatter_view called for {suffix} with vertical_layout={vertical_layout} plot_only={plot_only}")
        if not res:
            return None

        # isa_data is index 10 (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, mlm_model, head_specialization, isa_data, head_clusters)
        # Using res[-1] is dangerous now that we added head_clusters.
        if len(res) > 10:
            isa_data = res[10]
        else:
            isa_data = res[-1] # Fallback for old tuples? Should not happen.
            if isinstance(isa_data, list): # If it's the cluster list by mistake
                 isa_data = None

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

        # Clean tokens for display in hover_texts
        cleaned_sentences = [s.replace("Ġ", "").replace("##", "") for s in sentences]

        hover_texts = [
            f"Target ← {cleaned_sentences[int(r)][:60]}...<br>Source → {cleaned_sentences[int(c)][:60]}...<br>ISA = {s:.4f}"
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

        labels = [s[:30].replace("Ġ", "").replace("##", "") + "..." if len(s) > 30 else s.replace("Ġ", "").replace("##", "") for s in sentences]

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

        div_id = f"isa_scatter_plot{suffix}"
        
        # Generate HTML with unique ID
        plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id=div_id, config={'displayModeBar': False})
        
        # Custom JS to handle clicks, send to Shiny, AND stop loading state
        # This is placed here because the ISA plot is the heaviest component.
        # When this renders, we know the data is ready.
        js = f"""
        <script>
        (function() {{
            // Stop loading state (Button Reset)
            var btn = $('#generate_all');
            if (btn.data('original-content')) {{
                btn.html(btn.data('original-content'));
            }} else {{
                btn.html('Generate All');
            }}
            btn.prop('disabled', false).css('opacity', '1');
            
            // Show Dashboard
            $('#dashboard-container').removeClass('content-hidden').addClass('content-visible');

            // Trigger Simultaneous Reveal for Compare Mode
            $('#compare-content-body').removeClass('content-hidden').addClass('content-visible');

            console.log("DEBUG: Initializing ISA Plot Script for {div_id}");
            function initPlot() {{
                var plot = document.getElementById('{div_id}');
                if (plot) {{
                    console.log("DEBUG: ISA Plot found, attaching listener");
                    plot.on('plotly_click', function(data){{
                        var pt = data.points[0];
                        var x = pt.x; // source index
                        var y = pt.y; // target index
                        // Send to Shiny input 'isa_scatter_click' with suffix
                        Shiny.setInputValue('isa_scatter_click{suffix}', {{x: x, y: y}}, {{priority: 'event'}});
                    }});
                }} else {{
                    console.log("DEBUG: ISA Plot not found yet, retrying...");
                    setTimeout(initPlot, 100);
                }}
            }}
            initPlot();
        }})();
        </script>
        """
        
        # Determine model type for explanation
        _, _, _, _, _, _, _, encoder_model, *_ = res
        is_gpt2 = not hasattr(encoder_model, "encoder")
        model_type = "GPT-2" if is_gpt2 else "BERT"

        col_layout = [12, 12] if vertical_layout else [6, 6]

        if plot_only:
            return ui.HTML(plot_html + js)

        return ui.div(
            {"class": "card"},
            ui.div(
                {"class": "viz-header-with-info", "style": "margin-bottom: 8px;"},
                ui.h4(
                    "Inter-Sentence Attention (ISA)", 
                    style="margin: 0;"
                ),
                ui.div(
                    {"class": "info-tooltip-wrapper"},
                    ui.span({
                        "class": "info-tooltip-icon", 
                        "onclick": f"showISACalcExplanation('{model_type}')",
                        "style": "cursor: pointer;"
                    }, "i"),
                    ui.div(
                        {"class": "info-tooltip-content"},
                        ui.HTML("""
                            <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Inter-Sentence Attention (ISA)</strong>
                            
                            <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Measures how strongly tokens in one sentence attend to tokens in another sentence, aggregating across all layers and attention heads.</p>
                            
                            <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Three-step max aggregation:<br>
                            <code style='font-size:10px;background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px'>ISA = max<sub>heads</sub>(max<sub>tokens</sub>(max<sub>layers</sub>(α)))</code></p>
                            
                            <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                <strong style='color:#8b5cf6;font-size:11px'>Score Interpretation:</strong>
                                <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                    <span style='color:#22c55e'>● High (>0.8): Strong dependency</span>
                                    <span style='color:#eab308'>● Mid (0.4-0.8): Moderate</span>
                                    <span style='color:#ef4444'>● Low (<0.4): Weak link</span>
                                </div>
                            </div>
                            
                            <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                <em>Click icon for formulas & details</em>
                            </p>
                        """)
                    )
                )
            ),
            ui.div({"class": "viz-description"}, "Visualizes attention strength between sentence pairs. Click any point to see token-level attention details. ⚠️ Higher ISA scores indicate stronger cross-sentence attention, not necessarily semantic similarity."),
            ui.layout_columns(
                ui.div(
                    {"style": "width: 100%; display: flex; justify-content: center; align-items: center; margin-bottom: 20px;" if vertical_layout else "height: 500px; width: 100%; display: flex; justify-content: center; align-items: center;"},
                    ui.HTML(plot_html + js)
                ),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; height: 100%;"},
                    ui.div(ui.output_ui(f"isa_detail_info{suffix}"), style="flex: 0 0 auto; margin-bottom: 10px;"),
                    ui.div(ui.output_ui(f"isa_token_view{suffix}"), style="flex: 1; display: flex; flex-direction: column;"),
                ),
                col_widths=col_layout,
            ),
        )

    @output(id="isa_scatter")
    @render.ui
    def isa_scatter_renderer():
        res = get_active_result()
        return get_isa_scatter_view(res, suffix="", plot_only=False)

    @output(id="isa_scatter_A_compare")
    @render.ui
    def isa_scatter_renderer_A_compare():
        res = get_active_result()
        return get_isa_scatter_view(res, suffix="", vertical_layout=True)

    @output(id="isa_scatter_B_compare")
    @render.ui
    def isa_scatter_renderer_B_compare():
        res = get_active_result("_B")
        return get_isa_scatter_view(res, suffix="_B", vertical_layout=True)

    @output(id="isa_scatter_B")
    @render.ui
    def isa_scatter_renderer_B():
        res = get_active_result("_B")
        return get_isa_scatter_view(res, suffix="_B", vertical_layout=True)


    @output
    @render.ui
    def isa_row_dynamic():
        """
        Dynamic ISA Layout for Single Mode.
        Now simplified to just render the main isa_scatter, which handles its own Card/Layout/Description.
        """
        return ui.output_ui("isa_scatter")

    # @output(id="isa_scatter")
    # @render.ui
    def _legacy_isa_scatter_renderer():
        res = cached_result.get()
        if not res:
            return None

        isa_data = res[-2]

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

        # Clean tokens for display in hover_texts
        cleaned_sentences = [s.replace("Ġ", "").replace("##", "") for s in sentences]

        hover_texts = [
            f"Target ← {cleaned_sentences[int(r)][:60]}...<br>Source → {cleaned_sentences[int(c)][:60]}...<br>ISA = {s:.4f}"
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

        labels = [s[:30].replace("Ġ", "").replace("##", "") + "..." if len(s) > 30 else s.replace("Ġ", "").replace("##", "") for s in sentences]

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
        
        # Custom JS to handle clicks, send to Shiny, AND stop loading state
        # This is placed here because the ISA plot is the heaviest component.
        # When this renders, we know the data is ready.
        js = """
        <script>
        (function() {
            // Stop loading state (Button Reset)
            var btn = $('#generate_all');
            if (btn.data('original-content')) {
                btn.html(btn.data('original-content'));
            } else {
                btn.html('Generate All');
            }
            btn.prop('disabled', false).css('opacity', '1');
            
            // Show Dashboard
            $('#dashboard-container').removeClass('content-hidden').addClass('content-visible');

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
                    {"style": "display: flex; flex-direction: column; justify-content: center; height: 500px; padding-left: 20px;"},
                    ui.div(
                        {"style": "display: flex; flex-direction: column; align-items: center; width: 100%;"}, # Wrapper for visual grouping
                        ui.div(ui.output_ui("isa_detail_info"), style="margin-bottom: 5px; text-align: center;"),
                        ui.div(ui.output_ui("isa_token_view"), style="width: 100%; display: flex; justify-content: center;"),
                    )
                ),
                col_widths=[6, 6],
            ),
        )


    @output
    @render.ui
    def isa_token_view():
        pair = isa_selected_pair()
        res = get_active_result()

        if res is None or pair is None:
            return ui.div(
                ui.p("Select a point on the scatter plot to view token-to-token attention.", 
                     style="color: #94a3b8; font-size: 13px; font-style: italic;"),
                style="height: 350px; display: flex; align-items: center; justify-content: center; border: 1px dashed #e2e8f0; border-radius: 8px; background: #f8fafc;"
            )

        target_idx, source_idx = pair
        tokens, _, _, attentions, *_ = res
        isa_data = res[-2]
        boundaries = isa_data["sentence_boundaries_ids"]

        sub_att, tokens_combined, src_start = get_sentence_token_attention(
            attentions, tokens, target_idx, source_idx, boundaries
        )

        # Clean tokens for display in the heatmap
        toks_target = [t.replace("Ġ", "").replace("##", "") for t in tokens_combined[:src_start]]
        toks_source = [t.replace("Ġ", "").replace("##", "") for t in tokens_combined[src_start:]]

        # --- Highlight Selected Token Logic (Span Support) ---
        selected_indices = []
        try:
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        if not selected_indices:
            try: selected_idx = int(input.global_focus_token())
            except: selected_idx = -1
            if selected_idx != -1: selected_indices = [selected_idx]

        # Helper to get sentence range
        def get_range(idx):
            start = boundaries[idx]
            if idx < len(boundaries) - 1:
                end = boundaries[idx+1]
            else:
                end = len(tokens)
            return start, end

        t_start, t_end = get_range(target_idx)
        s_start, s_end = get_range(source_idx)

        # Identify which tokens in the heatmap correspond to selected global indices
        target_highlight_indices = []
        for s_idx in selected_indices:
            if t_start <= s_idx < t_end:
                 target_highlight_indices.append(s_idx - t_start)

        source_highlight_indices = []
        for s_idx in selected_indices:
            if s_start <= s_idx < s_end:
                 source_highlight_indices.append(s_idx - s_start)

        # Prepare Styled Ticks
        styled_target = []
        for i, tok in enumerate(toks_target):
            if i in target_highlight_indices:
                styled_target.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_target.append(tok)
                
        styled_source = []
        for i, tok in enumerate(toks_source):
            if i in source_highlight_indices:
                styled_source.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_source.append(tok)

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

        # Add Highlights
        for idx in target_highlight_indices:
             fig.add_shape(type="rect", 
                x0=-0.5, x1=len(toks_source)-0.5, 
                y0=idx-0.5, y1=idx+0.5,
                fillcolor="rgba(236, 72, 153, 0.15)", line=dict(color="#ec4899", width=1), layer="above"
             )
        for idx in source_highlight_indices:
             fig.add_shape(type="rect", 
                x0=idx-0.5, x1=idx+0.5, 
                y0=-0.5, y1=len(toks_target)-0.5,
                fillcolor="rgba(236, 72, 153, 0.15)", line=dict(color="#ec4899", width=1), layer="above"
             )

        fig.update_layout(
            title=dict(
                text=f"Token-to-Token — S{target_idx} ← S{source_idx} (Model A)",
                font=dict(size=14, color="#1e293b", family="Inter, system-ui, sans-serif")
            ),
            xaxis=dict(
                title=dict(
                    text="Source tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickmode='array',
                tickvals=list(range(len(toks_source))),
                ticktext=styled_source,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9"
            ),
            yaxis=dict(
                title=dict(
                    text="Target tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickmode='array',
                tickvals=list(range(len(toks_target))),
                ticktext=styled_target,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9",
                autorange="reversed" 
            ),
            height=420,
            width=440,
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
        res = get_active_result()
        score = 0.0
        if res and res[-2]:
            score = res[-2]["sentence_attention_matrix"][tx, sy]
        return ui.HTML(f"Sentence {tx} (target) ← Sentence {sy} (source) · ISA: <strong>{score:.4f}</strong>")

    @output(id="isa_token_view_B")
    @render.ui
    def isa_token_view_B():
        pair = isa_selected_pair_B()
        res = get_active_result("_B")

        if res is None or pair is None:
            return ui.div(
                ui.p("Select a point on the scatter plot to view token-to-token attention.", 
                     style="color: #94a3b8; font-size: 13px; font-style: italic;"),
                style="height: 350px; display: flex; align-items: center; justify-content: center; border: 1px dashed #e2e8f0; border-radius: 8px; background: #f8fafc;"
            )

        target_idx, source_idx = pair
        tokens, _, _, attentions, *_ = res
        isa_data = res[-2]
        boundaries = isa_data["sentence_boundaries_ids"]

        sub_att, tokens_combined, src_start = get_sentence_token_attention(
            attentions, tokens, target_idx, source_idx, boundaries
        )

        # Clean tokens for display in the heatmap
        toks_target = [t.replace("Ġ", "").replace("##", "") for t in tokens_combined[:src_start]]
        toks_source = [t.replace("Ġ", "").replace("##", "") for t in tokens_combined[src_start:]]

        # --- Highlight Selected Token Logic (Span Support) ---
        selected_indices = []
        try:
            val = input.global_selected_tokens_B()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        if not selected_indices:
            try: selected_idx = int(input.global_focus_token())
            except: selected_idx = -1
            if selected_idx != -1: selected_indices = [selected_idx]

        # Helper to get sentence range
        def get_range(idx):
            start = boundaries[idx]
            if idx < len(boundaries) - 1:
                end = boundaries[idx+1]
            else:
                end = len(tokens)
            return start, end

        t_start, t_end = get_range(target_idx)
        s_start, s_end = get_range(source_idx)

        # Identify which tokens in the heatmap correspond to selected global indices
        target_highlight_indices = []
        for s_idx in selected_indices:
            if t_start <= s_idx < t_end:
                 target_highlight_indices.append(s_idx - t_start)

        source_highlight_indices = []
        for s_idx in selected_indices:
            if s_start <= s_idx < s_end:
                 source_highlight_indices.append(s_idx - s_start)

        # Prepare Styled Ticks
        styled_target = []
        for i, tok in enumerate(toks_target):
            if i in target_highlight_indices:
                styled_target.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_target.append(tok)
                
        styled_source = []
        for i, tok in enumerate(toks_source):
            if i in source_highlight_indices:
                styled_source.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_source.append(tok)

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

        # Add Highlights
        for idx in target_highlight_indices:
             fig.add_shape(type="rect", 
                x0=-0.5, x1=len(toks_source)-0.5, 
                y0=idx-0.5, y1=idx+0.5,
                fillcolor="rgba(236, 72, 153, 0.15)", line=dict(color="#ec4899", width=1), layer="above"
             )
        for idx in source_highlight_indices:
             fig.add_shape(type="rect", 
                x0=idx-0.5, x1=idx+0.5, 
                y0=-0.5, y1=len(toks_target)-0.5,
                fillcolor="rgba(236, 72, 153, 0.15)", line=dict(color="#ec4899", width=1), layer="above"
             )

        fig.update_layout(
            title=dict(
                text=f"Token-to-Token — S{target_idx} ← S{source_idx} (Model B)",
                font=dict(size=14, color="#1e293b", family="Inter, system-ui, sans-serif")
            ),
            xaxis=dict(
                title=dict(
                    text="Source tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickmode='array',
                tickvals=list(range(len(toks_source))),
                ticktext=styled_source,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9"
            ),
            yaxis=dict(
                title=dict(
                    text="Target tokens",
                    font=dict(color="#475569", size=11)
                ),
                tickmode='array',
                tickvals=list(range(len(toks_target))),
                ticktext=styled_target,
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#f1f5f9",
                autorange="reversed" 
            ),
            height=420,
            width=440,
            autosize=True,
            margin=dict(l=60, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif")
        )
        return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="isa_token_view_plot_B", config={'displayModeBar': False}))

    @output(id="isa_detail_info_B")
    @render.ui
    def isa_detail_info_B():
        pair = isa_selected_pair_B()
        if pair is None:
            return ui.HTML("<em style='color:#94a3b8;'>Click a dot on the ISA chart.</em>")
        tx, sy = pair
        res = get_active_result("_B")
        score = 0.0
        if res and res[-2]:
            score = res[-2]["sentence_attention_matrix"][tx, sy]
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

    @reactive.effect
    @reactive.event(input.isa_scatter_click_B)
    def _handle_isa_click_B():
        click = input.isa_scatter_click_B()
        if not click or "x" not in click or "y" not in click:
            return
        # Plotly coordinates: x = source (B), y = target (A)
        source_idx = click["x"]
        target_idx = click["y"]
        isa_selected_pair_B.set((int(target_idx), int(source_idx)))



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
        res = get_active_result()
        if not res: return None
        tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0: return None
        
        # Check if we're in global mode
        use_global = global_metrics_mode.get() == "all"
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        # Get attention matrix - either specific layer/head or averaged across all
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            att = np.mean(att_layers, axis=(0, 1))
        else:
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
        
        # Determine if causal
        if hasattr(layer_block, "attention"): # BERT
             is_causal = False
             causal_desc = ""
        else: # GPT-2
             is_causal = True
             # Mask upper triangle (future tokens)
             # Use NaN to make them transparent/hidden in Plotly
             att = att.copy() # Ensure writable
             att[np.triu_indices_from(att, k=1)] = np.nan
             causal_desc = " <span style='color:#ef4444;font-weight:bold;'>(Causal Mask Applied)</span>"

        d_k = Q.shape[-1] // num_heads
        custom = np.empty((L, L, 5), dtype=object)
        for i in range(L):
            for j in range(L):
                # For causal models, skip masked cells (future tokens)
                if is_causal and j > i:
                    custom[i, j, :] = [None, None, None, None, None]
                    continue
                    
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

        # Clean tokens for display in the heatmap
        # Add index suffix ONLY to duplicate tokens (Plotly merges duplicate labels otherwise)
        base_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
        
        # Count occurrences of each token
        from collections import Counter
        token_counts = Counter(base_tokens)
        
        # Track occurrence number for each token
        occurrence_tracker = {}
        cleaned_tokens = []
        for t in base_tokens:
            if token_counts[t] > 1:
                # Duplicate token - add occurrence number
                occurrence_tracker[t] = occurrence_tracker.get(t, 0) + 1
                cleaned_tokens.append(f"{t}_{occurrence_tracker[t]}")
            else:
                # Unique token - no suffix
                cleaned_tokens.append(t)
        
        # Convert attention matrix: replace NaN with None for proper gap handling (no hover)
        att_list = []
        for i in range(len(att)):
            row = []
            for j in range(len(att[i])):
                if np.isnan(att[i, j]):
                    row.append(None)
                else:
                    row.append(float(att[i, j]))
            att_list.append(row)
        
        # Use go.Heatmap with list data for proper None/gap handling
        fig = go.Figure(data=go.Heatmap(
            z=att_list,
            x=cleaned_tokens,
            y=cleaned_tokens,
            colorscale=att_colorscale,
            zmin=0,
            zmax=1,
            customdata=custom,
            hovertemplate=hover,
            colorbar=dict(
                title=dict(
                    text="Attention",
                    font=dict(color="#64748b", size=11)
                ),
                tickfont=dict(color="#64748b", size=10)
            ),
            hoverongaps=False  # Don't show hover for None/gap cells
        ))
        
        # For causal models, add white rectangles to cover the forbidden cells (future tokens)
        if is_causal:
            n = len(cleaned_tokens)
            for i in range(n - 1):
                fig.add_shape(
                    type="rect",
                    x0=i + 0.5,
                    x1=n - 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor="white",
                    line=dict(width=0),
                    layer="above"
                )
        
        
        # --- Highlight Selected Token Logic (Span Support) ---
        selected_indices = []
        try:
            # Try to get span selection first
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback to single token if no span
        if not selected_indices:
            try: 
                single = int(input.global_focus_token())
                if single != -1:
                    selected_indices = [single]
            except: 
                pass
        
        # Prepare styled ticks
        styled_tokens = []
        for i, tok in enumerate(cleaned_tokens):
            if i in selected_indices:
                # Highlight selected token label (Pink & Bold)
                styled_tokens.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_tokens.append(tok)

        # Highlight Row/Column Shapes for ALL selected tokens
        for selected_idx in selected_indices:
            if selected_idx >= 0 and selected_idx < len(cleaned_tokens):
                 # Highlight Row
                 fig.add_shape(type="rect", 
                    x0=-0.5, x1=len(cleaned_tokens)-0.5, 
                    y0=selected_idx-0.5, y1=selected_idx+0.5,
                    fillcolor="rgba(236, 72, 153, 0.15)", 
                    line=dict(color="#ec4899", width=1),
                    layer="above"
                 )
                 # Highlight Column
                 fig.add_shape(type="rect", 
                    x0=selected_idx-0.5, x1=selected_idx+0.5, 
                    y0=-0.5, y1=len(cleaned_tokens)-0.5,
                    fillcolor="rgba(236, 72, 153, 0.15)", 
                    line=dict(color="#ec4899", width=1),
                    layer="above"
                 )

        # Dynamic title based on mode
        if use_global:
            title_text = "Attention Heatmap — Averaged (All Layers · Heads)"
        else:
            title_text = f"Attention Heatmap — Layer {layer_idx}, Head {head_idx}"
        
        fig.update_layout(
            xaxis_title="Key (attending to)", 
            yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40), 
            plot_bgcolor="#ffffff", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#64748b", family="Inter, system-ui, sans-serif"),
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(cleaned_tokens))),
                ticktext=styled_tokens,
                tickfont=dict(size=10), 
                title=dict(font=dict(size=11))
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(cleaned_tokens))),
                ticktext=styled_tokens,
                tickfont=dict(size=10), 
                title=dict(font=dict(size=11)),
                autorange='reversed'
            ),
            title=dict(
                text=title_text,
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color="#334155")
            )
        )
        return ui.div(
            {"class": "card", "style": "height: 100%; display: flex; flex-direction: column;"},
            ui.div(
                {"class": "header-simple"},
                ui.div(
                    {"class": "viz-header-with-info"},
                    ui.h4("Multi-Head Attention"),
                    ui.div(
                        {"class": "info-tooltip-wrapper"},
                        ui.span({"class": "info-tooltip-icon"}, "i"),
                        ui.div(
                            {"class": "info-tooltip-content"},
                            ui.HTML("""
                                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Multi-Head Attention</strong>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Shows how each token distributes its attention across all other tokens. Each cell (i,j) represents the attention weight from query token i to key token j.</p>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> <code style='font-size:10px;background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px'>Attention = softmax(QK<sup>T</sup>/√d<sub>k</sub>)V</code></p>
                                
                                <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                    <strong style='color:#8b5cf6;font-size:11px'>Color Scale (Blue):</strong>
                                    <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                        <span style='color:#1e40af'>● Dark blue: High weight</span>
                                        <span style='color:#3b82f6'>● Medium: Moderate</span>
                                        <span style='color:#93c5fd'>● Light: Low weight</span>
                                    </div>
                                </div>
                                
                                <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                    High attention ≠ importance
                                </p>
                            """)
                        )
                    )
                ),
            ),
            ui.div({"class": "viz-description", "style": "margin-top: 20px; flex-shrink: 0;"}, ui.HTML(f"Displays how much each token attends to every other token. Brighter cells indicate stronger attention weights. ⚠️ Note that high attention ≠ importance or influence.{causal_desc}")),
            ui.div(
                {"style": "flex: 1; display: flex; flex-direction: column; justify-content: center; width: 100%;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_map_plot"))
            ),
            # JavaScript to suppress hover on masked cells for causal models
            ui.HTML(f"""
            <script>
            (function() {{
                var isCausal = {'true' if is_causal else 'false'};
                if (!isCausal) return;
                
                function setupHoverSuppression() {{
                    var plot = document.getElementById('attention_map_plot');
                    if (!plot) {{
                        setTimeout(setupHoverSuppression, 100);
                        return;
                    }}
                    
                    plot.on('plotly_hover', function(data) {{
                        var pt = data.points[0];
                        var x = pt.pointIndex[1]; // column index
                        var y = pt.pointIndex[0]; // row index
                        
                        // If column > row, this is a masked cell - hide hover
                        if (x > y) {{
                            Plotly.Fx.hover(plot, []);
                        }}
                    }});
                }}
                setupHoverSuppression();
            }})();
            </script>
            """) if is_causal else ui.HTML("")
        )

    @output
    @render.ui
    def attention_flow():
        res = get_active_result()
        if not res: return None
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0: return None
        
        # Check if we're in global mode
        use_global = global_metrics_mode.get() == "all"
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        # Use global_selected_tokens for span support
        selected_indices = []
        try:
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass
        
        focus_indices = selected_indices if selected_indices else None # None means show all

        clean_tokens = tokens_data()
        
        # Get attention matrix - either specific layer/head or averaged across all
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            att = np.mean(att_layers, axis=(0, 1))
        else:
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
            # Clean token for display
            cleaned_tok = tok.replace("##", "").replace("Ġ", "")
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2
            
            is_selected = (i in focus_indices) if focus_indices else True

            # Dynamically adjust font size for many tokens
            if n_tokens > 30:
                font_size = 9 if is_selected else 8
            elif n_tokens > 20:
                font_size = 11 if is_selected else 10
            else:
                font_size = 13 if is_selected else 10

            text_color = color if (focus_indices and is_selected) else "#111827"
            weight = 'bold' if (focus_indices and is_selected) else 'normal'
            
            fig.add_trace(go.Scatter(x=[x_pos], y=[1.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight=weight), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight=weight), showlegend=False, hoverinfo='skip'))

        threshold = 0.04
        for i in range(n_tokens):
            for j in range(n_tokens):
                weight = att[i, j]
                if weight > threshold:
                    # Highlight line if source token is in selected set (or if no selection)
                    is_line_focused = (i in focus_indices) if focus_indices else True
                    
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
                    
                    # Clean tokens for hovertext
                    cleaned_token_i = tokens[i].replace("##", "").replace("Ġ", "")
                    cleaned_token_j = tokens[j].replace("##", "").replace("Ġ", "")
                    
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=line_color, width=line_width), opacity=line_opacity, showlegend=False, hoverinfo='text' if is_line_focused else 'skip', hovertext=f"<b>{cleaned_token_i} to {cleaned_token_j}</b><br>Attention: {weight:.4f}"))

        title_text = ""
        if focus_indices:
            token_spans = []
            for idx in focus_indices:
                focus_color = color_palette[idx % len(color_palette)]
                cleaned = tokens[idx].replace("##", "").replace("Ġ", "")
                token_spans.append(f"<span style='color:{focus_color}'>{cleaned}</span>")
            
            title_text += f" · <b>Focused: {', '.join(token_spans)}</b>"

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
                ui.div(
                    {"class": "viz-header-with-info"},
                    ui.h4("Attention Flow"),
                    ui.div(
                        {"class": "info-tooltip-wrapper"},
                        ui.span({"class": "info-tooltip-icon"}, "i"),
                        ui.div(
                            {"class": "info-tooltip-content"},
                            ui.HTML("""
                                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Flow</strong>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Sankey-style diagram showing attention distribution. Line width is proportional to attention weight α<sub>ij</sub> between token pairs.</p>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Threshold:</strong> Only connections with weight ≥ 0.04 are shown to reduce clutter.</p>
                                
                                <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                    <strong style='color:#8b5cf6;font-size:11px'>Line Width:</strong>
                                    <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                        <span style='color:#22c55e'>● Wide: High weight</span>
                                        <span style='color:#eab308'>● Medium: Moderate</span>
                                        <span style='color:#ef4444'>● Thin: Low weight</span>
                                    </div>
                                </div>
                                
                                <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                    Shows Query→Key, not information flow
                                </p>
                            """)
                        )
                    )
                )
            ),
            ui.div({"class": "viz-description"}, "Traces attention weight patterns between tokens. Thicker lines indicate stronger attention. ⚠️ This shows weight distribution, not actual information flow through the network."),
            ui.div(
                {"style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot"))
            )
        )

    @output
    @render.ui
    def _unused_duplicate_attention_flow_B():
        res = get_active_result("_B")
        if not res: return None
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0: return None

        # Check if we're in global mode
        use_global = global_metrics_mode.get() == "all"

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        # Use global_selected_tokens for span support (same as Model A)
        selected_indices = []
        try:
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass

        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass

        focus_indices = selected_indices if selected_indices else None

        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]

        # Get attention matrix - either specific layer/head or averaged across all
        if use_global:
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            att = np.mean(att_layers, axis=(0, 1))
        else:
            att = attentions[layer_idx][0, head_idx].cpu().numpy()

        n_tokens = len(tokens)
        color_palette = ['#ff5ca9', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#0ea5e9']

        fig = go.Figure()

        min_pixels_per_token = 50
        calculated_width = max(1000, n_tokens * min_pixels_per_token)
        block_width = 0.95 / n_tokens

        for i, tok in enumerate(tokens):
            cleaned_tok = tok.replace("##", "").replace("Ġ", "")
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2

            is_selected = (i in focus_indices) if focus_indices else True

            if n_tokens > 30: font_size = 9 if is_selected else 8
            elif n_tokens > 20: font_size = 11 if is_selected else 10
            else: font_size = 13 if is_selected else 10

            text_color = color if (focus_indices and is_selected) else "#111827"
            weight = 'bold' if (focus_indices and is_selected) else 'normal'

            fig.add_trace(go.Scatter(x=[x_pos], y=[1.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight=weight), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight=weight), showlegend=False, hoverinfo='skip'))

        threshold = 0.04
        for i in range(n_tokens):
            for j in range(n_tokens):
                weight = att[i, j]
                if weight > threshold:
                    is_line_focused = (i in focus_indices) if focus_indices else True
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

                    cleaned_token_i = tokens[i].replace("##", "").replace("Ġ", "")
                    cleaned_token_j = tokens[j].replace("##", "").replace("Ġ", "")

                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=line_color, width=line_width), opacity=line_opacity, showlegend=False, hoverinfo='text' if is_line_focused else 'skip', hovertext=f"<b>{cleaned_token_i} to {cleaned_token_j}</b><br>Attention: {weight:.4f}"))

        # Dynamic title based on mode
        if use_global:
            title_text = "Attention Flow — Averaged (All Layers · Heads)"
        else:
            title_text = f"Attention Flow — Layer {layer_idx}, Head {head_idx}"

        if focus_indices:
            for fidx in focus_indices[:3]:  # Show up to 3 focused tokens
                if fidx < len(tokens):
                    focus_color = color_palette[fidx % len(color_palette)]
                    cleaned_focus_token = tokens[fidx].replace("##", "").replace("Ġ", "")
                    title_text += f" · <b style='color:{focus_color}'>'{cleaned_focus_token}'</b>"

        fig.update_layout(
             title=title_text,
             xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.05, 1.05]),
             yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.25, 1.25]),
             plot_bgcolor='#ffffff',
             paper_bgcolor='#ffffff',
             font=dict(color='#111827'),
             height=400,
             width=calculated_width, 
             margin=dict(l=20, r=20, t=60, b=40),
             clickmode='event+select',
             hovermode='closest',
             dragmode=False,
             autosize=False
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
                        ui.div({"class": "select-compact"}, ui.input_select("flow_token_select_B", None, choices={"all": "All tokens", **{str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}}, selected=str(focus_idx) if focus_idx is not None else "all"))
                    )
                )
            ),
             ui.p("Visualizes how information flows between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            ui.div(
                {"style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot_B"))
            )
        )

    @output
    @render.ui
    def render_radar_view():
        res = get_active_result()
        if not res: return None
        if not res: return None
        # Removed dependency on input.radar_layer/head/mode to break infinite loop


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
             {"class": "card card-compact-height", "style": "height: 100%; display: flex; flex-direction: column;"},
             ui.div(
                {"class": "header-controls-stacked"},
                 ui.div(
                    {"class": "header-row-top"},
                    ui.div(
                        {"class": "viz-header-with-info"},
                        ui.h4("Head Specialization"),
                        ui.div(
                            {"class": "info-tooltip-wrapper"},
                            ui.span({"class": "info-tooltip-icon"}, "i"),
                            ui.div(
                                {"class": "info-tooltip-content"},
                                ui.HTML("""
                                    <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Head Specialization</strong>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Profiles each attention head across 7 dimensions using heuristic metrics from attention pattern analysis.</p>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Dimensions:</strong> Syntax, Semantics, Positional, Long-range, CLS Focus, Local, SEP Focus</p>
                                    
                                    <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                        <strong style='color:#8b5cf6;font-size:11px'>Score Interpretation:</strong>
                                        <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                            <span style='color:#22c55e'>● High (>0.7): Specialized</span>
                                            <span style='color:#eab308'>● Mid: Mixed role</span>
                                            <span style='color:#ef4444'>● Low (<0.3): Not focused</span>
                                        </div>
                                    </div>
                                    
                                    <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                        Labels are approximations, not verified roles
                                    </p>
                                """)
                            )
                        )
                    ),

                ),
                ui.div(
                    {"class": "header-row-bottom", "style": "margin-top: 4px;"},
                    ui.div({"class": "viz-description", "style": "margin: 0;"}, "Analyzes attention patterns to identify potential linguistic roles of each head. Each dimension represents a heuristic score for different attention behaviors. ⚠️ Labels like 'Syntax' and 'Semantics' are approximations based on attention patterns, not verified functional roles.")
                )
             ),
             ui.div(
                 {"style": "flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0;"},
                 ui.output_ui("radar_plot_internal")
             ),
             ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center; flex-shrink: 0;">
                    <strong style="color: #ff5ca9;">Specialization Dimensions</strong> — click any to see detailed explanation:<br>
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
    def radar_plot_internal():
        res = get_active_result()
        if not res: return None
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        # Use global mode from floating bar: "all" -> cluster, otherwise -> single
        use_global = global_metrics_mode.get() == "all"
        mode = "cluster" if use_global else "single"
        
        return head_specialization_radar(res, layer_idx, head_idx, mode, suffix="")

    # This function is now called directly from dashboard_content
    def head_specialization_radar(res, layer_idx, head_idx, mode, suffix=""):
        if not res: return None
        
        tokens, _, _, attentions, _, _, _, _, _, head_specialization, *_ = res
        
        # Safely extract head_clusters if available (new return value)
        head_clusters = []
        if len(res) >= 12:
            head_clusters = res[11]

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
        
        # Role Color Map for Clustering
        role_colors = {
            "Syntax": "#3b82f6",      # Blue
            "Semantics": "#8b5cf6",   # Purple
            "CLS Focus": "#64748b",   # Slate (Neutral)
            "Punctuation": "#f59e0b", # Amber
            "Entities": "#ec4899",    # Pink
            "Long-range": "#10b981",  # Emerald
            "Self-attention": "#ef4444" # Red
        }

        fig = go.Figure()

        if mode == "cluster":
            # Algorithmic Cluster Map (t-SNE + K-Means)
            if not head_clusters:
                return ui.HTML("<div style='text-align:center; padding:20px; color:#94a3b8; font-size:12px;'>" 
                             "Clustering data not found.<br>Please click <b>Generate All</b> to re-compute." 
                             "</div>")

            x_vals = []
            y_vals = []
            colors_list = []
            sizes = []
            hover_texts = []
            
            # Robust colors for up to 15 clusters
            cluster_colors = [
                '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', 
                '#06b6d4', '#84cc16', '#6366f1', '#d946ef', '#f97316', '#14b8a6',
                '#a855f7', '#fbbf24', '#f43f5e'
            ]
            
            for item in head_clusters:
                c_id = item['cluster']
                color = cluster_colors[c_id % len(cluster_colors)]
                
                x_vals.append(item['x'])
                y_vals.append(item['y'])
                colors_list.append(color)
                
                # Find dominant metric for context
                m = item['metrics']
                dom_role = max(m, key=m.get)
                score = m[dom_role]
                
                # Size by specialization confidence (score)
                sizes.append(6 + (score * 8))
                
                c_name = item.get('cluster_name', f"Cluster {c_id}")
                
                hover_texts.append(
                    f"<b>L{item['layer']}·H{item['head']}</b><br>" +
                    f"<b>{c_name}</b><br>" + 
                    f"Dominant: {dom_role} ({score:.2f})"
                )

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=colors_list,
                    size=sizes,
                    opacity=0.9,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add Legend for Clusters
            unique_clusters = sorted(list(set(c['cluster'] for c in head_clusters)))
            for c_id in unique_clusters:
                color = cluster_colors[c_id % len(cluster_colors)]
                
                # Find name for this cluster
                example_item = next((x for x in head_clusters if x['cluster'] == c_id), None)
                c_name = example_item.get('cluster_name', f"Cluster {c_id}") if example_item else f"Cluster {c_id}"
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color=color, size=10, line=dict(width=1, color='white')),
                    name=c_name,
                    showlegend=True
                ))

            title_text = "Head Specialization Clusters"
            
            fig.update_layout(
                xaxis=dict(title="t-SNE Dimension 1", showgrid=True, gridcolor='#f1f5f9', zeroline=False, showticklabels=False),
                yaxis=dict(title="t-SNE Dimension 2", showgrid=True, gridcolor='#f1f5f9', zeroline=False, showticklabels=False),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40)
            )

        elif mode == "single":
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
            
            title_text = f'Radar — Layer {layer_idx}, Head {head_idx}'
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
            
            title_text = f'Radar — Layer {layer_idx} (All Heads)'
        
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
                y=0.95,
                x=0.5,
                xanchor='center'
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, system-ui, sans-serif"),
            height=300,
            width=350,
            autosize=False
        )
        
        plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, div_id=f"radar_plot{suffix}", config={'displayModeBar': False})
        return ui.HTML(f'<div style="display: flex; justify-content: center; align-items: center; width: 100%; height: 100%;">{plot_html}</div>')


    # This function replaces the previous @output @render.ui def influence_tree():
    def get_influence_tree_ui(res, root_idx=0, layer_idx=0, head_idx=0, suffix="", use_global=False, max_depth=3, top_k=3):
        if not res:
            return ui.HTML("""
                <div style='padding: 20px; text-align: center;'>
                    <p style='font-size:11px;color:#9ca3af;'>Generate attention data to view the influence tree.</p>
                </div>
            """)
        
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0:
            return ui.HTML("<p style='font-size:11px;color:#6b7280;'>No attention data available.</p>")
        
        # Ensure valid indices
        root_idx = max(0, min(root_idx, len(tokens) - 1))
        
        try:
            tree_data = get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, max_depth)
            
            if tree_data is None:
                return ui.HTML("<p style='font-size:11px;color:#6b7280;'>Unable to generate tree.</p>")
            
            # Convert to JSON
            tree_json = json.dumps(tree_data)
        except Exception as e:
            return ui.HTML(f"<p style='font-size:11px;color:#ef4444;'>Error generating tree: {str(e)}</p>")
        
        html = f"""
    <div class="influence-tree-wrapper" style="height: 100%; display: flex; flex-direction: column; position: relative; overflow: auto;">
        <div id="tree-viz-container{suffix}" class="tree-viz-container" style="height: 100%; min-height: 400px; width: 100%; overflow: auto; text-align: center; display: block;"></div>
    </div>
        <script>
                (function() {{
                    function tryRender() {{
                        if (typeof d3 !== 'undefined' && typeof renderInfluenceTree !== 'undefined') {{
                            try {{
                                renderInfluenceTree({tree_json}, 'tree-viz-container{suffix}');
                            }} catch(e) {{
                                console.error('Error rendering tree:', e);
                                document.getElementById('tree-viz-container{suffix}').innerHTML = 
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

    # -------------------------------------------------------------------------
    # RENDERERS
    # -------------------------------------------------------------------------

    @output
    @render.ui
    def render_tree_view():
        res = get_active_result()
        if not res: return None
        
        # Check if we're in global mode
        use_global = global_metrics_mode.get() == "all"
        
        # Use global inputs for layer and head
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        # Use global_selected_tokens for span support - Tree View uses only the FIRST selected token as root
        selected_indices = []
        try:
            val = input.global_selected_tokens()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass
        
        root_idx = selected_indices[0] if selected_indices else 0
        
        tokens = res[0]
        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]

        try:
            top_k_val = int(input.global_topk())
        except:
            top_k_val = 3
            
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            ui.div(
                {"class": "header-controls-stacked"},
                ui.div(
                    {"class": "header-row-top"},
                    ui.div(
                        {"class": "viz-header-with-info"},
                        ui.h4("Attention Dependency Tree"),
                        ui.div(
                            {"class": "info-tooltip-wrapper"},
                            ui.span({"class": "info-tooltip-icon"}, "i"),
                            ui.div(
                                {"class": "info-tooltip-content"},
                                ui.HTML("""
                                    <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Dependency Tree</strong>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Hierarchical view of attention dependencies starting from a root token. Shows which tokens the root attends to (depth 1) and their dependencies (depth 2).</p>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Construction:</strong> Edges connect to top-k most attended tokens at each level recursively.</p>
                                    
                                    <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                        <strong style='color:#8b5cf6;font-size:11px'>Key Insights:</strong>
                                        <ul style='margin:6px 0 0 0;padding-left:14px;font-size:10px'>
                                            <li>Identify which tokens a word "looks at"</li>
                                            <li>Discover multi-hop attention chains</li>
                                            <li>Find syntactic or semantic clusters</li>
                                        </ul>
                                    </div>
                                    
                                    <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                        Not a syntactic parse tree
                                    </p>
                                """)
                            )
                        )
                    )
                ),
                ui.div(
                    {"class": "header-row-bottom", "style": "margin-top: 4px;"},
                    ui.div({"class": "viz-description", "style": "margin: 0;"}, "Visualizes how the root token attends to other tokens (Depth 1), and how those tokens attend to others (Depth 2). Click nodes to collapse/expand. Thicker edges = stronger influence. ⚠️ This represents attention patterns, not syntactic parse structure.")
                )
            ),
            get_influence_tree_ui(res, root_idx, layer_idx, head_idx, suffix="", use_global=use_global, max_depth=top_k_val, top_k=top_k_val)
        )

    # -------------------------------------------------------------------------
    # MODEL B RENDERERS (For Comparison Mode)
    # -------------------------------------------------------------------------

    @output
    @render.ui
    def render_embedding_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        return get_embedding_table(res)

    @output
    @render.ui
    def render_segment_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Segment Embeddings"), 
            ui.p("Segment ID (Sentence A/B)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_segment_embedding_view(res)
        )

    @output
    @render.ui
    def render_posenc_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Positional Embeddings"), 
            ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_posenc_table(res)
        )

    @output
    @render.ui
    def render_sum_layernorm_B():
        res = get_active_result("_B")
        if not res:
            return None
        _, _, _, _, _, _, _, encoder_model, *_ = res
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Sum & Layer Normalization"),
            ui.p("Sum of embeddings + Pre-Norm", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_sum_layernorm_view(res, encoder_model)
        )

    @output
    @render.ui
    def render_qkv_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-simple"},
                ui.h4("Q/K/V Projections")
            ),
            ui.p("Projects input to Query, Key, Value vectors.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_qkv_table(res, layer_idx)
        )

    @output
    @render.ui
    def render_scaled_attention_B():
        res = get_active_result("_B")
        if not res:
            return None
            
        # Use global_selected_tokens_B for span support
        selected_indices = []
        try:
            val = input.global_selected_tokens_B()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
             # Legacy fallback removed for B to ensure independence or default to [0]
             pass
        
        focus_indices = selected_indices if selected_indices else [0]
        
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        try: top_k = int(input.global_topk())
        except: top_k = 3
        
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-simple"},
                ui.h4("Scaled Dot-Product Attention")
            ),
            ui.p("Calculates attention scores between tokens.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=top_k)
        )

    @output
    @render.ui
    def render_ffn_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer = int(input.global_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"}, 
            ui.h4("Feed-Forward Network"), 
            ui.p("Expansion -> Activation -> Projection", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_ffn_view(res, layer)
        )

    @output
    @render.ui
    def render_add_norm_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer = int(input.global_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Add & Norm"),
            ui.p("Residual Connection + Layer Normalization", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_add_norm_view(res, layer)
        )

    @output
    @render.ui
    def render_add_norm_post_ffn_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer = int(input.global_layer())
        except: layer = 0
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Add & Norm (Post-FFN)"),
            ui.p("Residual Connection + Layer Normalization", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_add_norm_post_ffn_view(res, layer)
        )

    @output
    @render.ui
    def render_layer_output_B():
        res = get_active_result("_B")
        if not res:
            return None
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
    def render_mlm_predictions_B():
        res = get_active_result("_B")
        if not res:
            return None
        
        try:
            is_compare_prompts = input.compare_prompts_mode()
        except:
            is_compare_prompts = False

        try:
            if is_compare_prompts:
                 model_family = input.model_family()
            else:
                 model_family = input.model_family_B()
        except:
            model_family = "bert"
            
        try: use_mlm = input.use_mlm()
        except: use_mlm = False
        
        if model_family == "gpt2": use_mlm = True
            
        try: text = input.text_input()
        except: text = ""
        
        is_bert = model_family != "gpt2" # safer check
        
        try: top_k = int(input.global_topk())
        except: top_k = 3

        if is_bert:
            # BERT logic: Check local state
            use_mlm_B = show_mlm_B.get()
            
            title = "Masked Token Predictions (MLM)"
            desc = "Pseudo-Likelihood: Each token is individually masked and predicted using the bidirectional context."
            
            if not use_mlm_B:
                 return ui.div(
                    {"class": "card", "style": "height: 100%; display: flex; flex-direction: column; justify-content: space-between;"},
                    ui.div(
                        ui.h4(title),
                        ui.p(desc, style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    ),
                    ui.div(
                         {"style": "flex-grow: 1; display: flex; align-items: center; justify-content: center; padding: 20px;"},
                         ui.input_action_button("trigger_mlm_B", "Generate Predictions", class_="btn-primary")
                    )
                )
        else:
            use_mlm_B = True
            title = "Next Token Predictions (Causal)"
            desc = "Predicting the probability of the next token appearing after the sequence."

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4(title),
            ui.p(desc, style="font-size:11px; color:#6b7280; margin-bottom:8px; min-height: 32px;"),
            get_output_probabilities(res, use_mlm_B, text, suffix="_B", top_k=top_k)
        )

    @output
    @render.ui
    def render_radar_view_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        try: mode = input.radar_mode_B()
        except: mode = "single"
        
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%; display: flex; flex-direction: column;"},
            ui.div(
                {"class": "header-controls-stacked"},
                ui.div(
                    {"class": "header-row-top"},
                    ui.div(
                        {"class": "viz-header-with-info"},
                        ui.h4("Head Specialization"),
                        ui.div(
                            {"class": "info-tooltip-wrapper"},
                            ui.span({"class": "info-tooltip-icon"}, "i"),
                            ui.div(
                                {"class": "info-tooltip-content"},
                                ui.HTML("""
                                    <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Head Specialization</strong>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Profiles each attention head across 7 dimensions using heuristic metrics from attention pattern analysis.</p>
                                    
                                    <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Dimensions:</strong> Syntax, Semantics, Positional, Long-range, CLS Focus, Local, SEP Focus</p>
                                    
                                    <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                        <strong style='color:#8b5cf6;font-size:11px'>Score Interpretation:</strong>
                                        <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                            <span style='color:#22c55e'>● High (>0.7): Specialized</span>
                                            <span style='color:#eab308'>● Mid: Mixed role</span>
                                            <span style='color:#ef4444'>● Low (<0.3): Not focused</span>
                                        </div>
                                    </div>
                                    
                                    <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                        Labels are approximations, not verified roles
                                    </p>
                                """)
                            )
                        )
                    )
                ),
                ui.div(
                    {"class": "header-row-bottom", "style": "margin-top: 4px;"},
                    ui.div({"class": "viz-description", "style": "margin: 0;"}, "Analyzes attention patterns to identify potential linguistic roles of each head. Each dimension represents a heuristic score for different attention behaviors. ⚠️ Labels like 'Syntax' and 'Semantics' are approximations based on attention patterns, not verified functional roles.")
                )
             ),
            head_specialization_radar(res, layer_idx, head_idx, mode, suffix="_B"),
            ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; margin-top: 16px; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center;">
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
    def render_tree_view_B():
        res = get_active_result("_B")
        if not res:
            return None

        # Use global_selected_tokens_B for span support - Tree View uses only the FIRST selected token as root
        selected_indices = []
        try:
            val = input.global_selected_tokens_B()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass
        
        
        # Safety: Filter indices to be within bounds
        tokens_B = res[0]
        if selected_indices:
            selected_indices = [idx for idx in selected_indices if idx < len(tokens_B)]
            
        root_idx = selected_indices[0] if selected_indices else 0

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        

        try: top_k = int(input.global_topk())
        except: top_k = 3

        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens_B]
        choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            ui.div(
                {"class": "header-controls-stacked"},
                    ui.div(
                        {"class": "header-simple"},
                        ui.div(
                            {"class": "viz-header-with-info"},
                            ui.h4("Attention Dependency Tree"),
                            ui.div(
                                {"class": "info-tooltip-wrapper"},
                                ui.span({"class": "info-tooltip-icon"}, "i"),
                                ui.div(
                                    {"class": "info-tooltip-content"},
                                    ui.HTML("""
                                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Dependency Tree</strong>
                                        
                                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Hierarchical view of attention dependencies starting from a root token. Shows which tokens the root attends to (depth 1) and their dependencies (depth 2).</p>
                                        
                                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Construction:</strong> Edges connect to top-k most attended tokens at each level recursively.</p>
                                        
                                        <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                            <strong style='color:#8b5cf6;font-size:11px'>Key Insights:</strong>
                                            <ul style='margin:6px 0 0 0;padding-left:14px;font-size:10px'>
                                                <li>Identify which tokens a word "looks at"</li>
                                                <li>Discover multi-hop attention chains</li>
                                                <li>Find syntactic or semantic clusters</li>
                                            </ul>
                                        </div>
                                        
                                        <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                            Not a syntactic parse tree
                                        </p>
                                    """)
                                )
                            )
                        )
                    ),
                ui.div(
                    {"class": "header-row-bottom", "style": "margin-top: 4px;"},
                    ui.div({"class": "viz-description", "style": "margin: 0;"}, "Visualizes how the root token attends to other tokens (Depth 1), and how those tokens attend to others (Depth 2). Click nodes to collapse/expand. Thicker edges = stronger influence. ⚠️ This represents attention patterns, not syntactic parse structure.")
                )
            ),
            get_influence_tree_ui(res, root_idx, layer_idx, head_idx, suffix="_B", top_k=top_k, max_depth=top_k)
        )

    @output
    @render.ui
    def render_global_metrics_B():
        res = get_active_result("_B")
        if not res:
            return None
        
        # Check if we should use all layers/heads or specific selection
        use_all = global_metrics_mode.get() == "all"
        
        if use_all:
            layer_idx = None
            head_idx = None
            subtitle = "All Layers · All Heads"
        else:
            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0
            subtitle = f"Layer {layer_idx} · Head {head_idx}"

        return ui.div(
            {"class": "card"}, 
            ui.div(
                {"style": "display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;"},
                ui.h4("Global Attention Metrics", style="margin: 0;"),
                ui.span(subtitle, style="font-size: 11px; color: #94a3b8; font-weight: 500;")
            ),
            get_metrics_display(res, layer_idx=layer_idx, head_idx=head_idx)
        )

    @output
    @render.ui
    def attention_map_B():
        res = get_active_result("_B")
        if not res:
            return None
        tokens, _, _, attentions, _, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0:
            return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        
        # Clean tokens for display in the heatmap
        # Add index suffix ONLY to duplicate tokens (Plotly merges duplicate labels otherwise)
        base_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
        
        # Count occurrences of each token
        from collections import Counter
        token_counts = Counter(base_tokens)
        
        # Track occurrence number for each token
        occurrence_tracker = {}
        cleaned_tokens = []
        for t in base_tokens:
            if token_counts[t] > 1:
                # Duplicate token - add occurrence number
                occurrence_tracker[t] = occurrence_tracker.get(t, 0) + 1
                cleaned_tokens.append(f"{t}_{occurrence_tracker[t]}")
            else:
                # Unique token - no suffix
                cleaned_tokens.append(t)
        
        # Custom colorscale for attention map (White -> Blue -> Dark Blue)
        att_colorscale = [
            [0.0, '#ffffff'],
            [0.1, '#f0f9ff'],
            [0.3, '#bae6fd'],
            [0.6, '#3b82f6'],
            [1.0, '#1e3a8a']
        ]

        # Clean tokens for display in the heatmap
        base_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
        
        # Count occurrences of each token
        from collections import Counter
        token_counts = Counter(base_tokens)
        
        # Track occurrence number for each token
        occurrence_tracker = {}
        cleaned_tokens = []
        for t in base_tokens:
            if token_counts[t] > 1:
                # Duplicate token - add occurrence number
                occurrence_tracker[t] = occurrence_tracker.get(t, 0) + 1
                cleaned_tokens.append(f"{t}_{occurrence_tracker[t]}")
            else:
                # Unique token - no suffix
                cleaned_tokens.append(t)
        
        # Convert attention matrix: replace NaN with None for proper gap handling (no hover)
        att_list = []
        rows, cols = att.shape
        for i in range(rows):
            row = []
            for j in range(cols):
                if np.isnan(att[i, j]):
                    row.append(None)
                else:
                    row.append(float(att[i, j]))
            att_list.append(row)
        
        # Use go.Heatmap with list data for proper None/gap handling
        fig = go.Figure(data=go.Heatmap(
            z=att_list,
            x=cleaned_tokens,
            y=cleaned_tokens,
            colorscale=att_colorscale,
            zmin=0,
            zmax=1,
            hoverongaps=False,  # Don't show hover for None/gap cells
            colorbar=dict(
                title=dict(
                    text="Attention",
                    font=dict(color="#64748b", size=11)
                ),
                tickfont=dict(color="#64748b", size=10)
            ),
        ))

        # --- Highlight Selected Token Logic (Span Support) ---
        selected_indices = []
        try:
            val = input.global_selected_tokens_B()
            if val:
                selected_indices = json.loads(val)
        except:
            pass

        if not selected_indices:
            try:
                single = int(input.global_focus_token())
                if single != -1:
                    selected_indices = [single]
            except:
                pass

        # Prepare styled ticks
        styled_tokens = []
        for i, tok in enumerate(cleaned_tokens):
            if i in selected_indices:
                styled_tokens.append(f"<span style='color:#ec4899; font-weight:bold; font-size:12px'>{tok}</span>")
            else:
                styled_tokens.append(tok)

        # Highlight Row/Column Shapes for ALL selected tokens
        for selected_idx in selected_indices:
            if selected_idx >= 0 and selected_idx < len(cleaned_tokens):
                 fig.add_shape(type="rect",
                    x0=-0.5, x1=len(cleaned_tokens)-0.5,
                    y0=selected_idx-0.5, y1=selected_idx+0.5,
                    fillcolor="rgba(236, 72, 153, 0.15)",
                    line=dict(color="#ec4899", width=1),
                    layer="above"
                 )
                 fig.add_shape(type="rect",
                    x0=selected_idx-0.5, x1=selected_idx+0.5,
                    y0=-0.5, y1=len(cleaned_tokens)-0.5,
                    fillcolor="rgba(236, 72, 153, 0.15)",
                    line=dict(color="#ec4899", width=1),
                    layer="above"
                 )

        # Consistent Layout
        fig.update_layout(
            xaxis_title="Key (attending to)",
            yaxis_title="Query (attending from)",
            margin=dict(l=40, r=10, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", family="Inter"),
            xaxis=dict(
                tickfont=dict(size=10),
                ticktext=styled_tokens,
                tickvals=list(range(len(cleaned_tokens))),
                title=dict(font=dict(size=11))
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                ticktext=styled_tokens,
                tickvals=list(range(len(cleaned_tokens))),
                autorange='reversed',
                title=dict(font=dict(size=11))
            ),
            title=dict(
                text=f"Attention Heatmap — Layer {layer_idx}, Head {head_idx}",
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color="#334155")
            )
        )
        
        return ui.div(
            {"class": "card", "style": "height: 100%; display: flex; flex-direction: column;"},
            ui.div(
                {"class": "header-simple"},
                ui.div(
                    {"class": "viz-header-with-info"},
                    ui.h4("Multi-Head Attention"),
                    ui.div(
                        {"class": "info-tooltip-wrapper"},
                        ui.span({"class": "info-tooltip-icon"}, "i"),
                        ui.div(
                            {"class": "info-tooltip-content"},
                            ui.HTML("""
                                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Multi-Head Attention</strong>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Shows how each token distributes its attention across all other tokens. Each cell (i,j) represents the attention weight from query token i to key token j.</p>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> <code style='font-size:10px;background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px'>Attention = softmax(QK<sup>T</sup>/√d<sub>k</sub>)V</code></p>
                                
                                <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                    <strong style='color:#8b5cf6;font-size:11px'>Color Scale (Blue):</strong>
                                    <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                        <span style='color:#1e40af'>● Dark blue: High weight</span>
                                        <span style='color:#3b82f6'>● Medium: Moderate</span>
                                        <span style='color:#93c5fd'>● Light: Low weight</span>
                                    </div>
                                </div>
                                
                                <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                    High attention ≠ importance
                                </p>
                            """)
                        )
                    )
                ),
            ),
            ui.div({"class": "viz-description", "style": "margin-top: 20px; flex-shrink: 0;"}, "Displays how much each token attends to every other token. Darker cells indicate stronger attention weights. ⚠️ Note that high attention ≠ importance or influence."),
            ui.div(
                {"style": "flex: 1; display: flex; flex-direction: column; justify-content: center; width: 100%; height: 500px;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_heatmap_B")),
                ui.HTML("""<script>
                    (function() {
                        var attempts = 0;
                        function checkAndResize() {
                            var p = document.getElementById('attention_heatmap_B');
                            if(p && p.data) {
                                Plotly.Plots.resize(p);
                                console.log("Resized Heatmap B");
                            }
                            attempts++;
                            if (attempts < 10) setTimeout(checkAndResize, 200);
                        }
                        // Start polling for visibility/readiness
                        setTimeout(checkAndResize, 100);
                    })();
                </script>""")
            )
        )




    @output
    @render.ui
    def attention_flow_B():
        res = get_active_result("_B")
        if not res:
            return None
        tokens, _, _, attentions, *_ = res
        if attentions is None or len(attentions) == 0:
            return None

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        # Use global_selected_tokens_B for span support
        selected_indices = []
        try:
            val = input.global_selected_tokens_B()
            if val:
                selected_indices = json.loads(val)
        except:
            pass
            
        # Fallback
        if not selected_indices:
            try:
                global_token = int(input.global_focus_token())
                if global_token >= 0:
                    selected_indices = [global_token]
            except:
                pass
        
        # Safety: Filter indices to be within bounds of current tokens (crucial for Compare Prompts)
        if selected_indices:
            selected_indices = [idx for idx in selected_indices if idx < len(tokens)]
        
        focus_indices = selected_indices if selected_indices else None # None means show all

        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]
        
        att = attentions[layer_idx][0, head_idx].cpu().numpy()
        n_tokens = len(tokens)
        color_palette = ['#ff5ca9', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#6366f1', '#14b8a6', '#f43f5e', '#a855f7', '#0ea5e9']

        fig = go.Figure()

        min_pixels_per_token = 50
        calculated_width = max(1000, n_tokens * min_pixels_per_token)
        block_width = 0.95 / n_tokens 

        for i, tok in enumerate(tokens):
            cleaned_tok = tok.replace("##", "").replace("Ġ", "")
            color = color_palette[i % len(color_palette)]
            x_pos = i / n_tokens + block_width / 2
            is_selected = (i in focus_indices) if focus_indices else True

            if n_tokens > 30: font_size = 9 if is_selected else 8
            elif n_tokens > 20: font_size = 11 if is_selected else 10
            else: font_size = 13 if is_selected else 10

            text_color = color if (focus_indices and is_selected) else "#111827"
            fig.add_trace(go.Scatter(x=[x_pos], y=[1.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[x_pos], y=[-0.05], mode='text', text=cleaned_tok, textfont=dict(size=font_size, color=text_color, family='monospace', weight='bold'), showlegend=False, hoverinfo='skip'))

        threshold = 0.04
        for i in range(n_tokens):
            for j in range(n_tokens):
                weight = att[i, j]
                if weight > threshold:
                    is_line_focused = (i in focus_indices) if focus_indices else True
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
                    
                    cleaned_token_i = tokens[i].replace("##", "").replace("Ġ", "")
                    cleaned_token_j = tokens[j].replace("##", "").replace("Ġ", "")
                    
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=line_color, width=line_width), opacity=line_opacity, showlegend=False, hoverinfo='text' if is_line_focused else 'skip', hovertext=f"<b>{cleaned_token_i} to {cleaned_token_j}</b><br>Attention: {weight:.4f}"))

        title_text = ""
        if focus_indices:
            token_spans = []
            for idx in focus_indices:
                focus_color = color_palette[idx % len(color_palette)]
                cleaned = tokens[idx].replace("##", "").replace("Ġ", "")
                token_spans.append(f"<span style='color:{focus_color}'>{cleaned}</span>")
            
            title_text += f" · <b>Focused: {', '.join(token_spans)}</b>"

        fig.update_layout(
             title=title_text,
             xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.05, 1.05]),
             yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.25, 1.25]),
             plot_bgcolor='#ffffff',
             paper_bgcolor='#ffffff',
             font=dict(color='#111827'),
             height=500, # Matches Model A
             width=calculated_width, 
             margin=dict(l=20, r=20, t=60, b=40),
             clickmode='event+select',
             hovermode='closest',
             dragmode=False,
             autosize=False
         )
         
        # Configure the figure to prevent responsive resizing (matches A)
        fig.update_layout(
            modebar=dict(orientation='v')
        )

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-with-selectors"},
                ui.div(
                    {"class": "viz-header-with-info"},
                    ui.h4("Attention Flow"),
                    ui.div(
                        {"class": "info-tooltip-wrapper"},
                        ui.span({"class": "info-tooltip-icon"}, "i"),
                        ui.div(
                            {"class": "info-tooltip-content"},
                            ui.HTML("""
                                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Flow</strong>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Sankey-style diagram showing attention distribution. Line width is proportional to attention weight α<sub>ij</sub> between token pairs.</p>
                                
                                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Threshold:</strong> Only connections with weight ≥ 0.04 are shown to reduce clutter.</p>
                                
                                <div style='background:rgba(255,255,255,0.05);border-radius:6px;padding:10px;margin-top:8px'>
                                    <strong style='color:#8b5cf6;font-size:11px'>Line Width:</strong>
                                    <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:11px'>
                                        <span style='color:#22c55e'>● Wide: High weight</span>
                                        <span style='color:#eab308'>● Medium: Moderate</span>
                                        <span style='color:#ef4444'>● Thin: Low weight</span>
                                    </div>
                                </div>
                                
                                <p style='font-size:10px;color:#64748b;margin:10px 0 0 0;text-align:center;border-top:1px solid rgba(255,255,255,0.1);padding-top:8px'>
                                    Shows Query→Key, not information flow
                                </p>
                            """)
                        )
                    )
                ),
                ui.div(
                    {"class": "selection-boxes-container", "style": "visibility: hidden; pointer-events: none;"},
                    ui.tags.span("Filter:", style="font-size:12px; font-weight:600; color:#64748b; margin-right: 4px;"),
                    ui.div(
                        {"class": "selection-box"},
                        ui.div({"class": "select-compact", "style": "height: 24px; width: 100px;"}) # Dummy spacer
                    )
                )
            ),
            ui.div({"class": "viz-description", "style": "margin-top: -5px;"}, "Traces attention weight patterns between tokens. Thicker lines indicate stronger attention. ⚠️ This shows weight distribution, not actual information flow through the network."),
            ui.div(
                {"style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot_B"))
            )
        )

__all__ = ["server"]
