import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import re
import json
from datetime import datetime
import os
from pathlib import Path

# Create default download directories
for d in ["sessions", "csv", "images"]:
    Path(d).mkdir(exist_ok=True)

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
from ..isa import compute_isa
from ..isa import get_sentence_token_attention
from .baselines import compute_baselines

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

    # Inject JavaScript for scroll position preservation
    scroll_preservation_js = """
    (function() {
        // Save scroll position before DOM updates
        let savedScrollTop = 0;
        let pendingRestore = false;

        // Save current scroll position
        function saveScroll() {
            savedScrollTop = window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
            pendingRestore = true;
        }

        // Restore scroll position after a short delay (after DOM update)
        function restoreScroll() {
            if (!pendingRestore) return;
            setTimeout(function() {
                if (savedScrollTop > 0) {
                    window.scrollTo({top: savedScrollTop, behavior: 'instant'});
                }
                pendingRestore = false;
            }, 50);
        }

        // Listen for Shiny input changes on layer/head/view controls
        $(document).on('shiny:inputchanged', function(event) {
            if (event.name === 'global_layer' || event.name === 'global_head' ||
                event.name === 'global_topk' || event.name === 'global_norm' ||
                event.name === 'trigger_global_view' || event.name === 'global_rollout_layers') {
                saveScroll();
            }
        });

        // Restore scroll after Shiny value/output updates
        $(document).on('shiny:value', function(event) {
            if (pendingRestore) restoreScroll();
        });

        // Also restore on idle (backup)
        $(document).on('shiny:idle', function(event) {
            if (pendingRestore) restoreScroll();
        });
    })();
    """
    ui.insert_ui(selector="body", where="beforeEnd", ui=ui.tags.script(scroll_preservation_js))
    # Store the text used for generation to avoid reading input.text_input() in renderers
    cached_text_A = reactive.value("")
    cached_text_B = reactive.value("")
    
    # Snapshots of sidebar state - Updated ONLY on 'Generate All'
    active_compare_models = reactive.Value(False)
    active_compare_prompts = reactive.Value(False)
    active_view_mode = reactive.Value("basic")
    
    # Session Load Force Flags (to override UI lag)
    session_force_compare_mode = reactive.Value(False)
    session_force_compare_prompts_mode = reactive.Value(False)
    
    isa_selected_pair = reactive.Value(None)
    isa_selected_pair = reactive.Value(None)
    isa_selected_pair_B = reactive.Value(None) # For comparison
    
    # Baseline Logic
    baseline_stats = reactive.Value(None)
    current_baseline_model = reactive.Value(None)

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

    # --- Filename Generation Helper ---
    def generate_export_filename(section, ext="csv", is_b=False, incl_timestamp=True, data_type=None):
        """
        Generate export filename with format: <section>_<data>_<model>_<modelA/B>_<promptA/B>_<timestamp>.ext

        Args:
            section: The visualization section (e.g., 'attention_tree', 'isa', 'qkv')
            ext: File extension (csv, json, png)
            is_b: Whether this is Model B / Prompt B
            incl_timestamp: Include timestamp in filename
            data_type: Optional data type descriptor (e.g., 'topk', 'all_layers')
        """
        # Get model name
        try:
            if is_b:
                model = input.model_family_B() or "model"
            else:
                model = input.model_family() or "model"
        except:
            model = "model"

        # Check modes
        try: cm = input.compare_mode()
        except: cm = False
        try: cpm = input.compare_prompts_mode()
        except: cpm = False

        # Build filename parts
        parts = [section]

        if data_type:
            parts.append(data_type)

        parts.append(model)

        # Add model label if compare_mode
        if cm:
            parts.append("ModelB" if is_b else "ModelA")

        # Add prompt label if compare_prompts_mode
        if cpm:
            parts.append("PromptB" if is_b else "PromptA")

        if incl_timestamp:
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            parts.append(ts)

        filename = f"{'_'.join(parts)}.{ext}"
        return filename

    # --- Auto-save download helper ---
    _EXPORT_FOLDER_MAP = {"json": "sessions", "csv": "csv", "png": "images", "svg": "images"}

    def save_export_to_folder(content, filename):
        """Save export content to the appropriate project folder based on file extension."""
        ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
        folder = _EXPORT_FOLDER_MAP.get(ext)
        if folder and content and not content.startswith("Error") and not content.startswith("No data"):
            try:
                filepath = Path(folder) / filename
                filepath.write_text(content, encoding='utf-8')
            except Exception:
                pass

    def auto_save_download(section, ext, **gen_kwargs):
        """Decorator replacing @render.download that also saves a copy to the project folder.

        Usage:
            @auto_save_download("head_specialization", "csv", data_type="all_heads")
            def export_head_spec():
                yield content
        """
        import functools
        filename_fn = lambda: generate_export_filename(section, ext, **gen_kwargs)

        def decorator(fn):
            @render.download(filename=filename_fn)
            @functools.wraps(fn)
            def wrapper():
                parts = []
                for chunk in fn():
                    parts.append(str(chunk) if not isinstance(chunk, str) else chunk)
                    yield chunk
                # After all chunks yielded, save a copy to the project folder
                if parts:
                    content = "".join(parts)
                    fname = filename_fn()
                    save_export_to_folder(content, fname)
            return wrapper
        return decorator

    # --- PNG save handler (receives base64 data from JavaScript) ---
    @reactive.Effect
    @reactive.event(input._save_png)
    def _handle_save_png():
        data = input._save_png()
        if not data:
            return
        try:
            import base64
            filename = data.get("filename", "export.png")
            b64_data = data.get("data", "")
            # Strip data:image/png;base64, prefix
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            filepath = Path("images") / filename
            filepath.write_bytes(img_bytes)
        except Exception:
            pass

    # --- Session Persistence ---
    @auto_save_download("attention_atlas_session", "json")
    def save_session():
        def safe_get(input_fn, default=None):
            try:
                val = input_fn()
                # If silent exception occurs, it might return None or raise.
                # Just return val if no exception.
                return val
            except:
                return default

        # Basic Session Data
        session_data = {
            "text_input": safe_get(input.text_input, ""),
            "model_family": safe_get(input.model_family, "bert"),
            "model_name": safe_get(input.model_name, "bert-base-uncased"),
            # Global inputs might not exist if dashboard not rendered
            "layer": safe_get(input.global_layer, 0),
            "head": safe_get(input.global_head, 0),
            "topk": safe_get(input.global_topk, 3),
            "norm": safe_get(input.global_norm, "raw"),
            "view_mode": safe_get(input.view_mode, "basic"),
            "compare_mode": safe_get(input.compare_mode, False),
            "compare_prompts_mode": safe_get(input.compare_prompts_mode, False)
        }

        # Conditionally add Model B data ONLY if compare_mode is active
        if session_data["compare_mode"]:
            session_data.update({
                "model_family_B": safe_get(input.model_family_B, "bert"),
                "model_name_B": safe_get(input.model_name_B, "gpt2"),
            })
            
        # Text Input B is relevant if either compare mode is active
        if session_data["compare_mode"] or session_data["compare_prompts_mode"]:
             session_data["text_input_B"] = safe_get(input.text_input_B, "")
        
        yield json.dumps(session_data, indent=2)

    @reactive.Effect
    @reactive.event(input.load_session_upload)
    async def load_session():
        file_infos = input.load_session_upload()
        if not file_infos:
            return

        try:
            with open(file_infos[0]["datapath"], "r") as f:
                data = json.load(f)

            # 1. Update Layout/Model Controls first (triggers UI rebuild)
            if "compare_mode" in data:
                 val = data.get("compare_mode")
                 ui.update_switch("compare_mode", value=val)
                 if val:
                     active_compare_models.set(True)
                     session_force_compare_mode.set(True)
            
            if "compare_prompts_mode" in data:
                 val = data.get("compare_prompts_mode")
                 ui.update_switch("compare_prompts_mode", value=val)
                 if val:
                     active_compare_prompts.set(True)
                     session_force_compare_prompts_mode.set(True)
                     # Force Wizard to DONE so generation can proceed (overriding default "A" from effect)
                     prompt_entry_step.set("DONE")
            
            if "view_mode" in data:
                 val = data.get("view_mode")
                 ui.update_radio_buttons("view_mode", selected=val)
                 active_view_mode.set(val)

            # Update Models
            if "model_family" in data:
                ui.update_select("model_family", selected=data.get("model_family", "bert"))
            
            if data.get("compare_mode") and "model_family_B" in data:
                ui.update_select("model_family_B", selected=data.get("model_family_B"))

            # WAIT for UI to rebuild inputs based on model/mode changes
            await asyncio.sleep(0.5)

            # 2. Update Model Specifics (Names)
            if "model_name" in data:
                ui.update_select("model_name", selected=data.get("model_name"))
            if data.get("compare_mode") and "model_name_B" in data:
                 ui.update_select("model_name_B", selected=data.get("model_name_B"))

            # 3. Update Text Inputs (via JS)
            await session.send_custom_message("restore_session_text", {
                "text_input": data.get("text_input", ""),
                "text_input_B": data.get("text_input_B", "")
            })

            # 4. Update Custom Controls (Sliders & Radio)
            # These are custom HTML inputs, so we must use a custom message to update them
            await session.send_custom_message("restore_session_controls", {
                "layer": int(data.get("layer", 0)),
                "head": int(data.get("head", 0)),
                "topk": int(data.get("topk", 3)),
                "norm": data.get("norm", "raw")
            })

            ui.notification_show("Session loaded! Generating...", type="message")

            # 5. Trigger Generation
            await asyncio.sleep(0.2)
            await session.send_custom_message("trigger_generate", {})

        except Exception as e:
            ui.notification_show(f"Failed to load session: {str(e)}", type="error")
            print(f"Load Session Error: {e}")
            traceback.print_exc()

    # --- Data Export Handlers ---

    def get_model_short_name():
        """Get short model name for filenames (bert or gpt2)."""
        try:
            family = input.model_family()
            return family if family else "model"
        except:
            return "model"

    @auto_save_download("head_specialization", "csv", data_type="all_heads")
    def export_head_spec():
        """Export head specialization data as CSV - ALL layers and heads."""
        res = cached_result.get()
        if not res:
            yield "No data available"
            return

        try:
            # res structure: (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, mlm_model, head_specialization, isa_data, head_clusters)
            head_specialization = res[9] if len(res) > 9 else None
            if head_specialization is None:
                yield "No head specialization data available"
                return

            # Build CSV with all layers and heads
            lines = ["layer,head,syntax,semantics,cls_focus,punctuation,entities,long_range,self_attention"]
            for layer_idx in sorted(head_specialization.keys()):
                layer_data = head_specialization[layer_idx]
                for head_idx in sorted(layer_data.keys()):
                    metrics = layer_data[head_idx]
                    line = f"{layer_idx},{head_idx},{metrics.get('syntax',0):.4f},{metrics.get('semantics',0):.4f},{metrics.get('cls_focus',0):.4f},{metrics.get('punctuation',0):.4f},{metrics.get('entities',0):.4f},{metrics.get('long_range',0):.4f},{metrics.get('self_attention',0):.4f}"
                    lines.append(line)
            yield "\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("multi_head_attention", "csv", data_type="all_layers_heads")
    def export_multi_head_data():
        """Export multi-head attention matrices as CSV - ALL layers and heads."""
        res = cached_result.get()
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            # Build CSV: layer,head,query_token_idx,key_token_idx,query_token,key_token,weight
            lines = ["layer,head,query_idx,key_idx,query_token,key_token,weight"]
            seq_len = len(tokens)

            # Export ALL layers and ALL heads
            for layer_idx, layer_att in enumerate(attentions):
                # layer_att shape: (batch=1, num_heads, seq_len, seq_len)
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()  # Remove batch dim
                else:
                    layer_att_np = layer_att[0] if len(layer_att.shape) == 4 else layer_att

                num_heads = layer_att_np.shape[0]
                for h in range(num_heads):
                    for i in range(seq_len):
                        for j in range(seq_len):
                            weight = layer_att_np[h, i, j]
                            if weight > 1e-4:  # Skip near-zero values for file size
                                lines.append(f"{layer_idx},{h},{i},{j},{tokens[i]},{tokens[j]},{weight:.6f}")

            yield "\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("attention_metrics", "csv", data_type="all_layers_heads")
    def export_attention_metrics_single():
        """Export attention metrics for ALL layers and heads as CSV."""
        res = cached_result.get()
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            # Get metric names from first computation
            first_layer_att = attentions[0]
            if isinstance(first_layer_att, torch.Tensor):
                sample_matrix = first_layer_att[0, 0].detach().cpu().numpy()
            else:
                sample_matrix = first_layer_att[0, 0]
            sample_metrics = compute_all_attention_metrics(sample_matrix)
            metric_names = list(sample_metrics.keys())

            # Build CSV header
            lines = ["layer,head," + ",".join(metric_names)]

            # Compute metrics for ALL layers and ALL heads
            for layer_idx, layer_att in enumerate(attentions):
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()
                else:
                    layer_att_np = layer_att[0]

                num_heads = layer_att_np.shape[0]
                for head_idx in range(num_heads):
                    att_matrix = layer_att_np[head_idx]
                    metrics = compute_all_attention_metrics(att_matrix)
                    values = ",".join([f"{metrics.get(m, 0):.6f}" for m in metric_names])
                    lines.append(f"{layer_idx},{head_idx},{values}")

            # Also add global metrics (averaged across all)
            # Compute global by averaging attention across all heads/layers
            all_weights = []
            for layer_att in attentions:
                if isinstance(layer_att, torch.Tensor):
                    all_weights.append(layer_att[0].detach().cpu().numpy().mean(axis=0))
                else:
                    all_weights.append(layer_att[0].mean(axis=0))
            global_matrix = np.mean(all_weights, axis=0)
            global_metrics = compute_all_attention_metrics(global_matrix)
            global_values = ",".join([f"{global_metrics.get(m, 0):.6f}" for m in metric_names])
            lines.append(f"global,all,{global_values}")

            yield "\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("attention_tree", "json", data_type="tree_structure")
    def export_tree_data_json():
        """Export attention dependency tree as JSON."""
        res = cached_result.get()
        if not res:
            yield json.dumps({"error": "No data available"})
            return

        try:
            tokens = res[0]
            attentions = res[3]
            layer_idx = int(input.global_layer()) if input.global_layer() else 0
            head_idx = int(input.global_head()) if input.global_head() else 0
            try:
                focus_token = int(input.global_focus_token()) if input.global_focus_token() else 0
            except:
                focus_token = 0

            if attentions is None or len(attentions) == 0:
                yield json.dumps({"error": "No attention data available"})
                return

            # Get attention matrix - shape is (batch=1, num_heads, seq_len, seq_len)
            layer_att = attentions[layer_idx]
            if isinstance(layer_att, torch.Tensor):
                att_matrix = layer_att[0, head_idx].detach().cpu().numpy()
            else:
                att_matrix = layer_att[0, head_idx]

            # Build tree data
            tree_data = compute_influence_tree(att_matrix, tokens, root_idx=focus_token)

            export_data = {
                "model": get_model_short_name(),
                "layer": layer_idx,
                "head": head_idx,
                "focus_token": focus_token,
                "tokens": tokens,
                "tree": tree_data
            }

            yield json.dumps(export_data, indent=2)
        except Exception as e:
            yield json.dumps({"error": str(e)})

    @auto_save_download("isa", "json", data_type="sentence_attention")
    def export_isa_data():
        """Export ISA (Inter-sentence Attention) data as JSON."""
        res = cached_result.get()
        if not res:
            yield json.dumps({"error": "No data available"})
            return

        try:
            tokens = res[0]
            # ISA data is at index 10
            isa_data = res[10] if len(res) > 10 else None

            if isa_data is None:
                yield json.dumps({"error": "No ISA data available. Ensure input has multiple sentences."})
                return

            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                "model": get_model_short_name(),
                "tokens": tokens,
            }

            # Handle different ISA data structures
            if isinstance(isa_data, dict):
                for key, value in isa_data.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    else:
                        export_data[key] = value
            else:
                export_data["isa_scores"] = isa_data if not isinstance(isa_data, np.ndarray) else isa_data.tolist()

            yield json.dumps(export_data, indent=2, default=str)
        except Exception as e:
            yield json.dumps({"error": str(e)})

    @auto_save_download("attention_flow", "csv", data_type="all_layers")
    def export_attention_flow_data():
        """Export attention flow (rollout) data as CSV - ALL layers."""
        res = cached_result.get()
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]
            encoder_model = res[7]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            from .renderers import compute_attention_rollout

            lines = ["layer,head,source_idx,target_idx,source_token,target_token,weight"]
            seq_len = len(tokens)

            # Export per-layer, per-head attention
            for l_idx, layer_att in enumerate(attentions):
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()
                else:
                    layer_att_np = layer_att[0]

                num_heads = layer_att_np.shape[0]
                for h_idx in range(num_heads):
                    att_matrix = layer_att_np[h_idx]
                    for i in range(seq_len):
                        for j in range(seq_len):
                            val = att_matrix[i, j]
                            if val > 1e-4:
                                lines.append(f"{l_idx},{h_idx},{i},{j},{tokens[i]},{tokens[j]},{val:.6f}")

            # Export global rollout
            rollout_matrix = compute_attention_rollout(attentions)
            for i in range(seq_len):
                for j in range(seq_len):
                    val = rollout_matrix[i, j]
                    if val > 1e-4:
                        lines.append(f"global,rollout,{i},{j},{tokens[i]},{tokens[j]},{val:.6f}")

            yield "\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("multi_head_attention_B", "csv", data_type="all_layers_heads", is_b=True)
    def export_heatmap_data_B():
        """Export multi-head attention matrices for Prompt B as CSV - ALL layers and heads."""
        res = get_active_result("_B")
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            # Build CSV: layer,head,query_token_idx,key_token_idx,query_token,key_token,weight
            lines = ["layer,head,query_idx,key_idx,query_token,key_token,weight"]
            seq_len = len(tokens)

            # Export ALL layers and ALL heads
            for layer_idx, layer_att in enumerate(attentions):
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()
                else:
                    layer_att_np = layer_att[0] if len(layer_att.shape) == 4 else layer_att

                num_heads = layer_att_np.shape[0]
                for h in range(num_heads):
                    for i in range(seq_len):
                        for j in range(seq_len):
                            weight = layer_att_np[h, i, j]
                            if weight > 1e-4:
                                lines.append(f"{layer_idx},{h},{i},{j},{tokens[i]},{tokens[j]},{weight:.6f}")

            yield "\\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("attention_metrics_B", "csv", data_type="all_layers_heads", is_b=True)
    def export_attention_metrics_single_B():
        """Export attention metrics for Prompt B for ALL layers and heads as CSV."""
        res = get_active_result("_B")
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            # Get metric names from first computation
            first_layer_att = attentions[0]
            if isinstance(first_layer_att, torch.Tensor):
                sample_matrix = first_layer_att[0, 0].detach().cpu().numpy()
            else:
                sample_matrix = first_layer_att[0, 0]
            sample_metrics = compute_all_attention_metrics(sample_matrix)
            metric_names = list(sample_metrics.keys())

            lines = ["layer,head," + ",".join(metric_names)]

            for layer_idx, layer_att in enumerate(attentions):
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()
                else:
                    layer_att_np = layer_att[0]

                num_heads = layer_att_np.shape[0]
                for head_idx in range(num_heads):
                    att_matrix = layer_att_np[head_idx]
                    metrics = compute_all_attention_metrics(att_matrix)
                    values = ",".join([f"{metrics.get(m, 0):.6f}" for m in metric_names])
                    lines.append(f"{layer_idx},{head_idx},{values}")

            # Global metrics
            all_weights = []
            for layer_att in attentions:
                if isinstance(layer_att, torch.Tensor):
                    all_weights.append(layer_att[0].detach().cpu().numpy().mean(axis=0))
                else:
                    all_weights.append(layer_att[0].mean(axis=0))
            global_matrix = np.mean(all_weights, axis=0)
            global_metrics = compute_all_attention_metrics(global_matrix)
            global_values = ",".join([f"{global_metrics.get(m, 0):.6f}" for m in metric_names])
            lines.append(f"global,all,{global_values}")

            yield "\\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("attention_flow_B", "csv", data_type="all_layers", is_b=True)
    def export_flow_data_B():
        """Export attention flow data for Prompt B as CSV - ALL layers."""
        res = get_active_result("_B")
        if not res:
            yield "No data available"
            return

        try:
            tokens = res[0]
            attentions = res[3]

            if attentions is None or len(attentions) == 0:
                yield "No attention data available"
                return

            from .renderers import compute_attention_rollout

            lines = ["layer,head,source_idx,target_idx,source_token,target_token,weight"]
            seq_len = len(tokens)

            for l_idx, layer_att in enumerate(attentions):
                if isinstance(layer_att, torch.Tensor):
                    layer_att_np = layer_att[0].detach().cpu().numpy()
                else:
                    layer_att_np = layer_att[0]

                num_heads = layer_att_np.shape[0]
                for h_idx in range(num_heads):
                    att_matrix = layer_att_np[h_idx]
                    for i in range(seq_len):
                        for j in range(seq_len):
                            val = att_matrix[i, j]
                            if val > 1e-4:
                                lines.append(f"{l_idx},{h_idx},{i},{j},{tokens[i]},{tokens[j]},{val:.6f}")

            rollout_matrix = compute_attention_rollout(attentions)
            for i in range(seq_len):
                for j in range(seq_len):
                    val = rollout_matrix[i, j]
                    if val > 1e-4:
                        lines.append(f"global,rollout,{i},{j},{tokens[i]},{tokens[j]},{val:.6f}")

            yield "\\n".join(lines)
        except Exception as e:
            yield f"Error exporting data: {str(e)}"

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
    # Synchronize text inputs between tabs - MOVED TO GEN BUTTON TO PREVENT INPUT LAG
    # (Removed continuous sync)
    
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

    # Ensure Compare Modes are Mutually Exclusive
    @reactive.Effect
    @reactive.event(input.compare_mode)
    def handle_compare_mode_change():
        if input.compare_mode():
            ui.update_switch("compare_prompts_mode", value=False)

    @reactive.Effect
    @reactive.event(input.compare_prompts_mode)
    def handle_compare_prompts_change():
        if input.compare_prompts_mode():
            ui.update_switch("compare_mode", value=False)

    # Prompt Wizard Step State: 'A' or 'B' or 'DONE'
    prompt_entry_step = reactive.Value("A")

    # Update Baselines when model changes
    @reactive.Effect
    def update_baselines():
        res = cached_result.get()
        if not res: return
        
        # Unpack needed items
        # (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, ...)
        tokenizer = res[6]
        model = res[7]
        
        try: model_name = input.model_name()
        except: model_name = "unknown"
        
        # Check if we need to recompute (new model or first run)
        if current_baseline_model.get() != model_name:
            is_gpt2 = not hasattr(model, "encoder")
            try:
                # Run computation in a way that doesn't block if possible, 
                # but these are small baselines so synchronous is okay-ish for now.
                # Ideally this should be threaded if long.
                stats = compute_baselines(model, tokenizer, is_gpt2)
                baseline_stats.set(stats)
                current_baseline_model.set(model_name)
                print(f"DEBUG: Baselines updated for {model_name}")
            except Exception as e:
                print(f"ERROR computing baselines: {e}")
                traceback.print_exc()

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


    # EXCEPTION: Sync View Mode immediately (User Request)
    @reactive.Effect
    @reactive.event(input.view_mode)
    def sync_view_mode_live():
         active_view_mode.set(input.view_mode())






    
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
        # Wizard Logic - CHECK THIS FIRST to avoid premature reset
        try: mode = input.compare_prompts_mode()
        except: mode = False
        
        if mode:
            step = prompt_entry_step.get()
            if step == "A":
                # Transition to B, DO NOT COMPUTE, DO NOT SNAPSHOT, DO NOT RESET
                prompt_entry_step.set("B")
                await session.send_custom_message("switch_prompt_tab", "B")
                return

            # If we are here, we are in Step B (or DONE), so we proceed
            prompt_entry_step.set("DONE")

        # --- SNAPSHOT SIDEBAR STATE ---
        # Only these values will be used for rendering until the next 'Generate All'
        try: cm = input.compare_mode()
        except: cm = False
        try: cpm = input.compare_prompts_mode()
        except: cpm = False
        try: vm = input.view_mode()
        except: vm = "basic"
        
        # Force Flags Override (Session Load Fix)
        # Prioritize session load flags over potentially stale UI inputs
        if session_force_compare_mode.get():
             cm = True
             session_force_compare_mode.set(False)
             print("DEBUG: Force Compare Mode Applied")
             
        if session_force_compare_prompts_mode.get():
             cpm = True
             session_force_compare_prompts_mode.set(False)
             prompt_entry_step.set("DONE") # Ensure wizard doesn't block
             print("DEBUG: Force Compare Prompts Mode Applied")
        
        active_compare_models.set(cm)
        active_compare_prompts.set(cpm)
        active_view_mode.set(vm)

        # Clear existing results to provide a "fresh load" feel.
        # Set cached_text to current inputs so they appear in previews immediately while loading.
        cached_result.set(None)
        cached_result_B.set(None)
        cached_text_A.set(input.text_input().strip())
        try: cached_text_B.set(input.text_input_B().strip())
        except: cached_text_B.set("")
        
        # Reset MLM triggers on new computation
        show_mlm_A.set(False)
        show_mlm_B.set(False)

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
        
        # Sync to Bias Input
        ui.update_text_area("bias_input_text", value=text)

        # Reset MLM states on new generation
        show_mlm_A.set(False)
        show_mlm_B.set(False)

        running.set(True)
        await session.send_custom_message('start_loading', {})
        await asyncio.sleep(0.1)

        model_name = input.model_name()
        print(f"DEBUG: Model name A: {model_name}")
        
        # Check compare modes - Use the values we determined at the start (cm, cpm)
        # which account for the force flags.
        compare_models = cm
        compare_prompts = cpm
        
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                # Compute Model A (Prompt A)
                print("DEBUG: Starting heavy_compute A")
                result_A = await loop.run_in_executor(pool, heavy_compute, text, model_name)
                cached_result.set(result_A)
                cached_text_A.set(text)  # Store text used for generation

                # Compute Second Result if needed
                if compare_models:
                    # Case 1: Same Prompt (A), Different Model (B)
                    try: 
                        model_name_B = input.model_name_B()
                        if not model_name_B: model_name_B = "gpt2"
                    except: model_name_B = "gpt2"
                    
                    print(f"DEBUG: Starting heavy_compute B ({model_name_B}) for Compare Models")
                    result_B = await loop.run_in_executor(pool, heavy_compute, text, model_name_B)
                    cached_result_B.set(result_B)
                    cached_text_B.set(text)  # Same text for compare models

                elif compare_prompts:
                    # Case 2: Different Prompt (B), Same Model (A)
                    try: text_B = input.text_input_B().strip()
                    except: text_B = ""

                    if text_B:
                        print(f"DEBUG: Starting heavy_compute B (Prompt B) for Compare Prompts")
                        # Use Model A for Prompt B
                        result_B = await loop.run_in_executor(pool, heavy_compute, text_B, model_name)
                        cached_result_B.set(result_B)
                        cached_text_B.set(text_B)  # Store Prompt B text
                    else:
                        cached_result_B.set(None)
                        cached_text_B.set("")

                else:
                    cached_result_B.set(None)
                    cached_text_B.set("")
                    
        except Exception as e:
            print(f"ERROR in compute_all: {e}")
            traceback.print_exc()
            cached_result.set(None)
            cached_result_B.set(None)
        finally:
            running.set(False)
            await session.send_custom_message('stop_loading', {})

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
        is_comparing = active_compare_models.get() or active_compare_prompts.get()
        if global_metrics_mode.get() == "all":
            # Deselect: go back to specific layer/head
            global_metrics_mode.set("specific")
            ui.update_radio_buttons("radar_mode", selected="single")
            if is_comparing: ui.update_radio_buttons("radar_mode_B", selected="single")
        else:
            # Select: set to global (all layers/heads)
            global_metrics_mode.set("all")
            ui.update_radio_buttons("radar_mode", selected="all")
            if is_comparing: ui.update_radio_buttons("radar_mode_B", selected="all")
    
    # Reactive value to track if global metrics should use all layers/heads
    global_metrics_mode = reactive.Value("specific")  # "all" or "specific"

    # Reactive value for attention normalization mode
    global_norm_mode = reactive.Value("raw")  # "raw" | "col" | "rollout"

    # Reactive value for rollout layers mode ("current" = 0→selected layer, "all" = all layers)
    global_rollout_layers = reactive.Value("current")

    # Reactive values to preserve accordion state across re-renders
    accordion_state_single = reactive.Value(["overview"])  # Default: overview open
    accordion_state_compare = reactive.Value(["overview"])  # Default: overview open

    # Track accordion changes and save state
    @reactive.Effect
    @reactive.event(input.dashboard_accordion)
    def save_accordion_state_single():
        try:
            state = input.dashboard_accordion()
            if state is not None:
                accordion_state_single.set(list(state) if state else [])
        except:
            pass

    @reactive.Effect
    @reactive.event(input.dashboard_accordion_compare)
    def save_accordion_state_compare():
        try:
            state = input.dashboard_accordion_compare()
            if state is not None:
                accordion_state_compare.set(list(state) if state else [])
        except:
            pass

    # Reset Global view mode when layer/head sliders change
    @reactive.Effect
    @reactive.event(input.global_layer, input.global_head)
    def reset_metrics_mode():
        if global_metrics_mode.get() == "all":
            global_metrics_mode.set("specific")
            # Also update radar mode radio buttons
            is_comparing = active_compare_models.get() or active_compare_prompts.get()
            ui.update_radio_buttons("radar_mode", selected="single")
            if is_comparing:
                ui.update_radio_buttons("radar_mode_B", selected="single")
            # Force update button visual state after a delay to ensure it runs after re-renders
            js_code = """
            setTimeout(function() {
                var btn = document.getElementById('trigger_global_view');
                if (btn) btn.classList.remove('active');
            }, 100);
            """
            ui.insert_ui(selector="body", where="beforeEnd", ui=ui.tags.script(js_code))

    # Update norm mode when radio button changes
    @reactive.Effect
    @reactive.event(input.global_norm)
    def update_norm_mode():
        try:
            mode = input.global_norm()
            if mode in ["raw", "col", "rollout"]:
                global_norm_mode.set(mode)
        except:
            pass

    # Update rollout layers mode when dropdown changes
    @reactive.Effect
    @reactive.event(input.global_rollout_layers)
    def update_rollout_layers():
        try:
            mode = input.global_rollout_layers()
            if mode in ["current", "all"]:
                global_rollout_layers.set(mode)
        except:
            pass

    # -------------------------------------------------------------------------
    # ATTENTION NORMALIZATION HELPERS
    # -------------------------------------------------------------------------
    def attention_rollout(attentions, current_layer):
        """
        Compute attention rollout up to the specified layer.

        Attention rollout propagates attention through all layers accounting
        for residual connections, showing how information flows from input
        to the current layer.

        Args:
            attentions: tuple of attention tensors, one per layer
                        Each has shape (batch, num_heads, seq_len, seq_len)
            current_layer: int, compute rollout up to this layer (inclusive)

        Returns:
            numpy array of shape (seq_len, seq_len) with accumulated attention
        """
        # Average across heads for each layer
        att_per_layer = []
        for layer_idx in range(current_layer + 1):
            # Shape: (num_heads, seq_len, seq_len) -> (seq_len, seq_len)
            att_layer = attentions[layer_idx][0].cpu().numpy()
            att_avg = np.mean(att_layer, axis=0)
            att_per_layer.append(att_avg)

        seq_len = att_per_layer[0].shape[0]
        rollout = np.eye(seq_len)

        for layer_idx in range(current_layer + 1):
            attention = att_per_layer[layer_idx]
            # Add residual connection (0.5 * attention + 0.5 * identity)
            attention_with_residual = 0.5 * attention + 0.5 * np.eye(seq_len)
            # Re-normalize rows to sum to 1
            row_sums = attention_with_residual.sum(axis=-1, keepdims=True)
            attention_with_residual = attention_with_residual / (row_sums + 1e-8)
            # Accumulate: rollout = attention_with_residual @ rollout
            rollout = np.matmul(attention_with_residual, rollout)

        return rollout

    def get_normalized_attention(raw_attention, attentions, layer_idx, mode, is_causal=False, use_all_layers=False):
        """
        Apply normalization to attention weights based on the selected mode.

        Args:
            raw_attention: numpy array (seq_len, seq_len) - raw attention weights
            attentions: tuple of all layer attentions (needed for rollout)
            layer_idx: current layer index
            mode: "raw" | "col" | "rollout"
            is_causal: bool, whether to apply causal masking
            use_all_layers: bool, for rollout mode - use all layers instead of up to current

        Returns:
            numpy array (seq_len, seq_len) with normalized attention
        """
        if mode == "raw":
            return raw_attention
        elif mode == "col":
            # Column normalization: normalize by key dimension (columns sum to 1)
            # This shows "which tokens receive the most attention overall"
            col_sums = raw_attention.sum(axis=0, keepdims=True)
            # Add epsilon to avoid division by zero
            normalized = raw_attention / (col_sums + 1e-8)
            return normalized
        elif mode == "rollout":
            # Attention rollout: accumulated flow through layers
            # Use all layers or up to the current layer
            target_layer = len(attentions) - 1 if use_all_layers else layer_idx
            # Slice attentions up to target_layer (inclusive if 0-indexed?) 
            # attentions is a tuple/list. range 0 to target_layer inclusive -> [:target_layer+1]
            relevant_attentions = attentions[:target_layer+1]
            rollout = compute_attention_rollout(relevant_attentions)
            
            if is_causal:
                rollout = rollout.copy()
                rollout[np.triu_indices_from(rollout, k=1)] = np.nan
            return rollout
        else:
            return raw_attention

    def get_norm_mode_label(mode, layer_idx=None, use_all_layers=False, total_layers=None):
        """Get dynamic label for the current normalization mode."""
        if mode == "raw":
            return "Attention Weights (query perspective)"
        elif mode == "col":
            return "Attention Weights (key perspective - who receives attention)"
        elif mode == "rollout":
            if use_all_layers and total_layers is not None:
                layer_str = f"0→{total_layers - 1}"
            elif layer_idx is not None:
                layer_str = f"0→{layer_idx}"
            else:
                layer_str = "all"
            return f"Accumulated Attention Flow (layers {layer_str})"
        return "Attention Weights"

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
        
        # Determine mode (Snapshotted)
        compare_prompts = active_compare_prompts.get()
        compare_models = active_compare_models.get()

        # Get defaults
        try: current_layer = int(input.global_layer())
        except: current_layer = 0
        try: current_head = int(input.global_head())
        except: current_head = 0
        try: current_topk = int(input.global_topk())
        except: current_topk = 3

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

                // SYNC: Update MLM Tokens for the correct model (A or B)
                $(`.maskable-token[data-model="${{prefix}}"], .predicted-word[data-model="${{prefix}}"], .interactive-token[data-model="${{prefix}}"]`).removeClass('masked-active');
                selectedArray.forEach(i => {{
                     $(`[data-index="${{i}}"][data-model="${{prefix}}"]`).addClass('masked-active');
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
            
            // Global View Toggle - toggle visual state immediately on click
            // Using onclick (not addEventListener) to prevent multiple handlers on re-render
            const globalBtn = document.getElementById('trigger_global_view');
            if (globalBtn) {{
                globalBtn.onclick = function() {{
                    this.classList.toggle('active');
                }};
            }}

             // Scale Toggle
             const scaleToggle = document.getElementById('scale-toggle');
             if (scaleToggle) {{
                 scaleToggle.onclick = function() {{
                     this.classList.toggle('active');
                     const isFull = this.classList.contains('active');
                     Shiny.setInputValue('global_scale_full', isFull, {{priority: 'event'}});
                 }};
             }}
             
             // Norm Radio with Rollout layers visibility
            const radioGroup = document.getElementById('norm-radio-group');
            const rolloutControl = document.getElementById('rollout-layers-control');
            const rolloutLayersGroup = document.getElementById('rollout-layers-group');

            if (radioGroup) {{
                radioGroup.querySelectorAll('.radio-option').forEach(btn => {{
                    btn.onclick = function() {{
                        radioGroup.querySelectorAll('.radio-option').forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        const normValue = this.getAttribute('data-value');
                        Shiny.setInputValue('global_norm', normValue, {{priority: 'event'}});

                        // Show/hide rollout layers control
                        if (rolloutControl) {{
                            if (normValue === 'rollout') {{
                                rolloutControl.classList.add('visible');
                            }} else {{
                                rolloutControl.classList.remove('visible');
                            }}
                        }}
                    }};
                }});
            }}

            // Rollout layers radio buttons
            if (rolloutLayersGroup) {{
                rolloutLayersGroup.querySelectorAll('.radio-option').forEach(btn => {{
                    btn.onclick = function() {{
                        rolloutLayersGroup.querySelectorAll('.radio-option').forEach(b => b.classList.remove('active'));
                        this.classList.add('active');
                        const layersValue = this.getAttribute('data-value');
                        Shiny.setInputValue('global_rollout_layers', layersValue, {{priority: 'event'}});
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
                
                # Global View - class managed by JavaScript only to avoid render timing issues
                ui.div(
                    {"class": "control-group"},
                    ui.span("View", class_="control-label"),
                    ui.input_action_button("trigger_global_view", "Global", class_="btn-global")
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
                
                # Scale Toggle
                ui.div(
                    {"class": "control-group"},
                    ui.span("Scale", class_="control-label"),
                    ui.div(
                        {"class": "btn-global", "id": "scale-toggle", "style": "width: 40px; cursor: pointer; position: relative;", "title": "Toggle between Optimized Scale (green) and Full 0-1 Scale (grey)"},
                        ui.span("Full", style="font-size: 8px;")
                    )
                ),


                
                # Norm (horizontal layout: label left, buttons right)
                ui.div(
                    {"class": "control-group norm-control-group"},
                    ui.span("Norm", class_="control-label"),
                    ui.div(
                        {"class": "norm-control-wrapper"},
                        ui.div(
                            {"class": "radio-group", "id": "norm-radio-group"},
                            ui.span("Raw", class_="radio-option active", title="Direct attention from softmax. Each query (row) distributes 100% of its attention.", **{"data-value": "raw"}),
                            ui.span("Col", class_="radio-option", title="Normalized by keys (columns). Shows which tokens are most attended to overall.", **{"data-value": "col"}),
                            ui.span("Rollout", class_="radio-option", title="Accumulated attention flow through all layers up to current, accounting for residual connections.", **{"data-value": "rollout"}),
                        ),
                        ui.div(
                            {"class": "rollout-layers-control", "id": "rollout-layers-control"},
                            ui.div(
                                {"class": "radio-group", "id": "rollout-layers-group"},
                                ui.span("0→L", class_="radio-option active", title="Rollout from layer 0 to the currently selected layer", **{"data-value": "current"}),
                                ui.span("All", class_="radio-option", title="Rollout across all layers (0 to max layer)", **{"data-value": "all"}),
                            )
                        )
                    )
                ),
                
                # Top-K
                ui.div(
                    {"class": "control-group"},
                    ui.span("Top-K", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span(str(current_topk), id="topk-value", class_="slider-value"),
                        ui.tags.input(type="range", id="topk-slider", min="1", max="20", value=str(current_topk), step="1")
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
        
        # Restore live preview (Safe now that dashboard is isolated)
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
        
        try: t = input.text_input_B().strip()
        except: t = ""
        
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
        # Fallback to cached_text_A if no result yet (loading state)
        if res:
            tokens = res[0]
            tokenizer = res[6]
            encoder_model = res[7]
            text_used = tokenizer.convert_tokens_to_string(tokens)
            is_gpt2 = not hasattr(encoder_model, "encoder")
        else:
            text_used = cached_text_A.get()
            is_gpt2 = False # Default or detect from sidebar? 
            # Actually get_preview_text_view handles the display nicely without is_gpt2
            
        # Detect GPT-2 (has 'h' attribute, not 'encoder') and add note about Ġ removal
        footer = ""
        if res and not hasattr(res[7], "encoder"):  # GPT-2
            footer = '<span style="color:#94a3b8;">Ġ (space token) removed for visualization</span>'
        return get_preview_text_view(res, text_used, "", footer)

    @output
    @render.ui
    def preview_text_B():
        res = get_active_result("_B")
        # Fallback to cached_text_B if no result yet (loading state)
        if res:
            tokens = res[0]
            tokenizer = res[6]
            encoder_model = res[7]
            text_used = tokenizer.convert_tokens_to_string(tokens)
        else:
            text_used = cached_text_B.get()
            
        footer = ""
        if res and not hasattr(res[7], "encoder"):  # GPT-2
            footer = '<span style="color:#94a3b8;">Ġ (space token) removed for visualization</span>'
        return get_preview_text_view(res, text_used, "_B", footer)


    def get_gpt2_dashboard_ui(res, input, output, session):
        tokens, _, _, _, _, _, _, encoder_model, *_ = res
        num_layers = len(encoder_model.h)
        num_heads = encoder_model.h[0].attn.num_heads
        

        # Get current selections
        try: top_k = int(input.global_topk())
        except: top_k = 3

        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens]

            # Determine View Mode for Layout
        try: 
            view_mode = input.view_mode()
        except: 
            view_mode = "basic"

        # Define Layout Blocks
        
        # Block 1: Global Metrics Card
        metrics_card = ui.div({
            "class": "card"
        }, 
            ui.div(
                {"style": "display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px;"},
                ui.h4("Global Attention Metrics", style="margin: 0;"),
                ui.HTML("""
                <div class='info-tooltip-wrapper' style='display:inline-flex; align-items:center; vertical-align:middle; position:relative; top:-1px;'>
                    <span class='info-tooltip-icon' style='width:14px; height:14px; line-height:14px; font-size:9px;'>i</span>
                    <div class='info-tooltip-content'>
                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Metrics</strong>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Quantitative measures characterizing attention behavior for comparison across heads and layers—descriptive statistics, not quality judgments.</p>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Confidence = max weight. Focus = normalized entropy (low = concentrated, high = diffuse). Sparsity = % near-zero weights. Additional: Uniformity, Balance, Flow Change. Updates for selected Layer/Head or 'Global'. Normalization modes: 'Raw', 'Column', 'Rollout'.</p>
                        <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Metrics describe distribution shape but don't indicate whether attention is 'correct' or task-relevant.</p>
                    </div>
                </div>
                """),
                ui.span("All Layers · All Heads", style="font-size: 11px; color: #94a3b8; font-weight: 500;")
            ),
            get_metrics_display(res)
        )
        
        # Block 2: Attention Maps Layout
        maps_row = ui.layout_columns(
            ui.output_ui("attention_map"),
            ui.output_ui("attention_flow"),
            col_widths=[6, 6]
        )

        # Block 3: Head Specialization & Tree Layout
        radar_tree_row = ui.layout_columns(
            ui.output_ui("render_radar_dashboard"),
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
        )

        # Configurable Layout Order
        # User requested consistency: Head Specialization ALWAYS on Top.
        dynamic_rows = [radar_tree_row, metrics_card, maps_row]

        return ui.div(
            {"class": "dashboard-stack gpt2-layout"},
            
            # Row 1: Embeddings
            ui.layout_columns(
                ui.div(
                    {"class": "card"},
                    ui.h4("Token Embeddings"),
                    ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_embedding_table(res, top_k=top_k)
                ),
                ui.div(
                    {"class": "card"},
                    ui.h4("Positional Embeddings"),
                    ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
                    get_posenc_table(res, top_k=top_k)
                ),
                ui.div(
                    {"class": "card"},
                    ui.h4("Sum & Layer Normalization"),
                    ui.p("Combines token, position, and segment embeddings, with layer normalization to stabilize activations before attention.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                    ui.div(style="height: 34px;"), # Spacer to align with Q/K/V (which has 2 button rows)
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
                    ui.p("Query, Key, Value projections with magnitude, alignment, and directional analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                    get_qkv_table(res, qkv_layer, top_k=top_k)
                ),
                ui.output_ui("render_scaled_attention_dashboard"),
                ui.div(
                    {"class": "card"},
                    viz_header("Multi-Head Attention", "Grid of all heads in this layer. See global patterns.",
                               """
                               <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Multi-Head Attention</strong>
                               <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Visualizes attention weight distribution showing how each token attends to others—not a direct measure of information flow or causal influence.</p>
                               <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Matrix where cell (i,j) = attention from query i to key j. Options: specific Layer/Head or 'Global' (mean across heads). Normalization: 'Raw' (row-sum=1), 'Column' (key importance), 'Rollout' (accumulated across layers). Token click highlights row/column.</p>
                               <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Attention weights ≠ contribution to output; gradient-based methods provide more reliable importance estimates.</p>
                               """,
                               controls=[
                                   ui.download_button("export_multi_head_data", "JSON", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                               ]),
                    ui.output_ui(f"multi_head_view{suffix}")
                ),
                col_widths=[4, 4, 4]
            ),

            # Dynamic Rows (Swapped based on mode)
            *dynamic_rows,

            # Row 5: ISA

            # Row 5: ISA
            ui.output_ui("isa_row_dynamic"),

            # Row 6: Unembedding & Predictions
            ui.layout_columns(
                ui.div({"class": "card"}, ui.h4("Hidden States"), ui.p("Final vector representation before projection.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_layer_output_view(res, num_layers - 1)),
                ui.div({"class": "card"}, ui.h4("Masked Token Predictions (MLM)", style="display:flex;align-items:center;"), ui.p("Probabilities for the predicted token.", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_output_probabilities(res, use_mlm_val, text_val, top_k=top_k)),
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
            # Use snapshotted compare_mode if needed (though wl is usually header-based)
            cm = active_compare_models.get()
            wl = input.word_level_toggle()
            # Allow word level in single mode too if enabled
            if wl:
                use_word_level = True
        except Exception as e:
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
        try: top_k = int(input.global_topk())
        except: top_k = 3
        return get_embedding_table(res, top_k=top_k)

    @output
    @render.ui
    def render_segment_table():
        res = get_active_result()
        if not res: return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Segment Embeddings"),
            ui.p("Encodes sentence membership using binary embeddings (Segment A or Segment B), enabling BERT to distinguish between tokens from different sentences. Essential for sentence-pair tasks such as Natural Language Inference, Question Answering, and Next Sentence Prediction.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_segment_embedding_view(res)
        )

    @output
    @render.ui
    def render_posenc_table():
        res = get_active_result()
        if not res: return None
        try: top_k = int(input.global_topk())
        except: top_k = 3
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Positional Embeddings"),
            ui.p("Injects absolute position information into each token representation using learned embeddings. Without positional encoding, Transformers would be permutation-invariant—unable to distinguish word order. Both BERT and GPT-2 use learned (not sinusoidal) position embeddings.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_posenc_table(res, top_k=top_k)
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
            ui.p("Computes the element-wise sum of token, positional, and segment embeddings (where applicable), then applies Layer Normalization to stabilize the activation distribution before entering the first Transformer block.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_sum_layernorm_view(res, encoder_model)
        )

    @output
    @render.ui
    def render_qkv_table():
        res = get_active_result()
        if not res: return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: top_k = int(input.global_topk())
        except: top_k = 3

        # Get normalization mode for the alignment view
        norm_mode = global_norm_mode.get()

        # Check if global mode is active
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Q/K/V Projections"),
            ui.p("Transforms input hidden states into Query, Key, and Value representations through learned linear projections (Q=XW_Q, K=XW_K, V=XW_V). Queries encode 'what information to look for', Keys encode 'what information is available', and Values contain 'the information to aggregate'.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_qkv_table(res, layer_idx, top_k=top_k, norm_mode=norm_mode, use_global=use_global)
        )

    @output
    @render.ui
    def render_scaled_attention_dashboard():
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

        # Get normalization mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            viz_header(
                "Scaled Dot-Product Attention",
                "Per-token breakdown of Q·K scoring and softmax weighting for the selected query position.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Scaled Dot-Product Attention</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Shows how attention scores are computed between token pairs, determining which tokens influence each other's representations.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention(Q,K,V) = softmax(QK^T / √d_k)V. Dot product measures Query-Key similarity; scaling by √d_k prevents softmax saturation; output weights sum to 1 per row. Token selector focuses on specific query position; adjustable top-k controls how many key connections are displayed.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Raw attention scores don't account for Value vector magnitudes—high attention weight doesn't guarantee high influence on the output.</p>
                """,
                subtitle=f"(Layer {layer_idx} · Head {head_idx})",
                controls=[
                    ui.download_button("export_attention_metrics_dashboard", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                ]
            ),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=top_k, norm_mode=norm_mode, use_global=use_global)
        )

    @output
    @render.ui
    def render_scaled_attention():
        """Scaled Dot-Product Attention for non-compare mode (Advanced view in single model mode)."""
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

        # Get normalization mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            viz_header(
                "Scaled Dot-Product Attention",
                "Per-token breakdown of Q·K scoring and softmax weighting for the selected query position.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Scaled Dot-Product Attention</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Shows how attention scores are computed between token pairs, determining which tokens influence each other's representations.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention(Q,K,V) = softmax(QK^T / √d_k)V. Dot product measures Query-Key similarity; scaling by √d_k prevents softmax saturation; output weights sum to 1 per row. Token selector focuses on specific query position; adjustable top-k controls how many key connections are displayed.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Raw attention scores don't account for Value vector magnitudes—high attention weight doesn't guarantee high influence on the output.</p>
                """,
                subtitle=f"(Layer {layer_idx} · Head {head_idx})",
                controls=[
                    ui.download_button("export_scaled_attention", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                ]
            ),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=top_k, norm_mode=norm_mode, use_global=use_global)
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
            ui.p("Position-wise two-layer MLP applied independently to each token: FFN(x) = GELU(xW₁+b₁)W₂+b₂. The intermediate dimension expands to 4× the hidden size, creating a bottleneck architecture believed to store factual knowledge learned during pre-training.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
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
            ui.p("Applies residual connection (x + Sublayer(x)) followed by Layer Normalization. Residual connections enable gradient flow through deep networks; normalization stabilizes activations across layers. Applied after both attention and FFN sublayers.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
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
            ui.p("Second residual connection and Layer Normalization within the Transformer block, applied after the Feed-Forward Network. Completes the standard Transformer block architecture: Attention → Add&Norm → FFN → Add&Norm.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
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
            ui.p("Contextualized vector representations output by each Transformer layer. Unlike static embeddings, these encode both the token's original meaning and information aggregated from other positions through attention. Early layers capture syntax; deeper layers encode abstract semantics.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_layer_output_view(res, num_layers - 1)
        )

    @output
    @render.ui
    def render_mlm_predictions():
        res = get_active_result()
        if not res: return None

        # JS for Interactive Masking
        js_script = ui.HTML("""
        <script>
        function toggleMask(index, modelPrefix) {
            // SYNC: Trigger Floating Bar Click (Source of Truth)
            // Use the modelPrefix to target the correct floating bar chips
            modelPrefix = modelPrefix || 'A';
            const chip = document.querySelector(`.token-chip[data-idx="${index}"][data-prefix="${modelPrefix}"]`);
            if (chip) {
                chip.click();
            } else {
               // Fallback if bar not loaded? Just toggle locally for the correct model.
               $(`[data-index="${index}"][data-model="${modelPrefix}"]`).toggleClass('masked-active');
            }
        }

        $(document).on('click', '#run_custom_mask', function() {
            var indices = [];
            // Get indices only from Model A tokens
            $('.maskable-token[data-model="A"].masked-active').each(function() {
                indices.push(parseInt($(this).attr('data-index')));
            });
            Shiny.setInputValue('manual_mask_indices', indices, {priority: 'event'});
        });

        $(document).on('click', '#run_custom_mask_B', function() {
            var indices = [];
            // Get indices only from Model B tokens
            $('.maskable-token[data-model="B"].masked-active').each(function() {
                indices.push(parseInt($(this).attr('data-index')));
            });
            Shiny.setInputValue('manual_mask_indices_B', indices, {priority: 'event'});
        });
        </script>
        <style>
        .maskable-token {
            display: inline-block;
            margin: 2px;
            padding: 4px 8px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            
            /* Match Sentence Preview Style (.token-viz) */
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }
        .maskable-token:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 10;
            border-color: #3b82f6;
        }
        .maskable-token.masked-active {
            background: #fee2e2; /* Light red */
            border-color: #ef4444;
            color: transparent; /* Hide original text visually */
            position: relative;
        }
        .maskable-token.masked-active::after {
            content: "[MASK]";
            color: #b91c1c; /* Red text for mask */
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 10px;
            font-weight: bold;
            pointer-events: none;
        }
        
        /* Predicted Sentence Tokens - Sync Style */
        .predicted-word {
            display: inline-block;
            cursor: pointer;
            border-radius: 4px;
            padding: 0 4px; /* Slightly more padding */
            transition: all 0.2s ease;
            position: relative;
            
            /* Inactive state: Prediction is ignored/inactive */
            color: #7c3aed;
            opacity: 0.4;
            filter: grayscale(80%);
            border: 1px solid transparent;
        }
        
        /* Interactive tokens (Original words) */
        .interactive-token {
             display: inline-block;
             margin: 0 2px;
             padding: 0 4px;
             border-radius: 4px;
             cursor: pointer;
             transition: all 0.15s;
             border: 1px solid transparent;
             /* Match default text style but clickable */
        }
        .interactive-token:hover, .predicted-word:hover {
            background: #f1f5f9;
        }

        /* Active State logic */
        
        /* 1. Interactive Token (Original) -> Creates Mask */
        .interactive-token.masked-active {
            background: #fee2e2;
            border-color: #ef4444;
            color: transparent;
            position: relative;
            border: 1px solid #ef4444;
            min-width: 40px; /* Ensure width for [MASK] */
        }
        .interactive-token.masked-active::after {
            content: "[MASK]";
            color: #b91c1c;
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            font-size: 10px; font-weight: bold;
            pointer-events: none;
        }

        /* 2. Predicted Token (Result) -> HIGHLIGHTS Prediction */
        .predicted-word.masked-active {
            /* Active = We want to SEE the prediction */
            opacity: 1;
            filter: none;
            background: rgba(124, 58, 237, 0.1); 
            color: #7c3aed !important; 
            border: 1px solid rgba(124, 58, 237, 0.3);
            font-weight: 700;
        }
        
        .predicted-word.masked-active::after {
            content: none !important; /* Do NOT show [MASK] */
        }

        /* Toggle Switch Styling */
        .form-check-input:checked {
            background-color: #7c3aed;
            border-color: #7c3aed;
        }
        .form-check-input:focus {
            box-shadow: 0 0 0 0.25rem rgba(124, 58, 237, 0.25);
            border-color: #8b5cf6;
        }
        /* Interactive Label Font Fix */
        .form-check-label {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 13px;
            font-weight: 500;
            color: #475569;
        }
        
        .mask-selector-container {
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 12px;
            margin-bottom: 12px;
        }
        .mask-selector-label {
            font-size: 11px;
            font-weight: 600;
            color: #64748b;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-secondary-sm {
            background: white;
            border: 1px solid #cbd5e1;
            color: #475569;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
        }
        .btn-secondary-sm:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }
        /* Predicted Sentence Styles */
        .predicted-sentence-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #7c3aed; /* Violet accent */
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            text-align: left;
            position: relative;
        }
        .predicted-sentence-card::before {
            display: none;
        }
        .predicted-label {
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #64748b;
            margin-bottom: 8px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .predicted-text {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 16px;
            color: #1e293b;
            line-height: 1.5;
            font-weight: 400;
        }
        .predicted-word {
            color: #7c3aed;
            font-weight: 600;
            background: rgba(124, 58, 237, 0.05);
            padding: 1px 4px;
            border-radius: 4px;
            border-bottom: 1px solid rgba(124, 58, 237, 0.3);
            transition: all 0.2s ease;
        }
        .predicted-word:hover {
            background: rgba(124, 58, 237, 0.1);
            border-bottom-color: rgba(124, 58, 237, 0.6);
        }
        .empty-state-message {
            padding: 40px 20px;
            text-align: center;
            color: #64748b;
            font-style: italic;
            background: #f8fafc;
            border-radius: 12px;
            border: 2px dashed #e2e8f0;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        </style>
        """)

        # Determine if we should show predictions based on the ACTUAL loaded model
        _, _, _, _, _, _, _, encoder_model_local, *_ = res
        is_gpt2_local = not hasattr(encoder_model_local, "encoder")
        model_family = "gpt2" if is_gpt2_local else "bert"

        # Reconstruct text from tokenizer
        tokens = res[0]
        tokenizer = res[6]
        text = tokenizer.convert_tokens_to_string(tokens)
        try: top_k = int(input.global_topk())
        except: top_k = 3
        
        # Interactive Mode Logic (BERT only)
        manual_mode = False
        custom_mask_indices = None
        
        controls = []

        if model_family == "gpt2":
            use_mlm = True
            title = "Next Token Predictions (Causal)"
            desc = "Predicting the probability of the next token appearing after the sequence."
            
            tooltip_html = """
                <div class='info-tooltip-wrapper' style='display:flex; align-items:center; margin-left:2px;'>
                    <span class='info-tooltip-icon' style='width:14px; height:14px; line-height:14px; font-size:9px;'>i</span>
                    <div class='info-tooltip-content'>
                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Next Token Predictions (Causal) (GPT-2)</strong>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Displays the model's probabilistic forecast for the *next* token in the sequence, based on all preceding tokens.</p>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Standard causal language modeling: P(token_t | token_1...token_{t-1}). The model outputs a probability distribution over the vocabulary; top-k candidates shown.</p>
                        <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Purely statistical prediction based on training data; "plausible" continuations may not be factually correct or logically consistent.</p>
                    </div>
                </div>
            """
            title_header = ui.h4(
                title, 
                ui.HTML(tooltip_html),
                style="margin:0; display:flex; align-items:center;"
            )
        else:
            use_mlm = True # Always show
            
            # Interactive Mode Toggle
            try: manual_mode = input.mlm_interactive_mode()
            except: manual_mode = False
            
            if manual_mode:
                try: custom_mask_indices = input.manual_mask_indices()
                except: custom_mask_indices = None
            
            # Custom Toggle Button for Interactive Mode (Style: Norm/Similarity)
            # User wants "Toggle Masks" text, pink style like Norm buttons.
            # We use radio-group styling for consistency.
            active_class = "active" if manual_mode else ""
            button_label = "Go back" if manual_mode else "Toggle Masks"
            
            # Inject CSS for the button to ensure it looks correct even if classes are missing
            button_style = """
            <style>
            .toggle-masks-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 4px 12px;
                background: #f1f5f9;
                color: #64748b;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid transparent;
                user-select: none;
            }
            .toggle-masks-btn:hover {
                background: #e2e8f0;
                color: #334155;
            }
            .toggle-masks-btn.active {
                background: #ff5ca9; /* Pink */
                color: white;
                box-shadow: 0 2px 6px rgba(255, 92, 169, 0.3);
            }
            .toggle-masks-btn.active:hover {
                background: #f43f8e;
            }
            </style>
            """
            
            controls.append(ui.HTML(f"""
            {button_style}
            <div class='control-group' style='display:flex; align-items:center;'>
                <div class='radio-group'>
                    <span class='toggle-masks-btn {active_class}' 
                          onclick="Shiny.setInputValue('mlm_interactive_mode', !{str(manual_mode).lower()}, {{priority: 'event'}});">
                        {button_label}
                    </span>
                </div>
            </div>
            """))

            # Add tooltip to title for BERT
            tooltip_html = """
                <div class='info-tooltip-wrapper' style='display:flex; align-items:center; margin-left:2px;'>
                    <span class='info-tooltip-icon' style='width:14px; height:14px; line-height:14px; font-size:9px;'>i</span>
                    <div class='info-tooltip-content'>
                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Masked Token Predictions (MLM) (BERT)</strong>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Reveals BERT's predictions when each token is masked, showing what the model considers plausible given bidirectional context.</p>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Each token is iteratively masked; the model predicts the most likely original using context from both left and right. Top-k predictions with probabilities are displayed.</p>
                        <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Predictions reflect training data statistics—high confidence in stereotypical associations may indicate learned biases rather than linguistic understanding.</p>
                    </div>
                </div>
            """
            
            # Use ui.h4 directly with flexbox to match other headers perfectly while aligning content
            title_header = ui.h4(
                "Masked Token Predictions (MLM)", 
                ui.HTML(tooltip_html), 
                style="margin:0; display:flex; align-items:center;"
            )
            desc = "Demonstrates BERT's bidirectional language understanding by iteratively masking each token and predicting the most likely original based on full surrounding context from both directions."

        # Header Container: Separating Title Row from Description to allow vertical centering of Button vs Title
        header_row = ui.div(
            {"style": "display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 4px;"},
            title_header,
            ui.div(*controls)
        )
        
        description_row = ui.p(desc, style="font-size:11px; color:#6b7280; margin-bottom:8px; min-height: 20px; line-height: 1.4;")

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            js_script,
            header_row,
            description_row,
            get_output_probabilities(res, use_mlm, text, top_k=top_k, manual_mode=manual_mode, custom_mask_indices=custom_mask_indices)
        )

    @output
    @render.ui
    def render_radar_dashboard():
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
            viz_header(
                "Head Specialization",
                "Radar chart profiling this head's attention distribution across 7 linguistic dimensions.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Head Specialization</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Profiles what linguistic patterns each attention head focuses on—an approximation of functional specialization, not ground truth.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention mass aggregated by token category (Syntax, Semantics, Entities, Punctuation, CLS, Long-range, Self-attention) using POS tagging. Updates for selected Layer/Head; 'Global' view displays head specialization clusters across all heads.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> POS-based categorization is heuristic; heads may capture patterns not aligned with traditional linguistic categories.</p>
                """,
                controls=[
                    ui.download_button("export_head_spec_unique", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick=f"downloadPlotlyPNG('radar-chart-container', 'head_specialization_L{layer_idx}_H{head_idx}')" if mode == "single" else "downloadPlotlyPNG('radar-chart-container', 'head_specialization_all')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"id": "radar-chart-container"}, head_specialization_radar(res, layer_idx, head_idx, mode)),
            ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; margin-top: auto; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center;">
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

        # Get normalization mode and global mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"

        tree_png_filename = generate_export_filename("attention_tree", "png", is_b=False, incl_timestamp=False, data_type="dependency")
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            viz_header("Attention Dependency Tree",
                       "Recursive expansion of the focus token's top-k attention connections into a multi-level tree.",
                       """
                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Dependency Tree</strong>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Hierarchical view of which tokens the selected focus token attends to most strongly—shows attention structure, not syntactic dependencies.</p>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Tree built recursively from attention weights starting at the root token (selectable via floating toolbar). Each branch shows parent→child attention weight. Node size reflects attention strength.</p>
                        <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Tree structure imposes hierarchy on non-hierarchical attention; multiple strong connections may be underrepresented.</p>
                       """,
                       controls=[
                           ui.download_button("export_tree_data", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                           ui.download_button("export_topk_attention", "Top-K CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                           ui.tags.button(
                               "PNG",
                               onclick=f"downloadD3PNG('tree-viz-container', '{tree_png_filename}')",
                               style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                           )
                       ]),
            ui.div({"class": "viz-description"}, "Edge labels show attention weights; node size reflects strength. Use the floating toolbar to change the root token. ⚠️ This is attention structure, not syntactic parsing."),
            ui.div({"id": "tree-viz-container"}, get_influence_tree_ui(res, root_idx, layer_idx, head_idx, use_global=use_global, norm_mode=norm_mode))
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
        
        # Get Scale
        try: use_full_scale = input.global_scale_full()
        except: use_full_scale = False

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        return ui.div(
            {"class": "card"},
            viz_header("Attention Metrics", "Summary statistics for the selected head's attention distribution, or global aggregate across all heads.",
                               """
                               <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Metrics</strong>
                               <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Quantitative measures characterizing attention behavior for comparison across heads and layers—descriptive statistics, not quality judgments.</p>
                               <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Confidence = max weight. Focus = normalized entropy (low = concentrated, high = diffuse). Sparsity = % near-zero weights. Additional: Uniformity, Balance, Flow Change. Updates for selected Layer/Head or 'Global'. Normalization modes: 'Raw', 'Column', 'Rollout'.</p>
                               <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Metrics describe distribution shape but don't indicate whether attention is 'correct' or task-relevant.</p>
                               """,
                               subtitle=subtitle,
                               controls=[
                                   ui.download_button("export_attention_metrics_single", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                               ]),
            ui.div({"id": "metrics-chart-container"}, get_metrics_display(res, layer_idx=layer_idx, head_idx=head_idx, use_full_scale=use_full_scale, baseline_stats=baseline_stats.get(), norm_mode=norm_mode))
        )

    def dashboard_layout_helper(is_gpt2, num_layers, num_heads, clean_tokens, suffix=""):
        # Helper to generate choices dict
        def get_choices(items):
            return {str(i): f"{i}: {t}" for i, t in enumerate(items)}

        # --- Get View Mode (Snapshotted) ---
        view_mode = active_view_mode.get()
        is_advanced = view_mode == "advanced"

        # --- Word-Level Aggregation Logic ---
        res = get_active_result() if suffix == "" else get_active_result("_B")

        # Check if we should aggregate
        use_word_level = False
        try:
            # Note: wl is live because it's in the card header
            cm = active_compare_models.get()
            wl = input.word_level_toggle()
            # Allow word level in single mode too if enabled
            if wl:
                use_word_level = True
        except Exception as e:
            pass
        from ..utils import aggregate_data_to_words
        if use_word_level and res:
            res = aggregate_data_to_words(res, filter_special=True)
            if res:
                 tokens = res[0]
                 clean_tokens = tokens

        # Legend for Tokenization Mode
        tokenization_legend = ""
        if use_word_level:
            tokenization_legend = ui.div(
                {"class": "tokenization-legend", "style": "font-size: 10px; color: #6b7280; margin-bottom: 8px; font-style: italic; background: #f3f4f6; padding: 4px 8px; border-radius: 4px; display: inline-block;"},
                "Tokenization: Word-Level (Aggregated)"
            )

        # --- PANEL DEFINITIONS ---

        # 1. OVERVIEW PANEL (Preview, Predictions, Global Metrics, Radar, [Advanced: Hidden States])
        def create_overview_panel():
            if not is_advanced:
                row1 = ui.layout_columns(
                     ui.output_ui(f"render_mlm_predictions{suffix}"),
                     ui.output_ui(f"render_radar_view{suffix}"),
                     col_widths=[6, 6]
                )
                row2 = ui.layout_columns(
                     ui.output_ui(f"render_global_metrics{suffix}"),
                     col_widths=[12]
                )
                return ui.div(row1, row2)
            else:
                row1 = ui.layout_columns(
                    ui.output_ui(f"render_mlm_predictions{suffix}"),
                    ui.output_ui(f"render_radar_view{suffix}"),
                    col_widths=[6, 6]
                )
                row2 = ui.layout_columns(
                     ui.output_ui(f"render_global_metrics{suffix}"),
                     col_widths=[12]
                )
                row3 = ui.layout_columns(
                        ui.output_ui(f"render_layer_output{suffix}"),
                        col_widths=[12]
                )
                return ui.div(row1, row2, row3)

        # 2. EXPLORE ATTENTION PANEL
        def create_explore_panel():
            row1 = ui.layout_columns(
                ui.output_ui(f"attention_map{suffix}"),
                ui.output_ui(f"attention_flow{suffix}"),
                col_widths=[6, 6]
            )
            row2 = ui.layout_columns(
                ui.output_ui(f"render_tree_view{suffix}"),
                col_widths=[12]
            )
            input_list = [row1, row2]
            if suffix == "":
                input_list.append(ui.output_ui("isa_row_dynamic"))
            else:
                input_list.append(ui.output_ui(f"isa_scatter{suffix}"))
            if is_advanced:
                row_advanced = ui.layout_columns(
                    ui.output_ui(f"render_scaled_attention{suffix}"),
                    col_widths=[12]
                )
                input_list.append(ui.div(style="margin-top: 26px;"))
                input_list.append(row_advanced)
            return ui.div(*input_list)

        # 3. DEEP DIVE PANEL (ADVANCED ONLY)
        # Embeddings, Norms, FFNs - Built directly for faster loading
        def create_deep_dive_panel_bert():
            if not res:
                return ui.div("Loading...", style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;")

            try: top_k_val = int(input.global_topk())
            except: top_k_val = 3
            try: layer_idx_val = int(input.global_layer())
            except: layer_idx_val = 0
            norm_mode_val = global_norm_mode.get()
            _, _, _, _, _, _, _, encoder_model_local, *_ = res
            model_type_val = getattr(encoder_model_local.config, 'model_type', 'bert')

            return ui.div(
                # Embeddings Row
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type_val, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Maps each token ID to a learned dense vector representation (d=768 for base models) that captures semantic meaning from the vocabulary embedding matrix. At this stage, representations are context-independent—contextual disambiguation occurs in subsequent attention layers.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k_val))
                    ),
                    arrow("Token Embeddings", "Segment Embeddings", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div(
                            {"class": "card", "style": "height: 100%;"},
                            ui.h4("Segment Embeddings"),
                            ui.p("Encodes sentence membership using binary embeddings (Segment A or Segment B), enabling BERT to distinguish between tokens from different sentences. Essential for sentence-pair tasks such as Natural Language Inference, Question Answering, and Next Sentence Prediction.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                            get_segment_embedding_view(res)
                        ),
                        arrow("Segment Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                    ),
                    arrow("Segment Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Injects absolute position information into each token representation using learned embeddings. Without positional encoding, Transformers would be permutation-invariant—unable to distinguish word order. Both BERT and GPT-2 use learned (not sinusoidal) position embeddings.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k_val))
                    ),
                ),
                # Sum & Norm Row
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Computes the element-wise sum of token, positional, and segment embeddings (where applicable), then applies Layer Normalization to stabilize the activation distribution before entering the first Transformer block.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model_local))
                    ),
                    arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div(
                            {"class": "card", "style": "height: 100%;"},
                            ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                            ui.p("Transforms input hidden states into Query, Key, and Value representations through learned linear projections (Q=XW_Q, K=XW_K, V=XW_V). Queries encode 'what information to look for', Keys encode 'what information is available', and Values contain 'the information to aggregate'.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                            get_qkv_table(res, layer_idx_val, top_k=top_k_val, suffix=suffix, norm_mode=norm_mode_val, use_global=(global_metrics_mode.get() == "all"))
                        ),
                    ),
                ),
                # Residual Connections Row
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 22px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm"), ui.p("Applies residual connection (x + Sublayer(x)) followed by Layer Normalization. Residual connections enable gradient flow through deep networks; normalization stabilizes activations across layers. Applied after both attention and FFN sublayers.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_add_norm_view(res, layer_idx_val))
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), ui.p("Position-wise two-layer MLP applied independently to each token: FFN(x) = GELU(xW₁+b₁)W₂+b₂. The intermediate dimension expands to 4× the hidden size, creating a bottleneck architecture believed to store factual knowledge learned during pre-training.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_ffn_view(res, layer_idx_val))
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), ui.p("Second residual connection and Layer Normalization within the Transformer block, applied after the Feed-Forward Network. Completes the standard Transformer block architecture: Attention → Add&Norm → FFN → Add&Norm.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_add_norm_post_ffn_view(res, layer_idx_val)),
                        arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type_val, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                    ),
                ),
            )

        def create_deep_dive_panel_gpt2():
            if not res:
                return ui.div("Loading...", style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;")

            try: top_k_val = int(input.global_topk())
            except: top_k_val = 3
            try: layer_idx_val = int(input.global_layer())
            except: layer_idx_val = 0
            norm_mode_val = global_norm_mode.get()
            _, _, _, _, _, _, _, encoder_model_local, *_ = res
            model_type_val = "gpt2"

            return ui.div(
                # Embeddings Row (GPT-2: No Segment Embeddings)
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type_val, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Maps each token ID to a learned dense vector representation (d=768 for base models) that captures semantic meaning from the vocabulary embedding matrix. At this stage, representations are context-independent—contextual disambiguation occurs in subsequent attention layers.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k_val))
                    ),
                    arrow("Token Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Injects absolute position information into each token representation using learned embeddings. Without positional encoding, Transformers would be permutation-invariant—unable to distinguish word order. Both BERT and GPT-2 use learned (not sinusoidal) position embeddings.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k_val)),
                        arrow("Positional Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                    ),
                ),
                # Sum & Norm Row
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Computes the element-wise sum of token, positional, and segment embeddings (where applicable), then applies Layer Normalization to stabilize the activation distribution before entering the first Transformer block.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model_local))
                    ),
                    arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div(
                            {"class": "card", "style": "height: 100%;"},
                            ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                            ui.p("Transforms input hidden states into Query, Key, and Value representations through learned linear projections (Q=XW_Q, K=XW_K, V=XW_V). Queries encode 'what information to look for', Keys encode 'what information is available', and Values contain 'the information to aggregate'.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                            get_qkv_table(res, layer_idx_val, top_k=top_k_val, suffix=suffix, norm_mode=norm_mode_val, use_global=(global_metrics_mode.get() == "all"))
                        ),
                    ),
                ),
                # Residual Connections Row
                ui.div(
                    {"class": "flex-row-container", "style": "margin-bottom: 22px;"},
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm"), ui.p("Applies residual connection (x + Sublayer(x)) followed by Layer Normalization. Residual connections enable gradient flow through deep networks; normalization stabilizes activations across layers. Applied after both attention and FFN sublayers.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_add_norm_view(res, layer_idx_val))
                    ),
                    arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), ui.p("Position-wise two-layer MLP applied independently to each token: FFN(x) = GELU(xW₁+b₁)W₂+b₂. The intermediate dimension expands to 4× the hidden size, creating a bottleneck architecture believed to store factual knowledge learned during pre-training.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_ffn_view(res, layer_idx_val))
                    ),
                    arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                    ui.div(
                        {"class": "flex-card", "style": "position: relative;"},
                        ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), ui.p("Second residual connection and Layer Normalization within the Transformer block, applied after the Feed-Forward Network. Completes the standard Transformer block architecture: Attention → Add&Norm → FFN → Add&Norm.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_add_norm_post_ffn_view(res, layer_idx_val)),
                        arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type_val, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                    ),
                ),
            )

        # --- Build Accordion Panels ---
        accordion_panels = [
            ui.accordion_panel(
                ui.span("Overview", ui.span({"class": "accordion-panel-badge essential"}, "Essential")),
                create_overview_panel(),
                value="overview"
            ),
            ui.accordion_panel(
                ui.span("Explore Attention", ui.span({"class": "accordion-panel-badge explore"}, "Visual")),
                create_explore_panel(),
                value="explore"
            ),
        ]

        if is_advanced:
            deep_dive_content = create_deep_dive_panel_gpt2() if is_gpt2 else create_deep_dive_panel_bert()
            accordion_panels.append(
                ui.accordion_panel(
                    ui.span("Deep Dive / Internals", ui.span({"class": "accordion-panel-badge technical"}, "Technical")),
                    deep_dive_content,
                    value="deep_dive"
                )
            )

        # Use stored accordion state to preserve open panels across re-renders
        current_accordion_state = accordion_state_single.get()

        dashboard_accordion = ui.accordion(
            *accordion_panels,
            id=f"dashboard_accordion{suffix}",
            open=current_accordion_state if current_accordion_state else ["overview"],
            multiple=True,
        )

        return ui.div(
            {"id": f"dashboard-container{suffix}", "class": "dashboard-stack"},
            tokenization_legend,
            dashboard_accordion
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
        
        # Use SNAPSHOTTED compare modes instead of raw inputs to decouple sidebar
        compare_models = active_compare_models.get()
        compare_prompts = active_compare_prompts.get()

        # --- Get View Mode (Snapshotted) ---
        view_mode = active_view_mode.get()
        is_advanced = view_mode == "advanced"
        
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
        def create_preview_header(has_toggle=False, color_theme="blue", gpt2_note=None):
            # Left side content (Title + Info Tooltip)
            left_side = ui.div(
                {"style": "display: flex; align-items: center;"},
                ui.h4("Sentence Preview", style="margin: 0; margin-right: 8px;"),
                ui.div(
                    {"class": "info-tooltip-wrapper", "style": "margin-left: 6px;"},
                    ui.span({"class": "info-tooltip-icon", "style": "font-size:8px; width:14px; height:14px; line-height:14px; font-family:'PT Serif', serif;"}, "i"),
                    ui.div(
                        {"class": "info-tooltip-content"},
                        ui.HTML("""
                            <strong>Sentence Preview</strong>
                            <p>Visualizes the input text after model-specific tokenization, showing how raw text is decomposed into processable subword units.</p>
                            <div style='background:rgba(255,255,255,0.1); padding:8px; border-radius:4px; margin: 8px 0;'>
                                <strong style='color:#3b82f6;'>Tokenization Details:</strong>
                                <p style='font-size:10px; margin-top:4px;'>Tokenization differs between models: BERT uses WordPiece (splitting unknown words into subwords like 'play' + '##ing'), while GPT-2 uses Byte-Pair Encoding (BPE) operating at byte level.</p>
                                <p style='font-size:10px; margin-top:4px;'>Special tokens are model-specific: [CLS]/[SEP] for BERT, &lt;|endoftext|&gt; for GPT-2.</p>
                            </div>
                        """)
                    )
                )
            )

            # Center/Right side content (GPT-2 Note + Word Level Toggle)
            right_side_elements = []
            
            # GPT-2 note aligned to the right of the title (before the toggle)
            if gpt2_note:
                right_side_elements.append(
                    ui.span(gpt2_note, style="font-size: 9px; color: #94a3b8; font-weight: 500; font-style: italic; margin-left: auto; text-transform: none; letter-spacing: 0;")
                )

            # Word Level Toggle (appears on the far right)
            if has_toggle:
                # Add spacer if we have a GPT-2 note to push them apart
                if gpt2_note:
                    right_side_elements.append(ui.div(style="width: 12px;"))
                
                right_side_elements.append(create_word_level_toggle(color_theme))
            else:
                # Placeholder to maintain height if no toggle is present
                right_side_elements.append(ui.div(class_="toggle-placeholder", style="width: 1px;"))
            
            return ui.div(
                {"class": "viz-header-with-info", "style": "margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; min-height: 24px;"},
                left_side,
                ui.div({"style": "display: flex; align-items: center; flex-grow: 1; justify-content: flex-end;"}, *right_side_elements)
            )

        # Logic to place toggle on BERT when families differ
        toggle_on_A = False
        toggle_on_B = False
        
        # Calculate GPT-2 status for headers differently in compare modes
        actual_is_gpt2_A = False
        actual_is_gpt2_B = False

        if compare_models and res_A and res_B:
            _, _, _, _, _, _, _, encoder_model_A, *_ = res_A
            _, _, _, _, _, _, _, encoder_model_B, *_ = res_B
            actual_is_gpt2_A = not hasattr(encoder_model_A, "encoder")
            actual_is_gpt2_B = not hasattr(encoder_model_B, "encoder")
        elif compare_prompts and res_A:
            _, _, _, _, _, _, _, encoder_model_A, *_ = res_A
            actual_is_gpt2_A = not hasattr(encoder_model_A, "encoder")
            actual_is_gpt2_B = actual_is_gpt2_A

        if is_family_diff:
            # Place on whichever is NOT GPT-2 (i.e. is BERT)
            if actual_is_gpt2_A:
                toggle_on_B = True # A is GPT2, so B must be BERT
            else:
                toggle_on_A = True # A is BERT
        
        # Create headers with appropriate themes
        # Pass GPT-2 note only if in compare mode (user request)
        gpt2_note_text = "Ġ (space token) removed for visualization"
        note_A = gpt2_note_text if (compare_models or compare_prompts) and actual_is_gpt2_A else None
        note_B = gpt2_note_text if (compare_models or compare_prompts) and actual_is_gpt2_B else None

        preview_title = create_preview_header(toggle_on_A, "blue", note_A) 
        preview_title_B = create_preview_header(toggle_on_B, "pink", note_B)

        compare = compare_models or compare_prompts
        
        # Determine labels
        header_a = "Model A"
        header_b = "Model B"
        if compare_prompts:
            header_a = "Prompt A"
            header_b = "Prompt B"
        
        if not config:
            # BEFORE GENERATION - Show live preview (Safe because dashboard isn't loaded yet)
            t = input.text_input().strip()
            # Placeholder changed to '(input)' per user request
            preview_html = f'<div style="font-family:monospace;color:#6b7280;font-size:14px;">"{t}"</div>' if t else '<div style="color:#9ca3af;font-size:12px;">(input)</div>'

            if not compare:
                # Single mode live preview
                return ui.div(
                    ui.div(
                        {"class": "card"},
                        ui.h4("Sentence Preview"),
                        ui.HTML(preview_html),
                    ),
                    ui.HTML("<script>$('#generate_all').html('Generate All').prop('disabled', false).css('opacity', '1');</script>")
                )
            else:
                # Compare mode - live paired previews
                if compare_prompts:
                    try: t_b = input.text_input_B().strip()
                    except: t_b = ""
                else:
                    t_b = t # Same text if comparing models
                
                preview_b = f'<div style="font-family:monospace;color:#ff5ca9;font-size:14px;">"{t_b}"</div>' if t_b else '<div style="color:#9ca3af;font-size:12px;">(input)</div>'
                
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
                        ui.div({"class": "card", "style": "border: 2px solid #3b82f6;"}, ui.h4("Sentence Preview"), ui.HTML(preview_html)),
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
                # Use cached_text_A to avoid reactivity from input.text_input()
                return ui.div(
                    # Always use output_ui for preview - it handles grey/colored states internally
                    ui.div(
                        {"class": "card", "style": "margin-bottom: 32px;"},
                        preview_title,
                        get_preview_text_view(res, cached_text_A.get(), ""),
                    ),
                    ui.div(
                        {"style": "padding: 40px; text-align: center; color: #9ca3af;"},
                        ui.p("Generating data...", style="font-size: 14px; animation: pulse 1.5s infinite;")
                    )
                )

            # Data is ready - show preview + full dashboard
            # Reconstruct text from tokenizer to avoid reactivity
            tokens_A = res[0]
            tokenizer_A = res[6]
            text_A = tokenizer_A.convert_tokens_to_string(tokens_A)
            footer_A_main = '<span style="color:#94a3b8;">Ġ (space token) removed for visualization</span>' if is_gpt2 else ""
            return ui.div(
                ui.div(
                    {"class": "card", "style": "margin-bottom: 32px;"},
                    preview_title,
                    get_preview_text_view(res, text_A, "", footer_A_main),
                ),
                # Dashboard layout
                dashboard_layout_helper(is_gpt2, num_layers, num_heads, [], suffix="")
            )
        else:
            # SIDE-BY-SIDE MODE - Accordion Layout
            
            # Get top_k for summary tables
            try: top_k = int(input.global_topk())
            except: top_k = 3
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
                    ui.div({"class": "compare-wrapper-a", "style": "height: 100%;"}, output_a),
                    ui.div({"class": "compare-wrapper-b", "style": "height: 100%;"}, output_b),
                    col_widths=[6, 6]
                )

            # For outputs that need card wrapper (preview_text doesn't have one)
            def paired_with_card(title, output_a, output_b):
                header = ui.h4(title) if isinstance(title, str) else title
                return ui.layout_columns(
                    ui.div({"class": "card compare-card-a", "style": "height: 100%; display: flex; flex-direction: column;"}, header, output_a),
                    ui.div({"class": "card compare-card-b", "style": "height: 100%; display: flex; flex-direction: column;"}, header, output_b),
                    col_widths=[6, 6]
                )

            # Paired arrows - vertical arrows for both models
            def paired_arrows(from_section, to_section, model_type_A=None, model_type_B=None):
                return ui.layout_columns(
                    ui.div(
                        {"style": "display: flex; justify-content: center; padding: 0; margin: 0;"},
                        arrow(from_section, to_section, "vertical", suffix="_A", extra_class="arrow-blue", model_type=model_type_A)
                    ),
                    ui.div(
                        {"style": "display: flex; justify-content: center; padding: 0; margin: 0;"},
                        arrow(from_section, to_section, "vertical", suffix="_B", extra_class="arrow-pink", model_type=model_type_B)
                    ),
                    col_widths=[6, 6]
                )

            # --- PANEL DEFINITIONS ---

            def create_overview_panel_compare():
                content = [
                     # Predictions
                     paired(ui.output_ui("render_mlm_predictions"), ui.output_ui("render_mlm_predictions_B")),
                     # Radar
                     paired(ui.output_ui("render_radar_view"), ui.output_ui("render_radar_view_B")),
                     # Global Metrics
                     paired(ui.output_ui("render_global_metrics"), ui.output_ui("render_global_metrics_B")),
                ]
                
                # Hidden States (if advanced) - In Single Mode this is in Overview
                if is_advanced:
                    content.append(paired(ui.output_ui("render_layer_output"), ui.output_ui("render_layer_output_B")))
                    
                return ui.div(*content)

            def create_explore_panel_compare():
                # Base content
                content = [
                    # Map
                    paired(ui.output_ui("attention_map"), ui.output_ui("attention_map_B")),
                    # Flow
                    paired(ui.output_ui("attention_flow"), ui.output_ui("attention_flow_B")),
                    
                    # Tree
                    paired(ui.output_ui("render_tree_view"), ui.output_ui("render_tree_view_B")),
                    
                    # ISA
                    paired(ui.output_ui("isa_scatter_A_compare"), ui.output_ui("isa_scatter_B_compare"))
                ]
                
                # Advanced: Scaled Attention (if advanced)
                if is_advanced:
                     content.append(ui.div(style="height: 20px;"))
                     content.append(paired(ui.output_ui("render_scaled_attention"), ui.output_ui("render_scaled_attention_B")))
                
                return ui.div(*content)

            def create_deep_dive_panel_compare():
                 # Res unpack to check for model types for dynamic Deep Dive layout
                 _, _, _, _, _, _, _, encoder_model_A, *_ = res_A
                 _, _, _, _, _, _, _, encoder_model_B, *_ = res_B
                 is_gpt2_A = not hasattr(encoder_model_A, "encoder")
                 is_gpt2_B = not hasattr(encoder_model_B, "encoder")
                 model_type_A = "gpt2" if is_gpt2_A else "bert"
                 model_type_B = "gpt2" if is_gpt2_B else "bert"

                 # Get shared parameters
                 try: layer_idx = int(input.global_layer())
                 except: layer_idx = 0
                 norm_mode = global_norm_mode.get()
                 use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency

                 rows = []

                 # Helper to create a card with content for A or B
                 def make_card(title, desc, content, side="a"):
                     card_class = "card compare-card-a" if side == "a" else "card compare-card-b"
                     return ui.div(
                         {"class": card_class, "style": "height: 100%; display: flex; flex-direction: column;"},
                         ui.h4(title),
                         ui.p(desc, style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                         content
                     )

                 # Embeddings - Arrow must be OUTSIDE the .card to avoid overflow: hidden clipping
                 row_A = ui.div(
                     {"style": "position: relative; height: 100%;"},
                     arrow("Input", "Token Embeddings", "vertical", suffix="_A", model_type=model_type_A,
                           style="position: absolute; top: -32px; left: 50%; transform: translateX(-50%); width: auto; margin: 0; z-index: 100;"),
                      make_card("Token Embeddings", "Maps each token ID to a learned dense vector representation (d=768 for base models) that captures semantic meaning from the vocabulary embedding matrix. At this stage, representations are context-independent—contextual disambiguation occurs in subsequent attention layers.", get_embedding_table(res_A, top_k=top_k), "a")
                 )
                 row_B = ui.div(
                     {"style": "position: relative; height: 100%;"},
                     arrow("Input", "Token Embeddings", "vertical", suffix="_B", model_type=model_type_B,
                           style="position: absolute; top: -32px; left: 50%; transform: translateX(-50%); width: auto; margin: 0; z-index: 100;"),
                      make_card("Token Embeddings", "Maps each token ID to a learned dense vector representation (d=768 for base models) that captures semantic meaning from the vocabulary embedding matrix. At this stage, representations are context-independent—contextual disambiguation occurs in subsequent attention layers.", get_embedding_table(res_B, top_k=top_k, suffix="_B"), "b")
                 )
                 rows.append(ui.layout_columns(row_A, row_B, col_widths=[6, 6]))

                 next_from = "Token Embeddings"

                 # Segment Embeddings (Show ONLY if both are BERT - neither is GPT-2)
                 if not is_gpt2_A and not is_gpt2_B:
                     rows.append(ui.div(paired_arrows(next_from, "Segment Embeddings"), class_="arrow-row"))
                     seg_desc = "Encodes sentence membership using binary embeddings (Segment A or Segment B), enabling BERT to distinguish between tokens from different sentences. Essential for sentence-pair tasks such as Natural Language Inference, Question Answering, and Next Sentence Prediction."
                     rows.append(ui.layout_columns(
                         make_card("Segment Embeddings", seg_desc, get_segment_embedding_view(res_A), "a"),
                         make_card("Segment Embeddings", seg_desc, get_segment_embedding_view(res_B), "b"),
                         col_widths=[6, 6]
                     ))
                     next_from = "Segment Embeddings"

                 # Positional
                 rows.append(ui.div(paired_arrows(next_from, "Positional Embeddings", model_type_A=model_type_A, model_type_B=model_type_B), class_="arrow-row"))
                 pos_desc = "Injects absolute position information into each token representation using learned embeddings. Without positional encoding, Transformers would be permutation-invariant—unable to distinguish word order. Both BERT and GPT-2 use learned (not sinusoidal) position embeddings."
                 rows.append(paired_with_card("Positional Embeddings", make_card("Positional Embeddings", pos_desc, get_posenc_table(res_A, top_k=top_k), "a"), make_card("Positional Embeddings", pos_desc, get_posenc_table(res_B, top_k=top_k, suffix="_B"), "b")))

                 # Sum & Norm
                 rows.append(ui.div(paired_arrows("Positional Embeddings", "Sum & Layer Normalization"), class_="arrow-row"))
                 sum_desc = "Computes the element-wise sum of token, positional, and segment embeddings (where applicable), then applies Layer Normalization to stabilize the activation distribution before entering the first Transformer block."
                 rows.append(ui.layout_columns(
                     make_card("Sum & Layer Normalization", sum_desc, get_sum_layernorm_view(res_A, encoder_model_A), "a"),
                     make_card("Sum & Layer Normalization", sum_desc, get_sum_layernorm_view(res_B, encoder_model_B, suffix="_B"), "b"),
                     col_widths=[6, 6]
                 ))

                 # QKV
                 rows.append(ui.div(paired_arrows("Sum & Layer Normalization", "Q/K/V Projections"), class_="arrow-row"))
                 qkv_desc = "Transforms input hidden states into Query, Key, and Value representations through learned linear projections (Q=XW_Q, K=XW_K, V=XW_V). Queries encode 'what information to look for', Keys encode 'what information is available', and Values contain 'the information to aggregate'."
                 rows.append(ui.layout_columns(
                     make_card("Q/K/V Projections", qkv_desc, get_qkv_table(res_A, layer_idx, top_k=top_k, norm_mode=norm_mode, use_global=use_global), "a"),
                     make_card("Q/K/V Projections", qkv_desc, get_qkv_table(res_B, layer_idx, top_k=top_k, suffix="_B", norm_mode=norm_mode, use_global=use_global), "b"),
                     col_widths=[6, 6]
                 ))

                 # Add & Norm
                 rows.append(ui.div(paired_arrows("Q/K/V Projections", "Add & Norm"), class_="arrow-row"))
                 addnorm_desc = "Applies residual connection (x + Sublayer(x)) followed by Layer Normalization. Residual connections enable gradient flow through deep networks; normalization stabilizes activations across layers. Applied after both attention and FFN sublayers."
                 rows.append(ui.layout_columns(
                     make_card("Add & Norm", addnorm_desc, get_add_norm_view(res_A, layer_idx), "a"),
                     make_card("Add & Norm", addnorm_desc, get_add_norm_view(res_B, layer_idx, suffix="_B"), "b"),
                     col_widths=[6, 6]
                 ))

                 # FFN
                 rows.append(ui.div(paired_arrows("Add & Norm", "Feed-Forward Network"), class_="arrow-row"))
                 ffn_desc = "Position-wise two-layer MLP applied independently to each token: FFN(x) = GELU(xW₁+b₁)W₂+b₂. The intermediate dimension expands to 4× the hidden size, creating a bottleneck architecture believed to store factual knowledge learned during pre-training."
                 rows.append(ui.layout_columns(
                     make_card("Feed-Forward Network", ffn_desc, get_ffn_view(res_A, layer_idx), "a"),
                     make_card("Feed-Forward Network", ffn_desc, get_ffn_view(res_B, layer_idx, suffix="_B"), "b"),
                     col_widths=[6, 6]
                 ))

                 # Post FFN
                 rows.append(ui.div(paired_arrows("Feed-Forward Network", "Add & Norm (Post-FFN)"), class_="arrow-row"))
                 addnorm_post_desc = "Second residual connection and Layer Normalization within the Transformer block, applied after the Feed-Forward Network. Completes the standard Transformer block architecture: Attention → Add&Norm → FFN → Add&Norm."
                 rows.append(ui.layout_columns(
                     make_card("Add & Norm (Post-FFN)", addnorm_post_desc, get_add_norm_post_ffn_view(res_A, layer_idx), "a"),
                     make_card("Add & Norm (Post-FFN)", addnorm_post_desc, get_add_norm_post_ffn_view(res_B, layer_idx, suffix="_B"), "b"),
                     col_widths=[6, 6]
                 ))

                 # Exit Arrow
                 rows.append(ui.div(paired_arrows("Add & Norm (Post-FFN)", "Exit", model_type_A=model_type_A, model_type_B=model_type_B), class_="arrow-row"))

                 return ui.div(*rows)

            # Load results
            res_A = get_active_result()
            res_B = get_active_result("_B")

            # If strictly waiting for data, show only preview row + loading message
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

            # Render Paired Layout
            # Reconstruct text from tokenizers to avoid reactivity
            tokens_A = res_A[0]
            tokenizer_A = res_A[6]
            text_A_reconstructed = tokenizer_A.convert_tokens_to_string(tokens_A)
            tokens_B = res_B[0]
            tokenizer_B = res_B[6]
            text_B_reconstructed = tokenizer_B.convert_tokens_to_string(tokens_B)

            # Build Accordion Items
            accordion_items = [
                ui.accordion_panel(
                    ui.span("Overview", ui.span({"class": "accordion-panel-badge essential"}, "Essential")),
                    create_overview_panel_compare(),
                    value="overview"
                ),
                ui.accordion_panel(
                    ui.span("Explore Attention", ui.span({"class": "accordion-panel-badge explore"}, "Visual")),
                    create_explore_panel_compare(),
                    value="explore"
                ),
            ]
            
            if is_advanced:
                accordion_items.append(
                    ui.accordion_panel(
                        ui.span("Deep Dive / Internals", ui.span({"class": "accordion-panel-badge technical"}, "Technical")),
                        create_deep_dive_panel_compare(),
                        value="deep_dive"
                    )
                )

            # Detect GPT-2 for Model A and B independently to set footer notes for Preview
            _, _, _, _, _, _, _, encoder_model_A_local, *_ = res_A
            _, _, _, _, _, _, _, encoder_model_B_local, *_ = res_B
            
            # Note: GPT-2 notes are now handled in the header (preview_title / preview_title_B)
            # We only need to keep the tokenization info (WordPiece/BPE) in the footer if families differ.
            footer_A = tok_info_A if is_family_diff else ""
            footer_B = tok_info_B if is_family_diff else ""

            return ui.div(
                {"id": "dashboard-container-compare", "class": "dashboard-stack"},

                # Top Header: MODEL A | MODEL B
                ui.layout_columns(
                    ui.h3(header_a, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.h3(header_b, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:16px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    col_widths=[6, 6]
                ),

                # Row 0: Sentence Preview
                ui.layout_columns(
                    ui.div({"class": "card compare-card-a"}, preview_title, get_preview_text_view(res_A, text_A_reconstructed, "", footer_html=footer_A)),
                    ui.div({"class": "card compare-card-b"}, preview_title_B, get_preview_text_view(res_B, text_B_reconstructed, "_B", footer_html=footer_B)),
                    col_widths=[6, 6]
                ),
                
                # Accordion - use stored state to preserve open panels
                ui.accordion(
                    *accordion_items,
                    id="dashboard_accordion_compare",
                    open=accordion_state_compare.get() if accordion_state_compare.get() else ["overview"],
                    multiple=True,
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

        # Determine export button IDs based on suffix
        export_csv_id = "export_isa_csv_B" if suffix == "_B" else "export_isa_csv"
        export_json_id = "export_isa_json_B" if suffix == "_B" else "export_isa_json"
        png_filename = generate_export_filename("isa", "png", is_b=(suffix == "_B"), incl_timestamp=False, data_type="heatmap")
        token_png_filename = generate_export_filename("isa", "png", is_b=(suffix == "_B"), incl_timestamp=False, data_type="token2token")

        return ui.div(
            {"class": "card"},
            viz_header(
                "Inter-Sentence Attention (ISA)",
                "Heatmap of sentence-pair attention scores computed via three-level max pooling (layers → heads → tokens).",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Inter-Sentence Attention (ISA)</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Measures attention strength between sentence pairs—indicates cross-sentence attention flow, not semantic similarity or entailment.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> ISA = max_heads(max_tokens(max_layers(α_ij))) for tokens across sentence boundaries. Scores: High (&gt;0.8) = strong coupling; Medium (0.4-0.8) = moderate; Low (&lt;0.4) = independent processing. Token-by-token visualization highlights selected token's cross-sentence connections.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>BERT:</strong> Bidirectional attention — all tokens attend to all tokens regardless of position. The ISA matrix is fully populated: sentence A can attend to sentence B and vice versa, producing a roughly symmetric matrix.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>GPT-2:</strong> Causal (left-to-right) attention — each token can only attend to previous tokens due to the autoregressive mask. The ISA matrix upper triangle is near-zero: later sentences attend to earlier ones, but earlier sentences cannot attend forward.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Max aggregation captures peak attention but may miss nuanced distributed interactions; high ISA doesn't imply logical relationship.</p>
                """,
                        controls=[
                            ui.download_button(export_csv_id, "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;"),
                            ui.download_button(export_json_id, "JSON", style="padding: 2px 8px; font-size: 10px; height: 24px;"),
                            ui.tags.button(
                                "PNG",
                                onclick=f"downloadPlotlyPNG('isa-plot-container{suffix}', '{png_filename}')",
                                style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                            ),
                        ]
                    ),
            ui.div({"class": "viz-description"}, "Click any cell to inspect token-level attention between the two sentences. ⚠️ High scores reflect strong attention coupling, not semantic similarity."),
            ui.layout_columns(
                ui.div(
                    {"id": f"isa-plot-container{suffix}", "style": "width: 100%; display: flex; justify-content: center; align-items: center; margin-bottom: 20px;" if vertical_layout else "height: 500px; width: 100%; display: flex; justify-content: center; align-items: center;"},
                    ui.HTML(plot_html + js)
                ),
                ui.div(
                    {"style": "display: flex; flex-direction: column; justify-content: flex-start; height: 100%;"},
                    ui.div(ui.output_ui(f"isa_detail_info{suffix}"), style="flex: 0 0 auto; margin-bottom: 10px;"),
                    ui.div(
                        {"id": f"isa-token-container{suffix}"},
                        ui.div(
                            {"style": "display: flex; justify-content: flex-end; margin-bottom: 4px;"},
                            ui.tags.button(
                                "Token PNG",
                                onclick=f"downloadPlotlyPNG('isa-token-container{suffix}', '{token_png_filename}')",
                                style="padding: 2px 6px; font-size: 9px; height: 20px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                            ),
                        ),
                        ui.output_ui(f"isa_token_view{suffix}"),
                        style="flex: 1; display: flex; flex-direction: column;"
                    ),
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
            viz_header(
                "Inter-Sentence Attention (ISA)",
                "Heatmap of sentence-pair attention scores computed via three-level max pooling (layers → heads → tokens).",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Inter-Sentence Attention (ISA)</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Measures attention strength between sentence pairs—indicates cross-sentence attention flow, not semantic similarity or entailment.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> ISA = max_heads(max_tokens(max_layers(α_ij))) for tokens across sentence boundaries. Scores: High (&gt;0.8) = strong coupling; Medium (0.4-0.8) = moderate; Low (&lt;0.4) = independent processing. Token-by-token visualization highlights selected token's cross-sentence connections.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>BERT:</strong> Bidirectional attention — all tokens attend to all tokens regardless of position. The ISA matrix is fully populated: sentence A can attend to sentence B and vice versa, producing a roughly symmetric matrix.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>GPT-2:</strong> Causal (left-to-right) attention — each token can only attend to previous tokens due to the autoregressive mask. The ISA matrix upper triangle is near-zero: later sentences attend to earlier ones, but earlier sentences cannot attend forward.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Max aggregation captures peak attention but may miss nuanced distributed interactions; high ISA doesn't imply logical relationship.</p>
                """
            ),
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

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        layer_block = get_layer_block(encoder_model, layer_idx)

        if hasattr(layer_block, "attention"): # BERT
            num_heads = layer_block.attention.self.num_attention_heads
            num_layers = len(encoder_model.encoder.layer)
            is_causal = False
        else: # GPT-2
            num_heads = layer_block.attn.num_heads
            num_layers = len(encoder_model.h)
            is_causal = True

        # Get rollout layers setting
        use_all_layers = global_rollout_layers.get() == "all"

        # Get raw attention matrix - either specific layer/head or averaged across all
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            raw_att = np.mean(att_layers, axis=(0, 1))
        else:
            raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

        # Apply normalization based on mode
        att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal, use_all_layers=use_all_layers)

        hs_in = hidden_states[layer_idx]

        Q, K, V = extract_qkv(layer_block, hs_in)

        L = len(tokens)

        # Handle causal masking for non-rollout modes
        causal_desc = ""
        if is_causal and norm_mode != "rollout":
            # Mask upper triangle (future tokens) - rollout already handles this
            att = att.copy()
            att[np.triu_indices_from(att, k=1)] = np.nan
            causal_desc = " <span style='color:#ef4444;font-weight:bold;'>(Causal Mask Applied)</span>"
        elif is_causal and norm_mode == "rollout":
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

        # Dynamic title based on mode and normalization
        norm_label = get_norm_mode_label(norm_mode, layer_idx, use_all_layers=use_all_layers, total_layers=num_layers)
        if use_global:
            title_text = f"Attention Heatmap — Averaged (All Layers · Heads)"
        else:
            title_text = f"Attention Heatmap — Layer {layer_idx}, Head {head_idx}"

        # Add normalization indicator to title
        if norm_mode == "col":
            title_text += " · <span style='color:#8b5cf6'>Column-normalized</span>"
        elif norm_mode == "rollout":
            rollout_end = num_layers - 1 if use_all_layers else layer_idx
            title_text += f" · <span style='color:#06b6d4'>Rollout (0→{rollout_end})</span>"
        
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
            viz_header(
                "Multi-Head Attention (Heatmap)",
                "Full attention weight matrix for the selected head. Rows are query tokens, columns are key tokens.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Multi-Head Attention (Heatmap)</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Visualizes attention weight distribution showing how each token attends to others—not a direct measure of information flow or causal influence.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Matrix where cell (i,j) = attention from query i to key j. Options: specific Layer/Head or 'Global' (mean across heads). Normalization: 'Raw' (row-sum=1), 'Column' (key importance), 'Rollout' (accumulated across layers). Token click highlights row/column.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Attention weights ≠ contribution to output; gradient-based methods provide more reliable importance estimates.</p>
                """,
                show_calc_title=False,
                controls=[
                    ui.download_button("export_multi_head_data", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;"),
                    ui.tags.button(
                        "PNG",
                        onclick=f"downloadPlotlyPNG('multi-head-plot', 'multi_head_attention')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"class": "viz-description", "style": "margin-top: 20px; flex-shrink: 0;"}, ui.HTML(
                f"<strong style='color:#64748b'>{norm_label}</strong><br>" +
                ("Displays how much each token attends to every other token (rows sum to 1)." if norm_mode == "raw" else
                 "Normalized by columns: shows which tokens receive the most attention overall (columns sum to 1)." if norm_mode == "col" else
                 f"Accumulated attention flow from input through layers 0→{layer_idx}, accounting for residual connections.") +
                f" Brighter cells indicate stronger weights. ⚠️ Note that high attention ≠ importance or influence.{causal_desc}"
            )),
            ui.div(
                {"id": "multi-head-plot", "style": "flex: 1; display: flex; flex-direction: column; justify-content: center; width: 100%;"},
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
        tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0: return None

        # Check if we're in global mode
        use_global = global_metrics_mode.get() == "all"

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        # Determine if causal (GPT-2)
        layer_block = get_layer_block(encoder_model, layer_idx)
        is_causal = not hasattr(layer_block, "attention")

        # Get number of layers for rollout
        if is_causal:
            num_layers = len(encoder_model.h)
        else:
            num_layers = len(encoder_model.encoder.layer)

        # Get rollout layers setting
        use_all_layers = global_rollout_layers.get() == "all"

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

        # Get raw attention matrix - either specific layer/head or averaged across all
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            raw_att = np.mean(att_layers, axis=(0, 1))
        else:
            raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

        # Apply normalization based on mode
        att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=False, use_all_layers=use_all_layers)
        # Note: We don't apply causal masking here as flow visualization handles it differently
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

        # Build title with normalization indicator
        title_text = ""
        if norm_mode == "col":
            title_text = "<span style='color:#8b5cf6'>Column-normalized</span>"
        elif norm_mode == "rollout":
            rollout_end = num_layers - 1 if use_all_layers else layer_idx
            title_text = f"<span style='color:#06b6d4'>Rollout (0→{rollout_end})</span>"

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
            viz_header(
                "Attention Flow",
                "Directed graph connecting tokens by attention weight. Only connections ≥0.04 displayed.", 
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Flow</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Sankey-style diagram showing attention distribution between tokens—illustrates weight patterns, not actual information propagation.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Line width proportional to attention weight α_ij. Only connections ≥0.04 threshold displayed. Options: specific Layer/Head or 'Global' (mean across heads). Normalization: 'Raw' (row-sum=1), 'Column' (key importance), 'Rollout' (accumulated across layers). Token click highlights connected flows.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Visual emphasis on strong connections may obscure important distributed patterns; threshold filtering hides weak but potentially meaningful attention.</p>
                """,
                show_calc_title=False,
                controls=[
                    ui.download_button("export_attention_flow_data", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;"),
                    ui.tags.button(
                        "PNG",
                        onclick=f"downloadPlotlyPNG('attention-flow-plot', 'attention_flow')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"class": "viz-description"}, ui.HTML(
                f"<strong style='color:#64748b'>{get_norm_mode_label(norm_mode, layer_idx, use_all_layers=use_all_layers, total_layers=num_layers)}</strong><br>" +
                "Traces attention weight patterns between tokens. Thicker lines indicate stronger attention. ⚠️ This shows weight distribution, not actual information flow through the network."
            )),
            ui.div(
                {"id": "attention-flow-plot", "style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
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
                {"id": "attention-flow-plot-B", "style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot_B"))
            )
        )

    @output
    @render.ui
    def render_radar_view():
        res = get_active_result()
        if not res: return None
        # Restore input dependency for interactive updates
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        # Use global mode from floating bar
        use_global = global_metrics_mode.get() == "all"
        mode = "cluster" if use_global else "single"

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
            viz_header(
                "Head Specialization",
                "Radar chart profiling this head's attention distribution across 7 linguistic dimensions.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Head Specialization</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Profiles what linguistic patterns each attention head focuses on—an approximation of functional specialization, not ground truth.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention mass aggregated by token category (Syntax, Semantics, Entities, Punctuation, CLS, Long-range, Self-attention) using POS tagging. Updates for selected Layer/Head; 'Global' view displays head specialization clusters across all heads.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> POS-based categorization is heuristic; heads may capture patterns not aligned with traditional linguistic categories.</p>
                """,
                controls=[
                    ui.download_button("export_head_spec_unique_legacy", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick="downloadPlotlyPNG('radar-chart-container-legacy', 'head_specialization')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"id": "radar-chart-container-legacy"}, head_specialization_radar(res, layer_idx, head_idx, mode)),
             ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; margin-top: auto; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center;">
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
    def get_influence_tree_ui(res, root_idx=0, layer_idx=0, head_idx=0, suffix="", use_global=False, max_depth=3, top_k=3, norm_mode="raw"):
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
            att_override = None
            if use_global:
                # Calculate average attention matrix across all layers/heads
                # attentions is list of (1, heads, seq, seq) tensors
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                # mean over layers -> (heads, seq, seq)
                # mean over heads -> (seq, seq)
                att_override = np.mean([np.mean(l, axis=0) for l in att_layers], axis=0)
            
            tree_data = get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, max_depth, norm_mode=norm_mode, att_matrix_override=att_override)
            
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

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        tree_png_filename = generate_export_filename("attention_tree", "png", is_b=False, incl_timestamp=False, data_type="dependency")
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            viz_header("Attention Dependency Tree",
                        "Recursive expansion of the focus token's top-k attention connections into a multi-level tree.",
                        """
                        <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Dependency Tree</strong>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Hierarchical view of which tokens the selected focus token attends to most strongly—shows attention structure, not syntactic dependencies.</p>
                        <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Tree built recursively from attention weights starting at the root token (selectable via floating toolbar). Each branch shows parent→child attention weight. Node size reflects attention strength.</p>
                        <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Tree structure imposes hierarchy on non-hierarchical attention; multiple strong connections may be underrepresented.</p>
                        """,
                        controls=[
                            ui.download_button("export_tree_data_legacy", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                            ui.tags.button(
                                "PNG",
                                onclick=f"downloadD3PNG('tree-viz-container-legacy', '{tree_png_filename}')",
                                style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                            )
                        ]
            ),
            ui.div({"class": "viz-description"}, "Edge labels show attention weights; node size reflects strength. Use the floating toolbar to change the root token. ⚠️ This is attention structure, not syntactic parsing."),
            ui.div({"id": "tree-viz-container-legacy"}, get_influence_tree_ui(res, root_idx, layer_idx, head_idx, suffix="", use_global=use_global, max_depth=top_k_val, top_k=top_k_val, norm_mode=norm_mode))
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
        try: top_k = int(input.global_topk())
        except: top_k = 3
        return get_embedding_table(res, top_k=top_k, suffix="_B")

    @output
    @render.ui
    def render_segment_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Segment Embeddings"),
            ui.p("Encodes sentence membership (A or B), allowing BERT to reason about relationships between sentence pairs. When more than two sentences are provided, all sentences beyond the first are treated as Sentence B.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_segment_embedding_view(res)
        )

    @output
    @render.ui
    def render_posenc_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: top_k = int(input.global_topk())
        except: top_k = 3
        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.h4("Positional Embeddings"),
            ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"),
            get_posenc_table(res, top_k=top_k, suffix="_B")
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
            ui.p("Sum of all embeddings + layer normalization.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_sum_layernorm_view(res, encoder_model, suffix="_B")
        )

    @output
    @render.ui
    def render_qkv_table_B():
        res = get_active_result("_B")
        if not res:
            return None
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: top_k = int(input.global_topk())
        except: top_k = 3

        # Get normalization mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            ui.div(
                {"class": "header-simple"},
                ui.h4("Q/K/V Projections")
            ),
            ui.p("Query, Key, Value projections with magnitude, alignment, and directional analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_qkv_table(res, layer_idx, top_k=top_k, suffix="_B", norm_mode=norm_mode, use_global=use_global)
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

        # Get normalization mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"

        return ui.div(
            {"class": "card", "style": "height: 100%;"},
            viz_header(
                "Scaled Dot-Product Attention",
                "Per-token breakdown of Q·K scoring and softmax weighting for the selected query position.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Scaled Dot-Product Attention</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Shows how attention scores are computed between token pairs, determining which tokens influence each other's representations.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention(Q,K,V) = softmax(QK^T / √d_k)V. Dot product measures Query-Key similarity; scaling by √d_k prevents softmax saturation; output weights sum to 1 per row. Token selector focuses on specific query position; adjustable top-k controls how many key connections are displayed.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Raw attention scores don't account for Value vector magnitudes—high attention weight doesn't guarantee high influence on the output.</p>
                """,
                subtitle=f"(Layer {layer_idx} · Head {head_idx})",
                controls=[
                    ui.download_button("export_scaled_attention_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                ]
            ),
            get_scaled_attention_view(res, layer_idx, head_idx, focus_indices, top_k=top_k, norm_mode=norm_mode, use_global=use_global)
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
            get_ffn_view(res, layer, suffix="_B")
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
            get_add_norm_view(res, layer, suffix="_B")
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
            ui.p("Second residual connection and Layer Normalization within the Transformer block, applied after the Feed-Forward Network. Completes the standard Transformer block architecture: Attention → Add&Norm → FFN → Add&Norm.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_add_norm_post_ffn_view(res, layer, suffix="_B")
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
            ui.p("Contextualized vector representations output by each Transformer layer. Unlike static embeddings, these encode both the token's original meaning and information aggregated from other positions through attention. Early layers capture syntax; deeper layers encode abstract semantics.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
            get_layer_output_view(res, num_layers - 1, suffix="_B")
        )

    @output
    @render.ui
    def render_mlm_predictions_B():
        res = get_active_result("_B")
        if not res:
            return None

        try:
            is_cpm = active_compare_prompts.get()
        except:
            is_cpm = False

        # Determine if we should show predictions based on the ACTUAL loaded model for B
        # Use res_B directly if it exists, otherwise use Model Manager? No, res is source of truth.
        _, _, _, _, _, _, _, encoder_model_B_local, *_ = res
        is_gpt2_B_local = not hasattr(encoder_model_B_local, "encoder")
        model_family = "gpt2" if is_gpt2_B_local else "bert"

        # Reconstruct text from tokenizer to avoid reactivity from input.text_input()
        tokens = res[0]
        tokenizer = res[6]
        text = tokenizer.convert_tokens_to_string(tokens)

        is_bert = model_family != "gpt2"
        use_mlm = True  # Always show predictions

        try: top_k = int(input.global_topk())
        except: top_k = 3

        # Interactive Mode Logic (BERT only)
        manual_mode = False
        custom_mask_indices = None

        if is_bert:
            try: manual_mode = input.mlm_interactive_mode_B()
            except: manual_mode = False

            if manual_mode:
                try: custom_mask_indices = input.manual_mask_indices_B()
                except: custom_mask_indices = None

        # JS for Interactive Masking (Model B)
        js_script_B = ui.HTML("""
        <script>
        // Ensure toggleMask function exists (may already be defined by Model A)
        if (typeof window.toggleMask === 'undefined') {
            window.toggleMask = function(index, modelPrefix) {
                modelPrefix = modelPrefix || 'A';
                const chip = document.querySelector(`.token-chip[data-idx="${index}"][data-prefix="${modelPrefix}"]`);
                if (chip) {
                    chip.click();
                } else {
                    $(`[data-index="${index}"][data-model="${modelPrefix}"]`).toggleClass('masked-active');
                }
            };
        }

        // Handler for Model B Predict Masked button
        $(document).off('click', '#run_custom_mask_B').on('click', '#run_custom_mask_B', function() {
            var indices = [];
            $('.maskable-token[data-model="B"].masked-active').each(function() {
                indices.push(parseInt($(this).attr('data-index')));
            });
            Shiny.setInputValue('manual_mask_indices_B', indices, {priority: 'event'});
        });
        </script>
        """)

        # Prepare viz_header arguments
        if is_bert:
            title = "Masked Token Predictions (MLM)"
            desc = "Demonstrates BERT's bidirectional language understanding by iteratively masking each token and predicting the most likely original based on full surrounding context from both directions."
            tooltip = "<b>Masked Language Modeling:</b> We use BERT's Masked Language Modeling capability. To get these results, we iteratively mask each token in the sequence one by one and ask the model to predict the most likely original token based on the surrounding context (left and right)."
            
            # Interactive Mode Toggle
            try: manual_mode = input.mlm_interactive_mode_B()
            except: manual_mode = False
            
            if manual_mode:
                try: custom_mask_indices = input.manual_mask_indices_B()
                except: custom_mask_indices = None
            
            # Custom Toggle Button for Interactive Mode (Style: Norm/Similarity)
            # User wants "Toggle Masks" text, pink style like Norm buttons.
            # We use radio-group styling for consistency.
            active_class = "active" if manual_mode else ""
            
            # Inject CSS for the button to ensure it looks correct even if classes are missing
            # Note: We reuse the same class names but ensure style is present
            button_style = """
            <style>
            .toggle-masks-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 4px 12px;
                background: #f1f5f9;
                color: #64748b;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid transparent;
                user-select: none;
            }
            .toggle-masks-btn:hover {
                background: #e2e8f0;
                color: #334155;
            }
            .toggle-masks-btn.active {
                background: #ff5ca9; /* Pink */
                color: white;
                box-shadow: 0 2px 6px rgba(255, 92, 169, 0.3);
            }
            .toggle-masks-btn.active:hover {
                background: #f43f8e;
            }
            </style>
            """
            
            extra_controls = [ui.HTML(f"""
            {button_style}
            <div class='control-group' style='display:flex; align-items:center;'>
                <div class='radio-group'>
                    <span class='toggle-masks-btn {active_class}' 
                          onclick="Shiny.setInputValue('mlm_interactive_mode_B', !{str(manual_mode).lower()}, {{priority: 'event'}});">
                        {'Go back' if manual_mode else 'Toggle Masks'}
                    </span>
                </div>
            </div>
            """)]
        else:
            title = "Next Token Predictions (Causal)"
            desc = "Shows GPT-2's autoregressive next-token probability distribution, predicting what token is most likely to follow the sequence based solely on preceding (left) context."
            tooltip = "Causal Language Modeling: GPT-2 predicts P(token_t | token_1...token_{t-1}) using only left context due to causal masking. This enables text generation but prevents direct use of future context, unlike BERT's bidirectional approach."
            extra_controls = None

        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            js_script_B,
            viz_header(
                title, 
                desc, 
                tooltip, 
                controls=extra_controls
            ),
            get_output_probabilities(res, use_mlm, text, suffix="_B", top_k=top_k, manual_mode=manual_mode, custom_mask_indices=custom_mask_indices)
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
        
        # Use global mode from floating bar for consistency with Model A
        use_global = global_metrics_mode.get() == "all"
        mode = "cluster" if use_global else "single"
        
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            viz_header(
                "Head Specialization",
                "Radar chart profiling this head's attention distribution across 7 linguistic dimensions.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Head Specialization</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Profiles what linguistic patterns each attention head focuses on—an approximation of functional specialization, not ground truth.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Attention mass aggregated by token category (Syntax, Semantics, Entities, Punctuation, CLS, Long-range, Self-attention) using POS tagging. Updates for selected Layer/Head; 'Global' view displays head specialization clusters across all heads.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> POS-based categorization is heuristic; heads may capture patterns not aligned with traditional linguistic categories.</p>
                """,
                controls=[
                    ui.download_button("export_head_spec_unique_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick="downloadPlotlyPNG('radar-chart-container-B', 'head_specialization_B')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"id": "radar-chart-container-B"}, head_specialization_radar(res, layer_idx, head_idx, mode, suffix="_B")),
            ui.HTML(f"""
                <div class="radar-explanation" style="font-size: 11px; color: #64748b; line-height: 1.6; padding: 12px; background: white; border-radius: 8px; margin-top: 16px; border: 1px solid #e2e8f0; padding-bottom: 4px; text-align: center;">
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

        # Get normalization mode and global mode
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency

        clean_tokens = [t.replace("##", "") if t.startswith("##") else t.replace("Ġ", "") for t in tokens_B]
        choices = {str(i): f"{i}: {t}" for i, t in enumerate(clean_tokens)}

        tree_png_filename_b = generate_export_filename("attention_tree", "png", is_b=True, incl_timestamp=False, data_type="dependency")
        return ui.div(
            {"class": "card card-compact-height", "style": "height: 100%;"},
            viz_header(
                "Attention Dependency Tree",
                "Recursive expansion of the focus token's top-k attention connections into a multi-level tree.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Dependency Tree</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Hierarchical view of which tokens the selected focus token attends to most strongly—shows attention structure, not syntactic dependencies.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Tree built recursively from attention weights starting at the root token (selectable via floating toolbar). Each branch shows parent→child attention weight. Node size reflects attention strength.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Tree structure imposes hierarchy on non-hierarchical attention; multiple strong connections may be underrepresented.</p>
                """,
                controls=[
                    ui.download_button("export_tree_data_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick=f"downloadD3PNG('tree-viz-container-B', '{tree_png_filename_b}')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"id": "tree-viz-container-B"}, get_influence_tree_ui(res, root_idx, layer_idx, head_idx, suffix="_B", use_global=use_global, top_k=top_k, max_depth=top_k, norm_mode=norm_mode))
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

        # Get Scale
        try: use_full_scale = input.global_scale_full()
        except: use_full_scale = False

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        return ui.div(
            {"class": "card"},
            viz_header(
                "Attention Metrics",
                "Summary statistics for the selected head's attention distribution, or global aggregate across all heads.",
                """
                <strong style='color:#ff5ca9;font-size:13px;display:block;margin-bottom:8px'>Attention Metrics</strong>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Definition:</strong> Quantitative measures characterizing attention behavior for comparison across heads and layers—descriptive statistics, not quality judgments.</p>
                <p style='margin:0 0 10px 0'><strong style='color:#3b82f6'>Calculation:</strong> Confidence = max weight. Focus = normalized entropy (low = concentrated, high = diffuse). Sparsity = % near-zero weights. Additional: Uniformity, Balance, Flow Change. Updates for selected Layer/Head or 'Global'. Normalization modes: 'Raw', 'Column', 'Rollout'.</p>
                <p style='margin:0'><strong style='color:#3b82f6'>Limitation:</strong> Metrics describe distribution shape but don't indicate whether attention is 'correct' or task-relevant.</p>
                """,
                subtitle=subtitle,
                controls=[
                    ui.download_button("export_attention_metrics_single_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px;")
                ]
            ),

            get_metrics_display(res, layer_idx=layer_idx, head_idx=head_idx, use_full_scale=use_full_scale, baseline_stats=baseline_stats.get(), norm_mode=norm_mode)
        )

    @output
    @render.ui
    def attention_map_B():
        res = get_active_result("_B")
        if not res:
            return None
        tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0:
            return None

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        layer_block = get_layer_block(encoder_model, layer_idx)
        is_causal = not hasattr(layer_block, "attention")

        # Get number of layers
        if is_causal:
            num_layers = len(encoder_model.h)
        else:
            num_layers = len(encoder_model.encoder.layer)

        # Get rollout layers setting
        use_all_layers = global_rollout_layers.get() == "all"

        # Get raw attention
        use_global = global_metrics_mode.get() == "all"
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            raw_att = np.mean(att_layers, axis=(0, 1))
        else:
            raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

        # Apply normalization
        att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal, use_all_layers=use_all_layers)

        # Apply causal masking for non-rollout modes
        if is_causal and norm_mode != "rollout":
            att = att.copy()
            att[np.triu_indices_from(att, k=1)] = np.nan

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
        # Dynamic title based on mode and normalization
        norm_label = get_norm_mode_label(norm_mode, layer_idx, use_all_layers=use_all_layers, total_layers=num_layers)
        if use_global:
            title_text = f"Attention Heatmap — Averaged (All Layers · Heads)"
        else:
            title_text = f"Attention Heatmap — Layer {layer_idx}, Head {head_idx}"

        # Add normalization indicator to title
        if norm_mode == "col":
            title_text += " · <span style='color:#8b5cf6'>Column-normalized</span>"
        elif norm_mode == "rollout":
            rollout_end = num_layers - 1 if use_all_layers else layer_idx
            title_text += f" · <span style='color:#06b6d4'>Rollout (0→{rollout_end})</span>"
        
        fig.update_layout(
             title=dict(
                text=title_text,
                x=0.5,
                y=0.98,
                xanchor='center',
                yanchor='top',
                font=dict(size=14, color="#334155")
            ),
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

        )

        # Get normalization label
        norm_label = get_norm_mode_label(norm_mode, layer_idx, use_all_layers=use_all_layers, total_layers=num_layers)

        return ui.div(
            {"class": "card", "style": "height: 100%; display: flex; flex-direction: column;"},
            viz_header(
                "Multi-Head Attention",
                "",
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
                            """),
                controls=[
                    ui.download_button("export_heatmap_data_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick="downloadPlotlyPNG('attention_heatmap_B', 'multi_head_attention_B')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"class": "viz-description", "style": "margin-top: 20px; flex-shrink: 0;"}, ui.HTML(
                f"<strong style='color:#64748b'>{norm_label}</strong><br>" +
                ("Displays how much each token attends to every other token (rows sum to 1)." if norm_mode == "raw" else
                 "Normalized by columns: shows which tokens receive the most attention overall (columns sum to 1)." if norm_mode == "col" else
                 f"Accumulated attention flow from input through layers 0→{layer_idx}, accounting for residual connections.") +
                " Darker cells indicate stronger weights. ⚠️ Note that high attention ≠ importance or influence."
            )),
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
        tokens, _, _, attentions, hidden_states, _, _, encoder_model, *_ = res
        if attentions is None or len(attentions) == 0:
            return None

        # Get normalization mode
        norm_mode = global_norm_mode.get()

        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0

        # Determine if causal
        layer_block = get_layer_block(encoder_model, layer_idx)
        is_causal = not hasattr(layer_block, "attention")

        # Get number of layers
        if is_causal:
            num_layers = len(encoder_model.h)
        else:
            num_layers = len(encoder_model.encoder.layer)

        # Get rollout layers setting
        use_all_layers = global_rollout_layers.get() == "all"

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

        # Get raw attention and apply normalization - check global mode
        use_global = global_metrics_mode.get() == "all"
        if use_global:
            # Average attention across all layers and all heads
            att_layers = [layer[0].cpu().numpy() for layer in attentions]
            raw_att = np.mean(att_layers, axis=(0, 1))
        else:
            raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

        att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=False, use_all_layers=use_all_layers)
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

        # Build title with normalization indicator
        rollout_end = num_layers - 1 if use_all_layers else layer_idx
        title_text = ""
        if norm_mode == "col":
            title_text = "<span style='color:#8b5cf6'>Column-normalized</span>"
        elif norm_mode == "rollout":
            title_text = f"<span style='color:#06b6d4'>Rollout (0→{rollout_end})</span>"

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
            viz_header(
                "Attention Flow",
                "",
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
                            """),
                controls=[
                    ui.download_button("export_flow_data_B", "CSV", style="padding: 2px 8px; font-size: 10px; height: 24px; display: inline-flex; align-items: center; justify-content: center;"),
                    ui.tags.button(
                        "PNG",
                        onclick="downloadPlotlyPNG('attention_flow_plot_B', 'attention_flow_B')",
                        style="padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
                    )
                ]
            ),
            ui.div({"class": "viz-description", "style": "margin-top: -5px;"}, ui.HTML(
                f"<strong style='color:#64748b'>{get_norm_mode_label(norm_mode, layer_idx)}</strong><br>" +
                "Traces attention weight patterns between tokens. Thicker lines indicate stronger attention. ⚠️ This shows weight distribution, not actual information flow through the network."
            )),
            ui.div(
                {"id": "attention-flow-plot-B-legacy", "style": "width: 100%; overflow-x: auto; overflow-y: hidden;"},
                ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False, div_id="attention_flow_plot_B"))
            )
        )

    @output
    @render.ui
    def render_deep_dive_bert_atomic():
        res = get_active_result()
        if not res:
            return ui.div(
                "Loading...",
                style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;"
            )
        suffix = ""

        try: top_k = int(input.global_topk())
        except: top_k = 3
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency
        _, _, _, _, _, _, _, encoder_model, *_ = res
        model_type = getattr(encoder_model.config, 'model_type', 'bert')

        return ui.div(
            # Embeddings Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k))
                ),
                arrow("Token Embeddings", "Segment Embeddings", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.h4("Segment Embeddings"),
                        ui.p("Encodes sentence membership (A or B).", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_segment_embedding_view(res)
                    ),
                    arrow("Segment Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                ),
                arrow("Segment Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k))
                ),
            ),
            # Sum & Norm Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Sum of all embeddings + layer normalization.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model))
                ),
                arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                        ui.p("Query, Key, Value projections analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_qkv_table(res, layer_idx, top_k=top_k, suffix=suffix, norm_mode=norm_mode, use_global=(global_metrics_mode.get() == "all"))
                    ),

                ),
            ),
            # Residual Connections Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 22px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Pre-FFN)"), get_add_norm_view(res, layer_idx))
                ),
                arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), get_ffn_view(res, layer_idx))
                ),
                arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), get_add_norm_post_ffn_view(res, layer_idx)),
                    arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                ),
            ),
        )

    @output
    @render.ui
    def render_deep_dive_bert_atomic_B():
        res = get_active_result("_B")
        if not res:
            return ui.div(
                "Loading...",
                style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;"
            )
        suffix = "_B"

        try: top_k = int(input.global_topk())
        except: top_k = 3
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency
        _, _, _, _, _, _, _, encoder_model, *_ = res
        model_type = getattr(encoder_model.config, 'model_type', 'bert')

        return ui.div(
            # Embeddings Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k, suffix=suffix))
                ),
                arrow("Token Embeddings", "Segment Embeddings", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.h4("Segment Embeddings"),
                        ui.p("Encodes sentence membership (A or B).", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_segment_embedding_view(res)
                    ),
                    arrow("Segment Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                ),
                arrow("Segment Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k, suffix=suffix))
                ),
            ),
            # Sum & Norm Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Sum of all embeddings + layer normalization.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model, suffix=suffix))
                ),
                arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                        ui.p("Query, Key, Value projections analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_qkv_table(res, layer_idx, top_k=top_k, suffix=suffix, norm_mode=norm_mode)
                    ),

                ),
            ),
            # Residual Connections Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 25px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Pre-FFN)"), get_add_norm_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -35px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), get_ffn_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), get_add_norm_post_ffn_view(res, layer_idx, suffix=suffix)),
                    arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                ),
            ),
        )

    @output
    @render.ui
    def render_deep_dive_gpt2_atomic():
        res = get_active_result()
        if not res:
            return ui.div(
                "Loading...",
                style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;"
            )
        suffix = ""

        try: top_k = int(input.global_topk())
        except: top_k = 3
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency
        _, _, _, _, _, _, _, encoder_model, *_ = res
        model_type = getattr(encoder_model.config, 'model_type', 'gpt2')

        return ui.div(
            # Embeddings Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k, suffix=suffix))
                ),
                ui.div(
                    {"style": "position: relative; display: flex; align-items: center; justify-content: center;"},
                    arrow("Token Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    arrow("Positional Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                ),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k, suffix=suffix))
                ),
            ),
            # Sum & Norm Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Sum of all embeddings + layer normalization.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model, suffix=suffix))
                ),
                arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                        ui.p("Query, Key, Value projections analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_qkv_table(res, layer_idx, top_k=top_k, suffix=suffix, norm_mode=norm_mode, use_global=(global_metrics_mode.get() == "all"))
                    ),

                ),
            ),
            # Residual Connections Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 25px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Pre-FFN)"), get_add_norm_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -35px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), get_ffn_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), get_add_norm_post_ffn_view(res, layer_idx, suffix=suffix)),
                    arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                ),
            ),
        )

    @output
    @render.ui
    def render_deep_dive_gpt2_atomic_B():
        res = get_active_result("_B")
        if not res:
            return ui.div(
                "Loading...",
                style="color: #cbd5e1; font-size: 18px; text-align: center; padding: 50px;"
            )
        suffix = "_B"

        try: top_k = int(input.global_topk())
        except: top_k = 3
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        norm_mode = global_norm_mode.get()
        use_global = global_metrics_mode.get() == "all"  # Ensure reactive dependency
        _, _, _, _, _, _, _, encoder_model, *_ = res
        model_type = getattr(encoder_model.config, 'model_type', 'gpt2')

        return ui.div(
            # Embeddings Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px; margin-top: 15px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Input", "Token Embeddings", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; top: -28px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Token Embeddings"), ui.p("Token Lookup (Meaning)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_embedding_table(res, top_k=top_k, suffix=suffix))
                ),
                ui.div(
                    {"style": "position: relative; display: flex; align-items: center; justify-content: center;"},
                    arrow("Token Embeddings", "Positional Embeddings", "horizontal", suffix=suffix),
                    arrow("Positional Embeddings", "Sum & Layer Normalization", "vertical", suffix=suffix, style="position: absolute; bottom: -25px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;")
                ),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Positional Embeddings"), ui.p("Position Lookup (Order)", style="font-size:11px; color:#6b7280; margin-bottom:8px;"), get_posenc_table(res, top_k=top_k, suffix=suffix))
                ),
            ),
            # Sum & Norm Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 26px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Sum & Layer Normalization"), ui.p("Sum of all embeddings + layer normalization.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"), get_sum_layernorm_view(res, encoder_model, suffix=suffix))
                ),
                arrow("Sum & Layer Normalization", "Q/K/V Projections", "horizontal", suffix=suffix, style="margin-top: 15px;"),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div(
                        {"class": "card", "style": "height: 100%;"},
                        ui.div({"class": "header-simple"}, ui.h4("Q/K/V Projections")),
                        ui.p("Query, Key, Value projections analysis.", style="font-size:10px; color:#6b7280; margin-bottom:8px;"),
                        get_qkv_table(res, layer_idx, top_k=top_k, suffix=suffix, norm_mode=norm_mode, use_global=(global_metrics_mode.get() == "all"))
                    ),

                ),
            ),
            # Residual Connections Row
            ui.div(
                {"class": "flex-row-container", "style": "margin-bottom: 25px;"},
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Pre-FFN)"), get_add_norm_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Add & Norm", "Feed-Forward Network", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    arrow("Q/K/V Projections", "Add & Norm", "vertical", suffix=suffix, style="position: absolute; top: -35px; left: 50%; transform: translateX(-50%) rotate(45deg); width: auto; margin: 0; z-index: 10;"),
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Feed-Forward Network"), get_ffn_view(res, layer_idx, suffix=suffix))
                ),
                arrow("Feed-Forward Network", "Add & Norm (post-FFN)", "horizontal", suffix=suffix),
                ui.div(
                    {"class": "flex-card", "style": "position: relative;"},
                    ui.div({"class": "card", "style": "height: 100%;"}, ui.h4("Add & Norm (Post-FFN)"), get_add_norm_post_ffn_view(res, layer_idx, suffix=suffix)),
                    arrow("Add & Norm (post-FFN)", "Exit", "vertical", suffix=suffix, model_type=model_type, style="position: absolute; bottom: -30px; left: 50%; transform: translateX(-50%); width: auto; margin: 0;")
                ),
            ),
        )

    @auto_save_download("attention_heatmap", "csv")
    def export_attention_metrics_dashboard():
        res = get_active_result()
        if not res: 
            return None # Should handle "No data" gracefully instead of crashing
        try:
            tokens, _, _, attentions, _, _, _, encoder_model, *_ = res
            
            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0
            
            norm_mode = global_norm_mode.get()
            use_global = global_metrics_mode.get() == "all"
            use_all_layers = global_rollout_layers.get() == "all"
            
            is_causal = not hasattr(encoder_model, "encoder")
            
            if use_global:
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                raw_att = np.mean(att_layers, axis=(0, 1))
            else:
                # Fix: Access raw_att correctly for single head. 
                # attentions[layer_idx] is (batch, num_heads, seq, seq). We need [0, head_idx]
                # If attentions is a list of tensors
                raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()
                
            att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal, use_all_layers=use_all_layers)
            
            # Clean tokens for CSV headers/indices
            clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]
            
            import pandas as pd
            df = pd.DataFrame(att, index=clean_tokens, columns=clean_tokens)
            yield df.to_csv(index=True)
        except Exception as e:
            traceback.print_exc()
            yield f"Error exporting metrics: {str(e)}"

    @auto_save_download("scaled_attention", "csv", data_type="qkv_scores")
    def export_scaled_attention():
        """Export scaled dot-product attention data as CSV."""
        res = get_active_result()
        if not res:
            yield "No data available"
            return

        try:
            tokens, _, _, attentions, _, _, _, encoder_model, *_ = res

            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0

            norm_mode = global_norm_mode.get()
            use_global = global_metrics_mode.get() == "all"
            is_causal = not hasattr(encoder_model, "encoder")

            if use_global:
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                raw_att = np.mean(att_layers, axis=(0, 1))
            else:
                raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

            att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal)

            clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]

            import pandas as pd
            df = pd.DataFrame(att, index=clean_tokens, columns=clean_tokens)
            df.index.name = "Query_Token"
            yield df.to_csv(index=True)
        except Exception as e:
            traceback.print_exc()
            yield f"Error exporting data: {str(e)}"

    @auto_save_download("scaled_attention", "csv", is_b=True, data_type="qkv_scores")
    def export_scaled_attention_B():
        """Export scaled dot-product attention data as CSV for Model B."""
        res = get_active_result("_B")
        if not res:
            yield "No data available"
            return

        try:
            tokens, _, _, attentions, _, _, _, encoder_model, *_ = res

            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0

            norm_mode = global_norm_mode.get()
            use_global = global_metrics_mode.get() == "all"
            is_causal = not hasattr(encoder_model, "encoder")

            if use_global:
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                raw_att = np.mean(att_layers, axis=(0, 1))
            else:
                raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

            att = get_normalized_attention(raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal)

            clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]

            import pandas as pd
            df = pd.DataFrame(att, index=clean_tokens, columns=clean_tokens)
            df.index.name = "Query_Token"
            yield df.to_csv(index=True)
        except Exception as e:
            traceback.print_exc()
            yield f"Error exporting data: {str(e)}"

    def get_head_spec_csv(is_b=False):
        if is_b:
            res = get_active_result("_B")
        else:
            res = get_active_result()
            
        if not res: return None
        tokens, _, _, attentions, _, _, tokenizer, encoder_model, *_ = res
        
        # Determine logical params (layer/head)
        try: layer_idx = int(input.global_layer())
        except: layer_idx = 0
        try: head_idx = int(input.global_head())
        except: head_idx = 0
        
        is_gpt2 = not hasattr(encoder_model, "encoder")
        
        # We need text for POS tagging
        if hasattr(tokenizer, "convert_tokens_to_string"):
            text = tokenizer.convert_tokens_to_string(tokens)
        else:
            # Fallback for some tokenizers
            text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        
        from ..head_specialization import compute_all_heads_specialization
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')

        # Compute metrics for ALL heads (User Request)
        all_metrics = compute_all_heads_specialization(attentions, tokens, text)
        
        # Flatten to DataFrame
        rows = []
        # Structure: {layer_idx: {head_idx: {metric: val}}}
        for l_idx, heads in all_metrics.items():
            for h_idx, metrics in heads.items():
                row = {
                    "Layer": l_idx,
                    "Head": h_idx,
                    **metrics
                }
                rows.append(row)

        import pandas as pd
        df = pd.DataFrame(rows)
        # Reorder columns slightly for readability
        cols = ["Layer", "Head"] + [c for c in df.columns if c not in ["Layer", "Head"]]
        df = df[cols]
        return df.to_csv(index=False)

    @auto_save_download("head_specialization", "csv")
    def export_head_spec_unique():
        yield get_head_spec_csv()

    @auto_save_download("head_specialization_legacy", "csv")
    def export_head_spec_unique_legacy():
        yield get_head_spec_csv()

    @auto_save_download("head_specialization", "csv", is_b=True)
    def export_head_spec_unique_B():
        yield get_head_spec_csv(is_b=True)
    
    def get_tree_csv(res, suffix="", all_layers_heads=True):
        """
        Export attention dependency tree as CSV.

        If all_layers_heads=True, exports data for ALL layers, heads, and global mode.
        Format: Layer, Head, Mode, Parent, Child, Attention, Depth
        """
        if not res: return None

        try: root_idx = int(input.global_focus_token())
        except: root_idx = 0
        if root_idx == -1: root_idx = 0

        try: top_k = int(input.global_topk())
        except: top_k = 3

        norm_mode = global_norm_mode.get()

        from .renderers import get_influence_tree_data
        import pandas as pd

        # Get model dimensions
        # res = (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, ...)
        tokens = res[0]
        attentions = res[3]
        num_layers = len(attentions)
        # attentions[layer] has shape (batch, num_heads, seq_len, seq_len)
        num_heads = attentions[0].shape[1] if num_layers > 0 else 12

        all_rows = []

        def traverse(node, layer_label, head_label, mode_label, depth=0, parent_name="ROOT"):
            name = node.get("name", "Unknown")
            att = node.get("att", 0.0)

            if depth > 0:
                all_rows.append({
                    "Layer": layer_label,
                    "Head": head_label,
                    "Mode": mode_label,
                    "Parent": parent_name,
                    "Child": name,
                    "Attention": att,
                    "Depth": depth
                })

            if "children" in node and node["children"]:
                for child in node["children"]:
                    traverse(child, layer_label, head_label, mode_label, depth + 1, name)

        if all_layers_heads:
            # Export ALL layers and heads
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    tree_data = get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, top_k, norm_mode, att_matrix_override=None)
                    if tree_data:
                        traverse(tree_data, layer_idx, head_idx, "per_head", depth=0, parent_name="ROOT")

            # Export GLOBAL mode (averaged across all heads)
            for layer_idx in range(num_layers):
                # For global, we average across heads - using layer's mean attention
                layer_att = attentions[layer_idx]
                # Handle both PyTorch tensors and numpy arrays
                if hasattr(layer_att, 'cpu'):
                    avg_att = layer_att.mean(dim=1).squeeze().cpu().numpy()
                else:
                    avg_att = np.mean(layer_att, axis=1).squeeze()

                tree_data_global = get_influence_tree_data(res, layer_idx, 0, root_idx, top_k, top_k, norm_mode, att_matrix_override=avg_att)
                if tree_data_global:
                    traverse(tree_data_global, layer_idx, "all", "global", depth=0, parent_name="ROOT")
        else:
            # Single layer/head export (legacy)
            try: layer_idx = int(input.global_layer())
            except: layer_idx = 0
            try: head_idx = int(input.global_head())
            except: head_idx = 0

            tree_data = get_influence_tree_data(res, layer_idx, head_idx, root_idx, top_k, top_k, norm_mode, att_matrix_override=None)
            if tree_data:
                traverse(tree_data, layer_idx, head_idx, "per_head", depth=0, parent_name="ROOT")

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        return df.to_csv(index=False)

    @auto_save_download("attention_tree", "csv", data_type="all_layers_heads")
    def export_tree_data():
        res = get_active_result()
        csv_content = get_tree_csv(res, all_layers_heads=True)
        if csv_content:
            yield csv_content
        else:
            yield "No data available"

    @auto_save_download("attention_tree", "csv", is_b=True, data_type="all_layers_heads")
    def export_tree_data_B():
        res = get_active_result("_B")
        csv_content = get_tree_csv(res, suffix="_B", all_layers_heads=True)
        if csv_content:
            yield csv_content
        else:
            yield "No data available"

    @auto_save_download("attention_tree", "csv", data_type="all_layers_heads")
    def export_tree_data_legacy():
        res = get_active_result()
        csv_content = get_tree_csv(res, suffix="", all_layers_heads=True)
        if csv_content:
            yield csv_content
        else:
            yield "No data available"

    # --- Top K Attention Targets CSV ---
    def get_topk_attention_csv(res, suffix=""):
        """
        Export top K tokens that each token attends to most, across ALL layers, heads, and global.
        Format: Token_Idx, Token, Layer, Head, Mode, Rank, Target_Idx, Target_Token, Attention
        """
        if not res: return None

        try: top_k = int(input.global_topk())
        except: top_k = 3

        import pandas as pd
        import numpy as np

        # res = (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, ...)
        tokens = res[0]
        attentions = res[3]
        num_layers = len(attentions)
        # attentions[layer] has shape (batch, num_heads, seq_len, seq_len)
        num_heads = attentions[0].shape[1] if num_layers > 0 else 12
        num_tokens = len(tokens)

        all_rows = []

        # Per-head attention
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                layer_att = attentions[layer_idx][0, head_idx]
                # Handle CUDA tensors
                if hasattr(layer_att, 'cpu'):
                    att_matrix = layer_att.cpu().numpy()
                else:
                    att_matrix = np.array(layer_att)

                for token_idx in range(num_tokens):
                    att_row = att_matrix[token_idx]
                    top_indices = np.argsort(att_row)[::-1][:top_k]

                    for rank, target_idx in enumerate(top_indices):
                        all_rows.append({
                            "Token_Idx": token_idx,
                            "Token": tokens[token_idx],
                            "Layer": layer_idx,
                            "Head": head_idx,
                            "Mode": "per_head",
                            "Rank": rank + 1,
                            "Target_Idx": target_idx,
                            "Target_Token": tokens[target_idx],
                            "Attention": float(att_row[target_idx])
                        })

        # Global mode (average across heads per layer)
        for layer_idx in range(num_layers):
            layer_att = attentions[layer_idx][0]
            # Handle CUDA tensors
            if hasattr(layer_att, 'cpu'):
                att_matrix = layer_att.mean(dim=0).cpu().numpy()
            else:
                att_matrix = np.mean(layer_att, axis=0)

            for token_idx in range(num_tokens):
                att_row = att_matrix[token_idx]
                top_indices = np.argsort(att_row)[::-1][:top_k]

                for rank, target_idx in enumerate(top_indices):
                    all_rows.append({
                        "Token_Idx": token_idx,
                        "Token": tokens[token_idx],
                        "Layer": layer_idx,
                        "Head": "all",
                        "Mode": "global",
                        "Rank": rank + 1,
                        "Target_Idx": target_idx,
                        "Target_Token": tokens[target_idx],
                        "Attention": float(att_row[target_idx])
                    })

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        return df.to_csv(index=False)

    @auto_save_download("attention_topk", "csv", data_type="all_layers_heads")
    def export_topk_attention():
        res = get_active_result()
        csv_content = get_topk_attention_csv(res)
        if csv_content:
            yield csv_content
        else:
            yield "No data available"

    @auto_save_download("attention_topk", "csv", is_b=True, data_type="all_layers_heads")
    def export_topk_attention_B():
        res = get_active_result("_B")
        csv_content = get_topk_attention_csv(res, suffix="_B")
        if csv_content:
            yield csv_content
        else:
            yield "No data available"

    # --- ISA Export with Token-to-Token ---
    @auto_save_download("isa", "json", data_type="with_token2token")
    def export_isa_json():
        """Export ISA data as JSON including token-to-token attention."""
        res = get_active_result()
        if not res:
            yield json.dumps({"error": "No data available"})
            return

        try:
            # res = (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, ...)
            tokens = res[0]
            attentions = res[3]
            isa_data = res[10] if len(res) > 10 else None

            export_data = {
                "tokens": list(tokens),
                "model": input.model_family() or "model",
            }

            # Handle ISA data
            if isinstance(isa_data, dict):
                for key, value in isa_data.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    else:
                        export_data[key] = value

                # Add token-to-token attention for selected sentence pair
                if "sentence_boundaries" in isa_data:
                    boundaries = isa_data["sentence_boundaries"]
                    token2token = {}

                    # Compute token-to-token for each sentence pair
                    for i, (start_i, end_i) in enumerate(boundaries):
                        for j, (start_j, end_j) in enumerate(boundaries):
                            if i != j:
                                # Get max attention across layers and heads
                                max_att = None
                                for layer_att in attentions:
                                    att = layer_att[0].numpy() if hasattr(layer_att[0], 'numpy') else np.array(layer_att[0])
                                    att_slice = att[:, start_i:end_i, start_j:end_j].max(axis=0)
                                    if max_att is None:
                                        max_att = att_slice
                                    else:
                                        max_att = np.maximum(max_att, att_slice)

                                if max_att is not None:
                                    token2token[f"sent{i}_to_sent{j}"] = {
                                        "source_tokens": tokens[start_i:end_i],
                                        "target_tokens": tokens[start_j:end_j],
                                        "attention_matrix": max_att.tolist()
                                    }

                    export_data["token_to_token"] = token2token

            yield json.dumps(export_data, indent=2, default=str)
        except Exception as e:
            yield json.dumps({"error": str(e)})

    @auto_save_download("isa", "json", is_b=True, data_type="with_token2token")
    def export_isa_json_B():
        """Export ISA data as JSON for Model B."""
        res = get_active_result("_B")
        if not res:
            yield json.dumps({"error": "No data available"})
            return

        try:
            # res = (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, ...)
            tokens = res[0]
            attentions = res[3]
            isa_data = res[10] if len(res) > 10 else None

            export_data = {
                "tokens": list(tokens),
                "model": input.model_family_B() or "model",
            }

            if isinstance(isa_data, dict):
                for key, value in isa_data.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    else:
                        export_data[key] = value

                if "sentence_boundaries" in isa_data:
                    boundaries = isa_data["sentence_boundaries"]
                    token2token = {}

                    for i, (start_i, end_i) in enumerate(boundaries):
                        for j, (start_j, end_j) in enumerate(boundaries):
                            if i != j:
                                max_att = None
                                for layer_att in attentions:
                                    att = layer_att[0].numpy() if hasattr(layer_att[0], 'numpy') else np.array(layer_att[0])
                                    att_slice = att[:, start_i:end_i, start_j:end_j].max(axis=0)
                                    if max_att is None:
                                        max_att = att_slice
                                    else:
                                        max_att = np.maximum(max_att, att_slice)

                                if max_att is not None:
                                    token2token[f"sent{i}_to_sent{j}"] = {
                                        "source_tokens": tokens[start_i:end_i],
                                        "target_tokens": tokens[start_j:end_j],
                                        "attention_matrix": max_att.tolist()
                                    }

                    export_data["token_to_token"] = token2token

            yield json.dumps(export_data, indent=2, default=str)
        except Exception as e:
            yield json.dumps({"error": str(e)})

    # --- ISA CSV Export ---
    @auto_save_download("isa", "csv", data_type="sentence_matrix")
    def export_isa_csv():
        """Export ISA sentence attention matrix as CSV."""
        res = get_active_result()
        if not res:
            yield "No data available"
            return

        try:
            import pandas as pd
            isa_data = res[10] if len(res) > 10 else None

            if not isa_data or "sentence_attention_matrix" not in isa_data:
                yield "No ISA data available"
                return

            matrix = isa_data["sentence_attention_matrix"]
            sentences = isa_data.get("sentence_texts", [f"Sent_{i}" for i in range(len(matrix))])

            # Create DataFrame with sentence labels
            df = pd.DataFrame(matrix, index=sentences, columns=sentences)
            df.index.name = "Source_Sentence"
            yield df.to_csv()
        except Exception as e:
            yield f"Error: {str(e)}"

    @auto_save_download("isa", "csv", is_b=True, data_type="sentence_matrix")
    def export_isa_csv_B():
        """Export ISA sentence attention matrix as CSV for Model B."""
        res = get_active_result("_B")
        if not res:
            yield "No data available"
            return

        try:
            import pandas as pd
            isa_data = res[10] if len(res) > 10 else None

            if not isa_data or "sentence_attention_matrix" not in isa_data:
                yield "No ISA data available"
                return

            matrix = isa_data["sentence_attention_matrix"]
            sentences = isa_data.get("sentence_texts", [f"Sent_{i}" for i in range(len(matrix))])

            df = pd.DataFrame(matrix, index=sentences, columns=sentences)
            df.index.name = "Source_Sentence"
            yield df.to_csv()
        except Exception as e:
            yield f"Error: {str(e)}"

__all__ = ["server"]