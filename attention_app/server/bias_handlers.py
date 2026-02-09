import json
from datetime import datetime
from pathlib import Path
import traceback
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from shiny import ui, render, reactive

import traceback
from ..models import ModelManager
from ..bias import (
    GusNetDetector, EnsembleGusNetDetector,
    AttentionBiasAnalyzer,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_combined_bias_visualization,
    create_inline_bias_html,
    create_method_info_html,
    create_ratio_formula_html,
    create_bias_criteria_html,
    create_bias_sentence_preview,
    create_token_bias_strip,
    create_confidence_breakdown,
)
from ..ui.bias_ui import create_bias_accordion, create_floating_bias_toolbar
from ..ui.components import viz_header

def _wrap_card(content, title=None, subtitle=None, help_text=None, manual_header=None, style=None, controls=None):
    """Wrap content in a card with consistent header style."""
    base_style = "min-height: auto; display: flex; flex-direction: column;"
    if style:
        base_style += f" {style}"

    header = None
    if manual_header:
        # manual_header is (title, subtitle) tuple
        header_parts = [
            ui.h4(manual_header[0], style="margin:0;"),
            ui.p(manual_header[1], style="font-size:11px;color:#6b7280;margin:4px 0 0;flex:1;")
        ]
        if controls:
            header = ui.div(
                {"class": "viz-header", "style": "display:flex;align-items:center;gap:8px;flex-wrap:wrap;"},
                ui.h4(manual_header[0], style="margin:0;"),
                ui.p(manual_header[1], style="font-size:11px;color:#6b7280;margin:4px 0 0;flex:1;"),
                ui.div(
                    *controls,
                    style="margin-left:auto;display:flex;align-items:center;gap:8px;"
                )
            )
        else:
            header = ui.div(
                {"class": "viz-header"},
                ui.h4(manual_header[0], style="margin:0;"),
                ui.p(manual_header[1], style="font-size:11px;color:#6b7280;margin:4px 0 0;")
            )
    elif title:
        header = viz_header(title, subtitle, help_text, controls=controls)

    return ui.div(
        {"class": "card", "style": base_style},
        header,
        ui.div(content, style="flex: 1; display: flex; flex-direction: column;")
    )



# ─── Token-alignment helper ───────────────────────────────────────────────

def _align_gusnet_to_attention_tokens(gusnet_labels, attention_tokens, gusnet_special_tokens=None):
    """Align GUS-Net token labels to the attention model's tokens.

    Handles cross-tokenizer subword mismatches — e.g. when GPT-2 BPE
    produces a single token ``nurturing`` while BERT WordPiece splits it
    into ``nur`` + ``##turing``.  In such cases the GUS-Net label is
    propagated to all BERT subwords that form the original word.

    Returns a list (same length as *attention_tokens*) where each entry
    is either a GUS-Net label dict or ``None`` for unmatched positions.
    """
    if gusnet_special_tokens is None:
        gusnet_special_tokens = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    attn_special = {"[CLS]", "[SEP]", "[PAD]", "<|endoftext|>"}

    def _clean(tok):
        """Normalise a subword token for text matching."""
        return tok.replace("##", "").replace("\u0120", "").replace("Ġ", "").lower().strip()

    # Build cleaned GUS-Net content tokens
    gus_clean = []
    gus_data = []
    for label in gusnet_labels:
        if label["token"] in gusnet_special_tokens:
            continue
        gus_clean.append(_clean(label["token"]))
        gus_data.append(label)

    aligned = [None] * len(attention_tokens)
    gus_idx = 0
    # Tracks leftover characters from a partially consumed GUS-Net token
    gus_remainder = ""
    gus_current_label = None

    for bt_idx, bt in enumerate(attention_tokens):
        if bt in attn_special or (bt.startswith("[") and bt.endswith("]")):
            continue

        clean_bt = _clean(bt)
        if not clean_bt:
            continue

        # Case 1: continuing to consume a partially matched GUS-Net token
        if gus_remainder:
            if gus_remainder.startswith(clean_bt):
                aligned[bt_idx] = gus_current_label
                gus_remainder = gus_remainder[len(clean_bt):]
                if not gus_remainder:
                    gus_current_label = None
                continue
            else:
                # Mismatch mid-word — reset and fall through
                gus_remainder = ""
                gus_current_label = None

        # Case 2: exact match with current GUS-Net token
        if gus_idx < len(gus_clean) and gus_clean[gus_idx] == clean_bt:
            aligned[bt_idx] = gus_data[gus_idx]
            gus_idx += 1
            continue

        # Case 3: attention subword is a prefix of the GUS-Net token
        #         (e.g. BERT "nur" is prefix of GPT-2 "nurturing")
        if gus_idx < len(gus_clean) and gus_clean[gus_idx].startswith(clean_bt):
            aligned[bt_idx] = gus_data[gus_idx]
            gus_remainder = gus_clean[gus_idx][len(clean_bt):]
            gus_current_label = gus_data[gus_idx]
            if not gus_remainder:
                gus_idx += 1
                gus_current_label = None
            else:
                gus_idx += 1  # advance — remainder will be consumed by next subwords
            continue

        # Case 4: GUS-Net subword is a prefix of the attention token
        #         (e.g. GPT-2 splits more finely than BERT on rare words)
        if gus_idx < len(gus_clean):
            accumulated = ""
            scan = gus_idx
            while scan < len(gus_clean) and len(accumulated) < len(clean_bt):
                accumulated += gus_clean[scan]
                scan += 1
            if accumulated == clean_bt:
                # Merge: use the first GUS-Net label (highest specificity)
                aligned[bt_idx] = gus_data[gus_idx]
                gus_idx = scan
                continue

        # Case 5: look-ahead for misalignment recovery
        for look in range(gus_idx, min(gus_idx + 5, len(gus_clean))):
            if gus_clean[look] == clean_bt:
                aligned[bt_idx] = gus_data[look]
                gus_idx = look + 1
                break

    return aligned


# ─── Server handler registration ──────────────────────────────────────────

def _get_bias_model_label(res):
    """Return a human-readable label for the bias model used in the result."""
    from ..bias.gusnet_detector import MODEL_REGISTRY
    key = res.get("bias_model_key", "gusnet-bert")
    cfg = MODEL_REGISTRY.get(key, {})
    return cfg.get("display_name", key)


def bias_server_handlers(input, output, session):
    """Create server handlers for bias analysis tab."""

    bias_running = reactive.value(False)
    bias_results = reactive.value(None)
    bias_results_B = reactive.value(None)  # For comparison
    bias_history = reactive.Value([])

    # Snapshots of compare mode state - Updated ONLY on 'Analyze Bias'
    active_bias_compare_models = reactive.Value(False)
    active_bias_compare_prompts = reactive.Value(False)
    
    # Sequential Logic State
    bias_prompt_step = reactive.Value("A")

    # Cached texts for display
    bias_cached_text_A = reactive.value("")
    bias_cached_text_B = reactive.value("")

    # ── Constants for UI Consistency ──
    _BTN_STYLE_CSV = "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; color: #334155; text-decoration: none;"
    _BTN_STYLE_PNG = "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; color: #334155;"

    # ── History Logic ──
    
    @reactive.Effect
    @reactive.event(input.analyze_bias_btn)
    def update_bias_history():
        text = input.bias_input_text().strip()
        if not text:
            return

        # Deduplicate using normalized comparison
        hist = [h for h in bias_history() if h.strip() != text]
        hist.insert(0, text)
        hist = hist[:15]

        bias_history.set(hist)

    @output
    @render.ui
    def bias_history_list():
        hist = bias_history()
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
                    onclick=f"selectBiasHistoryItem('{safe_text}')"
                )
            )
        return ui.div(*items)

    @reactive.Effect
    @reactive.event(bias_history)
    def update_bias_history_list():
        """Inject history HTML directly into the dropdown (mirrors attention tab)."""
        history = bias_history.get()
        html_content = ""
        if not history:
            html_content = '<div style="padding:10px;color:#94a3b8;font-style:italic;">No history yet.</div>'
        else:
            for item in history:
                display = (item[:60] + "...") if len(item) > 60 else item
                safe = item.replace("\\", "\\\\").replace("'", "\\'").replace('"', '&quot;').replace('\n', ' ')
                display_safe = display.replace("`", "\\`").replace("${", "\\${")
                html_content += (
                    f'<div class="history-item" '
                    f"onclick=\"selectBiasHistoryItem('{safe}')\">"
                    f'{display_safe}</div>'
                )

        js_code = f"""
            var dropdown = document.getElementById('bias-history-dropdown');
            if (dropdown) {{
                dropdown.innerHTML = `{html_content}`;
            }}
        """
        ui.insert_ui(
            selector="body", where="beforeEnd",
            ui=ui.tags.script(js_code),
        )

    # ── Sequential Button Logic ──
    @reactive.Effect
    @reactive.event(input.bias_active_prompt_tab)
    def update_bias_step_state():
        # Keep python state in sync with JS state
        step = input.bias_active_prompt_tab()
        if step:
            bias_prompt_step.set(step)

    @reactive.Effect
    @reactive.event(input.bias_compare_prompts_mode, bias_prompt_step)
    async def update_bias_button_label():
        try:
            mode = input.bias_compare_prompts_mode()
        except Exception:
            mode = False
            
        step = bias_prompt_step.get()
        label = "Analyze Bias"
        
        if mode and step == "A":
            label = "Prompt B ➜"
            
        await session.send_custom_message("update_bias_button_label", {"label": label})

    # ── Export Infrastructure ──

    _BIAS_EXPORT_FOLDER_MAP = {
        "json": "downloads/json",
        "csv": "downloads/csv",
        "png": "downloads/png",
    }

    def generate_bias_export_filename(section, ext="csv", is_b=False, incl_timestamp=True, data_type=None):
        """Generate export filename for bias section exports."""
        try:
            bias_key = input.bias_model_key()
        except Exception:
            bias_key = "gusnet-bert"

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()

        parts = [f"bias_{section}"]
        if data_type:
            parts.append(data_type)
        parts.append(bias_key)

        if compare_models:
            parts.append("ModelB" if is_b else "ModelA")
        if compare_prompts:
            parts.append("PromptB" if is_b else "PromptA")
        if incl_timestamp:
            parts.append(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        return f"{'_'.join(parts)}.{ext}"

    def save_bias_export_to_folder(content, filename):
        """Save export content to the appropriate project folder."""
        ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
        folder = _BIAS_EXPORT_FOLDER_MAP.get(ext)
        if "session" in filename and ext == "json":
            folder = "downloads/sessions"
        if folder and content and not content.startswith("Error") and not content.startswith("No data"):
            try:
                filepath = Path(folder) / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding='utf-8')
            except Exception:
                pass

    def auto_save_bias_download(section, ext, **gen_kwargs):
        """Decorator replacing @render.download that also saves a copy."""
        filename_fn = lambda: generate_bias_export_filename(section, ext, **gen_kwargs)
        def decorator(fn):
            @render.download(filename=filename_fn)
            @functools.wraps(fn)
            def wrapper():
                parts = []
                for chunk in fn():
                    parts.append(str(chunk) if not isinstance(chunk, str) else chunk)
                    yield chunk
                if parts:
                    content = "".join(parts)
                    fname = filename_fn()
                    save_bias_export_to_folder(content, fname)
            return wrapper
        return decorator

    # ── Session Logic ──

    @render.download(filename=lambda: f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def save_bias_session():
        data = {
            "type": "bias_analysis",
            "timestamp": datetime.now().isoformat(),
            "text": input.bias_input_text(),
            "threshold": float(input.bias_threshold()) if input.bias_threshold() else 0.5,
        }

        # Bias model key
        try:
            data["bias_model_key"] = input.bias_model_key()
        except Exception:
            data["bias_model_key"] = "gusnet-bert"

        # Compare modes
        try:
            data["bias_compare_mode"] = bool(input.bias_compare_mode())
        except Exception:
            data["bias_compare_mode"] = False
        try:
            data["bias_compare_prompts_mode"] = bool(input.bias_compare_prompts_mode())
        except Exception:
            data["bias_compare_prompts_mode"] = False

        # Conditional: Model B / Text B
        if data["bias_compare_mode"]:
            try:
                data["bias_model_key_B"] = input.bias_model_key_B()
            except Exception:
                data["bias_model_key_B"] = "gusnet-gpt2"
        if data["bias_compare_prompts_mode"]:
            try:
                data["bias_input_text_B"] = input.bias_input_text_B()
            except Exception:
                data["bias_input_text_B"] = ""

        # Layer / Head / Top-K
        try:
            data["bias_attn_layer"] = input.bias_attn_layer()
        except Exception:
            pass
        try:
            data["bias_attn_head"] = input.bias_attn_head()
        except Exception:
            pass
        try:
            data["bias_top_k"] = int(input.bias_top_k())
        except Exception:
            pass

        content = json.dumps(data, indent=2)
        # Also save to downloads/sessions/
        save_bias_export_to_folder(content, f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return content

    @reactive.Effect
    @reactive.event(input.load_bias_session_upload)
    async def load_bias_session():
        file_info = input.load_bias_session_upload()
        if not file_info:
            return

        try:
            with open(file_info[0]["datapath"], "r") as f:
                data = json.load(f)

            # 1. Restore compare modes first (triggers UI rebuild)
            compare_mode = data.get("bias_compare_mode", False)
            compare_prompts = data.get("bias_compare_prompts_mode", False)

            # Use JS to set switch values (standard Shiny switches)
            if compare_mode:
                await session.send_custom_message("bias_eval_js",
                    "var sw=$('#bias_compare_mode'); if(sw.length && !sw.prop('checked')){sw.prop('checked',true).trigger('change');}")
            if compare_prompts:
                await session.send_custom_message("bias_eval_js",
                    "var sw=$('#bias_compare_prompts_mode'); if(sw.length && !sw.prop('checked')){sw.prop('checked',true).trigger('change');}")

            # 2. Wait for UI to rebuild, then restore controls via JS
            await asyncio.sleep(0.3)

            restore_data = {}
            if "bias_model_key" in data:
                restore_data["bias_model_key"] = data["bias_model_key"]
            if "bias_model_key_B" in data:
                restore_data["bias_model_key_B"] = data["bias_model_key_B"]
            if "threshold" in data:
                restore_data["threshold"] = data["threshold"]

            if restore_data:
                await session.send_custom_message("restore_bias_session_controls", restore_data)

            # 3. Restore text inputs
            if "text" in data:
                await session.send_custom_message("bias_eval_js",
                    f"var ta=document.getElementById('bias_input_text'); if(ta){{ta.value={json.dumps(data['text'])}; Shiny.setInputValue('bias_input_text',ta.value,{{priority:'event'}});}}")
            if "bias_input_text_B" in data:
                await session.send_custom_message("bias_eval_js",
                    f"var ta=document.getElementById('bias_input_text_B'); if(ta){{ta.value={json.dumps(data['bias_input_text_B'])}; Shiny.setInputValue('bias_input_text_B',ta.value,{{priority:'event'}});}}")

            # 4. Restore layer/head selections (after a small delay for selects to populate)
            await asyncio.sleep(0.2)
            if "bias_attn_layer" in data:
                ui.update_select("bias_attn_layer", selected=str(data["bias_attn_layer"]))
            if "bias_attn_head" in data:
                ui.update_select("bias_attn_head", selected=str(data["bias_attn_head"]))

            ui.notification_show("Bias session loaded successfully.", type="message", duration=3)

        except Exception as e:
            print(f"Error loading bias session: {e}")
            traceback.print_exc()
            ui.notification_show(f"Error loading session: {e}", type="error")

    # ── CSV Export Handlers (8 pairs: A and B) ──

    # 1. Bias Summary
    @auto_save_bias_download("summary", "csv")
    def export_bias_summary():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        s = res["bias_summary"]
        lines = ["metric,value"]
        for k, v in s.items():
            lines.append(f"{k},{v}")
        yield "\n".join(lines)

    @auto_save_bias_download("summary", "csv", is_b=True)
    def export_bias_summary_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        s = res["bias_summary"]
        lines = ["metric,value"]
        for k, v in s.items():
            lines.append(f"{k},{v}")
        yield "\n".join(lines)

    # 2. Bias Spans (Detected Bias Tokens)
    @auto_save_bias_download("spans", "csv")
    def export_bias_spans():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,bias_types,GEN,UNFAIR,STEREO,max_score"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = ";".join(lbl.get("bias_types", []))
            max_s = max((scores.get(t, 0) for t in lbl.get("bias_types", [])), default=0)
            lines.append(f"{lbl['token']},{lbl['index']},{types},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f},{max_s:.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("spans", "csv", is_b=True)
    def export_bias_spans_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,bias_types,GEN,UNFAIR,STEREO,max_score"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = ";".join(lbl.get("bias_types", []))
            max_s = max((scores.get(t, 0) for t in lbl.get("bias_types", [])), default=0)
            lines.append(f"{lbl['token']},{lbl['index']},{types},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f},{max_s:.4f}")
        yield "\n".join(lines)

    # 3. Token Bias Strip
    @auto_save_bias_download("strip", "csv")
    def export_bias_strip():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,is_biased,O,GEN,UNFAIR,STEREO"]
        for lbl in res["token_labels"]:
            scores = lbl.get("scores", {})
            lines.append(f"{lbl['token']},{lbl['index']},{lbl.get('is_biased',False)},{scores.get('O',0):.4f},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("strip", "csv", is_b=True)
    def export_bias_strip_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,is_biased,O,GEN,UNFAIR,STEREO"]
        for lbl in res["token_labels"]:
            scores = lbl.get("scores", {})
            lines.append(f"{lbl['token']},{lbl['index']},{lbl.get('is_biased',False)},{scores.get('O',0):.4f},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f}")
        yield "\n".join(lines)

    # 4. Confidence Breakdown
    @auto_save_bias_download("confidence", "csv")
    def export_bias_confidence():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,tier,max_score,bias_types"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = lbl.get("bias_types", [])
            max_s = max((scores.get(t, 0) for t in types), default=0)
            tier = "high" if max_s >= 0.85 else ("medium" if max_s >= 0.70 else "low")
            lines.append(f"{lbl['token']},{lbl['index']},{tier},{max_s:.4f},{';'.join(types)}")
        yield "\n".join(lines)

    @auto_save_bias_download("confidence", "csv", is_b=True)
    def export_bias_confidence_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        lines = ["token,index,tier,max_score,bias_types"]
        for lbl in res["token_labels"]:
            if not lbl.get("is_biased"):
                continue
            scores = lbl.get("scores", {})
            types = lbl.get("bias_types", [])
            max_s = max((scores.get(t, 0) for t in types), default=0)
            tier = "high" if max_s >= 0.85 else ("medium" if max_s >= 0.70 else "low")
            lines.append(f"{lbl['token']},{lbl['index']},{tier},{max_s:.4f},{';'.join(types)}")
        yield "\n".join(lines)

    # 5. Combined Bias View (Attention Matrix CSV)
    @auto_save_bias_download("combined", "csv")
    def export_bias_combined():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        try:
            l_idx = int(input.bias_attn_layer())
            h_idx = int(input.bias_attn_head())
        except Exception:
            l_idx, h_idx = 0, 0
        atts = res["attentions"]
        if not atts or l_idx >= len(atts):
            yield "No attention data"
            return
        tokens = res["tokens"]
        attn = atts[l_idx][0, h_idx].cpu().numpy()
        header = "query_token," + ",".join(tokens)
        lines = [header]
        for i, tok in enumerate(tokens):
            vals = ",".join(f"{attn[i,j]:.6f}" for j in range(len(tokens)))
            lines.append(f"{tok},{vals}")
        yield "\n".join(lines)

    @auto_save_bias_download("combined", "csv", is_b=True)
    def export_bias_combined_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        try:
            l_idx = int(input.bias_attn_layer())
            h_idx = int(input.bias_attn_head())
        except Exception:
            l_idx, h_idx = 0, 0
        atts = res["attentions"]
        if not atts or l_idx >= len(atts):
            yield "No attention data"
            return
        tokens = res["tokens"]
        attn = atts[l_idx][0, h_idx].cpu().numpy()
        header = "query_token," + ",".join(tokens)
        lines = [header]
        for i, tok in enumerate(tokens):
            vals = ",".join(f"{attn[i,j]:.6f}" for j in range(len(tokens)))
            lines.append(f"{tok},{vals}")
        yield "\n".join(lines)

    # 6. Attention Bias Matrix
    @auto_save_bias_download("matrix", "csv")
    def export_bias_matrix():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        lines = ["layer,head,bias_attention_ratio,specialized"]
        for m in metrics:
            lines.append(f"{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    @auto_save_bias_download("matrix", "csv", is_b=True)
    def export_bias_matrix_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        lines = ["layer,head,bias_attention_ratio,specialized"]
        for m in metrics:
            lines.append(f"{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    # 7. Bias Propagation
    @auto_save_bias_download("propagation", "csv")
    def export_bias_propagation():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        prop = res["propagation_analysis"]["layer_propagation"]
        if not prop:
            yield "No propagation data"
            return
        lines = ["layer,mean_bar,max_bar,min_bar"]
        for p in prop:
            lines.append(f"{p['layer']},{p['mean_ratio']:.4f},{p['max_ratio']:.4f},{p['min_ratio']:.4f}")
        yield "\n".join(lines)

    @auto_save_bias_download("propagation", "csv", is_b=True)
    def export_bias_propagation_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        prop = res["propagation_analysis"]["layer_propagation"]
        if not prop:
            yield "No propagation data"
            return
        lines = ["layer,mean_bar,max_bar,min_bar"]
        for p in prop:
            lines.append(f"{p['layer']},{p['mean_ratio']:.4f},{p['max_ratio']:.4f},{p['min_ratio']:.4f}")
        yield "\n".join(lines)

    # 8. Top Bias-Focused Heads
    @auto_save_bias_download("top_heads", "csv")
    def export_bias_top_heads():
        res = bias_results.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:5]
        lines = ["rank,layer,head,bar,specialized"]
        for i, m in enumerate(top, 1):
            lines.append(f"{i},{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    @auto_save_bias_download("top_heads", "csv", is_b=True)
    def export_bias_top_heads_B():
        res = bias_results_B.get()
        if not res:
            yield "No data available"
            return
        metrics = res.get("attention_metrics", [])
        if not metrics:
            yield "No metrics available"
            return
        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:5]
        lines = ["rank,layer,head,bar,specialized"]
        for i, m in enumerate(top, 1):
            lines.append(f"{i},{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
        yield "\n".join(lines)

    def log_debug(msg):
        with open("bias_debug.log", "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
            
    def heavy_bias_compute(text, model_name, threshold, bias_model_key):
        """Perform bias analysis computation (runs in thread pool).

        Parameters
        ----------
        text : str
            The input sentence to analyse.
        model_name : str
            HuggingFace model name for the attention model (e.g. "bert-base-uncased").
        threshold : float
            Sensitivity threshold for the bias detector.
        bias_model_key : str
            Key into ``MODEL_REGISTRY`` selecting the GUS-Net backbone
            ("gusnet-bert" or "gusnet-gpt2").
        """
        log_debug(f"Starting heavy_bias_compute (threshold={threshold}, bias_model={bias_model_key})")
        print(f"DEBUG: Starting heavy_bias_compute (threshold={threshold}, bias_model={bias_model_key})")
        if not text:
            log_debug("No text provided")
            print("DEBUG: No text provided")
            return None

        try:
            # Load attention model
            log_debug(f"Loading attention model {model_name}...")
            print(f"DEBUG: Loading attention model {model_name}...")
            tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
            device = ModelManager.get_device()
            log_debug(f"Model loaded. Device: {device}")
            print(f"DEBUG: Model loaded. Device: {device}")

            log_debug("Tokenizing text...")
            print("DEBUG: Tokenizing text...")
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            log_debug("Running encoder inference...")
            print("DEBUG: Running encoder inference...")
            with torch.no_grad():
                outputs = encoder_model(**inputs)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attentions = outputs.attentions
            log_debug(f"Got {len(tokens)} tokens, {len(attentions)} layers")
            print(f"DEBUG: Got {len(tokens)} tokens, {len(attentions)} layers")

            # ── Token-level bias detection (GUS-Net) ──
            log_debug(f"Initializing GusNetDetector (model_key={bias_model_key})...")
            print(f"DEBUG: Initializing GusNetDetector (model_key={bias_model_key})...")
            if bias_model_key == "gusnet-ensemble":
                gus_detector = EnsembleGusNetDetector(
                    model_key_a="gusnet-bert",
                    model_key_b="gusnet-bert-custom",
                    threshold=threshold,
                )
            else:
                gus_detector = GusNetDetector(model_key=bias_model_key, threshold=threshold)
            log_debug("Running detect_bias...")
            print("DEBUG: Running detect_bias...")
            gusnet_labels = gus_detector.detect_bias(text)
            log_debug(f"GUS-Net returned {len(gusnet_labels)} labels")
            print(f"DEBUG: GUS-Net returned {len(gusnet_labels)} labels")

            # Align GUS-Net labels to the attention model's tokens
            gus_special = gus_detector.config["special_tokens"]
            gus_aligned = _align_gusnet_to_attention_tokens(
                gusnet_labels, tokens, gusnet_special_tokens=gus_special
            )

            # Build unified token_labels aligned to BERT tokens
            token_labels = []
            for i, tok in enumerate(tokens):
                matched = gus_aligned[i]
                if matched is not None:
                    token_labels.append({
                        **matched,
                        "token": tok,
                        "index": i,
                    })
                else:
                    token_labels.append({
                        "token": tok,
                        "index": i,
                        "bias_types": [],
                        "is_biased": False,
                        "scores": {"O": 0.0, "GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
                        "method": "gusnet",
                        "explanation": "",
                        "threshold": threshold,
                    })

            bias_summary = gus_detector.get_bias_summary(token_labels)
            bias_spans = gus_detector.get_biased_spans(token_labels)
            log_debug(f"{len(bias_spans)} spans detected")
            print(f"DEBUG: {len(bias_spans)} spans detected")

            # ── Attention x Bias ──
            attention_analyzer = AttentionBiasAnalyzer()
            biased_indices = [i for i, l in enumerate(token_labels) if l["is_biased"]]

            if biased_indices and attentions:
                attention_metrics = attention_analyzer.analyze_attention_to_bias(
                    list(attentions), biased_indices, tokens
                )
                propagation_analysis = attention_analyzer.analyze_bias_propagation(
                    list(attentions), biased_indices, tokens
                )
                bias_matrix = attention_analyzer.create_attention_bias_matrix(
                    list(attentions), biased_indices
                )
            else:
                attention_metrics = []
                propagation_analysis = {
                    "layer_propagation": [], "peak_layer": None,
                    "propagation_pattern": "none",
                }
                bias_matrix = np.array([])
            
            log_debug("Finished heavy_bias_compute successfully")
            return {
                "tokens": tokens,
                "text": text,
                "attentions": attentions,
                "token_labels": token_labels,
                "bias_summary": bias_summary,
                "bias_spans": bias_spans,
                "biased_indices": biased_indices,
                "attention_metrics": attention_metrics,
                "propagation_analysis": propagation_analysis,
                "bias_matrix": bias_matrix,
                "bias_model_key": bias_model_key,
            }
        except Exception as e:
            msg = f"ERROR in heavy_bias_compute: {e}"
            log_debug(msg)
            print(msg)
            traceback.print_exc()
            return None

    # ── Trigger analysis ──

    @reactive.effect
    @reactive.event(input.analyze_bias_btn)
    async def compute_bias():
        log_debug("BUTTON CLICKED: compute_bias triggered")
        try:
            text = input.bias_input_text().strip()
            log_debug(f"Input text length: {len(text)}")
            if not text:
                log_debug("Text is empty, returning")
                return

            # ── Intercept Click for Sequential Logic ──
            try:
                compare_prompts_live = input.bias_compare_prompts_mode()
            except Exception:
                compare_prompts_live = False
            
            step = bias_prompt_step.get()

            if compare_prompts_live and step == "A":
                # User clicked "Prompt B ->". Do not analyze. Switch tab.
                log_debug("Sequential Logic: Switching to Tab B")
                await session.send_custom_message("switch_prompt_tab_bias", "B") # Need to add this handler to JS too
                # Or just use the existing one? No, I added switchBiasPromptTab function but not a message handler for it specifically?
                # Actually, I can just call the function via JS eval or add a handler. 
                # Let's add a simple script execution.
                await session.send_custom_message("bias_eval_js", "window.switchBiasPromptTab('B');")
                return

            log_debug("Starting loading UI...")
            bias_running.set(True)
            await session.send_custom_message('start_bias_loading', {})
            await asyncio.sleep(0.1)
            log_debug("Loading UI active")

            # Check compare modes and snapshot them
            try:
                compare_models = input.bias_compare_mode()
            except Exception:
                compare_models = False
            try:
                compare_prompts = input.bias_compare_prompts_mode()
            except Exception:
                compare_prompts = False

            active_bias_compare_models.set(compare_models)
            active_bias_compare_prompts.set(compare_prompts)

            # Clear previous results
            bias_results.set(None)
            bias_results_B.set(None)
            bias_cached_text_A.set(text)

            try:
                model_name = input.model_name()
                log_debug(f"Retrieved model_name: {model_name}")
            except Exception as e:
                log_debug(f"Failed to get model_name, using default. Error: {e}")
                model_name = "bert-base-uncased"

            try:
                threshold = float(input.bias_threshold())
                log_debug(f"Threshold: {threshold}")
            except Exception as e:
                log_debug(f"Warning: bias_threshold input missing or invalid ({e}). Using default 0.5")
                threshold = 0.5

            # Bias detection model (BERT or GPT-2 backbone) - Model A
            try:
                bias_model_key = input.bias_model_key()
                log_debug(f"Bias model key: {bias_model_key}")
            except Exception as e:
                log_debug(f"Warning: bias_model_key missing ({e}). Using default gusnet-bert")
                bias_model_key = "gusnet-bert"

            try:
                loop = asyncio.get_running_loop()
                log_debug("Entering ThreadPoolExecutor...")
                with ThreadPoolExecutor() as pool:
                    # Compute Result A
                    log_debug("Submitting heavy_bias_compute for Model A...")
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            pool, heavy_bias_compute, text, model_name, threshold, bias_model_key,
                        ),
                        timeout=180.0
                    )
                    log_debug("heavy_bias_compute A returned successfully")
                    bias_results.set(result)

                    # Compute Result B if needed
                    if compare_models:
                        # Case 1: Same Prompt (A), Different Model (B)
                        try:
                            bias_model_key_B = input.bias_model_key_B()
                            if not bias_model_key_B:
                                bias_model_key_B = "gusnet-gpt2"
                        except Exception:
                            bias_model_key_B = "gusnet-gpt2"

                        log_debug(f"Starting heavy_bias_compute B ({bias_model_key_B}) for Compare Models")
                        result_B = await asyncio.wait_for(
                            loop.run_in_executor(
                                pool, heavy_bias_compute, text, model_name, threshold, bias_model_key_B,
                            ),
                            timeout=180.0
                        )
                        bias_results_B.set(result_B)
                        bias_cached_text_B.set(text)  # Same text for compare models

                    elif compare_prompts:
                        # Case 2: Different Prompt (B), Same Model (A)
                        try:
                            text_B = input.bias_input_text_B().strip()
                        except Exception:
                            text_B = ""

                        if text_B:
                            log_debug(f"Starting heavy_bias_compute B (Prompt B) for Compare Prompts")
                            result_B = await asyncio.wait_for(
                                loop.run_in_executor(
                                    pool, heavy_bias_compute, text_B, model_name, threshold, bias_model_key,
                                ),
                                timeout=180.0
                            )
                            bias_results_B.set(result_B)
                            bias_cached_text_B.set(text_B)
                        else:
                            bias_results_B.set(None)
                            bias_cached_text_B.set("")
                    else:
                        bias_results_B.set(None)
                        bias_cached_text_B.set("")

            except asyncio.TimeoutError:
                msg = "ERROR: Bias analysis timed out (limit: 180s)"
                log_debug(msg)
                print(msg)
                ui.notification_show("Analysis timed out.", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            except Exception as e:
                msg = f"ERROR during execution: {e}"
                log_debug(msg)
                print(msg)
                traceback.print_exc()
                ui.notification_show(f"Analysis failed: {e}", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            finally:
                log_debug("Stopping loading UI")
                bias_running.set(False)
                await session.send_custom_message('stop_bias_loading', {})

        except Exception as e:
            msg = f"CRITICAL ERROR in compute_bias top level: {e}"
            log_debug(msg)
            print(msg)

    # ── Dashboard content (conditional rendering) ──

    @output
    @render.ui
    def bias_dashboard_content():
        res = bias_results.get()
        res_B = bias_results_B.get()
        running = bias_running.get()

        # Get snapshotted compare mode states
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()

        # Use cached texts when results exist to prevent re-rendering on input change
        # Only read live inputs when there are no results (pre-analysis state)
        if res or running:
            text = bias_cached_text_A.get()
            text_B = bias_cached_text_B.get()
        else:
            try:
                text = input.bias_input_text().strip()
            except Exception:
                text = ""

            try:
                text_B = input.bias_input_text_B().strip()
            except Exception:
                text_B = ""

        from .renderers import get_gusnet_architecture_section

        # ── Post-analysis: sentence preview + accordion ──
        if res:
            # Check if we're in compare mode
            if (compare_models or compare_prompts) and res_B:
                # Side-by-side comparison view
                preview_html_A = create_bias_sentence_preview(
                    res["tokens"], res["token_labels"]
                )
                preview_html_B = create_bias_sentence_preview(
                    res_B["tokens"], res_B["token_labels"]
                )

                # Determine labels based on mode
                if compare_models:
                    header_A = "MODEL A"
                    header_B = "MODEL B"
                    # Keep the detailed label for the card or just use generic?
                    # User asked for "MODEL A" header.
                else:
                    header_A = "PROMPT A"
                    header_B = "PROMPT B"

                return ui.div(
                    {"style": "display: flex; flex-direction: column; gap: 24px;"},
                    # Side-by-side sentence previews
                    ui.div(
                        {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: start;"},
                        # Column A
                        ui.div(
                            ui.h3(header_A, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                            ui.div(
                                {"class": "card compare-card-a", "style": "min-height: auto;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_A),
                            ),
                        ),
                        # Column B
                        ui.div(
                            ui.h3(header_B, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                            ui.div(
                                {"class": "card compare-card-b", "style": "min-height: auto;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_B),
                            ),
                        ),
                    ),
                    # Summary comparison cards
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.tags.script("Shiny.setInputValue('toggle_bias_toolbar_visible', true, {priority: 'event'});"),
                )
            else:
                # Single result view (original behavior)
                preview_html = create_bias_sentence_preview(
                    res["tokens"], res["token_labels"]
                )
                return ui.div(
                    {"style": "display: flex; flex-direction: column; gap: 24px;"},
                    ui.div(
                        {"class": "card", "style": "min-height: auto;"},
                        ui.h4("Sentence Preview"),
                        ui.HTML(preview_html),
                    ),
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.tags.script("Shiny.setInputValue('toggle_bias_toolbar_visible', true, {priority: 'event'});"),
                )

        # ── Pre-analysis: sentence preview + architecture ──
        if text:
            preview = ui.div(
                f'"{text}"',
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
        else:
            preview = ui.div(
                'Enter text in the sidebar and click "Analyze Bias" to begin.',
                style="color:#9ca3af;font-size:14px;font-family:monospace;",
            )

        # Get LIVE compare mode states for pre-analysis (instant feedback)
        try:
            live_compare_models = input.bias_compare_mode()
        except Exception:
            live_compare_models = False

        try:
            live_compare_prompts = input.bias_compare_prompts_mode()
        except Exception:
            live_compare_prompts = False

        if live_compare_prompts:
            # Side-by-side preview for Compare Prompts
            preview_A = ui.div(
                f'"{text}"' if text else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            preview_B = ui.div(
                f'"{text_B}"' if text_B else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            
            card = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px; min-height: auto; align-items: start;"},
                # Column A
                ui.div(
                    ui.h3("PROMPT A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a", "style": "min-height: auto;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_A,
                    ),
                ),
                # Column B
                ui.div(
                    ui.h3("PROMPT B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b", "style": "min-height: auto;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_B,
                    ),
                ),
            )
        elif live_compare_models:
            # Side-by-side preview for Compare Models (same text, different models)
            preview_content = ui.div(
                f'"{text}"' if text else "Waiting for input...",
                style="font-family:monospace;color:#6b7280;font-size:14px;"
            )
            
            card = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px; min-height: auto; align-items: start;"},
                # Column A
                ui.div(
                    ui.h3("MODEL A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a", "style": "min-height: auto;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_content,
                    ),
                ),
                # Column B
                ui.div(
                    ui.h3("MODEL B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b", "style": "min-height: auto;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        # We need a fresh object/string for the second separate div content
                        ui.div(
                            f'"{text}"' if text else "Waiting for input...",
                            style="font-family:monospace;color:#6b7280;font-size:14px;"
                        ),
                    ),
                ),
            )
        else:
            # Single preview
            if text:
                preview = ui.div(
                    f'"{text}"',
                    style="font-family:monospace;color:#6b7280;font-size:14px;"
                )
            else:
                preview = ui.div(
                    'Enter text in the sidebar and click "Analyze Bias" to begin.',
                    style="color:#9ca3af;font-size:14px;font-family:monospace;",
                )

            card = ui.div(
                {"class": "card", "style": "min-height: auto;"},
                ui.h4("Sentence Preview"),
                preview,
            )

        if running:
            return ui.div(
                {"style": "display: flex; flex-direction: column; gap: 24px;"},
                card
            )

        # Initial State: Show Architecture
        # Determine architecture display mode based on current compare mode
        try:
            selected_model = input.bias_model_key()
        except Exception:
            selected_model = "gusnet-bert"

        try:
            current_compare_models = input.bias_compare_mode()
        except Exception:
            current_compare_models = False

        try:
            text_B = input.bias_input_text_B().strip()
        except Exception:
            text_B = ""

        # Get Model B selection for Compare Models mode
        try:
            model_b = input.bias_model_key_B()
            if not model_b:
                model_b = "gusnet-gpt2"  # Default Model B
        except Exception:
            model_b = "gusnet-gpt2"

        # Determine if we're in compare prompts mode (text_B present but not compare_models)
        current_compare_prompts = bool(text_B) and not current_compare_models

        arch_section = ui.div(
            get_gusnet_architecture_section(
                selected_model=selected_model,
                compare_mode=current_compare_models,
                compare_prompts=current_compare_prompts,
                model_a=selected_model,  # Model A comes from main selector
                model_b=model_b,  # Model B comes from compare selector
            ),
        )

        return ui.div(
            {"style": "display: flex; flex-direction: column; gap: 24px;"},
            card,
            arch_section
        )


    def _create_bias_comparison_summary(res_A, res_B, is_model_compare):
        """Create a side-by-side summary comparison card."""
        summary_A = res_A["bias_summary"]
        summary_B = res_B["bias_summary"]

        if is_model_compare:
            label_A = _get_bias_model_label(res_A)
            label_B = _get_bias_model_label(res_B)
        else:
            label_A = "Prompt A"
            label_B = "Prompt B"

        # Create comparison metrics
        comparison_html = f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="border: 2px solid #3b82f6; border-radius: 8px; padding: 16px;">
                <h5 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 14px; font-weight: 700;">{label_A}</h5>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #1e293b;">{summary_A['biased_tokens']}</div>
                        <div style="font-size: 10px; color: #64748b;">Biased Tokens</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #ea580c;">{summary_A['generalization_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Generalizations</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #dc2626;">{summary_A['unfairness_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Unfair</div>
                    </div>
                </div>
                <div style="margin-top: 12px; text-align: center;">
                    <span style="font-size: 28px; font-weight: 700; color: #ff5ca9;">{summary_A['bias_percentage']:.1f}%</span>
                    <span style="font-size: 11px; color: #64748b; margin-left: 4px;">Bias</span>
                </div>
            </div>
            <div style="border: 2px solid #ff5ca9; border-radius: 8px; padding: 16px;">
                <h5 style="color: #ff5ca9; margin: 0 0 12px 0; font-size: 14px; font-weight: 700;">{label_B}</h5>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #1e293b;">{summary_B['biased_tokens']}</div>
                        <div style="font-size: 10px; color: #64748b;">Biased Tokens</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #ea580c;">{summary_B['generalization_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Generalizations</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; font-weight: 700; color: #dc2626;">{summary_B['unfairness_count']}</div>
                        <div style="font-size: 10px; color: #64748b;">Unfair</div>
                    </div>
                </div>
                <div style="margin-top: 12px; text-align: center;">
                    <span style="font-size: 28px; font-weight: 700; color: #ff5ca9;">{summary_B['bias_percentage']:.1f}%</span>
                    <span style="font-size: 11px; color: #64748b; margin-left: 4px;">Bias</span>
                </div>
            </div>
        </div>
        """

        return ui.div(
            {"class": "card", "style": "min-height: auto;"},
            ui.h4("Bias Comparison Summary"),
            ui.HTML(comparison_html),
        )

    # ── Method info (sidebar) ──

    @output
    @render.ui
    def bias_method_info():
        try:
            model_key = input.bias_model_key()
        except Exception:
            model_key = "gusnet-bert"
        html = create_method_info_html(model_key)
        return ui.HTML(html)

    # ── Summary with explicit criteria ──

    @output
    @render.ui
    def bias_summary():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;font-size:12px;">'
                'Enter text and click "Analyze Bias" to begin.</div>'
            )

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        def get_summary_cards(res_data):
            summary = res_data["bias_summary"]
            # Criteria breakdown
            criteria_html = create_bias_criteria_html(summary)
            # Metric cards
            cards = f"""
            <div class="metrics-grid" style="margin-top:16px;">
                <div class="metric-card">
                    <div class="metric-label">Biased Tokens</div>
                    <div class="metric-value">{summary['biased_tokens']} / {summary['total_tokens']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Bias %</div>
                    <div class="metric-value">{summary['bias_percentage']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Generalizations</div>
                    <div class="metric-value" style="color:#ea580c;">{summary['generalization_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Unfair Language</div>
                    <div class="metric-value" style="color:#dc2626;">{summary['unfairness_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stereotypes</div>
                    <div class="metric-value" style="color:#7b1fa2;">{summary['stereotype_count']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value">{summary.get('avg_confidence', 0):.2f}</div>
                </div>
            </div>
            """
            return criteria_html + cards
        
        if (compare_models or compare_prompts) and res_B:
            content_A = get_summary_cards(res)
            content_B = get_summary_cards(res_B)

            header_args = ("Bias Detection Summary",
                          "Composite bias level with explicit weighted criteria and per-category counts.",
                          "Level = weighted composite of token density (30%), generalizations (20%), unfair language (25%), and stereotypes (25%).")

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(content_A), *header_args, style="margin-bottom:0; border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(content_B), *header_args, style="margin-bottom:0; border: 2px solid #ff5ca9; height: 100%;")
            )
        else:
            header_args = ("Bias Detection Summary",
                          "Composite bias level with explicit weighted criteria and per-category counts.",
                          "Level = weighted composite of token density (30%), generalizations (20%), unfair language (25%), and stereotypes (25%).")
            return _wrap_card(ui.HTML(get_summary_cards(res)), *header_args, style="margin-bottom: 24px;")

    # ── Inline bias view (primary) ──

    @output
    @render.ui
    def inline_bias_view():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            threshold = input.bias_threshold()
            html = create_inline_bias_html(
                res["text"], res["token_labels"], res["bias_spans"],
                show_neutral=False, threshold=threshold,
            )
            return ui.HTML(html)
        except Exception as e:
            print(f"Error in inline_bias_view: {e}")
            traceback.print_exc()
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Token heatmap (technical view) ──

    @output
    @render.ui
    def token_bias_viz():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )
        try:
            fig = create_token_bias_heatmap(res["token_labels"], res["text"])
            return ui.HTML(
                fig.to_html(include_plotlyjs='cdn', full_html=False,
                            config={'displayModeBar': False})
            )
        except Exception as e:
            print(f"Error creating token bias viz: {e}")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {e}</div>')

    # ── Bias spans table (per-token, one line each) ──



    # ── Token bias strip (replaces Plotly heatmap) ──



    # ── Ratio formula panel (static) ──

    @output
    @render.ui
    def bias_ratio_formula():
        return ui.HTML(create_ratio_formula_html())

    # ── Update layer/head selectors ──

    @reactive.effect
    def update_bias_selectors():
        res = bias_results.get()
        if not res or not res["attentions"]:
            return
        attentions = res["attentions"]
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]

        layer_choices = {str(i): f"Layer {i}" for i in range(num_layers)}
        head_choices = {str(i): f"Head {i}" for i in range(num_heads)}

        ui.update_select("bias_attn_layer", choices=layer_choices, selected="0")
        ui.update_select("bias_attn_head", choices=head_choices, selected="0")


    # ── Toolkit Handlers ──

    @output
    @render.ui
    def bias_toolkit_spans():
        """Render biased tokens in the toolbar popover."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:8px;">No analysis yet.</div>')

        token_labels = res["token_labels"]
        biased = [
            lbl for lbl in token_labels
            if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if not biased:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:12px;">No bias detected.</div>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        badge_script = f"<script>$('#bias-span-count-badge').text('{len(biased)}');</script>"

        items = [badge_script]
        for lbl in biased:
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            types = lbl.get("bias_types", [])
            scores = lbl.get("scores", {})
            max_score = max((scores.get(t, 0) for t in types), default=0)
            score_color = "#ef4444" if max_score > 0.8 else "#f59e0b" if max_score > 0.5 else "#94a3b8"

            cats_html = ""
            for cat in types:
                bg = cat_colors.get(cat, "#ff5ca9")
                cats_html += (
                    f'<span style="background:{bg}25;color:{bg};padding:1px 5px;'
                    f'border-radius:3px;font-size:9px;margin-right:3px;">{cat}</span>'
                )

            items.append(
                f'<div class="bias-span-item">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;'
                f'color:#e2e8f0;font-weight:600;">{clean}</span>'
                f'<span style="color:{score_color};font-weight:600;font-size:11px;'
                f'font-family:JetBrains Mono,monospace;">{max_score:.2f}</span></div>'
                f'<div style="margin-top:3px;">{cats_html}</div>'
                f'</div>'
            )

        return ui.HTML("".join(items))

    @output
    @render.ui
    def bias_toolkit_heads():
        """Render Top K heads as horizontal chips."""
        res = bias_results.get()
        if not res:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">—</span>')

        metrics = res.get("attention_metrics", [])
        if not metrics:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">—</span>')

        try:
            k = int(input.bias_top_k())
        except Exception:
            k = 5

        top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:k]

        items = []
        for m in top:
            is_sig = m.specialized_for_bias
            border = "#ff5ca9" if is_sig else "rgba(255,255,255,0.15)"
            items.append(
                f'<div class="bias-head-chip" onclick="setBiasHead({m.layer},{m.head})" '
                f'title="Ratio: {m.bias_attention_ratio:.3f}" '
                f'style="border-color:{border};">'
                f'L{m.layer}&middot;H{m.head}'
                f'</div>'
            )

        return ui.HTML("".join(items))

    @output
    @render.ui
    def bias_toolbar_tokens():
        """Render biased tokens as clickable chips in the toolbar.

        Clicking a chip selects that token in the Combined View,
        highlighting its row/column with the category-specific colour.
        """
        res = bias_results.get()
        if not res:
            return ui.HTML('<span style="color:#64748b;font-size:10px;">No analysis yet</span>')

        token_labels = res["token_labels"]
        biased = [
            lbl for lbl in token_labels
            if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if not biased:
            return ui.HTML('<span style="color:#64748b;font-size:10px;">No bias detected</span>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        items = []
        for lbl in biased:
            clean = lbl["token"].replace("##", "").replace("\u0120", "")
            types = lbl.get("bias_types", [])
            scores = lbl.get("scores", {})
            primary_color = cat_colors.get(types[0], "#ff5ca9") if types else "#ff5ca9"
            max_score = max((scores.get(t, 0) for t in types), default=0)
            cat_abbrevs = "·".join(types) if types else ""
            tok_idx = lbl["index"]

            items.append(
                f'<span class="bias-token-chip" '
                f'data-token-idx="{tok_idx}" '
                f'onclick="selectBiasToken({tok_idx})" '
                f'style="display:inline-flex;align-items:center;gap:3px;'
                f'font-size:9px;font-family:JetBrains Mono,monospace;'
                f'padding:1px 6px;border-radius:4px;flex-shrink:0;'
                f'background:{primary_color}20;border:1px solid {primary_color}50;'
                f'color:{primary_color};white-space:nowrap;cursor:pointer;" '
                f'title="{cat_abbrevs} ({max_score:.2f})">'
                f'{clean}'
                f'</span>'
            )

        return ui.HTML("".join(items))


    # ── Comparison Refactored Renderers ──

    @output
    @render.ui
    def bias_spans_table():
        res = bias_results.get()
        if not res: return None
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        try: threshold = float(input.bias_threshold())
        except: threshold = 0.5

        def get_content_html(items, t, c, s_idxs):
            import math
            n_items = len(items)
            mid_point = math.ceil(n_items / 2)
            items_col1 = items[:mid_point]
            items_col2 = items[mid_point:]
            
            lbl_info = f'Threshold: <code>{t:.2f}</code>'
            
            return (
                f'<div style="display:flex;gap:16px;border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
                f'<div style="flex:1;display:flex;flex-direction:column;">{"".join(items_col1)}</div>'
                f'<div style="flex:1;display:flex;flex-direction:column;border-left:1px solid rgba(226,232,240,0.4);">{"".join(items_col2)}</div>'
                f'</div>'
                f'<div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">{lbl_info}</div>'
            )
            
        # Helper to build content list
        def build_items(data, s_idxs):
             # Re-implement item building logic here, or just call logic
             # For brevity in this replacement, assume get_view logic is split
             # Actually I need to fully rewrite get_view to return the HTML string, then wrap in card.
             pass

        # Let's just inline the wrapping into the main logic
        
        # ... logic to build HTML items ...
        # Copied from previous implementation
        def produce_html(data, is_B):
            token_labels = data["token_labels"]
            biased = [l for l in token_labels if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
            if not biased: return '<div style="color:#9ca3af;font-size:12px;padding:12px;">No biased tokens detected.</div>'
            
            cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
            items = []
            sel_indices = []
            if not is_B:
                try: 
                    s = input.bias_selected_tokens()
                    if s: sel_indices = [int(s)] if isinstance(s,(int,str)) else [int(x) for x in s if x]
                except: pass
                
            for lbl in biased:
                clean = lbl["token"].replace("##", "").replace("\u0120", "")
                types = lbl.get("bias_types", [])
                scores = lbl.get("scores", {})
                badges = "".join([f'<span style="display:inline-flex;align-items:center;gap:4px;background:{cat_colors.get(t,"#ff5ca9")}18;border:1px solid {cat_colors.get(t,"#ff5ca9")}40;color:{cat_colors.get(t,"#ff5ca9")};padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">{t}<span style="font-family:JetBrains Mono;font-weight:400;opacity:0.8;">{scores.get(t,0):.2f}</span></span>' for t in types])
                style = "display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid rgba(226,232,240,0.4);"
                if lbl["index"] in sel_indices: style += "background:rgba(255, 92, 169, 0.1); border-left: 3px solid #ff5ca9;"
                items.append(f'<div style="{style}"><span style="font-family:JetBrains Mono;font-size:13px;font-weight:600;color:#ec4899;min-width:70px;">{clean}</span><span style="display:flex;gap:4px;flex-wrap:wrap;">{badges}</span></div>')
            
            import math
            mid = math.ceil(len(items)/2)
            c1, c2 = items[:mid], items[mid:]
            lbl_info = f'Threshold: <code>{threshold:.2f}</code>'
            return (f'<div style="display:flex;gap:16px;border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
                    f'<div style="flex:1;display:flex;flex-direction:column;">{"".join(c1)}</div>'
                    f'<div style="flex:1;display:flex;flex-direction:column;border-left:1px solid rgba(226,232,240,0.4);">{"".join(c2)}</div>'
                    f'</div><div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">{lbl_info}</div>')

        # Use manual header for Detected Bias Tokens
        # old ui: h4("Detected Bias Tokens", ...), p(...)
        man_header = ("Detected Bias Tokens", "Each biased token with its category labels and confidence scores.")
        
        # Additional footer: bias_method_info (was separate)
        # Should we include it in the card? The previous UI had `ui.hr(), ui.output_ui("bias_method_info")` INSIDE the card.
        # So we should append it to content.
        # But bias_method_info is a separate renderer. We can't easily call it here unless we duplicate it or it returns string.
        # `create_method_info_html` returns string. So we can just call that.
        try:
            model_key_A = input.bias_model_key()
        except Exception:
            model_key_A = "gusnet-bert"
        
        if (compare_models or compare_prompts) and res_B:
            # Get Model B key for compare models mode
            if compare_models:
                try:
                    model_key_B = input.bias_model_key_B()
                    if not model_key_B:
                        model_key_B = "gusnet-gpt2"
                except Exception:
                    model_key_B = "gusnet-gpt2"
            else:
                # Compare prompts uses the same model for both
                model_key_B = model_key_A
            
            method_html_A = create_method_info_html(model_key_A)
            method_html_B = create_method_info_html(model_key_B)
            method_footer_A = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html_A}</div>'
            method_footer_B = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html_B}</div>'
            
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}</div>' + method_footer_A), manual_header=man_header, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res_B, True)}</div>' + method_footer_B), manual_header=man_header, style="border: 2px solid #ff5ca9; height: 100%;")
            )
        
        method_html = create_method_info_html(model_key_A)
        method_footer = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html}</div>'
        return _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}</div>' + method_footer), manual_header=man_header)

    @output
    @render.ui
    def token_bias_strip():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        # Get selected token index for highlighting
        try:
            sel = input.bias_selected_tokens()
            selected_idx = [int(sel)] if sel else None
        except:
            selected_idx = None
        
        def get_viz(data, sel_idx=None):
            try:
                return create_token_bias_strip(data["token_labels"], sel_idx)
            except Exception as e: return f'<div style="color:red">Error: {e}</div>'

        man_header = ("Token-Level Bias Distribution",
                      "Each token with per-category scores. Coloured dots and values show bias intensity across O, GEN, UNFAIR, and STEREO.")

        if (compare_models or compare_prompts) and res_B:
             return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_strip", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_viz(res_B)), manual_header=man_header, style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_strip_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header,
                          controls=[ui.download_button("export_bias_strip", "CSV", style=_BTN_STYLE_CSV)])

    @output
    @render.ui
    def confidence_breakdown():
        res = bias_results.get()
        if not res:
            return ui.HTML(
                '<div style="color:#9ca3af;padding:20px;text-align:center;">'
                'No analysis results yet.</div>'
            )

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        def get_viz(data):
            try:
                return create_confidence_breakdown(data["token_labels"])
            except Exception as e:
                return f'<div style="color:red">Error: {e}</div>'

        man_header = (
            "Confidence Breakdown",
            "Biased tokens grouped by confidence tier — Low (0.50–0.70), Medium (0.70–0.85), High (0.85+)."
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res)), manual_header=man_header, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(get_viz(res_B)), manual_header=man_header, style="border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _wrap_card(ui.HTML(get_viz(res)), manual_header=man_header)

    @output
    @render.ui
    def combined_bias_view():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')
        
        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except: l_idx, h_idx = 0, 0
        
        sel = None
        try:
            s = input.bias_selected_tokens()
            if s: sel = [int(s)] if isinstance(s,(int,str)) else [int(x) for x in s if x]
        except: pass
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        def get_viz(data, s_idxs, container_id="bias-combined-container"):
            atts = data["attentions"]
            if not atts or l_idx >= len(atts): return '<div style="color:#9ca3af;">No attention data.</div>'
            try:
                if l_idx >= len(atts): return '<div style="color:#9ca3af;">Layer out of bounds.</div>'
                attn = atts[l_idx][0, h_idx].cpu().numpy()
                fig = create_combined_bias_visualization(data["tokens"], data["token_labels"], attn, l_idx, h_idx, selected_token_idx=s_idxs)
                plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False, 'responsive': True})
                resize_script = f'<script>setTimeout(function(){{ var el = document.querySelector("#{container_id} .js-plotly-plot"); if(el) Plotly.Plots.resize(el); }}, 150);</script>'
                return f'<div id="{container_id}" style="height:600px; width:100%;">{plot_html}</div>{resize_script}'
            except Exception as e: return f'<div style="color:red">Error: {e}</div>'

        man_header = ("Combined Attention & Bias View",
                      "Attention weights for a single head with pink row/column highlights on biased tokens. Use the toolbar to change layer and head.")
        card_style = "margin-bottom: 24px; box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, sel, "bias-combined-container")), manual_header=man_header, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container', 'bias_combined_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_B, None, "bias-combined-container-B")), manual_header=man_header, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-B', 'bias_combined_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res, sel, "bias-combined-container")), manual_header=man_header, style=card_style,
                          controls=[
                              ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container', 'bias_combined')", style=_BTN_STYLE_PNG),
                          ])

    @output
    @render.ui
    def attention_bias_matrix():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        # Analyzer instance for metrics
        analyzer = AttentionBiasAnalyzer()

        def get_viz(data, container_id="bias-matrix-container"):
            try:
                if "attentions" not in data or not data["attentions"]:
                     return '<div style="color:#9ca3af;">No attention data.</div>'

                attentions = data["attentions"]
                biased_indices = [l["index"] for l in data["token_labels"] if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
                matrix = analyzer.create_attention_bias_matrix(attentions, biased_indices)
                metrics = data.get("attention_metrics")
                try: sl = int(input.bias_attn_layer())
                except: sl = None

                fig = create_attention_bias_matrix(matrix, metrics=metrics, selected_layer=sl)
                plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False, 'responsive': True})
                resize_script = f'<script>setTimeout(function(){{ var el = document.querySelector("#{container_id} .js-plotly-plot"); if(el) Plotly.Plots.resize(el); }}, 150);</script>'
                return f'<div id="{container_id}" style="height:600px; width:100%;">{plot_html}</div>{resize_script}'
            except Exception as e: return f'<div style="color:red">Error: {e}</div>'

        header_args = ("Bias Attention Matrix",
            "Each cell = ratio of attention a head pays to biased tokens vs. the uniform baseline.",
            "Values centred on 1.0 (neutral). Red > 1.5 = head specializes on biased tokens. Blue < 1.0 = head avoids them.")
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, "bias-matrix-container")), *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_B, "bias-matrix-container-B")), *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-B', 'bias_matrix_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res, "bias-matrix-container")), *header_args, style=card_style,
                          controls=[
                              ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix')", style=_BTN_STYLE_PNG),
                          ])

    @output
    @render.ui
    def bias_propagation_plot():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;">No results.</div>')
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        try: l_idx = int(input.bias_attn_layer())
        except: l_idx = None

        def get_viz(data, container_id="bias-propagation-container"):
            p = data["propagation_analysis"]["layer_propagation"]
            if not p: return "No data."
            fig = create_bias_propagation_plot(p, selected_layer=l_idx)
            plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'displayModeBar': False, 'responsive': True})
            resize_script = f'<script>setTimeout(function(){{ var el = document.querySelector("#{container_id} .js-plotly-plot"); if(el) Plotly.Plots.resize(el); }}, 150);</script>'
            return f'<div id="{container_id}" style="height:450px; width:100%;">{plot_html}</div>{resize_script}'

        header_args = ("Bias Propagation Across Layers",
                      "Mean bias attention ratio per layer — how bias focus evolves through model depth.",
                      "The dashed line at 1.0 represents neutral attention. Values above indicate increasing bias focus.")
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, "bias-propagation-container")), *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_propagation", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container', 'bias_propagation_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_B, "bias-propagation-container-B")), *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_propagation_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container-B', 'bias_propagation_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res, "bias-propagation-container")), *header_args, style=card_style,
                          controls=[
                              ui.download_button("export_bias_propagation", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container', 'bias_propagation')", style=_BTN_STYLE_PNG),
                          ])

    @output
    @render.ui
    def bias_focused_heads_table():
        res = bias_results.get()
        if not res: return None
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except: l_idx, h_idx = -1, -1

        def get_table(data):
            mets = data["attention_metrics"]
            if not mets: return '<div style="color:#9ca3af;padding:20px;text-align:center;font-size:12px;">No metrics available.</div>'
            top = sorted(mets, key=lambda x: x.bias_attention_ratio, reverse=True)[:5]
            rows = []
            for rank, m in enumerate(top, 1):
                is_sel = (m.layer == l_idx and m.head == h_idx)
                is_sig = m.specialized_for_bias

                # Row background
                if is_sel:
                    bg = "background:linear-gradient(90deg, rgba(99, 102, 241, 0.12), rgba(99, 102, 241, 0.05));"
                    border_left = "border-left: 3px solid #6366f1;"
                elif is_sig:
                    bg = "background:rgba(255, 92, 169, 0.03);"
                    border_left = ""
                else:
                    bg = ""
                    border_left = ""

                # Badge for specialization
                if is_sig:
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#22c55e,#16a34a);margin-left:6px;" title="Specialized"><span style="color:white;font-size:8px;font-weight:700;">✓</span></span>'
                else:
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#e2e8f0;margin-left:6px;" title="Not specialized"><span style="color:#94a3b8;font-size:8px;">○</span></span>'

                # Value color
                val_color = "#ff5ca9" if is_sig else "#475569"

                rows.append(
                    f'<tr style="{bg}{border_left}transition:all 0.2s ease;">'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{m.layer}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{m.head}</td>'
                    f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{val_color};">{m.bias_attention_ratio:.3f}{badge}</td>'
                    f'</tr>'
                )

            return (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(rows)}</tbody>'
                '</table>'
            )

        header_args = ("Top Attention Heads by Bias Focus",
                      "Top 5 heads ranked by bias attention ratio. Green dot = above specialization threshold (1.5).",
                      "List of attention heads with the strongest focus on biased tokens, ranked by their Bias Attention Ratio.")
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 16px;"

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_table(res)), *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_table(res_B)), *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_table(res)), *header_args, style=card_style,
                          controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)])


    def _get_label(data, is_A, compare_m):
        if compare_m: return _get_bias_model_label(data)
        return "Prompt A" if is_A else "Prompt B"

__all__ = ["bias_server_handlers"]
