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
    create_ablation_impact_chart,
    create_ig_correlation_chart_v2,
    create_ig_token_comparison_chart,
    create_ig_distribution_chart,
    create_ig_layer_summary_chart,
)
from ..bias.head_ablation import batch_ablate_top_heads, HeadAblationResult
from ..bias.integrated_gradients import batch_compute_ig_correlation, IGCorrelationResult, IGAnalysisBundle
from ..bias.stereoset import (
    load_stereoset_data,
    get_stereoset_scores,
    get_stereoset_examples,
    get_head_sensitivity_matrix,
    get_sensitive_heads,
    get_top_features,
    get_head_profile_stats,
    get_metadata,
)
from ..bias.visualizations import (
    create_stereoset_overview_html,
    create_stereoset_category_chart,
    create_stereoset_head_sensitivity_heatmap,
    create_stereoset_bias_distribution,
    create_stereoset_example_html,
    create_stereoset_attention_heatmaps,
    create_stereoset_attention_diff_heatmap,
)
from ..bias.feature_extraction import extract_attention_for_text
from ..ui.bias_ui import create_bias_accordion, create_floating_bias_toolbar
from ..ui.components import viz_header

# Map GUS-Net model key -> matching encoder model for attention analysis.
# Ensures the encoder tokenizer matches the GUS-Net architecture so that
# BERT tokens show ## subwords and GPT-2 tokens merge correctly with Ġ.
_GUSNET_TO_ENCODER = {
    "gusnet-bert": "bert-base-uncased",
    "gusnet-bert-custom": "bert-base-uncased",
    "gusnet-bert-large": "bert-large-uncased",
    "gusnet-gpt2": "gpt2",
    "gusnet-gpt2-medium": "gpt2-medium",
    "gusnet-ensemble": "bert-base-uncased",
}

def _deferred_plotly(fig, container_id, height=None, config=None):
    """Render a Plotly figure as deferred HTML - only calls Plotly.newPlot()
    when the container becomes visible. This avoids wrong dimensions when
    the container is inside a collapsed accordion panel."""
    import plotly.io as pio
    import base64, html as _html

    # Use figure's internal height if not explicitly overridden
    if height is None:
        fig_height = getattr(fig.layout, "height", None)
        if fig_height:
            height = f"{fig_height}px"
        else:
            height = "400px"

    fig_json = pio.to_json(fig, validate=False)
    cfg = json.dumps(config or {"displayModeBar": False, "responsive": True})
    # Base64-encode the figure JSON and store in a data attribute.
    # This avoids jQuery/Shiny re-parsing <script> tags via .html()
    # and sidesteps any HTML entity issues with <template> elements.
    b64_fig = base64.b64encode(fig_json.encode()).decode()
    escaped_cfg = _html.escape(cfg, quote=True)
    return (
        f'<div id="{container_id}" class="plotly-deferred" '
        f'style="width:100%;height:{height};min-height:50px;"'
        f' data-plotly-config="{escaped_cfg}"'
        f' data-plotly-fig="{b64_fig}">'
        f'</div>'
    )


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

    Handles cross-tokenizer subword mismatches - e.g. when GPT-2 BPE
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
                # Mismatch mid-word - reset and fall through
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
                gus_idx += 1  # advance - remainder will be consumed by next subwords
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
    from attention_app.bias.gusnet_detector import MODEL_REGISTRY
    key = res.get("bias_model_key", "gusnet-bert")
    cfg = MODEL_REGISTRY.get(key, {})
    return cfg.get("display_name", key)

def _process_raw_bias_result(raw_res, thresholds, use_optimized=False):
    """Apply thresholds to raw results and regeneration attention metrics."""
    if not raw_res:
        return None
    
    try:
        bias_model_key = raw_res["bias_model_key"]
        
        # 1. Re-instantiate detector (lightweight) to apply thresholds
        from attention_app.bias.gusnet_detector import GusNetDetector, EnsembleGusNetDetector
        
        if bias_model_key == "gusnet-ensemble":
            det = EnsembleGusNetDetector(model_key_a="gusnet-bert", model_key_b="gusnet-bert-custom")
        else:
            det = GusNetDetector(model_key=bias_model_key, use_optimized=use_optimized)
            
        # 2. Apply thresholds
        gusnet_labels = det.apply_thresholds(
            raw_res["gus_tokens"], 
            raw_res["gus_probs"], 
            thresholds=thresholds
        )
        
        # 3. Re-align (alignment needed because encoder tokens != gus tokens)
        tokens = raw_res["tokens"]
        gus_special = raw_res["gus_special"]
        
        # We need _align_gusnet function available here
        gus_aligned = _align_gusnet_to_attention_tokens(
            gusnet_labels, tokens, gusnet_special_tokens=gus_special
        )
        
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
                    "threshold": thresholds.get("GEN", 0.5) if thresholds else 0.5,
                })
        
        # 4. Re-calculate Summary & Spans
        bias_summary = det.get_bias_summary(token_labels)
        bias_spans = det.get_biased_spans(token_labels)
        
        # 5. Re-calculate Attention Metrics (Attention Analyzer)
        # Only if we have attention data
        attentions = raw_res["attentions"]
        biased_indices = [i for i, l in enumerate(token_labels) if l["is_biased"]]
        
        attention_metrics = []
        propagation_analysis = {
            "layer_propagation": [], "peak_layer": None,
            "propagation_pattern": "none",
        }
        bias_matrix = np.array([])
        
        if biased_indices and attentions:
            attention_analyzer = AttentionBiasAnalyzer()
            attention_metrics = attention_analyzer.analyze_attention_to_bias(
                list(attentions), biased_indices, tokens
            )
            propagation_analysis = attention_analyzer.analyze_bias_propagation(
                list(attentions), biased_indices, tokens
            )
            bias_doc_matrix = attention_analyzer.create_attention_bias_matrix(
                list(attentions), biased_indices
            )
            # bias_doc_matrix is a numpy array>= Helper usually returns something else>=
            # Warning: Existing code was: bias_matrix = attention_analyzer.create_attention_bias_matrix(...)
            bias_matrix = bias_doc_matrix

        # Return full result structure
        return {
            **raw_res,
            "token_labels": token_labels,
            "bias_summary": bias_summary,
            "bias_spans": bias_spans,
            "biased_indices": biased_indices,
            "attention_metrics": attention_metrics,
            "propagation_analysis": propagation_analysis,
            "bias_matrix": bias_matrix,
            "thresholds": thresholds,
            "use_optimized": use_optimized
        }
        
    except Exception as e:
        print(f"Error processing raw bias result: {e}")
        traceback.print_exc()
        return None


def bias_server_handlers(input, output, session):
    """Create server handlers for bias analysis tab."""

    bias_running = reactive.value(False)
    bias_raw_results = reactive.value(None)  # Raw probs (no threshold)
    bias_raw_results_B = reactive.value(None)
    bias_results = reactive.value(None)      # Final labels (after threshold)
    bias_results_B = reactive.value(None)    # For comparison
    bias_history = reactive.Value([])
    ablation_results = reactive.value(None)
    ablation_results_B = reactive.value(None)
    ablation_running = reactive.value(False)
    ig_results = reactive.value(None)
    ig_results_B = reactive.value(None)
    ig_running = reactive.value(False)

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

    # ── Defaults Logic ──
    @reactive.Effect
    def update_threshold_defaults():
        """Update UI sliders with model-specific optimized thresholds for Model A and B."""
        vk_A = input.bias_model_key()
        
        # Determine if we need thresholds for B (Models or Prompts comparison)
        compare_active = False
        vk_B = None
        try:
            if input.bias_compare_mode():
                compare_active = True
                vk_B = input.bias_model_key_B()
            elif input.bias_compare_prompts_mode():
                compare_active = True
                vk_B = vk_A # Same model for both prompts
        except Exception:
            pass

        from attention_app.bias.gusnet_detector import MODEL_REGISTRY
        
        def _get_model_defaults(vk):
            if not vk:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}
                
            cfg = MODEL_REGISTRY.get(vk)
            if not cfg:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}
            
            opt = cfg.get("optimized_thresholds")
            defaults = {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}
            
            if opt:
                cat_indices = cfg.get("category_indices", {})
                for cat, indices in cat_indices.items():
                    if cat in defaults:
                        vals = [opt[i] for i in indices if i < len(opt)]
                        if vals:
                            defaults[cat] = sum(vals) / len(vals)
            return defaults

        # Get thresholds for A
        defaults = _get_model_defaults(vk_A)
        
        # Get thresholds for B if any comparison mode is active
        if compare_active:
            defaults_B = _get_model_defaults(vk_B)
            # Prefix with _B for the UI handler
            for cat, val in defaults_B.items():
                defaults[f"{cat}_B"] = val
        
        # Send to UI
        if defaults:
            asyncio.create_task(session.send_custom_message("set_bias_thresholds", defaults))

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
    @reactive.event(input.bias_thresh_unfair, input.bias_thresh_gen, input.bias_thresh_stereo,
                   input.bias_thresh_unfair_b, input.bias_thresh_gen_b, input.bias_thresh_stereo_b)
    def update_bias_results_live():
        """Update bias results when threshold sliders change (without re-running inference).
        
        This only reacts to threshold slider changes, not to other reactive values.
        """
        # GUARD: Don't run if analysis is in progress
        if bias_running.get():
            return
            
        raw_A = bias_raw_results.get()
        raw_B = bias_raw_results_B.get()
        
        if not raw_A and not raw_B:
            return

        # GUARD: Only update if the LIVE model selection matches the ANALYZED model.
        try:
            live_key_A = input.bias_model_key()
            if raw_A and raw_A.get("bias_model_key") != live_key_A:
                return
        except Exception:
            pass

        # Check optimization flag
        try:
            use_opt = input.bias_use_optimized()
        except: use_opt = False

        # Read manual sliders
        try: t_unfair = float(input.bias_thresh_unfair())
        except: t_unfair = 0.5
        try: t_gen = float(input.bias_thresh_gen())
        except: t_gen = 0.5
        try: t_stereo = float(input.bias_thresh_stereo())
        except: t_stereo = 0.5
        
        manual_thresholds_A = {
            "UNFAIR": t_unfair,
            "GEN": t_gen,
            "STEREO": t_stereo
        }
        
        # Read B sliders
        try: t_unfair_b = float(input.bias_thresh_unfair_b())
        except: t_unfair_b = 0.5
        try: t_gen_b = float(input.bias_thresh_gen_b())
        except: t_gen_b = 0.5
        try: t_stereo_b = float(input.bias_thresh_stereo_b())
        except: t_stereo_b = 0.5

        manual_thresholds_B = {
            "UNFAIR": t_unfair_b,
            "GEN": t_gen_b,
            "STEREO": t_stereo_b
        }
        
        # Guard for B key match
        update_B = False
        if raw_B:
            try:
                in_compare_mode = active_bias_compare_models.get() or active_bias_compare_prompts.get()
                if in_compare_mode:
                    live_key_B = input.bias_model_key_B()
                    expected_key_B = raw_B.get("bias_model_key")
                    if active_bias_compare_prompts.get() and raw_A:
                        expected_key_B = raw_A.get("bias_model_key")
                    if live_key_B and expected_key_B and expected_key_B == live_key_B:
                        update_B = True
            except Exception:
                pass
        
        # Process A
        if raw_A:
            res_A = _process_raw_bias_result(raw_A, manual_thresholds_A, use_optimized=use_opt)
            bias_results.set(res_A)
            
        # Process B
        if raw_B and update_B:
            res_B = _process_raw_bias_result(raw_B, manual_thresholds_B, use_optimized=use_opt)
            bias_results_B.set(res_B)

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
    @reactive.event(input.bias_compare_prompts_mode, input.bias_compare_mode, bias_prompt_step)
    async def update_bias_button_label():
        """Update the Analyze Bias button label based on mode and step.
        
        - In Compare Prompts mode on Step A: Show "Prompt B ->"
        - In Compare Prompts mode on Step B: Show "Analyze Bias"
        - In Compare Models mode: Show "Analyze Bias" (both models at once)
        - In single mode: Show "Analyze Bias"
        
        Also resets prompt step when modes are toggled off.
        """
        try:
            prompts_mode = input.bias_compare_prompts_mode()
        except Exception:
            prompts_mode = False
            
        try:
            models_mode = input.bias_compare_mode()
        except Exception:
            models_mode = False
            
        step = bias_prompt_step.get()
        label = "Analyze Bias"
        
        # Reset to step A when compare prompts mode is turned off
        if not prompts_mode and step == "B":
            bias_prompt_step.set("A")
            # Also sync the UI tab to A
            await session.send_custom_message("bias_eval_js", 
                "if(window.switchBiasPromptTab) window.switchBiasPromptTab('A');")
        
        # In compare prompts mode on Step A: show "Prompt B ->"
        if prompts_mode and step == "A":
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

    def _safe_threshold():
        """Read bias_threshold input, falling back to 0.5 if unavailable."""
        try:
            return float(input.bias_threshold())
        except Exception:
            return 0.5

    # ── Session Logic ──

    @render.download(filename=lambda: f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def save_bias_session():
        data = {
            "type": "bias_analysis",
            "timestamp": datetime.now().isoformat(),
            "text": input.bias_input_text(),
            "threshold": _safe_threshold(),
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
            
    def heavy_bias_compute(text, model_name, threshold, bias_model_key, use_optimized=True):
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
        use_optimized : bool
            Whether to use per-label optimized thresholds (if available).
        """
        log_debug(f"Starting heavy_bias_compute (threshold={threshold}, bias_model={bias_model_key}, use_optimized={use_optimized})")
        print(f"DEBUG: Starting heavy_bias_compute (threshold={threshold}, bias_model={bias_model_key}, use_optimized={use_optimized})")
        if not text:
            log_debug("No text provided")
            print("DEBUG: No text provided")
            return None

        try:
            # Clear cache before starting heavy operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                try:
                    outputs = encoder_model(**inputs)
                except torch.cuda.OutOfMemoryError:
                    # Clear cache and try again>= No, just fail gracefully
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "GPU Out of Memory. usage is high. "
                        "Please stop the background 'generate_stereoset_json' script "
                        "or other heavy processes."
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                         torch.cuda.empty_cache()
                         raise RuntimeError(
                            "GPU Out of Memory. usage is high. "
                            "Please stop the background 'generate_stereoset_json' script "
                            "or other heavy processes."
                        )
                    raise e
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
                    threshold=0.5, # Placeholder
                )
            else:
                gus_detector = GusNetDetector(
                    model_key=bias_model_key, threshold=0.5,
                    use_optimized=False,
                )
            log_debug("Running predict_proba...")
            print("DEBUG: Running predict_proba...")
            
            # Helper to get raw probs.
            gus_tokens, gus_probs = gus_detector.predict_proba(text)
            
            log_debug(f"GUS-Net returned {len(gus_tokens)} tokens and {gus_probs.shape} probs")

            # Calculate effective thresholds for UI feedback
            effective_thresholds = {}
            if use_optimized and gus_detector.config.get("optimized_thresholds"):
                opt = gus_detector.config["optimized_thresholds"]
                cat_indices = gus_detector.config.get("category_indices", {})
                for cat in ["GEN", "UNFAIR", "STEREO"]:
                    if cat in cat_indices:
                        indices = cat_indices[cat]
                        # Average the per-label thresholds for the category to give a representative slider value
                        vals = [opt[i] for i in indices if i < len(opt)]
                        if vals:
                            effective_thresholds[cat] = sum(vals) / len(vals)
            else:
                if isinstance(threshold, dict):
                    effective_thresholds = threshold
                else:
                    effective_thresholds = {
                        "GEN": threshold, 
                        "UNFAIR": threshold, 
                        "STEREO": threshold
                    }

            # Return RAW data
            return {
                "tokens": tokens, # Encoder tokens
                "text": text,
                "attentions": attentions,
                "gus_tokens": gus_tokens, 
                "gus_probs": gus_probs,
                "bias_model_key": bias_model_key,
                "model_name": model_name,
                "gus_special": gus_detector.config["special_tokens"],
                "effective_thresholds": effective_thresholds
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

            # MODE SWITCHING CLEANUP: Reset state when modes change
            prev_compare_models = active_bias_compare_models.get()
            prev_compare_prompts = active_bias_compare_prompts.get()
            
            # Detect mode transitions
            mode_switched_to_prompts = compare_prompts and not prev_compare_prompts
            mode_switched_to_models = compare_models and not prev_compare_models
            mode_switched_off = (not compare_prompts and not compare_models) and (prev_compare_prompts or prev_compare_models)
            
            # Reset state when switching modes to ensure clean state
            if mode_switched_to_prompts or mode_switched_to_models or mode_switched_off:
                log_debug("Mode switch detected - resetting bias state")
                bias_raw_results.set(None)
                bias_raw_results_B.set(None)
                bias_results.set(None)
                bias_results_B.set(None)
                bias_cached_text_A.set("")
                bias_cached_text_B.set("")
                # Reset prompt step to A when switching modes
                bias_prompt_step.set("A")
                # Ensure UI reflects the reset
                await session.send_custom_message("bias_eval_js", 
                    "if(window.switchBiasPromptTab) window.switchBiasPromptTab('A');")

            active_bias_compare_models.set(compare_models)
            active_bias_compare_prompts.set(compare_prompts)

            # Clear previous results
            bias_results.set(None)
            bias_results_B.set(None)
            bias_cached_text_A.set(text)

            try:
                use_optimized = bool(input.bias_use_optimized())
            except Exception:
                use_optimized = True

            try:
                t_unfair = float(input.bias_thresh_unfair())
                t_gen = float(input.bias_thresh_gen())
                t_stereo = float(input.bias_thresh_stereo())
                thresholds_A = {"UNFAIR": t_unfair, "GEN": t_gen, "STEREO": t_stereo}
                
                # Fetch B thresholds if needed
                thresholds_B = {"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5}
                if compare_models:
                    try:
                        thresholds_B = {
                            "UNFAIR": float(input.bias_thresh_unfair_b()),
                            "GEN": float(input.bias_thresh_gen_b()),
                            "STEREO": float(input.bias_thresh_stereo_b())
                        }
                    except: pass
                elif compare_prompts:
                    thresholds_B = thresholds_A

                # Mean placeholder for legacy compatibility
                threshold = sum(thresholds_A.values()) / 3
                log_debug(f"Thresholds A: {thresholds_A}, B: {thresholds_B}")
            except Exception as e:
                log_debug(f"Warning: category thresholds missing ({e}). Using 0.5")
                threshold = 0.5
                thresholds_A = {"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5}
                thresholds_B = {"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5}
            log_debug(f"use_optimized={use_optimized}, threshold={threshold}")

            # Bias detection model (BERT or GPT-2 backbone) - Model A
            try:
                bias_model_key = input.bias_model_key()
                log_debug(f"Bias model key: {bias_model_key}")
            except Exception as e:
                log_debug(f"Warning: bias_model_key missing ({e}). Using default gusnet-bert")
                bias_model_key = "gusnet-bert"

            # Derive encoder model from bias_model_key so tokenization matches
            model_name = _GUSNET_TO_ENCODER.get(bias_model_key, "bert-base-uncased")
            log_debug(f"Encoder model (derived from {bias_model_key}): {model_name}")

            try:
                loop = asyncio.get_running_loop()
                log_debug("Entering ThreadPoolExecutor...")
                with ThreadPoolExecutor() as pool:
                    # Compute Result A
                    log_debug("Submitting heavy_bias_compute for Model A...")
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            pool, heavy_bias_compute, text, model_name, thresholds_A, bias_model_key, use_optimized,
                        ),
                        timeout=180.0
                    )
                    log_debug("heavy_bias_compute A returned successfully")
                    # Store RAW result so sliders can re-threshold live
                    bias_raw_results.set(result)
                    # Initial processing for Model A (using thresholds we just grabbed)
                    effective_A = result.get("effective_thresholds", thresholds_A)
                    res_processed = _process_raw_bias_result(result, effective_A, use_optimized=use_optimized)
                    bias_results.set(res_processed)

                    # Update UI sliders with the actual thresholds used
                    if result and result.get("effective_thresholds"):
                        log_debug(f"Sending effective thresholds to UI: {result['effective_thresholds']}")
                        await session.send_custom_message("set_bias_thresholds", result["effective_thresholds"])

                    # Compute Result B if needed
                    if compare_models:
                        # Case 1: Same Prompt (A), Different Model (B)
                        try:
                            bias_model_key_B = input.bias_model_key_B()
                            if not bias_model_key_B:
                                bias_model_key_B = "gusnet-gpt2"
                        except Exception:
                            bias_model_key_B = "gusnet-gpt2"

                        model_name_B = _GUSNET_TO_ENCODER.get(bias_model_key_B, "gpt2")
                        log_debug(f"Starting heavy_bias_compute B ({bias_model_key_B}, encoder={model_name_B}) for Compare Models")
                        result_B = await asyncio.wait_for(
                            loop.run_in_executor(
                                pool, heavy_bias_compute, text, model_name_B, thresholds_B, bias_model_key_B, use_optimized,
                            ),
                            timeout=180.0
                        )
                        bias_raw_results_B.set(result_B) # Store RAW result B
                        bias_cached_text_B.set(text)  # Same text for compare models
                        
                        # Process B immediately for UI
                        effective_B = result_B.get("effective_thresholds", thresholds_B)
                        res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                        bias_results_B.set(res_B_proc)
                        
                        # Send effective thresholds for B
                        if result_B and result_B.get("effective_thresholds"):
                            eff_B = result_B["effective_thresholds"]
                            msg_B = {
                                "UNFAIR_B": eff_B.get("UNFAIR"),
                                "GEN_B": eff_B.get("GEN"),
                                "STEREO_B": eff_B.get("STEREO")
                            }
                            log_debug(f"Sending effective thresholds B to UI: {msg_B}")
                            await session.send_custom_message("set_bias_thresholds", msg_B)

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
                                    pool, heavy_bias_compute, text_B, model_name, thresholds_B, bias_model_key, use_optimized,
                                ),
                                timeout=180.0
                            )
                            bias_raw_results_B.set(result_B)
                            bias_cached_text_B.set(text_B)
                            
                            # Process B immediately for UI
                            effective_B = result_B.get("effective_thresholds", thresholds_B)
                            res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                            bias_results_B.set(res_B_proc)
                            
                            # Send effective thresholds for B
                            if result_B and result_B.get("effective_thresholds"):
                                eff_B = result_B["effective_thresholds"]
                                msg_B = {
                                    "UNFAIR_B": eff_B.get("UNFAIR"),
                                    "GEN_B": eff_B.get("GEN"),
                                    "STEREO_B": eff_B.get("STEREO")
                                }
                                log_debug(f"Sending effective thresholds B to UI: {msg_B}")
                                await session.send_custom_message("set_bias_thresholds", msg_B)
                        else:
                            # Empty text B - clear results and show notification
                            bias_raw_results_B.set(None)
                            bias_cached_text_B.set("")
                            bias_results_B.set(None)
                            ui.notification_show("Prompt B is empty. Only analyzing Prompt A.", type="warning", duration=3)
                    else:
                        bias_raw_results_B.set(None)
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
                    # Keep the detailed label for the card or just use generic>=
                    # User asked for "MODEL A" header.
                else:
                    header_A = "PROMPT A"
                    header_B = "PROMPT B"

                return ui.div(
                    {"style": "display: flex; flex-direction: column; gap: 24px;"},
                    # Side-by-side sentence previews
                    ui.div(
                        {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                        # Column A
                        ui.div(
                            {"style": "display: flex; flex-direction: column;"},
                            ui.h3(header_A, style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                            ui.div(
                                {"class": "card compare-card-a", "style": "min-height: 140px; flex: 1;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_A),
                            ),
                        ),
                        # Column B
                        ui.div(
                            {"style": "display: flex; flex-direction: column;"},
                            ui.h3(header_B, style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                            ui.div(
                                {"class": "card compare-card-b", "style": "min-height: 140px; flex: 1;"},
                                ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                                ui.HTML(preview_html_B),
                            ),
                        ),
                    ),
                    # Summary comparison cards
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.div(style="height: 110px;"), # Spacer to clear floating bar
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
                        {"class": "card", "style": "min-height: 140px;"},
                        ui.h4("Sentence Preview"),
                        ui.HTML(preview_html),
                    ),
                    create_bias_accordion(),
                    create_floating_bias_toolbar(),
                    ui.div(style="height: 110px;"), # Spacer to clear floating bar
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
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                # Column A
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("PROMPT A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a", "style": "min-height: 140px; flex: 1;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_A,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("PROMPT B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b", "style": "min-height: 140px; flex: 1;"},
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
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 32px;"},
                # Column A
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("MODEL A", style="font-size:16px; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #3b82f6;"),
                    ui.div(
                        {"class": "card compare-card-a", "style": "min-height: 140px; flex: 1;"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_content,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("MODEL B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b", "style": "min-height: 140px; flex: 1;"},
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
                {"class": "card", "style": "min-height: 140px;"},
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
        if res:
            selected_model = res.get("bias_model_key", "gusnet-bert")
        else:
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
        # Use analyzed model key if results exist, otherwise live input
        res = bias_results.get()
        if res:
            model_key = res.get("bias_model_key", "gusnet-bert")
        else:
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
            threshold = _safe_threshold()
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
            return ui.HTML(_deferred_plotly(fig, "token-bias-heatmap"))
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

        # Use pre-calculated spans (merged subwords) instead of raw tokens
        bias_spans = res.get("bias_spans", [])
        
        if not bias_spans:
            return ui.HTML('<div style="color:#94a3b8;font-size:11px;padding:12px;">No bias detected.</div>')

        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}
        # Update badge count
        badge_script = f"<script>$('#bias-span-count-badge').text('{len(bias_spans)}');</script>"

        items = [badge_script]
        for span in bias_spans:
            text = span.get("text", " ".join(span["tokens"])) # Fallback if text not present
            score = span.get("avg_score", 0.0)
            types = span.get("bias_types", [])
            
            score_color = "#ef4444" if score > 0.8 else "#f59e0b" if score > 0.5 else "#94a3b8"

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
                f'color:#e2e8f0;font-weight:600;">{text}</span>'
                f'<span style="color:{score_color};font-weight:600;font-size:11px;'
                f'font-family:JetBrains Mono,monospace;">{score:.2f}</span></div>'
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
            return ui.HTML('<span style="color:#64748b;font-size:9px;">--</span>')

        metrics = res.get("attention_metrics", [])
        if not metrics:
            return ui.HTML('<span style="color:#64748b;font-size:9px;">--</span>')

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

        In compare prompts mode, matches the screenshot:
        - Row A (top): Blue label "A", tokens to the right.
        - Separator line.
        - Row B (bottom): Pink label "B", tokens to the right.
        """
        cat_colors = {"GEN": "#f97316", "UNFAIR": "#ef4444", "STEREO": "#9c27b0"}

        def _build_chips_html(res, variant="A"):
            """Build HTML chips for a single result set."""
            if not res:
                return '<span style="color:#64748b;font-size:10px;padding:2px;">No analysis</span>'
            token_labels = res["token_labels"]
            biased = [
                lbl for lbl in token_labels
                if lbl.get("is_biased") and lbl["token"] not in ("[CLS]", "[SEP]", "[PAD]")
            ]
            if not biased:
                return '<span style="color:#64748b;font-size:10px;padding:2px;">No bias detected</span>'

            # Base color for chips depending on variant
            base_color = "#3b82f6" if variant == "A" else "#ff5ca9"

            items = []
            for lbl in biased:
                clean = lbl["token"].replace("##", "").replace("\u0120", "")
                tok_idx = lbl["index"]
                
                # Use category color if available, else variant default
                types = lbl.get("bias_types", [])
                item_color = cat_colors.get(types[0], base_color) if types else base_color

                # Use native .token-chip class with data-prefix for styling
                items.append(
                    f'<span class="token-chip" '
                    f'data-token-idx="{tok_idx}" '
                    f'data-prefix="{variant}" '
                    f'onclick="selectBiasToken({tok_idx})">'
                    f'{clean}'
                    f'</span>'
                )
            return "".join(items)

        compare_prompts = active_bias_compare_prompts.get()

        if compare_prompts:
            # ── Compare Prompts Layout ──
            res_A = bias_results.get()
            res_B = bias_results_B.get()
            chips_A = _build_chips_html(res_A, "A")
            chips_B = _build_chips_html(res_B, "B")

            return ui.HTML(
                f'<div class="token-sentence" style="flex-direction:column;flex-wrap:nowrap;align-items:stretch;padding:2px 6px;">'
                # Row A
                f'<div style="display:flex;align-items:center;min-height:24px;">'
                f'<div style="margin-right:8px;margin-left:8px;min-width:15px;text-align:center;color:#60a5fa;font-size:10px;font-weight:600;">A</div>'
                f'<div id="bias-tokens-a" style="flex:1;display:flex;overflow-x:auto;scrollbar-width:none;align-items:center;gap:4px;">{chips_A}</div>'
                f'</div>'
                # Row B
                f'<div style="display:flex;align-items:center;min-height:24px;border-top:1px solid rgba(233, 30, 99, 0.3);">'
                f'<div style="margin-right:8px;margin-left:8px;min-width:15px;text-align:center;color:#E91E63;font-size:10px;font-weight:600;">B</div>'
                f'<div id="bias-tokens-b" onscroll="document.getElementById(\'bias-tokens-a\').scrollLeft = this.scrollLeft" '
                f'style="flex:1;display:flex;overflow-x:auto;scrollbar-width:none;align-items:center;gap:4px;">{chips_B}</div>'
                f'</div>'
                f'</div>'
            )
        else:
            # ── Single / Compare Models ──
            res = bias_results.get()
            chips = _build_chips_html(res, "A")
            return ui.HTML(
                f'<div class="token-sentence">'
                f'{chips}'
                f'</div>'
            )


    # ── Comparison Refactored Renderers ──

    @output
    @render.ui
    def bias_spans_table():
        res = bias_results.get()
        if not res: return None
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        # ... logic to build HTML items ...
        def produce_html(data, is_B):
            if not data:
                return f'<div style="color:#9ca3af;font-size:12px;padding:12px;">Run analysis to see {"Model B" if is_B else "Model A"} results.</div>'
                
            token_labels = data.get("token_labels", [])
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
            
            # Show per-category thresholds
            t_info = "Thresholds: "
            if data and data.get("thresholds"):
                ts = data["thresholds"]
                parts = []
                for k in ["UNFAIR", "GEN", "STEREO"]:
                    if k in ts:
                        parts.append(f'{k}: <code>{float(ts[k]):.2f}</code>')
                t_info += " &bull; ".join(parts)
                if data.get("use_optimized"):
                    t_info += " <span style='opacity:0.6;'>(Optimized)</span>"
            else:
                t_info += f"<code>{_safe_threshold():.2f}</code>"

            return (f'<div style="display:flex;gap:16px;border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
                    f'<div style="flex:1;display:flex;flex-direction:column;">{"".join(c1)}</div>'
                    f'<div style="flex:1;display:flex;flex-direction:column;border-left:1px solid rgba(226,232,240,0.4);">{"".join(c2)}</div>'
                    f'</div><div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">{t_info}</div>')

        # Use manual header for Detected Bias Tokens
        # old ui: h4("Detected Bias Tokens", ...), p(...)
        man_header = ("Detected Bias Tokens", "Each biased token with its category labels and confidence scores.")
        
        # Additional footer: bias_method_info (was separate)
        # Should we include it in the card? The previous UI had `ui.hr(), ui.output_ui("bias_method_info")` INSIDE the card.
        # So we should append it to content.
        # But bias_method_info is a separate renderer. We can't easily call it here unless we duplicate it or it returns string.
        # `create_method_info_html` returns string. So we can just call that.
        if res and "bias_model_key" in res:
            model_key_A = res["bias_model_key"]
        else:
            try:
                model_key_A = input.bias_model_key()
            except Exception:
                model_key_A = "gusnet-bert"
        
        if (compare_models or compare_prompts) and res_B:
            # Get Model B key for compare models mode
            if res_B and "bias_model_key" in res_B:
                model_key_B = res_B["bias_model_key"]
            else:
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
        selected_idx = None
        try:
            sel = input.bias_selected_tokens()
            if sel:
                selected_idx = (
                    [int(sel)] if isinstance(sel, (int, str))
                    else [int(x) for x in sel if x is not None]
                )
        except Exception:
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

        # Get selected token index for highlighting
        selected_idx = None
        try:
            sel = input.bias_selected_tokens()
            if sel:
                selected_idx = (
                    [int(sel)] if isinstance(sel, (int, str))
                    else [int(x) for x in sel if x is not None]
                )
        except Exception:
            selected_idx = None

        def get_viz(data, sel_idx=None):
            try:
                return create_confidence_breakdown(data["token_labels"], selected_token_idx=sel_idx)
            except Exception as e:
                return f'<div style="color:red">Error: {e}</div>'

        man_header = (
            "Confidence Breakdown",
            "Biased tokens grouped by confidence tier - Low (0.50-0.70), Medium (0.70-0.85), High (0.85+)."
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(get_viz(res_B)), manual_header=man_header, style="border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header)

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
                return _deferred_plotly(fig, container_id, height="600px")
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

                try: _bar_th = float(input.bias_bar_threshold())
                except Exception: _bar_th = 1.5
                fig = create_attention_bias_matrix(matrix, metrics=metrics, selected_layer=sl, bar_threshold=_bar_th)
                return _deferred_plotly(fig, container_id, height="600px")
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
            return _deferred_plotly(fig, container_id, height="450px")

        header_args = ("Bias Propagation Across Layers",
                      "Mean bias attention ratio per layer - how bias focus evolves through model depth.",
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

        # Read dynamic Top-K and BAR threshold
        try: k = int(input.bias_top_k())
        except Exception: k = 5
        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        def get_table(data):
            mets = data["attention_metrics"]
            if not mets: return '<div style="color:#9ca3af;padding:20px;text-align:center;font-size:12px;">No metrics available.</div>'
            top = sorted(mets, key=lambda x: x.bias_attention_ratio, reverse=True)[:k]
            any_above = any(m.bias_attention_ratio > bar_threshold for m in top)

            # Note when no heads exceed threshold
            note_html = ""
            if not any_above and top:
                note_html = (
                    f'<div style="padding:8px 12px;margin-bottom:8px;background:rgba(245,158,11,0.08);'
                    f'border:1px solid rgba(245,158,11,0.2);border-radius:6px;font-size:11px;color:#92400e;line-height:1.5;">'
                    f'No heads exceed the specialization threshold ({bar_threshold:.1f}). '
                    f'Showing top-{len(top)} by BAR value. Bias may be lexical rather than concentrated in specific attention heads.'
                    f'</div>'
                )

            rows = []
            for rank, m in enumerate(top, 1):
                is_sel = (m.layer == l_idx and m.head == h_idx)
                is_sig = m.bias_attention_ratio > bar_threshold

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
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:linear-gradient(135deg,#22c55e,#16a34a);margin-left:6px;" title="Specialized"><span style="color:white;font-size:8px;font-weight:700;">OK</span></span>'
                else:
                    badge = '<span style="display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#e2e8f0;margin-left:6px;" title="Not specialized"><span style="color:#94a3b8;font-size:8px;">o</span></span>'

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
                f'{note_html}'
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

        header_args = (f"Top {k} Attention Heads by Bias Focus",
                      f"Ranked by bias attention ratio. Green dot = above specialization threshold ({bar_threshold:.1f}).",
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


    # ── Ablation handlers ─────────────────────────────────────────────

    @reactive.effect
    async def compute_ablation():
        """Automatically run ablation when bias analysis is complete."""
        res_A = bias_results.get()
        if not res_A:
            return

        ablation_running.set(True)
        ablation_results.set(None)
        ablation_results_B.set(None)

        try:
            try: k = int(input.bias_top_k())
            except Exception: k = 5

            # Define helper for single computation
            def _compute_single(r):
                text = r["text"]
                metrics = r.get("attention_metrics", [])
                if not metrics: return None

                # Use top heads from THIS result
                top_heads_local = sorted(
                    metrics, key=lambda m: m.bias_attention_ratio, reverse=True
                )[:k]

                model_name = r.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name

                tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)
                
                # We need to run sync code in executor
                return (encoder_model, lm_head_model, tokenizer, text, top_heads_local, is_g)

            # Prepare args for A
            args_A = _compute_single(res_A)
            
            # Prepare args for B if needed
            args_B = None
            compare = (active_bias_compare_models.get() or active_bias_compare_prompts.get())
            if compare:
                res_B = bias_results_B.get()
                if res_B:
                    args_B = _compute_single(res_B)

            # Execute
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=2) as pool:
                # Launch A
                fut_A = None
                if args_A:
                    fut_A = loop.run_in_executor(
                        pool,
                        batch_ablate_top_heads,
                        *args_A
                    )
                
                # Launch B
                fut_B = None
                if args_B:
                    fut_B = loop.run_in_executor(
                        pool,
                        batch_ablate_top_heads,
                        *args_B
                    )

                # Wait results
                if fut_A:
                    res_A_calc = await fut_A
                    ablation_results.set(res_A_calc)
                
                if fut_B:
                    res_B_calc = await fut_B
                    ablation_results_B.set(res_B_calc)

        except Exception as e:
            print(f"Ablation error: {e}")
            traceback.print_exc()
        finally:
            ablation_running.set(False)

    @output
    @render.ui
    def ablation_results_display():
        running = ablation_running.get()
        results_A = ablation_results.get()

        if running or not results_A:
            return None
        
        results_B = ablation_results_B.get()
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        # Show comparison if comparing models OR comparing prompts
        show_comparison = (compare_models or compare_prompts) and results_B

        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        # Get selected head for highlighting
        selected_head = None
        try:
            sel_l = int(input.bias_attn_layer())
            sel_h = int(input.bias_attn_head())
            selected_head = (sel_l, sel_h)
        except Exception:
            pass

        def _render_ablation_single(results_data, container_suffix=""):
            if not results_data: return "No data"
            
            fig = create_ablation_impact_chart(results_data, bar_threshold=bar_threshold, selected_head=selected_head)
            c_id = f"ablation-chart-container{container_suffix}"
            chart_html = _deferred_plotly(fig, c_id)

            table_rows = []
            for rank, r in enumerate(results_data, 1):
                impact_color = "#ff5ca9" if r.representation_impact > 0.05 else "#64748b"
                kl_cell = f"{r.kl_divergence:.4f}" if r.kl_divergence is not None else "N/A"
                specialized = "Yes" if r.bar_original > bar_threshold else "No"
                row_bg = "background:rgba(255,92,169,0.12);" if selected_head and (r.layer, r.head) == selected_head else ""
                table_rows.append(
                    f'<tr style="{row_bg}">'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">#{rank}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{r.layer}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{r.head}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{impact_color};">{r.representation_impact:.4f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{kl_cell}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.bar_original:.3f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;">{specialized}</td>'
                    f'</tr>'
                )

            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:16px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Impact</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">KL Div</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Specialized</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(table_rows)}</tbody>'
                '</table>'
            )
            
            return ui.div(
                ui.HTML(chart_html),
                ui.HTML(table_html),
            )

        header_args = (
            "Head Ablation Results",
            "Measures the causal impact of each attention head by zeroing its output and measuring the change in the model's internal representation.",
            "Higher impact means removing this head significantly alters the output. Compare with BAR to assess faithfulness."
        )

        if show_comparison:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_ablation_single(results_A, "_A"), *header_args, 
                           style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_ablation_csv", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(_render_ablation_single(results_B, "_B"), *header_args, 
                           style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_ablation_csv_B", "CSV", style=_BTN_STYLE_CSV)])
            )

        return _wrap_card(
            _render_ablation_single(results_A),
            *header_args,
            controls=[
                ui.download_button("export_ablation_csv", "CSV", style=_BTN_STYLE_CSV),
                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ablation-chart-container', 'ablation_impact')", style=_BTN_STYLE_PNG),
            ],
        )

    @render.download(filename="ablation_results.csv")
    def export_ablation_csv():
        results = ablation_results.get()
        if not results:
            yield "No ablation data"
            return
        lines = ["rank,layer,head,representation_impact,kl_divergence,bar_original"]
        for i, r in enumerate(results, 1):
            kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else ""
            lines.append(f"{i},{r.layer},{r.head},{r.representation_impact:.6f},{kl},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="ablation_results_B.csv")
    def export_ablation_csv_B():
        results = ablation_results_B.get()
        if not results:
            yield "No ablation data"
            return
        lines = ["rank,layer,head,representation_impact,kl_divergence,bar_original"]
        for i, r in enumerate(results, 1):
            kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else ""
            lines.append(f"{i},{r.layer},{r.head},{r.representation_impact:.6f},{kl},{r.bar_original:.4f}")
        yield "\n".join(lines)

    # ── Integrated Gradients handlers ─────────────────────────────────

    @reactive.effect
    async def compute_ig():
        """Automatically run IG correlation when bias analysis is complete."""
        res_A = bias_results.get()
        if not res_A:
            return

        ig_running.set(True)
        ig_results.set(None)
        ig_results_B.set(None)

        try:
            # Helper for single computation config
            def _prepare_ig_args(r):
                text = r["text"]
                attentions = r.get("attentions")
                metrics = r.get("attention_metrics", [])
                if not attentions or not metrics: return None
                
                model_name = r.get("model_name", "bert-base-uncased")
                is_g = "gpt2" in model_name
                tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                
                return (encoder_model, tokenizer, text, list(attentions), metrics, is_g)

            args_A = _prepare_ig_args(res_A)
            
            args_B = None
            compare = (active_bias_compare_models.get() or active_bias_compare_prompts.get())
            if compare:
                res_B = bias_results_B.get()
                if res_B:
                    args_B = _prepare_ig_args(res_B)
            
            loop = asyncio.get_running_loop()
            # Use max_workers=1 to prevent OOM/concurrency issues with heavy IG computation
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut_A = None
                if args_A:
                    fut_A = loop.run_in_executor(
                        pool,
                        batch_compute_ig_correlation,
                        *args_A
                    )
                
                fut_B = None
                if args_B:
                    fut_B = loop.run_in_executor(
                        pool,
                        batch_compute_ig_correlation,
                        *args_B
                    )
                
                if fut_A:
                    res_val_A = await fut_A
                    ig_results.set(res_val_A)
                
                if fut_B:
                    res_val_B = await fut_B
                    ig_results_B.set(res_val_B)

        except Exception as e:
            print(f"IG error: {e}")
            traceback.print_exc()
        finally:
            ig_running.set(False)

    @output
    @render.ui
    def ig_results_display():
        try:
            return _ig_results_display_impl()
        except Exception as e:
            traceback.print_exc()
            return ui.div(f"Error rendering IG results: {e}", style="color:red; padding:10px; border:1px solid red;")

    def _ig_results_display_impl():
        running = ig_running.get()
        bundle_A = ig_results.get()

        if running or not bundle_A:
            return None
            
        bundle_B = ig_results_B.get()
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        # Show comparison if comparing models OR comparing prompts
        show_comparison = (compare_models or compare_prompts) and bundle_B

        try: bar_threshold = float(input.bias_bar_threshold())
        except Exception: bar_threshold = 1.5

        # Get selected head/layer for highlighting
        selected_head = None
        selected_layer = None
        try:
            sel_l = int(input.bias_attn_layer())
            sel_h = int(input.bias_attn_head())
            selected_head = (sel_l, sel_h)
            selected_layer = sel_l
        except Exception:
            pass

        try: top_k = int(input.bias_top_k())
        except Exception: top_k = 5
            
        def _render_ig_single(bundle, container_suffix="", context_results=None):
            if not bundle: return "No data"
            
            # Unpack bundle
            if isinstance(bundle, IGAnalysisBundle):
                results = bundle.correlations
                token_attrs = bundle.token_attributions
                tokens = bundle.tokens
            else:
                results = bundle
                token_attrs = None
                tokens = None
            
            # ── Chart 1: Correlation heatmap + BAR scatter ──
            # Vertical stack in compare mode to save horizontal space
            fig1 = create_ig_correlation_chart_v2(
                results, 
                bar_threshold=bar_threshold, 
                selected_head=selected_head,
                is_vertical=show_comparison
            )
            # Adjust container height based on layout
            chart1_height = "800px" if show_comparison else "450px"
            cid1 = f"ig-chart-container{container_suffix}"
            cid2 = f"ig-token-chart-container{container_suffix}"
            cid3 = f"ig-dist-chart-container{container_suffix}"
            cid4 = f"ig-layer-chart-container{container_suffix}"

            chart1_html = _deferred_plotly(fig1, cid1, height=chart1_height)

            # ── Chart 2: Token-level IG vs Attention comparison ──
            chart2_html = ""
            attentions = None
            if context_results and context_results.get("attentions"):
                attentions = context_results["attentions"]
            elif not show_comparison:
                 res = bias_results.get()
                 attentions = res.get("attentions") if res else None

            if token_attrs is not None and tokens and attentions:
                top_bar_heads = sorted(results, key=lambda r: r.bar_original, reverse=True)[:3]
                try:
                    fig2 = create_ig_token_comparison_chart(
                        tokens, token_attrs, list(attentions), top_bar_heads,
                    )
                    chart2_html = _deferred_plotly(fig2, cid2, height="400px")
                except Exception as e:
                     chart2_html = f"<div>Error generating token chart: {e}</div>"

            # ── Chart 3: Distribution violin ──
            fig3 = create_ig_distribution_chart(results, bar_threshold=bar_threshold, selected_head=selected_head)
            chart3_html = _deferred_plotly(fig3, cid3)

            # ── Chart 4: Layer-wise mean faithfulness ──
            fig4 = create_ig_layer_summary_chart(results, selected_layer=selected_layer)
            chart4_html = _deferred_plotly(fig4, cid4)

            # ── Summary stats ──
            sig_results = [r for r in results if r.spearman_pvalue < 0.05]
            mean_rho = np.mean([r.spearman_rho for r in results])
            n_positive = sum(1 for r in sig_results if r.spearman_rho > 0)
            n_negative = sum(1 for r in sig_results if r.spearman_rho < 0)

            summary_html = (
                f'<div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#2563eb;font-family:JetBrains Mono,monospace;">{mean_rho:.3f}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Mean Spearman ρ</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#22c55e;font-family:JetBrains Mono,monospace;">{len(sig_results)}/{len(results)}</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Significant (p&lt;0.05)</div></div>'
                f'<div style="flex:1;min-width:120px;padding:12px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:8px;text-align:center;">'
                f'<div style="font-size:20px;font-weight:700;color:#f59e0b;font-family:JetBrains Mono,monospace;">{n_positive}+ / {n_negative}-</div>'
                f'<div style="font-size:10px;color:#64748b;margin-top:4px;">Positive / Negative</div></div>'
                f'</div>'
            )

            # ── Top-K table ──
            top_heads = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)[:top_k]
            table_rows = []
            for rank, r in enumerate(top_heads, 1):
                rho_color = "#2563eb" if r.spearman_rho > 0 else "#dc2626"
                sig_badge = '<span style="color:#22c55e;font-weight:600;">★</span>' if r.spearman_pvalue < 0.05 else ""
                specialized = "Yes" if r.bar_original > bar_threshold else "No"
                row_bg = "background:rgba(255,92,169,0.12);" if selected_head and (r.layer, r.head) == selected_head else ""
                table_rows.append(
                    f'<tr style="{row_bg}">'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">#{rank}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">L{r.layer}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">H{r.head}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{rho_color};">{r.spearman_rho:.3f}{sig_badge}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.spearman_pvalue:.4f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:#475569;">{r.bar_original:.3f}</td>'
                    f'<td style="padding:8px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;">{specialized}</td>'
                    f'</tr>'
                )

            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:16px;">'
                '<thead>'
                '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Layer</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Head</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Spearman ρ</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">p-value</th>'
                '<th style="padding:8px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">BAR</th>'
                '<th style="padding:8px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Specialized</th>'
                '</tr>'
                '</thead>'
                f'<tbody>{"".join(table_rows)}</tbody>'
                '</table>'
            )
            
            sections = [
                ui.HTML(chart1_html),
                ui.HTML(summary_html),
                ui.HTML(table_html),
            ]

            if chart2_html:
                sections.append(ui.HTML('<hr style="border-color:rgba(100,116,139,0.15);margin:24px 0 16px;">'))
                sections.append(ui.HTML(chart2_html))

            sections.append(ui.HTML('<hr style="border-color:rgba(100,116,139,0.15);margin:24px 0 16px;">'))
            sections.append(ui.HTML(
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                f'{chart3_html}{chart4_html}'
                f'</div>'
            ))
            
            return ui.div(*sections)

        header_args = (
            "Attention vs Integrated Gradients",
            "Captum LayerIntegratedGradients - faithfulness analysis comparing attention patterns with gradient-based token attribution.",
            "Positive ρ = attention aligns with what gradients say matters. ★ = statistically significant (p&lt;0.05)."
        )

        if show_comparison:
             # We need context results (attentions etc) for Chart 2.
             # bias_results.get() is A, bias_results_B.get() is B
             res_A = bias_results.get()
             res_B = bias_results_B.get()
             
             return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(_render_ig_single(bundle_A, "_A", res_A), *header_args, 
                           style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_ig_csv", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(_render_ig_single(bundle_B, "_B", res_B), *header_args, 
                           style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_ig_csv_B", "CSV", style=_BTN_STYLE_CSV)])
            )

        return _wrap_card(
            _render_ig_single(bundle_A),
            *header_args,
            controls=[
                ui.download_button("export_ig_csv", "CSV", style=_BTN_STYLE_CSV),
                ui.tags.button("PNG", onclick="downloadPlotlyPNG('ig-chart-container', 'ig_correlation')", style=_BTN_STYLE_PNG),
            ],
        )

    @render.download(filename="ig_correlation_results.csv")
    def export_ig_csv():
        bundle = ig_results.get()
        if not bundle:
            yield "No IG data"
            return
        results = bundle.correlations if isinstance(bundle, IGAnalysisBundle) else bundle
        lines = ["rank,layer,head,spearman_rho,spearman_pvalue,bar_original"]
        sorted_results = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)
        for i, r in enumerate(sorted_results, 1):
            lines.append(f"{i},{r.layer},{r.head},{r.spearman_rho:.6f},{r.spearman_pvalue:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    @render.download(filename="ig_correlation_results_B.csv")
    def export_ig_csv_B():
        bundle = ig_results_B.get()
        if not bundle:
            yield "No IG data"
            return
        results = bundle.correlations if isinstance(bundle, IGAnalysisBundle) else bundle
        lines = ["rank,layer,head,spearman_rho,spearman_pvalue,bar_original"]
        sorted_results = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)
        for i, r in enumerate(sorted_results, 1):
            lines.append(f"{i},{r.layer},{r.head},{r.spearman_rho:.6f},{r.spearman_pvalue:.6f},{r.bar_original:.4f}")
        yield "\n".join(lines)

    # ── StereoSet Evaluation Handlers ─────────────────────────────────
    # These load from pre-computed JSON and render independently of bias_results.
    # They react to the selected GUS-Net model to show the matching base model's data.

    def _stereoset_model_key():
        """Derive the stereoset model key from the current GUS-Net selection."""
        # Prefer analyzed model key if available (prevents reactive updates)
        res = bias_results.get()
        if res and "bias_model_key" in res:
            return res["bias_model_key"]

        try:
            return input.bias_model_key()
        except Exception:
            # Fallback default if not yet available
            return "gusnet-bert"

    def _stereoset_model_key_B():
        """Derive the stereoset model key for Model B."""
        # Prefer analyzed model key if available
        res_B = bias_results_B.get()
        if res_B and "bias_model_key" in res_B:
            return res_B["bias_model_key"]

        try:
             # If comparing models, use the B selector
             if active_bias_compare_models.get():
                return input.bias_model_key_B()
             # If comparing prompts, we are usually on same model, so B = A
             if active_bias_compare_prompts.get():
                # Use analyzed A if available, else live input
                res = bias_results.get()
                if res and "bias_model_key" in res:
                    return res["bias_model_key"]
                return input.bias_model_key()
             return None
        except Exception:
            return None

    @output
    @render.ui
    def stereoset_overview():
        """Score card with SS/LMS/ICAT gauges."""
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, manual_header_override=None):
            scores = get_stereoset_scores(mk)
            metadata = get_metadata(mk)
            if scores is None or metadata is None:
                return ui.div(
                    ui.p(
                        "StereoSet data not available. Run ",
                        ui.code("python -m attention_app.bias.generate_stereoset_json"),
                        " to generate.",
                        style="color:#94a3b8;font-size:12px;",
                    ),
                )
            html = create_stereoset_overview_html(scores, metadata)
            if manual_header_override:
                header = manual_header_override
            else:
                model_label = metadata.get("model", "unknown")
                header = ("Benchmark Scores", f"StereoSet intersentence evaluation on {model_label}")
                
            return (html, header)
            
        # Check comparison
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None
        
        if compare_models and mk_B:
            res_A, header_A = _render_single(mk_A)
            res_B, header_B = _render_single(mk_B)
            
            # If render single returned a div (error/missing), handle it
            content_A = res_A if isinstance(res_A, str) else res_A
            content_B = res_B if isinstance(res_B, str) else res_B
            
            # If one is missing data, it might return a div directly, not tuple
            # _render_single returns tuple (html, header) on success, or UI object on failure
            # Let's standardize helper to always return tuple or (UI, None)
            
            # Adjusted helper strategy:
            def _render_safe(mk):
                s = get_stereoset_scores(mk)
                m = get_metadata(mk)
                if s is None or m is None:
                    return (None, None)
                return (create_stereoset_overview_html(s, m), m.get("model", "unknown"))

            html_A, model_A = _render_safe(mk_A)
            html_B, model_B = _render_safe(mk_B)
            
            if not html_A and not html_B:
                 return _wrap_card(ui.div("No StereoSet data available."), "Benchmark Scores")

            card_A = ui.div("No data for Model A")
            if html_A:
                card_A = _wrap_card(ui.HTML(html_A), manual_header=("Benchmark Scores", f"Model A: {model_A}"),
                                    style="border: 2px solid #3b82f6; height: 100%;")
            
            card_B = ui.div("No data for Model B")
            if html_B:
                card_B = _wrap_card(ui.HTML(html_B), manual_header=("Benchmark Scores", f"Model B: {model_B}"),
                                    style="border: 2px solid #ff5ca9; height: 100%;")

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                card_A, card_B
            )

        # Single mode legacy path
        scores = get_stereoset_scores(mk_A)
        metadata = get_metadata(mk_A)
        if scores is None or metadata is None:
            return ui.div(
                ui.p(
                    "StereoSet data not available. Run ",
                    ui.code("python -m attention_app.bias.generate_stereoset_json"),
                    " to generate.",
                    style="color:#94a3b8;font-size:12px;",
                ),
            )
        html = create_stereoset_overview_html(scores, metadata)
        model_label = metadata.get("model", "unknown")
        return _wrap_card(
            ui.HTML(html),
            manual_header=("Benchmark Scores", f"StereoSet intersentence evaluation on {model_label}"),
        )

    @output
    @render.ui
    def stereoset_category_breakdown():
        """Category bar chart + bias distribution violin side-by-side."""
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, style=None, layout="row", container_suffix=""):
            scores = get_stereoset_scores(mk)
            examples = get_stereoset_examples(mk)
            if scores is None or not examples:
                return None
            by_cat = scores.get("by_category", {})
            if not by_cat: return None

            fig_cat = create_stereoset_category_chart(by_cat)
            fig_dist = create_stereoset_bias_distribution(examples)

            cat_html = _deferred_plotly(fig_cat, f"stereoset-cat-chart{container_suffix}")
            dist_html = _deferred_plotly(fig_dist, f"stereoset-dist-chart{container_suffix}")
            
            card_style = style if style else ""
            
            container_style = "display:grid;grid-template-columns:1fr 1fr;gap:16px;" if layout == "row" else "display:flex;flex-direction:column;gap:16px;"

            return ui.div(
                {"style": container_style},
                _wrap_card(ui.HTML(cat_html), style=card_style),
                _wrap_card(ui.HTML(dist_html), style=card_style),
            )

        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None

        if compare_models and mk_B:
            content_A = _render_single(mk_A, style="border: 2px solid #3b82f6;", layout="column", container_suffix="_A") or ui.div("No data for Model A")
            content_B = _render_single(mk_B, style="border: 2px solid #ff5ca9;", layout="column", container_suffix="_B") or ui.div("No data for Model B")
            
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                content_A,
                content_B
            )

        return _render_single(mk_A) or ui.div()

    @output
    @render.ui
    def stereoset_demographic_slices():
        """Demographic target analysis with category filter."""
        from ..bias.visualizations import create_stereoset_demographic_chart
        
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, container_suffix=""):
            examples = get_stereoset_examples(mk)
            if not examples: return None
            
            categories = sorted(set(e.get("category", "") for e in examples))
            charts = []
            for i, cat in enumerate(categories):
                cat_examples = [e for e in examples if e.get("category") == cat]
                fig = create_stereoset_demographic_chart(cat_examples, category=cat, min_n=10)
                chart_html = _deferred_plotly(fig, f"stereoset-demo-{cat}{container_suffix}")
                charts.append(chart_html)
            
            from collections import Counter
            target_data = {}
            for ex in examples:
                t = ex.get("target", "unknown")
                if t not in target_data:
                    target_data[t] = {"stereo_wins": 0, "n": 0, "category": ex.get("category", "")}
                target_data[t]["n"] += 1
                if ex.get("stereo_pll", 0) > ex.get("anti_pll", 0):
                    target_data[t]["stereo_wins"] += 1
            
            targets = []
            for t, d in target_data.items():
                if d["n"] >= 10:
                    ss = d["stereo_wins"] / d["n"] * 100
                    targets.append({"target": t, "ss": ss, "n": d["n"], "category": d["category"]})
            targets.sort(key=lambda x: x["ss"], reverse=True)
            
            try: top_k = int(input.bias_top_k())
            except Exception: top_k = 5
            
            most_biased = targets[:top_k]
            least_biased = targets[-top_k:][::-1]
            STEREOSET_CAT_COLORS = {"gender": "#e74c3c", "race": "#3498db", "religion": "#2ecc71", "profession": "#f39c12"}
            
            def _build_target_rows(items, direction="high"):
                rows = []
                for rank, t in enumerate(items, 1):
                    cat_color = STEREOSET_CAT_COLORS.get(t["category"], "#94a3b8")
                    ss_color = "#ef4444" if t["ss"] > 60 else "#22c55e" if t["ss"] < 40 else "#eab308"
                    rows.append(f'<tr style="transition:all 0.2s ease;"><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">{t["target"]}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;"><span style="padding:1px 6px;border-radius:3px;font-size:9px;font-weight:600;background:rgba({int(cat_color[1:3],16)},{int(cat_color[3:5],16)},{int(cat_color[5:7],16)},0.2);color:{cat_color};text-transform:uppercase;">{t["category"]}</span></td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{ss_color};">{t["ss"]:.1f}%</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-size:11px;color:#64748b;">{t["n"]}</td></tr>')
                return "".join(rows)
            
            th_style = 'padding:10px 12px;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;'
            
            summary_html = (
                '<div style="display:flex;flex-direction:column;gap:16px;margin-bottom:16px;">'
                '<div><div style="font-size:10px;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Most Stereotyped Targets (highest SS)</div>'
                '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead>'
                f'<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;"><th style="{th_style}text-align:center;">Rank</th><th style="{th_style}text-align:left;">Target</th><th style="{th_style}text-align:center;">Category</th><th style="{th_style}text-align:right;">SS</th><th style="{th_style}text-align:center;">n</th></tr></thead>'
                f'<tbody>{_build_target_rows(most_biased, "high")}</tbody></table></div>'
                '<div><div style="font-size:10px;font-weight:700;color:#22c55e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Least Stereotyped Targets (lowest SS)</div>'
                '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead>'
                f'<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;"><th style="{th_style}text-align:center;">Rank</th><th style="{th_style}text-align:left;">Target</th><th style="{th_style}text-align:center;">Category</th><th style="{th_style}text-align:right;">SS</th><th style="{th_style}text-align:center;">n</th></tr></thead>'
                f'<tbody>{_build_target_rows(least_biased, "low")}</tbody></table></div></div>'
            )
            
            chart_grid = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">' + "".join([f'<div>{html}</div>' for html in charts]) + '</div>'
            
            return ui.div(ui.HTML(summary_html), ui.HTML(chart_grid)), len(targets)

        header_args = ("Demographic Slice Analysis", "Stereotype Score breakdown by target group.")
        
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None
        
        if compare_models and mk_B:
            res_A = _render_single(mk_A, "_A")
            res_B = _render_single(mk_B, "_B")
            
            c_A = res_A[0] if res_A else ui.div("No data")
            c_B = res_B[0] if res_B else ui.div("No data")
            
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(c_A, *header_args, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(c_B, *header_args, style="border: 2px solid #ff5ca9; height: 100%;")
            )
        
        res = _render_single(mk_A)
        content = res[0] if res else ui.div()
        n_targets = res[1] if res else 0
        
        return _wrap_card(content, *(header_args[0], f"Stereotype Score breakdown by target group ({n_targets} targets with n ≥ 10)"))

    @output
    @render.ui
    def stereoset_head_sensitivity():
        """12x12 head sensitivity heatmap."""
        mk_A = _stereoset_model_key()
        
        def _render_single(mk, container_suffix=""):
            matrix = get_head_sensitivity_matrix(mk)
            top_heads = get_sensitive_heads(mk)
            if matrix is None: return None

            fig = create_stereoset_head_sensitivity_heatmap(matrix, top_heads)
            chart_html = _deferred_plotly(fig, f"stereoset-sensitivity{container_suffix}")

            try: top_k = int(input.bias_top_k())
            except Exception: top_k = 5

            top_features = get_top_features(mk)
            if top_features:
                feat_rows = []
                for rank, f in enumerate(top_features[:top_k], 1):
                    p = f["p_value"]
                    p_color = "#16a34a" if p < 1e-10 else "#eab308" if p < 0.001 else "#94a3b8"
                    feat_rows.append(f'<tr style="transition:all 0.2s ease;"><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:center;font-weight:500;color:#64748b;font-size:11px;">#{rank}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:left;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:600;color:#334155;">{f["name"]}</td><td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;color:{p_color};">{p:.2e}</td></tr>')
                features_html = (
                    '<div style="margin-top:16px;"><div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">Top Discriminative Features (Kruskal-Wallis)</div>'
                    '<table style="width:100%;border-collapse:collapse;font-size:12px;"><thead><tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;"><th style="padding:10px 12px;text-align:center;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Rank</th><th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Feature</th><th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">p-value</th></tr></thead>'
                    f'<tbody>{"".join(feat_rows)}</tbody></table></div>'
                )
            else:
                features_html = ""
                
            return ui.div(ui.HTML(chart_html), ui.HTML(features_html))

        header_args = ("Head Sensitivity Analysis", "Which attention heads respond differently across bias categories?")

        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None
        
        if compare_models and mk_B:
            c_A = _render_single(mk_A, "_A") or ui.div("No data")
            c_B = _render_single(mk_B, "_B") or ui.div("No data")
            
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(c_A, *header_args, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(c_B, *header_args, style="border: 2px solid #ff5ca9; height: 100%;")
            )

        return _wrap_card(_render_single(mk_A) or ui.div(), *header_args)

    @output
    @render.ui
    def stereoset_example_explorer():
        """Interactive example explorer with category filter and detail view."""
        mk = _stereoset_model_key()
        examples = get_stereoset_examples(mk)
        
        # Comparison setup
        # Comparison setup
        compare_models = active_bias_compare_models.get()
        mk_B = _stereoset_model_key_B() if compare_models else None
        has_B = mk_B is not None
        examples_B = get_stereoset_examples(mk_B) if has_B else []
        # Map context -> example for quick lookup
        map_B = {e["context"]: e for e in examples_B}

        if not examples:
            return ui.div()

        # Category filter (inline select)
        categories = sorted(set(e["category"] for e in examples))
        cat_options = "".join(
            f'<option value="{c}">{c.capitalize()} ({sum(1 for e in examples if e["category"]==c)})</option>'
            for c in categories
        )

        # Build scrollable example table (show first 50 by default)
        STEREOSET_CAT_COLORS = {
            "gender": "#e74c3c", "race": "#3498db",
            "religion": "#2ecc71", "profession": "#f39c12",
        }

        table_rows = []
        for i, ex in enumerate(examples[:100]):
            ctx = ex.get("context", "")[:80]
            cat = ex.get("category", "")
            bs = ex.get("bias_score", 0)
            cat_color = STEREOSET_CAT_COLORS.get(cat, "#94a3b8")
            bs_color = "#ef4444" if bs > 0 else "#22c55e"
            
            # Model B data
            bs_B_cell = ""
            if has_B:
                ex_B = map_B.get(ex.get("context"))
                if ex_B:
                    bs_B = ex_B.get("bias_score", 0)
                    bs_B_color = "#ef4444" if bs_B > 0 else "#22c55e"
                    bs_B_cell = (
                        f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);'
                        f'font-size:11px;color:{bs_B_color};font-family:JetBrains Mono,monospace;'
                        f'font-weight:600;text-align:right;border-left:1px dashed #e2e8f0;">{bs_B:+.4f}</td>'
                    )
                else:
                    bs_B_cell = '<td style="border-bottom:1px solid rgba(226,232,240,0.5);border-left:1px dashed #e2e8f0;"></td>'

            table_rows.append(
                f'<tr class="stereoset-row" data-category="{cat}" data-idx="{i}" '
                f'onclick="window._selectStereoSetRow(this, {i})" '
                f'style="cursor:pointer;transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1);border-left:3px solid transparent;" '
                f'onmouseover="if(!this.classList.contains(\'stereoset-selected\')){{this.style.background=\'rgba(255,255,255,0.08)\';this.style.transform=\'translateX(4px)\';this.style.borderLeftColor=\'{cat_color}\';}}" '
                f'onmouseout="if(!this.classList.contains(\'stereoset-selected\')){{this.style.background=\'transparent\';this.style.transform=\'translateX(0)\';this.style.borderLeftColor=\'transparent\';}}">'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;">'
                f'<span style="padding:1px 6px;border-radius:3px;font-size:9px;font-weight:600;'
                f'background:rgba({int(cat_color[1:3],16)},{int(cat_color[3:5],16)},{int(cat_color[5:7],16)},0.2);'
                f'color:{cat_color};text-transform:uppercase;">{cat}</span></td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;color:#334155;max-width:400px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{ctx}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid rgba(226,232,240,0.5);font-size:11px;color:{bs_color};'
                f'font-family:JetBrains Mono,monospace;font-weight:600;text-align:right;">{bs:+.4f}</td>'
                f'{bs_B_cell}'
                f'</tr>'
            )

        # Header columns
        header_bias_A_label = "Bias A" if has_B else "Bias"
        header_bias_B_col = ""
        
        if has_B:
            header_bias_B_col = (
                '<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;'
                'color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:90px;border-left:1px dashed #cbd5e1;">Bias B</th>'
            )

        table_html = (
            '<div style="max-height:300px;overflow-y:auto;border:1px solid rgba(255,255,255,0.06);'
            'border-radius:8px;margin-top:8px;">'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead style="position:sticky;top:0;z-index:1;">'
            '<tr style="background:linear-gradient(135deg,#f8fafc,#f1f5f9);border-bottom:2px solid #e2e8f0;">'
            '<th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:100px;">Category</th>'
            '<th style="padding:10px 12px;text-align:left;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">Context</th>'
            f'<th style="padding:10px 12px;text-align:right;font-size:10px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;width:90px;">{header_bias_A_label}</th>'
            f'{header_bias_B_col}'
            '</tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody>'
            '</table></div>'
        )

        # Category filter + row selection JS
        filter_js = (
            '<script>'
            'window.filterStereoSetCategory = function(cat) {'
            '  var rows = document.querySelectorAll(".stereoset-row");'
            '  rows.forEach(function(r) {'
            '    if (cat === "all" || r.dataset.category === cat) {'
            '      r.style.display = "";'
            '    } else {'
            '      r.style.display = "none";'
            '    }'
            '  });'
            '};'
            'window._selectStereoSetRow = function(el, idx) {'
            '  document.querySelectorAll(".stereoset-row.stereoset-selected").forEach(function(r) {'
            '    r.classList.remove("stereoset-selected");'
            '    r.style.background = "transparent";'
            '    r.querySelectorAll("td").forEach(function(td) {'
            '      if (!td.querySelector("span")) td.style.color = "";'
            '    });'
            '  });'
            '  el.classList.add("stereoset-selected");'
            '  el.style.background = "#94a3b8";'
            '  el.querySelectorAll("td").forEach(function(td) {'
            '    if (!td.querySelector("span") && !td.style.fontFamily) td.style.color = "#0f172a";'
            '  });'
            '  Shiny.setInputValue("stereoset_selected_example", idx, {priority:"event"});'
            '};'
            '</script>'
        )

        filter_select = (
            f'<select onchange="window.filterStereoSetCategory(this.value)" '
            f'style="font-size:11px;padding:4px 8px;background:#f1f5f9;color:#475569;font-weight:700;'
            f'border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;outline:none;">'
            f'<option value="all">All Categories ({len(examples)})</option>'
            f'{cat_options}'
            f'</select>'
        )

        # Detail view (initially empty, populated on click)
        detail_html = ""
        try:
            selected_idx = input.stereoset_selected_example()
            if selected_idx is not None and 0 <= selected_idx < len(examples):
                ex_A = examples[selected_idx]
                ex_B_detail = map_B.get(ex_A.get("context")) if has_B else None
                # Fetch friendly model names
                meta_A = get_metadata(mk)
                name_A = meta_A.get("model", "Model A") if meta_A else "Model A"
                
                name_B = "Model B"
                if has_B:
                    meta_B = get_metadata(mk_B)
                    if meta_B:
                        name_B = meta_B.get("model", "Model B")

                # Safely get top_n (custom input might not be init)
                top_n = 5
                try:
                    val = input.bias_top_k()
                    if val is not None: top_n = int(val)
                except:
                    pass

                # ── Attention heatmaps (generate before detail HTML) ──
                heatmap_inner_html = ""
                try:
                    # Read layer/head from the floating toolbar controls
                    sensitive = get_sensitive_heads(mk) or []
                    try:
                        sel_layer = int(input.bias_attn_layer())
                    except Exception:
                        sel_layer = sensitive[0]["layer"] if sensitive else 0
                    try:
                        sel_head = int(input.bias_attn_head())
                    except Exception:
                        sel_head = sensitive[0]["head"] if sensitive else 0

                    def _generate_model_heatmaps(model_key, example_data, suffix=""):
                        """Generate trio + diff HTML for one model.
                        
                        Returns:
                            tuple: (trio_html, diff_html) for flexible layout
                        """
                        base = _GUSNET_TO_ENCODER.get(model_key, "bert-base-uncased")
                        sens = get_sensitive_heads(model_key) or []
                        is_sens = any(
                            h["layer"] == sel_layer and h["head"] == sel_head
                            for h in sens
                        )
                        ctx = example_data.get("context", "")
                        s_text = ctx + " " + example_data.get("stereo_sentence", "")
                        a_text = ctx + " " + example_data.get("anti_sentence", "")
                        u_text = ctx + " " + example_data.get("unrelated_sentence", "")

                        s_tok, s_a = extract_attention_for_text(s_text, base, ModelManager)
                        a_tok, a_a = extract_attention_for_text(a_text, base, ModelManager)
                        sd = {"tokens": s_tok, "attentions": s_a}
                        ad = {"tokens": a_tok, "attentions": a_a}
                        ud = None
                        if u_text.strip() and u_text.strip() != ctx.strip():
                            u_tok, u_a = extract_attention_for_text(u_text, base, ModelManager)
                            ud = {"tokens": u_tok, "attentions": u_a}

                        fig_t = create_stereoset_attention_heatmaps(
                            sd, ad, ud,
                            layer=sel_layer, head=sel_head,
                            is_sensitive=is_sens,
                        )
                        t_html = _deferred_plotly(fig_t, f"stereoset-attn-heatmap-{model_key}-{suffix}")
                        fig_d = create_stereoset_attention_diff_heatmap(
                            sd, ad, layer=sel_layer, head=sel_head,
                        )
                        d_html = _deferred_plotly(fig_d, f"stereoset-attn-diff-{model_key}-{suffix}")
                        return (t_html, d_html)

                    if has_B and ex_B_detail:
                        # ── Compare Models: side-by-side with aligned layout ──
                        trio_A, diff_A = _generate_model_heatmaps(mk, ex_A, "A")
                        trio_B, diff_B = _generate_model_heatmaps(mk_B, ex_B_detail, "B")
                        
                        # Layout: 
                        # Row 1: [Model A Trio] [Model B Trio]
                        # Row 2: [Model A Diff]  [Model B Diff]
                        # This ensures proper vertical alignment
                        heatmap_inner_html = (
                            f'<div style="display:flex;flex-direction:column;gap:16px;">'
                            # Row 1: Labels + Trios side by side
                            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;">'
                            # Model A column
                            f'  <div style="display:flex;flex-direction:column;gap:12px;">'
                            f'    <div style="font-size:10px;font-weight:700;color:#3b82f6;'
                            f'         text-transform:uppercase;letter-spacing:0.5px;'
                            f'         padding:4px 8px;background:rgba(59,130,246,0.1);border-radius:4px;'
                            f'         border:1px solid rgba(59,130,246,0.2);display:inline-block;width:fit-content;">{name_A}</div>'
                            f'    <div>{trio_A}</div>'
                            f'  </div>'
                            # Model B column
                            f'  <div style="display:flex;flex-direction:column;gap:12px;">'
                            f'    <div style="font-size:10px;font-weight:700;color:#ff5ca9;'
                            f'         text-transform:uppercase;letter-spacing:0.5px;'
                            f'         padding:4px 8px;background:rgba(255,92,169,0.1);border-radius:4px;'
                            f'         border:1px solid rgba(255,92,169,0.2);display:inline-block;width:fit-content;">{name_B}</div>'
                            f'    <div>{trio_B}</div>'
                            f'  </div>'
                            f'</div>'
                            # Row 2: Attention Differences side by side
                            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;">'
                            f'  <div>{diff_A}</div>'
                            f'  <div>{diff_B}</div>'
                            f'</div>'
                            f'</div>'
                        )
                    else:
                        # ── Single model ──
                        trio, diff = _generate_model_heatmaps(mk, ex_A, "single")
                        heatmap_inner_html = (
                            f'<div style="display:flex;flex-direction:column;gap:12px;">'
                            f'{trio}'
                            f'{diff}'
                            f'</div>'
                        )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    heatmap_inner_html = (
                        f'<div style="padding:12px;border:1px solid rgba(239,68,68,0.3);'
                        f'border-radius:8px;color:#f87171;font-size:11px;">'
                        f'Could not generate attention heatmaps: {e}</div>'
                    )

                # Build sensitive heads badge label
                sensitive_label = ""
                _sens = get_sensitive_heads(mk) or []
                if _sens:
                    badges_A = [f'L{h["layer"]}.H{h["head"]}' for h in _sens[:5]]
                    if has_B:
                        _sens_B = get_sensitive_heads(mk_B) or []
                        badges_B = [f'L{h["layer"]}.H{h["head"]}' for h in _sens_B[:5]]
                        sensitive_label = (
                            f'<div style="font-size:9px;margin-bottom:8px;display:flex;gap:16px;">'
                            f'<span style="color:#3b82f6;">★ {name_A}: {", ".join(badges_A)}</span>'
                            f'<span style="color:#ff5ca9;">★ {name_B}: {", ".join(badges_B)}</span>'
                            f'</div>'
                        )
                    else:
                        sensitive_label = (
                            f'<div style="font-size:9px;color:#f59e0b;margin-bottom:8px;">'
                            f'* Top sensitive: {", ".join(badges_A)}</div>'
                        )

                detail_html = create_stereoset_example_html(
                    ex_A,
                    example_B=ex_B_detail,
                    model_A_name=name_A,
                    model_B_name=name_B,
                    sensitive_heads=get_sensitive_heads(mk),
                    head_profile_stats=get_head_profile_stats(mk),
                    sensitive_heads_B=get_sensitive_heads(mk_B) if has_B else None,
                    head_profile_stats_B=get_head_profile_stats(mk_B) if has_B else None,
                    top_n=top_n,
                    heatmap_html=heatmap_inner_html,
                    sensitive_heads_label=sensitive_label,
                )

        except Exception as e:
            # Ignore SilentException (input not ready)
            if "SilentException" in str(type(e)):
                pass
            else:
                import traceback
                error_trace = traceback.format_exc()
                detail_html = (
                    f"<div style='color:#ef4444;padding:16px;border:1px solid rgba(239,68,68,0.2);background:rgba(239,68,68,0.05);border-radius:8px;'>"
                    f"<div style='font-weight:700;margin-bottom:8px;'>Error loading example details</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;white-space:pre-wrap;'>{e}</div>"
                    f"<details style='margin-top:8px;'><summary style='cursor:pointer;opacity:0.6;font-size:10px;'>Show Traceback</summary>"
                    f"<pre style='font-size:9px;opacity:0.8;margin-top:8px;'>{error_trace}</pre></details>"
                    f"</div>"
                )

        detail_section = (
            f'<div style="margin-top:16px;">{detail_html}</div>'
            if detail_html else
            '<div style="margin-top:16px;color:#64748b;font-size:11px;font-style:italic;'
            'text-align:center;padding:20px;">Click an example above to see details</div>'
        )

        return _wrap_card(
            ui.div(
                ui.HTML(filter_js),
                ui.div(
                    {"style": "display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;"},
                    ui.HTML(f'<span style="font-size:11px;color:#94a3b8;">'
                            f'Showing {min(100, len(examples))} of {len(examples)} examples</span>'),
                    ui.HTML(filter_select),
                ),
                ui.HTML(table_html),
                ui.HTML(detail_section),
            ),
            manual_header=(
                "Example Explorer",
                "Browse StereoSet examples - click to inspect with attention heatmaps",
            ),
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_label(data, is_A, compare_m):
        if compare_m: return _get_bias_model_label(data)
        return "Prompt A" if is_A else "Prompt B"

__all__ = ["bias_server_handlers"]
