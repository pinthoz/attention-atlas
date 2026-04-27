import json
import html as _html
import logging
from datetime import datetime
from pathlib import Path
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

_logger = logging.getLogger(__name__)


from .csv_utils import csv_safe as _csv_safe
from .bias_helpers import (
    _GUSNET_TO_ENCODER, _GUSNET_TO_BASE, _GUSNET_DISPLAY_NAMES,
    _clean_gusnet_label,
    _load_gusnet_attention_for_text, _load_base_encoder_attention_for_text,
    _deferred_plotly, _wrap_card, _chart_with_png_btn,
    _align_gusnet_to_attention_tokens,
    _get_bias_model_label, _process_raw_bias_result, _build_batch_report,
)
from .bias_styles import (
    BTN_STYLE_CSV as _BTN_STYLE_CSV, BTN_STYLE_PNG as _BTN_STYLE_PNG,
    TH as _TH, TR as _TR, TD as _TD, TC as _TC, TS as _TS,
    TBG as _TBG, TBR as _TBR, TBA as _TBA, TBB as _TBB, TBP as _TBP,
    TN as _TN,
)
from .bias_xai import register_xai_handlers
from .bias_stereoset import register_stereoset_handlers
from .bias_exports import (
    csv_summary as _csv_summary_fn, csv_spans as _csv_spans_fn,
    csv_strip as _csv_strip_fn, csv_confidence as _csv_confidence_fn,
    csv_combined as _csv_combined_fn, csv_matrix as _csv_matrix_fn,
    csv_propagation as _csv_propagation_fn, csv_top_heads as _csv_top_heads_fn,
    csv_ablation as _csv_ablation_fn,
    csv_ig_correlation as _csv_ig_correlation_fn,
    csv_topk_overlap as _csv_topk_overlap_fn,
    csv_perturbation as _csv_perturbation_fn,
    csv_lrp as _csv_lrp_fn,
    csv_stereoset_features as _csv_stereoset_features_fn,
    csv_stereoset_sensitivity as _csv_stereoset_sensitivity_fn,
    csv_stereoset_category as _csv_stereoset_category_fn,
    csv_stereoset_distribution as _csv_stereoset_distribution_fn,
    csv_stereoset_demographic as _csv_stereoset_demographic_fn,
    csv_perturb_attn as _csv_perturb_attn_fn,
    csv_cross_method as _csv_cross_method_fn,
)

import numpy as np
import torch
from shiny import ui, render, reactive

from ..models import ModelManager
from ..utils import get_reproducibility_info
from ..bias import (
    GusNetDetector, EnsembleGusNetDetector,
    AttentionBiasAnalyzer,
    create_attention_bias_matrix,
    create_bias_propagation_plot,
    create_bias_propagation_heads_plot,
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
    create_topk_overlap_chart,
    create_perturbation_comparison_chart,
    create_perturbation_attn_heatmap,
    create_lrp_comparison_chart,
    create_cross_method_agreement_chart,
)
from ..bias.counterfactual import find_swappable_terms, generate_counterfactual, get_swap_for_token
from ..bias.head_ablation import batch_ablate_top_heads, HeadAblationResult
from ..bias.integrated_gradients import (
    batch_compute_ig_correlation, IGCorrelationResult, IGAnalysisBundle,
    TopKOverlapResult,
    batch_compute_perturbation, PerturbationAnalysisBundle,
    batch_compute_lrp, LRPAnalysisBundle,
)
from ..bias.stereoset import (
    load_stereoset_data,
    get_stereoset_scores,
    get_stereoset_examples,
    get_head_sensitivity_matrix,
    get_sensitive_heads,
    get_top_features,
    get_head_profile_stats,
    get_metadata,
    get_gusnet_key,
    compute_model_similarity,
)
from ..bias.visualizations import (
    create_stereoset_overview_html,
    create_stereoset_category_chart,
    create_stereoset_head_sensitivity_heatmap,
    create_stereoset_bias_distribution,
    create_stereoset_example_html,
    create_stereoset_attention_heatmaps,
    create_stereoset_attention_diff_heatmap,
    create_stereoset_head_distributions,
    create_stereoset_attention_scatter,
)
from ..bias.visualizations import compute_cf_attention_consistency, create_cf_consistency_html
from ..bias.feature_extraction import extract_attention_for_text
from ..ui.bias_ui import create_bias_accordion, create_floating_bias_toolbar
from ..ui.components import viz_header

# Pure helpers, constants, and CSV body functions are now in separate
# modules — imported at the top of this file from bias_helpers,
# bias_styles, and bias_exports.
#

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
    ig_results_base = reactive.value(None)   # IG computed on base encoder
    ig_running = reactive.value(False)
    perturbation_results = reactive.value(None)
    perturbation_results_B = reactive.value(None)
    perturbation_running = reactive.value(False)
    lrp_results = reactive.value(None)
    lrp_results_B = reactive.value(None)
    lrp_running = reactive.value(False)

    # ── Propagation drilldown (layer click → per-head view) ──
    propagation_drilldown_layer = reactive.value(None)   # int or None

    # ── GUS-Net attention cache (computed on demand, keyed by text+model) ──
    _base_attn_cache = reactive.value(None)       # processed result dict
    _base_attn_cache_B = reactive.value(None)

    # ── Batch mode state ──
    batch_sentences = reactive.value([])
    batch_file_name = reactive.value("")

    # Snapshots of compare mode state - Updated ONLY on 'Analyze Bias'
    active_bias_compare_models = reactive.Value(False)
    active_bias_compare_prompts = reactive.Value(False)
    
    # Sequential Logic State
    bias_prompt_step = reactive.Value("A")

    # Cached texts for display
    bias_cached_text_A = reactive.value("")
    bias_cached_text_B = reactive.value("")

    # Counterfactual state - stores raw find_swappable_terms() result
    counterfactual_swaps = reactive.value(None)
    cf_applied_swaps = reactive.value(None)  # swaps applied in last CF compare (for consistency card)

    # Current thresholds (synced with UI sliders) - will be updated by update_threshold_defaults
    current_thresholds_A = reactive.value({"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5})
    current_thresholds_B = reactive.value({"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5})
    
    # Analysis generation counter - incremented each time analysis completes
    bias_analysis_generation = reactive.value(0)
    bias_last_processed_generation = reactive.value(-1)

    # Snapshot of previous run state - restored when Back is clicked
    bias_snapshot = reactive.value(None)

    # Style constants (_BTN_STYLE_*, _TH, _TR, etc.) are imported from
    # bias_styles at module level.


    # ── Selection reader helper ─────────────────────────────────────────

    def _read_sel(reader):
        """Read a Shiny reactive input and convert to ``list[int]``.

        Returns ``None`` when the input has not been set yet (first render)
        or is empty.  The broad ``except`` is intentional: Shiny raises
        ``SilentException`` before the first click, and we must not let
        that propagate or the renderer fails silently.
        """
        try:
            sel = reader()
            if not sel:
                return None
            return (
                [int(sel)] if isinstance(sel, (int, str))
                else [int(x) for x in sel if x is not None]
            )
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            return None

    def _map_sel_a_to_b(sel_a, res_a, res_b):
        """Map selected token indices from model A to model B by content.

        In compare-models mode BERT and GPT-2 tokenize the same text
        differently, so the same word lives at different indices.  This
        helper matches by cleaned token text (case-insensitive) and
        returns a list of B indices, or ``None`` if *sel_a* is empty.
        """
        if not sel_a or not res_a or not res_b:
            return None
        toks_a = [t.replace("##", "").replace("\u0120", "").lower()
                  for t in res_a.get("tokens", [])]
        toks_b = [t.replace("##", "").replace("\u0120", "").lower()
                  for t in res_b.get("tokens", [])]
        mapped = []
        used = set()
        for idx_a in sel_a:
            if idx_a >= len(toks_a):
                continue
            word = toks_a[idx_a]
            if not word:
                continue
            for j, tb in enumerate(toks_b):
                if tb == word and j not in used:
                    mapped.append(j)
                    used.add(j)
                    break
        return mapped if mapped else None

    # ── Attention Source helpers ──────────────────────────────────────────

    def _get_attn_source_mode(input_id: str = "bias_attn_source") -> str:
        """Read the current attention source mode from a hidden input.

        Default is ``"gusnet"`` because ``heavy_bias_compute`` loads
        ``_GUSNET_TO_ENCODER`` (the fine-tuned model) as the encoder.
        """
        try:
            val = getattr(input, input_id)()
            if val in ("base", "gusnet", "compare"):
                return val
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            pass
        return "gusnet"

    def _normalize_attn_source_mode(mode: str) -> str:
        """Passthrough - _get_attn_source_mode already returns a valid value."""
        return mode if mode in ("gusnet", "base", "compare") else "gusnet"

    def _get_base_resolved(res, cache_rv):
        """Return a processed base-encoder-attention result.

        ``heavy_bias_compute`` already loads the GUS-Net model as the encoder
        (via ``_GUSNET_TO_ENCODER``), so ``res`` contains **GUS-Net** attentions.
        This helper loads the original pretrained base model (e.g.
        ``bert-base-uncased``) on demand and caches the result.
        """
        if not res:
            return None
        cached = cache_rv.get()
        cache_key = (res.get("text"), res.get("bias_model_key"))
        if cached and (cached.get("text"), cached.get("bias_model_key")) == cache_key:
            return cached
        try:
            raw_base = _load_base_encoder_attention_for_text(res["text"], res["bias_model_key"])
            thresholds = raw_base.get("effective_thresholds", {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5})
            processed = _process_raw_bias_result(raw_base, thresholds, use_optimized=False)
            if processed:
                cache_rv.set(processed)
            return processed
        except Exception:
            _logger.exception("[attn-source] Base encoder attention load failed")
            return None

    def _resolve_source_for_render(res, base_cache, mode):
        """Pick the right result dict for the active attention source mode.

        ``res`` from ``heavy_bias_compute`` is always **GUS-Net** attention
        (because ``_GUSNET_TO_ENCODER`` maps to the fine-tuned model).

        Returns ``(data_for_render, source_label)``.
        """
        if mode == "base":
            base = _get_base_resolved(res, base_cache)
            if base:
                return base, "Base Encoder"
            return res, "GUS-Net"  # fallback if base load fails
        # mode == "gusnet" or default → use res as-is (already GUS-Net)
        return res, "GUS-Net"

    def _resolve_faithfulness_results():
        """Resolve which results faithfulness analyses should run against."""
        res_A = bias_results.get()
        if not res_A:
            return None, None, False

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        if compare_models or compare_prompts:
            res_B = bias_results_B.get()
            return res_A, res_B, bool(res_B)

        source_mode = _get_attn_source_mode("bias_attn_source")
        if source_mode == "compare":
            res_base = _get_base_resolved(res_A, _base_attn_cache)
            if res_base:
                return res_base, res_A, True

        res_render, _ = _resolve_source_for_render(res_A, _base_attn_cache, source_mode)
        return res_render, None, False

    def _source_badge_html(label: str) -> str:
        """Small inline badge indicating the attention source."""
        if "gus" in label.lower():
            color = "#ff5ca9"
            bg = "rgba(255,92,169,0.12)"
        else:
            color = "#60a5fa"
            bg = "rgba(96,165,250,0.12)"
        return (
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;'
            f'letter-spacing:0.4px;text-transform:uppercase;'
            f'color:{color};background:{bg};border:1px solid {color}20;'
            f'margin-left:8px;vertical-align:middle;">'
            f'{label}</span>'
        )


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
            _logger.debug("Suppressed exception", exc_info=True)
            pass

        from attention_app.bias.gusnet_detector import MODEL_REGISTRY

        def _get_model_defaults(vk):
            if not vk:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

            cfg = MODEL_REGISTRY.get(vk)
            if not cfg:
                return {"GEN": 0.5, "UNFAIR": 0.5, "STEREO": 0.5}

            opt = cfg.get("optimized_thresholds")
            # Auto-load from .npy if available (same as GusNetDetector)
            npy_path = Path(cfg["path"]) / "optimized_thresholds.npy"
            if npy_path.exists():
                opt = np.load(str(npy_path), allow_pickle=False).tolist()
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
        defaults_A = _get_model_defaults(vk_A)

        # Update server state immediately (so UI always shows latest thresholds)
        current_thresholds_A.set(defaults_A)

        # Get thresholds for B if any comparison mode is active
        msg_defaults = defaults_A.copy()
        if compare_active:
            defaults_B = _get_model_defaults(vk_B)
            current_thresholds_B.set(defaults_B)
            # Prefix with _B for the UI handler
            for cat, val in defaults_B.items():
                msg_defaults[f"{cat}_B"] = val

        # Send to UI
        if msg_defaults:
            asyncio.create_task(session.send_custom_message("set_bias_thresholds", msg_defaults))

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
            safe_text = _html.escape(text.replace('\n', ' '), quote=True)
            items.append(
                ui.div(
                    _html.escape(display_text),
                    class_="history-item",
                    onclick=f"selectBiasHistoryItem('{safe_text}')"
                )
            )
        return ui.div(*items)

    @reactive.Effect
    def update_bias_results_live():
        """Update bias results when threshold sliders change (without re-running inference).

        Sliders are the ONLY reactive triggers. current_thresholds_A/B are read via
        isolate() so that .set() calls inside this function do NOT cause re-triggers.
        """
        # ── Read slider inputs (these ARE reactive triggers) ──
        try:
            t_unfair = float(input.bias_thresh_unfair())
        except (Exception, ValueError, TypeError):
            t_unfair = 0.5
        try:
            t_gen = float(input.bias_thresh_gen())
        except (Exception, ValueError, TypeError):
            t_gen = 0.5
        try:
            t_stereo = float(input.bias_thresh_stereo())
        except (Exception, ValueError, TypeError):
            t_stereo = 0.5

        try:
            t_unfair_b = float(input.bias_thresh_unfair_b())
        except (Exception, ValueError, TypeError):
            t_unfair_b = 0.5
        try:
            t_gen_b = float(input.bias_thresh_gen_b())
        except (Exception, ValueError, TypeError):
            t_gen_b = 0.5
        try:
            t_stereo_b = float(input.bias_thresh_stereo_b())
        except (Exception, ValueError, TypeError):
            t_stereo_b = 0.5

        manual_thresholds_A = {"UNFAIR": t_unfair, "GEN": t_gen, "STEREO": t_stereo}
        manual_thresholds_B = {"UNFAIR": t_unfair_b, "GEN": t_gen_b, "STEREO": t_stereo_b}

        # ── Read everything else via isolate() to avoid extra re-triggers ──
        with reactive.isolate():
            # GUARD: Don't run if analysis is in progress
            if bias_running.get():
                return

            # GUARD: Skip the first call after a fresh analysis (stale slider values)
            current_gen = bias_analysis_generation.get()
            last_gen = bias_last_processed_generation.get()
            if current_gen > last_gen:
                bias_last_processed_generation.set(current_gen)
                return

            raw_A = bias_raw_results.get()
            raw_B = bias_raw_results_B.get()

            if not raw_A and not raw_B:
                return

            # Check A model-key consistency
            a_model_matches = True
            try:
                live_key_A = input.bias_model_key()
                if raw_A and live_key_A and raw_A.get("bias_model_key") != live_key_A:
                    a_model_matches = False
            except Exception:
                _logger.debug("Suppressed exception", exc_info=True)
                pass

            try:
                use_opt = input.bias_use_optimized()
            except Exception:
                use_opt = False

            # Compare with last-processed thresholds (no reactive dependency)
            prev_A = current_thresholds_A.get()
            prev_B = current_thresholds_B.get()
            changed_A = prev_A != manual_thresholds_A
            changed_B = prev_B != manual_thresholds_B

            if not changed_A and not changed_B:
                return

            # Persist new values (isolate prevents re-trigger)
            if changed_A:
                current_thresholds_A.set(manual_thresholds_A)
            if changed_B:
                current_thresholds_B.set(manual_thresholds_B)

            # Determine if compare mode is active
            try:
                live_compare_models = bool(input.bias_compare_mode())
            except Exception:
                live_compare_models = bool(active_bias_compare_models.get())
            try:
                live_compare_prompts = bool(input.bias_compare_prompts_mode())
            except Exception:
                live_compare_prompts = bool(active_bias_compare_prompts.get())

            update_B = bool(raw_B) and changed_B and (live_compare_models or live_compare_prompts)

            # Process A
            if raw_A and changed_A and a_model_matches:
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
                safe = _html.escape(item.replace('\n', ' '), quote=True)
                display_safe = _html.escape(display)
                html_content += (
                    f'<div class="history-item" '
                    f"onclick=\"selectBiasHistoryItem('{safe}')\">"
                    f'{display_safe}</div>'
                )

        js_code = f"""
            var dropdown = document.getElementById('bias-history-dropdown');
            if (dropdown) {{
                dropdown.innerHTML = {json.dumps(html_content)};
            }}
        """
        ui.insert_ui(
            selector="body", where="beforeEnd",
            ui=ui.tags.script(js_code),
        )

    @reactive.Effect
    async def sync_bias_history_storage():
        """Persist bias history to localStorage (mirrors attention tab)."""
        await session.send_custom_message("update_bias_history", bias_history())

    @reactive.Effect
    @reactive.event(input.restored_bias_history)
    def restore_bias_history():
        """Restore bias history from localStorage on page load."""
        if input.restored_bias_history():
            raw = input.restored_bias_history()
            unique = []
            for item in raw:
                clean_item = item.strip()
                if clean_item and clean_item not in unique:
                    unique.append(clean_item)
            bias_history.set(unique)

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
            await session.send_custom_message("bias_switch_prompt_tab", {"tab": "A"})
        
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
                filepath = Path(folder) / Path(filename).name
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding='utf-8')
            except Exception as e:
                logging.getLogger(__name__).warning("save_bias_export_to_folder failed: %s", e)

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
            _logger.debug("Suppressed exception", exc_info=True)
            return 0.5

    # ── Session Logic ──

    @render.download(filename=lambda: f"bias_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def save_bias_session():
        data = {
            "type": "bias_analysis",
            "timestamp": datetime.now().isoformat(),
            "text": input.bias_input_text(),
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

        # Per-class thresholds A
        for key in ["bias_thresh_unfair", "bias_thresh_gen", "bias_thresh_stereo"]:
            try:
                data[key] = float(getattr(input, key)())
            except Exception:
                _logger.debug("Suppressed exception", exc_info=True)
                pass
        # Per-class thresholds B
        for key in ["bias_thresh_unfair_b", "bias_thresh_gen_b", "bias_thresh_stereo_b"]:
            try:
                data[key] = float(getattr(input, key)())
            except Exception:
                _logger.debug("Suppressed exception", exc_info=True)
                pass

        # Layer / Head / Top-K
        try:
            data["bias_attn_layer"] = input.bias_attn_layer()
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            pass
        try:
            data["bias_attn_head"] = input.bias_attn_head()
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            pass
        try:
            data["bias_top_k"] = int(input.bias_top_k())
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            pass

        # Reproducibility metadata
        try:
            model_key = data.get("bias_model_key", "gusnet-bert")
            from ..bias.gusnet_detector import MODEL_REGISTRY
            hf_path = MODEL_REGISTRY.get(model_key, {}).get("path")
            data["reproducibility"] = get_reproducibility_info(
                data.get("text", ""), model_name=hf_path
            )
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
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
                await session.send_custom_message("bias_toggle_switch",
                    {"id": "bias_compare_mode", "checked": True})
            if compare_prompts:
                await session.send_custom_message("bias_toggle_switch",
                    {"id": "bias_compare_prompts_mode", "checked": True})

            # 2. Wait for UI to rebuild, then restore controls via JS
            await asyncio.sleep(0.3)

            restore_data = {}
            if "bias_model_key" in data:
                restore_data["bias_model_key"] = data["bias_model_key"]
            if "bias_model_key_B" in data:
                restore_data["bias_model_key_B"] = data["bias_model_key_B"]

            if restore_data:
                await session.send_custom_message("restore_bias_session_controls", restore_data)

            # 3. Restore text inputs
            if "text" in data:
                await session.send_custom_message("bias_set_textarea",
                    {"id": "bias_input_text", "value": data["text"]})
            if "bias_input_text_B" in data:
                await session.send_custom_message("bias_set_textarea",
                    {"id": "bias_input_text_B", "value": data["bias_input_text_B"]})

            # 4. Restore per-class thresholds via JS slider updates
            thresh_map = {
                "bias_thresh_unfair":   "bias-thresh-unfair",
                "bias_thresh_gen":      "bias-thresh-gen",
                "bias_thresh_stereo":   "bias-thresh-stereo",
                "bias_thresh_unfair_b": "bias-thresh-unfair-b",
                "bias_thresh_gen_b":    "bias-thresh-gen-b",
                "bias_thresh_stereo_b": "bias-thresh-stereo-b",
            }
            for input_key, slider_id in thresh_map.items():
                if input_key in data:
                    val = float(data[input_key])
                    await session.send_custom_message("bias_set_slider",
                        {"id": slider_id, "value": val, "inputId": input_key})

            # 5. Restore layer/head selections (after a small delay for selects to populate)
            await asyncio.sleep(0.2)
            if "bias_attn_layer" in data:
                ui.update_select("bias_attn_layer", selected=str(data["bias_attn_layer"]))
            if "bias_attn_head" in data:
                ui.update_select("bias_attn_head", selected=str(data["bias_attn_head"]))
            if "bias_top_k" in data:
                top_k = int(data["bias_top_k"])
                await session.send_custom_message("bias_set_slider",
                    {"id": "bias-topk-slider", "value": top_k, "inputId": "bias_top_k"})

            ui.notification_show("Bias session loaded successfully.", type="message", duration=3)

        except Exception as e:
            _logger.exception("Error loading bias session")
            ui.notification_show("Error loading session. Please check the file format.", type="error")

    # ── CSV Export Handlers (8 pairs: A and B) ──
    # Body helpers are imported from bias_exports; A/B stubs pick the reactive source.

    def _csv_combined(res):
        """Thin wrapper: reads reactive input for layer/head, delegates to pure fn."""
        try:
            l_idx = int(input.bias_attn_layer())
            h_idx = int(input.bias_attn_head())
        except Exception:
            l_idx, h_idx = 0, 0
        return _csv_combined_fn(res, l_idx, h_idx)

    def _bias_export_ab(section, ext, body_fn, name):
        """Register A + B download handlers that share *body_fn(res) -> str*."""
        def _a():
            res = bias_results.get()
            if not res:
                yield "No data available"; return
            yield body_fn(res)
        _a.__name__ = name
        auto_save_bias_download(section, ext)(_a)

        def _b():
            res = bias_results_B.get()
            if not res:
                yield "No data available"; return
            yield body_fn(res)
        _b.__name__ = f"{name}_B"
        auto_save_bias_download(section, ext, is_b=True)(_b)

    _bias_export_ab("summary",     "csv", _csv_summary_fn,     "export_bias_summary")
    _bias_export_ab("spans",       "csv", _csv_spans_fn,       "export_bias_spans")
    _bias_export_ab("strip",       "csv", _csv_strip_fn,       "export_bias_strip")
    _bias_export_ab("confidence",  "csv", _csv_confidence_fn,   "export_bias_confidence")
    _bias_export_ab("combined",    "csv", _csv_combined,        "export_bias_combined")
    _bias_export_ab("matrix",      "csv", _csv_matrix_fn,       "export_bias_matrix")
    _bias_export_ab("propagation", "csv", _csv_propagation_fn,  "export_bias_propagation")
    _bias_export_ab("top_heads",   "csv", _csv_top_heads_fn,    "export_bias_top_heads")

    def log_debug(msg):
        _logger.debug(msg)
            
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
        if not text:
            log_debug("No text provided")
            return None

        try:
            # Clear cache before starting heavy operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load attention model
            log_debug(f"Loading attention model {model_name}...")
            tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
            device = ModelManager.get_device()
            log_debug(f"Model loaded. Device: {device}")

            log_debug("Tokenizing text...")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            log_debug("Running encoder inference...")
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

            # ── Token-level bias detection (GUS-Net) ──
            log_debug(f"Initializing GusNetDetector (model_key={bias_model_key})...")
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
            log_debug(f"ERROR in heavy_bias_compute: {e}")
            _logger.exception("ERROR in heavy_bias_compute")
            return None

    # ── Trigger analysis ──

    @reactive.effect
    @reactive.event(input.analyze_bias_btn)
    async def compute_bias():
        log_debug("BUTTON CLICKED: compute_bias triggered")
        try:
            # ── Batch mode intercept ──
            try:
                is_batch = input.bias_batch_mode_active() == "true"
            except Exception:
                is_batch = False

            if is_batch:
                if not batch_sentences.get():
                    ui.notification_show("No file uploaded for batch mode.", type="warning", duration=3)
                    return
                bias_running.set(True)
                await session.send_custom_message('start_bias_loading', {})
                await asyncio.sleep(0.1)
                try:
                    await _run_batch_analysis()
                finally:
                    bias_running.set(False)
                    await session.send_custom_message('stop_bias_loading', {})
                return

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
                await session.send_custom_message("bias_switch_prompt_tab", {"tab": "B"})
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

            # Snapshot BEFORE any clearing (so had_prior_results is accurate)
            had_prior_results = bias_results.get() is not None
            if had_prior_results:
                bias_snapshot.set({
                    'results':         bias_results.get(),
                    'results_B':       bias_results_B.get(),
                    'text_A':          bias_cached_text_A.get(),
                    'text_B':          bias_cached_text_B.get(),
                    'compare_models':  prev_compare_models,
                    'compare_prompts': prev_compare_prompts,
                })

            # Detect mode transitions
            mode_switched_to_prompts = compare_prompts and not prev_compare_prompts
            mode_switched_to_models = compare_models and not prev_compare_models
            mode_switched_off = (not compare_prompts and not compare_models) and (prev_compare_prompts or prev_compare_models)

            # Reset state when switching modes to ensure clean state
            if mode_switched_to_prompts or mode_switched_to_models or mode_switched_off:
                log_debug("Mode switch detected - resetting bias state")
                # Preserve cf_applied_swaps — trigger_counterfactual sets it
                # right before this mode switch happens
                _saved_cf_swaps = cf_applied_swaps.get()
                bias_raw_results.set(None)
                bias_raw_results_B.set(None)
                bias_results.set(None)
                bias_results_B.set(None)
                bias_cached_text_A.set("")
                bias_cached_text_B.set("")
                cf_applied_swaps.set(_saved_cf_swaps)
                # Reset prompt step to A when switching modes
                bias_prompt_step.set("A")
                # Ensure UI reflects the reset
                await session.send_custom_message("bias_switch_prompt_tab", {"tab": "A"})

            active_bias_compare_models.set(compare_models)
            active_bias_compare_prompts.set(compare_prompts)

            # Clear previous results
            # cf_applied_swaps is preserved here — trigger_counterfactual sets it
            # right before triggering this handler. Clear it only for non-CF runs.
            if not compare_prompts:
                cf_applied_swaps.set(None)
            bias_results.set(None)
            bias_results_B.set(None)
            counterfactual_swaps.set(None)
            _base_attn_cache.set(None)
            _base_attn_cache_B.set(None)
            ig_results_base.set(None)
            bias_cached_text_A.set(text)

            try:
                use_optimized = bool(input.bias_use_optimized())
            except Exception:
                use_optimized = True

            # Use current thresholds from reactive values (synced with UI)
            thresholds_A = current_thresholds_A.get()
            
            # Fetch B thresholds if needed
            if compare_models:
                thresholds_B = current_thresholds_B.get()
            elif compare_prompts:
                thresholds_B = thresholds_A
            else:
                thresholds_B = {"UNFAIR": 0.5, "GEN": 0.5, "STEREO": 0.5}
            
            # Mean placeholder for legacy compatibility
            threshold = sum(thresholds_A.values()) / 3
            log_debug(f"Thresholds A: {thresholds_A}, B: {thresholds_B}")
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
                    
                    # Initial processing for Model A (using thresholds we just grabbed)
                    effective_A = result.get("effective_thresholds", thresholds_A)
                    res_processed = _process_raw_bias_result(result, effective_A, use_optimized=use_optimized)
                    
                    # Update current_thresholds to match what was actually used
                    current_thresholds_A.set(effective_A)
                    
                    # Store RAW result so sliders can re-threshold live
                    bias_raw_results.set(result)
                    
                    # Set processed results (this triggers update_bias_results_live, but generation counter protects it)
                    bias_results.set(res_processed)

                    # Prepare message for UI sliders (A thresholds always included)
                    msg_thresholds = {}
                    if result and result.get("effective_thresholds"):
                        eff_A = result["effective_thresholds"]
                        msg_thresholds.update({
                            "UNFAIR": eff_A.get("UNFAIR"),
                            "GEN": eff_A.get("GEN"),
                            "STEREO": eff_A.get("STEREO")
                        })

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
                        current_thresholds_B.set(effective_B)
                        res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                        bias_results_B.set(res_B_proc)
                        
                        # Add thresholds B to the message
                        if result_B and result_B.get("effective_thresholds"):
                            eff_B = result_B["effective_thresholds"]
                            msg_thresholds.update({
                                "UNFAIR_B": eff_B.get("UNFAIR"),
                                "GEN_B": eff_B.get("GEN"),
                                "STEREO_B": eff_B.get("STEREO")
                            })

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
                            current_thresholds_B.set(effective_B)
                            res_B_proc = _process_raw_bias_result(result_B, effective_B, use_optimized=use_optimized)
                            bias_results_B.set(res_B_proc)
                            
                            # Add thresholds B to the message
                            if result_B and result_B.get("effective_thresholds"):
                                eff_B = result_B["effective_thresholds"]
                                msg_thresholds.update({
                                    "UNFAIR_B": eff_B.get("UNFAIR"),
                                    "GEN_B": eff_B.get("GEN"),
                                    "STEREO_B": eff_B.get("STEREO")
                                })
                        else:
                            # Empty text B - clear results and show notification
                            bias_raw_results_B.set(None)
                            bias_cached_text_B.set("")
                            bias_results_B.set(None)
                            ui.notification_show("Prompt B is empty. Only analyzing Prompt A.", type="warning", duration=3)
                    else:
                        # Not in compare_prompts mode, clear B
                        bias_raw_results_B.set(None)
                        bias_cached_text_B.set("")

                        # Counterfactual detection (single-prompt mode only)
                        try:
                            cf_swaps = find_swappable_terms(text)
                            counterfactual_swaps.set(cf_swaps if cf_swaps else None)
                        except Exception:
                            counterfactual_swaps.set(None)

                    # Send all thresholds at once (A always included, B only if compare mode)
                    if msg_thresholds:
                        log_debug(f"Sending effective thresholds to UI: {msg_thresholds}")
                        await session.send_custom_message("set_bias_thresholds", msg_thresholds)

            except asyncio.TimeoutError:
                log_debug("ERROR: Bias analysis timed out (limit: 180s)")
                _logger.error("Bias analysis timed out (limit: 180s)")
                ui.notification_show("Analysis timed out.", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            except Exception as e:
                _logger.exception("Bias analysis failed")
                log_debug(f"ERROR during execution: {type(e).__name__}")
                ui.notification_show("Analysis failed. Please try again or check the server logs.", type="error")
                bias_results.set(None)
                bias_results_B.set(None)
            finally:
                log_debug("Stopping loading UI")
                # Increment generation counter to mark results as fresh
                # This causes the next update_bias_results_live call to be skipped
                new_gen = bias_analysis_generation.get() + 1
                bias_analysis_generation.set(new_gen)
                bias_running.set(False)
                await session.send_custom_message('stop_bias_loading', {})
                # Show back button whenever prior results existed (second run in same session)
                show_back = had_prior_results and bias_results.get() is not None
                await session.send_custom_message('bias_back_btn_update', {'show': show_back})

        except Exception as e:
            log_debug(f"CRITICAL ERROR in compute_bias top level: {e}")
            _logger.exception("CRITICAL ERROR in compute_bias top level")

    # ── Back button: restore previous run snapshot ──

    @reactive.effect
    @reactive.event(input.bias_go_back)
    async def bias_restore_from_snapshot():
        snap = bias_snapshot.get()
        if not snap:
            return
        bias_results.set(snap['results'])
        bias_results_B.set(snap['results_B'])
        bias_cached_text_A.set(snap.get('text_A', ''))
        bias_cached_text_B.set(snap.get('text_B', ''))
        active_bias_compare_models.set(snap.get('compare_models', False))
        active_bias_compare_prompts.set(snap.get('compare_prompts', False))
        bias_snapshot.set(None)
        await session.send_custom_message('bias_back_btn_update', {'show': False})
        await session.send_custom_message('bias_restore_ui', {
            'compare_models': snap.get('compare_models', False),
            'compare_prompts': snap.get('compare_prompts', False),
        })

    # ── Counterfactual click handler ──

    @reactive.effect
    @reactive.event(input.bias_trigger_counterfactual)
    async def trigger_counterfactual():
        orig = bias_cached_text_A.get()
        all_swaps = counterfactual_swaps.get()
        if not orig or not all_swaps:
            return

        # Get selected swap keys from JS (list of lowercase token strings)
        selected_raw = input.bias_trigger_counterfactual()
        if not selected_raw or not isinstance(selected_raw, (list, tuple)):
            return
        selected_keys = set(str(k).lower() for k in selected_raw)

        # Filter swaps to only the user-selected ones
        filtered = [s for s in all_swaps if s["term"].lower() in selected_keys]
        if not filtered:
            return

        # Generate counterfactual with selected swaps only
        cf_text, applied = generate_counterfactual(orig, filtered)
        cf_applied_swaps.set(applied)

        # 1. Enable Compare Prompts mode (toggle switch + explicit Shiny input)
        await session.send_custom_message("bias_toggle_switch",
            {"id": "bias_compare_prompts_mode", "checked": True,
             "shinyId": "bias_compare_prompts_mode"})
        await asyncio.sleep(0.3)

        # 2. Fill textareas A and B
        await session.send_custom_message("bias_set_textarea",
            {"id": "bias_input_text", "value": orig})
        await session.send_custom_message("bias_set_textarea",
            {"id": "bias_input_text_B", "value": cf_text})
        await asyncio.sleep(0.3)

        # 3. Set step to "B" so the analyze click skips the sequential intercept
        bias_prompt_step.set("B")
        await asyncio.sleep(0.1)

        # 4. Trigger analysis directly by setting the action button value
        #    AND clicking the DOM element just to be safe
        await session.send_custom_message("bias_click_analyze", {})

    # ── Batch file upload handler ──

    @reactive.effect
    @reactive.event(input.batch_file_upload)
    async def parse_batch_file():
        """Parse uploaded CSV or JSON file for batch mode."""
        import csv
        import io
        file_info = input.batch_file_upload()
        if not file_info:
            return
        try:
            file_info = file_info[0]
            file_path = file_info["datapath"]
            file_name = file_info["name"]

            sentences = []
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_name.lower().endswith(".json"):
                data = json.loads(content)
                if isinstance(data, list):
                    if all(isinstance(x, str) for x in data):
                        sentences = [s.strip() for s in data if s.strip()]
                    elif all(isinstance(x, dict) for x in data):
                        for item in data:
                            for key in ("text", "sentence", "input", "content"):
                                if key in item and isinstance(item[key], str) and item[key].strip():
                                    sentences.append(item[key].strip())
                                    break
                elif isinstance(data, dict):
                    for key in ("sentences", "texts", "data"):
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                if isinstance(item, str) and item.strip():
                                    sentences.append(item.strip())
                                elif isinstance(item, dict):
                                    for k in ("text", "sentence", "input"):
                                        if k in item and isinstance(item[k], str) and item[k].strip():
                                            sentences.append(item[k].strip())
                                            break
                            break
            elif file_name.lower().endswith(".csv"):
                reader = csv.DictReader(io.StringIO(content))
                cols = reader.fieldnames or []
                text_col = None
                for candidate in ("text", "sentence", "input", "content"):
                    for c in cols:
                        if c.lower().strip() == candidate:
                            text_col = c
                            break
                    if text_col:
                        break
                if not text_col and cols:
                    text_col = cols[0]
                if text_col:
                    for row in reader:
                        val = row.get(text_col, "").strip()
                        if val:
                            sentences.append(val)

            MAX_BATCH_SIZE = 500
            if len(sentences) > MAX_BATCH_SIZE:
                sentences = sentences[:MAX_BATCH_SIZE]

            if sentences:
                batch_sentences.set(sentences)
                batch_file_name.set(file_name)
                await session.send_custom_message("batch_file_parsed", {
                    "filename": file_name,
                    "count": len(sentences),
                })
            else:
                batch_sentences.set([])
                await session.send_custom_message("batch_file_parsed", {
                    "error": "No sentences found in file",
                })
        except Exception as e:
            batch_sentences.set([])
            await session.send_custom_message("batch_file_parsed", {
                "error": f"Parse error: {str(e)[:80]}",
            })

    # ── Batch processing pipeline ──

    async def _run_batch_analysis():
        """Run comprehensive analysis on all batch sentences."""
        import hashlib
        import time as _time

        sentences = batch_sentences.get()
        if not sentences:
            return

        t0 = _time.time()

        try:
            bias_model_key = input.bias_model_key()
        except Exception:
            bias_model_key = "gusnet-bert"
        model_name = _GUSNET_TO_ENCODER.get(bias_model_key, "bert-base-uncased")
        is_gpt2 = "gpt2" in model_name

        try:
            use_optimized = bool(input.bias_use_optimized())
        except Exception:
            use_optimized = True

        thresholds = current_thresholds_A.get()
        n_total = len(sentences)

        per_sentence_results = []
        all_bar_matrices = []
        all_bsr_matrices = []

        loop = asyncio.get_running_loop()

        for idx, text in enumerate(sentences):
            sentence_data = {
                "index": idx,
                "text": text,
                "is_biased": False,
                "error": None,
            }
            try:
                # ── Phase 1: GUS-Net bias detection ──
                await session.send_custom_message("batch_progress", {
                    "label": f"Analyse Bias ({idx+1}/{n_total})"
                })

                with ThreadPoolExecutor(max_workers=1) as pool:
                    raw = await asyncio.wait_for(
                        loop.run_in_executor(pool, heavy_bias_compute, text, model_name, thresholds, bias_model_key, use_optimized),
                        timeout=120.0
                    )

                if not raw:
                    sentence_data["error"] = "Bias detection failed"
                    per_sentence_results.append(sentence_data)
                    continue

                processed = _process_raw_bias_result(raw, raw.get("effective_thresholds", thresholds), use_optimized=use_optimized)
                if not processed:
                    sentence_data["error"] = "Result processing failed"
                    per_sentence_results.append(sentence_data)
                    continue

                # Extract bias info
                token_labels = processed.get("token_labels", [])
                bias_summary = processed.get("bias_summary", {})
                bias_spans = processed.get("bias_spans", [])
                metrics = processed.get("attention_metrics", [])
                attentions = processed.get("attentions")
                propagation = processed.get("propagation_analysis", {})
                bias_matrix = processed.get("bias_matrix")

                biased_count = sum(1 for t in token_labels if t.get("is_biased"))
                total_tokens = len(token_labels)
                bias_pct = round(biased_count / total_tokens * 100, 1) if total_tokens else 0
                is_biased = biased_count > 0

                # Token scores for report
                token_scores = []
                for tl in token_labels:
                    scores = tl.get("scores", {})
                    token_scores.append({
                        "token": tl.get("token", ""),
                        "is_biased": tl.get("is_biased", False),
                        "bias_types": tl.get("bias_types", []),
                        "scores": {k: round(float(v), 4) for k, v in scores.items()} if scores else {},
                    })

                # Biased spans
                span_data = []
                for sp in bias_spans:
                    span_data.append({
                        "text": sp.get("text", ""),
                        "bias_types": sp.get("bias_types", []),
                        "avg_score": round(float(sp.get("avg_score", 0)), 4),
                    })

                # Category counts
                cat_counts = {}
                for tl in token_labels:
                    if tl.get("is_biased"):
                        for bt in tl.get("bias_types", []):
                            cat_counts[bt] = cat_counts.get(bt, 0) + 1

                # Confidence
                confidences = []
                for tl in token_labels:
                    if tl.get("is_biased") and tl.get("scores"):
                        max_score = max(v for k, v in tl["scores"].items() if k != "O")
                        confidences.append(float(max_score))
                avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0

                sentence_data.update({
                    "is_biased": is_biased,
                    "bias_summary": {
                        "total_tokens": total_tokens,
                        "biased_tokens": biased_count,
                        "bias_percentage": bias_pct,
                        "generalization_count": cat_counts.get("GEN", 0),
                        "unfairness_count": cat_counts.get("UNFAIR", 0),
                        "stereotype_count": cat_counts.get("STEREO", 0),
                        "avg_confidence": avg_conf,
                        "categories_found": list(cat_counts.keys()),
                    },
                    "biased_spans": span_data,
                    "token_scores": token_scores,
                })

                # BAR/BSR matrix
                if bias_matrix is not None:
                    all_bar_matrices.append(np.array(bias_matrix))
                    sentence_data["bar_matrix_shape"] = list(np.array(bias_matrix).shape)

                # Propagation
                if propagation:
                    layer_bars = propagation.get("layer_propagation", [])
                    sentence_data["propagation"] = {
                        "layer_bars": [round(float(x), 4) for x in layer_bars],
                        "peak_layer": propagation.get("peak_layer"),
                        "pattern": propagation.get("propagation_pattern", "none"),
                    }

                # ── Phase 2: Integrated Gradients ──
                ig_data = None
                try:
                    tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        ig_bundle = await asyncio.wait_for(
                            loop.run_in_executor(pool, batch_compute_ig_correlation,
                                encoder_model, tokenizer, text, list(attentions), metrics, is_gpt2),
                            timeout=120.0
                        )
                    if ig_bundle and isinstance(ig_bundle, IGAnalysisBundle):
                        ig_corrs = []
                        for c in (ig_bundle.correlations or []):
                            ig_corrs.append({
                                "layer": c.layer, "head": c.head,
                                "spearman_rho": round(float(c.spearman_rho), 4),
                                "p_value": round(float(c.spearman_pvalue), 6) if c.spearman_pvalue is not None else None,
                                "bar": round(float(c.bar_original), 4) if c.bar_original is not None else None,
                            })
                        topk_data = []
                        for t in (ig_bundle.topk_overlaps or []):
                            topk_data.append({
                                "layer": t.layer, "head": t.head, "k": t.k,
                                "jaccard": round(float(t.jaccard), 4),
                                "rbo": round(float(t.rank_biased_overlap), 4) if t.rank_biased_overlap is not None else None,
                                "bar": round(float(t.bar_original), 4) if t.bar_original is not None else None,
                            })
                        ig_data = {
                            "token_attributions": [round(float(x), 6) for x in ig_bundle.token_attributions],
                            "top_ig_tokens": [
                                {"token": ig_bundle.tokens[i], "attribution": round(float(ig_bundle.token_attributions[i]), 6), "rank": r+1}
                                for r, i in enumerate(np.argsort(ig_bundle.token_attributions)[::-1][:10])
                            ],
                            "ig_attention_correlations": ig_corrs[:20],
                            "topk_overlaps": topk_data[:20],
                        }
                        sentence_data["ig_analysis"] = ig_data
                except Exception as e:
                    sentence_data["ig_error"] = str(e)[:100]

                # ── Phase 3: Head Ablation ──
                try:
                    top_k = min(10, len(metrics))
                    top_heads_local = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:top_k]
                    tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        ablation_list = await asyncio.wait_for(
                            loop.run_in_executor(pool, batch_ablate_top_heads,
                                encoder_model, lm_head_model, tokenizer, text, top_heads_local, is_gpt2),
                            timeout=120.0
                        )
                    if ablation_list:
                        abl_data = []
                        for a in ablation_list:
                            abl_data.append({
                                "layer": a.layer, "head": a.head,
                                "representation_impact": round(float(a.representation_impact), 6),
                                "kl_divergence": round(float(a.kl_divergence), 6) if a.kl_divergence is not None else None,
                                "bar": round(float(a.bar_original), 4) if a.bar_original is not None else None,
                            })
                        sentence_data["ablation_analysis"] = abl_data
                except Exception as e:
                    sentence_data["ablation_error"] = str(e)[:100]

                # ── Phase 4: Perturbation ──
                try:
                    if ig_data and ig_bundle:
                        tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            perturb_bundle = await asyncio.wait_for(
                                loop.run_in_executor(pool, batch_compute_perturbation,
                                    encoder_model, tokenizer, text, is_gpt2, ig_bundle.token_attributions, list(attentions)),
                                timeout=120.0
                            )
                        if perturb_bundle and isinstance(perturb_bundle, PerturbationAnalysisBundle):
                            tok_imp = [{"token": r.token, "importance": round(float(r.importance), 6)} for r in perturb_bundle.token_results]
                            attn_corrs = [{"layer": l, "head": h, "spearman_rho": round(float(r), 4)} for l, h, r in (perturb_bundle.perturb_vs_attn_spearman or [])[:10]]
                            sentence_data["perturbation_analysis"] = {
                                "token_importance": tok_imp,
                                "perturb_vs_ig_spearman": round(float(perturb_bundle.perturb_vs_ig_spearman), 4) if perturb_bundle.perturb_vs_ig_spearman is not None else None,
                                "perturb_vs_attention_top5": attn_corrs,
                            }
                except Exception as e:
                    sentence_data["perturbation_error"] = str(e)[:100]

                # ── Phase 5: LRP / DeepLift ──
                try:
                    if ig_data and ig_bundle:
                        tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            lrp_bundle = await asyncio.wait_for(
                                loop.run_in_executor(pool, batch_compute_lrp,
                                    encoder_model, tokenizer, text, is_gpt2, ig_bundle.token_attributions, list(attentions), metrics),
                                timeout=120.0
                            )
                        if lrp_bundle and isinstance(lrp_bundle, LRPAnalysisBundle):
                            lrp_corrs = [{"layer": l, "head": h, "spearman_rho": round(float(r), 4)} for l, h, r in (lrp_bundle.correlations or [])[:10]]
                            sentence_data["lrp_analysis"] = {
                                "token_attributions": [round(float(x), 6) for x in lrp_bundle.token_attributions],
                                "lrp_vs_ig_spearman": round(float(lrp_bundle.lrp_vs_ig_spearman), 4) if lrp_bundle.lrp_vs_ig_spearman is not None else None,
                                "lrp_vs_attention_top5": lrp_corrs,
                            }
                except Exception as e:
                    sentence_data["lrp_error"] = str(e)[:100]

                # ── Phase 6: Counterfactual swaps ──
                try:
                    cf_swaps = find_swappable_terms(text)
                    if cf_swaps:
                        sentence_data["counterfactual_swaps"] = [
                            {"term": s["term"], "swap_to": s["swap_to"], "category": s["category"]}
                            for s in cf_swaps
                        ]
                except Exception:
                    _logger.debug("Suppressed exception", exc_info=True)
                    pass

            except asyncio.TimeoutError:
                sentence_data["error"] = "Timeout (120s)"
            except Exception as e:
                sentence_data["error"] = str(e)[:200]
            finally:
                # Clear GPU memory between sentences
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_sentence_results.append(sentence_data)

        elapsed = round(_time.time() - t0, 1)

        # ── Build comprehensive report ──
        await session.send_custom_message("batch_progress", {
            "label": f"Analyse Bias ({n_total}/{n_total})"
        })

        report = _build_batch_report(
            per_sentence_results, all_bar_matrices,
            bias_model_key, model_name, thresholds, use_optimized,
            batch_file_name.get(), elapsed, n_total
        )

        # Save to disk
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"batch_report_{bias_model_key}_{ts}.json"
        save_path = Path("downloads/batch/bias") / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Auto-download
        await session.send_custom_message("batch_download_ready", {
            "json_content": json.dumps(report, indent=2, ensure_ascii=False, default=str),
            "filename": fname,
        })

        analyzed = sum(1 for r in per_sentence_results if not r.get("error"))
        ui.notification_show(
            f"Batch complete! {analyzed}/{n_total} sentences, {elapsed}s",
            type="message", duration=8
        )

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
                        {"class": "card compare-card-a"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_A,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("PROMPT B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b"},
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
                        {"class": "card compare-card-a"},
                        ui.h4("Sentence Preview", style="margin: 0 0 8px 0;"),
                        preview_content,
                    ),
                ),
                # Column B
                ui.div(
                    {"style": "display: flex; flex-direction: column;"},
                    ui.h3("MODEL B", style="font-size:16px; color:#ff5ca9; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:24px; padding-bottom:8px; border-bottom: 2px solid #ff5ca9;"),
                    ui.div(
                        {"class": "card compare-card-b"},
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
                {"class": "card"},
                ui.h4("Sentence Preview"),
                preview,
            )

        # Initial / Running State: Keep sentence preview + architecture visible
        # (mirrors the attention tab behavior before analysis results exist).
        # Defensive fallback: never let this block fail silently and hide UI.
        try:
            # Determine architecture display mode based on current compare mode
            if res:
                selected_model = res.get("bias_model_key", "gusnet-bert")
            else:
                try:
                    selected_model = input.bias_model_key()
                except BaseException:
                    selected_model = "gusnet-bert"

            try:
                current_compare_models = bool(input.bias_compare_mode())
            except BaseException:
                current_compare_models = False

            # Get Model B selection for Compare Models mode
            try:
                model_b = input.bias_model_key_B()
                if not model_b:
                    model_b = "gusnet-gpt2"  # Default Model B
            except BaseException:
                model_b = "gusnet-gpt2"

            # Determine compare-prompts from the actual toggle, not text content.
            try:
                current_compare_prompts = bool(input.bias_compare_prompts_mode())
            except BaseException:
                current_compare_prompts = False

            arch_section = ui.div(
                get_gusnet_architecture_section(
                    selected_model=selected_model,
                    compare_mode=current_compare_models,
                    compare_prompts=current_compare_prompts,
                    model_a=selected_model,  # Model A comes from main selector
                    model_b=model_b,  # Model B comes from compare selector
                ),
            )
        except Exception:
            # Hard fallback with safe defaults so pre-analysis UI never disappears.
            arch_section = ui.div(
                get_gusnet_architecture_section(
                    selected_model="gusnet-bert",
                    compare_mode=False,
                    compare_prompts=False,
                    model_a="gusnet-bert",
                    model_b="gusnet-gpt2",
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

        abl = ablation_results.get()
        ig = ig_results.get()
        abl_B = ablation_results_B.get()
        ig_B = ig_results_B.get()

        def get_summary_cards(res_data, abl_data=None, ig_bundle=None):
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

            # ── Benchmark cards (BAR / IG ρ / Ablation Δ) ──
            # Root cause of tooltip failure: `.card { overflow:hidden }` clips position:absolute
            # children that extend outside card bounds.
            # Fix: keep position:fixed from the CSS class (fixed elements escape overflow:hidden),
            # and use getBoundingClientRect() in JS to position the tooltip near the (i) icon.
            # transform:none on metric-card prevents the hover-transform from creating a new
            # containing block (which would make position:fixed behave like position:absolute).
            _oe = (
                "var t=this.querySelector('.bcard-tooltip');"
                "var r=this.getBoundingClientRect();"
                "t.style.bottom=(window.innerHeight-r.top+8)+'px';"
                "t.style.top='auto';"
                "t.style.left=(r.left+r.width/2)+'px';"
                "t.style.transform='translateX(-50%)';"
                "t.style.visibility='visible';"
                "t.style.opacity='1';"
            )
            _ol = (
                "var t=this.querySelector('.bcard-tooltip');"
                "t.style.visibility='hidden';"
                "t.style.opacity='0';"
            )

            # _TH without text-transform:uppercase for these benchmark tooltips
            _TH_lc = _TH.replace("text-transform:uppercase;", "")

            def _bcard(label, value, color="#334155", sub=None, tooltip=None):
                sub_html = f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">{sub}</div>' if sub else ""
                if tooltip:
                    # Use a unique class so CSS :hover rule does NOT apply
                    info = (
                        f'<span class="info-tooltip-wrapper" style="margin-left:4px;vertical-align:middle;"'
                        f' onmouseenter="{_oe}" onmouseleave="{_ol}">'
                        f'<span class="info-tooltip-icon">i</span>'
                        f'<div class="bcard-tooltip"'
                        f' style="visibility:hidden;opacity:0;position:fixed;z-index:9999999;'
                        f'background:linear-gradient(135deg,#1e293b 0%,#334155 100%);color:#f1f5f9;'
                        f'padding:16px 20px;border-radius:12px;font-size:12px;line-height:1.7;'
                        f'width:380px;max-width:420px;'
                        f'box-shadow:0 15px 50px rgba(0,0,0,0.5),0 0 0 1px rgba(255,92,169,0.3);'
                        f'border:1px solid rgba(255,255,255,0.15);'
                        f'transition:opacity 0.2s ease,visibility 0.2s ease;pointer-events:none;"'
                        f'>{tooltip}</div>'
                        f'</span>'
                    )
                else:
                    info = ""
                return (
                    f'<div class="metric-card" style="min-width:130px;transform:none;">'
                    f'<div class="metric-label" style="display:flex;align-items:center;text-transform:none;">{label}{info}</div>'
                    f'<div class="metric-value" style="color:{color};font-size:20px;">{value}</div>'
                    f'{sub_html}</div>'
                )

            # BAR
            attn = res_data.get("attention_metrics", [])
            if attn:
                bars = [m.bias_attention_ratio for m in attn]
                mean_bar = sum(bars) / len(bars)
                n_spec = sum(1 for m in attn if m.specialized_for_bias)
                bar_color = "#dc2626" if mean_bar > 1.5 else ("#ea580c" if mean_bar > 1.2 else "#22c55e")
                bar_val, bar_sub = f"{mean_bar:.3f}", f"{n_spec}/{len(attn)} heads specialized"
            else:
                bar_val, bar_color, bar_sub = "-", "#94a3b8", "run Analyze Bias first"

            # IG ρ
            if ig_bundle is not None:
                from ..bias.integrated_gradients import IGAnalysisBundle
                corrs = ig_bundle.correlations if isinstance(ig_bundle, IGAnalysisBundle) else ig_bundle
                if corrs:
                    rhos = [r.spearman_rho for r in corrs]
                    mean_rho = sum(rhos) / len(rhos)
                    n_sig = sum(1 for r in corrs if r.spearman_pvalue < 0.05)
                    rho_color = "#2563eb" if mean_rho > 0 else "#dc2626"
                    rho_val, rho_sub = f"{mean_rho:.3f}", f"{n_sig}/{len(corrs)} heads significant"
                else:
                    rho_val, rho_color, rho_sub = "-", "#94a3b8", "no IG data"
            else:
                rho_val, rho_color, rho_sub = "-", "#94a3b8", "run IG analysis first"

            # Ablation Δ
            if abl_data:
                max_delta = max(r.representation_impact for r in abl_data)
                top = abl_data[0]
                abl_color = "#ff5ca9" if max_delta > 0.05 else "#64748b"
                abl_val, abl_sub = f"{max_delta:.3f}", f"top: L{top.layer}H{top.head}"
            else:
                abl_val, abl_color, abl_sub = "-", "#94a3b8", "run ablation first"

            _tt_bar = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Ratio of attention paid to biased tokens vs. chance (μ̂_B / μ₀)</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#dc2626;'>●</span>"
                f"<span>BAR &gt; 1.5 → <span style='{_TBR}'>bias-specialized</span></span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ea580c;'>●</span>"
                f"<span>BAR &gt; 1.2 → <span style='{_TBA}'>elevated</span></span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span>BAR ≤ 1.2 → <span style='{_TBG}'>normal</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Higher values indicate stronger bias-driven attention patterns.</div>"
            )
            _tt_rho = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Spearman correlation between attention weights and Integrated Gradients attributions, per head</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Interpretation</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#2563eb;'>●</span>"
                f"<span>ρ ≈ 1 → <span style='{_TBB}'>attention faithful</span> to token importance</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>ρ ≈ 0 → attention is <span style='{_TBG}'>uninformative</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Significance threshold: p &lt; 0.05 per head.</div>"
            )
            _tt_abl = (
                f"<span style='{_TH_lc}'>What it measures</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Max representation change (1 − cosine similarity) from zeroing a single head</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH_lc}'>Threshold</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
                f"<span>Δ &gt; 0.05 → <span style='{_TBP}'>high-impact head</span></span></div>"
                f"<div style='{_TN};margin-top:6px;'>Identifies the head whose removal most disrupts the model's internal representation.</div>"
            )
            bench_cards = (
                '<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(226,232,240,0.4);">'
                '<div class="metrics-grid">'
                + _bcard("Mean BAR", bar_val, bar_color, bar_sub, _tt_bar)
                + _bcard("Mean IG ρ", rho_val, rho_color, rho_sub, _tt_rho)
                + _bcard("Max Ablation Δ", abl_val, abl_color, abl_sub, _tt_abl)
                + '</div></div>'
            )

            return criteria_html + cards + bench_cards

        if (compare_models or compare_prompts) and res_B:
            content_A = get_summary_cards(res, abl, ig)
            content_B = get_summary_cards(res_B, abl_B, ig_B)

            header_args = (
                "Bias Detection Summary",
                "Composite bias level with explicit weighted criteria and per-category counts.",
                f"<span style='{_TH}'>What this shows</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                f"<span>4-level severity: <span style='{_TBG}'>None</span> "
                f"<span style='{_TBB}'>Low</span> <span style='{_TBA}'>Moderate</span> "
                f"<span style='{_TBR}'>High</span></span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Weights</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span><span style='{_TBG}'>30%</span>&nbsp;Token Density: fraction of tokens flagged</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
                f"<span><span style='{_TBR}'>25%</span>&nbsp;Unfair Language: slurs, loaded framing, and explicit prejudice</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
                f"<span><span style='{_TBA}'>25%</span>&nbsp;Stereotypes: attribute-group co-occurrence patterns</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
                f"<span><span style='{_TBP}'>20%</span>&nbsp;Generalisations: universal claims about groups</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Low &lt; 0.15 &nbsp;·&nbsp; Moderate 0.15–0.40 &nbsp;·&nbsp; High &ge; 0.40</span></div>"
                f"<div style='{_TN}; margin-top:6px;'>High token density + low stereotype score → loaded language without group-specific targeting.</div>"
            )

            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(content_A), *header_args, style="margin-bottom:0; border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(content_B), *header_args, style="margin-bottom:0; border: 2px solid #ff5ca9; height: 100%;")
            )
        else:
            header_args = (
                "Bias Detection Summary",
                "Composite bias level with explicit weighted criteria and per-category counts.",
                f"<span style='{_TH}'>What this shows</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
                f"<span>4-level severity: <span style='{_TBG}'>None</span> "
                f"<span style='{_TBB}'>Low</span> <span style='{_TBA}'>Moderate</span> "
                f"<span style='{_TBR}'>High</span></span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Weights</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
                f"<span><span style='{_TBG}'>30%</span>&nbsp;Token Density: fraction of tokens flagged</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
                f"<span><span style='{_TBR}'>25%</span>&nbsp;Unfair Language: slurs, loaded framing, and explicit prejudice</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
                f"<span><span style='{_TBA}'>25%</span>&nbsp;Stereotypes: attribute-group co-occurrence patterns</span></div>"
                f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
                f"<span><span style='{_TBP}'>20%</span>&nbsp;Generalisations: universal claims about groups</span></div>"
                f"<hr style='{_TS}'>"
                f"<span style='{_TH}'>Thresholds</span>"
                f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
                f"<span>Low &lt; 0.15 &nbsp;·&nbsp; Moderate 0.15–0.40 &nbsp;·&nbsp; High &ge; 0.40</span></div>"
                f"<div style='{_TN}; margin-top:6px;'>High token density + low stereotype score → loaded language without group-specific targeting.</div>"
            )
            return _wrap_card(ui.HTML(get_summary_cards(res, abl, ig)), *header_args, style="margin-bottom: 24px;")

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
            _logger.exception("Error in inline_bias_view")
            return ui.HTML(f'<div style="color:#ef4444;">Error: {_html.escape(str(e))}</div>')

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

        def _hex_to_rgba(hex_color, alpha):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

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

            base_color = "#3b82f6" if variant == "A" else "#ff5ca9"

            items = []
            for lbl in biased:
                clean = _html.escape(lbl["token"].replace("##", "").replace("\u0120", ""))
                tok_idx = lbl["index"]

                # Pick color of category with highest score
                scores = lbl.get("scores", {})
                if scores:
                    max_cat = max(scores, key=lambda k: scores[k])
                    item_color = cat_colors.get(max_cat, base_color)
                else:
                    types = lbl.get("bias_types", [])
                    item_color = cat_colors.get(types[0], base_color) if types else base_color

                chip_style = (
                    f"background:{_hex_to_rgba(item_color, 0.18)};"
                    f"border-color:{_hex_to_rgba(item_color, 0.55)};"
                    f"color:white;"
                    f"--chip-color:{item_color};"
                )

                items.append(
                    f'<span class="token-chip bias-token-chip" '
                    f'data-token-idx="{tok_idx}" '
                    f'data-prefix="{variant}" '
                    f'style="{chip_style}" '
                    f'onclick="selectBiasToken({tok_idx}, \'{variant}\')">'
                    f'{clean}'
                    f'</span>'
                )
            return "".join(items)

        compare_prompts = active_bias_compare_prompts.get()
        compare_models = active_bias_compare_models.get()

        # Tell the JS whether we are in compare-models mode (same text,
        # two models) so selectBiasToken mirrors A selection to B.
        import json as _json
        mirror_flag = "true" if (compare_models and not compare_prompts) else "false"
        # Embed cleaned token lists so JS can match by content across models
        _res_A = bias_results.get()
        _res_B = bias_results_B.get() if compare_models else None
        _toks_a = [t.replace("##", "").replace("\u0120", "") for t in (_res_A or {}).get("tokens", [])]
        _toks_b = [t.replace("##", "").replace("\u0120", "") for t in (_res_B or {}).get("tokens", [])]
        mirror_script = ui.tags.script(
            f"window._biasCompareModels = {mirror_flag};"
            f"window._biasTokensA = {_json.dumps(_toks_a)};"
            f"window._biasTokensB = {_json.dumps(_toks_b)};"
        )

        if compare_prompts:
            # ── Compare Prompts Layout (same structure as attention tab) ──
            res_A = bias_results.get()
            res_B = bias_results_B.get()
            chips_A = _build_chips_html(res_A, "A")
            chips_B = _build_chips_html(res_B, "B")

            return ui.div(
                mirror_script,
                {"class": "token-row-split"},
                ui.div(
                    {"class": "token-split-item"},
                    ui.span("A", class_="model-label-a"),
                    ui.HTML(chips_A),
                ),
                ui.div(
                    {"class": "token-split-item item-b"},
                    ui.span("B", class_="model-label-b"),
                    ui.HTML(chips_B),
                ),
            )
        else:
            # ── Single / Compare Models ──
            res = bias_results.get()
            chips = _build_chips_html(res, "A")
            return ui.div(
                mirror_script,
                ui.HTML(
                    f'<div class="token-sentence">'
                    f'{chips}'
                    f'</div>'
                ),
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
            # Read the side-specific selection so A and B stay independent.
            try:
                s = input.bias_selected_tokens_B() if is_B else input.bias_selected_tokens_A()
                if s: sel_indices = [int(s)] if isinstance(s,(int,str)) else [int(x) for x in s if x]
            except Exception:
                _logger.debug("Suppressed exception", exc_info=True)
            # Compare-models: always derive B from A by token content
            if is_B and compare_models and not compare_prompts:
                sel_a = _read_sel(input.bias_selected_tokens_A)
                mapped = _map_sel_a_to_b(sel_a, res, res_B)
                if mapped:
                    sel_indices = mapped
                
            # Check if counterfactual swaps are available (single-prompt mode)
            show_cf = (not is_B and not compare_models and not compare_prompts
                       and counterfactual_swaps.get())

            for lbl in biased:
                clean = _html.escape(lbl["token"].replace("##", "").replace("\u0120", ""))
                types = lbl.get("bias_types", [])
                scores = lbl.get("scores", {})
                badges = "".join([f'<span style="display:inline-flex;align-items:center;gap:4px;background:{cat_colors.get(t,"#ff5ca9")}18;border:1px solid {cat_colors.get(t,"#ff5ca9")}40;color:{cat_colors.get(t,"#ff5ca9")};padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;">{_html.escape(t)}<span style="font-family:JetBrains Mono;font-weight:400;opacity:0.8;">{scores.get(t,0):.2f}</span></span>' for t in types])

                # Counterfactual swap badge (if token is swappable)
                cf_badge = ""
                if show_cf:
                    swap_info = get_swap_for_token(clean)
                    if swap_info:
                        swap_to, swap_cat = swap_info
                        cf_key = clean.lower()
                        cf_badge = (
                            f'<span class="cf-swap-badge" data-cf-key="{cf_key}" '
                            f'onclick="event.stopPropagation(); window.toggleCfSwap(\'{cf_key}\', \'{swap_to}\', \'{swap_cat}\')" '
                            f'style="display:inline-flex;align-items:center;gap:3px;'
                            f'background:#1e293b;border:1px solid rgba(255,92,169,0.5);'
                            f'color:#ff5ca9;padding:2px 8px;border-radius:4px;font-size:10px;'
                            f'font-weight:600;cursor:pointer;transition:all 0.15s ease;'
                            f'margin-left:4px;" '
                            f'title="Click to select this swap for counterfactual comparison">'
                            f'&#x21C4; {swap_to}</span>'
                        )

                style = "display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid rgba(226,232,240,0.4);"
                if lbl["index"] in sel_indices: style += "background:rgba(255, 92, 169, 0.1); border-left: 3px solid #ff5ca9;"
                items.append(f'<div style="{style}"><span style="font-family:JetBrains Mono;font-size:13px;font-weight:600;color:#ec4899;min-width:70px;">{clean}</span><span style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;">{badges}{cf_badge}</span></div>')
            
            import math
            mid = math.ceil(len(items)/2)
            c1, c2 = items[:mid], items[mid:]
            
            # Show per-category thresholds - always read from live slider inputs
            t_info = "Thresholds: "

            # Read directly from slider inputs (always reflects sidebar state)
            if not is_B:
                try: u = float(input.bias_thresh_unfair())
                except Exception: u = current_thresholds_A.get().get("UNFAIR", 0.5)
                try: g = float(input.bias_thresh_gen())
                except Exception: g = current_thresholds_A.get().get("GEN", 0.5)
                try: s = float(input.bias_thresh_stereo())
                except Exception: s = current_thresholds_A.get().get("STEREO", 0.5)
            else:
                try: u = float(input.bias_thresh_unfair_b())
                except Exception: u = current_thresholds_B.get().get("UNFAIR", 0.5)
                try: g = float(input.bias_thresh_gen_b())
                except Exception: g = current_thresholds_B.get().get("GEN", 0.5)
                try: s = float(input.bias_thresh_stereo_b())
                except Exception: s = current_thresholds_B.get().get("STEREO", 0.5)

            t_info += f'UNFAIR: <code>{u:.2f}</code> &bull; GEN: <code>{g:.2f}</code> &bull; STEREO: <code>{s:.2f}</code>'
            if data and data.get("use_optimized"):
                t_info += " <span style='opacity:0.6;'>(Optimized)</span>"

            return (f'<div style="display:flex;gap:16px;border:1px solid rgba(226,232,240,0.4);border-radius:8px;overflow:hidden;">'
                    f'<div style="flex:1;display:flex;flex-direction:column;">{"".join(c1)}</div>'
                    f'<div style="flex:1;display:flex;flex-direction:column;border-left:1px solid rgba(226,232,240,0.4);">{"".join(c2)}</div>'
                    f'</div><div style="margin-top:8px;font-size:10px;color:#94a3b8;text-align:center;">{t_info}</div>')

        # Use manual header for Detected Bias Tokens
        # old ui: h4("Detected Bias Tokens", ...), p(...)
        man_header = (
            "Detected Bias Tokens",
            "Each biased span with category labels and GUS-Net confidence scores (hover a token for details).",
        )
        _detected_bias_help = (
            f"<span style='{_TH}'>Categories: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>UNFAIR</span>&nbsp;explicit prejudice, slurs, loaded framing</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>GEN</span>&nbsp;overgeneralisations, for example 'all X are Y'</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>STEREO</span>&nbsp;stereotyped attribute–group co-occurrences</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Confidence score</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>GUS-Net softmax probability for predicted bias class</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span>Tokens below threshold (default 0.5) are not highlighted</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interaction</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▶</span>"
            f"<span>Click a token to highlight it across the token-level bias distribution and confidence breakdown views</span></div>"
            f"<div style='{_TN}; margin-top:4px;'>Multiple categories can fire on the same token. The primary label is the highest-confidence class.</div>"
        )
        
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

            side_by_side = ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}</div>' + method_footer_A), manual_header=man_header, help_text=_detected_bias_help, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res_B, True)}</div>' + method_footer_B), manual_header=man_header, help_text=_detected_bias_help, style="border: 2px solid #ff5ca9; height: 100%;")
            )

            # Counterfactual consistency card — only when triggered via CF swap
            swaps_applied = cf_applied_swaps.get()
            log_debug(f"CF consistency check: compare_prompts={compare_prompts}, swaps={bool(swaps_applied)}, attn_A={bool(res.get('attentions'))}, attn_B={bool(res_B.get('attentions') if res_B else False)}")
            if compare_prompts and swaps_applied and res.get("attentions") and res_B and res_B.get("attentions"):
                try:
                    consistency = compute_cf_attention_consistency(
                        res["tokens"], res["attentions"],
                        res_B["tokens"], res_B["attentions"],
                    )
                    cf_html = create_cf_consistency_html(
                        consistency,
                        res["tokens"], res_B["tokens"],
                        swaps_applied,
                        bias_summary_A=res.get("bias_summary"),
                        bias_summary_B=res_B.get("bias_summary"),
                    )
                    _cf_help = (
                        f"<span style='{_TH}'>Attention Consistency Metrics</span>"
                        f"<div style='{_TR}'><span style='{_TD};color:#3b82f6;'>&#x25CF;</span>"
                        f"<span><b>Attention Similarity</b>: cosine similarity of attention-received profiles "
                        f"for non-swapped tokens (1.0 = identical patterns)</span></div>"
                        f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>&#x25CF;</span>"
                        f"<span><b>Swap Focus</b>: fraction of total attention change concentrated at "
                        f"the swapped demographic tokens (higher = model attention is more localized)</span></div>"
                        f"<div style='{_TR}'><span style='{_TD};color:#3b82f6;'>&#x25CF;</span>"
                        f"<span><b>Top Sensitive Heads</b>: attention heads with the largest total attention "
                        f"delta between original and counterfactual</span></div>"
                    )
                    consistency_card = _wrap_card(
                        ui.HTML(cf_html),
                        manual_header=(
                            "Counterfactual Attention Consistency",
                            "How attention patterns change when demographic terms are swapped.",
                        ),
                        help_text=_cf_help,
                        style="margin-top: 16px;",
                    )
                    return ui.div(side_by_side, consistency_card)
                except Exception:
                    _logger.exception("CF consistency error")

            return side_by_side

        method_html = create_method_info_html(model_key_A)
        method_footer = f'<div style="margin-top: auto;"><hr style="margin:16px 0;opacity:0.3;"/>{method_html}</div>'

        # Counterfactual compare button (single-prompt mode only)
        cf_section = ""
        cf_swaps_raw = counterfactual_swaps.get()
        if cf_swaps_raw and not compare_models and not compare_prompts:
            # Reset JS selection state whenever the panel re-renders
            cf_section = (
                '<script>window.selectedCfSwaps = window.selectedCfSwaps || {};</script>'
                '<div id="cf-compare-btn-container" style="display:none;margin-top:12px;text-align:center;">'
                '<button onclick="window.triggerCfCompare()" '
                'style="background:#1e293b;color:#ff5ca9;border:1px solid rgba(255,92,169,0.3);padding:8px 20px;'
                'border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;'
                'transition:all 0.15s ease;letter-spacing:0.5px;" '
                'onmouseover="this.style.background=\'rgba(255,92,169,0.1)\';this.style.borderColor=\'rgba(255,92,169,0.7)\';this.style.color=\'#ff74b8\'" '
                'onmouseout="this.style.background=\'#1e293b\';this.style.borderColor=\'rgba(255,92,169,0.3)\';this.style.color=\'#ff5ca9\'">'
                '<span style="margin-right:6px;">&#x21C4;</span>'
                'Compare Counterfactuals (<span id="cf-count">0</span>)'
                '</button></div>'
            )

        return _wrap_card(ui.HTML(f'<div style="flex: 1; display: flex; flex-direction: column;">{produce_html(res, False)}{cf_section}</div>' + method_footer), manual_header=man_header, help_text=_detected_bias_help)

    @output
    @render.ui
    def token_bias_strip():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')
        
        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        
        selected_idx = _read_sel(input.bias_selected_tokens_A)
        selected_idx_B = _read_sel(input.bias_selected_tokens_B)
        # Compare-models: always derive B from A by token content
        if compare_models and not compare_prompts and selected_idx:
            selected_idx_B = _map_sel_a_to_b(selected_idx, res, res_B)

        def get_viz(data, sel_idx=None):
            try:
                return create_token_bias_strip(data["token_labels"], sel_idx)
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        man_header = (
            "Token-Level Bias Distribution",
            "Per-token bias scores across all four categories (O, GEN, UNFAIR, STEREO) shown as coloured dot strips.",
        )
        _strip_help = (
            f"<span style='{_TH}'>What this shows: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Coloured dot strip showing each token's softmax score across all four bias categories, scored independently by GUS-Net</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Reading the strip</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Dot colour</b>: category (O / GEN / UNFAIR / STEREO)</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Dot size / bar height</b>: overall bias magnitude for that token</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><b>Grey / no fill</b>: token is below the detection threshold</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Categories</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>●</span>"
            f"<span><span style='{_TBG};color:#9ca3af;'>O</span>&nbsp;neutral: not biased</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ca8a04;'>●</span>"
            f"<span><span style='{_TBA};color:#ca8a04;'>GEN</span>&nbsp;generalisation: broad claims about a group</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR};color:#ef4444;'>UNFAIR</span>&nbsp;prejudiced framing, slurs, or loaded language</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a855f7;'>●</span>"
            f"<span><span style='{_TBP};color:#a855f7;'>STEREO</span>&nbsp;stereotype: fixed cultural or social assumption</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Compare profiles across tokens to spot which part of the sentence drives the overall score and whether multiple bias types co-occur.</div>"
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_strip_help, style="border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_strip", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_viz(res_B, selected_idx_B)), manual_header=man_header, help_text=_strip_help, style="border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_strip_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_strip_help,
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

        selected_idx = _read_sel(input.bias_selected_tokens_A)
        selected_idx_B = _read_sel(input.bias_selected_tokens_B)
        # Compare-models: always derive B from A by token content
        if compare_models and not compare_prompts and selected_idx:
            selected_idx_B = _map_sel_a_to_b(selected_idx, res, res_B)

        def get_viz(data, sel_idx=None):
            try:
                return create_confidence_breakdown(data["token_labels"], selected_token_idx=sel_idx)
            except Exception as e:
                _logger.debug("Suppressed: %s", e)
                return f'<div style="color:red">Error: {e}</div>'

        man_header = (
            "Confidence Breakdown",
            "Biased tokens grouped by confidence tier: Low (0.50–0.70), Medium (0.70–0.85), and High (0.85+).",
        )
        _confidence_help = (
            f"<span style='{_TH}'>What this shows: GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Detected bias spans grouped into three confidence tiers based on GUS-Net's softmax probability for the predicted bias class</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Tiers</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBA};color:#22c55e;'>Low 0.50–0.70</span>&nbsp;marginal detection: figurative language, borderline phrasing, or domain terms</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#eab308;'>●</span>"
            f"<span><span style='{_TBB};color:#eab308;'>Medium 0.70–0.85</span>&nbsp;probable bias signal: likely, but not certain</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR};color:#ef4444;'>High 0.85+</span>&nbsp;strong model-confident signal: typically unambiguous bias indicators</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Low-confidence spans should be read cautiously because they may reflect context or style rather than bias</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>High-confidence spans are the primary evidence for bias presence in the text</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>Threshold (default 0.5): spans below this are suppressed entirely. Adjust via the toolbar to surface or hide borderline detections.</div>"
        )

        if (compare_models or compare_prompts) and res_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_confidence_help, style="border: 2px solid #3b82f6; height: 100%;"),
                _wrap_card(ui.HTML(get_viz(res_B, selected_idx_B)), manual_header=man_header, help_text=_confidence_help, style="border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _wrap_card(ui.HTML(get_viz(res, selected_idx)), manual_header=man_header, help_text=_confidence_help)

    @output
    @render.ui
    def combined_bias_view():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;text-align:center;">No analysis results yet.</div>')

        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except Exception: l_idx, h_idx = 0, 0

        sel = _read_sel(input.bias_selected_tokens_A)
        sel_B = _read_sel(input.bias_selected_tokens_B)

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # Compare-models: always derive B from A by token content
        if compare_models and not compare_prompts and sel:
            sel_B = _map_sel_a_to_b(sel, res, res_B)

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = None
        if src_mode == "compare":
            res_base_for_compare = _get_base_resolved(res, _base_attn_cache)
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        def get_viz(data, s_idxs, container_id="bias-combined-container",
                    other_data=None, delta_label="A \u2212 B"):
            atts = data["attentions"]
            if not atts or l_idx >= len(atts): return '<div style="color:#9ca3af;">No attention data.</div>'
            try:
                if l_idx >= len(atts): return '<div style="color:#9ca3af;">Layer out of bounds.</div>'
                attn = atts[l_idx][0, h_idx].cpu().numpy()
                attn_other = None
                if other_data:
                    try:
                        o_atts = other_data["attentions"]
                        if o_atts and l_idx < len(o_atts):
                            attn_other = o_atts[l_idx][0, h_idx].cpu().numpy()
                    except Exception:
                        attn_other = None
                fig = create_combined_bias_visualization(
                    data["tokens"], data["token_labels"], attn, l_idx, h_idx,
                    selected_token_idx=s_idxs,
                    attention_matrix_other=attn_other,
                    delta_label=delta_label,
                )
                return _deferred_plotly(fig, container_id, height="600px")
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        man_header = (
            f"Combined Attention & Bias View{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Attention weight matrix for one head, with pink highlights on biased token rows/columns.",
        )
        _combined_help = (
            f"<span style='{_TH}'>What this shows: Attention × GUS-Net (Powers et al., 2024)</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span>Full token × token attention weight matrix for one (layer, head), with pink highlights on rows and columns that GUS-Net flagged as biased</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Reading the heatmap</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▪</span>"
            f"<span>Cell <span style='{_TC}'>(i, j)</span>: attention weight from query token <i>i</i> to key token <i>j</i></span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>▪</span>"
            f"<span><span style='{_TBR}'>Pink row</span>&nbsp;token <i>i</i> is biased; its query attends across the full sequence</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▪</span>"
            f"<span><span style='{_TBP}'>Pink col</span>&nbsp;token <i>j</i> is biased; it receives attention from the full sequence</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>▪</span>"
            f"<span><span style='{_TBA}'>Intersection</span>&nbsp;biased token attending to another biased token; possible semantic coupling or co-reference</span></div>"
            f"<hr style='{_TS}'/>"
            f"<span style='{_TH}'>Toolbar controls</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
            f"<span>Change <b>layer</b> and <b>head</b> to explore different specialisation profiles</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>▪</span>"
            f"<span>Click a token in <i>Detected Bias Tokens</i> to set it as the query focus</span></div>"
            f"<hr style='{_TS}'/>"
            f"<div style='{_TN}'>High weights at two pink-cell intersections suggest the model is semantically linking two biased spans, which can indicate a reinforcement mechanism.</div>"
        )
        card_style = "margin-bottom: 24px; box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        # res = GUS-Net attentions (from heavy_bias_compute via _GUSNET_TO_ENCODER)
        # res_base_for_compare = original pretrained base model attentions
        if src_mode == "compare" and res_base_for_compare:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, sel, "bias-combined-container-base",
                                          other_data=res, delta_label="Base \u2212 GUS")),
                           manual_header=(f"Combined Attention & Bias View{_source_badge_html('Base Encoder')}",
                                          man_header[1]),
                           help_text=_combined_help,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-base', 'bias_combined_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, None, "bias-combined-container-gus",
                                          other_data=res_base_for_compare, delta_label="GUS \u2212 Base")),
                           manual_header=(f"Combined Attention & Bias View{_source_badge_html('GUS-Net')}",
                                          man_header[1]),
                           help_text=_combined_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-gus', 'bias_combined_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_render, sel, "bias-combined-container",
                                          other_data=res_render_B, delta_label="A \u2212 B")),
                           manual_header=man_header, help_text=_combined_help,
                           style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container', 'bias_combined_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_render_B, sel_B, "bias-combined-container-B",
                                          other_data=res_render, delta_label="B \u2212 A")),
                           manual_header=man_header, help_text=_combined_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_combined_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-combined-container-B', 'bias_combined_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res_render, sel, "bias-combined-container")), manual_header=man_header, help_text=_combined_help, style=card_style,
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

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = _get_base_resolved(res, _base_attn_cache) if src_mode == "compare" else None
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        # Analyzer instance for metrics
        analyzer = AttentionBiasAnalyzer()

        def get_viz(data, container_id="bias-matrix-container",
                    other_data=None, delta_label="A \u2212 B"):
            try:
                if "attentions" not in data or not data["attentions"]:
                     return '<div style="color:#9ca3af;">No attention data.</div>'

                attentions = data["attentions"]
                biased_indices = [l["index"] for l in data["token_labels"] if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
                matrix = analyzer.create_attention_bias_matrix(attentions, biased_indices)
                metrics = data.get("attention_metrics")
                try: sl = int(input.bias_attn_layer())
                except Exception: sl = None

                try: _bar_th = float(input.bias_bar_threshold())
                except Exception: _bar_th = 1.5

                matrix_other = None
                if other_data:
                    try:
                        o_att = other_data.get("attentions")
                        o_bi = [l["index"] for l in other_data["token_labels"] if l.get("is_biased") and l["token"] not in ("[CLS]","[SEP]","[PAD]")]
                        if o_att:
                            matrix_other = analyzer.create_attention_bias_matrix(o_att, o_bi)
                    except Exception:
                        matrix_other = None

                fig = create_attention_bias_matrix(
                    matrix, metrics=metrics, selected_layer=sl, bar_threshold=_bar_th,
                    bias_matrix_other=matrix_other, delta_label=delta_label,
                )
                return _deferred_plotly(fig, container_id, height="600px")
            except Exception as e: return f'<div style="color:red">Error: {_html.escape(str(e))}</div>'

        _bar_help = (
            f"<span style='{_TH}'>Formula</span>"
            f"<div style='{_TR};font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;'>"
            f"BAR(l,h) = mean_attn_biased / (n_biased / n_tokens)</div>"
            f"<div style='{_TN}; margin-top:2px; margin-bottom:4px;'>observed share / expected under uniform distribution</div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Interpretation</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ffffff;'>●</span>"
            f"<span><span style='{_TBA};color:#ffffff;'>BAR = 1.0</span>&nbsp;White, uniform attention, no bias focus</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>●</span>"
            f"<span><span style='{_TBR}'>BAR &gt; 1.5</span>&nbsp;head <b>over-attends</b> to biased tokens</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>BAR &lt; 1.0</span>&nbsp;head <b>avoids</b> biased tokens</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Red cells (BAR >= threshold) = candidate bias-specialised heads. Cross-reference with ablation to confirm causal impact.</div>"
        )
        header_args = (
            f"Bias Attention Matrix{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Each cell shows the Bias Attention Ratio (BAR) for one (layer, head), indicating how much that head over- or under-attends to biased tokens.",
            _bar_help,
        )
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare" and res_base_for_compare:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, "bias-matrix-container-base",
                                          other_data=res, delta_label="Base \u2212 GUS")),
                           f"Bias Attention Matrix{_source_badge_html('Base Encoder')}",
                           header_args[1], _bar_help,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-base', 'bias_matrix_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, "bias-matrix-container-gus",
                                          other_data=res_base_for_compare, delta_label="GUS \u2212 Base")),
                           f"Bias Attention Matrix{_source_badge_html('GUS-Net')}",
                           header_args[1], _bar_help,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-gus', 'bias_matrix_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_render, "bias-matrix-container",
                                          other_data=res_render_B, delta_label="A \u2212 B")),
                           *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix_A')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res_render_B, "bias-matrix-container-B",
                                          other_data=res_render, delta_label="B \u2212 A")),
                           *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_matrix_B", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container-B', 'bias_matrix_B')", style=_BTN_STYLE_PNG),
                           ])
            )
        return _wrap_card(ui.HTML(get_viz(res_render, "bias-matrix-container")), *header_args, style=card_style,
                          controls=[
                              ui.download_button("export_bias_matrix", "CSV", style=_BTN_STYLE_CSV),
                              ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-matrix-container', 'bias_matrix')", style=_BTN_STYLE_PNG),
                          ])

    # ── Propagation drilldown: click layer → per-head view ──
    @reactive.effect
    @reactive.event(input.propagation_layer_click)
    def _on_propagation_layer_click():
        val = input.propagation_layer_click()
        try:
            propagation_drilldown_layer.set(int(val))
        except (TypeError, ValueError):
            pass

    @reactive.effect
    @reactive.event(input.propagation_back)
    def _on_propagation_back():
        propagation_drilldown_layer.set(None)

    # Reset drilldown when new bias results arrive
    @reactive.effect
    @reactive.event(bias_results)
    def _reset_propagation_drilldown():
        propagation_drilldown_layer.set(None)

    @output
    @render.ui
    def bias_propagation_plot():
        res = bias_results.get()
        if not res: return ui.HTML('<div style="color:#9ca3af;padding:20px;">No results.</div>')

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()
        drilldown_layer = propagation_drilldown_layer.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_base_for_compare = _get_base_resolved(res, _base_attn_cache) if src_mode == "compare" else None
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        try: l_idx = int(input.bias_attn_layer())
        except Exception: l_idx = None

        def _get_head_bars(data, layer):
            """Extract per-head BAR values for a given layer."""
            metrics = data.get("attention_metrics", [])
            heads = [m for m in metrics if m.layer == layer]
            heads.sort(key=lambda m: m.head)
            return [m.bias_attention_ratio for m in heads]

        def get_viz(data, container_id="bias-propagation-container"):
            p = data["propagation_analysis"]["layer_propagation"]
            if not p: return "No data."
            fig = create_bias_propagation_plot(p, selected_layer=l_idx)
            return _deferred_plotly(fig, container_id, height="450px",
                                   click_input="propagation_layer_click")

        def get_heads_viz(data, layer, container_id="bias-propagation-heads-container"):
            head_bars = _get_head_bars(data, layer)
            if not head_bars: return "No per-head data for this layer."
            fig = create_bias_propagation_heads_plot(head_bars, layer)
            return _deferred_plotly(fig, container_id, height="450px")

        # ── Back button for drilldown ──
        back_btn = ui.tags.button(
            ui.HTML('<i class="fa-solid fa-arrow-left" style="margin-right:4px;"></i> Layers'),
            onclick="Shiny.setInputValue('propagation_back', Date.now(), {priority:'event'});",
            style=(
                "background:transparent;border:1px solid #94a3b8;color:#94a3b8;"
                "border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;"
                "font-weight:600;transition:all 0.2s;"
            ),
        )

        _layer_header_args = (
            f"Bias Propagation Across Layers{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
            "Mean BAR per transformer layer. Click a layer to drill down into its heads.",
            f"<span style='{_TH}'>Layer depth</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>Early (0–3)</span>&nbsp;syntactic or surface-focused; BAR is usually near neutral</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>Middle (4–8)</span>&nbsp;semantic associations; bias often peaks here</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>●</span>"
            f"<span><span style='{_TBP}'>Late (9–11)</span>&nbsp;task-specific or abstract; signal may consolidate or diffuse</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>Curve shapes</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ef4444;'>▲</span>"
            f"<span><b>Rising</b>: bias behaves like a learned semantic feature and deepens with depth</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#94a3b8;'>▬</span>"
            f"<span><b>Flat</b>: the signal is fairly uniform and not strongly layer-specific</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>▼</span>"
            f"<span><b>Peak then drop</b>: bias is built in middle layers and then refined or diluted later</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Click a layer point to see per-head BAR breakdown. Reference dashed line = BAR 1.0 (uniform).</div>"
        )

        _heads_tooltip = (
            f"<span style='{_TH}'>Per-head breakdown</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#ff5ca9;'>●</span>"
            f"<span>Each bar is one attention head's BAR value</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#dc2626;'>●</span>"
            f"<span><span style='{_TBR}'>Red bars</span>&nbsp;BAR &gt; 1.5 = bias-specialised head</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#ea580c;'>●</span>"
            f"<span><span style='{_TBA}'>Orange bars</span>&nbsp;BAR 1.2–1.5 = moderate bias focus</span></div>"
            f"<hr style='{_TS}'>"
            f"<div style='{_TN}'>Click ← Layers to return to the layer overview.</div>"
        )

        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"

        # ── Helper: render single-panel (layers or heads) ──
        def _single_panel(data, suffix="", extra_style=""):
            cid_layers = f"bias-propagation-container{suffix}"
            cid_heads = f"bias-propagation-heads-container{suffix}"
            if drilldown_layer is not None:
                hdr = (
                    f"Bias Propagation — Layer {drilldown_layer} Heads{_source_badge_html(src_label) if src_mode != 'compare' else ''}",
                    f"BAR per attention head in layer {drilldown_layer}.",
                    _heads_tooltip,
                )
                ctrls = [
                    back_btn,
                    ui.tags.button("PNG", onclick=f"downloadPlotlyPNG('{cid_heads}', 'bias_propagation_heads_L{drilldown_layer}')", style=_BTN_STYLE_PNG),
                ]
                return _wrap_card(ui.HTML(get_heads_viz(data, drilldown_layer, cid_heads)),
                                  *hdr, style=card_style + extra_style, controls=ctrls)
            else:
                ctrls = [
                    ui.download_button(f"export_bias_propagation{suffix}", "CSV", style=_BTN_STYLE_CSV),
                    ui.tags.button("PNG", onclick=f"downloadPlotlyPNG('{cid_layers}', 'bias_propagation{suffix}')", style=_BTN_STYLE_PNG),
                ]
                return _wrap_card(ui.HTML(get_viz(data, cid_layers)),
                                  *_layer_header_args, style=card_style + extra_style, controls=ctrls)

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare" and res_base_for_compare:
            hdr_base = (f"Bias Propagation{_source_badge_html('Base Encoder')}", _layer_header_args[1], _layer_header_args[2])
            hdr_gus = (f"Bias Propagation{_source_badge_html('GUS-Net')}", _layer_header_args[1], _layer_header_args[2])
            if drilldown_layer is not None:
                hdr_base = (f"Layer {drilldown_layer} Heads{_source_badge_html('Base Encoder')}",
                            f"BAR per head in layer {drilldown_layer}.", _heads_tooltip)
                hdr_gus = (f"Layer {drilldown_layer} Heads{_source_badge_html('GUS-Net')}",
                           f"BAR per head in layer {drilldown_layer}.", _heads_tooltip)
                return ui.div(
                    {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                    _wrap_card(ui.HTML(get_heads_viz(res_base_for_compare, drilldown_layer, "bias-propagation-heads-base")),
                               *hdr_base, style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                               controls=[back_btn, ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-heads-base', 'bias_propagation_heads_base')", style=_BTN_STYLE_PNG)]),
                    _wrap_card(ui.HTML(get_heads_viz(res, drilldown_layer, "bias-propagation-heads-gus")),
                               *hdr_gus, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                               controls=[ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-heads-gus', 'bias_propagation_heads_gus')", style=_BTN_STYLE_PNG)]),
                )
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_viz(res_base_for_compare, "bias-propagation-container-base")), *hdr_base,
                           style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                           controls=[
                               ui.download_button("export_bias_propagation", "CSV", style=_BTN_STYLE_CSV),
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container-base', 'bias_propagation_base')", style=_BTN_STYLE_PNG),
                           ]),
                _wrap_card(ui.HTML(get_viz(res, "bias-propagation-container-gus")), *hdr_gus,
                           style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[
                               ui.tags.button("PNG", onclick="downloadPlotlyPNG('bias-propagation-container-gus', 'bias_propagation_gus')", style=_BTN_STYLE_PNG),
                           ])
            )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _single_panel(res_render, "", " border: 2px solid #3b82f6; height: 100%;"),
                _single_panel(res_render_B, "_B", " border: 2px solid #ff5ca9; height: 100%;"),
            )
        return _single_panel(res_render)

    @output
    @render.ui
    def bias_focused_heads_table():
        res = bias_results.get()
        if not res: return None

        compare_models = active_bias_compare_models.get()
        compare_prompts = active_bias_compare_prompts.get()
        res_B = bias_results_B.get()

        # ── Attention source resolution ──
        src_mode = _normalize_attn_source_mode(_get_attn_source_mode("bias_attn_source"))
        res_render, src_label = _resolve_source_for_render(res, _base_attn_cache, src_mode)
        res_render_B = None
        if res_B:
            res_render_B, _ = _resolve_source_for_render(res_B, _base_attn_cache_B, src_mode)

        try: l_idx, h_idx = int(input.bias_attn_layer()), int(input.bias_attn_head())
        except Exception: l_idx, h_idx = -1, -1

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

        _heads_help = (
            f"<span style='{_TH}'>Columns</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#60a5fa;'>●</span>"
            f"<span><span style='{_TBB}'>Layer / Head</span>&nbsp;transformer block + head index (0-indexed)</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#f59e0b;'>●</span>"
            f"<span><span style='{_TBA}'>BAR</span>&nbsp;observed / expected attention - 1.0 = uniform</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#22c55e;'>●</span>"
            f"<span><span style='{_TBG}'>green</span>&nbsp;BAR exceeds specialisation threshold</span></div>"
            f"<hr style='{_TS}'>"
            f"<span style='{_TH}'>How to use</span>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Click a row to highlight that head in all visualisations</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Run <b>Head Ablation</b> to test causal impact</span></div>"
            f"<div style='{_TR}'><span style='{_TD};color:#a78bfa;'>▶</span>"
            f"<span>Run <b>Integrated Gradients</b> to check faithfulness</span></div>"
        )
        _heads_subtitle = f"Ranked by Bias Attention Ratio (BAR). Green dot = above specialisation threshold ({bar_threshold:.1f})."
        card_style = "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 16px;"

        # ── Compare attention sources (Base Encoder vs GUS-Net) ──
        if src_mode == "compare":
            res_base_for_compare = _get_base_resolved(res, _base_attn_cache)
            if res_base_for_compare:
                return ui.div(
                    {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                    _wrap_card(ui.HTML(get_table(res_base_for_compare)),
                               manual_header=(f"Top {k} Attention Heads by Bias Focus{_source_badge_html('Base Encoder')}", _heads_subtitle),
                               help_text=_heads_help,
                               style=card_style + " border: 2px solid #60a5fa; height: 100%;",
                               controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)]),
                    _wrap_card(ui.HTML(get_table(res)),
                               manual_header=(f"Top {k} Attention Heads by Bias Focus{_source_badge_html('GUS-Net')}", _heads_subtitle),
                               help_text=_heads_help,
                               style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                               controls=[ui.download_button("export_bias_top_heads_B", "CSV", style=_BTN_STYLE_CSV)])
                )

        header_args = (
            f"Top {k} Attention Heads by Bias Focus{_source_badge_html(src_label)}",
            _heads_subtitle,
            _heads_help,
        )

        if (compare_models or compare_prompts) and res_render_B:
            return ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 24px;"},
                _wrap_card(ui.HTML(get_table(res_render)), *header_args, style=card_style + " border: 2px solid #3b82f6; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)]),
                _wrap_card(ui.HTML(get_table(res_render_B)), *header_args, style=card_style + " border: 2px solid #ff5ca9; height: 100%;",
                           controls=[ui.download_button("export_bias_top_heads_B", "CSV", style=_BTN_STYLE_CSV)])
            )
        return _wrap_card(ui.HTML(get_table(res_render)), *header_args, style=card_style,
                          controls=[ui.download_button("export_bias_top_heads", "CSV", style=_BTN_STYLE_CSV)])


    # ── Ablation handlers ─────────────────────────────────────────────

    @reactive.effect
    async def compute_ablation():
        """Automatically run ablation when bias analysis is complete."""
        res_A, res_B, show_comparison = _resolve_faithfulness_results()
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
            args_B = _compute_single(res_B) if show_comparison and res_B else None

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

        except Exception:
            _logger.exception("Ablation error")
        finally:
            ablation_running.set(False)

    # ── XAI handlers (extracted to bias_xai.py) ────────────────────────
    register_xai_handlers(
        input=input, output=output,
        ablation_results=ablation_results,
        ablation_results_B=ablation_results_B,
        ablation_running=ablation_running,
        ig_results=ig_results,
        ig_results_B=ig_results_B,
        ig_running=ig_running,
        perturbation_results=perturbation_results,
        perturbation_results_B=perturbation_results_B,
        perturbation_running=perturbation_running,
        lrp_results=lrp_results,
        lrp_results_B=lrp_results_B,
        lrp_running=lrp_running,
        active_bias_compare_models=active_bias_compare_models,
        active_bias_compare_prompts=active_bias_compare_prompts,
        bias_results_B_rv=bias_results_B,
        _get_attn_source_mode=_get_attn_source_mode,
        _get_bias_model_label=_get_bias_model_label,
        _resolve_faithfulness_results=_resolve_faithfulness_results,
    )

    # ── StereoSet handlers (extracted to bias_stereoset.py) ──────────
    register_stereoset_handlers(
        input=input, output=output,
        bias_results=bias_results,
        bias_results_B=bias_results_B,
        active_bias_compare_models=active_bias_compare_models,
        active_bias_compare_prompts=active_bias_compare_prompts,
        ig_results=ig_results,
        ig_results_B=ig_results_B,
        lrp_results=lrp_results,
        lrp_results_B=lrp_results_B,
        perturbation_results=perturbation_results,
        perturbation_results_B=perturbation_results_B,
        _get_attn_source_mode=_get_attn_source_mode,
    )

    # Replaced block — original ablation_results_display was here
    # (see bias_xai.py and bias_stereoset.py for extracted code)

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_label(data, is_A, compare_m):
        if compare_m: return _get_bias_model_label(data)
        return "Prompt A" if is_A else "Prompt B"

__all__ = ["bias_server_handlers"]
