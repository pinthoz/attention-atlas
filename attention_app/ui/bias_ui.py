"""UI components for Bias Analysis tab."""

from shiny import ui


def create_bias_sidebar():
    """Create sidebar for bias analysis - matching Attention Atlas styling."""
    from .components import ICON_DATA_URL

    return ui.div(
        {"class": "sidebar"},
        ui.div(
            {"class": "app-title", "style": "display: flex; align-items: center; gap: 8px;"},
            ui.tags.img(src=ICON_DATA_URL or "/favicon.ico", alt="Logo"),
            ui.h3("Attention Atlas"),
        ),
        ui.div(
            {"class": "app-subtitle", "style": "margin-bottom: 12px; padding-bottom: 12px;"},
            "Detect and analyze social bias in text using GUS-Net neural NER."
        ),

        # ── Configuration Sections ──
        ui.div(
            {"class": "sidebar-section", "style": "margin-top: 4px;"},

            ui.tags.span("Sensitivity Threshold", class_="sidebar-label"),
            ui.div(
                {"style": "padding: 0 4px;"},
                ui.input_slider(
                    "bias_threshold",
                    None,
                    min=0.1, max=0.9, value=0.5, step=0.05,
                    width="100%",
                ),
            )
        ),

        # ── Input Text Area ──
        ui.div(
            {"class": "sidebar-section", "style": "margin-top: auto; padding-top: 16px;"},
            ui.tags.span("Input Text", class_="sidebar-label"),
            
            ui.div(
                {"class": "custom-input-container", "id": "bias-input-container"},
                ui.div(
                    {"class": "tabs-row"},
                    ui.div(
                        {"class": "history-tab", "onclick": "toggleBiasHistory()", "title": "History"},
                        ui.HTML("""<svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 448 512" fill="white"><path d="M0 96C0 78.3 14.3 64 32 64H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32s14.3-32 32-32H416c17.7 0 32 14.3 32 32z"/></svg>"""),
                    ),
                    ui.div(
                        {"class": "session-controls", "style": "display: flex; gap: 4px; align-items: flex-end; margin-left: auto;"},
                        ui.tags.label(ui.HTML('<i class="fa-solid fa-folder-open"></i>'), {"class": "session-btn-custom", "title": "Load Session", "for": "load_bias_session_upload"}),
                        ui.div(ui.input_file("load_bias_session_upload", None, accept=[".json"], multiple=False), style="display: none;"),
                        ui.download_button("save_bias_session", label=None, icon=ui.tags.i(class_="fa-solid fa-floppy-disk"), class_="session-btn-custom"),
                    ),
                ),
                ui.div({"id": "bias-history-dropdown", "class": "history-dropdown"}, ui.output_ui("bias_history_list")),
                ui.tags.textarea("All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.", id="bias_input_text", class_="custom-textarea", rows=6, oninput="Shiny.setInputValue('bias_input_text', this.value, {priority: 'event'})"),
            ),
            
            ui.div(
                {"style": "margin-top: 12px;"},
                ui.input_action_button("analyze_bias_btn", "Analyze Bias", class_="btn-primary", style="padding-top: 6px; padding-bottom: 6px; min-height: 0; height: auto;"),
            ),
            
            ui.tags.script("""
                function toggleBiasHistory() {
                    const dropdown = document.getElementById('bias-history-dropdown');
                    dropdown.classList.toggle('show');
                }
                function selectBiasHistoryItem(text) {
                    const textarea = document.getElementById('bias_input_text');
                    textarea.value = text;
                    Shiny.setInputValue('bias_input_text', text, {priority: 'event'});
                    document.getElementById('bias-history-dropdown').classList.remove('show');
                }
                document.addEventListener('click', function(event) {
                    const container = document.getElementById('bias-input-container');
                    const dropdown = document.getElementById('bias-history-dropdown');
                    if (container && !container.contains(event.target)) {
                        dropdown.classList.remove('show');
                    }
                });
            """)
        ),
    )


def create_bias_content():
    """Create main content area for bias analysis.

    The actual dashboard content is rendered dynamically by the server
    handler ``bias_dashboard_content`` so that accordion sections only
    appear after analysis.
    """
    return ui.div(
        {"class": "content bias-content", "style": "margin-top: 50px !important;"},

        # Accordion badge styles (shared with dynamic content)
        ui.tags.style("""
            .accordion-panel-badge {
                display: inline-flex !important;
                align-items: center !important;
                gap: 6px !important;
                font-size: 10px !important;
                font-weight: 500 !important;
                padding: 2px 8px !important;
                border-radius: 4px !important;
                margin-left: 10px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                vertical-align: middle !important;
            }
            .accordion-panel-badge.essential {
                background: rgba(34, 197, 94, 0.15) !important;
                color: #22c55e !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
            }
            .accordion-panel-badge.technical {
                background: rgba(139, 92, 246, 0.15) !important;
                color: #8b5cf6 !important;
                border: 1px solid rgba(139, 92, 246, 0.3) !important;
            }
            .accordion-panel-badge.explore {
                background: rgba(59, 130, 246, 0.15) !important;
                color: #3b82f6 !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
            }
            .accordion-button:not(.collapsed) .accordion-panel-badge {
                background: rgba(255, 255, 255, 0.2) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
            }
            /* Prevent attention tab's has-control-bar padding from leaking here */
            .bias-content.has-control-bar {
                padding-bottom: 0 !important;
            }
        """),

        ui.output_ui("bias_dashboard_content"),
    )


def create_bias_accordion():
    """Build the accordion panels for bias analysis (rendered after analysis)."""
    from .components import viz_header

    return ui.accordion(
        # Panel 1: Overview & Detection
        ui.accordion_panel(
            ui.span("Overview & Detection", ui.span({"class": "accordion-panel-badge essential"}, "Essential")),
            # Summary Card
            ui.div(
                {"class": "card", "style": "margin-bottom: 24px;"},
                viz_header(
                    "Bias Detection Summary",
                    "Composite bias level with explicit weighted criteria and per-category counts.",
                    "Level = weighted composite of token density (30%), generalizations (20%), unfair language (25%), and stereotypes (25%).",
                ),
                ui.output_ui("bias_summary"),
            ),
            # Detected Bias Tokens Card
            ui.div(
                {"class": "card"},
                ui.div(
                    {"style": "padding: 16px;"},
                    ui.h4("Detected Bias Tokens", style="font-size:16px; font-weight:700; color:#ff5ca9; margin-bottom:4px;"),
                    ui.p("Each biased token with its category labels and confidence scores.",
                         style="font-size:11px;color:#6b7280;margin-bottom:16px;"),
                    ui.output_ui("bias_spans_table"),
                    ui.hr(),
                    ui.output_ui("bias_method_info"),
                )
            ),
            value="overview"
        ),

        # Panel 2: Technical Analysis
        ui.accordion_panel(
            ui.span("Technical Analysis", ui.span({"class": "accordion-panel-badge technical"}, "Technical")),
            ui.div(
                {"class": "card"},
                ui.div(
                    {"class": "viz-header"},
                    ui.h4("Token-Level Bias Distribution", style="margin:0;"),
                    ui.p("Each token with per-category scores. Coloured dots and values show bias intensity across GEN, UNFAIR, and STEREO.",
                         style="font-size:11px;color:#6b7280;margin:4px 0 0;"),
                ),
                ui.output_ui("token_bias_strip"),
            ),
            value="technical"
        ),

        # Panel 3: Attention x Bias Correlation
        ui.accordion_panel(
            ui.span("Attention × Bias Correlation", ui.span({"class": "accordion-panel-badge explore"}, "Exploration")),
            # Formula definition panel
            ui.output_ui("bias_ratio_formula"),
            # Hidden selects (driven by the floating toolbar)
            ui.div(
                {"style": "display:none;"},
                ui.input_select("bias_attn_layer", "Layer", choices={}, selected="0"),
                ui.input_select("bias_attn_head", "Head", choices={}, selected="0"),
            ),
            # Combined Attention & Bias View (controlled by toolbar)
            ui.div(
                {"class": "card", "style": "margin-bottom: 24px; box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"},
                ui.div(
                    {"class": "viz-header"},
                    ui.h4("Combined Attention & Bias View", style="margin:0;"),
                    ui.p("Attention weights for a single head with pink row/column highlights on biased tokens. Use the toolbar to change layer and head.",
                         style="font-size:11px;color:#6b7280;margin:4px 0 0;"),
                ),
                ui.output_ui("combined_bias_view"),
            ),
            ui.layout_columns(
                ui.div(
                    {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"},
                    viz_header(
                        "Bias Attention Matrix",
                        "Each cell = ratio of attention a head pays to biased tokens vs. the uniform baseline.",
                        "Values centred on 1.0 (neutral). Red > 1.5 = head specializes on biased tokens. Blue < 1.0 = head avoids them.",
                    ),
                    ui.output_ui("attention_bias_matrix"),
                ),
                ui.div(
                    {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"},
                    viz_header(
                        "Bias Propagation Across Layers",
                        "Mean bias attention ratio per layer — how bias focus evolves through model depth.",
                        "The dashed line at 1.0 represents neutral attention. Values above indicate increasing bias focus.",
                    ),
                    ui.output_ui("bias_propagation_plot"),
                ),
                col_widths=[6, 6],
            ),
            ui.div(
                {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 16px;"},
                ui.h5("Top Attention Heads by Bias Focus", style="color:#ff5ca9; margin-bottom:12px; font-weight: 700;"),
                ui.p("Top 5 heads ranked by bias attention ratio. Green dot = above specialization threshold (1.5).", style="font-size:11px;color:#6b7280;margin-bottom:12px;"),
                ui.output_ui("bias_focused_heads_table"),
            ),
            value="attention_bias"
        ),
        id="bias_accordion",
        open="overview",
        multiple=True,
    )


def create_floating_bias_toolbar():
    """Create a floating bottom toolbar matching the attention tab's design exactly."""
    return ui.div(
        # Main Bar — reuses .floating-control-bar from styles.py
        ui.div(
            {"class": "floating-control-bar", "id": "bias-floating-toolbar"},

            ui.span("Bias Toolkit", class_="bar-title"),

            ui.div(
                {"class": "controls-row"},

                # ── Layer slider ──
                ui.div(
                    {"class": "control-group"},
                    ui.span("Layer", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span("0", id="bias-layer-value", class_="slider-value"),
                        ui.tags.input(type="range", id="bias-layer-slider", min="0", max="11", value="0", step="1"),
                    ),
                ),

                # ── Head slider ──
                ui.div(
                    {"class": "control-group"},
                    ui.span("Head", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span("0", id="bias-head-value", class_="slider-value"),
                        ui.tags.input(type="range", id="bias-head-slider", min="0", max="11", value="0", step="1"),
                    ),
                ),

                # ── Divider ──
                ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),

                # ── Top-K ──
                ui.div(
                    {"class": "control-group"},
                    ui.span("Top-K", class_="control-label"),
                    ui.div(
                        {"class": "slider-container"},
                        ui.tags.span("5", id="bias-topk-value", class_="slider-value"),
                        ui.tags.input(type="range", id="bias-topk-slider", min="1", max="10", value="5", step="1"),
                    ),
                ),

                # ── Top Focus Heads chips ──
                ui.div(
                    {"class": "control-group"},
                    ui.span("Focus Heads", class_="control-label"),
                    ui.div(
                        {"style": "display:flex;gap:4px;align-items:center;"},
                        ui.output_ui("bias_toolkit_heads"),
                    ),
                ),

                # ── Divider ──
                ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),

                # ── Detected Spans toggle ──
                ui.div(
                    {"class": "control-group"},
                    ui.span("Spans", class_="control-label"),
                    ui.tags.button(
                        "Detected ",
                        ui.span({"id": "bias-span-count-badge", "style": "background:rgba(0,0,0,0.3);font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px;"}, "0"),
                        type="button",
                        class_="btn-global",
                        style="height:20px;padding:0 8px;font-size:10px;border:1px solid #ff5ca9;color:#ff5ca9;background:rgba(255,92,169,0.1);",
                        onclick="toggleBiasSpans()",
                    ),
                ),
            ),

            # ── Popover: Detected Spans ──
            ui.div(
                {"id": "bias-spans-popover", "class": "bias-popover"},
                ui.div(
                    {"class": "popover-header"},
                    ui.span("Detected Bias Spans", style="font-weight:600;font-size:12px;color:white;"),
                    ui.tags.i(class_="fa-solid fa-xmark", style="cursor:pointer;color:#94a3b8;", onclick="toggleBiasSpans()"),
                ),
                ui.div(
                    {"class": "popover-content"},
                    ui.output_ui("bias_toolkit_spans"),
                ),
            ),
        ),

        # ── Extra styles (only for bias-specific additions) ──
        ui.tags.style("""
            #bias-floating-toolbar {
                animation: biasBarSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            }
            @keyframes biasBarSlideUp {
                from { transform: translateY(100px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .bias-head-chip {
                font-size: 10px;
                font-family: 'JetBrains Mono', monospace;
                padding: 2px 8px;
                border-radius: 4px;
                background: rgba(255, 92, 169, 0.15);
                border: 1px solid rgba(255, 92, 169, 0.3);
                color: #ff5ca9;
                cursor: pointer;
                transition: all 0.15s ease;
                white-space: nowrap;
            }
            .bias-head-chip:hover {
                background: rgba(255, 92, 169, 0.3);
                transform: translateY(-1px);
            }
            .bias-popover {
                position: absolute;
                bottom: 70px;
                right: 20px;
                width: 300px;
                max-height: 380px;
                background: var(--sidebar-bg);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                box-shadow: 0 -10px 40px rgba(0,0,0,0.5);
                display: none;
                flex-direction: column;
                overflow: hidden;
            }
            .bias-popover.visible { display: flex; }
            .popover-header {
                padding: 10px 14px;
                background: rgba(255,255,255,0.03);
                border-bottom: 1px solid rgba(255,255,255,0.05);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .popover-content {
                padding: 0;
                overflow-y: auto;
                max-height: 320px;
            }
            .popover-content::-webkit-scrollbar { width: 4px; }
            .popover-content::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
            .bias-span-item {
                padding: 8px 14px;
                border-bottom: 1px solid rgba(255,255,255,0.05);
                transition: background 0.15s;
            }
            .bias-span-item:last-child { border-bottom: none; }
            .bias-span-item:hover { background: rgba(255,255,255,0.05); }
        """),

        # ── Script ──
        ui.tags.script("""
        (function() {
            // ── Debounced Shiny setters ──
            function debounce(fn, ms) {
                var t; return function() { var a=arguments,c=this; clearTimeout(t); t=setTimeout(function(){fn.apply(c,a);},ms); };
            }
            function _setSelectize(id, val) {
                var el = document.getElementById(id);
                if (!el) return;
                var s = el.selectize || ($(el)[0] && $(el)[0].selectize);
                if (s) s.setValue(val.toString());
                else { el.value = val.toString(); $(el).trigger('change'); }
            }
            var setLayer = debounce(function(v){ _setSelectize('bias_attn_layer', v); }, 200);
            var setHead  = debounce(function(v){ _setSelectize('bias_attn_head', v); }, 200);
            var setTopK  = debounce(function(v){ Shiny.setInputValue('bias_top_k', parseInt(v), {priority:'event'}); }, 200);

            // ── Bind sliders ──
            function bindSlider(sliderId, valId, setter) {
                var el = document.getElementById(sliderId);
                if (el) el.oninput = function() {
                    document.getElementById(valId).textContent = this.value;
                    setter(this.value);
                };
            }
            bindSlider('bias-layer-slider', 'bias-layer-value', setLayer);
            bindSlider('bias-head-slider', 'bias-head-value', setHead);
            bindSlider('bias-topk-slider', 'bias-topk-value', setTopK);

            // ── Sync slider ranges from actual model ──
            setTimeout(function() {
                var lEl = document.getElementById('bias_attn_layer');
                var hEl = document.getElementById('bias_attn_head');
                var lSlider = document.getElementById('bias-layer-slider');
                var hSlider = document.getElementById('bias-head-slider');
                if (lEl && lEl.options && lSlider) {
                    lSlider.max = lEl.options.length - 1;
                    lSlider.value = lEl.value || '0';
                    document.getElementById('bias-layer-value').textContent = lEl.value || '0';
                }
                if (hEl && hEl.options && hSlider) {
                    hSlider.max = hEl.options.length - 1;
                    hSlider.value = hEl.value || '0';
                    document.getElementById('bias-head-value').textContent = hEl.value || '0';
                }
            }, 500);

            // ── setBiasHead (from top-K chips) ──
            window.setBiasHead = function(layer, head) {
                _setSelectize('bias_attn_layer', layer);
                _setSelectize('bias_attn_head', head);
                var ls = document.getElementById('bias-layer-slider');
                var hs = document.getElementById('bias-head-slider');
                if (ls) { ls.value = layer; document.getElementById('bias-layer-value').textContent = layer; }
                if (hs) { hs.value = head;  document.getElementById('bias-head-value').textContent = head; }
            };

            // ── Spans popover toggle ──
            window.toggleBiasSpans = function() {
                var pop = document.getElementById('bias-spans-popover');
                if (pop) pop.classList.toggle('visible');
            };

            // ── Sync on select changes (from server-side updates) ──
            $(document).on('change', '#bias_attn_layer', function() {
                var s = document.getElementById('bias-layer-slider');
                if (s) { s.value = this.value; document.getElementById('bias-layer-value').textContent = this.value; }
            });
            $(document).on('change', '#bias_attn_head', function() {
                var s = document.getElementById('bias-head-slider');
                if (s) { s.value = this.value; document.getElementById('bias-head-value').textContent = this.value; }
            });
        })();
        """),
    )


__all__ = ["create_bias_sidebar", "create_bias_content", "create_bias_accordion", "create_floating_bias_toolbar"]
