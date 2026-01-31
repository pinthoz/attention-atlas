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
            
            # 1. Detection Settings
            ui.tags.span("Detection Settings", class_="sidebar-label"),
            ui.tags.span("Method", class_="sidebar-label", style="margin-top: 10px; font-size: 11px; color:#94a3b8;"),
            ui.input_select(
                "bias_detection_method",
                None,
                choices={
                    "combined": "Combined (GUS-Net + Lexicon)",
                    "gusnet": "Neural (GUS-Net Model)",
                    "lexicon": "Lexicon (Fixed Dictionary)",
                },
                selected="combined",
                width="100%",
            ),
            
            ui.tags.span("Sensitivity Threshold", class_="sidebar-label", style="margin-top: 15px; font-size: 11px; color:#94a3b8;"),
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
    """Create main content area for bias analysis - Harmonica / Accordion Style."""
    from .components import viz_header, mini_select
    
    return ui.div(
        {"class": "content", "style": "margin-top: 50px !important;"},

        # Explicit styles for this tab to ensure badges and spacing are perfect
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
        """),

        ui.accordion(
            # Panel 1: Overview & Detection
            ui.accordion_panel(
                ui.span("Overview & Detection", ui.span({"class": "accordion-panel-badge essential"}, "Essential")),
                # Summary Card
                ui.div(
                    {"class": "card", "style": "margin-bottom: 24px;"},
                    viz_header(
                        "Bias Detection Summary",
                        "High-level overview of detected bias categories and their confidence scores.",
                        "Bias scores are computed by weighting the outputs of three GUS-Net specialized domains (GEN, UNFAIR, STEREO).",
                    ),
                    ui.output_ui("bias_summary"),
                ),
                # Detection View Card
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"style": "padding: 16px;"},
                        ui.output_ui("inline_bias_view"),
                        ui.hr(),
                        ui.h4("Detected Bias Spans", style="font-size:16px; font-weight:700; color:#ff5ca9; margin-bottom:16px;"),
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
                    {"class": "card", "style": "margin-bottom: 24px;"},
                    ui.div(
                        {"class": "viz-header"},
                        ui.h4("Token-Level Bias Distribution", style="margin:0;"),
                        ui.p("Binary classification heatmap showing active bias categories per token.", style="font-size:11px;color:#6b7280;margin:4px 0 0;"),
                    ),
                    ui.output_ui("token_bias_viz"),
                ),
                ui.div(
                    {"class": "card"},
                    ui.div(
                        {"class": "header-with-selectors"},
                        ui.h4("Combined Attention & Bias View"),
                        ui.div(
                            {"class": "selection-boxes-container"},
                            ui.div({"class": "selection-box"}, ui.input_select("bias_attn_layer", "Layer", choices={}, selected="0")),
                            ui.div({"class": "selection-box"}, ui.input_select("bias_attn_head", "Head", choices={}, selected="0")),
                        ),
                    ),
                    ui.p("Overlays attention weights with bias detection to see if model attends to controversial tokens.", style="font-size:11px;color:#6b7280;margin-bottom:12px;"),
                    ui.output_ui("combined_bias_view"),
                ),
                value="technical"
            ),

            # Panel 3: Attention x Bias Correlation
            ui.accordion_panel(
                ui.span("Attention x Bias Correlation", ui.span({"class": "accordion-panel-badge explore"}, "Exploration")),
                ui.layout_columns(
                    ui.div(
                        {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"},
                        viz_header("Bias Attention Matrix", "Identify which attention heads pay the most attention to biased tokens.", "A matrix visualization where each cell represents the average attention a head pays to biased tokens relative to non-biased tokens."),
                        ui.output_ui("attention_bias_matrix"),
                    ),
                    ui.div(
                        {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05);"},
                        viz_header("Bias Propagation", "Track how attention to biased tokens evolves across model layers.", "A line plot aggregate showing the mean attention focus on controversial or biased spans at each depth of the network."),
                        ui.output_ui("bias_propagation_plot"),
                    ),
                    col_widths=[6, 6],
                ),
                ui.div(
                    {"class": "card", "style": "box-shadow: none; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 16px;"},
                    ui.h5("Bias-Focused Attention Heads", style="color:#ff5ca9; margin-bottom:12px; font-weight: 700;"),
                    ui.p("Heads that pay significantly more attention to biased tokens (ratio > 1.5)", style="font-size:11px;color:#6b7280;margin-bottom:12px;"),
                    ui.output_ui("bias_focused_heads_table"),
                ),
                value="attention_bias"
            ),
            id="bias_accordion",
            open="overview",
            multiple=True
        ),
    )


__all__ = ["create_bias_sidebar", "create_bias_content"]
