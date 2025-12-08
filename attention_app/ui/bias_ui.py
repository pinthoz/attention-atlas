"""UI components for Bias Analysis tab."""

from shiny import ui


def create_bias_sidebar():
    """Create sidebar for bias analysis."""
    from .components import ICON_DATA_URL

    return ui.div(
        {"class": "sidebar"},
        ui.div(
            {"class": "app-title"},
            ui.tags.img(src=ICON_DATA_URL or "/favicon.ico", alt="Logo"),
            ui.h3("Attention Atlas"),
        ),
        ui.div(
            {"class": "app-subtitle"},
            "Detect and analyze bias in transformer models at token and attention levels."
        ),

        ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("Input Text", class_="sidebar-label"),
            ui.input_text_area(
                "bias_input_text",
                None,
                "All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.",
                rows=6
            ),
            ui.div(
                ui.input_action_button("analyze_bias_btn", "Analyze Bias", class_="btn-primary"),
                ui.div(
                    {"id": "bias_loading_spinner", "class": "loading-container", "style": "display:none;"},
                    ui.div({"class": "spinner"}),
                    ui.span("Analyzing...")
                ),
            ),
        ),



        ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("ℹ️ About Bias Detection", class_="sidebar-label"),
            ui.tags.p(
                "This module combines two approaches:",
                style="font-size:12px;color:#cbd5e1;margin-bottom:8px;"
            ),
            ui.tags.ul(
                ui.tags.li("Token-Level Detection (GUS-Net)", style="font-size:11px;color:#cbd5e1;"),
                ui.tags.li("Attention × Bias Analysis", style="font-size:11px;color:#cbd5e1;"),
                style="margin:0;padding-left:20px;"
            ),
        ),
    )


def create_bias_content():
    """Create main content area for bias analysis."""
    return ui.div(
        {"class": "content"},

        # Bias Summary Card
        ui.div(
            {"class": "card"},
            ui.h4("Bias Detection Summary"),
            ui.output_ui("bias_summary")
        ),

        # Visualizations in tabs
        ui.navset_card_tab(
            ui.nav_panel(
                "Token-Level Bias",
                ui.div(
                    {"style": "padding: 16px;"},
                    ui.output_ui("token_bias_viz"),
                    ui.hr(),
                    ui.h5("Detected Bias Spans", style="color:#ff5ca9;"),
                    ui.output_ui("bias_spans_table")
                )
            ),

            ui.nav_panel(
                "Attention × Bias",
                ui.div(
                    {"style": "padding: 16px;"},
                    ui.layout_columns(
                        ui.div(
                            {"class": "card"},
                            ui.h4("Bias Attention Matrix"),
                            ui.p(
                                "Shows which attention heads focus on biased tokens",
                                style="font-size:11px;color:#6b7280;margin-bottom:12px;"
                            ),
                            ui.output_ui("attention_bias_matrix")
                        ),
                        ui.div(
                            {"class": "card"},
                            ui.h4("Bias Propagation"),
                            ui.p(
                                "How bias attention changes across layers",
                                style="font-size:11px;color:#6b7280;margin-bottom:12px;"
                            ),
                            ui.output_ui("bias_propagation_plot")
                        ),
                        col_widths=[6, 6]
                    ),
                    ui.hr(),
                    ui.div(
                        {"class": "card"},
                        ui.div(
                            {"class": "header-with-selectors"},
                            ui.h4("Combined Attention & Bias View"),
                            ui.div(
                                {"class": "selection-boxes-container"},
                                ui.div(
                                    {"class": "selection-box"},
                                    ui.input_select("bias_attn_layer", "Layer", choices={}, selected="0")
                                ),
                                ui.div(
                                    {"class": "selection-box"},
                                    ui.input_select("bias_attn_head", "Head", choices={}, selected="0")
                                ),
                            )
                        ),
                        ui.p(
                            "Attention heatmap with bias-highlighted tokens",
                            style="font-size:11px;color:#6b7280;margin-bottom:12px;"
                        ),
                        ui.output_ui("combined_bias_view")
                    ),
                    ui.hr(),
                    ui.div(
                        {"class": "card"},
                        ui.h5("Bias-Focused Attention Heads", style="color:#ff5ca9;margin-bottom:12px;"),
                        ui.p(
                            "Heads that pay significantly more attention to biased tokens (ratio > 1.5)",
                            style="font-size:11px;color:#6b7280;margin-bottom:12px;"
                        ),
                        ui.output_ui("bias_focused_heads_table")
                    )
                )
            ),
            id="bias_tabs"
        )
    )


__all__ = ["create_bias_sidebar", "create_bias_content"]
