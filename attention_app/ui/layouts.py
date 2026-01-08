"""Main UI layout for Attention Atlas with tabs."""

from shiny import ui
from shinywidgets import output_widget

from .styles import CSS
from .scripts import JS_CODE, JS_INTERACTIVE, JS_TREE_VIZ, JS_TRANSITION_MODAL
from .components import ICON_DATA_URL
from .modals import metric_modal, isa_overlay_modal
from .bias_ui import create_bias_sidebar, create_bias_content


# Original attention analysis page
attention_analysis_page = ui.page_fluid(
    # Sidebar
    ui.div(
        {"class": "sidebar"},
        ui.div(
            {"class": "app-title"},
            ui.tags.img(src=ICON_DATA_URL or "/favicon.ico", alt="Logo"),
            ui.h3("Attention Atlas"),
        ),
        ui.div(
            {"class": "app-subtitle"},
            "An interactive visualization of Transformer internals with a focus on attention mechanisms."
        ),

        ui.div(
            {"class": "sidebar-section"},
            # Header
            ui.tags.span("Model Configuration", class_="sidebar-label"),
            
            # Compare Mode Toggle (Inline, styled like section header)
            ui.div(
                {"style": "margin-bottom: 12px;"},
                ui.input_switch("compare_mode", ui.span("Compare Models", style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #cbd5e1; font-weight: 600;"), value=False)
            ),

            # Model Configuration Container (Flex Row)
            ui.div(
                {"style": "display: flex; gap: 12px; align-items: flex-start;"},
                
                # LEFT COLUMN: Model A (Always Visible, Flex Grow)
                ui.div(
                    {"style": "flex: 1; min-width: 0; display: flex; flex-direction: column;"},
                    
                    # Model A Header (Conditional)
                    ui.panel_conditional(
                        "input.compare_mode",
                        ui.tags.span("Model A", class_="sidebar-label", style="color: #3b82f6; font-size: 10px; font-weight: 700; margin-bottom: 4px; display: block; border-bottom: 1px dashed rgba(59, 130, 246, 0.3); padding-bottom: 2px;")
                    ),

                    # Inputs A
                    ui.tags.span("Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b;"),
                    ui.input_select(
                        "model_family",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="bert",
                        width="100%"
                    ),
                    ui.tags.span("Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: 6px;"),
                    ui.input_select(
                        "model_name",
                        None,
                        choices={
                            "bert-base-uncased": "Base (Uncased)", # Shortened label
                            "bert-large-uncased": "Large (Uncased)", # Shortened label
                            "bert-base-multilingual-uncased": "Multilingual", # Shortened label
                        },
                        selected="bert-base-uncased",
                        width="100%"
                    ),
                ),

                # RIGHT COLUMN: Model B (Conditional)
                ui.panel_conditional(
                    "input.compare_mode",
                    {"style": "flex: 1; min-width: 0; display: flex; flex-direction: column;"}, # Attributes for the panel div
                    
                    # Model B Header
                    ui.tags.span("Model B", class_="sidebar-label", style="color: #ff5ca9; font-size: 10px; font-weight: 700; margin-bottom: 4px; display: block; border-bottom: 1px dashed rgba(255, 92, 169, 0.3); padding-bottom: 2px;"),
                    
                    # Inputs B
                    ui.tags.span("Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b;"),
                    ui.input_select(
                        "model_family_B",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="gpt2",
                        width="100%"
                    ),
                    ui.tags.span("Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: 6px;"),
                    ui.input_select(
                        "model_name_B",
                        None,
                        choices={
                            "gpt2": "GPT-2 Small",
                        },
                        selected="gpt2",
                        width="100%"
                    ),
                )
            )
        ),

        ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("Input Text", class_="sidebar-label"),
            ui.input_text_area("text_input", None, "All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.", rows=6),
            ui.div(
                ui.input_action_button("generate_all", "Generate All", class_="btn-primary"),
            ),
        ),
        
        ui.output_ui("visualization_options_container")
    ),

    # Main Content
    ui.div(
        {"class": "content"},

        # Dashboard Content (handles both static preview and generated content)
        ui.output_ui("dashboard_content")
    ),
)

# Bias analysis page
bias_analysis_page = ui.page_fluid(
    create_bias_sidebar(),
    create_bias_content()
)

# Main app UI with navbar
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Attention",
        attention_analysis_page
    ),
    ui.nav_panel(
        "Bias",
        bias_analysis_page
    ),
    title="Attention Atlas",
    id="main_navbar",

    # CSS Styles
    header=ui.tags.head(
        ui.tags.title("Attention Atlas"),
        ui.tags.style(CSS),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;700&display=swap"),
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
        ui.tags.script(src="https://d3js.org/d3.v7.min.js"),
        ui.tags.script(JS_CODE),
        ui.tags.script(JS_INTERACTIVE),
        ui.tags.script(JS_TREE_VIZ),
        ui.tags.script(JS_TRANSITION_MODAL),
    ),

    # Modals (shared across tabs)
    footer=ui.div(
        metric_modal(),
        isa_overlay_modal(),
    )
)


__all__ = ["app_ui"]
