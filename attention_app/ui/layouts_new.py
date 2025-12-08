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
            ui.tags.span("Model Configuration", class_="sidebar-label"),
            ui.input_select(
                "model_family",
                "Model Family",
                choices={"bert": "BERT", "gpt2": "GPT-2"},
                selected="bert"
            ),
            ui.input_select(
                "model_name",
                "Select Architecture",
                choices={
                    "bert-base-uncased": "BERT Base (Uncased)",
                    "bert-large-uncased": "BERT Large (Uncased)",
                    "bert-base-multilingual-uncased": "BERT Multilingual",
                },
                selected="bert-base-uncased"
            ),
        ),

        ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("Input Text", class_="sidebar-label"),
            ui.input_text_area("text_input", None, "Men are often expected to take leadership roles in the workplace. As a result, women are typically underrepresented in top management positions.", rows=6),
            ui.div(
                ui.input_action_button("generate_all", "Generate All", class_="btn-primary"),
                ui.div(
                    {"id": "loading_spinner", "class": "loading-container", "style": "display:none;"},
                    ui.div({"class": "spinner"}),
                    ui.span("Processing...")
                ),
            ),
        ),

        ui.output_ui("visualization_options_container")
    ),

    # Main Content
    ui.div(
        {"class": "content"},

        # Sentence Preview
        ui.div(
            {"class": "card"},
            ui.h4("Sentence Preview"),
            ui.output_ui("preview_text"),
        ),

        # Dashboard Content (Synchronized Rendering)
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
        "Attention Analysis",
        attention_analysis_page
    ),
    ui.nav_panel(
        "Bias Analysis",
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
