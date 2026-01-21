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
            {"class": "app-title", "style": "display: flex; align-items: center; gap: 8px;"},
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
            
            # Checkbox Row - forced single line with compact spacing
            ui.div(
                {"style": "margin-bottom: 0px; display: flex; align-items: center; gap: 4px; white-space: nowrap; overflow: hidden;"},
                ui.input_switch("compare_mode", ui.span("Compare Models", style="font-size: 10px; color: #cbd5e1; font-weight: 600;"), value=False),
                ui.input_switch("compare_prompts_mode", ui.span("Compare Prompts", style="font-size: 10px; color: #cbd5e1; font-weight: 600;"), value=False)
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
                    ui.tags.span("Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
                    ui.input_select(
                        "model_family",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="bert",
                        width="100%"
                    ),
                    ui.tags.span("Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
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
                    ui.tags.span("Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
                    ui.input_select(
                        "model_family_B",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="gpt2",
                        width="100%"
                    ),
                    ui.tags.span("Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
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
            
            # Custom History Input Component
            ui.div(
                {"class": "custom-input-container", "id": "input-container"},
                
                # Tabs Container (History + Compare Prompts Tabs)
                ui.div(
                    {"class": "tabs-row"},
                    
                    # History Tab (always visible, z-index highest)
                    ui.div(
                        {"class": "history-tab", "onclick": "toggleHistory()", "title": "History"},
                        ui.HTML("""<svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 448 512" fill="white"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M0 96C0 78.3 14.3 64 32 64H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32s14.3-32 32-32H416c17.7 0 32 14.3 32 32z"/></svg>"""),
                    ),
                    
                    # Compare Prompts Tabs A and B (Conditional, stacked behind history tab)
                    ui.panel_conditional(
                        "input.compare_prompts_mode",
                        ui.div(
                            {"class": "compare-tabs-inline"},
                            ui.div("A", id="tab-a", class_="prompt-tab tab-a active", onclick="switchPrompt('A')"),
                            ui.div("B", id="tab-b", class_="prompt-tab tab-b", onclick="switchPrompt('B')")
                        )
                    ),
                ),
                
                # History Dropdown (initially hidden)
                ui.div(
                    {"id": "history-dropdown", "class": "history-dropdown"},
                    ui.output_ui("history_list")
                ),
                
                # Custom Textarea A (Blue Border only in compare mode)
                ui.tags.textarea(
                    "All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.",
                    id="text_input",
                    class_="custom-textarea",
                    rows=6,
                    oninput="Shiny.setInputValue('text_input', this.value, {priority: 'event'})",
                ),

                # Custom Textarea B (Pink Border only in compare mode) - Initially Hidden
                ui.tags.textarea(
                    "Programmers are logical and rigorous. Artists are creative and emotional.",
                    id="text_input_B",
                    class_="custom-textarea",
                    rows=6,
                    oninput="Shiny.setInputValue('text_input_B', this.value)",
                    style="display: none;"
                ),
                
                # JS to handle history interactions and Prompt Switching
                ui.tags.script("""
                function toggleHistory() {
                    const dropdown = document.getElementById('history-dropdown');
                    dropdown.classList.toggle('show');
                }
                
                // Close dropdown when clicking outside
                document.addEventListener('click', function(event) {
                    const container = document.querySelector('.custom-input-container');
                    const dropdown = document.getElementById('history-dropdown');
                    if (container && !container.contains(event.target)) {
                        dropdown.classList.remove('show');
                    }
                });
                
                function selectHistoryItem(text) {
                    // Check which input is visible and set value there
                    const inputB = document.getElementById('text_input_B');
                    const isB = inputB && inputB.style.display !== 'none';
                    
                    const targetId = isB ? 'text_input_B' : 'text_input';
                    const textarea = document.getElementById(targetId);
                    
                    textarea.value = text;
                    Shiny.setInputValue(targetId, text, {priority: 'event'});
                    document.getElementById('history-dropdown').classList.remove('show');
                }

                function switchPrompt(mode) {
                    const tabA = document.getElementById('tab-a');
                    const tabB = document.getElementById('tab-b');
                    const inputA = document.getElementById('text_input');
                    const inputB = document.getElementById('text_input_B');

                    if (mode === 'A') {
                        if(tabA) {
                            tabA.style.opacity = '1.0';
                            tabA.style.zIndex = '20';
                        }
                        if(tabB) {
                            tabB.style.opacity = '0.5';
                            tabB.style.zIndex = '10';
                        }
                        inputA.style.display = 'block';
                        inputB.style.display = 'none';
                    } else {
                        if(tabA) {
                            tabA.style.opacity = '0.5';
                            tabA.style.zIndex = '10';
                        }
                        if(tabB) {
                            tabB.style.opacity = '1.0';
                            tabB.style.zIndex = '25';
                        }
                        inputA.style.display = 'none';
                        inputB.style.display = 'block';
                    }
                }
                
                // Handle server request to switch tabs
                Shiny.addCustomMessageHandler('switch_prompt_tab', function(message) {
                    switchPrompt(message);
                });
                
                
                // Handle compare_prompts_mode toggle
                $(document).on('shiny:inputchanged', function(event) {
                    if (event.name === 'compare_prompts_mode') {
                        const container = document.getElementById('input-container');
                        if (event.value === true) {
                            container.classList.add('compare-prompts-active');
                        } else {
                            container.classList.remove('compare-prompts-active');
                            switchPrompt('A');
                        }
                    }
                });

                // Persistence Logic
                Shiny.addCustomMessageHandler('update_history', function(message) {
                    localStorage.setItem('attention_atlas_history', JSON.stringify(message));
                });

                $(document).on('shiny:connected', function() {
                    const stored = localStorage.getItem('attention_atlas_history');
                    if (stored) {
                        Shiny.setInputValue('restored_history', JSON.parse(stored));
                    }
                });
                """)
            ),

            ui.div(
                {"style": "margin-bottom: 0px;"}, # Space for Visual Options
                ui.output_ui("dynamic_submit_button"),
            ),
        ),
        
        ui.output_ui("visualization_options_container")
    ),

    # Main Content
    ui.div(
        {"class": "content"},

        # Floating Control Bar (rendered dynamically based on model)
        ui.output_ui("floating_control_bar"),

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
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;700&family=PT+Serif:wght@400;700&display=swap"),
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
        ui.tags.script(src="https://d3js.org/d3.v7.min.js"),
        ui.tags.script(JS_CODE),
        ui.tags.script(JS_INTERACTIVE),
        ui.tags.script(JS_TREE_VIZ),
        ui.tags.script(JS_TREE_VIZ),
        ui.tags.script(JS_TRANSITION_MODAL),
        ui.tags.style("""
            /* FORCE HIDE NAVBAR TOGGLER GLOBALLY */
            .navbar-toggler, .navbar-toggler-icon {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                width: 0 !important;
                pointer-events: none !important;
            }

            /* FORCE HISTORY TAB GLUE AND SPACING */
            .custom-input-container {
                margin-bottom: 10px !important; /* EXACTLY 10px as requested */
                position: relative !important;
                display: block !important;
                margin-top: 32px !important; /* Adjusted to be "more glued" */
            }
            .sidebar-section {
                margin-top: 0 !important; /* Move everything UP (max) */
                padding-top: 0 !important;
                margin-bottom: 0 !important; 
            }
            .sidebar-label {
                display: block !important;
                margin-bottom: 6px !important; /* Slightly reduced */
            }
            .tabs-row {
                position: absolute !important;
                top: -26px !important;
                left: 0 !important;
            }
            .history-tab {
                position: relative !important;
                margin-bottom: 0 !important;
            }
            .history-dropdown {
                top: 0 !important; /* Start immediately below tab */
            }
            .custom-textarea {
                margin-top: 0 !important; /* Reset margin */
                position: relative !important;
                z-index: 40 !important;
            }
            /* Pull Visualization Options closer to Generate button */
            #visualization_options_container {
                margin-top: -20px !important;
            }
            #visualization_options_container .sidebar-section {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
        """)
    ),

    # Modals (shared across tabs)
    footer=ui.div(
        metric_modal(),
        isa_overlay_modal(),
    )
)


__all__ = ["app_ui"]
