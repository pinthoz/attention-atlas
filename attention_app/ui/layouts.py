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
            {"class": "app-subtitle", "style": "margin-bottom: 12px; padding-bottom: 12px;"}, # Tighter spacing
            "An interactive visualization of Transformer internals with a focus on attention mechanisms."
        ),

        # View Mode Toggle (Basic/Advanced) - Modern Pill Buttons with Icons
        ui.div(
            {"class": "sidebar-section", "style": "margin-top: 4px; margin-bottom: 4px;"}, # Move closer to line
            ui.tags.span("Mode", class_="sidebar-label"),
            ui.div(
                {"id": "view-mode-container", "style": "width: 100%; display: flex; justify-content: center; margin-top: 6px;"},
                ui.input_radio_buttons(
                    "view_mode",
                    None,
                    choices={
                        "basic": ui.HTML('''<span class="btn-content"><span>Basic</span><i class="fa-solid fa-gear"></i></span>'''),
                        "advanced": ui.HTML('''<span class="btn-content"><span>Advanced</span><i class="fa-solid fa-brain"></i></span>''')
                    },
                    selected="basic",
                    inline=True,
                ),
            ),
        ),

        ui.div(
            {"class": "sidebar-section", "style": "margin-top: 0; transform: translateY(-8px);"}, # Fine-tuned vertical spacing
            # Header
            ui.tags.span("Compare Modes", class_="sidebar-label"),
            
            # Checkbox Row - centered
            ui.div(
                {"id": "compare-modes-container", "style": "margin-bottom: 16px; display: flex; align-items: center; justify-content: center; width: 100%; gap: 48px; white-space: nowrap;"},
                ui.input_switch("compare_mode", ui.span("Models", style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #cbd5e1; font-weight: 600;"), value=False),
                ui.input_switch("compare_prompts_mode", ui.span("Prompts", style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #cbd5e1; font-weight: 600;"), value=False)
            ),

            # Model Configuration Container (Flex Row)
            ui.div(
                {"style": "display: flex; gap: 12px; align-items: flex-start;"},
                
                # LEFT COLUMN: Model A (Always Visible, Flex Grow)
                ui.div(
                    {"style": "flex: 1; min-width: 0; display: flex; flex-direction: column;"},

                    # Model A Header - Hidden by default via CSS, shown when compare_mode is on
                    ui.tags.span("Model A", id="model-a-header", class_="sidebar-label", style="color: #3b82f6; font-size: 10px; font-weight: 700; margin-bottom: 4px; border-bottom: 1px dashed rgba(59, 130, 246, 0.3); padding-bottom: 2px;"),

                    # Inputs A
                    ui.tags.span("Model Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
                    ui.input_select(
                        "model_family",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="bert",
                        width="100%"
                    ),
                    ui.tags.span("Model Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
                    ui.input_select(
                        "model_name",
                        None,
                        choices={
                            "bert-base-uncased": "BERT Base (Uncased)",
                            "bert-large-uncased": "BERT Large (Uncased)",
                            "bert-base-multilingual-uncased": "BERT Multilingual",
                        },
                        selected="bert-base-uncased",
                        width="100%"
                    ),
                ),

                # RIGHT COLUMN: Model B - Hidden by default via CSS, shown when compare_mode is on
                ui.div(
                    {"id": "model-b-panel", "style": "flex: 1; min-width: 0; flex-direction: column;"},

                    # Model B Header
                    ui.tags.span("Model B", class_="sidebar-label", style="color: #ff5ca9; font-size: 10px; font-weight: 700; margin-bottom: 4px; display: block; border-bottom: 1px dashed rgba(255, 92, 169, 0.3); padding-bottom: 2px;"),

                    # Inputs B
                    ui.tags.span("Model Family", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
                    ui.input_select(
                        "model_family_B",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="gpt2",
                        width="100%"
                    ),
                    ui.tags.span("Model Architecture", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px;"),
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
            {"class": "sidebar-section", "style": "margin-top: 0; transform: translateY(-14px);"}, # Force visual move up
            ui.tags.span("Input Text", class_="sidebar-label"),
            
            # Custom History Input Component
            ui.div(
                {"class": "custom-input-container", "id": "input-container"},
                
                # Tabs Container (History + Compare Prompts Tabs + Session Controls)
                ui.div(
                    {"class": "tabs-row"},

                    # History Tab (always visible, left side)
                    ui.div(
                        {"class": "history-tab", "onclick": "toggleHistory()", "title": "History"},
                        ui.HTML("""<svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 448 512" fill="white"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M0 96C0 78.3 14.3 64 32 64H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32s14.3-32 32-32H416c17.7 0 32 14.3 32 32z"/></svg>"""),
                    ),

                    # Compare Prompts Tabs A and B (Conditional, right after History tab)
                    ui.panel_conditional(
                        "input.compare_prompts_mode",
                        ui.div(
                            {"class": "compare-tabs-inline"},
                            ui.div("A", id="tab-a", class_="prompt-tab tab-a active", onclick="switchPrompt('A')"),
                            ui.div("B", id="tab-b", class_="prompt-tab tab-b", onclick="switchPrompt('B')")
                        ),
                    ),

                    # Session Controls (Right Side - margin-left: auto pushes to right)
                    ui.div(
                        {"class": "session-controls", "style": "display: flex; gap: 4px; align-items: flex-end; margin-left: auto;"},

                        # Load Button (Trigger for hidden input)
                        ui.tags.label(
                            ui.HTML('<i class="fa-solid fa-folder-open"></i>'),
                            {"class": "session-btn-custom", "title": "Load Session", "for": "load_session_upload"}
                        ),
                        # Hidden File Input
                        ui.div(
                            ui.input_file("load_session_upload", None, accept=[".json"], multiple=False),
                            style="display: none;"
                        ),

                        # Save Button
                        ui.download_button(
                            "save_session",
                            label=None,
                            icon=ui.tags.i(class_="fa-solid fa-floppy-disk"),
                            class_="session-btn-custom"
                        ),
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
                            switchPrompt('A'); // Ensure Tab A is active and B is dimmed
                        } else {
                            container.classList.remove('compare-prompts-active');
                            switchPrompt('A');
                        }
                    }
                    // Handle compare_mode toggle - show/hide Model A header and Model B panel smoothly
                    if (event.name === 'compare_mode') {
                        const modelAHeader = document.getElementById('model-a-header');
                        const modelBPanel = document.getElementById('model-b-panel');

                        if (event.value === true) {
                            // Show elements by adding class (CSS handles display and transition)
                            if (modelAHeader) modelAHeader.classList.add('compare-active');
                            if (modelBPanel) modelBPanel.classList.add('compare-active');
                        } else {
                            // Hide elements by removing class
                            if (modelAHeader) modelAHeader.classList.remove('compare-active');
                            if (modelBPanel) modelBPanel.classList.remove('compare-active');
                        }
                    }
                });

                // Persistence Logic
                Shiny.addCustomMessageHandler('update_history', function(message) {
                    localStorage.setItem('attention_atlas_history', JSON.stringify(message));
                });
                
                // Session Restore Logic for Textareas
                Shiny.addCustomMessageHandler('restore_session_text', function(message) {
                    if (message.text_input) {
                        const el = document.getElementById('text_input');
                        if (el) {
                            el.value = message.text_input;
                            Shiny.setInputValue('text_input', message.text_input, {priority: 'event'});
                        }
                    }
                    if (message.text_input_B) {
                        const elB = document.getElementById('text_input_B');
                        if (elB) {
                            elB.value = message.text_input_B;
                            Shiny.setInputValue('text_input_B', message.text_input_B);
                        }
                    }
                });

                $(document).on('shiny:connected', function() {
                    // Initialize inputs with default values
                    const inputA = document.getElementById('text_input');
                    if (inputA) Shiny.setInputValue('text_input', inputA.value);

                    const inputB = document.getElementById('text_input_B');
                    if (inputB) Shiny.setInputValue('text_input_B', inputB.value);

                    const stored = localStorage.getItem('attention_atlas_history');
                    if (stored) {
                        Shiny.setInputValue('restored_history', JSON.parse(stored));
                    }
                });
                """)
            ),

            ui.div(
                {"style": "margin-bottom: 0px;"}, # Space for Visual Options
                ui.input_action_button("generate_all", "Generate All", class_="btn-primary", style="padding-top: 6px; padding-bottom: 6px; min-height: 0; height: auto;"), # Shorter button
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
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"),
        ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
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
                /* margin-top: -20px !important; REMOVED to fix overlap */
            }
            #visualization_options_container .sidebar-section {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            /* View Mode Toggle Styling - Equal Width & Centered */
            #view-mode-container .shiny-input-container {
                width: 100% !important;
                display: flex !important;
                justify-content: center !important;
            }
            
            #view-mode-container .shiny-options-group {
                display: flex !important;
                gap: 12px !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                margin-top: 0 !important;
                justify-content: center !important;
                width: 100% !important;
            }
            
            #view-mode-container .form-check {
                padding: 0 !important;
                margin: 0 !important;
                min-height: 0 !important;
            }

            /* Target labels specifically inside the radio group */
            #view-mode-container .shiny-options-group label {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 4px 0 !important; /* Shorter vertical padding */
                width: 125px !important; /* FIXED WIDTH FOR EQUALITY */
                flex: 0 0 125px !important; /* Rigid flex sizing */
                font-size: 13px !important;
                font-weight: 700 !important;
                font-family: 'Inter', system-ui, sans-serif !important;
                text-transform: none !important; /* Allow Title Case */
                letter-spacing: 0.5px !important;
                color: #ff5ca9 !important;
                background: transparent !important;
                border: 2px solid #ff5ca9 !important;
                border-radius: 9999px !important;
                cursor: pointer !important;
                transition: all 0.2s ease-in-out !important;
                line-height: 1 !important;
                margin: 0 !important;
                opacity: 1 !important;
                box-shadow: none !important;
            }

            /* Hide inputs */
            #view-mode-container input[type="radio"],
            #view-mode-container .form-check-input {
                 position: absolute;
                 opacity: 0;
                 width: 0;
                 height: 0;
                 pointer-events: none;
            }

            /* Aggressively clear inner styles */
            #view-mode-container .btn-content,
            #view-mode-container .btn-content span,
            #view-mode-container label span,
            #view-mode-container i {
                background-color: transparent !important;
                background: transparent !important;
                color: inherit !important;
                border: none !important;
                box-shadow: none !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            #view-mode-container .btn-content {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
                gap: 8px !important;
            }

            #view-mode-container i {
                font-size: 14px !important;
            }

            /* Hover State */
            #view-mode-container .shiny-options-group label:hover {
                background: rgba(255, 92, 169, 0.1) !important;
                transform: translateY(-1px) !important;
            }

            /* Active State - Solid Pink (Less Bright) */
            #view-mode-container input:checked + label,
            #view-mode-container label:has(input:checked) {
                background-color: #e64090 !important; /* Slightly darker/less neon pink */
                color: #ffffff !important;
                border-color: #e64090 !important;
                box-shadow: none !important; /* Removed shadow/glow */
            }
            
            #view-mode-container input:checked + label:hover,
            #view-mode-container label:has(input:checked):hover {
                 background-color: #d63080 !important;
                 transform: translateY(-1px) !important;
                 box-shadow: none !important;
            }

            /* Compare Modes Centering Fix */
            #compare-modes-container .shiny-input-container {
                width: auto !important;
                margin-bottom: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            #compare-modes-container .form-check {
                margin: 0 !important;
                padding-left: 0 !important; /* Reset bootstrap padding */
                display: flex !important;
                align-items: center !important;
                min-height: auto !important;
                width: auto !important; /* Allow shrinking */
                justify-content: center !important;
            }
            
            #compare-modes-container .form-check-input {
                margin-left: 0 !important; /* Reset bootstrap */
                margin-right: 6px !important;
                float: none !important;
            }



            /* Accordion Styling for Dark Theme */
            .accordion {
                background: transparent !important;
                border: none !important;
            }
            .accordion-item {
                background: transparent !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 16px !important;
                margin-bottom: 16px !important;
                overflow: hidden !important;
            }
            .accordion-header {
                margin: 0 !important;
            }
            .accordion-button {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
                color: #e2e8f0 !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                padding: 14px 20px !important;
                border: none !important;
                box-shadow: none !important;
                transition: all 0.2s ease !important;
            }
            .accordion-button:not(.collapsed) {
                background: #ff5ca9 !important;
                color: #ffffff !important;
                box-shadow: 0 4px 6px -1px rgba(255, 92, 169, 0.2) !important;
            }
            .accordion-button:focus {
                z-index: 3;
                border-color: #ff5ca9;
                outline: 0;
                box-shadow: 0 0 0 0.25rem rgba(255, 92, 169, 0.25) !important;
            }
            .accordion-button:hover {
                background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
            }
            .accordion-button:not(.collapsed):hover {
                background: #e64090 !important;
            }
            .accordion-button::after {
                filter: invert(1) !important;
            }
            .accordion-button:not(.collapsed)::after {
                filter: invert(1) !important;
            }
            .accordion-body {
                background: #0f172a !important;
                padding: 16px !important;
                border-top: 1px solid #334155 !important;
                border-bottom-left-radius: 16px !important;
                border-bottom-right-radius: 16px !important;
            }
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
            }
            .accordion-panel-badge.essential {
                background: rgba(34, 197, 94, 0.15) !important;
                color: #22c55e !important;
                border: 1px solid rgba(34, 197, 94, 0.3) !important;
            }
            .accordion-panel-badge.explore {
                background: rgba(59, 130, 246, 0.15) !important;
                color: #3b82f6 !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
            }
            .accordion-panel-badge.technical {
                background: rgba(139, 92, 246, 0.15) !important;
                color: #8b5cf6 !important;
                border: 1px solid rgba(139, 92, 246, 0.3) !important;
            }
            
            /* Badges on Active (Pink) Header -> White */
            .accordion-button:not(.collapsed) .accordion-panel-badge {
                background: rgba(255, 255, 255, 0.2) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.5) !important;
            }

            /* Navbar Styling Overrides (Removed - Consolidated in styles.py) */
        """)
    ),

    # Modals (shared across tabs)
    footer=ui.div(
        metric_modal(),
        isa_overlay_modal(),
    )
)


__all__ = ["app_ui"]
