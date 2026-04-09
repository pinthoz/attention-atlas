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
    # BLAZING FAST INLINE CSS to completely kill FOUC before stylesheets load!
    ui.HTML('''<style>
        .navbar:not(.sidebar .navbar), .navbar-toggler, .navbar-toggler-icon { 
            display: none !important; 
            opacity: 0 !important; 
            visibility: hidden !important; 
        }
    </style>'''),
    
    # Sidebar
    ui.div(
        {"class": "sidebar"},
        ui.div(
            {"class": "app-title", "style": "display: flex; align-items: center; justify-content: space-between; width: 100%;"},
            ui.div(
                {"style": "display: flex; align-items: center; gap: 8px;"},
                ui.tags.img(src=ICON_DATA_URL or "/favicon.ico", alt="Logo"),
                ui.h3("Attention Atlas", style="margin: 0;"),
            ),
            # Back Button (Visible only in compare modes)
            ui.div(
                {"id": "attn-back-button-container", "style": "display: none;"},
                ui.HTML(
                    '<div onclick="Shiny.setInputValue(\'attn_go_back\',Date.now(),{priority:\'event\'});"'
                    ' class="sidebar-back-btn" title="Back to previous analysis">'
                    '<i class="fa-solid fa-arrow-left" style="font-size: 10px;"></i> Back'
                    '</div>'
                )
            ),
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
            ui.tags.span("Compare Modes", id="cmp-modes-label", class_="sidebar-label"),
            
            # Checkbox Row - centered
            ui.div(
                {"id": "compare-modes-container"},
                ui.input_switch("compare_mode", ui.span("Models", class_="compare-label"), value=False),
                ui.input_switch("compare_prompts_mode", ui.span("Prompts", class_="compare-label"), value=False)
            ),

            # Model Configuration Container (Flex Row)
            ui.div(
                {"style": "display: flex; gap: 12px; align-items: flex-start;"},
                
                # LEFT COLUMN: Model A (Always Visible, Flex Grow)
                ui.div(
                    {"style": "flex: 1; min-width: 0; display: flex; flex-direction: column;"},

                    # Inputs A
                    ui.tags.span("Model Family", id="lbl-fam-a", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px; transition: color 0.3s ease;"),
                    ui.input_select(
                        "model_family",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="bert",
                        width="100%"
                    ),
                    ui.tags.span("Model Configuration", id="lbl-conf-a", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #64748b; margin-top: -5px; transition: color 0.3s ease;"),
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

                    # Inputs B
                    ui.tags.span("Model Family - B", id="lbl-fam-b", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #ff5ca9; margin-top: -5px;"),
                    ui.input_select(
                        "model_family_B",
                        None,
                        choices={"bert": "BERT", "gpt2": "GPT-2"},
                        selected="gpt2",
                        width="100%"
                    ),
                    ui.tags.span("Model Config - B", id="lbl-conf-b", class_="sidebar-label", style="margin-bottom: 2px; font-size: 10px; color: #ff5ca9; margin-top: -5px;"),
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

                        # Batch Mode Button
                        ui.tags.button(
                            ui.HTML('<i class="fa-solid fa-chart-pie"></i>'),
                            id="attn_batch_mode_btn",
                            class_="session-btn-custom",
                            title="Batch Mode",
                            onclick="window.toggleAttnBatchMode && window.toggleAttnBatchMode();",
                        ),

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

                # Batch Upload Section (hidden by default)
                ui.div(
                    {"id": "attn-batch-upload-section", "style": "display: none;"},
                    ui.div(
                        {"class": "batch-upload-container"},
                        ui.div(
                            {"class": "batch-header"},
                            ui.HTML('<i class="fa-solid fa-chart-pie" style="margin-right: 6px;"></i>'),
                            ui.tags.span("Batch Mode", style="font-weight: 600; font-size: 13px;"),
                            ui.tags.span("Upload a CSV or JSON to analyse multiple sentences at once.", style="font-weight: 400; font-size: 11px; opacity: 0.85; margin-left: 4px;"),
                        ),
                        ui.div(
                            {"class": "batch-dropzone", "id": "attn-batch-dropzone", "onclick": "document.getElementById('attn_batch_file_upload').click();"},
                            ui.HTML('<i class="fa-solid fa-cloud-arrow-up attn-batch-dropzone-icon" style="font-size: 20px; color: #94a3b8; margin-bottom: 4px;"></i>'),
                            ui.tags.div("Drop CSV or JSON file here, or click to browse", id="attn-batch-dropzone-label", style="color: #94a3b8; font-size: 12px; font-weight: 500; text-align: center;"),
                            ui.div(
                                {"id": "attn-batch-file-info", "class": "batch-file-info", "style": "display: none;"},
                                ui.HTML('<i class="fa-solid fa-file-lines" style="margin-right: 6px;"></i>'),
                                ui.tags.span("", id="attn-batch-file-name"),
                                ui.tags.span("", id="attn-batch-file-count", style="margin-left: 8px; font-weight: 600;"),
                            ),
                        ),
                    ),
                    ui.div(ui.input_file("attn_batch_file_upload", None, accept=[".csv", ".json"], multiple=False), style="display: none;"),
                ),

                # Batch Mode Styles
                ui.tags.style("""
                    #attn_batch_mode_btn.batch-active {
                        background: #ff5ca9 !important;
                        color: white !important;
                    }
                """),

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
                    var backBtn = document.getElementById('attn-back-button-container');

                    if (event.name === 'compare_prompts_mode') {
                        const container = document.getElementById('input-container');
                        if (event.value === true) {
                            container.classList.add('compare-prompts-active');
                            switchPrompt('A'); // Ensure Tab A is active and B is dimmed
                            // Back button shown after Generate All completes
                        } else {
                            container.classList.remove('compare-prompts-active');
                            switchPrompt('A');
                            if (backBtn) backBtn.style.display = 'none';
                        }
                    }
                    // Handle compare_mode toggle - show/hide Model A header and Model B panel smoothly
                    if (event.name === 'compare_mode') {
                        const modelBPanel = document.getElementById('model-b-panel');
                        const lblFamA = document.getElementById('lbl-fam-a');
                        const lblConfA = document.getElementById('lbl-conf-a');

                        if (event.value === true) {
                            // Show B Panel
                            if (modelBPanel) modelBPanel.classList.add('compare-active');
                            // Back button shown after Generate All completes

                            // Update Model A Labels (Blue + Suffix)
                            if(lblFamA) {
                                lblFamA.innerText = "Model Family - A";
                                lblFamA.style.color = "#3b82f6";
                            }
                            if(lblConfA) {
                                lblConfA.innerText = "Model Config - A";
                                lblConfA.style.color = "#3b82f6";
                            }
                        } else {
                            // Hide B Panel
                            if (modelBPanel) modelBPanel.classList.remove('compare-active');
                            if (backBtn) backBtn.style.display = 'none';

                            // Revert Model A Labels (Grey + Original)
                            if(lblFamA) {
                                lblFamA.innerText = "Model Family";
                                lblFamA.style.color = "#64748b";
                            }
                            if(lblConfA) {
                                lblConfA.innerText = "Model Configuration";
                                lblConfA.style.color = "#64748b";
                            }
                        }
                    }
                });

                // Persistence Logic
                Shiny.addCustomMessageHandler('update_history', function(message) {
                    localStorage.setItem('attention_atlas_history', JSON.stringify(message));
                });
                
                // Back button visibility after Generate All completes
                Shiny.addCustomMessageHandler('attn_back_btn_update', function(message) {
                    var bb = document.getElementById('attn-back-button-container');
                    if (bb) bb.style.display = message.show ? 'block' : 'none';
                });

                // Restore UI switches when Back is clicked
                Shiny.addCustomMessageHandler('attn_restore_ui', function(message) {
                    var swm = $('#compare_mode');
                    var swp = $('#compare_prompts_mode');
                    if (swm.prop('checked') !== message.compare_models) {
                        swm.prop('checked', message.compare_models).trigger('change');
                        Shiny.setInputValue('compare_mode', message.compare_models, {priority: 'event'});
                    }
                    if (swp.prop('checked') !== message.compare_prompts) {
                        swp.prop('checked', message.compare_prompts).trigger('change');
                        Shiny.setInputValue('compare_prompts_mode', message.compare_prompts, {priority: 'event'});
                    }
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

                    // Drag-and-drop for attention batch dropzone
                    var dz = document.getElementById('attn-batch-dropzone');
                    if (dz) {
                        dz.addEventListener('dragover', function(e) {
                            e.preventDefault(); e.stopPropagation();
                            dz.style.borderColor = '#ff5ca9';
                            dz.style.background = 'rgba(255, 92, 169, 0.06)';
                        });
                        dz.addEventListener('dragleave', function(e) {
                            e.preventDefault(); e.stopPropagation();
                            dz.style.borderColor = '';
                            dz.style.background = '';
                        });
                        dz.addEventListener('drop', function(e) {
                            e.preventDefault(); e.stopPropagation();
                            dz.style.borderColor = '';
                            dz.style.background = '';
                            var files = e.dataTransfer.files;
                            if (!files || !files.length) return;
                            var name = files[0].name.toLowerCase();
                            if (!name.endsWith('.csv') && !name.endsWith('.json')) return;
                            var fi = document.getElementById('attn_batch_file_upload');
                            if (fi) {
                                var dt = new DataTransfer();
                                dt.items.add(files[0]);
                                fi.files = dt.files;
                                $(fi).trigger('change');
                            }
                        });
                    }
                });

                // ── Attention Batch Mode ──────────────────────────────
                window.toggleAttnBatchMode = function() {
                    var btn = document.getElementById('attn_batch_mode_btn');
                    var section = document.getElementById('attn-batch-upload-section');
                    var textA = document.getElementById('text_input');
                    var textB = document.getElementById('text_input_B');
                    var isActive = btn.classList.contains('batch-active');

                    if (isActive) {
                        btn.classList.remove('batch-active');
                        section.style.display = 'none';
                        textA.style.display = 'block';
                        // Reset dropzone
                        var icon = document.querySelector('.attn-batch-dropzone-icon');
                        var lbl = document.getElementById('attn-batch-dropzone-label');
                        var info = document.getElementById('attn-batch-file-info');
                        if (icon) icon.style.display = '';
                        if (lbl) lbl.style.display = '';
                        if (info) info.style.display = 'none';
                        Shiny.setInputValue('attn_batch_mode_active', 'false', {priority: 'event'});
                    } else {
                        var container = section.querySelector('.batch-upload-container');
                        if (container && textA.offsetHeight > 0) {
                            container.style.height = textA.offsetHeight + 'px';
                        }
                        btn.classList.add('batch-active');
                        section.style.display = 'block';
                        textA.style.display = 'none';
                        textB.style.display = 'none';
                        // Turn off compare modes
                        var ms = document.getElementById('compare_mode');
                        if (ms && ms.checked) { ms.checked = false; $(ms).trigger('change'); }
                        var ps = document.getElementById('compare_prompts_mode');
                        if (ps && ps.checked) { ps.checked = false; $(ps).trigger('change'); }
                        Shiny.setInputValue('attn_batch_mode_active', 'true', {priority: 'event'});
                    }
                };

                Shiny.addCustomMessageHandler('attn_batch_file_parsed', function(msg) {
                    var infoDiv = document.getElementById('attn-batch-file-info');
                    var nameSpan = document.getElementById('attn-batch-file-name');
                    var countSpan = document.getElementById('attn-batch-file-count');
                    var icon = document.querySelector('.attn-batch-dropzone-icon');
                    var label = document.getElementById('attn-batch-dropzone-label');
                    if (msg.error) {
                        infoDiv.style.display = 'inline-flex';
                        infoDiv.style.background = 'rgba(239, 68, 68, 0.1)';
                        infoDiv.style.color = '#dc2626';
                        nameSpan.textContent = msg.error;
                        countSpan.textContent = '';
                    } else {
                        infoDiv.style.display = 'inline-flex';
                        infoDiv.style.background = 'rgba(34, 197, 94, 0.1)';
                        infoDiv.style.color = '#16a34a';
                        nameSpan.textContent = msg.filename;
                        countSpan.textContent = msg.count + ' sentences';
                    }
                    if (icon) icon.style.display = 'none';
                    if (label) label.style.display = 'none';
                });

                Shiny.addCustomMessageHandler('attn_batch_progress', function(msg) {
                    var btn = document.getElementById('generate_all');
                    if (btn) {
                        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin" style="margin-right:6px;"></i>' + msg.label;
                    }
                });

                Shiny.addCustomMessageHandler('attn_batch_download_ready', function(msg) {
                    var blob = new Blob([msg.json_content], {type: 'application/json'});
                    var url = URL.createObjectURL(blob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = msg.filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                });
                """)
            ),

            ui.div(
                {"style": "margin-top: 12px;"}, # Match Bias tab spacing
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
    # BLAZING FAST INLINE CSS to completely kill FOUC before stylesheets load!
    ui.HTML('''<style>
        .navbar:not(.sidebar .navbar), .navbar-toggler, .navbar-toggler-icon { 
            display: none !important; 
            opacity: 0 !important; 
            visibility: hidden !important; 
        }
    </style>'''),
    
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
        # BLAZING FAST SYNCHRONOUS STYLE INJECTION - GUARANTEES NO FOUC!
        ui.HTML("<script>document.write('<style>.navbar-toggler, .navbar-toggler-icon { display: none !important; }</style>');</script>"),
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

            /* PREVENT FOUC: Hide navbar everywhere unless it has been moved into the sidebar */
            nav.navbar:not(.sidebar nav.navbar),
            .navbar:not(.sidebar .navbar) {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                position: absolute !important;
                top: -9999px !important;
            }
            
            /* Show navbar once it reaches its designated location */
            .sidebar .navbar,
            .sidebar > .navbar {
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                animation: fadeInNav 0.3s ease-out;
            }

            @keyframes fadeInNav {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
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
                margin: 0 !important; 
                padding: 0 !important;
            }
            #visualization_options_container:empty {
                display: none !important;
                height: 0 !important;
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

            /* Compare Modes Container and Switches */
            #cmp-modes-label {
                color: #cbd5e1 !important;
                margin-bottom: 8px !important;
                width: 100% !important;
                text-align: left !important;
                display: block !important;
            }

            #compare-modes-container {
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                gap: 44px;
                white-space: nowrap;
            }

            #compare-modes-container .shiny-input-container {
                width: auto !important;
                margin-bottom: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            #compare-modes-container .form-check {
                margin: 0 !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                min-height: auto !important;
                width: auto !important;
                justify-content: center !important;
                background: transparent !important;
                border: none !important;
            }
            
            /* Target the switch itself */
            #compare-modes-container .form-check-input {
                margin: 0 8px 0 0 !important;
                float: none !important;
                cursor: pointer;
                background-color: #1e293b !important;
                border-color: #334155 !important;
                width: 2.2em !important;
                height: 1.2em !important;
            }

            #compare-modes-container .form-check-input:checked {
                background-color: #ff5ca9 !important;
                border-color: #ff5ca9 !important;
            }

            /* Redefine label colors/styles for integration */
            #compare-modes-container .compare-label {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #cbd5e1;
                font-weight: 600;
                margin-bottom: 0 !important;
                cursor: pointer;
                line-height: 1;
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
        """),
        ui.tags.script("""
            // Move navbar (Attention/Bias buttons) inside the active sidebar
            // so it appears below "Generate All" both on desktop and mobile.
            var _relocating = false;
            function clearTabHashFromUrl() {
                try {
                    var clean = window.location.pathname + window.location.search;
                    window.history.replaceState(null, '', clean);
                } catch (e) {}
            }
            function relocateNavbarIntoSidebar() {
                if (_relocating) return;
                _relocating = true;
                try {
                    document.body.classList.add('navbar-ready');
                    var navbar = document.querySelector('nav.navbar, .navbar');
                    if (!navbar) return;
                    var sidebars = document.querySelectorAll('.sidebar');
                    if (!sidebars.length) return;
                    var target = null;
                    sidebars.forEach(function(s) {
                        // offsetParent is null for position:fixed — use rect+style instead
                        var r = s.getBoundingClientRect();
                        var st = window.getComputedStyle(s);
                        if (st.display !== 'none' && st.visibility !== 'hidden' && r.width > 0 && r.height > 0) {
                            // Also ensure no ancestor has display:none
                            var p = s.parentElement, ok = true;
                            while (p) {
                                if (window.getComputedStyle(p).display === 'none') { ok = false; break; }
                                p = p.parentElement;
                            }
                            if (ok) target = s;
                        }
                    });
                    if (!target) target = sidebars[0];
                    if (navbar.parentElement !== target) {
                        target.appendChild(navbar);
                    }
                    navbar.style.display = '';
                    navbar.style.visibility = '';
                    navbar.querySelectorAll('a').forEach(function(a) {
                        a.removeAttribute('title');
                        var h = a.getAttribute('href');
                        var t = a.getAttribute('data-bs-target');
                        var target = null;

                        if (h && h.charAt(0) === '#') {
                            target = h;
                        } else if (t && t.charAt(0) === '#') {
                            target = t;
                        }

                        if (target) {
                            a.dataset.tabTarget = target;
                            a.setAttribute('data-bs-target', target);
                            a.setAttribute('data-bs-toggle', 'tab');
                            // Keep href for Bootstrap/Shiny tab state consistency.
                            if (!h || h.charAt(0) !== '#') {
                                a.setAttribute('href', target);
                            }
                            a.setAttribute('role', 'tab');
                            a.style.cursor = 'pointer';
                        }
                    });
                } finally {
                    _relocating = false;
                }
            }
            document.addEventListener('DOMContentLoaded', relocateNavbarIntoSidebar);
            setTimeout(relocateNavbarIntoSidebar, 100);
            setTimeout(relocateNavbarIntoSidebar, 500);
            setTimeout(relocateNavbarIntoSidebar, 1500);
            setInterval(relocateNavbarIntoSidebar, 400);
            document.addEventListener('shown.bs.tab', function() {
                relocateNavbarIntoSidebar();
                clearTabHashFromUrl();
            });
            if (window.Shiny) {
                $(document).on('shiny:value shiny:bound', function() {
                    setTimeout(relocateNavbarIntoSidebar, 30);
                });
            }
            // Fire on nav-link clicks to catch tab switches
            document.addEventListener('click', function(e) {
                var link = e.target.closest('.nav-link, [data-bs-toggle="tab"], [data-toggle="tab"]');
                if (!link) return;
                var href = link.getAttribute('href') || '';
                // Only force manual tab show when href is missing.
                if (link.dataset && link.dataset.tabTarget && (!href || href === '#') && window.bootstrap && bootstrap.Tab) {
                    e.preventDefault();
                    try { bootstrap.Tab.getOrCreateInstance(link).show(); } catch(err) {}
                }
                [50, 150, 300, 600].forEach(function(t) {
                    setTimeout(relocateNavbarIntoSidebar, t);
                });
            }, true);

            // On mobile, relocate misplaced Deep Dive arrows so the flow reads
            // top-to-bottom: each listed arrow is moved out of its current
            // parent and inserted right after its row container.
            function relocateDeepDiveArrowsMobile() {
                if (window.innerWidth > 1024) return;
                // Each entry: arrow id prefix + the heading text of the FROM card.
                // We insert the arrow right after the row that contains the FROM card.
                var moves = [
                    { prefix: 'arrow_Segment_Embeddings_Sum__Layer_Normalization', from: 'Segment Embeddings' },
                    { prefix: 'arrow_Q/K/V_Projections_Add__Norm',                 from: 'Q/K/V Projections' }
                ];
                moves.forEach(function(move) {
                    var nodes = document.querySelectorAll('[id^="' + move.prefix + '"]');
                    nodes.forEach(function(el) {
                        if (el.dataset.relocated === '1') return;
                        // Find the row containing the FROM card by matching an h4
                        var row = null;
                        var headings = document.querySelectorAll('.flex-card h4');
                        for (var i = 0; i < headings.length; i++) {
                            if (headings[i].textContent.trim() === move.from) {
                                row = headings[i].closest('.flex-row-container');
                                if (row) break;
                            }
                        }
                        if (!row || !row.parentNode) return;
                        // Strip absolute positioning / rotation inline styles
                        el.style.position = 'static';
                        el.style.top = 'auto';
                        el.style.bottom = 'auto';
                        el.style.left = 'auto';
                        el.style.right = 'auto';
                        el.style.transform = 'none';
                        el.style.width = '100%';
                        el.style.margin = '6px 0';
                        el.style.zIndex = '1';
                        var inner = el.querySelector('span');
                        if (inner) inner.style.transform = 'none';
                        row.parentNode.insertBefore(el, row.nextSibling);
                        el.dataset.relocated = '1';
                    });
                });
            }
            document.addEventListener('DOMContentLoaded', relocateDeepDiveArrowsMobile);
            setTimeout(relocateDeepDiveArrowsMobile, 300);
            setTimeout(relocateDeepDiveArrowsMobile, 1200);
            setInterval(relocateDeepDiveArrowsMobile, 2000);

            // On mobile, restack 2-column Plotly subplots (Attention-IG
            // correlation + BAR scatter, Top-K Jaccard + Jaccard vs BAR)
            // into a single column by relayout-ing axis domains.
            function relayoutMobilePlots() {
                if (window.innerWidth > 1024) return;
                if (!window.Plotly) return;
                var sels = [
                    '[id^="ig-chart-container"]',
                    '[id^="ig-topk-chart-container"]'
                ];
                document.querySelectorAll(sels.join(',')).forEach(function(el) {
                    if (!el._fullLayout || el.dataset.mobileLaidOut === '1') return;
                    var lay = el._fullLayout;
                    if (!lay.xaxis2 || !lay.yaxis2) return;
                    var update = {
                        'xaxis.domain':  [0, 1],
                        'yaxis.domain':  [0.56, 1],
                        'xaxis2.domain': [0, 1],
                        'yaxis2.domain': [0, 0.44],
                        'height': 760
                    };
                    if (lay.annotations && lay.annotations.length >= 2) {
                        var newAnns = lay.annotations.map(function(a) { return Object.assign({}, a); });
                        newAnns[0].x = 0.5; newAnns[0].y = 1.0;
                        newAnns[0].xref = 'paper'; newAnns[0].yref = 'paper';
                        newAnns[0].xanchor = 'center'; newAnns[0].yanchor = 'bottom';
                        newAnns[1].x = 0.5; newAnns[1].y = 0.46;
                        newAnns[1].xref = 'paper'; newAnns[1].yref = 'paper';
                        newAnns[1].xanchor = 'center'; newAnns[1].yanchor = 'bottom';
                        update.annotations = newAnns;
                    }
                    try {
                        Plotly.relayout(el, update).then(function() {
                            el.dataset.mobileLaidOut = '1';
                        }).catch(function(){});
                    } catch (e) {}
                });
            }
            setInterval(relayoutMobilePlots, 1500);
            window.addEventListener('resize', function() {
                document.querySelectorAll('[data-mobile-laid-out="1"]').forEach(function(e) {
                    if (window.innerWidth > 1024) e.dataset.mobileLaidOut = '0';
                });
            });
            window.addEventListener('resize', function() {
                // Reset flag so re-render after resize re-applies if needed
                document.querySelectorAll('[data-relocated="1"]').forEach(function(e) {
                    if (window.innerWidth > 1024) e.dataset.relocated = '0';
                });
                relocateDeepDiveArrowsMobile();
            });
            $(document).on('shown.bs.tab', function(e) {
                setTimeout(relocateNavbarIntoSidebar, 50);
                clearTabHashFromUrl();
                var target = $(e.target).text().trim();
                if (target === 'Bias') {
                    document.body.style.setProperty('overflow', 'auto', 'important');
                }
            });
            
            // Robust check to ensure scroll is enabled on Bias tab
            setInterval(function() {
                var activeTab = $('.navbar .nav-link.active').text().trim();
                if (activeTab === 'Bias' && document.body.style.overflow === 'hidden') {
                    document.body.style.setProperty('overflow', 'auto', 'important');
                }
            }, 500);
        """)
    ),

    # Modals (shared across tabs)
    footer=ui.div(
        metric_modal(),
        isa_overlay_modal(),
    )
)


__all__ = ["app_ui"]
