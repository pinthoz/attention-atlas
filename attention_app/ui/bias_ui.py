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

        # ── Compare Modes Section ──
        ui.div(
            {"class": "sidebar-section", "style": "margin-top: 0; padding-bottom: 10px;"},
            # Header
            ui.tags.span("Compare Modes", id="bias-cmp-modes-label", class_="sidebar-label"),
            # Switches Row
            # Switches Row
            ui.div(
                {"id": "bias-compare-modes-container"},
                ui.input_switch("bias_compare_mode", ui.span("Models", class_="compare-label"), value=False),
                ui.input_switch("bias_compare_prompts_mode", ui.span("Prompts", class_="compare-label"), value=False)
            ),
        ),

        # Compare Modes Styling
        ui.tags.style("""
            #bias-cmp-modes-label {
                color: #cbd5e1 !important;
                margin-bottom: 8px !important;
                width: 100% !important;
                text-align: left !important;
                display: block !important;
            }

            #bias-compare-modes-container {
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                gap: 44px;
                white-space: nowrap;
            }

            #bias-compare-modes-container .shiny-input-container {
                width: auto !important;
                margin-bottom: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            #bias-compare-modes-container .form-check {
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
            
            #bias-compare-modes-container .form-check-input {
                margin: 0 8px 0 0 !important;
                float: none !important;
                cursor: pointer;
                background-color: #1e293b !important;
                border-color: #334155 !important;
                width: 2.2em !important;
                height: 1.2em !important;
            }

            #bias-compare-modes-container .form-check-input:checked {
                background-color: #ff5ca9 !important;
                border-color: #ff5ca9 !important;
            }

            .compare-label {
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #cbd5e1;
                font-weight: 600;
                margin-bottom: 0 !important;
                cursor: pointer;
                line-height: 1;
            }
        """),

        # ── Models Container (A and B side by side when compare mode) ──
        ui.div(
            {"id": "bias-models-container", "style": "display: flex; gap: 12px;"},

            # Model A Panel
            ui.div(
                {"id": "bias-model-a-panel", "style": "flex: 1;"},
                ui.div(
                    {"class": "sidebar-section", "style": "margin-top: 4px; margin-bottom: 4px;"},
                    ui.tags.span("Bias Detection Model", id="bias-model-a-label", class_="sidebar-label"),
                    ui.div(
                        {"class": "bias-model-selector-wrap", "style": "margin-top: 8px;"},
                        ui.tags.select(
                            ui.tags.option("GUS-Net (BERT)", value="gusnet-bert", selected="selected"),
                            ui.tags.option("GUS-Net Ensemble", value="gusnet-ensemble"),
                            ui.tags.option("GUS-Net (BERT Large)", value="gusnet-bert-large"),
                            ui.tags.option("GUS-Net (GPT-2)", value="gusnet-gpt2"),
                            ui.tags.option("GUS-Net (GPT-2 Medium)", value="gusnet-gpt2-medium"),
                            id="bias_model_key",
                            class_="bias-model-select",
                            onchange="Shiny.setInputValue('bias_model_key', this.value, {priority:'event'});",
                        ),
                    ),
                ),
            ),

            # Model B Panel (hidden by default)
            ui.div(
                {"id": "bias-model-b-panel", "style": "flex: 1; display: none;"},
                ui.div(
                    {"class": "sidebar-section", "style": "margin-top: 4px; margin-bottom: 4px;"},
                    ui.tags.span("Detect Model - B", id="bias-model-b-label", class_="sidebar-label", style="color: #ff5ca9;"),
                    ui.div(
                        {"class": "bias-model-selector-wrap", "style": "margin-top: 8px;"},
                        ui.tags.select(
                            ui.tags.option("GUS-Net (BERT)", value="gusnet-bert"),
                            ui.tags.option("GUS-Net Ensemble", value="gusnet-ensemble"),
                            ui.tags.option("GUS-Net (BERT Large)", value="gusnet-bert-large"),
                            ui.tags.option("GUS-Net (GPT-2)", value="gusnet-gpt2", selected="selected"),
                            ui.tags.option("GUS-Net (GPT-2 Medium)", value="gusnet-gpt2-medium"),
                            id="bias_model_key_B",
                            class_="bias-model-select bias-model-select-b",
                            onchange="Shiny.setInputValue('bias_model_key_B', this.value, {priority:'event'});",
                        ),
                    ),
                ),
            ),
        ),

        # ── Styles for model selects and toggle switches ──
        ui.tags.style("""
            .bias-model-select {
                width: 100%;
                padding: 6px 28px 6px 10px;
                font-size: 12px;
                height: 32px;
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                appearance: none;
                -webkit-appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394a3b8' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 8px center;
                background-size: 12px;
                cursor: pointer;
            }
            .bias-model-select:hover {
                border-color: rgba(255, 92, 169, 0.4);
            }
            .bias-model-select:focus {
                border-color: #ff5ca9;
                box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.15);
            }
            .bias-model-select option {
                background: #1e293b;
                color: #e2e8f0;
                padding: 6px;
            }

            /* Model A label when in compare mode */
            #bias-model-a-label.compare-active {
                color: #3b82f6 !important;
            }

            /* Model B select pink border */
            .bias-model-select-b {
                border-color: rgba(255, 92, 169, 0.3) !important;
            }
            .bias-model-select-b:hover {
                border-color: rgba(255, 92, 169, 0.5) !important;
            }
            .bias-model-select-b:focus {
                border-color: #ff5ca9 !important;
                box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.2) !important;
            }

            /* Toggle Switch Styling */
            .toggle-switch {
                position: relative;
                display: inline-block;
                width: 36px;
                height: 20px;
            }
            .toggle-switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #334155;
                transition: 0.3s;
                border-radius: 20px;
            }
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 14px;
                width: 14px;
                left: 3px;
                bottom: 3px;
                background-color: #94a3b8;
                transition: 0.3s;
                border-radius: 50%;
            }
            .toggle-switch input:checked + .toggle-slider {
                background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
            }
            .toggle-switch input:checked + .toggle-slider:before {
                transform: translateX(16px);
                background-color: white;
            }
        """),

        # ── Spacer ──
        ui.div(style="flex-grow: 1; min-height: 24px;"),

        # ── Sensitivity Thresholds (styled like the floating toolbar sliders) ──
        ui.tags.style("""
            .sidebar-thresh-group {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
                padding: 2px 0;
                min-width: 0;
                overflow: hidden;
            }
            .sidebar-thresh-group .thresh-label {
                font-size: 9px;
                font-weight: 600;
                color: #ffffff;
                text-transform: uppercase;
                letter-spacing: 0.3px;
                white-space: nowrap;
                width: auto;
                flex-shrink: 0;
            }
            .sidebar-thresh-group .thresh-val {
                min-width: 24px;
                height: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 9px;
                font-weight: 700;
                color: #fff;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
                font-family: 'JetBrains Mono', monospace;
                flex-shrink: 0;
            }
            .sidebar-thresh-group input[type="range"] {
                -webkit-appearance: none;
                flex: 1 1 0;
                min-width: 0;
                max-width: 70px;
                height: 3px;
                background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
                border-radius: 2px;
                outline: none;
                box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
            }
            .sidebar-thresh-group input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
                border: 1.5px solid rgba(255, 255, 255, 0.2);
                cursor: pointer;
                box-shadow: 0 1px 4px rgba(255, 92, 169, 0.4);
                transition: all 0.15s ease;
            }
            .sidebar-thresh-group input[type="range"]::-webkit-slider-thumb:hover {
                transform: scale(1.2);
                box-shadow: 0 2px 8px rgba(255, 92, 169, 0.6);
            }
            .sidebar-thresh-group input[type="range"]::-moz-range-thumb {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
                border: 1.5px solid rgba(255, 255, 255, 0.2);
                cursor: pointer;
                box-shadow: 0 1px 4px rgba(255, 92, 169, 0.4);
            }
            .sidebar-thresh-group input[type="range"]::-moz-range-track {
                background: transparent;
                border: 0;
            }
        """),
        ui.div(
            {"class": "sidebar-section", "style": "margin-top: 0; padding-bottom: 10px;"},
            ui.tags.span("Sensitivity Threshold", class_="sidebar-label"),
            ui.div(
                {"style": "display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px;"},
                # UNFAIR
                ui.div(
                    {"class": "sidebar-thresh-group"},
                    ui.span("UNFAIR", class_="thresh-label"),
                    ui.tags.input(type="range", id="bias-thresh-unfair", min="0.01", max="0.99", value="0.5", step="0.01"),
                    ui.tags.span("0.50", id="bias-thresh-unfair-val", class_="thresh-val"),
                ),
                # GEN
                ui.div(
                    {"class": "sidebar-thresh-group"},
                    ui.span("GEN", class_="thresh-label"),
                    ui.tags.input(type="range", id="bias-thresh-gen", min="0.01", max="0.99", value="0.5", step="0.01"),
                    ui.tags.span("0.50", id="bias-thresh-gen-val", class_="thresh-val"),
                ),
                # STEREO
                ui.div(
                    {"class": "sidebar-thresh-group"},
                    ui.span("STEREO", class_="thresh-label"),
                    ui.tags.input(type="range", id="bias-thresh-stereo", min="0.01", max="0.99", value="0.5", step="0.01"),
                    ui.tags.span("0.50", id="bias-thresh-stereo-val", class_="thresh-val"),
                ),
            ),
        ),

        # ── Input Text Area ──
        ui.div(
            {"class": "sidebar-section", "style": "padding-top: 16px;"},
            ui.tags.span("Input Text", class_="sidebar-label"),

            ui.div(
                {"class": "custom-input-container", "id": "bias-input-container"},
                ui.div(
                    {"class": "tabs-row"},
                    ui.div(
                        {"class": "history-tab", "onclick": "toggleBiasHistory()", "title": "History"},
                        ui.HTML("""<svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 448 512" fill="white"><path d="M0 96C0 78.3 14.3 64 32 64H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32C14.3 128 0 113.7 0 96zM0 256c0-17.7 14.3-32 32-32H416c17.7 0 32 14.3 32 32s-14.3 32-32 32H32c-17.7 0-32-14.3-32-32zM448 416c0 17.7-14.3 32-32 32H32c-17.7 0-32-14.3-32-32s14.3-32 32-32H416c17.7 0 32 14.3 32 32z"/></svg>"""),
                    ),
                    # Compare Prompts Tabs (hidden by default)
                    ui.div(
                        {"class": "bias-compare-tabs-inline", "id": "bias-prompt-tabs", "style": "display: none; gap: 2px; margin-right: 8px;"},
                        ui.tags.button("A", id="bias-prompt-tab-a", class_="bias-prompt-tab active", onclick="window.switchBiasPromptTab && window.switchBiasPromptTab('A');"),
                        ui.tags.button("B", id="bias-prompt-tab-b", class_="bias-prompt-tab", onclick="window.switchBiasPromptTab && window.switchBiasPromptTab('B');"),
                    ),
                    ui.div(
                        {"class": "session-controls", "style": "display: flex; gap: 4px; align-items: flex-end; margin-left: auto;"},
                        ui.tags.label(ui.HTML('<i class="fa-solid fa-folder-open"></i>'), {"class": "session-btn-custom", "title": "Load Session", "for": "load_bias_session_upload"}),
                        ui.div(ui.input_file("load_bias_session_upload", None, accept=[".json"], multiple=False), style="display: none;"),
                        ui.download_button("save_bias_session", label=None, icon=ui.tags.i(class_="fa-solid fa-floppy-disk"), class_="session-btn-custom"),
                    ),
                ),
                ui.div({"id": "bias-history-dropdown", "class": "history-dropdown"}, ui.output_ui("bias_history_list")),
                # Text Input A
                ui.tags.textarea("All women are naturally nurturing and emotional. Men are logical and suited for leadership positions.", id="bias_input_text", class_="custom-textarea", rows=6, oninput="Shiny.setInputValue('bias_input_text', this.value, {priority: 'event'})"),
                # Text Input B (hidden by default)
                ui.tags.textarea("Programmers are logical and rigorous. Artists are creative and emotional.", id="bias_input_text_B", class_="custom-textarea", rows=6, style="display: none;", oninput="Shiny.setInputValue('bias_input_text_B', this.value, {priority: 'event'})", placeholder="Enter second text for comparison..."),
            ),

            # Prompt tabs styling
            ui.tags.style("""
                .bias-compare-tabs-inline {
                    display: none;
                }
                /* Prompt Tab Base Styles (Matched to Attention Tab) */
                .bias-prompt-tab {
                    position: relative;
                    padding: 4px 12px;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    border-bottom-left-radius: 0;
                    border-bottom-right-radius: 0;
                    font-size: 14px;
                    font-weight: 700;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    height: 26px;
                    line-height: 1;
                    color: white;
                    border: none;
                    transition: opacity 0.2s ease, transform 0.2s ease;
                }

                /* Tab A - Blue */
                #bias-prompt-tab-a {
                    background: #3b82f6; /* Solid blue */
                    margin-left: -8px;
                    z-index: 20;
                    box-shadow: 2px -2px 5px rgba(0,0,0,0.1);
                    padding-left: 14px;
                    padding-right: 10px;
                }
                #bias-prompt-tab-a:hover {
                    background: #2563eb;
                }

                /* Tab B - Pink */
                #bias-prompt-tab-b {
                    background: #ff5ca9; /* Solid pink */
                    margin-left: -8px;
                    z-index: 10;
                    box-shadow: 2px -2px 5px rgba(0,0,0,0.1);
                    padding-left: 14px;
                    padding-right: 10px;
                }
                #bias-prompt-tab-b:hover {
                    background: #f43f8e;
                }
                
                /* Active State Logic (opacity/z-index toggled via JS, colors stay constant) */
                .bias-prompt-tab.active {
                    /* No specific active style needed here as we toggle opacity/zIndex in JS like Attention tab */
                }

                /* Text input styling in compare prompts mode */
                #bias-input-container.compare-prompts-active #bias_input_text {
                    border: 2px solid #3b82f6;
                }
                #bias-input-container.compare-prompts-active #bias_input_text_B {
                    border: 2px solid #ff5ca9;
                }
            """),


            ui.div(
                {"style": "margin-top: 12px;"},
                ui.input_action_button("analyze_bias_btn", "Analyze Bias", class_="btn-primary", style="padding-top: 6px; padding-bottom: 6px; min-height: 0; height: auto;"),
            ),

            # JavaScript for compare modes
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

                // StereoSet: inject text into the bias input
                window.analyzeStereoSetExample = function(text) {
                    var ta = document.getElementById('bias_input_text');
                    if (ta) {
                        ta.value = text;
                        Shiny.setInputValue('bias_input_text', text, {priority: 'event'});
                    }
                    window.scrollTo({top: 0, behavior: 'smooth'});
                };
                document.addEventListener('click', function(event) {
                    const container = document.getElementById('bias-input-container');
                    const dropdown = document.getElementById('bias-history-dropdown');
                    if (container && !container.contains(event.target)) {
                        dropdown.classList.remove('show');
                    }
                });

                // Compare Mode Logic (Models & Prompts) - Handling shiny:inputchanged for standard switches
                $(document).on('shiny:inputchanged', function(event) {
                    // Compare Models Logic
                    if (event.name === 'bias_compare_mode') {
                        const enabled = event.value;
                        const modelBPanel = document.getElementById('bias-model-b-panel');
                        const modelALabel = document.getElementById('bias-model-a-label');

                        if (enabled) {
                            modelBPanel.style.display = 'block';
                            modelALabel.classList.add('compare-active');
                            modelALabel.innerText = "Detect Model - A";
                            
                            // Mutex: Turn off Compare Prompts if on
                            // using jQuery to trigger change so shiny sees it (and executes that handler to update UI)
                            const promptSwitch = $('#bias_compare_prompts_mode');
                            // Only if it is currently visually checked (Shiny input might be true)
                            // We can check .prop('checked')
                            if (promptSwitch.length && promptSwitch.prop('checked')) {
                                // We need to update the checkbox and trigger change
                                // But prevent infinite loops? The prompt handler logic won't turn off models if prompts is becoming false.
                                // So it's safe.
                                promptSwitch.prop('checked', false).trigger('change');
                                // Note: trigger('change') might not be enough for shiny binding to pick up in all cases
                                // but standard behavior for shiny wrapper usually works with change. 
                            }
                        } else {
                            modelBPanel.style.display = 'none';
                            modelALabel.classList.remove('compare-active');
                            modelALabel.innerText = "Bias Detection Model";
                        }
                    }

                    // Compare Prompts Logic
                    if (event.name === 'bias_compare_prompts_mode') {
                        const enabled = event.value;
                        const promptTabs = document.getElementById('bias-prompt-tabs');
                        const inputContainer = document.getElementById('bias-input-container');
                        const textInputA = document.getElementById('bias_input_text');
                        const textInputB = document.getElementById('bias_input_text_B');
                        
                        if (enabled) {
                            promptTabs.style.display = 'flex';
                            inputContainer.classList.add('compare-prompts-active');
                            textInputA.style.display = 'block';
                            textInputB.style.display = 'none';
                            
                            // Mutex: Turn off Compare Models if on
                            const modelSwitch = $('#bias_compare_mode');
                            if (modelSwitch.length && modelSwitch.prop('checked')) {
                                modelSwitch.prop('checked', false).trigger('change');
                            }
                        } else {
                            promptTabs.style.display = 'none';
                            inputContainer.classList.remove('compare-prompts-active');
                            textInputA.style.display = 'block';
                            textInputB.style.display = 'none';
                        }
                    }
                });

                // Switch between Prompt A and B tabs
                window.switchBiasPromptTab = function(tab) {
                    const tabA = document.getElementById('bias-prompt-tab-a');
                    const tabB = document.getElementById('bias-prompt-tab-b');
                    const textInputA = document.getElementById('bias_input_text');
                    const textInputB = document.getElementById('bias_input_text_B');

                    if (tab === 'A') {
                        tabA.classList.add('active');
                        tabB.classList.remove('active');
                        tabA.style.opacity = '1.0';
                        tabA.style.zIndex = '20';
                        tabB.style.opacity = '0.5';
                        tabB.style.zIndex = '10';
                        textInputA.style.display = 'block';
                        textInputB.style.display = 'none';
                    } else {
                        tabA.classList.remove('active');
                        tabB.classList.add('active');
                        tabA.style.opacity = '0.5';
                        tabA.style.zIndex = '10';
                        tabB.style.opacity = '1.0';
                        tabB.style.zIndex = '25';
                        textInputA.style.display = 'none';
                        textInputB.style.display = 'block';
                    }
                    Shiny.setInputValue('bias_active_prompt_tab', tab, {priority: 'event'});
                };

                // Handle dynamic bias button label updates (Sequential Logic)
                Shiny.addCustomMessageHandler('update_bias_button_label', function(msg) {
                    var btn = $('#analyze_bias_btn');
                    if (btn.length === 0) return;

                    var newLabel = msg.label;
                    var isDisabled = btn.prop('disabled');

                    if (isDisabled) {
                        btn.data('original-content', newLabel);
                    } else {
                        btn.html(newLabel);
                        btn.data('original-content', newLabel); 
                    }
                });

                // Generic JS Evaluator for simple server commands
                Shiny.addCustomMessageHandler('bias_eval_js', function(code) {
                    eval(code);
                });

                // Handler for setting bias thresholds from server (e.g. optimized values)
                Shiny.addCustomMessageHandler('set_bias_thresholds', function(message) {
                    console.log("[BiasUI] Received thresholds:", message);
                    // Helper to update slider without triggering input (which would uncheck optimized box)
                    function updateSlider(id, valId, val) {
                        var el = document.getElementById(id);
                        var valDisplay = document.getElementById(valId);
                        if (el) {
                            el.value = val;
                            if (valDisplay) valDisplay.textContent = parseFloat(val).toFixed(2);
                            // We do NOT trigger 'input' or 'change' here to avoid side effects
                        }
                    }
                    if (message.UNFAIR !== undefined) {
                        updateSlider('bias-thresh-unfair', 'bias-thresh-unfair-val', message.UNFAIR);
                    }
                    if (message.GEN !== undefined) {
                        updateSlider('bias-thresh-gen', 'bias-thresh-gen-val', message.GEN);
                    }
                    if (message.STEREO !== undefined) {
                        updateSlider('bias-thresh-stereo', 'bias-thresh-stereo-val', message.STEREO);
                    }
                });

                function setupBiasToolbar() {
                    console.log("[BiasUI] Initializing controls...");
                    
                    function debounce(func, wait) {
                        let timeout;
                        return function(...args) {
                            const context = this;
                            clearTimeout(timeout);
                            timeout = setTimeout(() => func.apply(context, args), wait);
                        };
                    }

                    const inputs = [
                        {id: 'bias-layer-slider', valId: 'bias-layer-value', msg: 'bias_attn_layer'},
                        {id: 'bias-head-slider', valId: 'bias-head-value', msg: 'bias_attn_head'},
                        {id: 'bias-bar-threshold-slider', valId: 'bias-bar-threshold-value', msg: 'bias_bar_threshold'},
                        {id: 'bias-topk-slider', valId: 'bias-topk-value', msg: 'bias_top_k'},
                        
                        // Sidebar Thresholds
                        {id: 'bias-thresh-unfair', valId: 'bias-thresh-unfair-val', msg: 'bias_thresh_unfair', isThresh: true},
                        {id: 'bias-thresh-gen', valId: 'bias-thresh-gen-val', msg: 'bias_thresh_gen', isThresh: true},
                        {id: 'bias-thresh-stereo', valId: 'bias-thresh-stereo-val', msg: 'bias_thresh_stereo', isThresh: true}
                    ];

                    inputs.forEach(item => {
                        const el = document.getElementById(item.id);
                        const valDisplay = document.getElementById(item.valId);
                        
                        if (el) {
                            // Create debounced sender (500ms delay)
                            const sendValue = debounce(function(val) {
                                Shiny.setInputValue(item.msg, val, {priority: 'event'});
                            }, 500);

                            // Update display on drag AND trigger debounced send
                            el.addEventListener('input', function() {
                                var val = parseFloat(this.value);
                                if(valDisplay) valDisplay.textContent = val.toFixed(2);
                                
                                // Sync with counterpart if exists
                                if (item.syncId) {
                                    var syncEl = document.getElementById(item.syncId);
                                    var syncVal = document.getElementById(item.syncId + '-val');
                                    if (syncEl) syncEl.value = val;
                                    if (syncVal) syncVal.textContent = val.toFixed(2);
                                }

                                // If this is a threshold slider, manual adjustment should disable "Use Optimized" IMMEDIATELY
                                if (item.isThresh) {
                                    var optCheckbox = $('#bias_use_optimized'); 
                                    if (optCheckbox.length === 0) optCheckbox = $(document.getElementById('bias_use_optimized'));
                                    
                                    if (optCheckbox.length && optCheckbox.prop('checked')) {
                                        optCheckbox.prop('checked', false);
                                        Shiny.setInputValue('bias_use_optimized', false, {priority: 'event'});
                                    }
                                }
                                
                                // Send value to server (debounced)
                                sendValue(val);
                            });
                            
                            // Remove change listener (we rely on debounced input now)
                            // el.addEventListener('change', ...); // REMOVED
                        }
                    });
                }

                // Run once on load, and potentially re-run if re-rendered
                $(document).ready(function() {
                     setupBiasToolbar();
                });
                
                // Also trigger when inserted/visible
                $(document).on('shiny:value', function(event) {
                    if (event.name === 'bias_dashboard_content') {
                        setTimeout(setupBiasToolbar, 100);
                    }
                });

                // Restore bias session controls (model selects, threshold slider)
                Shiny.addCustomMessageHandler('restore_bias_session_controls', function(data) {
                    if (data.bias_model_key) {
                        var sel = document.getElementById('bias_model_key');
                        if (sel) {
                            sel.value = data.bias_model_key;
                            Shiny.setInputValue('bias_model_key', data.bias_model_key, {priority: 'event'});
                        }
                    }
                    if (data.bias_model_key_B) {
                        var selB = document.getElementById('bias_model_key_B');
                        if (selB) {
                            selB.value = data.bias_model_key_B;
                            Shiny.setInputValue('bias_model_key_B', data.bias_model_key_B, {priority: 'event'});
                        }
                    }
                    // Old custom threshold logic removed. New thresholds handled by setupBiasToolbar and individual inputs.
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
        {"class": "content bias-content", "style": "margin-top: 19px !important;"},

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
            .accordion-panel-badge.validation {
                background: rgba(245, 158, 11, 0.15) !important;
                color: #f59e0b !important;
                border: 1px solid rgba(245, 158, 11, 0.3) !important;
            }
            .accordion-panel-badge.benchmark {
                background: rgba(6, 182, 212, 0.15) !important;
                color: #06b6d4 !important;
                border: 1px solid rgba(6, 182, 212, 0.3) !important;
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

        ui.tags.style("""
            #bias_dashboard_content {
                margin-bottom: 0 !important;
                height: auto;
            }
        """),
        ui.output_ui("bias_dashboard_content"),
        ui.tags.script("document.body.style.overflow = 'auto';"),
    )


def create_bias_accordion():
    """Build the accordion panels for bias analysis (rendered after analysis)."""
    from .components import viz_header

    return ui.accordion(
        # Panel 1: Overview & Detection
        ui.accordion_panel(
            ui.span("Overview & Detection", ui.span({"class": "accordion-panel-badge essential"}, "Essential")),
            ui.output_ui("bias_summary"),
            ui.output_ui("bias_spans_table"),
            value="overview"
        ),

        # Panel 2: Technical Analysis
        ui.accordion_panel(
            ui.span("Technical Analysis", ui.span({"class": "accordion-panel-badge technical"}, "Technical")),
            ui.output_ui("token_bias_strip"),
            ui.output_ui("confidence_breakdown"),
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
            ui.output_ui("combined_bias_view"),
            
            # Using custom output_ui for these to handle layout internally (single vs dual columns)
            ui.output_ui("attention_bias_matrix"),
            ui.output_ui("bias_propagation_plot"),
            ui.output_ui("bias_focused_heads_table"),
            value="attention_bias"
        ),

        # Panel 4: Faithfulness & Ablation
        ui.accordion_panel(
            ui.span(
                "Faithfulness & Ablation",
                ui.span({"class": "accordion-panel-badge validation"}, "Validation"),
            ),
            ui.div(
                {"style": "padding: 8px 0;"},
                ui.output_ui("ablation_results_display"),

                # ── Integrated Gradients section ──
                ui.hr(style="border-color: rgba(100,116,139,0.2); margin: 24px 0 16px;"),
                ui.output_ui("ig_results_display"),
            ),
            value="ablation",
        ),

        # Panel 5: StereoSet Evaluation
        ui.accordion_panel(
            ui.span(
                "StereoSet Evaluation",
                ui.span({"class": "accordion-panel-badge benchmark"}, "Benchmark"),
            ),
            ui.output_ui("stereoset_overview"),
            ui.output_ui("stereoset_category_breakdown"),
            ui.output_ui("stereoset_demographic_slices"),
            ui.output_ui("stereoset_head_sensitivity"),
            ui.output_ui("stereoset_example_explorer"),
            value="stereoset",
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

            ui.span("CONFIGURATIONS", class_="bar-title"),

            ui.div(
                {"class": "controls-row", "style": "display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; width: 100%; gap: 0;"},

                # ── LEFT COL: Layer & Head ──
                ui.div(
                    {"style": "display: flex; align-items: center; justify-content: flex-end; gap: 16px; padding-right: 12px;"},

                    # Layer slider
                    ui.div(
                        {"class": "control-group"},
                        ui.span("Layer", class_="control-label"),
                        ui.div(
                            {"class": "slider-container"},
                            ui.tags.span("0", id="bias-layer-value", class_="slider-value"),
                            ui.tags.input(type="range", id="bias-layer-slider", min="0", max="11", value="0", step="1"),
                        ),
                    ),

                    # Head slider
                    ui.div(
                        {"class": "control-group"},
                        ui.span("Head", class_="control-label"),
                        ui.div(
                            {"class": "slider-container"},
                            ui.tags.span("0", id="bias-head-value", class_="slider-value"),
                            ui.tags.input(type="range", id="bias-head-slider", min="0", max="11", value="0", step="1"),
                        ),
                    ),

                    # Divider Left
                    ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),
                ),

                # ── CENTER COL: Bias Tokens ──
                ui.div(
                    {"style": "display: flex; justify-content: center; width: 100%; margin-top: 0;"},
                    ui.div(
                        {"class": "control-group", "style": "display: flex; align-items: center; justify-content: center; width: 100%; overflow: visible; margin: 0;"},
                        ui.div(
                            {"id": "bias-tokens-row", "class": "bias-token-container-compact"},
                            ui.output_ui("bias_toolbar_tokens"),
                        ),
                    ),
                ),

                # ── RIGHT COL: Custom Threshold toggle + BAR + Top-K ──
                ui.div(
                    {"style": "display: flex; align-items: center; justify-content: flex-start; gap: 16px; padding-left: 12px;"},

                    # Divider Right
                    ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),

                    ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),

                    # BAR Threshold slider
                    ui.div(
                        {"class": "control-group",
                         "title": "Bias Attention Ratio (BAR) specialization threshold.\n"
                                  "BAR = observed attention to biased tokens / expected under uniform.\n"
                                  "Heads with BAR above this value are marked as 'specialized'.\n"
                                  "= 1.0: head attends uniformly (no bias focus)\n"
                                  "> 1.5: head over-attends to biased tokens (default threshold)\n"
                                  "Lower values detect subtler patterns; higher values are stricter."},
                        ui.span("BAR Threshold", class_="control-label"),
                        ui.div(
                            {"class": "slider-container"},
                            ui.tags.span("1.5", id="bias-bar-threshold-value", class_="slider-value"),
                            ui.tags.input(type="range", id="bias-bar-threshold-slider", min="1.0", max="3.0", value="1.5", step="0.1"),
                        ),
                    ),

                    # Top-K slider
                    ui.div(
                        {"class": "control-group"},
                        ui.span("Top-K", class_="control-label"),
                        ui.div(
                            {"class": "slider-container"},
                            ui.tags.span("5", id="bias-topk-value", class_="slider-value"),
                            ui.tags.input(type="range", id="bias-topk-slider", min="1", max="10", value="5", step="1"),
                        ),
                    ),

                    ui.div({"class": "control-divider", "style": "display:block; width:1px; height:24px; background:rgba(255,255,255,0.1);"}),


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

            /* Outer Scroll & Visual Container */
            .bias-token-container-compact {
                display: block;
                background: rgba(30, 41, 59, 0.6);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                width: auto;
                max-width: 450px;
                max-height: 36px;
                overflow-y: auto;
                overflow-x: hidden;
                box-sizing: border-box;
                padding: 3px 4px;
                height: fit-content;
                align-self: center;
            }

            #bias_toolbar_tokens {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                align-items: center;
                align-content: flex-start;
                gap: 2px;
                width: 100%;
                margin: 0 !important;
                padding: 0 !important;
                height: fit-content;
            }

            #bias-tokens-row > div,
            #bias-tokens-row > div > div {
                margin: 0 !important;
                padding: 0 !important;
            }

            .bias-token-container-compact::-webkit-scrollbar { width: 3px; }
            .bias-token-container-compact::-webkit-scrollbar-track { background: transparent; }
            .bias-token-container-compact::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 2px; }

            .bias-token-chip {
                transition: all 0.15s ease;
            }
            .bias-token-chip:hover {
                transform: translateY(-1px);
                filter: brightness(1.2);
            }
            .bias-token-chip.selected {
                box-shadow: 0 0 0 1.5px currentColor !important;
                filter: brightness(1.3) !important;
                transform: translateY(-1px);
            }
        """),

        # ── Script ──
        ui.tags.script("""
        // ── Token selection for Combined View ──
        window.selectedBiasTokens = new Set();

        window.selectBiasToken = function(idx) {
            var chips = document.querySelectorAll('.bias-token-chip[data-token-idx="' + idx + '"]');

            if (window.selectedBiasTokens.has(idx)) {
                window.selectedBiasTokens.delete(idx);
                chips.forEach(function(c) { c.classList.remove('selected'); });
            } else {
                window.selectedBiasTokens.add(idx);
                chips.forEach(function(c) { c.classList.add('selected'); });
            }

            Shiny.setInputValue('bias_selected_tokens', Array.from(window.selectedBiasTokens), {priority: 'event'});
        };

        window.setBiasHead = function(layer, head) {
            var _set = function(id, val) {
                var el = document.getElementById(id);
                if (!el) return;
                var s = el.selectize || ($(el)[0] && $(el)[0].selectize);
                if (s) s.setValue(val.toString());
                else { el.value = val.toString(); $(el).trigger('change'); }
            };
            _set('bias_attn_layer', layer);
            _set('bias_attn_head', head);
            var lS = document.getElementById('bias-layer-slider');
            var hS = document.getElementById('bias-head-slider');
            if (lS) { lS.value = layer; document.getElementById('bias-layer-value').textContent = layer; }
            if (hS) { hS.value = head; document.getElementById('bias-head-value').textContent = head; }
        };

        (function() {
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
            var setBarThreshold = debounce(function(v){ Shiny.setInputValue('bias_bar_threshold', parseFloat(v), {priority:'event'}); }, 200);

            function bindSlider(sliderId, valId, setter) {
                var el = document.getElementById(sliderId);
                if (el) el.oninput = function() {
                    document.getElementById(valId).textContent = this.value;
                    setter(this.value);
                };
            }
            bindSlider('bias-layer-slider', 'bias-layer-value', setLayer);
            bindSlider('bias-head-slider', 'bias-head-value', setHead);
            bindSlider('bias-bar-threshold-slider', 'bias-bar-threshold-value', setBarThreshold);
            bindSlider('bias-topk-slider', 'bias-topk-value', setTopK);

            // Custom Threshold toggle + slider
            var setCustomThreshold = debounce(function(v){
                Shiny.setInputValue('bias_threshold', parseFloat(v), {priority:'event'});
            }, 200);
            bindSlider('bias-custom-threshold-slider', 'bias-custom-threshold-value', setCustomThreshold);

            var toggle = document.getElementById('bias-custom-threshold-toggle');
            var sliderWrap = document.getElementById('bias-custom-threshold-slider-wrap');
            if (toggle) {
                // Default: optimized thresholds ON (toggle unchecked)
                Shiny.setInputValue('bias_use_optimized', true, {priority:'event'});
                toggle.addEventListener('change', function() {
                    var customOn = this.checked;
                    sliderWrap.style.display = customOn ? 'block' : 'none';
                    Shiny.setInputValue('bias_use_optimized', !customOn, {priority:'event'});
                    if (customOn) {
                        var slider = document.getElementById('bias-custom-threshold-slider');
                        if (slider) Shiny.setInputValue('bias_threshold', parseFloat(slider.value), {priority:'event'});
                    }
                });
            }

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
                Shiny.setInputValue('bias_selected_tokens', [], {priority: 'event'});
            }, 500);

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
