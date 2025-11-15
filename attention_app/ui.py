from shiny import ui
from shinywidgets import output_widget


# Função reutilizável para mini selects
def mini_select(id_, selected="0", options=None):
    if options is None:
        options = {str(i): str(i) for i in range(12)}
    return ui.tags.div(
        {"class": "select-mini"},
        ui.tags.select(
            {"id": id_, "name": id_},
            *[ui.tags.option(label, value=value) for value, label in options.items()],
            selected=selected,
        ),
    )
    


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background-color:#f5f7fb;
            font-family:'Inter', system-ui, -apple-system, sans-serif;
            margin:0;
        }
        .sidebar {
            position:fixed;
            left:0; top:0; bottom:0;
            width:360px;
            background:#1e1e2e;
            color:white;
            padding:28px;
            overflow-y:auto;
            box-shadow:4px 0 24px rgba(0,0,0,0.3);
        }
        .sidebar h3 {
            color:#ff5ca9;
            font-weight:700;
            margin:0 0 8px;
            font-size:24px;
        }
        .sidebar small {
            color:#9ca3af;
            font-size:12px;
            display:block;
            margin-bottom:16px;
        }
        .sidebar ul {
            padding-left:20px;
            margin:0 0 16px;
            font-size:11px;
            color:#cbd5e1;
        }
        .sidebar li {margin-bottom:4px;}
        .content {
            margin-left:380px;
            padding:25px;
        }
        .btn-primary {
            background:#ff5ca9!important;
            border-color:#ff5ca9!important;
            padding:8px 20px;
            font-size:13px;
            font-weight:600;
            border-radius:999px;
            transition:all 0.2s;
        }
        .btn-primary:hover {
            background:#ff74b8!important;
            border-color:#ff74b8!important;
            transform:translateY(-1px);
            box-shadow:0 4px 12px rgba(255,92,169,0.3);
        }
        .btn-container {
            display:flex;
            align-items:center;
            gap:12px;
            margin-top:8px;
        }
        .mini-spinner {
            display:inline-block !important;
            width:20px;
            height:20px;
            border-radius:50%;
            border:3px solid rgba(255,92,169,0.3);
            border-top-color:#ff5ca9;
            border-right-color:#ff74b8;
            animation:spin 0.6s linear infinite;
            vertical-align:middle;
            box-sizing:border-box;
        }
        @keyframes spin {
            0% {transform:rotate(0deg);}
            100% {transform:rotate(360deg);}
        }
        @keyframes pulse {
            0%, 100% {opacity:0.5;}
            50% {opacity:1;}
        }
        .spinner {
            display:inline-block;
            width:22px;
            height:22px;
            border-radius:999px;
            border:3px solid rgba(255,92,169,0.15);
            border-top-color:#ff5ca9;
            border-right-color:#ff74b8;
            animation:spin 0.7s linear infinite;
            box-shadow:0 0 8px rgba(255,92,169,0.3);
        }
        .processing-text {
            font-size:11px;
            color:#ff5ca9;
            font-weight:600;
            letter-spacing:0.5px;
            animation:pulse 1.5s ease-in-out infinite;
        }
        .loading-container {
            display:none;
            align-items:center;
            gap:10px;
            padding:4px 12px;
            background:rgba(255,92,169,0.1);
            border-radius:999px;
            border:1px solid rgba(255,92,169,0.2);
        }
        .card {
            background:white;
            border-radius:15px;
            padding:18px;
            box-shadow:0 2px 8px rgba(0,0,0,0.1);
            margin-bottom:20px;
            overflow-x:auto;
            position:relative;
        }
        .card h4 {
            margin:0 0 12px;
            font-size:15px;
            font-weight:600;
            color:#111827;
        }
        .sub-label {
            font-size:10px;
            color:#6b7280;
            margin-bottom:8px;
        }
        .grid-3 {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:16px;
        }
        .token-table, .token-table-compact {
            width:100%;
            border-collapse:collapse;
            font-size:10px;
        }
        .token-table th, .token-table-compact th {
            text-align:left;
            padding:6px 4px;
            color:#6b7280;
            border-bottom:2px solid #e5e7eb;
            font-size:9px;
            font-weight:600;
            text-transform:uppercase;
        }
        .token-table td, .token-table-compact td {
            padding:6px 4px;
            vertical-align:middle;
            border-bottom:1px solid #f3f4f6;
        }
        .token-name {
            font-weight:600;
            color:#111827;
            min-width:50px;
            font-family:monospace;
            font-size:10px;
        }
        .card-scroll {
            max-height:300px;
            overflow-y:auto;
        }
        img.heatmap {
            border-radius:4px;
            display:block;
            cursor:help;
            width:100%;
            max-width:200px;
        }

        /* ---------- HEADER CONTROLS & MINI SELECTS ---------- */
        .header-controls {
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
            width: auto !important;
            max-width: fit-content !important;
            margin-bottom: 8px !important;
            flex-wrap: nowrap !important;
        }

        .header-controls h4 {
            margin: 0 !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
            flex-grow: 0 !important;
        }

        .header-right {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            background: #f3f4f6;
            padding: 3px 6px;
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px #e5e7eb;
            width: auto !important;
            max-width: fit-content !important;
            flex-shrink: 0 !important;
            flex-grow: 0 !important;
        }

        .selector-label {
            text-transform: uppercase;
            letter-spacing: 0.3px;
            font-size: 8px;
            font-weight: 700;
            color: #6b7280;
        }

        /* MINI SELECTS - FORÇA SOBRE BOOTSTRAP */
        .select-mini {
            position: relative;
            display: inline-block;
        }

        .select-mini select {
            appearance: none !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;

            width: 36px !important;
            min-width: 36px !important;
            max-width: 36px !important;
            height: 20px !important;

            padding: 0 16px 0 6px !important;
            margin: 0 !important;

            font-size: 10px !important;
            font-weight: 600 !important;
            font-variant-numeric: tabular-nums;

            border: 1px solid #d1d5db !important;
            border-radius: 6px !important;
            background: white !important;
            color: #111827 !important;

            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            transition: all 0.15s ease-in-out !important;

            flex: none !important;
            display: inline-block !important;
        }

        .select-mini select:focus {
            outline: none !important;
            border-color: #a5b4fc !important;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
        }

        .select-mini::after {
            content: "";
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 0;
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-top: 4px solid #9ca3af;
            pointer-events: none;
            z-index: 1;
        }

        .header-right > .select-mini {
            margin: 0 !important;
            padding: 0 !important;
        }

        .js-plotly-plot .plotly .scatterlayer .trace .points path {
            cursor: pointer !important;
        }
        .token-btn {
            display: inline-block;
            padding: 4px 10px;
            margin: 3px;
            border-radius: 999px;
            font-family: monospace;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s;
        }
        .token-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .token-btn.active {
            border-color: white;
            box-shadow: 0 0 12px rgba(255,255,255,0.4);
            transform: scale(1.1);
        }
        .token-btn-reset {
            background: #6b7280 !important;
            color: white;
            border: none;
            padding: 4px 14px;
            margin: 3px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .token-btn-reset:hover {
            background: #4b5563 !important;
            transform: translateY(-1px);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .metric-card {
            border-radius: 8px;
            padding: 12px 14px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
            border: 1px solid rgba(0,0,0,0.08);
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            border-color: rgba(0,0,0,0.12);
        }
        .metric-label {
            font-size: 9px;
            color: #78716c;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 20px;
            color: #292524;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }
        .metric-icon {
            position: absolute;
            right: 10px;
            top: 10px;
            font-size: 20px;
            opacity: 0.3;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 99999999 !important;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .modal-content {
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            margin: 5% auto;
            padding: 30px;
            border-radius: 20px;
            width: 90%;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 10px 50px rgba(0,0,0,0.5);
            animation: slideIn 0.3s;
            color: white;
            position: relative;
            z-index: 999999999 !important;
        }
        @keyframes slideIn {
            from {transform: translateY(-50px); opacity: 0;}
            to {transform: translateY(0); opacity: 1;}
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,92,169,0.3);
        }
        .modal-title {
            font-size: 20px;
            font-weight: 700;
            color: #ff5ca9;
        }
        .close-btn {
            font-size: 28px;
            font-weight: bold;
            color: #aaa;
            cursor: pointer;
            transition: color 0.3s;
        }
        .close-btn:hover {
            color: #ff5ca9;
        }
        .modal-body {
            font-size: 13px;
            line-height: 1.8;
            color: #cbd5e1;
        }
        .modal-formula {
            background: rgba(255,92,169,0.1);
            border-left: 3px solid #ff5ca9;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .modal-section {
            margin: 16px 0;
        }
        .modal-section h4 {
            color: #ff5ca9;
            font-size: 14px;
            margin-bottom: 8px;
        }
        """
    ),

    # Sidebar (fixed)
    ui.div(
        {"class": "sidebar"},
        ui.h3("Attention Atlas"),
        ui.tags.small("A full architectural map of BERT, centered on multi-head attention patterns, attention metrics, flow propagation, and head-level interpretability."),
        ui.tags.ul(
            ui.tags.li("Token Embeddings + Positional Encoding"),
            ui.tags.li("Q/K/V projections & Multi-Head Attention"),
            ui.tags.li("6 Attention Metrics (Confidence, Focus, Sparsity, etc.)"),
            ui.tags.li("Add & Norm + Feed Forward Network"),
            ui.tags.li("Linear Projection + Softmax"),
            ui.tags.li("Output Probabilities (Top-5)"),
        ),
        ui.input_text("text_input", "Input sentence:", "the cat sat on the mat"),
        ui.tags.div(
            {"class": "btn-container"},
            ui.input_action_button("generate_all", "Generate All", class_="btn btn-primary"),
            ui.tags.div(
                {"id": "loading_spinner", "class": "loading-container", "style": "display:none;"},
                ui.tags.div({"class": "mini-spinner"}),
                ui.tags.span({"class": "processing-text"}, "Processing..."),
            ),
        ),
        ui.hr(),
        ui.input_switch("use_mlm", "Use MLM head for predictions", value=True),
        ui.tags.small(
            "When enabled, shows real token probabilities from BertForMaskedLM.",
            style="display:block;color:#9ca3af;margin-top:4px;font-size:10px;",
        ),
        ui.hr(),
        ui.p(
            "Hover over heatmaps to inspect vectors",
            style="font-size:11px;color:#cbd5e1;margin-top:12px;",
        ),

        ui.tags.script(
            """
            var waitingToHide = false;
            var plotsRendered = new Set();
            var minimumWaitTime = 3000; // 3 seconds minimum
            var hideTimeout = null;

            function hideSpinner() {
                var spinner = document.getElementById('loading_spinner');
                var btn = document.getElementById('generate_all');
                if (spinner) {
                    spinner.style.display = 'none';
                }
                if (btn) {
                    btn.disabled = false;
                    btn.style.opacity = 1;
                    btn.style.cursor = 'pointer';
                }
                waitingToHide = false;
                plotsRendered.clear();
            }

            // Listen for Plotly plot rendering events
            $(document).on('plotly_afterplot', function(event) {
                if (!waitingToHide) return;

                var plotId = event.target.id;
                plotsRendered.add(plotId);

                // If we have both plots rendered, hide spinner after a short delay
                if (plotsRendered.size >= 2) {
                    if (hideTimeout) clearTimeout(hideTimeout);
                    hideTimeout = setTimeout(hideSpinner, 500);
                }
            });

            Shiny.addCustomMessageHandler('start_loading', function(message) {
                waitingToHide = false;
                plotsRendered.clear();
                if (hideTimeout) clearTimeout(hideTimeout);

                var spinner = document.getElementById('loading_spinner');
                var btn = document.getElementById('generate_all');
                if (spinner) {
                    spinner.style.display = 'flex';
                }
                if (btn) {
                    btn.disabled = true;
                    btn.style.opacity = 0.7;
                    btn.style.cursor = 'not-allowed';
                }
            });

            Shiny.addCustomMessageHandler('stop_loading', function(message) {
                waitingToHide = true;

                // Fallback: hide after 10 seconds no matter what
                hideTimeout = setTimeout(function() {
                    hideSpinner();
                }, 10000);
            });
        """
        ),
    ),

    # Main content
    ui.div(
        {"class": "content"},

        ui.card(
            ui.h4("Sentence Preview"),
            ui.output_text_verbatim("preview_text"),
        ),

        # Embeddings / Positional Encoding / QKV
        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Token Embeddings"),
                ui.div({"class": "sub-label"}, "Contextual word vectors (768 dims)"),
                ui.output_ui("embedding_table"),
            ),
            ui.card(
                ui.h4("Positional Encodings"),
                ui.div({"class": "sub-label"}, "Sinusoidal position encodings"),
                ui.output_ui("posenc_table"),
            ),
            ui.card(
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Q / K / V Projections"),
                    ui.div(
                        {"class": "header-right"},
                        ui.span("Layer", class_="selector-label"),
                        mini_select("qkv_layer", selected="0"),
                    ),
                ),
                ui.output_ui("qkv_table"),
            ),
        ),

        # Multi-head attention
        ui.card(
            ui.div(
                {"class": "header-controls"},
                ui.h4("Multi-Head Attention"),
                ui.div(
                    {"class": "header-right"},
                    ui.span("Layer", class_="selector-label"),
                    mini_select("att_layer", selected="0"),
                    ui.span("Head", class_="selector-label"),
                    mini_select("att_head", selected="0"),
                ),
            ),
            ui.div(
                {"style": "display:grid;grid-template-columns:1fr 1fr;gap:16px;"},
                ui.div(
                    ui.h4("Attention Map", style="font-size:14px;margin-bottom:8px;"),
                    output_widget("attention_map"),
                ),
                ui.div(
                    ui.h4("Attention Flow", style="font-size:14px;margin-bottom:8px;"),
                    ui.div(
                        {"class": "sub-label"},
                        "Click on token buttons below to focus attention",
                    ),
                    ui.div(
                        {"id": "token_buttons", "style": "margin-bottom: 12px;"},
                        ui.output_ui("token_selector_buttons"),
                    ),
                    output_widget("attention_flow"),
                ),
            ),
            ui.div(
                ui.h4("Attention Metrics", style="font-size:14px;margin-bottom:8px;margin-top:16px;"),
                ui.output_ui("metrics_display"),
            ),
        ),

        # Add&Norm / FFN / Layer output
        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Add & Norm"),
                ui.div({"class": "sub-label"}, "Residual connections + LayerNorm"),
                ui.output_ui("add_norm_view"),
            ),
            ui.card(
                ui.h4("Feed Forward"),
                ui.div({"class": "sub-label"}, "FFN layer activation patterns"),
                ui.output_ui("ffn_view"),
            ),
            ui.card(
                ui.h4("Layer Stats"),
                ui.div({"class": "sub-label"}, "Summary statistics per token"),
                ui.output_ui("layer_output_view"),
            ),
        ),

        # Final hidden & MLM Top-5
        ui.card(
            ui.h4("Output Layer: Linear Projection + Softmax"),
            ui.div(
                {"style": "display:grid;grid-template-columns:1fr 2fr;gap:20px;"},
                ui.div(
                    ui.div(
                        {"class": "sub-label"},
                        "Final hidden states (encoder output"),
                    ),
                    ui.output_ui("linear_projection_viz"),
                ),
                ui.div(
                    ui.div(
                        {"class": "sub-label"},
                        "Top-5 Token Predictions with Probabilities",
                    ),
                    ui.output_ui("output_probabilities"),
                ),
            ),
        ),

        # Modal for metric explanations
        ui.tags.div(
            {"id": "metric-modal", "class": "modal"},
            ui.tags.div(
                {"class": "modal-content"},
                ui.tags.div(
                    {"class": "modal-header"},
                    ui.tags.h3({"class": "modal-title", "id": "modal-title"}, "Metric Explanation"),
                    ui.tags.span({"class": "close-btn", "onclick": "document.getElementById('metric-modal').style.display='none'"}, "×"),
                ),
                ui.tags.div({"class": "modal-body", "id": "modal-body"}, "Loading..."),
            ),
        ),

        # JavaScript for modal functionality
        ui.tags.script("""
function showMetricModal(metricName, layer, head) {
    var modal = document.getElementById('metric-modal');
    var title = document.getElementById('modal-title');
    var body = document.getElementById('modal-body');

    title.textContent = metricName;

    var explanations = {
        'Confidence Max': {
            formula: 'C<sub>max</sub><sup>l,h</sup> = max<sub>i,j</sub>(A<sub>ij</sub><sup>l,h</sup>)',
            description: 'The maximum attention weight in the attention matrix. Measures the strongest connection between any query-key pair.',
            interpretation: 'Higher values indicate that this head has a very confident focus on a specific token. Values close to 1 suggest the head is highly specialized and focuses almost exclusively on one token-pair relationship.',
            paper: 'Attention Confidence metric from attention analysis literature'
        },
        'Confidence Avg': {
            formula: 'C<sub>avg</sub><sup>l,h</sup> = (1/n) Σ<sub>i=1</sub><sup>n</sup> max<sub>j</sub>(A<sub>ij</sub><sup>l,h</sup>)',
            description: 'Average of the maximum attention weight per row. Each row represents how a query token attends to all key tokens.',
            interpretation: 'This metric captures the overall confidence level of the attention head. High values (closer to 1) suggest the head consistently focuses strongly on specific tokens for each query, indicating specialized behavior across all positions.',
            paper: 'Attention Confidence metric from attention analysis literature'
        },
        'Focus': {
            formula: 'E<sub>l,h</sub> = -Σ<sub>i=1</sub><sup>n</sup> Σ<sub>j=1</sub><sup>n</sup> A<sub>ij</sub><sup>l,h</sup> log(A<sub>ij</sub><sup>l,h</sup>)',
            description: 'Shannon entropy measures the uncertainty or randomness in the attention distribution. Quantifies how spread out the attention is.',
            interpretation: 'Low entropy (e.g., < 2) = highly focused attention on few tokens. High entropy (e.g., > 4) = attention broadly distributed across many tokens. This reveals whether the head is specialized (low entropy) or generalist (high entropy).',
            paper: 'Attention Focus metric using entropy from information theory'
        },
        'Sparsity': {
            formula: 'S<sub>l,h</sub> = (1/n²) Σ<sub>i=1</sub><sup>n</sup> Σ<sub>j=1</sub><sup>n</sup> 𝟙(A<sub>ij</sub><sup>l,h</sup> < τ)',
            description: 'Proportion of attention weights below threshold τ = 0.01. Measures how many token connections the head effectively ignores.',
            interpretation: 'High sparsity (closer to 100%) = selective attention on very few tokens, most connections ignored. Low sparsity (closer to 0%) = attention distributed across many tokens. Reveals the head selectivity and whether it implements sparse attention patterns.',
            paper: 'Attention Sparsity metric with threshold τ = 0.01'
        },
        'Distribution': {
            formula: 'Q<sub>0.5</sub><sup>l,h</sup> = median(A<sub>l,h</sub>)',
            description: 'The median (50th percentile) of all attention weights in the matrix. A distribution attribute that captures the typical attention value.',
            interpretation: 'Compare median with max to understand attention concentration. Low median + high max = attention highly concentrated on few tokens. High median = attention more evenly distributed. This is one of the quantile-based distribution attributes.',
            paper: 'Attention Distribution Attributes using quantiles (Q_p for p ∈ {0, 0.25, 0.5, 0.75, 1.0})'
        },
        'Uniformity': {
            formula: 'U<sub>l,h</sub> = √[(1/n²) Σ<sub>i,j</sub> (A<sub>ij</sub><sup>l,h</sup> - μ<sub>l,h</sub>)²]',
            description: 'Standard deviation of all attention weights. Measures the variability in the attention distribution, where μ is the mean attention weight.',
            interpretation: 'High uniformity (larger std) = high variance in attention weights, indicating diverse and heterogeneous attention patterns across token pairs. Low uniformity (smaller std) = weights are similar, suggesting uniform/homogeneous attention distribution.',
            paper: 'Attention Uniformity metric measuring distribution variance'
        }
    };

    var info = explanations[metricName];
    if (info) {
        body.innerHTML = `
            <div class="modal-section">
                <h4>Formula</h4>
                <div class="modal-formula">${info.formula}</div>
                <p><em>Layer ${layer}, Head ${head}</em></p>
            </div>
            <div class="modal-section">
                <h4>Description</h4>
                <p>${info.description}</p>
            </div>
            <div class="modal-section">
                <h4>Interpretation</h4>
                <p>${info.interpretation}</p>
            </div>
            <div class="modal-section">
                <h4>Reference</h4>
                <p style="font-size:11px;line-height:1.6;">
                    Golshanrad, Pouria and Faghih, Fathiyeh, <em>From Attention to Assurance: Enhancing Transformer Encoder Reliability Through Advanced Testing and Online Error Prediction</em>.
                    <a href="https://ssrn.com/abstract=4856933" target="_blank" style="color:#ff5ca9;text-decoration:none;border-bottom:1px solid rgba(255,92,169,0.3);">Available at SSRN</a> or
                    <a href="http://dx.doi.org/10.2139/ssrn.4856933" target="_blank" style="color:#ff5ca9;text-decoration:none;border-bottom:1px solid rgba(255,92,169,0.3);">DOI</a>
                </p>
            </div>
        `;
    } else {
        body.innerHTML = '<p>No explanation available for this metric.</p>';
    }

    modal.style.display = 'block';
}

window.onclick = function(event) {
    var modal = document.getElementById('metric-modal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
        """),
    )