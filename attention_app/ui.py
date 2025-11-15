import base64
from pathlib import Path

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

_ICON_PATH = Path(__file__).resolve().parent.parent / "static" / "favicon.ico"
try:
    _ICON_DATA = base64.b64encode(_ICON_PATH.read_bytes()).decode()
    ICON_DATA_URL = f"data:image/x-icon;base64,{_ICON_DATA}"
except Exception:
    ICON_DATA_URL = ""


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
        .sidebar .app-title {
            display:flex;
            align-items:center;
            gap:10px;
            margin-bottom:12px;
        }
        .sidebar .app-title img {
            width:28px;
            height:28px;
            border-radius:6px;
        }
        .sidebar h3 {
            color:#ff5ca9;
            font-weight:700;
            margin:0;
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
        .grid-2 {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:16px;
        }
        .token-table, .token-table-compact, .token-table-segment {
            width:100%;
            border-collapse:collapse;
            font-size:10px;
        }
        .token-table th, .token-table-compact th, .token-table-segment th {
            text-align:left;
            padding:6px 4px;
            color:#6b7280;
            border-bottom:2px solid #e5e7eb;
            font-size:9px;
            font-weight:600;
            text-transform:uppercase;
        }
        .token-table td, .token-table-compact td, .token-table-segment td {
            padding:6px 4px;
            vertical-align:middle;
            border-bottom:1px solid #f3f4f6;
        }
        .token-table-segment td:last-child {
            text-align:right;
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
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            width: 100% !important;
            margin-bottom: 8px !important;
            flex-wrap: nowrap !important;
            gap: 16px !important;
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
            gap: 8px !important;
            background: linear-gradient(135deg, #f8f9fa 0%, #f1f3f5 100%);
            padding: 5px 11px;
            border-radius: 999px;
            box-shadow: inset 0 0 0 1px #e5e7eb, 0 1px 2px rgba(0,0,0,0.04);
            width: auto !important;
            max-width: fit-content !important;
            flex-shrink: 0 !important;
            flex-grow: 0 !important;
        }

        .selector-label {
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 8.5px;
            font-weight: 600;
            color: #64748b;
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

            width: 44px !important;
            min-width: 44px !important;
            max-width: 44px !important;
            height: 26px !important;

            padding: 0 22px 0 10px !important;
            margin: 0 !important;

            font-size: 12px !important;
            font-weight: 600 !important;
            font-variant-numeric: tabular-nums;

            border: 1.5px solid #d1d5db !important;
            border-radius: 7px !important;
            background: white !important;
            color: #0f172a !important;

            box-shadow: 0 1px 2px rgba(15,23,42,0.05), inset 0 1px 0 rgba(255,255,255,0.9) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;

            flex: none !important;
            display: inline-block !important;
            cursor: pointer !important;
        }

        .select-mini select:hover {
            border-color: #94a3b8 !important;
            box-shadow: 0 2px 5px rgba(15,23,42,0.1), inset 0 1px 0 rgba(255,255,255,0.9) !important;
            transform: translateY(-0.5px) !important;
        }

        .select-mini select:focus {
            outline: none !important;
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 2px 5px rgba(15,23,42,0.1) !important;
            transform: translateY(0) !important;
        }

        .select-mini::after {
            content: "";
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 0;
            border-left: 3.5px solid transparent;
            border-right: 3.5px solid transparent;
            border-top: 4.5px solid #475569;
            pointer-events: none;
            z-index: 1;
            transition: transform 0.2s ease;
        }

        .select-mini select:focus ~ .select-mini::after,
        .select-mini:has(select:focus)::after {
            border-top-color: #6366f1;
        }

        .header-right > .select-mini {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Token Visualization (Inspectus style) */
        .token-viz-container {
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            padding: 2px 0;
        }

        .token-viz {
            padding: 2px 5px;
            border-radius: 4px;
            transition: all 0.2s ease;
            cursor: help;
            border: 1px solid transparent;
            display: inline-block;
        }

        .token-viz:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            border-color: rgba(59, 130, 246, 0.8);
            z-index: 10;
        }

        .js-plotly-plot .plotly .scatterlayer .trace .points path {
            cursor: pointer !important;
        }
        .attention-flow-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .token-instructions {
            border-top: 1px solid #e5e7eb;
            margin-top: 12px;
            padding-top: 12px;
            text-align: center;
        }
        .token-instructions .sub-label {
            text-align: center;
            margin-bottom: 6px;
        }
        #token_buttons {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-top: 8px;
        }
        .projection-layout {
            display: grid;
            grid-template-columns: 2fr 1.3fr;
            gap: 24px;
            align-items: stretch;
        }
        .architecture-flow {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }
        .arch-step {
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            padding: 10px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }
        .arch-step-title {
            font-size: 12px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 4px;
        }
        .arch-step-desc {
            font-size: 10px;
            color: #475569;
            line-height: 1.4;
        }
        .projection-card {
            background: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e2e8f0;
        }
        .projection-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .projection-row-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 4px;
            font-size: 10px;
            font-weight: 600;
            color: #475569;
        }
        .projection-arrow {
            font-size: 18px;
            color: #94a3b8;
            font-weight: 700;
        }
        .projection-preview {
            background: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: inset 0 0 0 1px #e2e8f0;
        }
        .prediction-panel {
            border: 1px dashed #cbd5f5;
            border-radius: 12px;
            padding: 16px;
            background: #fdfcff;
            min-height: 100%;
        }
        .prediction-panel h5 {
            margin: 0 0 10px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: #5b21b6;
        }
        .focus-select select {
            border-radius: 10px;
            border: 1px solid #c7d2fe;
            padding: 6px 12px;
            font-size: 11px;
            background: #f8fafc;
        }
        .scaled-attention-box {
            background: #fefce8;
            border: 1px solid #facc15;
            border-radius: 12px;
            padding: 14px;
        }
        .scaled-attention-row {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            margin-bottom: 4px;
            font-family: monospace;
        }
        .repeat-flow {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
            gap: 8px;
        }
        .repeat-block {
            padding: 8px;
            border-radius: 8px;
            background: #eef2ff;
            text-align: center;
            font-size: 10px;
            font-weight: 600;
            color: #4338ca;
        }
        .repeat-block.current {
            background: #c7d2fe;
            border: 1px solid #4338ca;
            box-shadow: 0 0 0 2px rgba(67,56,202,0.15);
        }
        .prob-line {
            display:flex;
            align-items:center;
            gap:6px;
        }
        .prob-detail-btn {
            border:none;
            background:transparent;
            color:#5b21b6;
            font-size:10px;
            font-weight:600;
            cursor:pointer;
            padding:0;
        }
        .prob-detail-btn:hover {
            text-decoration:underline;
        }
        .prob-detail-box {
            display:none;
            margin-left:70px;
            margin-top:4px;
            font-size:10px;
            background:#f5f3ff;
            border-radius:8px;
            padding:6px 10px;
            color:#4c1d95;
            text-align:left;
        }
        .segment-chips {
            display:flex;
            flex-wrap:wrap;
            gap:4px;
            margin-bottom:8px;
        }
        .segment-chip {
            font-size:10px;
            padding:2px 8px;
            border-radius:999px;
            color:white;
            font-weight:600;
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
    ui.tags.head(
        # Multiple favicon formats for maximum browser compatibility
        ui.tags.link(
            rel="icon",
            type="image/x-icon",
            href="/static/favicon.ico",
        ),
        ui.tags.link(
            rel="shortcut icon",
            href="/static/favicon.ico",
        ),
        # Also try with explicit sizes
        ui.tags.link(
            rel="icon",
            type="image/x-icon",
            sizes="16x16",
            href="/static/favicon.ico",
        ),
        ui.tags.link(
            rel="icon",
            type="image/x-icon",
            sizes="32x32",
            href="/static/favicon.ico",
        ),
        # Apple touch icon for iOS devices
        ui.tags.link(
            rel="apple-touch-icon",
            href="/static/favicon.ico",
        ),
    ),

    # Sidebar (fixed)
    ui.div(
        {"class": "sidebar"},
        ui.div(
            {"class": "app-title"},
            ui.tags.img(src=ICON_DATA_URL or "/favicon.ico", alt="Attention Atlas logo"),
            ui.h3("Attention Atlas"),
        ),
        ui.tags.small("A full architectural map of BERT, centered on multi-head attention patterns, attention metrics, flow propagation, and head-level interpretability."),
        ui.tags.ul(
            ui.tags.li("Token & Positional Embeddings"),
            ui.tags.li("Q/K/V Projections & Multi-Head Attention"),
            ui.tags.li("Attention Metrics (Confidence, Focus, Sparsity, etc.)"),
            ui.tags.li("Residual Connections & Layer Normalization"),
            ui.tags.li("Feed Forward Network"),
            ui.tags.li("Output Predictions (MLM Top-5)"),
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
        ui.input_switch("use_mlm", "Use MLM head for predictions", value=False),
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
                }, 15000);
            });
        """
        ),
    ),

    # Main content
    ui.div(
        {"class": "content"},

        ui.card(
            ui.h4("Sentence Preview"),
            ui.div({"class": "sub-label"}, "Token-level attention visualization (hover for details)"),
            ui.output_ui("preview_text"),
        ),

        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Token Embeddings"),
                ui.div({"class": "sub-label"}, "Contextual word vectors (768 dims)"),
                ui.output_ui("embedding_table"),
            ),
            ui.card(
                ui.h4("Segment Embeddings"),
                ui.div({"class": "sub-label"}, "Token type IDs (A/B sequences)"),
                ui.output_ui("segment_embedding_view"),
            ),
            ui.card(
                ui.h4("Positional Embeddings"),
                ui.div({"class": "sub-label"}, "Sinusoidal position encodings"),
                ui.output_ui("posenc_table"),
            ),
        ),

        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Soma + LayerNorm"),
                ui.div({"class": "sub-label"}, "Word + segment + positional embeddings"),
                ui.output_ui("sum_layernorm_view"),
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
            ui.card(
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Scaled Dot-Product Attention"),
                    ui.div(
                        {"class": "header-right"},
                        ui.span("Token", class_="selector-label"),
                        ui.output_ui("scaled_attention_selector_inline"),
                    ),
                ),
                ui.output_ui("scaled_attention_view"),
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
                ui.div(output_widget("attention_map")),
                ui.div({"class": "attention-flow-panel"}, output_widget("attention_flow")),
            ),
            ui.div(
                {"class": "token-instructions"},
                ui.div(
                    {"class": "sub-label"},
                    "Click on a token below to focus its outgoing attention",
                ),
                ui.div(
                    {"id": "token_buttons"},
                    ui.output_ui("token_selector_buttons"),
                ),
            ),
            ui.div(
                ui.h4("Attention Metrics", style="font-size:14px;margin-bottom:8px;margin-top:16px;"),
                ui.output_ui("metrics_display"),
            ),
        ),

        ui.div(
            {"class": "grid-3"},
            ui.card(
                ui.h4("Add & Norm"),
                ui.div({"class": "sub-label"}, "Residual connection after attention"),
                ui.output_ui("add_norm_view"),
            ),
            ui.card(
                ui.h4("Feed Forward Network"),
                ui.div({"class": "sub-label"}, "Intermediate 3072 dims + projection"),
                ui.output_ui("ffn_view"),
            ),
            ui.card(
                ui.h4("Add & Norm (Post-FFN)"),
                ui.div({"class": "sub-label"}, "Residual connection after FFN"),
                ui.output_ui("add_norm_post_ffn_view"),
            ),
        ),


        ui.div(
            {"class": "grid-2"},
            ui.card(
                ui.h4("Hidden States"),
                ui.output_ui("linear_projection_viz"),
            ),
            ui.card(
                ui.h4("MLM / CLS / Token outputs"),
                ui.output_ui("output_probabilities"),
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
        ui.tags.script("""
function toggleProbDetail(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.style.display = (el.style.display === 'block') ? 'none' : 'block';
}
        """),
    )
)
