import base64
from pathlib import Path

from shiny import ui
from shinywidgets import output_widget


# Reusable function for mini selects
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
        :root {
            --primary-color: #ff5ca9;
            --primary-hover: #ff74b8;
            --bg-color: #f5f7fb;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --sidebar-bg: #1e1e2e;
            --sidebar-text: #e2e8f0;
        }

        body {
            background-color: var(--bg-color);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            color: var(--text-main);
        }

        /* Sidebar Styling */
        .sidebar {
            position: fixed;
            left: 0; top: 0; bottom: 0;
            width: 320px;
            background: var(--sidebar-bg);
            color: var(--sidebar-text);
            padding: 24px;
            overflow-y: auto;
            box-shadow: 4px 0 24px rgba(0,0,0,0.1);
            z-index: 100;
        }

        .sidebar .app-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }

        .sidebar .app-title img {
            width: 32px;
            height: 32px;
            border-radius: 8px;
        }

        .sidebar h3 {
            color: var(--primary-color);
            font-weight: 700;
            margin: 0;
            font-size: 20px;
            letter-spacing: -0.5px;
        }
        
        .sidebar .app-subtitle {
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 24px;
            line-height: 1.4;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .sidebar-section {
            margin-bottom: 24px;
        }

        .sidebar-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #9ca3af;
            font-weight: 600;
            margin-bottom: 8px;
            display: block;
        }

        .sidebar p {
            font-size: 13px;
            line-height: 1.5;
            color: #cbd5e1;
            margin: 0 0 16px 0;
        }

        /* Content Area */
        .content {
            margin-left: 320px;
            padding: 32px;
            max-width: 100%;
            box-sizing: border-box;
        }

        /* Buttons & Inputs */
        .btn-primary {
            background: var(--primary-color) !important;
            border: none !important;
            padding: 10px 24px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 999px;
            transition: all 0.2s;
            width: 100%;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(255, 92, 169, 0.2);
        }

        .btn-primary:hover {
            background: var(--primary-hover) !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 8px -1px rgba(255, 92, 169, 0.3);
        }

        .form-control {
            border-radius: 8px;
            border: 1px solid #334155;
            padding: 8px 12px;
            font-size: 13px;
            background: #2d2d44;
            color: white;
            height: auto;
        }

        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.2);
            outline: none;
        }

        /* Sidebar select styling */
        .sidebar select.form-control {
            padding: 6px 12px;
            font-size: 12px;
            height: 32px;
        }

        /* Cards */
        .card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
            margin-bottom: 0; /* Remove bottom margin to avoid gaps */
            border: 1px solid var(--border-color);
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        .card h4 {
            margin: 0 0 16px;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }

        /* Scrollable Containers */
        .card-scroll {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
            min-height: 0;
            max-height: 500px; /* Limit height for large sections */
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-top: 0;
        }

        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--border-color);
            transition: all 0.2s;
            cursor: pointer;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-color: var(--primary-color);
        }

        .metric-label {
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
        }

        /* Token Visualization */
        .token-viz-container {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 12px;
            background: #f8fafc;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .token-viz {
            padding: 4px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            cursor: help;
            transition: all 0.15s;
        }

        .token-viz:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 10;
        }

        /* Loading Spinner */
        .loading-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 12px;
            color: var(--primary-color);
            font-size: 13px;
            font-weight: 500;
        }

        .spinner {
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255, 92, 169, 0.2);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Tables */
        .token-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .token-table th {
            text-align: left;
            padding: 12px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-color);
            background: #f8fafc;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .token-table td {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }

        .token-name {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--primary-color);
            font-size: 11px;
        }

        /* Compact Select - Unified styling for all dropdowns */
        .select-compact,
        .select-mini {
            position: relative;
            display: inline-block;
        }

        .select-compact select,
        .select-mini select {
            appearance: none;
            background-color: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 2px 18px 2px 6px; /* Reduced padding */
            font-size: 11px;
            font-weight: 500;
            color: #475569;
            cursor: pointer;
            transition: all 0.2s;
            height: 22px; /* Reduced height */
            line-height: 16px;
            min-width: fit-content;
            max-width: 120px;
            width: auto;
        }

        .select-compact select:hover,
        .select-mini select:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
        }

        .select-compact::after,
        .select-mini::after {
            content: "▼";
            font-size: 7px;
            color: #94a3b8;
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
        }

        /* --- NEW: Segment Embeddings Table --- */
        .segment-table-clean {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }
        
        .segment-table-clean th {
            text-align: left;
            padding: 8px 12px;
            font-size: 10px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
            position: sticky;
            top: 0;
        }
        
        .segment-table-clean td {
            padding: 6px 12px;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .seg-row-0 {
            background-color: rgba(59, 130, 246, 0.05); /* Subtle blue */
        }
        
        .seg-row-0 td.segment-cell {
            color: #2563eb;
            font-weight: 600;
        }

        .seg-row-1 {
            background-color: rgba(239, 68, 68, 0.05); /* Subtle red */
        }
        
        .seg-row-1 td.segment-cell {
            color: #dc2626;
            font-weight: 600;
        }

        /* Ensure full width for specific containers */
        .qkv-container, .qkv-item {
            width: 100%;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Header Controls */
        .header-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px; /* Reduced margin */
            flex-wrap: nowrap;
            flex-shrink: 0;
        }

        .header-controls h4 {
            margin: 0;
            white-space: nowrap;
        }

        .header-right {
            display: flex;
            gap: 4px; /* Reduced gap */
            align-items: center;
            flex-shrink: 0;
        }
        
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.5);
            backdrop-filter: blur(4px);
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
        .modal-section h4 {
            color: #ff5ca9;
            font-size: 14px;
            margin-bottom: 8px;
        }


        /* Q/K/V Projections - Vertical layout */
        .qkv-container {
            display: flex;
            flex-direction: column;
            gap: 12px; /* Reduced gap */
        }

        .qkv-item {
            background: #f8fafc;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #e2e8f0;
        }

        .qkv-token-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 8px;
        }

        .qkv-row-item {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 4px;
            padding-left: 16px;
        }

        .qkv-label {
            font-size: 10px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            min-width: 12px;
        }

        /* Scaled Dot-Product Attention */
        .scaled-attention-box {
            background: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e2e8f0;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden; /* Prevent double scroll */
        }
        .scaled-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            flex-shrink: 0;
        }
        .scaled-label {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
        }
        .scaled-formula {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #475569;
            margin-bottom: 16px;
            padding: 8px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            flex-shrink: 0;
        }
        .scaled-computations {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto; /* Scroll only this part if needed */
            flex: 1;
            padding-right: 4px; /* Space for scrollbar */
        }
        .scaled-computation-row {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            background: white;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            transition: all 0.2s;
            flex-shrink: 0; /* Prevent shrinking */
        }
        .scaled-computation-row:hover {
            border-color: var(--primary-color);
            box-shadow: 0 2px 8px rgba(255, 92, 169, 0.1);
        }
        .scaled-rank {
            font-size: 11px;
            font-weight: 700;
            color: #64748b;
            min-width: 28px;
            text-align: center;
            background: #f1f5f9;
            border-radius: 6px;
            padding: 4px 6px;
        }
        .scaled-details {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .scaled-connection {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
        }
        .scaled-values {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        .scaled-step {
            font-size: 11px;
            color: #475569;
            font-family: 'JetBrains Mono', monospace;
            background: #f8fafc;
            padding: 4px 8px;
            border-radius: 4px;
            white-space: nowrap;
        }
        .scaled-step b {
            color: var(--primary-color);
            font-weight: 600;
        }

        /* Hidden States & MLM Alignment */
        .equal-height-row {
            display: flex;
            gap: 24px;
            align-items: stretch;
        }
        .equal-height-col {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .full-height-card {
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .full-height-scroll {
            flex: 1;
            overflow-y: auto;
            min-height: 300px; /* Ensure minimum height */
        }

        /* --- NEW: MLM Predictions Grid --- */
        .mlm-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            padding: 8px;
        }

        .mlm-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: all 0.2s;
        }

        .mlm-card:hover {
            border-color: var(--primary-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .mlm-token-header {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            color: var(--primary-color);
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 6px;
            margin-bottom: 4px;
        }

        .mlm-pred-row {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
        }

        .mlm-pred-token {
            font-family: 'JetBrains Mono', monospace;
            color: #334155;
            min-width: 60px;
            cursor: pointer;
            transition: color 0.2s;
        }
        
        .mlm-pred-token:hover {
            color: var(--primary-color);
            font-weight: 700;
        }

        .mlm-bar-bg {
            flex: 1;
            height: 6px;
            background: #f1f5f9;
            border-radius: 999px;
            overflow: hidden;
        }

        .mlm-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            border-radius: 999px;
        }

        .mlm-prob-text {
            font-size: 9px;
            color: #94a3b8;
            min-width: 32px;
            text-align: right;
        }

        /* --- NEW: MLM Details Panel --- */
        .mlm-details-panel {
            display: none;
            margin-top: 8px;
            padding: 8px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            font-size: 11px;
            color: #475569;
        }

        .mlm-math {
            font-family: 'JetBrains Mono', monospace;
            background: white;
            padding: 6px;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
            margin: 4px 0;
            font-size: 10px;
            color: #334155;
        }

        .mlm-step {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2px;
        }

        /* --- NEW: Loading & Sync Rendering --- */
        .loading-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(4px);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease-out;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .loading-spinner-large {
            width: 48px;
            height: 48px;
            border: 4px solid rgba(255, 92, 169, 0.2);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        }

        .loading-text {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }

        /* Hide content while computing */
        .content-hidden {
            opacity: 0;
            transition: opacity 0.1s;
        }
        
        .content-visible {
            opacity: 1;
            transition: opacity 0.5s ease-in;
        }
        """
    ),
    ui.tags.script(
        """
        $(document).on('shiny:connected', function() {
            // Toggle MLM Details
            window.toggleMlmDetails = function(id) {
                var el = document.getElementById(id);
                if (el.style.display === 'none' || el.style.display === '') {
                    el.style.display = 'block';
                } else {
                    el.style.display = 'none';
                }
            };
        });
        """
    ),
    ui.tags.head(
        ui.tags.title("Attention Atlas"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;700&display=swap"),
    ),

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
            ui.input_text_area("text_input", None, "The quick brown fox jumps over the lazy dog.", rows=3),
            ui.div(
                ui.input_action_button("generate_all", "Generate All", class_="btn-primary"),
                ui.div(
                    {"id": "loading_spinner", "class": "loading-container", "style": "display:none;"},
                    ui.div({"class": "spinner"}),
                    ui.span("Processing...")
                ),
            ),
        ),

        ui.div(
            {"class": "sidebar-section"},
            ui.tags.span("Visualization Options", class_="sidebar-label"),
            ui.input_switch("use_mlm", "Show MLM Predictions", value=False),
        )
    ),

    # Main Content
    ui.div(
        {"class": "content"},
        
        # Sentence Preview (Always first)
        ui.div(
            {"class": "card"},
            ui.h4("Sentence Preview"),
            ui.output_ui("preview_text"),
        ),

        # Dashboard Content (Synchronized Rendering)
        ui.output_ui("dashboard_content")
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

    # JavaScript for interactivity and modal
    ui.tags.script(
        """
        // Handle spinner visibility
        $(document).on('shiny:busy', function() {
            // Only show if we are not already showing the custom spinner
            if ($('#loading_spinner').css('display') === 'none') {
                 // Optional: show a global spinner if needed, but we use the sidebar one
            }
        });

        // Custom message handlers
        Shiny.addCustomMessageHandler('start_loading', function(msg) {
            $('#loading_spinner').css('display', 'flex');
            $('#generate_all').prop('disabled', true).css('opacity', '0.7');
        });

        Shiny.addCustomMessageHandler('stop_loading', function(msg) {
            $('#loading_spinner').css('display', 'none');
            $('#generate_all').prop('disabled', false).css('opacity', '1');
        });

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
                    interpretation: 'Low entropy (e.g., < 2) = highly focused attention on few tokens. High entropy (e.g., > 4) = attention broadly distributed across many tokens.',
                    paper: 'Attention Focus metric using entropy from information theory'
                },
                'Sparsity': {
                    formula: 'S<sub>l,h</sub> = (1/n²) Σ<sub>i=1</sub><sup>n</sup> Σ<sub>j</sub><sup>n</sup> 𝟙(A<sub>ij</sub><sup>l,h</sup> < τ)',
                    description: 'Proportion of attention weights below threshold τ = 0.01. Measures how many token connections the head effectively ignores.',
                    interpretation: 'High sparsity (closer to 100%) = selective attention on very few tokens, most connections ignored.',
                    paper: 'Attention Sparsity metric with threshold τ = 0.01'
                },
                'Distribution': {
                    formula: 'Q<sub>0.5</sub><sup>l,h</sup> = median(A<sub>l,h</sub>)',
                    description: 'The median (50th percentile) of all attention weights in the matrix.',
                    interpretation: 'Low median + high max = attention concentrated on few tokens. High median = more evenly distributed.',
                    paper: 'Attention Distribution Attributes using quantiles'
                },
                'Uniformity': {
                    formula: 'U<sub>l,h</sub> = √[(1/n²) Σ<sub>i,j</sub> (A<sub>ij</sub><sup>l,h</sub> - μ<sub>l,h</sub>)²]',
                    description: 'Standard deviation of all attention weights. Measures the variability in the attention distribution.',
                    interpretation: 'High uniformity = high variance, low uniformity = homogeneous attention distribution.',
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
        """
    )
)
