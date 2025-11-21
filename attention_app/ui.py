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
            max-width: 1400px;
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
            margin-bottom: 24px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            height: 100%;
        }

        .card h4 {
            margin: 0 0 16px;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Scrollable Containers */
        .card-scroll {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-top: 20px;
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

        /* Segment Embeddings Grid */
        .segment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            padding: 4px;
        }
        
        .segment-column {
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }
        
        .segment-header {
            padding: 8px 12px;
            background: white;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .segment-count {
            font-size: 9px;
            color: #94a3b8;
            font-weight: 500;
        }
        
        .segment-tokens-list {
            padding: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .segment-token {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #475569;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 2px;
            transition: all 0.15s;
        }
        
        .segment-token:hover {
            background: #e2e8f0;
            color: #1e293b;
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
            padding: 2px 20px 2px 6px;
            font-size: 11px;
            font-weight: 500;
            color: #475569;
            cursor: pointer;
            transition: all 0.2s;
            height: 24px;
            line-height: 18px;
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
        .segment-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }
        
        .segment-table th {
            text-align: left;
            padding: 8px;
            font-size: 10px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            border-bottom: 1px solid #e2e8f0;
            background: #f8fafc;
            position: sticky;
            top: 0;
        }
        
        .segment-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #f1f5f9;
        }
        
        .seg-0 {
            background-color: #dbeafe;
            color: #1e40af;
        }
        
        .seg-1 {
            background-color: #ffe4e6;
            color: #9f1239;
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
            margin-bottom: 16px;
            flex-wrap: nowrap;
        }

        .header-controls h4 {
            margin: 0;
            white-space: nowrap;
        }

        .header-right {
            display: flex;
            gap: 4px;
            align-items: center;
            flex-shrink: 0;
        }
        
        /* Removed - now unified with .select-compact above */
        
        /* Token Buttons */
        .token-btn {
            display: inline-block;
            padding: 6px 12px;
            margin: 3px;
            border-radius: 999px;
            font-family: monospace;
            font-size: 12px;
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
            box-shadow: 0 0 12px rgba(0,0,0,0.2);
            transform: scale(1.05);
        }
        .token-btn-reset {
            background: #6b7280 !important;
            color: white;
            border: none;
            padding: 6px 14px;
            margin: 3px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(4px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .fade-in {
            animation: fadeIn 0.4s ease-out forwards;
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
            gap: 16px;
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
        }
        .scaled-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
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
        }
        .scaled-computations {
            display: flex;
            flex-direction: column;
            gap: 12px;
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

        /* --- NEW: Segment Embeddings Grid --- */
        .segment-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 12px;
            padding: 8px;
        }

        .segment-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: all 0.2s;
        }
        
        .segment-card:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .segment-id-badge {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            background: white;
            padding: 4px 8px;
            border-radius: 6px;
            border: 1px solid #f1f5f9;
            align-self: flex-start;
        }

        .segment-tokens {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }

        .segment-token-item {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: #334155;
            background: white;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
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
        


        # Row 1: Token Embeddings, Segment Embeddings, Positional Embeddings (Side-by-side)
        ui.layout_column_wrap(
            ui.div(
                {"class": "card"},
                ui.h4("Token Embeddings"),
                ui.output_ui("embedding_table")
            ),
            ui.div(
                {"class": "card"},
                ui.h4("Segment Embeddings"),
                ui.output_ui("segment_embedding_view")
            ),
            ui.div(
                {"class": "card"},
                ui.h4("Positional Embeddings"),
                ui.output_ui("posenc_table")
            ),
            width=1/3,
        ),

        # Row 2: SUM + LayerNorm, Q/K/V, Scaled Dot-Product Attention (Side-by-side)
        ui.layout_column_wrap(
            ui.div(
                {"class": "card"},
                ui.h4("Sum & Layer Normalization"),
                ui.output_ui("sum_layernorm_view")
            ),
            ui.div(
                {"class": "card"},
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Q/K/V Projections", title="Query / Key / Value Projections"),
                    ui.div(
                        {"class": "select-mini"},
                        ui.output_ui("qkv_layer_selector")
                    )
                ),
                ui.output_ui("qkv_table")
            ),
            ui.div(
                {"class": "card"},
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Scaled Dot-Product Attention"),
                    ui.div(
                        {"class": "header-right"},
                        ui.tags.span("Focus:", style="font-size:10px; font-weight:600; color:#64748b;"),
                        ui.div(
                            {"class": "select-compact"},
                            ui.output_ui("scaled_attention_selector"),
                        )
                    )
                ),
                ui.output_ui("scaled_attention_view"),
            ),
            width=1/3,
        ),

        # Global Metrics (Full Width) - Moved here
        ui.div(
            {"class": "card"},
            ui.h4("Global Attention Metrics"),
            ui.output_ui("metrics_display"),
        ),

        # Row 3: Multi-Head Attention, Attention Flow (Side-by-side)
        ui.layout_column_wrap(
            ui.div(
                {"class": "card"},
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Multi-Head Attention"),
                    ui.div(
                        {"class": "header-right"},
                        ui.div(
                            {"class": "select-mini"},
                            ui.output_ui("att_layer_selector")
                        ),
                        ui.div(
                            {"class": "select-mini"},
                            ui.output_ui("att_head_selector")
                        ),
                    )
                ),
                output_widget("attention_map"),
            ),
            ui.div(
                {"class": "card"},
                ui.div(
                    {"class": "header-controls"},
                    ui.h4("Attention Flow"),
                    ui.div(
                        {"class": "header-right"},
                        ui.tags.span("Filter:", style="font-size:12px; font-weight:600; color:#64748b;"),
                        ui.div(
                            {"class": "select-mini"},
                            ui.output_ui("attention_flow_selector"),
                        )
                    )
                ),
                output_widget("attention_flow"),
            ),
            width=1/2,
        ),

        # Row 4: Add & Norm, FFN, Add & Norm Post FFN (Side-by-side)
        ui.layout_column_wrap(
            ui.div(
                {"class": "card"},
                ui.h4("Add & Norm"),
                ui.output_ui("add_norm_view")
            ),
            ui.div(
                {"class": "card"},
                ui.h4("Feed-Forward Network"),
                ui.output_ui("ffn_view")
            ),
            ui.div(
                {"class": "card"},
                ui.h4("Add & Norm (post-FFN)"),
                ui.output_ui("add_norm_post_ffn_view")
            ),
            width=1/3,
        ),

        # Row 5: Hidden States, Token Outputs (Side-by-side)
        ui.div(
            {"class": "equal-height-row"},
            ui.div(
                {"class": "equal-height-col"},
                ui.div(
                    {"class": "card full-height-card"},
                    ui.h4("Hidden States"),
                    ui.output_ui("layer_output_view")
                )
            ),
            ui.div(
                {"class": "equal-height-col"},
                ui.output_ui("mlm_view_container")
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

    # JavaScript for interactivity and modal
    ui.tags.script(
        """
        // Handle spinner visibility
        $(document).on('shiny:busy', function() {
            $('#loading_spinner').css('display', 'flex');
        });
        $(document).on('shiny:idle', function() {
            $('#loading_spinner').css('display', 'none');
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
