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
            
            /* Spacing variables */
            --section-gap: 32px;
            --card-padding: 24px;
            --input-gap: 8px;
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
            padding: 24px;
            max-width: calc(100% - 320px);
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: var(--section-gap);
        }
        
        /* Remove old spacing method */
        .content > * {
            margin-bottom: 0 !important;
        }
        
        .content > *:last-child {
            margin-bottom: 0;
        }

        /* Stack for dashboard content */
        .dashboard-stack > * {
            margin-bottom: var(--section-gap);
        }
        
        .dashboard-stack > .shiny-layout-columns {
            margin-bottom: var(--section-gap) !important;
        }
        
        .dashboard-stack > *:last-child {
            margin-bottom: 0 !important;
        }

        /* Ensure consistent spacing for generated content */
        .shiny-html-output {
            margin-bottom: var(--section-gap) !important;
            display: block;
        }
        
        /* Force spacing between dashboard stack elements */
        .dashboard-stack {
            display: flex;
            flex-direction: column;
            gap: var(--section-gap);
        }
        
        .dashboard-stack > * {
            margin-bottom: 0 !important; /* Let gap handle it */
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

        /* Compact selection boxes */
        select.form-control,
        .form-select {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 6px 32px 6px 10px;
            font-size: 13px;
            background: #f5f5f5;
            color: var(--text-main);
            height: auto;
            line-height: 1.4;
            transition: all 0.2s;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2364748b' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 8px center;
            background-size: 12px;
            cursor: pointer;
        }
        
        select.form-control:hover,
        .form-select:hover {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }

        select.form-control:focus,
        .form-select:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
            outline: none;
            background: white;
        }
        
        /* Compact select wrapper */
        .select-compact {
            margin-bottom: 0;
            display: inline-flex;
            align-items: center;
        }
        
        .select-compact .form-group {
            margin-bottom: 0 !important;
            width: auto !important;
        }
        
        .select-compact .shiny-input-container {
            width: auto !important;
            display: inline-block !important;
        }
        
        .select-compact select {
            width: auto;
            min-width: 80px;
        }

        /* Sidebar select styling */
        .sidebar select.form-control {
            padding: 6px 28px 6px 10px;
            font-size: 12px;
            height: 32px;
            background: #2d2d44;
            color: white;
            border-color: #334155;
        }
        
        .sidebar select.form-control:hover {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.1);
        }
        
        .sidebar select.form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.2);
        }

        /* Cards */
        .card {
            background: var(--card-bg);
            border-radius: 16px;
            padding: var(--card-padding);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
            margin-bottom: 0;
            border: 1px solid var(--border-color);
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        /* Compact card for Token Influence Tree */
        .card-compact {
            padding: 16px !important;
            margin-top: 0 !important;
        }
        
        .card-compact h4 {
            margin-bottom: 12px !important;
            font-size: 15px !important;
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
            appearance: none !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            background-image: none !important;
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
            padding-right: 20px; /* Space for custom arrow */
        }
        
        /* Hide default arrow in IE/Edge */
        .select-compact select::-ms-expand,
        .select-mini select::-ms-expand {
            display: none;
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

        /* Segment Embeddings Table */
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
            justify-content: flex-start;
            gap: 16px; /* Added gap to control spacing between h4 and header-right */
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: nowrap;
            flex-shrink: 0;
        }

        .header-controls h4 {
            margin: 0;
            white-space: nowrap;
        }

        .header-right {
            display: flex;
            gap: 8px !important; /* Reduced gap between selectors */
            align-items: center;
            flex-shrink: 0;
        }

        /* Stacked Header Controls (for Radar) */
        .header-controls-stacked {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
            width: 100%;
        }

        .header-row-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }

        .header-row-bottom {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 8px;
        }

        .toggle-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        /* Custom Radio Toggle */
        .shiny-options-group {
            display: inline-flex !important;
            background: #f1f5f9;
            padding: 2px;
            border-radius: 6px;
            gap: 0 !important;
        }

        .shiny-options-group .radio {
            margin: 0 !important;
            padding: 0 !important;
        }

        .shiny-options-group label {
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 600;
            color: #64748b;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .shiny-options-group input[type="radio"] {
            display: none;
        }

        .shiny-options-group input[type="radio"]:checked + span {
            background: white;
            color: var(--primary-color);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Fix for Shiny's default radio structure */
        .shiny-options-group .radio label {
            padding: 4px 12px !important;
            margin: 0 !important;
        }
        
        .shiny-options-group .radio input[type="radio"] {
            position: absolute;
            opacity: 0;
        }
        
        .shiny-options-group .radio input[type="radio"]:checked + span {
            background: white;
            color: var(--primary-color);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border-radius: 4px;
            padding: 2px 8px;
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

        /* MLM Predictions Grid */
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


        /* MLM Details Panel */
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

        /* Radar Chart Explanation */
        .radar-explanation {
            font-size: 11px;
            color: #64748b;
            line-height: 1.6;
            padding: 12px;
            background: #f8fafc;
            border-radius: 8px;
            margin-top: 12px;
            border: 1px solid #e2e8f0;
        }

        /* Token Influence Tree */
        .influence-tree-container {
            padding: 16px;
            background: #f8fafc;
            border-radius: 12px;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 600px;
        }

        .tree-node {
            position: relative;
            margin: 4px 0;
        }

        .tree-node-content {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: white;
            border-radius: 8px;
            border: 2px solid;
            cursor: pointer;
            transition: all 0.2s;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .tree-node-content:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .tree-node-root {
            border-color: #ff5ca9;
            background: linear-gradient(135deg, #fff 0%, #fff5f9 100%);
            font-weight: 700;
        }

        .tree-node-level1 {
            border-color: #3b82f6;
        }

        .tree-node-level2 {
            border-color: #8b5cf6;
        }

        .tree-node-level3 {
            border-color: #06b6d4;
        }

        .tree-children {
            margin-left: 24px;
            border-left: 2px solid #e2e8f0;
            padding-left: 12px;
            margin-top: 4px;
            transition: all 0.3s ease;
        }

        .tree-children.collapsed {
            display: none;
        }

        .tree-toggle {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--primary-color);
            color: white;
            font-size: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
            flex-shrink: 0;
        }

        /* D3.js Tree Visualization Styles */
        .influence-tree-wrapper {
            padding: 4px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .tree-explanation {
            border-top: 1px solid #e2e8f0;
            padding: 12px 0 0 0;
            margin-top: 16px;
            text-align: center;
        }
        
        .tree-explanation strong {
            color: #ff5ca9;
            font-weight: 600;
        }
        
        .tree-viz-container {
            background: #f8fafc;
            border-radius: 12px;
            padding: 10px;
        .tree-viz-container {
            background: #f8fafc;
            border-radius: 12px;
            padding: 10px;
            height: 450px; /* Increased height to fill section better */
            overflow: auto;
        }
        
        .metric-tag {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 10px;
            color: #475569;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        
        .metric-tag:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }
        
        .tree-viz-container svg {
            display: block;
            margin: 0 auto;
        }
        
        .tree-viz-container .node circle {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .tree-viz-container .node circle:hover {
            filter: brightness(1.2) drop-shadow(0 0 8px currentColor);
        }
        
        .tree-viz-container .node text {
            font-family: 'JetBrains Mono', monospace;
            user-select: none;
        }
        
        .tree-viz-container .link {
            transition: all 0.3s ease;
        }
        
        .tree-viz-container .link:hover {
            filter: brightness(1.3);
        }

        .tree-toggle.collapsed {
            transform: rotate(-90deg);
        }

        .tree-toggle:hover {
            background: var(--primary-hover);
            transform: scale(1.1);
        }

        .tree-toggle.collapsed:hover {
            transform: rotate(-90deg) scale(1.1);
        }

        /* Loading & Sync Rendering */

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
            
            // Toggle Tree Node
            window.toggleTreeNode = function(nodeId) {
                var children = document.getElementById(nodeId + '-children');
                var toggle = document.getElementById(nodeId + '-toggle');
                if (children && toggle) {
                    children.classList.toggle('collapsed');
                    toggle.classList.toggle('collapsed');
                }
            };
        });
        """
    ),
    ui.tags.head(
        ui.tags.title("Attention Atlas"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@500;700&display=swap"),
        ui.tags.script(src="https://cdn.plot.ly/plotly-2.24.1.min.js"),
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
        
        # Sentence Preview
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

        // Handle radar mode change - show/hide head selector
        $(document).on('change', 'input[name="radar_mode"]', function() {
            const mode = $(this).val();
            if (mode === 'all') {
                $('#radar_head_selector').hide();
            } else {
                $('#radar_head_selector').show();
            }
        });

        // MLM details toggle
        function toggleMlmDetails(id) {
            const panel = document.getElementById(id);
            if (panel) {
                panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
            }
        }
        window.toggleMlmDetails = toggleMlmDetails;
        """
    ),
    # D3.js library
    ui.tags.script(src="https://d3js.org/d3.v7.min.js"),
    # Tree visualization script - embedded to avoid 404 issues
    ui.tags.script(
        """
        // Token Influence Tree D3.js Visualization
        function renderInfluenceTree(treeData, containerId) {
            d3.select(`#${containerId}`).selectAll("*").remove();
            
            if (!treeData || !treeData.name) {
                d3.select(`#${containerId}`)
                    .append("p")
                    .style("font-size", "11px")
                    .style("color", "#9ca3af")
                    .style("padding", "20px")
                    .text("Generate attention data to view the influence tree.");
                return;
            }
            
            // Vertical tree configuration
            const margin = {top: 80, right: 20, bottom: 20, left: 20}; /* Increased top margin for root node */
            const width = 1000 - margin.right - margin.left;
            const height = 800 - margin.top - margin.bottom;
            
            const colors = {
                root: '#ff5ca9',
                level1: '#3b82f6',
                level2: '#8b5cf6',
                level3: '#06b6d4'
            };
            
            const svg = d3.select(`#${containerId}`)
                .append("svg")
                .attr("width", width + margin.right + margin.left)
                .attr("height", height + margin.top + margin.bottom)
                .style("font", "12px 'Inter', sans-serif");
            
            const g = svg.append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);
            
            // Vertical tree layout
            const tree = d3.tree().size([width, height]);
            const root = d3.hierarchy(treeData);
            
            // Don't collapse - show all nodes expanded by default
            // (removed the collapse logic)
            
            let i = 0;
            update(root);
            
            function update(source) {
                const treeData = tree(root);
                const nodes = treeData.descendants();
                const links = treeData.descendants().slice(1);
                
                // Vertical spacing by depth
                nodes.forEach(d => { d.y = d.depth * 130; }); /* Adjusted vertical spacing */
                
                const node = g.selectAll('g.node')
                    .data(nodes, d => d.id || (d.id = ++i));
                
                const nodeEnter = node.enter().append('g')
                    .attr('class', 'node')
                    .attr("transform", d => `translate(${source.x0 || width/2},${source.y0 || 0})`)
                    .on('click', click);
                
                nodeEnter.append('circle')
                    .attr('class', 'node-circle')
                    .attr('r', 1e-6)
                    .style("fill", d => getNodeColor(d))
                    .style("stroke", d => getNodeColor(d))
                    .style("stroke-width", d => 2 + (d.data.att || 0) * 3)
                    .style("opacity", d => 0.3 + (d.data.att || 0) * 0.7);
                
                nodeEnter.append('text')
                    .attr("dy", d => d.depth === 0 ? "-1.5em" : "-.5em") /* Move root text further up */
                    .attr("text-anchor", "middle")
                    .text(d => d.data.name)
                    .style("fill", d => getNodeColor(d))
                    .style("font-weight", d => d.depth === 0 ? "700" : "500")
                    .style("font-size", d => d.depth === 0 ? "14px" : "12px");
                
                nodeEnter.append('text')
                    .attr("dy", "1.8em")
                    .attr("text-anchor", "middle")
                    .text(d => d.depth > 0 ? `${(d.data.att || 0).toFixed(3)}` : "")
                    .style("fill", "#64748b")
                    .style("font-size", "10px");
                
                nodeEnter.append("title")
                    .text(d => {
                        if (d.depth === 0) return `Root: ${d.data.name}`;
                        return `Token: ${d.data.name}\\nAttention: ${(d.data.att || 0).toFixed(4)}\\nQ·K Similarity: ${(d.data.qk_sim || 0).toFixed(4)}`;
                    });
                
                const nodeUpdate = nodeEnter.merge(node);
                
                nodeUpdate.transition()
                    .duration(750)
                    .attr("transform", d => `translate(${d.x},${d.y})`);
                
                nodeUpdate.select('circle.node-circle')
                    .attr('r', d => 6 + (d.data.att || 0) * 4)
                    .style("fill", d => d._children ? getNodeColor(d) : "#fff")
                    .style("cursor", "pointer");
                
                const nodeExit = node.exit().transition()
                    .duration(750)
                    .attr("transform", d => `translate(${source.x},${source.y})`)
                    .remove();
                
                nodeExit.select('circle').attr('r', 1e-6);
                nodeExit.select('text').style('fill-opacity', 1e-6);
                
                const link = g.selectAll('path.link')
                    .data(links, d => d.id);
                
                const linkEnter = link.enter().insert('path', "g")
                    .attr("class", "link")
                    .attr('d', d => {
                        const o = {x: source.x0 || width/2, y: source.y0 || 0};
                        return diagonal(o, o);
                    })
                    .style("fill", "none")
                    .style("stroke", d => getNodeColor(d))
                    .style("stroke-width", d => 1 + (d.data.att || 0) * 4)
                    .style("opacity", d => 0.2 + (d.data.att || 0) * 0.6);
                
                const linkUpdate = linkEnter.merge(link);
                
                linkUpdate.transition()
                    .duration(750)
                    .attr('d', d => diagonal(d, d.parent));
                
                link.exit().transition()
                    .duration(750)
                    .attr('d', d => {
                        const o = {x: source.x, y: source.y};
                        return diagonal(o, o);
                    })
                    .remove();
                
                nodes.forEach(d => {
                    d.x0 = d.x;
                    d.y0 = d.y;
                });
                
                // Vertical diagonal path
                function diagonal(s, d) {
                    return `M ${s.x} ${s.y}
                            C ${s.x} ${(s.y + d.y) / 2},
                              ${d.x} ${(s.y + d.y) / 2},
                              ${d.x} ${d.y}`;
                }
                
                function click(event, d) {
                    if (d.children) {
                        d._children = d.children;
                        d.children = null;
                    } else {
                        d.children = d._children;
                        d._children = null;
                    }
                    update(d);
                }
            }
            
            function getNodeColor(d) {
                if (d.depth === 0) return colors.root;
                if (d.depth === 1) return colors.level1;
                if (d.depth === 2) return colors.level2;
                return colors.level3;
            }
        }
        
        window.renderInfluenceTree = renderInfluenceTree;
        console.log('D3.js version:', typeof d3 !== 'undefined' ? d3.version : 'not loaded');
        console.log('renderInfluenceTree:', typeof renderInfluenceTree !== 'undefined' ? 'loaded' : 'not loaded');
        """
    )
)
