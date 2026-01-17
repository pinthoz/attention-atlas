CSS = """
        :root {
            --primary-color: #ff5ca9;
            --primary-hover: #ff74b8;
            --accent-blue: #3b82f6;
            --bg-color: #f0f4f8;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --sidebar-bg: #0f172a;
            --sidebar-text: #e2e8f0;

            /* Spacing variables */
            --section-gap: 32px;
            --card-padding: 24px;
            --input-gap: 8px;
        }

        code {
            color: var(--primary-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }

        /* Ensure Plotly in modal is visible */
        .modal-content canvas,
        .modal-content .js-plotly-plot {
            width: 100% !important;
            height: auto !important;
        }

        body {
            background-color: var(--bg-color);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            color: var(--text-main);
        }

        /* Navbar Styling */
        /* Navbar Styling - Bottom of Sidebar */
        .navbar {
            left: 0 !important;
            width: 320px !important;
            height: auto !important;
            top: auto !important;
            bottom: 0 !important;
            position: fixed !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 24px 0 !important;
            background: transparent !important;
            z-index: 1002 !important; /* Above sidebar background */
            pointer-events: none; /* Allow clicks through empty areas */
            box-shadow: none !important;
        }

        /* Force explicit centering on all bootstrap containers */
        .navbar .container-fluid {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
        }

        .navbar-collapse {
            display: flex !important;
            justify-content: center !important;
            flex-grow: 1 !important;
            width: 100% !important; /* Force full width */
            margin: 0 !important;
        }
        
        .navbar-brand {
            display: none !important;
        }

        .navbar-nav {
            display: flex !important;
            flex-direction: row !important; /* Side-by-side buttons */
            justify-content: center !important;
            align-items: center !important;
            gap: 12px;
            
            /* Visuals */
            background: transparent;
            padding: 0 24px !important; /* Sidebar padding matches */
            border-radius: 0;
            border: none;
            box-shadow: none;
            backdrop-filter: none;
            
            pointer-events: auto;
            margin: 0 !important;
            width: 100% !important;
            box-sizing: border-box;
        }

        .navbar-nav > li {
            flex: 1 1 0; /* Force equal width ignoring content */
            width: 0;   /* Extra safety for flex layout */
            display: flex;
            min-width: 0; /* Allow shrinking below content size */
        }

        .navbar .nav-link {
            width: 100%;
            text-align: center;
            justify-content: center;
            display: flex !important;
            align-items: center;
            
            border-radius: 999px !important;
            padding: 10px 0 !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            margin: 0 !important;
            letter-spacing: 0.3px;
            
            /* Inactive State (Dark theme friendly) */
            background: rgba(255, 255, 255, 0.05) !important;
            color: #94a3b8 !important; /* Muted text */
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .navbar .nav-link:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-color: rgba(255,255,255,0.2) !important;
        }

        .navbar .nav-link.active {
            /* Active State matches btn-primary */
            background: var(--primary-color) !important;
            color: white !important;
            border-color: var(--primary-color) !important;
            box-shadow: 0 4px 6px -1px rgba(255, 92, 169, 0.2);
        }

        /* Sidebar Styling */
        .sidebar {
            position: fixed;
            left: 0; top: 0;
            bottom: 0;
            width: 320px;
            background: var(--sidebar-bg);
            color: var(--sidebar-text);
            padding: 20px;
            /* overflow-y: auto; Removed by request */
            box-shadow: 4px 0 24px rgba(0,0,0,0.1);
            z-index: 100;
        }

        .sidebar .app-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px; /* Reduced from 12px */
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
            margin-bottom: 16px; /* Reduced from 24px */
            line-height: 1.4;
            padding-bottom: 16px; /* Reduced from 20px */
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .sidebar-section {
            margin-bottom: 16px; /* Reduced from 24px */
        }

        .sidebar-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #9ca3af;
            font-weight: 600;
            margin-bottom: 6px; /* Reduced from 8px */
            display: block;
        }

        /* Centered Switch for Sidebar */
        .centered-switch .form-check {
            padding-left: 0 !important;
            margin-bottom: 0 !important;
            min-height: auto;
            display: flex;
            justify-content: center;
        }

        .centered-switch .form-check-input {
            margin-left: 0 !important;
            float: none !important;
            width: 2.2em !important; /* Slightly larger than standard 2em, but smaller than 3em */
            height: 1.2em !important; /* Proportionate height */
            background-size: 50% 100%; /* Adjust fill */
        }

        /* Pink Toggle Switch - Simple color override */
        .sidebar .form-switch .form-check-input:checked {
            background-color: #ff5ca9;
            border-color: #ff5ca9;
        }

        .sidebar .form-switch .form-check-input:focus {
            box-shadow: 0 0 0 0.25rem rgba(255, 92, 169, 0.25);
            border-color: #ff5ca9;
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
            /* padding-top removed as navbar is now in sidebar */
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

        /* Comparison mode - remove bottom margin from nested cards */
        #dashboard-container-compare .card {
            margin-bottom: 0 !important;
        }

        /* Ensure equal heights for cards in layout columns */
        .shiny-layout-columns {
            display: flex;
            align-items: stretch;
        }

        .shiny-layout-columns > div {
            display: flex;
            flex-direction: column;
        }

        .shiny-layout-columns .card {
            flex: 1;
            min-height: 500px;
        }

        .shiny-layout-columns .card.card-compact-height {
            min-height: 300px !important;
        }

        /* Ensure consistent spacing for generated content */
        .shiny-html-output {
            margin-bottom: var(--section-gap) !important;
            display: block;
            height: 100%; /* Ensure it fills parent */
        }

        /* Specific fix for flex cards to ensure children stretch */
        .flex-card .shiny-html-output {
            display: flex;
            flex-direction: column;
            flex: 1;
            height: 100%;
            margin-bottom: 0 !important;
        }

        .flex-card .shiny-html-output > .card {
            flex: 1;
            height: 100%;
            margin-bottom: 0 !important;
        }

        .shiny-layout-columns .shiny-html-output {
            display: flex;
            flex-direction: column;
            flex: 1;
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

        /* Compare mode: reduce gap for arrow rows */
        #dashboard-container-compare {
            gap: 16px !important;
        }

        /* Ensure content body mimics the flex stack for uniform spacing */
        #compare-content-body {
            display: flex;
            flex-direction: column;
            gap: 16px !important; /* MATCH PARENT GAP */
            width: 100%;
        }
        
        #compare-content-body > * {
            margin-bottom: 0 !important; /* MATCH DASHBOARD STACK BEHAVIOR */
        }
        
        /* Arrows in compare mode should have minimal spacing */
        #compare-content-body .arrow-row {
            margin-top: 5px !important;     /* Natural gap (16px) ensures "slight gap" above */
            margin-bottom: -25px !important; /* Aggressive negative margin tighters space below */
            padding: 0 !important;
            z-index: 10;
            position: relative;
        }

        /* Compare Mode Borders */
        /* Target the card directly inside the wrapper or the card itself */
        .compare-card-a, 
        .compare-wrapper-a .card,
        .compare-wrapper-a > .shiny-html-output > .card {
            border: 2px solid #3b82f6 !important;
            border-radius: 12px !important;
            box-shadow: none !important; /* Remove shadow to avoid visual clutter with strong border */
        }

        .compare-card-b, 
        .compare-wrapper-b .card,
        .compare-wrapper-b > .shiny-html-output > .card {
            border: 2px solid #ff5ca9 !important;
            border-radius: 12px !important;
             box-shadow: none !important;
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
            background: #1e293b;
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

        /* Card with horizontal scroll for Attention Flow */
        .card-scroll-horizontal {
            overflow-x: auto !important;
            overflow-y: hidden !important;
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
            display: flex;
            flex-direction: column;
            background: linear-gradient(135deg, #ffe5f3 0%, #ffd4ec 100%);
            border-radius: 12px;
            padding: 14px 16px 12px 16px;
            border: 1px solid #ffb8de;
            transition: all 0.2s;
            cursor: pointer;
            min-height: 140px;
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
            min-height: 28px; /* Fixed height for alignment */
            display: flex;
            align-items: flex-start; /* Align text to top */
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            flex-grow: 1;
        }

        /* Metric Card Enhancements */
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .metric-badge {
            font-size: 9px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 8px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .metric-gauge-wrapper {
            position: relative;
            margin-top: 12px;
        }

        .gauge-scale-label {
            position: absolute;
            font-size: 8px;
            color: #94a3b8;
            font-weight: 500;
            top: -2px;
        }

        .gauge-scale-label:first-child {
            left: 0;
        }

        .gauge-scale-label:last-child {
            right: 0;
        }

        .metric-gauge-fixed {
            position: relative;
            display: flex;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            overflow: visible;
            margin-top: 10px;
        }

        .gauge-zone {
            height: 100%;
        }

        .gauge-zone:first-child {
            border-radius: 3px 0 0 3px;
        }

        .gauge-zone:last-of-type {
            border-radius: 0 3px 3px 0;
        }

        .gauge-marker {
            position: absolute;
            top: -3px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transform: translateX(-50%);
            border: 2px solid white;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
            transition: left 0.3s ease;
        }

        .metric-badge-container {
            text-align: center;
            margin-top: 10px;
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

        /* Spinner inside primary button (needs to be white) */
        .btn-primary .spinner {
            border-color: rgba(255, 255, 255, 0.3);
            border-top-color: white;
        }

        /* Loading Dots Animation */
        .loading-dots:after {
            content: '.';
            animation: dots 1.5s steps(5, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }

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

        /* Uniform arrow styling - same thickness for all */
        .transition-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 20px; /* Uniform size */
            color: #94a3b8;
            opacity: 0.8;
            transition: all 0.2s ease;
            user-select: none;
            line-height: normal;
            z-index: 1000;
            position: relative;
        }

        .transition-arrow:hover {
            color: #ff5ca9;
            opacity: 1;
            transform: scale(1.3);
        }

        /* Horizontal arrows - vertically centered between cards */
        .arrow-horizontal {
            width: 24px;
            height: auto;
            align-self: stretch;
            display: flex;
            align-items: center;          /* center icon inside arrow box */
            justify-content: center;
            margin: 0 8px;
            flex-shrink: 0;
        }

        /* Single vertical arrow - horizontally centered */
        .arrow-vertical {
            width: 100%;
            height: 24px;
            display: flex;
            justify-content: center; /* Horizontal centering */
            align-items: center;
            margin: -16px auto; /* Negative margin to reduce gap */
        }

        /* Initial arrow indicator */
        .arrow-initial {
            position: absolute;
            left: -45px;
            top: 50%;
            transform: translateY(-70%);
            font-size: 20px;
        }

        /* Return Arrow Styling */
        .return-arrow {
            cursor: pointer;
            opacity: 0.6;
            transition: all 0.2s ease;
        }

        .return-arrow:hover {
            opacity: 1;
            transform: scale(1.02);
        }

        .return-line {
            background-color: #cbd5e1;
            transition: background-color 0.2s ease;
        }

        .return-head {
            color: #cbd5e1;
            transition: color 0.2s ease;
        }

        .return-arrow:hover .return-line {
            background-color: #ff5ca9;
        }

        .return-arrow:hover .return-head {
            color: #ff5ca9;
        }

        /* Flex Layout for Horizontal Arrows */
        .flex-row-container {
            display: flex;
            flex-direction: row;
            gap: 0; /* Minimal gap, let arrows handle margins */
            align-items: stretch; /* Stretch cards to same height */
            width: 100%;
        }

        .flex-card {
            flex: 1; /* Take available space */
            min-width: 0; /* Prevent overflow issues */
            display: flex;
            flex-direction: column;
        }

        .flex-card > .card {
            height: 100%; /* Fill the flex-card container */
            margin-bottom: 0 !important; /* Remove default margins */
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

        .header-controls-responsive {
            display: flex;
            justify-content: flex-start;
            gap: 16px;
            row-gap: 8px;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .header-controls-responsive h4 {
            margin: 0;
            white-space: nowrap;
        }

        .header-right {
            display: flex;
            gap: 8px !important; /* Reduced gap between selectors */
            align-items: center;
            flex-shrink: 0;
        }

        /* Selection boxes container - top-right alignment with wrapping */
        .selection-boxes-container {
            display: flex;
            justify-content: flex-end; /* Align to the right */
            flex-wrap: wrap;  /* Allow wrapping to the next line when needed */
            gap: 8px;
            align-items: center;
            margin-top: 0;
        }

        .selection-box {
            display: inline-flex;
            align-items: center;
        }

        /* Header with selection boxes at top-right */
        .header-with-selectors {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 12px;
        }

        .header-with-selectors h4 {
            margin: 0;
            flex-shrink: 0;
        }

        /* Responsive behavior for smaller screens */
        @media (max-width: 768px) {
            .selection-boxes-container {
                justify-content: flex-start;
                width: 100%;
                margin-top: 8px;
            }

            .header-with-selectors {
                flex-direction: column;
                align-items: flex-start;
            }
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
        .shiny-options-group .radio label span {
             margin-left: 4px;
        }

        /* Pink Bias Tabs Styling */
        .nav-tabs .nav-link {
            color: #64748b; /* Default muted slate */
            font-weight: 600;
            transition: all 0.2s;
        }

        .nav-tabs .nav-link:hover {
            color: var(--primary-color);
            background-color: rgba(255, 92, 169, 0.05);
            border-color: rgba(255, 92, 169, 0.2);
        }

        .nav-tabs .nav-link.active {
            color: var(--primary-color) !important;
            border-color: #e2e8f0 #e2e8f0 white !important;
            background-color: white !important;
            border-top: 2px solid var(--primary-color) !important; /* Top pink accent */
        }

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

        /* Visualization Info & Semantic Documentation */
        .viz-header-with-info {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 0;
        }

        .viz-header-with-info h4 {
            margin: 0 !important;
        }

        .info-tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: transparent;
            border: 1px dashed #94a3b8;
            color: #64748b;
            font-size: 10px;
            font-weight: 600;
            cursor: help;
            transition: all 0.2s ease;
            flex-shrink: 0;
            animation: info-pulse 2.5s ease-in-out infinite;
        }

        @keyframes info-pulse {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(255, 92, 169, 0);
                border-color: #94a3b8;
            }
            50% {
                box-shadow: 0 0 0 2px rgba(255, 92, 169, 0.2);
                border-color: #ff5ca9;
            }
        }

        .info-tooltip-icon:hover {
            background: #ff5ca9;
            border-color: #ff5ca9;
            border-style: solid;
            color: white;
            animation: none;
        }

        /* Tooltip container */
        .info-tooltip-wrapper {
            position: relative;
            display: inline-flex;
        }

        .info-tooltip-content {
            visibility: hidden;
            opacity: 0;
            position: fixed;
            z-index: 9999999;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: #f1f5f9;
            padding: 16px 20px;
            border-radius: 12px;
            font-size: 12px;
            line-height: 1.7;
            width: 380px;
            max-width: 420px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,92,169,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            transition: opacity 0.2s ease, visibility 0.2s ease;
            pointer-events: none;
        }

        .info-tooltip-wrapper:hover .info-tooltip-content {
            visibility: visible;
            opacity: 1;
        }

        .info-tooltip-content strong {
            color: #ff5ca9;
            display: block;
            margin-bottom: 6px;
            font-size: 12px;
        }

        .info-tooltip-content code {
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
        }

        /* Visualization description - with subtle border */
        .viz-description {
            font-size: 11px;
            color: #6b7280;
            line-height: 1.5;
            margin-bottom: 8px;
            padding: 8px 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
        }

        /* Limitation/warning note - inline style */
        .viz-limitation {
            font-size: 10px;
            color: #92400e;
            font-style: italic;
            margin-top: 0;
            margin-bottom: 0;
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
            overflow: hidden;
            max-width: 100%;
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
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }

        .qkv-row-item img.heatmap {
            flex: 1;
            width: 100%;
            height: auto;
            max-width: 100%;
        }

        .qkv-label {
            font-size: 10px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            min-width: 12px;
            flex-shrink: 0;
        }

        /* Scaled Dot-Product Attention */
        .scaled-attention-box {
            background: #f8fafc;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #e2e8f0;
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

        /* MLM Predictions - Prediction Panel Container */
        .prediction-panel {
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            height: 100%; /* Ensure it takes full height of parent */
            flex: 1;
            min-height: 0; /* Critical for flexbox scrolling */
        }

        .prediction-panel .card-scroll {
            flex: 1; /* Allow it to grow to fill remaining space */
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0;
            margin: 0;
            box-sizing: border-box;
            max-height: 500px; /* Enforce same max-height as other sections */
        }

        /* MLM Predictions Grid */
        .mlm-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            padding: 12px; /* Uniform padding */
            box-sizing: border-box;
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
            height: 100%;
            min-height: 500px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .tree-explanation {
            border-top: 1px solid #e2e8f0;
            padding: 12px 0 0 0;
            margin-top: auto;
            text-align: center;
            background: #f8fafc;
        }

        .tree-explanation strong {
            color: #ff5ca9;
            font-weight: 600;
        }


        .tree-viz-container {
            background: #f8fafc;
            border-radius: 12px;
            padding: 10px;
            flex: 1;
            min-height: 300px;
            overflow: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
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

        .metric-tag.specialization {
            background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
            border: 2px solid #ff3d94;
            color: #ffffff;
            font-weight: 700;
            font-size: 11px;
            padding: 8px 16px;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(255, 92, 169, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-tag:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-1px);
        }

        .metric-tag.specialization:hover {
            background: linear-gradient(135deg, #ff3d94 0%, #ff5ca9 100%);
            border-color: #ff2080;
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 6px 20px rgba(255, 92, 169, 0.5);
        }

        .metric-tag.specialization:active {
            transform: translateY(-1px) scale(1.02);
            box-shadow: 0 3px 10px rgba(255, 92, 169, 0.4);
        }

        .tree-viz-container svg {
            display: block;
            margin: auto;
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

        /* Simultaneous Reveal Classes */
        .content-hidden {
            opacity: 0;
            transition: opacity 0.5s ease-in-out;
        }

        .content-visible {
            opacity: 1;
            transition: opacity 0.5s ease-in-out;
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

        /* ISA Layout Adjustments */
        .token-to-token-container {
            height: 500px;
            overflow-y: auto;
            margin-top: 20px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: #ffffff;
        }

        .isa-explanation-block {
            margin-top: 24px;
            padding: 16px;
            background: #f8fafc;
            border-radius: 8px;
            font-size: 13px;
            color: #475569;
            line-height: 1.6;
            border: 1px solid #e2e8f0;
        }

        /* Attention Flow Plot Container - Horizontal Scrollable for Large Inputs */
        #attention_flow {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        #attention_flow .js-plotly-plot {
            display: block !important;
            overflow: visible !important;
        }

        #attention_flow .plotly {
            overflow: visible !important;
        }

        #attention_flow .plotly .main-svg {
            overflow: visible !important;
        }

        /* Prevent Plotly from auto-resizing */
        #attention_flow .plotly-graph-div {
            overflow-x: visible !important;
            overflow-y: visible !important;
        }

        /* ====================================
           BIAS ANALYSIS STYLES
           ==================================== */

        /* Bias Badge Styles */
        .bias-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0 4px;
        }

        .bias-badge-gen {
            background: linear-gradient(135deg, #f97316 0%, #fb923c 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2);
        }

        .bias-badge-unfair {
            background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
        }

        .bias-badge-stereo {
            background: linear-gradient(135deg, #9c27b0 0%, #ba68c8 100%);
            color: white;
            box-shadow: 0 2px 4px rgba(156, 39, 176, 0.2);
        }

        /* Bias Summary Card Enhancements */
        .bias-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }

        /* Bias Level Indicators */
        .bias-level-low {
            border-left: 4px solid #10b981;
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%);
        }

        .bias-level-moderate {
            border-left: 4px solid #f59e0b;
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, transparent 100%);
        }

        .bias-level-high {
            border-left: 4px solid #ef4444;
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%);
        }

        /* Bias Visualization Containers */
        .bias-viz-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        /* Token Bias Highlight Styles */
        .token-biased {
            position: relative;
            display: inline-block;
            padding: 2px 6px;
            margin: 0 2px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            transition: all 0.2s ease;
        }

        .token-biased:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            z-index: 10;
        }

        .token-biased-gen {
            background: rgba(249, 115, 22, 0.15);
            border-bottom: 2px solid #f97316;
        }

        .token-biased-unfair {
            background: rgba(239, 68, 68, 0.15);
            border-bottom: 2px solid #ef4444;
        }

        .token-biased-stereo {
            background: rgba(156, 39, 176, 0.15);
            border-bottom: 2px solid #9c27b0;
        }

        .token-biased-multiple {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(156, 39, 176, 0.15) 100%);
            border-bottom: 2px solid #ff5ca9;
        }

        /* Bias Analysis Info Box */
        .bias-info-box {
            background: linear-gradient(135deg, #fff5f9 0%, #ffe5f3 100%);
            border: 2px solid #ffcce5;
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
        }

        .bias-info-box h5 {
            color: #ff5ca9;
            font-size: 14px;
            font-weight: 700;
            margin: 0 0 8px 0;
        }

        .bias-info-box p {
            color: #64748b;
            font-size: 12px;
            line-height: 1.6;
            margin: 0;
        }

        /* Bias Attention Head Highlighting */
        .head-bias-specialized {
            background: linear-gradient(135deg, rgba(255, 92, 169, 0.2) 0%, rgba(255, 92, 169, 0.1) 100%);
            border: 2px solid #ff5ca9;
            box-shadow: 0 0 12px rgba(255, 92, 169, 0.3);
        }

        /* Bias Propagation Chart Styles */
        .propagation-increase {
            color: #ef4444;
            font-weight: 600;
        }

        .propagation-decrease {
            color: #10b981;
            font-weight: 600;
        }

        .propagation-stable {
            color: #64748b;
            font-weight: 600;
        }

        /* Navbar Styling for Multi-Tab Layout */
        .navbar {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 12px 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }

        .navbar a {
            color: #e2e8f0;
            font-weight: 600;
            font-size: 14px;
            padding: 8px 16px;
            border-radius: 8px;
            transition: all 0.2s;
        }

        .navbar a:hover {
            background: rgba(255, 92, 169, 0.1);
            color: #ff5ca9;
        }

        .navbar a.active {
            background: #ff5ca9;
            color: white;
        }

        /* Bias Tab Content Spacing */
        .nav-content {
            padding: 24px;
        }

        /* Bias Table Enhancements */
        .bias-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .bias-table thead {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        .bias-table th {
            padding: 12px;
            text-align: left;
            font-size: 11px;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e2e8f0;
        }

        .bias-table td {
            padding: 12px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 13px;
            color: #475569;
        }

        .bias-table tbody tr:hover {
            background: #fafafa;
            transition: background 0.15s ease;
        }

        /* Gender Direction Indicators */
        .gender-female {
            color: #ec4899;
            font-weight: 600;
        }

        .gender-male {
            color: #3b82f6;
            font-weight: 600;
        }

        .gender-neutral {
            color: #94a3b8;
        }

        /* Bias Tabs Custom Styling to match Specialization Buttons */
        #bias_tabs > .card-header {
            background: transparent !important;
            border-bottom: none !important;
            padding: 0 0 16px 0 !important;
        }

        #bias_tabs > .card-header .nav {
            border-bottom: none !important;
            gap: 12px; /* Increased gap */
        }
        
        #bias_tabs > .card-header .nav-link {
            background: white !important; /* White bg for better visibility */
            border: 1px solid #cbd5e1 !important; /* Darker border */
            border-radius: 999px !important; /* Fully rounded */
            padding: 8px 16px !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            color: #64748b !important;
            margin-right: 0 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            transition: all 0.2s ease !important;
            display: inline-block !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            width: auto !important;
        }

        #bias_tabs > .card-header .nav-link:hover {
            border-color: var(--primary-color) !important;
            color: var(--primary-color) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        }

        #bias_tabs > .card-header .nav-link.active {
            background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%) !important;
            border: 1px solid #ff3d94 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(255, 92, 169, 0.3) !important;
            padding: 8px 16px !important;
            transform: scale(1.05);
        }
        
        /* Side-by-Side Comparison Styling */
        .comparison-toggle-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Ensure inputs in comparison container are visible */
        .comparison-toggle-container label {
            color: #cbd5e1;
            font-size: 13px;
            font-weight: 500;
        }


        /* Arrow Variants for Comparison Mode */
        .arrow-blue {
            color: #3b82f6 !important; /* Force override grey */
        }
        .arrow-blue:hover {
            color: #2563eb !important;
            transform: scale(1.3);
        }

        .arrow-pink {
            color: #ec4899 !important;
        }
        .arrow-pink:hover {
             color: #db2777 !important;
             transform: scale(1.3);
        }

        """

__all__ = ["CSS"]

