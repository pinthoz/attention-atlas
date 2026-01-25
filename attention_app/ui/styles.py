CSS = """
        /* CRITICAL: Hide compare mode elements IMMEDIATELY to prevent flash on page load */
        #model-a-header,
        #model-b-panel {
            display: none !important;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
        }
        /* Class added by JS when compare_mode is active - overrides the hide rule */
        #model-a-header.compare-active {
            display: block !important;
        }
        #model-b-panel.compare-active {
            display: flex !important;
        }
        #model-a-header.compare-active,
        #model-b-panel.compare-active {
            opacity: 1;
        }

        /* Navbar buttons (Attention/Bias) - Apply styled appearance IMMEDIATELY */
        /* These rules load before Bootstrap can apply defaults */
        .navbar .nav-link {
            background: transparent !important;
            border: 2px solid #ff5ca9 !important;
            color: #ff5ca9 !important;
            border-radius: 9999px !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            padding: 6px 0 !important;
            opacity: 1 !important;
        }
        .navbar .nav-link.active {
            background: #e64090 !important;
            color: #ffffff !important;
            border-color: #e64090 !important;
        }

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
            padding-top: 0 !important; /* Override shiny navbar padding */
            color: var(--text-main);
        }

        /* Reset Bootstrap containers that cause top gap */
        .container-fluid, .tab-content, .tab-pane {
            padding-top: 0 !important;
            margin-top: 0 !important;
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
            padding: 12px 0 !important;
            background: transparent !important;
            z-index: 1002 !important; /* Above sidebar background */
            pointer-events: none; /* Allow clicks through empty areas */
            box-shadow: none !important;
            border-top: none !important;
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
            gap: 16px;
            
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
            
            border-radius: 9999px !important;
            padding: 6px 0 !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            border: 2px solid #ff5ca9 !important;
            margin: 0 !important;
            letter-spacing: 0.3px;
            height: 32px !important; /* Force explicit height */
            box-sizing: border-box !important;
            
            /* Inactive State (Dark theme friendly) */
            background: transparent !important;
            color: #ff5ca9 !important; /* Pink text */
            transition: background 0.2s, color 0.2s, transform 0.2s !important; /* Prevent layout transitions */
        }

        .navbar .nav-link:hover {
            background: rgba(255, 92, 169, 0.1) !important;
            color: #ff5ca9 !important;
            border-color: #ff5ca9 !important;
            transform: translateY(-1px);
        }

        .navbar .nav-link.active {
            /* Active State matches btn-primary */
            background: #e64090 !important;
            color: white !important;
            border-color: #e64090 !important;
            box-shadow: none !important;
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

        /* Word Level Toggle - Compact styling */
        .word-level-toggle .form-check {
            padding: 0 !important;
            margin: 0 !important;
            min-height: auto !important;
            display: flex;
            align-items: center;
        }
        .word-level-toggle .form-check-input {
            margin: 15px 0 0 0 !important;
            width: 28px !important;
            height: 14px !important;
            cursor: pointer;
        }
        .word-level-toggle .form-switch .form-check-input:checked {
            background-color: #0ea5e9;
            border-color: #0ea5e9;
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
            padding-top: 0 !important;
            margin-top: -45px !important; /* Force content up to counter hidden layout spacing */
            /* padding-top removed as navbar is now in sidebar */
            max-width: calc(100% - 320px);
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: var(--section-gap);
            transition: none !important;
        }

        /* Force first card to top */
        .sentence-preview-card {
            margin-top: 0 !important;
            min-height: 140px; /* Prevent collapse during load (stable height) */
            transition: none !important;
        }

        /* ... */

        /* Add padding to content to account for fixed bar */
        .content.has-control-bar {
            /* padding-top already set globally */
            padding-bottom: 110px !important;
        }

        /* Force first card to top */
        .sentence-preview-card {
            margin-top: 0 !important;
        }
        .content > .card:first-child h4:first-child,
        .content > .card:first-child h5:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Floating Control Bar - Dark Theme */
        .floating-control-bar {
            position: fixed;
            bottom: 24px;
            top: auto;
            left: 320px;
            right: 0;
            margin-left: 48px;
            margin-right: 48px;
            z-index: 1000;
            z-index: 1000;
            padding: 18px 20px 6px 20px;
            background: var(--sidebar-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 3px;
        }

        .floating-control-bar .bar-title {
            position: absolute;
            top: 5px;
            left: 0;
            right: 0;
            margin: 0 auto;
            pointer-events: none; /* Let clicks pass through if needed */
            font-size: 9px;
            font-weight: 600;
            color: white;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-align: center;
        }

        .floating-control-bar .controls-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            width: 100%;
        }

        .floating-control-bar .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
            flex-shrink: 0;
        }

        /* Norm control group - horizontal layout with label on left */
        .floating-control-bar .norm-control-group {
            flex-direction: row;
            align-items: center;
            gap: 6px;
        }

        .floating-control-bar .norm-control-group .control-label {
            writing-mode: horizontal-tb;
        }

        .floating-control-bar .control-label {
            font-size: 8px;
            font-weight: 600;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }

        .floating-control-bar .shiny-input-container {
            width: auto !important;
            margin-bottom: 0 !important;
        }

        /* Modern Dark Slider - Smaller */
        .floating-control-bar .slider-container {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .floating-control-bar .slider-value {
            min-width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 700;
            color: #fff;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
            font-family: 'JetBrains Mono', monospace;
        }

        .floating-control-bar input[type="range"] {
            -webkit-appearance: none;
            width: 60px;
            height: 4px;
            background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
            border-radius: 2px;
            outline: none;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.4);
        }

        .floating-control-bar input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
            border: 2px solid rgba(255, 255, 255, 0.2);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(255, 92, 169, 0.4);
            transition: all 0.15s ease;
        }

        .floating-control-bar input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.15);
            box-shadow: 0 2px 10px rgba(255, 92, 169, 0.6);
        }

        .floating-control-bar input[type="range"]::-moz-range-thumb {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff5ca9 0%, #ff74b8 100%);
            border: 2px solid rgba(255, 255, 255, 0.2);
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(255, 92, 169, 0.4);
        }

        .floating-control-bar input[type="range"]::-moz-range-track {
            background: transparent;
            border: 0;
        }

        .floating-control-bar .control-divider {
            display: none;
        }

        /* Radio button group for normalization - vertical layout */
        .floating-control-bar .radio-group {
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 1px;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 6px;
            padding: 2px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .floating-control-bar .radio-option {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 7px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            transition: all 0.15s ease;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            text-align: center;
            line-height: 1.2;
        }

        .floating-control-bar .radio-option:hover {
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.05);
        }

        .floating-control-bar .radio-option.active {
            background: var(--primary-color);
            color: white;
        }

        /* Norm control group with vertical layout */
        .floating-control-bar .norm-control-wrapper {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Rollout layers control - hidden by default */
        .floating-control-bar .rollout-layers-control {
            display: none;
            flex-direction: column;
            gap: 1px;
            margin-left: 4px;
            padding-left: 4px;
            border-left: 1px solid rgba(255, 255, 255, 0.1);
        }

        .floating-control-bar .rollout-layers-control.visible {
            display: flex;
        }

        /* Token sentence preview - flexible width */
        .floating-control-bar .token-sentence {
            display: flex;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 3px;
            padding: 3px 6px;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            flex: 1;
            max-width: 850px;
            min-width: 300px;
            max-height: 54px;
            overflow-y: auto;
            margin-top: 5px; /* Slight offset downwards */
        }

        .floating-control-bar .token-sentence::-webkit-scrollbar {
            width: 3px;
        }

        .floating-control-bar .token-sentence::-webkit-scrollbar-track {
            background: transparent;
        }

        .floating-control-bar .token-sentence::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
        }

        .floating-control-bar .btn-global {
            height: 20px;
            padding: 0 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06); /* Match radio group border */
            border-radius: 5px;
            color: #64748b; /* Match radio option grey */
            transition: all 0.2s;
            margin-top: 0;
            line-height: 1;
        }

        .floating-control-bar .btn-global:hover {
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.06);
        }

        .floating-control-bar .token-chip {
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
            background: rgba(51, 65, 85, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: white;
        }

        .floating-control-bar .token-chip:hover {
            border-color: var(--primary-color);
            background: rgba(255, 92, 169, 0.15);
            color: #fff;
        }

        .floating-control-bar .token-chip.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            box-shadow: 0 2px 6px rgba(255, 92, 169, 0.4);
        }

        /* Global View Button in Floating Bar */
        .floating-control-bar .btn-global {
            padding: 2px 8px;
            font-size: 10px;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 4px;
            transition: all 0.2s;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .floating-control-bar .btn-global:hover {
            background: var(--primary-color);
            border-color: var(--primary-color);
            color: white;
        }
        
        .floating-control-bar .btn-global:active,
        .floating-control-bar .btn-global.active {
            background: var(--primary-color);
            border-color: var(--primary-color);
            color: white;
            box-shadow: 0 0 10px rgba(255, 92, 169, 0.4);
        }

        /* Responsive Floating Bar */
        @media (max-width: 1400px) {
            .floating-control-bar {
                margin-left: 24px;
                margin-right: 24px;
            }
            .floating-control-bar .token-sentence {
                min-width: 200px;
            }
        }

        @media (max-width: 1200px) {
             .floating-control-bar {
                margin-left: 12px;
                margin-right: 12px;
                padding-left: 10px;
                padding-right: 10px;
            }
            .floating-control-bar .controls-row {
                gap: 8px;
            }
             .floating-control-bar .token-sentence {
                min-width: 120px;
            }
            .floating-control-bar input[type="range"] {
                width: 40px;
            }
        }



        /* Add padding to content to account for fixed bar */
        .content.has-control-bar {
            padding-top: 4px !important;
            padding-bottom: 110px !important;
        }

        /* Force first card to top */
        .sentence-preview-card {
            margin-top: -45px !important;
        }
        .content > .card:first-child h4:first-child,
        .content > .card:first-child h5:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
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

        /* Remove margin from shiny-html-output inside dashboard-stack to prevent double spacing */
        .dashboard-stack .shiny-html-output {
            margin-bottom: 0 !important;
        }

        /* Compare mode: reduce gap for arrow rows */
        #dashboard-container-compare {
            gap: 20px !important; /* Reduced from 24px */
        }

        /* Tighter vertical spacing for arrow rows in compare mode */
        #dashboard-container-compare .arrow-row {
            margin-top: -10px !important;
            margin-bottom: -10px !important;
            position: relative;
            top: 5px; /* Center arrows vertically in the margin space */
        }

        /* Last arrow (Exit) needs more top offset */
        #dashboard-container-compare .arrow-row:last-of-type {
            top: 12px;
        }

        /* Space for Input arrow at top and reduced bottom margin in deep dive compare */
        #dashboard-container-compare .accordion-item:last-child .accordion-body {
            padding-top: 40px !important;
            padding-bottom: 15px !important;
        }

        /* Ensure content body mimics the flex stack for uniform spacing */
        #compare-content-body {
            display: flex;
            flex-direction: column;
            gap: 12px !important; /* Reduced from 16px to match Single Mode density */
            width: 100%;
        }

        #compare-content-body > * {
            margin-bottom: 0 !important;
        }

        /* Reduce card padding specifically in compare mode for higher density */
        #compare-content-body .card {
            padding: 16px !important;
        }

        /* Arrows in compare mode should have minimal spacing */
        #compare-content-body .arrow-row {
            margin-top: -16px !important;    /* Tighter gap */
            margin-bottom: -16px !important; /* Tighter gap */
            padding: 0 !important;
            height: 10px; /* Minimal container height */
            z-index: 10;
            position: relative;
            display: flex;
            align-items: center;
        }

        #compare-content-body .arrow-vertical {
            height: 16px !important; /* Smaller arrow area */
            margin: 0 auto !important;
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

        /* Tabbed View Controls */
        .view-controls {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(0,0,0,0.05);
            padding-bottom: 8px;
        }

        .view-btn {
            padding: 1px 8px;
            border-radius: 4px;
            border: 1px solid transparent;
            background: rgba(0,0,0,0.03);
            color: var(--text-muted);
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            line-height: 1.4;
        }

        /* Q/K/V Projections - compact buttons centered */
        .qkv-controls {
            flex-wrap: wrap;
            gap: 4px 6px !important;
            justify-content: center;
        }

        .qkv-controls .view-btn {
            padding: 2px 6px !important;
            font-size: 9px !important;
            letter-spacing: 0.3px !important;
        }

        .view-btn:hover {
            background: rgba(0,0,0,0.06);
            color: var(--text-main);
        }

        .view-btn.active {
            background: var(--primary-color);
            color: white;
            box-shadow: 0 2px 4px rgba(255, 92, 169, 0.3);
        }
        
        .view-pane {
            display: none; /* Hidden by default, JS toggles this */
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(2px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Directional Alignment Visualization */
        .directional-alignment-container {
            padding: 4px 0;
        }

        .heatmap-comparison {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .heatmap-panel {
            flex: 1;
            min-width: 180px;
            max-width: 220px;
            text-align: center;
        }

        .heatmap-title {
            font-size: 11px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 2px;
        }

        .heatmap-subtitle {
            font-size: 9px;
            color: #9ca3af;
            margin-bottom: 8px;
        }

        .heatmap-wrapper {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .comparison-heatmap {
            width: 100%;
            max-width: 180px;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .heatmap-colorbar {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-top: 6px;
            font-size: 8px;
            color: #6b7280;
        }

        .colorbar-gradient {
            width: 60px;
            height: 8px;
            border-radius: 2px;
        }

        .cosine-gradient {
            background: linear-gradient(to right, #8b5cf6, #f8fafc, #ef4444);
        }

        .attention-gradient {
            background: linear-gradient(to right, #f8fafc, #ff5ca9);
        }

        .axis-labels-container {
            margin-top: 12px;
            padding-top: 8px;
            border-top: 1px solid rgba(0,0,0,0.05);
        }

        .axis-label-row {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            align-items: center;
            justify-content: center;
        }

        .axis-label {
            font-size: 8px;
            padding: 1px 4px;
            background: rgba(0,0,0,0.03);
            border-radius: 2px;
            color: #64748b;
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
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-top: 0;
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            background: linear-gradient(135deg, #ffe5f3 0%, #ffd4ec 100%);
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid #ffb8de;
            transition: all 0.2s;
            cursor: pointer;
            min-height: 100px;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-color: var(--primary-color);
        }

        .metric-label {
            font-size: 10px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            min-height: 0; 
            display: flex;
            align-items: center;
        }

        .metric-value {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            line-height: 1.1;
        }

        /* Metric Card Enhancements */
        .metric-header-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 4px;
        }

        .metric-value-row {
            display: flex;
            flex-direction: column;
            gap: 2px;
            margin-bottom: 8px;
            flex-grow: 1;
        }

        .metric-context {
            font-size: 9px;
            color: #64748b;
            line-height: 1.35;
            display: flex;
            flex-direction: column;
            white-space: nowrap;
        }

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
            z-index: 100;
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
            color: #94a3b8;
            font-size: 10px;
            font-weight: 600;
            font-family: 'PT Serif', serif;
            text-transform: lowercase;
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

        /* Global metric info icon (click to show popup) */
        .global-info-icon {
            animation: info-pulse 2.5s ease-in-out infinite;
            color: #64748b;
        }

        .global-info-icon:hover {
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



__all__ = ["CSS"]


        /* Custom Input History UI */
        .custom-input-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            position: relative;
            margin-top: 12px;
            margin-bottom: 40px !important; /* Force space below */
            font-family: 'Inter', sans-serif;
            overflow: visible;
        }

        /* Tabs Row Container */
        .tabs-row {
            display: flex;
            align-items: flex-end;
            position: absolute;
            top: -26px;
            left: 0;
            z-index: 50;
        }

        /* The Tab */
        .history-tab {
            position: relative;
            top: auto;
            left: auto;
            bottom: auto;
            background: #1e293b;
            color: white;
            padding: 4px 12px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
            font-size: 14px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            cursor: pointer;
            z-index: 30;
            transition: background 0.2s ease;
            height: 26px;
            line-height: 1;
            box-shadow: 2px -2px 5px rgba(0,0,0,0.1);
        }

        .history-tab:hover {
            background: #334155;
            z-index: 31;
        }

        /* Compare Prompts Tabs Container */
        .compare-tabs-inline {
            display: flex;
            align-items: flex-end;
        }

        /* Prompt Tab Base Styles */
        .prompt-tab {
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
            transition: opacity 0.2s ease, transform 0.2s ease;
        }

        /* Tab A - Blue */
        .prompt-tab.tab-a {
            background: #3b82f6;
            margin-left: -8px;
            z-index: 20;
            box-shadow: 2px -2px 5px rgba(0,0,0,0.1);
            padding-left: 14px;
            padding-right: 10px;
        }

        .prompt-tab.tab-a:hover {
            background: #2563eb;
        }

        /* Tab B - Pink */
        .prompt-tab.tab-b {
            background: #ff5ca9;
            margin-left: -8px;
            z-index: 10;
            box-shadow: 2px -2px 5px rgba(0,0,0,0.1);
            padding-left: 14px;
            padding-right: 10px;
        }

        .prompt-tab.tab-b:hover {
            background: #f43f8e;
        }

        /* The Textarea */
        .custom-textarea {
            width: 100%;
            min-height: 120px;
            padding: 12px;
            background: white;
            color: #1e293b;
            border: 2px solid white;
            border-radius: 0 8px 8px 8px; /* Top-left sharp to join with tab */
            border-top-left-radius: 0;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            z-index: 10;
            display: block;
            margin-top: 0; /* Ensure no gaps */
        }

        .custom-textarea:focus {
            border-color: white;
            box-shadow: none;
            z-index: 6; 
        }

        /* Compare models mode: colored borders for textareas */
        .compare-mode-active #text_input {
            border: 2px solid #3b82f6;
        }

        .compare-mode-active #text_input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .compare-mode-active #text_input_B {
            border: 2px solid #ff5ca9;
        }

        .compare-mode-active #text_input_B:focus {
            border-color: #ff5ca9;
            box-shadow: 0 0 0 3px rgba(255, 92, 169, 0.1);
        }

        /* Compare prompts mode: colored borders for textareas */
        .compare-prompts-active #text_input {
            border: 2px solid #3b82f6;
        }

        .compare-prompts-active #text_input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .compare-prompts-active #text_input_B {
            border: 2px solid #ff5ca9;
        }

        .compare-prompts-active #text_input_B:focus {
            border-color: #ff5ca9;
            box-shadow: 0 0 0 3px rgba(255, 92, 169, 0.1);
        }

        /* History Dropdown */
        .history-dropdown {
            position: absolute;
            top: 32px; /* Height of tab (approx) */
            left: 0;
            width: 100%; /* Match container width */
            max-height: 250px;
            overflow-y: auto;
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 0 8px 8px 8px;
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.15);
            z-index: 100; /* High z-index to float over everything */
            display: none;
        }

        .history-dropdown.show {
            display: block;
        }

        .history-item {
            padding: 10px 14px;
            border-bottom: 1px solid #f1f5f9;
            font-size: 13px;
            color: #475569;
            cursor: pointer;
            transition: background 0.15s ease;
            white-space: normal; /* Allow wrapping */
            line-height: 1.4;
        }

        .history-item:hover {
            background: #fff1f2;
            color: #be185d;
        }

        }

        /* Hide default navbar toggler (duplicate icon) */
        .navbar-toggler {
            display: none !important;
        }

        /* Dual Row Token Split Layout (Compare Prompts) */
        .floating-control-bar .token-row-split {
            display: flex;
            flex-direction: row; /* Side-by-side */
            flex-wrap: wrap;     /* Allow wrapping */
            align-items: stretch;
            gap: 4px;
            padding: 3px 6px;
            background: rgba(30, 41, 59, 0.6);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            flex: 1;
            max-width: 850px;
            min-width: 300px;
            max-height: 72px; /* Restrict height (approx 2 lines) */
            overflow-y: auto; /* Scroll vertically */
            margin-top: 5px;
        }

        .floating-control-bar .token-row-split::-webkit-scrollbar {
            width: 3px;
        }
        .floating-control-bar .token-row-split::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
        }

        .floating-control-bar .token-split-item {
            display: flex;
            align-items: center;
            gap: 6px;
            overflow-x: auto;
            white-space: nowrap;
            font-size: 11px;
            flex: 1 1 45%; /* Basis 45% allows side-by-side */
            min-width: 250px;
            padding: 2px 0;
            border-bottom: 0;
        }
        
        .floating-control-bar .token-split-item::-webkit-scrollbar {
             height: 2px;
        }
        .floating-control-bar .token-split-item::-webkit-scrollbar-thumb {
             background: rgba(255, 255, 255, 0.1);
        }

        /* Model Labels */
        .floating-control-bar .model-label-a {
            font-weight: bold;
            color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 10px;
        }

        .floating-control-bar .model-label-b {
            font-weight: bold;
            color: #ec4899;
            background: rgba(236, 72, 153, 0.1);
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 10px;
        }

        /* Specific Token Coloring by Prefix */
        .floating-control-bar .token-chip[data-prefix="A"] {
            background: rgba(59, 130, 246, 0.15); /* Blue tint */
            border-color: rgba(59, 130, 246, 0.2);
            color: #eff6ff;
        }

        .floating-control-bar .token-chip[data-prefix="A"]:hover {
            background: rgba(59, 130, 246, 0.3);
            border-color: #3b82f6;
        }

        .floating-control-bar .token-chip[data-prefix="A"].active {
            background: #3b82f6;
            border-color: #3b82f6;
            color: white;
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.4);
        }

        .floating-control-bar .token-chip[data-prefix="B"] {
            background: rgba(236, 72, 153, 0.15); /* Pink tint */
            border-color: rgba(236, 72, 153, 0.2);
            color: #fdf2f8;
        }

        .floating-control-bar .token-chip[data-prefix="B"]:hover {
            background: rgba(236, 72, 153, 0.3);
            border-color: #ec4899;
        }

        .floating-control-bar .token-chip[data-prefix="B"].active {
            background: #ec4899;
            border-color: #ec4899;
            color: white;
            box-shadow: 0 0 8px rgba(236, 72, 153, 0.4);
        }

        /* ==========================================
           Vector Summary Components
           Embedding, Position Encoding, QKV views
           ========================================== */

        .vector-summary-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 8px;
        }

        .vector-summary-section {
            background: #f8fafc;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid var(--border-color);
        }

        .summary-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .summary-icon {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            color: var(--primary-color);
            background: rgba(255, 92, 169, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
            min-width: 32px;
            text-align: center;
        }

        .summary-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-main);
            flex: 1;
        }

        .summary-stat {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            background: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Norm Table Styling */
        .norm-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .norm-table th {
            text-align: left;
            padding: 6px 8px;
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }

        .norm-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #f1f5f9;
        }

        .norm-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-main);
            min-width: 50px;
        }

        .norm-bar-cell {
            width: 40%;
        }

        .norm-bar-bg {
            background: #e2e8f0;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }

        .norm-bar-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #ff5ca9, #e64090);
            transition: width 0.3s ease;
        }

        .pos-norm-fill {
            background: linear-gradient(90deg, #ff5ca9, #8b5cf6);
        }

        .pos-index {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            width: 30px;
            text-align: center;
        }

        /* Similarity Table Styling */
        .sim-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .sim-table th {
            text-align: left;
            padding: 6px 8px;
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }

        .sim-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #f1f5f9;
        }

        .sim-neighbors {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .sim-neighbor {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent-blue);
            padding: 3px 8px;
            border-radius: 4px;
            cursor: default;
            white-space: nowrap;
        }

        .sim-neighbor:hover {
            background: rgba(59, 130, 246, 0.2);
        }

        .sim-neighbor small {
            opacity: 0.7;
            font-size: 9px;
        }

        /* Vector Details Toggle (Collapsible) */
        .vector-details {
            margin-top: 8px;
        }

        .vector-details-toggle {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            padding: 8px 12px;
            background: #f1f5f9;
            border-radius: 6px;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .vector-details-toggle::before {
            content: '▶';
            font-size: 8px;
            transition: transform 0.2s ease;
        }

        .vector-details[open] .vector-details-toggle::before {
            transform: rotate(90deg);
        }

        .vector-details-toggle:hover {
            background: #e2e8f0;
            color: var(--text-main);
        }

        .vector-details[open] .vector-details-toggle {
            border-radius: 6px 6px 0 0;
            margin-bottom: 0;
        }

        .vector-details > .token-table,
        .vector-details > .qkv-container {
            margin-top: 8px;
            padding: 12px;
            background: #fff;
            border-radius: 0 0 6px 6px;
            border: 1px solid var(--border-color);
            border-top: none;
        }

        /* QKV Specific Styles */
        .qkv-stats-row {
            display: flex;
            gap: 12px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .qkv-stat {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            padding: 4px 8px;
            border-radius: 4px;
        }

        .q-stat {
            background: rgba(34, 197, 94, 0.1);
            color: #16a34a;
        }

        .k-stat {
            background: rgba(249, 115, 22, 0.1);
            color: #ea580c;
        }

        .v-stat {
            background: rgba(168, 85, 247, 0.1);
            color: #9333ea;
        }

        .qkv-norm-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        .qkv-norm-table th {
            text-align: left;
            padding: 6px 8px;
            font-size: 10px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border-color);
        }

        .qkv-norm-table td {
            padding: 6px 8px;
            border-bottom: 1px solid #f1f5f9;
        }

        .qkv-norm-cell {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .qkv-mini-bar {
            height: 6px;
            border-radius: 3px;
            max-width: 60px;
            min-width: 4px;
        }

        .q-bar {
            background: linear-gradient(90deg, #22c55e, #16a34a);
        }

        .k-bar {
            background: linear-gradient(90deg, #f97316, #ea580c);
        }

        .v-bar {
            background: linear-gradient(90deg, #a855f7, #9333ea);
        }

        .qkv-norm-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            min-width: 30px;
        }

        .qk-neighbor {
            background: rgba(249, 115, 22, 0.1);
            color: #ea580c;
        }

        .qk-neighbor:hover {
            background: rgba(249, 115, 22, 0.2);
        }

        /* ==========================================
           PCA Visualization Styles
           ========================================== */

        .pca-container {
            background: #fff;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid var(--border-color);
        }

        .pca-svg {
            width: 100%;
            height: auto;
            max-height: 200px;
        }

        .pca-axis {
            stroke: var(--border-color);
            stroke-width: 1;
        }

        .pca-axis-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            fill: var(--text-muted);
            text-anchor: middle;
        }

        .pca-point {
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .pca-point:hover {
            r: 7;
            filter: brightness(1.1);
        }

        /* Color classes for different vector types */
        .pca-embedding {
            fill: var(--primary-color);
        }

        .pca-position {
            fill: #8b5cf6;
        }

        .pca-query {
            fill: #22c55e;
        }

        .pca-key {
            fill: #f97316;
        }

        .pca-value {
            fill: #a855f7;
        }

        .pca-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            fill: var(--text-main);
            pointer-events: none;
        }

        .pca-label-small {
            font-size: 8px;
            fill: var(--text-muted);
        }

        .pca-variance {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            text-align: center;
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border-color);
        }

        .pca-note {
            font-size: 11px;
            color: var(--text-muted);
            text-align: center;
            padding: 16px;
            font-style: italic;
        }

        /* QKV PCA specific styles */
        .qkv-pca-legend {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
            font-weight: 500;
            color: var(--text-muted);
        }

        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .legend-dot.pca-query {
            background: #22c55e;
        }

        .legend-dot.pca-key {
            background: #f97316;
        }

        .legend-dot.pca-value {
            background: #a855f7;
        }

        .qkv-connector {
            stroke: rgba(0, 0, 0, 0.08);
            stroke-width: 1;
            stroke-dasharray: 2, 2;
        }

        /* Combined Summary Table (L2 Norm + Cosine Similarity) */
        .combined-summary-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            table-layout: fixed;
        }

        .combined-summary-table th {
            text-align: left;
            padding: 4px 6px;
            font-size: 9px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            border-bottom: 1px solid var(--border-color);
            background: #f8fafc;
            position: sticky;
            top: 0;
            z-index: 10;
            white-space: nowrap;
        }

        .combined-summary-table th:first-child {
            width: 55px;
            min-width: 55px;
        }
        
        /* Default behavior: allow other columns to size naturally or equally */
        .combined-summary-table th:not(:first-child) {
             min-width: 80px; /* Prevent aggressive truncation like 'GELU Acti...' */
        }

        /* Helper for 50/50 split on data columns (excluding token) */
        .combined-summary-table.distribute-cols th:not(:first-child) {
            width: 45%; 
        }

        .combined-summary-table td {
            padding: 3px 6px;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: middle;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .combined-summary-table .token-name {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--primary-color);
            font-size: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .combined-summary-table .norm-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-main);
            text-align: center;
        }

        .combined-summary-table .sim-neighbors {
            display: flex;
            flex-wrap: wrap;
            gap: 2px;
            overflow: hidden;
        }

        .combined-summary-table .sim-neighbor {
            font-size: 9px;
            padding: 1px 4px;
        }

        /* Position index column for posenc table */
        .combined-summary-table .pos-index {
            width: 25px;
            text-align: center;
            font-size: 9px;
        }

        /* Compact vector summary sections */
        .vector-summary-container {
            gap: 10px;
            padding: 6px;
        }

        .vector-summary-section {
            padding: 8px;
        }

        .summary-header {
            margin-bottom: 6px;
            padding-bottom: 6px;
        }

        /* QKV Combined Table Specific */
        .qkv-combined-table th:first-child {
            width: 55px;
        }

        .qkv-combined-table th:nth-child(2),
        .qkv-combined-table th:nth-child(3),
        .qkv-combined-table th:nth-child(4) {
            width: 35px;
            text-align: center;
        }

        .qkv-combined-table th.q-header {
            color: #16a34a;
        }

        .qkv-combined-table th.k-header {
            color: #ea580c;
        }

        .qkv-combined-table th.v-header {
            color: #9333ea;
        }

        .qkv-combined-table .v-norm {
            color: #9333ea;
        }
        
        /* ==========================================
           Accordion Styling Overrides
           ========================================== */
           
        .accordion-button:not(.collapsed) {
            color: white !important;
            background: #ff5ca9 !important;
            box-shadow: 0 4px 6px -1px rgba(255, 92, 169, 0.2) !important;
        }

        .accordion-button:focus {
            z-index: 3;
            border-color: #ff5ca9;
            outline: 0;
            box-shadow: 0 0 0 0.25rem rgba(255, 92, 169, 0.25) !important;
        }

        .accordion-button:not(.collapsed)::after {
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='white'%3e%3cpath fill-rule='evenodd' d='M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z'/%3e%3c/svg%3e") !important;
        }

        .accordion-item {
            border-radius: 16px !important;
            overflow: hidden !important; /* Needed to clip child borders */
            border: 1px solid #e2e8f0 !important;
            margin-bottom: 16px; /* Spacing between items if desired, matching cards */
            background-color: transparent !important; /* Fix potential white corners */
        }

        .accordion-collapse {
            border-bottom-left-radius: 16px !important;
            border-bottom-right-radius: 16px !important;
        }

        /* Fix corners for first/last items specifically if Bootstrap overrides radius */
        .accordion-item:first-of-type {
            border-top-left-radius: 16px !important;
            border-top-right-radius: 16px !important;
        }

        .accordion-item:last-of-type {
            border-bottom-left-radius: 16px !important;
            border-bottom-right-radius: 16px !important;
        }
        """
