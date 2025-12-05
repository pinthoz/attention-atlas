from shiny import ui


def metric_modal():
    """Create the metric explanation modal.

    Returns:
        A Shiny UI div containing the metric modal structure
    """
    return ui.tags.div(
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
    )


def isa_overlay_modal():
    """Create the ISA (Inter-Sentence Attention) overlay modal.

    Returns:
        A Shiny UI div containing the ISA modal structure
    """
    return ui.tags.div(
        {"id": "isa-overlay-modal", "class": "modal"},
        ui.tags.div(
            {"class": "modal-content"},
            ui.tags.div(
                {"class": "modal-header"},
                ui.tags.h3({"class": "modal-title"}, "Inter-Sentence Attention Details"),
                ui.tags.span({"class": "close-btn", "onclick": "document.getElementById('isa-overlay-modal').style.display='none'"}, "×"),
            ),
            ui.tags.div(
                {"class": "modal-body"},
                ui.tags.div(
                    {"class": "isa-sentence-section", "style": "margin-bottom: 20px;"},
                    ui.tags.h4({"style": "color: #ff5ca9; font-size: 14px; margin-bottom: 8px;"}, "Sentence X (Target)"),
                    ui.tags.p({"id": "isa-sentence-x", "style": "font-size: 13px; line-height: 1.6; color: #cbd5e1; margin-bottom: 16px;"}, ""),
                    ui.tags.h4({"style": "color: #ff5ca9; font-size: 14px; margin-bottom: 8px;"}, "Sentence Y (Source)"),
                    ui.tags.p({"id": "isa-sentence-y", "style": "font-size: 13px; line-height: 1.6; color: #cbd5e1; margin-bottom: 16px;"}, ""),
                    ui.tags.div(
                        {"style": "background: rgba(255,92,169,0.1); border-left: 3px solid #ff5ca9; padding: 12px; margin: 12px 0; border-radius: 6px;"},
                        ui.tags.strong({"style": "color: #ff5ca9;"}, "ISA Score: "),
                        ui.tags.span({"id": "isa-score", "style": "color: #cbd5e1; font-family: 'JetBrains Mono', monospace;"}, ""),
                    ),
                ),
                ui.tags.div(
                    {"class": "isa-explanation", "style": "margin-bottom: 20px; font-size: 13px; line-height: 1.8; color: #cbd5e1;"},
                    ui.tags.h4({"style": "color: #ff5ca9; font-size: 14px; margin-bottom: 8px;"}, "What does this represent?"),
                    ui.tags.p(
                        "This value represents the maximum attention strength between any token in Sentence X and any token in Sentence Y, aggregated across all heads and layers."
                    ),
                    ui.tags.h4({"style": "color: #ff5ca9; font-size: 14px; margin-bottom: 8px; margin-top: 16px;"}, "Interpretation"),
                    ui.tags.ul(
                        {"style": "margin: 0; padding-left: 20px;"},
                        ui.tags.li({"style": "margin-bottom: 6px;"}, ui.tags.strong("High ISA"), " → strong dependency across sentences (semantic or syntactic connection)"),
                        ui.tags.li("Low ISA → weak or no cross-sentence influence"),
                    ),
                ),
                ui.tags.h4({"style": "color: #ff5ca9; font-size: 14px; margin-bottom: 12px;"}, "Token-to-Token Attention"),
                ui.tags.div(
                    {"id": "isa-heatmap-container", "style": "min-height: 400px;"},
                    # ui.output_image("isa_token_view", height="400px", width="100%") # Removed to avoid duplicate ID with server.py widget
                ),
            ),
        ),
    )


__all__ = ["metric_modal", "isa_overlay_modal"]
