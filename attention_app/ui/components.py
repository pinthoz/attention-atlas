import base64
from pathlib import Path

from shiny import ui


# Reusable function for mini selects
def mini_select(id_, selected="0", options=None):
    """Create a compact select dropdown component.

    Args:
        id_: The element ID for the select
        selected: The initially selected value
        options: Dict of {value: label} pairs for options

    Returns:
        A Shiny UI div containing the select element
    """
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


# Setup icon data URL
_ICON_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "favicon.ico"
try:
    _ICON_DATA = base64.b64encode(_ICON_PATH.read_bytes()).decode()
    ICON_DATA_URL = f"data:image/x-icon;base64,{_ICON_DATA}"
except Exception:
    ICON_DATA_URL = ""


def viz_header(title, definition, tooltip_text, limitation=None, controls=None, subtitle=None, show_calc_title=True):
    """Create a visualization header with semantic documentation and optional controls.
    
    This provides clear explanations of what visualizations show and don't show,
    helping users correctly interpret attention-based visualizations.
    
    Args:
        title: Visualization title (string)
        definition: Brief "This shows X, not Y" explanation
        tooltip_text: Technical calculation explanation for hover
        limitation: Optional 1-line limitation note
        controls: Optional list of UI elements (buttons, etc.) to display in the header
        subtitle: Optional text to display after the info icon (e.g. "(Layer 0 · Head 0)")
        show_calc_title: Whether to show "How is this calculated?" header in tooltip
    
    Returns:
        A Shiny UI div containing the header, info icon, definition, limitation, and controls
    """
    tooltip_content = [ui.HTML(tooltip_text)]
    # if show_calc_title:
    #     tooltip_content.insert(0, ui.tags.strong("How is this calculated?"))

    info_icon = ui.div(
        {"class": "info-tooltip-wrapper"},
        ui.span({"class": "info-tooltip-icon"}, "i"),
        ui.div(
            {"class": "info-tooltip-content"},
            *tooltip_content
        )
    )
    
    # Header row with title, info icon, subtitle, and optional controls
    header_content = [ui.h4(ui.HTML(title)), info_icon]
    
    if subtitle:
        header_content.append(
            ui.span(subtitle, style="font-size: 13px; color: #64748b; font-weight: 500; margin-left: 4px;")
        )
    
    if controls:
        header_content.append(
            ui.div(
                *controls,
                {"class": "header-controls", "style": "margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end;"}
            )
        )
    
    # Conditional spacing: Compact if controls exist (BERT), Expanded if not (GPT-2)
    if controls:
        pt = "4px"  # "Mete mais para cima" for BERT
        mt = "8px"
    else:
        pt = "18px"
        mt = "16px" # "Diminui o espaço" for GPT-2

    header_row = ui.div(
        {"class": "viz-header-with-info", "style": f"display: flex; align-items: center; gap: 8px; width: 100%; flex-wrap: wrap; padding-top: {pt}; min-height: 32px;"},
        *header_content
    )
    
    definition_box = ui.div(
        {"class": "viz-definition", "style": f"font-size:11px; color:#6b7280; margin-bottom:8px; margin-top: {mt};"},
        definition
    )
    
    elements = [header_row, definition_box]
    
    if limitation:
        limitation_box = ui.div(
            {"class": "viz-limitation"},
            limitation
        )
        elements.append(limitation_box)
    
    return ui.div(*elements)


__all__ = ["mini_select", "ICON_DATA_URL", "viz_header"]

