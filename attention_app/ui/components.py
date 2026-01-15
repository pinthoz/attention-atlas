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


def viz_header(title, definition, tooltip_text, limitation=None):
    """Create a visualization header with semantic documentation.
    
    This provides clear explanations of what visualizations show and don't show,
    helping users correctly interpret attention-based visualizations.
    
    Args:
        title: Visualization title (string)
        definition: Brief "This shows X, not Y" explanation
        tooltip_text: Technical calculation explanation for hover
        limitation: Optional 1-line limitation note
    
    Returns:
        A Shiny UI div containing the header, info icon, definition, and limitation
    """
    info_icon = ui.div(
        {"class": "info-tooltip-wrapper"},
        ui.span({"class": "info-tooltip-icon"}, "ⓘ"),
        ui.div(
            {"class": "info-tooltip-content"},
            ui.tags.strong("How is this calculated?"),
            tooltip_text
        )
    )
    
    header_row = ui.div(
        {"class": "viz-header-with-info"},
        ui.h4(title),
        info_icon
    )
    
    definition_box = ui.div(
        {"class": "viz-definition"},
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

