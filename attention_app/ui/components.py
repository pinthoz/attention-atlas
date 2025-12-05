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


__all__ = ["mini_select", "ICON_DATA_URL"]
