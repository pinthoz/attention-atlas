import sys
from pathlib import Path

# Fix relative imports when run directly as a script (e.g. via shiny run)
if not __package__ or __package__ == "":
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    import attention_app
    __package__ = "attention_app"

import warnings
warnings.filterwarnings("ignore", message="(?s).*google.generativeai.*", category=FutureWarning)

from shiny import App

from .server import server
from .ui import app_ui

STATIC_PATH = Path(__file__).resolve().parent.parent / "static"

app = App(app_ui, server, static_assets=STATIC_PATH)

__all__ = ["app"]
