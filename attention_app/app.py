"""Construct the Shiny application instance."""

from pathlib import Path

from shiny import App

from .server import server
from .ui import app_ui

STATIC_PATH = Path(__file__).resolve().parent.parent / "static"

app = App(app_ui, server, static_assets=STATIC_PATH)

__all__ = ["app"]
