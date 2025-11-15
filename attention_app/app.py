"""Construct the Shiny application instance."""

from shiny import App

from .server import server
from .ui import app_ui

app = App(app_ui, server)

__all__ = ["app"]
