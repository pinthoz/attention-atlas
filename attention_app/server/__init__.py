"""Server package for Attention Atlas.

This package contains:
- logic.py: Heavy computation and business logic
- renderers.py: HTML generation functions
- main.py: Main server function with reactive logic
"""

from .main import server

__all__ = ["server"]
