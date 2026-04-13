"""Shared CSV utilities for the Attention Atlas server."""


def csv_safe(val):
    """Prevent CSV formula injection by prefixing dangerous first characters."""
    s = str(val)
    if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + s
    return s
