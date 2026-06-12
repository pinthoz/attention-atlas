"""Shared CSV utilities for the Attention Atlas server."""


def csv_safe(val):
    """Make a value safe to embed in a comma-separated line.

    Two concerns:
    - CSV formula injection: prefix dangerous first characters.
    - RFC 4180 quoting: tokens like "," or quotes would otherwise shift
      every column after them (the comma is itself a frequent token).
    """
    s = str(val)
    if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
        s = "'" + s
    if any(c in s for c in (',', '"', '\n', '\r')):
        s = '"' + s.replace('"', '""') + '"'
    return s
