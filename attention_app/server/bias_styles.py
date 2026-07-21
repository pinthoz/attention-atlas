"""Shared style constants for bias UI tooltips, badges, and buttons."""

# Button styles
BTN_STYLE_CSV = (
    "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9;"
    " border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
    " display: inline-flex; align-items: center; justify-content: center;"
    " color: #334155; text-decoration: none;"
)
BTN_STYLE_PNG = (
    "padding: 2px 8px; font-size: 10px; height: 24px; background: #f1f5f9;"
    " border: 1px solid #e2e8f0; border-radius: 4px; cursor: pointer;"
    " display: inline-flex; align-items: center; justify-content: center;"
    " color: #334155;"
)

# Tooltip micro-styles (dark tooltip background, ~380px wide)
TH = "font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#94a3b8;margin:8px 0 4px;display:block;"
TR = "display:flex;gap:7px;align-items:flex-start;margin:2px 0;font-size:11.5px;line-height:1.45;color:#cbd5e1;"
TD = "font-size:8px;margin-top:3px;flex-shrink:0;"
TC = "background:rgba(255,255,255,0.09);border-radius:3px;padding:1px 5px;font-family:JetBrains Mono,monospace;font-size:10px;color:#e2e8f0;"
TS = "border:none;border-top:1px solid rgba(255,255,255,0.08);margin:7px 0;"

# Coloured inline badges
TBG = "background:rgba(34,197,94,0.18);color:#22c55e;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
TBR = "background:rgba(239,68,68,0.18);color:#ef4444;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
TBA = "background:rgba(245,158,11,0.18);color:#f59e0b;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
TBB = "background:rgba(96,165,250,0.18);color:#60a5fa;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"
TBP = "background:rgba(167,139,250,0.18);color:#a78bfa;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;white-space:nowrap;"

# Note / italics footer
TN = "font-size:10.5px;color:#64748b;font-style:italic;line-height:1.4;"


# ── Empty states ─────────────────────────────────────────────────────────
# Two situations that must NOT look alike. A grey "No data" for both makes a
# negative result indistinguishable from a broken panel, which is how the
# first outside user of the tool read it.
#
#   no_detections()      the detector ran and matched nothing. The negative IS
#                        the result, so it is stated as one, with the scope of
#                        the definition that produced it.
#   requires_detections() the panel cannot be computed because it has no input.
#                        Not an empty result - a dependency, named as such.

_EMPTY_WRAP = (
    "padding:16px 18px;border:1px dashed #cbd5e1;border-radius:8px;"
    "background:#f8fafc;line-height:1.55;"
)
_EMPTY_TITLE = "font-size:12.5px;font-weight:600;color:#334155;display:block;margin-bottom:6px;"
_EMPTY_BODY = "font-size:11.5px;color:#64748b;display:block;"


def no_detections_html(thresholds: dict | None = None, note: str = "") -> str:
    """Empty state for a DETECTION panel: ran, matched nothing.

    Says it is a result rather than a failure, and names the scope of the
    definition, so that a sentence carrying bias the detector does not model
    (implicature, role assignment, presupposition) is not read as 'clean'.
    """
    thr = ""
    if thresholds:
        parts = " &middot; ".join(
            f"{k} {v:.2f}" for k, v in thresholds.items() if isinstance(v, (int, float))
        )
        if parts:
            thr = f" at the current thresholds ({parts})"
    extra = f"<span style='{_EMPTY_BODY}margin-top:6px;'>{note}</span>" if note else ""
    return (
        f"<div style='{_EMPTY_WRAP}'>"
        f"<span style='{_EMPTY_TITLE}'>No tokens matched the GUS-Net categories{thr}.</span>"
        f"<span style='{_EMPTY_BODY}'>This is a result, not an error. The detector flags "
        f"<b>explicit generalisations about a group</b>; bias carried by implicature, role "
        f"assignment or presupposition falls outside its definition and will not appear here. "
        f"Lower the thresholds in the sidebar to inspect near-misses.</span>"
        f"{extra}</div>"
    )


def requires_detections_html(what: str = "This view") -> str:
    """Empty state for a DEPENDENT panel: no input to compute from."""
    return (
        f"<div style='{_EMPTY_WRAP}'>"
        f"<span style='{_EMPTY_TITLE}'>Requires at least one detected bias token.</span>"
        f"<span style='{_EMPTY_BODY}'>{what} is computed against the detected tokens, and the "
        f"detector flagged none in this prompt, so there is nothing to compute. This is a "
        f"dependency, not a failed analysis.</span></div>"
    )
