"""Server-side reactive logic for the Auditor Notebook.

This module owns:

- the ``notebook_entries`` reactive value (a list of entry dicts);
- the ``Add entry`` / ``Clear form`` / ``Clear all`` actions;
- the rendering of the entries list and the per-entry delete button;
- the JSON and Markdown export downloads;
- a small JSON-backed persistence layer that writes entries to
  ``downloads/sessions/auditor_notebook.json`` so that the analyst does
  not lose work when restarting the app.

The persistence layer is best-effort: any IO error is swallowed so that
the rest of the app keeps running.
"""

from __future__ import annotations

import html as _html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from shiny import reactive, render, ui


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
_NOTEBOOK_PATH = Path("downloads") / "sessions" / "auditor_notebook.json"

# Five required fields plus an optional title.
_REQUIRED_FIELDS = (
    "hypothesis",
    "conditions",
    "signals",
    "uncertainty",
    "next_steps",
)
_FIELD_LABELS = {
    "hypothesis": "Hypothesis",
    "conditions": "Conditions tested",
    "signals": "Signals observed",
    "uncertainty": "Uncertainty acknowledged",
    "next_steps": "Next steps",
}


def _load_entries() -> List[Dict[str, Any]]:
    try:
        if _NOTEBOOK_PATH.is_file():
            with _NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
    except Exception:
        _logger.exception("Could not load Auditor Notebook entries from disk")
    return []


def _save_entries(entries: List[Dict[str, Any]]) -> None:
    try:
        _NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception:
        _logger.exception("Could not save Auditor Notebook entries to disk")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slugify(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return "untitled"
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:40] or "untitled"


def _entry_to_markdown(entry: Dict[str, Any]) -> str:
    title = entry.get("title") or "Untitled entry"
    ts = entry.get("timestamp", "")
    lines = [f"## {title}", "", f"*Recorded: {ts}*", ""]
    for k in _REQUIRED_FIELDS:
        label = _FIELD_LABELS[k]
        body = (entry.get(k) or "").strip() or "_(empty)_"
        lines.append(f"### {label}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _entries_to_markdown(entries: List[Dict[str, Any]]) -> str:
    head = [
        "# Auditor Notebook",
        "",
        f"_Exported: {_now_iso()}_",
        "",
        f"**Total entries:** {len(entries)}",
        "",
        "---",
        "",
    ]
    if not entries:
        return "\n".join(head + ["_No entries recorded yet._", ""])
    body = "\n---\n\n".join(_entry_to_markdown(e) for e in entries)
    return "\n".join(head) + body


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------

def notebook_server_handlers(input, output, session):
    """Register all reactive components of the Auditor Notebook page."""

    entries = reactive.value(_load_entries())
    last_status = reactive.value(("", "ok"))  # (message, kind)

    # ---- Add entry ------------------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_add)
    def _add_entry():
        record = {
            "title": (input.nb_title() or "").strip(),
            "hypothesis": (input.nb_hypothesis() or "").strip(),
            "conditions": (input.nb_conditions() or "").strip(),
            "signals": (input.nb_signals() or "").strip(),
            "uncertainty": (input.nb_uncertainty() or "").strip(),
            "next_steps": (input.nb_next_steps() or "").strip(),
        }
        missing = [k for k in _REQUIRED_FIELDS if not record[k]]
        if missing:
            human = ", ".join(_FIELD_LABELS[k] for k in missing)
            last_status.set((f"Missing required field(s): {human}.", "error"))
            return
        record["timestamp"] = _now_iso()
        current = list(entries.get())
        current.append(record)
        entries.set(current)
        _save_entries(current)
        last_status.set(("Entry saved.", "ok"))
        # Reset form
        ui.update_text("nb_title", value="")
        ui.update_text_area("nb_hypothesis", value="")
        ui.update_text_area("nb_conditions", value="")
        ui.update_text_area("nb_signals", value="")
        ui.update_text_area("nb_uncertainty", value="")
        ui.update_text_area("nb_next_steps", value="")

    # ---- Clear form -----------------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_clear)
    def _clear_form():
        ui.update_text("nb_title", value="")
        ui.update_text_area("nb_hypothesis", value="")
        ui.update_text_area("nb_conditions", value="")
        ui.update_text_area("nb_signals", value="")
        ui.update_text_area("nb_uncertainty", value="")
        ui.update_text_area("nb_next_steps", value="")
        last_status.set(("Form cleared.", "ok"))

    # ---- Clear all entries ---------------------------------------------
    @reactive.effect
    @reactive.event(input.nb_clear_all)
    def _clear_all():
        if not entries.get():
            return
        entries.set([])
        _save_entries([])
        last_status.set(("All entries cleared.", "ok"))

    # ---- Delete individual entry ---------------------------------------
    @reactive.effect
    @reactive.event(input.nb_delete_idx, ignore_init=True)
    def _delete_one():
        idx = input.nb_delete_idx()
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            return
        current = list(entries.get())
        if 0 <= idx_int < len(current):
            del current[idx_int]
            entries.set(current)
            _save_entries(current)
            last_status.set(("Entry removed.", "ok"))

    # ---- Status banner -------------------------------------------------
    @output
    @render.ui
    def nb_status():
        msg, kind = last_status.get()
        if not msg:
            return ui.HTML("")
        klass = "nb-status nb-status-error" if kind == "error" else "nb-status"
        return ui.tags.div(msg, class_=klass)

    # ---- Entry counter badge -------------------------------------------
    @output
    @render.ui
    def nb_count():
        n = len(entries.get())
        label = "entry" if n == 1 else "entries"
        return ui.HTML(f'<span class="nb-count">{n} {label}</span>')

    # ---- Entries list --------------------------------------------------
    @output
    @render.ui
    def nb_entries():
        items = entries.get()
        if not items:
            return ui.tags.div(
                "No entries yet. Fill the form on the left and click "
                "\"Add entry\" to record your first audit move.",
                class_="nb-empty",
            )
        # Build with raw HTML — the delete buttons need explicit ids so
        # the JS shim below can set ``nb_delete_idx`` and trigger the
        # delete effect above.
        pieces = []
        for i, e in enumerate(items):
            title = _html.escape(e.get("title") or f"Entry {i + 1}")
            ts = _html.escape(e.get("timestamp", ""))
            fields_html = []
            for k in _REQUIRED_FIELDS:
                label = _FIELD_LABELS[k]
                body = _html.escape(e.get(k, "") or "")
                fields_html.append(
                    f'<div class="nb-entry-field">'
                    f'<span class="nb-entry-field-label">{label}</span>'
                    f'<div class="nb-entry-field-value">{body}</div>'
                    f"</div>"
                )
            pieces.append(
                f'<div class="nb-entry">'
                f'<div class="nb-entry-header">'
                f'<span class="nb-entry-title">{title}</span>'
                f'<span class="nb-entry-meta">{ts}'
                f' <button class="nb-entry-delete" '
                f'onclick="Shiny.setInputValue(\'nb_delete_idx\', {i}, '
                f"{{priority: 'event'}}); return false;\">"
                f"&times; delete</button>"
                f"</span></div>"
                f"{''.join(fields_html)}"
                f"</div>"
            )
        return ui.HTML("".join(pieces))

    # ---- Downloads -----------------------------------------------------
    @render.download(filename=lambda: f"auditor_notebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    def nb_download_md():
        yield _entries_to_markdown(list(entries.get()))

    @render.download(filename=lambda: f"auditor_notebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    def nb_download_json():
        payload = {
            "exported_at": _now_iso(),
            "n_entries": len(entries.get()),
            "entries": list(entries.get()),
        }
        yield json.dumps(payload, ensure_ascii=False, indent=2)
