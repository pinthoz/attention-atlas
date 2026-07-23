"""Interaction logging for the user study (Chapter 8 / RQ4).

Records, with timestamps, the four event types the study protocol requires,
one JSON file per session:

1. ``panel_open``    - opening of each bias panel, flagged when it is the
                       faithfulness panel (accordion value ``ablation``)
2. ``control_use``   - use of the calibration controls (BAR threshold,
                       top-K, detection threshold, alpha, correction)
3. ``view_change``   - navigation between the main sections
4. ``notebook_entry``- creation of an Auditor Notebook entry

Why this exists: under the open task instructions (guiao v2) the participant
is never told to visit the faithfulness panel, so *spontaneous* consultation
is only demonstrable from these logs. Because the inference rests on whether
the panel was opened **before** the conclusion was written, every event
carries a monotonic ``seq`` and a ``t_rel_s`` offset from session start.

Identifying the participant
---------------------------
Two deployments, two mechanisms:

* **Localhost** (one app process per session): ``ATLAS_PARTICIPANT_ID=P07``.
* **Hosted (Hugging Face Space)**: one process serves every participant, so
  an environment variable cannot identify them. The code is taken from the
  session URL instead - give each participant a link ending in ``?pid=P07``.

The URL is read reactively, so the participant is known slightly after the
session opens; the log file is renamed once it resolves.

Persistence
-----------
A Space has an ephemeral filesystem: anything written to disk is lost when
the Space restarts or rebuilds. Set ``ATLAS_LOG_HF_REPO`` (a **private**
dataset repo id) plus a write-scoped ``HF_TOKEN`` secret and each session
log is uploaded there, which is also what keeps the study data restricted
to the team. Without those variables the log stays on local disk only.

Environment
-----------
``ATLAS_INTERACTION_LOG=0``   disable logging entirely
``ATLAS_PARTICIPANT_ID``      participant code (localhost)
``ATLAS_LOG_HF_REPO``         private HF dataset repo id for upload
``HF_TOKEN``                  write token used for that upload
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from shiny import reactive

_logger = logging.getLogger(__name__)

_LOG_DIR = Path("downloads") / "sessions" / "interaction_logs"

# Accordion value of the Faithfulness Validation panel. The study's central
# cross-checking measure is whether this panel is opened before concluding.
_FAITHFULNESS_PANEL = "ablation"

_PANEL_LABELS = {
    "overview": "Overview & Detection",
    "technical": "Technical Analysis",
    "attention_bias": "Attention x Bias Correlation",
    "ablation": "Faithfulness Validation",
    "stereoset": "StereoSet Evaluation",
}

# Calibration controls: input id -> human label. These are the inputs the
# study's "calibration-sensitive reasoning" construct is coded from, so the
# significance controls matter as much as the sliders.
_CONTROL_INPUTS = {
    "bias_bar_threshold": "BAR specialisation threshold",
    "bias_top_k": "Top-K heads",
    "bias_threshold": "Detection threshold",
    "bias_alpha": "Significance level (alpha)",
    "bias_correction": "Multiple-comparison correction",
    "bias_use_optimized": "Optimised-thresholds toggle",
}

# Query-string keys accepted for the participant code, in order.
_PID_KEYS = ("pid", "participant", "p")


def _enabled() -> bool:
    return os.environ.get("ATLAS_INTERACTION_LOG", "1").strip() not in ("0", "false", "False")


def study_mode() -> bool:
    """True when the app is serving a user-study session.

    In study mode the Auditor Notebook starts empty and is scoped to the
    participant, instead of restoring the shared notebook from disk.
    """
    return os.environ.get("ATLAS_STUDY_MODE", "").strip() in ("1", "true", "True")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", str(value))[:60]


def read_participant(session) -> Optional[str]:
    """Resolve the participant code for this session.

    Must be called from inside a reactive context, because the URL is read
    reactively. Order: ``?pid=`` in the session URL (hosted deployments),
    then ``ATLAS_PARTICIPANT_ID`` (localhost). Returns ``None`` when neither
    is present.
    """
    pid = ""
    try:
        search = session.clientdata.url_search() or ""
    except Exception:
        search = ""
    if search:
        try:
            params = parse_qs(search.lstrip("?"))
            for key in _PID_KEYS:
                if params.get(key):
                    pid = params[key][0]
                    break
        except Exception:
            pid = ""
    if not pid:
        pid = os.environ.get("ATLAS_PARTICIPANT_ID", "")
    pid = _safe_slug(pid.strip())
    return pid or None


def hub_upload_configured() -> bool:
    """Whether a private HF dataset repo + token are set for durable copies."""
    return bool((os.environ.get("ATLAS_LOG_HF_REPO") or "").strip()
                and (os.environ.get("HF_TOKEN") or "").strip())


def upload_study_file(path: Path, subdir: str, participant: Optional[str]) -> None:
    """Copy a study artefact to the private HF dataset repo, if configured.

    A Space's disk does not survive a restart, so for hosted sessions this is
    the only durable copy. Used for both the interaction log and the Auditor
    Notebook (the study's primary coded artefact). Failures are logged and
    swallowed: losing an upload must never interrupt a running session.

    ``subdir`` groups the artefact type (``interaction_logs`` /
    ``notebooks``); the per-participant folder keeps a participant's files
    together and out of the next participant's view.
    """
    repo = (os.environ.get("ATLAS_LOG_HF_REPO") or "").strip()
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not repo or not token:
        return
    try:
        path = Path(path)
    except TypeError:
        return
    if not path.is_file():
        return
    try:
        from huggingface_hub import HfApi

        who = _safe_slug(participant) if participant else "unassigned"
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{subdir}/{who}/{path.name}",
            repo_id=repo,
            repo_type="dataset",
            token=token,
        )
        _logger.info("Uploaded %s to %s", subdir, repo)
    except Exception:
        _logger.exception("Could not upload %s to %s", subdir, repo)


def upload_in_background(path: Path, subdir: str, participant: Optional[str]) -> None:
    """Fire-and-forget upload, so a network round-trip never blocks a session."""
    if not hub_upload_configured():
        return
    threading.Thread(
        target=upload_study_file, args=(path, subdir, participant), daemon=True
    ).start()


def _upload_to_hub(path: Path, participant: Optional[str]) -> None:
    """Back-compat shim for the interaction log's own uploads."""
    upload_study_file(path, "interaction_logs", participant)


class _SessionLog:
    """Append-only event log for one Shiny session, persisted as JSON."""

    # A hosted Space has an ephemeral filesystem and can be restarted at any
    # time, so waiting for the session to end before uploading risks losing
    # a whole participant's log - which cannot be re-collected. Push a copy
    # every few events as well.
    _UPLOAD_EVERY = 10

    def __init__(self, session_id: str, participant: Optional[str]) -> None:
        self._events: List[Dict[str, Any]] = []
        self._seq = 0
        self._uploaded_at_seq = 0
        self._t0 = time.monotonic()
        self.started_at = datetime.now(timezone.utc)
        self.participant = participant
        self.session_id = session_id
        self.path = self._build_path()

    def _build_path(self) -> Path:
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        who = _safe_slug(self.participant) if self.participant else f"unassigned-{_safe_slug(self.session_id)}"
        return _LOG_DIR / f"session_{stamp}_{who}.json"

    def set_participant(self, participant: str) -> None:
        """Attach a participant code, renaming the file already on disk.

        On hosted deployments the code only arrives once the URL has been
        read, which is a moment after the session opens.
        """
        if not participant or participant == self.participant:
            return
        old = self.path
        self.participant = participant
        self.path = self._build_path()
        try:
            if old.is_file() and old != self.path:
                old.unlink()
        except Exception:
            _logger.debug("Could not remove provisional log %s", old)
        self.record("participant_identified", {"participant": participant})

    def record(self, event: str, detail: Optional[Dict[str, Any]] = None) -> None:
        """Append one event and flush to disk. Never raises."""
        try:
            self._seq += 1
            self._events.append({
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "t_rel_s": round(time.monotonic() - self._t0, 3),
                "event": event,
                "detail": detail or {},
            })
            self._flush()
            if self._seq - self._uploaded_at_seq >= self._UPLOAD_EVERY:
                self._uploaded_at_seq = self._seq
                self._upload_in_background()
        except Exception:
            _logger.exception("Interaction log: could not record event %s", event)

    def _upload_in_background(self) -> None:
        """Push the log to the Hub without blocking the session.

        The upload is a network round-trip; running it inline would stall
        the participant's interaction while they are being timed.
        """
        upload_in_background(self.path, "interaction_logs", self.participant)

    def _flush(self) -> None:
        payload = {
            "session_id": self.session_id,
            "participant": self.participant,
            "started_at": self.started_at.isoformat(),
            "n_events": len(self._events),
            "events": self._events,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first: a crash mid-write would otherwise
            # leave truncated JSON and lose the whole session.
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp.replace(self.path)
        except Exception:
            _logger.exception("Interaction log: could not write %s", self.path)

    def finish(self) -> None:
        self.record("session_end", {})
        # Final synchronous copy: this runs in the session-end hook, so the
        # daemon thread used mid-session might not outlive the process.
        upload_study_file(self.path, "interaction_logs", self.participant)


def register_interaction_logging(input, session) -> Optional[_SessionLog]:
    """Wire the four study event types to this session's log."""
    if not _enabled():
        _logger.info("Interaction logging disabled (ATLAS_INTERACTION_LOG=0).")
        return None

    session_id = str(getattr(session, "id", "unknown"))
    log = _SessionLog(session_id, os.environ.get("ATLAS_PARTICIPANT_ID") or None)
    log.record("session_start", {"participant": log.participant})
    _logger.info("Interaction logging -> %s", log.path)

    # ── Participant identity ────────────────────────────────────────
    # Read reactively so that a hosted session can be identified by its
    # ``?pid=`` URL, which is not available synchronously at startup.
    @reactive.effect
    def _resolve_participant():
        pid = read_participant(session)
        if pid:
            log.set_participant(pid)
        elif log.participant is None:
            # Recoverable before the session starts, unrecoverable after.
            _logger.warning(
                "Interaction log has NO participant code. Open the app with "
                "?pid=<code> (hosted) or set ATLAS_PARTICIPANT_ID (local), or "
                "this log cannot be joined to the participant's other data.")

    # ── 1. Panel opening ────────────────────────────────────────────
    # The accordion input holds the currently-open panels. Log only the
    # newly-opened ones, so a panel the participant returns to is a fresh
    # event while unrelated re-renders are not.
    seen_open: set = set()

    @reactive.effect
    def _log_panel_open():
        try:
            current = input.bias_accordion()
        except Exception:
            return
        if current is None:
            return
        values = {current} if isinstance(current, str) else set(current)
        for value in values - seen_open:
            log.record("panel_open", {
                "panel": value,
                "label": _PANEL_LABELS.get(value, value),
                "is_faithfulness": value == _FAITHFULNESS_PANEL,
            })
        seen_open.clear()
        seen_open.update(values)

    # ── Change watcher ──────────────────────────────────────────────
    # A plain effect with explicit previous-value tracking, rather than
    # @reactive.event(..., ignore_init=True). Study inputs only come into
    # existence once their section renders, so at session start reading
    # them raises and the effect is cancelled; reactive.event would then
    # treat the participant's *first real interaction* as the initial
    # value and silently swallow it. Verified against a live session.
    _prev: Dict[str, Any] = {}
    _UNSET = object()

    def _watch(input_id: str, on_change, *, skip_values: tuple = ()) -> None:
        @reactive.effect
        def _watcher():
            try:
                value = getattr(input, input_id)()
            except Exception:
                # Not rendered yet. The dependency is registered, so this
                # effect re-runs once the input exists.
                return
            prev = _prev.get(input_id, _UNSET)
            _prev[input_id] = value
            first = prev is _UNSET
            if not first and value == prev:
                return
            if value in skip_values:
                return
            try:
                on_change(value, first)
            except Exception:
                _logger.exception("Interaction log: watcher failed for %s", input_id)

    # ── 2. Calibration controls ─────────────────────────────────────
    # These are raw HTML controls pushed through Shiny.setInputValue on
    # interaction, so the server never sees them until the participant
    # first touches one: the first observation IS a use, not a default.
    # It is still flagged, because a few controls are also initialised
    # once by the toolbar's setup script, and only the analyst can tell
    # a genuine first adjustment from that initialisation.
    for _cid, _label in _CONTROL_INPUTS.items():
        def _on_control(value, first, control=_cid, label=_label):
            log.record("control_use", {
                "control": control, "label": label, "value": value,
                "first_observation": first,
            })
        _watch(_cid, _on_control)

    # ── 3. View changes ─────────────────────────────────────────────
    def _on_view(value, first):
        log.record("view_change", {"view": value, "first_observation": first})

    _watch("main_navbar", _on_view)

    # ── 4. Auditor Notebook entries ─────────────────────────────────
    # Only the judgement fields are recorded (never the free text), so the
    # log stays an ordering artefact rather than a second copy of the entry.
    def _on_notebook_entry(_value, _first):
        def _filled(input_id: str) -> bool:
            try:
                return bool((getattr(input, input_id)() or "").strip())
            except Exception:
                return False

        log.record("notebook_entry", {
            "has_hypothesis": _filled("nb_hypothesis"),
            "has_uncertainty": _filled("nb_uncertainty"),
            "has_next_steps": _filled("nb_next_steps"),
            "faithfulness_opened_before": _FAITHFULNESS_PANEL in seen_open,
        })

    # The un-clicked button reports 0; only an actual click is an entry.
    _watch("nb_add", _on_notebook_entry, skip_values=(0,))

    # ── Session end: final flush + durable copy ──────────────────────
    try:
        session.on_ended(log.finish)
    except Exception:
        _logger.debug("Could not register session end hook for the log.")

    return log


__all__ = [
    "register_interaction_logging", "read_participant", "study_mode",
    "upload_in_background", "upload_study_file", "hub_upload_configured",
]
