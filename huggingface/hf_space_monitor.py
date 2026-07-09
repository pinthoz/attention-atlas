"""Hugging Face Space health monitor with automatic restart.

Checks the runtime stage of a Space and restarts it when it is in
RUNTIME_ERROR (the stage HF reports for transient scheduling failures
such as "Scheduling failure: unable to schedule"). Build/config errors
are reported but NOT auto-restarted: those come from the app code and a
restart cannot fix them.

Usage (single check, suitable for cron / Task Scheduler / GitHub Actions):

    python huggingface/hf_space_monitor.py --space pinthoz/attention-atlas

The token needs WRITE access to the Space. It is read from --token, or
the HF_TOKEN / HUGGINGFACE_TOKEN environment variables.

Exit codes: 0 = healthy (or restart issued), 1 = unexpected failure,
2 = Space in an error state this script does not auto-fix.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SPACE = "pinthoz/attention-atlas"

# Stages that a restart can plausibly fix (transient infrastructure).
RESTARTABLE_STAGES = {"RUNTIME_ERROR"}
# Healthy / transitional stages: nothing to do.
OK_STAGES = {"RUNNING", "RUNNING_BUILDING", "BUILDING", "RUNNING_APP_STARTING",
             "APP_STARTING", "SLEEPING"}
# Intentional or code-level states: report, never auto-restart.
MANUAL_STAGES = {"PAUSED", "STOPPED", "BUILD_ERROR", "CONFIG_ERROR",
                 "NO_APP_FILE", "DELETING"}


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[{stamp}] {msg}", flush=True)


def _read_state(state_file: Path) -> dict:
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state_file: Path, state: dict) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state), encoding="utf-8")
    except Exception as e:
        _log(f"warning: could not persist state file ({e})")


def check_and_restart(space: str, token: str, factory: bool,
                      min_gap_minutes: float, state_file: Path) -> int:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    runtime = api.get_space_runtime(repo_id=space)
    stage = str(getattr(runtime, "stage", "") or "").upper()
    _log(f"{space}: stage={stage}")

    if stage in OK_STAGES:
        return 0

    if stage in MANUAL_STAGES:
        _log(f"stage {stage} is not auto-restartable (paused/stopped are "
             f"intentional; build/config errors need a code fix). "
             f"Inspect: https://huggingface.co/spaces/{space}")
        return 2

    if stage not in RESTARTABLE_STAGES:
        _log(f"unknown stage {stage!r}; not restarting. Inspect the Space.")
        return 2

    # Cooldown: avoid hammering HF if the error persists across checks.
    state = _read_state(state_file)
    last = float(state.get("last_restart_ts", 0.0))
    gap_s = min_gap_minutes * 60.0
    since = time.time() - last
    if last and since < gap_s:
        _log(f"stage {stage}, but last restart was {since / 60.0:.1f} min ago "
             f"(< {min_gap_minutes:g} min cooldown); skipping this round.")
        return 0

    _log(f"stage {stage}: issuing {'factory reboot' if factory else 'restart'}...")
    api.restart_space(repo_id=space, factory_reboot=factory)
    _write_state(state_file, {"last_restart_ts": time.time(),
                              "last_stage": stage,
                              "factory": factory})
    _log("restart requested. The Space takes a few minutes to come back; "
         "the next check will confirm.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=DEFAULT_SPACE,
                        help=f"Space id (default: {DEFAULT_SPACE})")
    parser.add_argument("--token", default=None,
                        help="HF token with write access (default: HF_TOKEN / "
                             "HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--factory", action="store_true",
                        help="Use a factory reboot (full image rebuild) instead "
                             "of a plain restart")
    parser.add_argument("--min-gap-minutes", type=float, default=10.0,
                        help="Cooldown between automatic restarts (default: 10)")
    parser.add_argument("--state-file", default=None,
                        help="Where to remember the last restart time "
                             "(default: <script dir>/.hf_space_monitor_state.json)")
    parser.add_argument("--watch", type=float, default=None, metavar="MINUTES",
                        help="Keep running, re-checking every MINUTES instead "
                             "of exiting after one check")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Fall back to the token cached by `huggingface-cli login`.
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if not token:
        _log("error: no token. Either run `huggingface-cli login`, or set the "
             "env var (PowerShell: $env:HF_TOKEN = \"hf_...\"), or pass --token. "
             "The token needs WRITE access to the Space.")
        return 1

    state_file = Path(args.state_file) if args.state_file else (
        Path(__file__).resolve().parent / ".hf_space_monitor_state.json")

    while True:
        try:
            code = check_and_restart(args.space, token, args.factory,
                                     args.min_gap_minutes, state_file)
        except Exception as e:
            _log(f"error: {type(e).__name__}: {e}")
            code = 1
        if args.watch is None:
            return code
        time.sleep(max(60.0, args.watch * 60.0))


if __name__ == "__main__":
    sys.exit(main())
