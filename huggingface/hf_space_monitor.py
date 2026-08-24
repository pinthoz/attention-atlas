"""Hugging Face Space health monitor: keeps the Space awake and restarts it.

Two jobs, on every check:

1. Keep-awake. A free (cpu-basic) Space sleeps after `gcTimeout` seconds
   with no traffic - 48h, and NOT configurable on free hardware, so the
   only way to stay up is to keep sending traffic. Any HTTP request to
   the public *.hf.space URL resets that timer, so a healthy Space gets
   pinged. If it is already SLEEPING, a restart wakes it (deterministic;
   an HTTP GET only starts the wake-up and then serves a holding page).

2. Health. Restarts the Space when it is in RUNTIME_ERROR (the stage HF
   reports for transient scheduling failures such as "Scheduling failure:
   unable to schedule"). Build/config errors are reported but NOT
   auto-restarted: those come from the app code and a restart cannot fix
   them.

Usage (single check, suitable for cron / Task Scheduler / GitHub Actions):

    python huggingface/hf_space_monitor.py --space pinthoz/attention-atlas

The token needs WRITE access to the Space. It is read from --token, or
the HF_TOKEN / HUGGINGFACE_TOKEN environment variables.

Exit codes: 0 = healthy (pinged), 1 = unexpected failure, 2 = Space in an
error state this script does not auto-fix, 3 = restart issued after an
error, 4 = woken from SLEEPING.

3 is deliberately distinct from 0: a caller that treats "restarted" as
success cannot tell a one-off blip from a Space that is being restarted
every cycle and never recovering. 4 is distinct from 3 because waking a
sleeping Space is routine and needs no human attention, while repeated
error restarts do.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SPACE = "pinthoz/attention-atlas"
# The FastAPI liveness probe (service/api.py). Cheaper than "/", which renders
# the whole Shiny dashboard, and it still counts as traffic to the app.
DEFAULT_PING_PATH = "/api/health"

# Stages that a restart can plausibly fix (transient infrastructure).
RESTARTABLE_STAGES = {"RUNTIME_ERROR"}
# Asleep after gcTimeout with no traffic. Restarting is how you wake it.
WAKEABLE_STAGES = {"SLEEPING"}
# Serving, or on its way to serving: nothing to restart.
OK_STAGES = {"RUNNING", "RUNNING_BUILDING", "BUILDING", "RUNNING_APP_STARTING",
             "APP_STARTING"}
# Only a RUNNING Space serves the ping; the transitional stages return the
# HF holding page, which does not count as app traffic.
PINGABLE_STAGES = {"RUNNING", "RUNNING_BUILDING"}
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


def _space_url(space: str, runtime, path: str = "/") -> str:
    """Public app URL. Prefer the domain HF reports; derive it otherwise."""
    raw = getattr(runtime, "raw", None) or {}
    host = None
    for entry in (raw.get("domains") or []):
        if entry.get("domain"):
            host = entry["domain"]
            break
    if host is None:
        # Fallback: owner/name -> owner-name.hf.space, with the same character
        # folding the Hub applies when it builds the subdomain.
        host = space.replace("/", "-").lower()
        for ch in ("_", "."):
            host = host.replace(ch, "-")
        host = f"{host}.hf.space"
    return f"https://{host}/{path.lstrip('/')}"


def _ping(url: str, timeout: float) -> bool:
    """GET the Space so HF sees traffic and resets the inactivity timer.

    Best-effort: a slow or failing ping is worth a log line but must not
    fail the check, because the Space itself may be fine and the next tick
    will try again. Even a 4xx/5xx counts as traffic, so those are not
    treated as a ping failure.
    """
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "hf-space-monitor"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(2048)  # touch the body so the request is a real hit
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:
        _log(f"keep-alive ping to {url} failed "
             f"({type(e).__name__}: {e}); the Space may still be fine.")
        return False
    _log(f"keep-alive ping {url} -> {status} "
         f"in {time.monotonic() - started:.1f}s")
    return True


def _get_runtime(api, space: str, attempts: int = 3):
    # A DNS blip or a 5xx from the Hub is not a Space problem; without the
    # retry the scheduled check reports a failure for something transient.
    delay = 5.0
    for i in range(1, attempts + 1):
        try:
            return api.get_space_runtime(repo_id=space)
        except Exception as e:
            if i == attempts:
                raise
            _log(f"could not read runtime ({type(e).__name__}: {e}); "
                 f"retrying in {delay:.0f}s ({i}/{attempts - 1})")
            time.sleep(delay)
            delay *= 2


def check_and_restart(space: str, token: str, factory: bool,
                      min_gap_minutes: float, state_file: Path,
                      ping: bool = True, ping_timeout: float = 60.0,
                      ping_path: str = DEFAULT_PING_PATH) -> int:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    runtime = _get_runtime(api, space)
    stage = str(getattr(runtime, "stage", "") or "").upper()
    _log(f"{space}: stage={stage}")

    if stage in OK_STAGES:
        # This ping is the whole point of the schedule: without traffic the
        # Space sleeps again after gcTimeout regardless of how often we look.
        if ping and stage in PINGABLE_STAGES:
            _ping(_space_url(space, runtime, ping_path), ping_timeout)
        else:
            _log("not serving yet; skipping the keep-alive ping this round.")
        return 0

    if stage in WAKEABLE_STAGES:
        _log(f"stage {stage}: waking the Space (restart)...")
        api.restart_space(repo_id=space, factory_reboot=False)
        _write_state(state_file, {"last_restart_ts": time.time(),
                                  "last_stage": stage,
                                  "factory": False})
        _log("wake-up requested. It takes a few minutes to serve again; the "
             "next check pings it and resets the inactivity timer.")
        return 4

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
    return 3


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
    parser.add_argument("--no-ping", dest="ping", action="store_false",
                        help="Only report the stage; do not send the keep-alive "
                             "request that stops the Space going to sleep")
    parser.add_argument("--ping-timeout", type=float, default=60.0,
                        help="Seconds to wait for the keep-alive request "
                             "(default: 60; the app is slow on the first hit "
                             "after a restart)")
    parser.add_argument("--ping-path", default=DEFAULT_PING_PATH,
                        help=f"Path to request on the Space "
                             f"(default: {DEFAULT_PING_PATH})")
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
                                     args.min_gap_minutes, state_file,
                                     ping=args.ping,
                                     ping_timeout=args.ping_timeout,
                                     ping_path=args.ping_path)
        except Exception as e:
            _log(f"error: {type(e).__name__}: {e}")
            code = 1
        if args.watch is None:
            return code
        time.sleep(max(60.0, args.watch * 60.0))


if __name__ == "__main__":
    sys.exit(main())
