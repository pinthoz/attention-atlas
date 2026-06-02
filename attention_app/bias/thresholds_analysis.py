"""Empirical threshold calibration for the Attention Atlas bias pipeline.

This script produces defensible values for four currently-hardcoded
thresholds:

  1. BAR (Bias Attention Ratio) — currently 1.5 in
     ``attention_app/bias/attention_bias.py``.
  2. BSR (Bias Self-Reinforcement) — same family of metric as BAR.
  3. GUS-Net per-category thresholds (GEN / UNFAIR / STEREO) —
     currently 0.5 in the dashboard sliders.
  4. Top-K heads for ablation — currently ``min(10, len(metrics))``
     in ``attention_app/server/bias_handlers.py``.

The methodology is:

  - For (1) and (2): permutation null. For each sentence in a
    stratified subsample of ``bias_sentences_v9.json`` we run the model
    once, compute BAR/BSR for each (layer, head) at the observed
    biased-token positions, then re-compute under N random reshuffles
    of those positions. The 95th and 99th percentiles of the
    permutation null give thresholds with α=0.05 and α=0.01 false
    positive rates respectively.

  - For (3): we score every sentence in ``human_audit_sample_v9_300.csv``
    with GUS-Net, taking the max per-category probability across
    non-special tokens. Per-category we fit a precision-recall curve
    against the LLM-refined sentence-level ground truth (``original_label``
    flipped by the ``MISLABELED`` verdicts, ``WEAK_BIAS`` excluded) and
    report the threshold that maximises F1.

  - For (4): on a smaller subsample we ablate the top-1, top-2, …,
    top-30 heads ranked by BAR and record the cumulative drop in the
    bias-class probability. The elbow of the cumulative-impact curve
    suggests a defensible default K.

Outputs land in ``dataset/thresholds_results/`` (kept next to the
input datasets so all calibration artefacts live together) as JSON + PNG.

Usage:
    # Quick smoke test (5 sentences, 20 perms, 5 ablation sentences)
    python -m attention_app.bias.thresholds_analysis --all --n 5 --perms 20 --abl-n 5

    # Full run (~30-60 min on CPU, ~5-10 min on GPU)
    python -m attention_app.bias.thresholds_analysis --all --n 300 --perms 200 --abl-n 100

    # Individual parts
    python -m attention_app.bias.thresholds_analysis --bar --n 300 --perms 200
    python -m attention_app.bias.thresholds_analysis --gusnet
    python -m attention_app.bias.thresholds_analysis --ablation --abl-n 100
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import logging
import pickle
import random
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent           # attention_app/bias/
ROOT = HERE.parent.parent                        # repository root
RESULTS_DIR = ROOT / "dataset" / "thresholds_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Make the app package importable when this is run as a script (e.g.
# python attention_app/bias/thresholds_analysis.py) rather than as
# `python -m attention_app.bias.thresholds_analysis`.
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────
# Pattern: each long-running analysis periodically writes its full state
# (accumulators + set of processed sentence keys + counters + a hash of
# the run parameters) to ``dataset/thresholds_results/checkpoints/``.
# On restart, the analysis loads the checkpoint, checks that the
# parameters match, and skips sentences whose ``(model, text_hash)`` pair
# is already in the processed set. Writes are atomic (tmp + rename) so a
# crash mid-save does not corrupt the on-disk state.

def _text_hash(text: str) -> str:
    """Stable short hash of a sentence — used to key processed entries."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _args_signature(*parts: Any) -> str:
    """Hash that identifies the parameters of a run. A checkpoint is only
    reused when the new run has the same signature; otherwise we start
    fresh because the accumulators would mix incompatible data."""
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:16]


def _save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Atomic pickle write — temp file then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    except Exception:
        _logger.exception("Could not save checkpoint at %s", path)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        _logger.warning("Could not load checkpoint at %s; starting fresh", path)
        return None


class _GracefulInterrupt:
    """Catch Ctrl-C once. The first SIGINT sets a flag so the running
    loop can save its checkpoint and exit cleanly; a second Ctrl-C
    aborts immediately (the default handler is re-installed)."""

    def __init__(self) -> None:
        self.interrupted = False
        self._previous = None

    def __enter__(self):
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *exc):
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, signum, frame):
        if self.interrupted:
            # Second Ctrl-C — restore default handler and re-raise.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\nSecond interrupt — aborting now.", flush=True)
            raise KeyboardInterrupt()
        self.interrupted = True
        print(
            "\nInterrupt received. Will save the checkpoint after the current "
            "sentence and exit. Press Ctrl-C again to abort immediately.",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────
# BAR / BSR helpers
# ──────────────────────────────────────────────────────────────────────

def compute_bar_bsr(attention_matrix: np.ndarray,
                    biased_mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return (BAR, BSR) for one head. Mirrors the definitions in
    ``attention_app/bias/attention_bias.py``."""
    seq_len = attention_matrix.shape[0]
    n_biased = int(biased_mask.sum())
    if n_biased == 0:
        return None, None
    mu_expected = n_biased / seq_len
    if mu_expected <= 0:
        return None, None
    mu_observed = attention_matrix[:, biased_mask].sum() / seq_len
    bar = float(mu_observed / mu_expected)
    nu_observed = attention_matrix[biased_mask][:, biased_mask].sum() / n_biased
    bsr = float(nu_observed / mu_expected)
    return bar, bsr


def permute_bar_bsr(attention_matrix: np.ndarray, n_biased: int,
                    n_permutations: int, rng: np.random.Generator
                    ) -> Tuple[List[float], List[float]]:
    """N permutations of ``n_biased`` indices, return null BAR/BSR lists."""
    seq_len = attention_matrix.shape[0]
    if n_biased == 0 or n_biased > seq_len:
        return [], []
    bar_null: List[float] = []
    bsr_null: List[float] = []
    indices = np.arange(seq_len)
    for _ in range(n_permutations):
        perm = rng.choice(indices, size=n_biased, replace=False)
        mask = np.zeros(seq_len, dtype=bool)
        mask[perm] = True
        b, s = compute_bar_bsr(attention_matrix, mask)
        if b is not None:
            bar_null.append(b)
            bsr_null.append(s)
    return bar_null, bsr_null


# ──────────────────────────────────────────────────────────────────────
# Subsample loader
# ──────────────────────────────────────────────────────────────────────

def load_v9_stratified(n_sentences: int, seed: int = 42,
                       max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
    """Stratified subsample (half biased, half neutral) from v9.

    Parameters
    ----------
    n_sentences
        Target subsample size. ``0`` or a negative value means "use the
        entire v9 dataset" (both classes, full size).
    seed
        Shuffle seed for reproducibility.
    max_tokens
        Optional cap on whitespace-token count. Sentences longer than
        this are dropped before sampling — a cheap proxy that keeps
        runtime bounded and avoids pathologically slow forward passes.
    """
    with open(ROOT / "dataset" / "bias_sentences_v9.json", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise RuntimeError("Unexpected v9 structure")
    if max_tokens is not None and max_tokens > 0:
        entries = [
            e for e in entries
            if len(str(e.get("text", "")).split()) <= max_tokens
        ]
    biased = [e for e in entries if e.get("has_bias")]
    neutral = [e for e in entries if not e.get("has_bias")]
    rng = random.Random(seed)
    rng.shuffle(biased)
    rng.shuffle(neutral)
    if n_sentences is None or n_sentences <= 0:
        # Use everything (still stratify-balanced order via interleave).
        sample = biased + neutral
        rng.shuffle(sample)
        return sample
    n_each = max(1, n_sentences // 2)
    sample = biased[:n_each] + neutral[:n_each]
    rng.shuffle(sample)
    return sample[:n_sentences]


# ──────────────────────────────────────────────────────────────────────
# Lazy imports of app modules (kept lazy so smoke-test of `--gusnet`
# alone does not pay the price of loading transformers)
# ──────────────────────────────────────────────────────────────────────

def _get_heavy_compute():
    from attention_app.server.logic import heavy_compute  # type: ignore
    return heavy_compute


def _get_gusnet_detector():
    from attention_app.bias.gusnet_detector import GusNetDetector  # type: ignore
    return GusNetDetector


def _detect_biased_indices_for_attention_tokens(
    text: str, attention_tokens: List[str], detector
) -> Optional[np.ndarray]:
    """Run GUS-Net on the text and return a boolean mask aligned with
    the *attention model's* tokens. We approximate alignment by
    matching the GUS-Net biased word substrings against the attention
    tokens (case-insensitive). This is the same shortcut used in
    ``_align_gusnet_to_attention_tokens`` and is sufficient for
    distribution-level analysis.
    """
    try:
        gus_tokens, gus_probs = detector.predict_proba(text)
        labeled = detector.apply_thresholds(gus_tokens, gus_probs)
    except Exception:
        return None
    biased_tokens_text = {
        str(item.get("token", "")).strip("#").lower()
        for item in labeled
        if item.get("is_biased")
    }
    biased_tokens_text.discard("")
    if not biased_tokens_text:
        return None
    n = len(attention_tokens)
    mask = np.zeros(n, dtype=bool)
    for i, tok in enumerate(attention_tokens):
        clean = tok.replace("##", "").replace("Ġ", "").lower()
        if clean in biased_tokens_text:
            mask[i] = True
    return mask if mask.any() else None


# ──────────────────────────────────────────────────────────────────────
# Analysis 1 — BAR / BSR permutation null
# ──────────────────────────────────────────────────────────────────────

def run_bar_bsr_analysis(n_sentences: int = 300, n_permutations: int = 200,
                         models: Optional[List[str]] = None,
                         max_tokens: Optional[int] = None,
                         resume: bool = True,
                         checkpoint_every: int = 20) -> Dict[str, Any]:
    """Compute observed and permutation-null BAR/BSR across a v9 subsample.

    Supports resumable runs: periodic checkpoints are written to
    ``dataset/thresholds_results/checkpoints/bar_bsr.pkl`` so that
    Ctrl-C or a crash does not lose progress. Pass ``resume=False`` to
    force a fresh run.
    """
    if models is None:
        models = ["bert-base-uncased", "gpt2"]
    sample = load_v9_stratified(n_sentences, max_tokens=max_tokens)
    print(f"BAR/BSR: {len(sample)} sentences x {len(models)} models, "
          f"{n_permutations} permutations per (sentence, head)", flush=True)

    heavy_compute = _get_heavy_compute()
    GusNetDetector = _get_gusnet_detector()

    rng = np.random.default_rng(42)
    detector_cache: Dict[str, Any] = {}

    checkpoint_path = CHECKPOINT_DIR / "bar_bsr.pkl"
    sig = _args_signature("bar_bsr_v1", tuple(models), n_permutations, max_tokens)

    # Per-model aggregates + bookkeeping. ``array.array('f')`` stores
    # 4 bytes per float vs ~28 for a Python list of float objects, so
    # we can hold ~300M permutation samples in ~1.2 GB instead of ~9 GB.
    bar_obs: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bar_null: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bsr_obs: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bsr_null: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    processed_keys: set = set()
    n_skipped = 0
    n_ok = 0

    def _as_arr(maybe_list) -> "array.array":
        """Promote a legacy list-typed accumulator (from older checkpoints)
        to ``array.array('f')`` without losing data."""
        if isinstance(maybe_list, array.array):
            return maybe_list
        a = array.array("f")
        if maybe_list:
            a.extend(maybe_list)
        return a

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state is not None and state.get("signature") == sig:
            bar_obs = {m: _as_arr(state["bar_obs"][m]) for m in models}
            bar_null = {m: _as_arr(state["bar_null"][m]) for m in models}
            bsr_obs = {m: _as_arr(state["bsr_obs"][m]) for m in models}
            bsr_null = {m: _as_arr(state["bsr_null"][m]) for m in models}
            processed_keys = state["processed_keys"]
            n_ok = state.get("n_ok", 0)
            n_skipped = state.get("n_skipped", 0)
            print(f"Resuming from checkpoint: {len(processed_keys)} "
                  f"(model, sentence) pairs already done.", flush=True)
        elif state is not None:
            print("Checkpoint exists but parameters differ — starting fresh.",
                  flush=True)

    def _snapshot() -> Dict[str, Any]:
        return {
            "signature": sig,
            "bar_obs": bar_obs, "bar_null": bar_null,
            "bsr_obs": bsr_obs, "bsr_null": bsr_null,
            "processed_keys": processed_keys,
            "n_ok": n_ok, "n_skipped": n_skipped,
        }

    start = time.time()
    interrupted = False
    with _GracefulInterrupt() as gi:
        for model_name in models:
            if interrupted:
                break
            print(f"\n── {model_name} ──", flush=True)
            bias_key = "gusnet-bert" if "bert" in model_name.lower() else "gusnet-gpt2"
            if bias_key not in detector_cache:
                detector_cache[bias_key] = GusNetDetector(model_key=bias_key)
            detector = detector_cache[bias_key]

            since_last_save = 0
            for i, entry in enumerate(sample):
                if gi.interrupted:
                    interrupted = True
                    break
                if i % 20 == 0:
                    el = time.time() - start
                    print(f"  [{i+1}/{len(sample)}] elapsed {el:.0f}s, "
                          f"obs n_ok={n_ok}, skipped={n_skipped}", flush=True)
                text = entry.get("text", "")
                if not text:
                    n_skipped += 1
                    continue
                key = (model_name, _text_hash(text))
                if key in processed_keys:
                    # Already done in a previous run.
                    continue
                try:
                    result = heavy_compute(text, model_name)
                    if result is None:
                        n_skipped += 1
                        processed_keys.add(key)
                        continue
                    tokens = list(getattr(result, "tokens", []) or [])
                    attentions = getattr(result, "attentions", None)
                    if attentions is None or not tokens:
                        n_skipped += 1
                        processed_keys.add(key)
                        continue
                    biased_mask = _detect_biased_indices_for_attention_tokens(
                        text, tokens, detector
                    )
                    if biased_mask is None:
                        n_skipped += 1
                        processed_keys.add(key)
                        continue
                    seq_len = len(tokens)
                    biased_mask = biased_mask[:seq_len]
                    n_biased = int(biased_mask.sum())
                    if n_biased == 0 or n_biased >= seq_len:
                        n_skipped += 1
                        processed_keys.add(key)
                        continue
                    for li, lt in enumerate(attentions):
                        try:
                            arr = lt[0].detach().cpu().numpy()  # (heads, seq, seq)
                        except Exception:
                            continue
                        arr = arr[:, :seq_len, :seq_len]
                        for hi in range(arr.shape[0]):
                            head_attn = arr[hi]
                            b, s = compute_bar_bsr(head_attn, biased_mask)
                            if b is None:
                                continue
                            bar_obs[model_name].append(b)
                            bsr_obs[model_name].append(s)
                            bn, sn = permute_bar_bsr(
                                head_attn, n_biased, n_permutations, rng
                            )
                            bar_null[model_name].extend(bn)
                            bsr_null[model_name].extend(sn)
                    n_ok += 1
                    processed_keys.add(key)
                except Exception as e:
                    n_skipped += 1
                    processed_keys.add(key)
                    _logger.debug("Sentence %d failed: %s", i, e)
                    continue

                since_last_save += 1
                if since_last_save >= checkpoint_every:
                    _save_checkpoint(checkpoint_path, _snapshot())
                    since_last_save = 0
        # Save the final state — even on interrupt — so the next run resumes.
        _save_checkpoint(checkpoint_path, _snapshot())

    elapsed = time.time() - start
    if interrupted:
        print(f"\nInterrupted after {elapsed:.1f}s — checkpoint saved at "
              f"{checkpoint_path}. Re-run the same command to resume.",
              flush=True)
        return {"interrupted": True, "checkpoint": str(checkpoint_path),
                "processed_so_far": len(processed_keys),
                "n_ok": n_ok, "n_skipped": n_skipped}
    print(f"\nDone in {elapsed:.1f}s — ok={n_ok}, skipped={n_skipped}")

    def _to_np(acc) -> np.ndarray:
        """Zero-copy view of an ``array.array('f')`` as a float32 numpy
        array. Falls back to ``np.asarray`` for legacy Python lists."""
        if isinstance(acc, array.array):
            if len(acc) == 0:
                return np.empty(0, dtype=np.float32)
            return np.frombuffer(acc, dtype=np.float32)
        return np.asarray(acc, dtype=np.float32)

    def summarise(obs_acc, null_acc, default: float) -> Dict[str, Any]:
        obs_n = len(obs_acc)
        null_n = len(null_acc)
        if obs_n == 0 or null_n == 0:
            return {"n_obs": obs_n, "n_null": null_n, "note": "insufficient data"}
        obs_arr = _to_np(obs_acc)
        null_arr = _to_np(null_acc)
        # Compute percentiles in one np.percentile call (single sort).
        null_q = np.percentile(null_arr, [50, 95, 99])
        obs_q = np.percentile(obs_arr, [50, 95, 99])
        return {
            "n_obs": obs_n,
            "n_null": null_n,
            "obs_mean": float(obs_arr.mean()),
            "obs_median": float(obs_q[0]),
            "obs_p95": float(obs_q[1]),
            "obs_p99": float(obs_q[2]),
            "null_mean": float(null_arr.mean()),
            "null_median": float(null_q[0]),
            "null_p95": float(null_q[1]),
            "null_p99": float(null_q[2]),
            "current_default": default,
            "recommended_threshold_alpha_0.05": float(null_q[1]),
            "recommended_threshold_alpha_0.01": float(null_q[2]),
            "fraction_obs_above_recommended_0.05": float(
                (obs_arr >= null_q[1]).mean()
            ),
        }

    summary: Dict[str, Any] = {
        "n_sentences_attempted": len(sample),
        "n_sentences_ok": n_ok,
        "n_sentences_skipped": n_skipped,
        "n_permutations_per_obs": n_permutations,
        "models": models,
        "by_model": {},
    }

    # Combined (across models) summary. Concatenate per-model arrays
    # only as numpy views — never materialise a giant Python list.
    def _concat(per_model_accs) -> np.ndarray:
        parts = [_to_np(per_model_accs[m]) for m in per_model_accs
                 if len(per_model_accs[m]) > 0]
        if not parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(parts)

    all_bar_obs_a = _concat(bar_obs)
    all_bar_null_a = _concat(bar_null)
    all_bsr_obs_a = _concat(bsr_obs)
    all_bsr_null_a = _concat(bsr_null)
    summary["combined"] = {
        "BAR": summarise(all_bar_obs_a, all_bar_null_a, default=1.5),
        "BSR": summarise(all_bsr_obs_a, all_bsr_null_a, default=1.5),
    }
    for m in models:
        summary["by_model"][m] = {
            "BAR": summarise(bar_obs[m], bar_null[m], default=1.5),
            "BSR": summarise(bsr_obs[m], bsr_null[m], default=1.5),
        }

    out_path = RESULTS_DIR / "bar_bsr_thresholds.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")

    # Plot histograms — subsample to keep matplotlib responsive.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOT_SAMPLE_CAP = 1_000_000  # at most 1M points per histogram

        def _sample_for_plot(arr: np.ndarray) -> np.ndarray:
            if arr.size <= PLOT_SAMPLE_CAP:
                return arr
            rng_plot = np.random.default_rng(0)
            idx = rng_plot.choice(arr.size, size=PLOT_SAMPLE_CAP, replace=False)
            return arr[idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (name, obs_a, null_a, default) in zip(axes, [
            ("BAR", all_bar_obs_a, all_bar_null_a, 1.5),
            ("BSR", all_bsr_obs_a, all_bsr_null_a, 1.5),
        ]):
            if obs_a.size == 0 or null_a.size == 0:
                continue
            ax.hist(_sample_for_plot(null_a), bins=80, range=(0, 5),
                    alpha=0.55, label="Null (permutation)",
                    color="#94a3b8", density=True)
            ax.hist(_sample_for_plot(obs_a), bins=80, range=(0, 5),
                    alpha=0.55, label="Observed",
                    color="#ff5ca9", density=True)
            p95 = float(np.percentile(null_a, 95))
            p99 = float(np.percentile(null_a, 99))
            ax.axvline(p95, color="#16a34a", linestyle="--",
                       label=f"95th null = {p95:.2f}")
            ax.axvline(p99, color="#dc2626", linestyle="--",
                       label=f"99th null = {p99:.2f}")
            ax.axvline(default, color="#1d4ed8", linestyle=":",
                       label=f"Current default = {default}")
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.set_title(f"{name}: observed vs permutation-null")
            ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = RESULTS_DIR / "bar_bsr_distribution.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"Saved {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
    # Only remove the checkpoint *after* a successful summary so that an
    # OOM or other crash at the summary stage does not throw away the
    # ~15 minutes of accumulated data.
    try:
        checkpoint_path.unlink(missing_ok=True)
    except Exception:
        pass
    return summary


# ──────────────────────────────────────────────────────────────────────
# Analysis 2 — GUS-Net per-category threshold calibration
# ──────────────────────────────────────────────────────────────────────

def _truth_from_verdict(row: pd.Series) -> Optional[bool]:
    """Refine the dataset's original_label with the LLM auditor verdict.

    Returns ``None`` when the verdict is ``WEAK_BIAS`` (ambiguous) or
    unknown. Otherwise: ``CORRECT`` keeps the original_label, ``MISLABELED``
    flips it.
    """
    verdict = row.get("prompt1_verdict")
    if pd.isna(verdict):
        return None
    if verdict == "WEAK_BIAS":
        return None
    if verdict == "CORRECT":
        return bool(row.get("original_label"))
    if verdict == "MISLABELED":
        return not bool(row.get("original_label"))
    return None


def run_gusnet_calibration(model_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sweep thresholds per category against the LLM-refined audit set."""
    if model_keys is None:
        model_keys = ["gusnet-bert", "gusnet-gpt2"]
    audit_path = ROOT / "dataset" / "human_audit_sample_v9_300.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit set not found at {audit_path}")
    audit = pd.read_csv(audit_path)
    audit["truth"] = audit.apply(_truth_from_verdict, axis=1)
    clear = audit.dropna(subset=["truth"]).copy()
    clear["truth"] = clear["truth"].astype(bool)
    print(f"GUS-Net calibration: {len(clear)} unambiguous sentences "
          f"({len(audit) - len(clear)} WEAK_BIAS / NaN excluded)")
    print(f"  positives={int(clear['truth'].sum())}, "
          f"negatives={int((~clear['truth']).sum())}")

    GusNetDetector = _get_gusnet_detector()
    try:
        from sklearn.metrics import (
            precision_recall_curve, roc_auc_score, f1_score
        )
    except Exception as e:
        raise RuntimeError("scikit-learn is required for calibration") from e

    summary: Dict[str, Any] = {
        "n_sentences": int(len(clear)),
        "n_positives": int(clear["truth"].sum()),
        "n_negatives": int((~clear["truth"]).sum()),
        "models": {},
    }
    for model_key in model_keys:
        print(f"\n── {model_key} ──", flush=True)
        try:
            detector = GusNetDetector(model_key=model_key)
        except Exception as e:
            print(f"Skipping {model_key}: {e}")
            continue
        cfg = detector.config
        cat_idx = cfg["category_indices"]
        special = cfg["special_tokens"]

        cat_scores: Dict[str, List[float]] = {"GEN": [], "UNFAIR": [], "STEREO": []}
        max_any: List[float] = []
        truths: List[bool] = []
        start = time.time()
        for i, (_, row) in enumerate(clear.iterrows()):
            if i % 20 == 0:
                el = time.time() - start
                print(f"  [{i+1}/{len(clear)}] elapsed {el:.0f}s", flush=True)
            text = row.get("text", "")
            try:
                tokens, probs = detector.predict_proba(text)
            except Exception as e:
                _logger.debug("predict_proba failed: %s", e)
                continue
            arr = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
            valid_rows = [i_ for i_, t in enumerate(tokens) if t not in special]
            if not valid_rows:
                continue
            sub = arr[valid_rows]
            sentence_max = 0.0
            for cat, idxs in cat_idx.items():
                # per-token score = max(B-prob, I-prob); per-sentence = max over tokens.
                if not idxs:
                    continue
                col = sub[:, idxs].max(axis=1)
                s = float(col.max()) if col.size else 0.0
                cat_scores[cat].append(s)
                if s > sentence_max:
                    sentence_max = s
            max_any.append(sentence_max)
            truths.append(bool(row["truth"]))

        truths_np = np.asarray(truths)
        model_result: Dict[str, Any] = {"n_used": int(len(truths))}
        for cat, scores in cat_scores.items():
            if len(scores) != len(truths) or not scores:
                continue
            scores_np = np.asarray(scores)
            try:
                precision, recall, ths = precision_recall_curve(truths_np, scores_np)
                # Exclude the trivial recall=0 endpoint when computing F1.
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                if len(ths) == 0:
                    continue
                # ths corresponds to precision[:-1] / recall[:-1]; align lengths.
                f1_aligned = f1[:-1]
                best_idx = int(np.argmax(f1_aligned)) if len(f1_aligned) else 0
                best_th = float(ths[best_idx])
                best_f1 = float(f1_aligned[best_idx])
                auc = float(roc_auc_score(truths_np, scores_np))
            except Exception as e:
                _logger.debug("PR curve for %s failed: %s", cat, e)
                continue
            model_result[cat] = {
                "current_default": 0.5,
                "recommended_threshold_f1_opt": best_th,
                "best_f1": best_f1,
                "auc_roc": auc,
                "score_mean_pos": float(scores_np[truths_np].mean()) if truths_np.any() else None,
                "score_mean_neg": float(scores_np[~truths_np].mean()) if (~truths_np).any() else None,
            }
        # Combined: use max across categories.
        if max_any and len(max_any) == len(truths):
            try:
                precision, recall, ths = precision_recall_curve(truths_np, np.asarray(max_any))
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                f1_aligned = f1[:-1]
                if len(f1_aligned):
                    best_idx = int(np.argmax(f1_aligned))
                    model_result["ANY_CATEGORY"] = {
                        "recommended_threshold_f1_opt": float(ths[best_idx]),
                        "best_f1": float(f1_aligned[best_idx]),
                        "auc_roc": float(roc_auc_score(truths_np, np.asarray(max_any))),
                    }
            except Exception:
                pass
        summary["models"][model_key] = model_result

    out_path = RESULTS_DIR / "gusnet_thresholds.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Analysis 3 — top-K cumulative ablation impact
# ──────────────────────────────────────────────────────────────────────

def run_topk_ablation(n_sentences: int = 50, k_max: int = 20,
                      model_name: str = "bert-base-uncased",
                      max_tokens: Optional[int] = None,
                      resume: bool = True,
                      checkpoint_every: int = 10) -> Dict[str, Any]:
    """For each sentence: rank heads by BAR, ablate top-1..top-K, record
    cumulative effect on the bias-class probability.

    Resumable via ``dataset/thresholds_results/checkpoints/topk_ablation.pkl``.
    """
    sample = load_v9_stratified(n_sentences, max_tokens=max_tokens)
    biased_only = [e for e in sample if e.get("has_bias")]
    print(f"Top-K ablation: {len(biased_only)} biased sentences, K_max={k_max}, "
          f"model={model_name}")
    if not biased_only:
        return {"note": "no biased sentences in subsample"}

    heavy_compute = _get_heavy_compute()
    GusNetDetector = _get_gusnet_detector()
    from attention_app.bias.attention_bias import AttentionBiasAnalyzer  # type: ignore
    from attention_app.bias.head_ablation import batch_ablate_top_heads, HeadAblationResult  # noqa
    from attention_app.models import ModelManager  # type: ignore

    bias_key = "gusnet-bert" if "bert" in model_name.lower() else "gusnet-gpt2"
    detector = GusNetDetector(model_key=bias_key)
    analyzer = AttentionBiasAnalyzer()
    is_gpt2 = "gpt2" in model_name.lower()

    checkpoint_path = CHECKPOINT_DIR / "topk_ablation.pkl"
    sig = _args_signature("topk_ablation_v1", model_name, k_max, max_tokens)

    cumulative_impact: Dict[int, List[float]] = defaultdict(list)
    processed_keys: set = set()
    n_ok = 0

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state is not None and state.get("signature") == sig:
            cumulative_impact = defaultdict(list, state["cumulative_impact"])
            processed_keys = state["processed_keys"]
            n_ok = state.get("n_ok", 0)
            print(f"Resuming from checkpoint: {len(processed_keys)} "
                  f"sentences already done.", flush=True)
        elif state is not None:
            print("Checkpoint exists but parameters differ — starting fresh.",
                  flush=True)

    def _snapshot() -> Dict[str, Any]:
        return {
            "signature": sig,
            "cumulative_impact": dict(cumulative_impact),
            "processed_keys": processed_keys,
            "n_ok": n_ok,
        }

    start = time.time()
    interrupted = False
    with _GracefulInterrupt() as gi:
        since_last_save = 0
        for i, entry in enumerate(biased_only):
            if gi.interrupted:
                interrupted = True
                break
            if i % 5 == 0:
                print(f"  [{i+1}/{len(biased_only)}] elapsed {time.time() - start:.0f}s",
                      flush=True)
            text = entry.get("text", "")
            key = _text_hash(text)
            if key in processed_keys:
                continue
            try:
                result = heavy_compute(text, model_name)
                if result is None:
                    processed_keys.add(key)
                    continue
                tokens = list(getattr(result, "tokens", []) or [])
                attentions = getattr(result, "attentions", None)
                if attentions is None or not tokens:
                    processed_keys.add(key)
                    continue
                biased_mask = _detect_biased_indices_for_attention_tokens(
                    text, tokens, detector
                )
                if biased_mask is None or biased_mask.sum() == 0:
                    processed_keys.add(key)
                    continue
                biased_indices = set(int(i_) for i_, b in enumerate(biased_mask) if b)
                metrics = analyzer.analyze_attention_to_bias(
                    list(attentions), biased_indices, tokens
                )
                if not metrics:
                    processed_keys.add(key)
                    continue
                ranked = sorted(metrics, key=lambda m: m.bias_attention_ratio,
                                reverse=True)
                top_for_ablation = ranked[:k_max]
                tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)
                ablations = batch_ablate_top_heads(
                    encoder_model, lm_head_model, tokenizer, text,
                    top_for_ablation, is_gpt2,
                )
                if not ablations:
                    processed_keys.add(key)
                    continue
                running = 0.0
                for k, abl in enumerate(ablations, start=1):
                    running += abs(abl.representation_impact or 0.0)
                    cumulative_impact[k].append(running)
                n_ok += 1
                processed_keys.add(key)
            except Exception as e:
                processed_keys.add(key)
                _logger.debug("ablation sentence %d failed: %s", i, e)
                continue

            since_last_save += 1
            if since_last_save >= checkpoint_every:
                _save_checkpoint(checkpoint_path, _snapshot())
                since_last_save = 0
        _save_checkpoint(checkpoint_path, _snapshot())

    if interrupted:
        print(f"\nInterrupted — checkpoint saved at {checkpoint_path}. "
              f"Re-run to resume.", flush=True)
        return {"interrupted": True, "checkpoint": str(checkpoint_path),
                "processed_so_far": len(processed_keys), "n_ok": n_ok}

    summary: Dict[str, Any] = {
        "n_sentences_attempted": len(biased_only),
        "n_sentences_ok": n_ok,
        "model": model_name,
        "k_max": k_max,
        "current_default_k": 10,
        "mean_cumulative_impact_by_k": {},
        "elbow_recommendation": None,
    }
    if n_ok > 0:
        means = [
            (k, float(np.mean(cumulative_impact[k]))) for k in sorted(cumulative_impact)
        ]
        summary["mean_cumulative_impact_by_k"] = {str(k): m for k, m in means}
        # Find K such that incremental impact from k-1 to k is < 5% of total
        # (rule-of-thumb elbow).
        total = means[-1][1] if means else 0.0
        if total > 0:
            for k, m in means[1:]:
                prev_m = next(mm for kk, mm in means if kk == k - 1)
                marginal = m - prev_m
                if marginal < 0.05 * total:
                    summary["elbow_recommendation"] = int(k - 1)
                    break

    out_path = RESULTS_DIR / "topk_ablation.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if cumulative_impact:
            ks = sorted(cumulative_impact)
            ys = [float(np.mean(cumulative_impact[k])) for k in ks]
            std = [float(np.std(cumulative_impact[k])) for k in ks]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.errorbar(ks, ys, yerr=std, fmt="-o", color="#ff5ca9",
                        ecolor="#ffd5e7", capsize=3, label="Mean cumulative impact")
            ax.axvline(10, color="#1d4ed8", linestyle=":",
                       label="Current default K=10")
            if summary["elbow_recommendation"] is not None:
                ax.axvline(summary["elbow_recommendation"], color="#16a34a",
                           linestyle="--",
                           label=f"Elbow K={summary['elbow_recommendation']}")
            ax.set_xlabel("K (heads ablated, ranked by BAR)")
            ax.set_ylabel("Cumulative |representation impact|")
            ax.set_title(f"Top-K head ablation impact curve ({model_name})")
            ax.legend()
            fig.tight_layout()
            plot_path = RESULTS_DIR / "topk_ablation_curve.png"
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"Saved {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
    # Successful completion: remove the checkpoint so the next run starts fresh.
    try:
        checkpoint_path.unlink(missing_ok=True)
    except Exception:
        pass
    return summary


# ──────────────────────────────────────────────────────────────────────
# Recovery mode: summarise from an existing checkpoint
# ──────────────────────────────────────────────────────────────────────

def _summarise_existing_checkpoints(args) -> int:
    """Re-emit the summary JSON + plot from the on-disk checkpoint(s).

    Useful when the live run completed the data-collection loop but the
    summary step itself raised (e.g. OOM building giant arrays). The
    checkpoint is preserved on every save, so the recorded data is
    intact even if the summary code path crashed.
    """
    print("Recovery mode: summarising from checkpoint(s) on disk.")
    bar_ckpt = CHECKPOINT_DIR / "bar_bsr.pkl"
    if bar_ckpt.is_file():
        state = _load_checkpoint(bar_ckpt)
        if state is None:
            print(f"Could not load {bar_ckpt}.")
        else:
            n_models = len(state.get("bar_obs", {}))
            print(f"Found BAR/BSR checkpoint with {n_models} model(s), "
                  f"{len(state.get('processed_keys', set()))} entries.")
            # Re-run the function body would re-process; instead, pass
            # n=0 and re-use existing checkpoint, then it will skip
            # everything (all keys are in processed_keys) and go
            # straight to summary.
            try:
                run_bar_bsr_analysis(
                    n_sentences=0,
                    n_permutations=state.get("signature", "")[-3:] or 200,  # ignored
                    models=list(state.get("bar_obs", {}).keys()),
                    max_tokens=None,
                    resume=True,
                    checkpoint_every=10**9,  # never re-save
                )
            except Exception:
                traceback.print_exc()
    else:
        print(f"No BAR/BSR checkpoint at {bar_ckpt}.")

    abl_ckpt = CHECKPOINT_DIR / "topk_ablation.pkl"
    if abl_ckpt.is_file():
        print(f"Found ablation checkpoint at {abl_ckpt}.")
        state = _load_checkpoint(abl_ckpt)
        if state is not None:
            try:
                run_topk_ablation(
                    n_sentences=0, k_max=20,
                    model_name="bert-base-uncased",
                    max_tokens=None,
                    resume=True,
                    checkpoint_every=10**9,
                )
            except Exception:
                traceback.print_exc()
    else:
        print(f"No ablation checkpoint at {abl_ckpt}.")
    return 0


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _parse_n_arg(value: str) -> int:
    """Accept ``all``/``0``/``-1`` as 'use the whole dataset'."""
    if value is None:
        return 0
    s = str(value).strip().lower()
    if s in {"all", "full", "max", "0", "-1"}:
        return 0
    try:
        n = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--n must be an integer or 'all', got {value!r}"
        ) from e
    return max(0, n)


def _estimate_runtime(n_sentences: int, n_models: int = 2,
                      seconds_per_unit: float = 3.65) -> str:
    """Rough human-readable runtime estimate based on past benchmarks.

    The 3.65 s/(sentence, model) baseline comes from a measured CPU run
    of the BAR/BSR analysis at n=300 (BERT+GPT-2). GPU is much faster.
    """
    secs = n_sentences * n_models * seconds_per_unit
    if secs < 120:
        return f"~{secs:.0f}s"
    if secs < 3600:
        return f"~{secs/60:.0f}min"
    return f"~{secs/3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Attention Atlas bias-pipeline thresholds."
    )
    parser.add_argument("--bar", action="store_true",
                        help="Run BAR/BSR permutation-null analysis.")
    parser.add_argument("--gusnet", action="store_true",
                        help="Run GUS-Net per-category threshold calibration.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run top-K head ablation cumulative-impact analysis.")
    parser.add_argument("--all", action="store_true",
                        help="Run all three analyses.")
    parser.add_argument("--n", type=_parse_n_arg, default=200,
                        help="Sentences for BAR/BSR (int or 'all'; default 200).")
    parser.add_argument("--perms", type=int, default=200,
                        help="Permutations per (sentence, head) (default 200).")
    parser.add_argument("--abl-n", type=_parse_n_arg, default=50,
                        help="Sentences for top-K ablation (int or 'all'; default 50).")
    parser.add_argument("--k-max", type=int, default=20,
                        help="Max K for ablation curve (default 20).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Skip sentences with more than this many whitespace "
                             "tokens (cheap filter to keep runtime bounded). "
                             "Default: no filter.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing checkpoint and start fresh. "
                             "Default: resume if a matching checkpoint exists.")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save checkpoint every N processed sentences "
                             "(default 20 for BAR/BSR, half that for ablation).")
    parser.add_argument("--summarise-from-checkpoint", action="store_true",
                        help="Skip the analysis loop and only emit the summary "
                             "JSON + plot from the existing checkpoint. Useful "
                             "to recover when the summary step itself crashed.")
    args = parser.parse_args()

    if args.summarise_from_checkpoint:
        return _summarise_existing_checkpoints(args)

    if args.all:
        args.bar = args.gusnet = args.ablation = True
    if not (args.bar or args.gusnet or args.ablation):
        parser.print_help()
        return 1

    # Estimate runtime if --n is large or 'all'.
    if args.bar:
        if args.n == 0:
            # Total v9 size after optional max-tokens filter is computed by
            # the loader; just give a ballpark of the worst case.
            est = _estimate_runtime(10304, n_models=2)
            print(f"NOTE: --bar on the full v9 dataset (~10,304 sentences x 2 models) "
                  f"will take {est} on CPU. Use a GPU or reduce --n to shorten.")
        elif args.n >= 1000:
            est = _estimate_runtime(args.n, n_models=2)
            print(f"NOTE: --bar with --n {args.n} will take {est} on CPU.")

    # Ensure UTF-8 output on Windows consoles that default to cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    resume = not args.no_resume
    if args.bar:
        print("\n========== BAR/BSR analysis ==========")
        try:
            run_bar_bsr_analysis(
                n_sentences=args.n,
                n_permutations=args.perms,
                max_tokens=args.max_tokens,
                resume=resume,
                checkpoint_every=args.checkpoint_every,
            )
        except Exception:
            traceback.print_exc()
    if args.gusnet:
        print("\n========== GUS-Net calibration ==========")
        try:
            run_gusnet_calibration()
        except Exception:
            traceback.print_exc()
    if args.ablation:
        print("\n========== Top-K ablation curve ==========")
        try:
            run_topk_ablation(
                n_sentences=args.abl_n,
                k_max=args.k_max,
                max_tokens=args.max_tokens,
                resume=resume,
                checkpoint_every=max(1, args.checkpoint_every // 2),
            )
        except Exception:
            traceback.print_exc()

    print(f"\nResults in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
