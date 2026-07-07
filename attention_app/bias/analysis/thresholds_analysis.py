"""Empirical threshold calibration for the Attention Atlas bias pipeline.

This script produces defensible values for four currently-hardcoded
thresholds:

  1. BAR (Bias Attention Ratio) - calibrated to 2.5 in
     ``attention_app/bias/attention_bias.py`` (was 1.5 pre-calibration;
     the ``current_default=1.5`` in the JSON output is kept as a
     historical anchor so the calibration delta stays visible).
  2. BSR (Bias Self-Reinforcement) - same family of metric as BAR.
  3. GUS-Net per-category thresholds (GEN / UNFAIR / STEREO) -
     currently 0.5 in the dashboard sliders.
  4. Top-K heads for ablation - currently ``min(10, len(metrics))``
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
    python -m attention_app.bias.analysis.thresholds_analysis --all --n 5 --perms 20 --abl-n 5

    # Full run (~30-60 min on CPU, ~5-10 min on GPU)
    python -m attention_app.bias.analysis.thresholds_analysis --all --n 300 --perms 200 --abl-n 100

    # Individual parts
    python -m attention_app.bias.analysis.thresholds_analysis --bar --n 300 --perms 200
    python -m attention_app.bias.analysis.thresholds_analysis --gusnet
    python -m attention_app.bias.analysis.thresholds_analysis --ablation --abl-n 100

    # Recalibrate on the GUS-Net fine-tuned trunks (the dashboard's default
    # attention source) - outputs/checkpoints are tagged '_gusnet' /
    # 'gusnet_bert' so they never overwrite the base-model calibration:
    python -m attention_app.bias.analysis.thresholds_analysis --bar --n 0 --perms 200 --encoders gusnet
    python -m attention_app.bias.analysis.thresholds_analysis --faithfulness --faith-n 0 --faith-model both --encoders gusnet
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
HERE = Path(__file__).resolve().parent           # attention_app/bias/analysis/
ROOT = HERE.parent.parent.parent                 # repository root
RESULTS_DIR = ROOT / "dataset" / "thresholds_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Per-analysis subfolders ────────────────────────────────────────
# Keeps the results folder navigable: each analysis writes only to
# its own subfolder. Created on import so callers can write directly.
BAR_BSR_DIR = RESULTS_DIR / "bar_bsr"
TOPK_DIR = RESULTS_DIR / "topk_ablation"
FAITHFULNESS_DIR = RESULTS_DIR / "faithfulness"
GUSNET_DIR = RESULTS_DIR / "gusnet"
for _d in (BAR_BSR_DIR, TOPK_DIR, FAITHFULNESS_DIR, GUSNET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Make the app package importable when this is run as a script (e.g.
# python attention_app/bias/analysis/thresholds_analysis.py) rather than as
# `python -m attention_app.bias.analysis.thresholds_analysis`.
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
    """Stable short hash of a sentence - used to key processed entries."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _args_signature(*parts: Any) -> str:
    """Hash that identifies the parameters of a run. A checkpoint is only
    reused when the new run has the same signature; otherwise we start
    fresh because the accumulators would mix incompatible data."""
    return hashlib.sha1(repr(parts).encode("utf-8")).hexdigest()[:16]


def _save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Atomic pickle write - temp file then rename."""
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
            # Second Ctrl-C - restore default handler and re-raise.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\nSecond interrupt - aborting now.", flush=True)
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
        this are dropped before sampling - a cheap proxy that keeps
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


_ATTN_SPECIAL_TOKENS = frozenset({
    "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
    "<|endoftext|>", "<s>", "</s>", "<pad>", "<unk>",
})


def _is_pure_punctuation(token: str) -> bool:
    """True if the token contains no alphanumeric characters (after stripping
    WordPiece/BPE markers). Punctuation is never a bias carrier."""
    clean = token.replace("##", "").replace("Ġ", "").replace("Ġ", "")
    return bool(clean) and not any(c.isalnum() for c in clean)


def _merge_subtokens_to_words(
    tokens: List[str],
    special_tokens: Optional[set] = None,
) -> List[Tuple[str, List[int]]]:
    """Merge a sub-token sequence into whole words.

    Returns a list of ``(lowercased_word, [sub_token_indices])``. Handles
    both WordPiece (``##`` = continuation) and BPE (``Ġ`` = word start)
    auto-detected from marker presence. Punctuation and special tokens act
    as word separators and are not themselves emitted as words - both are
    never bias carriers so dropping them keeps the alignment clean.
    """
    if special_tokens is None:
        special_tokens = _ATTN_SPECIAL_TOKENS
    has_bert_marker = any(t.startswith("##") for t in tokens)
    has_gpt2_marker = any("Ġ" in t for t in tokens)

    words: List[Tuple[str, List[int]]] = []
    current_word = ""
    current_indices: List[int] = []

    def flush():
        nonlocal current_word, current_indices
        if current_indices:
            words.append((current_word.lower(), list(current_indices)))
        current_word = ""
        current_indices = []

    for i, tok in enumerate(tokens):
        if tok in special_tokens:
            flush()
            continue
        if _is_pure_punctuation(tok):
            # Break the current word but don't emit punctuation as one.
            flush()
            continue

        if has_gpt2_marker:
            # BPE: Ġ marks a new word; otherwise continuation of the current.
            if "Ġ" in tok:
                flush()
                current_word = tok.replace("Ġ", "")
                current_indices = [i]
            else:
                current_word += tok
                current_indices.append(i)
        elif has_bert_marker:
            # WordPiece: ## marks continuation; otherwise new word.
            if tok.startswith("##"):
                current_word += tok[2:]
                current_indices.append(i)
            else:
                flush()
                current_word = tok
                current_indices = [i]
        else:
            # No marker characters detected - treat each token as its own word.
            flush()
            current_word = tok
            current_indices = [i]
    flush()
    return words


def _detect_biased_indices_for_attention_tokens(
    text: str, attention_tokens: List[str], detector
) -> Optional[np.ndarray]:
    """Run GUS-Net and return a boolean mask over the *attention model's*
    sub-tokens, marking those that belong to a word GUS-Net labelled biased.

    Whole-word alignment: rather than matching raw sub-token strings (which
    failed badly for BPE sub-words like GPT-2's ``["Ġster","eot","yp","ical"]``
    where no sub-piece equals the GUS-Net biased word ``"stereotypical"``),
    we reconstruct whole words on both sides and compare those.

    Returns ``None`` when GUS-Net found nothing or the mask ends up empty.
    """
    try:
        gus_tokens, gus_probs = detector.predict_proba(text)
        labeled = detector.apply_thresholds(gus_tokens, gus_probs)
    except Exception:
        return None

    gus_special = set(detector.config.get("special_tokens", _ATTN_SPECIAL_TOKENS))
    biased_sub_indices = {
        int(item["index"]) for item in labeled if item.get("is_biased")
    }
    if not biased_sub_indices:
        return None

    # Reconstruct whole words on the GUS-Net side; a word is biased if any
    # of its sub-tokens was flagged.
    gus_words = _merge_subtokens_to_words(list(gus_tokens), gus_special)
    biased_words: set = set()
    for word, sub_ix in gus_words:
        if not word:
            continue
        if any(s in biased_sub_indices for s in sub_ix):
            biased_words.add(word)
    if not biased_words:
        return None

    # Reconstruct whole words on the attention side and mark every sub-token
    # of a matched word.
    n = len(attention_tokens)
    mask = np.zeros(n, dtype=bool)
    attn_words = _merge_subtokens_to_words(list(attention_tokens))
    for word, sub_ix in attn_words:
        if word and word in biased_words:
            for s in sub_ix:
                if 0 <= s < n:
                    mask[s] = True
    return mask if mask.any() else None


CATEGORIES = ("GEN", "UNFAIR", "STEREO")


# ──────────────────────────────────────────────────────────────────────
# Audited gold labels (human token-level audit of 300-sample, n=125)
# ──────────────────────────────────────────────────────────────────────
# The audit set replaces GUS-Net's live predictions with human-verified
# token labels, so re-running BAR/BSR and faithfulness against it
# quantifies how much the calibrated thresholds depend on GUS-Net label
# noise. Labels are per whitespace token, multi-label via "+"
# (e.g. "GEN+STEREO"); alignment to attention sub-tokens reuses the
# whole-word strategy from _detect_biased_indices_for_attention_tokens.

AUDITED_LABELS_DEFAULT = ROOT / "dataset" / "human_token_labels_v9_300.jsonl"


def _clean_word(tok: str) -> str:
    """Normalise a whitespace token for whole-word matching: lowercase
    and strip leading/trailing punctuation (audit tokens keep attached
    punctuation, e.g. ``"drivers."``)."""
    return tok.strip().strip(".,!?;:\"'()[]{}").lower()


def load_audited_labels(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load the human-audited token labels into a lookup keyed by text hash.

    Returns ``{text_hash: {"entry": <v9-like dict>, "biased_words": set,
    "cat_words": {GEN|UNFAIR|STEREO: set}}}``. Only records with
    ``audited=True`` are included.
    """
    path = path or AUDITED_LABELS_DEFAULT
    lookup: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("audited"):
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            biased_words: set = set()
            cat_words: Dict[str, set] = {c: set() for c in CATEGORIES}
            for tok, lab in zip(rec.get("tokens", []), rec.get("labels", [])):
                if not lab or lab.strip() == "O":
                    continue
                word = _clean_word(tok)
                if not word:
                    continue
                biased_words.add(word)
                for part in lab.replace(",", "+").split("+"):
                    part = part.strip()
                    if part in cat_words:
                        cat_words[part].add(word)
            lookup[_text_hash(text)] = {
                "entry": {
                    "id": rec.get("id"),
                    "text": text,
                    "has_bias": rec.get("has_bias", False),
                    "bias_type": rec.get("bias_type"),
                },
                "biased_words": biased_words,
                "cat_words": cat_words,
            }
    return lookup


def _mask_from_word_set(words: set, attention_tokens: List[str]) -> Optional[np.ndarray]:
    """Boolean mask over attention sub-tokens whose merged whole word is
    in ``words``. Returns None when the mask comes out empty."""
    if not words:
        return None
    n = len(attention_tokens)
    mask = np.zeros(n, dtype=bool)
    for word, sub_ix in _merge_subtokens_to_words(list(attention_tokens)):
        if word and word in words:
            for s in sub_ix:
                if 0 <= s < n:
                    mask[s] = True
    return mask if mask.any() else None


def _make_audited_mask_fns(audited_lookup: Dict[str, Dict[str, Any]]):
    """Return drop-in replacements for the two GUS-Net mask functions that
    read from the audited gold labels instead. Same signatures (the
    ``detector`` argument is accepted and ignored)."""

    def detect_combined(text: str, attention_tokens: List[str], detector=None):
        rec = audited_lookup.get(_text_hash((text or "").strip()))
        if rec is None:
            return None
        return _mask_from_word_set(rec["biased_words"], attention_tokens)

    def detect_per_category(text: str, attention_tokens: List[str], detector=None):
        rec = audited_lookup.get(_text_hash((text or "").strip()))
        if rec is None:
            return None
        n = len(attention_tokens)
        cat_masks = {}
        any_hit = False
        for c in CATEGORIES:
            m = _mask_from_word_set(rec["cat_words"][c], attention_tokens)
            if m is None:
                m = np.zeros(n, dtype=bool)
            elif m.any():
                any_hit = True
            cat_masks[c] = m
        return cat_masks if any_hit else None

    return detect_combined, detect_per_category


def _detect_per_category_indices(
    text: str, attention_tokens: List[str], detector
) -> Optional[Dict[str, np.ndarray]]:
    """Like ``_detect_biased_indices_for_attention_tokens`` but returns one
    boolean mask per GUS-Net category (GEN, UNFAIR, STEREO).

    Categories overlap: a token may be both GEN and STEREO. The returned dict
    always has all three keys, but a category with no detected tokens gets an
    all-False mask. Returns ``None`` when GUS-Net found nothing at all.
    """
    try:
        gus_tokens, gus_probs = detector.predict_proba(text)
        labeled = detector.apply_thresholds(gus_tokens, gus_probs)
    except Exception:
        return None

    gus_special = set(detector.config.get("special_tokens", _ATTN_SPECIAL_TOKENS))

    # Per-category sub-token indices on the GUS-Net side.
    cat_sub_indices: Dict[str, set] = {c: set() for c in CATEGORIES}
    for item in labeled:
        if not item.get("is_biased"):
            continue
        for cat in item.get("bias_types", []) or []:
            if cat in cat_sub_indices:
                cat_sub_indices[cat].add(int(item["index"]))

    if not any(cat_sub_indices.values()):
        return None

    # Whole-word reconstruction on the GUS-Net side (per category).
    gus_words = _merge_subtokens_to_words(list(gus_tokens), gus_special)
    cat_words: Dict[str, set] = {c: set() for c in CATEGORIES}
    for word, sub_ix in gus_words:
        if not word:
            continue
        for cat, indices in cat_sub_indices.items():
            if any(s in indices for s in sub_ix):
                cat_words[cat].add(word)

    # Map back to attention sub-tokens.
    n = len(attention_tokens)
    attn_words = _merge_subtokens_to_words(list(attention_tokens))
    cat_masks: Dict[str, np.ndarray] = {
        c: np.zeros(n, dtype=bool) for c in CATEGORIES
    }
    for word, sub_ix in attn_words:
        if not word:
            continue
        for cat, words_set in cat_words.items():
            if word in words_set:
                for s in sub_ix:
                    if 0 <= s < n:
                        cat_masks[cat][s] = True

    if not any(m.any() for m in cat_masks.values()):
        return None
    return cat_masks


# ──────────────────────────────────────────────────────────────────────
# Analysis 1 - BAR / BSR permutation null
# ──────────────────────────────────────────────────────────────────────

def run_bar_bsr_analysis(n_sentences: int = 300, n_permutations: int = 200,
                         models: Optional[List[str]] = None,
                         max_tokens: Optional[int] = None,
                         resume: bool = True,
                         checkpoint_every: int = 20,
                         per_category: bool = False,
                         audited_labels: Optional[Path] = None) -> Dict[str, Any]:
    """Compute observed and permutation-null BAR/BSR across a v9 subsample.

    Supports resumable runs: periodic checkpoints are written to
    ``dataset/thresholds_results/checkpoints/bar_bsr.pkl`` so that
    Ctrl-C or a crash does not lose progress. Pass ``resume=False`` to
    force a fresh run.

    When ``per_category=True`` we also compute separate BAR/BSR distributions
    for each GUS-Net category (GEN, UNFAIR, STEREO), with their own
    permutation nulls (random mask of size |B_C|). Adds roughly 3x the
    permutation work but uses the same forward passes, so wall-clock cost
    is about 1.5-2x.
    """
    if models is None:
        models = ["bert-base-uncased", "gpt2"]

    # Audited mode: restrict the corpus to the human-audited sentences and
    # swap the GUS-Net mask function for gold-label lookups.
    audited_lookup = None
    detect_combined_fn = _detect_biased_indices_for_attention_tokens
    detect_per_cat_fn = _detect_per_category_indices
    if audited_labels is not None:
        audited_lookup = load_audited_labels(audited_labels)
        detect_combined_fn, detect_per_cat_fn = _make_audited_mask_fns(audited_lookup)
        sample = [rec["entry"] for rec in audited_lookup.values()]
        print(f"AUDITED mode: {len(sample)} gold-labelled sentences "
              f"from {audited_labels}", flush=True)
    else:
        sample = load_v9_stratified(n_sentences, max_tokens=max_tokens)
    print(f"BAR/BSR: {len(sample)} sentences x {len(models)} models, "
          f"{n_permutations} permutations per (sentence, head)"
          f"{' [per-category mode]' if per_category else ''}"
          f"{' [audited labels]' if audited_labels else ''}", flush=True)

    heavy_compute = _get_heavy_compute()
    GusNetDetector = _get_gusnet_detector()

    rng = np.random.default_rng(42)
    detector_cache: Dict[str, Any] = {}

    # Encoder-family tag: GUS-Net-trunk calibrations get their own
    # checkpoint AND output files - overwriting the base-model artefacts
    # would silently mix two different null distributions.
    _enc_tag = "_gusnet" if any("gus" in m.lower() for m in models) else ""
    _aud_tag = "_audited" if audited_labels else ""
    checkpoint_path = CHECKPOINT_DIR / (
        f"bar_bsr_per_category{_enc_tag}{_aud_tag}.pkl" if per_category
        else f"bar_bsr{_enc_tag}{_aud_tag}.pkl"
    )
    sig = _args_signature(
        "bar_bsr_v2" if per_category else "bar_bsr_v1",
        tuple(models), n_permutations, max_tokens, per_category,
        str(audited_labels) if audited_labels else None,
    )

    # Per-model aggregates + bookkeeping. ``array.array('f')`` stores
    # 4 bytes per float vs ~28 for a Python list of float objects, so
    # we can hold ~300M permutation samples in ~1.2 GB instead of ~9 GB.
    bar_obs: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bar_null: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bsr_obs: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    bsr_null: Dict[str, "array.array"] = {m: array.array("f") for m in models}
    # Per-category accumulators: only populated when per_category=True.
    # Structure: dict[model][category] -> array.array("f")
    def _empty_per_cat():
        return {m: {c: array.array("f") for c in CATEGORIES} for m in models}
    bar_obs_cat = _empty_per_cat() if per_category else None
    bar_null_cat = _empty_per_cat() if per_category else None
    bsr_obs_cat = _empty_per_cat() if per_category else None
    bsr_null_cat = _empty_per_cat() if per_category else None
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

    def _as_arr_cat(state_cat):
        """Restore per-category accumulator dict from checkpoint state."""
        return {m: {c: _as_arr(state_cat[m][c]) for c in CATEGORIES}
                for m in models}

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state is not None and state.get("signature") == sig:
            bar_obs = {m: _as_arr(state["bar_obs"][m]) for m in models}
            bar_null = {m: _as_arr(state["bar_null"][m]) for m in models}
            bsr_obs = {m: _as_arr(state["bsr_obs"][m]) for m in models}
            bsr_null = {m: _as_arr(state["bsr_null"][m]) for m in models}
            if per_category and "bar_obs_cat" in state:
                bar_obs_cat = _as_arr_cat(state["bar_obs_cat"])
                bar_null_cat = _as_arr_cat(state["bar_null_cat"])
                bsr_obs_cat = _as_arr_cat(state["bsr_obs_cat"])
                bsr_null_cat = _as_arr_cat(state["bsr_null_cat"])
            processed_keys = state["processed_keys"]
            n_ok = state.get("n_ok", 0)
            n_skipped = state.get("n_skipped", 0)
            print(f"Resuming from checkpoint: {len(processed_keys)} "
                  f"(model, sentence) pairs already done.", flush=True)
        elif state is not None:
            print("Checkpoint exists but parameters differ. Starting fresh.",
                  flush=True)

    def _snapshot() -> Dict[str, Any]:
        snap = {
            "signature": sig,
            "bar_obs": bar_obs, "bar_null": bar_null,
            "bsr_obs": bsr_obs, "bsr_null": bsr_null,
            "processed_keys": processed_keys,
            "n_ok": n_ok, "n_skipped": n_skipped,
        }
        if per_category:
            snap["bar_obs_cat"] = bar_obs_cat
            snap["bar_null_cat"] = bar_null_cat
            snap["bsr_obs_cat"] = bsr_obs_cat
            snap["bsr_null_cat"] = bsr_null_cat
        return snap

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
                    biased_mask = detect_combined_fn(
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
                    # Per-category masks (only when requested).
                    cat_masks = None
                    cat_n: Dict[str, int] = {}
                    if per_category:
                        cat_masks = detect_per_cat_fn(
                            text, tokens, detector
                        )
                        if cat_masks is not None:
                            cat_masks = {
                                c: m[:seq_len] for c, m in cat_masks.items()
                            }
                            cat_n = {
                                c: int(m.sum()) for c, m in cat_masks.items()
                            }
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
                            # Per-category metrics + nulls.
                            if cat_masks is not None:
                                for c in CATEGORIES:
                                    nc = cat_n.get(c, 0)
                                    if nc == 0 or nc >= seq_len:
                                        continue
                                    bc, sc = compute_bar_bsr(
                                        head_attn, cat_masks[c]
                                    )
                                    if bc is None:
                                        continue
                                    bar_obs_cat[model_name][c].append(bc)
                                    bsr_obs_cat[model_name][c].append(sc)
                                    bnc, snc = permute_bar_bsr(
                                        head_attn, nc, n_permutations, rng
                                    )
                                    bar_null_cat[model_name][c].extend(bnc)
                                    bsr_null_cat[model_name][c].extend(snc)
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
        # Save the final state - even on interrupt - so the next run resumes.
        _save_checkpoint(checkpoint_path, _snapshot())

    elapsed = time.time() - start
    if interrupted:
        print(f"\nInterrupted after {elapsed:.1f}s - checkpoint saved at "
              f"{checkpoint_path}. Re-run the same command to resume.",
              flush=True)
        return {"interrupted": True, "checkpoint": str(checkpoint_path),
                "processed_so_far": len(processed_keys),
                "n_ok": n_ok, "n_skipped": n_skipped}
    print(f"\nDone in {elapsed:.1f}s - ok={n_ok}, skipped={n_skipped}")

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
    # only as numpy views - never materialise a giant Python list.
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

    # Per-category breakdown (combined across models + per model).
    if per_category:
        per_cat_combined: Dict[str, Any] = {}
        per_cat_by_model: Dict[str, Dict[str, Any]] = {m: {} for m in models}
        for c in CATEGORIES:
            all_cat_bar_obs = _concat({m: bar_obs_cat[m][c] for m in models})
            all_cat_bar_null = _concat({m: bar_null_cat[m][c] for m in models})
            all_cat_bsr_obs = _concat({m: bsr_obs_cat[m][c] for m in models})
            all_cat_bsr_null = _concat({m: bsr_null_cat[m][c] for m in models})
            per_cat_combined[c] = {
                "BAR": summarise(all_cat_bar_obs, all_cat_bar_null, default=1.5),
                "BSR": summarise(all_cat_bsr_obs, all_cat_bsr_null, default=1.5),
            }
            for m in models:
                per_cat_by_model[m][c] = {
                    "BAR": summarise(bar_obs_cat[m][c], bar_null_cat[m][c],
                                     default=1.5),
                    "BSR": summarise(bsr_obs_cat[m][c], bsr_null_cat[m][c],
                                     default=1.5),
                }
        summary["per_category"] = {
            "combined": per_cat_combined,
            "by_model": per_cat_by_model,
        }

    out_path = BAR_BSR_DIR / (
        f"per_category{_enc_tag}{_aud_tag}.json" if per_category
        else f"thresholds{_enc_tag}{_aud_tag}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")

    # Plot histograms - subsample to keep matplotlib responsive.
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
        plot_path = BAR_BSR_DIR / f"distribution{_enc_tag}{_aud_tag}.png"
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
# Analysis 2 - GUS-Net per-category threshold calibration
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
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        raise RuntimeError("scikit-learn is required for calibration") from e

    def _f1_opt_threshold(y, s):
        """Threshold that maximises F1 on (y, s); returns (th, f1)."""
        precision, recall, ths = precision_recall_curve(y, s)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_aligned = f1[:-1]
        if len(f1_aligned) == 0 or len(ths) == 0:
            return None, None
        best_idx = int(np.argmax(f1_aligned))
        return float(ths[best_idx]), float(f1_aligned[best_idx])

    def _cv5_f1(y, s, seed=42):
        """5-fold CV estimate of the F1-optimal-threshold procedure.

        The in-sample best_f1 selects the threshold on the SAME data it is
        evaluated on, which is optimistic on a ~300-sentence audit set.
        Here the threshold is chosen on the train folds and F1 measured on
        the held-out fold - the honest generalisation estimate.
        Returns (f1_mean, f1_std, threshold_mean) or (None, None, None).
        """
        y = np.asarray(y)
        s = np.asarray(s)
        if len(np.unique(y)) < 2 or len(y) < 10:
            return None, None, None
        f1s, ths_used = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(s.reshape(-1, 1), y):
            th, _ = _f1_opt_threshold(y[tr_idx], s[tr_idx])
            if th is None:
                continue
            preds = s[te_idx] >= th
            f1s.append(float(f1_score(y[te_idx], preds, zero_division=0)))
            ths_used.append(th)
        if not f1s:
            return None, None, None
        return (float(np.mean(f1s)), float(np.std(f1s)),
                float(np.mean(ths_used)))

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
                best_th, best_f1 = _f1_opt_threshold(truths_np, scores_np)
                if best_th is None:
                    continue
                auc = float(roc_auc_score(truths_np, scores_np))
                cv_f1, cv_f1_std, cv_th = _cv5_f1(truths_np, scores_np)
            except Exception as e:
                _logger.debug("PR curve for %s failed: %s", cat, e)
                continue
            model_result[cat] = {
                "current_default": 0.5,
                "recommended_threshold_f1_opt": best_th,
                # In-sample: the threshold is selected AND evaluated on the
                # same audit set, so this F1 is optimistic. Cite cv5_f1_mean.
                "best_f1_in_sample": best_f1,
                "cv5_f1_mean": cv_f1,
                "cv5_f1_std": cv_f1_std,
                "cv5_threshold_mean": cv_th,
                "auc_roc": auc,
                "score_mean_pos": float(scores_np[truths_np].mean()) if truths_np.any() else None,
                "score_mean_neg": float(scores_np[~truths_np].mean()) if (~truths_np).any() else None,
            }
        # Combined: use max across categories.
        if max_any and len(max_any) == len(truths):
            try:
                any_np = np.asarray(max_any)
                best_th, best_f1 = _f1_opt_threshold(truths_np, any_np)
                if best_th is not None:
                    cv_f1, cv_f1_std, cv_th = _cv5_f1(truths_np, any_np)
                    model_result["ANY_CATEGORY"] = {
                        "recommended_threshold_f1_opt": best_th,
                        "best_f1_in_sample": best_f1,
                        "cv5_f1_mean": cv_f1,
                        "cv5_f1_std": cv_f1_std,
                        "cv5_threshold_mean": cv_th,
                        "auc_roc": float(roc_auc_score(truths_np, any_np)),
                    }
            except Exception:
                pass
        summary["models"][model_key] = model_result

    out_path = GUSNET_DIR / "thresholds.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Analysis 2b - GUS-Net per-category TOKEN-level threshold calibration
# against the human-audited gold labels
# ──────────────────────────────────────────────────────────────────────

def run_gusnet_token_calibration(
    audited_labels: Optional[Path] = None,
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Re-derive GUS-Net's per-category thresholds on the audited gold set.

    The registry ``optimized_thresholds`` arrays were tuned during GUS-Net
    training on the Powers et al. validation split (see MD section 18).
    This function derives audit-grounded alternatives: for each category
    C, the per-token score is ``max(B-prob, I-prob)`` (matching the
    detector's ``apply_thresholds`` logic) and the threshold is swept over
    the precision-recall curve against the gold token labels.

    Alignment: gold labels live on whitespace tokens; GUS-Net predictions
    live on its own sub-tokens. Both sides are merged to whole words
    (section 6 machinery) and aligned positionally with
    ``difflib.SequenceMatcher`` so repeated words are handled correctly.
    Sub-tokens of unmatched words are excluded from the sweep.

    The gold labels are flat category sets per token (no BIO structure),
    so the sweep is per-category, not per-BIO-tag: one threshold per
    category, applied to both its B- and I- probabilities.

    Saves ``gusnet/token_thresholds_audited.json``.
    """
    import difflib

    if model_keys is None:
        model_keys = ["gusnet-bert", "gusnet-gpt2"]
    path = audited_labels or AUDITED_LABELS_DEFAULT
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("audited"):
                    recs.append(rec)
    print(f"GUS-Net token calibration: {len(recs)} audited sentences")

    GusNetDetector = _get_gusnet_detector()
    try:
        from sklearn.metrics import precision_recall_curve, roc_auc_score
    except Exception as e:
        raise RuntimeError("scikit-learn is required for calibration") from e

    def _gold_words(rec) -> List[Tuple[str, set]]:
        """Cleaned gold whole words with their category sets, in order."""
        out = []
        for tok, lab in zip(rec.get("tokens", []), rec.get("labels", [])):
            w = _clean_word(tok)
            if not w:
                continue
            cats = set()
            if lab and lab.strip() != "O":
                for part in lab.replace(",", "+").split("+"):
                    part = part.strip()
                    if part in CATEGORIES:
                        cats.add(part)
            out.append((w, cats))
        return out

    summary: Dict[str, Any] = {
        "n_sentences": len(recs),
        "audited_labels": str(path),
        "note": ("Per-category thresholds swept on gold token labels. "
                 "Score per token = max(B-prob, I-prob), matching "
                 "GusNetDetector.apply_thresholds."),
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
        special = set(cfg["special_tokens"])
        registry_opt = cfg.get("optimized_thresholds")

        scores: Dict[str, List[float]] = {c: [] for c in CATEGORIES}
        golds: Dict[str, List[int]] = {c: [] for c in CATEGORIES}
        n_aligned_words = 0
        n_gold_words = 0
        start = time.time()

        for i, rec in enumerate(recs):
            if i % 50 == 0:
                print(f"  [{i+1}/{len(recs)}] elapsed {time.time()-start:.0f}s",
                      flush=True)
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            try:
                g_tokens, g_probs = detector.predict_proba(text)
            except Exception as e:
                _logger.debug("predict_proba failed: %s", e)
                continue
            arr = (g_probs.detach().cpu().numpy()
                   if hasattr(g_probs, "detach") else np.asarray(g_probs))

            # Merge GUS-Net sub-tokens to whole words; per-word score per
            # category = max over the word's sub-tokens of max(B, I).
            gus_words = _merge_subtokens_to_words(list(g_tokens), special)
            gus_seq = [w for w, _ in gus_words]
            gold = _gold_words(rec)
            gold_seq = [w for w, _ in gold]
            n_gold_words += len(gold_seq)

            sm = difflib.SequenceMatcher(a=gold_seq, b=gus_seq, autojunk=False)
            for block in sm.get_matching_blocks():
                for off in range(block.size):
                    gold_w, gold_cats = gold[block.a + off]
                    _, sub_ix = gus_words[block.b + off]
                    valid_ix = [s for s in sub_ix if s < arr.shape[0]]
                    if not valid_ix:
                        continue
                    n_aligned_words += 1
                    for c in CATEGORIES:
                        idxs = cat_idx[c]
                        word_score = float(
                            max(arr[s, j] for s in valid_ix for j in idxs)
                        )
                        scores[c].append(word_score)
                        golds[c].append(1 if c in gold_cats else 0)

        model_result: Dict[str, Any] = {
            "n_aligned_words": n_aligned_words,
            "n_gold_words": n_gold_words,
            "alignment_rate": (round(n_aligned_words / n_gold_words, 4)
                               if n_gold_words else None),
        }
        for c in CATEGORIES:
            y = np.asarray(golds[c])
            s = np.asarray(scores[c], dtype=np.float64)
            if y.sum() == 0 or y.sum() == len(y):
                model_result[c] = {"note": "degenerate gold distribution"}
                continue
            precision, recall, ths = precision_recall_curve(y, s)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            f1_aligned = f1[:-1]
            best_idx = int(np.argmax(f1_aligned)) if len(f1_aligned) else 0
            # Metrics at the registry threshold for comparison: use the
            # B-tag entry (apply_thresholds checks each prob against its
            # own per-label cut, but B and I share the category here).
            reg_th = None
            reg_f1 = None
            if registry_opt is not None:
                reg_th = float(min(registry_opt[j] for j in cat_idx[c]))
                pred = (s >= reg_th).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                p_ = tp / max(1, tp + fp)
                r_ = tp / max(1, tp + fn)
                reg_f1 = 2 * p_ * r_ / max(1e-9, p_ + r_)
            model_result[c] = {
                "n_words": int(len(y)),
                "n_positive": int(y.sum()),
                "audit_threshold_f1_opt": float(ths[best_idx]),
                "audit_best_f1": float(f1_aligned[best_idx]),
                "audit_precision_at_opt": float(precision[best_idx]),
                "audit_recall_at_opt": float(recall[best_idx]),
                "auc_roc": float(roc_auc_score(y, s)),
                "registry_threshold_min_BI": reg_th,
                "registry_f1_on_audit": reg_f1,
            }
        summary["models"][model_key] = model_result

    out_path = GUSNET_DIR / "token_thresholds_audited.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nSaved {out_path}")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Analysis 3 - top-K cumulative ablation impact
# ──────────────────────────────────────────────────────────────────────

def _model_short(model_name: str) -> str:
    """Short tag used in output / checkpoint filenames.

    Distinguishes the GUS-Net fine-tuned encoders from the pretrained base
    models so that a --encoders gusnet calibration never overwrites the
    base-model artefacts (their nulls are NOT interchangeable)."""
    base = "gpt2" if "gpt2" in model_name.lower() else "bert"
    return f"gusnet_{base}" if "gus" in model_name.lower() else base


def run_topk_ablation(n_sentences: int = 50, k_max: int = 20,
                      model_name: str = "bert-base-uncased",
                      max_tokens: Optional[int] = None,
                      resume: bool = True,
                      checkpoint_every: int = 10,
                      per_category: bool = False) -> Dict[str, Any]:
    """For each sentence: rank heads by BAR, ablate top-1..top-K, record
    cumulative effect on the bias-class probability.

    Resumable via ``dataset/thresholds_results/checkpoints/topk_ablation_{model}.pkl``.
    Output JSON / plot are also model-suffixed so BERT and GPT-2 runs do
    not clobber each other.

    When ``per_category=True`` we additionally compute a separate
    cumulative-impact curve for each GUS-Net category (GEN, UNFAIR,
    STEREO) by re-ranking heads against the category-specific mask. The
    output JSON includes a ``per_category`` section with one elbow
    recommendation per category. Cost is ~4x because each sentence now
    runs 1 + 3 ablation batches (combined + 3 categories), though shared
    heads across rankings reduce that in practice.
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

    short = _model_short(model_name)
    checkpoint_path = CHECKPOINT_DIR / (
        f"topk_ablation_per_category_{short}.pkl" if per_category
        else f"topk_ablation_{short}.pkl"
    )
    sig = _args_signature(
        "topk_ablation_v3_per_cat" if per_category else "topk_ablation_v2",
        model_name, k_max, max_tokens, per_category,
    )

    cumulative_impact: Dict[int, List[float]] = defaultdict(list)
    # Per-category accumulators: only populated when per_category=True.
    cumulative_impact_cat: Dict[str, Dict[int, List[float]]] = {
        c: defaultdict(list) for c in CATEGORIES
    } if per_category else None
    processed_keys: set = set()
    n_ok = 0

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state is not None and state.get("signature") == sig:
            cumulative_impact = defaultdict(list, state["cumulative_impact"])
            if per_category and "cumulative_impact_cat" in state:
                cumulative_impact_cat = {
                    c: defaultdict(list, state["cumulative_impact_cat"].get(c, {}))
                    for c in CATEGORIES
                }
            processed_keys = state["processed_keys"]
            n_ok = state.get("n_ok", 0)
            print(f"Resuming from checkpoint: {len(processed_keys)} "
                  f"sentences already done.", flush=True)
        elif state is not None:
            print("Checkpoint exists but parameters differ - starting fresh.",
                  flush=True)

    def _snapshot() -> Dict[str, Any]:
        snap = {
            "signature": sig,
            "cumulative_impact": dict(cumulative_impact),
            "processed_keys": processed_keys,
            "n_ok": n_ok,
        }
        if per_category:
            snap["cumulative_impact_cat"] = {
                c: dict(cumulative_impact_cat[c]) for c in CATEGORIES
            }
        return snap

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

                # Per-category: re-rank heads against each category's mask
                # and ablate top-K_C heads independently. The category-specific
                # rankings will differ from the combined ranking, so we have
                # to call batch_ablate_top_heads once per category.
                if per_category:
                    cat_masks = _detect_per_category_indices(
                        text, tokens, detector
                    )
                    if cat_masks is not None:
                        for cat in CATEGORIES:
                            mask_c = cat_masks[cat]
                            n_c = int(mask_c.sum())
                            if n_c == 0 or n_c >= len(tokens):
                                continue
                            biased_idx_c = set(
                                int(i_) for i_, b in enumerate(mask_c) if b
                            )
                            metrics_c = analyzer.analyze_attention_to_bias(
                                list(attentions), biased_idx_c, tokens
                            )
                            if not metrics_c:
                                continue
                            ranked_c = sorted(
                                metrics_c,
                                key=lambda m: m.bias_attention_ratio,
                                reverse=True,
                            )
                            top_c = ranked_c[:k_max]
                            ablations_c = batch_ablate_top_heads(
                                encoder_model, lm_head_model, tokenizer,
                                text, top_c, is_gpt2,
                            )
                            if not ablations_c:
                                continue
                            running_c = 0.0
                            for k, abl in enumerate(ablations_c, start=1):
                                running_c += abs(abl.representation_impact or 0.0)
                                cumulative_impact_cat[cat][k].append(running_c)
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
        print(f"\nInterrupted - checkpoint saved at {checkpoint_path}. "
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
    def _summarise_curve(curve: Dict[int, List[float]]) -> Tuple[Dict[str, float], Optional[int]]:
        if not curve:
            return {}, None
        means = [
            (k, float(np.mean(curve[k]))) for k in sorted(curve)
        ]
        mean_by_k = {str(k): m for k, m in means}
        total = means[-1][1] if means else 0.0
        elbow = None
        if total > 0:
            for k, m in means[1:]:
                prev_m = next(mm for kk, mm in means if kk == k - 1)
                marginal = m - prev_m
                if marginal < 0.05 * total:
                    elbow = int(k - 1)
                    break
        return mean_by_k, elbow

    if n_ok > 0:
        mean_by_k, elbow = _summarise_curve(cumulative_impact)
        summary["mean_cumulative_impact_by_k"] = mean_by_k
        summary["elbow_recommendation"] = elbow

        if per_category:
            per_cat_summary: Dict[str, Any] = {}
            for cat in CATEGORIES:
                mb_k, el = _summarise_curve(cumulative_impact_cat[cat])
                # n_sentences_with_signal differs per category because some
                # sentences have no tokens of that category.
                n_sent = (
                    len(cumulative_impact_cat[cat][1])
                    if 1 in cumulative_impact_cat[cat] else 0
                )
                per_cat_summary[cat] = {
                    "n_sentences": n_sent,
                    "mean_cumulative_impact_by_k": mb_k,
                    "elbow_recommendation": el,
                }
            summary["per_category"] = per_cat_summary

    out_path = TOPK_DIR / (
        f"{short}_per_category.json" if per_category else f"{short}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if cumulative_impact:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ks = sorted(cumulative_impact)
            ys = [float(np.mean(cumulative_impact[k])) for k in ks]
            ax.plot(ks, ys, "-o", color="#0f172a", linewidth=2.2,
                    label="Combined (all biased tokens)")
            ax.axvline(10, color="#1d4ed8", linestyle=":",
                       label="Pre-calibration K=10")
            if summary["elbow_recommendation"] is not None:
                ax.axvline(summary["elbow_recommendation"], color="#16a34a",
                           linestyle="--",
                           label=f"Combined elbow K={summary['elbow_recommendation']}")
            if per_category:
                cat_colors = {"GEN": "#f59e0b", "UNFAIR": "#ef4444", "STEREO": "#ec4899"}
                for cat in CATEGORIES:
                    curve = cumulative_impact_cat[cat]
                    if not curve:
                        continue
                    ks_c = sorted(curve)
                    ys_c = [float(np.mean(curve[k])) for k in ks_c]
                    el = summary["per_category"][cat]["elbow_recommendation"]
                    label_extra = f" (elbow K={el})" if el is not None else ""
                    ax.plot(ks_c, ys_c, "-s", color=cat_colors[cat], alpha=0.85,
                            linewidth=1.5, markersize=4,
                            label=f"{cat}{label_extra}")
            ax.set_xlabel("K (heads ablated, ranked by BAR or BAR_category)")
            ax.set_ylabel("Cumulative |representation impact|")
            title_suffix = "per-category" if per_category else "combined"
            ax.set_title(f"Top-K head ablation impact curve ({model_name}, {title_suffix})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            plot_path = TOPK_DIR / (
                f"{short}_per_category_curve.png" if per_category
                else f"{short}_curve.png"
            )
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
# Analysis 4 - Faithfulness validation: representation_impact bands
# ──────────────────────────────────────────────────────────────────────

def run_faithfulness_calibration(
    n_sentences: int = 500,
    k_max: int = 20,
    model_name: str = "bert-base-uncased",
    max_tokens: Optional[int] = None,
    resume: bool = True,
    checkpoint_every: int = 10,
    per_category: bool = False,
    audited_labels: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calibrate the dashboard's ``representation_impact`` cut-off AND
    test whether the BERT-vs-GPT-2 magnitude gap is a normalisation
    artefact by also collecting KL divergence on the LM-head logits.

    When ``per_category=True`` we additionally compute observed pools
    per GUS-Net category (GEN, UNFAIR, STEREO) by re-ranking heads
    against the category-specific BAR. The random-null pool is also
    **filtered per category** to include only impacts from sentences that
    were eligible for that category (had at least one C-labelled token).
    This addresses the audit finding that a single null over the full
    biased corpus does not control for per-category sentence-pool and
    mask-size differences. No extra ablations are needed, only per-sample
    eligibility bookkeeping (~330 KB extra memory at v9 scale).
    Per-category mode therefore adds ~3 extra observation batches per
    sentence (~2.5x total cost rather than 4x).

    The current default in ``bias_xai.py`` is ``> 0.05`` for "high impact"
    coloring of an ablated head. That number is arbitrary. We make it
    defensible by comparing it to a null distribution built from ablating
    random heads instead of BAR-ranked heads.

    For each biased sentence in a v9 subsample, we:

    1. Compute BAR per (layer, head) and pick the top-K by BAR.
    2. Ablate each of those K heads individually, record the
       per-head ``representation_impact`` AND ``kl_divergence``
       (observed pool).
    3. Sample K random (layer, head) pairs from the full set and
       ablate each, record both metrics (null pool).

    The 95th / 99th percentile of the null pool give the
    ``representation_impact`` thresholds that catch real causal heads
    with false-positive rates of 5% and 1% respectively.

    KL divergence is included to test the hypothesis that the
    BERT-vs-GPT-2 100x gap in ``representation_impact`` is a LayerNorm
    scaling artefact. KL is scale-invariant on the logit space, so if
    the gap closes under KL the difference was metric-specific; if it
    persists the architectures genuinely differ in head-level
    contribution.

    Saves ``faithfulness/{model_short}.json`` (now with both metric
    pools) and a 2-panel histogram. Checkpoint:
    ``checkpoints/faithfulness_impact_{model_short}.pkl``.
    """
    # Audited mode: restrict the corpus to the human-audited sentences and
    # swap the GUS-Net mask functions for gold-label lookups.
    audited_lookup = None
    detect_combined_fn = _detect_biased_indices_for_attention_tokens
    detect_per_cat_fn = _detect_per_category_indices
    if audited_labels is not None:
        audited_lookup = load_audited_labels(audited_labels)
        detect_combined_fn, detect_per_cat_fn = _make_audited_mask_fns(audited_lookup)
        sample = [rec["entry"] for rec in audited_lookup.values()]
        print(f"AUDITED mode: {len(sample)} gold-labelled sentences "
              f"from {audited_labels}", flush=True)
    else:
        sample = load_v9_stratified(n_sentences, max_tokens=max_tokens)
    biased_only = [e for e in sample if e.get("has_bias")]
    print(f"Faithfulness calibration: {len(biased_only)} biased sentences, "
          f"K={k_max}, model={model_name}"
          f"{' [audited labels]' if audited_labels else ''}")
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

    short = _model_short(model_name)
    _aud_tag = "_audited" if audited_labels else ""
    checkpoint_path = CHECKPOINT_DIR / (
        f"faithfulness_per_category_{short}{_aud_tag}.pkl" if per_category
        else f"faithfulness_impact_{short}{_aud_tag}.pkl"
    )
    sig = _args_signature(
        # v4 bumps the signature because per-category mode now also tracks
        # null-pool sentence eligibility per category, so the saved state
        # is structurally different from v3.
        "faithfulness_v4_per_cat" if per_category else "faithfulness_v2",
        model_name, k_max, max_tokens, per_category,
        str(audited_labels) if audited_labels else None,
    )

    obs_impacts = array.array("f")    # BAR-ranked heads' impacts
    null_impacts = array.array("f")   # random heads' impacts
    obs_kls = array.array("f")        # BAR-ranked heads' KL divergence
    null_kls = array.array("f")       # random heads' KL divergence
    # Per-category observed pools. The null pool is now FILTERED per category
    # (not shared) by tracking, for each null sample, whether the source
    # sentence was eligible for category C (has at least one C-labelled token).
    # This addresses the "shared null pool" weakness flagged in the audit:
    # category C's null is now the impact distribution of random heads on
    # sentences eligible for C, not on the entire biased corpus.
    obs_impacts_cat: Dict[str, "array.array"] = (
        {c: array.array("f") for c in CATEGORIES} if per_category else None
    )
    obs_kls_cat: Dict[str, "array.array"] = (
        {c: array.array("f") for c in CATEGORIES} if per_category else None
    )
    # Per-sample eligibility flags for the null pool. Parallel to null_impacts
    # and null_kls respectively (impacts and KLs may have different lengths
    # since either can be None per ablation, so they need separate flags).
    null_imp_elig_cat: Dict[str, "array.array"] = (
        {c: array.array("B") for c in CATEGORIES} if per_category else None
    )
    null_kl_elig_cat: Dict[str, "array.array"] = (
        {c: array.array("B") for c in CATEGORIES} if per_category else None
    )
    processed_keys: set = set()
    n_ok = 0
    rng = np.random.default_rng(42)

    def _as_arr(maybe_list) -> "array.array":
        a = array.array("f")
        if maybe_list:
            a.extend(maybe_list)
        return a

    def _as_arr_b(maybe_list) -> "array.array":
        a = array.array("B")
        if maybe_list:
            a.extend(maybe_list)
        return a

    if resume:
        state = _load_checkpoint(checkpoint_path)
        if state is not None and state.get("signature") == sig:
            obs_impacts = _as_arr(state["obs_impacts"])
            null_impacts = _as_arr(state["null_impacts"])
            obs_kls = _as_arr(state.get("obs_kls", []))
            null_kls = _as_arr(state.get("null_kls", []))
            if per_category and "obs_impacts_cat" in state:
                obs_impacts_cat = {
                    c: _as_arr(state["obs_impacts_cat"].get(c, []))
                    for c in CATEGORIES
                }
                obs_kls_cat = {
                    c: _as_arr(state.get("obs_kls_cat", {}).get(c, []))
                    for c in CATEGORIES
                }
                null_imp_elig_cat = {
                    c: _as_arr_b(state.get("null_imp_elig_cat", {}).get(c, []))
                    for c in CATEGORIES
                }
                null_kl_elig_cat = {
                    c: _as_arr_b(state.get("null_kl_elig_cat", {}).get(c, []))
                    for c in CATEGORIES
                }
            processed_keys = state["processed_keys"]
            n_ok = state.get("n_ok", 0)
            print(f"Resuming from checkpoint: {len(processed_keys)} done.",
                  flush=True)
        elif state is not None:
            print("Checkpoint exists but parameters differ. Starting fresh.",
                  flush=True)

    def _snapshot() -> Dict[str, Any]:
        snap = {
            "signature": sig,
            "obs_impacts": list(obs_impacts),
            "null_impacts": list(null_impacts),
            "obs_kls": list(obs_kls),
            "null_kls": list(null_kls),
            "processed_keys": processed_keys,
            "n_ok": n_ok,
        }
        if per_category:
            snap["obs_impacts_cat"] = {
                c: list(obs_impacts_cat[c]) for c in CATEGORIES
            }
            snap["obs_kls_cat"] = {
                c: list(obs_kls_cat[c]) for c in CATEGORIES
            }
            snap["null_imp_elig_cat"] = {
                c: list(null_imp_elig_cat[c]) for c in CATEGORIES
            }
            snap["null_kl_elig_cat"] = {
                c: list(null_kl_elig_cat[c]) for c in CATEGORIES
            }
        return snap

    start = time.time()
    interrupted = False
    with _GracefulInterrupt() as gi:
        since_last_save = 0
        for i, entry in enumerate(biased_only):
            if gi.interrupted:
                interrupted = True
                break
            if i % 5 == 0:
                print(f"  [{i+1}/{len(biased_only)}] elapsed "
                      f"{time.time() - start:.0f}s, obs={len(obs_impacts)}, "
                      f"null={len(null_impacts)}", flush=True)
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
                biased_mask = detect_combined_fn(
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
                top_obs = ranked[:k_max]

                # Random heads: sample K uniformly from all (layer, head) pairs.
                # Re-use HeadBiasMetrics objects from the same metrics list so
                # batch_ablate_top_heads receives the type it expects.
                if len(metrics) <= k_max:
                    top_null = list(metrics)
                else:
                    rand_idx = rng.choice(len(metrics), size=k_max, replace=False)
                    top_null = [metrics[int(j)] for j in rand_idx]

                tokenizer, encoder_model, lm_head_model = ModelManager.get_model(model_name)

                obs_abl = batch_ablate_top_heads(
                    encoder_model, lm_head_model, tokenizer, text,
                    top_obs, is_gpt2,
                )
                null_abl = batch_ablate_top_heads(
                    encoder_model, lm_head_model, tokenizer, text,
                    top_null, is_gpt2,
                )
                if not obs_abl or not null_abl:
                    processed_keys.add(key)
                    continue
                for r in obs_abl:
                    if r.representation_impact is not None:
                        obs_impacts.append(abs(r.representation_impact))
                    if r.kl_divergence is not None:
                        obs_kls.append(abs(r.kl_divergence))

                # Per-category mode: determine sentence eligibility for each
                # category BEFORE processing null ablations, because we need
                # to tag each null sample with whether the source sentence
                # was eligible for category C. The eligibility array runs in
                # parallel to null_impacts / null_kls.
                cat_masks_sentence: Optional[Dict[str, np.ndarray]] = None
                if per_category:
                    cat_masks_sentence = detect_per_cat_fn(
                        text, tokens, detector
                    )
                sent_elig: Dict[str, int] = {c: 0 for c in CATEGORIES}
                if per_category and cat_masks_sentence is not None:
                    seq_len = len(tokens)
                    for cat in CATEGORIES:
                        m = cat_masks_sentence[cat][:seq_len]
                        n_c = int(m.sum())
                        sent_elig[cat] = 1 if 0 < n_c < seq_len else 0

                for r in null_abl:
                    if r.representation_impact is not None:
                        null_impacts.append(abs(r.representation_impact))
                        if per_category:
                            for cat in CATEGORIES:
                                null_imp_elig_cat[cat].append(sent_elig[cat])
                    if r.kl_divergence is not None:
                        null_kls.append(abs(r.kl_divergence))
                        if per_category:
                            for cat in CATEGORIES:
                                null_kl_elig_cat[cat].append(sent_elig[cat])

                # Per-category obs: re-rank by BAR_C and ablate the top-K
                # heads per category. We reuse the cat_masks computed above
                # rather than recompute them.
                if per_category:
                    cat_masks = cat_masks_sentence
                    if cat_masks is not None:
                        for cat in CATEGORIES:
                            mask_c = cat_masks[cat]
                            n_c = int(mask_c.sum())
                            if n_c == 0 or n_c >= len(tokens):
                                continue
                            biased_idx_c = set(
                                int(i_) for i_, b in enumerate(mask_c) if b
                            )
                            metrics_c = analyzer.analyze_attention_to_bias(
                                list(attentions), biased_idx_c, tokens
                            )
                            if not metrics_c:
                                continue
                            ranked_c = sorted(
                                metrics_c,
                                key=lambda m: m.bias_attention_ratio,
                                reverse=True,
                            )
                            top_c = ranked_c[:k_max]
                            obs_abl_c = batch_ablate_top_heads(
                                encoder_model, lm_head_model, tokenizer,
                                text, top_c, is_gpt2,
                            )
                            if not obs_abl_c:
                                continue
                            for r in obs_abl_c:
                                if r.representation_impact is not None:
                                    obs_impacts_cat[cat].append(
                                        abs(r.representation_impact)
                                    )
                                if r.kl_divergence is not None:
                                    obs_kls_cat[cat].append(
                                        abs(r.kl_divergence)
                                    )

                n_ok += 1
                processed_keys.add(key)
            except Exception as e:
                processed_keys.add(key)
                _logger.debug("faithfulness sentence %d failed: %s", i, e)
                continue

            since_last_save += 1
            if since_last_save >= checkpoint_every:
                _save_checkpoint(checkpoint_path, _snapshot())
                since_last_save = 0
        _save_checkpoint(checkpoint_path, _snapshot())

    if interrupted:
        print(f"\nInterrupted. Checkpoint saved at {checkpoint_path}. "
              f"Re-run to resume.", flush=True)
        return {"interrupted": True, "checkpoint": str(checkpoint_path),
                "processed_so_far": len(processed_keys), "n_ok": n_ok}

    def _arr(a: "array.array") -> np.ndarray:
        return np.frombuffer(a, dtype=np.float32) if len(a) else np.empty(0, dtype=np.float32)

    def _pool_stats(arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {"n": 0, "mean": None, "median": None,
                    "p80": None, "p95": None, "p99": None}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.percentile(arr, 50)),
            "p80": float(np.percentile(arr, 80)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    obs_np = _arr(obs_impacts)
    null_np = _arr(null_impacts)
    obs_kl_np = _arr(obs_kls)
    null_kl_np = _arr(null_kls)
    obs_np_cat = (
        {c: _arr(obs_impacts_cat[c]) for c in CATEGORIES}
        if per_category else None
    )
    obs_kl_np_cat = (
        {c: _arr(obs_kls_cat[c]) for c in CATEGORIES}
        if per_category else None
    )

    summary: Dict[str, Any] = {
        "n_sentences_attempted": len(biased_only),
        "n_sentences_ok": n_ok,
        "model": model_name,
        "k_max": k_max,
        "current_default_high_impact": 0.05,
        # ── representation_impact (norm of hidden-state delta) ────────
        "obs_pool": _pool_stats(obs_np),
        "null_pool": _pool_stats(null_np),
        "recommended_thresholds": {
            "above_noise_alpha_0.20": float(np.percentile(null_np, 80)) if null_np.size else None,
            "high_impact_alpha_0.05": float(np.percentile(null_np, 95)) if null_np.size else None,
            "very_high_impact_alpha_0.01": float(np.percentile(null_np, 99)) if null_np.size else None,
        },
        "fraction_obs_above_alpha_0.05": (
            float((obs_np >= np.percentile(null_np, 95)).mean())
            if obs_np.size and null_np.size else None
        ),
        # ── KL divergence on LM-head logits (alternative metric) ──────
        # Tests whether the BERT/GPT-2 ~100x gap in representation_impact
        # is a LayerNorm-scale artefact (KL is scale-invariant: a unit
        # change in logit space gives the same KL regardless of the
        # absolute hidden-state norm).
        "obs_kl_pool": _pool_stats(obs_kl_np),
        "null_kl_pool": _pool_stats(null_kl_np),
        "recommended_kl_thresholds": {
            "high_kl_alpha_0.05": float(np.percentile(null_kl_np, 95)) if null_kl_np.size else None,
            "very_high_kl_alpha_0.01": float(np.percentile(null_kl_np, 99)) if null_kl_np.size else None,
        },
        "fraction_obs_kl_above_alpha_0.05": (
            float((obs_kl_np >= np.percentile(null_kl_np, 95)).mean())
            if obs_kl_np.size and null_kl_np.size else None
        ),
    }

    if per_category:
        # Category-specific nulls: filter the random-ablation pool to only
        # those impacts whose source sentence was eligible for category C
        # (had at least one C-labelled token). This addresses the audit
        # finding that a shared null over the full biased corpus does not
        # control for per-category sentence-pool and mask-size differences.
        per_cat_summary: Dict[str, Any] = {}
        for cat in CATEGORIES:
            ocp = obs_np_cat[cat]
            okp = obs_kl_np_cat[cat]
            # Build the category-specific null arrays via boolean mask.
            imp_elig = np.frombuffer(null_imp_elig_cat[cat], dtype=np.uint8).astype(bool) if len(null_imp_elig_cat[cat]) else np.empty(0, dtype=bool)
            kl_elig = np.frombuffer(null_kl_elig_cat[cat], dtype=np.uint8).astype(bool) if len(null_kl_elig_cat[cat]) else np.empty(0, dtype=bool)
            # Defensive sizing: arrays are appended in lock-step with
            # null_np / null_kl_np, but defend against any size mismatch.
            imp_mask = imp_elig[: null_np.size] if imp_elig.size else np.zeros(null_np.size, dtype=bool)
            kl_mask = kl_elig[: null_kl_np.size] if kl_elig.size else np.zeros(null_kl_np.size, dtype=bool)
            null_cat_np = null_np[imp_mask] if imp_mask.size and imp_mask.any() else np.empty(0, dtype=np.float32)
            null_kl_cat_np = null_kl_np[kl_mask] if kl_mask.size and kl_mask.any() else np.empty(0, dtype=np.float32)
            per_cat_summary[cat] = {
                "obs_pool": _pool_stats(ocp),
                "null_pool": _pool_stats(null_cat_np),
                "recommended_thresholds": {
                    "high_impact_alpha_0.05": (
                        float(np.percentile(null_cat_np, 95))
                        if null_cat_np.size else None
                    ),
                    "very_high_impact_alpha_0.01": (
                        float(np.percentile(null_cat_np, 99))
                        if null_cat_np.size else None
                    ),
                },
                "fraction_obs_above_alpha_0.05": (
                    float((ocp >= np.percentile(null_cat_np, 95)).mean())
                    if ocp.size and null_cat_np.size else None
                ),
                "obs_kl_pool": _pool_stats(okp),
                "null_kl_pool": _pool_stats(null_kl_cat_np),
                "recommended_kl_thresholds": {
                    "high_kl_alpha_0.05": (
                        float(np.percentile(null_kl_cat_np, 95))
                        if null_kl_cat_np.size else None
                    ),
                    "very_high_kl_alpha_0.01": (
                        float(np.percentile(null_kl_cat_np, 99))
                        if null_kl_cat_np.size else None
                    ),
                },
                "fraction_obs_kl_above_alpha_0.05": (
                    float((okp >= np.percentile(null_kl_cat_np, 95)).mean())
                    if okp.size and null_kl_cat_np.size else None
                ),
            }
        summary["per_category"] = per_cat_summary

    out_path = FAITHFULNESS_DIR / (
        f"{short}_per_category{_aud_tag}.json" if per_category
        else f"{short}{_aud_tag}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Saved {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _hist_panel(ax, obs_arr: np.ndarray, null_arr: np.ndarray,
                        metric_label: str, current_default: Optional[float]):
            if obs_arr.size == 0 or null_arr.size == 0:
                return
            xmax = float(np.percentile(np.concatenate([obs_arr, null_arr]), 99.5))
            if xmax <= 0:
                xmax = max(float(obs_arr.max()), float(null_arr.max()), 1e-6)
            ax.hist(null_arr, bins=80, range=(0, xmax), alpha=0.55,
                    color="#94a3b8", density=True,
                    label=f"Null (random heads, n={null_arr.size})")
            ax.hist(obs_arr, bins=80, range=(0, xmax), alpha=0.55,
                    color="#ff5ca9", density=True,
                    label=f"Observed (top-{k_max} BAR heads, n={obs_arr.size})")
            p95 = float(np.percentile(null_arr, 95))
            p99 = float(np.percentile(null_arr, 99))
            ax.axvline(p95, color="#16a34a", linestyle="--",
                       label=f"95th null = {p95:.4g} (α=0.05)")
            ax.axvline(p99, color="#dc2626", linestyle="--",
                       label=f"99th null = {p99:.4g} (α=0.01)")
            if current_default is not None:
                ax.axvline(current_default, color="#1d4ed8", linestyle=":",
                           label=f"Current default = {current_default}")
            ax.set_xlabel(metric_label)
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        has_impact = obs_np.size and null_np.size
        has_kl = obs_kl_np.size and null_kl_np.size

        if has_impact and has_kl:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            _hist_panel(axes[0], obs_np, null_np,
                        "|representation_impact|", 0.05)
            axes[0].set_title("representation_impact (cosine-distance based)")
            _hist_panel(axes[1], obs_kl_np, null_kl_np,
                        "|KL divergence|", None)
            axes[1].set_title("KL divergence (LM-head logits, scale-invariant)")
            fig.suptitle(
                f"Per-head ablation: BAR-ranked vs random null  -  {model_name}",
                fontsize=11, y=1.02,
            )
        elif has_impact:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _hist_panel(ax, obs_np, null_np, "|representation_impact|", 0.05)
            ax.set_title(f"Per-head ablation impact ({model_name}): BAR-ranked vs random null")
        else:
            fig = None

        if fig is not None:
            fig.tight_layout()
            plot_path = FAITHFULNESS_DIR / (
                f"{short}_per_category{_aud_tag}.png" if per_category
                else f"{short}{_aud_tag}.png"
            )
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

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

    for _short, _model in [("bert", "bert-base-uncased"), ("gpt2", "gpt2")]:
        abl_ckpt = CHECKPOINT_DIR / f"topk_ablation_{_short}.pkl"
        if abl_ckpt.is_file():
            print(f"Found ablation checkpoint at {abl_ckpt}.")
            state = _load_checkpoint(abl_ckpt)
            if state is not None:
                try:
                    run_topk_ablation(
                        n_sentences=0, k_max=20,
                        model_name=_model,
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
    parser.add_argument("--gusnet-token", action="store_true",
                        help="Re-derive GUS-Net per-category thresholds at the "
                             "TOKEN level against the human-audited gold labels "
                             "(uses --audited-labels path or its default). "
                             "Output: gusnet/token_thresholds_audited.json.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run top-K head ablation cumulative-impact analysis.")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Calibrate the representation_impact threshold for "
                             "faithfulness validation by comparing BAR-ranked "
                             "vs random head ablations.")
    parser.add_argument("--all", action="store_true",
                        help="Run all four analyses.")
    parser.add_argument("--n", type=_parse_n_arg, default=200,
                        help="Sentences for BAR/BSR (int or 'all'; default 200).")
    parser.add_argument("--perms", type=int, default=200,
                        help="Permutations per (sentence, head) (default 200).")
    parser.add_argument("--bar-per-category", action="store_true",
                        help="Also compute BAR/BSR distributions per GUS-Net "
                             "category (GEN, UNFAIR, STEREO) with their own "
                             "permutation nulls. Output goes to "
                             "bar_bsr/per_category.json.")
    parser.add_argument("--abl-n", type=_parse_n_arg, default=50,
                        help="Sentences for top-K ablation (int or 'all'; default 50).")
    parser.add_argument("--k-max", type=int, default=20,
                        help="Max K for ablation curve (default 20).")
    parser.add_argument("--abl-model", choices=["bert", "gpt2", "both"],
                        default="both",
                        help="Which attention model(s) to ablate (default both). "
                             "Each model writes its own checkpoint + output JSON.")
    parser.add_argument("--abl-per-category", action="store_true",
                        help="Also compute per-category Top-K elbow curves "
                             "(GEN, UNFAIR, STEREO) using category-specific BAR "
                             "rankings. Output goes to "
                             "topk_ablation/{model}_per_category.json. "
                             "Approximately 3x slower because each sentence runs "
                             "3 extra ablation batches.")
    parser.add_argument("--faith-n", type=_parse_n_arg, default=500,
                        help="Sentences for faithfulness calibration "
                             "(int or 'all'; default 500). Lower than --abl-n "
                             "because each sentence does 2x the ablations.")
    parser.add_argument("--faith-model", choices=["bert", "gpt2", "both"],
                        default="both",
                        help="Which attention model(s) for faithfulness calibration "
                             "(default both).")
    parser.add_argument("--faith-per-category", action="store_true",
                        help="Also collect per-category (GEN, UNFAIR, STEREO) "
                             "observed pools, sharing the random-null pool with "
                             "the combined run. Output goes to "
                             "faithfulness/{model}_per_category.json. ~2.5x slower.")
    parser.add_argument("--audited-labels", nargs="?", type=Path,
                        const=AUDITED_LABELS_DEFAULT, default=None,
                        metavar="PATH",
                        help="Use the human-audited token labels instead of "
                             "GUS-Net live predictions for the biased masks, "
                             "restricting the corpus to the audited sentences. "
                             "Applies to --bar and --faithfulness. Optional PATH "
                             f"defaults to {AUDITED_LABELS_DEFAULT.name}. "
                             "Outputs get an _audited suffix.")
    parser.add_argument("--encoders", choices=["base", "gusnet"], default="base",
                        help="Which encoder family to calibrate: 'base' = the "
                             "pretrained bert-base-uncased / gpt2 (historical "
                             "default), 'gusnet' = the fine-tuned GUS-Net trunks "
                             "(pinthoz/gus-net-*). The dashboard's default "
                             "attention source is GUS-Net, so thresholds quoted "
                             "for that view should come from --encoders gusnet; "
                             "outputs/checkpoints are tagged so the two families "
                             "never overwrite each other.")
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
        args.bar = args.gusnet = args.ablation = args.faithfulness = True
    if not (args.bar or args.gusnet or args.gusnet_token or args.ablation
            or args.faithfulness):
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

    # Same warning for ablation. Each biased sentence does ~(K_max+2) forward
    # passes (baseline + K ablations + GUS-Net), so it is ~10x more expensive
    # per sentence than BAR/BSR - adjust seconds_per_unit accordingly.
    if args.ablation:
        n_models_abl = 2 if args.abl_model == "both" else 1
        # Empirical: ~23 s / (sentence, model) at K_max=20 on CPU.
        sec_per_unit = max(1.0, 1.15 * (args.k_max + 3))
        if args.abl_n == 0:
            # v9 is ~50% biased (~5152 biased after filter).
            est = _estimate_runtime(5152, n_models=n_models_abl,
                                    seconds_per_unit=sec_per_unit)
            print(f"NOTE: --ablation on the full v9 corpus "
                  f"(~5,152 biased sentences x {n_models_abl} model(s), K_max={args.k_max}) "
                  f"will take {est} on CPU. Checkpoints save every "
                  f"{max(1, args.checkpoint_every // 2)} sentences; safe to "
                  f"Ctrl-C and re-run.")
        elif args.abl_n >= 500:
            est = _estimate_runtime(args.abl_n // 2, n_models=n_models_abl,
                                    seconds_per_unit=sec_per_unit)
            print(f"NOTE: --ablation with --abl-n {args.abl_n} "
                  f"(~{args.abl_n // 2} biased after filter) "
                  f"will take {est} on CPU.")

    # Ensure UTF-8 output on Windows consoles that default to cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    resume = not args.no_resume
    # Encoder-family map: which HF model each short name resolves to.
    _ENC = {
        "base":   {"bert": "bert-base-uncased",      "gpt2": "gpt2"},
        "gusnet": {"bert": "pinthoz/gus-net-bert",   "gpt2": "pinthoz/gus-net-gpt2"},
    }[args.encoders]
    if args.encoders == "gusnet":
        print("Encoder family: GUS-Net fine-tuned trunks "
              f"({_ENC['bert']}, {_ENC['gpt2']}) - outputs tagged '_gusnet'.")
    if args.bar:
        print("\n========== BAR/BSR analysis ==========")
        try:
            run_bar_bsr_analysis(
                n_sentences=args.n,
                n_permutations=args.perms,
                models=[_ENC["bert"], _ENC["gpt2"]],
                max_tokens=args.max_tokens,
                resume=resume,
                checkpoint_every=args.checkpoint_every,
                per_category=args.bar_per_category,
                audited_labels=args.audited_labels,
            )
        except Exception:
            traceback.print_exc()
    if args.gusnet:
        print("\n========== GUS-Net calibration ==========")
        try:
            run_gusnet_calibration()
        except Exception:
            traceback.print_exc()
    if args.gusnet_token:
        print("\n========== GUS-Net token-level calibration (audited) ==========")
        try:
            run_gusnet_token_calibration(audited_labels=args.audited_labels)
        except Exception:
            traceback.print_exc()
    if args.ablation:
        ablation_models: List[str] = []
        if args.abl_model in ("bert", "both"):
            ablation_models.append(_ENC["bert"])
        if args.abl_model in ("gpt2", "both"):
            ablation_models.append(_ENC["gpt2"])
        for _m in ablation_models:
            print(f"\n========== Top-K ablation curve ({_m}) ==========")
            try:
                run_topk_ablation(
                    n_sentences=args.abl_n,
                    k_max=args.k_max,
                    model_name=_m,
                    max_tokens=args.max_tokens,
                    resume=resume,
                    checkpoint_every=max(1, args.checkpoint_every // 2),
                    per_category=args.abl_per_category,
                )
            except Exception:
                traceback.print_exc()
    if args.faithfulness:
        faith_models: List[str] = []
        if args.faith_model in ("bert", "both"):
            faith_models.append(_ENC["bert"])
        if args.faith_model in ("gpt2", "both"):
            faith_models.append(_ENC["gpt2"])
        for _m in faith_models:
            print(f"\n========== Faithfulness calibration ({_m}) ==========")
            try:
                run_faithfulness_calibration(
                    n_sentences=args.faith_n,
                    k_max=args.k_max,
                    model_name=_m,
                    max_tokens=args.max_tokens,
                    resume=resume,
                    checkpoint_every=max(1, args.checkpoint_every // 2),
                    per_category=args.faith_per_category,
                    audited_labels=args.audited_labels,
                )
            except Exception:
                traceback.print_exc()

    print(f"\nResults in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
