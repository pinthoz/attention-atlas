"""GUS-Net Neural Bias Detector.

Supports two backbone architectures for token-level social bias detection:

  1. BERT  (pinthoz/gus-net-bert)   - 7-label BIO scheme
  2. GPT-2 (pinthoz/gus-net-gpt2)  - 7-label BIO scheme

Both models share the same label layout:

    Index  Label
    ─────  ─────────
      0    O
      1    B-STEREO
      2    I-STEREO
      3    B-GEN
      4    I-GEN
      5    B-UNFAIR
      6    I-UNFAIR

Both output per-token sigmoid probabilities that are thresholded to produce
categorical bias labels (GEN, UNFAIR, STEREO).
"""

import torch
import logging
import threading
import numpy as np
from pathlib import Path
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    AutoTokenizer,
    GPT2ForTokenClassification,
)
from typing import List, Dict

_logger = logging.getLogger(__name__)


# NOTE on GPT-2 masking: the GUS-Net GPT-2 models are fine-tuned with the
# standard CAUSAL mask (see attention_app/bias/training/ - no mask removal
# anywhere), so inference below intentionally keeps the causal mask too.
# A `_make_gpt2_bidirectional` helper used to live here with a docstring
# claiming it "must be called" after loading; it was never called and its
# premise (bidirectional training) does not match the training scripts, so
# it was removed to avoid suggesting the inference path is misconfigured.


# ── Punctuation filter ───────────────────────────────────────────────────────
# Punctuation tokens are never bias carriers - suppress any model predictions on them.
_PUNCTUATION_TOKENS: frozenset = frozenset({
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "-", "–", "-", "'", '"', "`", "``", "''", "'s", "/", "\\",
})


# ── Shared label scheme (7 labels - identical for BERT and GPT-2) ────────────

NUM_LABELS = 7
CATEGORY_INDICES = {
    "STEREO": [1, 2],   # B-STEREO, I-STEREO
    "GEN":    [3, 4],   # B-GEN,    I-GEN
    "UNFAIR": [5, 6],   # B-UNFAIR, I-UNFAIR
}
O_INDEX = 0


# ── GPT-2 tokenizer compatibility shim ───────────────────────────────────────

# Cache of downgraded tokenizer dirs, keyed by source model path, so we only
# rewrite the JSON once per process.
_DOWNGRADED_TOK_DIRS: Dict[str, str] = {}


def _is_tokenizer_format_error(exc: Exception) -> bool:
    """True when a tokenizer.json was written by a newer `tokenizers` than the
    one installed, so the Rust deserializer rejects it."""
    msg = str(exc)
    return "did not match any variant" in msg and "ModelWrapper" in msg


def _load_gpt2_tokenizer_compat(model_path: str, **kwargs):
    """Load a GPT-2 fast tokenizer, downgrading the tokenizer.json format if the
    installed `tokenizers` is too old to parse it.

    Newer `tokenizers` (>= 0.20) serialise BPE merges as `[a, b]` pairs and add
    an `ignore_merges` field (>= 0.15). Older runtimes (e.g. 0.13.x, pinned by
    transformers 4.28) only understand space-joined `"a b"` merges and reject
    the extra field. The vocab/merges are semantically identical, so we rewrite
    a compatible copy into a temp dir and load from there. The original model
    artefact is left untouched."""
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except Exception as exc:
        if not _is_tokenizer_format_error(exc):
            raise

        import os
        import json
        import shutil
        import tempfile

        cached = _DOWNGRADED_TOK_DIRS.get(model_path)
        if cached is None:
            if not os.path.isdir(model_path):
                # Repo id / cached snapshot: resolve the real files on disk.
                from transformers.utils import cached_file
                src_json = cached_file(model_path, "tokenizer.json", local_files_only=True)
                src_cfg = cached_file(
                    model_path, "tokenizer_config.json",
                    local_files_only=True, _raise_exceptions_for_missing_entries=False,
                )
                src_dir = os.path.dirname(src_json)
            else:
                src_dir = model_path
                src_json = os.path.join(src_dir, "tokenizer.json")

            with open(src_json, encoding="utf-8") as f:
                tok = json.load(f)
            model = tok.get("model", {})
            # `ignore_merges` (tokenizers >= 0.15) is unknown to older runtimes.
            model.pop("ignore_merges", None)
            merges = model.get("merges")
            if merges and isinstance(merges[0], list):
                # `[a, b]` pairs -> "a b" strings.
                model["merges"] = [" ".join(pair) for pair in merges]

            tmp_dir = tempfile.mkdtemp(prefix="gusnet_tok_")
            with open(os.path.join(tmp_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tok, f, ensure_ascii=False)
            # Carry the config so pad/bos/eos special tokens are preserved.
            for fname in ("tokenizer_config.json", "special_tokens_map.json"):
                src_extra = os.path.join(src_dir, fname)
                if os.path.isfile(src_extra):
                    shutil.copy(src_extra, os.path.join(tmp_dir, fname))

            _DOWNGRADED_TOK_DIRS[model_path] = tmp_dir
            cached = tmp_dir
            _logger.warning(
                "[GUS-Net] Rewrote %s tokenizer.json to a format compatible "
                "with the installed tokenizers; loading from %s.",
                model_path, tmp_dir,
            )

        return AutoTokenizer.from_pretrained(cached, **kwargs)


# ── Model registry ───────────────────────────────────────────────────────────

# Version: 2026-02-09-v2 - Models now loaded from HuggingFace Hub
_BIAS_DIR = Path(__file__).parent / "models"

MODEL_REGISTRY = {
    "gusnet-bert": {
        "path": "pinthoz/gus-net-bert",
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT)",
        "public": True,
        # The checkpoint published as pinthoz/gus-net-bert IS the
        # sparsity-regularised training run; the registry key and display name
        # intentionally omit the training detail. Re-trained 2026-07-16 on the
        # CLEANED corpus (see gus-net-bert-sparse-clean below) - the previous
        # run's thresholds were [0.476, 0.375, 0.3475, 0.3251, 0.3975, 0.342,
        # 0.3309]. These are the per-label F1-optimised values produced by that
        # run (optimized_thresholds.npy). If the published checkpoint is ever
        # retrained, refresh these together.
        "optimized_thresholds": [0.4265, 0.4071, 0.3938, 0.3462, 0.3669, 0.3184, 0.3630],
    },
    "gusnet-bert-large": {
        "path": "pinthoz/gus-net-bert-large",
        "architecture": "bert",
        "tokenizer": "bert-large-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT Large)",
        "public": True,
        "optimized_thresholds": [0.4761, 0.4478, 0.4573, 0.4178, 0.4278, 0.4000, 0.3497],
    },
    "gusnet-bert-custom": {
        "path": "pinthoz/gus-net-bert-custom",
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT Custom)",
        "public": False,
        "optimized_thresholds": None,
    },
    "gusnet-gpt2": {
        "path": "pinthoz/gus-net-gpt2",
        "architecture": "gpt2",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"<|endoftext|>", "[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (GPT-2)",
        "public": True,
        # Sparsity-regularised run, re-trained 2026-07-16 on the CLEANED corpus
        # (see gus-net-gpt2-sparse-clean below). The previous published run was
        # trained on the RAW corpus and predicted bias on sentence-final
        # punctuation in 38% of sentences; this one does not.
        "optimized_thresholds": [0.4250, 0.3977, 0.3500, 0.3617, 0.3913, 0.3278, 0.4174],
    },
    "gusnet-gpt2-medium": {
        "path": "pinthoz/gus-net-gpt2-medium",
        "architecture": "gpt2",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"<|endoftext|>", "[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (GPT-2 Medium)",
        "public": True,
        "optimized_thresholds": [0.4912, 0.5042, 0.4213, 0.4204, 0.4000, 0.4618, 0.3848],
    },
    # ── Locally trained models (ASL + LLRD + BIO postprocessing) ─────────
    "gusnet-bert-new": {
        "path": str(_BIAS_DIR / "gus-net-bert-final-new"),
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT v2)",
        "public": False,
        # [O, B-STEREO, I-STEREO, B-GEN, I-GEN, B-UNFAIR, I-UNFAIR]
        # I-STEREO raised to 0.85 to prevent stereotype span bleed to function words
        # I-GEN and I-UNFAIR raised to 0.65 for the same reason
        "optimized_thresholds": [0.4992, 0.5111, 0.8500, 0.4276, 0.6500, 0.4614, 0.6500],
    },
    "gusnet-gpt2-new": {
        "path": str(_BIAS_DIR / "gus-net-gpt2-final-new"),
        "architecture": "gpt2",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"<|endoftext|>", "[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (GPT-2 v2)",
        "public": False,
        # [O, B-STEREO, I-STEREO, B-GEN, I-GEN, B-UNFAIR, I-UNFAIR]
        # I-STEREO raised to 0.85 to prevent stereotype span bleed to function words
        # I-GEN and I-UNFAIR raised to 0.65 for the same reason
        "optimized_thresholds": [0.4607, 0.4328, 0.8500, 0.4537, 0.6500, 0.3500, 0.6500],
    },
    # ── Sparse models (sparsity-regularised training) ────────────────────
    "gusnet-bert-sparse": {
        "path": str(_BIAS_DIR / "gus-net-bert-sparse"),
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT Sparse)",
        "public": False,
        "optimized_thresholds": [0.476, 0.375, 0.3475, 0.3251, 0.3975, 0.342, 0.3309],
    },
    "gusnet-gpt2-sparse": {
        "path": str(_BIAS_DIR / "gus-net-gpt2-sparse"),
        "architecture": "gpt2",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"<|endoftext|>", "[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (GPT-2 Sparse)",
        "public": False,
        "optimized_thresholds": [0.4664, 0.3656, 0.2993, 0.3696, 0.4, 0.3501, 0.3659],
    },
    # ── Paper-faithful models (standard focal loss, no LLRD/span decay) ──
    "gusnet-bert-paper": {
        "path": str(_BIAS_DIR / "gus-net-bert-paper"),
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT Paper)",
        "public": False,
        # [O, B-STEREO, I-STEREO, B-GEN, I-GEN, B-UNFAIR, I-UNFAIR]
        "optimized_thresholds": [0.5039, 0.4993, 0.541, 0.463, 0.4658, 0.4632, 0.414],
    },
    "gusnet-gpt2-paper": {
        "path": str(_BIAS_DIR / "gus-net-gpt2-paper"),
        "architecture": "gpt2",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"<|endoftext|>", "[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (GPT-2 Paper)",
        "public": False,
        # [O, B-STEREO, I-STEREO, B-GEN, I-GEN, B-UNFAIR, I-UNFAIR]
        "optimized_thresholds": [0.5, 0.4862, 0.4946, 0.5012, 0.5, 0.4748, 0.475],
    },
}


class GusNetDetector:
    """Neural bias detector supporting BERT and GPT-2 GUS-Net backbones.

    Models are cached at class level to avoid redundant reloading.  When the
    requested model differs from the cached one, only the new model is loaded
    (previous entries stay in cache until memory pressure forces eviction).
    """

    _cache: Dict[str, tuple] = {}
    # Serialises loads so concurrent sessions/executors don't load the same
    # model twice (mirrors ModelManager._lock).
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_key: str = "gusnet-bert",
        threshold: float = 0.5,
        use_optimized: bool = True,
    ):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_key = model_key
        self.config = dict(MODEL_REGISTRY[model_key])  # shallow copy
        self.threshold = threshold
        self.use_optimized = use_optimized
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-load thresholds from .npy (written by training script)
        npy_path = Path(self.config["path"]) / "optimized_thresholds.npy"
        if npy_path.exists():
            self.config["optimized_thresholds"] = np.load(str(npy_path), allow_pickle=False).tolist()

    # ── Model loading ────────────────────────────────────────────────────

    @classmethod
    def _load_model(cls, model_key: str, device: str):
        """Load and cache the model identified by *model_key*.

        If the model is already cached but on a different device, it is
        moved to the requested device so callers don't silently get a CPU
        model when they asked for CUDA (or vice versa).
        """
        with cls._cache_lock:
            if model_key in cls._cache:
                cached_tok, cached_model = cls._cache[model_key]
                # Ensure the cached model sits on the requested device.
                if str(cached_model.device) != str(device):
                    cached_model.to(device)
                return cls._cache[model_key]

            cfg = MODEL_REGISTRY[model_key]
            model_path = cfg["path"]
            arch = cfg["architecture"]

            _logger.info("[GUS-Net] Loading %s from %s ...", cfg['display_name'], model_path)

            try:
                if arch == "gpt2":
                    tokenizer = _load_gpt2_tokenizer_compat(
                        model_path, add_prefix_space=True,
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    # Silence "Using sep_token, but it is not set yet." -
                    # GPT-2 has no sep/cls/mask tokens and transformers logs
                    # this on every internal special-token access otherwise.
                    tokenizer.verbose = False
                    # attn_implementation="eager" is REQUIRED: the transformers
                    # default (SDPA) uses a fused kernel that never materialises
                    # the attention weights, so output_attentions=True returns a
                    # tuple of None on GPT-2. That breaks the Chefer-LRP
                    # cross-check (no attention gradients -> IG fallback) and any
                    # attention extraction on the decoder. Eager materialises the
                    # softmax weights in the graph, so gradients flow through them.
                    model = GPT2ForTokenClassification.from_pretrained(
                        model_path, attn_implementation="eager"
                    )
                else:
                    # BERT: prefer the tokenizer PUBLISHED WITH the
                    # fine-tuned checkpoint - if a future checkpoint ever
                    # extends the vocabulary, a generic bert-base-uncased
                    # tokenizer would produce ids misaligned with the
                    # model's embedding matrix and the predictions would be
                    # silently wrong. Fall back to the registry name only
                    # when the repo ships no tokenizer files.
                    try:
                        tokenizer = BertTokenizerFast.from_pretrained(model_path)
                    except Exception:
                        tok_name = cfg.get("tokenizer", "bert-base-uncased")
                        _logger.info(
                            "[GUS-Net] %s ships no tokenizer - falling back to %s",
                            model_path, tok_name)
                        tokenizer = BertTokenizerFast.from_pretrained(tok_name)
                    # eager attention for the same reason as GPT-2 above: BERT
                    # with SDPA falls back to eager per-call when attentions are
                    # requested (with a warning), so loading eager up front keeps
                    # the attention weights differentiable and silences it.
                    model = BertForTokenClassification.from_pretrained(
                        model_path, num_labels=cfg["num_labels"],
                        attn_implementation="eager",
                    )

                # Guard against the silent-misalignment failure mode: every
                # tokenizer id must exist in the model's embedding matrix.
                try:
                    n_emb = model.get_input_embeddings().num_embeddings
                    if len(tokenizer) > n_emb:
                        _logger.warning(
                            "[GUS-Net] Tokenizer vocab (%d) exceeds model "
                            "embeddings (%d) for %s - predictions may be "
                            "wrong for out-of-range tokens.",
                            len(tokenizer), n_emb, cfg["display_name"])
                except Exception:
                    pass

                model.eval()
                model.to(device)
                cls._cache[model_key] = (tokenizer, model)
                _logger.info("[GUS-Net] %s loaded successfully on %s.", cfg['display_name'], device)
            except Exception as e:
                _logger.error("[GUS-Net] Failed to load %s: %s", cfg['display_name'], e)
                raise

            return cls._cache[model_key]

    @property
    def is_available(self) -> bool:
        """Check whether the model can be loaded."""
        try:
            self._load_model(self.model_key, self._device)
            return True
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            return False

    # ── Inference ────────────────────────────────────────────────────────

    # ── Inference ────────────────────────────────────────────────────────

    def predict_proba(self, text: str):
        """Run model inference and return raw probabilities.

        Returns:
            tokens: List[str]
            probabilities: torch.Tensor [seq_len, num_labels]
        """
        tokenizer, model = self._load_model(self.model_key, self._device)
        
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, num_labels]

        probabilities = torch.sigmoid(logits)[0].cpu()  # [seq_len, num_labels]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        
        return tokens, probabilities

    def apply_thresholds(
        self, 
        tokens: List[str], 
        probabilities: torch.Tensor, 
        thresholds: Dict[str, float] = None
    ) -> List[Dict]:
        """Apply thresholds to raw probabilities to generate labels.
        
        Args:
            tokens: List of tokens from predict_proba
            probabilities: Tensor of shape [seq_len, num_labels]
            thresholds: Dict mapping 'GEN', 'UNFAIR', 'STEREO' to float values.
                       If None, uses self.threshold or self.use_optimized defaults.
        """
        cfg = self.config
        results: List[Dict] = []
        
        # Resolve thresholds
        if thresholds is None:
            # Fallback to single scalar or optimized
            active_thresholds = {
                "GEN": self.threshold, 
                "UNFAIR": self.threshold, 
                "STEREO": self.threshold
            }
            use_opt = self.use_optimized
        else:
            active_thresholds = thresholds
            use_opt = False # If explicit thresholds provided, override optimized
            
        for idx, (token, probs) in enumerate(zip(tokens, probabilities)):
            # Skip special tokens
            if token in cfg["special_tokens"]:
                results.append(self._empty_label(token, idx))
                continue

            # Per-category score = max(B-prob, I-prob)
            scores: Dict[str, float] = {}
            for cat, indices in cfg["category_indices"].items():
                scores[cat] = float(max(probs[i].item() for i in indices))

            # O score: direct from model output
            scores["O"] = float(probs[cfg["o_index"]].item())

            # Threshold to determine active bias types
            bias_types = []
            
            opt = cfg.get("optimized_thresholds")
            
            # ``fired`` records, per detected category, the EXACT probability
            # and threshold pair that triggered the decision, so the UI can
            # show a reproducible rule instead of a max(B,I) score paired
            # with an unrelated threshold value.
            fired: Dict[str, Dict] = {}

            if use_opt and opt is not None:
                # Optimized per-label thresholds logic (per B-/I- index)
                for cat in ("GEN", "UNFAIR", "STEREO"):
                    indices = cfg["category_indices"][cat]
                    hits = [i for i in indices if probs[i].item() >= opt[i]]
                    if hits:
                        bias_types.append(cat)
                        best = max(hits, key=lambda i: probs[i].item())
                        prefix = "B-" if best == indices[0] else "I-"
                        fired[cat] = {
                            "prob": float(probs[best].item()),
                            "threshold": float(opt[best]),
                            "label": f"{prefix}{cat}",
                        }
            else:
                # Dynamic / Manual thresholds (per-category max(B, I) score)
                for cat in ("GEN", "UNFAIR", "STEREO"):
                    thresh = active_thresholds.get(cat, 0.5)
                    if scores.get(cat, 0.0) >= thresh:
                        bias_types.append(cat)
                        fired[cat] = {
                            "prob": float(scores[cat]),
                            "threshold": float(thresh),
                            "label": cat,
                        }

            # Punctuation is never a bias carrier - suppress span-bleed scores
            if bias_types and token.lstrip("\u0120") in _PUNCTUATION_TOKENS:
                bias_types = []
                fired = {}

            # Human-readable explanation: show the firing rule itself.
            _CAT_LABEL = {
                "GEN": "Generalization",
                "UNFAIR": "Unfair language",
                "STEREO": "Stereotype",
            }
            explanations = [
                f"{_CAT_LABEL[cat]} ({fired[cat]['prob']:.2f} >= "
                f"{fired[cat]['threshold']:.2f}, {fired[cat]['label']})"
                for cat in bias_types
            ]

            # ``threshold``: the threshold of the primary (highest-prob)
            # fired category \u2014 the value the decision actually used.
            if fired:
                primary = max(fired, key=lambda c: fired[c]["prob"])
                token_threshold = fired[primary]["threshold"]
            else:
                token_threshold = active_thresholds.get("GEN", 0.5)

            results.append({
                "token": token,
                "index": idx,
                "bias_types": bias_types,
                "is_biased": len(bias_types) > 0,
                "scores": scores,
                "fired": fired,
                "method": "gusnet",
                "explanation": "; ".join(explanations),
                "threshold": token_threshold,
            })

        return results

    def detect_bias(self, text: str) -> List[Dict]:
        """Run GUS-Net inference on *text*.

        Returns a list of per-token dicts with keys:
            token, index, bias_types, is_biased, scores, method,
            explanation, threshold
        """
        tokens, probs = self.predict_proba(text)
        return self.apply_thresholds(tokens, probs)

    # ── Post-processing ──────────────────────────────────────────────────

    def get_bias_summary(self, token_labels: List[Dict]) -> Dict:
        """Generate summary statistics from detect_bias output."""
        special = self.config["special_tokens"]
        
        # Determine tokenizer type from tokens
        # BERT has ##, GPT-2 has Ġ (u0120)
        has_gpt2_tokens = any("\u0120" in t["token"] for t in token_labels)
        
        # Merge subwords to count "whole words"
        whole_words = []
        current_word = None
        
        for t in token_labels:
            tok = t["token"]
            if tok in special:
                if current_word:
                    whole_words.append(current_word)
                    current_word = None
                continue
            
            # Decide if we merge
            should_merge = False
            
            if has_gpt2_tokens:
                # GPT-2: Merge if NO Ġ (continuation) but NOT standalone punctuation
                clean_tok = tok.replace("\u0120", "")
                is_punct = clean_tok and not any(c.isalnum() for c in clean_tok)
                if "\u0120" not in tok and current_word and not is_punct:
                    should_merge = True
            else:
                # BERT / fallback: '##' marks a continuation of the previous
                # word. The SUMMARY counts whole words for BOTH tokenizers so
                # that total_tokens / bias_percentage / category counts are
                # word-level and comparable across BERT and GPT-2 in
                # compare-mode. (Display views such as the span list may
                # still show BERT subwords split - that is presentation,
                # not the denominator of a statistic.)
                should_merge = tok.startswith("##") and current_word is not None

            if should_merge:
                # Merge logic: if any part is biased, the whole word is biased
                current_word["is_biased"] = current_word["is_biased"] or t["is_biased"]
                # Collect all bias types
                current_word["bias_types"] = list(set(current_word["bias_types"] + t["bias_types"]))
                # Merge scores (max)
                for k, v in t["scores"].items():
                    current_word["scores"][k] = max(current_word["scores"].get(k, 0.0), v)
            else:
                if current_word:
                    whole_words.append(current_word)
                # Start new word (deep copy to avoid mutating original)
                current_word = {
                    "token": tok,
                    "is_biased": t["is_biased"],
                    "bias_types": list(t["bias_types"]),
                    "scores": t["scores"].copy()
                }
        if current_word:
            whole_words.append(current_word)

        total = len(whole_words)
        biased = [t for t in whole_words if t["is_biased"]]

        gen_count = sum(1 for t in biased if "GEN" in t["bias_types"])
        unfair_count = sum(1 for t in biased if "UNFAIR" in t["bias_types"])
        stereo_count = sum(1 for t in biased if "STEREO" in t["bias_types"])

        avg_confidence = 0.0
        if biased:
            all_scores = [
                t["scores"].get(cat, 0.0)
                for t in biased
                for cat in t["bias_types"]
            ]
            avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "total_tokens": total,
            "biased_tokens": len(biased),
            "bias_percentage": (len(biased) / total * 100) if total > 0 else 0,
            "generalization_count": gen_count,
            "unfairness_count": unfair_count,
            "stereotype_count": stereo_count,
            "avg_confidence": avg_confidence,
            "categories_found": [
                cat
                for cat, count in [
                    ("GEN", gen_count),
                    ("UNFAIR", unfair_count),
                    ("STEREO", stereo_count),
                ]
                if count > 0
            ],
        }

    def get_biased_spans(self, token_labels: List[Dict]) -> List[Dict]:
        """Group contiguous biased tokens into spans.

        Returns list of dicts with: start_idx, end_idx, tokens, bias_types,
        avg_score, method, explanation
        """
        special = self.config["special_tokens"]
        spans: List[Dict] = []
        current = None

        for label in token_labels:
            if label["token"] in special:
                if current is not None:
                    spans.append(self._finalize_span(current))
                    current = None
                continue

            if label["is_biased"]:
                # Check for explicit split required (BERT subword)
                # If we encounter ##, user wants it split from previous span.
                force_split = label["token"].startswith("##")
                
                if force_split and current is not None:
                     spans.append(self._finalize_span(current))
                     current = None

                if current is None:
                    current = {
                        "start": label["index"],
                        "end": label["index"],
                        "tokens": [label["token"]],
                        "bias_types": set(label["bias_types"]),
                        "scores": [label["scores"]],
                        "explanations": [label["explanation"]],
                    }
                else:
                    current["end"] = label["index"]
                    current["tokens"].append(label["token"])
                    current["bias_types"].update(label["bias_types"])
                    current["scores"].append(label["scores"])
                    current["explanations"].append(label["explanation"])
            else:
                if current is not None:
                    spans.append(self._finalize_span(current))
                    current = None

        if current is not None:
            spans.append(self._finalize_span(current))

        return spans

    # ── Helpers ───────────────────────────────────────────────────────────

    def _empty_label(self, token: str, index: int) -> Dict:
        """Return a zeroed label dict for special tokens."""
        return {
            "token": token,
            "index": index,
            "bias_types": [],
            "is_biased": False,
            "scores": {"O": 0.0, "GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
            "fired": {},
            "method": "gusnet",
            "explanation": "",
            "threshold": self.threshold,
        }

    @staticmethod
    def _finalize_span(current: Dict) -> Dict:
        """Compute averaged scores for a span."""
        all_cats = list(current["bias_types"])
        avg_scores: Dict[str, float] = {}
        for cat in ("GEN", "UNFAIR", "STEREO"):
            cat_scores = [s.get(cat, 0.0) for s in current["scores"]]
            avg_scores[cat] = (
                sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            )

        overall_avg = 0.0
        if all_cats:
            overall_avg = sum(avg_scores[c] for c in all_cats) / len(all_cats)

        span_text = ""
        for t in current["tokens"]:
            # For BERT, do we remove ## in the span text?
            # If user wants them separated, the span text for "##turing" should probably be "##turing".
            # The previous logic merged ##: span_text += t[2:].
            # But now spans are split. So current["tokens"] will trigger ONCE per token.
            # So "##turing" startswith ##.
            # If we do t[2:], we get "turing".
            # If we want to show ##, we should keep it.
            # Let's clean it but maybe keep ## if it's the start?
            # Actually, standard is to show "##" to indicate subword.
            if t.startswith("##"):
                 span_text += t # Keep ## for display in list
            else:
                if span_text:
                    span_text += " "
                
                # Clean GPT-2 markers
                clean_t = t.replace("\u0120", "")
                span_text += clean_t

        return {
            "start_idx": current["start"],
            "end_idx": current["end"],
            "tokens": current["tokens"],
            "text": span_text,
            "bias_types": all_cats,
            "avg_score": overall_avg,
            "method": "gusnet",
            "explanation": " | ".join(filter(None, current["explanations"])),
        }


class EnsembleGusNetDetector:
    """Ensemble bias detector combining two GUS-Net BERT models.

    Runs both models on the same input and combines their per-category
    sigmoid probabilities using configurable weights.  This allows
    leveraging each model's strengths (e.g. Model A for GEN, Model B
    for STEREO / UNFAIR).

    Both models must share the same tokenizer (bert-base-uncased) so
    tokens align 1-to-1.
    """

    # Default per-category weights: (model_a_weight, model_b_weight)
    # Model A = pinthoz/gus-net-bert  (better STEREO/UNFAIR)
    # Model B = custom gus-net-bert-final-my          (better GEN)
    DEFAULT_WEIGHTS = {
        "GEN":    (0.3, 0.7),   # favor custom model
        "UNFAIR": (0.7, 0.3),   # favor HF model
        "STEREO": (0.7, 0.3),   # favor HF model
        "O":      (0.5, 0.5),   # equal
    }

    def __init__(
        self,
        model_key_a: str = "gusnet-bert",
        model_key_b: str = "gusnet-bert-custom",
        threshold: float = 0.5,
        weights: Dict[str, tuple] = None,
    ):
        self.model_key_a = model_key_a
        self.model_key_b = model_key_b
        self.threshold = threshold
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Both models must be BERT with the same tokenizer
        cfg_a = MODEL_REGISTRY[model_key_a]
        cfg_b = MODEL_REGISTRY[model_key_b]
        self.config = cfg_a  # use model A's config for special tokens, etc.

    @property
    def is_available(self) -> bool:
        try:
            GusNetDetector._load_model(self.model_key_a, self._device)
            GusNetDetector._load_model(self.model_key_b, self._device)
            return True
        except Exception:
            _logger.debug("Suppressed exception", exc_info=True)
            return False

    def _get_raw_probs(self, model_key: str, text: str) -> torch.Tensor:
        """Run a single model and return sigmoid probabilities [seq_len, 7]."""
        tokenizer, model = GusNetDetector._load_model(model_key, self._device)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.sigmoid(logits)[0].cpu()

    def predict_proba(self, text: str):
        """Run ensemble inference and return combined probabilities."""
        probs_a = self._get_raw_probs(self.model_key_a, text)
        probs_b = self._get_raw_probs(self.model_key_b, text)

        # Get tokens from model A's tokenizer (shared)
        tokenizer_a, _ = GusNetDetector._load_model(self.model_key_a, self._device)
        inputs = tokenizer_a(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        tokens = tokenizer_a.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        
        # Combine probabilities based on weights
        # We need to return a "fused" probability tensor that apply_thresholds can use.
        # But apply_thresholds expects a single tensor where indices match CATEGORY_INDICES.
        # Since both models share the same 7-label scheme, we can just fuse the tensors.
        
        cfg = self.config
        # Weighted average of the probabilities tensor
        # But wait, weights are per-category.
        # We need to construct a new fused probability tensor.
        
        fused_probs = torch.zeros_like(probs_a)
        
        # O label
        w_o_a, w_o_b = self.weights.get("O", (0.5, 0.5))
        o_idx = cfg["o_index"]
        fused_probs[:, o_idx] = w_o_a * probs_a[:, o_idx] + w_o_b * probs_b[:, o_idx]
        
        # Categories
        for cat, cat_indices in cfg["category_indices"].items():
            w_a, w_b = self.weights.get(cat, (0.5, 0.5))
            for idx in cat_indices:
                fused_probs[:, idx] = w_a * probs_a[:, idx] + w_b * probs_b[:, idx]
                
        return tokens, fused_probs

    def apply_thresholds(self, tokens, probabilities, thresholds=None):
        """Reuse GusNetDetector's threshold logic on the fused probabilities."""
        det = GusNetDetector.__new__(GusNetDetector)
        det.config = self.config
        det.threshold = self.threshold
        det.use_optimized = False # Ensemble doesn't support optimized thresholds yet
        return det.apply_thresholds(tokens, probabilities, thresholds)

    def detect_bias(self, text: str) -> List[Dict]:
        """Run ensemble inference combining both models."""
        tokens, fused_probs = self.predict_proba(text)
        return self.apply_thresholds(tokens, fused_probs)

    def get_bias_summary(self, token_labels: List[Dict]) -> Dict:
        """Reuse GusNetDetector's summary logic."""
        det = GusNetDetector.__new__(GusNetDetector)
        det.config = self.config
        det.threshold = self.threshold
        return det.get_bias_summary(token_labels)

    def get_biased_spans(self, token_labels: List[Dict]) -> List[Dict]:
        """Reuse GusNetDetector's span logic."""
        det = GusNetDetector.__new__(GusNetDetector)
        det.config = self.config
        det.threshold = self.threshold
        return det.get_biased_spans(token_labels)


__all__ = ["GusNetDetector", "EnsembleGusNetDetector", "MODEL_REGISTRY"]
