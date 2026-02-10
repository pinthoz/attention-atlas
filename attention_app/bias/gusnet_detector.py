"""GUS-Net Neural Bias Detector.

Supports two backbone architectures for token-level social bias detection:

  1. BERT  (ethical-spectacle/social-bias-ner)  — 7-label BIO scheme
  2. GPT-2 (locally fine-tuned)                — 7-label BIO scheme

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
import numpy as np
from pathlib import Path
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    AutoTokenizer,
    GPT2ForTokenClassification,
)
from typing import List, Dict


# ── Bidirectional fix for GPT-2 inference ────────────────────────────────────

def _make_gpt2_bidirectional(model):
    """Remove the causal attention mask from a GPT-2 model.

    The ``attn.bias`` buffer is registered with ``persistent=False`` so it
    is NOT included in ``save_pretrained`` / ``state_dict``.  After loading
    a bidirectionally-trained GPT-2 model, this function must be called to
    restore full self-attention (replace lower-triangular with all-ones).

    Supports both legacy attention and newer SDPA-based implementations.
    """
    fixed = 0
    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "bias") and attn.bias is not None:
            attn.bias.fill_(1)
            fixed += 1
    if fixed:
        print(f"[GUS-Net] Bidirectional fix applied to {fixed} attention blocks (attn.bias)")
    else:
        # Newer transformers: disable causal masking via config
        print("[GUS-Net] No attn.bias found — setting is_cross_attention=True as fallback")
        model.config.is_decoder = False
        for block in model.transformer.h:
            attn = block.attn
            attn.is_cross_attention = True


# ── Shared label scheme (7 labels — identical for BERT and GPT-2) ────────────

NUM_LABELS = 7
CATEGORY_INDICES = {
    "STEREO": [1, 2],   # B-STEREO, I-STEREO
    "GEN":    [3, 4],   # B-GEN,    I-GEN
    "UNFAIR": [5, 6],   # B-UNFAIR, I-UNFAIR
}
O_INDEX = 0


# ── Model registry ───────────────────────────────────────────────────────────

# Version: 2026-02-09-v2 - Models now loaded from HuggingFace Hub
_BIAS_DIR = Path(__file__).parent

MODEL_REGISTRY = {
    "gusnet-bert": {
        "path": "ethical-spectacle/social-bias-ner",
        "architecture": "bert",
        "tokenizer": "bert-base-uncased",
        "num_labels": NUM_LABELS,
        "has_o_label": True,
        "o_index": O_INDEX,
        "category_indices": CATEGORY_INDICES,
        "special_tokens": {"[CLS]", "[SEP]", "[PAD]"},
        "display_name": "GUS-Net (BERT)",
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
    },
}


class GusNetDetector:
    """Neural bias detector supporting BERT and GPT-2 GUS-Net backbones.

    Models are cached at class level to avoid redundant reloading.  When the
    requested model differs from the cached one, only the new model is loaded
    (previous entries stay in cache until memory pressure forces eviction).
    """

    _cache: Dict[str, tuple] = {}

    def __init__(
        self,
        model_key: str = "gusnet-bert",
        threshold: float = 0.5,
    ):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_key = model_key
        self.config = MODEL_REGISTRY[model_key]
        self.threshold = threshold
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model loading ────────────────────────────────────────────────────

    @classmethod
    def _load_model(cls, model_key: str, device: str):
        """Load and cache the model identified by *model_key*."""
        if model_key in cls._cache:
            return cls._cache[model_key]

        cfg = MODEL_REGISTRY[model_key]
        model_path = cfg["path"]
        arch = cfg["architecture"]

        print(f"[GUS-Net] Loading {cfg['display_name']} from {model_path} ...")

        try:
            if arch == "gpt2":
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, add_prefix_space=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = GPT2ForTokenClassification.from_pretrained(model_path)
            else:
                # BERT: use tokenizer from registry (model repo may lack one)
                tok_name = cfg.get("tokenizer", "bert-base-uncased")
                tokenizer = BertTokenizerFast.from_pretrained(tok_name)
                model = BertForTokenClassification.from_pretrained(
                    model_path, num_labels=cfg["num_labels"]
                )

            model.eval()
            model.to(device)
            cls._cache[model_key] = (tokenizer, model)
            print(f"[GUS-Net] {cfg['display_name']} loaded successfully.")
        except Exception as e:
            print(f"[GUS-Net] WARNING — failed to load {cfg['display_name']}: {e}")
            raise

        return cls._cache[model_key]

    @property
    def is_available(self) -> bool:
        """Check whether the model can be loaded."""
        try:
            self._load_model(self.model_key, self._device)
            return True
        except Exception:
            return False

    # ── Inference ────────────────────────────────────────────────────────

    def detect_bias(self, text: str) -> List[Dict]:
        """Run GUS-Net inference on *text*.

        Returns a list of per-token dicts with keys:
            token, index, bias_types, is_biased, scores, method,
            explanation, threshold
        """
        tokenizer, model = self._load_model(self.model_key, self._device)
        cfg = self.config

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, num_labels]

        probabilities = torch.sigmoid(logits)[0].cpu()  # [seq_len, num_labels]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        results: List[Dict] = []
        for idx, (token, probs) in enumerate(zip(tokens, probabilities)):
            # Skip special tokens
            if token in cfg["special_tokens"]:
                results.append(self._empty_label(token, idx))
                continue

            # Per-category score = max(B-prob, I-prob)
            scores: Dict[str, float] = {}
            for cat, indices in cfg["category_indices"].items():
                scores[cat] = float(max(probs[i].item() for i in indices))

            # O score: direct from model output (both BERT and GPT-2 have O at index 0)
            scores["O"] = float(probs[cfg["o_index"]].item())

            # Threshold to determine active bias types
            bias_types = [
                cat for cat in ("GEN", "UNFAIR", "STEREO")
                if scores.get(cat, 0.0) >= self.threshold
            ]

            # Human-readable explanation
            _CAT_LABEL = {
                "GEN": "Generalization",
                "UNFAIR": "Unfair language",
                "STEREO": "Stereotype",
            }
            explanations = [
                f"{_CAT_LABEL[cat]} (score: {scores[cat]:.2f})"
                for cat in bias_types
            ]

            results.append({
                "token": token,
                "index": idx,
                "bias_types": bias_types,
                "is_biased": len(bias_types) > 0,
                "scores": scores,
                "method": "gusnet",
                "explanation": "; ".join(explanations),
                "threshold": self.threshold,
            })

        return results

    # ── Post-processing ──────────────────────────────────────────────────

    def get_bias_summary(self, token_labels: List[Dict]) -> Dict:
        """Generate summary statistics from detect_bias output."""
        special = self.config["special_tokens"]
        
        # Determine tokenizer type from tokens
        # BERT has ##, GPT-2 has Ġ (u0120)
        has_gpt2_tokens = any("\u0120" in t["token"] for t in token_labels)
        has_bert_tokens = any("##" in t["token"] for t in token_labels)
        
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
            elif has_bert_tokens:
                # BERT: User explicitly wants SPLIT. So never merge ##.
                should_merge = False
            else:
                # Fallback/Ambiguous: Only merge if explicitly starting with ## (Old behavior)
                # BUT user wants split for BERT. So actually, default to False.
                should_merge = False

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
    # Model A = HF ethical-spectacle/social-bias-ner  (better STEREO/UNFAIR)
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

    def detect_bias(self, text: str) -> List[Dict]:
        """Run ensemble inference combining both models."""
        probs_a = self._get_raw_probs(self.model_key_a, text)
        probs_b = self._get_raw_probs(self.model_key_b, text)

        # Get tokens from model A's tokenizer (shared)
        tokenizer_a, _ = GusNetDetector._load_model(self.model_key_a, self._device)
        inputs = tokenizer_a(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        tokens = tokenizer_a.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        cfg = self.config
        results: List[Dict] = []

        for idx, token in enumerate(tokens):
            if token in cfg["special_tokens"]:
                results.append({
                    "token": token, "index": idx, "bias_types": [],
                    "is_biased": False,
                    "scores": {"O": 0.0, "GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
                    "method": "gusnet-ensemble",
                    "explanation": "", "threshold": self.threshold,
                })
                continue

            pa = probs_a[idx]
            pb = probs_b[idx]

            # Weighted combination per category
            scores: Dict[str, float] = {}
            for cat, cat_indices in cfg["category_indices"].items():
                w_a, w_b = self.weights.get(cat, (0.5, 0.5))
                score_a = float(max(pa[i].item() for i in cat_indices))
                score_b = float(max(pb[i].item() for i in cat_indices))
                scores[cat] = w_a * score_a + w_b * score_b

            w_o_a, w_o_b = self.weights.get("O", (0.5, 0.5))
            scores["O"] = w_o_a * pa[cfg["o_index"]].item() + w_o_b * pb[cfg["o_index"]].item()

            bias_types = [
                cat for cat in ("GEN", "UNFAIR", "STEREO")
                if scores.get(cat, 0.0) >= self.threshold
            ]

            _CAT_LABEL = {
                "GEN": "Generalization",
                "UNFAIR": "Unfair language",
                "STEREO": "Stereotype",
            }
            explanations = [
                f"{_CAT_LABEL[cat]} (score: {scores[cat]:.2f})"
                for cat in bias_types
            ]

            results.append({
                "token": token,
                "index": idx,
                "bias_types": bias_types,
                "is_biased": len(bias_types) > 0,
                "scores": scores,
                "method": "gusnet-ensemble",
                "explanation": "; ".join(explanations),
                "threshold": self.threshold,
            })

        return results

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
