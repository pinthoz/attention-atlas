"""GUS-Net Neural Bias Detector.

Wraps the pre-trained ethical-spectacle/social-bias-ner model for
token-level social bias detection using NER.

Label scheme (BIO):
  O, B-STEREO, I-STEREO, B-GEN, I-GEN, B-UNFAIR, I-UNFAIR
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from typing import List, Dict, Optional
from dataclasses import dataclass


LABEL2ID = {
    'O': 0,
    'B-STEREO': 1, 'I-STEREO': 2,
    'B-GEN': 3, 'I-GEN': 4,
    'B-UNFAIR': 5, 'I-UNFAIR': 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# Merged category indices for score extraction
CATEGORY_INDICES = {
    "STEREO": [1, 2],   # B-STEREO, I-STEREO
    "GEN": [3, 4],      # B-GEN, I-GEN
    "UNFAIR": [5, 6],   # B-UNFAIR, I-UNFAIR
}


class GusNetDetector:
    """Neural bias detector using the GUS-Net (social-bias-ner) model."""

    _instance = None
    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_name: str = "ethical-spectacle/social-bias-ner",
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        """Load and cache the GUS-Net model (singleton)."""
        if cls._model is not None:
            return cls._tokenizer, cls._model

        print(f"Loading GUS-Net model: {model_name}...")
        try:
            cls._tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            cls._model = BertForTokenClassification.from_pretrained(
                model_name, num_labels=NUM_LABELS
            )
            cls._model.eval()
            cls._model.to(device)
            print("GUS-Net model loaded successfully.")
        except Exception as e:
            print(f"WARNING: Failed to load GUS-Net model: {e}")
            cls._model = None
            cls._tokenizer = None
            raise

        return cls._tokenizer, cls._model

    @property
    def is_available(self) -> bool:
        """Check if the model was loaded successfully."""
        try:
            self._load_model(self.model_name, self._device)
            return self._model is not None
        except Exception:
            return False

    def detect_bias(self, text: str) -> List[Dict]:
        """Run GUS-Net inference on text.

        Args:
            text: Raw input text.

        Returns:
            List of per-token dicts with keys:
                token, index, bias_types, is_biased, scores, method,
                explanation, threshold
        """
        tokenizer, model = self._load_model(self.model_name, self._device)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, num_labels]

        probabilities = torch.sigmoid(logits)[0].cpu()  # [seq_len, num_labels]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        results = []
        for idx, (token, probs) in enumerate(zip(tokens, probabilities)):
            # Skip special tokens
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                results.append({
                    "token": token,
                    "index": idx,
                    "bias_types": [],
                    "is_biased": False,
                    "scores": {"GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
                    "method": "gusnet",
                    "explanation": "",
                    "threshold": self.threshold,
                })
                continue

            # Compute per-category score (max of B/I probabilities)
            scores = {}
            for cat, indices in CATEGORY_INDICES.items():
                scores[cat] = float(max(probs[i].item() for i in indices))

            # Determine which categories exceed threshold
            bias_types = [cat for cat, score in scores.items() if score >= self.threshold]

            # Build explanation
            explanations = []
            for cat in bias_types:
                if cat == "GEN":
                    explanations.append(f"Generalization (score: {scores[cat]:.2f})")
                elif cat == "UNFAIR":
                    explanations.append(f"Unfair language (score: {scores[cat]:.2f})")
                elif cat == "STEREO":
                    explanations.append(f"Stereotype (score: {scores[cat]:.2f})")

            results.append({
                "token": token,
                "index": idx,
                "bias_types": bias_types,
                "is_biased": len(bias_types) > 0,
                "scores": scores,
                "method": "gusnet",
                "explanation": "; ".join(explanations) if explanations else "",
                "threshold": self.threshold,
            })

        return results

    def get_bias_summary(self, token_labels: List[Dict]) -> Dict:
        """Generate summary statistics from detect_bias output."""
        content_tokens = [t for t in token_labels if t["token"] not in ("[CLS]", "[SEP]", "[PAD]")]
        total = len(content_tokens)
        biased = [t for t in content_tokens if t["is_biased"]]

        gen_count = sum(1 for t in biased if "GEN" in t["bias_types"])
        unfair_count = sum(1 for t in biased if "UNFAIR" in t["bias_types"])
        stereo_count = sum(1 for t in biased if "STEREO" in t["bias_types"])

        # Average confidence across biased tokens
        avg_confidence = 0.0
        if biased:
            all_scores = []
            for t in biased:
                for cat in t["bias_types"]:
                    all_scores.append(t["scores"].get(cat, 0.0))
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
                cat for cat, count in [("GEN", gen_count), ("UNFAIR", unfair_count), ("STEREO", stereo_count)]
                if count > 0
            ],
        }

    def get_biased_spans(self, token_labels: List[Dict]) -> List[Dict]:
        """Group contiguous biased tokens into spans.

        Returns list of dicts with: start_idx, end_idx, tokens, bias_types,
        avg_score, method, explanation
        """
        spans = []
        current = None

        for label in token_labels:
            if label["token"] in ("[CLS]", "[SEP]", "[PAD]"):
                if current is not None:
                    spans.append(self._finalize_span(current))
                    current = None
                continue

            if label["is_biased"]:
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

    @staticmethod
    def _finalize_span(current: Dict) -> Dict:
        """Compute averaged scores for a span."""
        all_cats = list(current["bias_types"])
        avg_scores = {}
        for cat in ("GEN", "UNFAIR", "STEREO"):
            cat_scores = [s.get(cat, 0.0) for s in current["scores"]]
            avg_scores[cat] = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0

        overall_avg = 0.0
        if all_cats:
            overall_avg = sum(avg_scores[c] for c in all_cats) / len(all_cats)

        return {
            "start_idx": current["start"],
            "end_idx": current["end"],
            "tokens": current["tokens"],
            "bias_types": all_cats,
            "avg_score": overall_avg,
            "method": "gusnet",
            "explanation": " | ".join(filter(None, current["explanations"])),
        }


__all__ = ["GusNetDetector"]
