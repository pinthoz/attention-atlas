"""Token-Level Bias Detector based on GUS-Net approach.

This module implements span-level bias detection for three categories:
- Generalizations (GEN): Overgeneralizations about groups
- Unfairness (UNFAIR): Unfair or derogatory language
- Stereotypes (STEREO): Stereotypical associations
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass


@dataclass
class BiasSpan:
    """Represents a detected bias span in text."""
    start_idx: int
    end_idx: int
    tokens: List[str]
    bias_types: List[str]  # One or more of: GEN, UNFAIR, STEREO
    confidence: float
    explanation: str
    method: str = "lexicon"          # "lexicon" or "gusnet" or "combined"
    avg_score: float = 1.0           # Average confidence score (0-1)


class TokenBiasDetector:
    """Detects token-level bias patterns using rule-based methods."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize detector with bias lexicons.

        Args:
            data_dir: Path to directory containing JSON data files.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"

        self.data_dir = Path(data_dir)
        self._load_lexicons()

    def _load_lexicons(self):
        """Load bias lexicons from JSON files."""
        with open(self.data_dir / "bias_lexicon.json", "r") as f:
            lexicon = json.load(f)
            self.generalization_markers = set(lexicon["generalization_markers"])
            self.group_nouns = set(lexicon["group_nouns"])
            self.unfair_negative = set(lexicon["unfair_words"]["negative_traits"])
            self.unfair_limiting = set(lexicon["unfair_words"]["limiting_descriptors"])
            self.stereotype_gender = set(lexicon["stereotype_patterns"]["gender"])
            self.stereotype_race = set(lexicon["stereotype_patterns"]["race_ethnicity"])
            self.stereotype_age = set(lexicon["stereotype_patterns"]["age"])

        with open(self.data_dir / "gender_pairs.json", "r") as f:
            gender_data = json.load(f)
            self.gender_specific = (
                set(gender_data["gender_specific_words"]["female"]) |
                set(gender_data["gender_specific_words"]["male"])
            )

    def detect_bias(self, text: str, tokens: List[str]) -> List[Dict]:
        """Detect bias in tokenized text.

        Args:
            text: Original text
            tokens: List of tokens (from tokenizer)

        Returns:
            List of dictionaries containing bias information for each token:
                - token: The token text
                - index: Token position
                - bias_types: List of bias categories (GEN, UNFAIR, STEREO)
                - is_biased: Boolean indicating if token is biased
        """
        text_lower = text.lower()
        token_labels = []

        for i, token in enumerate(tokens):
            # Skip special tokens
            if token.startswith("[") and token.endswith("]"):
                token_labels.append({
                    "token": token,
                    "index": i,
                    "bias_types": [],
                    "is_biased": False,
                    "scores": {"GEN": 0.0, "UNFAIR": 0.0, "STEREO": 0.0},
                    "method": "lexicon",
                    "explanation": "",
                    "threshold": 0.5,
                })
                continue

            # Clean token (remove ## for BERT subwords)
            clean_token = token.replace("##", "").lower()

            bias_types = []
            explanation_parts = []

            # Check for generalization patterns
            if self._is_generalization(i, tokens, text_lower):
                bias_types.append("GEN")
                explanation_parts.append("Part of overgeneralization")

            # Check for unfair language
            if clean_token in self.unfair_negative:
                bias_types.append("UNFAIR")
                explanation_parts.append("Negative/derogatory term")
            elif clean_token in self.unfair_limiting:
                bias_types.append("UNFAIR")
                explanation_parts.append("Limiting descriptor")

            # Check for stereotype patterns
            stereotype_type = self._check_stereotype(i, tokens, text_lower)
            if stereotype_type:
                bias_types.append("STEREO")
                explanation_parts.append(f"Stereotypical {stereotype_type} association")

            # Build scores dict (1.0 for matched rules, 0.0 otherwise)
            scores = {
                "GEN": 1.0 if "GEN" in bias_types else 0.0,
                "UNFAIR": 1.0 if "UNFAIR" in bias_types else 0.0,
                "STEREO": 1.0 if "STEREO" in bias_types else 0.0,
            }

            token_labels.append({
                "token": token,
                "index": i,
                "bias_types": bias_types,
                "is_biased": len(bias_types) > 0,
                "scores": scores,
                "method": "lexicon",
                "explanation": "; ".join(explanation_parts) if explanation_parts else "",
                "threshold": 0.5,
            })

        return token_labels

    def _is_generalization(self, idx: int, tokens: List[str], text_lower: str) -> bool:
        """Check if token is part of a generalization pattern.

        Pattern: [generalization_marker] + [group_noun] + [verb/attribute]
        Example: "All women are emotional"
        """
        if idx >= len(tokens):
            return False

        clean_token = tokens[idx].replace("##", "").lower()

        # Check if current token is a generalization marker
        if clean_token in self.generalization_markers:
            # Look ahead for group noun
            for j in range(idx + 1, min(idx + 3, len(tokens))):
                next_token = tokens[j].replace("##", "").lower()
                if next_token in self.group_nouns:
                    return True

        # Check if current token is a group noun preceded by generalization marker
        if clean_token in self.group_nouns:
            # Look back for generalization marker
            for j in range(max(0, idx - 2), idx):
                prev_token = tokens[j].replace("##", "").lower()
                if prev_token in self.generalization_markers:
                    return True

            # CHECK FOR IMPLIED GENERALIZATION (e.g., "Women should not...")
            # If group noun is subject and followed by prescriptive/limiting markers
            prescriptive_markers = {"should", "must", "cannot", "cant", "can't", "ought", "need", "do", "don't", "dont", "are"}
            
            # Simple heuristic: Look ahead 1-3 tokens for these markers
            for j in range(idx + 1, min(idx + 4, len(tokens))):
                next_token = tokens[j].replace("##", "").lower()
                if next_token in prescriptive_markers:
                    return True

        return False

    def _check_stereotype(self, idx: int, tokens: List[str], text_lower: str) -> Optional[str]:
        """Check if token is part of a stereotypical association.

        Returns:
            Type of stereotype (e.g., "gender", "race") or None
        """
        if idx >= len(tokens):
            return None

        clean_token = tokens[idx].replace("##", "").lower()

        # Check for gender stereotypes
        if clean_token in self.stereotype_gender:
            # Look for gender-specific words in context (window of 5 tokens)
            window_start = max(0, idx - 5)
            window_end = min(len(tokens), idx + 6)

            for j in range(window_start, window_end):
                context_token = tokens[j].replace("##", "").lower()
                if context_token in self.gender_specific:
                    return "gender"

        # Check for race/ethnicity stereotypes
        if clean_token in self.stereotype_race:
            # Simple presence check
            return "race/ethnicity"

        # Check for age stereotypes
        if clean_token in self.stereotype_age:
            age_indicators = {"old", "young", "elderly", "senior", "youth", "teenager", "boomer", "millennial"}
            window_start = max(0, idx - 5)
            window_end = min(len(tokens), idx + 6)

            for j in range(window_start, window_end):
                context_token = tokens[j].replace("##", "").lower()
                if context_token in age_indicators:
                    return "age"

        return None

    def get_bias_summary(self, token_labels: List[Dict]) -> Dict:
        """Generate summary statistics for detected bias.

        Args:
            token_labels: Output from detect_bias()

        Returns:
            Dictionary with summary statistics
        """
        total_tokens = len([t for t in token_labels if not t["token"].startswith("[")])
        biased_tokens = [t for t in token_labels if t["is_biased"]]

        gen_count = sum(1 for t in biased_tokens if "GEN" in t["bias_types"])
        unfair_count = sum(1 for t in biased_tokens if "UNFAIR" in t["bias_types"])
        stereo_count = sum(1 for t in biased_tokens if "STEREO" in t["bias_types"])

        # Average confidence across biased tokens
        avg_confidence = 0.0
        if biased_tokens:
            all_scores = []
            for t in biased_tokens:
                for cat in t["bias_types"]:
                    all_scores.append(t.get("scores", {}).get(cat, 1.0))
            avg_confidence = sum(all_scores) / len(all_scores) if all_scores else 1.0

        return {
            "total_tokens": total_tokens,
            "biased_tokens": len(biased_tokens),
            "bias_percentage": (len(biased_tokens) / total_tokens * 100) if total_tokens > 0 else 0,
            "generalization_count": gen_count,
            "unfairness_count": unfair_count,
            "stereotype_count": stereo_count,
            "avg_confidence": avg_confidence,
            "categories_found": [
                cat for cat, count in [("GEN", gen_count), ("UNFAIR", unfair_count), ("STEREO", stereo_count)]
                if count > 0
            ]
        }

    def get_biased_spans(self, token_labels: List[Dict]) -> List[BiasSpan]:
        """Extract contiguous spans of biased tokens.

        Args:
            token_labels: Output from detect_bias()

        Returns:
            List of BiasSpan objects
        """
        spans = []
        current_span = None

        for i, label in enumerate(token_labels):
            if label["is_biased"]:
                if current_span is None:
                    current_span = {
                        "start": i,
                        "end": i,
                        "tokens": [label["token"]],
                        "bias_types": set(label["bias_types"]),
                        "explanations": [label["explanation"]],
                        "scores": [label.get("scores", {})],
                        "methods": [label.get("method", "lexicon")],
                    }
                else:
                    current_span["end"] = i
                    current_span["tokens"].append(label["token"])
                    current_span["bias_types"].update(label["bias_types"])
                    current_span["explanations"].append(label["explanation"])
                    current_span["scores"].append(label.get("scores", {}))
                    current_span["methods"].append(label.get("method", "lexicon"))
            else:
                if current_span is not None:
                    spans.append(self._build_span(current_span))
                    current_span = None

        if current_span is not None:
            spans.append(self._build_span(current_span))

        return spans

    @staticmethod
    def _build_span(current: Dict) -> 'BiasSpan':
        """Build a BiasSpan from accumulated span data."""
        bias_types = list(current["bias_types"])
        # Compute average score across all tokens in span
        all_scores = []
        for scores_dict in current["scores"]:
            for cat in bias_types:
                all_scores.append(scores_dict.get(cat, 1.0))
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 1.0

        # Determine method (if all same, use that; otherwise "combined")
        methods = set(current["methods"])
        method = methods.pop() if len(methods) == 1 else "combined"

        return BiasSpan(
            start_idx=current["start"],
            end_idx=current["end"],
            tokens=current["tokens"],
            bias_types=bias_types,
            confidence=avg_score,
            explanation=" | ".join(filter(None, current["explanations"])),
            method=method,
            avg_score=avg_score,
        )


__all__ = ["TokenBiasDetector", "BiasSpan"]
