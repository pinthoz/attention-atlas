"""Counterfactual / Minimal-Pair Generation for Bias Analysis.

Rule-based demographic term swapping inspired by CrowS-Pairs
(Nangia et al., EMNLP 2020).  Bidirectional swap dictionary covers
gender, race, religion, age, and nationality categories.
"""

import re
from typing import Dict, List, Optional, Tuple

Category = str  # "gender" | "race" | "religion" | "age" | "nationality"

# ── Bidirectional swap pairs ──────────────────────────────────────────
# (term_a, term_b, category) — both directions are valid.

SWAP_PAIRS: List[Tuple[str, str, Category]] = [
    # Gender
    ("man", "woman", "gender"),
    ("men", "women", "gender"),
    ("boy", "girl", "gender"),
    ("boys", "girls", "gender"),
    ("he", "she", "gender"),
    ("him", "her", "gender"),
    ("his", "her", "gender"),
    ("himself", "herself", "gender"),
    ("father", "mother", "gender"),
    ("son", "daughter", "gender"),
    ("sons", "daughters", "gender"),
    ("brother", "sister", "gender"),
    ("brothers", "sisters", "gender"),
    ("husband", "wife", "gender"),
    ("king", "queen", "gender"),
    ("prince", "princess", "gender"),
    ("uncle", "aunt", "gender"),
    ("nephew", "niece", "gender"),
    ("grandfather", "grandmother", "gender"),
    ("grandson", "granddaughter", "gender"),
    ("male", "female", "gender"),
    ("males", "females", "gender"),
    ("mr", "mrs", "gender"),
    ("sir", "madam", "gender"),
    ("gentleman", "lady", "gender"),
    ("gentlemen", "ladies", "gender"),
    ("boyfriend", "girlfriend", "gender"),

    # Race / Ethnicity
    ("white", "black", "race"),
    ("whites", "blacks", "race"),
    ("caucasian", "african american", "race"),
    ("european", "african", "race"),
    ("hispanic", "latino", "race"),

    # Religion
    ("christian", "muslim", "religion"),
    ("christians", "muslims", "religion"),
    ("christianity", "islam", "religion"),
    ("church", "mosque", "religion"),
    ("churches", "mosques", "religion"),
    ("bible", "quran", "religion"),
    ("jewish", "muslim", "religion"),
    ("judaism", "islam", "religion"),
    ("synagogue", "mosque", "religion"),
    ("hindu", "buddhist", "religion"),
    ("catholic", "protestant", "religion"),

    # Age
    ("young", "old", "age"),
    ("younger", "older", "age"),
    ("youth", "elderly", "age"),
    ("teenager", "senior", "age"),
    ("teenagers", "seniors", "age"),

    # Nationality
    ("american", "mexican", "nationality"),
    ("americans", "mexicans", "nationality"),
    ("british", "indian", "nationality"),
    ("chinese", "japanese", "nationality"),
    ("german", "french", "nationality"),
]


# ── Compiled bidirectional lookup (built once at import) ─────────────

_SWAP_MAP: Dict[str, List[Tuple[str, Category]]] = {}


def _build_swap_map() -> None:
    for a, b, cat in SWAP_PAIRS:
        al, bl = a.lower(), b.lower()
        _SWAP_MAP.setdefault(al, []).append((bl, cat))
        if al != bl:
            _SWAP_MAP.setdefault(bl, []).append((al, cat))


_build_swap_map()

# Pre-sorted keys by length (longest first) for greedy matching.
_SORTED_KEYS: List[str] = sorted(_SWAP_MAP.keys(), key=len, reverse=True)


# ── Public API ────────────────────────────────────────────────────────

def get_swap_for_token(token: str) -> Optional[Tuple[str, str]]:
    """Return ``(swap_target, category)`` if *token* is swappable, else *None*.

    Lookup is case-insensitive.  Strips BERT ``##`` and GPT-2 ``Ġ`` markers
    before matching.
    """
    clean = token.replace("##", "").replace("\u0120", "").strip().lower()
    targets = _SWAP_MAP.get(clean)
    return targets[0] if targets else None


def find_swappable_terms(text: str) -> List[Dict]:
    """Scan *text* for demographic terms that have counterfactual swaps.

    Returns a list of dicts sorted by position::

        [{"term": "women", "start": 4, "end": 9,
          "swap_to": "men", "category": "gender"}, ...]

    Multi-word terms are checked first (longest-match priority).
    Matching is case-insensitive; ``term`` preserves original case.
    """
    results: List[Dict] = []
    occupied: List[Tuple[int, int]] = []  # spans already matched

    for key in _SORTED_KEYS:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # Skip if this span overlaps with an already-matched term.
            if any(s <= start < e or s < end <= e for s, e in occupied):
                continue
            targets = _SWAP_MAP[key]
            swap_to, category = targets[0]  # primary swap target
            results.append({
                "term": m.group(),
                "start": start,
                "end": end,
                "swap_to": swap_to,
                "category": category,
            })
            occupied.append((start, end))

    results.sort(key=lambda d: d["start"])
    return results


def generate_counterfactual(
    text: str,
    swaps: Optional[List[Dict]] = None,
) -> Tuple[str, List[Dict]]:
    """Generate a counterfactual sentence by applying demographic swaps.

    Parameters
    ----------
    text : str
        Original input text.
    swaps : list of dict, optional
        Pre-computed swaps from :func:`find_swappable_terms`.
        If *None*, calls ``find_swappable_terms(text)`` automatically.

    Returns
    -------
    (counterfactual_text, applied_swaps)
        *applied_swaps* is a list of dicts with keys
        ``original``, ``replacement``, ``category``, ``start``, ``end``
        (positions refer to the **original** text).
    """
    if swaps is None:
        swaps = find_swappable_terms(text)
    if not swaps:
        return text, []

    applied: List[Dict] = []
    # Process right-to-left so earlier offsets stay valid.
    for swap in sorted(swaps, key=lambda s: s["start"], reverse=True):
        original_fragment = text[swap["start"]:swap["end"]]
        replacement = _match_case(original_fragment, swap["swap_to"])
        text = text[:swap["start"]] + replacement + text[swap["end"]:]
        applied.append({
            "original": original_fragment,
            "replacement": replacement,
            "category": swap["category"],
            "start": swap["start"],
            "end": swap["end"],
        })

    text = _fix_articles(text)
    applied.sort(key=lambda d: d["start"])
    return text, applied


# ── Helpers ───────────────────────────────────────────────────────────

def _match_case(original: str, replacement: str) -> str:
    """Transfer casing pattern from *original* to *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    if original.islower():
        return replacement.lower()
    # Mixed case — keep replacement as-is (lower by default in SWAP_PAIRS).
    return replacement


# Words where "a" is correct despite starting with a vowel letter
# (because the pronunciation starts with a consonant sound).
_A_EXCEPTIONS = {"european", "united", "university", "uniform", "unique",
                 "universal", "unicorn", "union", "usage", "usual", "one"}

_VOWELS = set("aeiouAEIOU")


def _fix_articles(text: str) -> str:
    """Fix a/an article agreement after a demographic swap."""

    def _replace(m: re.Match) -> str:
        article = m.group(1)          # "a" or "an" (original)
        space = m.group(2)            # whitespace between article and word
        word = m.group(3)             # the following word

        word_lower = word.lower()
        starts_vowel_sound = (
            word[0] in _VOWELS and word_lower not in _A_EXCEPTIONS
        )

        if starts_vowel_sound:
            correct = "an"
        else:
            correct = "a"

        # Preserve original capitalisation of article.
        if article[0].isupper():
            correct = correct.title()

        return correct + space + word

    return re.sub(r"\b(an?|An?)\b(\s+)(\w+)", _replace, text)


__all__ = [
    "SWAP_PAIRS",
    "get_swap_for_token",
    "find_swappable_terms",
    "generate_counterfactual",
]
