"""Counterfactual / Minimal-Pair Generation for Bias Analysis.

Rule-based demographic term swapping inspired by CrowS-Pairs
(Nangia et al., EMNLP 2020).  Bidirectional swap dictionary covers
gender, race, religion, age, and nationality categories.
"""

import re
from typing import Dict, List, Optional, Tuple

Category = str  # "gender" | "race" | "religion" | "age" | "nationality"

# ── Bidirectional swap pairs ──────────────────────────────────────────
# (term_a, term_b, category) - both directions are valid.

SWAP_PAIRS: List[Tuple[str, str, Category]] = [
    # ── Gender (CrowS-Pairs: 262 examples) ────────────────────────────
    ("he", "she", "gender"),
    ("him", "her", "gender"),
    ("his", "her", "gender"),
    ("himself", "herself", "gender"),
    ("he'd", "she'd", "gender"),
    ("man", "woman", "gender"),
    ("men", "women", "gender"),
    ("boy", "girl", "gender"),
    ("boys", "girls", "gender"),
    ("male", "female", "gender"),
    ("males", "females", "gender"),
    ("father", "mother", "gender"),
    ("dad", "mom", "gender"),
    ("son", "daughter", "gender"),
    ("sons", "daughters", "gender"),
    ("brother", "sister", "gender"),
    ("brothers", "sisters", "gender"),
    ("husband", "wife", "gender"),
    ("boyfriend", "girlfriend", "gender"),
    ("king", "queen", "gender"),
    ("prince", "princess", "gender"),
    ("uncle", "aunt", "gender"),
    ("nephew", "niece", "gender"),
    ("grandfather", "grandmother", "gender"),
    ("grandson", "granddaughter", "gender"),
    ("mr", "mrs", "gender"),
    ("sir", "madam", "gender"),
    ("ma'am", "sir", "gender"),
    ("gentleman", "lady", "gender"),
    ("gentlemen", "ladies", "gender"),
    ("guys", "women", "gender"),
    ("masculine", "feminine", "gender"),
    ("cisgender", "transgender", "gender"),
    ("cis", "trans", "gender"),

    # ── Race / Ethnicity (CrowS-Pairs: 516 examples) ─────────────────
    ("black", "white", "race-color"),
    ("blacks", "whites", "race-color"),
    ("african", "caucasian", "race-color"),
    ("african", "european", "race-color"),
    ("african", "white", "race-color"),
    ("african", "asian", "race-color"),
    ("african american", "white", "race-color"),
    ("african american", "caucasian", "race-color"),
    ("african americans", "whites", "race-color"),
    ("african americans", "caucasians", "race-color"),
    ("african-american", "white american", "race-color"),
    ("african-americans", "white americans", "race-color"),
    ("afro-american", "caucasian", "race-color"),
    ("asian", "white", "race-color"),
    ("asian", "black", "race-color"),
    ("asians", "whites", "race-color"),
    ("asian-americans", "white americans", "race-color"),
    ("hispanic", "white", "race-color"),
    ("hispanics", "whites", "race-color"),
    ("latino", "white", "race-color"),
    ("latinos", "white people", "race-color"),
    ("mexican", "white", "race-color"),
    ("mexicans", "whites", "race-color"),
    ("chinese", "white", "race-color"),
    ("indian", "white", "race-color"),
    ("indians", "whites", "race-color"),
    ("native", "white", "race-color"),
    ("native americans", "whites", "race-color"),
    ("native americans", "white people", "race-color"),
    ("colored", "white", "race-color"),
    ("dark-skinned", "light-skinned", "race-color"),
    ("minorities", "white people", "race-color"),
    ("puerto ricans", "white people", "race-color"),
    ("anglo americans", "hispanics", "race-color"),
    ("caucasian", "native american", "race-color"),
    ("black", "chinese", "race-color"),
    ("black", "spanish", "race-color"),
    ("black", "mexican", "race-color"),

    # ── Religion (CrowS-Pairs: 105 examples) ──────────────────────────
    ("christian", "muslim", "religion"),
    ("christians", "muslims", "religion"),
    ("christian", "jewish", "religion"),
    ("christian", "hindu", "religion"),
    ("christian", "jew", "religion"),
    ("christian", "wiccan", "religion"),
    ("christians", "jews", "religion"),
    ("christianity", "islam", "religion"),
    ("catholic", "muslim", "religion"),
    ("catholic", "protestant", "religion"),
    ("catholic", "jew", "religion"),
    ("catholics", "protestants", "religion"),
    ("atheist", "christian", "religion"),
    ("atheist", "muslim", "religion"),
    ("atheist", "jewish", "religion"),
    ("atheists", "catholics", "religion"),
    ("atheists", "christians", "religion"),
    ("buddhist", "christian", "religion"),
    ("baptist", "wiccan", "religion"),
    ("baptists", "muslims", "religion"),
    ("mormon", "muslim", "religion"),
    ("mormons", "muslims", "religion"),
    ("hindu", "lutheran", "religion"),
    ("jewish", "methodist", "religion"),
    ("jewish", "muslim", "religion"),
    ("judaism", "islam", "religion"),
    ("amish", "christian", "religion"),
    ("muslim", "pentecostal", "religion"),
    ("muslim", "unitarian", "religion"),
    ("church", "mosque", "religion"),
    ("churches", "mosques", "religion"),
    ("church", "temple", "religion"),
    ("synagogue", "mosque", "religion"),
    ("bible", "quran", "religion"),
    ("bible", "torah", "religion"),
    ("jesus", "mohammad", "religion"),
    ("christmas", "ramadan", "religion"),

    # ── Age (CrowS-Pairs: 87 examples) ────────────────────────────────
    ("young", "old", "age"),
    ("younger", "older", "age"),
    ("youngest", "oldest", "age"),
    ("youth", "elderly", "age"),
    ("teenager", "senior", "age"),
    ("teenagers", "seniors", "age"),
    ("adult", "teenage", "age"),
    ("adults", "teenagers", "age"),
    ("adults", "children", "age"),
    ("child", "adult", "age"),
    ("children", "elderly", "age"),
    ("middle aged", "young", "age"),

    # ── Nationality (CrowS-Pairs: 159 examples) ──────────────────────
    ("american", "mexican", "nationality"),
    ("americans", "mexicans", "nationality"),
    ("american", "italian", "nationality"),
    ("american", "chinese", "nationality"),
    ("american", "indian", "nationality"),
    ("american", "asian", "nationality"),
    ("american", "armenian", "nationality"),
    ("american", "vietnamese", "nationality"),
    ("american", "japanese", "nationality"),
    ("american", "polish", "nationality"),
    ("american", "russian", "nationality"),
    ("american", "arab", "nationality"),
    ("american", "dutch", "nationality"),
    ("american", "european", "nationality"),
    ("american", "latino", "nationality"),
    ("american", "pakistani", "nationality"),
    ("american", "scottish", "nationality"),
    ("american", "syrian", "nationality"),
    ("american", "thai", "nationality"),
    ("american", "saudi arabian", "nationality"),
    ("american", "immigrant", "nationality"),
    ("americans", "immigrants", "nationality"),
    ("americans", "foreigners", "nationality"),
    ("americans", "chinese", "nationality"),
    ("americans", "greeks", "nationality"),
    ("americans", "italians", "nationality"),
    ("americans", "asians", "nationality"),
    ("americans", "indians", "nationality"),
    ("americans", "middle easterners", "nationality"),
    ("british", "nigerian", "nationality"),
    ("british", "korean", "nationality"),
    ("british", "mexican", "nationality"),
    ("british", "indian", "nationality"),
    ("canadian", "mexican", "nationality"),
    ("canadian", "indian", "nationality"),
    ("canadian", "chinese", "nationality"),
    ("canadian", "middle eastern", "nationality"),
    ("european", "middle eastern", "nationality"),
    ("european", "mexican", "nationality"),
    ("chinese", "russian", "nationality"),
    ("chinese", "japanese", "nationality"),
    ("chinese", "french", "nationality"),
    ("chinese", "hispanic", "nationality"),
    ("english", "russian", "nationality"),
    ("english", "japanese", "nationality"),
    ("english", "italian", "nationality"),
    ("english", "korean", "nationality"),
    ("german", "french", "nationality"),
    ("german", "russian", "nationality"),
    ("german", "korean", "nationality"),
    ("italian", "japanese", "nationality"),
    ("japanese", "mongolian", "nationality"),
    ("greek", "norwegian", "nationality"),
    ("australian", "german", "nationality"),
    ("citizen", "refugee", "nationality"),
    ("citizens", "immigrants", "nationality"),

    # ── Sexual Orientation (CrowS-Pairs: 84 examples) ────────────────
    ("gay", "straight", "sexual-orientation"),
    ("gays", "straights", "sexual-orientation"),
    ("lesbian", "straight", "sexual-orientation"),
    ("homosexual", "heterosexual", "sexual-orientation"),
    ("homosexuals", "heterosexuals", "sexual-orientation"),
    ("bisexual", "heterosexual", "sexual-orientation"),
    ("bisexual", "straight", "sexual-orientation"),

    # ── Disability (CrowS-Pairs: 60 examples) ────────────────────────
    ("able-bodied", "disabled", "disability"),
    ("abled", "crippled", "disability"),
    ("disabled", "nondisabled", "disability"),
    ("handicapped", "healthy", "disability"),
    ("athletic", "paraplegic", "disability"),

    # ── Physical Appearance (CrowS-Pairs: 63 examples) ───────────────
    ("fat", "thin", "physical-appearance"),
    ("fat", "slim", "physical-appearance"),
    ("fat", "skinny", "physical-appearance"),
    ("fat", "fit", "physical-appearance"),
    ("short", "tall", "physical-appearance"),
    ("shortest", "tallest", "physical-appearance"),
    ("smallest", "tallest", "physical-appearance"),
    ("overweight", "thin", "physical-appearance"),
    ("overweight", "slim", "physical-appearance"),
    ("overweight", "underweight", "physical-appearance"),
    ("overweight", "fit", "physical-appearance"),
    ("beautiful", "ugly", "physical-appearance"),
    ("handsome", "ugly", "physical-appearance"),
    ("large", "small", "physical-appearance"),
    ("heavy", "light", "physical-appearance"),
    ("blonde", "brunette", "physical-appearance"),
    ("buff", "skinny", "physical-appearance"),

    # ── Socioeconomic Status (CrowS-Pairs: 172 examples) ─────────────
    ("poor", "rich", "socioeconomic"),
    ("poor", "wealthy", "socioeconomic"),
    ("poor", "privileged", "socioeconomic"),
    ("poverty", "wealth", "socioeconomic"),
    ("homeless", "landlord", "socioeconomic"),
    ("ghetto", "suburb", "socioeconomic"),
    ("educated", "uneducated", "socioeconomic"),
    ("doctor", "janitor", "socioeconomic"),
    ("ceo", "janitor", "socioeconomic"),
    ("lawyer", "waiter", "socioeconomic"),
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


# ── POS-based disambiguation of ambiguous reverse mappings ───────────
# English collapses distinct grammatical words into one surface form, so a
# purely lexical reverse lookup is ambiguous: "her" is both the object
# pronoun (→ "him") and the possessive determiner (→ "his"). For these keys
# the correct target depends on the token's POS tag in context; everywhere
# else the first-declared pair remains the deterministic choice.
_POS_DISAMBIGUATION: Dict[str, Dict[str, Tuple[str, Category]]] = {
    "her": {
        "PRP$": ("his", "gender"),   # possessive determiner: her car → his car
        "PRP":  ("him", "gender"),   # object pronoun: saw her → saw him
    },
}


# ── Polysemous terms: only swap in person-referential contexts ────────
# Colour / size / wealth / weight adjectives double as ordinary descriptors:
# "a large dataset", "a white shirt", "poor performance", "a heavy box".
# Swapping those produces invalid counterfactuals - the pair no longer
# differs only in the demographic attribute, which is the whole premise of
# minimal-pair analysis (Nangia et al., 2020). For the terms below we
# require evidence that the mention refers to people: the token is a
# NORP/PERSON entity, a noun used nominally ("the poor"), or an adjective
# whose syntactic head is a person-denoting noun.
_POLYSEMOUS_TERMS = frozenset({
    "black", "blacks", "white", "whites", "colored", "native",
    "short", "shortest", "tall", "tallest", "small", "smallest", "large",
    "heavy", "light", "fat", "thin", "slim", "skinny", "fit", "buff",
    "poor", "rich", "straight", "young", "younger", "youngest",
    "old", "older", "oldest", "educated", "uneducated",
    "beautiful", "handsome", "ugly", "blonde", "brunette",
    "adult", "adults", "child", "children", "youth", "indian", "indians",
})

# Lemmas of head nouns that mark an adjective as person-referential.
_PERSON_HEAD_LEMMAS = frozenset({
    "person", "people", "man", "woman", "men", "women", "guy", "girl",
    "boy", "child", "children", "kid", "adult", "teenager", "senior",
    "individual", "citizen", "immigrant", "refugee", "worker", "employee",
    "student", "teacher", "doctor", "nurse", "patient", "customer",
    "neighbor", "neighbour", "family", "community", "population", "folk",
    "lady", "gentleman", "male", "female", "resident", "voter", "parent",
    "mother", "father", "friend", "colleague", "americans", "american",
})


def _is_person_referential(tok) -> bool:
    """True when a spaCy token plausibly refers to people (not an object)."""
    if tok.ent_type_ in ("PERSON", "NORP"):
        return True
    if tok.pos_ in ("NOUN", "PROPN"):
        # Nominal use ("the poor", "young adults", "the elderly")
        return True
    if tok.pos_ == "ADJ":
        head = tok.head
        if head is not None and head is not tok:
            if head.ent_type_ in ("PERSON", "NORP"):
                return True
            if head.lemma_.lower() in _PERSON_HEAD_LEMMAS:
                return True
    return False


def _spacy_doc_for(text: str):
    """Parse *text* with the shared spaCy model; None when unavailable."""
    try:
        from ..head_specialization import get_spacy_model
        return get_spacy_model()(text)
    except Exception:
        return None


def _pos_tags_for(text: str) -> Dict[int, str]:
    """Return ``{char_offset: PTB tag}`` for *text* via the shared spaCy model.

    Returns an empty dict when spaCy (or its model) is unavailable so the
    caller silently falls back to the lexical first-pair behaviour.
    """
    doc = _spacy_doc_for(text)
    if doc is None:
        return {}
    return {tok.idx: tok.tag_ for tok in doc}


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
    doc = None          # lazy spaCy parse: only when needed
    doc_loaded = False
    tok_by_offset: Dict[int, object] = {}

    def _ensure_doc():
        nonlocal doc, doc_loaded, tok_by_offset
        if not doc_loaded:
            doc = _spacy_doc_for(text)
            doc_loaded = True
            if doc is not None:
                tok_by_offset = {tok.idx: tok for tok in doc}

    for key in _SORTED_KEYS:
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # Skip if this span overlaps with an already-matched term.
            if any(s <= start < e or s < end <= e for s, e in occupied):
                continue

            # Polysemous terms ("black", "poor", "large"…): only swap when
            # the mention is person-referential - otherwise the pair is not
            # a minimal demographic pair ("a large dataset" ≠ demographic).
            # When spaCy is unavailable the term is SKIPPED (a missed swap
            # is recoverable; a corrupted counterfactual is not).
            if key in _POLYSEMOUS_TERMS:
                _ensure_doc()
                tok = tok_by_offset.get(start)
                if tok is None or not _is_person_referential(tok):
                    continue

            targets = _SWAP_MAP[key]
            swap_to, category = targets[0]  # primary swap target
            ambiguous = _POS_DISAMBIGUATION.get(key)
            if ambiguous:
                _ensure_doc()
                tok = tok_by_offset.get(start)
                tag = tok.tag_ if tok is not None else None
                if tag in ambiguous:
                    swap_to, category = ambiguous[tag]
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
        # Fix ONLY the article immediately preceding this replacement.
        # A global pass would rewrite articles the swap never touched
        # (and any heuristic will get some of those wrong).
        text = _fix_article_before(text, swap["start"])
        applied.append({
            "original": original_fragment,
            "replacement": replacement,
            "category": swap["category"],
            "start": swap["start"],
            "end": swap["end"],
        })

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
    # Mixed case - keep replacement as-is (lower by default in SWAP_PAIRS).
    return replacement


# Words where "a" is correct despite starting with a vowel letter
# (because the pronunciation starts with a consonant sound).
_A_EXCEPTIONS = {"european", "united", "university", "uniform", "unique",
                 "universal", "unicorn", "union", "usage", "usual", "one",
                 "unitarian", "uneducated"}

# Words where "an" is correct despite starting with a consonant letter
# (silent h → the pronunciation starts with a vowel sound).
_AN_EXCEPTIONS = {"hour", "hourly", "honest", "honestly", "honor", "honour",
                  "honorable", "honourable", "heir", "heiress", "herb"}

_VOWELS = set("aeiouAEIOU")


def _correct_article(word: str) -> str:
    """Return "a" or "an" for *word* using letter + silent-h heuristics."""
    word_lower = word.lower()
    starts_vowel_sound = (
        (word[0] in _VOWELS and word_lower not in _A_EXCEPTIONS)
        or word_lower in _AN_EXCEPTIONS
    )
    return "an" if starts_vowel_sound else "a"


def _fix_article_before(text: str, word_start: int) -> str:
    """Fix a/an agreement for the article immediately preceding the word
    at *word_start* (a replacement site). Leaves the rest of the text
    untouched - the swap cannot have broken articles anywhere else."""
    m = re.search(r"\b(an?|An?)(\s+)$", text[:word_start])
    if not m:
        return text
    word_m = re.match(r"\w+", text[word_start:])
    if not word_m:
        return text

    article = m.group(1)
    correct = _correct_article(word_m.group())
    if article[0].isupper():
        correct = correct.title()
    if correct == article:
        return text
    return text[:m.start(1)] + correct + text[m.end(1):]


def _fix_articles(text: str) -> str:
    """Fix a/an agreement for EVERY article in *text*.

    Kept for backward compatibility only - prefer the targeted
    :func:`_fix_article_before`, which cannot corrupt articles the
    counterfactual swap never touched."""

    def _replace(m: re.Match) -> str:
        article = m.group(1)          # "a" or "an" (original)
        space = m.group(2)            # whitespace between article and word
        word = m.group(3)             # the following word

        correct = _correct_article(word)

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
