import logging
import re
from dataclasses import dataclass
from typing import Any

import torch

from ..utils import positional_encoding
from ..models import ModelManager
from ..head_specialization import compute_all_heads_specialization, compute_head_clusters
from ..isa import compute_isa

_logger = logging.getLogger(__name__)


@dataclass
class ComputeResult:
    """Result of heavy_compute().

    Attributes:
        tokens: List of token strings
        embeddings: Numpy array of token embeddings
        pos_enc: Positional encoding array
        attentions: Tuple of attention tensors from each layer
        hidden_states: Tuple of hidden state tensors from each layer
        inputs: Tokenized inputs dict
        model_name: HuggingFace model id - tokenizer / encoder / MLM head
            are resolved lazily through ModelManager (see properties below)
        head_specialization: Head specialization metrics
        isa_data: Inter-sentence attention data
        head_clusters: Algorithmic clustering results (t-SNE + K-Means)

    Memory note: results used to hold hard references to the tokenizer and
    both model instances, so ModelManager's LRU eviction could never
    actually free an evicted model while any session kept a result alive
    (compare-mode holds two). Storing only ``model_name`` and resolving via
    the ModelManager cache keeps the attribute API identical while letting
    eviction reclaim the weights; a rare cache miss just reloads from the
    local HF snapshot.
    """
    tokens: Any
    embeddings: Any
    pos_enc: Any
    attentions: Any
    hidden_states: Any
    inputs: Any
    model_name: Any
    head_specialization: Any
    isa_data: Any
    head_clusters: Any

    def _resolve_models(self):
        if not self.model_name:
            return (None, None, None)
        return ModelManager.get_model(self.model_name)

    @property
    def tokenizer(self):
        return self._resolve_models()[0]

    @property
    def encoder_model(self):
        return self._resolve_models()[1]

    @property
    def mlm_model(self):
        return self._resolve_models()[2]

    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        return iter((
            self.tokens, self.embeddings, self.pos_enc, self.attentions,
            self.hidden_states, self.inputs, self.tokenizer, self.encoder_model,
            self.mlm_model, self.head_specialization, self.isa_data,
            self.head_clusters,
        ))


def tokenize_with_segments(text: str, tokenizer):
    """Tokenize text with automatic sentence segmentation.

    For BERT-style tokenizers (those with a ``[SEP]`` token), multi-sentence
    inputs are encoded as a sentence pair so the segment (token_type_ids)
    machinery is exercised. The split point is the sentence boundary closest
    to the middle of the text, so 3+ sentences produce balanced A/B segments
    instead of A=first sentence, B=everything else.

    GPT-2-style tokenizers have no segment semantics (``sep_token is None``):
    pair encoding would silently concatenate the two texts with nothing in
    between, so the text is tokenized whole.

    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Dict containing tokenized inputs with input_ids, attention_mask, and
        (for BERT-style pairs) token_type_ids
    """
    has_pair_semantics = getattr(tokenizer, "sep_token", None) is not None
    if has_pair_semantics:
        # An explicit [SEP] is the analyst declaring the split themselves,
        # which is how the reranker's query and passage arrive. Encode those
        # as the two segments; otherwise a query with no final stop would be
        # tokenised as one segment and every token would carry type id 0.
        sep_token = tokenizer.sep_token
        if sep_token in text:
            left, _, right = text.partition(sep_token)
            sentence_a, sentence_b = left.strip(), right.strip()
            if sentence_a and sentence_b:
                return tokenizer(sentence_a, sentence_b, return_tensors="pt",
                                 truncation=True, max_length=512)

        boundaries = [m.end(1) for m in re.finditer(r"([.!?])\s+([A-Za-z])", text)]
        if boundaries:
            # Sentence boundary closest to the midpoint -> balanced segments.
            split_idx = min(boundaries, key=lambda b: abs(b - len(text) // 2))
            sentence_a = text[:split_idx].strip()
            sentence_b = text[split_idx:].strip()
            if sentence_a and sentence_b:
                return tokenizer(sentence_a, sentence_b, return_tensors="pt",
                                 truncation=True, max_length=512)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=512)


def heavy_compute(text, model_name, *, with_specialization=True,
                  with_clusters=True, with_isa=True):
    """Perform heavy computation for attention analysis.

    Loads the model, tokenizes the input, runs inference, and computes
    all necessary metrics for visualization.

    Args:
        text: Input text to analyze
        model_name: Name of the model to use
        with_specialization: compute head specialization metrics (spaCy
            tagging + per-head metrics). Skipping it also skips clustering,
            which is derived from it.
        with_clusters: compute the t-SNE + K-Means head clusters.
        with_isa: compute the inter-sentence attention data.

    The three flags are keyword-only and default to ``True``, i.e. to the
    behaviour every existing caller already gets - they exist for the HTTP
    API, which only needs tokens + attentions for most requests and would
    otherwise pay for spaCy, t-SNE and ISA on every call. Skipped stages
    leave their ComputeResult field at the same value the corresponding
    failure path produces (``None`` / ``[]``).

    Returns:
        ComputeResult dataclass (also iterable for backward compat).
    """
    _logger.debug("Starting heavy_compute")
    if not text:
        return None

    tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
    _logger.debug("Models loaded")

    device = ModelManager.get_device()
    inputs = tokenize_with_segments(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs)

    _logger.debug("Model inference complete")

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Token embeddings for the Deep Dive "Token Embeddings" card: the
    # context-independent vocabulary vectors (word_embeddings / wte), as the
    # card text claims. The previous behaviour exposed last_hidden_state -
    # the OUTPUT of the final layer, maximally contextual - which is still
    # available as hidden_states[-1] where needed.
    with torch.no_grad():
        if hasattr(encoder_model, "embeddings"):  # BERT
            emb_layer = encoder_model.embeddings.word_embeddings
        else:  # GPT-2
            emb_layer = encoder_model.wte
        embeddings = emb_layer(inputs["input_ids"])[0].cpu().numpy()
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    # Positional embeddings: show the model's REAL learned position
    # embeddings (BERT: embeddings.position_embeddings; GPT-2: wpe). Both
    # models learn them - the Deep Dive card says so explicitly - so the
    # sinusoidal formula from the original Transformer (previous behaviour)
    # displayed values the models never use. Kept as a fallback only.
    try:
        if hasattr(encoder_model, "embeddings"):  # BERT
            pe_weight = encoder_model.embeddings.position_embeddings.weight
        else:  # GPT-2
            pe_weight = encoder_model.wpe.weight
        pos_enc = pe_weight[: len(tokens)].detach().cpu().numpy()
    except Exception:
        _logger.warning("Falling back to sinusoidal positional encoding")
        pos_enc = positional_encoding(len(tokens), embeddings.shape[1])

    # Compute head specialization metrics
    head_specialization = None
    if with_specialization:
        try:
            _logger.debug("Computing head specialization")
            head_specialization = compute_all_heads_specialization(attentions, tokens, text)
            _logger.debug("Head specialization complete")
        except Exception as e:
            _logger.warning("Could not compute head specialization: %s", e)

    # Compute head clusters (t-SNE + KMeans)
    head_clusters = []
    if with_clusters and head_specialization:
        try:
            _logger.debug("Computing head clusters")
            head_clusters = compute_head_clusters(
                head_specialization,
                is_gpt_style=any("Ġ" in t for t in tokens),
            )
            _logger.debug("Head clusters complete")
        except Exception as e:
            _logger.warning("Could not compute head clusters: %s", e)

    # Compute ISA. Pass the correct model_type: GPT-2-style tokenisations
    # carry the Ġ marker; compute_isa also auto-detects causality from the
    # matrix, but the explicit flag keeps the metadata honest.
    isa_data = None
    if with_isa:
        try:
            _logger.debug("Computing ISA")
            isa_model_type = "gpt" if any("Ġ" in t for t in tokens) else "bert"
            isa_data = compute_isa(attentions, tokens, text, tokenizer, inputs,
                                   model_type=isa_model_type)
            _logger.debug("ISA complete")
        except Exception as e:
            _logger.warning("Could not compute ISA: %s", e)

    _logger.debug("heavy_compute finished")
    return ComputeResult(
        tokens=tokens, embeddings=embeddings, pos_enc=pos_enc,
        attentions=attentions, hidden_states=hidden_states, inputs=inputs,
        model_name=model_name,
        head_specialization=head_specialization, isa_data=isa_data,
        head_clusters=head_clusters,
    )




# ── Batch file parsing ──────────────────────────────────────────────────

# Column and key names accepted for each half of a reranker pair. A pair
# file is recognised by having one of each, so a plain text corpus with a
# "query" column but no passage still parses as single sentences.
_QUERY_KEYS = ("query", "q", "question")
_PASSAGE_KEYS = ("passage", "document", "doc", "passage_text", "text_b")
_TEXT_KEYS = ("text", "sentence", "input", "content")

PAIR_SEPARATOR = " [SEP] "


@dataclass
class BatchPayload:
    """What a batch upload turned into.

    ``sentences`` is always a flat list of strings, because every downstream
    stage takes one string per item. A reranker pair is carried as
    ``"query [SEP] passage"``, the same shape the prompt box produces, so
    ``tokenize_with_segments`` encodes it as two segments.
    """
    sentences: list
    is_pairs: bool = False
    skipped: int = 0


def _match_key(keys, candidates):
    """First key in ``keys`` whose lowercased name is in ``candidates``."""
    for candidate in candidates:
        for k in keys:
            if isinstance(k, str) and k.lower().strip() == candidate:
                return k
    return None


def _pair_from_mapping(item, q_key, p_key):
    """Join one mapping into a pair string, or None when a half is missing."""
    query = (item.get(q_key) or "")
    passage = (item.get(p_key) or "")
    if not isinstance(query, str) or not isinstance(passage, str):
        return None
    query, passage = query.strip(), passage.strip()
    if not query or not passage:
        return None
    return f"{query}{PAIR_SEPARATOR}{passage}"


def parse_batch_payload(content: str, file_name: str, max_items: int = 500):
    """Turn an uploaded CSV or JSON batch file into a ``BatchPayload``.

    Accepts three shapes:

    * a list of strings, or a column of them, for plain sentence batches;
    * records with a text field (``text`` / ``sentence`` / ``input`` /
      ``content``);
    * records with both a query and a passage field, which is how a
      cross-encoder reranker batch is given. Those are joined with an
      explicit ``[SEP]`` so the pair survives into the segment ids.

    Rows missing one half of a pair are counted in ``skipped`` rather than
    silently dropped: a reranker cannot score half a pair, and an analyst
    who uploads 200 rows and gets 180 results should be told why.
    """
    import csv
    import io

    sentences = []
    is_pairs = False
    skipped = 0

    lower_name = (file_name or "").lower()

    if lower_name.endswith(".json"):
        import json as _json
        data = _json.loads(content)
        records = None
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                sentences = [s.strip() for s in data if s.strip()]
            elif all(isinstance(x, dict) for x in data):
                records = data
        elif isinstance(data, dict):
            for key in ("pairs", "sentences", "texts", "data"):
                if key in data and isinstance(data[key], list):
                    inner = data[key]
                    if all(isinstance(x, str) for x in inner):
                        sentences = [s.strip() for s in inner if s.strip()]
                    else:
                        records = [x for x in inner if isinstance(x, dict)]
                    break

        if records:
            keys = set()
            for item in records:
                keys.update(item.keys())
            q_key = _match_key(keys, _QUERY_KEYS)
            p_key = _match_key(keys, _PASSAGE_KEYS)
            if q_key and p_key:
                is_pairs = True
                for item in records:
                    joined = _pair_from_mapping(item, q_key, p_key)
                    if joined:
                        sentences.append(joined)
                    else:
                        skipped += 1
            else:
                t_key = _match_key(keys, _TEXT_KEYS)
                for item in records:
                    val = item.get(t_key) if t_key else None
                    if isinstance(val, str) and val.strip():
                        sentences.append(val.strip())

    elif lower_name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content))
        cols = reader.fieldnames or []
        q_col = _match_key(cols, _QUERY_KEYS)
        p_col = _match_key(cols, _PASSAGE_KEYS)

        if q_col and p_col:
            is_pairs = True
            for row in reader:
                joined = _pair_from_mapping(row, q_col, p_col)
                if joined:
                    sentences.append(joined)
                else:
                    skipped += 1
        else:
            text_col = _match_key(cols, _TEXT_KEYS)
            if not text_col and cols:
                text_col = cols[0]
            if text_col:
                for row in reader:
                    val = (row.get(text_col) or "").strip()
                    if val:
                        sentences.append(val)

    if len(sentences) > max_items:
        sentences = sentences[:max_items]

    return BatchPayload(sentences=sentences, is_pairs=is_pairs, skipped=skipped)


__all__ = ["ComputeResult", "tokenize_with_segments", "heavy_compute",
           "BatchPayload", "parse_batch_payload", "PAIR_SEPARATOR"]
