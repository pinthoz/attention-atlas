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
        model_name: HuggingFace model id — tokenizer / encoder / MLM head
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


def heavy_compute(text, model_name):
    """Perform heavy computation for attention analysis.

    Loads the model, tokenizes the input, runs inference, and computes
    all necessary metrics for visualization.

    Args:
        text: Input text to analyze
        model_name: Name of the model to use

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
    # card text claims. The previous behaviour exposed last_hidden_state —
    # the OUTPUT of the final layer, maximally contextual — which is still
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
    # models learn them — the Deep Dive card says so explicitly — so the
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
    try:
        _logger.debug("Computing head specialization")
        head_specialization = compute_all_heads_specialization(attentions, tokens, text)
        _logger.debug("Head specialization complete")
    except Exception as e:
        _logger.warning("Could not compute head specialization: %s", e)

    # Compute head clusters (t-SNE + KMeans)
    head_clusters = []
    if head_specialization:
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


__all__ = ["ComputeResult", "tokenize_with_segments", "heavy_compute"]
