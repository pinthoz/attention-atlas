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
        tokenizer: Tokenizer instance
        encoder_model: Encoder model instance
        mlm_model: MLM model instance
        head_specialization: Head specialization metrics
        isa_data: Inter-sentence attention data
        head_clusters: Algorithmic clustering results (t-SNE + K-Means)
    """
    tokens: Any
    embeddings: Any
    pos_enc: Any
    attentions: Any
    hidden_states: Any
    inputs: Any
    tokenizer: Any
    encoder_model: Any
    mlm_model: Any
    head_specialization: Any
    isa_data: Any
    head_clusters: Any

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

    Args:
        text: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Dict containing tokenized inputs with input_ids, attention_mask, and token_type_ids
    """
    pattern = re.search(r"([.!?])\s+([A-Za-z])", text)
    if pattern:
        split_idx = pattern.end(1)
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
    embeddings = outputs.last_hidden_state[0].cpu().numpy()
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
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
            head_clusters = compute_head_clusters(head_specialization)
            _logger.debug("Head clusters complete")
        except Exception as e:
            _logger.warning("Could not compute head clusters: %s", e)

    # Compute ISA
    isa_data = None
    try:
        _logger.debug("Computing ISA")
        isa_data = compute_isa(attentions, tokens, text, tokenizer, inputs)
        _logger.debug("ISA complete")
    except Exception as e:
        _logger.warning("Could not compute ISA: %s", e)

    _logger.debug("heavy_compute finished")
    return ComputeResult(
        tokens=tokens, embeddings=embeddings, pos_enc=pos_enc,
        attentions=attentions, hidden_states=hidden_states, inputs=inputs,
        tokenizer=tokenizer, encoder_model=encoder_model, mlm_model=mlm_model,
        head_specialization=head_specialization, isa_data=isa_data,
        head_clusters=head_clusters,
    )


__all__ = ["ComputeResult", "tokenize_with_segments", "heavy_compute"]
