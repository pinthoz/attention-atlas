"""Head Ablation for Faithfulness Validation.

Measures the causal impact of individual attention heads by zeroing
each head's output and computing the resulting representation change.

Approach
--------
For each head (l, h):
    1. Register a forward hook on the self-attention module that zeros
       the output slice for head h.
    2. Run a forward pass through the encoder.
    3. Compare the final-layer hidden states with the original (non-ablated)
       hidden states using cosine similarity.
    4. Optionally compute KL divergence of LM-head logits.
    5. Remove the hook (guaranteed via ``finally``).

This tells us whether a head that *attends* to biased tokens actually
*affects* the model's output — bridging the gap between attention-based
analysis and causal faithfulness.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch.nn.functional import cosine_similarity


@dataclass
class HeadAblationResult:
    """Result of ablating a single attention head.

    Attributes
    ----------
    layer : int
        Layer index of the ablated head.
    head : int
        Head index within the layer.
    representation_impact : float
        1 − mean cosine_similarity(original_hidden, ablated_hidden)
        over all token positions.  Higher = removing this head changes
        the representation more.
    kl_divergence : float | None
        KL(original_logits ∥ ablated_logits) averaged over positions.
        None if no LM head is available.
    bar_original : float
        Original BAR value for this head (for cross-referencing).
    """
    layer: int
    head: int
    representation_impact: float
    kl_divergence: Optional[float]
    bar_original: float


def _get_head_dim(model, is_gpt2: bool) -> int:
    """Return per-head dimension from model config."""
    cfg = model.config
    return cfg.hidden_size // cfg.num_attention_heads


def _get_attention_module(model, layer_idx: int, is_gpt2: bool):
    """Return the self-attention module for the given layer."""
    if is_gpt2:
        return model.transformer.h[layer_idx].attn
    return model.encoder.layer[layer_idx].attention.self


def _run_single_ablation(
    encoder_model,
    inputs: dict,
    original_hidden: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    is_gpt2: bool,
    lm_head_model=None,
    original_logits: Optional[torch.Tensor] = None,
) -> Tuple[float, Optional[float]]:
    """Ablate one head and measure impact.

    Returns (representation_impact, kl_divergence_or_None).
    """
    head_dim = _get_head_dim(encoder_model, is_gpt2)
    attn_module = _get_attention_module(encoder_model, layer_idx, is_gpt2)

    def ablation_hook(module, input, output):
        modified = output[0].clone()
        start = head_idx * head_dim
        end = start + head_dim
        modified[:, :, start:end] = 0.0
        return (modified,) + output[1:]

    handle = attn_module.register_forward_hook(ablation_hook)
    try:
        with torch.no_grad():
            ablated_outputs = encoder_model(**inputs)
        ablated_hidden = ablated_outputs.last_hidden_state  # [1, seq, hidden]

        # Representation impact: 1 − mean cosine similarity
        cos_sim = cosine_similarity(
            original_hidden.squeeze(0),   # [seq, hidden]
            ablated_hidden.squeeze(0),    # [seq, hidden]
            dim=-1,
        )  # [seq]
        rep_impact = 1.0 - cos_sim.mean().item()

        # KL divergence on LM-head logits (optional)
        kl_div_val = None
        if lm_head_model is not None and original_logits is not None:
            with torch.no_grad():
                if is_gpt2:
                    ablated_logits = lm_head_model.lm_head(ablated_hidden)
                else:
                    ablated_logits = lm_head_model.cls(ablated_hidden)
            log_q = torch.log_softmax(ablated_logits, dim=-1)
            log_p = torch.log_softmax(original_logits, dim=-1)
            p = log_p.exp()
            kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
            kl_div_val = float(kl)
    finally:
        handle.remove()

    return float(rep_impact), kl_div_val


def batch_ablate_top_heads(
    encoder_model,
    lm_head_model,
    tokenizer,
    text: str,
    top_heads: list,
    is_gpt2: bool,
) -> List[HeadAblationResult]:
    """Ablate each of the top-K heads and return impact results.

    Parameters
    ----------
    encoder_model : PreTrainedModel
        Cached encoder (BertModel or GPT2Model) with output_attentions=True.
    lm_head_model : PreTrainedModel or None
        LM head (BertForMaskedLM or GPT2LMHeadModel).  If None, KL
        divergence is skipped.
    tokenizer : PreTrainedTokenizer
    text : str
        The same sentence used in the bias analysis.
    top_heads : list[HeadBiasMetrics]
        Heads to ablate (typically sorted by BAR descending).
    is_gpt2 : bool
        True for GPT-2 models, False for BERT.

    Returns
    -------
    list[HeadAblationResult]
        Sorted by representation_impact descending.
    """
    device = next(encoder_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Baseline forward pass
    with torch.no_grad():
        baseline_outputs = encoder_model(**inputs)
    original_hidden = baseline_outputs.last_hidden_state  # [1, seq, hidden]

    # Baseline LM logits (if available)
    original_logits = None
    if lm_head_model is not None:
        try:
            with torch.no_grad():
                if is_gpt2:
                    original_logits = lm_head_model.lm_head(original_hidden)
                else:
                    original_logits = lm_head_model.cls(original_hidden)
        except Exception:
            original_logits = None

    results = []
    for head_metric in top_heads:
        rep_impact, kl_val = _run_single_ablation(
            encoder_model,
            inputs,
            original_hidden,
            head_metric.layer,
            head_metric.head,
            is_gpt2,
            lm_head_model,
            original_logits,
        )
        results.append(HeadAblationResult(
            layer=head_metric.layer,
            head=head_metric.head,
            representation_impact=rep_impact,
            kl_divergence=kl_val,
            bar_original=head_metric.bias_attention_ratio,
        ))

    results.sort(key=lambda r: r.representation_impact, reverse=True)
    return results


__all__ = ["HeadAblationResult", "batch_ablate_top_heads"]
