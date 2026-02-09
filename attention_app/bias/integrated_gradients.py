"""Attention vs Integrated Gradients — Faithfulness Validation.

Compares attention-based token importance with gradient-based attribution
(Integrated Gradients via Captum) to assess whether attention patterns
faithfully reflect what actually drives the model's output.

Integrated Gradients (Sundararajan et al., 2017)
-------------------------------------------------
Uses Captum's LayerIntegratedGradients to compute per-token attributions
at the word-embedding layer.  Baseline = PAD-token embeddings (semantic
"no information" reference).

Target scalar F = L2 norm of mean-pooled final hidden state.

Faithfulness metric
-------------------
For each head (l, h):
    attn_importance[j] = mean_i(attention[l,h,i,j])  (column-mean)
    ig_importance[j]   = |IG(x)_j|
    ρ(l,h) = Spearman(attn_importance, ig_importance)

High positive ρ → attention faithfully reflects gradient importance.
Low / negative ρ → attention is not a reliable proxy for this head.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.stats import spearmanr
from captum.attr import LayerIntegratedGradients


@dataclass
class IGCorrelationResult:
    """Correlation between attention and IG attribution for one head.

    Attributes
    ----------
    layer : int
    head : int
    spearman_rho : float
        Spearman rank correlation between attention-based and IG-based
        token importance.  Range [-1, 1].
    spearman_pvalue : float
        Two-sided p-value for the Spearman test.
    bar_original : float
        BAR value for this head (for cross-referencing with bias analysis).
    """
    layer: int
    head: int
    spearman_rho: float
    spearman_pvalue: float
    bar_original: float


@dataclass
class IGAnalysisBundle:
    """Complete IG analysis results — correlations + raw attributions.

    Attributes
    ----------
    correlations : list[IGCorrelationResult]
        Per-head Spearman correlations, sorted by |ρ| descending.
    token_attributions : np.ndarray
        Absolute IG attribution per token, shape [seq_len].
    tokens : list[str]
        Tokenized text (sub-word tokens).
    """
    correlations: List[IGCorrelationResult]
    token_attributions: np.ndarray
    tokens: List[str]


def _get_embedding_layer(encoder_model, is_gpt2):
    """Return the word-embedding nn.Embedding layer."""
    if is_gpt2:
        return encoder_model.wte
    return encoder_model.embeddings.word_embeddings


def compute_token_attributions(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
    n_steps: int = 30,
) -> np.ndarray:
    """Compute per-token Integrated Gradients attributions via Captum.

    Uses ``LayerIntegratedGradients`` targeting the word-embedding layer
    with a PAD-token baseline.

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    is_gpt2 : bool
    n_steps : int
        Number of interpolation steps (higher = more accurate, slower).

    Returns
    -------
    np.ndarray of shape [seq_len]
        Absolute IG attribution per token.
    """
    device = next(encoder_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Forward function: input_ids → scalar (L2 norm of mean-pooled hidden state)
    def forward_fn(input_ids, attention_mask):
        outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        return pooled.norm(dim=-1)

    embedding_layer = _get_embedding_layer(encoder_model, is_gpt2)
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    # Baseline: PAD-token ids (semantic "no information" reference)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    baseline_ids = torch.full_like(input_ids, pad_id)

    # Captum handles gradient enablement internally
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )
    # attributions: [1, seq_len, hidden_dim]

    # Sum over embedding dimension → per-token attribution, take absolute value
    token_attrs = attributions.squeeze(0).sum(dim=-1).abs()

    return token_attrs.detach().cpu().numpy()


def compute_ig_attention_correlation(
    attentions: list,
    token_attributions: np.ndarray,
    attention_metrics: list,
) -> List[IGCorrelationResult]:
    """Correlate attention patterns with IG attributions for each head.

    Parameters
    ----------
    attentions : list of torch.Tensor
        Attention weights per layer: [batch, heads, seq, seq].
    token_attributions : np.ndarray
        IG attribution per token, shape [seq_len].
    attention_metrics : list of HeadBiasMetrics
        For cross-referencing BAR values.

    Returns
    -------
    list of IGCorrelationResult
        One per (layer, head), sorted by |spearman_rho| descending.
    """
    # Build BAR lookup
    bar_lookup = {(m.layer, m.head): m.bias_attention_ratio for m in attention_metrics}

    results = []
    ig_imp = token_attributions  # [seq_len]

    for layer_idx, layer_attn in enumerate(attentions):
        num_heads = layer_attn.shape[1]
        for head_idx in range(num_heads):
            # Attention-based token importance: column mean
            attn_matrix = layer_attn[0, head_idx].cpu().numpy()  # [seq, seq]
            attn_imp = attn_matrix.mean(axis=0)  # [seq]

            # Ensure same length (should be, but guard)
            min_len = min(len(attn_imp), len(ig_imp))
            a = attn_imp[:min_len]
            g = ig_imp[:min_len]

            # Spearman rank correlation
            if np.std(a) < 1e-10 or np.std(g) < 1e-10:
                # Constant vector → correlation undefined
                rho, pval = 0.0, 1.0
            else:
                rho, pval = spearmanr(a, g)

            results.append(IGCorrelationResult(
                layer=layer_idx,
                head=head_idx,
                spearman_rho=float(rho) if not np.isnan(rho) else 0.0,
                spearman_pvalue=float(pval) if not np.isnan(pval) else 1.0,
                bar_original=bar_lookup.get((layer_idx, head_idx), 0.0),
            ))

    results.sort(key=lambda r: abs(r.spearman_rho), reverse=True)
    return results


def batch_compute_ig_correlation(
    encoder_model,
    tokenizer,
    text: str,
    attentions: list,
    attention_metrics: list,
    is_gpt2: bool,
    n_steps: int = 30,
) -> IGAnalysisBundle:
    """Full pipeline: compute IG attributions, then correlate with attention.

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    attentions : list of torch.Tensor
        Raw attention weights from the encoder.
    attention_metrics : list of HeadBiasMetrics
        For BAR cross-referencing.
    is_gpt2 : bool
    n_steps : int

    Returns
    -------
    IGAnalysisBundle
        Correlations sorted by |ρ| descending, plus raw token attributions
        and tokenized text.
    """
    token_attributions = compute_token_attributions(
        encoder_model, tokenizer, text, is_gpt2, n_steps=n_steps,
    )
    correlations = compute_ig_attention_correlation(
        attentions, token_attributions, attention_metrics,
    )
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(text, truncation=True, max_length=512)["input_ids"]
    )
    return IGAnalysisBundle(
        correlations=correlations,
        token_attributions=token_attributions,
        tokens=tokens,
    )


__all__ = ["IGCorrelationResult", "IGAnalysisBundle", "batch_compute_ig_correlation"]
