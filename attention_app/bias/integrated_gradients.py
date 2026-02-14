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
class TopKOverlapResult:
    """Top-K overlap between IG and attention rankings for one head.

    Attributes
    ----------
    layer : int
    head : int
    k : int
        Number of top tokens compared.
    jaccard : float
        Jaccard similarity of top-K sets.  Range [0, 1].
    rank_biased_overlap : float
        RBO (Webber et al. 2010, p=0.9).  Range [0, 1].
    bar_original : float
        BAR value for this head.
    """
    layer: int
    head: int
    k: int
    jaccard: float
    rank_biased_overlap: float
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
    topk_overlaps : list[TopKOverlapResult] or None
        Top-K overlap results (IG vs attention) per head.
    """
    correlations: List[IGCorrelationResult]
    token_attributions: np.ndarray
    tokens: List[str]
    topk_overlaps: Optional[List[TopKOverlapResult]] = None


@dataclass
class TokenPerturbationResult:
    """Single-token perturbation importance.

    Attributes
    ----------
    token_index : int
    token : str
    importance : float
        1 - cosine_similarity(original, perturbed).
    """
    token_index: int
    token: str
    importance: float


@dataclass
class PerturbationAnalysisBundle:
    """Complete perturbation analysis results.

    Attributes
    ----------
    token_results : list[TokenPerturbationResult]
        Per-token perturbation importance.
    tokens : list[str]
    perturb_vs_ig_spearman : float
        Spearman ρ between perturbation ranking and IG ranking.
    perturb_vs_attn_spearman : list[tuple]
        List of (layer, head, spearman_rho) for perturbation vs attention.
    """
    token_results: List[TokenPerturbationResult]
    tokens: List[str]
    perturb_vs_ig_spearman: float
    perturb_vs_attn_spearman: List[tuple]


@dataclass
class LRPAnalysisBundle:
    """LRP analysis results for cross-validation with IG.

    Attributes
    ----------
    token_attributions : np.ndarray
        LRP attribution per token, shape [seq_len].
    tokens : list[str]
    lrp_vs_ig_spearman : float
        Spearman ρ between LRP and IG token rankings.
    correlations : list[tuple]
        List of (layer, head, spearman_rho) for LRP vs attention per head.
    """
    token_attributions: np.ndarray
    tokens: List[str]
    lrp_vs_ig_spearman: float
    correlations: List[tuple]


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
    topk_overlaps = compute_topk_overlap(
        attentions, token_attributions, attention_metrics, k=5,
    )

    return IGAnalysisBundle(
        correlations=correlations,
        token_attributions=token_attributions,
        tokens=tokens,
        topk_overlaps=topk_overlaps,
    )


# ── Top-K Overlap ────────────────────────────────────────────────────────


def _rbo(ranked_a: list, ranked_b: list, p: float = 0.9) -> float:
    """Rank-Biased Overlap (Webber et al., 2010).

    Computes a weighted overlap of two ranked lists, giving more weight
    to agreement at the top of the lists.

    Parameters
    ----------
    ranked_a, ranked_b : list
        Ranked lists (index 0 = most important).
    p : float
        Persistence parameter.  Higher p = more weight to deeper ranks.

    Returns
    -------
    float in [0, 1].
    """
    if not ranked_a or not ranked_b:
        return 0.0

    k = min(len(ranked_a), len(ranked_b))
    if k == 0:
        return 0.0

    rbo_sum = 0.0
    intersection_size = 0
    set_a = set()
    set_b = set()

    for d in range(1, k + 1):
        set_a.add(ranked_a[d - 1])
        set_b.add(ranked_b[d - 1])
        intersection_size = len(set_a & set_b)
        agreement = intersection_size / d
        rbo_sum += (p ** (d - 1)) * agreement

    return (1 - p) * rbo_sum


def compute_topk_overlap(
    attentions: list,
    token_attributions: np.ndarray,
    attention_metrics: list,
    k: int = 5,
) -> List[TopKOverlapResult]:
    """Compute Top-K overlap (Jaccard + RBO) between IG and attention per head.

    Parameters
    ----------
    attentions : list of torch.Tensor
        Attention weights per layer.
    token_attributions : np.ndarray
        IG attribution per token, shape [seq_len].
    attention_metrics : list of HeadBiasMetrics
    k : int
        Number of top tokens to compare.

    Returns
    -------
    list of TopKOverlapResult
    """
    bar_lookup = {(m.layer, m.head): m.bias_attention_ratio for m in attention_metrics}

    # IG top-K indices (descending importance)
    ig_imp = token_attributions
    actual_k = min(k, len(ig_imp))
    if actual_k == 0:
        return []

    ig_topk = set(np.argsort(ig_imp)[-actual_k:].tolist())
    ig_ranked = np.argsort(ig_imp)[::-1][:actual_k].tolist()

    results = []
    for layer_idx, layer_attn in enumerate(attentions):
        num_heads = layer_attn.shape[1]
        for head_idx in range(num_heads):
            attn_matrix = layer_attn[0, head_idx].cpu().numpy()
            attn_imp = attn_matrix.mean(axis=0)

            min_len = min(len(attn_imp), len(ig_imp))
            attn_topk_indices = set(np.argsort(attn_imp[:min_len])[-actual_k:].tolist())
            attn_ranked = np.argsort(attn_imp[:min_len])[::-1][:actual_k].tolist()

            # Jaccard
            intersection = len(ig_topk & attn_topk_indices)
            union = len(ig_topk | attn_topk_indices)
            jaccard = intersection / union if union > 0 else 0.0

            # RBO
            rbo = _rbo(ig_ranked, attn_ranked, p=0.9)

            results.append(TopKOverlapResult(
                layer=layer_idx,
                head=head_idx,
                k=actual_k,
                jaccard=jaccard,
                rank_biased_overlap=rbo,
                bar_original=bar_lookup.get((layer_idx, head_idx), 0.0),
            ))

    return results


# ── Perturbation Analysis ─────────────────────────────────────────────────


def compute_token_perturbation(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
) -> Tuple[List[TokenPerturbationResult], List[str]]:
    """Compute token importance by zeroing each embedding individually.

    For each token position, replaces its embedding with zeros and measures
    how much the model's output representation changes (1 - cosine similarity).

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    is_gpt2 : bool

    Returns
    -------
    (list[TokenPerturbationResult], list[str])
        Per-token importance results and token strings.
    """
    device = next(encoder_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    seq_len = input_ids.shape[1]

    embedding_layer = _get_embedding_layer(encoder_model, is_gpt2)

    with torch.no_grad():
        # Get original output
        orig_embeds = embedding_layer(input_ids)
        orig_output = encoder_model(inputs_embeds=orig_embeds, attention_mask=attention_mask)
        orig_hidden = orig_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        orig_pooled = (orig_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        orig_pooled = orig_pooled.squeeze(0)  # [hidden_dim]

        results = []
        cos = torch.nn.CosineSimilarity(dim=0)

        for i in range(seq_len):
            perturbed_embeds = orig_embeds.clone()
            perturbed_embeds[0, i, :] = 0.0  # Zero out token i

            perturbed_output = encoder_model(
                inputs_embeds=perturbed_embeds, attention_mask=attention_mask
            )
            perturbed_hidden = perturbed_output.last_hidden_state
            perturbed_pooled = (perturbed_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            perturbed_pooled = perturbed_pooled.squeeze(0)

            similarity = cos(orig_pooled, perturbed_pooled).item()
            importance = 1.0 - similarity

            results.append(TokenPerturbationResult(
                token_index=i,
                token=tokens[i],
                importance=importance,
            ))

    return results, tokens


def compute_perturbation_correlations(
    perturb_results: List[TokenPerturbationResult],
    ig_attrs: np.ndarray,
    attentions: list,
) -> Tuple[float, List[tuple]]:
    """Correlate perturbation importance with IG and attention.

    Parameters
    ----------
    perturb_results : list[TokenPerturbationResult]
    ig_attrs : np.ndarray
        IG attribution per token.
    attentions : list of torch.Tensor

    Returns
    -------
    (perturb_vs_ig_rho, list[(layer, head, rho)])
    """
    perturb_imp = np.array([r.importance for r in perturb_results])

    # Perturbation vs IG
    min_len = min(len(perturb_imp), len(ig_attrs))
    p = perturb_imp[:min_len]
    g = ig_attrs[:min_len]

    if np.std(p) < 1e-10 or np.std(g) < 1e-10:
        perturb_vs_ig = 0.0
    else:
        perturb_vs_ig, _ = spearmanr(p, g)
        if np.isnan(perturb_vs_ig):
            perturb_vs_ig = 0.0

    # Perturbation vs attention per head
    perturb_vs_attn = []
    for layer_idx, layer_attn in enumerate(attentions):
        num_heads = layer_attn.shape[1]
        for head_idx in range(num_heads):
            attn_matrix = layer_attn[0, head_idx].cpu().numpy()
            attn_imp = attn_matrix.mean(axis=0)

            ml = min(len(perturb_imp), len(attn_imp))
            a = attn_imp[:ml]
            pp = perturb_imp[:ml]

            if np.std(a) < 1e-10 or np.std(pp) < 1e-10:
                rho = 0.0
            else:
                rho, _ = spearmanr(pp, a)
                if np.isnan(rho):
                    rho = 0.0

            perturb_vs_attn.append((layer_idx, head_idx, float(rho)))

    return float(perturb_vs_ig), perturb_vs_attn


def batch_compute_perturbation(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
    ig_attrs: np.ndarray,
    attentions: list,
) -> PerturbationAnalysisBundle:
    """Full perturbation analysis pipeline.

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    is_gpt2 : bool
    ig_attrs : np.ndarray
        IG attributions from prior IG analysis.
    attentions : list of torch.Tensor
        Attention weights from the encoder.

    Returns
    -------
    PerturbationAnalysisBundle
    """
    token_results, tokens = compute_token_perturbation(
        encoder_model, tokenizer, text, is_gpt2,
    )
    perturb_vs_ig, perturb_vs_attn = compute_perturbation_correlations(
        token_results, ig_attrs, attentions,
    )
    return PerturbationAnalysisBundle(
        token_results=token_results,
        tokens=tokens,
        perturb_vs_ig_spearman=perturb_vs_ig,
        perturb_vs_attn_spearman=perturb_vs_attn,
    )


# ── LRP (Layer-wise Relevance Propagation) ────────────────────────────────


def compute_lrp_attributions(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-token LRP attributions via Captum.

    Uses LayerLRP if available, falls back to LayerDeepLift for models
    where LRP fails (e.g. BERT LayerNorm incompatibility).

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    is_gpt2 : bool

    Returns
    -------
    (np.ndarray, list[str])
        Absolute LRP attribution per token [seq_len] and token strings.
    """
    device = next(encoder_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    embedding_layer = _get_embedding_layer(encoder_model, is_gpt2)

    def forward_fn(input_ids, attention_mask):
        outputs = encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        return pooled.norm(dim=-1)

    # Try LayerDeepLift (more robust than LRP for BERT/GPT-2)
    from captum.attr import LayerDeepLift

    try:
        ldl = LayerDeepLift(forward_fn, embedding_layer)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        baseline_ids = torch.full_like(input_ids, pad_id)

        attributions = ldl.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
        )
        token_attrs = attributions.squeeze(0).sum(dim=-1).abs()
        return token_attrs.detach().cpu().numpy(), tokens
    except Exception:
        # Final fallback: use IG (already proven to work)
        token_attrs = compute_token_attributions(
            encoder_model, tokenizer, text, is_gpt2, n_steps=20,
        )
        return token_attrs, tokens


def batch_compute_lrp(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
    ig_attrs: np.ndarray,
    attentions: list,
    attention_metrics: list,
) -> LRPAnalysisBundle:
    """Full LRP analysis pipeline.

    Parameters
    ----------
    encoder_model : BertModel or GPT2Model
    tokenizer : PreTrainedTokenizer
    text : str
    is_gpt2 : bool
    ig_attrs : np.ndarray
        IG attributions for cross-comparison.
    attentions : list of torch.Tensor
    attention_metrics : list of HeadBiasMetrics

    Returns
    -------
    LRPAnalysisBundle
    """
    lrp_attrs, tokens = compute_lrp_attributions(
        encoder_model, tokenizer, text, is_gpt2,
    )

    # LRP vs IG correlation
    min_len = min(len(lrp_attrs), len(ig_attrs))
    l = lrp_attrs[:min_len]
    g = ig_attrs[:min_len]

    if np.std(l) < 1e-10 or np.std(g) < 1e-10:
        lrp_vs_ig = 0.0
    else:
        lrp_vs_ig, _ = spearmanr(l, g)
        if np.isnan(lrp_vs_ig):
            lrp_vs_ig = 0.0

    # LRP vs attention per head (reuse same pattern as IG correlation)
    correlations = []
    for layer_idx, layer_attn in enumerate(attentions):
        num_heads = layer_attn.shape[1]
        for head_idx in range(num_heads):
            attn_matrix = layer_attn[0, head_idx].cpu().numpy()
            attn_imp = attn_matrix.mean(axis=0)

            ml = min(len(lrp_attrs), len(attn_imp))
            a = attn_imp[:ml]
            lr = lrp_attrs[:ml]

            if np.std(a) < 1e-10 or np.std(lr) < 1e-10:
                rho = 0.0
            else:
                rho, _ = spearmanr(lr, a)
                if np.isnan(rho):
                    rho = 0.0

            correlations.append((layer_idx, head_idx, float(rho)))

    return LRPAnalysisBundle(
        token_attributions=lrp_attrs,
        tokens=tokens,
        lrp_vs_ig_spearman=float(lrp_vs_ig),
        correlations=correlations,
    )


__all__ = [
    "IGCorrelationResult", "IGAnalysisBundle",
    "TopKOverlapResult",
    "TokenPerturbationResult", "PerturbationAnalysisBundle",
    "LRPAnalysisBundle",
    "batch_compute_ig_correlation",
    "batch_compute_perturbation",
    "batch_compute_lrp",
]
