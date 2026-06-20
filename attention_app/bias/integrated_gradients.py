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
# Shim: NumPy >=2 removed np.matrix, but older SciPy's spearmanr/corrcoef
# still does isinstance(a, np.matrix). Provide a harmless placeholder.
if not hasattr(np, "matrix"):
    class _NoMatrix(np.ndarray):
        pass
    np.matrix = _NoMatrix  # type: ignore[attr-defined]
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
    # Which attribution method actually produced token_attributions:
    # "AttnLRP" (Achtibat et al. 2024, via lxt), "Chefer-LRP" (Chefer et al.
    # 2021), or "IG-fallback" when both failed. In the fallback case
    # lrp_vs_ig_spearman compares IG with IG and is NOT an independent
    # cross-validation — renderers must surface this.
    method: str = "AttnLRP"
    # Optional alternate-method bundle (e.g. Chefer-LRP when this one is
    # AttnLRP), so the panel can show both ρ(method, IG) and switch charts
    # between the two LRP variants. None when only one method was available.
    alt: Optional["LRPAnalysisBundle"] = None


def _get_embedding_layer(encoder_model, is_gpt2):
    """Return the word-embedding nn.Embedding layer."""
    if is_gpt2:
        return encoder_model.wte
    return encoder_model.embeddings.word_embeddings


def _baseline_token_id(tokenizer) -> int:
    """Neutral baseline token id for IG / DeepLift.

    BERT has a real [PAD] token. GPT-2 has none — ``pad_token_id`` is None,
    and falling through to vocabulary id 0 would make the baseline a
    sequence of ``"!"`` tokens. Use ``<|endoftext|>`` instead: it is the
    closest thing GPT-2 has to a semantic "no information" reference.
    """
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return eos
    return 0


def _attention_token_importance(attn_matrix: np.ndarray) -> np.ndarray:
    """Per-token importance = mean attention RECEIVED by each position.

    For causal matrices (upper triangle structurally zero) position j can
    only receive attention from queries i ≥ j, so the plain column mean
    divides by the full N and systematically deflates later positions —
    a positional artefact, not model behaviour. Divide by the number of
    valid queries instead. Bidirectional matrices keep the plain mean.
    """
    n = attn_matrix.shape[0]
    if n > 1 and float(np.triu(attn_matrix, k=1).sum()) < 1e-6:
        valid_queries = n - np.arange(n)
        return attn_matrix.sum(axis=0) / valid_queries
    return attn_matrix.mean(axis=0)


def compute_token_attributions(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
    n_steps: int = 50,
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

    # Baseline: PAD (BERT) / <|endoftext|> (GPT-2) — see _baseline_token_id
    baseline_ids = torch.full_like(input_ids, _baseline_token_id(tokenizer))

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
            # Attention-based token importance: mean attention received,
            # causal-support-aware (see _attention_token_importance)
            attn_matrix = layer_attn[0, head_idx].cpu().numpy()  # [seq, seq]
            attn_imp = _attention_token_importance(attn_matrix)  # [seq]

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
            attn_imp = _attention_token_importance(attn_matrix)

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
            attn_imp = _attention_token_importance(attn_matrix)

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


def _attnlrp_attributions(encoder_model, tokenizer, text, is_gpt2):
    """AttnLRP relevance (Achtibat et al., 2024) via the ``lxt`` library.

    AttnLRP is a conservation-preserving layer-wise relevance propagation with
    rules tailored to the transformer non-linearities (softmax, matmul, layer
    normalisation), and is more rigorous than the attention-rollout relevance
    of Chefer et al. (2021). The ``lxt`` implementation works by monkey-patching
    the Hugging Face modeling module so that the backward pass redistributes
    relevance; the forward pass is numerically unchanged.

    Because the patch is global (it rewrites methods on the shared modeling
    classes) and only alters the backward, leaving it in place would corrupt the
    gradients of every other diagnostic (integrated gradients, the Chefer
    cross-check). We therefore snapshot the patched classes/module, patch,
    compute, and restore the snapshot, so the rest of the dashboard is
    unaffected. Returns ``(abs attribution per token, tokens)`` or ``None`` if
    ``lxt`` is unavailable or the computation fails (the caller then falls back
    to Chefer).
    """
    try:
        from lxt.efficient import monkey_patch
        if is_gpt2:
            import lxt.efficient.models.gpt2 as _lm
            from transformers.models.gpt2 import modeling_gpt2 as _mod
        else:
            import lxt.efficient.models.bert as _lm
            from transformers.models.bert import modeling_bert as _mod
        patch_map = _lm.attnLRP
    except Exception:
        return None  # lxt not installed / incompatible — caller uses Chefer

    targets = list(patch_map.keys())
    snapshot = {t: dict(vars(t)) for t in targets}  # pristine class/module state
    try:
        device = next(encoder_model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        encoder_model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            monkey_patch(_mod, patch_map, verbose=False)
            emb = encoder_model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
            outputs = encoder_model(inputs_embeds=emb, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            pooled.norm(dim=-1).sum().backward()

        rel = (emb.grad * emb).sum(dim=-1)[0].abs().detach().cpu().numpy()
        if rel.size == 0 or not np.isfinite(rel).all() or float(rel.std()) < 1e-12:
            return None
        return rel, tokens
    except Exception:
        return None
    finally:
        # Restore the pristine modeling state so no other diagnostic is affected.
        for t, original in snapshot.items():
            for k in list(vars(t).keys()):
                if k not in original:
                    try:
                        delattr(t, k)
                    except Exception:
                        pass
            for k, v in original.items():
                try:
                    setattr(t, k, v)
                except Exception:
                    pass
        encoder_model.zero_grad(set_to_none=True)


def compute_lrp_attributions(
    encoder_model,
    tokenizer,
    text: str,
    is_gpt2: bool,
    force: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, List[str], str]]:
    """Compute per-token transformer-LRP attributions for the IG cross-check.

    Three methods are tried in order of rigour, and the one that succeeds is
    reported so the interface can label it:

    1. ``"AttnLRP"`` — the conservation-preserving Attention-Aware LRP of
       \\citet{Achtibat2024} via the ``lxt`` library, when it is installed and
       compatible with the loaded Transformers version.
    2. ``"Chefer-LRP"`` — the attention-rollout relevance of Chefer, Gur & Wolf
       (CVPR 2021): for every layer take the head-averaged positive part of
       (attention ⊙ ∂target/∂attention) and roll it through the layers with a
       residual, ``R₀ = I, Āₗ = mean_h((Aₗ ⊙ ∇Aₗ)⁺), Rₗ = Rₗ₋₁ + Āₗ·Rₗ₋₁``.
       Self-contained, so it always runs on BERT/GPT-2.
    3. ``"IG-fallback"`` — re-uses integrated gradients if both relevance
       methods fail. The caller MUST surface this, because the agreement is
       then IG-vs-IG and proves nothing.

    All three attribute the same masked-mean-pooled hidden-state norm that IG
    explains, so the cross-validation compares like with like.

    Returns ``(abs attribution per token [seq_len], token strings, method)``.
    """
    device = next(encoder_model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 1) AttnLRP (most rigorous) when lxt is available, unless Chefer is forced.
    if force != "chefer":
        _attn = _attnlrp_attributions(encoder_model, tokenizer, text, is_gpt2)
        if _attn is not None:
            return _attn[0], _attn[1], "AttnLRP"
        if force == "attnlrp":
            return None  # AttnLRP explicitly requested but unavailable

    # 2) Chefer transformer-LRP (always available).
    try:
        encoder_model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = encoder_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            attns = outputs.attentions  # tuple of L tensors [1, H, N, N]
            if not attns:
                raise RuntimeError("model returned no attentions")
            for a in attns:
                a.retain_grad()
            hidden = outputs.last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            target = pooled.norm(dim=-1).sum()
            target.backward()

        N = input_ids.shape[1]
        R = torch.eye(N, device=device)
        used = 0
        for a in attns:
            g = a.grad
            if g is None:
                continue
            # head-averaged positive relevance for this layer  → [N, N]
            cam = (a * g).clamp(min=0).mean(dim=1)[0]
            R = R + cam @ R          # Chefer rollout with residual
            used += 1
        if used == 0:
            raise RuntimeError("no attention gradients captured")

        # Per-token relevance: mean relevance flowing into each key token.
        rel = R.mean(dim=0).abs().detach().cpu().numpy()
        encoder_model.zero_grad(set_to_none=True)
        if rel.size == 0 or not np.isfinite(rel).all() or float(rel.std()) < 1e-12:
            raise RuntimeError("degenerate relevance")
        return rel, tokens, "Chefer-LRP"
    except Exception:
        # Honest fallback: reuse IG. The caller MUST surface this — the
        # "LRP vs IG" agreement is then IG-vs-IG and proves nothing.
        encoder_model.zero_grad(set_to_none=True)
        token_attrs = compute_token_attributions(
            encoder_model, tokenizer, text, is_gpt2, n_steps=20,
        )
        return token_attrs, tokens, "IG-fallback"


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
    def _build(lrp_attrs, tokens, method):
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
                attn_matrix = layer_attn[0, head_idx].detach().cpu().numpy()
                attn_imp = _attention_token_importance(attn_matrix)
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
            method=method,
        )

    # Compute AttnLRP (preferred) and Chefer separately, so the panel can show
    # both ρ(method, IG) and let the user check whether the two LRP variants
    # agree with IG. AttnLRP returns None when lxt is unavailable; Chefer always
    # produces a result (or the IG fallback as the last resort).
    attn = compute_lrp_attributions(encoder_model, tokenizer, text, is_gpt2, force="attnlrp")
    chef = compute_lrp_attributions(encoder_model, tokenizer, text, is_gpt2, force="chefer")

    bundles = []
    if attn is not None:
        bundles.append(_build(attn[0], attn[1], attn[2]))
    if chef is not None:
        bundles.append(_build(chef[0], chef[1], chef[2]))

    if not bundles:
        d = compute_lrp_attributions(encoder_model, tokenizer, text, is_gpt2)
        return _build(d[0], d[1], d[2])

    primary = bundles[0]
    if len(bundles) > 1:
        primary.alt = bundles[1]
    return primary


__all__ = [
    "IGCorrelationResult", "IGAnalysisBundle",
    "TopKOverlapResult",
    "TokenPerturbationResult", "PerturbationAnalysisBundle",
    "LRPAnalysisBundle",
    "batch_compute_ig_correlation",
    "batch_compute_perturbation",
    "batch_compute_lrp",
]
