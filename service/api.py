"""JSON HTTP API in front of the Attention Atlas Shiny app.

Purely additive: the Shiny application is mounted unchanged at ``/`` and keeps
serving the full dashboard, while ``/api/*`` exposes the same computation
pipeline as JSON for an external frontend.

Run with::

    uvicorn api:api --port 8000

Design notes:

* The analysis itself is :func:`attention_app.server.logic.heavy_compute` -
  the exact function the Shiny server uses - and the metrics are
  :func:`attention_app.metrics.compute_all_attention_metrics`, called the
  same way the Shiny renderer calls it, so the numbers served here are the
  numbers the dashboard displays. Nothing is reimplemented.
* ``ComputeResult`` is never serialised wholesale: its ``tokenizer`` /
  ``encoder_model`` / ``mlm_model`` attributes are properties resolving real
  models through ModelManager. Fields go through
  :mod:`attention_app.serialize` one at a time.
* The Shiny mount is registered LAST. A mount at ``/`` matches every path, so
  anything registered after it would be unreachable.
"""

import hmac
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import serialize as ser
from attention_app.metrics import calculate_flow_change, compute_all_attention_metrics
from attention_app.models import ModelManager
from attention_app.server.logic import heavy_compute, tokenize_with_segments

# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------

# heavy_compute truncates at 512 tokens, but the metrics grid is
# n_layers x n_heads (144 for BERT base) full passes over a seq x seq matrix.
# At 512 tokens that is 144 x 512^2 cells per request, far too slow on a free
# CPU Space, so the API enforces a much tighter ceiling of its own.
MAX_TOKENS = 100

# Cheap pre-filter so a multi-megabyte body is rejected before any tokenizer
# runs. Deliberately loose - the real limit is MAX_TOKENS, checked after
# tokenization (which is fast; only the forward pass is not).
MAX_CHARS = 4000

_logger = logging.getLogger(__name__)

# ComputeResult holds the attention and hidden-state tensors for the whole
# forward pass, so the cache stays small on purpose.
CACHE_SIZE = 12

DEFAULT_MODEL = "bert-base-uncased"


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL
    layer: Optional[int] = Field(default=None, ge=0)
    head: Optional[int] = Field(default=None, ge=0)
    include_clusters: bool = False
    include_isa: bool = False


# --------------------------------------------------------------------------
# Compute + cache
# --------------------------------------------------------------------------

@lru_cache(maxsize=CACHE_SIZE)
def _cached_compute(text: str, model_name: str, with_clusters: bool, with_isa: bool):
    """Memoised ``heavy_compute``.

    Keyed on the analysis inputs only - ``layer`` / ``head`` merely index into
    the result - so clicking through the heads of one sentence reuses a single
    forward pass. The optional-stage flags are part of the key because they
    change what the result contains; the common case (both ``False``) collapses
    to a plain ``(text, model_name)`` key.
    """
    return heavy_compute(
        text,
        model_name,
        # Clusters are derived from head specialization, so it is only needed
        # when clusters are requested.
        with_specialization=with_clusters,
        with_clusters=with_clusters,
        with_isa=with_isa,
    )


def _known_models() -> List[str]:
    """Model ids ModelManager will accept, in a stable order."""
    return sorted(ModelManager._ALLOWED_MODELS)


def _display_name(model_id: str) -> str:
    """Human-readable label derived from the model id."""
    name = model_id.split("/")[-1]
    parts = name.replace("_", "-").split("-")
    pretty = {
        "bert": "BERT", "gpt2": "GPT-2", "gus": "GUS", "net": "Net",
        "uncased": "(Uncased)", "multilingual": "Multilingual",
        "xl": "XL", "base": "Base", "large": "Large", "medium": "Medium",
        "custom": "Custom",
    }
    return " ".join(pretty.get(p, p.capitalize()) for p in parts)


def _validate_request(payload: AnalyzeRequest) -> str:
    """Validate the request and return the token count. Raises HTTP 400."""
    text = payload.text.strip() if payload.text else ""
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(f"`text` is {len(text)} characters; the limit is {MAX_CHARS}. "
                    f"Send at most ~{MAX_TOKENS} tokens."),
        )
    if payload.model not in ModelManager._ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{payload.model}'. See GET /api/models.",
        )
    return text


#: Markers of a load that ran out of device memory. "meta tensor" is included
#: because that is how the shortage actually surfaces: transformers initialises
#: on the meta device, cannot materialise the weights when the device is full,
#: and the error only appears later at ``.to(device)``.
_DEVICE_MEMORY_MARKERS = (
    "meta tensor",
    "to_empty(",
    "out of memory",
    "outofmemoryerror",
    "cannot allocate",
)


def _looks_like_device_memory(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _DEVICE_MEMORY_MARKERS)


def _check_token_budget(text: str, model_name: str) -> None:
    """Reject over-long inputs before paying for the forward pass.

    Tokenization is cheap; the forward pass and the n_layers x n_heads metrics
    grid are not, so the ceiling is enforced here rather than after the fact.
    """
    try:
        tokenizer, _, _ = ModelManager.get_model(model_name)
    except Exception as exc:  # unavailable weights, no network, out of memory
        # A device that has filled up with previously cached models cannot
        # materialise the next one, and transformers reports that as a meta
        # tensor rather than as the out-of-memory it is. Free everything and
        # give it one clean attempt before giving up, so a session does not
        # have to be restarted to change model.
        if _looks_like_device_memory(exc):
            _logger.warning(
                "Load of %s failed on a full device; clearing cached models "
                "and retrying once.", model_name,
            )
            ModelManager.free_all()
            try:
                tokenizer, _, _ = ModelManager.get_model(model_name)
            except Exception as retry_exc:
                raise HTTPException(
                    status_code=503, detail=f"Could not load model: {retry_exc}",
                ) from retry_exc
        else:
            raise HTTPException(
                status_code=503, detail=f"Could not load model: {exc}",
            ) from exc

    n_tokens = int(tokenize_with_segments(text, tokenizer)["input_ids"].shape[1])
    if n_tokens > MAX_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input is {n_tokens} tokens; the API limit is {MAX_TOKENS}. "
                "Every request computes metrics for all layers x heads over a "
                "seq x seq attention matrix, which grows quadratically with "
                "sequence length. Send a shorter text."
            ),
        )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

# The interactive documentation is switched off deliberately. FastAPI would
# otherwise publish /docs, /redoc and /openapi.json, which advertise every
# route to anyone who opens the Space and make this look like a documented
# public interface. It is complementary tooling for one frontend, not a
# product surface, so the routes stay unlisted. They keep working exactly as
# before; only the generated documentation is gone.
api = FastAPI(
    title="Attention Atlas",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# localhost for local development; the deployed frontend origin is configured
# through FRONTEND_ORIGIN rather than hardcoded here.
_allowed_origins = ["http://localhost:3000"]
_frontend_origin = os.environ.get("FRONTEND_ORIGIN", "").strip()
if _frontend_origin and _frontend_origin not in _allowed_origins:
    _allowed_origins.append(_frontend_origin)

api.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@api.get("/api/health")
def health() -> Dict[str, Any]:
    """Liveness probe plus which models are currently resident."""
    return {
        "status": "ok",
        "models_loaded": list(ModelManager._instances.keys()),
        "device": ModelManager.get_device(),
    }


@api.get("/api/models")
def models() -> Dict[str, Any]:
    """Models ModelManager accepts, with display names."""
    return {
        "models": [
            {
                "id": model_id,
                "name": _display_name(model_id),
                "family": "gpt2" if "gpt2" in model_id else "bert",
            }
            for model_id in _known_models()
        ],
        "default": DEFAULT_MODEL,
    }


@api.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Run the attention pipeline over ``text`` and return JSON.

    ``metrics`` is a ``[layer][head]`` grid straight from
    ``compute_all_attention_metrics``, called exactly as the Shiny renderer
    calls it (``has_cls`` only, so ``causal`` takes its documented default of
    ``not has_cls``) - the values match the dashboard for the same input.
    """
    text = _validate_request(payload)
    _check_token_budget(text, payload.model)

    try:
        result = _cached_compute(
            text, payload.model, payload.include_clusters, payload.include_isa
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if result is None:
        raise HTTPException(status_code=400, detail="`text` produced no tokens.")

    tokens = result.tokens
    # Same derivation the Shiny renderer uses: a [CLS] summary token means a
    # BERT-style bidirectional encoder. The "G-with-dot" marker is the
    # tokenizer-level signal for a GPT-2-style causal model; the absence of
    # [CLS] covers single-token inputs, where no marker appears.
    has_cls = "[CLS]" in tokens
    is_causal = any("Ġ" in t for t in tokens) or not has_cls

    # (n_layers, n_heads, seq, seq): batch dimension dropped, as in the app.
    att_layers = [layer[0].detach().cpu().numpy() for layer in result.attentions]
    n_layers = len(att_layers)
    n_heads = int(att_layers[0].shape[0]) if n_layers else 0

    layer_idx, head_idx = payload.layer, payload.head
    if layer_idx is not None and layer_idx >= n_layers:
        raise HTTPException(
            status_code=400,
            detail=f"layer {layer_idx} out of range; this model has {n_layers} layers.",
        )
    if head_idx is not None and head_idx >= n_heads:
        raise HTTPException(
            status_code=400,
            detail=f"head {head_idx} out of range; this model has {n_heads} heads.",
        )

    grid = [
        [compute_all_attention_metrics(att_layers[l][h], has_cls=has_cls)
         for h in range(n_heads)]
        for l in range(n_layers)
    ]

    response: Dict[str, Any] = {
        "tokens": ser.serialize_tokens(tokens),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "has_cls": has_cls,
        "is_causal": is_causal,
        "model": payload.model,
        "metrics": ser.serialize_metrics_grid(grid),
        "flow_change": ser.to_float(calculate_flow_change(att_layers)),
    }

    # The full matrix is the bulk of the payload, so it ships only when a
    # specific head was asked for.
    if layer_idx is not None and head_idx is not None:
        response["attention"] = ser.serialize_matrix(att_layers[layer_idx][head_idx])
        response["layer"] = layer_idx
        response["head"] = head_idx

    segments = ser.serialize_segments(result.inputs)
    if segments is not None:
        response["segments"] = segments

    if payload.include_clusters:
        response["clusters"] = ser.serialize_clusters(result.head_clusters)
    if payload.include_isa:
        response["isa"] = ser.serialize_isa(result.isa_data)

    return response


# --------------------------------------------------------------------------
# Bias
# --------------------------------------------------------------------------

# Base model -> the GUS-Net backbone fine-tuned from it. The inverse of
# attention_app/server/bias_helpers._GUSNET_TO_BASE, restricted to pairs that
# share a tokenizer vocabulary.
#
# The restriction is the whole point. GUS-Net labels tokens by INDEX, and those
# indices are only meaningful against the same token sequence the attention was
# computed over. Every GPT-2 size shares one 50257 BPE vocab and both BERT
# uncased sizes share one 30522 WordPiece vocab, so those pairs align.
# bert-base-multilingual-uncased has a different vocabulary entirely and is
# deliberately absent: pairing it would silently map bias labels onto the wrong
# tokens, which is worse than having no bias view at all.
BASE_TO_GUSNET = {
    "bert-base-uncased": "gusnet-bert",
    "bert-large-uncased": "gusnet-bert-large",
    "gpt2": "gusnet-gpt2",
    "gpt2-medium": "gusnet-gpt2-medium",
    "gpt2-large": "gusnet-gpt2",
    "openai-community/gpt2-xl": "gusnet-gpt2",
    "pinthoz/gus-net-bert": "gusnet-bert",
    "pinthoz/gus-net-bert-large": "gusnet-bert-large",
    "pinthoz/gus-net-bert-custom": "gusnet-bert-custom",
    "pinthoz/gus-net-gpt2": "gusnet-gpt2",
    "pinthoz/gus-net-gpt2-medium": "gusnet-gpt2-medium",
}


class BiasRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL


def _plain_attention(text: str, model_name: str):
    """Attention over a PLAIN tokenization of ``text``.

    Deliberately not ``heavy_compute``: that path runs ``tokenize_with_segments``,
    which encodes a multi-sentence BERT input as a sentence PAIR and so emits an
    extra ``[SEP]``. GUS-Net tokenizes the text plainly, so reusing the paired
    sequence would shift every bias index by one from the second sentence
    onwards. Both sides tokenize the same way here, and the caller still
    verifies the two sequences match before trusting any index.
    """
    import torch

    tokenizer, encoder_model, _ = ModelManager.get_model(model_name)
    device = ModelManager.get_device()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
    return tokens, list(outputs.attentions)


@lru_cache(maxsize=CACHE_SIZE)
def _cached_bias(text: str, model_name: str):
    """Memoised bias analysis, keyed exactly like ``_cached_compute``."""
    from attention_app.bias import AttentionBiasAnalyzer, GusNetDetector

    gusnet_key = BASE_TO_GUSNET[model_name]

    tokens, attentions = _plain_attention(text, model_name)

    detector = GusNetDetector(model_key=gusnet_key, threshold=0.5, use_optimized=True)
    token_labels = detector.detect_bias(text)

    # The alignment check the whole endpoint rests on. The map above should
    # guarantee it, but a wrong pairing would produce plausible-looking numbers
    # attached to the wrong words - the exact failure a reader could not spot -
    # so it is verified rather than assumed.
    gus_tokens = [entry["token"] for entry in token_labels]
    if gus_tokens != list(tokens):
        raise HTTPException(
            status_code=409,
            detail=(
                f"The bias detector and '{model_name}' tokenized this text "
                f"differently ({len(gus_tokens)} vs {len(tokens)} tokens), so "
                "bias labels cannot be matched to attention. This is a model "
                "pairing problem, not a problem with the text."
            ),
        )

    biased_indices = [e["index"] for e in token_labels if e.get("is_biased")]

    analyzer = AttentionBiasAnalyzer()
    head_metrics = analyzer.analyze_attention_to_bias(
        attentions, biased_indices, list(tokens)
    )
    propagation = analyzer.analyze_bias_propagation(
        attentions, biased_indices, list(tokens), precomputed_metrics=head_metrics
    )

    n_layers = len(attentions)
    n_heads = int(attentions[0].shape[1]) if n_layers else 0

    # Flat list -> [layer][head], matching the shape /api/analyze uses for its
    # own metrics so the frontend can index both grids identically.
    grid = [[None] * n_heads for _ in range(n_layers)]
    for m in head_metrics:
        grid[m.layer][m.head] = {
            "bias_attention_ratio": ser.to_float(m.bias_attention_ratio),
            "amplification_score": ser.to_float(m.amplification_score),
            "max_bias_attention": ser.to_float(m.max_bias_attention),
            "specialized_for_bias": bool(m.specialized_for_bias),
        }

    return {
        "tokens": ser.serialize_tokens(tokens),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "model": model_name,
        "bias_model": gusnet_key,
        "bias_model_name": detector.config.get("display_name", gusnet_key),
        "tokens_biased": biased_indices,
        "token_labels": ser.to_builtin(token_labels),
        "summary": ser.to_builtin(detector.get_bias_summary(token_labels)),
        # Empty when nothing was flagged: with no biased tokens there is no
        # ratio to compute, and the analyzer returns [] rather than zeros.
        "metrics": grid if biased_indices else [],
        "propagation": ser.to_builtin(propagation),
        # BAR > this counts as specialisation (empirical alpha=0.05).
        "bar_threshold": 2.5,
    }


@api.post("/api/bias")
def bias(payload: BiasRequest) -> Dict[str, Any]:
    """Token-level bias from GUS-Net, crossed with each head's attention.

    Separate from ``/api/analyze`` on purpose: this runs a second forward pass
    through a different model, and the attention view should not have to wait
    for it.
    """
    text = _validate_request(payload)  # type: ignore[arg-type]

    # Before _check_token_budget, which loads the model to tokenize. Pairing is
    # a static fact about the model, so an unsupported one should say so
    # immediately rather than after paying to download weights it cannot use.
    if payload.model not in BASE_TO_GUSNET:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No GUS-Net model shares a tokenizer with '{payload.model}', so "
                "bias labels could not be lined up with its tokens. Try "
                "bert-base-uncased or gpt2."
            ),
        )

    _check_token_budget(text, payload.model)

    try:
        return _cached_bias(text, payload.model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bias analysis failed: {exc}") from exc


# --------------------------------------------------------------------------
# Faithfulness
# --------------------------------------------------------------------------

# Ablating a head costs one forward pass, so the count is capped. The heads
# worth testing are the ones that attend to the flagged tokens most; past the
# first handful the BAR values are indistinguishable from each other anyway.
DEFAULT_TOP_K = 8
MAX_TOP_K = 16


class FaithfulnessRequest(BaseModel):
    text: str
    model: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


def _ig_correlations(text, model_name, ranked, is_gpt2, tokenizer, encoder_model):
    """Correlate each head's attention with Integrated Gradients attribution.

    A second, independent faithfulness signal: ablation asks whether removing a
    head changes the output, this asks whether the tokens a head attends to are
    the tokens the gradient says the decision rests on.

    ATTRIBUTION TARGET. Gradients only mean something relative to a decision.
    IG can attribute the GUS-Net bias evidence, but only when the attention
    maps come from the GUS-Net trunk itself - that is, when the analysed model
    IS a ``pinthoz/gus-net-*`` id. For a plain encoder the attentions belong to
    the pretrained model while the gradients would flow through the fine-tuned
    one, which is a model mismatch, so the target falls back to the pooled norm
    and the response says which was used. Pooled-norm correlations do NOT
    validate the bias explanations, and the panel has to say so.

    Step count follows the same rule the dashboard uses: the bias-evidence
    target is a sum of sigmoids that saturates, and its integral converges
    slowly (residual ~68% at 30 steps against ~3% at 64).
    """
    from attention_app.bias import batch_compute_ig_correlation
    from attention_app.bias.gusnet_detector import GusNetDetector, MODEL_REGISTRY

    target_model = None
    if "gus-net" in model_name:
        key = BASE_TO_GUSNET.get(model_name)
        if key in MODEL_REGISTRY:
            try:
                _tok, target_model = GusNetDetector._load_model(
                    key, ModelManager.get_device(),
                )
            except Exception:
                target_model = None

    _tokens, attentions = _plain_attention(text, model_name)

    try:
        bundle = batch_compute_ig_correlation(
            encoder_model,
            tokenizer,
            text,
            attentions,
            ranked,
            is_gpt2,
            n_steps=64 if target_model is not None else 30,
            target_model=target_model,
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}

    overlaps = {
        (o.layer, o.head): o for o in (bundle.topk_overlaps or [])
    }

    return {
        "available": True,
        # "gusnet-bias-logits" or "pooled-norm". The renderer must surface it.
        "target": bundle.target,
        # Relative error of the IG path integral. Above ~0.05 the attributions
        # have not converged and every correlation below is approximate.
        "convergence_delta": ser.to_float(bundle.convergence_delta),
        "token_attributions": ser.to_builtin(bundle.token_attributions),
        "tokens": ser.serialize_tokens(bundle.tokens),
        "correlations": [
            {
                "layer": c.layer,
                "head": c.head,
                "spearman_rho": ser.to_float(c.spearman_rho),
                # Raw p is NOT interpretable on its own: one test per head over
                # ~144 heads yields ~7 hits below 0.05 by chance. The q value is
                # the Benjamini-Hochberg FDR adjustment and the one any
                # significance claim must use.
                "spearman_pvalue": ser.to_float(c.spearman_pvalue),
                "spearman_qvalue": ser.to_float(c.spearman_qvalue),
                "bias_attention_ratio": ser.to_float(c.bar_original),
                "jaccard": ser.to_float(
                    getattr(overlaps.get((c.layer, c.head)), "jaccard", None)
                ),
                "rank_biased_overlap": ser.to_float(
                    getattr(
                        overlaps.get((c.layer, c.head)), "rank_biased_overlap", None
                    )
                ),
            }
            for c in bundle.correlations[:24]
        ],
    }


@lru_cache(maxsize=CACHE_SIZE)
def _cached_faithfulness(text: str, model_name: str, top_k: int):
    """Ablate the most bias-focused heads and measure what actually changes.

    Attention says where a head looks; this says whether the model's output
    depends on it. A head can attend heavily to the flagged words and still
    matter not at all, which is the single most important caveat about reading
    attention as explanation - so the two numbers are returned together.
    """
    from types import SimpleNamespace

    from attention_app.bias import batch_ablate_top_heads

    bias = _cached_bias(text, model_name)
    grid = bias.get("metrics") or []
    if not grid:
        return {
            "model": model_name,
            "ablation_mode": "zero",
            "heads": [],
            "reason": "no_flagged_tokens",
        }

    # Rank by BAR, then ablate the top of that list.
    ranked = []
    for layer_idx, row in enumerate(grid):
        for head_idx, cell in enumerate(row):
            bar = (cell or {}).get("bias_attention_ratio")
            if bar is None:
                continue
            ranked.append(SimpleNamespace(
                layer=layer_idx, head=head_idx, bias_attention_ratio=float(bar),
            ))
    ranked.sort(key=lambda m: m.bias_attention_ratio, reverse=True)
    top_heads = ranked[:top_k]
    if not top_heads:
        return {
            "model": model_name,
            "ablation_mode": "zero",
            "heads": [],
            "reason": "no_ratios",
        }

    tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
    is_gpt2 = "gpt2" in model_name.lower()

    results = batch_ablate_top_heads(
        encoder_model, mlm_model, tokenizer, text, top_heads, is_gpt2, mode="zero",
    )

    ig = _ig_correlations(text, model_name, ranked, is_gpt2, tokenizer, encoder_model)

    return {
        "model": model_name,
        "ig": ig,
        # Recorded because it changes how the numbers should be read: zeroing a
        # head pushes activations off the manifold the model was trained on,
        # which tends to OVERSTATE impact. Good for ranking heads against each
        # other, not for absolute claims.
        "ablation_mode": "zero",
        "heads": [
            {
                "layer": r.layer,
                "head": r.head,
                "representation_impact": ser.to_float(r.representation_impact),
                "kl_divergence": ser.to_float(r.kl_divergence),
                "bias_attention_ratio": ser.to_float(r.bar_original),
            }
            for r in results
        ],
        "reason": None,
    }


@api.post("/api/faithfulness")
def faithfulness(payload: FaithfulnessRequest) -> Dict[str, Any]:
    """Does the model's output actually depend on its most bias-focused heads?"""
    text = _validate_request(payload)  # type: ignore[arg-type]

    if payload.model not in BASE_TO_GUSNET:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Bias analysis is unavailable for '{payload.model}', so there "
                "are no heads to test. Try bert-base-uncased or gpt2."
            ),
        )

    _check_token_budget(text, payload.model)

    try:
        return _cached_faithfulness(text, payload.model, payload.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Faithfulness analysis failed: {exc}",
        ) from exc


@api.post("/api/session-recorded")
async def session_recorded(request: Request) -> Dict[str, Any]:
    """Hugging Face webhook: the study's log dataset changed.

    Fans the event out to Discord and email so the team knows the session was
    saved. The notification carries no participant code - see
    ``service.session_notify``.
    """
    from .session_notify import handle_event, notify_enabled

    if not notify_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    secret = (os.environ.get("ATLAS_NOTIFY_SECRET") or "").strip()
    if not hmac.compare_digest(request.headers.get("x-webhook-secret", ""), secret):
        raise HTTPException(status_code=401, detail="Bad webhook secret")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return handle_event(payload)


# --------------------------------------------------------------------------
# Shiny mount - MUST stay last: a mount at "/" matches every path, so any
# route registered after it would never be reached.
# --------------------------------------------------------------------------

from attention_app.app import app as shiny_app  # noqa: E402

api.mount("/", shiny_app)

__all__ = ["api"]
