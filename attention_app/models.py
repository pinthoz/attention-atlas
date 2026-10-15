import logging
import os
import threading
import warnings
import torch
from transformers import (
    BertTokenizerFast,
    BertModel,
    BertForMaskedLM,
    BertForSequenceClassification,
)
from transformers.utils import logging as transformers_logging
from transformers import GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
import gc

_logger = logging.getLogger(__name__)


def _eager_kwargs():
    """Force eager attention so ``output_attentions=True`` returns the maps.

    From transformers 4.36 the default attention is SDPA, which silently
    drops the attention weights when ``output_attentions=True`` (and on 5.x
    raises). The whole dashboard is built on those weights, so we request the
    eager implementation. The kwarg does not exist before 4.36, hence the
    version guard (keeps the code working if transformers is downgraded)."""
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse("4.36"):
            return {"attn_implementation": "eager"}
    except Exception:
        pass
    return {}


def _release_to_cpu(entry):
    """Move an evicted entry's weights off the accelerator before dropping it.

    Dropping the cache's reference does NOT free device memory on its own: a
    caller that fetched the tuple earlier, or a cached result built from it,
    can still be holding the same modules alive. Moving them to CPU releases
    the VRAM whoever else is holding on.

    This matters more than it looks. When device memory runs out mid-load,
    transformers cannot materialise the weights it initialised on the meta
    device, and the failure surfaces from the later ``.to(device)`` as

        Cannot copy out of meta tensor; no data!

    which reads like a model or version problem rather than the plain
    out-of-memory it is.
    """
    if not entry:
        return
    for obj in entry:
        # Tokenizers have no .parameters(); only move real modules.
        if hasattr(obj, "parameters") and hasattr(obj, "to"):
            try:
                obj.to("cpu")
            except Exception:
                _logger.debug("Could not move evicted module to CPU", exc_info=True)


# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


# Rerankers offered in the "Rerankers" model family. Single source of truth:
# the allowlist, the cross-encoder test and the sidebar all read from here.
RERANKER_CHOICES = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "MS MARCO MiniLM L-6",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "MS MARCO MiniLM L-12",
}
RERANKER_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ModelManager:
    """Manages loading and caching of BERT / GPT-2 models.

    Thread-safe: a ``threading.Lock`` serialises cache reads and writes so
    two Shiny sessions loading the same model concurrently won't corrupt
    the shared ``_instances`` dict.
    """

    _ALLOWED_MODELS = {
        # Base encoder models
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-multilingual-uncased",
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "openai-community/gpt2-xl",
        # GUS-Net fine-tuned models (HuggingFace)
        "pinthoz/gus-net-bert",
        "pinthoz/gus-net-bert-large",
        "pinthoz/gus-net-bert-custom",
        "pinthoz/gus-net-gpt2",
        "pinthoz/gus-net-gpt2-medium",
        # Cross-encoder rerankers (BERT trunk, single relevance logit)
        *RERANKER_CHOICES,
    }

    # Rerankers scored as a query/passage pair. These carry a sequence
    # classification head with one output (the relevance logit) and no
    # masked-language head, so the prediction surfaces have to read the
    # logit instead of a vocabulary distribution.
    _CROSS_ENCODERS = set(RERANKER_CHOICES)

    # Relevance heads, kept beside the cache so ``get_model`` keeps returning
    # the (tokenizer, encoder, mlm) triple every caller unpacks.
    _heads: dict = {}

    # LRU cache - keep up to _MAX_CACHED models loaded simultaneously.
    # Compare-mode + bias-mode routinely needs {target_model, base_encoder,
    # bias_model} at the same time; a cache of 2 caused ping-pong evictions.
    from collections import OrderedDict as _OrderedDict
    _instances: "_OrderedDict[str, tuple]" = _OrderedDict()
    _MAX_CACHED = 4
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str):
        """
        Returns (tokenizer, encoder_model, mlm_model) for the specified model_name.
        Loads from cache if available, otherwise loads from HuggingFace.
        """
        # Local directories (e.g. a not-yet-published checkpoint) are trusted:
        # allow them through so calibration/eval can run on a local folder
        # before it is uploaded to the Hub.
        if model_name not in cls._ALLOWED_MODELS and not os.path.isdir(model_name):
            raise ValueError(
                f"Model '{model_name}' is not in the allowed list. "
                f"Allowed: {sorted(cls._ALLOWED_MODELS)}"
            )

        with cls._lock:
            # Check if model is already loaded (mark as most-recently-used)
            if model_name in cls._instances:
                cls._instances.move_to_end(model_name)
                return cls._instances[model_name]

            # Evict least-recently-used entries until we have room for one more
            while len(cls._instances) >= cls._MAX_CACHED:
                old_name, old_entry = cls._instances.popitem(last=False)
                _logger.info("Unloading previous model (%s) to free memory...", old_name)
                _release_to_cpu(old_entry)
                old_head = cls._heads.pop(old_name, None)
                _release_to_cpu((old_head,) if old_head is not None else ())
                del old_entry, old_head
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Load outside the lock so other threads can still access the cache
        # for already-loaded models while this (slow) download runs.
        _logger.info("Loading model: %s...", model_name)

        try:
            is_gpt2 = "gpt2" in model_name

            if is_gpt2:
                tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # GPT-2 has no sep/cls/mask tokens; transformers logs
                # "Using sep_token, but it is not set yet." on every
                # internal access unless verbose is off. Pure log noise.
                tokenizer.verbose = False

                encoder = GPT2Model.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    **_eager_kwargs(),
                )
                encoder.eval()

                try:
                    mlm = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    mlm.eval()
                except Exception:
                    _logger.warning("Could not load LM head for %s", model_name, exc_info=True)
                    mlm = None
            elif cls.is_cross_encoder(model_name):
                # Reranker: the trunk is a plain BERT encoder, so every
                # attention view works unchanged. The head is a one-output
                # classifier, and there is no MLM head to load: asking for one
                # would silently hand back randomly initialised weights and
                # the Overview would show invented predictions.
                tokenizer = BertTokenizerFast.from_pretrained(model_name)

                encoder = BertModel.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    **_eager_kwargs(),
                )
                encoder.eval()

                mlm = None

                head = BertForSequenceClassification.from_pretrained(
                    model_name,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                head.eval()
                head.to("cuda" if torch.cuda.is_available() else "cpu")
                with cls._lock:
                    cls._heads[model_name] = head
            else:
                tokenizer = BertTokenizerFast.from_pretrained(model_name)

                encoder = BertModel.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    **_eager_kwargs(),
                )
                encoder.eval()

                try:
                    mlm = BertForMaskedLM.from_pretrained(
                        model_name,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    mlm.eval()
                except Exception:
                    _logger.warning("Could not load MLM head for %s", model_name, exc_info=True)
                    mlm = None

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder.to(device)
            if mlm:
                mlm.to(device)

            result = (tokenizer, encoder, mlm)

            with cls._lock:
                # Another thread may have loaded the same model while we were
                # downloading - check again to avoid duplicates.
                if model_name in cls._instances:
                    cls._instances.move_to_end(model_name)
                    return cls._instances[model_name]
                cls._instances[model_name] = result

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

    @classmethod
    def free_all(cls):
        """Drop every cached model and hand the device memory back.

        For callers that hit an out-of-memory during a load and want to retry
        with a clean device rather than fail. Already-fetched models keep
        working: they are moved to CPU, not destroyed.
        """
        with cls._lock:
            entries = list(cls._instances.values())
            cls._instances.clear()
            entries.extend((head,) for head in cls._heads.values())
            cls._heads.clear()
        for entry in entries:
            _release_to_cpu(entry)
        del entries
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def is_cross_encoder(cls, model_name: str) -> bool:
        """True for rerankers scored as a query/passage pair."""
        return model_name in cls._CROSS_ENCODERS

    @classmethod
    def get_relevance_head(cls, model_name: str):
        """Return the one-output classification head of a reranker.

        Loads the model first if it is not cached. Returns ``None`` for every
        model that is not a reranker, so callers can branch on the result.
        """
        if not cls.is_cross_encoder(model_name):
            return None
        with cls._lock:
            head = cls._heads.get(model_name)
        if head is not None:
            return head
        cls.get_model(model_name)          # populates _heads
        with cls._lock:
            return cls._heads.get(model_name)

    @staticmethod
    def encode_pair(tokenizer, query: str, passage: str, max_length: int = 512):
        """Tokenise a query and a passage as the two segments of one input.

        The reranker is trained on pairs, so the passage tokens must carry
        token type id 1. Encoding ``"query [SEP] passage"`` as a single string
        leaves every token in segment 0 and changes what the model sees.
        """
        return tokenizer(
            query,
            passage,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    @classmethod
    def score_pair(cls, model_name: str, query: str, passage: str) -> float:
        """Relevance logit the reranker gives this query/passage pair.

        Higher means more relevant. The value is the raw logit, which is what
        the ranking is built on; it is not a probability.
        """
        head = cls.get_relevance_head(model_name)
        if head is None:
            raise ValueError(f"Model '{model_name}' is not a reranker.")
        tokenizer, _encoder, _mlm = cls.get_model(model_name)
        inputs = cls.encode_pair(tokenizer, query, passage)
        device = cls.get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = head(**inputs).logits
        return float(logits.reshape(-1)[0])


__all__ = ["ModelManager", "RERANKER_CHOICES", "RERANKER_DEFAULT"]
