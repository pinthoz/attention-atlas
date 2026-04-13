import logging
import threading
import warnings
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from transformers.utils import logging as transformers_logging
from transformers import GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
import gc

_logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


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
    }

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
        if model_name not in cls._ALLOWED_MODELS:
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
                old_name, _ = cls._instances.popitem(last=False)
                _logger.info("Unloading previous model (%s) to free memory...", old_name)
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

                encoder = GPT2Model.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
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
            else:
                tokenizer = BertTokenizerFast.from_pretrained(model_name)

                encoder = BertModel.from_pretrained(
                    model_name,
                    output_attentions=True,
                    output_hidden_states=True,
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
                # downloading — check again to avoid duplicates.
                if model_name in cls._instances:
                    cls._instances.move_to_end(model_name)
                    return cls._instances[model_name]
                cls._instances[model_name] = result

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

__all__ = ["ModelManager"]
