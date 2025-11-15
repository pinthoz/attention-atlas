"""Model loading utilities kept in a single place."""

import logging
import warnings

from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers.utils import logging as transformers_logging

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

encoder_model = BertModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    output_hidden_states=True,
)
encoder_model.eval()

mlm_model = BertForMaskedLM.from_pretrained(
    MODEL_NAME,
    output_attentions=False,
    output_hidden_states=False,
)
mlm_model.eval()

__all__ = ["MODEL_NAME", "tokenizer", "encoder_model", "mlm_model"]
