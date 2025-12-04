"""Model loading utilities with caching and dynamic selection."""

import logging
import warnings
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers.utils import logging as transformers_logging

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
    """Manages loading and caching of BERT models."""
    
    _instances = {}

    @classmethod
    def get_model(cls, model_name: str):
        """
        Returns (tokenizer, encoder_model, mlm_model) for the specified model_name.
        Loads from cache if available, otherwise loads from HuggingFace.
        """
        # Check if model is already loaded
        if model_name in cls._instances:
            return cls._instances[model_name]

        # Clear existing cache to free memory
        if cls._instances:
            print(f"Unloading previous models to free memory...")
            cls._instances.clear()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Loading model: {model_name}...")
        
        try:

            is_gpt2 = "gpt2" in model_name
            
            if is_gpt2:
                from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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
                except Exception as e:
                    print(f"Warning: Could not load LM head for {model_name}: {e}")
                    mlm = None
            else:
                tokenizer = BertTokenizer.from_pretrained(model_name)
                
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
                except Exception as e:
                    print(f"Warning: Could not load MLM head for {model_name}: {e}")
                    mlm = None

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder.to(device)
            if mlm:
                mlm.to(device)

            cls._instances[model_name] = (tokenizer, encoder, mlm)
            return tokenizer, encoder, mlm
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

__all__ = ["ModelManager"]
