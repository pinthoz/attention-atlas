import re
import torch
import traceback

from ..utils import positional_encoding
from ..models import ModelManager
from ..head_specialization import compute_all_heads_specialization
from ..isa import compute_isa


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
            return tokenizer(sentence_a, sentence_b, return_tensors="pt")
    return tokenizer(text, return_tensors="pt")


def heavy_compute(text, model_name):
    """Perform heavy computation for attention analysis.

    This function loads the model, tokenizes the input, runs inference,
    and computes all necessary metrics for visualization.

    Args:
        text: Input text to analyze
        model_name: Name of the model to use

    Returns:
        Tuple containing:
            - tokens: List of token strings
            - embeddings: Numpy array of token embeddings
            - pos_enc: Positional encoding array
            - attentions: Tuple of attention tensors from each layer
            - hidden_states: Tuple of hidden state tensors from each layer
            - inputs: Tokenized inputs dict
            - tokenizer: Tokenizer instance
            - encoder_model: Encoder model instance
            - mlm_model: MLM model instance
            - head_specialization: Head specialization metrics
            - isa_data: Inter-sentence attention data
    """
    print("DEBUG: Starting heavy_compute")
    if not text:
        return None

    tokenizer, encoder_model, mlm_model = ModelManager.get_model(model_name)
    print("DEBUG: Models loaded")

    device = ModelManager.get_device()
    inputs = tokenize_with_segments(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder_model(**inputs)

    print("DEBUG: Model inference complete")

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    embeddings = outputs.last_hidden_state[0].cpu().numpy()
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    pos_enc = positional_encoding(len(tokens), embeddings.shape[1])

    # Compute head specialization metrics
    head_specialization = None
    try:
        print("DEBUG: Computing head specialization")
        head_specialization = compute_all_heads_specialization(attentions, tokens, text)
        print("DEBUG: Head specialization complete")
    except Exception as e:
        print(f"Warning: Could not compute head specialization: {e}")
        traceback.print_exc()

    # Compute ISA
    isa_data = None
    try:
        print("DEBUG: Computing ISA")
        isa_data = compute_isa(attentions, tokens, text, tokenizer, inputs)
        print("DEBUG: ISA complete")
    except Exception as e:
        print(f"Warning: Could not compute ISA: {e}")
        traceback.print_exc()

    print("DEBUG: heavy_compute finished")
    return (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, tokenizer, encoder_model, mlm_model, head_specialization, isa_data)


__all__ = ["tokenize_with_segments", "heavy_compute"]
