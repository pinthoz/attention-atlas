import torch
from typing import Tuple, List, Any

def extract_attention_for_text(text: str, model_key: str, model_manager: Any) -> Tuple[List[str], Tuple[torch.Tensor, ...]]:
    """
    Runs a lightweight forward pass to extract attention weights for a given text.
    
    Parameters
    ----------
    text : str
        The input text to analyze.
    model_key : str
        The model identifier (e.g., 'bert-base-uncased', 'gpt2').
    model_manager : Any
        The ModelManager instance/class to retrieve model and tokenizer.
        
    Returns
    -------
    tokens : List[str]
        List of tokens corresponding to the input text.
    attentions : Tuple[torch.Tensor, ...]
        Tuple of attention tensors (one per layer), shape (1, num_heads, seq_len, seq_len).
    """
    # Retrieve model and tokenizer
    # ModelManager.get_model returns (tokenizer, encoder, mlm)
    tokenizer, model, _ = model_manager.get_model(model_key)
    
    # Ensure model is in eval mode (though likely already is)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Move inputs to same device as model parameters
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        # Fallback if model device check fails (e.g. empty model)
        pass
    
    # Forward pass with attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get attentions tuple: (layer 1, layer 2, ...)
    # Each tensor shape: (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    
    # Convert token IDs back to tokens for visualization
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    return tokens, attentions
