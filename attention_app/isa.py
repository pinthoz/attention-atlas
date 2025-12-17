import torch
import numpy as np
import nltk
from typing import List, Dict, Tuple, Optional

# Ensure nltk data is downloaded
# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt_tab')

def get_sentence_boundaries(text: str, tokens: List[str], tokenizer, inputs) -> Tuple[List[str], List[int]]:
    """
    Split text into sentences and map tokens to sentences.
    Works for both BERT-style and GPT-style tokenizers.
    
    Returns:
        sentence_texts: List of sentence strings
        sentence_boundaries_ids: List of start token indices for each sentence
    """
    # 1. Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # 2. Construct a map of char_index -> sentence_index
    char_to_sent = {}
    current_pos = 0
    for i, sent in enumerate(sentences):
        # Find start of sentence in text (handling potential whitespace gaps)
        start = text.find(sent, current_pos)
        if start == -1:
            # Fallback: just assume it follows immediately
            start = current_pos
        end = start + len(sent)
        for k in range(start, end):
            char_to_sent[k] = i
        current_pos = end
    
    # 3. Map tokens to sentences using greedy character-based matching
    token_to_sent = []
    current_text_pos = 0
    
    # Detect tokenizer type
    is_gpt_style = any(tok.startswith("Ġ") or tok.startswith("Â") for tok in tokens[:min(10, len(tokens))])
    is_bert_style = any("##" in tok for tok in tokens[:min(10, len(tokens))])
    
    for i, tok in enumerate(tokens):
        # Handle special tokens
        if tok in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>", "<|pad|>"]:
            token_to_sent.append(-1)
            continue
        
        # Clean token based on tokenizer type
        if is_gpt_style:
            clean_tok = tok.replace("Ġ", " ").replace("Â", "").strip()
        elif is_bert_style:
            clean_tok = tok.replace("##", "")
        else:
            clean_tok = tok
        
        if not clean_tok:
            token_to_sent.append(-1)
            continue
        
        # Skip whitespace in text
        while current_text_pos < len(text) and text[current_text_pos].isspace():
            current_text_pos += 1
        
        # Try to match token
        match_len = len(clean_tok)
        if current_text_pos + match_len <= len(text):
            # Check for match (case-insensitive for robustness)
            text_slice = text[current_text_pos:current_text_pos+match_len]
            if text_slice.lower() == clean_tok.lower():
                # Found match - determine sentence index
                center_pos = current_text_pos + match_len // 2
                sent_idx = char_to_sent.get(center_pos, -1)
                token_to_sent.append(sent_idx)
                current_text_pos += match_len
            else:
                token_to_sent.append(-1)
        else:
            token_to_sent.append(-1)
    
    # 4. Extract sentence boundaries (first token of each sentence)
    final_boundaries = []
    for s_idx in range(len(sentences)):
        # Find first token with this s_idx
        found = False
        for t_idx, assigned_s_idx in enumerate(token_to_sent):
            if assigned_s_idx == s_idx:
                final_boundaries.append(t_idx)
                found = True
                break
        if not found:
            # Fallback: use previous boundary or 0
            final_boundaries.append(final_boundaries[-1] if final_boundaries else 0)
    
    return sentences, final_boundaries


def compute_isa(attentions, tokens: List[str], text: str, tokenizer, inputs, 
                model_type: str = "bert") -> Dict:
    """
    Compute Inter-Sentence Attention (ISA) matrix.
    
    Args:
        attentions: Tuple of tensors (num_layers, batch, num_heads, seq_len, seq_len)
                   or list of tensors.
        tokens: List of token strings.
        text: Original input text.
        tokenizer: The tokenizer used.
        inputs: The inputs dict (optional, for offset mapping if we could use it).
        model_type: "bert" for bidirectional models, "gpt" for causal models
        
    Returns:
        dict: {
            "sentence_texts": List[str],
            "sentence_boundaries_ids": List[int],
            "sentence_attention_matrix": np.ndarray (num_sentences, num_sentences),
            "model_type": str,
            "is_causal": bool
        }
    """
    # 1. Sentence Segmentation
    sentence_texts, sentence_boundaries_ids = get_sentence_boundaries(text, tokens, tokenizer, inputs)
    num_sentences = len(sentence_texts)
    
    if num_sentences < 1:
        # If no sentences, return empty result
        return {
            "sentence_texts": sentence_texts,
            "sentence_boundaries_ids": sentence_boundaries_ids,
            "sentence_attention_matrix": np.zeros((0, 0)),
            "model_type": model_type,
            "is_causal": model_type == "gpt"
        }
    
    if num_sentences == 1:
        # If only one sentence, return trivial result
        return {
            "sentence_texts": sentence_texts,
            "sentence_boundaries_ids": sentence_boundaries_ids,
            "sentence_attention_matrix": np.ones((1, 1)),
            "model_type": model_type,
            "is_causal": model_type == "gpt"
        }

    # 2. Integrate Token-Level Attention (Layer Aggregation)
    # Stack attentions: (num_layers, num_heads, seq_len, seq_len)
    # attentions is usually a tuple of tensors, each (batch, num_heads, seq_len, seq_len)
    # We assume batch_size=1
    
    # Handle different attention formats
    if isinstance(attentions[0], tuple):
        # For some models, attentions might be nested
        stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
    else:
        # Standard format: just stack directly (removing batch dimension if needed)
        if len(attentions[0].shape) == 4:  # (batch, heads, seq, seq)
            stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
        else:  # Already (heads, seq, seq)
            stacked_attentions = torch.stack(attentions, dim=0)
    
    # stacked_attentions shape: (num_layers, num_heads, seq_len, seq_len)
    
    # Max across layers - A shape: (num_heads, seq_len, seq_len)
    A = torch.max(stacked_attentions, dim=0)[0]
    
    # 3. Aggregate Token Attention -> Sentence Attention
    # ISA(Sa, Sb) = max_h max_{i in Sa, j in Sb} A[h, i, j]
    
    # We need token ranges for each sentence.
    # sentence_boundaries_ids gives start indices.
    # End index is the start of next sentence, or end of sequence.
    
    ranges = []
    for k in range(num_sentences):
        start = sentence_boundaries_ids[k]
        if k < num_sentences - 1:
            end = sentence_boundaries_ids[k+1]
        else:
            # For the last sentence, go until the end of sequence
            end = len(tokens)
        ranges.append((start, end))
    
    isa_matrix = np.zeros((num_sentences, num_sentences))
    
    for r_idx, (start_row, end_row) in enumerate(ranges):
        for c_idx, (start_col, end_col) in enumerate(ranges):
            if start_row >= end_row or start_col >= end_col:
                continue
            
            # Extract submatrix for this sentence pair: (H, Sa_len, Sb_len)
            sub_att = A[:, start_row:end_row, start_col:end_col]
            
            # Max over heads and token positions
            if sub_att.numel() > 0:
                val = torch.max(sub_att).item()
            else:
                val = 0.0
            
            isa_matrix[r_idx, c_idx] = val
    
    # Final safety check for NaNs
    isa_matrix = np.nan_to_num(isa_matrix, nan=0.0)
    
    # Add metadata about causality
    # For causal models (GPT), the upper triangle should be mostly zeros
    # We can detect this automatically
    is_causal = False
    if num_sentences > 1:
        # Check if upper triangle is mostly zero (causal attention mask)
        upper_triangle = np.triu(isa_matrix, k=1)
        if np.mean(upper_triangle) < 0.01:  # Threshold for "mostly zero"
            is_causal = True
    
    return {
        "sentence_texts": sentence_texts,
        "sentence_boundaries_ids": sentence_boundaries_ids,
        "sentence_attention_matrix": isa_matrix,
        "model_type": model_type,
        "is_causal": is_causal
    }


def get_sentence_token_attention(attentions, tokens: List[str], sent_x_idx: int, sent_y_idx: int, 
                                   sentence_boundaries_ids: List[int]) -> Tuple[np.ndarray, List[str], int]:
    """
    Extract token-level attention for a specific sentence pair.
    Works for both bidirectional (BERT) and causal (GPT) models.
    
    Args:
        attentions: Tuple of attention tensors (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: List of token strings
        sent_x_idx: Index of source sentence X (target/query)
        sent_y_idx: Index of source sentence Y (source/key)
        sentence_boundaries_ids: List of sentence start indices
        
    Returns:
        attention_data: numpy array (len(sent_x_tokens), len(sent_y_tokens)) - max attention across layers & heads
        tokens_combined: concatenated list of [sent_x_tokens, sent_y_tokens]
        sentence_b_start: index where sent_y tokens start in tokens_combined
    """
    # Handle different attention formats
    if isinstance(attentions[0], tuple):
        stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
    else:
        if len(attentions[0].shape) == 4:
            stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
        else:
            stacked_attentions = torch.stack(attentions, dim=0)
    
    # stacked_attentions shape: (num_layers, num_heads, seq_len, seq_len)
    
    # Max over layers: A shape: (num_heads, seq_len, seq_len)
    A = torch.max(stacked_attentions, dim=0)[0]
    
    # Max over heads: A_max shape: (seq_len, seq_len)
    A_max = torch.max(A, dim=0)[0].cpu().numpy()
    
    # Get token ranges for the two sentences
    num_sentences = len(sentence_boundaries_ids)
    
    def get_range(idx):
        start = sentence_boundaries_ids[idx]
        if idx < num_sentences - 1:
            end = sentence_boundaries_ids[idx + 1]
        else:
            end = len(tokens)
        return start, end
    
    start_x, end_x = get_range(sent_x_idx)
    start_y, end_y = get_range(sent_y_idx)
    
    # Extract submatrix: rows=sent_x (query), cols=sent_y (key)
    sub_att = A_max[start_x:end_x, start_y:end_y]
    
    # Extract tokens
    toks_x = tokens[start_x:end_x]
    toks_y = tokens[start_y:end_y]
    
    # Combine tokens for visualization [sent_x_tokens, sent_y_tokens]
    tokens_combined = list(toks_x) + list(toks_y)
    sentence_b_start = len(toks_x)
    
    return sub_att, tokens_combined, sentence_b_start


# Additional utility functions for GPT-specific analysis

def compute_isa_with_aggregation(attentions, tokens: List[str], text: str, tokenizer, inputs,
                                  model_type: str = "bert", 
                                  aggregation_method: str = "max") -> Dict:
    """
    Compute ISA with different aggregation methods.
    
    Args:
        attentions: Attention tensors
        tokens: Token strings
        text: Original text
        tokenizer: Tokenizer
        inputs: Input dict
        model_type: "bert" or "gpt"
        aggregation_method: "max" (default), "mean", or "last_layer"
        
    Returns:
        ISA result dictionary with aggregation_method field
    """
    sentence_texts, sentence_boundaries_ids = get_sentence_boundaries(text, tokens, tokenizer, inputs)
    num_sentences = len(sentence_texts)
    
    if num_sentences < 1:
        return {
            "sentence_texts": sentence_texts,
            "sentence_boundaries_ids": sentence_boundaries_ids,
            "sentence_attention_matrix": np.zeros((0, 0)),
            "model_type": model_type,
            "is_causal": model_type == "gpt",
            "aggregation_method": aggregation_method
        }
    
    # Handle different attention formats
    if isinstance(attentions[0], tuple):
        stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
    else:
        if len(attentions[0].shape) == 4:
            stacked_attentions = torch.stack([att[0] for att in attentions], dim=0)
        else:
            stacked_attentions = torch.stack(attentions, dim=0)
    
    # stacked_attentions shape: (num_layers, num_heads, seq_len, seq_len)
    
    # Apply aggregation method
    if aggregation_method == "max":
        # Max across layers, then heads
        A = torch.max(stacked_attentions, dim=0)[0]  # (num_heads, seq_len, seq_len)
    elif aggregation_method == "mean":
        # Mean across layers and heads
        A = stacked_attentions.mean(dim=0)  # (num_heads, seq_len, seq_len)
    elif aggregation_method == "last_layer":
        # Only use last layer
        A = stacked_attentions[-1]  # (num_heads, seq_len, seq_len)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Compute sentence ranges
    ranges = []
    for k in range(num_sentences):
        start = sentence_boundaries_ids[k]
        if k < num_sentences - 1:
            end = sentence_boundaries_ids[k+1]
        else:
            end = len(tokens)
        ranges.append((start, end))
    
    isa_matrix = np.zeros((num_sentences, num_sentences))
    
    for r_idx, (start_row, end_row) in enumerate(ranges):
        for c_idx, (start_col, end_col) in enumerate(ranges):
            if start_row >= end_row or start_col >= end_col:
                continue
            
            # Extract submatrix: (num_heads, Sa_len, Sb_len)
            sub_att = A[:, start_row:end_row, start_col:end_col]
            
            # Max over heads and positions
            if sub_att.numel() > 0:
                val = torch.max(sub_att).item()
            else:
                val = 0.0
            
            isa_matrix[r_idx, c_idx] = val
    
    # Safety check
    isa_matrix = np.nan_to_num(isa_matrix, nan=0.0)
    
    # Detect causality
    is_causal = False
    if num_sentences > 1:
        upper_triangle = np.triu(isa_matrix, k=1)
        if np.mean(upper_triangle) < 0.01:
            is_causal = True
    
    return {
        "sentence_texts": sentence_texts,
        "sentence_boundaries_ids": sentence_boundaries_ids,
        "sentence_attention_matrix": isa_matrix,
        "model_type": model_type,
        "is_causal": is_causal,
        "aggregation_method": aggregation_method
    }


def analyze_sentence_dependencies(isa_result: Dict) -> Dict:
    """
    Analyze sentence dependencies from ISA matrix.
    Particularly useful for causal models (GPT).
    
    Args:
        isa_result: Result dictionary from compute_isa
        
    Returns:
        Dictionary with dependency analysis
    """
    isa_matrix = isa_result["sentence_attention_matrix"]
    num_sentences = len(isa_result["sentence_texts"])
    is_causal = isa_result.get("is_causal", False)
    
    if num_sentences < 2:
        return {"dependencies": [], "influences": [], "spans": []}
    
    # For each sentence, find top dependencies
    dependencies = []
    for i in range(num_sentences):
        if i > 0:
            # Get dependencies (excluding self)
            deps = isa_matrix[i, :i]
            if len(deps) > 0:
                top_idx = np.argmax(deps)
                top_score = deps[top_idx]
                avg_score = np.mean(deps)
                dependencies.append({
                    "sentence_idx": i,
                    "top_dependency": int(top_idx),
                    "top_score": float(top_score),
                    "avg_dependency": float(avg_score)
                })
        else:
            dependencies.append({
                "sentence_idx": i,
                "top_dependency": None,
                "top_score": 0.0,
                "avg_dependency": 0.0
            })
    
    # For each sentence, compute influence on later sentences
    influences = []
    for j in range(num_sentences):
        if j < num_sentences - 1:
            # Get influence scores
            infl = isa_matrix[j+1:, j]
            if len(infl) > 0:
                max_infl = np.max(infl)
                avg_infl = np.mean(infl)
                influences.append({
                    "sentence_idx": j,
                    "max_influence": float(max_infl),
                    "avg_influence": float(avg_infl)
                })
        else:
            influences.append({
                "sentence_idx": j,
                "max_influence": 0.0,
                "avg_influence": 0.0
            })
    
    # Compute attention span (average distance attended)
    spans = []
    for i in range(num_sentences):
        if i > 0:
            deps = isa_matrix[i, :i]
            if np.sum(deps) > 0:
                distances = np.arange(1, i+1)[::-1]  # [i, i-1, ..., 1]
                avg_span = np.sum(distances * deps) / np.sum(deps)
                spans.append({
                    "sentence_idx": i,
                    "avg_span": float(avg_span)
                })
            else:
                spans.append({"sentence_idx": i, "avg_span": 0.0})
        else:
            spans.append({"sentence_idx": i, "avg_span": 0.0})
    
    return {
        "dependencies": dependencies,
        "influences": influences,
        "spans": spans,
        "is_causal": is_causal
    }


# Backward compatibility
def compute_isa_bert(attentions, tokens: List[str], text: str, tokenizer, inputs):
    """Backward compatible function for BERT models"""
    return compute_isa(attentions, tokens, text, tokenizer, inputs, model_type="bert")


def compute_isa_gpt(attentions, tokens: List[str], text: str, tokenizer, inputs):
    """Function specifically for GPT models"""
    return compute_isa(attentions, tokens, text, tokenizer, inputs, model_type="gpt")
