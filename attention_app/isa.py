import torch
import numpy as np
import nltk
from typing import List, Dict, Tuple, Optional

# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_sentence_boundaries(text: str, tokens: List[str], tokenizer, inputs) -> Tuple[List[str], List[int]]:
    """
    Split text into sentences and map tokens to sentences.
    Returns:
        sentence_texts: List of sentence strings
        sentence_boundaries_ids: List of start token indices for each sentence
    """
    # 1. Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # 2. Map tokens to sentences
    # We need to determine which sentence each token belongs to.
    # We'll use the character offsets if possible.
    
    # Re-run tokenizer to get offsets, matching the logic in server.py's tokenize_with_segments
    # But here we might not know exactly how it was tokenized (single or pair).
    # However, we have the 'tokens' list and 'inputs'.
    # Let's try to align based on reconstructed text or character spans.
    
    # Strategy:
    # 1. Find character spans of each sentence in the original text.
    # 2. Find character spans of each token in the original text.
    # 3. Assign token to sentence if it overlaps significantly.
    
    # To get token offsets, we can use the tokenizer with the original text(s).
    # Since we don't know if it was split into A/B in server.py easily without passing that info,
    # we'll try to use the tokenizer's ability to align if we pass the same input.
    # But server.py logic is: split on first [.!?] -> A, B.
    # If we just use the full text and nltk, we get S1, S2, S3...
    # The tokens might include [SEP] in the middle if it was A/B.
    
    # Let's assume we can rely on the tokens themselves to some extent, but subwords make it hard.
    # Better approach:
    # Iterate through tokens and sentences simultaneously.
    
    sentence_boundaries = [0]
    current_sent_idx = 0
    
    # Clean tokens for matching (remove ##, special tokens)
    # But we need to keep indices aligned with the original 'tokens' list.
    
    # Let's try a character-based alignment.
    # Construct a map of char_index -> sentence_index
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
        
    # Now map tokens to character positions.
    # This is tricky without offset_mapping from the tokenizer.
    # We will try to obtain offset_mapping by re-tokenizing.
    # We'll assume the input was just 'text' for simplicity of getting offsets, 
    # or we try to match tokens to text greedily.
    
    # Greedy matching of tokens to text:
    token_to_sent = []
    current_text_pos = 0
    
    for i, tok in enumerate(tokens):
        if tok in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
            token_to_sent.append(-1) # Special token
            continue
            
        clean_tok = tok.replace("##", "")
        # Find this token in text starting from current_text_pos
        # We need to be careful about case and whitespace.
        # BERT tokens are usually lowercased if uncased model.
        # We'll search case-insensitive.
        
        # Skip whitespace in text
        while current_text_pos < len(text) and text[current_text_pos].isspace():
            current_text_pos += 1
            
        # Check if text matches token
        # This is a heuristic and might fail for some complex tokenizations,
        # but is often sufficient for visualization.
        # A more robust way is to require offset_mapping from the caller.
        # For now, let's try this greedy match.
        
        # Advance position
        match_len = len(clean_tok)
        # Check if we have a match (ignoring case)
        if text[current_text_pos:current_text_pos+match_len].lower() == clean_tok.lower():
            # Found match
            # Determine sentence index for the center of this token
            center_pos = current_text_pos + match_len // 2
            sent_idx = char_to_sent.get(center_pos, -1)
            token_to_sent.append(sent_idx)
            current_text_pos += match_len
        else:
            # Mismatch or special char?
            # If we can't match, assign to previous or -1
            token_to_sent.append(-1)
            
    # Now extract boundaries
    # A boundary is where the sentence index changes.
    # We want the start index of each sentence.
    
    boundaries = []
    seen_sentences = set()
    
    for i, sent_idx in enumerate(token_to_sent):
        if sent_idx != -1 and sent_idx not in seen_sentences:
            boundaries.append(i)
            seen_sentences.add(sent_idx)
            
    # If we missed the first sentence (e.g. only [CLS] before it), ensure it's captured?
    # Actually, boundaries should correspond to 'sentences' list.
    # If we have 3 sentences, we expect 3 boundaries.
    
    # Refine boundaries: ensure we have one start index for each sentence found in 'sentences'.
    # If a sentence wasn't matched (e.g. very short or skipped), we might have issues.
    # Let's fill in gaps.
    
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

def compute_isa(attentions, tokens: List[str], text: str, tokenizer, inputs):
    """
    Compute Inter-Sentence Attention (ISA) matrix.
    
    Args:
        attentions: Tuple of tensors (num_layers, batch, num_heads, seq_len, seq_len)
                   or list of tensors.
        tokens: List of token strings.
        text: Original input text.
        tokenizer: The tokenizer used.
        inputs: The inputs dict (optional, for offset mapping if we could use it).
        
    Returns:
        dict: {
            "sentence_texts": List[str],
            "sentence_boundaries_ids": List[int],
            "sentence_attention_matrix": np.ndarray (num_sentences, num_sentences)
        }
    """
    # 1. Sentence Segmentation
    sentence_texts, sentence_boundaries_ids = get_sentence_boundaries(text, tokens, tokenizer, inputs)
    num_sentences = len(sentence_texts)
    
    if num_sentences < 2:
        # If only one sentence, return trivial result
        return {
            "sentence_texts": sentence_texts,
            "sentence_boundaries_ids": sentence_boundaries_ids,
            "sentence_attention_matrix": np.ones((1, 1))
        }

    # 2. Integrate Token-Level Attention (Layer Aggregation)
    # Stack attentions: (num_layers, num_heads, seq_len, seq_len)
    # attentions is usually a tuple of tensors, each (batch, num_heads, seq_len, seq_len)
    # We assume batch_size=1
    
    # Convert to single tensor
    # layer_attentions: [ (1, H, L, L), ... ]
    stacked_attentions = torch.cat(attentions, dim=0) # (num_layers, H, L, L)
    
    # Max across layers
    # A shape: (H, L, L)
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
            # For the last sentence, go until the first special token after it or end of seq
            end = len(tokens)
        ranges.append((start, end))
        
    isa_matrix = np.zeros((num_sentences, num_sentences))
    
    for r_idx, (start_row, end_row) in enumerate(ranges):
        for c_idx, (start_col, end_col) in enumerate(ranges):
            if start_row >= end_row or start_col >= end_col:
                continue
                
            # Extract submatrix for this sentence pair: (H, Sa_len, Sb_len)
            sub_att = A[:, start_row:end_row, start_col:end_col]
            
            # Max over Sa, Sb
            if sub_att.numel() > 0:
                val = torch.max(sub_att).item()
            else:
                val = 0.0
            
            isa_matrix[r_idx, c_idx] = val
            
    # Final safety check for NaNs
    isa_matrix = np.nan_to_num(isa_matrix, nan=0.0)
            
    return {
        "sentence_texts": sentence_texts,
        "sentence_boundaries_ids": sentence_boundaries_ids,
        "sentence_attention_matrix": isa_matrix
    }

def get_sentence_token_attention(attentions, tokens: List[str], sent_x_idx: int, sent_y_idx: int, 
                                   sentence_boundaries_ids: List[int]) -> Tuple[np.ndarray, List[str], int]:
    """
    Extract token-level attention for a specific sentence pair.
    
    Args:
        attentions: Tuple of attention tensors (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: List of token strings
        sent_x_idx: Index of source sentence X (target)
        sent_y_idx: Index of source sentence Y (source)
        sentence_boundaries_ids: List of sentence start indices
        
    Returns:
        attention_data: numpy array (len(sent_x_tokens), len(sent_y_tokens)) - max attention across layers & heads
        tokens_combined: concatenated list of [sent_x_tokens, sent_y_tokens]
        sentence_b_start: index where sent_y tokens start in tokens_combined
    """
    # Stack attentions and max over layers
    stacked_attentions = torch.cat(attentions, dim=0)  # (num_layers, H, L, L)
    A = torch.max(stacked_attentions, dim=0)[0]  # (H, L, L)
    # Max over heads
    A_max = torch.max(A, dim=0)[0].cpu().numpy()  # (L, L)
    
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
    
    # Extract submatrix: rows=sent_x, cols=sent_y
    sub_att = A_max[start_x:end_x, start_y:end_y]
    
    # Extract tokens
    toks_x = tokens[start_x:end_x]
    toks_y = tokens[start_y:end_y]
    
    # Combine tokens for visualization [sent_x_tokens, sent_y_tokens]
    tokens_combined = list(toks_x) + list(toks_y)
    sentence_b_start = len(toks_x)
    
    return sub_att, tokens_combined, sentence_b_start
