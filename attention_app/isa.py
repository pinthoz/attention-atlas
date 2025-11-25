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
            
        clean_tok = tok.replace("##", "").replace("Ġ", "")
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

def aggregate_attention(attentions):
    """
    Aggregates attention across layers and heads in a more refined way.
    """
    # Stack attentions: (num_layers, num_heads, seq_len, seq_len)
    stacked_attentions = torch.cat(attentions, dim=0)
    
    # Mean across layers
    avg_attention = torch.mean(stacked_attentions, dim=0)  # (num_heads, seq_len, seq_len)
    
    # Max across heads
    max_attention = torch.max(avg_attention, dim=0)[0]  # (seq_len, seq_len)
    return max_attention

def apply_attention_mask(attention_matrix, sentence_boundaries_ids):
    """
    Applies a mask to the attention matrix, highlighting inter-sentence interactions.
    Penalizes interactions within the same sentence.
    """
    mask = np.ones_like(attention_matrix)
    num_tokens = attention_matrix.shape[0]
    
    # Define ranges for each sentence
    ranges = []
    for k in range(len(sentence_boundaries_ids)):
        start = sentence_boundaries_ids[k]
        if k < len(sentence_boundaries_ids) - 1:
            end = sentence_boundaries_ids[k+1]
        else:
            end = num_tokens
        ranges.append((start, end))
        
    # Set intra-sentence blocks to 0 (or very low value)
    for start, end in ranges:
        # Ensure indices are within bounds
        s = max(0, min(start, num_tokens))
        e = max(0, min(end, num_tokens))
        mask[s:e, s:e] = 0.0
        
    attention_matrix *= mask
    return attention_matrix

def normalize_attention_matrix(isa_matrix):
    """
    Normalizes the inter-sentence attention matrix for easier visualization.
    """
    max_value = np.max(isa_matrix)
    min_value = np.min(isa_matrix)
    if max_value - min_value == 0:
        return np.zeros_like(isa_matrix)
        
    normalized_matrix = (isa_matrix - min_value) / (max_value - min_value)
    return normalized_matrix

def compute_isa(attentions, tokens: List[str], text: str, tokenizer, inputs):
    """
    Modifies the ISA calculation in GPT-2 to better highlight inter-sentence interactions.
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

    # 2. Aggregate Attention
    # A is (seq_len, seq_len)
    A = aggregate_attention(attentions)
    
    # 3. Apply Mask (Penalize Intra-Sentence)
    A_masked = apply_attention_mask(A.cpu().numpy(), sentence_boundaries_ids)
    
    # 4. Normalize
    A_normalized = normalize_attention_matrix(A_masked)
    
    # 5. Aggregate to Sentence-Level Matrix
    # We need to reduce the token-level matrix (seq_len, seq_len) to (num_sentences, num_sentences)
    # Since we masked intra-sentence, the diagonal blocks are 0.
    
    isa_matrix = np.zeros((num_sentences, num_sentences))
    
    ranges = []
    for k in range(num_sentences):
        start = sentence_boundaries_ids[k]
        if k < num_sentences - 1:
            end = sentence_boundaries_ids[k+1]
        else:
            end = len(tokens)
        ranges.append((start, end))
        
    for r_idx, (start_row, end_row) in enumerate(ranges):
        for c_idx, (start_col, end_col) in enumerate(ranges):
            if start_row >= end_row or start_col >= end_col:
                continue
                
            # Extract submatrix
            sub_att = A_normalized[start_row:end_row, start_col:end_col]
            
            if sub_att.size > 0:
                # Use mean over tokens to represent the strength of connection between sentences
                val = np.mean(sub_att)
            else:
                val = 0.0
            
            isa_matrix[r_idx, c_idx] = val
            
    return {
        "sentence_texts": sentence_texts,
        "sentence_boundaries_ids": sentence_boundaries_ids,
        "sentence_attention_matrix": isa_matrix
    }

def get_sentence_token_attention(attentions, tokens: List[str], sent_x_idx: int, sent_y_idx: int, 
                                   sentence_boundaries_ids: List[int]) -> Tuple[np.ndarray, List[str], int]:
    """
    Extract token-level attention for a specific sentence pair.
    Uses the same aggregation logic as compute_isa (Mean layers -> Max heads).
    """
    # Aggregate attention: Mean layers, Max heads
    A = aggregate_attention(attentions).cpu().numpy()  # (seq_len, seq_len)
    
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
    sub_att = A[start_x:end_x, start_y:end_y]
    
    # Extract tokens
    toks_x = tokens[start_x:end_x]
    toks_y = tokens[start_y:end_y]
    
    # Combine tokens for visualization [sent_x_tokens, sent_y_tokens]
    tokens_combined = list(toks_x) + list(toks_y)
    sentence_b_start = len(toks_x)
    
    return sub_att, tokens_combined, sentence_b_start
