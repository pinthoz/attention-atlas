from io import BytesIO
import base64
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def positional_encoding(position: int, d_model: int = 768) -> np.ndarray:
    """Sinusoidal positional encodings to mimic transformer inputs."""
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pe


def array_to_base64_img(array: np.ndarray, cmap: str = "Blues", height: float = 0.22) -> str:
    """Encode a 1D numpy array as a small PNG strip for inline HTML usage."""
    plt.figure(figsize=(3, height))
    plt.imshow(array[np.newaxis, :], cmap=cmap, aspect="auto")
    plt.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_influence_tree(attention_matrix, tokens, Q_matrix, K_matrix, d_k, root_token_idx, top_k=3, max_depth=5):
    """
    Compute hierarchical influence tree from attention weights.
    
    This function builds a multi-hop attention tree starting from a root token,
    showing which tokens receive the highest attention at each level.
    
    Args:
        attention_matrix: numpy array of shape (seq_len, seq_len) containing attention weights
        tokens: list of token strings
        Q_matrix: numpy array of Query vectors
        K_matrix: numpy array of Key vectors
        d_k: dimension for scaling (d_k = d_model / num_heads)
        root_token_idx: int, index of the root token to analyze
        top_k: int, number of top children to select at each level (default: 3)
        max_depth: int, maximum depth of the tree (default: 3)
    
    Returns:
        dict: Tree structure with the following format:
            {
                'name': str (token text),
                'att': float (attention weight),
                'qk_sim': float (Q·K dot product),
                'children': [tree_node, ...]
            }
    """
    def build_tree(token_idx, depth, current_path=None, parent_idx=None):
        if current_path is None:
            current_path = set()
            
        # Prevent cycles: strict check against current path
        if depth >= max_depth or token_idx in current_path:
            return None
        
        # New path for this branch includes current token
        new_path = current_path.copy()
        new_path.add(token_idx)
        
        # Get attention scores from this token
        attention_scores = attention_matrix[token_idx]
        
        # Compute Q·K similarity if we have a parent
        qk_sim = 0.0
        if parent_idx is not None:
            qk_dot = float(np.dot(Q_matrix[parent_idx], K_matrix[token_idx]))
            qk_sim = qk_dot / np.sqrt(d_k)
        
        # Get attention weight (from parent to this node)
        if parent_idx is not None:
            att_weight = float(attention_matrix[parent_idx][token_idx])
        else:
            att_weight = 1.0  # Root node
        
        # Get top-k tokens, excluding those in the CURRENT path to avoid cycles
        # But allowing tokens used in other branches
        top_indices = []
        sorted_indices = np.argsort(attention_scores)[::-1]
        
        for idx in sorted_indices:
            if len(top_indices) >= top_k:
                break
            # Skip if it would form a cycle in this branch
            if idx in new_path:
                continue
            top_indices.append(idx)
        
        # Build children recursively
        children = []
        for child_idx in top_indices:
            child_tree = build_tree(child_idx, depth + 1, new_path, token_idx)
            if child_tree:
                children.append(child_tree)
        
        return {
            'name': tokens[token_idx].replace("##", "").replace("Ġ", ""),
            'att': att_weight,
            'qk_sim': qk_sim,
            'token_idx': int(token_idx),
            'children': children
        }
    
    return build_tree(root_token_idx, 0)


def aggregate_data_to_words(res, filter_special=True):
    """
    Aggregates sub-token attention data into word-level data.
    
    Logic:
    1. Filter out special tokens ([CLS], [SEP], <pad>, <s>, </s>) if requested.
    2. Group remaining tokens into words based on heuristics:
       - BERT: '##' indicates continuation.
       - GPT-2: 'Ġ' (U+0120) indicates start of new word (except index 0 which is always start).
    3. Aggregate:
       - Tokens: Concatenated string + count (e.g., "word (2)")
       - Vectors (Embeddings/Hidden States): Mean pooling.
       - Attention Matrix: 
           - From Word (Rows): Average of sub-token rows.
           - To Word (Cols): Sum of sub-token columns.
    4. Re-normalize: Ensure each row sums to 1 after filtering/aggregation.
    
    Args:
        res: Tuple from heavy_compute containing (tokens, embeddings, pos_enc, attentions, hidden_states, ...)
        filter_special: Boolean, whether to remove special tokens.
        
    Returns:
        New 'res' tuple with aggregated data.
    """
    if res is None:
        return None
        
    (tokens, embeddings, pos_enc, attentions, hidden_states, inputs, 
     tokenizer, encoder_model, mlm_model, head_specialization, 
     isa_data, head_clusters) = res

    # 1. Identify valid indices (Filtering)
    # Common special tokens across models
    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>", "<s>", "</s>", "<pad>", "<mask>"}
    
    valid_indices = []
    for i, t in enumerate(tokens):
        if filter_special and t in special_tokens:
            continue
        valid_indices.append(i)
        
    if not valid_indices:
        return res # Fallback if everything is filtered (shouldn't happen)

    # 2. Grouping Heuristics
    word_groups = [] # List of lists of original indices [[0, 1], [2], [3, 4]]
    current_group = []
    
    # Pre-compute GPT-2 mode detection once (not inside the loop)
    has_gpt2_markers = any("Ġ" in tok for tok in tokens)
    punct_set = {".", ",", "!", "?", ";", ":", "-", "(", ")", '"', "'", "'", """, """}

    prev_idx = None  # Track previous token index for punctuation boundary detection

    for idx in valid_indices:
        t = tokens[idx]
        is_start_of_word = False

        # Heuristics
        if "##" in t: # BERT continuation
            is_start_of_word = False
        elif "Ġ" in t: # GPT-2 start (contains space)
            is_start_of_word = True
        elif "Ċ" in t: # GPT-2 newline marker - always start new word
            is_start_of_word = True
        else:
            # No '##' and no 'Ġ'.
            # For BERT, this is a new word (punctuation or normal word).
            # For GPT-2, this is usually a continuation (e.g., "world" after "Hello"), EXCEPT index 0.
            # To be robust, let's treat "Start of Sentence" (first valid index) as absolute start.
            if len(word_groups) == 0 and not current_group:
                is_start_of_word = True
            elif has_gpt2_markers:
                # GPT-2 Mode:
                # 1. New word if current token is a punctuation mark
                # 2. New word if PREVIOUS token was/ended with punctuation

                # Clean current token for check: remove Ġ, ##, and Ċ (newline)
                clean_t = t.replace("Ġ", "").replace("##", "").replace("Ċ", "").strip()

                # Current token is punctuation - start new word
                if clean_t in punct_set:
                    is_start_of_word = True
                elif prev_idx is not None:
                    # Check the previous token in the sequence (use raw index, not filtered)
                    prev_token = tokens[prev_idx]
                    clean_prev = prev_token.replace("Ġ", "").replace("##", "").replace("Ċ", "").strip()

                    # Split if previous was exact punctuation
                    if clean_prev in punct_set:
                        is_start_of_word = True
                    # Split if previous ENDED in punctuation (e.g. "word.")
                    elif clean_prev and clean_prev[-1] in punct_set:
                        is_start_of_word = True
                    # Split if previous was just markers/newline (empty after cleaning)
                    # This handles cases like [".", "Ċ", "Men"] where "Men" follows invisible chars
                    elif not clean_prev:
                        is_start_of_word = True
            else:
                # BERT Mode: No '##' implies new word
                is_start_of_word = True

        # Update prev_idx for next iteration
        prev_idx = idx
        
        if is_start_of_word:
            if current_group:
                word_groups.append(current_group)
            current_group = [idx]
        else:
            current_group.append(idx)
            
    if current_group:
        word_groups.append(current_group)
        
    # 3. Aggregation
    
    # New Tokens
    new_tokens = []
    for group in word_groups:
        # Construct clean text
        # BERT: remove ##
        # GPT-2: remove Ġ and Ċ, then lowercase
        text_parts = []
        for idx in group:
            clean = tokens[idx].replace("##", "").replace("Ġ", "").replace("Ċ", "")
            text_parts.append(clean)

        full_word = "".join(text_parts)

        # GPT-2: convert to lowercase for display
        if has_gpt2_markers:
            full_word = full_word.lower()

        count = len(group)
        if count > 1:
            full_word += " (*)" # User requested (*) instead of (count)
        new_tokens.append(full_word)
        
    # Mappings for vector ops
    # We need to transform data structures.
    
    # helper to aggregate tensor/array (SEQ_LEN, DIM) -> (NUM_WORDS, DIM)
    def aggregate_sequence_vector(vec_seq):
        # vec_seq is numpy array (seq_len, dim)
        if isinstance(vec_seq, list):
            vec_seq = np.array(vec_seq)
        elif isinstance(vec_seq, torch.Tensor):
            vec_seq = vec_seq.cpu().numpy()
            
        new_vecs = []
        for group in word_groups:
            # Mean pooling
            sub_vecs = vec_seq[group]
            mean_vec = np.mean(sub_vecs, axis=0)
            new_vecs.append(mean_vec)
        return np.array(new_vecs)

    new_embeddings = aggregate_sequence_vector(embeddings)
    new_pos_enc = aggregate_sequence_vector(pos_enc)
    
    # Hidden States: Tuple of (1, seq_len, dim) tensors.
    new_hidden_states = []
    device = hidden_states[0].device
    dtype = hidden_states[0].dtype
    
    for layer_hs in hidden_states:
        # layer_hs is (1, seq_len, dim)
        hs_np = layer_hs[0].cpu().numpy()
        agg_np = aggregate_sequence_vector(hs_np) # (num_words, dim)
        # Reshape to (1, num_words, dim) and tensorify
        new_tensor = torch.tensor(agg_np[np.newaxis, :], device=device, dtype=dtype)
        new_hidden_states.append(new_tensor)
    
    new_hidden_states = tuple(new_hidden_states)
    
    # Attentions: Tuple of (1, num_heads, seq_len, seq_len)
    new_attentions = []
    for layer_att in attentions:
        # layer_att is (1, heads, seq, seq)
        att_np = layer_att[0].cpu().numpy() # (heads, seq, seq)
        num_heads, s_len, _ = att_np.shape
        num_words = len(word_groups)
        
        new_layer_att = np.zeros((num_heads, num_words, num_words), dtype=att_np.dtype)
        
        for h in range(num_heads):
            old_mat = att_np[h] # (seq, seq)
            
            # Step 1: Compress Columns (Target Aggregation -> SUM)
            # "How much attention does Source X pay to [Sub1, Sub2]?" -> Sum(Att(X, Sub1), Att(X, Sub2))
            mid_mat = np.zeros((s_len, num_words))
            for w_col, group in enumerate(word_groups):
                mid_mat[:, w_col] = np.sum(old_mat[:, group], axis=1)
                
            # Step 2: Compress Rows (Source Aggregation -> AVERAGE)
            # "How much attention does [Sub1, Sub2] pay to Target Y?" -> Mean(Att(Sub1, Y), Att(Sub2, Y))
            final_mat = np.zeros((num_words, num_words))
            for w_row, group in enumerate(word_groups):
                final_mat[w_row, :] = np.mean(mid_mat[group, :], axis=0)
            
            # Step 3: Re-normalize Rows
            # Filtered tokens might have contained attention mass.
            # Also, averaging sources usually preserves sum=1 IF targets were complete.
            # But we filtered targets (special tokens). So sum < 1.
            row_sums = final_mat.sum(axis=1, keepdims=True) + 1e-9
            final_mat = final_mat / row_sums
            
            new_layer_att[h] = final_mat

        new_tensor = torch.tensor(new_layer_att[np.newaxis, :], device=device, dtype=dtype)
        new_attentions.append(new_tensor)
        
    new_attentions = tuple(new_attentions)
    
    # Recompute derived metrics on aggregated data
    # We do this here so visualizations downstream don't crash or show nothing
    try:
        from .isa import compute_isa
        from .head_specialization import compute_all_heads_specialization, compute_head_clusters
        
        # 1. ISA
        # reconstruct text for ISA (approximation)
        agg_text = " ".join(new_tokens) # Basic joining
        new_isa_data = compute_isa(new_attentions, new_tokens, agg_text, tokenizer, inputs)
        
        # 2. Head Specialization
        new_head_specialization = compute_all_heads_specialization(new_attentions, new_tokens, agg_text)
        
        # 3. Head Clusters
        new_head_clusters = compute_head_clusters(new_head_specialization)
        
    except Exception as e:
        print(f"Warning: Failed to recompute aggregated metrics: {e}")
        new_isa_data = None
        new_head_specialization = None
        new_head_clusters = None
    
    return (new_tokens, new_embeddings, new_pos_enc, new_attentions, 
            new_hidden_states, inputs, tokenizer, encoder_model, mlm_model, 
            new_head_specialization, new_isa_data, new_head_clusters)


__all__ = ["positional_encoding", "array_to_base64_img", "compute_influence_tree", "aggregate_data_to_words"]
