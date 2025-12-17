import numpy as np
import torch
from ..metrics import compute_all_attention_metrics
from ..head_specialization import compute_head_metrics, get_linguistic_tags
from ..isa import compute_isa

def pad_or_truncate_vector(vector, target_length, padding_value=0.0):
    """Ensure vector is exactly target_length. Truncates or pads with padding_value."""
    vector = np.array(vector).flatten()
    if len(vector) > target_length:
        return vector[:target_length]
    elif len(vector) < target_length:
        return np.pad(vector, (0, target_length - len(vector)), constant_values=padding_value)
    return vector

def extract_global_metrics(attentions):
    """
    Extract Global Attention Metrics (GAM) for all heads in all layers.
    Returns a unified dictionary of granular features.
    
    Args:
        attentions: List of tensors (one per layer), each (1, num_heads, seq_len, seq_len)
    """
    features = {}
    
    for layer_idx, layer_att in enumerate(attentions):
        # layer_att shape: (1, num_heads, seq, seq) -> (num_heads, seq, seq)
        layer_att = layer_att[0].cpu().numpy()
        num_heads = layer_att.shape[0]
        
        for head_idx in range(num_heads):
            att_matrix = layer_att[head_idx]
            metrics = compute_all_attention_metrics(att_matrix)
            
            # Flatten dict with granular keys
            prefix = f"GAM_L{layer_idx}_H{head_idx}"
            for key, val in metrics.items():
                features[f"{prefix}_{key}"] = float(val)
                
    return features

def extract_head_specialization_features(attentions, tokens, text):
    """
    Extract Head Specialization metrics for all heads in all layers.
    Granular extraction: 7 metrics per head.
    """
    features = {}
    
    # Get linguistic tags once
    pos_tags, ner_tags = get_linguistic_tags(tokens, text)
    
    for layer_idx, layer_att in enumerate(attentions):
        layer_att = layer_att[0].cpu().numpy()
        num_heads = layer_att.shape[0]
        
        for head_idx in range(num_heads):
            att_matrix = layer_att[head_idx]
            # Get raw metrics (unnormalized is better for raw feature extraction)
            metrics = compute_head_metrics(att_matrix, tokens, pos_tags, ner_tags)
            
            prefix = f"Spec_L{layer_idx}_H{head_idx}"
            for key, val in metrics.items():
                features[f"{prefix}_{key}"] = float(val)
                
    return features

def extract_isa_features(attentions, tokens, text, tokenizer, inputs, max_sentences=5):
    """
    Extract features from Inter-Sentence Attention (ISA).
    Pads/truncates sentence interaction matrix to fixed size (max_sentences x max_sentences).
    """
    features = {}
    
    # Compute ISA
    result = compute_isa(attentions, tokens, text, tokenizer, inputs)
    isa_matrix = result.get("sentence_attention_matrix", np.array([]))
    
    # Flatten and pad/truncate matrix
    # We want a fixed feature vector size of max_sentences * max_sentences
    
    # Resize matrix to fixed square size
    padded_matrix = np.zeros((max_sentences, max_sentences))
    
    rows = min(isa_matrix.shape[0], max_sentences)
    cols = min(isa_matrix.shape[1], max_sentences)
    
    if rows > 0 and cols > 0:
        padded_matrix[:rows, :cols] = isa_matrix[:rows, :cols]
        
    # Flatten
    flat_matrix = padded_matrix.flatten()
    
    for i, val in enumerate(flat_matrix):
        features[f"ISA_flat_{i}"] = float(val)
        
    return features

def _collect_tree_leaves(attentions, tokens, root_idx, layer_idx, head_idx, max_depth, top_k):
    """
    Helper to traverse tree and collect leaf cumulative influence.
    Similar logic to server.main._generate_tree_data but purely recursive collection.
    """
    
    try:
        att_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
    except:
        return []

    leaves = []

    def traverse(current_idx, current_depth, current_value):
        # Stop condition: max depth reached
        if current_depth >= max_depth:
            leaves.append(current_value)
            return

        # Get attention weights for this token
        row = att_matrix[current_idx]
        
        # Get top-k indices
        top_indices = np.argsort(row)[-top_k:][::-1]
        
        # If no children (shouldn't happen in attention usually), it's a leaf
        if len(top_indices) == 0:
            leaves.append(current_value)
            return

        for child_idx in top_indices:
            raw_att = float(row[child_idx])
            # Cumulative influence
            child_value = current_value * raw_att if current_depth > 0 else raw_att
            traverse(child_idx, current_depth + 1, child_value)

    # Start traversal from root
    traverse(root_idx, 0, 1.0)
    return leaves

def extract_tree_features(attentions, tokens, max_leaves=50):
    """
    Extract influence values for leaves of the attention tree.
    We'll do this for the [CLS] token (index 0) as root, 
    using the last layer (usually most semantic) and mean head aggregation (or a specific head).
    
    To get "all leaves" in a stable vector, we'll run it for the last layer (L11 for base) 
    averaged across heads, or just Head 0. Let's do Average Head for stability.
    """
    features = {}
    
    # Aggregate heads for Tree (mean attention) to simplify
    # stacked shape: (num_layers, num_heads, seq, seq)
    stacked = torch.stack([att[0] for att in attentions])
    
    # Use Last Layer, Mean Head
    last_layer_att = stacked[-1].mean(dim=0).cpu().numpy() # (seq, seq)
    
    # Mock attentions structure for helper compatibility
    # shape (1, 1, seq, seq)
    # We need attentions[layer_idx] to be a tensor (batch, heads, seq, seq)
    mock_tensor = torch.tensor(last_layer_att).unsqueeze(0).unsqueeze(0)
    mock_att = [mock_tensor]
    
    # Extract leaves
    # Root = 0 ([CLS]), Depth=3, TopK=3 (matches UI default)
    # 3^3 = 27 leaves max usually, but branching can vary
    leaves = _collect_tree_leaves(mock_att, tokens, root_idx=0, layer_idx=0, head_idx=0, max_depth=3, top_k=5)
    
    # Pad/Truncate
    padded_leaves = pad_or_truncate_vector(leaves, max_leaves, padding_value=0.0)
    
    for i, val in enumerate(padded_leaves):
        features[f"Tree_Leaf_{i}"] = float(val)
        
    return features



def extract_raw_attention_features(attentions, max_seq_len=50):
    """
    Extract raw attention weights for all heads in all layers.
    Matrix Size: num_layers * num_heads * max_seq_len * max_seq_len
    
    WARNING: This produces a massive number of features.
    """
    features = {}
    
    for layer_idx, layer_att in enumerate(attentions):
        layer_att = layer_att[0].cpu().numpy() # (num_heads, seq, seq)
        num_heads = layer_att.shape[0]
        curr_seq_len = layer_att.shape[1]
        
        for head_idx in range(num_heads):
            att_matrix = layer_att[head_idx]
            
            # Pad or truncate matrix to fixed size (max_seq_len x max_seq_len)
            padded_matrix = np.zeros((max_seq_len, max_seq_len))
            
            rows = min(curr_seq_len, max_seq_len)
            cols = min(curr_seq_len, max_seq_len)
            
            if rows > 0 and cols > 0:
                padded_matrix[:rows, :cols] = att_matrix[:rows, :cols]
            
            # Flatten
            flat = padded_matrix.flatten()
            
            # Store features (e.g. AttMap_L0_H0_0, AttMap_L0_H0_1...)
            # To save space and time, we iterate efficiently or just vector assign?
            # Assigning 2500 individual keys per head (144 heads) = 360k keys is slow in Python dict.
            # We will try to be efficient with naming or just brute force it as requested.
            
            prefix = f"AttMap_L{layer_idx}_H{head_idx}"
            for i, val in enumerate(flat):
                # Optimization: Only store non-zero values? 
                # No, for a feature vector we need dense representation usually, 
                # or at least consistent keys. "Try all" means try all.
                features[f"{prefix}_{i}"] = float(val)
                
    return features


def extract_features_for_sentence(text, model_name, model_manager):
    """
    Main entry point to extract all features for a single sentence.
    """
    tokenizer, model, _ = model_manager.get_model(model_name)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions # tuple of (batch, heads, seq, seq)
        
    # Container for all features
    all_features = {}
    
    # 1. Global Metrics (Granular)
    all_features.update(extract_global_metrics(attentions))
    
    # 2. Head Specialization (Granular)
    all_features.update(extract_head_specialization_features(attentions, tokens, text))
    
    # 3. ISA features
    all_features.update(extract_isa_features(attentions, tokens, text, tokenizer, inputs))
    
    # 4. Tree features
    all_features.update(extract_tree_features(attentions, tokens))
    
    # 5. Raw Attention (Attributes requested: "try all")
    all_features.update(extract_raw_attention_features(attentions, max_seq_len=50))
    
    return all_features

