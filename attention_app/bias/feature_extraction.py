import numpy as np
import torch
from ..metrics import compute_all_attention_metrics, calculate_flow_change
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

            # Calculate Normalized Focus (0-1 range)
            # raw entropy / (n * log(n))
            seq_len = att_matrix.shape[0]
            max_entropy = seq_len * np.log(seq_len) if seq_len > 1 else 1.0
            focus_normalized = metrics['focus_entropy'] / max_entropy if max_entropy > 0 else 0.0
            features[f"{prefix}_focus_normalized"] = float(focus_normalized)
                
    return features

def extract_global_aggregate_metrics(attentions):
    """
    Extract Global Attention Metrics aggregated across ALL heads and layers.
    Provides a holistic view of the model's attention behavior.
    
    Args:
        attentions: List of tensors (one per layer), each (1, num_heads, seq_len, seq_len)
    """
    features = {}
    
    # Stack all attention matrices
    # Shape: (num_layers, num_heads, seq, seq)
    stacked = torch.stack([att[0] for att in attentions])
    
    # Aggregate across layers and heads (mean)
    # Shape: (seq, seq)
    global_att_matrix = stacked.mean(dim=(0, 1)).cpu().numpy()
    
    # Compute metrics on the aggregated matrix
    metrics = compute_all_attention_metrics(global_att_matrix)
    
    # Add with GAM_global prefix
    for key, val in metrics.items():
        features[f"GAM_global_{key}"] = float(val)
        
    # Calculate Global Normalized Focus
    seq_len = global_att_matrix.shape[0]
    max_entropy = seq_len * np.log(seq_len) if seq_len > 1 else 1.0
    focus_normalized = metrics['focus_entropy'] / max_entropy if max_entropy > 0 else 0.0
    features["GAM_global_focus_normalized"] = float(focus_normalized)
        
    # Calculate Flow Change (JSD between first and last layer)
    # Get list of layer arrays (num_heads, seq, seq)
    att_layers_full = [layer[0].cpu().numpy() for layer in attentions]
    features["GAM_global_flow_change"] = calculate_flow_change(att_layers_full)
    
    # Also add layer-wise aggregations (mean across all heads in each layer)
    for layer_idx, layer_att in enumerate(attentions):
        layer_mean_att = layer_att[0].mean(dim=0).cpu().numpy()  # (seq, seq)
        layer_metrics = compute_all_attention_metrics(layer_mean_att)
        
        for key, val in layer_metrics.items():
            features[f"GAM_layer{layer_idx}_{key}"] = float(val)
            
        # Calculate Layer Normalized Focus
        seq_len = layer_mean_att.shape[0]
        max_entropy = seq_len * np.log(seq_len) if seq_len > 1 else 1.0
        focus_normalized = layer_metrics['focus_entropy'] / max_entropy if max_entropy > 0 else 0.0
        features[f"GAM_layer{layer_idx}_focus_normalized"] = float(focus_normalized)
    
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



def extract_raw_attention_features(attentions):
    """
    Extract aggregated attention statistics for all heads in all layers.
    Returns multiple statistics per head: mean, max, min, std.
    
    Total features: num_layers * num_heads * 4 stats (e.g., 12 * 12 * 4 = 576 for BERT-base)
    """
    features = {}
    
    for layer_idx, layer_att in enumerate(attentions):
        layer_att = layer_att[0].cpu().numpy()  # (num_heads, seq, seq)
        num_heads = layer_att.shape[0]
        
        for head_idx in range(num_heads):
            att_matrix = layer_att[head_idx]
            
            # Extract multiple statistics from attention matrix
            prefix = f"AttMap_L{layer_idx}_H{head_idx}"
            
            features[f"{prefix}_mean"] = float(np.mean(att_matrix))
            features[f"{prefix}_max"] = float(np.max(att_matrix))
            features[f"{prefix}_min"] = float(np.min(att_matrix))
            features[f"{prefix}_std"] = float(np.std(att_matrix))
                
    return features

def extract_word_attention_statistics(attentions, tokens):
    """
    Extract statistics of total attention received by each word.
    Excludes special tokens ([CLS], [SEP], [PAD], etc).
    Returns size-independent features.
    
    Total attention = sum of all attention weights pointing TO each token
    across all layers and heads.
    """
    features = {}
    
    # Stack all attention matrices
    # Shape: (num_layers, num_heads, seq, seq)
    stacked = torch.stack([att[0] for att in attentions])
    
    # Sum across layers and heads to get total attention matrix
    # Shape: (seq, seq)
    total_att_matrix = stacked.sum(dim=(0, 1)).cpu().numpy()
    
    # Sum along source dimension to get total attention received by each token
    # total_att_matrix[i, j] = attention from token i to token j
    # We want sum over i (total attention TO each token j)
    word_attentions = total_att_matrix.sum(axis=0)  # Shape: (seq,)
    
    # Filter out special tokens
    # Common special tokens in BERT: [CLS], [SEP], [PAD]
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'}
    
    # Get attention for real words only
    filtered_attentions = []
    for i, token in enumerate(tokens):
        if token not in special_tokens and i < len(word_attentions):
            filtered_attentions.append(word_attentions[i])
    
    # If no real words (shouldn't happen), return zeros
    if len(filtered_attentions) == 0:
        features['word_att_mean'] = 0.0
        features['word_att_max'] = 0.0
        features['word_att_min'] = 0.0
        features['word_att_std'] = 0.0
        features['word_att_range'] = 0.0
        features['word_att_q25'] = 0.0
        features['word_att_q50'] = 0.0
        features['word_att_q75'] = 0.0
        return features
    
    # Convert to numpy array for statistics
    word_att_array = np.array(filtered_attentions)
    
    # Calculate statistics
    features['word_att_mean'] = float(np.mean(word_att_array))
    features['word_att_max'] = float(np.max(word_att_array))
    features['word_att_min'] = float(np.min(word_att_array))
    features['word_att_std'] = float(np.std(word_att_array))
    features['word_att_range'] = float(np.max(word_att_array) - np.min(word_att_array))
    features['word_att_q25'] = float(np.percentile(word_att_array, 25))
    features['word_att_q50'] = float(np.percentile(word_att_array, 50))  # median
    features['word_att_q75'] = float(np.percentile(word_att_array, 75))
    
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
    
    # 1. Global Metrics (Granular per head)
    all_features.update(extract_global_metrics(attentions))
    
    # 1b. Global Metrics (Aggregated across all heads/layers)
    all_features.update(extract_global_aggregate_metrics(attentions))
    
    # 2. Head Specialization (Granular)
    all_features.update(extract_head_specialization_features(attentions, tokens, text))
    
    # 3. ISA features
    all_features.update(extract_isa_features(attentions, tokens, text, tokenizer, inputs))
    
    # 4. Tree features
    all_features.update(extract_tree_features(attentions, tokens))
    
    # 5. Raw Attention (Aggregated per head)
    all_features.update(extract_raw_attention_features(attentions))
    
    # 6. Word Attention Statistics (size-independent distribution)
    all_features.update(extract_word_attention_statistics(attentions, tokens))
    
    return all_features

