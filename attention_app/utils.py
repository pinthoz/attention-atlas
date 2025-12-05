"""Shared helper utilities for the Shiny attention explorer."""

from io import BytesIO
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def compute_influence_tree(attention_matrix, tokens, Q_matrix, K_matrix, d_k, root_token_idx, top_k=3, max_depth=3):
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
    visited = set()
    
    def build_tree(token_idx, depth, parent_idx=None):
        if depth >= max_depth or token_idx in visited:
            return None
        
        visited.add(token_idx)
        
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
        
        # Get top-k tokens (excluding already visited)
        top_indices = []
        for idx in np.argsort(attention_scores)[::-1]:
            if idx not in visited and len(top_indices) < top_k:
                top_indices.append(idx)
        
        # Build children recursively
        children = []
        for child_idx in top_indices:
            child_tree = build_tree(child_idx, depth + 1, token_idx)
            if child_tree:
                children.append(child_tree)
        
        return {
            'name': tokens[token_idx],
            'att': att_weight,
            'qk_sim': qk_sim,
            'token_idx': int(token_idx),
            'children': children
        }
    
    return build_tree(root_token_idx, 0)


__all__ = ["positional_encoding", "array_to_base64_img", "compute_influence_tree"]
