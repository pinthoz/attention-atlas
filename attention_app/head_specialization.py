import string
import numpy as np
from functools import lru_cache


# Cache for spaCy model to avoid reloading
_SPACY_NLP = None


def get_spacy_model():
    """Load and cache spaCy model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded, try to download it
            import subprocess
            import sys
            print("Downloading spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def align_spacy_to_bert_tokens(bert_tokens, spacy_doc):
    """
    Align spaCy word-level tags to BERT subword tokens.
    
    Args:
        bert_tokens: List of BERT tokens (including subwords like '##ing')
        spacy_doc: spaCy Doc object with POS and NER tags
    
    Returns:
        Tuple of (pos_tags, ner_tags) aligned to BERT tokens
    """
    pos_tags = []
    ner_tags = []
    
    # Build mapping from spaCy words to their tags
    spacy_words = [token.text.lower() for token in spacy_doc]
    spacy_pos = [token.pos_ for token in spacy_doc]
    spacy_ner = [token.ent_type_ if token.ent_type_ else "O" for token in spacy_doc]
    
    spacy_idx = 0
    current_word = ""
    
    for bert_token in bert_tokens:
        # Skip special tokens
        if bert_token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
            pos_tags.append("X")  # Special tag
            ner_tags.append("O")
            continue
        
        # Handle subword tokens
        if bert_token.startswith("##"):
            # Continuation of previous word - use same tags
            if pos_tags:
                pos_tags.append(pos_tags[-1])
                ner_tags.append(ner_tags[-1])
            else:
                pos_tags.append("X")
                ner_tags.append("O")
        else:
            # New word - find corresponding spaCy token
            clean_token = bert_token.lower().replace("ġ", "") # Handle GPT-2 prefix
            
            # Try to match with current spaCy word
            if spacy_idx < len(spacy_words):
                # Simple heuristic: if BERT token is prefix of spaCy word, it's a match
                if spacy_words[spacy_idx].startswith(clean_token) or clean_token.startswith(spacy_words[spacy_idx]):
                    pos_tags.append(spacy_pos[spacy_idx])
                    ner_tags.append(spacy_ner[spacy_idx])
                    
                    # Check if we've consumed the full spaCy word
                    if clean_token == spacy_words[spacy_idx]:
                        spacy_idx += 1
                else:
                    # Try next spaCy word
                    spacy_idx += 1
                    if spacy_idx < len(spacy_words):
                        pos_tags.append(spacy_pos[spacy_idx])
                        ner_tags.append(spacy_ner[spacy_idx])
                    else:
                        pos_tags.append("X")
                        ner_tags.append("O")
            else:
                pos_tags.append("X")
                ner_tags.append("O")
    
    return pos_tags, ner_tags


def get_linguistic_tags(tokens, text):
    """
    Extract POS tags and NER labels using spaCy, aligned to BERT tokens.
    
    Args:
        tokens: List of BERT tokens
        text: Original input text
    
    Returns:
        Tuple of (pos_tags, ner_tags) lists aligned to tokens
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    return align_spacy_to_bert_tokens(tokens, doc)


def compute_head_metrics(attention_matrix, tokens, pos_tags, ner_tags):
    """
    Compute all 7 behavioral metrics for a single attention head.
    
    Args:
        attention_matrix: numpy array of shape (seq_len, seq_len)
        tokens: List of token strings
        pos_tags: List of POS tags aligned to tokens
        ner_tags: List of NER tags aligned to tokens
    
    Returns:
        Dict with keys: syntax, semantics, cls, punct, entities, long_range, self
    """
    seq_len = len(tokens)
    
    # 1. CLS focus - average attention to [CLS] token (index 0)
    cls_focus = float(attention_matrix[:, 0].mean())
    
    # 2. Self-attention - diagonal mean
    self_att = float(np.diag(attention_matrix).mean())
    
    # 3. Long-range attention - distance >= 5
    long_range_mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) >= 5:
                long_range_mask[i, j] = True
    
    if long_range_mask.any():
        long_range_att = float(attention_matrix[long_range_mask].mean())
    else:
        long_range_att = 0.0
    
    # 4. Punctuation focus
    punct_indices = [i for i, tok in enumerate(tokens) if tok in string.punctuation]
    if punct_indices:
        punct_focus = float(attention_matrix[:, punct_indices].sum() / attention_matrix.sum())
    else:
        punct_focus = 0.0
    
    # 5. Entities focus (NER tags that are not "O")
    entity_indices = [i for i, tag in enumerate(ner_tags) if tag != "O"]
    if entity_indices:
        entity_focus = float(attention_matrix[:, entity_indices].sum() / attention_matrix.sum())
    else:
        entity_focus = 0.0
    
    # 6. Syntax focus - syntactic POS tags
    syntax_pos = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}
    syntax_indices = [i for i, tag in enumerate(pos_tags) if tag in syntax_pos]
    if syntax_indices:
        syntax_focus = float(attention_matrix[:, syntax_indices].sum() / attention_matrix.sum())
    else:
        syntax_focus = 0.0
    
    # 7. Semantics focus - semantic POS tags
    semantic_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
    semantic_indices = [i for i, tag in enumerate(pos_tags) if tag in semantic_pos]
    if semantic_indices:
        semantic_focus = float(attention_matrix[:, semantic_indices].sum() / attention_matrix.sum())
    else:
        semantic_focus = 0.0
    
    return {
        "syntax": syntax_focus,
        "semantics": semantic_focus,
        "cls": cls_focus,
        "punct": punct_focus,
        "entities": entity_focus,
        "long_range": long_range_att,
        "self": self_att
    }


def normalize_metrics(all_head_metrics):
    """
    Normalize metrics across all heads using min-max normalization.
    
    Args:
        all_head_metrics: Dict of {head_idx: metrics_dict}
    
    Returns:
        Dict of {head_idx: normalized_metrics_dict}
    """
    if not all_head_metrics:
        return {}
    
    # Collect all values for each dimension
    dimensions = ["syntax", "semantics", "cls", "punct", "entities", "long_range", "self"]
    dim_values = {dim: [] for dim in dimensions}
    
    for metrics in all_head_metrics.values():
        for dim in dimensions:
            dim_values[dim].append(metrics[dim])
    
    # Compute min and max for each dimension
    dim_ranges = {}
    for dim in dimensions:
        values = dim_values[dim]
        min_val = min(values)
        max_val = max(values)
        dim_ranges[dim] = (min_val, max_val)
    
    # Normalize each head's metrics
    normalized = {}
    for head_idx, metrics in all_head_metrics.items():
        norm_metrics = {}
        for dim in dimensions:
            min_val, max_val = dim_ranges[dim]
            if max_val - min_val > 1e-10:  # Avoid division by zero
                norm_metrics[dim] = (metrics[dim] - min_val) / (max_val - min_val)
            else:
                norm_metrics[dim] = 0.5  # All values identical
        normalized[head_idx] = norm_metrics
    
    return normalized


def compute_all_heads_specialization(attentions, tokens, text):
    """
    Compute normalized specialization metrics for all heads in all layers.
    
    Args:
        attentions: List of attention tensors from BERT (one per layer)
        tokens: List of BERT tokens
        text: Original input text
    
    Returns:
        Dict of {layer_idx: {head_idx: normalized_metrics_dict}}
    """
    # Get linguistic tags once for all heads
    pos_tags, ner_tags = get_linguistic_tags(tokens, text)
    
    all_layers = {}
    
    for layer_idx, layer_attention in enumerate(attentions):
        # layer_attention shape: (batch, num_heads, seq_len, seq_len)
        num_heads = layer_attention.shape[1]
        
        # Compute raw metrics for all heads in this layer
        layer_metrics = {}
        for head_idx in range(num_heads):
            att_matrix = layer_attention[0, head_idx].cpu().numpy()
            metrics = compute_head_metrics(att_matrix, tokens, pos_tags, ner_tags)
            layer_metrics[head_idx] = metrics
        
        # Normalize across heads in this layer
        normalized_metrics = normalize_metrics(layer_metrics)
        all_layers[layer_idx] = normalized_metrics
    
    return all_layers


__all__ = ["compute_all_heads_specialization", "get_linguistic_tags"]
