import string
import numpy as np
from functools import lru_cache
import spacy
import subprocess
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

def _manual_silhouette_score(X, labels):
    """
    Compute Silhouette Score manually using NumPy.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters <= 1 or n_clusters >= n_samples:
        return -1.0
        
    silhouette_vals = []
    for i in range(n_samples):
        point = X[i]
        label = labels[i]
        
        # a: mean dist to same cluster
        same_mask = labels == label
        other_in_cluster = X[same_mask]
        if len(other_in_cluster) > 1:
            a = np.mean(np.sqrt(np.sum((other_in_cluster - point)**2, axis=1)))
        else:
            a = 0.0
            
        # b: mean dist to nearest other cluster
        b = float('inf')
        for l in unique_labels:
            if l == label: continue
            other_cluster = X[labels == l]
            if len(other_cluster) == 0: continue
            dist = np.mean(np.sqrt(np.sum((other_cluster - point)**2, axis=1)))
            b = min(b, dist)
            
        if max(a, b) == 0:
            s_i = 0
        else:
            s_i = (b - a) / max(a, b)
        silhouette_vals.append(s_i)
        
    return np.mean(silhouette_vals)


def _manual_pca_kmeans(X, n_clusters=None):
    """
    Robust fallback implementation using only NumPy.
    Supports auto-K selection.
    """
    import numpy as np
    
    # 1. Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1 # Avoid division by zero
    X_scaled = (X - X_mean) / X_std
    
    # 2. PCA via SVD
    # X = U S V^T
    # Projection = U * S
    try:
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        X_2d = U[:, :2] * S[:2]
    except:
        X_2d = X_scaled[:, :2] # Fallback to first 2 dims
        
    # 3. K-Means (Manual)
    
    def run_kmeans_step(data, k):
        n = data.shape[0]
        k = min(k, n)
        indices = np.random.choice(n, k, replace=False)
        centers = data[indices]
        labels = np.zeros(n, dtype=int)
        
        for _ in range(20):
            dists = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels): break
            labels = new_labels
            for i in range(k):
                mask = labels == i
                if np.any(mask): centers[i] = data[mask].mean(axis=0)
        return labels

    if n_clusters is not None:
        best_labels = run_kmeans_step(X_scaled, n_clusters)
    else:
        # Auto-detect K
        best_score = -1.0
        best_labels = None
        # Check k=2 to 8
        for k in range(2, 9):
            lbls = run_kmeans_step(X_scaled, k)
            score = _manual_silhouette_score(X_scaled, lbls)
            if score > best_score:
                best_score = score
                best_labels = lbls
        if best_labels is None:
             best_labels = run_kmeans_step(X_scaled, 4)

    return X_2d, best_labels



# Cache for spaCy model to avoid reloading
_SPACY_NLP = None


def get_spacy_model():
    """Load and cache spaCy model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded, try to download it
            print("Downloading spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _SPACY_NLP = spacy.load("en_core_web_sm")
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
            clean_token = bert_token.lower()
            
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


    return all_layers



def _assign_cluster_names(results):
    """
    Analyze cluster centroids and assign descriptive names based on dominant metrics.
    """
    if not results: return results
    
    import numpy as np
    
    # 1. Group by cluster
    clusters = {}
    for r in results:
        c_id = r['cluster']
        if c_id not in clusters: clusters[c_id] = []
        clusters[c_id].append(r['metrics'])
        
    # 2. Compute Centroids
    cluster_names = {}
    
    # Metric to Label mapping (Priority order)
    # If score > 0.4 implies significance
    metric_labels = {
        "self": "Self-Attention",
        "punct": "Separator/Punctuation",
        "cls": "CLS/Global",
        "syntax": "Syntactic",
        "position": "Positional/Locality",
        "semantics": "Semantic",
        "long_range": "Long-Range"
    }
    
    
    # helper to get name from dim
    def get_dim_name(dim, score=0):
        if dim == "syntax": return "Syntactic"
        if dim == "position": return "Positional"
        if dim == "semantics": return "Semantic"
        if dim == "struct": return "Structural"
        if dim == "diffuse": return "Diffuse"
        if dim in ["cls", "sep", "punct"]: return "Token Ops"
        return metric_labels.get(dim, dim.title())

    # Temporary storage for name candidates
    # c_id -> (primary_dim, secondary_dim, primary_score, secondary_score)
    candidates = {}

    for c_id, metrics_list in clusters.items():
        # Compute average score for each metric dimension
        dims = metrics_list[0].keys()
        centroid = {d: np.mean([m[d] for m in metrics_list]) for d in dims}
        
        # Sort dimensions by score descending
        sorted_dims = sorted(centroid.items(), key=lambda x: x[1], reverse=True)
        
        p_dim, p_score = sorted_dims[0]
        s_dim, s_score = sorted_dims[1] if len(sorted_dims) > 1 else (None, 0)
        
        candidates[c_id] = {
            "p_dim": p_dim, "p_score": p_score,
            "s_dim": s_dim, "s_score": s_score,
            "centroid": centroid
        }

    # Resolve collisions
    # Group clusters by their primary dimension
    by_primary = {}
    for c_id, data in candidates.items():
        p_dim = data["p_dim"]
        if p_dim not in by_primary: by_primary[p_dim] = []
        by_primary[p_dim].append(c_id)
        
    for p_dim, c_ids in by_primary.items():
        if len(c_ids) == 1:
            # Unique primary - simple name
            c_id = c_ids[0]
            data = candidates[c_id]
            if data["p_score"] < 0.25:
                name = "Diffuse/Noise"
            else:
                name = get_dim_name(p_dim) + " Specialists"
            cluster_names[c_id] = name
        else:
            # Collision - use secondary dimension
            # Sort colliding clusters by their secondary score to differentiate? 
            # Or just name them by secondary.
            for c_id in c_ids:
                data = candidates[c_id]
                base_name = get_dim_name(p_dim)
                sec_name = get_dim_name(data["s_dim"])
                
                # Check variance - if secondary is also close?
                if data["p_score"] < 0.25:
                     name = f"Diffuse ({sec_name})"
                elif data["s_score"] > 0.6 * data["p_score"]: # Strong secondary
                     name = f"{base_name} & {sec_name}"
                else:
                     name = f"{base_name} ({sec_name})"
                     
                cluster_names[c_id] = name
    
    # Final cleanup: Check for exact string duplicates again (rare but possible if secondary same)
    # and append ID if needed
    name_counts = {}
    for name in cluster_names.values():
        name_counts[name] = name_counts.get(name, 0) + 1
        
    for c_id in cluster_names:
        name = cluster_names[c_id]
        if name_counts[name] > 1:
             # Append ID to disambiguate
             cluster_names[c_id] = f"{name} {c_id}"

    # 3. Apply names back to results
    for r in results:
        r['cluster_name'] = cluster_names.get(r['cluster'], f"Cluster {r['cluster']}")
        
    return results

def compute_head_clusters(head_specialization_data):
    """
    Compute t-SNE coordinates and K-Means clusters for attention heads.
    
    Args:
        head_specialization_data: Output from compute_all_heads_specialization
                                 Dict {layer_idx: {head_idx: metrics_dict}}
    
    Returns:
        List of dicts: [{'layer': l, 'head': h, 'x': x, 'y': y, 'cluster': c, 'metrics': ...}, ...]
    """
    if not head_specialization_data:
        return []

    # 1. Flatten Data
    heads = []
    features = []
    dimensions = ["syntax", "semantics", "cls", "punct", "entities", "long_range", "self"]
    
    for l_idx, heads_map in head_specialization_data.items():
        for h_idx, metrics in heads_map.items():
            heads.append({'layer': l_idx, 'head': h_idx, 'metrics': metrics})
            row = [metrics[dim] for dim in dimensions]
            features.append(row)
            
    if not features:
        return []
        
    X = np.array(features)
    
    X = np.array(features)
    
    
    
    X_embedded = None
    cluster_labels = None
    
    # Try Sklearn (Safe Mode) - Re-enabled with method='exact'
    force_fallback = False
    
    try:
        if force_fallback:
             raise RuntimeError("Forcing manual fallback")

        with threadpool_limits(limits=1): 
            # 2. Preprocessing (Standard Scaling)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 3. t-SNE (2D Projection)
            n_samples = X.shape[0]
            perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
            
            # method='exact' avoids Barnes-Hut (OpenMP source of deadlock)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto', method='exact')
            X_embedded = tsne.fit_transform(X_scaled)

            # 4. K-Means Clustering (Auto-Detect K)
            # Use Silhouette Score to find optimal K
            best_score = -1.0
            best_k = 4
            best_labels = None
            
            for k in range(2, 9):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbls = kmeans.fit_predict(X_scaled)
                # Use sklearn silhouette if available, or manual? 
                # Since we are in sklearn block, use sklearn impl? 
                # Actually, let's use the manual one we wrote or just trust K=4?
                # The user LIKED the auto-K.
                # Let's keep the manual silhouette logic or implement it here for sklearn.
                
                # We can reuse _manual_silhouette_score for consistence even with sklearn labels
                score = _manual_silhouette_score(X_scaled, lbls)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = lbls
            
            if best_labels is None: # Should not happen
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                best_labels = kmeans.fit_predict(X_scaled)
                
            cluster_labels = best_labels 
        
    except Exception as e:
        print(f"Sklearn clustering failed ({e}), using manual fallback.")
        try:
            # Pass n_clusters=None to trigger auto-detection
            X_embedded, cluster_labels = _manual_pca_kmeans(X, n_clusters=None)
        except Exception as e2:
            print(f"Manual clustering failed: {e2}")
            return []

    # 5. Combine Results
    results = []
    if X_embedded is not None and cluster_labels is not None:
        for i, head_info in enumerate(heads):
            results.append({
                'layer': head_info['layer'],
                'head': head_info['head'],
                'x': float(X_embedded[i, 0]),
                'y': float(X_embedded[i, 1]),
                'cluster': int(cluster_labels[i]),
                'metrics': head_info['metrics']
            })
            
    # 6. Assign Semantic Names
    results = _assign_cluster_names(results)
        
    return results



def compute_head_specialization_metrics(tokens, attentions, layer_idx, head_idx, text, is_gpt2=False):
    """
    Compute specialization metrics for a specific single head.
    Wrapper around compute_head_metrics with tag extraction.
    """
    # 1. Get tags
    pos_tags, ner_tags = get_linguistic_tags(tokens, text)
    
    # 2. Get matrix
    # attentions is list of tensors (batch, num_heads, seq_len, seq_len)
    att_matrix = attentions[layer_idx][0, head_idx].cpu().numpy()
    
    # 3. Compute
    return compute_head_metrics(att_matrix, tokens, pos_tags, ner_tags)


__all__ = ["compute_all_heads_specialization", "get_linguistic_tags", "compute_head_clusters", "compute_head_specialization_metrics"]
