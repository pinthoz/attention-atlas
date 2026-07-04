import logging
import string
import threading
import numpy as np
from functools import lru_cache
import spacy
import subprocess
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

_logger = logging.getLogger(__name__)

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

        # a: mean dist to OTHER members of the same cluster. Per Rousseeuw
        # (1987) — and sklearn — a singleton cluster has silhouette 0, and
        # the point itself is excluded from the mean (otherwise small
        # clusters get inflated scores and auto-K is biased toward many
        # tiny clusters: a singleton would score a=0 → s=1.0).
        same_mask = labels == label
        n_same = int(np.sum(same_mask))
        if n_same <= 1:
            silhouette_vals.append(0.0)
            continue
        dists_same = np.sqrt(np.sum((X[same_mask] - point) ** 2, axis=1))
        a = float(dists_same.sum()) / (n_same - 1)

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
    except Exception:
        X_2d = X_scaled[:, :2] # Fallback to first 2 dims
        
    # 3. K-Means (Manual)

    # Fixed seed: without it the fallback clustering (and therefore the
    # auto-K choice) changed between runs on the same sentence.
    rng = np.random.default_rng(42)

    def run_kmeans_step(data, k):
        n = data.shape[0]
        k = min(k, n)
        indices = rng.choice(n, k, replace=False)
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
# heavy_compute runs in a thread pool: without the lock two concurrent
# sessions could both see None and load the model twice.
_SPACY_LOCK = threading.Lock()


def get_spacy_model():
    """Load and cache spaCy model (thread-safe)."""
    global _SPACY_NLP
    with _SPACY_LOCK:
        if _SPACY_NLP is None:
            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except OSError:
                raise OSError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install it with: python -m spacy download en_core_web_sm"
                )
        return _SPACY_NLP


_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>"}


def align_spacy_to_bert_tokens(model_tokens, spacy_doc):
    """
    Align spaCy word-level tags to model subword tokens.

    Handles both tokenizer conventions:
      - BERT WordPiece: ``##`` prefix marks a continuation of the previous word.
      - GPT-2 BPE: ``Ġ`` prefix marks the START of a new word; tokens without
        it continue the previous word. ``Ċ`` is a newline marker.

    Args:
        model_tokens: List of sub-tokens from the attention model's tokenizer.
        spacy_doc: spaCy Doc object with POS and NER tags.

    Returns:
        Tuple of (pos_tags, ner_tags) aligned to model_tokens.
    """
    pos_tags = []
    ner_tags = []

    spacy_words = [token.text.lower() for token in spacy_doc]
    spacy_pos = [token.pos_ for token in spacy_doc]
    spacy_ner = [token.ent_type_ if token.ent_type_ else "O" for token in spacy_doc]

    is_gpt_style = any("Ġ" in tok for tok in model_tokens)

    spacy_idx = 0

    for raw_token in model_tokens:
        if raw_token in _SPECIAL_TOKENS:
            pos_tags.append("X")  # Special tag
            ner_tags.append("O")
            continue

        # Determine continuation vs new word per tokenizer convention, and
        # strip the marker characters before matching against spaCy text.
        if is_gpt_style:
            is_continuation = "Ġ" not in raw_token and "Ċ" not in raw_token \
                and len(pos_tags) > 0
            clean_token = raw_token.replace("Ġ", "").replace("Ċ", "").lower()
        else:
            is_continuation = raw_token.startswith("##")
            clean_token = raw_token.replace("##", "").lower()

        if is_continuation:
            # Continuation of previous word - inherit its tags
            if pos_tags:
                pos_tags.append(pos_tags[-1])
                ner_tags.append(ner_tags[-1])
            else:
                pos_tags.append("X")
                ner_tags.append("O")
            continue

        if not clean_token:
            # Pure marker token (e.g. a bare newline)
            pos_tags.append("X")
            ner_tags.append("O")
            continue

        # New word - find the corresponding spaCy token
        if spacy_idx < len(spacy_words):
            if (spacy_words[spacy_idx].startswith(clean_token)
                    or clean_token.startswith(spacy_words[spacy_idx])):
                pos_tags.append(spacy_pos[spacy_idx])
                ner_tags.append(spacy_ner[spacy_idx])
                if clean_token == spacy_words[spacy_idx]:
                    spacy_idx += 1
            else:
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


def compute_head_metrics(attention_matrix, tokens, pos_tags, ner_tags, is_gpt_style=None):
    """
    Compute all 7 behavioral metrics for a single attention head.

    Causal-model handling: for GPT-2-style tokenisations (detected via the
    ``Ġ`` marker) the first row of the attention matrix is degenerate —
    token 0 can only attend to itself, with weight 1.0 by construction — so
    the ``cls`` (first-token / sink focus) and ``self`` metrics exclude row 0
    to avoid a constant inflation artefact. For GPT-2 the ``cls`` metric
    measures attention received by the FIRST token (the attention-sink
    position), not a [CLS] summary token, which GPT-2 does not have.

    Args:
        attention_matrix: numpy array of shape (seq_len, seq_len)
        tokens: List of token strings
        pos_tags: List of POS tags aligned to tokens
        ner_tags: List of NER tags aligned to tokens

    Returns:
        Dict with keys: syntax, semantics, cls, punct, entities, long_range, self
    """
    seq_len = len(tokens)
    # Auto-detect from the Ġ marker unless the caller passes the flag
    # explicitly — word-aggregated tokens have the marker stripped, so the
    # aggregated path must pass is_gpt_style itself.
    if is_gpt_style is None:
        is_gpt_style = any("Ġ" in tok for tok in tokens)

    # 1. CLS / first-token focus - average attention to position 0.
    #    For causal models, exclude the degenerate first row (always 1.0).
    if is_gpt_style and seq_len > 1:
        cls_focus = float(attention_matrix[1:, 0].mean())
    else:
        cls_focus = float(attention_matrix[:, 0].mean())

    # 2. Self-attention - diagonal mean (excluding the forced 1.0 at row 0
    #    for causal models).
    diag = np.diag(attention_matrix)
    if is_gpt_style and seq_len > 1:
        self_att = float(diag[1:].mean())
    else:
        self_att = float(diag.mean())

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

    # 4. Punctuation focus — strip tokenizer markers before checking, so
    #    GPT-2 tokens like "Ġ," are recognised as punctuation.
    def _is_punct(tok):
        clean = tok.replace("Ġ", "").replace("Ċ", "").replace("##", "")
        return bool(clean) and all(c in string.punctuation for c in clean)

    punct_indices = [i for i, tok in enumerate(tokens) if _is_punct(tok)]
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


def compute_all_heads_specialization(attentions, tokens, text, is_gpt_style=None):
    """
    Compute normalized specialization metrics for all heads in all layers.

    Args:
        attentions: List of attention tensors from BERT (one per layer)
        tokens: List of BERT tokens
        text: Original input text
        is_gpt_style: True for causal models (row-0 exclusion in cls/self).
                      None = auto-detect from the Ġ marker; pass explicitly
                      for word-aggregated tokens where the marker is stripped.

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
            metrics = compute_head_metrics(att_matrix, tokens, pos_tags, ner_tags,
                                           is_gpt_style=is_gpt_style)
            layer_metrics[head_idx] = metrics
        
        # Normalize across heads in this layer
        normalized_metrics = normalize_metrics(layer_metrics)
        # Keep the raw (un-normalised) values alongside under flat
        # ``raw_<dim>`` keys. The per-layer min-max values only carry rank
        # information WITHIN a layer; cross-layer views (head clustering)
        # need absolute behaviour, otherwise every layer is forced to span
        # [0, 1] per dimension and clusters reflect within-layer rank
        # instead of head function.
        for h_idx, raw in layer_metrics.items():
            for dim, val in raw.items():
                normalized_metrics[h_idx][f"raw_{dim}"] = float(val)
        all_layers[layer_idx] = normalized_metrics

    return all_layers



def _assign_cluster_names(results, is_gpt_style=False):
    """
    Analyze cluster centroids and assign descriptive names based on dominant metrics.

    Naming happens in global z-score space (``z_metrics``) when available:
    a dimension is "dominant" if the cluster centroid stands out against the
    average head of the whole model, independent of each dimension's raw
    scale. Falls back to the per-layer-normalised ``metrics`` for cached
    results that predate ``z_metrics``.
    """
    if not results: return results

    import numpy as np

    # Use z-scores when every result carries them (post-2026-06 pipeline).
    use_z = all(r.get('z_metrics') for r in results)
    # "No dimension stands out" cutoff: z < 0.3 ≈ less than 0.3 SD above the
    # average head (Cohen-style small effect). The legacy cutoff of 0.25 was
    # tuned for the per-layer min-max scale.
    diffuse_cutoff = 0.3 if use_z else 0.25

    # 1. Group by cluster
    clusters = {}
    for r in results:
        c_id = r['cluster']
        if c_id not in clusters: clusters[c_id] = []
        clusters[c_id].append(r['z_metrics'] if use_z else r['metrics'])

    # 2. Compute Centroids
    cluster_names = {}

    # Metric to Label mapping (Priority order). GPT-2 has no [CLS] token:
    # the "cls" metric there measures attention received by the first token
    # (the attention-sink position), so label it honestly.
    metric_labels = {
        "self": "Self-Attention",
        "punct": "Separator/Punctuation",
        "cls": "First-Token (Sink)" if is_gpt_style else "CLS/Global",
        "syntax": "Syntactic",
        "entities": "Entity",
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
            if data["p_score"] < diffuse_cutoff:
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
                if data["p_score"] < diffuse_cutoff:
                     name = f"Diffuse ({sec_name})"
                elif data["s_score"] > 0.6 * data["p_score"] and data["s_score"] > 0: # Strong secondary
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

def compute_head_clusters(head_specialization_data, is_gpt_style=False):
    """
    Compute t-SNE coordinates and K-Means clusters for attention heads.

    Args:
        head_specialization_data: Output from compute_all_heads_specialization
                                 Dict {layer_idx: {head_idx: metrics_dict}}
        is_gpt_style: True for causal models without a [CLS] token — only
                      affects cluster naming (the "cls" dimension is labelled
                      as first-token sink focus).

    Returns:
        List of dicts: [{'layer': l, 'head': h, 'x': x, 'y': y, 'cluster': c, 'metrics': ...}, ...]
    """
    if not head_specialization_data:
        return []

    # 1. Flatten Data
    heads = []
    features = []
    dimensions = ["syntax", "semantics", "cls", "punct", "entities", "long_range", "self"]

    # Cluster on the RAW metrics (absolute behaviour) with a single global
    # standardisation below — not on the per-layer min-max values, which
    # encode rank within a layer and destroy cross-layer comparability.
    # ``raw_<dim>`` keys are written by compute_all_heads_specialization;
    # fall back to the normalised values for cached results that predate them.
    for l_idx, heads_map in head_specialization_data.items():
        for h_idx, metrics in heads_map.items():
            # Keep only the 7 display dimensions in the result payload so
            # downstream consumers (hover "Dominant" lookup) are unaffected.
            heads.append({
                'layer': l_idx,
                'head': h_idx,
                'metrics': {dim: metrics[dim] for dim in dimensions},
            })
            row = [metrics.get(f"raw_{dim}", metrics[dim]) for dim in dimensions]
            features.append(row)

    if not features:
        return []

    X = np.array(features)

    # Global z-scores: used for cluster naming so "dominant dimension" means
    # "stands out against the average head of the whole model", independent
    # of each dimension's raw scale.
    _mean = X.mean(axis=0)
    _std = X.std(axis=0)
    _std[_std == 0] = 1
    X_z = (X - _mean) / _std
    for i, head_info in enumerate(heads):
        head_info['z_metrics'] = {dim: float(X_z[i, j]) for j, dim in enumerate(dimensions)}
    
    
    
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
        _logger.warning("Sklearn clustering failed (%s), using manual fallback.", e)
        try:
            # Pass n_clusters=None to trigger auto-detection
            X_embedded, cluster_labels = _manual_pca_kmeans(X, n_clusters=None)
        except Exception as e2:
            _logger.warning("Manual clustering failed: %s", e2)
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
                'metrics': head_info['metrics'],
                'z_metrics': head_info.get('z_metrics'),
            })
            
    # 6. Assign Semantic Names
    results = _assign_cluster_names(results, is_gpt_style=is_gpt_style)

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
    return compute_head_metrics(att_matrix, tokens, pos_tags, ner_tags,
                                is_gpt_style=is_gpt2 or None)


__all__ = ["compute_all_heads_specialization", "get_linguistic_tags", "compute_head_clusters", "compute_head_specialization_metrics"]
