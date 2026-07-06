import torch
import numpy as np
import nltk
from typing import List, Dict, Tuple

# Ensure nltk data is downloaded
# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt_tab', quiet=True)

def _token_to_sent_via_offsets(text, tokens, tokenizer, char_to_sent):
    """Map each token to its sentence index using tokenizer char offsets.

    Returns the per-token sentence list, or ``None`` when offsets are
    unavailable or the re-encoded token sequence does not reproduce
    ``tokens`` exactly (word-aggregated pseudo-tokens, sentence-pair
    encodings with a mid-sequence [SEP], slow tokenizers) — the caller then
    uses the character-matching heuristic.
    """
    try:
        enc = tokenizer(text, return_offsets_mapping=True,
                        truncation=True, max_length=512)
        enc_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets = enc["offset_mapping"]
    except Exception:
        return None
    if list(enc_tokens) != list(tokens) or len(offsets) != len(tokens):
        return None

    token_to_sent = []
    for (start, end) in offsets:
        if end <= start:  # special tokens carry (0, 0)
            token_to_sent.append(-1)
            continue
        mid = start + (end - start) // 2
        token_to_sent.append(char_to_sent.get(mid, char_to_sent.get(start, -1)))
    return token_to_sent


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
    
    # 3a. Preferred path: exact character offsets from the fast tokenizer.
    #     The greedy character matching below is a heuristic that can lose
    #     sync on curly quotes / accents / emoji; offsets cannot. Only
    #     usable when re-encoding the plain text reproduces the exact token
    #     sequence we were given (it does NOT for BERT sentence-pair
    #     encoding, whose mid-sequence [SEP] changes the layout — that case
    #     falls through to the heuristic).
    token_to_sent = _token_to_sent_via_offsets(text, tokens, tokenizer, char_to_sent)
    if token_to_sent is not None:
        final_boundaries = []
        for s_idx in range(len(sentences)):
            found = next((t for t, s in enumerate(token_to_sent) if s == s_idx), None)
            if found is not None:
                final_boundaries.append(found)
            else:
                final_boundaries.append(final_boundaries[-1] if final_boundaries else 0)
        return sentences, final_boundaries

    # 3b. Fallback: map tokens to sentences using greedy character matching
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
                # Resync: an exact match at the cursor failed (curly quote,
                # accent normalisation, unicode artefact...). Without
                # advancing, EVERY subsequent token would also fail — a
                # cascade that silently degrades the whole mapping. Search
                # ahead in a small window and resync the cursor if found.
                found = text.lower().find(clean_tok.lower(), current_text_pos,
                                          current_text_pos + 50 + match_len)
                if found != -1:
                    center_pos = found + match_len // 2
                    token_to_sent.append(char_to_sent.get(center_pos, -1))
                    current_text_pos = found + match_len
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


_ISA_SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|endoftext|>", "<|pad|>"}


def _stack_attentions(attentions) -> torch.Tensor:
    """Stack per-layer attention tensors into (layers, heads, seq, seq)."""
    if isinstance(attentions[0], tuple):
        return torch.stack([att[0] for att in attentions], dim=0)
    if len(attentions[0].shape) == 4:  # (batch, heads, seq, seq)
        return torch.stack([att[0] for att in attentions], dim=0)
    return torch.stack(attentions, dim=0)


def _sentence_token_indices(tokens: List[str], sentence_boundaries_ids: List[int],
                            num_sentences: int) -> List[List[int]]:
    """Per-sentence token index lists, EXCLUDING special tokens.

    Contiguous ranges would otherwise sweep in [SEP] (both the final one and
    the mid-sequence one from sentence-pair encoding) — and attention to
    [SEP] is a well-documented sink (Clark et al., 2019), which would
    distort the affected sentence's ISA row/column.
    """
    sent_indices: List[List[int]] = []
    for k in range(num_sentences):
        start = sentence_boundaries_ids[k]
        end = sentence_boundaries_ids[k + 1] if k < num_sentences - 1 else len(tokens)
        sent_indices.append(
            [i for i in range(start, end) if tokens[i] not in _ISA_SPECIAL]
        )
    return sent_indices


def _isa_pair_score(stacked: torch.Tensor, row_ix: List[int], col_ix: List[int],
                    aggregation_method: str) -> float:
    """Score one sentence pair (Sa=row queries, Sb=col keys).

    "mean" (default since 2026-06-12) — attention MASS:
        ᾱ = mean over layers and heads → (seq, seq)
        ISA(Sa, Sb) = (1/|Sa|) Σ_{i∈Sa} Σ_{j∈Sb} ᾱ[i, j]
        = the average fraction of Sa's attention that lands on Sb.
        Bounded [0, 1], robust, directly interpretable. Rows do not sum
        to 1 across sentences because attention to special tokens (the
        [CLS]/[SEP] sink mass) is deliberately excluded.

    "max" — legacy salience measure (pre-2026-06-12 default):
        ISA(Sa, Sb) = max over layers, heads and token pairs of α.
        A single strong token link sets the whole pair's score; kept as
        an explicit option for "does ANY strong link exist?" questions.

    "last_layer" — mass computed on the last layer only (mean over heads).
    """
    if not row_ix or not col_ix:
        return 0.0
    if aggregation_method == "max":
        A = torch.max(stacked, dim=0)[0]            # (heads, seq, seq)
        sub = A[:, row_ix][:, :, col_ix]
        return float(torch.max(sub).item()) if sub.numel() else 0.0
    if aggregation_method == "last_layer":
        A = stacked[-1].mean(dim=0)                 # (seq, seq)
    elif aggregation_method == "mean":
        A = stacked.mean(dim=(0, 1))                # (seq, seq)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    sub = A[row_ix][:, col_ix]
    if sub.numel() == 0:
        return 0.0
    # Mass: sum over keys (Sb), mean over queries (Sa).
    return float(sub.sum(dim=1).mean().item())


def compute_isa(attentions, tokens: List[str], text: str, tokenizer, inputs,
                model_type: str = "bert",
                aggregation_method: str = "mean") -> Dict:
    """
    Compute Inter-Sentence Attention (ISA) matrix.

    Default aggregation is attention MASS ("mean", since 2026-06-12):
    ISA(Sa, Sb) = average fraction of Sa's attention received by Sb,
    with attention to special tokens excluded. The previous default
    ("max": max over layers, heads and token pairs) is available via
    ``aggregation_method="max"`` — it measures the existence of a single
    strong link, not coupling, and is outlier-dominated by construction.

    Args:
        attentions: Tuple of tensors (num_layers, batch, num_heads, seq_len, seq_len)
                   or list of tensors.
        tokens: List of token strings.
        text: Original input text.
        tokenizer: The tokenizer used.
        inputs: The inputs dict (optional, for offset mapping if we could use it).
        model_type: "bert" for bidirectional models, "gpt" for causal models
        aggregation_method: "mean" (default), "max", or "last_layer"

    Returns:
        dict: {
            "sentence_texts": List[str],
            "sentence_boundaries_ids": List[int],
            "sentence_attention_matrix": np.ndarray (num_sentences, num_sentences),
            "model_type": str,
            "is_causal": bool,
            "aggregation_method": str
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
            "is_causal": model_type == "gpt",
            "aggregation_method": aggregation_method
        }

    if num_sentences == 1:
        # If only one sentence, return trivial result (display convention)
        return {
            "sentence_texts": sentence_texts,
            "sentence_boundaries_ids": sentence_boundaries_ids,
            "sentence_attention_matrix": np.ones((1, 1)),
            "model_type": model_type,
            "is_causal": model_type == "gpt",
            "aggregation_method": aggregation_method
        }

    # 2. Stack attentions: (num_layers, num_heads, seq_len, seq_len)
    stacked_attentions = _stack_attentions(attentions)

    # 3. Aggregate Token Attention -> Sentence Attention
    #    (see _isa_pair_score for the per-method definitions)
    sent_indices = _sentence_token_indices(tokens, sentence_boundaries_ids, num_sentences)

    isa_matrix = np.zeros((num_sentences, num_sentences))
    for r_idx, row_ix in enumerate(sent_indices):
        for c_idx, col_ix in enumerate(sent_indices):
            isa_matrix[r_idx, c_idx] = _isa_pair_score(
                stacked_attentions, row_ix, col_ix, aggregation_method
            )

    # Final safety check for NaNs
    isa_matrix = np.nan_to_num(isa_matrix, nan=0.0)

    # Causality metadata. The caller's model_type is authoritative (it is
    # derived from the tokenizer, which cannot be wrong about the
    # architecture); the matrix heuristic is only a fallback for callers
    # that pass the default. NOTE: the previous heuristic averaged
    # np.triu(...) over the FULL n×n matrix (diluting by the zero diagonal
    # and lower triangle), so a bidirectional model with weak inter-sentence
    # coupling and many sentences could be misflagged as causal.
    if model_type == "gpt":
        is_causal = True
    elif model_type == "bert":
        is_causal = False
    else:
        # Unknown model type: mean over the strict upper-triangle cells only.
        iu = np.triu_indices(num_sentences, k=1)
        is_causal = bool(iu[0].size) and float(np.mean(isa_matrix[iu])) < 0.01

    return {
        "sentence_texts": sentence_texts,
        "sentence_boundaries_ids": sentence_boundaries_ids,
        "sentence_attention_matrix": isa_matrix,
        "model_type": model_type,
        "is_causal": is_causal,
        "aggregation_method": aggregation_method
    }


def get_sentence_token_attention(attentions, tokens: List[str], sent_x_idx: int, sent_y_idx: int,
                                   sentence_boundaries_ids: List[int],
                                   aggregation_method: str = "mean") -> Tuple[np.ndarray, List[str], int]:
    """
    Extract token-level attention for a specific sentence pair.
    Works for both bidirectional (BERT) and causal (GPT) models.

    The token matrix uses the SAME layer/head aggregation as the ISA matrix
    (default: mean over layers and heads), so the drill-down explains the
    cell the user clicked. ``aggregation_method="max"`` reproduces the
    legacy max-over-layers-and-heads view.

    Args:
        attentions: Tuple of attention tensors (num_layers, batch, num_heads, seq_len, seq_len)
        tokens: List of token strings
        sent_x_idx: Index of source sentence X (target/query)
        sent_y_idx: Index of source sentence Y (source/key)
        sentence_boundaries_ids: List of sentence start indices
        aggregation_method: "mean" (default), "max", or "last_layer"

    Returns:
        attention_data: numpy array (len(sent_x_tokens), len(sent_y_tokens))
        tokens_combined: concatenated list of [sent_x_tokens, sent_y_tokens]
        sentence_b_start: index where sent_y tokens start in tokens_combined
    """
    stacked_attentions = _stack_attentions(attentions)
    # stacked_attentions shape: (num_layers, num_heads, seq_len, seq_len)

    if aggregation_method == "max":
        # Legacy: max over layers, then heads
        A_tok = torch.max(torch.max(stacked_attentions, dim=0)[0], dim=0)[0].cpu().numpy()
    elif aggregation_method == "last_layer":
        A_tok = stacked_attentions[-1].mean(dim=0).cpu().numpy()
    else:  # "mean"
        A_tok = stacked_attentions.mean(dim=(0, 1)).cpu().numpy()

    # Get token index lists for the two sentences, excluding special tokens
    # so the drill-down stays consistent with the ISA matrix (which also
    # excludes them; see compute_isa).
    num_sentences = len(sentence_boundaries_ids)

    def get_indices(idx):
        start = sentence_boundaries_ids[idx]
        if idx < num_sentences - 1:
            end = sentence_boundaries_ids[idx + 1]
        else:
            end = len(tokens)
        return [i for i in range(start, end) if tokens[i] not in _ISA_SPECIAL]

    ix_x = get_indices(sent_x_idx)
    ix_y = get_indices(sent_y_idx)

    # Extract submatrix: rows=sent_x (query), cols=sent_y (key)
    sub_att = A_tok[np.ix_(ix_x, ix_y)] if ix_x and ix_y else np.zeros((0, 0))

    # Extract tokens
    toks_x = [tokens[i] for i in ix_x]
    toks_y = [tokens[i] for i in ix_y]
    
    # Combine tokens for visualization [sent_x_tokens, sent_y_tokens]
    tokens_combined = list(toks_x) + list(toks_y)
    sentence_b_start = len(toks_x)
    
    return sub_att, tokens_combined, sentence_b_start


# Additional utility functions for GPT-specific analysis

def compute_isa_with_aggregation(attentions, tokens: List[str], text: str, tokenizer, inputs,
                                  model_type: str = "bert",
                                  aggregation_method: str = "mean") -> Dict:
    """
    Compute ISA with an explicit aggregation method.

    Thin wrapper over compute_isa, kept for backward compatibility. The
    methods now have honest semantics (see _isa_pair_score): the previous
    implementation's "mean" averaged over layers only and still took the
    max over heads and token pairs.

    Args:
        aggregation_method: "mean" (default; attention mass), "max"
            (legacy salience), or "last_layer" (mass on last layer).
    """
    return compute_isa(attentions, tokens, text, tokenizer, inputs,
                       model_type=model_type,
                       aggregation_method=aggregation_method)


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
