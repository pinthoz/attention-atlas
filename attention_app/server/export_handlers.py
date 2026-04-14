"""Download / export handlers for the attention-section CSV and JSON exports.

Extracted from ``main.py`` to reduce monolith size.  Each handler is
registered via the caller-supplied ``auto_save_download`` decorator,
which wraps ``@render.download`` and also saves a copy to the project
folder.
"""

import json
import logging

import numpy as np

from .renderers import get_influence_tree_data

_logger = logging.getLogger(__name__)


def register_export_handlers(
    input,
    auto_save_download,
    get_active_result,
    global_norm_mode,
    global_metrics_mode,
    global_rollout_layers,
    get_normalized_attention,
):
    """Wire up all attention-section download handlers.

    Parameters are the reactive values / helpers that live inside
    ``server()`` — passed in to avoid closure coupling.
    """

    # ── Attention heatmap CSV ───────────────────────────────────────────

    @auto_save_download("attention_heatmap", "csv")
    def export_attention_metrics_dashboard():
        res = get_active_result()
        if not res:
            return None
        try:
            tokens, attentions, encoder_model = (
                res.tokens, res.attentions, res.encoder_model,
            )
            try: layer_idx = int(input.global_layer())
            except Exception: layer_idx = 0
            try: head_idx = int(input.global_head())
            except Exception: head_idx = 0

            norm_mode = global_norm_mode.get()
            use_global = global_metrics_mode.get() == "all"
            use_all_layers = global_rollout_layers.get() == "all"
            is_causal = not hasattr(encoder_model, "encoder")

            if use_global:
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                raw_att = np.mean(att_layers, axis=(0, 1))
            else:
                raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

            att = get_normalized_attention(
                raw_att, attentions, layer_idx, norm_mode,
                is_causal=is_causal, use_all_layers=use_all_layers,
            )
            clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]

            import pandas as pd
            df = pd.DataFrame(att, index=clean_tokens, columns=clean_tokens)
            yield df.to_csv(index=True)
        except Exception:
            _logger.exception("Error exporting metrics")
            yield "Error exporting metrics"

    # ── Scaled attention (Q·K scores) — A / B ───────────────────────────

    def _export_scaled_attention(suffix=""):
        res = get_active_result(suffix)
        if not res:
            yield "No data available"
            return
        try:
            tokens, attentions, encoder_model = (
                res.tokens, res.attentions, res.encoder_model,
            )
            try: layer_idx = int(input.global_layer())
            except Exception: layer_idx = 0
            try: head_idx = int(input.global_head())
            except Exception: head_idx = 0

            norm_mode = global_norm_mode.get()
            use_global = global_metrics_mode.get() == "all"
            is_causal = not hasattr(encoder_model, "encoder")

            if use_global:
                att_layers = [layer[0].cpu().numpy() for layer in attentions]
                raw_att = np.mean(att_layers, axis=(0, 1))
            else:
                raw_att = attentions[layer_idx][0, head_idx].cpu().numpy()

            att = get_normalized_attention(
                raw_att, attentions, layer_idx, norm_mode, is_causal=is_causal,
            )
            clean_tokens = [t.replace("##", "").replace("Ġ", "") for t in tokens]

            import pandas as pd
            df = pd.DataFrame(att, index=clean_tokens, columns=clean_tokens)
            df.index.name = "Query_Token"
            yield df.to_csv(index=True)
        except Exception:
            _logger.exception("Error exporting data")
            yield "Error exporting data"

    @auto_save_download("scaled_attention", "csv", data_type="qkv_scores")
    def export_scaled_attention():
        """Export scaled dot-product attention data as CSV."""
        yield from _export_scaled_attention()

    @auto_save_download("scaled_attention", "csv", is_b=True, data_type="qkv_scores")
    def export_scaled_attention_B():
        """Export scaled dot-product attention data as CSV for Model B."""
        yield from _export_scaled_attention("_B")

    # ── Head specialization CSV ─────────────────────────────────────────

    def get_head_spec_csv(is_b=False):
        res = get_active_result("_B") if is_b else get_active_result()
        if not res:
            return None
        tokens, attentions, tokenizer, encoder_model = (
            res.tokens, res.attentions, res.tokenizer, res.encoder_model,
        )
        try: layer_idx = int(input.global_layer())
        except Exception: layer_idx = 0
        try: head_idx = int(input.global_head())
        except Exception: head_idx = 0

        if hasattr(tokenizer, "convert_tokens_to_string"):
            text = tokenizer.convert_tokens_to_string(tokens)
        else:
            text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))

        from ..head_specialization import compute_all_heads_specialization
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            nltk.download("universal_tagset")

        all_metrics = compute_all_heads_specialization(attentions, tokens, text)
        rows = []
        for l_idx, heads in all_metrics.items():
            for h_idx, metrics in heads.items():
                rows.append({"Layer": l_idx, "Head": h_idx, **metrics})

        import pandas as pd
        df = pd.DataFrame(rows)
        cols = ["Layer", "Head"] + [c for c in df.columns if c not in ("Layer", "Head")]
        df = df[cols]
        return df.to_csv(index=False)

    @auto_save_download("head_specialization", "csv")
    def export_head_spec_unique():
        yield get_head_spec_csv()

    @auto_save_download("head_specialization_legacy", "csv")
    def export_head_spec_unique_legacy():
        yield get_head_spec_csv()

    @auto_save_download("head_specialization", "csv", is_b=True)
    def export_head_spec_unique_B():
        yield get_head_spec_csv(is_b=True)

    # ── Attention dependency tree CSV ───────────────────────────────────

    def get_tree_csv(res, suffix="", all_layers_heads=True):
        if not res:
            return None
        try: root_idx = int(input.global_focus_token())
        except Exception: root_idx = 0
        if root_idx == -1:
            root_idx = 0
        try: top_k = int(input.global_topk())
        except Exception: top_k = 3

        norm_mode = global_norm_mode.get()

        import pandas as pd

        tokens = res.tokens
        attentions = res.attentions
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1] if num_layers > 0 else 12

        all_rows = []

        def traverse(node, layer_label, head_label, mode_label,
                     depth=0, parent_name="ROOT"):
            name = node.get("name", "Unknown")
            att = node.get("att", 0.0)
            if depth > 0:
                all_rows.append({
                    "Layer": layer_label, "Head": head_label,
                    "Mode": mode_label, "Parent": parent_name,
                    "Child": name, "Attention": att, "Depth": depth,
                })
            for child in node.get("children", []):
                traverse(child, layer_label, head_label, mode_label,
                         depth + 1, name)

        if all_layers_heads:
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    tree_data = get_influence_tree_data(
                        res, layer_idx, head_idx, root_idx,
                        top_k, top_k, norm_mode, att_matrix_override=None,
                    )
                    if tree_data:
                        traverse(tree_data, layer_idx, head_idx, "per_head")

                layer_att = attentions[layer_idx]
                if hasattr(layer_att, "cpu"):
                    avg_att = layer_att.mean(dim=1).squeeze().cpu().numpy()
                else:
                    avg_att = np.mean(layer_att, axis=1).squeeze()
                tree_data_global = get_influence_tree_data(
                    res, layer_idx, 0, root_idx, top_k, top_k,
                    norm_mode, att_matrix_override=avg_att,
                )
                if tree_data_global:
                    traverse(tree_data_global, layer_idx, "all", "global")
        else:
            try: layer_idx = int(input.global_layer())
            except Exception: layer_idx = 0
            try: head_idx = int(input.global_head())
            except Exception: head_idx = 0

            tree_data = get_influence_tree_data(
                res, layer_idx, head_idx, root_idx,
                top_k, top_k, norm_mode, att_matrix_override=None,
            )
            if tree_data:
                traverse(tree_data, layer_idx, head_idx, "per_head")

        if not all_rows:
            return None
        df = pd.DataFrame(all_rows)
        return df.to_csv(index=False)

    @auto_save_download("attention_tree", "csv", data_type="all_layers_heads")
    def export_tree_data():
        csv_content = get_tree_csv(get_active_result(), all_layers_heads=True)
        yield csv_content or "No data available"

    @auto_save_download("attention_tree", "csv", is_b=True, data_type="all_layers_heads")
    def export_tree_data_B():
        csv_content = get_tree_csv(get_active_result("_B"), suffix="_B",
                                   all_layers_heads=True)
        yield csv_content or "No data available"

    # ── Top-K attention targets CSV ─────────────────────────────────────

    def get_topk_attention_csv(res, suffix=""):
        if not res:
            return None
        try: top_k = int(input.global_topk())
        except Exception: top_k = 3

        import pandas as pd

        tokens = res.tokens
        attentions = res.attentions
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1] if num_layers > 0 else 12
        num_tokens = len(tokens)

        all_rows = []

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                layer_att = attentions[layer_idx][0, head_idx]
                att_matrix = (layer_att.cpu().numpy()
                              if hasattr(layer_att, "cpu")
                              else np.array(layer_att))
                for token_idx in range(num_tokens):
                    att_row = att_matrix[token_idx]
                    top_indices = np.argsort(att_row)[::-1][:top_k]
                    for rank, target_idx in enumerate(top_indices):
                        all_rows.append({
                            "Token_Idx": token_idx,
                            "Token": tokens[token_idx],
                            "Layer": layer_idx, "Head": head_idx,
                            "Mode": "per_head", "Rank": rank + 1,
                            "Target_Idx": target_idx,
                            "Target_Token": tokens[target_idx],
                            "Attention": float(att_row[target_idx]),
                        })

        for layer_idx in range(num_layers):
            layer_att = attentions[layer_idx][0]
            att_matrix = (layer_att.mean(dim=0).cpu().numpy()
                          if hasattr(layer_att, "cpu")
                          else np.mean(layer_att, axis=0))
            for token_idx in range(num_tokens):
                att_row = att_matrix[token_idx]
                top_indices = np.argsort(att_row)[::-1][:top_k]
                for rank, target_idx in enumerate(top_indices):
                    all_rows.append({
                        "Token_Idx": token_idx,
                        "Token": tokens[token_idx],
                        "Layer": layer_idx, "Head": "all",
                        "Mode": "global", "Rank": rank + 1,
                        "Target_Idx": target_idx,
                        "Target_Token": tokens[target_idx],
                        "Attention": float(att_row[target_idx]),
                    })

        if not all_rows:
            return None
        df = pd.DataFrame(all_rows)
        return df.to_csv(index=False)

    @auto_save_download("attention_topk", "csv", data_type="all_layers_heads")
    def export_topk_attention():
        csv_content = get_topk_attention_csv(get_active_result())
        yield csv_content or "No data available"

    @auto_save_download("attention_topk", "csv", is_b=True, data_type="all_layers_heads")
    def export_topk_attention_B():
        csv_content = get_topk_attention_csv(get_active_result("_B"), suffix="_B")
        yield csv_content or "No data available"

    # ── ISA JSON export (with token-to-token) ───────────────────────────

    def _export_isa_json(suffix="", model_family_fn=None):
        res = get_active_result(suffix)
        if not res:
            yield json.dumps({"error": "No data available"})
            return
        try:
            tokens = res.tokens
            attentions = res.attentions
            isa_data = res.isa_data

            export_data = {
                "tokens": list(tokens),
                "model": (model_family_fn() if model_family_fn else "model") or "model",
            }

            if isinstance(isa_data, dict):
                for key, value in isa_data.items():
                    if isinstance(value, np.ndarray):
                        export_data[key] = value.tolist()
                    else:
                        export_data[key] = value

                if "sentence_boundaries" in isa_data:
                    boundaries = isa_data["sentence_boundaries"]
                    token2token = {}
                    for i, (start_i, end_i) in enumerate(boundaries):
                        for j_idx, (start_j, end_j) in enumerate(boundaries):
                            if i != j_idx:
                                max_att = None
                                for layer_att in attentions:
                                    att = (layer_att[0].numpy()
                                           if hasattr(layer_att[0], "numpy")
                                           else np.array(layer_att[0]))
                                    att_slice = att[:, start_i:end_i,
                                                    start_j:end_j].max(axis=0)
                                    if max_att is None:
                                        max_att = att_slice
                                    else:
                                        max_att = np.maximum(max_att, att_slice)
                                if max_att is not None:
                                    token2token[f"sent{i}_to_sent{j_idx}"] = {
                                        "source_tokens": tokens[start_i:end_i],
                                        "target_tokens": tokens[start_j:end_j],
                                        "attention_matrix": max_att.tolist(),
                                    }
                    export_data["token_to_token"] = token2token

            yield json.dumps(export_data, indent=2, default=str)
        except Exception as e:
            yield json.dumps({"error": str(e)})

    @auto_save_download("isa", "json", data_type="with_token2token")
    def export_isa_json():
        """Export ISA data as JSON including token-to-token attention."""
        yield from _export_isa_json("", lambda: input.model_family())

    @auto_save_download("isa", "json", is_b=True, data_type="with_token2token")
    def export_isa_json_B():
        """Export ISA data as JSON for Model B."""
        yield from _export_isa_json("_B", lambda: input.model_family_B())

    # ── ISA sentence matrix CSV ─────────────────────────────────────────

    def _export_isa_csv(suffix=""):
        res = get_active_result(suffix)
        if not res:
            yield "No data available"
            return
        try:
            import pandas as pd
            isa_data = res.isa_data
            if not isa_data or "sentence_attention_matrix" not in isa_data:
                yield "No ISA data available"
                return
            matrix = isa_data["sentence_attention_matrix"]
            sentences = isa_data.get(
                "sentence_texts",
                [f"Sent_{i}" for i in range(len(matrix))],
            )
            df = pd.DataFrame(matrix, index=sentences, columns=sentences)
            df.index.name = "Source_Sentence"
            yield df.to_csv()
        except Exception as e:
            yield f"Error: {e}"

    @auto_save_download("isa", "csv", data_type="sentence_matrix")
    def export_isa_csv():
        """Export ISA sentence attention matrix as CSV."""
        yield from _export_isa_csv()

    @auto_save_download("isa", "csv", is_b=True, data_type="sentence_matrix")
    def export_isa_csv_B():
        """Export ISA sentence attention matrix as CSV for Model B."""
        yield from _export_isa_csv("_B")
