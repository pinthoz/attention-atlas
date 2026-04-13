"""Pure CSV body functions for bias export handlers.

Each function takes a result dict/bundle and returns a CSV string.
No reactive dependencies — these can be imported and tested independently.
"""

import numpy as np

from .csv_utils import csv_safe as _csv_safe
from ..bias.integrated_gradients import IGAnalysisBundle
from ..bias.stereoset import (
    get_stereoset_scores,
    get_stereoset_examples,
    get_sensitive_heads,
    get_top_features,
)


# ── Basic bias exports ───────────────────────────────────────────────────

def csv_summary(res):
    s = res["bias_summary"]
    lines = ["metric,value"]
    for k, v in s.items():
        lines.append(f"{k},{v}")
    return "\n".join(lines)


def csv_spans(res):
    lines = ["token,index,bias_types,GEN,UNFAIR,STEREO,max_score"]
    for lbl in res["token_labels"]:
        if not lbl.get("is_biased"):
            continue
        scores = lbl.get("scores", {})
        types = ";".join(lbl.get("bias_types", []))
        max_s = max((scores.get(t, 0) for t in lbl.get("bias_types", [])), default=0)
        lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{types},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f},{max_s:.4f}")
    return "\n".join(lines)


def csv_strip(res):
    lines = ["token,index,is_biased,O,GEN,UNFAIR,STEREO"]
    for lbl in res["token_labels"]:
        scores = lbl.get("scores", {})
        lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{lbl.get('is_biased',False)},{scores.get('O',0):.4f},{scores.get('GEN',0):.4f},{scores.get('UNFAIR',0):.4f},{scores.get('STEREO',0):.4f}")
    return "\n".join(lines)


def csv_confidence(res):
    lines = ["token,index,tier,max_score,bias_types"]
    for lbl in res["token_labels"]:
        if not lbl.get("is_biased"):
            continue
        scores = lbl.get("scores", {})
        types = lbl.get("bias_types", [])
        max_s = max((scores.get(t, 0) for t in types), default=0)
        tier = "high" if max_s >= 0.85 else ("medium" if max_s >= 0.70 else "low")
        lines.append(f"{_csv_safe(lbl['token'])},{lbl['index']},{tier},{max_s:.4f},{';'.join(types)}")
    return "\n".join(lines)


def csv_combined(res, l_idx=0, h_idx=0):
    """Attention matrix CSV for a given layer/head.

    The layer/head indices are passed explicitly so this function stays
    free of reactive dependencies (the caller reads ``input`` and passes
    the values in).
    """
    atts = res["attentions"]
    if not atts or l_idx >= len(atts):
        return "No attention data"
    tokens = res["tokens"]
    attn = atts[l_idx][0, h_idx].cpu().numpy()
    header = "query_token," + ",".join(tokens)
    lines = [header]
    for i, tok in enumerate(tokens):
        vals = ",".join(f"{attn[i,j]:.6f}" for j in range(len(tokens)))
        lines.append(f"{tok},{vals}")
    return "\n".join(lines)


def csv_matrix(res):
    metrics = res.get("attention_metrics", [])
    if not metrics:
        return "No metrics available"
    lines = ["layer,head,bias_attention_ratio,specialized"]
    for m in metrics:
        lines.append(f"{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
    return "\n".join(lines)


def csv_propagation(res):
    prop = res["propagation_analysis"]["layer_propagation"]
    if not prop:
        return "No propagation data"
    lines = ["layer,mean_bar,max_bar,min_bar"]
    for p in prop:
        lines.append(f"{p['layer']},{p['mean_ratio']:.4f},{p['max_ratio']:.4f},{p['min_ratio']:.4f}")
    return "\n".join(lines)


def csv_top_heads(res):
    metrics = res.get("attention_metrics", [])
    if not metrics:
        return "No metrics available"
    top = sorted(metrics, key=lambda m: m.bias_attention_ratio, reverse=True)[:5]
    lines = ["rank,layer,head,bar,specialized"]
    for i, m in enumerate(top, 1):
        lines.append(f"{i},{m.layer},{m.head},{m.bias_attention_ratio:.4f},{m.specialized_for_bias}")
    return "\n".join(lines)


# ── Ablation export ──────────────────────────────────────────────────────

def csv_ablation(results):
    lines = ["rank,layer,head,representation_impact,kl_divergence,bar_original"]
    for i, r in enumerate(results, 1):
        kl = f"{r.kl_divergence:.6f}" if r.kl_divergence is not None else ""
        lines.append(f"{i},{r.layer},{r.head},{r.representation_impact:.6f},{kl},{r.bar_original:.4f}")
    return "\n".join(lines)


# ── IG exports ───────────────────────────────────────────────────────────

def csv_ig_correlation(bundle):
    results = bundle.correlations if isinstance(bundle, IGAnalysisBundle) else bundle
    lines = ["rank,layer,head,spearman_rho,spearman_pvalue,bar_original"]
    sorted_results = sorted(results, key=lambda r: abs(r.spearman_rho), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        lines.append(f"{i},{r.layer},{r.head},{r.spearman_rho:.6f},{r.spearman_pvalue:.6f},{r.bar_original:.4f}")
    return "\n".join(lines)


def csv_topk_overlap(bundle):
    lines = ["layer,head,k,jaccard,rank_biased_overlap,bar_original"]
    for r in bundle.topk_overlaps:
        lines.append(f"{r.layer},{r.head},{r.k},{r.jaccard:.6f},{r.rank_biased_overlap:.6f},{r.bar_original:.4f}")
    return "\n".join(lines)


# ── Perturbation / LRP exports ──────────────────────────────────────────

def csv_perturbation(bundle):
    lines = ["token_index,token,importance"]
    for r in bundle.token_results:
        lines.append(f"{r.token_index},{_csv_safe(r.token)},{r.importance:.6f}")
    return "\n".join(lines)


def csv_lrp(bundle):
    lines = ["layer,head,spearman_rho_vs_attention"]
    for l, h, rho in bundle.correlations:
        lines.append(f"{l},{h},{rho:.6f}")
    return "\n".join(lines)


def csv_perturb_attn(bundle):
    lines = ["layer,head,spearman_rho"]
    for layer, head, rho in bundle.perturb_vs_attn_spearman:
        lines.append(f"{layer},{head},{rho:.6f}")
    return "\n".join(lines)


def csv_cross_method(lrp_bundle, ig_bundle):
    ig_dict = {(r.layer, r.head): r.spearman_rho for r in ig_bundle.correlations}
    lines = ["layer,head,ig_rho,lrp_rho"]
    for layer, head, lrp_rho in lrp_bundle.correlations:
        ig_rho = ig_dict.get((layer, head), float("nan"))
        lines.append(f"{layer},{head},{ig_rho:.6f},{lrp_rho:.6f}")
    return "\n".join(lines)


# ── StereoSet exports ────────────────────────────────────────────────────

def csv_stereoset_features(mk):
    top_features = get_top_features(mk)
    if not top_features:
        return "No feature data"
    lines = ["rank,feature,p_value"]
    for rank, f in enumerate(top_features, 1):
        lines.append(f'{rank},{f["name"]},{f["p_value"]:.6e}')
    return "\n".join(lines)


def csv_stereoset_sensitivity(mk):
    heads = get_sensitive_heads(mk)
    if not heads:
        return "No sensitivity data"
    lines = ["rank,layer,head,variance,correlation,best_feature"]
    for rank, h in enumerate(heads, 1):
        lines.append(f'{rank},{h["layer"]},{h["head"]},{h["variance"]:.6f},{h["correlation"]:.6f},{h["best_feature"]}')
    return "\n".join(lines)


def csv_stereoset_category(mk):
    scores = get_stereoset_scores(mk)
    if not scores:
        return "No scores data"
    by_cat = scores.get("by_category", {})
    lines = ["category,ss,lms,icat,n,mean_bias_score"]
    for cat, v in by_cat.items():
        lines.append(f'{cat},{v["ss"]:.2f},{v["lms"]:.2f},{v["icat"]:.2f},{v["n"]},{v.get("mean_bias_score", 0):.6f}')
    return "\n".join(lines)


def csv_stereoset_distribution(mk):
    examples = get_stereoset_examples(mk)
    if not examples:
        return "No examples data"
    lines = ["category,bias_score"]
    for e in examples:
        lines.append(f'{e.get("category","")},{e.get("bias_score", e.get("stereo_prob", 0)):.6f}')
    return "\n".join(lines)


def csv_stereoset_demographic(mk):
    examples = get_stereoset_examples(mk)
    if not examples:
        return "No examples data"
    target_data = {}
    for ex in examples:
        t = ex.get("target", "unknown")
        if t not in target_data:
            target_data[t] = {"stereo_wins": 0, "n": 0, "category": ex.get("category", "")}
        target_data[t]["n"] += 1
        if ex.get("stereo_pll", 0) > ex.get("anti_pll", 0):
            target_data[t]["stereo_wins"] += 1
    lines = ["target,category,ss_pct,n"]
    for t, d in sorted(target_data.items(), key=lambda x: x[1]["stereo_wins"] / max(x[1]["n"], 1), reverse=True):
        if d["n"] >= 5:
            ss = d["stereo_wins"] / d["n"] * 100
            lines.append(f'{t},{d["category"]},{ss:.1f},{d["n"]}')
    return "\n".join(lines)
