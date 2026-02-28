"""
GUS-Net Focal Loss Hyperparameter Sweep
=========================================
Grid search over ALPHA × GAMMA for the focal loss.
Reutiliza toda a lógica de gus_net_training_paper.py.
Resultados guardados no training_log.json.
"""

import sys
import json
import importlib
import numpy as np
from pathlib import Path

# ============================================================================
# SWEEP CONFIG — edita aqui
# ============================================================================

DATASET_SOURCE = "clean"    # "clean" | "hf" | "gemini"
BACKBONE       = "bert"     # "bert" | "gpt2"

ALPHAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GAMMAS = [2.0, 2.5, 3.0, 3.5, 4.0]

# ============================================================================


def _patch_module(alpha: float, gamma: float):
    """Reload gus_net_training_paper com novos ALPHA e GAMMA."""
    mod_name = "gus_net_training_paper"

    # Remove cached version para forçar reload limpo
    for key in list(sys.modules):
        if key == mod_name:
            del sys.modules[key]

    m = importlib.import_module(mod_name)

    # Patching dos globals
    m.ALPHA = alpha
    m.GAMMA = gamma

    # Patching dos default args da focal_loss_paper
    # (defaults são avaliados na definição — precisamos de os actualizar)
    m.focal_loss_paper.__defaults__ = (alpha, gamma)

    return m


def _read_latest_log(log_path: Path) -> dict:
    with open(log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    return log[-1]


def run_sweep():
    combos = [(a, g) for a in ALPHAS for g in GAMMAS]
    total  = len(combos)

    print("=" * 60)
    print(f"GUS-Net Focal Loss Sweep  —  {total} combinações")
    print(f"Backbone: {BACKBONE.upper()}  |  Dataset: {DATASET_SOURCE}")
    print(f"ALPHAS: {ALPHAS}")
    print(f"GAMMAS: {GAMMAS}")
    print("=" * 60)

    results = []

    for i, (alpha, gamma) in enumerate(combos, 1):
        print(f"\n{'=' * 60}")
        print(f"RUN {i}/{total}  —  ALPHA={alpha:.2f}, GAMMA={gamma:.1f}")
        print("=" * 60)

        try:
            m = _patch_module(alpha, gamma)

            # Seed consistente por run
            import pytorch_lightning as pl
            pl.seed_everything(m.SEED, workers=True)

            if BACKBONE == "bert":
                m.train_bert_paper(dataset_source=DATASET_SOURCE)
            else:
                m.train_gpt2_paper(dataset_source=DATASET_SOURCE)

            # Lê o último entry do log (esta run)
            log_path = m._SCRIPT_DIR / "training_log.json"
            entry = _read_latest_log(log_path)

            cat = entry["test_fixed_thr"]["category_level"]
            macro_f1 = cat["macro avg"]["f1-score"]
            weighted_f1 = cat["weighted avg"]["f1-score"]
            em = entry["test_fixed_thr"]["exact_match"]

            cat_opt = entry["test_optimised_thr"]["category_level"]
            macro_f1_opt = cat_opt["macro avg"]["f1-score"]
            em_opt = entry["test_optimised_thr"]["exact_match"]

            results.append({
                "alpha":        alpha,
                "gamma":        gamma,
                "macro_f1":     round(macro_f1, 4),
                "weighted_f1":  round(weighted_f1, 4),
                "exact_match":  round(em, 4),
                "macro_f1_opt": round(macro_f1_opt, 4),
                "em_opt":       round(em_opt, 4),
            })

            print(f"\n  → Macro F1 (fixed): {macro_f1:.4f}  |  EM: {em:.4f}")

        except Exception as exc:
            print(f"\n  ✗ ERRO: {exc}")
            results.append({
                "alpha":       alpha,
                "gamma":       gamma,
                "macro_f1":    0.0,
                "weighted_f1": 0.0,
                "exact_match": 0.0,
                "macro_f1_opt":0.0,
                "em_opt":      0.0,
                "error":       str(exc),
            })

    # ----------------------------------------------------------------
    # Tabela final
    # ----------------------------------------------------------------
    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    print("\n\n" + "=" * 72)
    print("SWEEP COMPLETO — RESULTADOS (ordenados por Macro F1 fixo)")
    print("=" * 72)
    header = f"{'ALPHA':>6}  {'GAMMA':>6}  {'MacroF1':>8}  {'WgtF1':>7}  {'EM':>7}  {'MacroF1opt':>10}  {'EMopt':>7}"
    print(header)
    print("-" * 72)

    for r in results:
        marker = "  ← MELHOR" if r is results[0] else ""
        err    = f"  [ERRO: {r['error'][:30]}]" if "error" in r else ""
        print(
            f"  {r['alpha']:.2f}   {r['gamma']:.1f}   "
            f"{r['macro_f1']:.4f}   {r['weighted_f1']:.4f}   "
            f"{r['exact_match']:.4f}   {r['macro_f1_opt']:.4f}   "
            f"{r['em_opt']:.4f}"
            f"{marker}{err}"
        )

    best = results[0]
    print(f"\nMelhor configuração:")
    print(f"  ALPHA = {best['alpha']:.2f}")
    print(f"  GAMMA = {best['gamma']:.1f}")
    print(f"  Macro F1 (fixed thr) = {best['macro_f1']:.4f}")
    print(f"  Exact Match          = {best['exact_match']:.4f}")

    # Guarda tabela em JSON
    sweep_log = Path(__file__).resolve().parent / "sweep_results.json"
    with open(sweep_log, "w", encoding="utf-8") as f:
        json.dump({
            "backbone":       BACKBONE,
            "dataset":        DATASET_SOURCE,
            "alphas_tested":  ALPHAS,
            "gammas_tested":  GAMMAS,
            "best_alpha":     best["alpha"],
            "best_gamma":     best["gamma"],
            "results":        results,
        }, f, indent=2)
    print(f"\nTabela guardada em {sweep_log}")


if __name__ == "__main__":
    run_sweep()
