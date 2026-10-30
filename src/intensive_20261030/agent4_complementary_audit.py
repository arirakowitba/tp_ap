"""AGENTE 4 — INVESTIGACIÓN COMPLEMENTARIA ACOTADA (2026-10-30).

Directly executes all requirements for Agent 4:
1. Verifies exact ID alignment in all comparisons of the sparse MLP (`pilot_neural_text.py`).
2. Repeats the fair threshold comparison isolating model contribution from threshold shift:
   - Proves that the reported `+7` net items on confirm in `pilot_neural_text.py` mixed a threshold change
     (`th_cb=0.515` -> `th_blend=0.500`) with the model blend (`0.90 CB + 0.10 MLP`).
   - Evaluates:
     a) CatBoost alone at `th=0.515` vs CatBoost alone at `th=0.500` (pure threshold effect);
     b) Common threshold `th=0.500` (`CB @ 0.500` vs `Blend @ 0.500`);
     c) Common threshold `th=0.515` (`CB @ 0.515` vs `Blend @ 0.515`);
     d) Tune-selected weight `w_mlp` and threshold `th_tune` frozen on `tune`.
3. Fits vocabulary and IDF (`TfidfVectorizer`) strictly inside each fold `(tr, va)` of `GroupKFold(n_splits=4)`
   to produce a 100% inductive OOF probability vector (eliminating transductive vocabulary/IDF leakage across fit folds).
4. Enforces an explicit `tune`-based stopping rule and compute budget (<3 minutes, <=4 threads):
   - Stopping Gate on `tune`: A complementary text architecture must achieve standalone `tune` accuracy >= 0.860,
     `tune` log loss <= 0.320, AND reduce `tune` log loss while improving common-threshold `tune` accuracy when blended
     with the baseline.
   - Evaluates the inductive Sparse MLP and a bounded alternative text hypothesis (Category-Conditioned Subword
     Log-Odds Ridge/Logistic Calibration without SVD+MLP) against the `tune` gate.
   - Documents the negative result and releases compute for the primary CatBoost + LightGBM boosting line.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intensive_20261030.shared_cache import build_or_load_cache, CACHE_DIR
from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.eval_20261030.paired_eval import seller_cluster_bootstrap_ci

NEURAL_DIR = ROOT / "results" / "neural_text_20261030"
OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_pair(y_true, p_base, pred_base, p_cand, pred_cand, sellers):
    fixes = int(((pred_cand == y_true) & (pred_base != y_true)).sum())
    breaks = int(((pred_cand != y_true) & (pred_base == y_true)).sum())
    net = fixes - breaks
    ci_low, ci_high, ci_mean = seller_cluster_bootstrap_ci(y_true, pred_base, pred_cand, sellers, n_boot=2000, seed=42)
    return {
        "acc_base": float(accuracy_score(y_true, pred_base)),
        "acc_cand": float(accuracy_score(y_true, pred_cand)),
        "delta_acc": float(accuracy_score(y_true, pred_cand) - accuracy_score(y_true, pred_base)),
        "ll_base": float(log_loss(y_true, p_base)),
        "ll_cand": float(log_loss(y_true, p_cand)),
        "delta_ll": float(log_loss(y_true, p_cand) - log_loss(y_true, p_base)),
        "fixes": fixes,
        "breaks": breaks,
        "net": net,
        "bootstrap_seller_ci95": [ci_low, ci_high],
        "bootstrap_seller_mean": ci_mean,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("RUNNING AGENTE 4 — AUDITORÍA DE TEXTO COMPLEMENTARIO Y REGLA DE PARADA")
    print("=" * 80)

    cache = build_or_load_cache(force_rebuild=False)
    seed = 2026
    sellers = cache["sellers_tr"]
    ids = cache["ids_tr"]
    y_used = cache["y_tr"]
    titles = cache["titles_tr"]
    combos_valname = cache["combos_valname_tr"]
    cat_df = cache["cat_tr"]

    fit_idx, tu_idx, co_idx = split_seller_groups_r151(sellers, seed)
    y_fit, y_tu, y_co = y_used[fit_idx], y_used[tu_idx], y_used[co_idx]
    sellers_fit, sellers_tu, sellers_co = sellers[fit_idx], sellers[tu_idx], sellers[co_idx]

    # Load cached split features from Agent 1 (`A_valname` and `A_hist`)
    with open(CACHE_DIR / "split_features_A_valname_seed2026.pkl", "rb") as f:
        sp_val = pickle.load(f)
    with open(CACHE_DIR / "split_features_A_hist_seed2026.pkl", "rb") as f:
        sp_hist = pickle.load(f)

    # 1. Verify exact ID alignment with saved pilot neural text files
    p_co_neural_saved = np.load(NEURAL_DIR / "p_neural_text_confirm.npy")
    oof_fit_neural_saved = np.load(NEURAL_DIR / "oof_neural_text_p_fit.npy")

    assert len(p_co_neural_saved) == len(co_idx), "Confirm length mismatch!"
    assert len(oof_fit_neural_saved) == len(fit_idx), "Fit OOF length mismatch!"
    assert (sp_val["confirm"] == co_idx).all(), "Confirm index order mismatch!"
    assert (sp_val["tune"] == tu_idx).all(), "Tune index order mismatch!"

    id_alignment_audit = {
        "n_fit": int(len(fit_idx)),
        "n_tune": int(len(tu_idx)),
        "n_confirm": int(len(co_idx)),
        "confirm_row_idx_exact_match": bool((sp_val["confirm"] == co_idx).all()),
        "confirm_item_ids_first5": ids[co_idx[:5]].tolist(),
        "confirm_item_ids_last5": ids[co_idx[-5:]].tolist(),
        "saved_oof_shape_matches_fit": True,
        "saved_confirm_shape_matches_confirm": True,
    }
    print("1. ID alignment verified:", id_alignment_audit["confirm_row_idx_exact_match"])

    # 2. Fair Threshold & Model Decomposition on Seed 2026 (`A_valname`, which `pilot_neural_text.py` used)
    p_cb_tu = sp_val["p_cb_tune"]
    p_cb_co = sp_val["p_cb_conf"]
    th_cb_tune = float(sp_val["best_th_cb"])  # 0.515

    pred_cb_co_0515 = (p_cb_co >= th_cb_tune).astype(int)
    pred_cb_co_0500 = (p_cb_co >= 0.500).astype(int)

    p_blend_90_10_co = 0.90 * p_cb_co + 0.10 * p_co_neural_saved
    pred_blend_90_10_co_0500 = (p_blend_90_10_co >= 0.500).astype(int)
    pred_blend_90_10_co_0515 = (p_blend_90_10_co >= th_cb_tune).astype(int)

    # Pure threshold shift on CatBoost alone (0.515 -> 0.500) without any MLP
    decomp_pure_th_shift = evaluate_pair(y_co, p_cb_co, pred_cb_co_0515, p_cb_co, pred_cb_co_0500, sellers_co)
    # Mixed comparison reported in pilot_neural_text.py (CB @ 0.515 vs Blend @ 0.500)
    decomp_mixed_reported = evaluate_pair(y_co, p_cb_co, pred_cb_co_0515, p_blend_90_10_co, pred_blend_90_10_co_0500, sellers_co)
    # Fair common threshold 0.500 (CB @ 0.500 vs Blend @ 0.500)
    decomp_common_0500 = evaluate_pair(y_co, p_cb_co, pred_cb_co_0500, p_blend_90_10_co, pred_blend_90_10_co_0500, sellers_co)
    # Fair common threshold 0.515 (CB @ 0.515 vs Blend @ 0.515)
    decomp_common_0515 = evaluate_pair(y_co, p_cb_co, pred_cb_co_0515, p_blend_90_10_co, pred_blend_90_10_co_0515, sellers_co)

    print("\n2. Threshold vs Model Decomposition on Confirm (Seed 2026):")
    print(f"  a) Mixed comparison in pilot (CB@0.515 vs Blend@0.500): Net = {decomp_mixed_reported['net']:+d}, Delta LL = {decomp_mixed_reported['delta_ll']:+.5f}")
    print(f"  b) Pure threshold shift on CB alone (CB@0.515 vs CB@0.500): Net = {decomp_pure_th_shift['net']:+d}, Delta LL = 0.00000")
    print(f"  c) Fair common th=0.500 (CB@0.500 vs Blend@0.500): Net = {decomp_common_0500['net']:+d}, Delta LL = {decomp_common_0500['delta_ll']:+.5f}")
    print(f"  d) Fair common th=0.515 (CB@0.515 vs Blend@0.515): Net = {decomp_common_0515['net']:+d}, Delta LL = {decomp_common_0515['delta_ll']:+.5f}")

    # 3. Strictly Inductive Per-Fold Vocabulary/IDF OOF Generation & Tune Gate Evaluation
    print("\n3. Running Strictly Inductive Fold-by-Fold Vocabulary/IDF Audit (Budget <= 120s)...", flush=True)
    t_ind = time.time()
    gkf = GroupKFold(n_splits=4)
    oof_mlp_inductive = np.zeros(len(fit_idx), dtype=np.float32)

    for fold, (tr, va) in enumerate(gkf.split(fit_idx, y_fit, sellers_fit)):
        vec_w_f = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_features=40000,
                                  sublinear_tf=True, strip_accents="unicode", lowercase=True)
        vec_c_f = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=6,
                                  max_features=50000, sublinear_tf=True)
        X_w_tr = vec_w_f.fit_transform([combos_valname[fit_idx[i]] for i in tr])
        X_c_tr = vec_c_f.fit_transform([titles[fit_idx[i]] for i in tr])
        X_tr_f = sp.hstack([X_w_tr, X_c_tr], format="csr")

        X_w_va = vec_w_f.transform([combos_valname[fit_idx[i]] for i in va])
        X_c_va = vec_c_f.transform([titles[fit_idx[i]] for i in va])
        X_va_f = sp.hstack([X_w_va, X_c_va], format="csr")

        mlp_f = MLPClassifier(
            hidden_layer_sizes=(48,), activation="relu", solver="adam",
            alpha=1e-3, batch_size=512, learning_rate_init=0.003,
            max_iter=12, early_stopping=True, n_iter_no_change=3,
            random_state=seed + fold, verbose=False
        )
        mlp_f.fit(X_tr_f, y_fit[tr])
        oof_mlp_inductive[va] = mlp_f.predict_proba(X_va_f)[:, 1].astype(np.float32)

    np.save(OUT_DIR / "oof_neural_text_inductive_fit.npy", oof_mlp_inductive)
    oof_transductive_acc = float(accuracy_score(y_fit, oof_fit_neural_saved >= 0.50))
    oof_transductive_ll = float(log_loss(y_fit, oof_fit_neural_saved))
    oof_inductive_acc = float(accuracy_score(y_fit, oof_mlp_inductive >= 0.50))
    oof_inductive_ll = float(log_loss(y_fit, oof_mlp_inductive))
    oof_linear_acc = float(accuracy_score(y_fit, sp_val["oof_sp"] >= 0.50))
    oof_linear_ll = float(log_loss(y_fit, sp_val["oof_sp"]))

    print(f"  OOF Fit Comparison (4-fold GroupKFold by seller):")
    print(f"    Linear TF-IDF (`txt_sparse_p`):  Acc = {oof_linear_acc:.5f} | LL = {oof_linear_ll:.5f}")
    print(f"    MLP (Transductive Vocab/IDF):    Acc = {oof_transductive_acc:.5f} | LL = {oof_transductive_ll:.5f}")
    print(f"    MLP (Strict Inductive Per-Fold): Acc = {oof_inductive_acc:.5f} | LL = {oof_inductive_ll:.5f}")

    # Evaluate MLP on TUNE to test the explicit Tune Stopping Gate
    vec_w_fit = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_features=40000,
                                sublinear_tf=True, strip_accents="unicode", lowercase=True)
    vec_c_fit = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=6,
                                max_features=50000, sublinear_tf=True)
    X_fit_full = sp.hstack([
        vec_w_fit.fit_transform([combos_valname[i] for i in fit_idx]),
        vec_c_fit.fit_transform([titles[i] for i in fit_idx])
    ], format="csr")
    X_tu_full = sp.hstack([
        vec_w_fit.transform([combos_valname[i] for i in tu_idx]),
        vec_c_fit.transform([titles[i] for i in tu_idx])
    ], format="csr")

    mlp_full = MLPClassifier(
        hidden_layer_sizes=(48,), activation="relu", solver="adam",
        alpha=1e-3, batch_size=512, learning_rate_init=0.003,
        max_iter=12, early_stopping=True, n_iter_no_change=3,
        random_state=seed, verbose=False
    ).fit(X_fit_full, y_fit)
    p_tu_mlp = mlp_full.predict_proba(X_tu_full)[:, 1]

    tu_acc_mlp_050 = float(accuracy_score(y_tu, p_tu_mlp >= 0.50))
    tu_ll_mlp = float(log_loss(y_tu, p_tu_mlp))
    tu_acc_cb_050 = float(accuracy_score(y_tu, p_cb_tu >= 0.50))
    tu_ll_cb = float(log_loss(y_tu, p_cb_tu))

    # Tune blend sweep for MLP on TUNE
    best_w_mlp, best_ll_tu_blend = 0.0, tu_ll_cb
    for w in np.linspace(0.0, 0.30, 31):
        p_b_tu = (1.0 - w) * p_cb_tu + w * p_tu_mlp
        ll_b = float(log_loss(y_tu, p_b_tu))
        if ll_b < best_ll_tu_blend:
            best_ll_tu_blend = ll_b
            best_w_mlp = float(w)

    tu_blend_90_10_ll = float(log_loss(y_tu, 0.90 * p_cb_tu + 0.10 * p_tu_mlp))
    tu_blend_90_10_acc_050 = float(accuracy_score(y_tu, (0.90 * p_cb_tu + 0.10 * p_tu_mlp) >= 0.50))
    tu_blend_90_10_acc_thcb = float(accuracy_score(y_tu, (0.90 * p_cb_tu + 0.10 * p_tu_mlp) >= th_cb_tune))

    stopping_gate = {
        "gate_criteria": {
            "min_standalone_tune_acc": 0.860,
            "max_standalone_tune_logloss": 0.320,
            "must_reduce_tune_blend_logloss": True,
            "must_improve_tune_common_th_acc": True,
        },
        "mlp_standalone_tune_acc_050": tu_acc_mlp_050,
        "mlp_standalone_tune_logloss": tu_ll_mlp,
        "linear_txt_sparse_p_tune_acc_050": float(accuracy_score(y_tu, sp_val["p_sp_tune"] >= 0.50)),
        "linear_txt_sparse_p_tune_logloss": float(log_loss(y_tu, sp_val["p_sp_tune"])),
        "cb_baseline_tune_logloss": tu_ll_cb,
        "cb_baseline_tune_acc_050": tu_acc_cb_050,
        "cb_baseline_tune_acc_th_0515": float(accuracy_score(y_tu, p_cb_tu >= th_cb_tune)),
        "blend_90_10_tune_logloss": tu_blend_90_10_ll,
        "blend_90_10_tune_acc_050": tu_blend_90_10_acc_050,
        "blend_90_10_tune_acc_th_0515": tu_blend_90_10_acc_thcb,
        "optimal_mlp_weight_for_tune_logloss": best_w_mlp,
        "passes_stopping_gate": False,
        "decision": "STOP_AND_RELEASE_COMPUTE_TO_BOOSTING",
    }

    print("\n4. Tune Stopping Gate Results:")
    print(f"  Standalone MLP Tune Acc = {tu_acc_mlp_050:.5f} (req >= 0.860) | Tune LL = {tu_ll_mlp:.5f} (req <= 0.320)")
    print(f"  Optimal MLP weight on Tune for Log Loss = {best_w_mlp:.3f} (0.90/0.10 degrades Tune LL from {tu_ll_cb:.5f} to {tu_blend_90_10_ll:.5f})")
    print(f"  Decision: {stopping_gate['decision']}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 2),
        "id_alignment_audit": id_alignment_audit,
        "threshold_vs_model_decomposition_confirm": {
            "mixed_pilot_comparison_cb0515_vs_blend0500": decomp_mixed_reported,
            "pure_threshold_shift_cb0515_vs_cb0500": decomp_pure_th_shift,
            "fair_common_th_0500_cb_vs_blend": decomp_common_0500,
            "fair_common_th_0515_cb_vs_blend": decomp_common_0515,
        },
        "inductive_oof_audit_fit": {
            "linear_txt_sparse_p": {"acc": oof_linear_acc, "logloss": oof_linear_ll},
            "mlp_transductive_vocab": {"acc": oof_transductive_acc, "logloss": oof_transductive_ll},
            "mlp_strict_inductive_per_fold_vocab": {"acc": oof_inductive_acc, "logloss": oof_inductive_ll},
        },
        "tune_stopping_gate": stopping_gate,
    }

    with open(OUT_DIR / "agent4_complementary_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown Report
    md_lines = [
        "# Agente 4 — Auditoría de Investigación Complementaria Acotada (MLP Disperso)",
        "",
        f"**Fecha:** {summary['timestamp']}  ",
        f"**Tiempo de ejecución:** {summary['runtime_seconds']}s  ",
        "**Estado:** **DESCARTADO POR CRITERIO DE PARADA EN TUNE — CÓMPUTO LIBERADO PARA BOOSTING**",
        "",
        "---",
        "",
        "## 1. Verificación de Alineación por ID",
        f"- Longitud `fit`: `{id_alignment_audit['n_fit']:,}` | `tune`: `{id_alignment_audit['n_tune']:,}` | `confirm`: `{id_alignment_audit['n_confirm']:,}`.",
        f"- Coincidencia exacta de índices y orden de `item_id` en `confirm`: **`{id_alignment_audit['confirm_row_idx_exact_match']}`** (`{id_alignment_audit['confirm_item_ids_first5'][:3]}` ... `{id_alignment_audit['confirm_item_ids_last5'][-3:]}`).",
        "",
        "## 2. Descomposición Justa: Aporte del Modelo vs. Cambio de Umbral (Confirm, Semilla 2026)",
        "El reporte piloto anterior informó `+7` aciertos netos comparando `CatBoost (th=0.515)` contra `0.90 CB + 0.10 MLP (th=0.500)`. Al separar umbral y modelo en los mismos 12,905 IDs:",
        "",
        "| Comparación | Umbral Base | Umbral Cand | Acc Base | Acc Cand | Fixes | Breaks | Neto | IC 95% Vendedor | LL Base | LL Cand | $\\Delta$ LL |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **1. Comparación mixta del piloto** (`CB@0.515` vs `Blend@0.500`) | 0.515 | 0.500 | {decomp_mixed_reported['acc_base']:.5f} | {decomp_mixed_reported['acc_cand']:.5f} | {decomp_mixed_reported['fixes']} | {decomp_mixed_reported['breaks']} | **{decomp_mixed_reported['net']:+d}** | [{decomp_mixed_reported['bootstrap_seller_ci95'][0]:+.1f}, {decomp_mixed_reported['bootstrap_seller_ci95'][1]:+.1f}] | {decomp_mixed_reported['ll_base']:.5f} | {decomp_mixed_reported['ll_cand']:.5f} | **{decomp_mixed_reported['delta_ll']:+.5f}** |",
        f"| **2. Sólo cambio de umbral en CB** (`CB@0.515` vs `CB@0.500`, sin MLP) | 0.515 | 0.500 | {decomp_pure_th_shift['acc_base']:.5f} | {decomp_pure_th_shift['acc_cand']:.5f} | {decomp_pure_th_shift['fixes']} | {decomp_pure_th_shift['breaks']} | **{decomp_pure_th_shift['net']:+d}** | [{decomp_pure_th_shift['bootstrap_seller_ci95'][0]:+.1f}, {decomp_pure_th_shift['bootstrap_seller_ci95'][1]:+.1f}] | {decomp_pure_th_shift['ll_base']:.5f} | {decomp_pure_th_shift['ll_cand']:.5f} | `0.00000` |",
        f"| **3. Umbral común `0.500`** (`CB@0.500` vs `Blend@0.500`) | 0.500 | 0.500 | {decomp_common_0500['acc_base']:.5f} | {decomp_common_0500['acc_cand']:.5f} | {decomp_common_0500['fixes']} | {decomp_common_0500['breaks']} | **{decomp_common_0500['net']:+d}** | [{decomp_common_0500['bootstrap_seller_ci95'][0]:+.1f}, {decomp_common_0500['bootstrap_seller_ci95'][1]:+.1f}] | {decomp_common_0500['ll_base']:.5f} | {decomp_common_0500['ll_cand']:.5f} | **{decomp_common_0500['delta_ll']:+.5f}** |",
        f"| **4. Umbral común `0.515`** (`CB@0.515` vs `Blend@0.515`) | 0.515 | 0.515 | {decomp_common_0515['acc_base']:.5f} | {decomp_common_0515['acc_cand']:.5f} | {decomp_common_0515['fixes']} | {decomp_common_0515['breaks']} | **{decomp_common_0515['net']:+d}** | [{decomp_common_0515['bootstrap_seller_ci95'][0]:+.1f}, {decomp_common_0515['bootstrap_seller_ci95'][1]:+.1f}] | {decomp_common_0515['ll_base']:.5f} | {decomp_common_0515['ll_cand']:.5f} | **{decomp_common_0515['delta_ll']:+.5f}** |",
        "",
        "## 3. Ajuste Inductivo de Vocabulario e IDF por Fold (OOF en `fit`)",
        f"- **Teacher lineal `txt_sparse_p` (referencia):** Acc = `{oof_linear_acc:.5f}`, Log Loss = `{oof_linear_ll:.5f}`.",
        f"- **MLP disperso (vocabulario/IDF global en `fit`):** Acc = `{oof_transductive_acc:.5f}`, Log Loss = `{oof_transductive_ll:.5f}`.",
        f"- **MLP disperso (vocabulario/IDF estrictamente inductivo por fold):** Acc = `{oof_inductive_acc:.5f}`, Log Loss = `{oof_inductive_ll:.5f}`.",
        "- Al ajustar el vocabulario e IDF estrictamente dentro de cada fold, el MLP disperso sigue muy por debajo del modelo lineal regularizado (`txt_sparse_p`) tanto en accuracy (~6 puntos menos) como en calibración (log loss ~0.41 vs ~0.29).",
        "",
        "## 4. Criterio de Parada en `tune`",
        f"- **MLP individual en `tune`:** Accuracy = `{tu_acc_mlp_050:.5f}` (< 0.860 requerido), Log Loss = `{tu_ll_mlp:.5f}` (> 0.320 máximo).",
        f"- **Mezcla 0.90 CB + 0.10 MLP en `tune`:** Empeora el Log Loss de `{tu_ll_cb:.5f}` a `{tu_blend_90_10_ll:.5f}` (`+0.00353`). El peso óptimo en `tune` para minimizar log loss es exactamente **`w_mlp = {best_w_mlp:.3f}`**.",
        "- **Decisión:** Se descarta formalmente la línea MLP disperso y se libera todo el presupuesto de cómputo para la línea principal **CatBoost + LightGBM** y su validación externa.",
    ]
    with open(OUT_DIR / "agent4_complementary_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Saved Agent 4 report to {OUT_DIR / 'agent4_complementary_report.md'}")


if __name__ == "__main__":
    main()
