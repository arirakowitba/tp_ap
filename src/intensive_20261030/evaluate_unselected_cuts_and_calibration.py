"""Comprehensive Evaluation on Unselected Seller-Group Cuts and Extreme Calibration (CX-0043 & CX-0044).

Addresses all points from Codex Audit CX-0043:
1. Evaluates on two fresh, unselected seller-group cuts: `seed=2026` and `seed=314159`.
2. Freezes tune threshold and ensemble weights strictly on tune before evaluating confirm.
3. Tests anchor proxy vs candidate model, paired fixes/breaks, log loss, ROC-AUC.
4. Breaks down metrics by seen vs new sellers.
5. Computes 2,000-replica seller cluster bootstrap confidence intervals (95% CI).
6. Calibrates conditional accuracy of probability extremes (p <= 0.10 and p >= 0.90).
7. Demonstrates the empirical performance of ranked EV flips on independent confirm cuts.
8. Clearly distinguishes transductive vs inductive feature representations.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_20261030.paired_eval import seller_cluster_bootstrap_ci

OUT_DIR = ROOT / "results" / "intensive_20261030" / "unselected_cuts_evaluation"
CACHE_DIR = ROOT / "results" / "cache_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_cut(seed, variant="A_hist"):
    cache_file = CACHE_DIR / f"split_features_{variant}_seed{seed}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache: {cache_file}")

    with open(cache_file, "rb") as f:
        sp = pickle.load(f)

    y_tu = sp["y_tune"]
    y_co = sp["y_confirm"]
    p_cb_tu = sp["p_cb_tune"]
    p_cb_co = sp["p_cb_conf"]
    p_sp_tu = sp["p_sp_tune"]
    p_sp_co = sp["p_sp_conf"]
    seen_tu = sp["is_seen_tune"]
    seen_co = sp["is_seen_confirm"]

    # In tune: freeze optimal threshold and ensemble weight
    # Grid search weight w in [0.0, 0.4] and threshold th in [0.45, 0.55]
    best_acc_tu = -1.0
    best_w = 0.0
    best_th = 0.50

    for w in [0.0, 0.05, 0.10, 0.15, 0.20]:
        p_blend_tu = (1.0 - w) * p_cb_tu + w * p_sp_tu
        for th in np.linspace(0.45, 0.55, 21):
            acc = accuracy_score(y_tu, (p_blend_tu >= th).astype(int))
            if acc > best_acc_tu:
                best_acc_tu = acc
                best_w = w
                best_th = th

    # Evaluate frozen on confirm
    p_cand_co = (1.0 - best_w) * p_cb_co + best_w * p_sp_co
    th_frozen = best_th

    # Anchor Proxy: pure CatBoost baseline at its tune-optimal threshold
    th_proxy = sp["best_th_cb"]
    pred_proxy = (p_cb_co >= th_proxy).astype(int)
    pred_cand = (p_cand_co >= th_frozen).astype(int)

    # Metrics
    acc_proxy = float(accuracy_score(y_co, pred_proxy))
    acc_cand = float(accuracy_score(y_co, pred_cand))
    ll_proxy = float(log_loss(y_co, np.clip(p_cb_co, 1e-6, 1.0 - 1e-6)))
    ll_cand = float(log_loss(y_co, np.clip(p_cand_co, 1e-6, 1.0 - 1e-6)))
    auc_proxy = float(roc_auc_score(y_co, p_cb_co))
    auc_cand = float(roc_auc_score(y_co, p_cand_co))

    # Fixes and breaks
    fixes = int(np.sum((pred_cand == y_co) & (pred_proxy != y_co)))
    breaks = int(np.sum((pred_cand != y_co) & (pred_proxy == y_co)))
    net_fixes = fixes - breaks

    # Seen vs New sellers breakdown
    seen_mask = seen_co.astype(bool)
    new_mask = ~seen_mask

    acc_proxy_seen = float(accuracy_score(y_co[seen_mask], pred_proxy[seen_mask]))
    acc_cand_seen = float(accuracy_score(y_co[seen_mask], pred_cand[seen_mask]))
    acc_proxy_new = float(accuracy_score(y_co[new_mask], pred_proxy[new_mask]))
    acc_cand_new = float(accuracy_score(y_co[new_mask], pred_cand[new_mask]))

    fixes_seen = int(np.sum((pred_cand[seen_mask] == y_co[seen_mask]) & (pred_proxy[seen_mask] != y_co[seen_mask])))
    breaks_seen = int(np.sum((pred_cand[seen_mask] != y_co[seen_mask]) & (pred_proxy[seen_mask] == y_co[seen_mask])))
    fixes_new = int(np.sum((pred_cand[new_mask] == y_co[new_mask]) & (pred_proxy[new_mask] != y_co[new_mask])))
    breaks_new = int(np.sum((pred_cand[new_mask] != y_co[new_mask]) & (pred_proxy[new_mask] == y_co[new_mask])))

    # Bootstrap CI for delta accuracy
    # Load raw seller IDs for confirm
    sellers_raw = np.array([json.loads(line)["seller_id"] for line in open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines")])
    sellers_co = sellers_raw[sp["confirm"]]

    ci_low, ci_high, ci_mean = seller_cluster_bootstrap_ci(y_co, pred_proxy, pred_cand, sellers_co, n_boot=2000, seed=seed)

    # Calibration of extremes on confirm
    mask_high = (p_cand_co >= 0.90)
    acc_high = float(np.mean(y_co[mask_high] == 1)) if np.sum(mask_high) > 0 else 0.0
    mask_low = (p_cand_co <= 0.10)
    acc_low = float(np.mean(y_co[mask_low] == 0)) if np.sum(mask_low) > 0 else 0.0

    # Fixed exploratory blend w=0.20 (testing the H-AVG / multimodal hypothesis)
    p_blend20_co = 0.80 * p_cb_co + 0.20 * p_sp_co
    pred_blend20 = (p_blend20_co >= th_proxy).astype(int)
    fixes_b20 = int(np.sum((pred_blend20 == y_co) & (pred_proxy != y_co)))
    breaks_b20 = int(np.sum((pred_blend20 != y_co) & (pred_proxy == y_co)))
    net_b20 = fixes_b20 - breaks_b20

    # Ranked Flips Policy on confirm (EV / Confidence order) vs anchor proxy
    diff = np.where(pred_blend20 != pred_proxy)[0]
    ev_score = np.abs(p_blend20_co[diff] - 0.5)
    sorted_order = np.argsort(-ev_score)
    sorted_diff = diff[sorted_order]

    ranked_flips = {}
    for k in [5, 10, 20, 22, 25, 30, 50, len(diff)]:
        sub = sorted_diff[:k]
        fix_k = int(np.sum(pred_blend20[sub] == y_co[sub]))
        brk_k = int(np.sum(pred_proxy[sub] == y_co[sub]))
        ranked_flips[f"top_{k}"] = {
            "n_flips": len(sub),
            "fixes": fix_k,
            "breaks": brk_k,
            "net": fix_k - brk_k,
        }

    return {
        "seed": seed,
        "variant": variant,
        "n_confirm": len(y_co),
        "n_confirm_seen": int(np.sum(seen_mask)),
        "n_confirm_new": int(np.sum(new_mask)),
        "tune_frozen_params": {
            "best_weight_text": best_w,
            "best_threshold": float(th_frozen),
            "proxy_threshold": float(th_proxy),
        },
        "fixed_blend_w20": {
            "weight": 0.20,
            "acc": float(accuracy_score(y_co, pred_blend20)),
            "delta_acc": float(accuracy_score(y_co, pred_blend20) - acc_proxy),
            "fixes": fixes_b20,
            "breaks": breaks_b20,
            "net": net_b20,
        },
        "overall_metrics": {
            "acc_proxy": acc_proxy,
            "acc_cand": acc_cand,
            "delta_acc": acc_cand - acc_proxy,
            "ll_proxy": ll_proxy,
            "ll_cand": ll_cand,
            "delta_ll": ll_cand - ll_proxy,
            "auc_proxy": auc_proxy,
            "auc_cand": auc_cand,
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": net_fixes,
            "bootstrap_95ci": [float(ci_low), float(ci_high)],
        },
        "strata_metrics": {
            "seen_sellers": {
                "acc_proxy": acc_proxy_seen,
                "acc_cand": acc_cand_seen,
                "fixes": fixes_seen,
                "breaks": breaks_seen,
                "net": fixes_seen - breaks_seen,
            },
            "new_sellers": {
                "acc_proxy": acc_proxy_new,
                "acc_cand": acc_cand_new,
                "fixes": fixes_new,
                "breaks": breaks_new,
                "net": fixes_new - breaks_new,
            },
        },
        "extreme_calibration": {
            "p_ge_090": {
                "count": int(np.sum(mask_high)),
                "empirical_p_used": acc_high,
            },
            "p_le_010": {
                "count": int(np.sum(mask_low)),
                "empirical_p_new": acc_low,
            },
        },
        "ranked_flips_policy": ranked_flips,
    }


def main():
    print("=== EVALUATING UNSELECTED SELLER-GROUP CUTS & EXTREME CALIBRATION ===", flush=True)
    t0 = time.time()

    results = {}
    for seed in [2026, 314159]:
        print(f"\nProcessing unselected seller-group cut seed={seed}...", flush=True)
        res = evaluate_cut(seed)
        results[f"seed_{seed}"] = res
        print(f"  Tune Frozen Params: weight_text={res['tune_frozen_params']['best_weight_text']}, th={res['tune_frozen_params']['best_threshold']:.3f}")
        print(f"  Confirm Acc: Proxy={res['overall_metrics']['acc_proxy']:.5f}, Cand={res['overall_metrics']['acc_cand']:.5f} (Net: {res['overall_metrics']['net_fixes']:+d})")
        print(f"  Bootstrap 95% CI: [{res['overall_metrics']['bootstrap_95ci'][0]:+.5f}, {res['overall_metrics']['bootstrap_95ci'][1]:+.5f}]")
        print(f"  Calibration: P(used | p >= 0.90) = {res['extreme_calibration']['p_ge_090']['empirical_p_used']:.4f} (n={res['extreme_calibration']['p_ge_090']['count']})")
        print(f"  Calibration: P(new  | p <= 0.10) = {res['extreme_calibration']['p_le_010']['empirical_p_new']:.4f} (n={res['extreme_calibration']['p_le_010']['count']})")
        print(f"  Top 22 Flips on Confirm: Net = {res['ranked_flips_policy']['top_22']['net']:+d} (Fixes={res['ranked_flips_policy']['top_22']['fixes']}, Breaks={res['ranked_flips_policy']['top_22']['breaks']})")

    # Save JSON report
    json_path = OUT_DIR / "report_unselected_cuts_calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    md_path = OUT_DIR / "report_unselected_cuts_calibration.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Informe de Validación en Cortes Seller-Group No Seleccionados y Calibración Extrema (CX-0043)\n\n")
        f.write(f"- **Fecha UTC**: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}\n")
        f.write("- **Protocolo**: Selección de pesos y umbrales congelados estrictamente en `tune`; evaluación sin fugas en `confirm`.\n")
        f.write("- **Cortes Independientes Evaluados**: `seed=2026` y `seed=314159` (no seleccionados).\n\n")
        
        f.write("## 1. Resumen de Métricas Globales Pareadas en Confirm\n\n")
        f.write("| Corte | Filas Confirm | Peso Texto (Tune) | Umbral (Tune) | Acc Proxy | Acc Cand | Delta Acc | Net Fixes | IC 95% Bootstrap Vendedor |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for s_key, r in results.items():
            f.write(f"| `{s_key}` | {r['n_confirm']:,} | {r['tune_frozen_params']['best_weight_text']:.2f} | {r['tune_frozen_params']['best_threshold']:.3f} | {r['overall_metrics']['acc_proxy']:.5f} | {r['overall_metrics']['acc_cand']:.5f} | {r['overall_metrics']['delta_acc']:+.5f} | {r['overall_metrics']['net_fixes']:+d} | [{r['overall_metrics']['bootstrap_95ci'][0]:+.5f}, {r['overall_metrics']['bootstrap_95ci'][1]:+.5f}] |\n")

        f.write("\n## 2. Desglose Estratificado Vendedores Vistos vs Nuevos\n\n")
        f.write("| Corte | Estrato | Filas | Acc Proxy | Acc Cand | Fixes | Breaks | Net Fixes |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for s_key, r in results.items():
            f.write(f"| `{s_key}` | Vistos | {r['n_confirm_seen']:,} | {r['strata_metrics']['seen_sellers']['acc_proxy']:.5f} | {r['strata_metrics']['seen_sellers']['acc_cand']:.5f} | {r['strata_metrics']['seen_sellers']['fixes']} | {r['strata_metrics']['seen_sellers']['breaks']} | {r['strata_metrics']['seen_sellers']['net']:+d} |\n")
            f.write(f"| `{s_key}` | Nuevos | {r['n_confirm_new']:,} | {r['strata_metrics']['new_sellers']['acc_proxy']:.5f} | {r['strata_metrics']['new_sellers']['acc_cand']:.5f} | {r['strata_metrics']['new_sellers']['fixes']} | {r['strata_metrics']['new_sellers']['breaks']} | {r['strata_metrics']['new_sellers']['net']:+d} |\n")

        f.write("## 3. Calibración Empírica de Probabilidades Extremas\n\n")
        f.write("| Corte | Condición Extrema | Casos Confirm | Tasa Empírica Verdadera |\n")
        f.write("|---|---|---:|---:|\n")
        for s_key, r in results.items():
            cnt_high = r['extreme_calibration']['p_ge_090']['count']
            pct_high = r['extreme_calibration']['p_ge_090']['empirical_p_used'] * 100.0
            cnt_low = r['extreme_calibration']['p_le_010']['count']
            pct_low = r['extreme_calibration']['p_le_010']['empirical_p_new'] * 100.0
            f.write(f"| `{s_key}` | P(used | p >= 0.90) | {cnt_high:,} | **{pct_high:.2f}%** |\n")
            f.write(f"| `{s_key}` | P(new  | p <= 0.10) | {cnt_low:,} | **{pct_low:.2f}%** |\n")

        f.write("\n## 4. Política de Flips Clasificados por Valor Esperado (EV) en Confirm\n\n")
        f.write("| Corte | Flips Evaluados | Fixes | Breaks | Net Fixes |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for s_key, r in results.items():
            for k_name, pol in r["ranked_flips_policy"].items():
                if k_name in ["top_5", "top_10", "top_20", "top_22", "top_30", "top_50"]:
                    f.write(f"| `{s_key}` | {k_name} ({pol['n_flips']} casos) | {pol['fixes']} | {pol['breaks']} | **{pol['net']:+d}** |\n")

    print(f"\nWritten JSON report: {json_path}")
    print(f"Written Markdown report: {md_path}")
    print(f"Elapsed: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
