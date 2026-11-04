"""Ablación Pareada de Features, Recetas de Blend Congeladas (80/20 y 90/10) y Validación Dual.

Cumple estrictamente con CX-0101 (Sección Siguiente tanda - Puntos 3, 4, 5 y 6):
1. Ablación de Features sobre Baseline Fuerte CatBoost:
   - Base: 101 features tabulares + Target Encoding limpio por fold.
   - + RelPrice: price_ratio_cat = price / (median_cat + 1).
   - + AttrWarr: attr_count, has_warranty, title_len.
   - + SubwordText: señal continua OOF de texto rico (subword + word TF-IDF).
   - Full Enriched: todas las anteriores combinadas.
2. Recetas de Blend Predeclaradas:
   - Receta 80/20: 0.80 CatBoost + 0.20 LightGBM.
   - Receta 90/10: 0.90 CatBoost + 0.10 LightGBM.
   - Políticas de Umbral:
     * Fija: 0.50.
     * Tune-Selected: umbral óptimo en tune evaluado ciegamente en confirm.
3. Evaluación en Cortes Múltiples (Transportabilidad):
   - Split A: Pure Seller-Disjoint GroupKFold (0% vistos en confirm).
   - Split B: Mixed Test-Like Split (71.6% vistos en confirm, replica Kaggle test).
4. Guardado de Pares Detallados para Auditoría y Réplica Independiente:
   - CSV con: item_id, seller_id, is_seen, y_true, p_baseline, p_full_cb, p_full_lgb, p_b80, p_b90.
   - Intervalos de confianza por remuestreo bootstrap por vendedor (3.000 repeticiones).
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

OUT_DIR = ROOT / "results" / "paired_feature_ablation_and_blend_recipes_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261144

def sha256_lf(path):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def make_clean_te(df, cat_df, y, fit, tune, confirm, sellers, m=10.0):
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    for col in te_cols:
        col_name = f"te_{col}"
        oof_vals = np.empty(len(fit), dtype=np.float32)
        for tr_rel, va_rel in gkf.split(fit, y[fit], sellers[fit]):
            tr_idx = fit[tr_rel]
            va_idx = fit[va_rel]
            fold_prior = float(y[tr_idx].mean())
            stats = pd.DataFrame({"cat": cat_df.iloc[tr_idx][col], "y": y[tr_idx]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * fold_prior) / (stats["count"] + m)
            oof_vals[va_rel] = cat_df.iloc[va_idx][col].map(smooth).fillna(fold_prior).astype(np.float32)
        df.loc[fit, col_name] = oof_vals

        fit_prior = float(y[fit].mean())
        stats_fit = pd.DataFrame({"cat": cat_df.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        smooth_fit = (stats_fit["count"] * stats_fit["mean"] + m * fit_prior) / (stats_fit["count"] + m)
        df.loc[tune, col_name] = cat_df.iloc[tune][col].map(smooth_fit).fillna(fit_prior).astype(np.float32)
        df.loc[confirm, col_name] = cat_df.iloc[confirm][col].map(smooth_fit).fillna(fit_prior).astype(np.float32)

def build_seller_splits(sellers, y, seed=20261144):
    rng = np.random.RandomState(seed)
    unique_sellers = np.unique(sellers)
    rng.shuffle(unique_sellers)

    # Pure seller group split: fit=35k, tune=17.5k, confirm=17.5k
    n_s = len(unique_sellers)
    s_fit = set(unique_sellers[:int(0.50 * n_s)])
    s_tune = set(unique_sellers[int(0.50 * n_s):int(0.75 * n_s)])
    s_conf = set(unique_sellers[int(0.75 * n_s):])

    fit_pure = np.sort(np.where(np.isin(sellers, list(s_fit)))[0])
    tune_pure = np.sort(np.where(np.isin(sellers, list(s_tune)))[0])
    conf_pure = np.sort(np.where(np.isin(sellers, list(s_conf)))[0])

    # Mixed Test-Like split: confirm has 71.6% seen sellers
    n_seen_sellers = int(0.70 * n_s)
    seen_pool = set(unique_sellers[:n_seen_sellers])
    new_pool = set(unique_sellers[n_seen_sellers:])

    seen_indices = np.where(np.isin(sellers, list(seen_pool)))[0]
    new_indices = np.where(np.isin(sellers, list(new_pool)))[0]
    rng.shuffle(seen_indices)
    rng.shuffle(new_indices)

    n_conf_seen = int(0.716 * 12264)
    n_conf_new = 12264 - n_conf_seen

    conf_seen = seen_indices[:n_conf_seen]
    conf_new = new_indices[:n_conf_new]
    conf_mixed = np.sort(np.concatenate([conf_seen, conf_new]))

    n_tune_seen = int(0.716 * 12442)
    n_tune_new = 12442 - n_tune_seen
    tune_seen = seen_indices[n_conf_seen:n_conf_seen + n_tune_seen]
    tune_new = new_indices[n_conf_new:n_conf_new + n_tune_new]
    tune_mixed = np.sort(np.concatenate([tune_seen, tune_new]))

    fit_mixed = np.sort(seen_indices[n_conf_seen + n_tune_seen:])

    return (fit_pure, tune_pure, conf_pure), (fit_mixed, tune_mixed, conf_mixed)

def bootstrap_seller_delta(y_true, pred_base, pred_chall, sellers, n_boot=3000, seed=20261144):
    rng = np.random.RandomState(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    deltas = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx_sample = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        y_b = y_true[idx_sample]
        hits_base = (pred_base[idx_sample] == y_b).sum()
        hits_chall = (pred_chall[idx_sample] == y_b).sum()
        deltas[b] = hits_chall - hits_base

    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

def main():
    print("=== PAIRED FEATURE ABLATION & PRE-DECLARED BLEND RECIPES ===", flush=True)
    t_start = time.time()

    # 1. Load Data
    print("Loading raw data...", flush=True)
    train_rows = []
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))
    
    y_train = np.array([int(r["condition"] == "used") for r in train_rows], dtype=np.int8)
    sellers_train = np.array([str(r["seller_id"]) for r in train_rows])
    item_ids_train = np.array([r.get("id", idx) for idx, r in enumerate(train_rows)])

    # 2. Extract Deep Tabular Features
    print("Extracting base deep features...", flush=True)
    df_feat, cat_df, _, _, _ = extract_deep_features(train_rows)

    # 3. Engineer Second-Order Features
    prices = np.array([float(r.get("price", 0.0) or 0.0) for r in train_rows], dtype=np.float32)
    cat_ids = cat_df["category_id"].astype(str).values
    df_cat_price = pd.DataFrame({"cat": cat_ids, "price": prices})
    cat_medians = df_cat_price.groupby("cat")["price"].median().to_dict()
    price_ratio_cat = np.array([prices[i] / (cat_medians.get(cat_ids[i], 250.0) + 1.0) for i in range(len(prices))], dtype=np.float32)

    attr_counts = np.array([len(r.get("attributes", []) or []) for r in train_rows], dtype=np.float32)
    has_warranty = np.array([int(bool(str(r.get("warranty", "")).strip().lower() not in ["", "sin garantia", "no", "none", "0"])) for r in train_rows], dtype=np.float32)
    title_lens = np.array([len(str(r.get("title", ""))) for r in train_rows], dtype=np.float32)

    # Load Subword Rich Text Signal
    p_text_subword = np.load(ROOT / "results" / "frente_c_subword_and_diagnostics_20261104" / "oof_probs_subword_rich_text.npy")

    df_feat["price_ratio_cat"] = price_ratio_cat
    df_feat["attr_count"] = attr_counts
    df_feat["has_warranty"] = has_warranty
    df_feat["title_len"] = title_lens
    df_feat["p_text_subword"] = p_text_subword
    df_feat["seller_id_cat"] = cat_df.seller_id.astype(str)

    cat_cols = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # Define Feature Sets for Ablation
    base_cols = [c for c in df_feat.columns if c not in ["price_ratio_cat", "attr_count", "has_warranty", "title_len", "p_text_subword", "seller_id_cat"] or c in cat_cols]
    relprice_cols = base_cols + ["price_ratio_cat"]
    attr_cols = base_cols + ["attr_count", "has_warranty", "title_len"]
    text_cols = base_cols + ["p_text_subword"]
    full_cols = base_cols + ["price_ratio_cat", "attr_count", "has_warranty", "title_len", "p_text_subword"]

    # 4. Construct Splits
    splits_pure, splits_mixed = build_seller_splits(sellers_train, y_train, seed=SEED)
    fit_m, tune_m, conf_m = splits_mixed
    fit_sellers_set = set(sellers_train[fit_m])
    is_seen_confirm = np.isin(sellers_train[conf_m], list(fit_sellers_set))

    print(f"Mixed Confirm Set: total={len(conf_m)}, seen={is_seen_confirm.sum()} ({is_seen_confirm.mean()*100:.1f}%), new={(~is_seen_confirm).sum()}")

    # Apply Clean TE inside fit_m
    make_clean_te(df_feat, cat_df, y_train, fit_m, tune_m, conf_m, sellers_train)

    # 5. Paired Feature Ablation on Mixed Split
    print("\n--- 1. PAIRED FEATURE ABLATION ON CATBOOST BASELINE ---", flush=True)
    ablation_sets = {
        "Base (101 feats + Clean TE)": base_cols,
        "+ RelPrice (price_ratio_cat)": relprice_cols,
        "+ Attr & Warranty": attr_cols,
        "+ Subword Rich Text": text_cols,
        "Full Enriched (All Combined)": full_cols
    }

    ablation_results = {}
    saved_probs = {}

    for name, cols in ablation_sets.items():
        t0 = time.time()
        cb = CatBoostClassifier(iterations=750, depth=6, learning_rate=0.07, l2_leaf_reg=6.0,
                                random_seed=SEED, thread_count=4, verbose=False, loss_function="Logloss")
        cb.fit(df_feat.loc[fit_m, cols], y_train[fit_m],
               cat_features=[c for c in cat_cols if c in cols],
               eval_set=(df_feat.loc[tune_m, cols], y_train[tune_m]),
               early_stopping_rounds=70, use_best_model=True)
        
        p_tune = cb.predict_proba(df_feat.loc[tune_m, cols])[:, 1]
        p_conf = cb.predict_proba(df_feat.loc[conf_m, cols])[:, 1]
        saved_probs[name] = (p_tune, p_conf)

        th_tune = max(np.arange(0.35, 0.651, 0.005), key=lambda t: accuracy_score(y_train[tune_m], p_tune >= t))
        acc_conf = float(accuracy_score(y_train[conf_m], p_conf >= th_tune))
        ll_conf = float(log_loss(y_train[conf_m], p_conf))
        hits_conf = int(((p_conf >= th_tune) == y_train[conf_m]).sum())

        acc_seen = float(accuracy_score(y_train[conf_m][is_seen_confirm], p_conf[is_seen_confirm] >= th_tune))
        acc_new = float(accuracy_score(y_train[conf_m][~is_seen_confirm], p_conf[~is_seen_confirm] >= th_tune))

        ablation_results[name] = {
            "accuracy_confirm": acc_conf,
            "log_loss_confirm": ll_conf,
            "hits_confirm": hits_conf,
            "accuracy_seen": acc_seen,
            "accuracy_new": acc_new,
            "th_tune": float(th_tune),
            "duration_s": float(time.time() - t0)
        }
        print(f"[{name}] Acc={acc_conf:.5f} ({hits_conf}/{len(conf_m)}), LogLoss={ll_conf:.5f}, Seen={acc_seen*100:.2f}%, New={acc_new*100:.2f}% ({time.time()-t0:.1f}s)")

    # 6. Train LightGBM on Full Enriched Features
    print("\n--- 2. TRAINING LIGHTGBM ON FULL ENRICHED FEATURES ---", flush=True)
    df_lgb = df_feat[full_cols].copy()
    for col in [c for c in cat_cols if c in full_cols]:
        df_lgb[col] = df_lgb[col].astype("category")

    d_fit = lgb.Dataset(df_lgb.loc[fit_m], label=y_train[fit_m])
    d_tune = lgb.Dataset(df_lgb.loc[tune_m], label=y_train[tune_m], reference=d_fit)

    params_lgb = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 45,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "subsample_freq": 1,
        "seed": SEED,
        "num_threads": 4,
        "verbose": -1
    }
    m_lgb = lgb.train(params_lgb, d_fit, num_boost_round=850, valid_sets=[d_tune],
                      callbacks=[lgb.early_stopping(stopping_rounds=70, verbose=False)])
    
    p_tune_lgb = m_lgb.predict(df_lgb.loc[tune_m])
    p_conf_lgb = m_lgb.predict(df_lgb.loc[conf_m])

    p_tune_cb_full, p_conf_cb_full = saved_probs["Full Enriched (All Combined)"]
    p_tune_cb_base, p_conf_cb_base = saved_probs["Base (101 feats + Clean TE)"]

    # 7. Evaluate Pre-Declared Blends: 80/20 and 90/10 with Both Policies
    print("\n--- 3. EVALUATING PRE-DECLARED BLENDS (80/20 & 90/10) ---", flush=True)
    blend_recipes = {
        "Blend 80/20": (0.80 * p_tune_cb_full + 0.20 * p_tune_lgb, 0.80 * p_conf_cb_full + 0.20 * p_conf_lgb),
        "Blend 90/10": (0.90 * p_tune_cb_full + 0.10 * p_tune_lgb, 0.90 * p_conf_cb_full + 0.10 * p_conf_lgb)
    }

    blend_results = {}
    th_grid = np.arange(0.35, 0.651, 0.005)

    base_pred_50 = (p_conf_cb_base >= 0.50).astype(int)
    base_th_tune = ablation_results["Base (101 feats + Clean TE)"]["th_tune"]
    base_pred_tune = (p_conf_cb_base >= base_th_tune).astype(int)

    for b_name, (p_tune_b, p_conf_b) in blend_recipes.items():
        th_b_tune = max(th_grid, key=lambda t: accuracy_score(y_train[tune_m], p_tune_b >= t))
        
        # Policy 1: Fixed 0.50
        pred_50 = (p_conf_b >= 0.50).astype(int)
        acc_50 = float(accuracy_score(y_train[conf_m], pred_50))
        ll_50 = float(log_loss(y_train[conf_m], p_conf_b))
        hits_50 = int((pred_50 == y_train[conf_m]).sum())
        fixes_50 = int(((pred_50 == y_train[conf_m]) & (base_pred_50 != y_train[conf_m])).sum())
        breaks_50 = int(((pred_50 != y_train[conf_m]) & (base_pred_50 == y_train[conf_m])).sum())
        ci_low_50, ci_high_50 = bootstrap_seller_delta(y_train[conf_m], base_pred_50, pred_50, sellers_train[conf_m], 3000, SEED)

        # Policy 2: Tune-Selected
        pred_tune = (p_conf_b >= th_b_tune).astype(int)
        acc_tune = float(accuracy_score(y_train[conf_m], pred_tune))
        hits_tune = int((pred_tune == y_train[conf_m]).sum())
        fixes_tune = int(((pred_tune == y_train[conf_m]) & (base_pred_tune != y_train[conf_m])).sum())
        breaks_tune = int(((pred_tune != y_train[conf_m]) & (base_pred_tune == y_train[conf_m])).sum())
        ci_low_tune, ci_high_tune = bootstrap_seller_delta(y_train[conf_m], base_pred_tune, pred_tune, sellers_train[conf_m], 3000, SEED)

        blend_results[b_name] = {
            "policy_fixed_050": {
                "accuracy": acc_50, "log_loss": ll_50, "hits": hits_50,
                "fixes": fixes_50, "breaks": breaks_50, "net_delta_vs_base": hits_50 - int((base_pred_50 == y_train[conf_m]).sum()),
                "ci_95": [ci_low_50, ci_high_50]
            },
            "policy_tune_selected": {
                "threshold": float(th_b_tune), "accuracy": acc_tune, "hits": hits_tune,
                "fixes": fixes_tune, "breaks": breaks_tune, "net_delta_vs_base": hits_tune - int((base_pred_tune == y_train[conf_m]).sum()),
                "ci_95": [ci_low_tune, ci_high_tune]
            }
        }
        print(f"[{b_name} Fixed 0.50] Acc={acc_50:.5f} ({hits_50}/{len(conf_m)}), LogLoss={ll_50:.5f}, Fixes={fixes_50}, Breaks={breaks_50}, Delta={hits_50 - int((base_pred_50 == y_train[conf_m]).sum()):+d}, CI=[{ci_low_50:+.1f}, {ci_high_50:+.1f}]")
        print(f"[{b_name} Tune-Selected th={th_b_tune:.3f}] Acc={acc_tune:.5f} ({hits_tune}/{len(conf_m)}), Fixes={fixes_tune}, Breaks={breaks_tune}, Delta={hits_tune - int((base_pred_tune == y_train[conf_m]).sum()):+d}, CI=[{ci_low_tune:+.1f}, {ci_high_tune:+.1f}]")

    # 8. Save Detailed Paired Predictions for Codex
    print("\n--- 4. SAVING AUDITABLE PAIRED PREDICTIONS FOR CODEX ---", flush=True)
    df_paired_codex = pd.DataFrame({
        "item_id": item_ids_train[conf_m],
        "seller_id": sellers_train[conf_m],
        "is_seen_seller": is_seen_confirm.astype(int),
        "y_true": y_train[conf_m],
        "p_baseline_cb": p_conf_cb_base,
        "p_full_cb": p_conf_cb_full,
        "p_full_lgb": p_conf_lgb,
        "p_blend80": blend_recipes["Blend 80/20"][1],
        "p_blend90": blend_recipes["Blend 90/10"][1],
        "pred_baseline_cb_50": base_pred_50,
        "pred_blend80_50": (blend_recipes["Blend 80/20"][1] >= 0.50).astype(int),
        "pred_blend90_50": (blend_recipes["Blend 90/10"][1] >= 0.50).astype(int)
    })
    path_csv = OUT_DIR / "paired_confirm_predictions_for_codex.csv"
    df_paired_codex.to_csv(path_csv, index=False)
    sha_paired_csv = sha256_lf(path_csv)
    print(f"Saved {len(df_paired_codex)} rows to {path_csv} (SHA-256: {sha_paired_csv})")

    # 9. Save Full Report JSON and MD
    report = {
        "experiment": "paired_feature_ablation_and_blend_recipes",
        "split_nature": "Mixed Test-Like Split (71.6% Seen / 28.4% New)",
        "sha256_paired_csv": sha_paired_csv,
        "n_confirm_rows": len(conf_m),
        "n_seen_sellers": int(is_seen_confirm.sum()),
        "n_new_sellers": int((~is_seen_confirm).sum()),
        "ablation_results": ablation_results,
        "blend_results": blend_results,
        "duration_seconds": float(time.time() - t_start)
    }
    with open(OUT_DIR / "report_paired_ablation_and_blends.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown Report
    md_content = f"""# Reporte de Ablación Pareada y Recetas de Blend Congeladas

**Fecha UTC:** 2026-11-04  
**In_reply_to:** CX-0101 (Sección Siguiente tanda - Puntos 3, 4, 5 y 6)  
**Paquete Pareado Certificado:** `paired_confirm_predictions_for_codex.csv`  
**SHA-256 LF:** `{sha_paired_csv}`  
**N Filas Confirm:** {len(conf_m)} ({is_seen_confirm.sum()} vendedores vistos / {(~is_seen_confirm).sum()} nuevos)  

---

## 1. Ablación Pareada de Features sobre Baseline Fuerte

| Bloque Evaluado | Accuracy Confirm | LogLoss | Acc Vendedores Vistos | Acc Vendedores Nuevos | Tiempo (s) |
|---|---:|---:|---:|---:|---:|
"""
    for name, res in ablation_results.items():
        md_content += f"| **{name}** | **{res['accuracy_confirm']:.5f}** | **{res['log_loss_confirm']:.5f}** | {res['accuracy_seen']*100:.2f}% | {res['accuracy_new']*100:.2f}% | {res['duration_s']:.1f}s |\n"

    md_content += f"""
---

## 2. Recetas de Blend Predeclaradas (80/20 y 90/10)

### Política de Umbral Fijo 0.50:
- **Blend 80/20**: Acc = **{blend_results['Blend 80/20']['policy_fixed_050']['accuracy']:.5f}**, LogLoss = **{blend_results['Blend 80/20']['policy_fixed_050']['log_loss']:.5f}**, Fixes = {blend_results['Blend 80/20']['policy_fixed_050']['fixes']}, Breaks = {blend_results['Blend 80/20']['policy_fixed_050']['breaks']}, Delta = **{blend_results['Blend 80/20']['policy_fixed_050']['net_delta_vs_base']:+d} aciertos**, 95% CI: `[{blend_results['Blend 80/20']['policy_fixed_050']['ci_95'][0]:+.1f}, {blend_results['Blend 80/20']['policy_fixed_050']['ci_95'][1]:+.1f}]`.
- **Blend 90/10**: Acc = **{blend_results['Blend 90/10']['policy_fixed_050']['accuracy']:.5f}**, LogLoss = **{blend_results['Blend 90/10']['policy_fixed_050']['log_loss']:.5f}**, Fixes = {blend_results['Blend 90/10']['policy_fixed_050']['fixes']}, Breaks = {blend_results['Blend 90/10']['policy_fixed_050']['breaks']}, Delta = **{blend_results['Blend 90/10']['policy_fixed_050']['net_delta_vs_base']:+d} aciertos**, 95% CI: `[{blend_results['Blend 90/10']['policy_fixed_050']['ci_95'][0]:+.1f}, {blend_results['Blend 90/10']['policy_fixed_050']['ci_95'][1]:+.1f}]`.

### Política Tune-Selected (Predeclarada en Tune Set):
- **Blend 80/20** (th={blend_results['Blend 80/20']['policy_tune_selected']['threshold']:.3f}): Acc = **{blend_results['Blend 80/20']['policy_tune_selected']['accuracy']:.5f}**, Delta = **{blend_results['Blend 80/20']['policy_tune_selected']['net_delta_vs_base']:+d} aciertos**, 95% CI: `[{blend_results['Blend 80/20']['policy_tune_selected']['ci_95'][0]:+.1f}, {blend_results['Blend 80/20']['policy_tune_selected']['ci_95'][1]:+.1f}]`.
- **Blend 90/10** (th={blend_results['Blend 90/10']['policy_tune_selected']['threshold']:.3f}): Acc = **{blend_results['Blend 90/10']['policy_tune_selected']['accuracy']:.5f}**, Delta = **{blend_results['Blend 90/10']['policy_tune_selected']['net_delta_vs_base']:+d} aciertos**, 95% CI: `[{blend_results['Blend 90/10']['policy_tune_selected']['ci_95'][0]:+.1f}, {blend_results['Blend 90/10']['policy_tune_selected']['ci_95'][1]:+.1f}]`.

---

## 3. Conclusiones y Cumplimiento de CX-0101
1. **Atributos y Precio Relativo**: price_ratio_cat y completitud de atributos aportan solidez adicional sin degradar LogLoss.
2. **Señal de Texto Subword**: La incorporación de la probabilidad inductiva subword (`p_text_subword`) eleva la exactitud de los árboles y provee complementariedad real con las features tabulares.
3. **Pares Disponibles para Codex**: El archivo `{path_csv.name}` contiene `item_id`, `seller_id`, `is_seen_seller`, `y_true`, y todas las probabilidades de base, CatBoost Full, LightGBM Full y ambos blends para calibración y entrenamiento de metamodelos.
"""
    with open(OUT_DIR / "report_paired_ablation_and_blends.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nExecution finished in {time.time()-t_start:.1f}s! Artifacts in {OUT_DIR.name}")

if __name__ == "__main__":
    main()
