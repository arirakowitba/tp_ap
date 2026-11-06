"""Exportador de predicciones de probabilidad de Variant C en Tune y Confirm (CX-0017).

Genera archivos CSV comprimidos con row_idx, item_id, split ('tune', 'confirm'), y_true, y p_variant_c
para las tres semillas (2026, 314159, 20261030), con el objetivo de compartirlas con Claude
para su auditoría y contraste pareado independiente con B10.
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_20261030.feature_extractor import compute_txt_sparse_p
from src.benchmark_20261030.split_utils import split_seller_groups_r151
from src.coordination_lock import ProjectLock
from src.eval_20261030.submission_guard import sha256_bytes
from src.intensive_20261030.seller_te_ablation import compute_target_encodings_ablation
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_variant_c_tune_confirm_probs():
    lock = ProjectLock(owner_id="antigravity_exporter_variant_c")
    print("Acquiring ProjectLock for Variant C tune/confirm probability export...", flush=True)
    with lock:
        cache = build_or_load_cache()
        F_r151_tr = cache["F_r151_tr"]
        cat_tr = cache["cat_tr"]
        y_tr = cache["y_tr"]
        sellers_tr = cache["sellers_tr"]
        ids_tr = cache["ids_tr"]
        combos = cache["combos_hist_tr"]
        titles = cache["titles_tr"]

        seeds = [2026, 314159, 20261030]
        manifest = {}

        for seed in seeds:
            print(f"\n--- Generating Variant C Tune + Confirm Probs for Seed {seed} ---", flush=True)
            t0 = time.time()
            fit, tune, confirm = split_seller_groups_r151(sellers_tr, seed)
            y_fit = y_tr[fit]
            y_tu = y_tr[tune]
            y_co = y_tr[confirm]
            sellers_fit = sellers_tr[fit]

            # 1. txt_sparse_p
            print("  Computing txt_sparse_p...", flush=True)
            oof_sp, (p_sp_tune, p_sp_conf) = compute_txt_sparse_p(
                combos, titles, fit, [tune, confirm], y_fit, sellers_fit, seed=seed
            )

            # 2. Target Encoding Variant A (Group TE estándar)
            print("  Computing Target Encodings...", flush=True)
            fit_te, tune_te, conf_te = compute_target_encodings_ablation(
                cat_tr, fit, tune, confirm, y_fit, sellers_fit, variant="variant_a"
            )

            # 3. DataFrames
            X_fit = F_r151_tr.iloc[fit].copy().reset_index(drop=True)
            X_tu = F_r151_tr.iloc[tune].copy().reset_index(drop=True)
            X_co = F_r151_tr.iloc[confirm].copy().reset_index(drop=True)

            for c in ["seller_id", "category_id", "city"]:
                X_fit[f"te_{c}"] = fit_te[c]
                X_tu[f"te_{c}"] = tune_te[c]
                X_co[f"te_{c}"] = conf_te[c]

            X_fit["txt_sparse_p"] = oof_sp
            X_tu["txt_sparse_p"] = p_sp_tune
            X_co["txt_sparse_p"] = p_sp_conf

            cb_cat_features = list(CAT_FEATURES) + ["seller_id"]
            X_fit["seller_id"] = cat_tr["seller_id"].iloc[fit].astype(str).values
            X_tu["seller_id"] = cat_tr["seller_id"].iloc[tune].astype(str).values
            X_co["seller_id"] = cat_tr["seller_id"].iloc[confirm].astype(str).values

            for c in CAT_FEATURES:
                X_fit[c] = cat_tr[c].iloc[fit].astype(str).values
                X_tu[c] = cat_tr[c].iloc[tune].astype(str).values
                X_co[c] = cat_tr[c].iloc[confirm].astype(str).values

            # 4. Train CatBoost
            print("  Training CatBoost Variant C...", flush=True)
            cb = CatBoostClassifier(
                iterations=900,
                learning_rate=0.07,
                depth=6,
                l2_leaf_reg=6.0,
                random_seed=seed,
                thread_count=4,
                verbose=0,
            )
            pool_fit = Pool(X_fit, y_fit, cat_features=cb_cat_features)
            pool_tu = Pool(X_tu, y_tu, cat_features=cb_cat_features)
            pool_co = Pool(X_co, cat_features=cb_cat_features)

            cb.fit(pool_fit, eval_set=pool_tu, early_stopping_rounds=80, verbose=0)
            best_iter = int(cb.get_best_iteration())
            print(f"  Best iteration on tune: {best_iter}", flush=True)

            p_tu = cb.predict_proba(pool_tu)[:, 1]
            p_co = cb.predict_proba(pool_co)[:, 1]

            # 5. Build combined DataFrame
            df_tu = pd.DataFrame({
                "row_idx": tune,
                "item_id": ids_tr[tune],
                "split": "tune",
                "y_true": y_tu,
                "p_variant_c": p_tu.astype(np.float32),
            })
            df_co = pd.DataFrame({
                "row_idx": confirm,
                "item_id": ids_tr[confirm],
                "split": "confirm",
                "y_true": y_co,
                "p_variant_c": p_co.astype(np.float32),
            })
            df_combined = pd.concat([df_tu, df_co], ignore_index=True)

            out_csv = OUT_DIR / f"preds_variant_c_seed{seed}_tune_confirm.csv.gz"
            df_combined.to_csv(out_csv, index=False, compression="gzip")
            file_bytes = out_csv.read_bytes()
            file_sha = sha256_bytes(file_bytes)

            manifest[str(seed)] = {
                "file_name": out_csv.name,
                "sha256": file_sha,
                "n_tune": len(df_tu),
                "n_confirm": len(df_co),
                "best_iteration_tune": best_iter,
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            print(f"  Saved {out_csv.name} (SHA256: {file_sha[:16]}... in {manifest[str(seed)]['elapsed_seconds']}s)", flush=True)

        meta_path = OUT_DIR / "preds_variant_c_tune_confirm_manifest.json"
        meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nAll done! Manifest written to {meta_path}", flush=True)


if __name__ == "__main__":
    export_variant_c_tune_confirm_probs()
