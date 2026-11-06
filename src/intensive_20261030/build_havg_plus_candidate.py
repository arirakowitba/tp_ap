"""Entrenamiento y ensamble del modelo H-AVG-Plus (B10 + Variant C Blend con Bagging 3-seed).

Aborda integralmente las observaciones de CL-0015 y CL-0016:
1. Reemplaza el modelo single-seed por un ensamble embolsado (Bagging) de CatBoost con 3 semillas
   (2026, 314159, 20261030, 600 iteraciones cada una), eliminando la variabilidad no determinista de CPU.
2. Incorpora el componente LightGBM entrenado sobre los 70k completos con n_estimators=227
   (la mediana exacta de las mejores iteraciones de tune reportadas por Claude y Antigravity: 189, 227, 257).
3. Construye el blend formal de Variant C: 0.75 * CB_Bagged + 0.25 * LGB_Full70k.
4. Ensamble H-AVG-Plus: 0.50 * B10 (Claude) + 0.50 * Variant_C_Blend.
5. Inferencia estricta sobre 30.000 filas de test con umbral 0.50.
6. Verificación criptográfica con submission_guard.py.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordination_lock import ProjectLock
from src.eval_20261030.submission_guard import check_submission_file, sha256_bytes
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "candidate_model_havg_plus_full70k.csv"
META_OUT = OUT_DIR / "candidate_model_havg_plus_metadata.json"


def train_and_build_havg_plus():
    lock = ProjectLock(owner_id="antigravity_builder_havg_plus")
    print("Acquiring ProjectLock for H-AVG-Plus construction...", flush=True)
    with lock:
        t0 = time.time()
        print("1. Loading shared feature cache...", flush=True)
        cache = build_or_load_cache()

        F_tr = cache["F_r151_tr"].copy()
        F_te = cache["F_ind_te"].copy()
        cat_tr = cache["cat_tr"]
        cat_te = cache["cat_te"]
        y_tr = cache["y_tr"]
        sellers_tr = cache["sellers_tr"]
        sellers_te = cache["sellers_te"]
        combos_hist_tr = cache["combos_hist_tr"]
        combos_hist_te = cache["combos_hist_te"]
        titles_tr = cache["titles_tr"]
        titles_te = cache["titles_te"]

        # 2. Compute txt_sparse_p teacher
        print("2. Computing txt_sparse_p teacher features...", flush=True)
        vec_combo = TfidfVectorizer(max_features=12000, min_df=2, ngram_range=(1, 2), norm="l2")
        vec_title = TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2), norm="l2")

        X_combo_tr = vec_combo.fit_transform(combos_hist_tr)
        X_combo_te = vec_combo.transform(combos_hist_te)
        X_title_tr = vec_title.fit_transform(titles_tr)
        X_title_te = vec_title.transform(titles_te)

        X_sp_tr = hstack([X_combo_tr, X_title_tr]).tocsr()
        X_sp_te = hstack([X_combo_te, X_title_te]).tocsr()

        gkf = GroupKFold(n_splits=4)
        oof_sp = np.zeros(len(y_tr), dtype=np.float32)
        for tr_idx, va_idx in gkf.split(F_tr, y_tr, sellers_tr):
            lr_sp = LogisticRegression(C=3.0, max_iter=250, solver="liblinear", random_state=2026)
            lr_sp.fit(X_sp_tr[tr_idx], y_tr[tr_idx])
            oof_sp[va_idx] = lr_sp.predict_proba(X_sp_tr[va_idx])[:, 1]

        lr_full = LogisticRegression(C=3.0, max_iter=300, solver="liblinear", random_state=2026).fit(X_sp_tr, y_tr)
        p_sp_te = lr_full.predict_proba(X_sp_te)[:, 1].astype(np.float32)

        # 3. Target encodings
        print("3. Computing Group Target Encodings...", flush=True)
        prior = float(y_tr.mean())
        te_cols = ["seller_id", "category_id", "city"]
        alpha = 5.0

        fit_te = {c: np.full(len(F_tr), prior, dtype=np.float32) for c in te_cols}
        test_te = {c: np.full(len(F_te), prior, dtype=np.float32) for c in te_cols}

        for tr_idx, va_idx in gkf.split(F_tr, y_tr, sellers_tr):
            for c in ["category_id", "city"]:
                k_tr = cat_tr[c].iloc[tr_idx]
                k_va = cat_tr[c].iloc[va_idx]
                stats = pd.DataFrame({"k": k_tr, "y": y_tr[tr_idx]}).groupby("k").y.agg(["sum", "count"])
                smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
                fit_te[c][va_idx] = k_va.map(smoothed).fillna(prior).values.astype(np.float32)

        # Map to test from full 70k
        for c in te_cols:
            k_fit = cat_tr[c]
            stats = pd.DataFrame({"k": k_fit, "y": y_tr}).groupby("k").y.agg(["sum", "count"])
            smoothed = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            test_te[c] = cat_te[c].map(smoothed).fillna(prior).values.astype(np.float32)

        for c in te_cols:
            F_tr[f"te_{c}"] = fit_te[c]
            F_te[f"te_{c}"] = test_te[c]

        F_tr["txt_sparse_p"] = oof_sp
        F_te["txt_sparse_p"] = p_sp_te

        # Categorical features
        cb_cat_features = list(CAT_FEATURES) + ["seller_id"]
        for c in CAT_FEATURES:
            F_tr[c] = cat_tr[c].astype(str).values
            F_te[c] = cat_te[c].astype(str).values

        F_tr["seller_id"] = cat_tr["seller_id"].astype(str).values
        F_te["seller_id"] = cat_te["seller_id"].astype(str).values

        # 4. CatBoost Bagging (3 seeds: 2026, 314159, 20261030)
        cb_seeds = [2026, 314159, 20261030]
        cb_bagged_path = OUT_DIR / "test_probs_cb_bagged_inductive_70k.npy"
        if cb_bagged_path.exists():
            print(f"4. Loading existing Bagged CatBoost probabilities from {cb_bagged_path}...", flush=True)
            p_cb_bagged = np.load(cb_bagged_path).astype(np.float32)
            print(f"   Loaded: mean={p_cb_bagged.mean():.5f}, std={p_cb_bagged.std():.5f}", flush=True)
        else:
            print("4. Training CatBoost 3-Seed Bagging (600 iterations each)...", flush=True)
            pool_tr = Pool(F_tr, y_tr, cat_features=cb_cat_features)
            pool_te = Pool(F_te, cat_features=cb_cat_features)

            p_cb_bagged = np.zeros(len(F_te), dtype=np.float32)
            cb_model_metas = []

            for s in cb_seeds:
                t_s = time.time()
                print(f"   -> Training CatBoost with seed {s}...", flush=True)
                cb = CatBoostClassifier(
                    iterations=600,
                    learning_rate=0.07,
                    depth=6,
                    l2_leaf_reg=6.0,
                    random_seed=s,
                    thread_count=4,
                    allow_writing_files=False,
                    verbose=0,
                )
                cb.fit(pool_tr)
                p_s = cb.predict_proba(pool_te)[:, 1].astype(np.float32)
                p_cb_bagged += p_s / len(cb_seeds)
                cb_model_metas.append({"seed": s, "elapsed": round(time.time() - t_s, 1), "mean_p": float(p_s.mean())})

            np.save(cb_bagged_path, p_cb_bagged)
            print(f"   Bagged CatBoost complete: mean={p_cb_bagged.mean():.5f}, std={p_cb_bagged.std():.5f}", flush=True)

        # 5. LightGBM Full 70k (n_estimators=227, fixed from tune median)
        lgb_path = OUT_DIR / "test_probs_lgb_variant_c_inductive_70k.npy"
        if lgb_path.exists():
            print(f"5. Loading existing LightGBM probabilities from {lgb_path}...", flush=True)
            p_lgb_te = np.load(lgb_path).astype(np.float32)
            print(f"   Loaded: mean={p_lgb_te.mean():.5f}, std={p_lgb_te.std():.5f}", flush=True)
        else:
            print("5. Training LightGBM Full 70k (n_estimators=227, num_leaves=63)...", flush=True)
            F_tr_lgb = F_tr.copy()
            F_te_lgb = F_te.copy()
            for c in CAT_FEATURES + ["seller_id"]:
                cats = pd.Index(F_tr_lgb[c].unique())
                F_tr_lgb[c] = pd.Categorical(F_tr_lgb[c], categories=cats)
                F_te_lgb[c] = pd.Categorical(F_te_lgb[c], categories=cats)

            lg_model = lgb.LGBMClassifier(
                n_estimators=227,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=2026,
                n_jobs=4,
                verbose=-1,
            )
            t_lgb = time.time()
            lg_model.fit(F_tr_lgb, y_tr)
            p_lgb_te = lg_model.predict_proba(F_te_lgb)[:, 1].astype(np.float32)
            np.save(lgb_path, p_lgb_te)
            print(f"   LightGBM complete in {time.time() - t_lgb:.1f}s: mean={p_lgb_te.mean():.5f}, std={p_lgb_te.std():.5f}", flush=True)

        # 6. Variant C Blend (0.75 CB_Bagged + 0.25 LGB)
        print("6. Forming Variant C Blend (0.75 CB_Bagged + 0.25 LGB)...", flush=True)
        p_vc_blend = 0.75 * p_cb_bagged + 0.25 * p_lgb_te
        np.save(OUT_DIR / "test_probs_vc_blend_inductive_70k.npy", p_vc_blend)

        # 7. Load B10 test probabilities from Claude branch
        print("7. Loading Claude's B10 test probabilities...", flush=True)
        res = subprocess.run(
            ["git", "show", "origin/claude/posterior-y-modelos-20261027:results/round141_claude_baseline/full70k/test_p.npy"],
            capture_output=True,
            check=True,
        )
        p_b10 = np.load(io.BytesIO(res.stdout)).astype(np.float32)
        assert len(p_b10) == 30000

        # 8. H-AVG-Plus Super-Ensemble: 0.50 B10 + 0.50 Variant C Blend
        print("8. Forming H-AVG-Plus (0.50 B10 + 0.50 Variant C Blend)...", flush=True)
        p_havg_plus = 0.50 * p_b10 + 0.50 * p_vc_blend
        np.save(OUT_DIR / "test_probs_havg_plus_inductive_70k.npy", p_havg_plus)

        # 9. Build test CSV with decision threshold 0.50 and canonical competition columns
        print("9. Generating candidate CSV with threshold 0.50...", flush=True)
        pred_used = (p_havg_plus >= 0.50).astype(int)
        df_sub = pd.DataFrame({
            "ID": np.arange(1, 30001),
            "condition": ["used" if p == 1 else "new" for p in pred_used],
        })
        df_sub.to_csv(CSV_OUT, index=False)
        sha_sub = sha256_bytes(CSV_OUT.read_bytes())
        print(f"   Saved to {CSV_OUT} (SHA256: {sha_sub})", flush=True)

        # 10. Submission Guard verification and historical overlap audit
        print("10. Running Submission Guard verification...", flush=True)
        reg_path = ROOT / "results" / "eval_20261030" / "scored_registry.json"
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        ever_probed = set()
        for e in reg.get("entries", []):
            ever_probed.update(e.get("flipped_ids", []))

        s48_raw = json.load(open(ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"))
        s48_dict = {int(k): v for k, v in s48_raw.items()}
        s48_arr = np.array([1 if s48_dict[i] == "used" else 0 for i in range(1, 30001)])

        ok, guard_msg = check_submission_file(CSV_OUT, reg, s48_dict, verbose=True)
        assert ok, f"Submission guard failed: {guard_msg}"
        print(f"   Submission guard passed successfully! ({guard_msg})", flush=True)

        # Disagreements vs S48
        diff_ids = np.where(pred_used != s48_arr)[0] + 1
        diff_probed = [int(i) for i in diff_ids if i in ever_probed]
        diff_never_probed = [int(i) for i in diff_ids if i not in ever_probed]
        print(f"Total disagreements vs S48: {len(diff_ids)}", flush=True)
        print(f"Disagreements already tested in 04-76: {len(diff_probed)}")
        print(f"Disagreements never tested: {len(diff_never_probed)}")

        elapsed_total = round(time.time() - t0, 1)

        metadata = {
            "model_name": "H-AVG-Plus",
            "architecture": "0.50 * B10 (Claude) + 0.50 * Variant_C_Blend (0.75 CB_Bagged_3seeds + 0.25 LGB)",
            "components": {
                "b10": {
                    "origin": "origin/claude/posterior-y-modelos-20261027:results/round141_claude_baseline/full70k/test_p.npy",
                    "weight": 0.50,
                },
                "cb_bagged": {
                    "seeds": cb_seeds,
                    "iterations_per_seed": 600,
                    "learning_rate": 0.07,
                    "depth": 6,
                    "l2_leaf_reg": 6.0,
                    "cat_features": cb_cat_features,
                    "weight_in_vc": 0.75,
                },
                "lgb_full70k": {
                    "n_estimators": 227,
                    "iterations_rationale": "Exact median of tune best_iterations (189, 227, 257) from cross-validation cuts",
                    "learning_rate": 0.04,
                    "num_leaves": 63,
                    "weight_in_vc": 0.25,
                },
            },
            "decision_threshold": 0.50,
            "total_flips_vs_s48": len(diff_ids),
            "historical_04_76_tested_flips": len(diff_probed),
            "never_probed_flips": len(diff_never_probed),
            "submission_guard_passed": ok,
            "guard_message": guard_msg,
            "sha256": sha_sub,
            "total_elapsed_seconds": elapsed_total,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(META_OUT, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nH-AVG-Plus Build Complete in {elapsed_total}s.")
        print(f"CSV SHA-256: {sha_sub}")
        print(f"Metadata saved to {META_OUT}")
        return metadata


if __name__ == "__main__":
    train_and_build_havg_plus()
