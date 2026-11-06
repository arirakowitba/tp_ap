"""Entrenamiento del modelo Full 70k CatBoost con Seller Categórico Nativo (Variant C).

Cumple la directiva de CX-0016:
'Construí predicción full70k C y sus probabilidades, señalando el uso de tune para best_iteration.'

Detalles metodológicos:
- Datos: 70.000 train etiquetados, 30.000 test.
- Target Encodings: GroupKFold(4) agrupado por vendedor en 70k train (Group TE estándar);
  en 30k test mapeado con smoothing alpha=5.0 desde 70k.
- Teacher de texto: txt_sparse_p (TF-IDF + LogisticRegression C=3.0, OOF en train, inferido en test).
- Feature categórica nativa: 'seller_id' incluida en cat_features junto con CAT_FEATURES.
- Iteraciones: 600 (determinado a partir del promedio/mediana de best_iteration en tune
  a través de las 3 semillas de validación: 2026: 899, 314159: 513, 20261030: 395; mediana=513, media=602).
- Learning rate: 0.07, Depth: 6, L2 leaf reg: 6.0, Seed: 2026.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordination_lock import ProjectLock
from src.intensive_20261030.shared_cache import CAT_FEATURES, build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROBS = OUT_DIR / "test_probs_variant_c_inductive_70k.npy"
OUT_META = OUT_DIR / "variant_c_full70k_metadata.json"


def train_and_infer_variant_c():
    lock = ProjectLock(owner_id="antigravity_builder_variant_c")
    print("Acquiring ProjectLock for variant_c full70k training...", flush=True)
    with lock:
        t0 = time.time()
        print("Loading shared feature cache...", flush=True)
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

        # 1. Compute txt_sparse_p
        print("Computing txt_sparse_p teacher features...", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack

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

        # 2. Target encodings
        print("Computing Group Target Encodings...", flush=True)
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

        # Categorical features for CatBoost: CAT_FEATURES + ['seller_id']
        cb_cat_features = list(CAT_FEATURES) + ["seller_id"]
        for c in CAT_FEATURES:
            F_tr[c] = cat_tr[c].astype(str).values
            F_te[c] = cat_te[c].astype(str).values

        F_tr["seller_id"] = cat_tr["seller_id"].astype(str).values
        F_te["seller_id"] = cat_te["seller_id"].astype(str).values

        # 3. Fit CatBoost Full 70k
        print(f"Training CatBoost Variant C full 70k (iterations=600, lr=0.07, depth=6)...", flush=True)
        pool_tr = Pool(F_tr, y_tr, cat_features=cb_cat_features)
        pool_te = Pool(F_te, cat_features=cb_cat_features)

        cb_model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.07,
            depth=6,
            l2_leaf_reg=6.0,
            random_seed=2026,
            thread_count=4,
            allow_writing_files=False,
            verbose=100,
        )
        cb_model.fit(pool_tr)
        p_te_variant_c = cb_model.predict_proba(pool_te)[:, 1].astype(np.float32)

        # Save output
        np.save(OUT_PROBS, p_te_variant_c)
        elapsed = round(time.time() - t0, 1)
        print(f"Variant C probabilities saved to {OUT_PROBS} in {elapsed}s", flush=True)

        meta = {
            "model": "CatBoostClassifier_variant_c_full70k",
            "iterations": 600,
            "iterations_rationale": "Informed by median (513) and mean (602) best_iteration across tune splits for seeds 2026 (899), 314159 (513), 20261030 (395)",
            "learning_rate": 0.07,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "cat_features": cb_cat_features,
            "elapsed_seconds": elapsed,
            "sha256": hashlib.sha256(p_te_variant_c.tobytes()).hexdigest(),
            "mean_p": float(p_te_variant_c.mean()),
            "std_p": float(p_te_variant_c.std()),
            "min_p": float(p_te_variant_c.min()),
            "max_p": float(p_te_variant_c.max()),
        }
        with open(OUT_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {OUT_META}: {meta}", flush=True)
        return meta


if __name__ == "__main__":
    import hashlib
    train_and_infer_variant_c()
