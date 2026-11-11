#!/usr/bin/env python3
"""T98 Line 1 — Original Text & Subword Specialist.

Implements CX-0164 & T98 protocol strictly on the frozen shared split and parsed caches:
- Loads `data/splits/t98_parsed_train.pkl` (70,000 rows), `data/splits/t98_parsed_test.pkl` (30,000 rows),
  and `data/splits/t98_dual_regime_outer_seller_group_splits.csv` (70,000 rows).
- For each outer fold k in 0..4:
  * fit_idx: role_fold_{k} == "fit"
  * seen_val_idx: role_fold_{k} == "seen_val"
  * unseen_val_idx: role_fold_{k} == "unseen_val"
  * inner_fold_{k}: 0..3 on fit_idx (StratifiedGroupKFold(4) by seller_id, 0 seller overlap)
- Models trained and evaluated strictly inside each fold (zero leakage outside fit_idx or inner_train_idx):
  * M0_podium_text: Exact Podium Text recipe (title + tags_str), Word TF-IDF (1,2) 40k min_df=3 +
    Char char_wb (3,5) 60k min_df=5, sublinear_tf=True, LogisticRegression(C=1.5, solver='saga', max_iter=800, random_state=42).
  * M1_word_char_enriched_saga: Enriched item-intrinsic text corpus (title, subtitle, warranty_str, tags_str,
    attr_text, listing_type_id, buying_mode) + explicit domain normalization tokens + Word TF-IDF (1,3) 80k min_df=2 +
    Char char_wb (3,5) 80k min_df=3, sublinear_tf=True, LogisticRegression(solver='saga', C tuned in [0.75, 1.25, 2.0] on inner folds ONLY).
  * M2_subword_char_nb_svm: Subword specialist combining char (2,6) + char_wb (3,6) (120k features total, min_df=3,
    sublinear_tf=True) with Naive Bayes log-count ratio weighting computed strictly on y_fit (and inner y_train).
  * M3_nested_text_blend: Convex/logit blend of M1 + M2 selected ONLY on 4-fold inner_oof inside fit_idx,
    frozen to freeze_line1_outer{k}.json BEFORE predicting seen_val_idx, unseen_val_idx, and test.
- Status: DEVELOPMENT_ONLY. Does NOT touch submissions/ or Kaggle.
"""

import os
import sys
import gc
import re
import json
import time
import hashlib
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "t98_line1_text_subword_20261111"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = 4
BOOT_SEED = 20261111
BOOT_REPS = 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


# Explicit domain normalization patterns (item-intrinsic text cues)
DOMAIN_PATTERNS = [
    ("DOM_SIN_USO", re.compile(r"\bsin\s+uso\b|\bnunca\s+usad[oa]\b|\bnunca\s+se\s+uso\b|\bsin\s+usar\b", re.I)),
    ("DOM_0KM", re.compile(r"\b0\s*km\b|\b0km\b|\bcero\s*km\b|\bsin\s+rodar\b", re.I)),
    ("DOM_COMO_NUEVO", re.compile(r"\bcomo\s+nuev[oa]s?\b|\bigual\s+a\s+nuev[oa]\b", re.I)),
    ("DOM_POCO_USO", re.compile(r"\bpoco\s+uso\b|\bmuy\s+poco\s+uso\b|\bpoquisimo\s+uso\b", re.I)),
    ("DOM_REACONDICIONADO", re.compile(r"\breacondicionad[oa]s?\b|\brefurbished\b|\brecertificad[oa]\b", re.I)),
    ("DOM_PARA_REPARAR", re.compile(r"\bpara\s+reparar\b|\ba\s+reparar\b|\bno\s+funciona\b|\bsin\s+funcionar\b|\bpara\s+arreglar\b", re.I)),
    ("DOM_REPUESTO", re.compile(r"\brepuestos?\b|\bpara\s+repuesto\b|\bdesarme\b|\bdespiece\b|\bdesguace\b", re.I)),
    ("DOM_USADO", re.compile(r"\busad[oa]s?\b|\busadit[oa]s?\b|\bsegunda\s+mano\b|\b2da\s+mano\b", re.I)),
    ("DOM_NUEVO", re.compile(r"\bnuev[oa]s?\b|\ba\s+estrenar\b", re.I)),
    ("DOM_SELLADO", re.compile(r"\bsellad[oa]s?\b|\bcerrad[oa]s?\s+de\s+fabrica\b|\bprecintad[oa]s?\b|\ben\s+blister\b|\bblister\s+cerrado\b", re.I)),
    ("DOM_EN_CAJA", re.compile(r"\ben\s+caja\b|\bcaja\s+cerrada\b|\bcaja\s+original\b", re.I)),
    ("DOM_IMPECABLE", re.compile(r"\bimpecables?\b|\bexcelente\s+estado\b|\bmuy\s+buen\s+estado\b|\bperfecto\s+estado\b|\bestado\s+impecable\b", re.I)),
    ("DOM_ANTIGUO", re.compile(r"\bantigu[oa]s?\b|\bantiguedad\b|\banos\s+\d{2}\b", re.I)),
    ("DOM_VINTAGE", re.compile(r"\bvintage\b|\bretro\b|\bcoleccion\b|\bde\s+epoca\b", re.I)),
    ("DOM_DETALLES", re.compile(r"\bcon\s+detalles\b|\bmarcas\s+de\s+uso\b|\brayones\b|\bdesgaste\b|\bdetalle\s+estetico\b", re.I)),
    ("DOM_GARANTIA_OFICIAL", re.compile(r"\bgarantia\s+oficial\b|\bgarantia\s+de\s+fabrica\b|\bfabrica\b", re.I)),
    ("DOM_SIN_GARANTIA", re.compile(r"\bsin\s+garantia\b|\bno\s+tiene\s+garantia\b", re.I)),
]


def strip_accents(s: str) -> str:
    if not s:
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ASCII", "ignore").decode("utf-8")


def build_podium_corpus(df: pd.DataFrame) -> list[str]:
    """Exact M0 Podium Text corpus: title + ' ' + tags_str."""
    titles = df["title"].fillna("").astype(str).tolist()
    tags = df["tags_str"].fillna("").astype(str).tolist()
    return [f"{t} {tg}" for t, tg in zip(titles, tags)]


def build_enriched_corpus(df: pd.DataFrame) -> list[str]:
    """Build enriched item-intrinsic corpus for M1 and M2.
    Uses ONLY item-intrinsic text fields:
    title, subtitle, warranty_str, tags_str, attr_text, listing_type_id, buying_mode.
    """
    titles = df["title"].fillna("").astype(str).tolist()
    subtitles = df["subtitle"].fillna("").astype(str).tolist()
    warranties = df["warranty_str"].fillna("").astype(str).tolist()
    tags = df["tags_str"].fillna("").astype(str).tolist()
    attrs = df["attr_text"].fillna("").astype(str).tolist()
    lt_ids = df["listing_type_id"].fillna("MISSING").astype(str).tolist()
    b_modes = df["buying_mode"].fillna("MISSING").astype(str).tolist()

    enriched = []
    for t, sub, w, tg, at, lt, bm in zip(titles, subtitles, warranties, tags, attrs, lt_ids, b_modes):
        raw_combined = f"{t} {sub} {w} {at}"
        norm_combined = strip_accents(raw_combined).lower()
        # Normalize common inline surface variants
        norm_combined = re.sub(r"\b0\s*km\b", "0km DOM_0KM", norm_combined)
        norm_combined = re.sub(r"\bsin\s+uso\b", "sin_uso DOM_SIN_USO", norm_combined)
        norm_combined = re.sub(r"\bcomo\s+nuev([oa]s?)\b", r"como_nuev\1 DOM_COMO_NUEVO", norm_combined)
        norm_combined = re.sub(r"\bpoco\s+uso\b", "poco_uso DOM_POCO_USO", norm_combined)
        norm_combined = re.sub(r"\bpara\s+reparar\b", "para_reparar DOM_PARA_REPARAR", norm_combined)
        norm_combined = re.sub(r"\ba\s+reparar\b", "a_reparar DOM_PARA_REPARAR", norm_combined)
        norm_combined = re.sub(r"\ben\s+caja\b", "en_caja DOM_EN_CAJA", norm_combined)

        dom_tokens = []
        for tok_name, pat in DOMAIN_PATTERNS:
            if pat.search(norm_combined):
                dom_tokens.append(f"__{tok_name}__")
        if not w.strip():
            dom_tokens.append("__WARR_EMPTY__")

        lt_tok = f"__LT_{lt.strip().lower()}__"
        bm_tok = f"__BM_{bm.strip().lower()}__"
        tg_tokens = " ".join([f"__TAG_{x}__" for x in tg.split() if x])
        dom_str = " ".join(dom_tokens)

        doc = (
            f"{t} {sub} __WARR__ {w} __ATTR__ {at} "
            f"__NORM__ {norm_combined} "
            f"{lt_tok} {bm_tok} {tg_tokens} {dom_str}"
        )
        enriched.append(doc)
    return enriched


def compute_nb_log_ratio(X: sparse.csr_matrix, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Compute Naive Bayes log-count ratio r = log((p / ||p||_1) / (q / ||q||_1)) strictly on y."""
    y_bin = np.asarray(y, dtype=np.int32)
    pos_mask = (y_bin == 1)
    neg_mask = (y_bin == 0)
    p = np.asarray(X[pos_mask].sum(axis=0), dtype=np.float64).ravel() + alpha
    q = np.asarray(X[neg_mask].sum(axis=0), dtype=np.float64).ravel() + alpha
    p /= p.sum()
    q /= q.sum()
    r = np.log(p / q).astype(np.float32)
    return r


def fit_predict_m0(
    c_tr: list[str],
    y_tr: np.ndarray,
    c_eval_list: list[list[str]],
) -> tuple[list[np.ndarray], dict]:
    """Exact Podium Text recipe (M0_podium_text)."""
    t0 = time.time()
    vw = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vc = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=60000,
        min_df=5,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tr = sparse.hstack([vw.fit_transform(c_tr), vc.fit_transform(c_tr)], format="csr", dtype=np.float32)
    lr = LogisticRegression(C=1.5, solver="saga", max_iter=800, random_state=42, n_jobs=N_JOBS)
    lr.fit(X_tr, y_tr)

    preds = []
    for c_ev in c_eval_list:
        X_ev = sparse.hstack([vw.transform(c_ev), vc.transform(c_ev)], format="csr", dtype=np.float32)
        preds.append(lr.predict_proba(X_ev)[:, 1].astype(np.float64))
        del X_ev

    diag = {
        "vocab_word": int(len(vw.vocabulary_)),
        "vocab_char": int(len(vc.vocabulary_)),
        "n_iter": int(np.max(lr.n_iter_)),
        "seconds": round(time.time() - t0, 2),
    }
    del X_tr, vw, vc, lr
    return preds, diag


def fit_m1_vectorizers_and_transform(
    c_tr: list[str],
    c_eval_list: list[list[str]],
) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix], dict]:
    """Fit M1 Word (1,3) 80k min_df=2 + Char char_wb (3,5) 80k min_df=3 strictly on c_tr."""
    vw = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=80000,
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vc = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tr = sparse.hstack([vw.fit_transform(c_tr), vc.fit_transform(c_tr)], format="csr", dtype=np.float32)
    X_evals = [
        sparse.hstack([vw.transform(c_ev), vc.transform(c_ev)], format="csr", dtype=np.float32)
        for c_ev in c_eval_list
    ]
    v_diag = {
        "vocab_word": int(len(vw.vocabulary_)),
        "vocab_char": int(len(vc.vocabulary_)),
    }
    del vw, vc
    return X_tr, X_evals, v_diag


def fit_m2_vectorizers_and_nb_transform(
    c_tr: list[str],
    y_tr: np.ndarray,
    c_eval_list: list[list[str]],
) -> tuple[sparse.csr_matrix, list[sparse.csr_matrix], dict]:
    """Fit M2 subword char (2,6) 60k + char_wb (3,6) 60k (120k total, min_df=3, sublinear_tf=True)
    and multiply elementwise by Naive Bayes log-count ratio r computed strictly on y_tr.
    """
    vc_raw = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        max_features=60000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vc_wb = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        max_features=60000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tr_raw = sparse.hstack([vc_raw.fit_transform(c_tr), vc_wb.fit_transform(c_tr)], format="csr", dtype=np.float32)
    r = compute_nb_log_ratio(X_tr_raw, y_tr, alpha=1.0)
    r_diag = sparse.diags(r, format="csr", dtype=np.float32)

    X_tr_nb = X_tr_raw.dot(r_diag).tocsr()
    del X_tr_raw

    X_evals_nb = []
    for c_ev in c_eval_list:
        X_ev_raw = sparse.hstack([vc_raw.transform(c_ev), vc_wb.transform(c_ev)], format="csr", dtype=np.float32)
        X_evals_nb.append(X_ev_raw.dot(r_diag).tocsr())
        del X_ev_raw

    v_diag = {
        "vocab_char": int(len(vc_raw.vocabulary_)),
        "vocab_char_wb": int(len(vc_wb.vocabulary_)),
        "total_subword_features": int(X_tr_nb.shape[1]),
        "nb_r_min": float(r.min()),
        "nb_r_max": float(r.max()),
        "nb_r_std": float(r.std()),
    }
    del vc_raw, vc_wb, r_diag
    return X_tr_nb, X_evals_nb, v_diag


def fit_blend_parameters(
    p_m1_oof: np.ndarray,
    p_m2_oof: np.ndarray,
    y_fit: np.ndarray,
) -> dict:
    """Fit convex probability blend and regularized logit blend strictly on 4-fold inner OOF."""
    # 1. Convex probability blend: w * p_m1 + (1 - w) * p_m2
    def prob_ll(w_arr):
        w = float(w_arr[0])
        p_b = np.clip(w * p_m1_oof + (1.0 - w) * p_m2_oof, 1e-7, 1.0 - 1e-7)
        return log_loss(y_fit, p_b)

    res_prob = minimize(prob_ll, x0=[0.55], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    w_prob = float(res_prob.x[0])
    p_prob_oof = np.clip(w_prob * p_m1_oof + (1.0 - w_prob) * p_m2_oof, 1e-7, 1.0 - 1e-7)
    ll_prob = float(log_loss(y_fit, p_prob_oof))
    acc_prob = float(accuracy_score(y_fit, p_prob_oof >= 0.5))

    # 2. Logit convex/stack blend: tune C_blend in [0.25, 0.5, 1.0, 2.0] on inner OOF
    Z_oof = np.column_stack([logit(p_m1_oof), logit(p_m2_oof)])
    best_logit = None
    for c_b in [0.25, 0.5, 1.0, 2.0]:
        stk = LogisticRegression(C=c_b, solver="lbfgs", max_iter=500, random_state=42)
        stk.fit(Z_oof, y_fit)
        # Enforce non-negative weights check
        if np.all(stk.coef_[0] >= 0):
            p_l = stk.predict_proba(Z_oof)[:, 1]
            ll_l = float(log_loss(y_fit, np.clip(p_l, 1e-7, 1.0 - 1e-7)))
            acc_l = float(accuracy_score(y_fit, p_l >= 0.5))
            if best_logit is None or ll_l < best_logit["ll"]:
                best_logit = {
                    "C_blend": float(c_b),
                    "coef": [float(x) for x in stk.coef_[0]],
                    "intercept": float(stk.intercept_[0]),
                    "ll": ll_l,
                    "acc": acc_l,
                }

    # Select blend family with lower inner-OOF logloss
    if best_logit is not None and best_logit["ll"] < ll_prob:
        selected_type = "logit_blend"
    else:
        selected_type = "prob_convex_blend"

    return {
        "selected_type": selected_type,
        "prob_convex": {
            "w_m1": w_prob,
            "w_m2": 1.0 - w_prob,
            "inner_oof_logloss": ll_prob,
            "inner_oof_acc": acc_prob,
        },
        "logit_blend": best_logit,
    }


def apply_frozen_blend(
    p_m1: np.ndarray,
    p_m2: np.ndarray,
    blend_cfg: dict,
) -> np.ndarray:
    """Apply frozen M3 blend parameters to M1 and M2 probability vectors."""
    if blend_cfg["selected_type"] == "logit_blend" and blend_cfg["logit_blend"] is not None:
        coef = blend_cfg["logit_blend"]["coef"]
        intercept = blend_cfg["logit_blend"]["intercept"]
        z = intercept + coef[0] * logit(p_m1) + coef[1] * logit(p_m2)
        return sigmoid(z)
    w_m1 = blend_cfg["prob_convex"]["w_m1"]
    w_m2 = blend_cfg["prob_convex"]["w_m2"]
    return np.clip(w_m1 * p_m1 + w_m2 * p_m2, 1e-7, 1.0 - 1e-7)


def boot_two_way(
    diff: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    n_boot: int = BOOT_REPS,
    seed: int = BOOT_SEED,
) -> np.ndarray:
    """Two-way crossed cluster bootstrap (seller_id x category_id) with Poisson(1) product weights."""
    rng = np.random.default_rng(seed)
    su, si = np.unique(sellers, return_inverse=True)
    cu, ci = np.unique(cats, return_inverse=True)
    n_s = len(su)
    n_c = len(cu)
    diff_f = np.asarray(diff, dtype=np.float64)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        w = rng.poisson(1.0, n_s)[si] * rng.poisson(1.0, n_c)[ci]
        out[b] = (w * diff_f).sum() / w.sum()
    return out


def compute_paired_comparison(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_ref: np.ndarray,
    folds: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    n_boot: int = BOOT_REPS,
    seed: int = BOOT_SEED,
) -> dict:
    """Compute comprehensive paired metrics and 2,000-replicate two-way cluster bootstrap."""
    n = len(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(np.int32)
    c_ref = ((p_ref >= 0.5) == y_true).astype(np.int32)
    diff_hits = c_cand - c_ref

    fixes = int(((c_cand == 1) & (c_ref == 0)).sum())
    breaks = int(((c_cand == 0) & (c_ref == 1)).sum())
    net_hits = int(diff_hits.sum())
    delta_acc = float(net_hits / n)

    p_c_clip = np.clip(p_cand, 1e-7, 1.0 - 1e-7)
    p_r_clip = np.clip(p_ref, 1e-7, 1.0 - 1e-7)
    ll_cand = float(log_loss(y_true, p_c_clip))
    ll_ref = float(log_loss(y_true, p_r_clip))
    delta_ll = float(ll_cand - ll_ref)

    # Per-fold deltas
    unique_folds = np.sort(np.unique(folds))
    fold_deltas_hits = []
    fold_deltas_acc = []
    fold_deltas_ll = []
    per_fold_detail = []
    for fk in unique_folds:
        m = (folds == fk)
        nf = int(m.sum())
        df_hits = int(diff_hits[m].sum())
        df_fixes = int(((c_cand[m] == 1) & (c_ref[m] == 0)).sum())
        df_breaks = int(((c_cand[m] == 0) & (c_ref[m] == 1)).sum())
        df_acc = float(df_hits / nf)
        df_ll = float(log_loss(y_true[m], p_c_clip[m]) - log_loss(y_true[m], p_r_clip[m]))
        fold_deltas_hits.append(df_hits)
        fold_deltas_acc.append(df_acc)
        fold_deltas_ll.append(df_ll)
        per_fold_detail.append({
            "outer_fold": int(fk),
            "n_rows": nf,
            "cand_hits": int(c_cand[m].sum()),
            "ref_hits": int(c_ref[m].sum()),
            "delta_hits": df_hits,
            "fixes": df_fixes,
            "breaks": df_breaks,
            "delta_acc": round(df_acc, 6),
            "delta_logloss": round(df_ll, 6),
        })

    fold_deltas_hits_arr = np.array(fold_deltas_hits)
    fold_deltas_acc_arr = np.array(fold_deltas_acc)

    # Two-way cluster bootstrap on accuracy diff (and scaled to total hits N)
    boot_acc_diff = boot_two_way(diff_hits.astype(np.float64), sellers, cats, n_boot=n_boot, seed=seed)
    boot_hits_diff = boot_acc_diff * n

    # Two-way cluster bootstrap on per-item logloss diff
    ll_item_cand = -(y_true * np.log(p_c_clip) + (1.0 - y_true) * np.log(1.0 - p_c_clip))
    ll_item_ref = -(y_true * np.log(p_r_clip) + (1.0 - y_true) * np.log(1.0 - p_r_clip))
    boot_ll_diff = boot_two_way(ll_item_cand - ll_item_ref, sellers, cats, n_boot=n_boot, seed=seed + 1)

    return {
        "n_rows": int(n),
        "cand_hits": int(c_cand.sum()),
        "ref_hits": int(c_ref.sum()),
        "cand_acc": round(float(c_cand.mean()), 6),
        "ref_acc": round(float(c_ref.mean()), 6),
        "delta_acc": round(delta_acc, 6),
        "delta_hits": net_hits,
        "fixes": fixes,
        "breaks": breaks,
        "cand_logloss": round(ll_cand, 6),
        "ref_logloss": round(ll_ref, 6),
        "delta_logloss": round(delta_ll, 6),
        "positive_folds_count": int((fold_deltas_hits_arr > 0).sum()),
        "zero_folds_count": int((fold_deltas_hits_arr == 0).sum()),
        "negative_folds_count": int((fold_deltas_hits_arr < 0).sum()),
        "worst_fold_delta_hits": int(fold_deltas_hits_arr.min()),
        "worst_fold_delta_acc": round(float(fold_deltas_acc_arr.min()), 6),
        "boot_delta_hits_median": round(float(np.median(boot_hits_diff)), 3),
        "boot_delta_hits_q05": round(float(np.quantile(boot_hits_diff, 0.05)), 3),
        "boot_delta_hits_q95": round(float(np.quantile(boot_hits_diff, 0.95)), 3),
        "boot_delta_acc_median": round(float(np.median(boot_acc_diff)), 6),
        "boot_delta_acc_q05": round(float(np.quantile(boot_acc_diff, 0.05)), 6),
        "boot_delta_acc_q95": round(float(np.quantile(boot_acc_diff, 0.95)), 6),
        "boot_prob_gt0": round(float((boot_hits_diff > 0).mean()), 4),
        "boot_delta_logloss_median": round(float(np.median(boot_ll_diff)), 6),
        "boot_delta_logloss_q05": round(float(np.quantile(boot_ll_diff, 0.05)), 6),
        "boot_delta_logloss_q95": round(float(np.quantile(boot_ll_diff, 0.95)), 6),
        "per_fold": per_fold_detail,
    }


def main() -> None:
    t_start = time.time()
    log("=== T98 LINE 1: ORIGINAL TEXT & SUBWORD SPECIALIST ===")

    # 1. Load frozen shared split and parsed caches ONLY
    train_pkl_path = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
    test_pkl_path = ROOT / "data" / "splits" / "t98_parsed_test.pkl"
    splits_csv_path = ROOT / "data" / "splits" / "t98_dual_regime_outer_seller_group_splits.csv"
    t97_paired_path = ROOT / "results" / "seller_group_nested_podium_20261111" / "paired_outer_val_predictions.csv"

    df_train = pd.read_pickle(train_pkl_path)
    df_test = pd.read_pickle(test_pkl_path)
    df_splits = pd.read_csv(splits_csv_path)
    df_t97 = pd.read_csv(t97_paired_path)

    assert len(df_train) == 70000, f"Expected 70,000 train rows, got {len(df_train)}"
    assert len(df_test) == 30000, f"Expected 30,000 test rows, got {len(df_test)}"
    assert len(df_splits) == 70000, f"Expected 70,000 split rows, got {len(df_splits)}"
    assert np.array_equal(df_train["row"].values, np.arange(70000))
    assert np.array_equal(df_splits["row"].values, np.arange(70000))
    assert np.array_equal(df_train["id"].values, df_splits["id"].values)

    y_all = df_train["y"].values.astype(np.int32)
    sellers_all = df_train["seller_id"].astype(str).values
    cats_all = df_train["category_id"].astype(str).values
    ids_all = df_train["id"].astype(str).values

    # Build corpora once (pure string formatting per item, zero cross-item information)
    log("Building M0 (Podium title+tags) and Enriched (M1/M2 item-intrinsic + domain tokens) corpora...")
    corpus_m0_train = build_podium_corpus(df_train)
    corpus_m0_test = build_podium_corpus(df_test)

    corpus_enr_train = build_enriched_corpus(df_train)
    corpus_enr_test = build_enriched_corpus(df_test)

    m1_c_grid = [0.75, 1.25, 2.0]
    m2_c_grid = [0.5, 1.0, 1.5]

    # Global accumulators across the 5 outer folds
    oof_unseen = {
        m: np.zeros(70000, dtype=np.float64)
        for m in ["m0", "m1", "m2", "m3"]
    }
    unseen_fold_of = np.full(70000, -1, dtype=np.int32)
    seen_records = []
    inner_oof_all_records = []
    test_preds_per_fold = {
        m: np.zeros((5, 30000), dtype=np.float64)
        for m in ["m0", "m1", "m2", "m3"]
    }
    per_fold_metrics_rows = []
    frozen_policies = []

    for k in range(5):
        f_start = time.time()
        role_col = f"role_fold_{k}"
        inner_col = f"inner_fold_{k}"

        roles = df_splits[role_col].values
        inner_folds = df_splits[inner_col].values

        fit_idx = np.where(roles == "fit")[0]
        seen_val_idx = np.where(roles == "seen_val")[0]
        unseen_val_idx = np.where(roles == "unseen_val")[0]

        # Strict hygiene assertions
        assert len(fit_idx) + len(seen_val_idx) + len(unseen_val_idx) == 70000
        assert len(unseen_val_idx) == 14000
        assert np.all(df_splits["outer_fold"].values[unseen_val_idx] == k)
        assert len(set(sellers_all[fit_idx]) & set(sellers_all[unseen_val_idx])) == 0, "Seller leakage into unseen_val!"
        assert len(set(sellers_all[seen_val_idx]) & set(sellers_all[unseen_val_idx])) == 0, "Seller leakage between seen_val and unseen_val!"

        inner_f_fit = inner_folds[fit_idx].astype(np.int32)
        assert set( np.unique(inner_f_fit) ) == {0, 1, 2, 3}, f"Unexpected inner folds: {np.unique(inner_f_fit)}"

        log(
            f"=== Outer Fold {k}/4: fit={len(fit_idx)} | seen_val={len(seen_val_idx)} | unseen_val={len(unseen_val_idx)} ==="
        )

        fold_npz_path = OUT_DIR / f"fold_{k}_predictions.npz"
        freeze_json_path = OUT_DIR / f"freeze_line1_outer{k}.json"
        fold_cache_meta_path = CACHE_DIR / f"fold_{k}_meta.json"

        if fold_npz_path.exists() and freeze_json_path.exists() and fold_cache_meta_path.exists():
            log(f"  [Cache Hit] Loading fold {k} predictions from {fold_npz_path.name}")
            z = np.load(fold_npz_path)
            p_m0_fit_oof = z["p_m0_fit_oof"]
            p_m1_fit_oof = z["p_m1_fit_oof"]
            p_m2_fit_oof = z["p_m2_fit_oof"]
            p_m3_fit_oof = z["p_m3_fit_oof"]

            p_m0_seen_val = z["p_m0_seen_val"]
            p_m1_seen_val = z["p_m1_seen_val"]
            p_m2_seen_val = z["p_m2_seen_val"]
            p_m3_seen_val = z["p_m3_seen_val"]

            p_m0_unseen_val = z["p_m0_unseen_val"]
            p_m1_unseen_val = z["p_m1_unseen_val"]
            p_m2_unseen_val = z["p_m2_unseen_val"]
            p_m3_unseen_val = z["p_m3_unseen_val"]

            p_m0_test = z["p_m0_test"]
            p_m1_test = z["p_m1_test"]
            p_m2_test = z["p_m2_test"]
            p_m3_test = z["p_m3_test"]

            frozen_info = json.loads(freeze_json_path.read_text(encoding="utf-8"))
        else:
            y_fit = y_all[fit_idx]
            p_m0_fit_oof = np.zeros(len(fit_idx), dtype=np.float64)
            p_m1_grid_oof = {c: np.zeros(len(fit_idx), dtype=np.float64) for c in m1_c_grid}
            p_m2_grid_oof = {c: np.zeros(len(fit_idx), dtype=np.float64) for c in m2_c_grid}

            # ---- Step 1: 4-fold inner seller-group OOF inside fit_idx ONLY ----
            for j in range(4):
                t_in = time.time()
                in_va_local = np.where(inner_f_fit == j)[0]
                in_tr_local = np.where(inner_f_fit != j)[0]

                in_tr_global = fit_idx[in_tr_local]
                in_va_global = fit_idx[in_va_local]

                # Assert zero seller overlap in inner split
                assert len(set(sellers_all[in_tr_global]) & set(sellers_all[in_va_global])) == 0

                y_in_tr = y_all[in_tr_global]

                # 1.1 M0 inner fold
                c_m0_tr = [corpus_m0_train[i] for i in in_tr_global]
                c_m0_va = [corpus_m0_train[i] for i in in_va_global]
                preds_m0_in, _ = fit_predict_m0(c_m0_tr, y_in_tr, [c_m0_va])
                p_m0_fit_oof[in_va_local] = preds_m0_in[0]

                # 1.2 M1 inner fold (tune C in [0.75, 1.25, 2.0] with warm_start)
                c_enr_tr = [corpus_enr_train[i] for i in in_tr_global]
                c_enr_va = [corpus_enr_train[i] for i in in_va_global]
                X_m1_tr, [X_m1_va], _ = fit_m1_vectorizers_and_transform(c_enr_tr, [c_enr_va])
                lr_m1 = LogisticRegression(
                    C=m1_c_grid[0],
                    solver="saga",
                    max_iter=800,
                    random_state=42,
                    warm_start=True,
                    n_jobs=N_JOBS,
                )
                for c_val in m1_c_grid:
                    lr_m1.set_params(C=c_val)
                    lr_m1.fit(X_m1_tr, y_in_tr)
                    p_m1_grid_oof[c_val][in_va_local] = lr_m1.predict_proba(X_m1_va)[:, 1]
                del X_m1_tr, X_m1_va, lr_m1

                # 1.3 M2 inner fold (Subword char(2,6)+char_wb(3,6) + NB log-count ratio on y_in_tr)
                X_m2_tr, [X_m2_va], _ = fit_m2_vectorizers_and_nb_transform(c_enr_tr, y_in_tr, [c_enr_va])
                lr_m2 = LogisticRegression(
                    C=m2_c_grid[0],
                    solver="saga",
                    max_iter=800,
                    random_state=42,
                    warm_start=True,
                    n_jobs=N_JOBS,
                )
                for c_val in m2_c_grid:
                    lr_m2.set_params(C=c_val)
                    lr_m2.fit(X_m2_tr, y_in_tr)
                    p_m2_grid_oof[c_val][in_va_local] = lr_m2.predict_proba(X_m2_va)[:, 1]
                del X_m2_tr, X_m2_va, lr_m2
                gc.collect()

                log(
                    f"  [Outer {k} | Inner {j}/3] Done in {time.time() - t_in:.1f}s | "
                    f"M0 acc={accuracy_score(y_fit[in_va_local], p_m0_fit_oof[in_va_local] >= 0.5):.5f}"
                )

            # ---- Step 2: Select hyperparameters and blend weights strictly on 4-fold inner_oof ----
            m1_tuning_summary = {}
            best_c_m1 = m1_c_grid[0]
            best_ll_m1 = float("inf")
            for c_val in m1_c_grid:
                pr = np.clip(p_m1_grid_oof[c_val], 1e-7, 1.0 - 1e-7)
                ll_v = float(log_loss(y_fit, pr))
                acc_v = float(accuracy_score(y_fit, pr >= 0.5))
                m1_tuning_summary[str(c_val)] = {"logloss": round(ll_v, 6), "acc": round(acc_v, 6)}
                if ll_v < best_ll_m1:
                    best_ll_m1 = ll_v
                    best_c_m1 = c_val

            m2_tuning_summary = {}
            best_c_m2 = m2_c_grid[0]
            best_ll_m2 = float("inf")
            for c_val in m2_c_grid:
                pr = np.clip(p_m2_grid_oof[c_val], 1e-7, 1.0 - 1e-7)
                ll_v = float(log_loss(y_fit, pr))
                acc_v = float(accuracy_score(y_fit, pr >= 0.5))
                m2_tuning_summary[str(c_val)] = {"logloss": round(ll_v, 6), "acc": round(acc_v, 6)}
                if ll_v < best_ll_m2:
                    best_ll_m2 = ll_v
                    best_c_m2 = c_val

            p_m1_fit_oof = p_m1_grid_oof[best_c_m1]
            p_m2_fit_oof = p_m2_grid_oof[best_c_m2]

            # Fit blend parameters on 4-fold inner OOF for freezing before outer predictions
            blend_cfg = fit_blend_parameters(p_m1_fit_oof, p_m2_fit_oof, y_fit)

            # Compute strictly out-of-fold M3 predictions on fit_idx by fitting blend on the other 3 inner folds
            p_m3_fit_oof = np.zeros(len(fit_idx), dtype=np.float64)
            for j in range(4):
                in_va_local = np.where(inner_f_fit == j)[0]
                in_tr_local = np.where(inner_f_fit != j)[0]
                sub_blend_cfg = fit_blend_parameters(
                    p_m1_fit_oof[in_tr_local],
                    p_m2_fit_oof[in_tr_local],
                    y_fit[in_tr_local],
                )
                p_m3_fit_oof[in_va_local] = apply_frozen_blend(
                    p_m1_fit_oof[in_va_local],
                    p_m2_fit_oof[in_va_local],
                    sub_blend_cfg,
                )

            frozen_info = {
                "outer_fold": int(k),
                "status": "DEVELOPMENT_ONLY",
                "n_fit": int(len(fit_idx)),
                "fit_rows_sha256": sha256_array(np.sort(fit_idx)),
                "m1_selected_C": float(best_c_m1),
                "m1_inner_oof_grid": m1_tuning_summary,
                "m2_selected_C": float(best_c_m2),
                "m2_inner_oof_grid": m2_tuning_summary,
                "m3_blend_config": blend_cfg,
                "inner_oof_metrics": {
                    "M0_podium_text": {
                        "acc": round(float(accuracy_score(y_fit, p_m0_fit_oof >= 0.5)), 6),
                        "logloss": round(float(log_loss(y_fit, np.clip(p_m0_fit_oof, 1e-7, 1.0 - 1e-7))), 6),
                    },
                    "M1_word_char_enriched_saga": {
                        "acc": round(float(accuracy_score(y_fit, p_m1_fit_oof >= 0.5)), 6),
                        "logloss": round(float(log_loss(y_fit, np.clip(p_m1_fit_oof, 1e-7, 1.0 - 1e-7))), 6),
                    },
                    "M2_subword_char_nb_svm": {
                        "acc": round(float(accuracy_score(y_fit, p_m2_fit_oof >= 0.5)), 6),
                        "logloss": round(float(log_loss(y_fit, np.clip(p_m2_fit_oof, 1e-7, 1.0 - 1e-7))), 6),
                    },
                    "M3_nested_text_blend": {
                        "acc": round(float(accuracy_score(y_fit, p_m3_fit_oof >= 0.5)), 6),
                        "logloss": round(float(log_loss(y_fit, np.clip(p_m3_fit_oof, 1e-7, 1.0 - 1e-7))), 6),
                    },
                },
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # Freeze to disk BEFORE predicting seen_val_idx, unseen_val_idx, and test!
            freeze_json_path.write_text(json.dumps(frozen_info, indent=2), encoding="utf-8")
            log(
                f"  [Outer {k}] Frozen policy written to {freeze_json_path.name}: "
                f"C_m1={best_c_m1}, C_m2={best_c_m2}, blend={blend_cfg['selected_type']} "
                f"(w_m1={blend_cfg['prob_convex']['w_m1']:.3f}) | "
                f"Inner OOF Acc: M0={frozen_info['inner_oof_metrics']['M0_podium_text']['acc']:.5f}, "
                f"M1={frozen_info['inner_oof_metrics']['M1_word_char_enriched_saga']['acc']:.5f}, "
                f"M2={frozen_info['inner_oof_metrics']['M2_subword_char_nb_svm']['acc']:.5f}, "
                f"M3={frozen_info['inner_oof_metrics']['M3_nested_text_blend']['acc']:.5f}"
            )

            # ---- Step 3: Outer Refit on fit_idx ONLY, predict seen_val, unseen_val, test ----
            t_refit = time.time()
            c_m0_fit = [corpus_m0_train[i] for i in fit_idx]
            c_m0_seen = [corpus_m0_train[i] for i in seen_val_idx]
            c_m0_unseen = [corpus_m0_train[i] for i in unseen_val_idx]
            (p_m0_seen_val, p_m0_unseen_val, p_m0_test), m0_diag = fit_predict_m0(
                c_m0_fit, y_fit, [c_m0_seen, c_m0_unseen, corpus_m0_test]
            )

            c_enr_fit = [corpus_enr_train[i] for i in fit_idx]
            c_enr_seen = [corpus_enr_train[i] for i in seen_val_idx]
            c_enr_unseen = [corpus_enr_train[i] for i in unseen_val_idx]

            # Refit M1 on fit_idx with frozen best_c_m1
            X_m1_fit, [X_m1_seen, X_m1_unseen, X_m1_test], m1_diag = fit_m1_vectorizers_and_transform(
                c_enr_fit, [c_enr_seen, c_enr_unseen, corpus_enr_test]
            )
            lr_m1_full = LogisticRegression(
                C=best_c_m1, solver="saga", max_iter=800, random_state=42, n_jobs=N_JOBS
            )
            lr_m1_full.fit(X_m1_fit, y_fit)
            p_m1_seen_val = lr_m1_full.predict_proba(X_m1_seen)[:, 1].astype(np.float64)
            p_m1_unseen_val = lr_m1_full.predict_proba(X_m1_unseen)[:, 1].astype(np.float64)
            p_m1_test = lr_m1_full.predict_proba(X_m1_test)[:, 1].astype(np.float64)
            del X_m1_fit, X_m1_seen, X_m1_unseen, X_m1_test, lr_m1_full
            gc.collect()

            # Refit M2 on fit_idx with frozen best_c_m2
            X_m2_fit, [X_m2_seen, X_m2_unseen, X_m2_test], m2_diag = fit_m2_vectorizers_and_nb_transform(
                c_enr_fit, y_fit, [c_enr_seen, c_enr_unseen, corpus_enr_test]
            )
            lr_m2_full = LogisticRegression(
                C=best_c_m2, solver="saga", max_iter=800, random_state=42, n_jobs=N_JOBS
            )
            lr_m2_full.fit(X_m2_fit, y_fit)
            p_m2_seen_val = lr_m2_full.predict_proba(X_m2_seen)[:, 1].astype(np.float64)
            p_m2_unseen_val = lr_m2_full.predict_proba(X_m2_unseen)[:, 1].astype(np.float64)
            p_m2_test = lr_m2_full.predict_proba(X_m2_test)[:, 1].astype(np.float64)
            del X_m2_fit, X_m2_seen, X_m2_unseen, X_m2_test, lr_m2_full
            gc.collect()

            # Apply frozen M3 blend to seen_val, unseen_val, and test
            p_m3_seen_val = apply_frozen_blend(p_m1_seen_val, p_m2_seen_val, blend_cfg)
            p_m3_unseen_val = apply_frozen_blend(p_m1_unseen_val, p_m2_unseen_val, blend_cfg)
            p_m3_test = apply_frozen_blend(p_m1_test, p_m2_test, blend_cfg)

            # Save fold_{k}_predictions.npz for downstream lines (Line 2 CatBoost, Line 3 LightGBM, Line 5 Ensembles)
            np.savez_compressed(
                fold_npz_path,
                fit_idx=fit_idx.astype(np.int32),
                p_m0_fit_oof=p_m0_fit_oof.astype(np.float64),
                p_m1_fit_oof=p_m1_fit_oof.astype(np.float64),
                p_m2_fit_oof=p_m2_fit_oof.astype(np.float64),
                p_m3_fit_oof=p_m3_fit_oof.astype(np.float64),
                seen_val_idx=seen_val_idx.astype(np.int32),
                p_m0_seen_val=p_m0_seen_val.astype(np.float64),
                p_m1_seen_val=p_m1_seen_val.astype(np.float64),
                p_m2_seen_val=p_m2_seen_val.astype(np.float64),
                p_m3_seen_val=p_m3_seen_val.astype(np.float64),
                unseen_val_idx=unseen_val_idx.astype(np.int32),
                p_m0_unseen_val=p_m0_unseen_val.astype(np.float64),
                p_m1_unseen_val=p_m1_unseen_val.astype(np.float64),
                p_m2_unseen_val=p_m2_unseen_val.astype(np.float64),
                p_m3_unseen_val=p_m3_unseen_val.astype(np.float64),
                p_m0_test=p_m0_test.astype(np.float64),
                p_m1_test=p_m1_test.astype(np.float64),
                p_m2_test=p_m2_test.astype(np.float64),
                p_m3_test=p_m3_test.astype(np.float64),
            )
            fold_cache_meta_path.write_text(
                json.dumps({"m0_diag": m0_diag, "m1_diag": m1_diag, "m2_diag": m2_diag}, indent=2),
                encoding="utf-8",
            )
            log(f"  [Outer {k}] Refit + predictions saved to {fold_npz_path.name} in {time.time() - t_refit:.1f}s")

        frozen_policies.append(frozen_info)

        # Populate global unseen_val arrays
        oof_unseen["m0"][unseen_val_idx] = p_m0_unseen_val
        oof_unseen["m1"][unseen_val_idx] = p_m1_unseen_val
        oof_unseen["m2"][unseen_val_idx] = p_m2_unseen_val
        oof_unseen["m3"][unseen_val_idx] = p_m3_unseen_val
        unseen_fold_of[unseen_val_idx] = k

        # Populate test fold arrays
        test_preds_per_fold["m0"][k] = p_m0_test
        test_preds_per_fold["m1"][k] = p_m1_test
        test_preds_per_fold["m2"][k] = p_m2_test
        test_preds_per_fold["m3"][k] = p_m3_test

        # Populate seen_val records for fold k
        df_seen_k = pd.DataFrame({
            "outer_fold": k,
            "row": seen_val_idx,
            "id": ids_all[seen_val_idx],
            "seller_id": sellers_all[seen_val_idx],
            "category_id": cats_all[seen_val_idx],
            "y_true": y_all[seen_val_idx],
            "p_m0_text": p_m0_seen_val,
            "p_m1_text": p_m1_seen_val,
            "p_m2_subword": p_m2_seen_val,
            "p_m3_blend": p_m3_seen_val,
        })
        seen_records.append(df_seen_k)

        # Populate inner_oof fit records for fold k
        df_inner_k = pd.DataFrame({
            "outer_fold": k,
            "row": fit_idx,
            "id": ids_all[fit_idx],
            "inner_fold": inner_f_fit,
            "y_true": y_all[fit_idx],
            "p_m0_fit_oof": p_m0_fit_oof,
            "p_m1_fit_oof": p_m1_fit_oof,
            "p_m2_fit_oof": p_m2_fit_oof,
            "p_m3_fit_oof": p_m3_fit_oof,
        })
        inner_oof_all_records.append(df_inner_k)

        # Per-fold evaluation metrics (unseen_val and seen_val)
        y_un = y_all[unseen_val_idx]
        y_sn = y_all[seen_val_idx]
        p_t97_un = df_t97["p_text"].values[unseen_val_idx]
        p_t97_sn = df_t97["p_text"].values[seen_val_idx]

        fold_row = {
            "outer_fold": k,
            "n_fit": int(len(fit_idx)),
            "n_seen_val": int(len(seen_val_idx)),
            "n_unseen_val": int(len(unseen_val_idx)),
            "m1_best_C": float(frozen_info["m1_selected_C"]),
            "m2_best_C": float(frozen_info["m2_selected_C"]),
            "m3_blend_type": frozen_info["m3_blend_config"]["selected_type"],
            "m3_w_m1": round(float(frozen_info["m3_blend_config"]["prob_convex"]["w_m1"]), 4),
        }

        for regime, y_reg, preds_reg in [
            ("unseen", y_un, {"m0": p_m0_unseen_val, "m1": p_m1_unseen_val, "m2": p_m2_unseen_val, "m3": p_m3_unseen_val, "t97": p_t97_un}),
            ("seen", y_sn, {"m0": p_m0_seen_val, "m1": p_m1_seen_val, "m2": p_m2_seen_val, "m3": p_m3_seen_val, "t97": p_t97_sn}),
        ]:
            c_m0 = (preds_reg["m0"] >= 0.5) == y_reg
            c_t97 = (preds_reg["t97"] >= 0.5) == y_reg
            for m_key, m_pred in preds_reg.items():
                pr = np.clip(m_pred, 1e-7, 1.0 - 1e-7)
                cm = (pr >= 0.5) == y_reg
                fold_row[f"{regime}_hits_{m_key}"] = int(cm.sum())
                fold_row[f"{regime}_acc_{m_key}"] = round(float(cm.mean()), 6)
                fold_row[f"{regime}_ll_{m_key}"] = round(float(log_loss(y_reg, pr)), 6)
                if m_key in ("m1", "m2", "m3"):
                    fold_row[f"{regime}_delta_hits_{m_key}_vs_m0"] = int(cm.sum()) - int(c_m0.sum())
                    fold_row[f"{regime}_fixes_{m_key}_vs_m0"] = int((cm & ~c_m0).sum())
                    fold_row[f"{regime}_breaks_{m_key}_vs_m0"] = int((~cm & c_m0).sum())
                    fold_row[f"{regime}_delta_hits_{m_key}_vs_t97"] = int(cm.sum()) - int(c_t97.sum())
                    fold_row[f"{regime}_fixes_{m_key}_vs_t97"] = int((cm & ~c_t97).sum())
                    fold_row[f"{regime}_breaks_{m_key}_vs_t97"] = int((~cm & c_t97).sum())

        per_fold_metrics_rows.append(fold_row)
        pd.DataFrame(per_fold_metrics_rows).to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

        log(
            f"  [Outer {k} Summary ({time.time() - f_start:.1f}s)] "
            f"UNSEEN Acc: M0={fold_row['unseen_acc_m0']:.5f}, M1={fold_row['unseen_acc_m1']:.5f} ({fold_row['unseen_delta_hits_m1_vs_m0']:+d}), "
            f"M2={fold_row['unseen_acc_m2']:.5f} ({fold_row['unseen_delta_hits_m2_vs_m0']:+d}), "
            f"M3={fold_row['unseen_acc_m3']:.5f} ({fold_row['unseen_delta_hits_m3_vs_m0']:+d} vs M0, {fold_row['unseen_delta_hits_m3_vs_t97']:+d} vs T97) | "
            f"SEEN Acc: M0={fold_row['seen_acc_m0']:.5f}, M3={fold_row['seen_acc_m3']:.5f} ({fold_row['seen_delta_hits_m3_vs_m0']:+d})"
        )

    assert np.all(unseen_fold_of >= 0), "Not all 70,000 unseen_val rows were populated!"

    # ---- Step 4: Save OOF and Test Deliverable CSVs ----
    log("Saving OOF and Test prediction deliverables...")

    # 4.1 inner_oof_fit_predictions.csv
    df_inner_all = pd.concat(inner_oof_all_records, ignore_index=True)
    inner_oof_csv_path = OUT_DIR / "inner_oof_fit_predictions.csv"
    df_inner_all.to_csv(inner_oof_csv_path, index=False)

    # 4.2 oof_unseen_70k.csv (aligned with row 0..69999)
    df_oof_unseen = pd.DataFrame({
        "row": np.arange(70000, dtype=np.int32),
        "id": ids_all,
        "seller_id": sellers_all,
        "category_id": cats_all,
        "outer_fold": unseen_fold_of,
        "y_true": y_all,
        "p_t97_text": df_t97["p_text"].values,
        "p_m0_text": oof_unseen["m0"],
        "p_m1_text": oof_unseen["m1"],
        "p_m2_subword": oof_unseen["m2"],
        "p_m3_blend": oof_unseen["m3"],
    })
    oof_unseen_path = OUT_DIR / "oof_unseen_70k.csv"
    df_oof_unseen.to_csv(oof_unseen_path, index=False)

    # 4.3 oof_seen_all_folds.csv
    df_oof_seen = pd.concat(seen_records, ignore_index=True)
    df_oof_seen["p_t97_text"] = df_t97["p_text"].values[df_oof_seen["row"].values]
    oof_seen_path = OUT_DIR / "oof_seen_all_folds.csv"
    df_oof_seen.to_csv(oof_seen_path, index=False)

    # 4.4 test_predictions_30k.csv (columns: ID, p_m0_text, p_m1_text, p_m2_subword, p_m3_blend — NO inferred class labels)
    s4_path = ROOT / "submissions" / "SUBMISSION_4_TODAY_RECOMMENDED.csv"
    df_s4 = pd.read_csv(s4_path)
    assert len(df_s4) == 30000

    p_m0_test_avg = test_preds_per_fold["m0"].mean(axis=0)
    p_m1_test_avg = test_preds_per_fold["m1"].mean(axis=0)
    p_m2_test_avg = test_preds_per_fold["m2"].mean(axis=0)
    p_m3_test_avg = test_preds_per_fold["m3"].mean(axis=0)

    df_test_preds = pd.DataFrame({
        "ID": df_s4["ID"].values,
        "p_m0_text": p_m0_test_avg,
        "p_m1_text": p_m1_test_avg,
        "p_m2_subword": p_m2_test_avg,
        "p_m3_blend": p_m3_test_avg,
    })
    test_preds_path = OUT_DIR / "test_predictions_30k.csv"
    df_test_preds.to_csv(test_preds_path, index=False)

    # 4.5 test_disagreements_vs_s4_probs_only.csv (only IDs and model probabilities where (p_m3_blend >= 0.5) disagrees with S4)
    s4_used_indicator = (df_s4["condition"].astype(str).str.lower() == "used").values
    m3_used_indicator = (p_m3_test_avg >= 0.5)
    disagree_mask = (m3_used_indicator != s4_used_indicator)
    df_disagree = df_test_preds.loc[disagree_mask, ["ID", "p_m0_text", "p_m1_text", "p_m2_subword", "p_m3_blend"]].copy()
    test_disagree_path = OUT_DIR / "test_disagreements_vs_s4_probs_only.csv"
    df_disagree.to_csv(test_disagree_path, index=False)

    # ---- Step 5: Paired Evaluations & 2,000-Replicate Two-Way Cluster Bootstrap ----
    log("Running 2,000-replicate two-way cluster bootstrap (seller_id x category_id, seed=20261111)...")

    paired_evaluations = {
        "unseen_val": {},
        "seen_val": {},
    }

    # Unseen_val (70,000 rows)
    for cand_key, cand_label in [
        ("m0", "M0_podium_text"),
        ("m1", "M1_word_char_enriched_saga"),
        ("m2", "M2_subword_char_nb_svm"),
        ("m3", "M3_nested_text_blend"),
    ]:
        if cand_key != "m0":
            paired_evaluations["unseen_val"][f"{cand_label}_vs_M0_podium_text"] = compute_paired_comparison(
                y_true=y_all,
                p_cand=oof_unseen[cand_key],
                p_ref=oof_unseen["m0"],
                folds=unseen_fold_of,
                sellers=sellers_all,
                cats=cats_all,
                n_boot=BOOT_REPS,
                seed=BOOT_SEED,
            )
        paired_evaluations["unseen_val"][f"{cand_label}_vs_T97_p_text"] = compute_paired_comparison(
            y_true=y_all,
            p_cand=oof_unseen[cand_key],
            p_ref=df_t97["p_text"].values,
            folds=unseen_fold_of,
            sellers=sellers_all,
            cats=cats_all,
            n_boot=BOOT_REPS,
            seed=BOOT_SEED,
        )

    # Seen_val (all seen_val rows across 5 folds)
    y_seen_all = df_oof_seen["y_true"].values.astype(np.int32)
    sellers_seen_all = df_oof_seen["seller_id"].astype(str).values
    cats_seen_all = df_oof_seen["category_id"].astype(str).values
    folds_seen_all = df_oof_seen["outer_fold"].values.astype(np.int32)
    seen_map = {
        "m0": df_oof_seen["p_m0_text"].values,
        "m1": df_oof_seen["p_m1_text"].values,
        "m2": df_oof_seen["p_m2_subword"].values,
        "m3": df_oof_seen["p_m3_blend"].values,
        "t97": df_oof_seen["p_t97_text"].values,
    }
    for cand_key, cand_label in [
        ("m0", "M0_podium_text"),
        ("m1", "M1_word_char_enriched_saga"),
        ("m2", "M2_subword_char_nb_svm"),
        ("m3", "M3_nested_text_blend"),
    ]:
        if cand_key != "m0":
            paired_evaluations["seen_val"][f"{cand_label}_vs_M0_podium_text"] = compute_paired_comparison(
                y_true=y_seen_all,
                p_cand=seen_map[cand_key],
                p_ref=seen_map["m0"],
                folds=folds_seen_all,
                sellers=sellers_seen_all,
                cats=cats_seen_all,
                n_boot=BOOT_REPS,
                seed=BOOT_SEED,
            )
        paired_evaluations["seen_val"][f"{cand_label}_vs_T97_p_text"] = compute_paired_comparison(
            y_true=y_seen_all,
            p_cand=seen_map[cand_key],
            p_ref=seen_map["t97"],
            folds=folds_seen_all,
            sellers=sellers_seen_all,
            cats=cats_seen_all,
            n_boot=BOOT_REPS,
            seed=BOOT_SEED,
        )

    # Overall standalone summaries
    standalone_summary = {
        "unseen_val_70k": {},
        "seen_val_all_folds": {},
    }
    for m_key, m_name in [
        ("t97", "T97_p_text"),
        ("m0", "M0_podium_text"),
        ("m1", "M1_word_char_enriched_saga"),
        ("m2", "M2_subword_char_nb_svm"),
        ("m3", "M3_nested_text_blend"),
    ]:
        p_u = df_t97["p_text"].values if m_key == "t97" else oof_unseen[m_key]
        pr_u = np.clip(p_u, 1e-7, 1.0 - 1e-7)
        c_u = (pr_u >= 0.5) == y_all
        standalone_summary["unseen_val_70k"][m_name] = {
            "n_rows": 70000,
            "hits": int(c_u.sum()),
            "accuracy": round(float(c_u.mean()), 6),
            "logloss": round(float(log_loss(y_all, pr_u)), 6),
            "roc_auc": round(float(roc_auc_score(y_all, pr_u)), 6),
        }

        p_s = seen_map[m_key]
        pr_s = np.clip(p_s, 1e-7, 1.0 - 1e-7)
        c_s = (pr_s >= 0.5) == y_seen_all
        standalone_summary["seen_val_all_folds"][m_name] = {
            "n_rows": int(len(y_seen_all)),
            "hits": int(c_s.sum()),
            "accuracy": round(float(c_s.mean()), 6),
            "logloss": round(float(log_loss(y_seen_all, pr_s)), 6),
            "roc_auc": round(float(roc_auc_score(y_seen_all, pr_s)), 6),
        }

    # ---- Step 6: Save manifest.json with SHA-256 hashes ----
    deliverable_files = [
        "per_fold_metrics.csv",
        "inner_oof_fit_predictions.csv",
        "oof_unseen_70k.csv",
        "oof_seen_all_folds.csv",
        "test_predictions_30k.csv",
        "test_disagreements_vs_s4_probs_only.csv",
        *[f"fold_{k}_predictions.npz" for k in range(5)],
        *[f"freeze_line1_outer{k}.json" for k in range(5)],
    ]
    file_hashes = {
        fname: sha256_file(OUT_DIR / fname)
        for fname in deliverable_files
        if (OUT_DIR / fname).exists()
    }

    manifest = {
        "task": "T98_Line1_Text_Subword_Specialist",
        "status": "DEVELOPMENT_ONLY",
        "submission_generated": False,
        "kaggle_api_used": False,
        "script": "scripts/t98_line1_text_subword.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "t98_parsed_train_pkl": str(train_pkl_path.relative_to(ROOT)),
            "t98_parsed_train_sha256": sha256_file(train_pkl_path),
            "t98_parsed_test_pkl": str(test_pkl_path.relative_to(ROOT)),
            "t98_parsed_test_sha256": sha256_file(test_pkl_path),
            "t98_splits_csv": str(splits_csv_path.relative_to(ROOT)),
            "t98_splits_sha256": sha256_file(splits_csv_path),
            "t97_paired_csv": str(t97_paired_path.relative_to(ROOT)),
            "t97_paired_sha256": sha256_file(t97_paired_path),
        },
        "models": {
            "M0_podium_text": "title + tags_str, Word(1,2) 40k min_df=3 + Char_wb(3,5) 60k min_df=5, sublinear_tf=True, LogisticRegression(C=1.5, solver='saga', max_iter=800, random_state=42)",
            "M1_word_char_enriched_saga": "Enriched item-intrinsic corpus (title, subtitle, warranty_str, tags_str, attr_text, listing_type_id, buying_mode + domain tokens), Word(1,3) 80k min_df=2 + Char_wb(3,5) 80k min_df=3, sublinear_tf=True, LogisticRegression(solver='saga', C tuned in [0.75, 1.25, 2.0] on inner folds ONLY)",
            "M2_subword_char_nb_svm": "Subword specialist combining char(2,6) 60k + char_wb(3,6) 60k (120k features, min_df=3, sublinear_tf=True) weighted by Naive Bayes log-count ratio r computed strictly on fit labels, LogisticRegression(solver='saga', C tuned in [0.5, 1.0, 1.5] on inner folds ONLY)",
            "M3_nested_text_blend": "Convex/logit blend of M1 + M2 selected strictly on 4-fold inner_oof inside fit_idx and frozen to disk before outer evaluation",
        },
        "bootstrap_config": {
            "type": "two-way crossed cluster bootstrap seller_id x category_id (Poisson(1) product weights)",
            "n_replicates": BOOT_REPS,
            "seed": BOOT_SEED,
        },
        "standalone_summary": standalone_summary,
        "paired_evaluations": paired_evaluations,
        "test_diagnostics": {
            "n_test_rows": 30000,
            "m3_disagreements_vs_s4_count": int(disagree_mask.sum()),
            "m3_mean_prob_test": round(float(p_m3_test_avg.mean()), 6),
            "m0_mean_prob_test": round(float(p_m0_test_avg.mean()), 6),
        },
        "frozen_policies": frozen_policies,
        "output_file_sha256": file_hashes,
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"=== T98 LINE 1 COMPLETE in {time.time() - t_start:.1f}s | Manifest saved to {manifest_path} ===")


if __name__ == "__main__":
    main()
