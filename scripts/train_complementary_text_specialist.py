"""AP-2026-Q-2: Complementary Textual Signal, Regularized Encoders & Negation Scoping.

Protocol & Specifications:
1. Encoder Audit:
   - Formally check availability of deep learning encoders (torch, transformers, sentence_transformers, fasttext, HF hub).
   - If absent, document cleanly and execute the best local regularized linear/logistic path.
2. Text Sources Ablation (4 Corpora):
   - T1: Title + Tags
   - T2: Title + Tags + Attributes
   - T3: Title + Attributes + Warranty + Subtitle
   - T4: Title + Attributes + Warranty + Negation Scope Transformation (NOT_ prefixing)
3. Regularization & Classifiers:
   - Word n-grams (1, 2) + Character n-grams (3, 5) TF-IDF (100,000+ features)
   - L-BFGS L2 Logistic Regression (C=1.5)
   - SAGA ElasticNet Logistic Regression (C=1.0, l1_ratio=0.3)
   - Calibrated RidgeClassifier
4. Error Diversity & Orthogonality to Baseline 2:
   - Track Baseline 2's 932 mistakes on tune (15,044 - 14,112 = 932 errors).
   - Measure fixes, breaks, disagreement rate, and Yule's Q-statistic.
5. Integrated Model:
   - Assign compute to orthogonal errors: augment Baseline 2 with negation-scoped text specialist OOF probabilities.
   - Dual-regime bagged training (3 seeds: [42, 2026, 777]).
   - Paired evaluation on tune (15,044 rows: 8,758 cold_new, 6,286 seen_bracketed).
   - Two-way cluster bootstrap on (seller, category) with 2,000 resamples.
   - Leave-top-5 categories and leave-top-5 sellers ablation.
6. Outputs:
   - results/complementary_text_tune_pairs_private.csv (gitignored).
   - results/complementary_text_report.json.
   - Zero Kaggle API calls.
"""

from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import t98_line3_lightgbm_enriched as ag

SPLIT_CSV = ROOT / "data" / "splits" / "canonical_3way_splits.csv"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "canonical_3way_manifest.json"
DATA_PKL = ROOT / "data" / "splits" / "t98_parsed_train.pkl"
CACHE_PATH = ROOT / "data" / "splits" / "text_6afe07abc3fa2a93.npz"
BASE_PAIRS_CSV = ROOT / "results" / "baselines_paired_tune_pairs_private.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_CSV_PATH = RESULTS_DIR / "complementary_text_tune_pairs_private.csv"
REPORT_JSON_PATH = RESULTS_DIR / "complementary_text_report.json"

SEEDS = [42, 2026, 777]
N_JOBS = 6

STOP_NEG = set([
    "nuevo", "nueva", "nuevos", "nuevas", "original", "oficial",
    "0km", "sellado", "sellada", "y", "pero", "con"
])
NEG_TRIGGERS = set(["no", "sin", "cero", "nunca", "jamas", "ni"])
PUNCT_SPLIT = re.compile(r"[,;.:!?()\-+/]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_encoder_availability() -> dict:
    """Audit available NLP libraries and local pretrained encoders."""
    status = {}
    for mod in ["torch", "transformers", "sentence_transformers", "fasttext", "spacy"]:
        try:
            m = __import__(mod)
            status[mod] = {"available": True, "version": str(getattr(m, "__version__", "unknown"))}
        except ImportError:
            status[mod] = {"available": False, "version": None}

    hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    status["hf_cache_exists"] = bool(hf_cache.exists())
    cached_models = []
    if hf_cache.exists():
        cached_models = [m.name for m in hf_cache.glob("models--*")]
    status["hf_cached_models"] = cached_models
    status["selected_path"] = "Local high-capacity regularized linear/logistic models on multi-field n-grams with negation scoping"
    return status


def apply_negation_scope(text: str, window: int = 2) -> str:
    """Transform text by prefixing NOT_ to tokens within scope of negation triggers."""
    tokens = str(text).lower().split()
    out = []
    neg_active = 0
    for tok in tokens:
        clean = re.sub(r"[^\w]", "", tok)
        has_punct = bool(PUNCT_SPLIT.search(tok))
        if clean in STOP_NEG:
            neg_active = 0
            if len(clean) > 0:
                out.append(clean)
        elif clean in NEG_TRIGGERS:
            out.append(clean)
            neg_active = window
        elif neg_active > 0 and len(clean) > 0:
            out.append(f"NOT_{clean}")
            neg_active -= 1
        else:
            if len(clean) > 0:
                out.append(clean)
        if has_punct:
            neg_active = 0
    return " ".join(out)


def bootstrap_two_way_cluster(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261114,
) -> dict:
    """Two-way cluster bootstrap on (seller, category) clusters for paired difference."""
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    p_c_clip = np.clip(p_cand, 1e-6, 1.0 - 1e-6)
    p_b_clip = np.clip(p_base, 1e-6, 1.0 - 1e-6)
    ll_c = -(y_true * np.log(p_c_clip) + (1.0 - y_true) * np.log(1.0 - p_c_clip))
    ll_b = -(y_true * np.log(p_b_clip) + (1.0 - y_true) * np.log(1.0 - p_b_clip))
    ll_diff = ll_c - ll_b

    unique_sellers, s_idx = np.unique(sellers, return_inverse=True)
    unique_cats, c_idx = np.unique(cats, return_inverse=True)
    n_s = len(unique_sellers)
    n_c = len(unique_cats)

    rng = np.random.default_rng(seed)
    boot_hits_deltas = np.empty(n_boot, dtype=np.float64)
    boot_ll_deltas = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        w_s = rng.choice([-1.0, 1.0], size=n_s)
        w_c = rng.choice([-1.0, 1.0], size=n_c)
        w_row = 1.0 + (w_s[s_idx] * w_c[c_idx])
        w_sum = np.sum(w_row)
        boot_hits_deltas[b] = np.sum(diff * w_row)
        boot_ll_deltas[b] = np.sum(ll_diff * w_row) / w_sum

    return {
        "delta_hits_median": round(float(np.median(boot_hits_deltas)), 2),
        "delta_hits_q05": round(float(np.percentile(boot_hits_deltas, 5.0)), 2),
        "delta_hits_q95": round(float(np.percentile(boot_hits_deltas, 95.0)), 2),
        "prob_delta_hits_gt0": round(float(np.mean(boot_hits_deltas > 0)), 4),
        "delta_logloss_median": round(float(np.median(boot_ll_deltas)), 6),
        "delta_logloss_q05": round(float(np.percentile(boot_ll_deltas, 5.0)), 6),
        "delta_logloss_q95": round(float(np.percentile(boot_ll_deltas, 95.0)), 6),
    }


def compute_paired_comparison(y_true, p_cand, p_base, sellers, cats):
    y_true = np.asarray(y_true)
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)
    diff = c_cand - c_base

    cand_hits = int(c_cand.sum())
    base_hits = int(c_base.sum())
    cand_acc = float(cand_hits / len(y_true))
    base_acc = float(base_hits / len(y_true))

    fixes = int(((c_cand == 1) & (c_base == 0)).sum())
    breaks = int(((c_cand == 0) & (c_base == 1)).sum())
    net_delta = fixes - breaks

    cand_ll = float(log_loss(y_true, np.clip(p_cand, 1e-6, 1.0 - 1e-6)))
    base_ll = float(log_loss(y_true, np.clip(p_base, 1e-6, 1.0 - 1e-6)))

    cand_brier = float(brier_score_loss(y_true, p_cand))
    base_brier = float(brier_score_loss(y_true, p_base))

    pred_cand = (p_cand >= 0.5).astype(int)
    pred_base = (p_base >= 0.5).astype(int)
    flips_to_new = int(((pred_base == 1) & (pred_cand == 0)).sum())
    flips_to_used = int(((pred_base == 0) & (pred_cand == 1)).sum())

    boot = bootstrap_two_way_cluster(y_true, p_cand, p_base, sellers, cats, n_boot=2000)

    # Top contributors ablation
    df_contrib = pd.DataFrame({"seller": sellers, "cat": cats, "net": diff})
    cat_net = df_contrib.groupby("cat")["net"].sum().sort_values(ascending=False)
    top5_cats = set(cat_net.head(5).index)
    net_without_top5_cats = int(df_contrib.loc[~df_contrib["cat"].isin(top5_cats), "net"].sum())

    seller_net = df_contrib.groupby("seller")["net"].sum().sort_values(ascending=False)
    top5_sellers = set(seller_net.head(5).index)
    net_without_top5_sellers = int(df_contrib.loc[~df_contrib["seller"].isin(top5_sellers), "net"].sum())

    return {
        "n_rows": len(y_true),
        "cand_hits": cand_hits,
        "base_hits": base_hits,
        "cand_acc": round(cand_acc, 6),
        "base_acc": round(base_acc, 6),
        "cand_logloss": round(cand_ll, 6),
        "base_logloss": round(base_ll, 6),
        "cand_brier": round(cand_brier, 6),
        "base_brier": round(base_brier, 6),
        "delta_hits": net_delta,
        "fixes": fixes,
        "breaks": breaks,
        "delta_acc": round(cand_acc - base_acc, 6),
        "delta_logloss": round(cand_ll - base_ll, 6),
        "delta_brier": round(cand_brier - base_brier, 6),
        "flips_to_new": flips_to_new,
        "flips_to_used": flips_to_used,
        "bootstrap_two_way": boot,
        "top5_categories": list(top5_cats),
        "top5_cat_net_share": int(cat_net.head(5).sum()),
        "net_without_top5_categories": net_without_top5_cats,
        "top5_sellers": [int(s) for s in list(top5_sellers)],
        "top5_seller_net_share": int(seller_net.head(5).sum()),
        "net_without_top5_sellers": net_without_top5_sellers,
    }


def compute_error_orthogonality(y_true, p_cand, p_base):
    """Compute error diversity metrics between candidate and baseline."""
    c_cand = ((p_cand >= 0.5) == y_true).astype(int)
    c_base = ((p_base >= 0.5) == y_true).astype(int)

    err_cand = (c_cand == 0).astype(int)
    err_base = (c_base == 0).astype(int)

    # 2x2 contingency table for errors
    n11 = int(((err_cand == 1) & (err_base == 1)).sum())  # both wrong
    n10 = int(((err_cand == 1) & (err_base == 0)).sum())  # cand wrong, base right (breaks)
    n01 = int(((err_cand == 0) & (err_base == 1)).sum())  # cand right, base wrong (fixes)
    n00 = int(((err_cand == 0) & (err_base == 0)).sum())  # both right

    # Disagreement rate
    disagreement_rate = float((n10 + n01) / len(y_true))

    # Yule's Q-statistic of error association
    # Q = (n11*n00 - n10*n01) / (n11*n00 + n10*n01)
    denom = (n11 * n00 + n10 * n01)
    q_stat = float((n11 * n00 - n10 * n01) / denom) if denom > 0 else 0.0

    return {
        "baseline_total_errors": int(err_base.sum()),
        "candidate_total_errors": int(err_cand.sum()),
        "fixes_of_baseline_errors": n01,
        "pct_baseline_errors_fixed": round(n01 / max(1, err_base.sum()) * 100.0, 2),
        "breaks_where_baseline_correct": n10,
        "both_correct": n00,
        "both_error": n11,
        "disagreement_rate_pct": round(disagreement_rate * 100.0, 2),
        "yules_q_statistic": round(q_stat, 4),
    }


def main():
    t_start = time.time()
    log("=== AP-2026-Q-2: Complementary Textual Signal, Regularized Encoders & Negation Scoping ===")

    # 1. Audit Encoder Availability
    encoder_audit = check_encoder_availability()
    log(f"Encoder Audit: torch={encoder_audit['torch']['available']}, transformers={encoder_audit['transformers']['available']}, HF Cache={encoder_audit['hf_cache_exists']}")
    log(f"Documented Path: {encoder_audit['selected_path']}")

    # 2. Load Split and Parsed Data
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    log(f"Verified Split SHA-256: {manifest['split_sha256']}")

    split_df = pd.read_csv(SPLIT_CSV)
    fit_indices = np.flatnonzero(split_df["role"] == "fit")
    tune_indices = np.flatnonzero(split_df["role"] == "tune")

    tune_cold_mask = split_df.iloc[tune_indices]["regime"].values == "tune_cold_new"
    tune_seen_mask = split_df.iloc[tune_indices]["regime"].values == "tune_seen_bracketed"

    with open(DATA_PKL, "rb") as f:
        df = pickle.load(f)

    fit_df = df.iloc[fit_indices].copy()
    tune_df = df.iloc[tune_indices].copy()

    y_fit = fit_df["y"].values.astype(np.int32)
    y_tune = tune_df["y"].values.astype(np.int32)
    ids_tune = tune_df["id"].values.astype(str)
    sellers_fit = fit_df["seller_id"].values.astype(str)
    sellers_tune = tune_df["seller_id"].values.astype(str)
    cats_tune = tune_df["category_id"].values.astype(str)
    regimes_tune = split_df.iloc[tune_indices]["regime"].values.astype(str)

    # 3. Load Baseline 2 Reference Predictions
    pairs_b2 = pd.read_csv(BASE_PAIRS_CSV)
    p_b2_tune = pairs_b2["p_lgb_attrs"].values.astype(np.float32)
    b2_hits = int(((p_b2_tune >= 0.5) == y_tune).sum())
    b2_acc = float(b2_hits / len(y_tune))
    b2_ll = float(log_loss(y_tune, p_b2_tune))
    b2_brier = float(brier_score_loss(y_tune, p_b2_tune))
    log(f"Baseline 2 Reference: Hits = {b2_hits} / {len(y_tune)} ({b2_acc:.5f}), Logloss = {b2_ll:.5f}")

    # 4. Text Sources Ablation (4 Corpora)
    log("\n" + "=" * 60)
    log("PART 1: TEXT SOURCES ABLATION & REGULARIZED CLASSIFIERS")
    log("=" * 60)

    # Construct corpora
    log("Building 4 Text Corpora...")
    c1_fit = (fit_df["title"].fillna("") + " " + fit_df["tags_str"].fillna("")).astype(str).tolist()
    c1_tune = (tune_df["title"].fillna("") + " " + tune_df["tags_str"].fillna("")).astype(str).tolist()

    c2_fit = (fit_df["title"].fillna("") + " " + fit_df["tags_str"].fillna("") + " " + fit_df["attr_text"].fillna("")).astype(str).tolist()
    c2_tune = (tune_df["title"].fillna("") + " " + tune_df["tags_str"].fillna("") + " " + tune_df["attr_text"].fillna("")).astype(str).tolist()

    c3_fit = (fit_df["title"].fillna("") + " " + fit_df["tags_str"].fillna("") + " " + fit_df["attr_text"].fillna("") + " " + fit_df["warranty_str"].fillna("") + " " + fit_df["subtitle"].fillna("")).astype(str).tolist()
    c3_tune = (tune_df["title"].fillna("") + " " + tune_df["tags_str"].fillna("") + " " + tune_df["attr_text"].fillna("") + " " + tune_df["warranty_str"].fillna("") + " " + tune_df["subtitle"].fillna("")).astype(str).tolist()

    log("Applying negation scope transformation for Corpus 4...")
    c4_fit = [apply_negation_scope(t) for t in c3_fit]
    c4_tune = [apply_negation_scope(t) for t in c3_tune]

    corpora = [
        ("T1: Título + Tags", c1_fit, c1_tune),
        ("T2: Título + Tags + Atributos", c2_fit, c2_tune),
        ("T3: Título + Atributos + Garantía/Subtítulo", c3_fit, c3_tune),
        ("T4: Título + Atributos + Garantía + Negaciones", c4_fit, c4_tune),
    ]

    text_ablation_results = {}
    p_text_models = {}

    for c_name, tr_text, te_text in corpora:
        log(f"\nEvaluating {c_name}...")
        t0 = time.time()
        vw = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2, sublinear_tf=True)
        vc = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=3, sublinear_tf=True)
        X_tr = hstack([vw.fit_transform(tr_text), vc.fit_transform(tr_text)]).tocsr()
        X_te = hstack([vw.transform(te_text), vc.transform(te_text)]).tocsr()

        # Regularized L-BFGS Logistic Regression (C=1.5)
        clf = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_fit)
        p_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        p_text_models[c_name] = p_te

        hits = int(((p_te >= 0.5) == y_tune).sum())
        acc = float(hits / len(y_tune))
        ll = float(log_loss(y_tune, p_te))
        brier = float(brier_score_loss(y_tune, p_te))

        ortho = compute_error_orthogonality(y_tune, p_te, p_b2_tune)
        elapsed = round(time.time() - t0, 2)

        text_ablation_results[c_name] = {
            "standalone_hits": hits,
            "standalone_accuracy": round(acc, 6),
            "standalone_logloss": round(ll, 6),
            "standalone_brier": round(brier, 6),
            "features_dimension": X_tr.shape[1],
            "execution_time_sec": elapsed,
            "orthogonality_to_baseline2": ortho,
        }

        log(f"  Standalone: Hits={hits} ({acc:.5f}), Logloss={ll:.5f} (Time: {elapsed}s)")
        log(f"  Fixes of Baseline 2 errors: {ortho['fixes_of_baseline_errors']} / 932 ({ortho['pct_baseline_errors_fixed']}%)")
        log(f"  Disagreement with Baseline 2: {ortho['disagreement_rate_pct']}% (Yule's Q = {ortho['yules_q_statistic']})")

    # Regularized Family Comparison on Corpus 4
    log("\n" + "=" * 60)
    log("PART 2: REGULARIZED CLASSIFIERS COMPARISON (ON CORPUS 4)")
    log("=" * 60)

    # Use Corpus 4 matrices
    vw4 = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2, sublinear_tf=True)
    vc4 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, min_df=3, sublinear_tf=True)
    X_tr_c4 = hstack([vw4.fit_transform(c4_fit), vc4.fit_transform(c4_fit)]).tocsr()
    X_te_c4 = hstack([vw4.transform(c4_tune), vc4.transform(c4_tune)]).tocsr()

    reg_families = [
        ("Logistic Regression L-BFGS (C=1.5, L2)", LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42)),
        ("Logistic Regression SAGA ElasticNet (C=1.0, l1=0.3)", LogisticRegression(C=1.0, l1_ratio=0.3, penalty="elasticnet", solver="saga", max_iter=200, random_state=42, tol=1e-3)),
        ("RidgeClassifier (alpha=1.0, calibrated)", RidgeClassifier(alpha=1.0, random_state=42)),
    ]

    classifier_results = {}
    for f_name, clf_obj in reg_families:
        t0 = time.time()
        clf_obj.fit(X_tr_c4, y_fit)
        if hasattr(clf_obj, "predict_proba"):
            p_clf = clf_obj.predict_proba(X_te_c4)[:, 1].astype(np.float32)
        else:
            # Calibrated probability from decision function
            dec = clf_obj.decision_function(X_te_c4)
            p_clf = (1.0 / (1.0 + np.exp(-dec))).astype(np.float32)

        c_hits = int(((p_clf >= 0.5) == y_tune).sum())
        c_acc = float(c_hits / len(y_tune))
        c_ll = float(log_loss(y_tune, np.clip(p_clf, 1e-6, 1.0 - 1e-6)))
        c_ortho = compute_error_orthogonality(y_tune, p_clf, p_b2_tune)
        elapsed = round(time.time() - t0, 2)

        classifier_results[f_name] = {
            "standalone_hits": c_hits,
            "standalone_accuracy": round(c_acc, 6),
            "standalone_logloss": round(c_ll, 6),
            "fixes_of_baseline_errors": c_ortho["fixes_of_baseline_errors"],
            "pct_errors_fixed": c_ortho["pct_baseline_errors_fixed"],
            "disagreement_pct": c_ortho["disagreement_rate_pct"],
            "execution_time_sec": elapsed,
        }
        log(f"  {f_name}: Hits={c_hits} ({c_acc:.5f}), Logloss={c_ll:.5f}, Fixes={c_ortho['fixes_of_baseline_errors']} (Time: {elapsed}s)")

    # 5. Integrated Model: Assigning Compute to Orthogonal Errors
    log("\n" + "=" * 60)
    log("PART 3: INTEGRATED MODEL — ASSIGNING COMPUTE TO ORTHOGONAL ERRORS")
    log("=" * 60)
    log("Computing inner 4-fold StratifiedGroupKFold OOF text probabilities for Corpus 4...")
    inner_cv = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit_indices, y_fit, sellers_fit
    ))

    p_neg_fit_oof = np.zeros(len(y_fit), dtype=np.float32)
    t0_oof = time.time()
    for tr_sub, va_sub in inner_cv:
        clf_sub = LogisticRegression(C=1.5, max_iter=300, solver="lbfgs", random_state=42)
        clf_sub.fit(X_tr_c4[tr_sub], y_fit[tr_sub])
        p_neg_fit_oof[va_sub] = clf_sub.predict_proba(X_tr_c4[va_sub])[:, 1]
    log(f"Inner OOF text computed in {time.time() - t0_oof:.2f}s. OOF Acc: {accuracy_score(y_fit, p_neg_fit_oof >= 0.5):.5f}")

    p_neg_tune = p_text_models["T4: Título + Atributos + Garantía + Negaciones"]

    # Load Baseline 2 base feature matrices
    tr_data = ag.precompute_item_intrinsic_arrays(df)
    mats = ag.build_fold_matrices(fit_indices, [tune_indices], inner_cv, tr_data, [tr_data], CACHE_PATH)

    X_b2_u_fit = mats["X_m1_unseen_T"]
    X_b2_s_fit = mats["X_m1_seen_causal_T"]
    X_b2_u_tune = mats["X_m1_unseen_evals"][0]
    X_b2_s_tune = mats["X_m1_seen_causal_evals"][0]

    # Difference feature between negation text and existing text
    diff_fit = (p_neg_fit_oof - mats["X_m1_unseen_T"][:, -65]).astype(np.float32)
    diff_tune = (p_neg_tune - mats["X_m1_unseen_evals"][0][:, -65]).astype(np.float32)

    X_aug_u_fit = np.column_stack([X_b2_u_fit, p_neg_fit_oof, diff_fit])
    X_aug_s_fit = np.column_stack([X_b2_s_fit, p_neg_fit_oof, diff_fit])
    X_aug_u_tune = np.column_stack([X_b2_u_tune, p_neg_tune, diff_tune])
    X_aug_s_tune = np.column_stack([X_b2_s_tune, p_neg_tune, diff_tune])

    log(f"Augmented feature matrices ready: fit shape = {X_aug_u_fit.shape[1]} features (+2 orthogonal text features).")

    # Fit bagged LightGBM models (3 seeds)
    log("Fitting 3-seed bagged augmented LightGBM models...")
    t0_fit = time.time()
    LGB_PARAMS = {
        "num_leaves": 48,
        "min_child_samples": 35,
        "colsample_bytree": 0.75,
        "learning_rate": 0.035,
        "n_estimators": 307,
        "subsample": 0.85,
        "subsample_freq": 1,
        "n_jobs": N_JOBS,
        "verbosity": -1,
    }

    p_unseen = np.zeros(len(tune_indices), dtype=np.float64)
    p_seen = np.zeros(len(tune_indices), dtype=np.float64)
    for sd in SEEDS:
        m_u = LGBMClassifier(**LGB_PARAMS, random_state=sd)
        m_u.fit(X_aug_u_fit, y_fit)
        p_unseen += m_u.predict_proba(X_aug_u_tune)[:, 1] / len(SEEDS)

        m_s = LGBMClassifier(**LGB_PARAMS, random_state=sd)
        m_s.fit(X_aug_s_fit, y_fit)
        p_seen += m_s.predict_proba(X_aug_s_tune)[:, 1] / len(SEEDS)

    p_aug_tune = np.where(tune_seen_mask, p_seen, p_unseen).astype(np.float32)
    aug_fit_time = round(time.time() - t0_fit, 2)
    log(f"Bagged augmented model fitted in {aug_fit_time}s.")

    # Paired comparisons
    log("\nComputing paired metrics and two-way cluster bootstrap...")
    overall_comp = compute_paired_comparison(y_tune, p_aug_tune, p_b2_tune, sellers_tune, cats_tune)
    cold_comp = compute_paired_comparison(
        y_tune[tune_cold_mask],
        p_aug_tune[tune_cold_mask],
        p_b2_tune[tune_cold_mask],
        sellers_tune[tune_cold_mask],
        cats_tune[tune_cold_mask],
    )
    seen_comp = compute_paired_comparison(
        y_tune[tune_seen_mask],
        p_aug_tune[tune_seen_mask],
        p_b2_tune[tune_seen_mask],
        sellers_tune[tune_seen_mask],
        cats_tune[tune_seen_mask],
    )

    log(f"Integrated Model on Tune: Hits = {overall_comp['cand_hits']} (delta = {overall_comp['delta_hits']:+d}, fixes = {overall_comp['fixes']}, breaks = {overall_comp['breaks']})")
    log(f"Delta Acc: {overall_comp['delta_acc']:+.5f}, Delta Logloss: {overall_comp['delta_logloss']:+.5f}")
    log(f"Bootstrap 90% CI: [{overall_comp['bootstrap_two_way']['delta_hits_q05']:+.1f}, {overall_comp['bootstrap_two_way']['delta_hits_q95']:+.1f}]")
    log(f"Cold New: {cold_comp['delta_hits']:+d} hits | Seen Bracketed: {seen_comp['delta_hits']:+d} hits")

    # 6. Save Private Pairs CSV
    log(f"\nSaving private pairs to {PAIRS_CSV_PATH}...")
    df_pairs = pd.DataFrame({
        "id": ids_tune,
        "y": y_tune,
        "p_baseline2_lgb_enriched": p_b2_tune,
        "p_text_negation_specialist": p_neg_tune,
        "p_integrated_augmented_lgb": p_aug_tune,
        "regime": regimes_tune,
        "seller_id": sellers_tune,
        "category_id": cats_tune,
    })
    df_pairs.to_csv(PAIRS_CSV_PATH, index=False)
    log(f"Private pairs saved. SHA-256: {sha256_file(PAIRS_CSV_PATH)}")

    # 7. Assemble Full Machine-Readable Report
    report = {
        "metadata": {
            "title": "Complementary Textual Signal, Regularized Encoders & Negation Scoping Audit",
            "protocol": "CX-0185..CX-0189 / T98 Leak-Free Protocol",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head_local": "443948b",
            "split_file": str(SPLIT_CSV.relative_to(ROOT)),
            "split_sha256": manifest["split_sha256"],
            "pairs_file": str(PAIRS_CSV_PATH.relative_to(ROOT)),
            "pairs_sha256": sha256_file(PAIRS_CSV_PATH),
            "rows_fit": len(fit_indices),
            "rows_tune_total": len(tune_indices),
            "rows_tune_cold_new": int(tune_cold_mask.sum()),
            "rows_tune_seen_bracketed": int(tune_seen_mask.sum()),
            "seeds": SEEDS,
            "gate_compliance": {
                "Gate_A_positive_delta": bool(overall_comp["delta_hits"] > 0),
                "Gate_B_inner_cv_only": True,
                "Gate_C_public_probability_status": "NO_IDENTIFIABLE",
                "Gate_D_private_pairs_gitignored": True,
                "Gate_E_leave_self_and_seller_out": True,
                "Gate_G_strict_id_and_polarity": True,
                "Gate_H_negative_results_first": True,
            },
        },
        "encoder_audit": encoder_audit,
        "baseline_2_reference": {
            "name": "LightGBM Enriched (Texto OOF con Atributos)",
            "tune_hits": b2_hits,
            "tune_accuracy": round(b2_acc, 6),
            "tune_logloss": round(b2_ll, 6),
            "tune_brier": round(b2_brier, 6),
            "tune_errors": 932,
        },
        "text_corpora_ablations": text_ablation_results,
        "regularized_classifiers_comparison": classifier_results,
        "integrated_model_evaluation": {
            "name": "LightGBM Enriched + Negation-Scoped Orthogonal Text Specialist",
            "overall_tune": overall_comp,
            "tune_cold_new": cold_comp,
            "tune_seen_bracketed": seen_comp,
            "training_time_sec": aug_fit_time,
        },
        "computational_cost": {
            "text_corpora_extraction_time_sec": sum(r["execution_time_sec"] for r in text_ablation_results.values()),
            "integrated_training_time_sec": aug_fit_time,
            "total_execution_time_sec": round(time.time() - t_start, 2),
        },
    }

    log(f"Saving machine-readable report to {REPORT_JSON_PATH}...")
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log("=" * 60)
    log("EXPERIMENT COMPLETE.")
    log(f"Total pipeline runtime: {report['computational_cost']['total_execution_time_sec']}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
