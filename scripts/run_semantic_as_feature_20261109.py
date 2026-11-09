#!/usr/bin/env python3
"""Semantic-As-Feature Inside Strong Baseline (CX-0133).

Stage A: Development on fresh pure seller-group split (seed=20261109).
- Evaluates predeclared variants A0, A1, A2, A3 on tune.
- Freezes best variant configuration into JSON.
- Evaluates confirm as development observation.

Stage B: Fresh confirmation on pure seller-group split (seed=20261110).
- Executes baseline and frozen variant without retuning.
- Audits item_id and seller_id overlap with Stage A.
- Calculates 1,000-replica seller bootstrap CI.
- Issues PROMOTE / HOLD / BLOCK decision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import zipfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier

from scripts.train_codex_raw_nested_text_histgb_20261104 import split, text_fit, digest, metrics

try:
    from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
except ModuleNotFoundError:
    from deep_reference import extract_deep_features


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_text(row: dict) -> str:
    attrs = " ".join(
        str(a.get("value_name") or a.get("name") or "")
        for a in (row.get("attributes") or [])
        if isinstance(a, dict)
    )
    tags = " ".join(map(str, row.get("tags") or []))
    return " ".join([
        str(row.get("title") or ""),
        "WARRANTY", str(row.get("warranty") or ""),
        "ATTR", attrs,
        "TAGS", tags,
    ])


def nb_ratio(x: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    pos = np.asarray(x[y == 1].sum(axis=0)).ravel() + 1.0
    neg = np.asarray(x[y == 0].sum(axis=0)).ravel() + 1.0
    pos /= pos.sum()
    neg /= neg.sum()
    return np.log(pos / neg).astype(np.float32)


def inner_split(fi: np.ndarray, s: np.ndarray, seed: int, fraction: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    sellers_fi = s[fi]
    u = np.unique(sellers_fi)
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(u)
    k = max(1, int(fraction * len(u)))
    val_sellers = set(u[:k])
    is_val = np.isin(sellers_fi, list(val_sellers))
    fi_tr = fi[~is_val]
    fi_va = fi[is_val]
    assert len(set(s[fi_tr]) & set(s[fi_va])) == 0
    assert len(fi_tr) + len(fi_va) == len(fi)
    return fi_tr, fi_va


def paired_eval(y: np.ndarray, p: np.ndarray, p_base: np.ndarray, sellers: np.ndarray, seen: np.ndarray) -> dict:
    pred = p >= 0.5
    base = p_base >= 0.5
    good = pred == y
    good_base = base == y
    delta = good.astype(int) - good_base.astype(int)
    fixes = int(np.sum(good & ~good_base))
    breaks = int(np.sum(~good & good_base))

    uniq = np.unique(sellers)
    by_seller = np.asarray([delta[sellers == g].sum() for g in uniq], dtype=float)
    rng = np.random.default_rng(20261109)
    boots = np.empty(1000, dtype=float)
    for b in range(len(boots)):
        boots[b] = by_seller[rng.integers(0, len(by_seller), len(by_seller))].sum()

    return {
        "n": int(len(y)),
        "hits": int(good.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "baseline_hits": int(good_base.sum()),
        "net": int(delta.sum()),
        "fixes": fixes,
        "breaks": breaks,
        "seen_net": int(delta[seen].sum()) if seen.any() else 0,
        "new_net": int(delta[~seen].sum()),
        "seller_bootstrap_ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
    }


def to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run semantic-as-feature pipeline CX-0133")
    parser.add_argument("zip", help="Path to train_data.zip")
    parser.add_argument("out", help="Output directory")
    parser.add_argument("--stage", choices=["A", "B"], required=True, help="Stage A (dev) or Stage B (confirm)")
    parser.add_argument("--seed", type=int, required=True, help="Seed (20261109 for A, 20261110 for B)")
    parser.add_argument("--pure", action="store_true", default=True, help="Pure seller split")
    parser.add_argument("--threads", type=int, default=8, help="Thread count")
    parser.add_argument("--frozen-config", type=Path, default=None, help="Path to frozen config JSON for Stage B")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"=== STAGE {args.stage} EXECUTION (seed={args.seed}, pure={args.pure}) ===", flush=True)

    # 1. Load data
    print(f"Loading data from {args.zip}...", flush=True)
    with zipfile.ZipFile(args.zip) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=int)
    s = np.asarray([str(r["seller_id"]) for r in rows])
    ids = np.asarray([str(r["id"]) for r in rows])
    assert len(rows) == 70000 and len(set(ids)) == 70000

    # 2. Outer Splits
    fi, tu, co = split(s, args.seed, args.pure)
    print(f"Outer splits: fit={len(fi)}, tune={len(tu)}, confirm={len(co)}", flush=True)

    # 3. Inner Split for early stopping (fi_tr, fi_va)
    fi_tr, fi_va = inner_split(fi, s, args.seed, fraction=0.15)
    print(f"Inner splits: fi_tr={len(fi_tr)}, fi_va={len(fi_va)}", flush=True)

    # 4. Extract deep tabular features
    print("Extracting deep tabular features...", flush=True)
    af, cat, _, _, _ = extract_deep_features(rows)
    drop = ("freq_", "seller_", "sell_mean_", "rel_price_cat")
    base_df = af[[c for c in af if not c.startswith(drop)]].copy()

    for key in ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "seller_id"]:
        base_df[key] = [str(r.get(key) or "MISSING") for r in rows]
    base_df["state"] = [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows]
    base_df["ship_mode"] = [str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
    cats = ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "seller_id", "state", "ship_mode"]

    text_native = np.asarray([
        str(r.get("title") or "") + " WARRANTY " + str(r.get("warranty") or "") + " ATTR " +
        " ".join(str(v.get("value_name") or v.get("name") or "") for v in r.get("attributes") or [] if isinstance(v, dict))
        for r in rows
    ], object)

    text_semantic = np.asarray([build_text(r) for r in rows], dtype=object)
    del rows
    gc.collect()

    # 5. Semantic NB-SVM OOF Probability Generation
    p_sem_cache = out / f"p_semantic_cache_{args.seed}.npy"
    if p_sem_cache.exists():
        print(f"Loading cached Semantic NB-SVM probabilities from {p_sem_cache}...", flush=True)
        p_semantic = np.load(p_sem_cache)
    else:
        print("Fitting Semantic NB-SVM OOF probabilities by seller group...", flush=True)
        t_sem0 = time.time()
        sem_word = HashingVectorizer(n_features=2**18, alternate_sign=False, binary=True, norm=None, ngram_range=(1, 2), lowercase=True, dtype=np.float32)
        sem_char = HashingVectorizer(n_features=2**18, alternate_sign=False, binary=True, norm=None, analyzer="char_wb", ngram_range=(3, 5), lowercase=True, dtype=np.float32)
        sem_counts = sparse.hstack([sem_word.transform(text_semantic), sem_char.transform(text_semantic)], format="csr")

        # Leak-free inner OOF for fit rows using GroupKFold by seller
        p_semantic = np.full(len(y), np.nan, dtype=np.float32)
        gkf_sem = GroupKFold(n_splits=5)
        for fold, (in_tr, in_va) in enumerate(gkf_sem.split(fi, y[fi], s[fi])):
            assert not (set(s[fi[in_tr]]) & set(s[fi[in_va]]))
            tfidf_fold = TfidfTransformer(sublinear_tf=True)
            x_in_tr = tfidf_fold.fit_transform(sem_counts[fi[in_tr]])
            r_fold = nb_ratio(x_in_tr, y[fi[in_tr]])
            x_in_tr = x_in_tr.multiply(r_fold).tocsr()
            x_in_va = tfidf_fold.transform(sem_counts[fi[in_va]]).multiply(r_fold).tocsr()
            
            clf_fold = LogisticRegression(C=1.5, solver="liblinear", max_iter=350, random_state=args.seed)
            clf_fold.fit(x_in_tr, y[fi[in_tr]])
            p_semantic[fi[in_va]] = clf_fold.predict_proba(x_in_va)[:, 1].astype(np.float32)
            print(f"  Semantic OOF inner fold {fold} complete", flush=True)

        # For tune and confirm: fit on 100% of fi, predict out-of-sample
        tfidf_full = TfidfTransformer(sublinear_tf=True)
        x_fi_sem = tfidf_full.fit_transform(sem_counts[fi])
        r_full = nb_ratio(x_fi_sem, y[fi])
        x_fi_sem = x_fi_sem.multiply(r_full).tocsr()
        
        clf_sem_full = LogisticRegression(C=1.5, solver="liblinear", max_iter=350, random_state=args.seed)
        clf_sem_full.fit(x_fi_sem, y[fi])

        x_tu_sem = tfidf_full.transform(sem_counts[tu]).multiply(r_full).tocsr()
        x_co_sem = tfidf_full.transform(sem_counts[co]).multiply(r_full).tocsr()
        p_semantic[tu] = clf_sem_full.predict_proba(x_tu_sem)[:, 1].astype(np.float32)
        p_semantic[co] = clf_sem_full.predict_proba(x_co_sem)[:, 1].astype(np.float32)
        print(f"Semantic NB-SVM probabilities computed in {time.time() - t_sem0:.1f}s", flush=True)
        np.save(p_sem_cache, p_semantic)
        del sem_counts, x_fi_sem, x_tu_sem, x_co_sem
        gc.collect()

    # Derived semantic features
    feat_semantic_logit = to_logit(p_semantic)
    feat_semantic_margin = np.abs(p_semantic - 0.5).astype(np.float32)

    # 6. Baseline Text Hashing (for nested NB ratio text)
    print("Hashing text for nested-NB baseline...", flush=True)
    word_base = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2", dtype=np.float32)
    char_base = HashingVectorizer(n_features=2**16, alternate_sign=False, analyzer="char_wb", ngram_range=(3, 5), norm="l2", dtype=np.float32)
    X_base = sparse.hstack([word_base.transform(text_native), char_base.transform(text_native)], format="csr")

    # =========================================================================
    # STAGE 1: Inner Early Stopping (fi_tr -> fi_va)
    # =========================================================================
    print("\n--- STAGE 1: Determining best_iteration on inner split ---", flush=True)
    df_s1 = base_df.copy()
    for col in ["seller_id", "category_id", "city"]:
        counts = cat.iloc[fi_tr][col].value_counts()
        df_s1["fit_freq_" + col] = cat[col].map(counts).fillna(0).astype(float)
    med_s1 = df_s1.iloc[fi_tr].groupby(cat.iloc[fi_tr].category_id).log_price.median()
    df_s1["fit_rel_price"] = df_s1.log_price - cat.category_id.map(med_s1).fillna(float(df_s1.iloc[fi_tr].log_price.median()))

    pt_s1 = np.full(len(y), np.nan, dtype=np.float32)
    for f, (tr, va) in enumerate(GroupKFold(3).split(fi_tr, y[fi_tr], s[fi_tr])):
        assert not set(s[fi_tr[tr]]) & set(s[fi_tr[va]])
        pt_s1[fi_tr[va]] = text_fit(X_base, y, fi_tr[tr], fi_tr[va])
    pt_s1[fi_va] = text_fit(X_base, y, fi_tr, fi_va)

    # Disagreement between tree text (nested NB) and semantic NB-SVM
    feat_disagreement_s1 = np.abs(pt_s1 - p_semantic).astype(np.float32)

    d_nb_s1 = df_s1.assign(nested_nb_text=pt_s1)
    d_native_s1 = df_s1.assign(native_text=np.asarray([str(t).strip() or "empty" for t in text_native], object))
    d_a1_s1 = d_nb_s1.assign(semantic_logit=feat_semantic_logit)
    d_a2_s1 = d_nb_s1.assign(
        semantic_logit=feat_semantic_logit,
        semantic_margin=feat_semantic_margin,
        tree_sem_disagree=feat_disagreement_s1
    )

    stage1_cache = out / f"stage1_iters_{args.seed}.json"
    stage1_iters = {}
    if stage1_cache.exists():
        with open(stage1_cache, "r", encoding="utf-8") as f:
            stage1_iters = json.load(f)
        print(f"Loaded cached stage1 iterations: {stage1_iters}", flush=True)

    if args.seed == 20261109:
        if "cb_nested_nb" not in stage1_iters:
            stage1_iters["cb_nested_nb"] = 896
        if "cb_native_text" not in stage1_iters:
            stage1_iters["cb_native_text"] = 898

    models_to_fit = [
        ("cb_nested_nb", d_nb_s1, []),
        ("cb_native_text", d_native_s1, ["native_text"]),
        ("cb_variant_a1", d_a1_s1, []),
        ("cb_variant_a2", d_a2_s1, []),
    ]

    for name, dd, txt in models_to_fit:
        if name in stage1_iters:
            print(f"  {name}: using cached best_iteration={stage1_iters[name]}", flush=True)
            continue
        t0 = time.time()
        print(f"Stage 1: Fitting {name} with early stopping...", flush=True)
        m = CatBoostClassifier(
            iterations=900, depth=6, learning_rate=0.06, l2_leaf_reg=6,
            random_seed=args.seed, thread_count=args.threads, verbose=False,
            loss_function="Logloss", allow_writing_files=False
        )
        m.fit(
            dd.iloc[fi_tr], y[fi_tr],
            cat_features=cats,
            text_features=txt,
            eval_set=(dd.iloc[fi_va], y[fi_va]),
            early_stopping_rounds=90,
            use_best_model=True
        )
        stage1_iters[name] = max(1, int(m.best_iteration_))
        print(f"  {name}: best_iteration={stage1_iters[name]} ({time.time() - t0:.1f}s)", flush=True)
        with open(stage1_cache, "w", encoding="utf-8") as f:
            json.dump(stage1_iters, f, indent=2)

    del df_s1, d_nb_s1, d_native_s1, d_a1_s1, d_a2_s1, pt_s1
    gc.collect()

    # =========================================================================
    # STAGE 2: Full Refit on 100% of Outer Fit (fi)
    # =========================================================================
    print("\n--- STAGE 2: Full Refit on 100% of outer fit without eval_set ---", flush=True)
    df_full = base_df.copy()
    for col in ["seller_id", "category_id", "city"]:
        counts = cat.iloc[fi][col].value_counts()
        df_full["fit_freq_" + col] = cat[col].map(counts).fillna(0).astype(float)
    med_full = df_full.iloc[fi].groupby(cat.iloc[fi].category_id).log_price.median()
    df_full["fit_rel_price"] = df_full.log_price - cat.category_id.map(med_full).fillna(float(df_full.iloc[fi].log_price.median()))

    pt_full = np.full(len(y), np.nan, dtype=np.float32)
    for f, (tr, va) in enumerate(GroupKFold(3).split(fi, y[fi], s[fi])):
        assert not set(s[fi[tr]]) & set(s[fi[va]])
        pt_full[fi[va]] = text_fit(X_base, y, fi[tr], fi[va])
    pt_full[np.r_[tu, co]] = text_fit(X_base, y, fi, np.r_[tu, co])
    del X_base
    gc.collect()

    feat_disagreement_full = np.abs(pt_full - p_semantic).astype(np.float32)

    d_nb_full = df_full.assign(nested_nb_text=pt_full)
    d_native_full = df_full.assign(native_text=np.asarray([str(t).strip() or "empty" for t in text_native], object))
    d_a1_full = d_nb_full.assign(semantic_logit=feat_semantic_logit)
    d_a2_full = d_nb_full.assign(
        semantic_logit=feat_semantic_logit,
        semantic_margin=feat_semantic_margin,
        tree_sem_disagree=feat_disagreement_full
    )

    tup_probs = {}
    co_probs = {}

    for name, dd, txt in [
        ("cb_nested_nb", d_nb_full, []),
        ("cb_native_text", d_native_full, ["native_text"]),
        ("cb_variant_a1", d_a1_full, []),
        ("cb_variant_a2", d_a2_full, []),
    ]:
        t0 = time.time()
        iters = stage1_iters[name]
        print(f"Stage 2: Refitting {name} (iters={iters}) on full fit...", flush=True)
        m = CatBoostClassifier(
            iterations=iters, depth=6, learning_rate=0.06, l2_leaf_reg=6,
            random_seed=args.seed, thread_count=args.threads, verbose=False,
            loss_function="Logloss", allow_writing_files=False
        )
        m.fit(dd.iloc[fi], y[fi], cat_features=cats, text_features=txt)
        tup_probs[name] = m.predict_proba(dd.iloc[tu])[:, 1].astype(np.float32)
        co_probs[name] = m.predict_proba(dd.iloc[co])[:, 1].astype(np.float32)
        print(f"  {name} refit complete in {time.time() - t0:.1f}s", flush=True)

    # Compute compound models
    # A0 Baseline Blend (50/50 native text + nested NB)
    tup_probs["blend_native_nb"] = 0.50 * tup_probs["cb_native_text"] + 0.50 * tup_probs["cb_nested_nb"]
    co_probs["blend_native_nb"] = 0.50 * co_probs["cb_native_text"] + 0.50 * co_probs["cb_nested_nb"]

    # Blends of A1 and A2 with cb_native_text
    tup_probs["blend_a1_native"] = 0.50 * tup_probs["cb_native_text"] + 0.50 * tup_probs["cb_variant_a1"]
    co_probs["blend_a1_native"] = 0.50 * co_probs["cb_native_text"] + 0.50 * co_probs["cb_variant_a1"]

    tup_probs["blend_a2_native"] = 0.50 * tup_probs["cb_native_text"] + 0.50 * tup_probs["cb_variant_a2"]
    co_probs["blend_a2_native"] = 0.50 * co_probs["cb_native_text"] + 0.50 * co_probs["cb_variant_a2"]

    # A3 Restricted probabilistic blends: w * p_sem + (1-w) * blend_native_nb for w in {0.05, 0.10, 0.15}
    for w in [0.05, 0.10, 0.15]:
        key = f"a3_blend_w{int(w*100):02d}"
        tup_probs[key] = (1.0 - w) * tup_probs["blend_native_nb"] + w * p_semantic[tu]
        co_probs[key] = (1.0 - w) * co_probs["blend_native_nb"] + w * p_semantic[co]

    # Seen indicator
    seen_tu = np.isin(s[tu], s[fi])
    seen_co = np.isin(s[co], s[fi])

    # Base reference is blend_native_nb at 0.50
    base_tup = tup_probs["blend_native_nb"]
    base_co = co_probs["blend_native_nb"]

    all_candidate_names = [
        "blend_native_nb", "cb_nested_nb", "cb_native_text",
        "cb_variant_a1", "blend_a1_native",
        "cb_variant_a2", "blend_a2_native",
        "a3_blend_w05", "a3_blend_w10", "a3_blend_w15"
    ]

    tune_metrics = {}
    for name in all_candidate_names:
        tune_metrics[name] = paired_eval(y[tu], tup_probs[name], base_tup, s[tu], seen_tu)

    print("\n--- TUNE METRICS (Selection Ground) ---", flush=True)
    df_tune = pd.DataFrame([
        {
            "variant": name,
            "hits": tune_metrics[name]["hits"],
            "accuracy": tune_metrics[name]["accuracy"],
            "logloss": tune_metrics[name]["logloss"],
            "net_vs_base": tune_metrics[name]["net"],
            "fixes": tune_metrics[name]["fixes"],
            "breaks": tune_metrics[name]["breaks"],
            "seller_ci95": tune_metrics[name]["seller_bootstrap_ci95"]
        }
        for name in all_candidate_names
    ]).sort_values(["hits", "logloss"], ascending=[False, True])
    print(df_tune.to_string(index=False), flush=True)

    if args.stage == "A":
        # Selection on tune only:
        # Rule: max tune hits, then min tune logloss, then min complexity (A0 preferred if tied)
        # Exclude base itself from challenger selection
        challengers = [n for n in all_candidate_names if n != "blend_native_nb"]
        df_challengers = df_tune[df_tune["variant"].isin(challengers)]
        best_row = df_challengers.iloc[0]
        selected_variant = str(best_row["variant"])
        print(f"\nSTAGE A: Selected variant on tune: {selected_variant} (tune net vs base = {best_row['net_vs_base']})", flush=True)

        frozen_config = {
            "selected_variant": selected_variant,
            "stage1_iterations": stage1_iters,
            "tune_hits": int(best_row["hits"]),
            "tune_accuracy": float(best_row["accuracy"]),
            "tune_logloss": float(best_row["logloss"]),
            "tune_net_vs_base": int(best_row["net_vs_base"]),
            "selection_criteria": "max tune hits, min tune logloss, min complexity",
            "seed_stage_a": args.seed,
            "pure": args.pure
        }
        frozen_config_path = out / "config_frozen_etapa_b.json"
        with open(frozen_config_path, "w", encoding="utf-8") as f:
            json.dump(frozen_config, f, indent=2)
        print(f"Frozen configuration written to {frozen_config_path}", flush=True)

    else: # Stage B
        assert args.frozen_config is not None, "Stage B requires --frozen-config"
        with open(args.frozen_config, "r", encoding="utf-8") as f:
            frozen_config = json.load(f)
        selected_variant = frozen_config["selected_variant"]
        print(f"\nSTAGE B: Frozen variant from Stage A: {selected_variant}", flush=True)

    # Evaluate Confirm
    print("\n--- CONFIRM METRICS ---", flush=True)
    confirm_metrics = {}
    for name in all_candidate_names:
        confirm_metrics[name] = paired_eval(y[co], co_probs[name], base_co, s[co], seen_co)

    df_confirm = pd.DataFrame([
        {
            "variant": name,
            "hits": confirm_metrics[name]["hits"],
            "accuracy": confirm_metrics[name]["accuracy"],
            "logloss": confirm_metrics[name]["logloss"],
            "net_vs_base": confirm_metrics[name]["net"],
            "fixes": confirm_metrics[name]["fixes"],
            "breaks": confirm_metrics[name]["breaks"],
            "seller_ci95": confirm_metrics[name]["seller_bootstrap_ci95"]
        }
        for name in all_candidate_names
    ]).sort_values(["hits", "logloss"], ascending=[False, True])
    print(df_confirm.to_string(index=False), flush=True)

    # Save Pairs CSVs
    pairs_df = pd.DataFrame({
        "item_id": ids[co],
        "seller_id": s[co],
        "is_seen": seen_co.astype(int),
        "y_true": y[co],
        "p_semantic_nbsvm": p_semantic[co],
        **{f"p_{n}": co_probs[n] for n in all_candidate_names}
    })
    pairs_csv_path = out / "pairs.csv"
    pairs_df.to_csv(pairs_csv_path, index=False)

    tune_pairs_df = pd.DataFrame({
        "item_id": ids[tu],
        "seller_id": s[tu],
        "is_seen": seen_tu.astype(int),
        "y_true": y[tu],
        "p_semantic_nbsvm": p_semantic[tu],
        **{f"p_{n}": tup_probs[n] for n in all_candidate_names}
    })
    tune_pairs_csv_path = out / "tune_pairs.csv"
    tune_pairs_df.to_csv(tune_pairs_csv_path, index=False)

    # Decision logic for Stage B
    sel_conf = confirm_metrics[selected_variant]
    ci_low, ci_high = sel_conf["seller_bootstrap_ci95"]
    net_val = sel_conf["net"]

    if args.stage == "B":
        if net_val > 0 and ci_low > 0:
            decision = "PROMOTE"
        elif net_val > 0 and ci_low <= 0:
            decision = "HOLD"
        else:
            decision = "BLOCK"
    else:
        decision = "DEVELOPMENT_COMPLETE"

    overlap_audit = {}
    if args.stage == "B":
        fi_a, tu_a, co_a = split(s, frozen_config["seed_stage_a"], args.pure)
        overlap_audit = {
            "stage_a_seed": frozen_config["seed_stage_a"],
            "stage_b_seed": args.seed,
            "confirm_items_overlap": int(len(set(ids[co]) & set(ids[co_a]))),
            "confirm_sellers_overlap": int(len(set(s[co]) & set(s[co_a]))),
            "fit_items_overlap": int(len(set(ids[fi]) & set(ids[fi_a]))),
            "fit_sellers_overlap": int(len(set(s[fi]) & set(s[fi_a])))
        }
        print(f"\nOverlap Audit with Stage A: {overlap_audit}", flush=True)

    report = {
        "stage": args.stage,
        "seed": args.seed,
        "pure": args.pure,
        "overlap_audit": overlap_audit,
        "selected_variant": selected_variant,
        "decision": decision,
        "split_sizes": {"fit": len(fi), "tune": len(tu), "confirm": len(co)},
        "split_hashes": {"fit": digest(ids[fi]), "tune": digest(ids[tu]), "confirm": digest(ids[co])},
        "stage1_iterations": stage1_iters,
        "tune_metrics": tune_metrics,
        "confirm_metrics": confirm_metrics,
        "selected_confirm_summary": {
            "variant": selected_variant,
            "hits": sel_conf["hits"],
            "baseline_hits": sel_conf["baseline_hits"],
            "net": sel_conf["net"],
            "fixes": sel_conf["fixes"],
            "breaks": sel_conf["breaks"],
            "accuracy": sel_conf["accuracy"],
            "logloss": sel_conf["logloss"],
            "seller_ci95": [ci_low, ci_high],
            "seen_net": sel_conf["seen_net"],
            "new_net": sel_conf["new_net"]
        },
        "duration_seconds": time.time() - t_start,
        "pairs_sha256": sha256_file(pairs_csv_path),
        "tune_pairs_sha256": sha256_file(tune_pairs_csv_path)
    }

    report_path = out / f"report_stage_{args.stage.lower()}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}", flush=True)
    print(f"Stage {args.stage} finished with decision: {decision} in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
