"""Ablation: Injecting Prefixed Structured Text OOF into Deep CatBoost.

Hypothesis:
Post-hoc linear probability blending failed because static weights cannot condition
on seller or tabular context. Supplying `p_prefixed_oof` (trained on title, warranty,
and structured aid/name key-values) as a continuous feature directly to CatBoost
allows tree splits to interact structured text signals with seller, category, and price.

Protocol:
- Split SEED = 20261144 (fit, tune, confirm with test-like seller distribution).
- Generate 4-fold GroupKFold OOF predictions on fit for:
  1) Baseline historical text (word 1-2 + char 3-5 on title/combos).
  2) Prefixed structured text (title 1-3w/3-6c + warranty + structured aid/name attributes + normalized units).
- Train CatBoost Single & Bag3:
  - Configuration A (Control): 101 deep features + seller_id + TE + historical text OOF.
  - Configuration B (Enriched): Control features + structured prefixed text OOF.
- Evaluate on tune (for early stopping and threshold selection) and confirm (frozen).
- Report:
  - Accuracy, log-loss, net delta vs Control.
  - Discrepancies, fixes, breaks, seen vs new sellers.
  - 3,000 bootstrap seller resamples for 95% CI.
"""

import os, sys, json, hashlib, zipfile, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import split_testlike
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261144
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "prefixed_catboost_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"(\d+)\s*(gb|mb|tb|ram|rom|ghz|mhz|mah|mp|px|cm|mm|kg|km|v|w)\b", r"\1_\2", text)
    text = re.sub(r"\b0\s*km\b", "cerokm", text)
    text = re.sub(r"[^a-z0-9_=\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_structured_attributes(attrs_list):
    if not attrs_list or not isinstance(attrs_list, list):
        return ""
    tokens = []
    for a in attrs_list:
        if not isinstance(a, dict):
            continue
        aid = normalize_text(a.get("id") or "")
        aname = normalize_text(a.get("name") or "")
        vname = normalize_text(a.get("value_name") or "")
        if aid and vname:
            tokens.append(f"aid_{aid}={vname.replace(' ', '_')}")
        if aname and vname:
            tokens.append(f"name_{aname.replace(' ', '_')}={vname.replace(' ', '_')}")
        if vname:
            tokens.append(f"val_{vname.replace(' ', '_')}")
    return " ".join(tokens)

def make_te(a, cat, y, fit, tune, confirm, sellers, m=10.0):
    g_mean = float(y[fit].mean())
    te_cols = ["category_id", "listing_type_id", "state"]
    gkf = GroupKFold(n_splits=4)
    for col in te_cols:
        col_name = f"te_{col}"
        a[col_name] = g_mean
        oof_vals = np.full(len(fit), g_mean, dtype=np.float32)
        for tr, va in gkf.split(fit, y[fit], sellers[fit]):
            s_tr = cat.iloc[fit[tr]][col]
            stats = pd.DataFrame({"cat": s_tr, "y": y[fit[tr]]}).groupby("cat")["y"].agg(["count", "mean"])
            smooth = (stats["count"] * stats["mean"] + m * g_mean) / (stats["count"] + m)
            oof_vals[va] = cat.iloc[fit[va]][col].map(smooth).fillna(g_mean).astype(np.float32)
        a.loc[fit, col_name] = oof_vals
        fit_stats = pd.DataFrame({"cat": cat.iloc[fit][col], "y": y[fit]}).groupby("cat")["y"].agg(["count", "mean"])
        fit_smooth = (fit_stats["count"] * fit_stats["mean"] + m * g_mean) / (fit_stats["count"] + m)
        a.loc[tune, col_name] = cat.iloc[tune][col].map(fit_smooth).fillna(g_mean).astype(np.float32)
        a.loc[confirm, col_name] = cat.iloc[confirm][col].map(fit_smooth).fillna(g_mean).astype(np.float32)

def generate_prefixed_oof(titles, warranties, attrs, y, fit, tune, confirm, sellers):
    print("Generating GroupKFold OOF predictions for Prefixed Structured Text Model...", flush=True)
    gkf = GroupKFold(n_splits=4)
    oof_fit = np.empty(len(fit), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(fit, y[fit], sellers[fit])):
        f_tr = fit[tr_idx]
        f_va = fit[va_idx]
        v_t_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=100000, sublinear_tf=True)
        v_t_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=4, max_features=120000, sublinear_tf=True)
        v_w_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
        v_a_s = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True)

        X_tr = sp.hstack([
            v_t_w.fit_transform([titles[i] for i in f_tr]),
            v_t_c.fit_transform([titles[i] for i in f_tr]),
            v_w_w.fit_transform([warranties[i] for i in f_tr]),
            v_a_s.fit_transform([attrs[i] for i in f_tr])
        ], format="csr")

        X_va = sp.hstack([
            v_t_w.transform([titles[i] for i in f_va]),
            v_t_c.transform([titles[i] for i in f_va]),
            v_w_w.transform([warranties[i] for i in f_va]),
            v_a_s.transform([attrs[i] for i in f_va])
        ], format="csr")

        lr = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=SEED)
        lr.fit(X_tr, y[f_tr])
        oof_fit[va_idx] = lr.predict_proba(X_va)[:, 1]

    # Full fit on entire fit set to score tune and confirm
    v_t_w_full = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=100000, sublinear_tf=True)
    v_t_c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=4, max_features=120000, sublinear_tf=True)
    v_w_w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    v_a_s_full = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True)

    X_full_fit = sp.hstack([
        v_t_w_full.fit_transform([titles[i] for i in fit]),
        v_t_c_full.fit_transform([titles[i] for i in fit]),
        v_w_w_full.fit_transform([warranties[i] for i in fit]),
        v_a_s_full.fit_transform([attrs[i] for i in fit])
    ], format="csr")

    X_tune = sp.hstack([
        v_t_w_full.transform([titles[i] for i in tune]),
        v_t_c_full.transform([titles[i] for i in tune]),
        v_w_w_full.transform([warranties[i] for i in tune]),
        v_a_s_full.transform([attrs[i] for i in tune])
    ], format="csr")

    X_conf = sp.hstack([
        v_t_w_full.transform([titles[i] for i in confirm]),
        v_t_c_full.transform([titles[i] for i in confirm]),
        v_w_w_full.transform([warranties[i] for i in confirm]),
        v_a_s_full.transform([attrs[i] for i in confirm])
    ], format="csr")

    lr_full = LogisticRegression(C=2.5, max_iter=200, solver="liblinear", random_state=SEED)
    lr_full.fit(X_full_fit, y[fit])

    p_tune = lr_full.predict_proba(X_tune)[:, 1]
    p_conf = lr_full.predict_proba(X_conf)[:, 1]

    return oof_fit, p_tune, p_conf

def generate_hist_text_oof(combos, titles, y, fit, tune, confirm, sellers):
    print("Generating GroupKFold OOF predictions for Historical Text Model...", flush=True)
    gkf = GroupKFold(n_splits=4)
    oof_fit = np.empty(len(fit), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(fit, y[fit], sellers[fit])):
        f_tr = fit[tr_idx]
        f_va = fit[va_idx]
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000, sublinear_tf=True)

        X_tr = sp.hstack([w.fit_transform([combos[i] for i in f_tr]), c.fit_transform([titles[i] for i in f_tr])], format="csr")
        X_va = sp.hstack([w.transform([combos[i] for i in f_va]), c.transform([titles[i] for i in f_va])], format="csr")

        lr = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=SEED)
        lr.fit(X_tr, y[f_tr])
        oof_fit[va_idx] = lr.predict_proba(X_va)[:, 1]

    w_full = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    c_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120000, sublinear_tf=True)

    X_full_fit = sp.hstack([w_full.fit_transform([combos[i] for i in fit]), c_full.fit_transform([titles[i] for i in fit])], format="csr")
    X_tune = sp.hstack([w_full.transform([combos[i] for i in tune]), c_full.transform([titles[i] for i in tune])], format="csr")
    X_conf = sp.hstack([w_full.transform([combos[i] for i in confirm]), c_full.transform([titles[i] for i in confirm])], format="csr")

    lr_full = LogisticRegression(C=3.0, max_iter=200, solver="liblinear", random_state=SEED)
    lr_full.fit(X_full_fit, y[fit])

    p_tune = lr_full.predict_proba(X_tune)[:, 1]
    p_conf = lr_full.predict_proba(X_conf)[:, 1]

    return oof_fit, p_tune, p_conf

def run_bootstrap_seller(y_true, pred_a, pred_b, sellers, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    diffs = []
    for _ in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        acc_a = (pred_a[idx] == y_true[idx]).sum()
        acc_b = (pred_b[idx] == y_true[idx]).sum()
        diffs.append(acc_b - acc_a)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def main():
    print(f"=== CATBOOST PREFIXED STRUCTURED TEXT OOF ABLATION (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)

    # 1. Extract raw and tabular features
    print("Extracting deep tabular features...", flush=True)
    a, cat, titles_raw, warranties_raw, combos_raw = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)

    # 2. Historical text OOF
    oof_hist, pt_hist, pc_hist = generate_hist_text_oof(combos_raw, titles_raw, y, fit, tune, confirm, sellers)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_hist, pt_hist, pc_hist)):
        a.loc[idx, "txt_sparse_p"] = pred

    # 3. Prefixed structured text OOF
    titles_norm = [normalize_text(r.get("title")) for r in rows]
    warranties_norm = [normalize_text(r.get("warranty")) for r in rows]
    attrs_norm = [extract_structured_attributes(r.get("attributes")) for r in rows]

    oof_pref, pt_pref, pc_pref = generate_prefixed_oof(titles_norm, warranties_norm, attrs_norm, y, fit, tune, confirm, sellers)
    a["p_prefixed_text"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof_pref, pt_pref, pc_pref)):
        a.loc[idx, "p_prefixed_text"] = pred

    a["seller_id_cat"] = cat.seller_id.astype(str)
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # Features sets:
    cols_control = [c for c in a.columns if c != "p_prefixed_text"]
    cols_enriched = list(a.columns)

    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    grid_th = np.arange(0.35, 0.651, 0.005)

    # --- Train Control Configuration (Historical text only) ---
    print("\n--- Training Control Models (Historical Text Only) ---", flush=True)
    tune_probs_ctrl = []
    conf_probs_ctrl = []
    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting Control CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit, cols_control], y[fit], cat_features=cats, eval_set=(a.loc[tune, cols_control], y[tune]), early_stopping_rounds=100, use_best_model=True)
        tune_probs_ctrl.append(cb.predict_proba(a.loc[tune, cols_control])[:, 1])
        conf_probs_ctrl.append(cb.predict_proba(a.loc[confirm, cols_control])[:, 1])

    p_tune_ctrl_bag3 = np.mean(tune_probs_ctrl, axis=0)
    p_conf_ctrl_bag3 = np.mean(conf_probs_ctrl, axis=0)
    th_ctrl = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_ctrl_bag3 >= t))

    # --- Train Enriched Configuration (Historical + Prefixed Structured Text) ---
    print("\n--- Training Enriched Models (Historical + Prefixed Structured Text OOF) ---", flush=True)
    tune_probs_enr = []
    conf_probs_enr = []
    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting Enriched CatBoost seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.loc[fit, cols_enriched], y[fit], cat_features=cats, eval_set=(a.loc[tune, cols_enriched], y[tune]), early_stopping_rounds=100, use_best_model=True)
        tune_probs_enr.append(cb.predict_proba(a.loc[tune, cols_enriched])[:, 1])
        conf_probs_enr.append(cb.predict_proba(a.loc[confirm, cols_enriched])[:, 1])

    p_tune_enr_bag3 = np.mean(tune_probs_enr, axis=0)
    p_conf_enr_bag3 = np.mean(conf_probs_enr, axis=0)
    th_enr = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_enr_bag3 >= t))

    # Frozen Evaluation on Confirm
    print("\n=== EVALUATING FROZEN ON CONFIRM ===", flush=True)
    pred_conf_ctrl = (p_conf_ctrl_bag3 >= th_ctrl).astype(int)
    pred_conf_enr = (p_conf_enr_bag3 >= th_enr).astype(int)

    acc_ctrl = float(accuracy_score(y[confirm], pred_conf_ctrl))
    ll_ctrl = float(log_loss(y[confirm], p_conf_ctrl_bag3))

    acc_enr = float(accuracy_score(y[confirm], pred_conf_enr))
    ll_enr = float(log_loss(y[confirm], p_conf_enr_bag3))

    hits_ctrl = int((pred_conf_ctrl == y[confirm]).sum())
    hits_enr = int((pred_conf_enr == y[confirm]).sum())
    delta_hits = hits_enr - hits_ctrl

    fixes = int(((pred_conf_ctrl != y[confirm]) & (pred_conf_enr == y[confirm])).sum())
    breaks = int(((pred_conf_ctrl == y[confirm]) & (pred_conf_enr != y[confirm])).sum())
    discrepancies = int((pred_conf_ctrl != pred_conf_enr).sum())

    seen_mask = np.isin(sellers[confirm], sellers[fit])
    net_seen = int(((pred_conf_enr[seen_mask] == y[confirm][seen_mask]).sum()) - ((pred_conf_ctrl[seen_mask] == y[confirm][seen_mask]).sum()))
    net_new = int(((pred_conf_enr[~seen_mask] == y[confirm][~seen_mask]).sum()) - ((pred_conf_ctrl[~seen_mask] == y[confirm][~seen_mask]).sum()))

    ci_low, ci_high = run_bootstrap_seller(y[confirm], pred_conf_ctrl, pred_conf_enr, sellers[confirm], 3000, SEED)

    print(f"Control Bag3 Confirm:  Acc={acc_ctrl:.5f} ({hits_ctrl}/{len(confirm)}), LL={ll_ctrl:.5f}, th={th_ctrl:.3f}")
    print(f"Enriched Bag3 Confirm: Acc={acc_enr:.5f} ({hits_enr}/{len(confirm)}), LL={ll_enr:.5f}, th={th_enr:.3f}")
    print(f"Delta: {delta_hits:+d} hits ({fixes} fixes, {breaks} breaks, {discrepancies} discrepancies)")
    print(f"Seen sellers delta: {net_seen:+d}, New sellers delta: {net_new:+d}")
    print(f"Bootstrap 95% CI (3,000 resamples): [{ci_low:+.1f}, {ci_high:+.1f}]")

    report = {
        "experiment": "catboost_prefixed_text_oof_ablation",
        "split_seed": SEED,
        "n_confirm": len(confirm),
        "acc_control": acc_ctrl,
        "ll_control": ll_ctrl,
        "th_control": float(th_ctrl),
        "acc_enriched": acc_enr,
        "ll_enriched": ll_enr,
        "th_enriched": float(th_enr),
        "delta_hits": delta_hits,
        "fixes": fixes,
        "breaks": breaks,
        "discrepancies": discrepancies,
        "net_seen": net_seen,
        "net_new": net_new,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high
    }

    out_file = OUT_DIR / f"report_prefixed_catboost_{SEED}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save paired CSV
    df_pairs = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "is_seen": seen_mask.astype(int),
        "y_true": y[confirm],
        "p_ctrl": p_conf_ctrl_bag3,
        "p_enr": p_conf_enr_bag3,
        "pred_ctrl": pred_conf_ctrl,
        "pred_enr": pred_conf_enr
    })
    pairs_file = OUT_DIR / f"paired_catboost_prefixed_{SEED}.csv"
    df_pairs.to_csv(pairs_file, index=False)

    print(f"Report saved to {out_file}")
    print(f"Pairs saved to {pairs_file}")

if __name__ == "__main__":
    main()
