"""Line A: High-Capacity Text Linear Model & Paired Confirmation vs Strong Baseline & Bag3.

Protocol:
1. Fresh seller-group split 20261143 (fit, tune, confirm).
2. High-capacity sparse text representation:
   - Word 1-3 grams, char 3-6 grams, normalized text (accents, units, numbers, cues).
   - Dedicated channels: title, warranty, attributes (prefixed attr_id=value_name, names, values).
   - Fit-only vocabulary and IDF.
3. Linear classifiers (LogisticRegression / SGDClassifier) tuned strictly on tune.
4. Paired baselines on identical split:
   - Baseline 1: Historical strong single model (CatBoost depth 6, lr 0.07, word+char OOF + seller).
   - Baseline 2: CatBoost bag3 (average of 3 seeds: 20261143, 20262140, 20263137).
5. Predeclared blend tuned strictly on tune.
6. Single frozen evaluation on confirm:
   - Accuracy, log-loss, net fixes, breaks, seen/new breakdown.
   - Error correlation, disagreement count.
   - 3,000 bootstrap seller resamples for 95% CI.
   - Pairs CSV, JSON report, and Markdown summary.
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

def text_oof(xf, xt, xv, y, fit, sellers, C=3.0):
    oof = np.empty(len(fit), dtype=np.float32)
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(fit, y[fit], sellers[fit]):
        lr = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=20261143)
        lr.fit(xf[tr], y[fit[tr]])
        oof[va] = lr.predict_proba(xf[va])[:, 1]
    lr_full = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=20261143)
    lr_full.fit(xf, y[fit])
    p_tune = lr_full.predict_proba(xt)[:, 1]
    p_conf = lr_full.predict_proba(xv)[:, 1]
    return oof, p_tune, p_conf

SEED = 20261143
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "line_a_text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    # Unicode normalize
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    # Normalize units & model numbers
    text = re.sub(r"(\d+)\s*(gb|mb|tb|ram|rom|ghz|mhz|mah|mp|px|cm|mm|kg|km|v|w)\b", r"\1 \2", text)
    text = re.sub(r"\b0\s*km\b", "cerokm", text)
    text = re.sub(r"[^a-z0-9_=\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_attribute_repr(attrs_list):
    if not attrs_list or not isinstance(attrs_list, list):
        return "", "", ""
    prefixed_tokens = []
    names = []
    values = []
    for a in attrs_list:
        aid = normalize_text(a.get("id") or "")
        aname = normalize_text(a.get("name") or "")
        vname = normalize_text(a.get("value_name") or "")
        if aid and vname:
            prefixed_tokens.append(f"attr_{aid}={vname.replace(' ', '_')}")
        elif aname and vname:
            prefixed_tokens.append(f"attr_{aname.replace(' ', '_')}={vname.replace(' ', '_')}")
        if aname:
            names.append(aname)
        if vname:
            values.append(vname)
    return " ".join(prefixed_tokens), " ".join(names), " ".join(values)

def main():
    print(f"=== LINE A: HIGH-CAPACITY TEXT LINEAR MODEL & PAIRED CONFIRMATION (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)
    print(f"Split testlike {SEED}: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

    # 1. Build text representations
    print("Normalizing and tokenizing multi-channel text...", flush=True)
    titles = [normalize_text(r.get("title")) for r in rows]
    warranties = [normalize_text(r.get("warranty")) for r in rows]
    
    attr_prefixed = []
    attr_names = []
    attr_values = []
    for r in rows:
        pref, nms, vals = build_attribute_repr(r.get("attributes"))
        attr_prefixed.append(pref)
        attr_names.append(nms)
        attr_values.append(vals)

    # 2. Extract High-Capacity Sparse Representation strictly on FIT
    print("Fitting text vectorizers strictly on fit...", flush=True)
    # Channel 1: Title word 1-3 grams + char 3-6 grams
    vec_title_word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=120000, sublinear_tf=True)
    vec_title_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=4, max_features=180000, sublinear_tf=True)

    # Channel 2: Warranty word 1-2 grams + char 3-5 grams
    vec_warr_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    vec_warr_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)

    # Channel 3: Attribute prefixed tokens and value strings
    vec_attr_pref = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    vec_attr_vals = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, sublinear_tf=True)

    # Transform fit
    print("Transforming fit sparse matrix...", flush=True)
    X_fit = sp.hstack([
        vec_title_word.fit_transform([titles[i] for i in fit]),
        vec_title_char.fit_transform([titles[i] for i in fit]),
        vec_warr_word.fit_transform([warranties[i] for i in fit]),
        vec_warr_char.fit_transform([warranties[i] for i in fit]),
        vec_attr_pref.fit_transform([attr_prefixed[i] for i in fit]),
        vec_attr_vals.fit_transform([attr_values[i] for i in fit])
    ], format="csr", dtype=np.float32)
    print(f"Fit sparse matrix shape: {X_fit.shape}", flush=True)

    # Transform tune
    print("Transforming tune sparse matrix...", flush=True)
    X_tune = sp.hstack([
        vec_title_word.transform([titles[i] for i in tune]),
        vec_title_char.transform([titles[i] for i in tune]),
        vec_warr_word.transform([warranties[i] for i in tune]),
        vec_warr_char.transform([warranties[i] for i in tune]),
        vec_attr_pref.transform([attr_prefixed[i] for i in tune]),
        vec_attr_vals.transform([attr_values[i] for i in tune])
    ], format="csr", dtype=np.float32)

    # Transform confirm
    print("Transforming confirm sparse matrix...", flush=True)
    X_conf = sp.hstack([
        vec_title_word.transform([titles[i] for i in confirm]),
        vec_title_char.transform([titles[i] for i in confirm]),
        vec_warr_word.transform([warranties[i] for i in confirm]),
        vec_warr_char.transform([warranties[i] for i in confirm]),
        vec_attr_pref.transform([attr_prefixed[i] for i in confirm]),
        vec_attr_vals.transform([attr_values[i] for i in confirm])
    ], format="csr", dtype=np.float32)

    # 3. Regularization Calibration strictly on TUNE
    print("Calibrating Line A regularization C strictly on tune...", flush=True)
    best_c = 1.0
    best_tune_ll = float("inf")
    best_lr = None

    for c_val in [0.3, 1.0, 2.5, 5.0]:
        lr = LogisticRegression(C=c_val, max_iter=250, solver="liblinear", random_state=SEED)
        lr.fit(X_fit, y[fit])
        p_tune = lr.predict_proba(X_tune)[:, 1]
        ll = log_loss(y[tune], p_tune)
        print(f"C={c_val}: tune log_loss={ll:.5f}", flush=True)
        if ll < best_tune_ll:
            best_tune_ll = ll
            best_c = c_val
            best_lr = lr

    print(f"Selected best C={best_c} (tune LL={best_tune_ll:.5f})", flush=True)
    p_tune_line_a = best_lr.predict_proba(X_tune)[:, 1]
    p_conf_line_a = best_lr.predict_proba(X_conf)[:, 1]

    # Grid search threshold for Line A on tune
    grid_th = np.arange(0.35, 0.651, 0.005)
    best_th_line_a = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_line_a >= t))
    acc_tune_line_a = accuracy_score(y[tune], p_tune_line_a >= best_th_line_a)
    print(f"Line A Tune: Acc={acc_tune_line_a:.5f}, th={best_th_line_a:.3f}", flush=True)

    # 4. Train Historical Single Model & CatBoost Bag3
    print("Extracting deep tabular + OOF text features for CatBoost baselines...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)

    # Standard historical text OOF
    w_hist = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
    c_hist = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150000, sublinear_tf=True)
    
    xf_hist = sp.hstack([w_hist.fit_transform([combos_raw[i] for i in fit]), c_hist.fit_transform([titles_raw[i] for i in fit])], format="csr")
    xt_hist = sp.hstack([w_hist.transform([combos_raw[i] for i in tune]), c_hist.transform([titles_raw[i] for i in tune])], format="csr")
    xv_hist = sp.hstack([w_hist.transform([combos_raw[i] for i in confirm]), c_hist.transform([titles_raw[i] for i in confirm])], format="csr")

    oof, pt_text, pc_text = text_oof(xf_hist, xt_hist, xv_hist, y, fit, sellers, 3.0)
    a["txt_sparse_p"] = np.nan
    for idx, pred in zip((fit, tune, confirm), (oof, pt_text, pc_text)):
        a.loc[idx, "txt_sparse_p"] = pred
    a["seller_id_cat"] = cat.seller_id.astype(str)
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # Fit 3 CatBoost models for bag3
    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    tune_probs_cb = []
    conf_probs_cb = []

    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting CatBoost baseline seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.iloc[fit], y[fit], cat_features=cats, eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100, use_best_model=True)
        tune_probs_cb.append(cb.predict_proba(a.iloc[tune])[:, 1])
        conf_probs_cb.append(cb.predict_proba(a.iloc[confirm])[:, 1])

    # Single model (historical strong baseline)
    p_tune_single = tune_probs_cb[0]
    p_conf_single = conf_probs_cb[0]
    th_single = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_single >= t))

    # Bag3 model
    p_tune_bag3 = np.mean(tune_probs_cb, axis=0)
    p_conf_bag3 = np.mean(conf_probs_cb, axis=0)
    th_bag3 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_bag3 >= t))

    print(f"CatBoost Single Tune: Acc={accuracy_score(y[tune], p_tune_single >= th_single):.5f}, th={th_single:.3f}", flush=True)
    print(f"CatBoost Bag3 Tune:   Acc={accuracy_score(y[tune], p_tune_bag3 >= th_bag3):.5f}, th={th_bag3:.3f}", flush=True)

    # 5. Predeclared Mixtures Tuned STRICTLY on TUNE
    print("Optimizing mixture Line A + Bag3 strictly on tune...", flush=True)
    best_w = 0.0
    best_th_blend = 0.5
    best_blend_tune_acc = 0.0

    blend_grid_records = []
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        p_blend_tune = w * p_tune_line_a + (1 - w) * p_tune_bag3
        th_w = max(grid_th, key=lambda t: accuracy_score(y[tune], p_blend_tune >= t))
        acc_w = accuracy_score(y[tune], p_blend_tune >= th_w)
        ll_w = log_loss(y[tune], p_blend_tune)
        blend_grid_records.append({"w_line_a": w, "tune_acc": float(acc_w), "tune_ll": float(ll_w), "th": float(th_w)})
        if acc_w > best_blend_tune_acc:
            best_blend_tune_acc = acc_w
            best_w = w
            best_th_blend = th_w

    print(f"Selected Blend on Tune: w_line_a={best_w:.2f}, w_bag3={1-best_w:.2f}, th={best_th_blend:.3f} (Tune Acc={best_blend_tune_acc:.5f})", flush=True)

    # 6. FROZEN SINGLE EVALUATION ON CONFIRM
    print("\n=== FROZEN SINGLE EVALUATION ON CONFIRM ===", flush=True)
    pred_conf_line_a = (p_conf_line_a >= best_th_line_a).astype(int)
    pred_conf_single = (p_conf_single >= th_single).astype(int)
    pred_conf_bag3 = (p_conf_bag3 >= th_bag3).astype(int)

    p_conf_blend = best_w * p_conf_line_a + (1 - best_w) * p_conf_bag3
    pred_conf_blend = (p_conf_blend >= best_th_blend).astype(int)

    # Confirm Accuracies & Log Losses
    acc_line_a = accuracy_score(y[confirm], pred_conf_line_a)
    ll_line_a = log_loss(y[confirm], p_conf_line_a)

    acc_single = accuracy_score(y[confirm], pred_conf_single)
    ll_single = log_loss(y[confirm], p_conf_single)

    acc_bag3 = accuracy_score(y[confirm], pred_conf_bag3)
    ll_bag3 = log_loss(y[confirm], p_conf_bag3)

    acc_blend = accuracy_score(y[confirm], pred_conf_blend)
    ll_blend = log_loss(y[confirm], p_conf_blend)

    seen_mask = np.isin(sellers[confirm], sellers[fit])

    # Error correlations
    err_line_a = (pred_conf_line_a != y[confirm]).astype(int)
    err_single = (pred_conf_single != y[confirm]).astype(int)
    err_bag3 = (pred_conf_bag3 != y[confirm]).astype(int)
    corr_line_a_bag3 = float(np.corrcoef(p_conf_line_a, p_conf_bag3)[0, 1])
    corr_err_line_a_bag3 = float(np.corrcoef(err_line_a, err_bag3)[0, 1])
    n_disagreements = int((pred_conf_line_a != pred_conf_bag3).sum())

    # Net fixes calculations
    def compute_paired(pred_cand, pred_base):
        net = int((pred_cand == y[confirm]).sum() - (pred_base == y[confirm]).sum())
        fixes = int(((pred_cand == y[confirm]) & (pred_base != y[confirm])).sum())
        breaks = int(((pred_cand != y[confirm]) & (pred_base == y[confirm])).sum())
        net_seen = int((pred_cand[seen_mask] == y[confirm][seen_mask]).sum() - (pred_base[seen_mask] == y[confirm][seen_mask]).sum())
        net_new = int((pred_cand[~seen_mask] == y[confirm][~seen_mask]).sum() - (pred_base[~seen_mask] == y[confirm][~seen_mask]).sum())
        return {"net": net, "fixes": fixes, "breaks": breaks, "net_seen": net_seen, "net_new": net_new}

    paired_line_a_vs_single = compute_paired(pred_conf_line_a, pred_conf_single)
    paired_line_a_vs_bag3 = compute_paired(pred_conf_line_a, pred_conf_bag3)
    paired_bag3_vs_single = compute_paired(pred_conf_bag3, pred_conf_single)
    paired_blend_vs_single = compute_paired(pred_conf_blend, pred_conf_single)
    paired_blend_vs_bag3 = compute_paired(pred_conf_blend, pred_conf_bag3)

    # 3,000 bootstrap seller resamples for blend vs bag3 and blend vs single
    print("Computing 3,000 bootstrap seller resamples for 95% CI...", flush=True)
    unique_sellers = np.unique(sellers[confirm])
    rng = np.random.default_rng(SEED)
    seller_to_indices = {s: np.where(sellers[confirm] == s)[0] for s in unique_sellers}

    boot_diffs_blend_vs_bag3 = []
    boot_diffs_blend_vs_single = []
    boot_diffs_bag3_vs_single = []

    for _ in range(3000):
        sampled_s = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
        sampled_idx = np.concatenate([seller_to_indices[s] for s in sampled_s])
        
        diff_blend_bag3 = (pred_conf_blend[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_bag3[sampled_idx] == y[confirm][sampled_idx]).sum()
        diff_blend_single = (pred_conf_blend[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_single[sampled_idx] == y[confirm][sampled_idx]).sum()
        diff_bag3_single = (pred_conf_bag3[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_single[sampled_idx] == y[confirm][sampled_idx]).sum()

        boot_diffs_blend_vs_bag3.append(diff_blend_bag3)
        boot_diffs_blend_vs_single.append(diff_blend_single)
        boot_diffs_bag3_vs_single.append(diff_bag3_single)

    ci95_blend_vs_bag3 = [float(np.percentile(boot_diffs_blend_vs_bag3, 2.5)), float(np.percentile(boot_diffs_blend_vs_bag3, 97.5))]
    ci95_blend_vs_single = [float(np.percentile(boot_diffs_blend_vs_single, 2.5)), float(np.percentile(boot_diffs_blend_vs_single, 97.5))]
    ci95_bag3_vs_single = [float(np.percentile(boot_diffs_bag3_vs_single, 2.5)), float(np.percentile(boot_diffs_bag3_vs_single, 97.5))]

    # 7. Persist Pairs and Report
    pairs_df = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "is_seen": seen_mask.astype(int),
        "y_true": y[confirm],
        "p_line_a": p_conf_line_a,
        "p_single": p_conf_single,
        "p_bag3": p_conf_bag3,
        "p_blend": p_conf_blend,
        "pred_line_a": pred_conf_line_a,
        "pred_single": pred_conf_single,
        "pred_bag3": pred_conf_bag3,
        "pred_blend": pred_conf_blend
    })
    pairs_path = OUT_DIR / f"paired_{SEED}.csv"
    pairs_df.to_csv(pairs_path, index=False)

    report = {
        "experiment": "line_a_high_capacity_text_linear_model",
        "split_seed": SEED,
        "confirm_rows": len(confirm),
        "seen_fraction": float(seen_mask.mean()),
        "sparse_vocabulary_dim": int(X_fit.shape[1]),
        "line_a_standalone": {
            "best_c": float(best_c),
            "threshold": float(best_th_line_a),
            "tune_acc": float(acc_tune_line_a),
            "confirm_acc": float(acc_line_a),
            "confirm_ll": float(ll_line_a),
            "paired_vs_single": paired_line_a_vs_single,
            "paired_vs_bag3": paired_line_a_vs_bag3
        },
        "baseline_single": {
            "threshold": float(th_single),
            "confirm_acc": float(acc_single),
            "confirm_ll": float(ll_single)
        },
        "baseline_bag3": {
            "threshold": float(th_bag3),
            "confirm_acc": float(acc_bag3),
            "confirm_ll": float(ll_bag3),
            "paired_vs_single": paired_bag3_vs_single,
            "bootstrap_ci95_vs_single": ci95_bag3_vs_single
        },
        "predeclared_blend": {
            "w_line_a": float(best_w),
            "w_bag3": float(1 - best_w),
            "threshold": float(best_th_blend),
            "tune_acc": float(best_blend_tune_acc),
            "confirm_acc": float(acc_blend),
            "confirm_ll": float(ll_blend),
            "paired_vs_single": paired_blend_vs_single,
            "paired_vs_bag3": paired_blend_vs_bag3,
            "bootstrap_ci95_vs_bag3": ci95_blend_vs_bag3,
            "bootstrap_ci95_vs_single": ci95_blend_vs_single
        },
        "diversity_diagnostics": {
            "prob_correlation_line_a_bag3": corr_line_a_bag3,
            "error_correlation_line_a_bag3": corr_err_line_a_bag3,
            "disagreements_count": n_disagreements,
            "disagreements_rate": float(n_disagreements / len(confirm))
        },
        "tune_grid_records": blend_grid_records
    }

    report_json_path = OUT_DIR / f"report_line_a_text_{SEED}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown Summary
    md_lines = [
        f"# Reporte Línea A: Modelo Lineal Textual de Alta Capacidad (Corte {SEED})",
        "",
        f"- **Dimensión del vocabulario sparse fit-only**: `{X_fit.shape[1]}` features",
        f"- **Observaciones confirm**: `{len(confirm)}` (fracción vendedores vistos: `{seen_mask.mean()*100:.1f}%`)",
        f"- **Discrepancias entre Línea A y CatBoost Bag3**: `{n_disagreements}` ({n_disagreements/len(confirm)*100:.2f}%)",
        f"- **Correlación de predicciones Línea A vs Bag3**: `{corr_line_a_bag3:.4f}`",
        f"- **Correlación de errores Línea A vs Bag3**: `{corr_err_line_a_bag3:.4f}`",
        "",
        "## Resumen de Rendimiento Congelado en Confirm",
        "",
        "| Modelo / Configuración | Confirm Accuracy | Confirm Log Loss | Umbral Tune | Delta vs Single Histórico | Delta vs Bag3 | Bootstrap 95% CI (3.000 réplicas) |",
        "|---|---|---|---|---|---|---|",
        f"| **CatBoost Single (Histórico)** | {acc_single:.5f} | {ll_single:.5f} | {th_single:.3f} | Control (0) | {-paired_bag3_vs_single['net']} | — |",
        f"| **CatBoost Bag3** | {acc_bag3:.5f} | {ll_bag3:.5f} | {th_bag3:.3f} | **+{paired_bag3_vs_single['net']}** | Control (0) | `[{ci95_bag3_vs_single[0]:.1f}, {ci95_bag3_vs_single[1]:.1f}]` |",
        f"| **Línea A Standalone** | {acc_line_a:.5f} | {ll_line_a:.5f} | {best_th_line_a:.3f} | **{paired_line_a_vs_single['net']:+d}** | **{paired_line_a_vs_bag3['net']:+d}** | — |",
        f"| **Blend Predeclarado ({best_w:.2f} A + {1-best_w:.2f} Bag3)** | **{acc_blend:.5f}** | **{ll_blend:.5f}** | {best_th_blend:.3f} | **{paired_blend_vs_single['net']:+d}** | **{paired_blend_vs_bag3['net']:+d}** | `[{ci95_blend_vs_bag3[0]:.1f}, {ci95_blend_vs_bag3[1]:.1f}]` vs Bag3 |",
        "",
        "## Desglose Pareado por Tipo de Vendedor (Visto vs Nuevo)",
        "",
        "| Comparación | Fixes | Breaks | Net Total | Net Vendedores Vistos | Net Vendedores Nuevos |",
        "|---|---|---|---|---|---|",
        f"| **Bag3 vs Single** | {paired_bag3_vs_single['fixes']} | {paired_bag3_vs_single['breaks']} | **{paired_bag3_vs_single['net']:+d}** | {paired_bag3_vs_single['net_seen']:+d} | {paired_bag3_vs_single['net_new']:+d} |",
        f"| **Línea A Standalone vs Single** | {paired_line_a_vs_single['fixes']} | {paired_line_a_vs_single['breaks']} | **{paired_line_a_vs_single['net']:+d}** | {paired_line_a_vs_single['net_seen']:+d} | {paired_line_a_vs_single['net_new']:+d} |",
        f"| **Línea A Standalone vs Bag3** | {paired_line_a_vs_bag3['fixes']} | {paired_line_a_vs_bag3['breaks']} | **{paired_line_a_vs_bag3['net']:+d}** | {paired_line_a_vs_bag3['net_seen']:+d} | {paired_line_a_vs_bag3['net_new']:+d} |",
        f"| **Blend ({best_w:.2f} A + {1-best_w:.2f} Bag3) vs Bag3** | {paired_blend_vs_bag3['fixes']} | {paired_blend_vs_bag3['breaks']} | **{paired_blend_vs_bag3['net']:+d}** | {paired_blend_vs_bag3['net_seen']:+d} | {paired_blend_vs_bag3['net_new']:+d} |",
        "",
        "## Archivos y Trazabilidad",
        "",
        f"- Reporte JSON: `{report_json_path.relative_to(ROOT)}`",
        f"- Pares por ID: `{pairs_path.relative_to(ROOT)}`",
        f"- Hashes SHA-256 de pares: `{hashlib.sha256(open(pairs_path, 'rb').read()).hexdigest()}`",
        ""
    ]

    report_md_path = OUT_DIR / f"report_line_a_text_{SEED}.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nExecution finished! Report saved to {report_md_path}", flush=True)

if __name__ == "__main__":
    main()
