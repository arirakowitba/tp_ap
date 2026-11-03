"""Predeclared Blend: CatBoost Bag3 + Independent NB-SVM on Fresh Seller-Group Split.

Protocol:
1. Fresh seller-group split 20261144 (fit, tune, confirm).
2. CatBoost Single & Bag3:
   - Deep tabular features (101 columns).
   - Fit-only smoothed target encodings (category_id, listing_type_id, state).
   - Historical word+char TF-IDF OOF text probability (txt_sparse_p).
   - 3 CatBoost models (iterations 900, depth 6, lr 0.07, l2 6.0) early-stopped on tune.
3. Independent NB-SVM:
   - Binary word 1-3 + char 3-5 n-grams fitted strictly on fit.
   - Naive Bayes log-count ratio r computed strictly on fit.
   - Regularization C and threshold tuned strictly on tune.
4. Predeclared Blend:
   - Weight alpha and threshold chosen exclusively on tune.
   - Frozen single evaluation on confirm.
5. Metrics:
   - Accuracy, log-loss, net fixes, breaks, seen/new.
   - Discrepancies vs Bag3 and Single.
   - 3,000 bootstrap seller resamples for 95% CI.
   - Pairs CSV with item_id, probabilities, and hard labels.
"""
import os, sys, json, hashlib, zipfile, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
OUT_DIR = ROOT / "results" / "nbsvm_bag3_blend"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"(\d+)\s*(gb|mb|tb|ram|rom|ghz|mhz|mah|mp|px|cm|mm|kg|km|v|w)\b", r"\1 \2", text)
    text = re.sub(r"\b0\s*km\b", "cerokm", text)
    text = re.sub(r"[^a-z0-9_=\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
        lr = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=SEED)
        lr.fit(xf[tr], y[fit[tr]])
        oof[va] = lr.predict_proba(xf[va])[:, 1]
    lr_full = LogisticRegression(C=C, max_iter=200, solver="liblinear", random_state=SEED)
    lr_full.fit(xf, y[fit])
    p_tune = lr_full.predict_proba(xt)[:, 1]
    p_conf = lr_full.predict_proba(xv)[:, 1]
    return oof, p_tune, p_conf

def compute_nb_ratio(X, y, alpha=1.0):
    p = np.asarray(X[y == 1].sum(axis=0)).flatten() + alpha
    q = np.asarray(X[y == 0].sum(axis=0)).flatten() + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.log(p / q)

def main():
    print(f"=== EXPERIMENT: CATBOOST BAG3 + INDEPENDENT NB-SVM (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)
    print(f"Split testlike {SEED}: fit={len(fit)}, tune={len(tune)}, confirm={len(confirm)}", flush=True)

    # 1. Train Independent NB-SVM
    print("Training Independent NB-SVM...", flush=True)
    raw_texts = [
        normalize_text(r.get("title")) + " " +
        normalize_text(r.get("warranty")) + " " +
        " ".join(normalize_text(v.get("value_name") or "") for v in r.get("attributes") or [])
        for r in rows
    ]

    c_word = CountVectorizer(ngram_range=(1, 3), min_df=3, max_features=100000, binary=True)
    c_char = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=4, max_features=120000, binary=True)

    X_fit_nb = sp.hstack([
        c_word.fit_transform([raw_texts[i] for i in fit]),
        c_char.fit_transform([raw_texts[i] for i in fit])
    ], format="csr", dtype=np.float32)

    X_tune_nb = sp.hstack([
        c_word.transform([raw_texts[i] for i in tune]),
        c_char.transform([raw_texts[i] for i in tune])
    ], format="csr", dtype=np.float32)

    X_conf_nb = sp.hstack([
        c_word.transform([raw_texts[i] for i in confirm]),
        c_char.transform([raw_texts[i] for i in confirm])
    ], format="csr", dtype=np.float32)

    r_nb = compute_nb_ratio(X_fit_nb, y[fit], alpha=1.0)
    r_diag = sp.diags(r_nb)
    X_fit_scaled = X_fit_nb.dot(r_diag)
    X_tune_scaled = X_tune_nb.dot(r_diag)
    X_conf_scaled = X_conf_nb.dot(r_diag)

    # Tune C strictly on tune
    best_c_nb = 0.5
    best_ll_nb = float("inf")
    best_model_nb = None
    for c_val in [0.2, 0.5, 1.0, 2.0]:
        lr_nb = LogisticRegression(C=c_val, max_iter=250, solver="liblinear", random_state=SEED)
        lr_nb.fit(X_fit_scaled, y[fit])
        p_tu = lr_nb.predict_proba(X_tune_scaled)[:, 1]
        ll_tu = log_loss(y[tune], p_tu)
        if ll_tu < best_ll_nb:
            best_ll_nb = ll_tu
            best_c_nb = c_val
            best_model_nb = lr_nb

    p_tune_nbsvm = best_model_nb.predict_proba(X_tune_scaled)[:, 1]
    p_conf_nbsvm = best_model_nb.predict_proba(X_conf_scaled)[:, 1]

    grid_th = np.arange(0.35, 0.651, 0.005)
    best_th_nbsvm = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_nbsvm >= t))
    print(f"NB-SVM Tune: Acc={accuracy_score(y[tune], p_tune_nbsvm >= best_th_nbsvm):.5f}, C={best_c_nb}, th={best_th_nbsvm:.3f}", flush=True)

    # 2. Train CatBoost Single & Bag3
    print("Extracting deep tabular + OOF text features for CatBoost baselines...", flush=True)
    a, cat, titles_raw, _, combos_raw = extract_deep_features(rows)
    make_te(a, cat, y, fit, tune, confirm, sellers)

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

    # 3 seeds for bag3
    cb_seeds = [SEED, SEED + 997, SEED + 1994]
    tune_probs_cb = []
    conf_probs_cb = []

    for s_idx, s in enumerate(cb_seeds):
        print(f"Fitting CatBoost baseline seed={s} ({s_idx+1}/3)...", flush=True)
        cb = CatBoostClassifier(iterations=900, depth=6, learning_rate=0.07, l2_leaf_reg=6.0, random_seed=s, thread_count=14, verbose=False, loss_function="Logloss")
        cb.fit(a.iloc[fit], y[fit], cat_features=cats, eval_set=(a.iloc[tune], y[tune]), early_stopping_rounds=100, use_best_model=True)
        tune_probs_cb.append(cb.predict_proba(a.iloc[tune])[:, 1])
        conf_probs_cb.append(cb.predict_proba(a.iloc[confirm])[:, 1])

    # Single
    p_tune_single = tune_probs_cb[0]
    p_conf_single = conf_probs_cb[0]
    th_single = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_single >= t))

    # Bag3
    p_tune_bag3 = np.mean(tune_probs_cb, axis=0)
    p_conf_bag3 = np.mean(conf_probs_cb, axis=0)
    th_bag3 = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_bag3 >= t))

    print(f"CatBoost Single Tune: Acc={accuracy_score(y[tune], p_tune_single >= th_single):.5f}, th={th_single:.3f}", flush=True)
    print(f"CatBoost Bag3 Tune:   Acc={accuracy_score(y[tune], p_tune_bag3 >= th_bag3):.5f}, th={th_bag3:.3f}", flush=True)

    # 3. Predeclared Blend Optimization Strictly on TUNE
    print("Optimizing blend weight alpha and threshold strictly on tune...", flush=True)
    best_alpha = 0.0
    best_th_blend = 0.5
    best_blend_acc_tune = 0.0

    blend_records = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        p_b_tune = alpha * p_tune_nbsvm + (1.0 - alpha) * p_tune_bag3
        th_a = max(grid_th, key=lambda t: accuracy_score(y[tune], p_b_tune >= t))
        acc_a = accuracy_score(y[tune], p_b_tune >= th_a)
        ll_a = log_loss(y[tune], p_b_tune)
        blend_records.append({"alpha": alpha, "tune_acc": float(acc_a), "tune_ll": float(ll_a), "th": float(th_a)})
        if acc_a > best_blend_acc_tune:
            best_blend_acc_tune = acc_a
            best_alpha = alpha
            best_th_blend = th_a

    print(f"Selected Blend on Tune: alpha_nbsvm={best_alpha:.2f}, w_bag3={1.0-best_alpha:.2f}, th={best_th_blend:.3f} (Tune Acc={best_blend_acc_tune:.5f})", flush=True)

    # 4. FROZEN Single Evaluation on CONFIRM
    print("\n=== FROZEN EVALUATION ON CONFIRM ===", flush=True)
    pred_conf_single = (p_conf_single >= th_single).astype(int)
    pred_conf_bag3 = (p_conf_bag3 >= th_bag3).astype(int)
    pred_conf_nbsvm = (p_conf_nbsvm >= best_th_nbsvm).astype(int)

    p_conf_blend = best_alpha * p_conf_nbsvm + (1.0 - best_alpha) * p_conf_bag3
    pred_conf_blend = (p_conf_blend >= best_th_blend).astype(int)

    acc_single = float(accuracy_score(y[confirm], pred_conf_single))
    ll_single = float(log_loss(y[confirm], p_conf_single))

    acc_bag3 = float(accuracy_score(y[confirm], pred_conf_bag3))
    ll_bag3 = float(log_loss(y[confirm], p_conf_bag3))

    acc_nbsvm = float(accuracy_score(y[confirm], pred_conf_nbsvm))
    ll_nbsvm = float(log_loss(y[confirm], p_conf_nbsvm))

    acc_blend = float(accuracy_score(y[confirm], pred_conf_blend))
    ll_blend = float(log_loss(y[confirm], p_conf_blend))

    seen_mask = np.isin(sellers[confirm], sellers[fit])

    # Discrepancy counts
    disagr_blend_vs_bag3 = int((pred_conf_blend != pred_conf_bag3).sum())
    disagr_blend_vs_single = int((pred_conf_blend != pred_conf_single).sum())
    disagr_bag3_vs_single = int((pred_conf_bag3 != pred_conf_single).sum())
    disagr_nbsvm_vs_bag3 = int((pred_conf_nbsvm != pred_conf_bag3).sum())

    def compute_paired(pred_cand, pred_base):
        net = int((pred_cand == y[confirm]).sum() - (pred_base == y[confirm]).sum())
        fixes = int(((pred_cand == y[confirm]) & (pred_base != y[confirm])).sum())
        breaks = int(((pred_cand != y[confirm]) & (pred_base == y[confirm])).sum())
        net_seen = int((pred_cand[seen_mask] == y[confirm][seen_mask]).sum() - (pred_base[seen_mask] == y[confirm][seen_mask]).sum())
        net_new = int((pred_cand[~seen_mask] == y[confirm][~seen_mask]).sum() - (pred_base[~seen_mask] == y[confirm][~seen_mask]).sum())
        return {"net": net, "fixes": fixes, "breaks": breaks, "net_seen": net_seen, "net_new": net_new}

    paired_blend_vs_bag3 = compute_paired(pred_conf_blend, pred_conf_bag3)
    paired_blend_vs_single = compute_paired(pred_conf_blend, pred_conf_single)
    paired_bag3_vs_single = compute_paired(pred_conf_bag3, pred_conf_single)
    paired_nbsvm_vs_bag3 = compute_paired(pred_conf_nbsvm, pred_conf_bag3)
    paired_nbsvm_vs_single = compute_paired(pred_conf_nbsvm, pred_conf_single)

    # 3,000 bootstrap seller resamples
    print("Computing 3,000 bootstrap seller resamples...", flush=True)
    unique_sellers = np.unique(sellers[confirm])
    rng = np.random.default_rng(SEED)
    seller_to_indices = {s: np.where(sellers[confirm] == s)[0] for s in unique_sellers}

    boot_diffs_blend_vs_bag3 = []
    boot_diffs_bag3_vs_single = []

    for _ in range(3000):
        sampled_s = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
        sampled_idx = np.concatenate([seller_to_indices[s] for s in sampled_s])
        diff_blend_bag = (pred_conf_blend[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_bag3[sampled_idx] == y[confirm][sampled_idx]).sum()
        diff_bag_single = (pred_conf_bag3[sampled_idx] == y[confirm][sampled_idx]).sum() - (pred_conf_single[sampled_idx] == y[confirm][sampled_idx]).sum()
        boot_diffs_blend_vs_bag3.append(diff_blend_bag)
        boot_diffs_bag3_vs_single.append(diff_bag_single)

    ci95_blend_vs_bag3 = [float(np.percentile(boot_diffs_blend_vs_bag3, 2.5)), float(np.percentile(boot_diffs_blend_vs_bag3, 97.5))]
    ci95_bag3_vs_single = [float(np.percentile(boot_diffs_bag3_vs_single, 2.5)), float(np.percentile(boot_diffs_bag3_vs_single, 97.5))]

    # 5. Persist Pairs and Report
    pairs_df = pd.DataFrame({
        "item_id": ids[confirm],
        "seller_id": sellers[confirm],
        "is_seen": seen_mask.astype(int),
        "y_true": y[confirm],
        "p_single": p_conf_single,
        "p_bag3": p_conf_bag3,
        "p_nbsvm": p_conf_nbsvm,
        "p_blend": p_conf_blend,
        "pred_single": pred_conf_single,
        "pred_bag3": pred_conf_bag3,
        "pred_nbsvm": pred_conf_nbsvm,
        "pred_blend": pred_conf_blend
    })
    pairs_path = OUT_DIR / f"paired_{SEED}.csv"
    pairs_df.to_csv(pairs_path, index=False)
    pairs_hash = hashlib.sha256(open(pairs_path, "rb").read()).hexdigest()

    report = {
        "experiment": "nbsvm_bag3_predeclared_blend",
        "split_seed": SEED,
        "confirm_rows": len(confirm),
        "seen_fraction": float(seen_mask.mean()),
        "baseline_single": {
            "threshold": float(th_single),
            "confirm_acc": acc_single,
            "confirm_ll": ll_single
        },
        "baseline_bag3": {
            "threshold": float(th_bag3),
            "confirm_acc": acc_bag3,
            "confirm_ll": ll_bag3,
            "paired_vs_single": paired_bag3_vs_single,
            "bootstrap_ci95_vs_single": ci95_bag3_vs_single
        },
        "nbsvm_standalone": {
            "best_c": float(best_c_nb),
            "threshold": float(best_th_nbsvm),
            "confirm_acc": acc_nbsvm,
            "confirm_ll": ll_nbsvm,
            "paired_vs_bag3": paired_nbsvm_vs_bag3,
            "paired_vs_single": paired_nbsvm_vs_single
        },
        "predeclared_blend": {
            "alpha_nbsvm": float(best_alpha),
            "w_bag3": float(1.0 - best_alpha),
            "threshold": float(best_th_blend),
            "tune_acc": float(best_blend_acc_tune),
            "confirm_acc": acc_blend,
            "confirm_ll": ll_blend,
            "paired_vs_bag3": paired_blend_vs_bag3,
            "paired_vs_single": paired_blend_vs_single,
            "bootstrap_ci95_vs_bag3": ci95_blend_vs_bag3,
            "discrepancies_vs_bag3": disagr_blend_vs_bag3,
            "discrepancies_vs_single": disagr_blend_vs_single
        },
        "pairs_file": str(pairs_path.relative_to(ROOT)),
        "pairs_sha256": pairs_hash,
        "tune_grid_records": blend_records
    }

    report_json_path = OUT_DIR / f"report_nbsvm_bag3_blend_{SEED}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        f"# Reporte: Mezcla Predeclarada CatBoost Bag3 + NB-SVM (Corte Fresco {SEED})",
        "",
        f"- **Observaciones confirm**: `{len(confirm)}` (fracción vendedores vistos: `{seen_mask.mean()*100:.1f}%`)",
        f"- **Regla congelada en tune**: alpha_nbsvm = `{best_alpha:.2f}`, w_bag3 = `{1.0-best_alpha:.2f}`, umbral = `{best_th_blend:.3f}`",
        f"- **Discrepancias entre Blend y Bag3**: `{disagr_blend_vs_bag3}` ({disagr_blend_vs_bag3/len(confirm)*100:.2f}%)",
        "",
        "## Rendimiento Pareado en Confirm",
        "",
        "| Modelo / Configuración | Confirm Accuracy | Confirm Log Loss | Umbral Tune | Delta vs Single Histórico | Delta vs Bag3 | Bootstrap 95% CI (3.000 réplicas) |",
        "|---|---|---|---|---|---|---|",
        f"| **CatBoost Single (Histórico)** | {acc_single:.5f} | {ll_single:.5f} | {th_single:.3f} | Control (0) | {-paired_bag3_vs_single['net']} | — |",
        f"| **CatBoost Bag3** | {acc_bag3:.5f} | {ll_bag3:.5f} | {th_bag3:.3f} | **{paired_bag3_vs_single['net']:+d}** | Control (0) | `[{ci95_bag3_vs_single[0]:.1f}, {ci95_bag3_vs_single[1]:.1f}]` vs Single |",
        f"| **NB-SVM Standalone** | {acc_nbsvm:.5f} | {ll_nbsvm:.5f} | {best_th_nbsvm:.3f} | **{paired_nbsvm_vs_single['net']:+d}** | **{paired_nbsvm_vs_bag3['net']:+d}** | — |",
        f"| **Blend Predeclarado (alpha={best_alpha:.2f})** | **{acc_blend:.5f}** | **{ll_blend:.5f}** | {best_th_blend:.3f} | **{paired_blend_vs_single['net']:+d}** | **{paired_blend_vs_bag3['net']:+d}** | `[{ci95_blend_vs_bag3[0]:.1f}, {ci95_blend_vs_bag3[1]:.1f}]` vs Bag3 |",
        "",
        "## Desglose Pareado por Vendedor (Visto vs Nuevo)",
        "",
        "| Comparación | Fixes | Breaks | Net Total | Net Vendedores Vistos | Net Vendedores Nuevos |",
        "|---|---|---|---|---|---|",
        f"| **Bag3 vs Single** | {paired_bag3_vs_single['fixes']} | {paired_bag3_vs_single['breaks']} | **{paired_bag3_vs_single['net']:+d}** | {paired_bag3_vs_single['net_seen']:+d} | {paired_bag3_vs_single['net_new']:+d} |",
        f"| **NB-SVM Standalone vs Bag3** | {paired_nbsvm_vs_bag3['fixes']} | {paired_nbsvm_vs_bag3['breaks']} | **{paired_nbsvm_vs_bag3['net']:+d}** | {paired_nbsvm_vs_bag3['net_seen']:+d} | {paired_nbsvm_vs_bag3['net_new']:+d} |",
        f"| **Blend Predeclarado vs Bag3** | {paired_blend_vs_bag3['fixes']} | {paired_blend_vs_bag3['breaks']} | **{paired_blend_vs_bag3['net']:+d}** | {paired_blend_vs_bag3['net_seen']:+d} | {paired_blend_vs_bag3['net_new']:+d} |",
        "",
        "## Archivos y Trazabilidad",
        "",
        f"- Pares por ID: `{pairs_path.relative_to(ROOT)}`",
        f"- Hash SHA-256 de pares: `{pairs_hash}`",
        f"- Reporte JSON: `{report_json_path.relative_to(ROOT)}`",
        ""
    ]

    report_md_path = OUT_DIR / f"report_nbsvm_bag3_blend_{SEED}.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nExecution finished! Report saved to {report_md_path}", flush=True)

if __name__ == "__main__":
    main()
