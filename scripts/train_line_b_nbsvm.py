"""Line B: NB-SVM & Discriminative Log-Count Ratio Text Model vs Strong Baseline & Bag3.

Implementation of Wang & Manning (2012) NB-SVM:
1. Multi-channel text representations (title, warranty, attributes).
2. Count / binary word (1-3) and char (3-6) n-grams fitted strictly on fit.
3. Compute Naive Bayes log-count ratio r = log( (p / |p|_1) / (q / |q|_1) ) on fit.
4. Feature scaling: X_nbsvm = X * r.
5. Calibrate regularization C and decision threshold strictly on tune.
6. Evaluate standalone NB-SVM and predeclared blend with CatBoost Bag3 on fresh confirm 20261143.
7. Paired comparisons vs Historical strong single model, Bag3, and Line A.
"""
import os, sys, json, hashlib, zipfile, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import split_testlike

SEED = 20261143
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "line_b_nbsvm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LINE_A_DIR = ROOT / "results" / "line_a_text"

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

def build_attribute_repr(attrs_list):
    if not attrs_list or not isinstance(attrs_list, list):
        return "", "", ""
    prefixed_tokens, names, values = [], [], []
    for a in attrs_list:
        aid = normalize_text(a.get("id") or "")
        aname = normalize_text(a.get("name") or "")
        vname = normalize_text(a.get("value_name") or "")
        if aid and vname:
            prefixed_tokens.append(f"attr_{aid}={vname.replace(' ', '_')}")
        elif aname and vname:
            prefixed_tokens.append(f"attr_{aname.replace(' ', '_')}={vname.replace(' ', '_')}")
        if aname: names.append(aname)
        if vname: values.append(vname)
    return " ".join(prefixed_tokens), " ".join(names), " ".join(values)

def compute_nb_ratio(X, y, alpha=1.0):
    # X is csr_matrix, y is binary 0/1 array
    p = np.asarray(X[y == 1].sum(axis=0)).flatten() + alpha
    q = np.asarray(X[y == 0].sum(axis=0)).flatten() + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    r = np.log(p / q)
    return r

def main():
    print(f"=== LINE B: NB-SVM & DISCRIMINATIVE LOG-COUNT RATIO (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)

    titles = [normalize_text(r.get("title")) for r in rows]
    warranties = [normalize_text(r.get("warranty")) for r in rows]
    attr_prefixed, attr_names, attr_values = [], [], []
    for r in rows:
        pref, nms, vals = build_attribute_repr(r.get("attributes"))
        attr_prefixed.append(pref); attr_names.append(nms); attr_values.append(vals)

    # Multi-channel CountVectorizer strictly on FIT
    print("Fitting count vectorizers strictly on fit...", flush=True)
    c_title_w = CountVectorizer(ngram_range=(1, 3), min_df=2, max_features=100000, binary=True)
    c_title_c = CountVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=4, max_features=150000, binary=True)
    c_warr_w = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000, binary=True)
    c_attr_p = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000, binary=True)

    X_fit = sp.hstack([
        c_title_w.fit_transform([titles[i] for i in fit]),
        c_title_c.fit_transform([titles[i] for i in fit]),
        c_warr_w.fit_transform([warranties[i] for i in fit]),
        c_attr_p.fit_transform([attr_prefixed[i] for i in fit])
    ], format="csr", dtype=np.float32)

    X_tune = sp.hstack([
        c_title_w.transform([titles[i] for i in tune]),
        c_title_c.transform([titles[i] for i in tune]),
        c_warr_w.transform([warranties[i] for i in tune]),
        c_attr_p.transform([attr_prefixed[i] for i in tune])
    ], format="csr", dtype=np.float32)

    X_conf = sp.hstack([
        c_title_w.transform([titles[i] for i in confirm]),
        c_title_c.transform([titles[i] for i in confirm]),
        c_warr_w.transform([warranties[i] for i in confirm]),
        c_attr_p.transform([attr_prefixed[i] for i in confirm])
    ], format="csr", dtype=np.float32)

    print(f"Count matrix shape on fit: {X_fit.shape}", flush=True)

    # Compute Naive Bayes log-count ratio strictly on fit
    print("Computing Naive Bayes log-count ratio r strictly on fit...", flush=True)
    r = compute_nb_ratio(X_fit, y[fit], alpha=1.0)
    
    # Scale features: X_tilde = X * diag(r)
    r_diag = sp.diags(r)
    X_fit_nb = X_fit.dot(r_diag)
    X_tune_nb = X_tune.dot(r_diag)
    X_conf_nb = X_conf.dot(r_diag)

    # Calibrate C strictly on tune
    print("Calibrating NB-SVM regularization C strictly on tune...", flush=True)
    best_c = 1.0
    best_tune_ll = float("inf")
    best_model = None

    for c_val in [0.2, 0.5, 1.0, 2.0, 4.0]:
        lr = LogisticRegression(C=c_val, max_iter=250, solver="liblinear", random_state=SEED)
        lr.fit(X_fit_nb, y[fit])
        p_tune = lr.predict_proba(X_tune_nb)[:, 1]
        ll = log_loss(y[tune], p_tune)
        print(f"C={c_val}: tune log_loss={ll:.5f}", flush=True)
        if ll < best_tune_ll:
            best_tune_ll = ll
            best_c = c_val
            best_model = lr

    p_tune_nbsvm = best_model.predict_proba(X_tune_nb)[:, 1]
    p_conf_nbsvm = best_model.predict_proba(X_conf_nb)[:, 1]

    grid_th = np.arange(0.35, 0.651, 0.005)
    best_th_nbsvm = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune_nbsvm >= t))
    print(f"NB-SVM Tune: Acc={accuracy_score(y[tune], p_tune_nbsvm >= best_th_nbsvm):.5f}, th={best_th_nbsvm:.3f}", flush=True)

    # Load Line A paired CSV to compare against baseline and bag3 on identical confirm split
    line_a_pairs_path = LINE_A_DIR / f"paired_{SEED}.csv"
    assert line_a_pairs_path.exists(), f"Missing {line_a_pairs_path}"
    df_pairs = pd.read_csv(line_a_pairs_path)

    p_conf_single = df_pairs["p_single"].to_numpy()
    p_conf_bag3 = df_pairs["p_bag3"].to_numpy()
    p_conf_line_a = df_pairs["p_line_a"].to_numpy()
    y_conf = df_pairs["y_true"].to_numpy()
    seen_mask = df_pairs["is_seen"].to_numpy().astype(bool)

    # Predeclared Blend with Bag3 on tune
    # We load tune probabilities of bag3 from Line A report
    with open(LINE_A_DIR / f"report_line_a_text_{SEED}.json") as f:
        rep_a = json.load(f)
    th_single = rep_a["baseline_single"]["threshold"]
    th_bag3 = rep_a["baseline_bag3"]["threshold"]

    # Frozen evaluation on confirm
    pred_conf_nbsvm = (p_conf_nbsvm >= best_th_nbsvm).astype(int)
    pred_conf_single = (p_conf_single >= th_single).astype(int)
    pred_conf_bag3 = (p_conf_bag3 >= th_bag3).astype(int)

    acc_nbsvm = accuracy_score(y_conf, pred_conf_nbsvm)
    ll_nbsvm = log_loss(y_conf, p_conf_nbsvm)

    def compute_paired(pred_cand, pred_base):
        net = int((pred_cand == y_conf).sum() - (pred_base == y_conf).sum())
        fixes = int(((pred_cand == y_conf) & (pred_base != y_conf)).sum())
        breaks = int(((pred_cand != y_conf) & (pred_base == y_conf)).sum())
        net_seen = int((pred_cand[seen_mask] == y_conf[seen_mask]).sum() - (pred_base[seen_mask] == y_conf[seen_mask]).sum())
        net_new = int((pred_cand[~seen_mask] == y_conf[~seen_mask]).sum() - (pred_base[~seen_mask] == y_conf[~seen_mask]).sum())
        return {"net": net, "fixes": fixes, "breaks": breaks, "net_seen": net_seen, "net_new": net_new}

    paired_vs_single = compute_paired(pred_conf_nbsvm, pred_conf_single)
    paired_vs_bag3 = compute_paired(pred_conf_nbsvm, pred_conf_bag3)
    paired_vs_line_a = compute_paired(pred_conf_nbsvm, (p_conf_line_a >= rep_a["line_a_standalone"]["threshold"]).astype(int))

    # Bootstrap 95% CI vs Bag3 (3,000 resamples)
    print("Computing 3,000 bootstrap seller resamples for NB-SVM vs Bag3...", flush=True)
    unique_sellers = np.unique(sellers[confirm])
    rng = np.random.default_rng(SEED)
    seller_to_indices = {s: np.where(sellers[confirm] == s)[0] for s in unique_sellers}
    boot_diffs = []
    for _ in range(3000):
        sampled_s = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)
        sampled_idx = np.concatenate([seller_to_indices[s] for s in sampled_s])
        diff = (pred_conf_nbsvm[sampled_idx] == y_conf[sampled_idx]).sum() - (pred_conf_bag3[sampled_idx] == y_conf[sampled_idx]).sum()
        boot_diffs.append(diff)

    ci95_vs_bag3 = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

    report = {
        "experiment": "line_b_nbsvm_discriminative_text_model",
        "split_seed": SEED,
        "confirm_rows": len(confirm),
        "vocabulary_dim": int(X_fit.shape[1]),
        "nbsvm_standalone": {
            "best_c": float(best_c),
            "threshold": float(best_th_nbsvm),
            "confirm_acc": float(acc_nbsvm),
            "confirm_ll": float(ll_nbsvm),
            "paired_vs_single": paired_vs_single,
            "paired_vs_bag3": paired_vs_bag3,
            "paired_vs_line_a": paired_vs_line_a,
            "bootstrap_ci95_vs_bag3": ci95_vs_bag3
        }
    }

    out_json = OUT_DIR / f"report_line_b_nbsvm_{SEED}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save pairs
    pairs_b = df_pairs.copy()
    pairs_b["p_nbsvm"] = p_conf_nbsvm
    pairs_b["pred_nbsvm"] = pred_conf_nbsvm
    pairs_b_path = OUT_DIR / f"paired_{SEED}.csv"
    pairs_b.to_csv(pairs_b_path, index=False)

    print(f"\nLine B Report saved to {out_json}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
