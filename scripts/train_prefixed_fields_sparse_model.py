"""Prefixed Fields & Key-Value Structured Sparse Text Model.

Protocol:
1. Channels:
   - title: words 1-3, chars 3-6 (normalized accents, units, numbers, cues).
   - warranty: separate channel.
   - attributes: prefixed tokens (aid:<id>=<val>, aname:<name>=<val>, val:<val>).
   - condition / brand / model tokens.
2. Fitted strictly on fit.
3. Linear regularized model (C tuned on tune).
4. Evaluated on confirm against Bag3.
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261144
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_DIR = ROOT / "results" / "prefixed_sparse_model"
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

def compute_nb_ratio(X, y, alpha=1.0):
    p = np.asarray(X[y == 1].sum(axis=0)).flatten() + alpha
    q = np.asarray(X[y == 0].sum(axis=0)).flatten() + alpha
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.log(p / q)

def main():
    print(f"=== PREFIXED FIELDS STRUCTURED SPARSE TEXT MODEL (SEED {SEED}) ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000

    y = np.array([int(r["condition"] == "used") for r in rows], dtype=np.int8)
    sellers = np.array([str(r["seller_id"]) for r in rows])
    ids = np.array([str(r["id"]) for r in rows])

    fit, tune, confirm = split_testlike(sellers, SEED)

    titles = [normalize_text(r.get("title")) for r in rows]
    warranties = [normalize_text(r.get("warranty")) for r in rows]
    structured_attrs = [extract_structured_attributes(r.get("attributes")) for r in rows]

    print("Fitting multi-channel vectorizers strictly on fit...", flush=True)
    v_t_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=120000, sublinear_tf=True)
    v_t_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=4, max_features=150000, sublinear_tf=True)
    v_w_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
    v_a_s = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)

    X_fit = sp.hstack([
        v_t_w.fit_transform([titles[i] for i in fit]),
        v_t_c.fit_transform([titles[i] for i in fit]),
        v_w_w.fit_transform([warranties[i] for i in fit]),
        v_a_s.fit_transform([structured_attrs[i] for i in fit])
    ], format="csr", dtype=np.float32)

    X_tune = sp.hstack([
        v_t_w.transform([titles[i] for i in tune]),
        v_t_c.transform([titles[i] for i in tune]),
        v_w_w.transform([warranties[i] for i in tune]),
        v_a_s.transform([structured_attrs[i] for i in tune])
    ], format="csr", dtype=np.float32)

    X_conf = sp.hstack([
        v_t_w.transform([titles[i] for i in confirm]),
        v_t_c.transform([titles[i] for i in confirm]),
        v_w_w.transform([warranties[i] for i in confirm]),
        v_a_s.transform([structured_attrs[i] for i in confirm])
    ], format="csr", dtype=np.float32)

    print(f"Structured sparse matrix shape on fit: {X_fit.shape}", flush=True)

    # Calibrate C strictly on tune
    print("Calibrating regularization C strictly on tune...", flush=True)
    best_c = 1.0
    best_ll = float("inf")
    best_m = None
    for c_val in [0.5, 1.0, 2.5, 5.0]:
        m = LogisticRegression(C=c_val, max_iter=250, solver="liblinear", random_state=SEED)
        m.fit(X_fit, y[fit])
        p_tu = m.predict_proba(X_tune)[:, 1]
        ll_tu = log_loss(y[tune], p_tu)
        print(f"C={c_val}: tune log_loss={ll_tu:.5f}", flush=True)
        if ll_tu < best_ll:
            best_ll = ll_tu
            best_c = c_val
            best_m = m

    p_tune = best_m.predict_proba(X_tune)[:, 1]
    p_conf = best_m.predict_proba(X_conf)[:, 1]

    grid_th = np.arange(0.35, 0.651, 0.005)
    best_th = max(grid_th, key=lambda t: accuracy_score(y[tune], p_tune >= t))
    acc_tune = float(accuracy_score(y[tune], p_tune >= best_th))
    acc_conf = float(accuracy_score(y[confirm], p_conf >= best_th))
    ll_conf = float(log_loss(y[confirm], p_conf))

    print(f"\nPrefixed Sparse Model (Confirm): Acc={acc_conf:.5f}, LL={ll_conf:.5f}, C={best_c}, th={best_th:.3f}", flush=True)

    report = {
        "experiment": "prefixed_fields_structured_sparse_model",
        "split_seed": SEED,
        "n_confirm": len(confirm),
        "vocab_dim": int(X_fit.shape[1]),
        "best_c": float(best_c),
        "best_th": float(best_th),
        "tune_acc": acc_tune,
        "confirm_acc": acc_conf,
        "confirm_ll": ll_conf
    }

    out_json = OUT_DIR / f"report_prefixed_sparse_{SEED}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save confirm probabilities
    np.savez(OUT_DIR / f"confirm_probs_{SEED}.npz", p_conf=p_conf, y_true=y[confirm], ids=ids[confirm])
    print(f"Report saved to {out_json}")

if __name__ == "__main__":
    main()
