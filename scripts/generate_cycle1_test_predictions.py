"""Generate Cycle 1 Full-Train Multi-Modal Candidate Predictions on Test (30,000 rows).

Trains on full 70,000 training set using the validated Cycle 1 architecture:
- 5-fold seller-grouped OOF for Char TF-IDF (3,5) and Word TF-IDF (1,2) on full train.
- Domain Spanish condition cues (MeLi literature).
- Deep feature matrix + seller_id as categorical feature.
- CatBoost (depth=6, lr=0.05, iters=1246).
- Calibrated tune threshold (0.420).
- Generates test CSV and audits flips vs D1+D2 control.
"""
import os, json, hashlib, zipfile, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_deep_text_20261101 import DROP_AGG
from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features

ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"
OUT_CSV = ROOT / "submissions" / "candidate_cycle1_multimodal_full70k.csv"
OUT_REPORT = ROOT / "results" / "cycle1_training" / "cycle1_test_audit_vs_d1d2.json"
CONTROL_CSV = ROOT / "submissions" / "candidate_d1_d2_combined_on48.csv"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== GENERATING CYCLE 1 FULL-TRAIN PREDICTIONS ===", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        train = [json.loads(line) for line in z.open("train_data.jsonlines")]
        test = [json.loads(line) for line in z.open("test_data.jsonlines")]

    N_train = len(train)
    N_test = len(test)
    print(f"Loaded {N_train} train rows, {N_test} test rows.", flush=True)

    y_train = np.array([int(r["condition"] == "used") for r in train])
    sellers_train = np.array([str(r["seller_id"]) for r in train])
    sellers_test = np.array([str(r["seller_id"]) for r in test])
    unique_train_sellers = set(sellers_train)

    # 1. Base deep features
    all_data = train + test
    ftr, ctr, *_ = extract_deep_features(all_data)
    keep = [c for c in ftr if not c.startswith(DROP_AGG)]
    df = ftr[keep].copy()

    # Train frequencies and relative prices
    train_idx = np.arange(N_train)
    test_idx = np.arange(N_train, N_train + N_test)

    for c in ("seller_id", "category_id", "city"):
        cnt = ctr.iloc[train_idx][c].value_counts()
        df["fit_freq_" + c] = ctr[c].map(cnt).fillna(1).astype(np.float32).values

    med = df.iloc[train_idx].groupby(ctr.iloc[train_idx].category_id).log_price.median()
    global_med = df.iloc[train_idx].log_price.median()
    df["fit_rel_price_cat"] = df.log_price - ctr.category_id.map(med).fillna(global_med)
    df["seller_id_cat"] = ctr.seller_id
    cats = ["category_id", "listing_type_id", "buying_mode", "state", "ship_mode", "seller_id_cat"]

    # 2. Text features
    print("Building text features...", flush=True)
    texts_title = np.array([str(r.get("title") or "") for r in all_data], dtype=object)
    texts_warranty = np.array([str(r.get("warranty") or "") for r in all_data], dtype=object)
    texts_attrs = np.array([" ".join(str(v.get("value_name") or "") for v in r.get("attributes") or []) for r in all_data], dtype=object)
    texts_combined = np.array([t + " " + w + " " + a for t, w, a in zip(texts_title, texts_warranty, texts_attrs)], dtype=object)

    # 5-fold OOF text models on full train
    print("Computing 5-fold OOF text models on full train...", flush=True)
    oof_char = np.empty(N_train, np.float32)
    oof_word = np.empty(N_train, np.float32)

    vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=180000, sublinear_tf=True, dtype=np.float32)
    vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True, dtype=np.float32)

    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(train_idx, y_train, sellers_train):
        # Char
        xc_tr = vec_char.fit_transform(texts_combined[tr])
        xc_va = vec_char.transform(texts_combined[va])
        m_char = LogisticRegression(C=2.0, max_iter=200, solver="liblinear")
        m_char.fit(xc_tr, y_train[tr])
        oof_char[va] = m_char.predict_proba(xc_va)[:, 1]

        # Word
        xw_tr = vec_word.fit_transform(texts_combined[tr])
        xw_va = vec_word.transform(texts_combined[va])
        m_word = LogisticRegression(C=1.5, max_iter=200, solver="liblinear")
        m_word.fit(xw_tr, y_train[tr])
        oof_word[va] = m_word.predict_proba(xw_va)[:, 1]

    # Full train models for test
    print("Fitting text models on 70k train for test inference...", flush=True)
    xc_train = vec_char.fit_transform(texts_combined[train_idx])
    m_char_full = LogisticRegression(C=2.0, max_iter=200, solver="liblinear")
    m_char_full.fit(xc_train, y_train)
    p_test_char = m_char_full.predict_proba(vec_char.transform(texts_combined[test_idx]))[:, 1]

    xw_train = vec_word.fit_transform(texts_combined[train_idx])
    m_word_full = LogisticRegression(C=1.5, max_iter=200, solver="liblinear")
    m_word_full.fit(xw_train, y_train)
    p_test_word = m_word_full.predict_proba(vec_word.transform(texts_combined[test_idx]))[:, 1]

    df["txt_char_oof_p"] = 0.5
    df.iloc[train_idx, df.columns.get_loc("txt_char_oof_p")] = oof_char
    df.iloc[test_idx, df.columns.get_loc("txt_char_oof_p")] = p_test_char

    df["txt_word_oof_p"] = 0.5
    df.iloc[train_idx, df.columns.get_loc("txt_word_oof_p")] = oof_word
    df.iloc[test_idx, df.columns.get_loc("txt_word_oof_p")] = p_test_word

    # Cues
    cues = {
        "cue_nuevo_garantia": [r"\bnuev[oa]s?\b", r"\b0km\b", r"\bgarant[ií]a\b", r"\bsellad[oa]\b", r"\bblister\b", r"\bcaja cerrada\b"],
        "cue_usado_detalles": [r"\busad[oa]s?\b", r"\bsegunda mano\b", r"\bbuen estado\b", r"\bexcelente estado\b", r"\bcon detalles?\b", r"\breparar\b", r"\brepuestos?\b", r"\blote\b", r"\bfalla\b", r"\bfunciona\b"]
    }
    for cue_name, patterns in cues.items():
        pat = "|".join(patterns)
        cnt = [len(re.findall(pat, t, re.IGNORECASE)) for t in texts_combined]
        df[cue_name] = np.array(cnt, dtype=np.float32)

    # 3. Train CatBoost on full 70k
    print("Fitting CatBoost on full 70,000 train rows...", flush=True)
    cb = CatBoostClassifier(iterations=1246, depth=6, learning_rate=0.05, l2_leaf_reg=5, loss_function="Logloss", verbose=False, thread_count=14, random_seed=20261109)
    cb.fit(df.iloc[train_idx], y_train, cat_features=cats)

    # 4. Predict test
    print("Predicting test probabilities...", flush=True)
    p_test = cb.predict_proba(df.iloc[test_idx])[:, 1]
    
    # Save raw test probabilities for future ensembling
    prob_path = ROOT / "results" / "cycle1_training" / "cycle1_test_probs.npy"
    np.save(prob_path, p_test)
    print(f"Saved test probabilities to {prob_path}")

    # Threshold with optimal tune threshold = 0.420
    pred_test_labels = np.where(p_test >= 0.420, "used", "new")

    # Save CSV
    submission_df = pd.DataFrame({
        "ID": np.arange(1, N_test + 1),
        "condition": pred_test_labels
    })
    submission_df.to_csv(OUT_CSV, index=False)
    sha_csv = sha256_file(OUT_CSV)
    print(f"Generated {OUT_CSV} with SHA-256: {sha_csv}")

    # 5. Audit vs D1+D2 Control
    ctrl_df = pd.read_csv(CONTROL_CSV)
    diff_mask = submission_df["condition"] != ctrl_df["condition"]
    diff_ids = submission_df.loc[diff_mask, "ID"].tolist()
    n_flips = len(diff_ids)

    # Check seen sellers among flipped IDs
    seen_flips = [fid for fid in diff_ids if sellers_test[fid - 1] in unique_train_sellers]
    new_flips = [fid for fid in diff_ids if sellers_test[fid - 1] not in unique_train_sellers]

    # Load ledger 83 to check overlap
    with open(ROOT / "results" / "intensive_20261030" / "submission_ledger_master_83.json") as f:
        ledger = json.load(f)
    intervened_history = set()
    for e in ledger["entries"]:
        intervened_history.update(e.get("flipped_ids", []))

    virgin_flips = [fid for fid in diff_ids if fid not in intervened_history]
    touched_flips = [fid for fid in diff_ids if fid in intervened_history]

    audit = {
        "candidate_file": str(OUT_CSV.relative_to(ROOT)),
        "sha256": sha_csv,
        "n_test": N_test,
        "n_flips_vs_d1d2": n_flips,
        "flip_rate": round(n_flips / N_test, 4),
        "flips_seen_sellers": len(seen_flips),
        "flips_new_sellers": len(new_flips),
        "flips_virgin_history": len(virgin_flips),
        "flips_touched_history": len(touched_flips),
        "tune_threshold": 0.420,
        "sample_flips_first_20": [
            {
                "ID": fid,
                "d1d2_pred": ctrl_df.loc[fid - 1, "condition"],
                "candidate_pred": submission_df.loc[fid - 1, "condition"],
                "prob_used": round(float(p_test[fid - 1]), 4),
                "seller_seen": bool(sellers_test[fid - 1] in unique_train_sellers),
                "virgin_history": bool(fid not in intervened_history)
            }
            for fid in diff_ids[:20]
        ]
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(audit, f, indent=2)

    print("\n=== FLIP AUDIT VS D1+D2 CONTROL ===")
    print(f"Total flips vs D1+D2: {n_flips} ({n_flips/N_test*100:.2f}%)")
    print(f"Seen sellers: {len(seen_flips)} | New sellers: {len(new_flips)}")
    print(f"Virgin IDs (never tested in 83 submissions): {len(virgin_flips)}")
    print(f"Touched IDs (tested in past submissions): {len(touched_flips)}")
    print(f"Audit saved to {OUT_REPORT}")

if __name__ == "__main__":
    main()
