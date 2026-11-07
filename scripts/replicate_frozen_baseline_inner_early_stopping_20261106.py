"""Replication of frozen 12024 baseline recipe (50/50 native text + nested NB)
with strictly internal early stopping inside fit, preserving tune and confirm untouched.
"""
import argparse, gc, hashlib, json, time, zipfile, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier
import lightgbm as lgb
from scripts.train_codex_raw_nested_text_histgb_20261104 import split, text_fit, digest, metrics

def inner_split(fi, s, seed, fraction=0.15):
    """Partition fi into inner_train and inner_val by seller group."""
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

def main():
    ap = argparse.ArgumentParser(description="Replicate frozen baseline with inner early stopping")
    ap.add_argument("zip", help="Path to input zip file containing train_data.jsonlines")
    ap.add_argument("out", help="Output directory")
    ap.add_argument("--seed", type=int, default=20261107, help="Random seed for split")
    ap.add_argument("--pure", action="store_true", help="Pure seller-group split")
    ap.add_argument("--threads", type=int, default=4, help="Thread count for models")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Inner validation fraction of fit sellers")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"Loading data from {args.zip}...", flush=True)
    with zipfile.ZipFile(args.zip) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=int)
    s = np.asarray([str(r["seller_id"]) for r in rows])
    ids = np.asarray([str(r["id"]) for r in rows])
    assert len(rows) == 70000 and len(set(ids)) == 70000

    # 1. Outer Split: fit, tune, confirm
    fi, tu, co = split(s, args.seed, args.pure)
    print(f"OUTER SPLITS: fit={len(fi)}, tune={len(tu)}, confirm={len(co)} (pure={args.pure})", flush=True)

    # 2. Inner Split inside fit: fi_tr, fi_va (strictly zero seller overlap)
    fi_tr, fi_va = inner_split(fi, s, args.seed, args.val_frac)
    print(f"INNER SPLITS: inner_train={len(fi_tr)}, inner_val={len(fi_va)} (sellers: tr={len(set(s[fi_tr]))}, va={len(set(s[fi_va]))})", flush=True)

    # 3. Deep feature extraction
    try:
        from src.revision_submissions.deep_catboost_nbsvm_ablation_20261029 import extract_deep_features
    except ModuleNotFoundError:
        from deep_reference import extract_deep_features

    print("Extracting deep tabular features...", flush=True)
    af, cat, _, _, _ = extract_deep_features(rows)
    drop = ("freq_", "seller_", "sell_mean_", "rel_price_cat")
    df = af[[c for c in af if not c.startswith(drop)]].copy()

    # Frequencies and relative prices computed strictly on fi_tr to prevent any leak to fi_va, tu, co
    for col in ["seller_id", "category_id", "city"]:
        counts = cat.iloc[fi_tr][col].value_counts()
        df["fit_freq_" + col] = cat[col].map(counts).fillna(0).astype(float)
    med = df.iloc[fi_tr].groupby(cat.iloc[fi_tr].category_id).log_price.median()
    df["fit_rel_price"] = df.log_price - cat.category_id.map(med).fillna(float(df.iloc[fi_tr].log_price.median()))

    text = np.asarray([
        str(r.get("title") or "") + " WARRANTY " + str(r.get("warranty") or "") + " ATTR " +
        " ".join(str(v.get("value_name") or v.get("name") or "") for v in r.get("attributes") or [] if isinstance(v, dict))
        for r in rows
    ], object)

    for key in ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "seller_id"]:
        df[key] = [str(r.get(key) or "MISSING") for r in rows]
    df["state"] = [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows]
    df["ship_mode"] = [str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
    cats = ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "seller_id", "state", "ship_mode"]
    del rows; gc.collect()

    # 4. Stateless Hashing + Strictly inner-train Supervised NB/LR
    print("Hashing text...", flush=True)
    word = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2", dtype=np.float32)
    char = HashingVectorizer(n_features=2**16, alternate_sign=False, analyzer="char_wb", ngram_range=(3, 5), norm="l2", dtype=np.float32)
    X = sparse.hstack([word.transform(text), char.transform(text)], format="csr")
    pt = np.full(len(y), np.nan, dtype=np.float32)

    # 3-fold GroupKFold by seller strictly within fi_tr
    for f, (tr, va) in enumerate(GroupKFold(3).split(fi_tr, y[fi_tr], s[fi_tr])):
        assert not set(s[fi_tr[tr]]) & set(s[fi_tr[va]])
        pt[fi_tr[va]] = text_fit(X, y, fi_tr[tr], fi_tr[va])
        print(f"INNER_TEXT fold {f} elapsed {time.time() - start_time:.1f}s", flush=True)

    # Outer projection: fit on all fi_tr, predict on fi_va, tu, co
    outer_targets = np.r_[fi_va, tu, co]
    pt[outer_targets] = text_fit(X, y, fi_tr, outer_targets)
    del X; gc.collect()

    d_nb = df.assign(nested_nb_text=pt)
    d_native = df.assign(native_text=np.asarray([str(t).strip() or "empty" for t in text], object))

    probs = {}
    tup = {}
    vap = {}
    metadata = {}

    # 5. Train CatBoost models with eval_set on fi_va (early stopping purely internal to fit)
    for name, dd, txt in [
        ("cb_tabular", df, []),
        ("cb_nested_nb", d_nb, []),
        ("cb_native_text", d_native, ["native_text"]),
    ]:
        t0 = time.time()
        print(f"Starting CatBoost {name}...", flush=True)
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
        vap[name] = m.predict_proba(dd.iloc[fi_va])[:, 1]
        tup[name] = m.predict_proba(dd.iloc[tu])[:, 1]
        probs[name] = m.predict_proba(dd.iloc[co])[:, 1]
        metadata[name] = {
            "best_iteration": int(m.best_iteration_),
            "seconds": time.time() - t0
        }
        print(f"TRAINED {name}: best_iter={m.best_iteration_}, time={time.time() - t0:.1f}s", flush=True)

    # 6. Train LightGBM nested NB
    print("Starting LightGBM lgb_nested_nb...", flush=True)
    ldf = d_nb.copy()
    for c in cats:
        ldf[c] = ldf[c].astype("category")
    m = lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=45, max_depth=7, learning_rate=0.05,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5, random_state=args.seed, n_jobs=args.threads, verbosity=-1
    )
    t0 = time.time()
    m.fit(
        ldf.iloc[fi_tr], y[fi_tr],
        eval_set=[(ldf.iloc[fi_va], y[fi_va])],
        callbacks=[lgb.early_stopping(90, verbose=False)]
    )
    vap["lgb_nested_nb"] = m.predict_proba(ldf.iloc[fi_va])[:, 1]
    tup["lgb_nested_nb"] = m.predict_proba(ldf.iloc[tu])[:, 1]
    probs["lgb_nested_nb"] = m.predict_proba(ldf.iloc[co])[:, 1]
    metadata["lgb_nested_nb"] = {
        "best_iteration": int(m.best_iteration_),
        "seconds": time.time() - t0
    }
    print(f"TRAINED lgb_nested_nb: best_iter={m.best_iteration_}, time={time.time() - t0:.1f}s", flush=True)

    # 7. Predeclared Blends:
    # - blend_native_nb: 0.50 cb_native_text + 0.50 cb_nested_nb (FROZEN BASELINE RECIPE)
    # - blend_cb_lgb80: 0.80 cb_nested_nb + 0.20 lgb_nested_nb
    for name, (w, k1, k2) in {
        "blend_native_nb": (0.50, "cb_native_text", "cb_nested_nb"),
        "blend_cb_lgb80": (0.80, "cb_nested_nb", "lgb_nested_nb")
    }.items():
        vap[name] = w * vap[k1] + (1.0 - w) * vap[k2]
        tup[name] = w * tup[k1] + (1.0 - w) * tup[k2]
        probs[name] = w * probs[k1] + (1.0 - w) * probs[k2]

    # 8. Evaluation at Fixed Threshold 0.50
    # Seen mask on confirm and tune
    seen_co = np.isin(s[co], s[fi])
    seen_tu = np.isin(s[tu], s[fi])

    base_co_nested_nb = probs["cb_nested_nb"] >= 0.50
    base_tu_nested_nb = tup["cb_nested_nb"] >= 0.50

    base_co_native_text = probs["cb_native_text"] >= 0.50
    base_tu_native_text = tup["cb_native_text"] >= 0.50

    confirm_metrics_vs_nested = {
        name: metrics(y[co], p, base_co_nested_nb, s[co], seen_co)
        for name, p in probs.items()
    }
    tune_metrics_vs_nested = {
        name: metrics(y[tu], p, base_tu_nested_nb, s[tu], seen_tu)
        for name, p in tup.items()
    }

    confirm_metrics_vs_native = {
        name: metrics(y[co], probs[name], base_co_native_text, s[co], seen_co)
        for name in ["blend_native_nb"]
    }
    tune_metrics_vs_native = {
        name: metrics(y[tu], tup[name], base_tu_native_text, s[tu], seen_tu)
        for name in ["blend_native_nb"]
    }

    # Save Pairs
    pairs_df = pd.DataFrame({
        "item_id": ids[co], "seller_id": s[co], "is_seen": seen_co.astype(int), "y_true": y[co],
        **{"p_" + n: probs[n] for n in probs}
    })
    pairs_csv_path = out / "pairs.csv"
    pairs_df.to_csv(pairs_csv_path, index=False)

    tune_pairs_df = pd.DataFrame({
        "item_id": ids[tu], "seller_id": s[tu], "is_seen": seen_tu.astype(int), "y_true": y[tu],
        **{"p_" + n: tup[n] for n in tup}
    })
    tune_pairs_csv_path = out / "tune_pairs.csv"
    tune_pairs_df.to_csv(tune_pairs_csv_path, index=False)

    val_pairs_df = pd.DataFrame({
        "item_id": ids[fi_va], "seller_id": s[fi_va], "y_true": y[fi_va],
        **{"p_" + n: vap[n] for n in vap}
    })
    val_pairs_csv_path = out / "inner_val_pairs.csv"
    val_pairs_df.to_csv(val_pairs_csv_path, index=False)

    report = {
        "seed": args.seed,
        "pure": args.pure,
        "split_sizes": {
            "outer_fit": len(fi),
            "inner_train": len(fi_tr),
            "inner_val": len(fi_va),
            "tune": len(tu),
            "confirm": len(co)
        },
        "split_hashes": {
            "fit": digest(ids[fi]),
            "inner_train": digest(ids[fi_tr]),
            "inner_val": digest(ids[fi_va]),
            "tune": digest(ids[tu]),
            "confirm": digest(ids[co])
        },
        "confirm_metrics_at_050": confirm_metrics_vs_nested,
        "tune_metrics_at_050": tune_metrics_vs_nested,
        "blend_vs_native_text": {
            "confirm": confirm_metrics_vs_native["blend_native_nb"],
            "tune": tune_metrics_vs_native["blend_native_nb"]
        },
        "training_metadata": metadata,
        "pairs_sha256": hashlib.sha256(pairs_csv_path.read_bytes()).hexdigest(),
        "tune_pairs_sha256": hashlib.sha256(tune_pairs_csv_path.read_bytes()).hexdigest(),
        "inner_val_pairs_sha256": hashlib.sha256(val_pairs_csv_path.read_bytes()).hexdigest(),
        "duration_seconds": time.time() - start_time
    }

    report_json_path = out / "report.json"
    report_json_path.write_text(json.dumps(report, indent=2) + "\n")

    # Generate Markdown Summary Table
    md_lines = [
        f"# Reporte de Replicación del Baseline Congelado — Seed {args.seed} ({'Pure Seller-Group' if args.pure else 'Mixed Split'})",
        "",
        f"- **Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Pure:** {args.pure}",
        f"- **Seed:** {args.seed}",
        f"- **Tamaños de Corte:** fit={len(fi)} (inner_train={len(fi_tr)}, inner_val={len(fi_va)}), tune={len(tu)}, confirm={len(co)}",
        f"- **Duración total:** {time.time() - start_time:.1f} segundos",
        "",
        "## 1. Métricas en Confirm a Umbral Fijo 0.50",
        "",
        "| Modelo | Confirm Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"
    ]

    for name in ["cb_tabular", "cb_nested_nb", "cb_native_text", "blend_native_nb", "lgb_nested_nb", "blend_cb_lgb80"]:
        m_co = confirm_metrics_vs_nested[name]
        ci_str = f"[{m_co['seller_ci95'][0]:.1f}, {m_co['seller_ci95'][1]:.1f}]"
        seen_net = m_co["seen_new"]["seen"]["net"]
        new_net = m_co["seen_new"]["new"]["net"]
        md_lines.append(
            f"| **{name}** | {m_co['hits']} | {m_co['accuracy']:.5f} | {m_co['logloss']:.5f} | "
            f"{m_co['net']:+d} | {m_co['fixes']} | {m_co['breaks']} | {ci_str} | {seen_net:+d} | {new_net:+d} |"
        )

    md_lines.extend([
        "",
        "## 2. Métricas en Tune a Umbral Fijo 0.50",
        "",
        "| Modelo | Tune Hits | Exactitud | Logloss | Net vs Nested NB | Fixes | Breaks | Bootstrap IC 95% | Seen Net | New Net |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"
    ])

    for name in ["cb_tabular", "cb_nested_nb", "cb_native_text", "blend_native_nb", "lgb_nested_nb", "blend_cb_lgb80"]:
        m_tu = tune_metrics_vs_nested[name]
        ci_str = f"[{m_tu['seller_ci95'][0]:.1f}, {m_tu['seller_ci95'][1]:.1f}]"
        seen_net = m_tu["seen_new"]["seen"]["net"]
        new_net = m_tu["seen_new"]["new"]["net"]
        md_lines.append(
            f"| **{name}** | {m_tu['hits']} | {m_tu['accuracy']:.5f} | {m_tu['logloss']:.5f} | "
            f"{m_tu['net']:+d} | {m_tu['fixes']} | {m_tu['breaks']} | {ci_str} | {seen_net:+d} | {new_net:+d} |"
        )

    md_lines.extend([
        "",
        "## 3. Comparación Pareada Específica de `blend_native_nb` (50/50) vs `cb_native_text`",
        "",
        f"- **Confirm:** Net vs Native Text = {confirm_metrics_vs_native['blend_native_nb']['net']:+d} (Fixes: {confirm_metrics_vs_native['blend_native_nb']['fixes']}, Breaks: {confirm_metrics_vs_native['blend_native_nb']['breaks']}), IC 95%: [{confirm_metrics_vs_native['blend_native_nb']['seller_ci95'][0]:.1f}, {confirm_metrics_vs_native['blend_native_nb']['seller_ci95'][1]:.1f}]",
        f"- **Tune:** Net vs Native Text = {tune_metrics_vs_native['blend_native_nb']['net']:+d} (Fixes: {tune_metrics_vs_native['blend_native_nb']['fixes']}, Breaks: {tune_metrics_vs_native['blend_native_nb']['breaks']}), IC 95%: [{tune_metrics_vs_native['blend_native_nb']['seller_ci95'][0]:.1f}, {tune_metrics_vs_native['blend_native_nb']['seller_ci95'][1]:.1f}]",
        "",
        "## 4. Hashes de Reproducibilidad",
        "",
        f"- `pairs.csv`: `{report['pairs_sha256']}`",
        f"- `tune_pairs.csv`: `{report['tune_pairs_sha256']}`",
        f"- `inner_val_pairs.csv`: `{report['inner_val_pairs_sha256']}`",
        f"- Fit hash: `{report['split_hashes']['fit']}`",
        f"- Inner train hash: `{report['split_hashes']['inner_train']}`",
        f"- Inner val hash: `{report['split_hashes']['inner_val']}`",
        f"- Tune hash: `{report['split_hashes']['tune']}`",
        f"- Confirm hash: `{report['split_hashes']['confirm']}`"
    ])

    report_md_path = out / "REPORT.md"
    report_md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nCompleted! Written to {out}", flush=True)

if __name__ == "__main__":
    main()
