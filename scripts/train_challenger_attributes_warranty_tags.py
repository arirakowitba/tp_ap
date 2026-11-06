#!/usr/bin/env python3
"""Challenger training: Attributes, Warranty, and Tags (Strictly NO Title).

Implements the single P0 experiment commanded in CX-0117, CX-0118, and CX-0120:
- Corpus strictly excludes 'title' (radical ablation).
- Contains solely attributes.name, attributes.value_name, warranty, and tags.
- Stateless HashingVectorizer (word 1-2 + char_wb 3-5).
- Strictly nested NB/LR fitting on fit split (44,834 items) via GroupKFold(3) by seller.
- Out-of-fold calibration on fit; fit-only model evaluated on tune (12,542) and confirm (12,624).
- Paired comparison against the canonical baseline 'p_blend_native_nb' (12,016 hits).
- Router selection (z=3.5) on tune only; single evaluation on confirm.
- Zero ID overlap between tune and confirm.
- Full reporting: accuracy, logloss, fixes, breaks, net, seen/new, and seller bootstrap (3000 reps, 95% CI).
"""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold

# Import canonical split and metrics from existing verified script
try:
    from train_codex_raw_nested_text_histgb_20261104 import split, text_fit, digest, metrics
except ModuleNotFoundError:
    from scripts.train_codex_raw_nested_text_histgb_20261104 import split, text_fit, digest, metrics


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_attribute_corpus(rows):
    """Extract corpus strictly from attributes, warranty, and tags. NO title."""
    texts = []
    for r in rows:
        warranty = str(r.get("warranty") or "").strip()
        attrs = r.get("attributes") or []
        attr_parts = []
        for a in attrs:
            if isinstance(a, dict):
                name = str(a.get("name") or "").strip()
                val = str(a.get("value_name") or "").strip()
                if name or val:
                    attr_parts.append(f"{name} {val}".strip())
        attr_text = " ".join(attr_parts)
        tags = r.get("tags") or []
        tag_parts = [str(t).strip() for t in tags if t]
        tags_text = " ".join(tag_parts)

        parts = []
        if warranty:
            parts.append(f"WARRANTY {warranty}")
        if attr_text:
            parts.append(f"ATTR {attr_text}")
        if tags_text:
            parts.append(f"TAGS {tags_text}")

        full_text = " ".join(parts).strip()
        texts.append(full_text if full_text else "empty")
    return np.asarray(texts, dtype=object)


def extract_attribute_tabular(rows, fit_idx, sellers):
    """Extract tabular metadata related to attributes, warranty, and listing structure.
    Strictly without title. All cross-row/frequency calculations strictly within fit_idx.
    """
    df = pd.DataFrame()
    df["n_attrs"] = [len(r.get("attributes") or []) for r in rows]
    df["has_warranty"] = [int(bool(r.get("warranty"))) for r in rows]
    df["len_warranty"] = [len(str(r.get("warranty") or "")) for r in rows]
    df["n_tags"] = [len(r.get("tags") or []) for r in rows]
    df["n_pictures"] = [len(r.get("pictures") or []) for r in rows]
    df["n_variations"] = [len(r.get("variations") or []) for r in rows]
    df["has_mp"] = [int(bool(r.get("accepts_mercadopago"))) for r in rows]
    df["has_free_shipping"] = [int(bool((r.get("shipping") or {}).get("free_shipping"))) for r in rows]
    df["has_local_pick_up"] = [int(bool((r.get("shipping") or {}).get("local_pick_up"))) for r in rows]
    
    # Prices log1p
    prices = [float(r.get("price") or 0.0) for r in rows]
    df["log_price"] = np.log1p(np.maximum(0, prices))
    base_prices = [float(r.get("base_price") or 0.0) for r in rows]
    df["log_base_price"] = np.log1p(np.maximum(0, base_prices))
    
    # Categoricals
    df["category_id"] = [str(r.get("category_id") or "MISSING") for r in rows]
    df["listing_type_id"] = [str(r.get("listing_type_id") or "MISSING") for r in rows]
    df["buying_mode"] = [str(r.get("buying_mode") or "MISSING") for r in rows]
    df["currency_id"] = [str(r.get("currency_id") or "MISSING") for r in rows]
    df["status"] = [str(r.get("status") or "MISSING") for r in rows]
    df["shipping_mode"] = [str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
    
    # Seller frequency strictly computed on fit
    seller_counts = pd.Series(sellers[fit_idx]).value_counts()
    df["fit_seller_freq"] = np.log1p([seller_counts.get(s, 0) for s in sellers])
    
    return df


def evaluate_router_tune_confirm(tune_df, confirm_df, base_col, challenger_col, selection_z=3.5, min_routed=20):
    """Mirror of evaluate_tune_locked_challenger_router.py logic."""
    overlap = set(tune_df["item_id"].astype(str)) & set(confirm_df["item_id"].astype(str))
    assert not overlap, f"Forbidden tune/confirm ID overlap: {len(overlap)}"

    def mask_for(df, rule):
        pb = df[base_col].to_numpy(float)
        pc = df[challenger_col].to_numpy(float)
        b = pb >= 0.5
        c = pc >= 0.5
        mask = b != c
        mask &= np.abs(pc - 0.5) >= rule["challenger_margin_min"]
        mask &= np.abs(pb - 0.5) <= rule["baseline_margin_max"]
        direction = rule["direction"]
        if direction == "0_to_1":
            mask &= (~b) & c
        elif direction == "1_to_0":
            mask &= b & (~c)
        return mask

    def eval_mask(df, mask):
        y = df["y_true"].to_numpy(int)
        b = df[base_col].to_numpy(float) >= 0.5
        c = df[challenger_col].to_numpy(float) >= 0.5
        routed = b.copy()
        routed[mask] = c[mask]
        delta = (routed == y).astype(int) - (b == y).astype(int)
        n = int(mask.sum())
        fixes = int((delta == 1).sum())
        breaks = int((delta == -1).sum())
        return {
            "n_routed": n,
            "fixes": fixes,
            "breaks": breaks,
            "net": fixes - breaks,
            "baseline_hits": int((b == y).sum()),
            "routed_hits": int((routed == y).sum()),
            "baseline_accuracy": float((b == y).mean()),
            "routed_accuracy": float((routed == y).mean()),
            "net_lcb_normal95": float((fixes - breaks) - 1.96 * math.sqrt(max(n, 1))),
        }

    def seller_bootstrap(df, mask, seed=20261106, reps=3000):
        y = df["y_true"].to_numpy(int)
        b = df[base_col].to_numpy(float) >= 0.5
        c = df[challenger_col].to_numpy(float) >= 0.5
        routed = b.copy()
        routed[mask] = c[mask]
        delta = (routed == y).astype(int) - (b == y).astype(int)
        by_seller = pd.Series(delta).groupby(df["seller_id"].astype(str)).sum().to_numpy()
        rng = np.random.default_rng(seed)
        boot = np.empty(reps)
        for i in range(reps):
            boot[i] = by_seller[rng.integers(0, len(by_seller), len(by_seller))].sum()
        return np.quantile(boot, [0.025, 0.975]).tolist()

    candidates = []
    for cm in np.arange(0.0, 0.401, 0.05):
        for bm in np.arange(0.05, 0.501, 0.05):
            for direction in ("both", "0_to_1", "1_to_0"):
                rule = {
                    "challenger_margin_min": round(float(cm), 4),
                    "baseline_margin_max": round(float(bm), 4),
                    "direction": direction,
                }
                mask = mask_for(tune_df, rule)
                metrics_res = eval_mask(tune_df, mask)
                metrics_res["selection_lcb"] = float(
                    metrics_res["net"] - selection_z * math.sqrt(max(metrics_res["n_routed"], 1))
                )
                if metrics_res["n_routed"] >= min_routed:
                    candidates.append((metrics_res["selection_lcb"], metrics_res["net"], -metrics_res["n_routed"], rule, metrics_res))

    no_op = {
        "challenger_margin_min": 1.0,
        "baseline_margin_max": 0.0,
        "direction": "both",
    }
    selected = max(candidates, key=lambda x: x[:3]) if candidates else None
    if selected is None or selected[0] <= 0:
        rule = no_op
        tune_metrics = eval_mask(tune_df, mask_for(tune_df, no_op))
        selection_note = "No candidate had positive conservative tune LCB; selected no-op."
    else:
        _, _, _, rule, tune_metrics = selected
        selection_note = f"Selected by maximum tune-only multiplicity-guarded lower bound (z={selection_z:g})."

    confirm_mask = mask_for(confirm_df, rule)
    confirm_metrics = eval_mask(confirm_df, confirm_mask)
    confirm_metrics["seller_bootstrap_net_ci95"] = seller_bootstrap(confirm_df, confirm_mask)
    
    # Logloss comparison
    y_co = confirm_df["y_true"].to_numpy(int)
    base_p = confirm_df[base_col].to_numpy(float)
    chal_p = confirm_df[challenger_col].to_numpy(float)
    routed_p = base_p.copy()
    routed_p[confirm_mask] = chal_p[confirm_mask]
    confirm_metrics["baseline_logloss"] = float(log_loss(y_co, base_p))
    confirm_metrics["routed_logloss"] = float(log_loss(y_co, routed_p))
    
    # Seen vs new breakdown
    seen_mask = confirm_df["is_seen"].to_numpy(bool)
    b_co = base_p >= 0.5
    c_co = chal_p >= 0.5
    r_co = b_co.copy()
    r_co[confirm_mask] = c_co[confirm_mask]
    delta_co = (r_co == y_co).astype(int) - (b_co == y_co).astype(int)
    confirm_metrics["seen_breakdown"] = {
        "seen": {
            "n": int(seen_mask.sum()),
            "n_routed": int(confirm_mask[seen_mask].sum()),
            "net": int(delta_co[seen_mask].sum()),
        },
        "new": {
            "n": int((~seen_mask).sum()),
            "n_routed": int(confirm_mask[~seen_mask].sum()),
            "net": int(delta_co[~seen_mask].sum()),
        }
    }

    return {
        "selected_rule": rule,
        "selection_note": selection_note,
        "selection_z": selection_z,
        "n_candidate_rules": len(candidates),
        "tune": tune_metrics,
        "confirm": confirm_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path, help="Path to raw ap-2026-q-2.zip")
    parser.add_argument("--baseline_pairs", type=Path, default=Path("results/baseline_native_nested_20261104/pairs.csv"))
    parser.add_argument("--baseline_tune_pairs", type=Path, default=Path("results/baseline_native_nested_20261104/tune_pairs.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/challenger_attributes_warranty_tags_20261106"))
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--pure", action="store_true", help="Pure seller group split")
    parser.add_argument("--selection_z", type=float, default=3.5)
    args = parser.parse_args()

    t_start = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CHALLENGER: Attributes, Warranty, and Tags (Strictly NO Title)")
    print(f"Seed: {args.seed} | Pure: {args.pure} | Selection z: {args.selection_z}")
    print("=" * 80, flush=True)

    # 1. Load data
    print(f"\n1. Loading raw data from {args.zip_path}...", flush=True)
    with zipfile.ZipFile(args.zip_path) as z:
        rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
    assert len(rows) == 70000
    y = np.asarray([int(r["condition"] == "used") for r in rows], dtype=int)
    sellers = np.asarray([str(r["seller_id"]) for r in rows])
    ids = np.asarray([str(r["id"]) for r in rows])
    assert len(set(ids)) == 70000

    # 2. Split
    fi, tu, co = split(sellers, args.seed, pure=args.pure)
    print(f"Splits: fit={len(fi)}, tune={len(tu)}, confirm={len(co)}", flush=True)
    assert not (set(fi) & set(tu) or set(fi) & set(co) or set(tu) & set(co))
    seen = np.isin(sellers[co], sellers[fi])
    print(f"Confirm seen/new: seen={seen.sum()}, new={(~seen).sum()}", flush=True)

    # 3. Extract Corpus strictly without title
    print("\n2. Extracting attribute+warranty+tags corpus (strictly NO title)...", flush=True)
    corpus = extract_attribute_corpus(rows)
    print(f"Corpus sample 0: {corpus[0][:100]}...", flush=True)
    print(f"Corpus sample 2: {corpus[2][:100]}...", flush=True)
    
    # 4. Extract Tabular attributes metadata
    print("\n3. Extracting attribute tabular features...", flush=True)
    df_tab = extract_attribute_tabular(rows, fi, sellers)
    del rows
    gc.collect()

    # 5. Stateless Hashing + Strictly Nested NB/LR on fit
    print("\n4. Stateless HashingVectorizer (word 1-2 + char_wb 3-5)...", flush=True)
    t0 = time.time()
    word = HashingVectorizer(n_features=2**16, alternate_sign=False, ngram_range=(1, 2), norm="l2", dtype=np.float32)
    char = HashingVectorizer(n_features=2**16, alternate_sign=False, analyzer="char_wb", ngram_range=(3, 5), norm="l2", dtype=np.float32)
    X = sparse.hstack([word.transform(corpus), char.transform(corpus)], format="csr")
    print(f"Sparse matrix shape: {X.shape} in {time.time() - t0:.2f}s", flush=True)

    print("\n5. Running strictly nested GroupKFold(3) NB/LR on fit...", flush=True)
    pt_attr = np.full(len(y), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=3)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(fi, y[fi], sellers[fi])):
        assert not (set(sellers[fi[tr_idx]]) & set(sellers[fi[va_idx]]))
        t_fold = time.time()
        pt_attr[fi[va_idx]] = text_fit(X, y, fi[tr_idx], fi[va_idx])
        print(f"  Inner fold {fold}: {len(va_idx)} items in {time.time() - t_fold:.2f}s", flush=True)

    print("Fitting NB/LR on full fit and projecting onto tune and confirm...", flush=True)
    t0 = time.time()
    pt_attr[np.r_[tu, co]] = text_fit(X, y, fi, np.r_[tu, co])
    print(f"Projection completed in {time.time() - t0:.2f}s", flush=True)
    del X
    gc.collect()

    # 6. Build Shallow CatBoost on attribute tabular + nested NB
    print("\n6. Training shallow CatBoost on attributes tabular + nested NB probability...", flush=True)
    cats = ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "shipping_mode"]
    df_model = df_tab.assign(nested_attr_nb=pt_attr)
    cb_attr = CatBoostClassifier(
        iterations=600,
        depth=5,
        learning_rate=0.06,
        l2_leaf_reg=8,
        random_seed=args.seed,
        thread_count=4,
        verbose=False,
        loss_function="Logloss",
        allow_writing_files=False,
    )
    cb_attr.fit(
        df_model.iloc[fi],
        y[fi],
        cat_features=cats,
        eval_set=(df_model.iloc[tu], y[tu]),
        early_stopping_rounds=60,
        use_best_model=True,
    )
    p_cb_attr_tu = cb_attr.predict_proba(df_model.iloc[tu])[:, 1]
    p_cb_attr_co = cb_attr.predict_proba(df_model.iloc[co])[:, 1]
    print(f"CatBoost best iteration: {cb_attr.best_iteration_}", flush=True)

    # 7. Load Baseline Pairs
    print(f"\n7. Loading baseline pairs from {args.baseline_pairs}...", flush=True)
    df_base_co = pd.read_csv(args.baseline_pairs)
    df_base_tu = pd.read_csv(args.baseline_tune_pairs)

    # Identify base column
    base_col = "p_blend_native_nb"
    if base_col not in df_base_co.columns:
        # Fall back to other columns if present
        for candidate in ["p_blend_native_nb", "blend_native_nb", "p_cb_nested_nb"]:
            if candidate in df_base_co.columns:
                base_col = candidate
                break
    print(f"Official baseline column: '{base_col}'", flush=True)

    # Validate alignment
    assert (df_base_co["item_id"].astype(str) == ids[co]).all(), "Confirm item_id alignment mismatch!"
    assert (df_base_tu["item_id"].astype(str) == ids[tu]).all(), "Tune item_id alignment mismatch!"

    # 8. Create unified pair DataFrames
    tune_pairs_df = pd.DataFrame({
        "item_id": ids[tu],
        "seller_id": sellers[tu],
        "y_true": y[tu],
        base_col: df_base_tu[base_col].to_numpy(float),
        "p_attr_nb_lr": pt_attr[tu].astype(float),
        "p_attr_catboost": p_cb_attr_tu.astype(float),
    })
    
    confirm_pairs_df = pd.DataFrame({
        "item_id": ids[co],
        "seller_id": sellers[co],
        "is_seen": seen.astype(int),
        "y_true": y[co],
        base_col: df_base_co[base_col].to_numpy(float),
        "p_attr_nb_lr": pt_attr[co].astype(float),
        "p_attr_catboost": p_cb_attr_co.astype(float),
    })

    # Save pairs
    tune_pairs_path = out_dir / "tune_pairs.csv"
    pairs_path = out_dir / "pairs.csv"
    tune_pairs_df.to_csv(tune_pairs_path, index=False)
    confirm_pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved: {tune_pairs_path} ({sha256_file(tune_pairs_path)})")
    print(f"Saved: {pairs_path} ({sha256_file(pairs_path)})", flush=True)

    # Baseline confirm hits
    base_hits = int(((confirm_pairs_df[base_col] >= 0.5) == y[co]).sum())
    print(f"\nBaseline confirm hits ({base_col}): {base_hits} / {len(co)} (Acc: {base_hits / len(co):.5f})", flush=True)

    # 9. Standalone Challenger Performance on Confirm
    for chal_name, chal_probs in [("p_attr_nb_lr", pt_attr[co]), ("p_attr_catboost", p_cb_attr_co)]:
        m_chal = metrics(y[co], chal_probs, confirm_pairs_df[base_col] >= 0.5, sellers[co], seen)
        print(f"\nStandalone {chal_name} on Confirm:")
        print(f"  Hits: {m_chal['hits']} / {len(co)} (Acc: {m_chal['accuracy']:.5f}) | Logloss: {m_chal['logloss']:.5f}")
        print(f"  Net vs Baseline: {m_chal['net']} (fixes: {m_chal['fixes']}, breaks: {m_chal['breaks']})")
        print(f"  Seller 95% CI: {m_chal['seller_ci95']}")
        print(f"  Seen: {m_chal['seen_new']['seen']['net']} net | New: {m_chal['seen_new']['new']['net']} net")

    # 10. Run Tune-Locked Router for both challengers
    router_results = {}
    for chal_col in ["p_attr_nb_lr", "p_attr_catboost"]:
        print(f"\nEvaluating tune-locked router for challenger: {chal_col} (selection_z={args.selection_z})...", flush=True)
        res = evaluate_router_tune_confirm(
            tune_pairs_df, confirm_pairs_df, base_col, chal_col, selection_z=args.selection_z
        )
        router_results[chal_col] = res
        print(f"  Selected rule: {res['selected_rule']}")
        print(f"  Selection note: {res['selection_note']}")
        print(f"  Tune: n_routed={res['tune']['n_routed']}, fixes={res['tune']['fixes']}, breaks={res['tune']['breaks']}, net={res['tune']['net']}")
        print(f"  Confirm: n_routed={res['confirm']['n_routed']}, fixes={res['confirm']['fixes']}, breaks={res['confirm']['breaks']}, net={res['confirm']['net']}")
        print(f"  Confirm Routed Hits: {res['confirm']['routed_hits']} / {len(co)} (Acc: {res['confirm']['routed_accuracy']:.5f})")
        print(f"  Confirm Bootstrap 95% CI: {res['confirm']['seller_bootstrap_net_ci95']}")

    # 11. Full report
    report = {
        "contract": "Tune-locked router on attributes/warranty/tags challenger vs canonical baseline",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "pure": args.pure,
        "selection_z": args.selection_z,
        "splits": {"fit": len(fi), "tune": len(tu), "confirm": len(co)},
        "split_hashes": {n: digest(ids[ix]) for n, ix in [("fit", fi), ("tune", tu), ("confirm", co)]},
        "baseline_col": base_col,
        "baseline_confirm_hits": base_hits,
        "baseline_confirm_accuracy": float(base_hits / len(co)),
        "router_results": router_results,
        "files": {
            "tune_pairs": str(tune_pairs_path),
            "tune_pairs_sha256": sha256_file(tune_pairs_path),
            "confirm_pairs": str(pairs_path),
            "confirm_pairs_sha256": sha256_file(pairs_path),
            "source_zip": str(args.zip_path),
            "source_zip_sha256": sha256_file(args.zip_path),
        },
        "duration_seconds": time.time() - t_start,
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
