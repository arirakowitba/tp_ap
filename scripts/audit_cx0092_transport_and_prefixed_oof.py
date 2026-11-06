"""Audit of CX-0092 Transported Thresholds and Prefixed Text OOF on Split 20261144.

Computes exact paired metrics requested in CX-0092:
1. CatBoost Bag3 vs Single on seed 20261144:
   - Common threshold 0.50
   - Own tune thresholds (.485 single, .515 bag3)
   - Transported thresholds from seed 20261127 (.510 single, .505 bag3)
2. CatBoost Bag3 Control vs Enriched (Prefixed text OOF) on seed 20261144:
   - Common threshold 0.50
   - Own tune thresholds (.525 ctrl, .560 enr)
   - Transported threshold (.505)
3. Bootstrap 95% CI (3,000 seller resamples) and seen vs new breakdown.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]

def run_bootstrap(y, pred_a, pred_b, sellers, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    seller_to_indices = {s: np.where(sellers == s)[0] for s in unique_sellers}

    diffs = []
    for _ in range(n_boot):
        sample_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        idx = np.concatenate([seller_to_indices[s] for s in sample_sellers])
        acc_a = (pred_a[idx] == y[idx]).sum()
        acc_b = (pred_b[idx] == y[idx]).sum()
        diffs.append(acc_b - acc_a)

    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def analyze_pair(df, p_col_a, p_col_b, name_a, name_b, th_a, th_b, label=""):
    y = df["y_true"].values
    sellers = df["seller_id"].values
    is_seen = df["is_seen"].values.astype(bool)

    pred_a = (df[p_col_a].values >= th_a).astype(int)
    pred_b = (df[p_col_b].values >= th_b).astype(int)

    acc_a = float(accuracy_score(y, pred_a))
    acc_b = float(accuracy_score(y, pred_b))
    ll_a = float(log_loss(y, df[p_col_a].values))
    ll_b = float(log_loss(y, df[p_col_b].values))

    hits_a = int((pred_a == y).sum())
    hits_b = int((pred_b == y).sum())
    delta = hits_b - hits_a

    fixes = int(((pred_a != y) & (pred_b == y)).sum())
    breaks = int(((pred_a == y) & (pred_b != y)).sum())
    discrepancies = int((pred_a != pred_b).sum())

    seen_hits_a = int((pred_a[is_seen] == y[is_seen]).sum())
    seen_hits_b = int((pred_b[is_seen] == y[is_seen]).sum())
    new_hits_a = int((pred_a[~is_seen] == y[~is_seen]).sum())
    new_hits_b = int((pred_b[~is_seen] == y[~is_seen]).sum())

    ci_low, ci_high = run_bootstrap(y, pred_a, pred_b, sellers, 3000)

    print(f"\n--- {label}: {name_b} vs {name_a} (th_a={th_a:.3f}, th_b={th_b:.3f}) ---")
    print(f"Acc: {name_a}={acc_a:.5f} ({hits_a}), {name_b}={acc_b:.5f} ({hits_b}) -> Delta={delta:+d} hits")
    print(f"LogLoss: {name_a}={ll_a:.5f}, {name_b}={ll_b:.5f} -> Delta={ll_b - ll_a:+.5f}")
    print(f"Fixes={fixes}, Breaks={breaks}, Discrepancies={discrepancies}")
    print(f"Seen Delta: {seen_hits_b - seen_hits_a:+d}, New Delta: {new_hits_b - new_hits_a:+d}")
    print(f"Bootstrap 95% CI: [{ci_low:+.1f}, {ci_high:+.1f}]")

    return {
        "comparison": label,
        "model_a": name_a,
        "model_b": name_b,
        "th_a": th_a,
        "th_b": th_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "ll_a": ll_a,
        "ll_b": ll_b,
        "delta_hits": delta,
        "fixes": fixes,
        "breaks": breaks,
        "discrepancies": discrepancies,
        "net_seen": seen_hits_b - seen_hits_a,
        "net_new": new_hits_b - new_hits_a,
        "ci_95": [ci_low, ci_high]
    }

def main():
    print("=== AUDIT CX-0092: TRANSPORT AND PREFIXED OOF ON 20261144 ===")
    
    # 1. Bag3 vs Single
    df_bag = pd.read_csv(ROOT / "results" / "nbsvm_bag3_blend" / "paired_20261144.csv")
    print(f"Loaded Bag3/Single pairs: {len(df_bag)} rows")
    
    res_b_common = analyze_pair(df_bag, "p_single", "p_bag3", "Single", "Bag3", 0.50, 0.50, "Bag3 vs Single (Common th=0.50)")
    res_b_own = analyze_pair(df_bag, "p_single", "p_bag3", "Single", "Bag3", 0.485, 0.515, "Bag3 vs Single (Own Tune th)")
    res_b_trans = analyze_pair(df_bag, "p_single", "p_bag3", "Single", "Bag3", 0.510, 0.505, "Bag3 vs Single (Transported from 20261127)")

    # 2. Control Bag3 vs Enriched Bag3 (Prefixed text OOF)
    df_pref = pd.read_csv(ROOT / "results" / "prefixed_catboost_ablation" / "paired_catboost_prefixed_20261144.csv")
    print(f"\nLoaded Prefixed CatBoost pairs: {len(df_pref)} rows")

    res_p_common = analyze_pair(df_pref, "p_ctrl", "p_enr", "Control_Bag3", "Enriched_Bag3", 0.50, 0.50, "Enriched vs Control (Common th=0.50)")
    res_p_own = analyze_pair(df_pref, "p_ctrl", "p_enr", "Control_Bag3", "Enriched_Bag3", 0.525, 0.560, "Enriched vs Control (Own Tune th)")
    res_p_trans = analyze_pair(df_pref, "p_ctrl", "p_enr", "Control_Bag3", "Enriched_Bag3", 0.505, 0.505, "Enriched vs Control (Transported th=0.505)")

    out = {
        "split_seed": 20261144,
        "n_confirm": len(df_bag),
        "bag3_vs_single": {
            "common_50": res_b_common,
            "own_tune": res_b_own,
            "transported_20261127": res_b_trans
        },
        "enriched_vs_control": {
            "common_50": res_p_common,
            "own_tune": res_p_own,
            "transported_505": res_p_trans
        }
    }

    out_file = ROOT / "results" / "prefixed_catboost_ablation" / "cx0092_audit_transport_20261144.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nAudit saved to {out_file}")

if __name__ == "__main__":
    main()
