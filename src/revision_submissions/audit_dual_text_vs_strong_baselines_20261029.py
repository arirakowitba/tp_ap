"""Audit Dual-Text CatBoost against Strong Baselines (Round 142) on identical confirm IDs.

Loads paired predictions from Round 142 (HistGB, LightGBM, CatBoost fuerte, Ensemble)
and Round 144 (Dual-Text CatBoost with NB-SVM) across both Seed 2026 and Seed 314159.
Computes:
- Paired fixes and breaks
- Net accuracy differences
- Log loss comparisons
- Stratified metrics: Seen Sellers vs New Sellers
- 2000-iteration cluster bootstrap 95% CIs grouped by seller
- Confirm set overlap audit
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "revision_submissions"


def cluster_bootstrap_ci(y_true, base_pred, test_pred, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)
    clust_to_idx = defaultdict(list)
    for i, c in enumerate(clusters):
        clust_to_idx[c].append(i)
    base_correct = (base_pred == y_true)
    test_correct = (test_pred == y_true)
    diff = test_correct.astype(int) - base_correct.astype(int)
    clust_sums = np.array([np.sum(diff[clust_to_idx[c]]) for c in unique_clusters])
    boot_nets = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_clust, size=n_clust)
        boot_nets[b] = np.sum(clust_sums[sample_idx])
    return float(np.percentile(boot_nets, 2.5)), float(np.percentile(boot_nets, 97.5)), float(np.mean(boot_nets))


def main():
    audit_results = {}
    seeds = [2026, 314159]
    
    confirm_ids_by_seed = {}
    
    for seed in seeds:
        print(f"\n=======================================================", flush=True)
        print(f"AUDITING SEED {seed} AGAINST STRONG BASELINES", flush=True)
        print(f"=======================================================", flush=True)
        
        df_142 = pd.read_csv(RESULTS_DIR / f"paired_predictions_seed_{seed}.csv")
        df_144 = pd.read_csv(RESULTS_DIR / f"paired_dual_text_seed_{seed}.csv")
        
        assert (df_142["id"] == df_144["id"]).all(), f"ID mismatch on seed {seed}"
        assert (df_142["y_true_used"] == df_144["y_true"]).all(), f"Label mismatch on seed {seed}"
        
        confirm_ids_by_seed[seed] = set(df_142["id"])
        
        y_true = df_142["y_true_used"].values
        sellers = df_142["seller_id"].values
        is_seen = df_142["is_seen_seller"].values.astype(bool)
        is_new = ~is_seen
        
        p_dual = df_144["p_dual"].values
        pred_dual = (p_dual >= 0.5).astype(int)
        
        baselines = {
            "HistGB": df_142["p_hist_used"].values,
            "LightGBM": df_142["p_lgb_used"].values,
            "CatBoost_r142": df_142["p_cat_used"].values,
            "Ensemble_r142": df_142["p_ensemble_used"].values,
            "Baseline_CatBoost_r144": df_144["p_base"].values,
        }
        
        seed_audit = {}
        for base_name, p_base in baselines.items():
            pred_base = (p_base >= 0.5).astype(int)
            
            acc_base = accuracy_score(y_true, pred_base)
            acc_dual = accuracy_score(y_true, pred_dual)
            
            ll_base = log_loss(y_true, np.clip(p_base, 1e-6, 1 - 1e-6))
            ll_dual = log_loss(y_true, np.clip(p_dual, 1e-6, 1 - 1e-6))
            
            fixes = int(np.sum((pred_base != y_true) & (pred_dual == y_true)))
            breaks = int(np.sum((pred_base == y_true) & (pred_dual != y_true)))
            net = fixes - breaks
            
            ci_low, ci_high, ci_mean = cluster_bootstrap_ci(y_true, pred_base, pred_dual, sellers)
            
            # Stratified
            fixes_seen = int(np.sum((pred_base[is_seen] != y_true[is_seen]) & (pred_dual[is_seen] == y_true[is_seen])))
            breaks_seen = int(np.sum((pred_base[is_seen] == y_true[is_seen]) & (pred_dual[is_seen] != y_true[is_seen])))
            net_seen = fixes_seen - breaks_seen
            
            fixes_new = int(np.sum((pred_base[is_new] != y_true[is_new]) & (pred_dual[is_new] == y_true[is_new])))
            breaks_new = int(np.sum((pred_base[is_new] == y_true[is_new]) & (pred_dual[is_new] != y_true[is_new])))
            net_new = fixes_new - breaks_new
            
            print(f"Dual-Text vs {base_name:22s} | Acc: {acc_base:.5f}->{acc_dual:.5f} | Fixes/Breaks: {fixes:3d}/{breaks:3d} (Net: {net:+3d}) | CI95: [{ci_low:+5.1f}, {ci_high:+5.1f}] | Seen: {net_seen:+3d} | New: {net_new:+3d}", flush=True)
            
            seed_audit[base_name] = {
                "acc_base": float(acc_base),
                "acc_dual": float(acc_dual),
                "ll_base": float(ll_base),
                "ll_dual": float(ll_dual),
                "fixes": fixes,
                "breaks": breaks,
                "net": net,
                "ci_95": [ci_low, ci_high],
                "net_seen": net_seen,
                "net_new": net_new,
            }
            
        audit_results[seed] = seed_audit
        
    overlap_ids = len(confirm_ids_by_seed[2026] & confirm_ids_by_seed[314159])
    print(f"\n=======================================================", flush=True)
    print(f"CONFIRM PARTITION OVERLAP AUDIT", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Seed 2026 Confirm count:   {len(confirm_ids_by_seed[2026])}")
    print(f"Seed 314159 Confirm count: {len(confirm_ids_by_seed[314159])}")
    print(f"Shared Confirm rows:       {overlap_ids} ({overlap_ids / len(confirm_ids_by_seed[2026])*100:.2f}%)")
    print(f"Distinct Confirm rows:     {len(confirm_ids_by_seed[2026] | confirm_ids_by_seed[314159])}")
    
    out_file = RESULTS_DIR / "round147_audit_dual_text_vs_strong_baselines.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "overlap_confirm_ids": overlap_ids,
            "seeds": audit_results
        }, f, indent=2)
    print(f"\nAudit results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
