"""Paired Model Evaluation Library.

Provides strict matched-pair evaluation between a baseline and candidate:
- Accuracy, Log Loss, and deltas
- Fixes (base wrong, cand right), Breaks (base right, cand wrong), Net
- Segment breakdown (seen vs new sellers)
- Direction of prediction changes (base used -> cand new, etc.) with precision
- Cluster bootstrap 95% confidence intervals by seller (2000 replicates)
- Confidence band breakdown
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

def seller_cluster_bootstrap_ci(y_true, base_pred, cand_pred, sellers, n_boot=2000, seed=42):
    """Computes 95% bootstrap confidence interval for net correct items clustered by seller."""
    rng = np.random.default_rng(seed)
    unique_sellers = np.unique(sellers)
    n_sellers = len(unique_sellers)
    
    seller_to_idx = defaultdict(list)
    for idx, s in enumerate(sellers):
        seller_to_idx[s].append(idx)
        
    diff = (cand_pred == y_true).astype(int) - (base_pred == y_true).astype(int)
    seller_sums = np.array([np.sum(diff[seller_to_idx[s]]) for s in unique_sellers])
    
    boot_nets = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_sellers, size=n_sellers)
        boot_nets[b] = np.sum(seller_sums[sample_idx])
        
    ci_low = float(np.percentile(boot_nets, 2.5))
    ci_high = float(np.percentile(boot_nets, 97.5))
    ci_mean = float(np.mean(boot_nets))
    return ci_low, ci_high, ci_mean

def evaluate_paired_frames(df_base, df_cand, id_col="id", base_pred_col="pred", cand_pred_col="pred",
                           base_prob_col="p_used", cand_prob_col="p_used", y_col="y_used",
                           seen_col="is_seen", seller_col="seller_id", n_boot=2000, seed=42):
    """Evaluates two dataframes on common IDs with strict paired metrics."""
    # Find common IDs
    common_ids = sorted(list(set(df_base[id_col]) & set(df_cand[id_col])))
    assert len(common_ids) > 0, "No common IDs found between base and candidate frames"
    
    b_sub = df_base.set_index(id_col).loc[common_ids].reset_index()
    c_sub = df_cand.set_index(id_col).loc[common_ids].reset_index()
    
    y = b_sub[y_col].values.astype(int)
    assert (y == c_sub[y_col].values.astype(int)).all(), "Labels mismatch between base and candidate"
    
    sellers = b_sub[seller_col].values.astype(str)
    seen = b_sub[seen_col].values.astype(int) == 1
    new = ~seen
    
    p_b = b_sub[base_prob_col].values.astype(float)
    p_c = c_sub[cand_prob_col].values.astype(float)
    pred_b = b_sub[base_pred_col].values.astype(int)
    pred_c = c_sub[cand_pred_col].values.astype(int)
    
    # Global metrics
    acc_b = float(accuracy_score(y, pred_b))
    acc_c = float(accuracy_score(y, pred_c))
    ll_b = float(log_loss(y, p_b))
    ll_c = float(log_loss(y, p_c))
    
    c_base = (pred_b == y)
    c_cand = (pred_c == y)
    
    fixes = int(np.sum(~c_base & c_cand))
    breaks = int(np.sum(c_base & ~c_cand))
    net = fixes - breaks
    
    # Bootstrap CI
    ci_low, ci_high, ci_mean = seller_cluster_bootstrap_ci(y, pred_b, pred_c, sellers, n_boot=n_boot, seed=seed)
    
    # Segment breakdown
    def seg_stats(mask):
        sub_y = y[mask]
        sub_pb = pred_b[mask]
        sub_pc = pred_c[mask]
        sub_fixes = int(np.sum((~c_base & c_cand)[mask]))
        sub_breaks = int(np.sum((c_base & ~c_cand)[mask]))
        return {
            "n": int(np.sum(mask)),
            "acc_base": float(accuracy_score(sub_y, sub_pb)),
            "acc_cand": float(accuracy_score(sub_y, sub_pc)),
            "delta_acc": float(accuracy_score(sub_y, sub_pc) - accuracy_score(sub_y, sub_pb)),
            "logloss_base": float(log_loss(sub_y, p_b[mask])),
            "logloss_cand": float(log_loss(sub_y, p_c[mask])),
            "fixes": sub_fixes,
            "breaks": sub_breaks,
            "net": sub_fixes - sub_breaks,
        }
        
    stats_seen = seg_stats(seen)
    stats_new = seg_stats(new)
    
    # Direction breakdown
    # Base used (1) -> Cand new (0)
    flip_u2n = (pred_b == 1) & (pred_c == 0)
    # Base new (0) -> Cand used (1)
    flip_n2u = (pred_b == 0) & (pred_c == 1)
    
    dir_stats = {
        "used_to_new": {
            "count": int(np.sum(flip_u2n)),
            "fixes": int(np.sum(flip_u2n & (y == 0))),
            "breaks": int(np.sum(flip_u2n & (y == 1))),
            "net": int(np.sum(flip_u2n & (y == 0))) - int(np.sum(flip_u2n & (y == 1))),
            "precision": float(np.mean(y[flip_u2n] == 0)) if np.sum(flip_u2n) > 0 else 0.0,
        },
        "new_to_used": {
            "count": int(np.sum(flip_n2u)),
            "fixes": int(np.sum(flip_n2u & (y == 1))),
            "breaks": int(np.sum(flip_n2u & (y == 0))),
            "net": int(np.sum(flip_n2u & (y == 1))) - int(np.sum(flip_n2u & (y == 0))),
            "precision": float(np.mean(y[flip_n2u] == 1)) if np.sum(flip_n2u) > 0 else 0.0,
        }
    }
    
    results = {
        "n_rows": len(common_ids),
        "accuracy_base": acc_b,
        "accuracy_cand": acc_c,
        "delta_accuracy": acc_c - acc_b,
        "logloss_base": ll_b,
        "logloss_cand": ll_c,
        "delta_logloss": ll_c - ll_b,
        "fixes": fixes,
        "breaks": breaks,
        "net": net,
        "bootstrap_seller_ci95": [ci_low, ci_high],
        "bootstrap_seller_mean": ci_mean,
        "seen_sellers": stats_seen,
        "new_sellers": stats_new,
        "directions": dir_stats,
    }
    return results

def format_markdown_report(metrics, base_name="Baseline", cand_name="Candidate"):
    md = f"""### Comparacion Pareada: {cand_name} vs {base_name}
- **Filas Evaluadas:** {metrics['n_rows']}
- **Accuracy:** Base = {metrics['accuracy_base']:.5f} | Cand = {metrics['accuracy_cand']:.5f} (**Delta = {metrics['delta_accuracy']:+.5f}**)
- **Log Loss:** Base = {metrics['logloss_base']:.5f} | Cand = {metrics['logloss_cand']:.5f} (**Delta = {metrics['delta_logloss']:+.5f}**)
- **Aciertos Netos:** Fixes = {metrics['fixes']}, Breaks = {metrics['breaks']} -> **Net = {metrics['net']:+d}**
- **Bootstrap 95% CI (por vendedor):** [{metrics['bootstrap_seller_ci95'][0]:+.1f}, {metrics['bootstrap_seller_ci95'][1]:+.1f}] (media: {metrics['bootstrap_seller_mean']:+.1f})

#### Desglose por Segmento
| Segmento | N | Acc Base | Acc Cand | Delta Acc | Fixes | Breaks | Net |
|---|---|---|---|---|---|---|---|
| **Vistos** | {metrics['seen_sellers']['n']} | {metrics['seen_sellers']['acc_base']:.5f} | {metrics['seen_sellers']['acc_cand']:.5f} | {metrics['seen_sellers']['delta_acc']:+.5f} | {metrics['seen_sellers']['fixes']} | {metrics['seen_sellers']['breaks']} | **{metrics['seen_sellers']['net']:+d}** |
| **Nuevos** | {metrics['new_sellers']['n']} | {metrics['new_sellers']['acc_base']:.5f} | {metrics['new_sellers']['acc_cand']:.5f} | {metrics['new_sellers']['delta_acc']:+.5f} | {metrics['new_sellers']['fixes']} | {metrics['new_sellers']['breaks']} | **{metrics['new_sellers']['net']:+d}** |

#### Desglose por Direccion de Cambio
| Direccion | Total Flips | Fixes (Aciertos) | Breaks (Errores) | Neto | Precision |
|---|---|---|---|---|---|
| **used -> new** | {metrics['directions']['used_to_new']['count']} | {metrics['directions']['used_to_new']['fixes']} | {metrics['directions']['used_to_new']['breaks']} | **{metrics['directions']['used_to_new']['net']:+d}** | {metrics['directions']['used_to_new']['precision']*100:.1f}% |
| **new -> used** | {metrics['directions']['new_to_used']['count']} | {metrics['directions']['new_to_used']['fixes']} | {metrics['directions']['new_to_used']['breaks']} | **{metrics['directions']['new_to_used']['net']:+d}** | {metrics['directions']['new_to_used']['precision']*100:.1f}% |
"""
    return md

def main():
    parser = argparse.ArgumentParser(description="Paired Model Evaluation CLI")
    parser.add_argument("--base", type=str, required=True, help="Path to baseline CSV/CSV.gz")
    parser.add_argument("--cand", type=str, required=True, help="Path to candidate CSV/CSV.gz")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--base-pred-col", type=str, default="pred")
    parser.add_argument("--cand-pred-col", type=str, default="pred")
    parser.add_argument("--base-prob-col", type=str, default="p_used")
    parser.add_argument("--cand-prob-col", type=str, default="p_used")
    parser.add_argument("--y-col", type=str, default="y_used")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    df_base = pd.read_csv(args.base)
    df_cand = pd.read_csv(args.cand)

    metrics = evaluate_paired_frames(
        df_base, df_cand,
        id_col=args.id_col,
        base_pred_col=args.base_pred_col,
        cand_pred_col=args.cand_pred_col,
        base_prob_col=args.base_prob_col,
        cand_prob_col=args.cand_prob_col,
        y_col=args.y_col
    )
    print(format_markdown_report(metrics, Path(args.base).stem, Path(args.cand).stem))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
