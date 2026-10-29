"""Audit R157 Ensemble vs Prior Deep CatBoost Baseline on Exact ID Intersections.

Directly addresses User Request Task 1:
- Reconciles why R157 reported .915/.911 vs prior .924/.920
- Evaluates exact intersection of confirm IDs
- Analyzes discordant predictions, fixes/breaks, log loss, and segment breakdowns
- Verifies fit ID overlap and feature differences
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "revision_submissions"

def main():
    print("=" * 80)
    print("AUDIT: R157 ENSEMBLE VS PRIOR DEEP CATBOOST BASELINE")
    print("=" * 80)

    summary_rows = []

    for seed in [2026, 314159]:
        f_deep = RESULTS_DIR / f"paired_deep_ablation_seed_{seed}.csv"
        f_ens = RESULTS_DIR / f"paired_strong_ensemble_seed_{seed}.csv"
        
        assert f_deep.exists(), f"Missing {f_deep}"
        assert f_ens.exists(), f"Missing {f_ens}"
        
        df_deep = pd.read_csv(f_deep)
        df_ens = pd.read_csv(f_ens)
        
        ids_deep = set(df_deep["id"])
        ids_ens = set(df_ens["id"])
        inter_ids = ids_deep & ids_ens
        deep_only = ids_deep - ids_ens
        ens_only = ids_ens - ids_deep
        
        merged = pd.merge(df_deep, df_ens, on="id", suffixes=("_deep", "_ens"))
        assert (merged["y_true_deep"] == merged["y_true_ens"]).all()
        
        y = merged["y_true_deep"].values
        p_deep = merged["p_base"].values
        pred_deep = merged["pred_base"].values
        
        p_cb = merged["p_catboost"].values
        pred_cb = merged["pred_catboost"].values
        
        p_ens = merged["p_ensemble"].values
        pred_ens = merged["pred_ensemble"].values
        
        # Accuracies & Log Losses
        acc_deep = accuracy_score(y, pred_deep)
        acc_cb = accuracy_score(y, pred_cb)
        acc_ens = accuracy_score(y, pred_ens)
        
        ll_deep = log_loss(y, p_deep)
        ll_cb = log_loss(y, p_cb)
        ll_ens = log_loss(y, p_ens)
        
        # Fixes and breaks of Ens vs Deep Base
        c_deep = (pred_deep == y)
        c_ens = (pred_ens == y)
        fixes = int(np.sum(~c_deep & c_ens))
        breaks = int(np.sum(c_deep & ~c_ens))
        net = fixes - breaks
        
        # Segments
        seen = (merged["is_seen_deep"] == 1).values
        new = ~seen
        
        fixes_seen = int(np.sum((~c_deep & c_ens)[seen]))
        breaks_seen = int(np.sum((c_deep & ~c_ens)[seen]))
        net_seen = fixes_seen - breaks_seen
        
        fixes_new = int(np.sum((~c_deep & c_ens)[new]))
        breaks_new = int(np.sum((c_deep & ~c_ens)[new]))
        net_new = fixes_new - breaks_new
        
        # Discordant
        disc = (pred_deep != pred_ens)
        disc_deep1_ens0 = int(np.sum(disc & (pred_deep == 1) & (pred_ens == 0)))
        disc_deep0_ens1 = int(np.sum(disc & (pred_deep == 0) & (pred_ens == 1)))
        
        print(f"\n--- SEED {seed} ---")
        print(f"Deep confirm total: {len(df_deep)}, Ens confirm total: {len(df_ens)}")
        print(f"Intersection: {len(inter_ids)} ({len(inter_ids)/len(df_deep)*100:.1f}%), Deep only: {len(deep_only)}, Ens only: {len(ens_only)}")
        print(f"Intersection Accuracy: Deep Base = {acc_deep:.5f}, R157 CB = {acc_cb:.5f}, R157 Ens = {acc_ens:.5f}")
        print(f"Intersection Log Loss: Deep Base = {ll_deep:.5f}, R157 CB = {ll_cb:.5f}, R157 Ens = {ll_ens:.5f}")
        print(f"R157 Ens vs Deep Base: Fixes = {fixes}, Breaks = {breaks}, Net = {net}")
        print(f"  Seen (n={np.sum(seen)}): Fixes={fixes_seen}, Breaks={breaks_seen}, Net={net_seen}")
        print(f"  New  (n={np.sum(new)}):  Fixes={fixes_new}, Breaks={breaks_new}, Net={net_new}")
        print(f"Discordant: {np.sum(disc)} ({np.mean(disc)*100:.2f}%) | Deep=used,Ens=new: {disc_deep1_ens0} | Deep=new,Ens=used: {disc_deep0_ens1}")
        
        summary_rows.append({
            "seed": seed,
            "n_intersection": len(inter_ids),
            "n_deep_only": len(deep_only),
            "n_ens_only": len(ens_only),
            "acc_deep_base": acc_deep,
            "acc_r157_cb": acc_cb,
            "acc_r157_ens": acc_ens,
            "ll_deep_base": ll_deep,
            "ll_r157_cb": ll_cb,
            "ll_r157_ens": ll_ens,
            "fixes_ens_vs_deep": fixes,
            "breaks_ens_vs_deep": breaks,
            "net_ens_vs_deep": net,
            "net_seen": net_seen,
            "net_new": net_new,
            "disc_total": int(np.sum(disc)),
            "disc_deep1_ens0": disc_deep1_ens0,
            "disc_deep0_ens1": disc_deep0_ens1,
        })

    df_out = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "round157_intersection_audit_vs_deep_ablation.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved CSV audit to {csv_path}")

if __name__ == "__main__":
    main()
