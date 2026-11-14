import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

def run():
    print("=== PREQUENTIAL BACKTEST BY FAMILY & SHIFT SENSITIVITY ===")
    df = pd.read_csv('results/reconciliation_all_historical_submissions.csv')
    print(f"Total historical submissions in ledger: {len(df)}")

    # Clean scored submissions
    scored = df[df['public_score'].notna()].copy()
    scored['hits'] = scored['public_score'] * 20998.0
    scored['delta_vs_s4'] = scored['hits'] - 19899.0
    print(f"Scored submissions: {len(scored)}")

    # Group by policy family
    family_stats = []
    for fam, grp in scored.groupby('policy_family'):
        n_ev = len(grp)
        deltas = grp['delta_vs_s4'].values
        flips = grp['flips_vs_s4_count'].dropna().values
        
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        min_d = float(np.min(deltas))
        max_d = float(np.max(deltas))
        med_d = float(np.median(deltas))
        
        # Losses / drawdowns
        n_loss = int((deltas < 0).sum())
        n_gain = int((deltas > 0).sum())
        n_neutral = int((deltas == 0).sum())
        
        # Max loss event
        worst_row = grp.loc[grp['delta_vs_s4'].idxmin()]
        worst_id = worst_row['event_id']
        worst_loss = worst_row['delta_vs_s4']
        worst_flips = worst_row['flips_vs_s4_count']

        family_stats.append({
            "family": fam,
            "n_submissions": n_ev,
            "mean_delta_vs_s4": round(mean_d, 2),
            "median_delta_vs_s4": round(med_d, 2),
            "min_delta_vs_s4": round(min_d, 2),
            "max_delta_vs_s4": round(max_d, 2),
            "n_losses": n_loss,
            "n_gains": n_gain,
            "worst_event": f"{worst_id} (Delta={worst_loss:+.0f}, Flips={worst_flips})",
        })

    df_fam = pd.DataFrame(family_stats).sort_values('n_submissions', ascending=False)
    print("\n--- PREQUENTIAL AUDIT BY POLICY FAMILY ---")
    for _, r in df_fam.iterrows():
        print(f"Family: {r['family']:35} | N={r['n_submissions']:2d} | Med Delta: {r['median_delta_vs_s4']:+6.1f} | Min Delta: {r['min_delta_vs_s4']:+6.1f} | Losses: {r['n_losses']}/{r['n_submissions']} | Worst: {r['worst_event']}")

    # Specific audit of the large retrocessions
    print("\n--- AUDIT OF MAJOR RETROCESSIONS & RECENT EVENTS ---")
    focus_events = ['EVT-142', 'EVT-151', 'EVT-152', 'EVT-153', 'EVT-154', 'EVT-155']
    for eid in focus_events:
        row = scored[scored['event_id'] == eid]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {r['event_id']}: Score={r['public_score']:.5f} | Hits={r['hits']:.0f} | Delta={r['delta_vs_s4']:+4.0f} | Flips={r['flips_vs_s4_count']:4.0f} | Family={r['policy_family']}")

    # Overall prequential error and empirical coverage audit
    # From previous audit in AG-0220:
    # 117 events: nominal 90% coverage was 30.8% (gamma = 5.53x needed)
    # MAE = 27.99, RMSE = 57.02
    print("\n--- PREQUENTIAL CALIBRATION SUMMARY (117 EVENTS) ---")
    print("  Nominal 90% Coverage: 30.8%")
    print("  Empirical Coverage Failure: 69.2% of events landed outside internal CV 90% intervals!")
    print("  Inflation Factor Required for 90% Empirical Coverage: gamma_90 = 5.53x")
    print("  Macro Routers & Ensembles Coverage: 0.0% (Drawdowns up to -349 hits)")
    print("  Gate C Mandate: CALLING INTERNAL CV Q05 '95% CONFIDENCE' IS STATISTICALLY INVALID.")

if __name__ == '__main__':
    run()
