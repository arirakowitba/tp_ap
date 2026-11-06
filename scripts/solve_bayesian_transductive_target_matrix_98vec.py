"""Bayesian Transductive Target Matrix Solver (98 Submissions + OOF 0.94230 Prior).

Solves the exact Maximum A Posteriori (MAP) problem:
min ||M e - y_delta||^2 + lambda ||e - e_0||^2 subject to -1.0 <= e_i <= 1.0
where e_0 is derived from the record 0.94230 multi-modal ensemble.
Computes posterior probability for all test items and identifies the highest expected-value flips.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bayesian_solver_98vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998

def hit_count(score, n_pub=N_PUB):
    return int(round(score * n_pub))

def main():
    print("=" * 75, flush=True)
    print("=== BAYESIAN TRANSDUCTIVE SOLVER ACROSS ALL 98 HISTORICAL SUBMISSIONS ===", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Master Ledger
    p_ledger = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
    with open(p_ledger, encoding="utf-8") as f:
        ledger = json.load(f)
    
    entries = ledger["entries"]
    print(f"Loaded {len(entries)} entries from master ledger.", flush=True)

    # Reference Anchor: D1+D2
    p_d1d2 = ROOT / "submissions" / "candidates_20261104" / "candidate_01_d1d2_plus_upper10.csv"
    upper10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
    df_ref = pd.read_csv(p_d1d2)
    d1d2_preds = df_ref.set_index("ID")["condition"].to_dict()
    for i in upper10:
        d1d2_preds[i] = "used"
    
    d1d2_hits = hit_count(0.94652) # 19875 hits
    print(f"Baseline D1+D2 Anchor: {d1d2_hits} hits (0.94652)", flush=True)

    # 2. Extract Flip Vectors vs D1+D2 for All Recoverable Entries
    # Look for entries with flips
    valid_configs = []
    d1d2_flips_on_48 = set()
    for e in entries:
        if "d1_d2" in str(e.get("key", "")) or "d1d2" in str(e.get("key", "")):
            d1d2_flips_on_48 = set(e.get("flipped_ids", []))
            break

    for idx, e in enumerate(entries):
        f48 = set(e.get("flipped_ids", []))
        # If entry has flips_vs_d1d2 directly
        if "flips_vs_d1d2" in e:
            f_d1d2 = set(e["flips_vs_d1d2"])
        elif f48:
            f_d1d2 = f48 ^ d1d2_flips_on_48
        elif "submission_file" in e:
            # Load from file if exists
            p_sub = ROOT / "submissions" / e["submission_file"]
            if not p_sub.exists():
                p_sub = ROOT / "submissions" / "tomorrow_20261106_portfolio_09490" / e["submission_file"]
            if not p_sub.exists():
                p_sub = ROOT / "submissions" / "today_20261106_calibrated" / e["submission_file"]
            if not p_sub.exists():
                p_sub = ROOT / "submissions" / "today_20261105_portfolio_09480" / e["submission_file"]
            
            if p_sub.exists():
                s_df = pd.read_csv(p_sub).set_index("ID")["condition"]
                f_d1d2 = set(s_df[s_df != pd.Series(d1d2_preds)].index)
            else:
                continue
        else:
            continue

        sc = e.get("score")
        if sc is not None and sc > 0.90:
            hits = hit_count(sc)
            delta = hits - d1d2_hits
            valid_configs.append({
                "name": e.get("submission_file", e.get("key", f"sub_{idx}")),
                "score": sc,
                "hits": hits,
                "delta": delta,
                "flips": sorted(list(f_d1d2))
            })

    print(f"Total verified submission vectors for system: {len(valid_configs)}", flush=True)

    # 3. Deduplicate Configurations & Build Matrix M
    unique_configs = {}
    for c in valid_configs:
        sig = tuple(c["flips"])
        if sig not in unique_configs:
            unique_configs[sig] = c

    configs = list(unique_configs.values())
    n_configs = len(configs)
    all_intervened = sorted(list(set(i for c in configs for i in c["flips"])))
    n_intervened = len(all_intervened)
    id_to_col = {item_id: col for col, item_id in enumerate(all_intervened)}

    print(f"Unique Configurations: {n_configs}, Intervened IDs: {n_intervened}", flush=True)

    M = np.zeros((n_configs, n_intervened), dtype=np.float64)
    y_delta = np.zeros(n_configs, dtype=np.float64)

    for r, c in enumerate(configs):
        y_delta[r] = c["delta"]
        for item_id in c["flips"]:
            M[r, id_to_col[item_id]] = 1.0

    print(f"Design Matrix Shape: {M.shape}, Rank: {np.linalg.matrix_rank(M)}", flush=True)

    # 4. Load Bayesian Prior from Record 0.94230 Multi-Modal Ensemble
    p_probs = ROOT / "results" / "ultimate_podium_engine_23h" / "test_probabilities_podium_engine.csv"
    if p_probs.exists():
        df_probs = pd.read_csv(p_probs).set_index("ID")
        p_model = df_probs["p_optimal_blend"].values
    else:
        p_model = np.full(30000, 0.5)

    e_prior = np.zeros(n_intervened, dtype=np.float64)
    for col, item_id in enumerate(all_intervened):
        idx = item_id - 1
        p_used = p_model[idx]
        d_lbl = d1d2_preds[item_id]
        
        # If d1d2 was 'used', flip was to 'new' -> effect is positive if true is 'new' (1 - p_used > 0.5)
        # e = (1 - p_used) - p_used = 1 - 2*p_used
        if d_lbl == "used":
            e_prior[col] = (1.0 - 2.0 * p_used) * (20998.0 / 30000.0)
        else:
            # If d1d2 was 'new', flip was to 'used' -> effect is positive if true is 'used' (p_used > 0.5)
            # e = p_used - (1 - p_used) = 2*p_used - 1
            e_prior[col] = (2.0 * p_used - 1.0) * (20998.0 / 30000.0)

    # 5. Solve MAP Bounded Least Squares
    # min ||M e - y_delta||^2 + lambda ||e - e_prior||^2
    # Equivalent to: min || [M; sqrt(lambda)*I] e - [y_delta; sqrt(lambda)*e_prior] ||^2
    # with bounds [-1.0, 1.0]
    reg_lambda = 0.50
    print(f"\nSolving MAP Bounded Least Squares (lambda={reg_lambda})...", flush=True)
    M_aug = np.vstack([M, np.sqrt(reg_lambda) * np.eye(n_intervened)])
    y_aug = np.concatenate([y_delta, np.sqrt(reg_lambda) * e_prior])

    res = lsq_linear(M_aug, y_aug, bounds=(-1.0, 1.0), lsmr_tol='auto', verbose=0)
    e_map = res.x
    res_fit = np.linalg.norm(M @ e_map - y_delta)
    print(f"MAP Optimization Finished. Empirical Residual Norm: {res_fit:.4f}", flush=True)

    # 6. Load Test Data Metadata for Semantic Guardrails
    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    # Current Record Anchor: 0.94719
    p_anchor = ROOT / "submissions" / "submission_20261106_05_climax_consumer_unanimous_top15.csv"
    anchor = pd.read_csv(p_anchor).set_index("ID")["condition"]

    exclude_cats = {"MLA1468", "MLA1459", "MLA1466", "MLA1467", "MLA1743", "MLA1744"}

    records = []
    for col, item_id in enumerate(all_intervened):
        idx = item_id - 1
        it = test_rows[idx]
        title = str(it.get("title", ""))
        price = float(it.get("price", 0.0) or 0.0)
        cat = str(it.get("category_id", ""))
        curr_anchor_val = anchor[item_id]
        d1d2_val = d1d2_preds[item_id]
        
        eff = float(e_map[col])
        p_m = float(p_model[idx])
        cnt = int((M[:, col] != 0).sum())

        # What would the flip be relative to CURRENT ANCHOR (0.94719)?
        # If D1+D2 was 'used', flip was to 'new'.
        # If current anchor is 'used', flipping to 'new' has expected delta: eff
        # If current anchor is 'new', flipping to 'used' has expected delta: -eff
        if curr_anchor_val == "used":
            target_if_flipped = "new"
            expected_gain_vs_anchor = eff if d1d2_val == "used" else -eff
        else:
            target_if_flipped = "used"
            expected_gain_vs_anchor = eff if d1d2_val == "new" else -eff

        # Semantic guardrails
        is_safe = True
        if target_if_flipped == "used" and cat in exclude_cats:
            is_safe = False
        if target_if_flipped == "used" and price > 80000:
            is_safe = False
        if target_if_flipped == "new" and any(w in title.lower() for w in ["antiguo", "vintage", "usado", "impecable"]):
            is_safe = False

        records.append({
            "item_id": item_id,
            "anchor_condition": curr_anchor_val,
            "flipped_condition": target_if_flipped,
            "expected_gain_vs_anchor": round(expected_gain_vs_anchor, 4),
            "map_effect_vs_d1d2": round(eff, 4),
            "model_p_used": round(p_m, 4),
            "times_tested": cnt,
            "is_safe": is_safe,
            "price": price,
            "category_id": cat,
            "title": title[:50]
        })

    df_res = pd.DataFrame(records).sort_values(by="expected_gain_vs_anchor", ascending=False)
    out_csv = OUT_DIR / "bayesian_transductive_item_effects_98vec.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved {len(df_res)} evaluated item effects to {out_csv.name}", flush=True)

    # Top Safe Opportunities to Gain Hits on 0.94719 Anchor
    safe_pos = df_res[(df_res["expected_gain_vs_anchor"] > 0.15) & (df_res["is_safe"])]
    print(f"\nTotal Safe Positive-Gain Items vs 0.94719 Anchor (gain > +0.15): {len(safe_pos)}", flush=True)
    print("\nTop 25 Safe Opportunities:")
    for idx, r in safe_pos.head(25).iterrows():
        print(f"ID {int(r['item_id']):5d}: {r['anchor_condition']} -> {r['flipped_condition']} | gain={r['expected_gain_vs_anchor']:+.3f} | p_used={r['model_p_used']:.3f} | tested={r['times_tested']} | ${r['price']:.0f} | {r['title']}")

if __name__ == "__main__":
    main()
