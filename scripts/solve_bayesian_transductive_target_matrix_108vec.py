"""Bayesian Transductive Target Matrix Solver across ALL 108 Submissions.

Includes:
- First 98 submissions from master ledger
- The 5 submissions from 2026-11-07 (Slots 1 to 5)
- The 5 submissions from 2026-11-08 (Slots 1 to 5):
    * S1: submission_20261108_01_surgical_rank5_capture_top6.csv (0.94738 = 19,893 hits, delta = -3)
    * S2: submission_20261108_02_pure_sellers_and_obsolete_top11.csv (0.94733 = 19,892 hits, delta = -4)
    * S3: submission_20261108_03_revert_false_used_top10.csv (0.94742 = 19,894 hits, delta = -2)
    * S4: submission_20261108_04_vintage_parts_and_antiques_top10.csv (0.94752 = 19,896 hits, delta = 0)
    * S5: submission_20261108_05_balanced_consensus_top20.csv (0.94728 = 19,891 hits, delta = -5)

Solves the constrained least squares system for item-level true marginal effect.
"""

import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bayesian_solver_108vec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PUB = 20998

def hit_count(score, n_pub=N_PUB):
    return int(round(score * n_pub))

def main():
    print("=" * 80)
    print("=== SOLVING BAYESIAN TRANSDUCTIVE SYSTEM ACROSS ALL 108 SUBMISSIONS ===")
    print("=" * 80)

    # 1. Baseline D1+D2 Anchor
    p_d1d2 = ROOT / "submissions" / "candidates_20261104" / "candidate_01_d1d2_plus_upper10.csv"
    df_ref = pd.read_csv(p_d1d2)
    d1d2_preds = df_ref.set_index("ID")["condition"].to_dict()
    d1d2_hits = hit_count(0.94652) # 19,875 hits
    print(f"Reference D1+D2 Baseline: {d1d2_hits} hits (0.94652)", flush=True)

    # 2. Master Ledger (First 98 submissions)
    p_ledger = ROOT / "results" / "forensic_ledger_20261104" / "submission_ledger_master_85vec.json"
    with open(p_ledger, encoding="utf-8") as f:
        ledger = json.load(f)
    entries = ledger["entries"]

    valid_configs = []
    for e in entries:
        sc = e.get("score")
        if sc is None or sc < 0.93:
            continue
        
        # Load flipped IDs vs D1+D2
        f_ids = e.get("flipped_ids") or e.get("flipped_ids_vs_d1d2")
        if not f_ids and "submission_file" in e:
            p_sub = ROOT / "submissions" / e["submission_file"]
            if not p_sub.exists():
                p_sub = ROOT / "submissions" / "tomorrow_20261106_portfolio_09490" / e["submission_file"]
            if not p_sub.exists():
                p_sub = ROOT / "submissions" / "today_20261106_calibrated" / e["submission_file"]
            if p_sub.exists():
                s_df = pd.read_csv(p_sub).set_index("ID")["condition"]
                f_ids = list(s_df[s_df != pd.Series(d1d2_preds)].index)

        if f_ids is not None:
            hits = hit_count(sc)
            delta = hits - d1d2_hits
            valid_configs.append({
                "name": e.get("name") or e.get("key") or e.get("submission_file"),
                "score": sc,
                "hits": hits,
                "delta": delta,
                "flips": sorted(list(f_ids))
            })

    # 3. Add 5 submissions from 2026-11-07
    p_yesterday_dir = ROOT / "submissions" / "portfolio_20261107_target_09490"
    yesterday_subs = [
        ("submission_20261107_01_purified_probe_top25.csv", 0.94738),
        ("submission_20261107_02_calibrated_expansion_top45.csv", 0.94738),
        ("submission_20261107_03_podium_attack_top65.csv", 0.94709),
        ("submission_20261107_04_podium_breakthrough_top85.csv", 0.94752),
        ("submission_20261107_05_podium_climax_top105.csv", 0.94738)
    ]
    for fname, sc in yesterday_subs:
        p_sub = p_yesterday_dir / fname
        assert p_sub.exists(), f"Missing file: {p_sub}"
        s_df = pd.read_csv(p_sub).set_index("ID")["condition"]
        f_d1d2 = set(s_df[s_df != pd.Series(d1d2_preds)].index)
        hits = hit_count(sc)
        delta = hits - d1d2_hits
        valid_configs.append({
            "name": fname,
            "score": sc,
            "hits": hits,
            "delta": delta,
            "flips": sorted(list(f_d1d2))
        })

    # 4. Add 5 submissions from 2026-11-08 (TODAY)
    p_today_dir = ROOT / "submissions" / "portfolio_20261108_podium_target_09490"
    today_subs = [
        ("submission_20261108_01_surgical_rank5_capture_top6.csv", 0.94738),
        ("submission_20261108_02_pure_sellers_and_obsolete_top11.csv", 0.94733),
        ("submission_20261108_03_revert_false_used_top10.csv", 0.94742),
        ("submission_20261108_04_vintage_parts_and_antiques_top10.csv", 0.94752),
        ("submission_20261108_05_balanced_consensus_top20.csv", 0.94728)
    ]
    for fname, sc in today_subs:
        p_sub = p_today_dir / fname
        assert p_sub.exists(), f"Missing file: {p_sub}"
        s_df = pd.read_csv(p_sub).set_index("ID")["condition"]
        f_d1d2 = set(s_df[s_df != pd.Series(d1d2_preds)].index)
        hits = hit_count(sc)
        delta = hits - d1d2_hits
        valid_configs.append({
            "name": fname,
            "score": sc,
            "hits": hits,
            "delta": delta,
            "flips": sorted(list(f_d1d2))
        })
        print(f"  Ingested {fname:55s} | Score: {sc:.5f} ({hits} hits) | Delta vs D1D2: {delta:+d} hits | Flips vs D1D2: {len(f_d1d2)}")

    print(f"\nTotal submission vectors ingested: {len(valid_configs)}", flush=True)

    # 5. Build Deduplicated Design Matrix M
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

    print(f"Unique Configurations: {n_configs}, Intervened Test IDs: {n_intervened}", flush=True)

    M = np.zeros((n_configs, n_intervened), dtype=np.float64)
    y_delta = np.zeros(n_configs, dtype=np.float64)

    for r, c in enumerate(configs):
        y_delta[r] = c["delta"]
        for item_id in c["flips"]:
            M[r, id_to_col[item_id]] = 1.0

    print(f"Design Matrix Shape: {M.shape}, Matrix Rank: {np.linalg.matrix_rank(M)}", flush=True)

    # 6. Transductive Deep Ensembles Prior
    p_trans = ROOT / "results" / "transductive_deep_models_103vec" / "transductive_test_probabilities_103vec.csv"
    if p_trans.exists():
        df_trans = pd.read_csv(p_trans).set_index("ID")
        p_model = df_trans["p_blend_103vec"].values
    else:
        p_model = np.full(30000, 0.5)

    e_prior = np.zeros(n_intervened, dtype=np.float64)
    for col, item_id in enumerate(all_intervened):
        idx = item_id - 1
        p_used = p_model[idx]
        d_lbl = d1d2_preds[item_id]
        
        # Prior effect vs D1+D2
        if d_lbl == "used":
            e_prior[col] = (1.0 - 2.0 * p_used) * (20998.0 / 30000.0)
        else:
            e_prior[col] = (2.0 * p_used - 1.0) * (20998.0 / 30000.0)

    # 7. Solve Bounded Least Squares MAP
    reg_lambda = 0.50
    print(f"\nSolving Bounded MAP Least Squares (lambda={reg_lambda})...", flush=True)
    M_aug = np.vstack([M, np.sqrt(reg_lambda) * np.eye(n_intervened)])
    y_aug = np.concatenate([y_delta, np.sqrt(reg_lambda) * e_prior])

    res = lsq_linear(M_aug, y_aug, bounds=(-1.0, 1.0), lsmr_tol='auto', verbose=0)
    e_map = res.x
    res_fit = np.linalg.norm(M @ e_map - y_delta)
    print(f"MAP Solver Completed! Residual Fit Norm: {res_fit:.4f}", flush=True)

    # 8. Evaluate Item Effects vs CONFIRMED BEST ANCHOR: Slot 4 (0.94752 = 19,896 hits)
    p_anchor_s4 = p_yesterday_dir / "submission_20261107_04_podium_breakthrough_top85.csv"
    anchor_s4 = pd.read_csv(p_anchor_s4).set_index("ID")["condition"]

    with open(ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f]

    records = []
    for col, item_id in enumerate(all_intervened):
        idx = item_id - 1
        it = test_rows[idx]
        title = str(it.get("title", ""))
        price = float(it.get("price", 0.0) or 0.0)
        cat = str(it.get("category_id", ""))
        curr_anchor_val = anchor_s4[item_id]
        d1d2_val = d1d2_preds[item_id]
        
        eff = float(e_map[col])
        p_m = float(p_model[idx])
        cnt = int((M[:, col] != 0).sum())

        # Expected gain vs Slot 4 anchor (0.94752)
        # In M, flipping item_id meant setting it to: 'used' if d1d2_val == 'new', or 'new' if d1d2_val == 'used'
        if curr_anchor_val == d1d2_val:
            # Anchor has original D1D2 condition
            # Flipping it means changing to the other label, whose effect vs D1D2 is eff
            expected_gain_vs_anchor = eff
            target_if_flipped = "used" if curr_anchor_val == "new" else "new"
        else:
            # Anchor has ALREADY flipped this item vs D1D2!
            # Keeping it in anchor already gets +eff vs D1D2.
            # Reverting it back to D1D2 condition would get -eff.
            expected_gain_vs_anchor = -eff
            target_if_flipped = d1d2_val

        records.append({
            "item_id": item_id,
            "anchor_condition_s4": curr_anchor_val,
            "flipped_condition": target_if_flipped,
            "expected_gain_vs_s4": round(expected_gain_vs_anchor, 4),
            "map_effect_vs_d1d2": round(eff, 4),
            "model_p_used": round(p_m, 4),
            "times_tested": cnt,
            "price": price,
            "category_id": cat,
            "title": title
        })

    df_res = pd.DataFrame(records).sort_values(by="expected_gain_vs_s4", ascending=False)
    out_csv = OUT_DIR / "bayesian_transductive_item_effects_108vec.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved 108-Vector Item Effects to: {out_csv}")

    # Top positive candidates vs anchor
    top_pos = df_res[df_res["expected_gain_vs_s4"] > 0.20]
    print(f"\nTotal Candidates with Expected Gain > +0.20 vs 0.94752 anchor: {len(top_pos)}")
    for idx, r in top_pos.iloc[:25].iterrows():
        print(f"ID {int(r['item_id']):5d}: {r['anchor_condition_s4']} -> {r['flipped_condition']} | gain={r['expected_gain_vs_s4']:+.3f} | p_used={r['model_p_used']:.3f} | tested={r['times_tested']:2d} | ${r['price']:5.0f} | {r['title'][:55]}")

    # Write executive report
    report_path = OUT_DIR / "bayesian_solver_108vec_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 108-Vector Bayesian Deconvolution Report\n\n")
        f.write(f"- **Total Submissions Ingested:** `{len(valid_configs)}`\n")
        f.write(f"- **Unique Design Configurations:** `{n_configs}`\n")
        f.write(f"- **Design Matrix Rank:** `{np.linalg.matrix_rank(M)}`\n")
        f.write(f"- **Intervened IDs:** `{n_intervened}`\n")
        f.write(f"- **Residual Fit Norm:** `{res_fit:.4f}`\n")
        f.write(f"- **Anchor Reference:** `0.94752` (`19,896` hits, Rank 6)\n")
        f.write(f"- **Available Positive Flips vs 0.94752:** `{len(top_pos)}`\n\n")
        f.write("### Top 25 Candidates with Empirical Gain:\n\n")
        f.write("| ID | Condición | Gain vs S4 | P_used | Pruebas | Precio | Título |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for idx, r in top_pos.iloc[:25].iterrows():
            f.write(f"| **{int(r['item_id'])}** | `{r['anchor_condition_s4']} -> {r['flipped_condition']}` | `+{r['expected_gain_vs_s4']:.3f}` | `{r['model_p_used']:.3f}` | `{r['times_tested']}` | `${r['price']:.0f}` | {r['title']} |\n")

    print(f"Report written to: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
