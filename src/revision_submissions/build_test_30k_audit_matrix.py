"""Build Comprehensive 30,000-Row Test Audit Matrix.

Directly addresses User Request Task 3:
- 30,000 rows by ID (1 to 30000).
- Columns:
  * id
  * pred_s48 (benchmark label 'used'/'new')
  * is_seen (1 if seller in train 70k, 0 if new seller)
  * seller_id
  * category_id
  * Model probabilities (uniform polarity P(used=1)):
    - p_dual_catboost_used (R150 dual catboost)
    - p_strong_ensemble_used (R157 restored ensemble)
    - p_posterior_mcmc_used (MCMC posterior where available)
  * Agreement / disagreement direction vs S48:
    - dir_dual_catboost_vs_s48 ('agree', 'used->new', 'new->used')
    - dir_strong_ensemble_vs_s48 ('agree', 'used->new', 'new->used')
  * Historical Block & Probe Membership:
    - in_probe_66_new_to_new
    - in_probe_67_new_to_used
    - in_probe_68_virgin
    - in_probe_68_uncertainty
    - in_probe_68_draft_seller
    - in_probe_69_unanimous
    - in_candidate_71_ensemble
    - in_public_57
    - in_public_58
    - in_public_59_full
    - in_public_fin_stack_new150
    - in_public_fin_stack_seen150
    - in_public_41_44
    - in_public_45_50
  * Availability Mask (bitmask / string indicating which models/features are present)
  * Column provenance documentation
"""
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
RESULTS_DIR = ROOT / "results" / "revision_submissions"
SUB_DIR = ROOT / "submissions"

def load_submission_flips(csv_path, s48_dict):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    ids = df["ID"].astype(int).values
    conds = df["condition"].astype(str).str.strip().values
    s48_vals = np.array([s48_dict[i] for i in ids])
    diff_mask = (conds != s48_vals)
    return set(ids[diff_mask])

def main():
    print("Building 30,000-row test audit matrix...", flush=True)

    # 1. Load S48
    s48_json = RESULTS_DIR / "posterior_flips_20261027" / "s48.json"
    with open(s48_json) as f:
        s48_raw = json.load(f)
    s48_dict = {int(k): v for k, v in s48_raw.items()}

    # 2. Load Train sellers to identify seen sellers
    print("Loading train sellers...", flush=True)
    train_sellers = set()
    with open(DATA_DIR / "train_data.jsonlines", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            train_sellers.add(str(r["seller_id"]))

    # 3. Load Test Data
    print("Loading test data...", flush=True)
    test_ids = []
    item_ids = []
    seller_ids = []
    category_ids = []
    is_seens = []

    with open(DATA_DIR / "test_data.jsonlines", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            r = json.loads(line)
            test_ids.append(idx)
            item_ids.append(str(r["id"]))
            s_id = str(r["seller_id"])
            seller_ids.append(s_id)
            category_ids.append(str(r.get("category_id") or ""))
            is_seens.append(1 if s_id in train_sellers else 0)

    df_test = pd.DataFrame({
        "id": test_ids,
        "item_id": item_ids,
        "seller_id": seller_ids,
        "category_id": category_ids,
        "is_seen": is_seens,
    }).sort_values("id").reset_index(drop=True)

    df_test["pred_s48"] = [s48_dict[i] for i in df_test["id"]]

    # 4. Model Probabilities (Polarity uniform used=1)
    print("Loading model test probabilities...", flush=True)
    p_dual = np.load(RESULTS_DIR / "dual_catboost_test_probs.npy")
    p_ens = np.load(RESULTS_DIR / "strong_ensemble_test_probs.npy")
    assert len(p_dual) == 30000
    assert len(p_ens) == 30000
    df_test["p_dual_catboost_used"] = p_dual
    df_test["p_strong_ensemble_used"] = p_ens

    # Posterior MCMC where available
    post_file = RESULTS_DIR / "posterior_flips_20261027" / "posterior_por_id.csv"
    df_post = pd.read_csv(post_file)
    # P_48_mal is P(wrong). If pred_48 is 'used', P(used=1) = 1 - P_48_mal. If pred_48 is 'new', P(used=1) = P_48_mal.
    post_map = {}
    for _, row in df_post.iterrows():
        i = int(row["ID"])
        p_mal = float(row["P_48_mal"])
        p_used = (1.0 - p_mal) if s48_dict[i] == "used" else p_mal
        post_map[i] = p_used

    df_test["p_posterior_mcmc_used"] = [post_map.get(i, np.nan) for i in df_test["id"]]

    # 5. Flip Directions vs S48
    # Dual CatBoost
    pred_dual = ["used" if p >= 0.50 else "new" for p in p_dual]
    df_test["pred_dual_catboost"] = pred_dual
    def get_dir(p_cand, p_base):
        if p_cand == p_base:
            return "agree"
        return f"{p_base}->{p_cand}"

    df_test["dir_dual_vs_s48"] = [get_dir(c, b) for c, b in zip(pred_dual, df_test["pred_s48"])]

    # Strong Ensemble (threshold 0.49 from tune)
    pred_ens = ["used" if p >= 0.49 else "new" for p in p_ens]
    df_test["pred_strong_ensemble"] = pred_ens
    df_test["dir_ensemble_vs_s48"] = [get_dir(c, b) for c, b in zip(pred_ens, df_test["pred_s48"])]

    # 6. Candidate Probe & Historical Block Membership
    print("Loading probe and historical block memberships...", flush=True)
    p66_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_66_probe1_new_sellers_to_new.csv", s48_dict)
    p67_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_67_probe2_new_sellers_to_used.csv", s48_dict)
    p68_vir_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_68_probe3_alt_virgin_lexical.csv", s48_dict)
    p68_unc_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_68_probe3_uncertainty_flips.csv", s48_dict)
    p68_drf_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_68_probe3_seen_sellers_to_used.csv", s48_dict)
    p69_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_69_probe4_unanimous_consensus.csv", s48_dict)
    p71_flips = load_submission_flips(SUB_DIR / "20261029" / "submission_71_strong_ensemble_candidate.csv", s48_dict)

    df_test["in_probe_66_new_to_new"] = df_test["id"].isin(p66_flips).astype(int)
    df_test["in_probe_67_new_to_used"] = df_test["id"].isin(p67_flips).astype(int)
    df_test["in_probe_68_virgin"] = df_test["id"].isin(p68_vir_flips).astype(int)
    df_test["in_probe_68_uncertainty"] = df_test["id"].isin(p68_unc_flips).astype(int)
    df_test["in_probe_68_draft_seller"] = df_test["id"].isin(p68_drf_flips).astype(int)
    df_test["in_probe_69_unanimous"] = df_test["id"].isin(p69_flips).astype(int)
    df_test["in_candidate_71_ensemble"] = df_test["id"].isin(p71_flips).astype(int)

    # Public history blocks
    sub57_flips = load_submission_flips(SUB_DIR / "20261027" / "submission_57_posterior_sobre48.csv", s48_dict)
    sub58_flips = load_submission_flips(SUB_DIR / "20261027" / "submission_58_posterior_sobre48.csv", s48_dict)
    sub59_flips = load_submission_flips(SUB_DIR / "20261028" / "submission_59_fin_stack_banda_0.85_1.00_sobre48.csv", s48_dict)

    df_test["in_public_57"] = df_test["id"].isin(sub57_flips).astype(int)
    df_test["in_public_58"] = df_test["id"].isin(sub58_flips).astype(int)
    df_test["in_public_59_full"] = df_test["id"].isin(sub59_flips).astype(int)

    # Public 41, 44, 45, 46, 47, 48, 49, 50 IDs
    ids_41_44 = {1219, 8862, 22148, 18412, 26663, 26668}
    ids_45_50 = {1897, 6308, 7422, 24802, 2820, 9009, 10858, 14208, 8532, 13583, 16436, 22635, 25426, 25656, 26861,
                 18487, 1873, 22138, 12528, 27639, 9880, 28453, 14589, 25906, 17321, 40}
    df_test["in_public_41_44"] = df_test["id"].isin(ids_41_44).astype(int)
    df_test["in_public_45_50"] = df_test["id"].isin(ids_45_50).astype(int)

    # 7. Availability Mask
    # Bit 0: p_dual_catboost (1=yes)
    # Bit 1: p_strong_ensemble (1=yes)
    # Bit 2: p_posterior_mcmc (1=yes, 0=NaN)
    # Bit 3: tested in public submission (1=yes, 0=no)
    is_tested_public = (
        df_test["in_public_57"] | df_test["in_public_58"] | df_test["in_public_59_full"] |
        df_test["in_public_41_44"] | df_test["in_public_45_50"]
    )
    has_post = (~df_test["p_posterior_mcmc_used"].isna()).astype(int)

    mask_str = []
    for p_post_val, pub_val in zip(has_post, is_tested_public):
        flags = ["CB_DUAL", "ENS_R157"]
        if p_post_val:
            flags.append("MCMC_POST")
        if pub_val:
            flags.append("PUBLIC_TESTED")
        mask_str.append("+".join(flags))

    df_test["availability_mask"] = mask_str

    # Save to csv and csv.gz
    out_csv = RESULTS_DIR / "test_30k_full_audit_matrix.csv"
    out_csv_gz = RESULTS_DIR / "test_30k_full_audit_matrix.csv.gz"
    
    df_test.to_csv(out_csv, index=False)
    print(f"Saved CSV matrix: {out_csv} ({out_csv.stat().st_size / (1024*1024):.2f} MB)")
    df_test.to_csv(out_csv_gz, index=False, compression="gzip")
    print(f"Saved compressed CSV: {out_csv_gz} ({out_csv_gz.stat().st_size / (1024*1024):.2f} MB)")

    print("\nSummary Statistics of Audit Matrix:")
    print(f"Total Rows: {len(df_test)}")
    print(f"Seen Sellers: {df_test['is_seen'].sum()} ({df_test['is_seen'].mean()*100:.2f}%)")
    print(f"New Sellers:  {(df_test['is_seen']==0).sum()} ({(df_test['is_seen']==0).mean()*100:.2f}%)")
    print(f"Pred S48: Used = {(df_test['pred_s48']=='used').sum()}, New = {(df_test['pred_s48']=='new').sum()}")
    print(f"Dual CatBoost Disagreements vs S48: {(df_test['dir_dual_vs_s48']!='agree').sum()}")
    print(f"Strong Ensemble Disagreements vs S48: {(df_test['dir_ensemble_vs_s48']!='agree').sum()}")
    print(f"Rows with MCMC Posterior: {has_post.sum()}")
    print(f"Rows Tested in Public Submissions: {is_tested_public.sum()}")

if __name__ == "__main__":
    main()
