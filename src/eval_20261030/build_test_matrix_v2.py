"""Assemble Comprehensive 30,000-Row Test Audit Matrix v2.

Directly addresses User Request & Agente 3 Task 5:
- 30,000 rows by ID (1..30000).
- Columns:
  * id, item_id, seller_id, category_id, is_seen, pred_48
  * Model test probabilities with uniform polarity used=1 (p_dual_catboost_used, p_strong_ensemble_used, p_posterior_mcmc_used)
  * Public scored block membership columns (one per scored submission where flips differ from 48)
  * Draft candidate block membership columns (prefixed draft_)
  * availability_mask
- Accompanied by provenance JSON describing every column (source file + SHA256).
"""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "ap_2026_q_2" / "raw"
REV_DIR = ROOT / "results" / "revision_submissions"
EVAL_DIR = ROOT / "results" / "eval_20261030"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def compute_file_sha256(path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_flips_from_csv(csv_path, s48_dict):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    ids = df["ID"].astype(int).values
    conds = df["condition"].astype(str).str.strip().values
    s48_vals = np.array([s48_dict[i] for i in ids])
    return set(ids[conds != s48_vals])

def main():
    print("Building 30,000-row test matrix v2...", flush=True)

    # 1. Load S48
    s48_path = REV_DIR / "posterior_flips_20261027" / "s48.json"
    s48_dict = {int(k): v for k, v in json.load(open(s48_path)).items()}

    # 2. Train sellers
    train_sellers = set()
    for line in open(DATA_DIR / "train_data.jsonlines", encoding="utf-8"):
        train_sellers.add(str(json.loads(line)["seller_id"]))

    # 3. Test rows
    test_ids = []
    item_ids = []
    seller_ids = []
    category_ids = []
    is_seens = []

    for idx, line in enumerate(open(DATA_DIR / "test_data.jsonlines", encoding="utf-8"), start=1):
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
        "pred_48": [s48_dict[i] for i in test_ids],
    })

    provenance = {
        "id": {"description": "Kaggle submission ID (1 to 30000 sequential)", "source": "test line index"},
        "item_id": {"description": "MercadoLibre item ID MLA...", "source": "test_data.jsonlines", "sha256": compute_file_sha256(DATA_DIR / "test_data.jsonlines")},
        "seller_id": {"description": "Seller ID", "source": "test_data.jsonlines"},
        "category_id": {"description": "Product category ID", "source": "test_data.jsonlines"},
        "is_seen": {"description": "1 if seller appears in 70k train, 0 if new seller", "source": "train_data.jsonlines", "sha256": compute_file_sha256(DATA_DIR / "train_data.jsonlines")},
        "pred_48": {"description": "Submission 48 condition (benchmark .94633)", "source": "s48.json", "sha256": compute_file_sha256(s48_path)},
    }

    # 4. Model Probabilities (Polarity uniform used=1)
    p_pipe_a_path = ROOT / "results" / "benchmark_20261030" / "test_probs_pipeline_A.npy"
    if p_pipe_a_path.exists():
        df_test["p_pipeline_A_used"] = np.load(p_pipe_a_path)
        provenance["p_pipeline_A_used"] = {"description": "P(used=1) from Restored Pipeline A Benchmark (Full 70k)", "source": str(p_pipe_a_path.name), "sha256": compute_file_sha256(p_pipe_a_path)}

    p_comp_path = ROOT / "results" / "complementary_20261030" / "test_probs_complementary.npy"
    if p_comp_path.exists():
        df_test["p_complementary_used"] = np.load(p_comp_path)
        provenance["p_complementary_used"] = {"description": "P(used=1) from Restored Complementary Ensemble (CB+LGB, Full 70k)", "source": str(p_comp_path.name), "sha256": compute_file_sha256(p_comp_path)}

    p_dual_path = REV_DIR / "dual_catboost_test_probs.npy"
    if p_dual_path.exists():
        df_test["p_dual_catboost_used"] = np.load(p_dual_path)
        provenance["p_dual_catboost_used"] = {"description": "P(used=1) from Dual CatBoost R150", "source": str(p_dual_path.name), "sha256": compute_file_sha256(p_dual_path)}

    p_ens_path = REV_DIR / "strong_ensemble_test_probs.npy"
    if p_ens_path.exists():
        df_test["p_strong_ensemble_used"] = np.load(p_ens_path)
        provenance["p_strong_ensemble_used"] = {"description": "P(used=1) from Strong Ensemble R157", "source": str(p_ens_path.name), "sha256": compute_file_sha256(p_ens_path)}

    p_post_path = REV_DIR / "posterior_flips_20261027" / "posterior_por_id.csv"
    if p_post_path.exists():
        df_post = pd.read_csv(p_post_path)
        post_map = {}
        for _, row in df_post.iterrows():
            i = int(row["ID"])
            p_mal = float(row["P_48_mal"])
            post_map[i] = (1.0 - p_mal) if s48_dict[i] == "used" else p_mal
        df_test["p_posterior_mcmc_used"] = [post_map.get(i, np.nan) for i in df_test["id"]]
        provenance["p_posterior_mcmc_used"] = {"description": "P(used=1) from MCMC posterior where tested historically", "source": str(p_post_path.name), "sha256": compute_file_sha256(p_post_path)}

    # 5. Public Scored Blocks (from scored_registry.json)
    reg_path = EVAL_DIR / "scored_registry.json"
    if reg_path.exists():
        reg_data = json.load(open(reg_path))
        for entry in reg_data["entries"]:
            k = entry["key"]
            flips = set(entry["flipped_ids"])
            if len(flips) > 0 and len(flips) < 500: # Exclude massive 39
                col_name = f"block_pub_{k}"
                df_test[col_name] = df_test["id"].isin(flips).astype(int)
                provenance[col_name] = {
                    "description": f"1 if flipped vs 48 in public submission {k} (Score: {entry['score']}, Delta: {entry['delta_vs_48']:+.5f})",
                    "n_flips": len(flips),
                    "status": entry["status"]
                }

    # 6. Draft Candidates in submissions/2026102*
    draft_files = {
        "draft_sub64_dual_cb_pure": ROOT / "submissions" / "20261029" / "submission_64_dual_catboost_pure.csv",
        "draft_sub65_dual_cb_new_margin": ROOT / "submissions" / "20261029" / "submission_65_dual_catboost_new_sellers_top_margin.csv",
        "draft_sub66_probe1_new_to_new": ROOT / "submissions" / "20261029" / "submission_66_probe1_new_sellers_to_new.csv",
        "draft_sub67_probe2_new_to_used": ROOT / "submissions" / "20261029" / "submission_67_probe2_new_sellers_to_used.csv",
        "draft_sub68_probe3_virgin_lexical": ROOT / "submissions" / "20261029" / "submission_68_probe3_alt_virgin_lexical.csv",
        "draft_sub68_probe3_refuted_seller": ROOT / "submissions" / "20261029" / "submission_68_probe3_seen_sellers_to_used.csv",
        "draft_sub69_probe4_unanimous": ROOT / "submissions" / "20261029" / "submission_69_probe4_unanimous_consensus.csv",
        "draft_sub71_strong_ensemble": ROOT / "submissions" / "20261029" / "submission_71_strong_ensemble_candidate.csv",
    }
    for col_name, path in draft_files.items():
        if path.exists():
            flips = get_flips_from_csv(path, s48_dict)
            df_test[col_name] = df_test["id"].isin(flips).astype(int)
            provenance[col_name] = {
                "description": f"1 if flipped vs 48 in offline draft {path.name}",
                "n_flips": len(flips),
                "sha256": compute_file_sha256(path)
            }

    # 7. Availability mask
    masks = []
    has_pipe_a = "p_pipeline_A_used" in df_test.columns
    has_comp = "p_complementary_used" in df_test.columns
    has_dual = "p_dual_catboost_used" in df_test.columns
    has_ens = "p_strong_ensemble_used" in df_test.columns
    for _, row in df_test.iterrows():
        flags = []
        if has_pipe_a and not np.isnan(row["p_pipeline_A_used"]):
            flags.append("PIPE_A")
        if has_comp and not np.isnan(row["p_complementary_used"]):
            flags.append("ENS_COMP")
        if has_dual and not np.isnan(row["p_dual_catboost_used"]):
            flags.append("CB_DUAL")
        if has_ens and not np.isnan(row["p_strong_ensemble_used"]):
            flags.append("ENS_R157")
        if "p_posterior_mcmc_used" in df_test.columns and not np.isnan(row["p_posterior_mcmc_used"]):
            flags.append("MCMC_POST")
        masks.append("+".join(flags))
    df_test["availability_mask"] = masks
    provenance["availability_mask"] = {"description": "String flags indicating which model test probabilities are present"}

    # Save CSV.GZ
    out_csv_gz = EVAL_DIR / "test_matrix_by_id.csv.gz"
    df_test.to_csv(out_csv_gz, index=False, compression="gzip")
    print(f"Saved test matrix v2 to {out_csv_gz} ({out_csv_gz.stat().st_size / (1024*1024):.2f} MB)")

    # Save provenance JSON
    prov_path = EVAL_DIR / "test_matrix_provenance.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Saved test matrix provenance to {prov_path}")

if __name__ == "__main__":
    main()
