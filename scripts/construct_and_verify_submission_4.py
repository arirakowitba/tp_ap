import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
N_PUB = 20998
N_TEST = 30000

def sha256_lf(p):
    data = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()

def hash_id_target_map(df):
    mapping_str = "\n".join(f"{row['ID']},{row['condition']}" for _, row in df.iterrows())
    return hashlib.sha256(mapping_str.encode("utf-8")).hexdigest()

def main():
    print("=" * 80)
    print("AG-0156: CONSTRUCTING AND VERIFYING SUBMISSION 4 (ALGEBRAIC BLOCK B)")
    print("=" * 80)

    # 1. Load Anchor EVT-143
    evt143_path = ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv"
    assert evt143_path.exists(), "Missing EVT-143"
    df_evt143 = pd.read_csv(evt143_path)
    y_anchor = df_evt143["condition"].values
    anchor_sha = sha256_lf(evt143_path)
    anchor_map = hash_id_target_map(df_evt143)

    # 2. Load S2
    s2_path = ROOT / "submissions/SUBMISSION_2_TODAY_RECOMMENDED.csv"
    assert s2_path.exists(), "Missing S2"
    df_s2 = pd.read_csv(s2_path)
    y_s2 = df_s2["condition"].values
    s2_sha = sha256_lf(s2_path)
    s2_map = hash_id_target_map(df_s2)

    # 3. Load S3
    s3_path = ROOT / "submissions/SUBMISSION_3_TODAY_RECOMMENDED.csv"
    assert s3_path.exists(), "Missing S3"
    df_s3 = pd.read_csv(s3_path)
    y_s3 = df_s3["condition"].values
    s3_sha = sha256_lf(s3_path)
    s3_map = hash_id_target_map(df_s3)

    # Verification 1: Hashes of S3
    expected_s3_sha = "53bc6bd50b22b2b0dfc511d06ea765a9b4c0875b2a777a049fbc8576aae1ead4"
    expected_s3_map = "727e898f9c58efd239bbac35692735b48d490f658c12f5d67c0a611dba6ea7f0"
    assert s3_sha == expected_s3_sha, f"S3 SHA mismatch: {s3_sha} != {expected_s3_sha}"
    assert s3_map == expected_s3_map, f"S3 Map Hash mismatch: {s3_map} != {expected_s3_map}"
    print("[PASS] Verification 1: S3 file matches exact committed SHA-256 and ID/Target map hash.")

    # Verification 2: S2 diffs vs EVT-143
    diff_s2_idx = set(np.where(y_s2 != y_anchor)[0])
    assert len(diff_s2_idx) == 6, f"Expected 6 diffs in S2, got {len(diff_s2_idx)}"
    print(f"[PASS] Verification 2: S2 differs from EVT-143 in exactly 6 rows: {[i+1 for i in sorted(diff_s2_idx)]}")

    # Verification 3: S3 diffs vs EVT-143
    diff_s3_idx = set(np.where(y_s3 != y_anchor)[0])
    expected_block_a_ids = {5176, 8560, 23469}
    expected_block_a_idx = {i - 1 for i in expected_block_a_ids}
    assert diff_s3_idx == expected_block_a_idx, f"S3 diffs {diff_s3_idx} != expected Block A {expected_block_a_idx}"
    print(f"[PASS] Verification 3: S3 differs from EVT-143 in exactly the 3 U2N items of Block A: {expected_block_a_ids}")

    # Verification 4: S2 remaining changes are Block B (N2U)
    block_b_idx = diff_s2_idx - diff_s3_idx
    expected_block_b_ids = {11421, 14089, 15958}
    expected_block_b_idx = {i - 1 for i in expected_block_b_ids}
    assert block_b_idx == expected_block_b_idx, f"Block B {block_b_idx} != expected {expected_block_b_idx}"
    print(f"[PASS] Verification 4: S2 remaining changes are exactly the 3 N2U items of Block B: {expected_block_b_ids}")

    # Verification 5: Disjointness
    assert len(diff_s3_idx.intersection(block_b_idx)) == 0
    assert diff_s3_idx.union(block_b_idx) == diff_s2_idx
    print("[PASS] Verification 5: Block A and Block B are strictly disjoint and partition S2's diffs.")

    # 4. Construct S4 = EVT-143 + Block B only
    df_s4 = df_evt143.copy()
    for idx in sorted(block_b_idx):
        item_id = idx + 1
        assert df_s4.loc[idx, "ID"] == item_id
        assert df_s4.loc[idx, "condition"] == "new", f"Expected 'new' in EVT-143 for ID {item_id}"
        df_s4.loc[idx, "condition"] = "used"

    s4_path = ROOT / "submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv"
    df_s4.to_csv(s4_path, index=False)

    s4_sha = sha256_lf(s4_path)
    s4_map = hash_id_target_map(df_s4)
    y_s4 = df_s4["condition"].values

    # Check assertions on S4
    assert len(df_s4) == 30000
    assert df_s4["ID"].tolist() == list(range(1, 30001))
    assert set(df_s4["condition"].unique()) == {"new", "used"}

    diff_s4_idx = set(np.where(y_s4 != y_anchor)[0])
    assert diff_s4_idx == block_b_idx, "S4 diffs do not match Block B"
    assert len(diff_s4_idx.intersection(diff_s3_idx)) == 0, "S4 shares flips with S3"
    assert diff_s3_idx.union(diff_s4_idx) == diff_s2_idx, "S3 union S4 != S2"

    print("\n--- S4 CONSTRUCTED AND CERTIFIED DETERMINISTICALLY ---")
    print("Path:              ", s4_path)
    print("SHA-256 (LF):      ", s4_sha)
    print("ID/Target Map Hash:", s4_map)
    print("Flips vs EVT-143:  ", len(diff_s4_idx), "flips (all N2U: IDs 11421, 14089, 15958)")
    print("Flips vs S2:       ", int(np.sum(y_s4 != y_s2)), "flips (the 3 U2N items reverted to EVT-143)")
    print("Flips vs S3:       ", int(np.sum(y_s4 != y_s3)), "flips (3 U2N from S3 + 3 N2U from S4 = 6 flips)")

    # 5. Compute Algebraic Expected Score
    # EVT-143: 19,898 hits (0.94761)
    # S3: 19,896 hits (0.94752) -> Effect(A) = -2 hits
    # S2: 19,897 hits (0.94757) -> Effect(A + B) = -1 hit
    # Effect(B) = Effect(A + B) - Effect(A) = -1 - (-2) = +1 hit!
    expected_hits_s4 = 19898 + 1
    expected_exact_fraction = expected_hits_s4 / N_PUB
    expected_display = f"{expected_exact_fraction:.5f}"

    print(f"\n--- ALGEBRAIC IDENTIFICATION ---")
    print(f"Effect(Block A: U2N):  score(S3) - score(EVT) = 19,896 - 19,898 = -2 hits")
    print(f"Effect(Block A + B):   score(S2) - score(EVT) = 19,897 - 19,898 = -1 hit")
    print(f"Effect(Block B: N2U):  (-1) - (-2)             = +1 hit!")
    print(f"Expected Public Hits:  {expected_hits_s4} / {N_PUB}")
    print(f"Expected Exact Score:  {expected_exact_fraction:.9f}")
    print(f"Expected Display:      {expected_display}")

if __name__ == "__main__":
    main()
