"""Generador y verificador formal de H-AVG (0.5 B10 + 0.5 Variant C) para test set (30.000 filas).

Cumple los requerimientos de CX-0016 y CL-0014:
- Combina estrictamente las probabilidades de B10 full70k (Claude CL-0005) y Variant C full70k (Antigravity AG-0013).
- Genera el vector de inferencia para test set con umbral 0.50 (th=0.50 fijo, sin optimización en confirm).
- Valida con `submission_guard.py` (30.000 filas, IDs 1..30000, columna used, sin NaNs).
- Mide la distancia de Hamming frente a S48 y clasifica los desacuerdos entre:
  * Probados históricamente (04-76).
  * Nunca probados (vistos y nuevos).
- El CSV generado se almacena en disco para decisión exclusiva de Ariel (cero uploads a Kaggle).
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.coordination_lock import ProjectLock
from src.eval_20261030.submission_guard import check_submission_file, sha256_bytes
from src.intensive_20261030.shared_cache import build_or_load_cache

OUT_DIR = ROOT / "results" / "intensive_20261030"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_OUT = OUT_DIR / "candidate_model_havg_b10_variant_c_full70k.csv"
META_OUT = OUT_DIR / "candidate_model_havg_metadata.json"


def build_and_verify_havg():
    lock = ProjectLock(owner_id="antigravity_builder_havg")
    print("Acquiring ProjectLock for H-AVG build and verification...", flush=True)
    with lock:
        # 1. Load B10 test probabilities from Claude branch
        print("Loading B10 test probabilities from origin/claude/posterior-y-modelos-20261027...", flush=True)
        res = subprocess.run(
            ["git", "show", "origin/claude/posterior-y-modelos-20261027:results/round141_claude_baseline/full70k/test_p.npy"],
            capture_output=True,
            check=True,
        )
        p_b10 = np.load(io.BytesIO(res.stdout)).astype(np.float32)
        assert len(p_b10) == 30000, f"Expected 30000 rows for B10, got {len(p_b10)}"

        # 2. Load Variant C test probabilities
        p_vc_path = OUT_DIR / "test_probs_variant_c_inductive_70k.npy"
        print(f"Loading Variant C test probabilities from {p_vc_path}...", flush=True)
        p_vc = np.load(p_vc_path).astype(np.float32)
        assert len(p_vc) == 30000, f"Expected 30000 rows for Variant C, got {len(p_vc)}"

        # 3. Simple unweighted average H-AVG (0.5 B10 + 0.5 Variant C)
        p_havg = 0.5 * p_b10 + 0.5 * p_vc
        out_probs_path = OUT_DIR / "test_probs_havg_b10_variant_c_inductive_70k.npy"
        np.save(out_probs_path, p_havg)

        # 4. Generate candidate CSV with fixed threshold 0.50 and canonical competition columns
        pred_havg = (p_havg >= 0.50).astype(int)
        df_sub = pd.DataFrame({
            "ID": np.arange(1, 30001),
            "condition": ["used" if p == 1 else "new" for p in pred_havg],
        })
        df_sub.to_csv(SUB_OUT, index=False)
        sub_sha = sha256_bytes(SUB_OUT.read_bytes())
        print(f"Generated {SUB_OUT.name} (SHA256: {sub_sha})", flush=True)

        # 5. Audit overlap with 04-76
        reg = json.load(open(ROOT / "results" / "eval_20261030" / "scored_registry.json"))
        ever_probed = set()
        for e in reg["entries"]:
            ever_probed.update(e.get("flipped_ids", []))

        s48_raw = json.load(open(ROOT / "results" / "revision_submissions" / "posterior_flips_20261027" / "s48.json"))
        s48_dict = {int(k): v for k, v in s48_raw.items()}
        s48_arr = np.array([1 if s48_dict[i] == "used" else 0 for i in range(1, 30001)])

        # Validate with submission_guard
        print("Validating with submission_guard...", flush=True)
        ok, guard_msg = check_submission_file(SUB_OUT, reg, s48_dict, verbose=True)
        assert ok, f"Submission guard failed: {guard_msg}"
        print(f"  Submission guard passed successfully! ({guard_msg})", flush=True)

        diff_ids = np.where(pred_havg != s48_arr)[0] + 1
        diff_probed = [i for i in diff_ids if i in ever_probed]
        diff_never_probed = [i for i in diff_ids if i not in ever_probed]

        cache = build_or_load_cache()
        sellers_tr = pd.Series(cache["sellers_tr"])
        sellers_te = cache["sellers_te"]
        freq_in_tr = pd.Series(sellers_te).map(sellers_tr.value_counts()).fillna(0).astype(int).values
        is_seen = (freq_in_tr > 0).astype(int)

        np_seen = [i for i in diff_never_probed if is_seen[i - 1] == 1]
        np_new = [i for i in diff_never_probed if is_seen[i - 1] == 0]

        meta = {
            "model_name": "H_AVG_0.5_B10_plus_0.5_Variant_C",
            "weights": {"b10": 0.5, "variant_c": 0.5},
            "threshold": 0.50,
            "sub_file": SUB_OUT.name,
            "sub_sha256": sub_sha,
            "probs_file": out_probs_path.name,
            "n_flips_vs_48": len(diff_ids),
            "probed_in_04_76": len(diff_probed),
            "never_probed": len(diff_never_probed),
            "never_probed_seen": len(np_seen),
            "never_probed_new": len(np_new),
            "never_probed_seen_ids": [int(x) for x in np_seen],
            "p_havg_mean": float(np.mean(p_havg)),
            "p_havg_std": float(np.std(p_havg)),
        }
        META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Metadata written to {META_OUT.name}", flush=True)


if __name__ == "__main__":
    build_and_verify_havg()
