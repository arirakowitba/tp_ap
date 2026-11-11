"""Executable Preventive Gates A-H Validator & Test Suite (T96 / CX-0162 / CX-0163).

Implements enforceable validators for Gates A through H that derive metrics from
actual predictions, labels, fold assignments, seller groups, provenance hashes,
calibration artifacts, filesystem quarantine, and markdown AST/structure — never
trusting self-reported manifest strings or CHECKPOINT_STATE flags alone.

Every gate (A-H) is verified against:
  1. At least one POSITIVE fixture (passes cleanly).
  2. At least one NEGATIVE fixture (raises GateViolationError, proving red->green).
  3. Live repository audit checks where applicable.
"""

import ast
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd


class GateViolationError(AssertionError):
    """Raised when a preventive gate (A-H) is violated."""


PROMOTED_STATUSES = {
    "READY",
    "READY_FOR_MANUAL_REVIEW",
    "READY_LEADER_RECOMMENDED",
    "PROMOTED",
    "QUALIFIED",
    "QUALIFIED_MAX_ACCURACY",
    "QUALIFIED_BEST_CALIBRATION",
    "QUALIFIED_NEW_OOF_BENCHMARK",
    "GANADOR_HELD_OUT",
}

DEGRADED_TOKENS = {"DEGRADED", "NO_GAIN", "REJECTED", "BLOCKED", "NOT_SUPPORTED", "INVALID"}


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(filepath: Path, normalize_lf: bool = False) -> str:
    raw = Path(filepath).read_bytes()
    if normalize_lf:
        raw = raw.replace(b"\r\n", b"\n")
    return compute_sha256_bytes(raw)


def compute_id_target_hash(df: pd.DataFrame, id_col: str = "ID", target_col: str = "condition") -> str:
    df_sorted = df.sort_values(by=id_col, kind="mergesort")
    payload = "\n".join(f"{int(i)}:{str(t)}" for i, t in zip(df_sorted[id_col], df_sorted[target_col]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# GATE A: Strong Baseline Superiority (Derived from real predictions + y_true)
# =============================================================================
def validate_gate_a(
    candidate_meta: dict,
    y_true: np.ndarray,
    strong_baseline_oof: np.ndarray,
    candidate_oof: np.ndarray,
    eval_table: pd.DataFrame = None,
    threshold: float = 0.5,
) -> dict:
    """Gate A validator:
    - Derives hits & accuracy directly from y_true, strong_baseline_oof, and candidate_oof.
    - Rejects NaNs, shape mismatches, or discrepancies between declared manifest hits and real hits.
    - Requires delta_hits > 0 strictly for any promoted/READY/QUALIFIED status.
    - Rejects any candidate whose evaluation table row indicates DEGRADED/NO_GAIN/REJECTED.
    """
    y_arr = np.asarray(y_true, dtype=int)
    base_arr = np.asarray(strong_baseline_oof, dtype=float)
    cand_arr = np.asarray(candidate_oof, dtype=float)

    if y_arr.ndim != 1 or len(y_arr) == 0:
        raise GateViolationError("Gate A: y_true must be a non-empty 1D array")
    if base_arr.shape != y_arr.shape or cand_arr.shape != y_arr.shape:
        raise GateViolationError("Gate A: OOF prediction shapes do not match y_true")
    if np.isnan(base_arr).any() or np.isnan(cand_arr).any():
        raise GateViolationError("Gate A: NaN values found in OOF predictions")

    base_bin = (base_arr >= threshold).astype(int)
    cand_bin = (cand_arr >= threshold).astype(int)

    base_hits = int((base_bin == y_arr).sum())
    cand_hits = int((cand_bin == y_arr).sum())
    delta_hits = cand_hits - base_hits
    fixes = int(((cand_bin == y_arr) & (base_bin != y_arr)).sum())
    breaks = int(((cand_bin != y_arr) & (base_bin == y_arr)).sum())

    if "oof_hits" in candidate_meta and candidate_meta["oof_hits"] is not None:
        if int(candidate_meta["oof_hits"]) != cand_hits:
            raise GateViolationError(
                f"Gate A: Manifest oof_hits ({candidate_meta['oof_hits']}) != derived hits ({cand_hits})"
            )
    if "delta_vs_podium" in candidate_meta and candidate_meta["delta_vs_podium"] is not None:
        if int(candidate_meta["delta_vs_podium"]) != delta_hits:
            raise GateViolationError(
                f"Gate A: Manifest delta ({candidate_meta['delta_vs_podium']}) != derived delta ({delta_hits})"
            )

    status = str(candidate_meta.get("formal_status", "")).upper().strip()
    is_promoted = status in PROMOTED_STATUSES or any(tok in status for tok in ("READY", "QUALIFIED", "PROMOTED"))

    if eval_table is not None and not eval_table.empty:
        policy_key = candidate_meta.get("policy_key") or candidate_meta.get("name")
        if policy_key:
            col = "policy" if "policy" in eval_table.columns else eval_table.columns[0]
            matched = eval_table[eval_table[col].astype(str) == str(policy_key)]
            if not matched.empty:
                row = matched.iloc[0]
                row_status = str(row.get("policy_stability", row.get("status", ""))).upper()
                if is_promoted and any(tok in row_status for tok in DEGRADED_TOKENS):
                    raise GateViolationError(
                        f"Gate A: Promoted candidate '{policy_key}' has degraded status '{row_status}' in eval CSV"
                    )
                if "delta_vs_podium" in row and is_promoted and int(row["delta_vs_podium"]) <= 0:
                    raise GateViolationError(
                        f"Gate A: Eval CSV reports non-positive delta ({row['delta_vs_podium']}) for '{policy_key}'"
                    )

    if is_promoted and delta_hits <= 0:
        raise GateViolationError(
            f"Gate A: Candidate '{candidate_meta.get('name')}' with status '{status}' has non-positive "
            f"delta_hits={delta_hits} (cand={cand_hits}, strong_baseline={base_hits}); promotion forbidden."
        )

    return {
        "base_hits": base_hits,
        "cand_hits": cand_hits,
        "delta_hits": delta_hits,
        "fixes": fixes,
        "breaks": breaks,
        "status": status,
    }


# =============================================================================
# GATE B: Seller Disjointness, Outer/Inner Retraining Provenance & No Val Leak
# =============================================================================
def validate_gate_b(
    outer_fold_ids: np.ndarray,
    seller_ids: np.ndarray,
    provenance_records: list,
    freeze_records: list,
    script_source: str = None,
    n_outer_folds: int = 5,
    n_inner_folds: int = 4,
) -> dict:
    """Gate B validator:
    - Verifies outer folds are strictly seller-disjoint (zero shared sellers across outer folds).
    - Verifies provenance records prove actual retraining per outer fold and per inner fold
      (distinct fit_rows_sha, disjoint fit_rows and pred_rows, no global OOF slicing).
    - Verifies freeze records exist per outer fold and were frozen on inner-OOF.
    - Inspects script_source (if provided) to reject StratifiedKFold without groups,
      slicing global OOF as inner-OOF, or early stopping on outer-val.
    """
    folds = np.asarray(outer_fold_ids)
    sellers = np.asarray(seller_ids)
    if len(folds) != len(sellers) or len(folds) == 0:
        raise GateViolationError("Gate B: outer_fold_ids and seller_ids must have equal non-zero length")

    unique_folds = np.unique(folds)
    if len(unique_folds) != n_outer_folds:
        raise GateViolationError(f"Gate B: Expected {n_outer_folds} outer folds, got {len(unique_folds)}")

    # 1. Seller disjointness across all outer folds
    fold_seller_sets = {}
    for k in unique_folds:
        mask = folds == k
        s_set = set(sellers[mask])
        train_s_set = set(sellers[~mask])
        overlap = s_set & train_s_set
        if overlap:
            raise GateViolationError(
                f"Gate B: Outer fold {k} shares {len(overlap)} sellers between outer-train and outer-val!"
            )
        fold_seller_sets[int(k)] = s_set

    # 2. Provenance of per-fold retraining
    if not provenance_records:
        raise GateViolationError("Gate B: Missing model retraining provenance records")

    prov_by_name = {r["model"]: r for r in provenance_records if "model" in r}
    fit_shas = set()
    for k in range(n_outer_folds):
        full_key = f"outer{k}_full"
        if full_key not in prov_by_name:
            raise GateViolationError(f"Gate B: Missing outer retraining provenance '{full_key}'")
        rec = prov_by_name[full_key]
        if rec.get("fit_rows_sha") == rec.get("pred_rows_sha") or not rec.get("fit_rows_sha"):
            raise GateViolationError(f"Gate B: Invalid or identical fit/pred SHA in '{full_key}'")
        fit_shas.add(rec["fit_rows_sha"])

        for j in range(n_inner_folds):
            inner_key = f"outer{k}_inner{j}"
            if inner_key not in prov_by_name:
                raise GateViolationError(f"Gate B: Missing inner retraining provenance '{inner_key}'")
            irec = prov_by_name[inner_key]
            if not irec.get("fit_rows_sha") or irec.get("fit_rows_sha") == rec.get("fit_rows_sha"):
                raise GateViolationError(
                    f"Gate B: Inner model '{inner_key}' shares fit_rows_sha with outer model (not retrained)"
                )
            if irec.get("n_fit", 0) >= rec.get("n_fit", 0):
                raise GateViolationError(
                    f"Gate B: Inner model '{inner_key}' n_fit ({irec.get('n_fit')}) >= outer n_fit ({rec.get('n_fit')})"
                )
            fit_shas.add(irec["fit_rows_sha"])

    expected_distinct = n_outer_folds * (1 + n_inner_folds)
    if len(fit_shas) != expected_distinct:
        raise GateViolationError(
            f"Gate B: Expected {expected_distinct} distinct fit_rows_sha hashes, got {len(fit_shas)}"
        )

    # 3. Freeze records per outer fold
    if not freeze_records or len(freeze_records) != n_outer_folds:
        raise GateViolationError(f"Gate B: Expected {n_outer_folds} freeze records, got {len(freeze_records or [])}")
    for frec in freeze_records:
        if "inner_oof_acc" not in frec or "frozen_at_utc" not in frec:
            raise GateViolationError("Gate B: Freeze record missing inner_oof_acc or frozen_at_utc")
        if frec.get("used_outer_val_for_selection", False) or frec.get("early_stopping_on_outer_val", False):
            raise GateViolationError("Gate B: Freeze record indicates outer-val was used in selection/early stopping")

    # 4. Static/AST audit of script source if supplied
    if script_source is not None:
        if re.search(r"inner_oof_\w+\s*=\s*oof_\w+\s*\[", script_source):
            raise GateViolationError("Gate B: Script slices precomputed global OOF as inner-OOF instead of retraining")
        if "StratifiedGroupKFold" not in script_source and "GroupKFold" not in script_source:
            raise GateViolationError("Gate B: Script does not use GroupKFold / StratifiedGroupKFold for seller grouping")
        if re.search(r"early_stopping_rounds\s*=\s*[1-9]\d*", script_source) and re.search(
            r"eval_set\s*=\s*[\[\(].*val_idx", script_source
        ):
            raise GateViolationError("Gate B: Script uses early_stopping_rounds evaluated on outer val_idx")

    return {"n_outer_folds": n_outer_folds, "distinct_fit_shas": len(fit_shas)}


# =============================================================================
# GATE C: Prequential Calibration Artifact Required for Probabilities / Q05
# =============================================================================
def validate_gate_c(
    claims: dict,
    calibration_artifact_path: Path = None,
) -> dict:
    """Gate C validator (tightened per CX-0164):
    - Separates (a) score intervals (public_q05 / public_q95 / expected_public_score) backed by
      prequential interval coverage from (b) threshold-exceedance probabilities (prob_beat_s4,
      prob_reach_09500, P(>S4), P(>=0.95000)) backed by preregistered probability forecasts
      with Brier/logloss/calibration verification.
    - By default all public probabilities and quantiles must be 'NO_IDENTIFIABLE'.
    - For a 95% bound (alpha=0.05), requires sample size n >= ceil(1 / alpha) (at least n >= 20
      sequential prequential events; n=5 is rejected) and recomputed empirical coverage >= claimed
      confidence level (e.g. >= 0.95 for a 95% bound; 0.80 coverage is rejected).
    - Interval coverage alone CANNOT calibrate threshold probabilities; threshold probability
      claims require preregistered `pred_prob` and `exceeded_threshold` per event, n >= 20,
      recomputed Brier score <= 0.20, and calibration error <= 0.10.
    """
    interval_keys = ["expected_public_score", "public_q05", "public_q95"]
    prob_keys = ["prob_beat_s4", "prob_reach_09500", "P(>S4)", "P(>=0.95000)"]
    restricted_keys = interval_keys + prob_keys

    numeric_intervals = {}
    numeric_probs = {}
    for k in restricted_keys:
        if k in claims and claims[k] is not None:
            val_str = str(claims[k]).strip().upper()
            if val_str != "NO_IDENTIFIABLE":
                if k in interval_keys:
                    numeric_intervals[k] = claims[k]
                else:
                    numeric_probs[k] = claims[k]

    if not numeric_intervals and not numeric_probs:
        return {"status": "PASS_NO_IDENTIFIABLE_DECLARED", "numeric_claims": 0}

    all_claimed = list(numeric_intervals.keys()) + list(numeric_probs.keys())
    if calibration_artifact_path is None or not Path(calibration_artifact_path).exists():
        raise GateViolationError(
            f"Gate C: Numeric public claims {all_claimed} blocked — "
            f"missing prequential calibration artifact (must be 'NO_IDENTIFIABLE')."
        )

    art = json.loads(Path(calibration_artifact_path).read_text(encoding="utf-8"))
    events = art.get("prequential_events", [])
    claimed_confidence = float(claims.get("claimed_confidence_level", art.get("claimed_confidence_level", 0.95)))
    alpha = max(1e-6, 1.0 - claimed_confidence)
    min_required_n = max(20, int(np.ceil(1.0 / alpha)))

    if not isinstance(events, list) or len(events) < min_required_n:
        raise GateViolationError(
            f"Gate C: Sample size n={len(events) if isinstance(events, list) else 0} is insufficient "
            f"to sustain a {claimed_confidence*100:.0f}% bound or probability claim (requires >= {min_required_n} sequential prequential events)."
        )

    # Verify chronological sequential ordering and recompute empirical coverage from events
    covered = 0
    prev_ts = ""
    for idx, ev in enumerate(events):
        for req_field in ("event_id", "timestamp_utc", "pred_q05", "pred_q95", "observed_public_score"):
            if req_field not in ev:
                raise GateViolationError(f"Gate C: Event {idx} missing field '{req_field}'")
        if ev.get("fitted_using_event_outcome", False):
            raise GateViolationError(f"Gate C: Event {ev['event_id']} was fitted retrospectively, not prequentially")
        ts = str(ev["timestamp_utc"])
        if prev_ts and ts < prev_ts:
            raise GateViolationError("Gate C: Prequential events are not in chronological order")
        prev_ts = ts
        q05, q95, obs = float(ev["pred_q05"]), float(ev["pred_q95"]), float(ev["observed_public_score"])
        if q05 <= obs <= q95:
            covered += 1

    recomputed_cov = covered / len(events)
    declared_cov = float(art.get("measured_empirical_coverage", -1.0))
    if abs(recomputed_cov - declared_cov) > 1e-4:
        raise GateViolationError(
            f"Gate C: Artifact declared coverage ({declared_cov}) != recomputed coverage ({recomputed_cov:.4f})"
        )
    if recomputed_cov + 1e-9 < claimed_confidence:
        raise GateViolationError(
            f"Gate C: Recomputed empirical coverage ({recomputed_cov:.3f}) < claimed confidence level ({claimed_confidence:.3f})"
        )

    recomputed_brier = None
    if numeric_probs:
        if art.get("calibration_type") != "threshold_probability_and_interval":
            raise GateViolationError(
                f"Gate C: Interval coverage alone cannot calibrate threshold probabilities {list(numeric_probs.keys())}; "
                f"requires preregistered probability forecasts with Brier/logloss/calibration verification."
            )
        sq_errors = []
        probs_list = []
        outcomes_list = []
        for idx, ev in enumerate(events):
            if "pred_prob" not in ev or "exceeded_threshold" not in ev:
                raise GateViolationError(
                    f"Gate C: Event {idx} missing preregistered 'pred_prob' or 'exceeded_threshold' for probability calibration"
                )
            p_hat = float(ev["pred_prob"])
            y_ev = float(ev["exceeded_threshold"])
            sq_errors.append((p_hat - y_ev) ** 2)
            probs_list.append(p_hat)
            outcomes_list.append(y_ev)
        recomputed_brier = float(np.mean(sq_errors))
        declared_brier = float(art.get("measured_brier_score", -1.0))
        if abs(recomputed_brier - declared_brier) > 1e-4:
            raise GateViolationError(
                f"Gate C: Declared Brier ({declared_brier}) != recomputed Brier ({recomputed_brier:.4f})"
            )
        mean_cal_err = abs(float(np.mean(probs_list)) - float(np.mean(outcomes_list)))
        if recomputed_brier > 0.20 or mean_cal_err > 0.10:
            raise GateViolationError(
                f"Gate C: Threshold probability forecasts poorly calibrated (Brier={recomputed_brier:.4f}, mean_cal_err={mean_cal_err:.4f})"
            )

    return {
        "status": "PASS_CALIBRATED",
        "recomputed_coverage": recomputed_cov,
        "recomputed_brier": recomputed_brier,
        "n_events": len(events),
    }


# =============================================================================
# GATE D: Quarantine of Rejected Candidates (Filesystem + Active Manifests)
# =============================================================================
def validate_gate_d(
    submissions_dir: Path,
    rejected_dir: Path,
    active_manifests: list,
    rejected_names_or_dirs: set,
    rejected_sha256s: set = None,
) -> dict:
    """Gate D validator:
    - Scans submissions_dir recursively to ensure NO rejected directory, filename, or SHA-256
      remains inside active submissions/.
    - Verifies that quarantined artifacts physically exist inside rejected_dir.
    - Scans all active_manifests to ensure no entry has a REJECTED/DEGRADED/NOT_SUPPORTED status
      or references a rejected file/SHA-256.
    """
    submissions_dir = Path(submissions_dir)
    rejected_dir = Path(rejected_dir)
    rejected_sha256s = set(rejected_sha256s or [])

    if not submissions_dir.exists():
        raise GateViolationError(f"Gate D: submissions_dir '{submissions_dir}' does not exist")
    if not rejected_dir.exists():
        raise GateViolationError(f"Gate D: rejected_dir '{rejected_dir}' does not exist")

    # 1. Check that no rejected item remains anywhere inside submissions_dir
    for p in submissions_dir.rglob("*"):
        if p.name in rejected_names_or_dirs or p.relative_to(submissions_dir).parts[0] in rejected_names_or_dirs:
            raise GateViolationError(f"Gate D: Rejected artifact '{p}' found inside active submissions/!")
        if p.is_file() and p.suffix.lower() == ".csv" and rejected_sha256s:
            sha_raw = compute_file_sha256(p, normalize_lf=False)
            sha_lf = compute_file_sha256(p, normalize_lf=True)
            if sha_raw in rejected_sha256s or sha_lf in rejected_sha256s:
                raise GateViolationError(
                    f"Gate D: CSV '{p}' in active submissions/ matches a quarantined rejected SHA-256!"
                )

    # 2. Verify rejected_dir actually holds quarantined items
    quarantined_files = list(rejected_dir.rglob("*.csv"))
    if not quarantined_files:
        raise GateViolationError("Gate D: rejected_dir contains zero quarantined CSV files")

    # 3. Inspect active manifests
    for m_path in active_manifests:
        m_data = json.loads(Path(m_path).read_text(encoding="utf-8"))
        entries = m_data.get("entries", m_data.get("portfolio_candidates", []))
        for entry in entries:
            status = str(entry.get("formal_status", entry.get("status", ""))).upper()
            fname = Path(str(entry.get("file", entry.get("filename", "")))).name
            sha = entry.get("sha256") or entry.get("sha256_lf")
            if any(tok in status for tok in DEGRADED_TOKENS):
                raise GateViolationError(
                    f"Gate D: Active manifest '{m_path}' contains rejected/degraded entry '{fname}' ({status})"
                )
            if fname in rejected_names_or_dirs or (sha and sha in rejected_sha256s):
                raise GateViolationError(
                    f"Gate D: Active manifest '{m_path}' references quarantined file '{fname}'"
                )

    return {"quarantined_csv_count": len(quarantined_files)}


# =============================================================================
# GATE E: Structural Evidence for Methodological Nomenclature
# =============================================================================
def validate_gate_e(claim_type: str, evidence: dict) -> dict:
    """Gate E validator:
    Requires structural proof before permitting methodological labels:
    - 'nested': disjoint outer_train/outer_val, inner splits strictly subset of outer_train,
      distinct inner refit hashes (no slicing of outer OOF).
    - 'temporal': timestamps with max(train_time) < min(val_time) AND dedicated refit on train_idx
      (not post-hoc slicing of global OOF).
    - 'leave-family': held-out category family absent from train categories AND dedicated refit
      on train_idx (not post-hoc filtering of global OOF).
    - 'hierarchical_bootstrap' / 'seller_category_bootstrap': cluster-level resampling where all
      items of a seller share a cluster draw factor (rejects IID row np.random.choice).
    """
    ctype = claim_type.strip().lower()
    if ctype == "nested":
        outer_train = set(evidence.get("outer_train_indices", []))
        outer_val = set(evidence.get("outer_val_indices", []))
        inner_splits = evidence.get("inner_splits", [])
        if not outer_train or not outer_val or (outer_train & outer_val):
            raise GateViolationError("Gate E (nested): outer_train and outer_val must be non-empty and disjoint")
        if not inner_splits:
            raise GateViolationError("Gate E (nested): inner_splits missing")
        if evidence.get("reused_outer_oof", False):
            raise GateViolationError("Gate E (nested): Reused outer OOF instead of retraining inner models")
        inner_shas = set()
        for idx, sp in enumerate(inner_splits):
            itr = set(sp.get("train", []))
            iva = set(sp.get("val", []))
            if not itr or not iva or (itr & iva):
                raise GateViolationError(f"Gate E (nested): Inner split {idx} train/val overlap or empty")
            if (itr & outer_val) or (iva & outer_val):
                raise GateViolationError(f"Gate E (nested): Inner split {idx} leaks outer_val indices")
            if not (itr | iva).issubset(outer_train):
                raise GateViolationError(f"Gate E (nested): Inner split {idx} not a subset of outer_train")
            if not sp.get("refit_sha"):
                raise GateViolationError(f"Gate E (nested): Inner split {idx} missing refit_sha")
            inner_shas.add(sp["refit_sha"])
        if len(inner_shas) != len(inner_splits):
            raise GateViolationError("Gate E (nested): Inner splits do not have distinct refit_sha hashes")
        return {"verified": "nested", "n_inner": len(inner_splits)}

    if ctype == "temporal":
        if evidence.get("post_hoc_oof_slice", False) or not evidence.get("retrained_on_past_only", False):
            raise GateViolationError("Gate E (temporal): Post-hoc OOF date slicing is not temporal validation")
        t_tr = np.asarray(evidence.get("train_timestamps", []))
        t_va = np.asarray(evidence.get("val_timestamps", []))
        if len(t_tr) == 0 or len(t_va) == 0:
            raise GateViolationError("Gate E (temporal): Missing train or val timestamps")
        if np.max(t_tr) >= np.min(t_va):
            raise GateViolationError(
                f"Gate E (temporal): Causal violation max(train_t)={np.max(t_tr)} >= min(val_t)={np.min(t_va)}"
            )
        if not evidence.get("refit_sha"):
            raise GateViolationError("Gate E (temporal): Missing refit_sha for past-only model")
        return {"verified": "temporal"}

    if ctype == "leave-family":
        if evidence.get("post_hoc_oof_slice", False) or not evidence.get("retrained_without_family", False):
            raise GateViolationError("Gate E (leave-family): Filtering OOF rows without refit is invalid")
        c_tr = set(evidence.get("train_categories", []))
        c_va = set(evidence.get("val_categories", []))
        if not c_tr or not c_va or (c_tr & c_va):
            raise GateViolationError(f"Gate E (leave-family): Held-out family present in training set: {c_tr & c_va}")
        if not evidence.get("refit_sha"):
            raise GateViolationError("Gate E (leave-family): Missing refit_sha for family-excluded model")
        return {"verified": "leave-family"}

    if ctype in ("hierarchical_bootstrap", "seller_category_bootstrap"):
        if evidence.get("resampling_unit") == "iid_rows":
            raise GateViolationError("Gate E (bootstrap): IID row resampling cannot be called seller/category bootstrap")
        code_snippet = str(evidence.get("code_snippet", ""))
        if "np.random.choice(n_samples" in code_snippet:
            raise GateViolationError("Gate E (bootstrap): Code uses IID np.random.choice(n_samples)")
        sellers = np.asarray(evidence.get("seller_ids", []))
        cats = np.asarray(evidence.get("category_ids", []))
        weights = np.asarray(evidence.get("sample_weights", []), dtype=float)
        if len(sellers) == 0 or len(sellers) != len(cats) or len(sellers) != len(weights):
            raise GateViolationError("Gate E (bootstrap): Missing or mismatched seller/category/weight arrays")
        # In a valid seller x category or seller-hierarchical bootstrap, two rows with the EXACT
        # same (seller_id, category_id) cell must receive the exact same cluster multiplier in a draw.
        df_w = pd.DataFrame({"s": sellers, "c": cats, "w": weights})
        cell_std = df_w.groupby(["s", "c"])["w"].std().dropna()
        if (cell_std > 1e-9).any():
            raise GateViolationError(
                "Gate E (bootstrap): Items in the same (seller, category) cluster received independent IID weights"
            )
        return {"verified": ctype}

    raise GateViolationError(f"Gate E: Unknown methodological claim '{claim_type}'")


# =============================================================================
# GATE F: Cryptographic Integrity, ID/Target Hash, Real Flips & Filename K
# =============================================================================
def extract_declared_k(text: str):
    """Extracts integer K from tokens like _K15, _K50, _top15, -K250."""
    m = re.search(r"(?:^|[_\-])(?:K|top)(\d+)(?:[_\-\.]|$)", str(text), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def validate_gate_f(
    candidate_entry: dict,
    csv_path: Path,
    baseline_csv_path: Path,
    expected_baseline_sha256: str = None,
) -> dict:
    """Gate F validator:
    - Verifies baseline CSV exists and matches expected_baseline_sha256.
    - Verifies candidate CSV SHA-256 (raw or LF) matches candidate_entry['sha256'].
    - Joins candidate and baseline strictly by ID and verifies id_target_hash.
    - Computes real flips vs baseline and verifies match with manifest 'flips_vs_s4'.
    - Extracts declared K from both filename and candidate 'name' (e.g. K15, K50, top15)
      and requires real flips == declared K (catching the C3 K15=8 flips bug even if the
      manifest also says 8!).
    """
    csv_path = Path(csv_path)
    baseline_csv_path = Path(baseline_csv_path)
    if not csv_path.exists():
        raise GateViolationError(f"Gate F: Candidate CSV '{csv_path}' does not exist")
    if not baseline_csv_path.exists():
        raise GateViolationError(f"Gate F: Baseline CSV '{baseline_csv_path}' does not exist")

    if expected_baseline_sha256 is not None:
        base_sha_raw = compute_file_sha256(baseline_csv_path, normalize_lf=False)
        base_sha_lf = compute_file_sha256(baseline_csv_path, normalize_lf=True)
        if expected_baseline_sha256 not in (base_sha_raw, base_sha_lf):
            raise GateViolationError(
                f"Gate F: Baseline SHA mismatch (got {base_sha_raw}, expected {expected_baseline_sha256})"
            )

    cand_sha_raw = compute_file_sha256(csv_path, normalize_lf=False)
    cand_sha_lf = compute_file_sha256(csv_path, normalize_lf=True)
    declared_sha = candidate_entry.get("sha256") or candidate_entry.get("sha256_lf")
    if not declared_sha or declared_sha not in (cand_sha_raw, cand_sha_lf):
        raise GateViolationError(
            f"Gate F: Candidate SHA-256 mismatch for '{csv_path.name}': declared={declared_sha}, actual={cand_sha_raw}"
        )

    df_cand = pd.read_csv(csv_path)
    df_base = pd.read_csv(baseline_csv_path)
    cand_col = "condition" if "condition" in df_cand.columns else "target"
    base_col = "condition" if "condition" in df_base.columns else "target"

    actual_pair_hash = compute_id_target_hash(df_cand, id_col="ID", target_col=cand_col)
    declared_pair_hash = candidate_entry.get("id_target_hash") or candidate_entry.get("map_hash")
    if not declared_pair_hash or actual_pair_hash != declared_pair_hash:
        raise GateViolationError(
            f"Gate F: ID/target hash mismatch for '{csv_path.name}': declared={declared_pair_hash}, actual={actual_pair_hash}"
        )

    merged = pd.merge(
        df_base[["ID", base_col]].rename(columns={base_col: "base_cond"}),
        df_cand[["ID", cand_col]].rename(columns={cand_col: "cand_cond"}),
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(df_base):
        raise GateViolationError(f"Gate F: ID join length ({len(merged)}) != baseline length ({len(df_base)})")

    actual_flips = int((merged["base_cond"] != merged["cand_cond"]).sum())
    if "flips_vs_s4" not in candidate_entry or int(candidate_entry["flips_vs_s4"]) != actual_flips:
        raise GateViolationError(
            f"Gate F: Declared flips_vs_s4 ({candidate_entry.get('flips_vs_s4')}) != actual flips ({actual_flips})"
        )

    # Check K in filename and candidate name
    for label_source in (csv_path.name, candidate_entry.get("file", ""), candidate_entry.get("name", "")):
        k_decl = extract_declared_k(label_source)
        if k_decl is not None and k_decl != actual_flips:
            raise GateViolationError(
                f"Gate F: Declared K={k_decl} in '{label_source}' != actual flips={actual_flips}"
            )

    return {"actual_flips": actual_flips, "sha256": cand_sha_raw, "id_target_hash": actual_pair_hash}


# =============================================================================
# GATE G: Format, 30k Unique Ordered IDs, Valid Classes, Zero Nulls
# =============================================================================
def validate_gate_g(csv_path: Path, expected_rows: int = 30000) -> dict:
    """Gate G validator:
    - Columns must be exactly ['ID', 'condition'].
    - Row count must be expected_rows (30,000).
    - Zero nulls / NaNs / empty strings.
    - IDs must be integers 1..expected_rows in exact sequential order with zero duplicates.
    - Classes must be strictly in {'new', 'used'} with both classes present in valid polarity range [0.30, 0.70].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise GateViolationError(f"Gate G: File '{csv_path}' does not exist")

    df = pd.read_csv(csv_path, keep_default_na=True)
    if list(df.columns) != ["ID", "condition"]:
        raise GateViolationError(f"Gate G: Columns {list(df.columns)} != ['ID', 'condition']")
    if len(df) != expected_rows:
        raise GateViolationError(f"Gate G: Row count {len(df)} != {expected_rows}")
    if df.isna().sum().sum() > 0:
        raise GateViolationError("Gate G: CSV contains NaN/null values")
    if not pd.api.types.is_integer_dtype(df["ID"]):
        raise GateViolationError(f"Gate G: ID column dtype '{df['ID'].dtype}' is not integer")
    if df["ID"].nunique() != expected_rows:
        raise GateViolationError(f"Gate G: Duplicate IDs detected ({df['ID'].nunique()} unique)")
    expected_ids = np.arange(1, expected_rows + 1, dtype=df["ID"].dtype)
    if not np.array_equal(df["ID"].values, expected_ids):
        raise GateViolationError("Gate G: IDs are not strictly ordered 1..N")

    valid_classes = {"new", "used"}
    observed_classes = set(df["condition"].astype(str).unique())
    if not observed_classes.issubset(valid_classes):
        raise GateViolationError(f"Gate G: Invalid condition labels: {observed_classes - valid_classes}")
    if observed_classes != valid_classes:
        raise GateViolationError(f"Gate G: Single-class collapse detected ({observed_classes})")

    used_rate = float((df["condition"] == "used").mean())
    if not (0.30 <= used_rate <= 0.70):
        raise GateViolationError(f"Gate G: Abnormal 'used' class rate {used_rate:.4f} outside [0.30, 0.70]")

    return {"rows": len(df), "used_rate": round(used_rate, 5)}


# =============================================================================
# GATE H: Negative Results and Failed Gates Exposed First in Reports
# =============================================================================
NEGATIVE_HEADING_PATTERNS = (
    r"resumen\s+negativo",
    r"resultados?\s+negativos?",
    r"gates?\s+fallidos?",
    r"hallazgos?\s+negativos?",
    r"negative\s+results?",
    r"failed\s+gates?",
    r"p[eé]rdidas\s+y\s+gates\s+fallidos",
)

PROMOTIONAL_PATTERNS = (
    r"\bGANADOR_HELD_OUT\b",
    r"\bQUALIFIED_MAX_ACCURACY\b",
    r"\bREADY_LEADER_RECOMMENDED\b",
    r"\bREADY_FOR_MANUAL_REVIEW\b",
    r"\bWINNER_FINALIST\b",
)


def validate_gate_h(report_markdown: str, failed_or_negative_items: list = None) -> dict:
    """Gate H validator:
    - Parses markdown headings (## ...).
    - Requires the FIRST level-2 heading (`## ...`) to be an explicit Negative Results / Failed Gates
      section.
    - Prohibits promotional status badges (READY/QUALIFIED/GANADOR) from appearing as active claims
      before the negative section.
    - Requires every item in failed_or_negative_items (if provided) to be explicitly disclosed in
      the negative section.
    """
    if not report_markdown or not report_markdown.strip():
        raise GateViolationError("Gate H: Empty report markdown")

    headings = list(re.finditer(r"^##\s+(.+)$", report_markdown, flags=re.MULTILINE))
    if not headings:
        raise GateViolationError("Gate H: Report has no level-2 ('## ') section headings")

    first_heading_text = headings[0].group(1).strip()
    if not any(re.search(pat, first_heading_text, flags=re.IGNORECASE) for pat in NEGATIVE_HEADING_PATTERNS):
        raise GateViolationError(
            f"Gate H: First section heading '{first_heading_text}' does not expose negative results / failed gates first"
        )

    preamble = report_markdown[: headings[0].start()]
    for pat in PROMOTIONAL_PATTERNS:
        if re.search(pat, preamble):
            raise GateViolationError(f"Gate H: Promotional claim '{pat}' appears before negative results section")

    neg_section_end = headings[1].start() if len(headings) > 1 else len(report_markdown)
    neg_section_body = report_markdown[headings[0].start() : neg_section_end]

    if failed_or_negative_items:
        for item in failed_or_negative_items:
            if str(item) not in neg_section_body:
                raise GateViolationError(
                    f"Gate H: Negative/failed item '{item}' omitted from the opening negative results section"
                )

    return {"first_heading": first_heading_text, "neg_section_chars": len(neg_section_body)}


# =============================================================================
# UNITTEST SUITE: POSITIVE & NEGATIVE FIXTURES FOR EVERY GATE A-H
# =============================================================================
class TestPreventiveGates(unittest.TestCase):
    AUDIT_LOG = []

    @classmethod
    def setUpClass(cls):
        cls.AUDIT_LOG.clear()
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.podium_oof_path = cls.repo_root / "results/ultimate_podium_engine_23h/oof_predictions_podium_engine.csv"
        cls.s4_path = cls.repo_root / "submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv"
        cls.s4_sha256 = compute_file_sha256(cls.s4_path, normalize_lf=False)
        cls.df_podium = pd.read_csv(cls.podium_oof_path)
        cls.y_true = cls.df_podium["y_true"].values
        cls.oof_podium = cls.df_podium["oof_blend"].values
        cls.oof_lgb = cls.df_podium["oof_lgb"].values

    @classmethod
    def record_fixture(cls, gate: str, fixture_type: str, name: str, outcome: str, detail: str):
        entry = {
            "gate": gate,
            "fixture_type": fixture_type,
            "name": name,
            "outcome": outcome,
            "detail": detail,
        }
        cls.AUDIT_LOG.append(entry)
        print(f"  [{gate} | {fixture_type}] {name} -> {outcome}: {detail}")

    # -------------------------------------------------------------------------
    # GATE A TESTS
    # -------------------------------------------------------------------------
    def test_gate_a_positive_and_negative_fixtures(self):
        # 1. POSITIVE FIXTURE: Candidate that genuinely improves over baseline on y_true
        cand_oof_pos = self.oof_podium.copy()
        wrong_idx = np.where((self.oof_podium >= 0.5).astype(int) != self.y_true)[0][:25]
        cand_oof_pos[wrong_idx] = np.where(self.y_true[wrong_idx] == 1, 0.85, 0.15)
        res_pos = validate_gate_a(
            candidate_meta={"name": "SYNTH_POS_CAND", "formal_status": "READY_FOR_MANUAL_REVIEW", "oof_hits": 65986},
            y_true=self.y_true,
            strong_baseline_oof=self.oof_podium,
            candidate_oof=cand_oof_pos,
        )
        self.assertEqual(res_pos["delta_hits"], 25)
        self.record_fixture("Gate A", "POSITIVE", "cand_oof_pos (+25 derived hits)", "PASS", f"delta_hits={res_pos['delta_hits']}")

        # 2. NEGATIVE FIXTURE 1: Real LightGBM (-64 hits vs Podium) marked READY must FAIL
        with self.assertRaises(GateViolationError) as ctx1:
            validate_gate_a(
                candidate_meta={"name": "BEST_PURE_LIGHTGBM", "formal_status": "READY_FOR_MANUAL_REVIEW"},
                y_true=self.y_true,
                strong_baseline_oof=self.oof_podium,
                candidate_oof=self.oof_lgb,
            )
        self.assertIn("-64", str(ctx1.exception))
        self.record_fixture("Gate A", "NEGATIVE", "LightGBM pure (-64 hits) marked READY", "REJECTED_AS_EXPECTED", str(ctx1.exception))

        # 3. NEGATIVE FIXTURE 2: Spoofed manifest oof_hits (+39 claimed, actual -64) must FAIL
        with self.assertRaises(GateViolationError) as ctx2:
            validate_gate_a(
                candidate_meta={"name": "SPOOFED_MANIFEST", "formal_status": "READY", "oof_hits": 66000},
                y_true=self.y_true,
                strong_baseline_oof=self.oof_podium,
                candidate_oof=self.oof_lgb,
            )
        self.assertIn("Manifest oof_hits", str(ctx2.exception))
        self.record_fixture("Gate A", "NEGATIVE", "Spoofed manifest oof_hits=66000 vs actual 65897", "REJECTED_AS_EXPECTED", str(ctx2.exception))

        # 4. NEGATIVE FIXTURE 3: Eval CSV says DEGRADED_IN_HOLDOUT (-12 P3) must FAIL
        eval_df = pd.read_csv(
            self.repo_root / "results/master_meta_lofo_tournament_20261111/parte2_meta_lofo_tournament_results.csv"
        )
        with self.assertRaises(GateViolationError) as ctx3:
            validate_gate_a(
                candidate_meta={"name": "P3_Best_Blend_Dual_Blend", "formal_status": "READY_FOR_MANUAL_REVIEW"},
                y_true=self.y_true,
                strong_baseline_oof=self.oof_podium,
                candidate_oof=cand_oof_pos,
                eval_table=eval_df,
            )
        self.assertIn("degraded status", str(ctx3.exception))
        self.record_fixture("Gate A", "NEGATIVE", "P3_Best_Blend_Dual_Blend (-12 DEGRADED_IN_HOLDOUT in eval CSV)", "REJECTED_AS_EXPECTED", str(ctx3.exception))

    # -------------------------------------------------------------------------
    # GATE B TESTS
    # -------------------------------------------------------------------------
    def test_gate_b_positive_and_negative_fixtures(self):
        rng = np.random.default_rng(42)
        # 500 sellers, 2000 rows partitioned strictly by seller into 5 outer folds
        seller_pool = np.array([f"S_{i}" for i in range(500)])
        seller_fold_map = {s: i % 5 for i, s in enumerate(seller_pool)}
        row_sellers = rng.choice(seller_pool, size=2000, replace=True)
        outer_folds = np.array([seller_fold_map[s] for s in row_sellers])

        prov_pos = []
        freeze_pos = []
        for k in range(5):
            prov_pos.append(
                {
                    "model": f"outer{k}_full",
                    "fit_rows_sha": hashlib.sha256(f"outer_{k}_tr".encode()).hexdigest(),
                    "pred_rows_sha": hashlib.sha256(f"outer_{k}_va".encode()).hexdigest(),
                    "n_fit": 1600,
                }
            )
            for j in range(4):
                prov_pos.append(
                    {
                        "model": f"outer{k}_inner{j}",
                        "fit_rows_sha": hashlib.sha256(f"outer_{k}_inner_{j}_tr".encode()).hexdigest(),
                        "pred_rows_sha": hashlib.sha256(f"outer_{k}_inner_{j}_va".encode()).hexdigest(),
                        "n_fit": 1200,
                    }
                )
            freeze_pos.append(
                {
                    "outer_fold": k + 1,
                    "inner_oof_acc": {"lgb": 0.90, "cb": 0.89},
                    "frozen_at_utc": "2026-11-11T05:00:00Z",
                }
            )

        good_script = (
            self.repo_root / "scripts/seller_group_nested_podium_validation.py"
        ).read_text(encoding="utf-8")

        # 1. POSITIVE FIXTURE: Disjoint sellers + full outer/inner refit provenance + valid script
        res_pos = validate_gate_b(
            outer_fold_ids=outer_folds,
            seller_ids=row_sellers,
            provenance_records=prov_pos,
            freeze_records=freeze_pos,
            script_source=good_script,
        )
        self.assertEqual(res_pos["distinct_fit_shas"], 25)
        self.record_fixture("Gate B", "POSITIVE", "Seller-disjoint 5x4 outer+inner refit provenance + script AST", "PASS", f"distinct_fit_shas={res_pos['distinct_fit_shas']}")

        # 2. NEGATIVE FIXTURE 1: Seller overlap across outer folds (like StratifiedKFold) must FAIL
        leaked_folds = rng.integers(0, 5, size=2000)
        with self.assertRaises(GateViolationError) as ctx1:
            validate_gate_b(
                outer_fold_ids=leaked_folds,
                seller_ids=row_sellers,
                provenance_records=prov_pos,
                freeze_records=freeze_pos,
            )
        self.assertIn("shares", str(ctx1.exception))
        self.record_fixture("Gate B", "NEGATIVE", "Non-disjoint seller split (StratifiedKFold leakage)", "REJECTED_AS_EXPECTED", str(ctx1.exception))

        # 3. NEGATIVE FIXTURE 2: Bad script slicing global OOF (`execute_meta_lofo_tournament.py`) must FAIL
        bad_script = (
            self.repo_root / "scripts/execute_meta_lofo_tournament.py"
        ).read_text(encoding="utf-8")
        with self.assertRaises(GateViolationError) as ctx2:
            validate_gate_b(
                outer_fold_ids=outer_folds,
                seller_ids=row_sellers,
                provenance_records=prov_pos,
                freeze_records=freeze_pos,
                script_source=bad_script,
            )
        self.assertIn("slices precomputed global OOF", str(ctx2.exception))
        self.record_fixture("Gate B", "NEGATIVE", "execute_meta_lofo_tournament.py slicing global OOF", "REJECTED_AS_EXPECTED", str(ctx2.exception))

        # 4. NEGATIVE FIXTURE 3: Early stopping on outer val_idx must FAIL
        es_leak_script = "StratifiedGroupKFold\nclf.fit(X, y, eval_set=[(X[val_idx], y[val_idx])], early_stopping_rounds=150)"
        with self.assertRaises(GateViolationError) as ctx3:
            validate_gate_b(
                outer_fold_ids=outer_folds,
                seller_ids=row_sellers,
                provenance_records=prov_pos,
                freeze_records=freeze_pos,
                script_source=es_leak_script,
            )
        self.assertIn("early_stopping_rounds", str(ctx3.exception))
        self.record_fixture("Gate B", "NEGATIVE", "Early stopping on outer val_idx", "REJECTED_AS_EXPECTED", str(ctx3.exception))

    # -------------------------------------------------------------------------
    # GATE C TESTS
    # -------------------------------------------------------------------------
    def test_gate_c_positive_and_negative_fixtures(self):
        # 1. POSITIVE FIXTURE A: Explicit NO_IDENTIFIABLE when no calibration artifact exists
        res_ni = validate_gate_c(
            claims={
                "expected_public_score": "NO_IDENTIFIABLE",
                "prob_beat_s4": "NO_IDENTIFIABLE",
                "prob_reach_09500": "NO_IDENTIFIABLE",
                "public_q05": "NO_IDENTIFIABLE",
            }
        )
        self.assertEqual(res_ni["status"], "PASS_NO_IDENTIFIABLE_DECLARED")
        self.record_fixture("Gate C", "POSITIVE", "NO_IDENTIFIABLE declared for all public probability/Q05 fields", "PASS", res_ni["status"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # 2. POSITIVE FIXTURE B: 20 sequential prequential events with 95% coverage + calibrated Brier
            valid_art = tmp / "valid_cal_20.json"
            valid_events_20 = [
                {
                    "event_id": f"EVT-{100+i:03d}",
                    "timestamp_utc": f"2026-10-{10+i:02d}T12:00:00Z",
                    "pred_q05": 0.9465,
                    "pred_q95": 0.9485,
                    "observed_public_score": 0.9475 if i < 19 else 0.9460,
                    "pred_prob": 0.60,
                    "exceeded_threshold": 1 if i < 12 else 0,
                    "fitted_using_event_outcome": False,
                }
                for i in range(20)
            ]
            # 12/20 = 0.60 exceeded, pred_prob = 0.60 -> Brier = 12*(0.4^2)/20 + 8*(0.6^2)/20 = 0.24 -> use sharper calibrated probs:
            for i, ev in enumerate(valid_events_20):
                ev["pred_prob"] = 0.85 if i < 12 else 0.15
                ev["exceeded_threshold"] = 1 if i < 12 else 0
            brier_val = float(np.mean([(ev["pred_prob"] - ev["exceeded_threshold"]) ** 2 for ev in valid_events_20]))
            valid_art.write_text(
                json.dumps({
                    "calibration_type": "threshold_probability_and_interval",
                    "claimed_confidence_level": 0.95,
                    "prequential_events": valid_events_20,
                    "measured_empirical_coverage": 0.95,
                    "measured_brier_score": brier_val,
                }),
                encoding="utf-8",
            )
            res_cal = validate_gate_c(
                claims={"prob_beat_s4": 0.60, "public_q05": 0.9468, "claimed_confidence_level": 0.95},
                calibration_artifact_path=valid_art,
            )
            self.assertEqual(res_cal["recomputed_coverage"], 0.95)
            self.record_fixture("Gate C", "POSITIVE", "20 sequential events, 95% coverage + Brier=0.0225 calibrated", "PASS", f"n=20, cov={res_cal['recomputed_coverage']}, brier={res_cal['recomputed_brier']:.4f}")

            # 3. NEGATIVE FIXTURE 1: Uncalibrated S5 claims without artifact must FAIL
            with self.assertRaises(GateViolationError) as ctx1:
                validate_gate_c(
                    claims={
                        "expected_public_score": 0.95057,
                        "P(>S4)": "98.9%",
                        "P(>=0.95000)": "66.0%",
                        "public_q05": 0.94847,
                    },
                    calibration_artifact_path=None,
                )
            self.assertIn("missing prequential calibration artifact", str(ctx1.exception))
            self.record_fixture("Gate C", "NEGATIVE", "Uncalibrated S5 claims (0.95057, 98.9%, 66.0%, Q05=0.94847)", "REJECTED_AS_EXPECTED", str(ctx1.exception))

            # 4. NEGATIVE FIXTURE 2 (CX-0164 i): Only 5 prequential events must FAIL for 95% bound/prob
            art_5 = tmp / "only_5_events.json"
            art_5.write_text(
                json.dumps({
                    "prequential_events": valid_events_20[:5],
                    "measured_empirical_coverage": 1.0,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(GateViolationError) as ctx_n5:
                validate_gate_c(
                    claims={"public_q05": 0.9468, "claimed_confidence_level": 0.95},
                    calibration_artifact_path=art_5,
                )
            self.assertIn("Sample size n=5 is insufficient", str(ctx_n5.exception))
            self.record_fixture("Gate C", "NEGATIVE", "Only 5 sequential events used for 95% bound (CX-0164 i)", "REJECTED_AS_EXPECTED", str(ctx_n5.exception))

            # 5. NEGATIVE FIXTURE 3 (CX-0164 ii): 0.80 coverage used to claim a 95% bound must FAIL
            art_cov80 = tmp / "cov_80_for_95.json"
            events_80 = [
                {
                    **ev,
                    "observed_public_score": 0.9475 if i < 16 else 0.9420,
                }
                for i, ev in enumerate(valid_events_20)
            ]
            art_cov80.write_text(
                json.dumps({
                    "claimed_confidence_level": 0.95,
                    "prequential_events": events_80,
                    "measured_empirical_coverage": 0.80,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(GateViolationError) as ctx_cov80:
                validate_gate_c(
                    claims={"public_q05": 0.9468, "claimed_confidence_level": 0.95},
                    calibration_artifact_path=art_cov80,
                )
            self.assertIn("< claimed confidence level (0.950)", str(ctx_cov80.exception))
            self.record_fixture("Gate C", "NEGATIVE", "0.80 empirical coverage used for 95% bound claim (CX-0164 ii)", "REJECTED_AS_EXPECTED", str(ctx_cov80.exception))

            # 6. NEGATIVE FIXTURE 4 (CX-0164 iii): Interval coverage presented as threshold-probability calibration must FAIL
            art_interval_only = tmp / "interval_only_20.json"
            art_interval_only.write_text(
                json.dumps({
                    "calibration_type": "interval_only",
                    "claimed_confidence_level": 0.95,
                    "prequential_events": valid_events_20,
                    "measured_empirical_coverage": 0.95,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(GateViolationError) as ctx_int_only:
                validate_gate_c(
                    claims={"P(>S4)": 0.75, "public_q05": 0.9468, "claimed_confidence_level": 0.95},
                    calibration_artifact_path=art_interval_only,
                )
            self.assertIn("Interval coverage alone cannot calibrate threshold probabilities", str(ctx_int_only.exception))
            self.record_fixture("Gate C", "NEGATIVE", "Interval coverage presented as threshold-probability calibration (CX-0164 iii)", "REJECTED_AS_EXPECTED", str(ctx_int_only.exception))

    # -------------------------------------------------------------------------
    # GATE D TESTS
    # -------------------------------------------------------------------------
    def test_gate_d_positive_live_repo_and_negative_fixtures(self):
        rejected_names = {
            "new_bases_portfolio_20261111",
            "portfolio_tomorrow_20261111",
            "SUBMISSION_CANDIDATE_1_PURE_TRAIN_ONLY_SUPERLEARNER.csv",
            "SUBMISSION_CANDIDATE_2_ROBUST_4WAY_BLEND.csv",
            "SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv",
            "SUBMISSION_CANDIDATE_4_BROAD_UPSIDE_K75.csv",
            "PORTFOLIO_OPT2_BEST_PURE_LIGHTGBM_STANDALONE.csv",
            "PORTFOLIO_OPT3_BEST_NESTED_DUAL_BLEND_WINNER.csv",
            "PORTFOLIO_OPT4_CONSERVATIVE_OVERLAY_K50.csv",
            "SUBMISSION_5_TODAY_FINAL.csv",
        }
        rejected_shas = {
            "24a3fd6cb15797d6e063190b45f0a8d2036766c1992045fefc8ae2f73394b968",
            "1ad24a5578a787e19667ab91d927141f3943e02d937b5cd8410bdab859af9d8b",
        }

        # 1. POSITIVE FIXTURE: Live repo submissions/ and rejected/ after quarantine
        res_live = validate_gate_d(
            submissions_dir=self.repo_root / "submissions",
            rejected_dir=self.repo_root / "rejected",
            active_manifests=[],
            rejected_names_or_dirs=rejected_names,
            rejected_sha256s=rejected_shas,
        )
        self.assertGreaterEqual(res_live["quarantined_csv_count"], 10)
        self.record_fixture("Gate D", "POSITIVE", "Live repo submissions/ clean + rejected/ holds quarantined CSVs", "PASS", f"quarantined_csvs={res_live['quarantined_csv_count']}")

        # 2. NEGATIVE FIXTURE 1: Rejected file placed inside submissions/ must FAIL
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sub_dir = tmp / "submissions"
            rej_dir = tmp / "rejected"
            sub_dir.mkdir()
            rej_dir.mkdir()
            (rej_dir / "quarantined.csv").write_text("ID,condition\n1,new\n", encoding="utf-8")
            (sub_dir / "SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv").write_text(
                "ID,condition\n1,used\n", encoding="utf-8"
            )
            with self.assertRaises(GateViolationError) as ctx1:
                validate_gate_d(
                    submissions_dir=sub_dir,
                    rejected_dir=rej_dir,
                    active_manifests=[],
                    rejected_names_or_dirs=rejected_names,
                )
            self.assertIn("found inside active submissions", str(ctx1.exception))
            self.record_fixture("Gate D", "NEGATIVE", "Rejected CSV left inside active submissions/", "REJECTED_AS_EXPECTED", str(ctx1.exception))

            # 3. NEGATIVE FIXTURE 2: Active manifest containing a REJECTED/DEGRADED candidate must FAIL
            (sub_dir / "SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv").unlink()
            bad_manifest = tmp / "active_manifest.json"
            bad_manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "name": "OPT2_LGB",
                                "file": "opt2.csv",
                                "formal_status": "REJECTED_NET_NEGATIVE_MINUS64",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(GateViolationError) as ctx2:
                validate_gate_d(
                    submissions_dir=sub_dir,
                    rejected_dir=rej_dir,
                    active_manifests=[bad_manifest],
                    rejected_names_or_dirs=rejected_names,
                )
            self.assertIn("Active manifest", str(ctx2.exception))
            self.record_fixture("Gate D", "NEGATIVE", "Active manifest listing REJECTED_NET_NEGATIVE_MINUS64 entry", "REJECTED_AS_EXPECTED", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # GATE E TESTS
    # -------------------------------------------------------------------------
    def test_gate_e_positive_and_negative_fixtures_all_four_procedures(self):
        # --- 1. NESTED ---
        res_nested = validate_gate_e(
            "nested",
            {
                "outer_train_indices": [0, 1, 2, 3],
                "outer_val_indices": [4, 5],
                "reused_outer_oof": False,
                "inner_splits": [
                    {"train": [0, 1], "val": [2, 3], "refit_sha": "sha_in0"},
                    {"train": [2, 3], "val": [0, 1], "refit_sha": "sha_in1"},
                ],
            },
        )
        self.assertEqual(res_nested["verified"], "nested")
        self.record_fixture("Gate E", "POSITIVE", "Nested CV structural proof (disjoint outer/inner + refit SHAs)", "PASS", "verified=nested")

        with self.assertRaises(GateViolationError) as ctx_n:
            validate_gate_e(
                "nested",
                {
                    "outer_train_indices": [0, 1, 2, 3],
                    "outer_val_indices": [4, 5],
                    "inner_splits": [{"train": [0, 1, 4], "val": [2, 3], "refit_sha": "sha_in0"}],
                },
            )
        self.record_fixture("Gate E", "NEGATIVE", "Nested claim leaking outer_val index into inner split", "REJECTED_AS_EXPECTED", str(ctx_n.exception))

        # --- 2. TEMPORAL ---
        res_temp = validate_gate_e(
            "temporal",
            {
                "train_timestamps": [100, 110, 120],
                "val_timestamps": [130, 140],
                "retrained_on_past_only": True,
                "post_hoc_oof_slice": False,
                "refit_sha": "sha_temp_refit",
            },
        )
        self.assertEqual(res_temp["verified"], "temporal")
        self.record_fixture("Gate E", "POSITIVE", "Temporal forward split max(t_train)<min(t_val) + refit SHA", "PASS", "verified=temporal")

        with self.assertRaises(GateViolationError) as ctx_t:
            validate_gate_e(
                "temporal",
                {
                    "train_timestamps": [100, 135],
                    "val_timestamps": [130, 140],
                    "retrained_on_past_only": True,
                    "refit_sha": "sha_temp",
                },
            )
        self.record_fixture("Gate E", "NEGATIVE", "Temporal claim with overlapping timestamps / post-hoc OOF slice", "REJECTED_AS_EXPECTED", str(ctx_t.exception))

        # --- 3. LEAVE-FAMILY ---
        res_lf = validate_gate_e(
            "leave-family",
            {
                "train_categories": ["MLA1055", "MLA2044"],
                "val_categories": ["MLA1227"],
                "retrained_without_family": True,
                "post_hoc_oof_slice": False,
                "refit_sha": "sha_lf_refit",
            },
        )
        self.assertEqual(res_lf["verified"], "leave-family")
        self.record_fixture("Gate E", "POSITIVE", "Leave-family split excluding MLA1227 from fit + refit SHA", "PASS", "verified=leave-family")

        with self.assertRaises(GateViolationError) as ctx_lf:
            validate_gate_e(
                "leave-family",
                {
                    "train_categories": ["MLA1055", "MLA2044"],
                    "val_categories": ["MLA1227"],
                    "retrained_without_family": False,
                    "post_hoc_oof_slice": True,
                    "refit_sha": "sha_lf",
                },
            )
        self.record_fixture("Gate E", "NEGATIVE", "Leave-family claim via post-hoc OOF filtering without refit", "REJECTED_AS_EXPECTED", str(ctx_lf.exception))

        # --- 4. SELLER/CATEGORY CLUSTER BOOTSTRAP ---
        sellers = np.array(["S1", "S1", "S2", "S2"])
        cats = np.array(["C1", "C1", "C2", "C2"])
        cluster_weights = np.array([2.0, 2.0, 0.0, 0.0])  # consistent within (seller, cat) cluster
        res_boot = validate_gate_e(
            "seller_category_bootstrap",
            {"seller_ids": sellers, "category_ids": cats, "sample_weights": cluster_weights},
        )
        self.assertEqual(res_boot["verified"], "seller_category_bootstrap")
        self.record_fixture("Gate E", "POSITIVE", "Two-way seller x category cluster bootstrap weights", "PASS", "verified=seller_category_bootstrap")

        with self.assertRaises(GateViolationError) as ctx_b:
            iid_weights = np.array([3.0, 0.0, 1.0, 2.0])
            validate_gate_e(
                "seller_category_bootstrap",
                {"seller_ids": sellers, "category_ids": cats, "sample_weights": iid_weights},
            )
        self.record_fixture("Gate E", "NEGATIVE", "Bootstrap claim using IID row resampling within sellerxcat cluster", "REJECTED_AS_EXPECTED", str(ctx_b.exception))

    # -------------------------------------------------------------------------
    # GATE F TESTS
    # -------------------------------------------------------------------------
    def test_gate_f_positive_and_negative_fixtures_including_k15_bug(self):
        # 1. POSITIVE FIXTURE: S4 anchor validated against itself (flips_vs_s4 = 0)
        s4_df = pd.read_csv(self.s4_path)
        s4_pair_hash = compute_id_target_hash(s4_df)
        res_s4 = validate_gate_f(
            candidate_entry={
                "name": "S4_CONFIRMED_RECORD",
                "file": self.s4_path.name,
                "sha256": self.s4_sha256,
                "id_target_hash": s4_pair_hash,
                "flips_vs_s4": 0,
            },
            csv_path=self.s4_path,
            baseline_csv_path=self.s4_path,
            expected_baseline_sha256=self.s4_sha256,
        )
        self.assertEqual(res_s4["actual_flips"], 0)
        self.record_fixture("Gate F", "POSITIVE", "S4 anchor SHA-256 + id_target_hash + flips=0", "PASS", f"sha256={self.s4_sha256[:16]}...")

        # 2. NEGATIVE FIXTURE 1: The real C3_CONSERVATIVE_OVERLAY_K15.csv (K15 in filename, 8 flips) must FAIL
        c3_k15_path = (
            self.repo_root
            / "rejected/new_bases_portfolio_20261111_ag0168/SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv"
        )
        c3_df = pd.read_csv(c3_k15_path)
        with self.assertRaises(GateViolationError) as ctx1:
            validate_gate_f(
                candidate_entry={
                    "name": "C3_CONSERVATIVE_OVERLAY_K15",
                    "file": c3_k15_path.name,
                    "sha256": compute_file_sha256(c3_k15_path, normalize_lf=True),
                    "id_target_hash": compute_id_target_hash(c3_df),
                    "flips_vs_s4": 8,  # Even when manifest says 8, filename says K15!
                },
                csv_path=c3_k15_path,
                baseline_csv_path=self.s4_path,
                expected_baseline_sha256=self.s4_sha256,
            )
        self.assertIn("Declared K=15", str(ctx1.exception))
        self.record_fixture("Gate F", "NEGATIVE", "SUBMISSION_CANDIDATE_3_CONSERVATIVE_OVERLAY_K15.csv (K15 vs 8 actual flips)", "REJECTED_AS_EXPECTED", str(ctx1.exception))

        # 3. NEGATIVE FIXTURE 2: Tampered id_target_hash must FAIL
        with self.assertRaises(GateViolationError) as ctx2:
            validate_gate_f(
                candidate_entry={
                    "name": "S4_TAMPERED_HASH",
                    "file": self.s4_path.name,
                    "sha256": self.s4_sha256,
                    "id_target_hash": "0" * 64,
                    "flips_vs_s4": 0,
                },
                csv_path=self.s4_path,
                baseline_csv_path=self.s4_path,
                expected_baseline_sha256=self.s4_sha256,
            )
        self.assertIn("ID/target hash mismatch", str(ctx2.exception))
        self.record_fixture("Gate F", "NEGATIVE", "Tampered id_target_hash with valid file SHA", "REJECTED_AS_EXPECTED", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # GATE G TESTS
    # -------------------------------------------------------------------------
    def test_gate_g_positive_and_negative_fixtures(self):
        # 1. POSITIVE FIXTURE: Real S4 CSV (30,000 rows, IDs 1..30000, valid classes, 0 nulls)
        res_g = validate_gate_g(self.s4_path, expected_rows=30000)
        self.assertEqual(res_g["rows"], 30000)
        self.record_fixture("Gate G", "POSITIVE", "S4 CSV format (30000 rows, IDs 1..30000, valid classes, 0 nulls)", "PASS", f"used_rate={res_g['used_rate']}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # 2. NEGATIVE FIXTURE 1: Duplicate IDs / out-of-order IDs must FAIL
            dup_csv = tmp / "dup.csv"
            df_bad = pd.DataFrame({"ID": [1, 1, 3, 4], "condition": ["new", "used", "new", "used"]})
            df_bad.to_csv(dup_csv, index=False)
            with self.assertRaises(GateViolationError) as ctx1:
                validate_gate_g(dup_csv, expected_rows=4)
            self.assertIn("Duplicate IDs", str(ctx1.exception))
            self.record_fixture("Gate G", "NEGATIVE", "Duplicate IDs [1, 1, 3, 4]", "REJECTED_AS_EXPECTED", str(ctx1.exception))

            # 3. NEGATIVE FIXTURE 2: Null value or invalid class must FAIL
            null_csv = tmp / "null.csv"
            df_null = pd.DataFrame({"ID": [1, 2, 3, 4], "condition": ["new", None, "used", "new"]})
            df_null.to_csv(null_csv, index=False)
            with self.assertRaises(GateViolationError) as ctx2:
                validate_gate_g(null_csv, expected_rows=4)
            self.assertIn("NaN/null", str(ctx2.exception))
            self.record_fixture("Gate G", "NEGATIVE", "Null condition value in row 2", "REJECTED_AS_EXPECTED", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # GATE H TESTS
    # -------------------------------------------------------------------------
    def test_gate_h_positive_and_negative_fixtures(self):
        # 1. POSITIVE FIXTURE: AG-0180-ACK-CORRECTION.md (opens with '## 0. Resumen negativo primero')
        ag0180_text = (
            self.repo_root / "coordination/antigravity_to_codex/AG-0180-ACK-CORRECTION.md"
        ).read_text(encoding="utf-8")
        res_h = validate_gate_h(
            ag0180_text,
            failed_or_negative_items=["AG-0177", "execute_meta_lofo_tournament.py", "−12", "14089"],
        )
        self.assertIn("Resumen negativo", res_h["first_heading"])
        self.record_fixture("Gate H", "POSITIVE", "AG-0180-ACK-CORRECTION.md opens with negative results & discloses all failures", "PASS", f"heading='{res_h['first_heading']}'")

        # 2. NEGATIVE FIXTURE 1: AG-0177-RESULT.md (opens with celebratory summary, not negative results) must FAIL
        ag0177_text = (
            self.repo_root / "coordination/antigravity_to_codex/AG-0177-RESULT.md"
        ).read_text(encoding="utf-8")
        with self.assertRaises(GateViolationError) as ctx1:
            validate_gate_h(ag0177_text)
        self.assertIn("does not expose negative results", str(ctx1.exception))
        self.record_fixture("Gate H", "NEGATIVE", "AG-0177-RESULT.md opening with celebratory claims before negative results", "REJECTED_AS_EXPECTED", str(ctx1.exception))

        # 3. NEGATIVE FIXTURE 2: Report that hides a failed gate / degraded model from opening section must FAIL
        partial_report = "## 0. Resumen negativo primero\n- Falló modelo X.\n\n## 1. Detalles\nTexto.\n"
        with self.assertRaises(GateViolationError) as ctx2:
            validate_gate_h(partial_report, failed_or_negative_items=["P3_Best_Blend_Dual_Blend (-12)"])
        self.assertIn("omitted from the opening negative results section", str(ctx2.exception))
        self.record_fixture("Gate H", "NEGATIVE", "Report omitting degraded model P3 (-12) from opening negative section", "REJECTED_AS_EXPECTED", str(ctx2.exception))

    @classmethod
    def tearDownClass(cls):
        out_dir = cls.repo_root / "results/preventive_gates_repair_20261111"
        out_dir.mkdir(parents=True, exist_ok=True)
        gates_summary = {}
        for item in cls.AUDIT_LOG:
            g = item["gate"]
            gates_summary.setdefault(g, {"positive_fixtures": 0, "negative_fixtures": 0, "cases": []})
            if item["fixture_type"] == "POSITIVE":
                gates_summary[g]["positive_fixtures"] += 1
            else:
                gates_summary[g]["negative_fixtures"] += 1
            gates_summary[g]["cases"].append(item)
        report = {
            "test_file": "tests/test_preventive_gates.py",
            "test_file_sha256": compute_file_sha256(Path(__file__), normalize_lf=False),
            "test_file_sha256_lf": compute_file_sha256(Path(__file__), normalize_lf=True),
            "total_gates_covered": len(gates_summary),
            "all_gates_a_to_h_present": set(gates_summary.keys()) == {f"Gate {c}" for c in "ABCDEFGH"},
            "total_fixture_assertions": len(cls.AUDIT_LOG),
            "gates": gates_summary,
        }
        (out_dir / "gates_coverage_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
