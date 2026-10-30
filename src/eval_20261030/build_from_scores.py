"""Content-Addressed 5th Slot Decision Engine & Disjoint Combiner (Agent 5).

Strict design principles:
1. ZERO filename-number heuristics:
   - Never inspects numeric substrings in filenames.
   - Never uses hardcoded ID sets for legacy candidate blocks.
2. Content-addressed & structural identification:
   - Every candidate CSV and baseline reference is identified strictly by:
     * sha256: exact SHA256 digest of the file bytes on disk.
     * vector_sha256: exact SHA256 digest of the 30,000 binary label vector.
     * baseline_sha256: SHA256 digest of the active baseline vector (S48 = 3f8d08e0...).
     * flips: exact dictionary {id: target_label} where target_label != baseline[id].
     * directions: exact dictionary {id: f"{baseline[id]}_to_{target_label}"}.
     * flip_set_hash: deterministic SHA256 of sorted (id, baseline[id], target_label) tuples.
3. Strict input, schema, score, and duplicate validation prior to any decision.
4. Strict combiner & contrast rules:
   - Ties (Delta == 0) and Losses (Delta < 0) are never promoted.
   - Never extracts an untested difference set (B \\ C or B \\ A) as a synthetic winning block.
   - Admits nested contrast (A strict subset of B) ONLY when both A and B are evaluated against
     the same baseline and A won (score(A) > base_score):
       * If score(B) > score(A): resolves {A, B} to B.
       * If score(B) <= score(A): rejects marginal flips B \\ A and resolves {A, B} to A.
   - If A is a strict subset of B and A lost/tied (score(A) <= base_score) while B won:
     raises SubmissionBuildError (refuses to extract unprobed B \\ A or promote B containing refuted A).
   - Partially overlapping winning blocks (0 < |W_i & W_j| < min(|W_i|, |W_j|)) raise SubmissionBuildError.
   - Single resolved winner already scored on Kaggle returns SINGLE_WINNER_ALREADY_SCORED without
     emitting a duplicate CSV.
   - Never sums public deltas of constituent blocks as an expected score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_20261030.submission_guard import (
    REGISTRY_JSON,
    S48_JSON,
    check_submission_file,
)

EXPECTED_N_ROWS = 30000
VALID_CONDITIONS = {"new", "used"}
VALID_BASE_IDS = {"48", "s48", "submission_48"}


class SubmissionBuildError(ValueError):
    """Raised when input probes, schema, hashes, or combination rules are violated."""


def compute_sha256(path: Path | str) -> str:
    """Compute exact SHA256 hex digest of raw file bytes on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_vector_sha256(labels_dict: dict[int, str]) -> str:
    """Compute SHA256 hex digest of the 30,000-element binary prediction vector."""
    vec = "".join("1" if labels_dict[i] == "used" else "0" for i in range(1, EXPECTED_N_ROWS + 1))
    return hashlib.sha256(vec.encode("ascii")).hexdigest()


def compute_canonical_csv_sha256(labels_dict: dict[int, str]) -> str:
    """Compute SHA256 hex digest of canonical LF-delimited CSV bytes (3f8d08e0... for S48)."""
    lines = ["ID,condition\n"]
    for i in range(1, EXPECTED_N_ROWS + 1):
        lines.append(f"{i},{labels_dict[i]}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def compute_flip_set_hash(flips: dict[int, str], baseline_dict: dict[int, str]) -> str:
    """Compute deterministic SHA256 hex digest of sorted (id, baseline_label, target_label) tuples."""
    canonical_tuples = [
        [int(i), str(baseline_dict[int(i)]), str(target)]
        for i, target in sorted(flips.items(), key=lambda kv: int(kv[0]))
    ]
    payload = json.dumps(canonical_tuples, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_baseline_dict(s48_dict: dict[Any, Any]) -> tuple[dict[int, str], str]:
    """Validate and normalize the 30,000-row baseline dictionary and compute its canonical CSV SHA256."""
    if not isinstance(s48_dict, dict):
        raise SubmissionBuildError("Baseline s48_dict must be a dictionary mapping ID -> condition.")
    if len(s48_dict) != EXPECTED_N_ROWS:
        raise SubmissionBuildError(
            f"Baseline s48_dict must contain exactly {EXPECTED_N_ROWS} rows, got {len(s48_dict)}."
        )

    normalized: dict[int, str] = {}
    for k, v in s48_dict.items():
        try:
            ik = int(k)
        except (ValueError, TypeError) as exc:
            raise SubmissionBuildError(f"Invalid baseline ID key: {k!r}") from exc
        if not isinstance(v, str) or v not in VALID_CONDITIONS:
            raise SubmissionBuildError(f"Invalid baseline condition for ID {ik}: {v!r}")
        normalized[ik] = v

    expected_ids = set(range(1, EXPECTED_N_ROWS + 1))
    if set(normalized.keys()) != expected_ids:
        raise SubmissionBuildError("Baseline s48_dict IDs must be strictly 1..30000.")

    baseline_sha256 = compute_canonical_csv_sha256(normalized)
    return normalized, baseline_sha256


def validate_score(score: Any, label: str) -> float:
    """Validate that a public evaluation score is a finite float in [0.0, 1.0]."""
    if score is None or isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SubmissionBuildError(
            f"Invalid or missing score for '{label}': got {score!r}. Score must be a finite float."
        )
    fscore = float(score)
    if math.isnan(fscore) or math.isinf(fscore):
        raise SubmissionBuildError(
            f"Non-finite score for '{label}': got {score!r}. NaN and Inf are strictly rejected."
        )
    if not (0.0 <= fscore <= 1.0):
        raise SubmissionBuildError(
            f"Out-of-range score for '{label}': got {fscore}. Expected score in [0.0, 1.0]."
        )
    return fscore


def load_and_validate_candidate_csv(
    csv_path: Path | str,
    s48_dict: dict[int, str],
    baseline_sha256: str,
    expected_sha256: str | None = None,
    entry_base_sha256: str | None = None,
    entry_base_id: str | None = None,
) -> dict[str, Any]:
    """Load, validate schema/integrity, and compute content-addressed descriptors for a candidate CSV."""
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise SubmissionBuildError(f"Candidate CSV file not found: {path}")

    if entry_base_sha256 is not None and str(entry_base_sha256) != baseline_sha256:
        raise SubmissionBuildError(
            f"Baseline SHA256 mismatch for '{path.name}': entry specifies base_sha256={entry_base_sha256}, "
            f"active baseline is {baseline_sha256}."
        )

    if entry_base_id is not None and str(entry_base_id).lower() not in VALID_BASE_IDS:
        raise SubmissionBuildError(
            f"Incompatible baseline ID for '{path.name}': got base_id={entry_base_id!r}, "
            f"expected one of {sorted(VALID_BASE_IDS)}."
        )

    actual_sha256 = compute_sha256(path)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise SubmissionBuildError(
            f"File SHA256 mismatch for '{path.name}': expected {expected_sha256}, got {actual_sha256}."
        )

    try:
        df = pd.read_csv(path, dtype=object, keep_default_na=True)
    except Exception as exc:
        raise SubmissionBuildError(f"Failed to parse CSV '{path.name}': {exc}") from exc

    if list(df.columns) != ["ID", "condition"]:
        raise SubmissionBuildError(
            f"Invalid CSV header in '{path.name}': expected ['ID', 'condition'], got {list(df.columns)}."
        )

    if df["ID"].isna().any() or df["condition"].isna().any():
        raise SubmissionBuildError(f"CSV '{path.name}' contains NaN or missing values in 'ID' or 'condition'.")

    if len(df) != EXPECTED_N_ROWS:
        raise SubmissionBuildError(
            f"Invalid row count in '{path.name}': expected {EXPECTED_N_ROWS}, got {len(df)}."
        )

    raw_ids = df["ID"].tolist()
    parsed_ids: list[int] = []
    for row_idx, raw_id in enumerate(raw_ids, start=1):
        s_id = str(raw_id)
        if s_id != s_id.strip() or not s_id.isdigit():
            raise SubmissionBuildError(
                f"Invalid non-integer ID at row {row_idx} in '{path.name}': {raw_id!r}."
            )
        parsed_ids.append(int(s_id))

    if len(set(parsed_ids)) != EXPECTED_N_ROWS:
        seen: set[int] = set()
        dups: list[int] = []
        for i in parsed_ids:
            if i in seen:
                dups.append(i)
            seen.add(i)
        raise SubmissionBuildError(
            f"Duplicate IDs detected in '{path.name}': {sorted(set(dups))[:10]}."
        )

    expected_id_seq = list(range(1, EXPECTED_N_ROWS + 1))
    if parsed_ids != expected_id_seq:
        missing = sorted(set(expected_id_seq) - set(parsed_ids))
        extra = sorted(set(parsed_ids) - set(expected_id_seq))
        if missing or extra:
            raise SubmissionBuildError(
                f"ID range error in '{path.name}': missing={missing[:10]}, extra={extra[:10]}."
            )
        raise SubmissionBuildError(
            f"ID sequencing error in '{path.name}': IDs must be strictly ordered 1..{EXPECTED_N_ROWS}."
        )

    raw_conds = df["condition"].tolist()
    labels: dict[int, str] = {}
    for idx_0, raw_c in enumerate(raw_conds):
        if not isinstance(raw_c, str) or raw_c != raw_c.strip() or raw_c not in VALID_CONDITIONS:
            raise SubmissionBuildError(
                f"Invalid condition label at ID {idx_0 + 1} in '{path.name}': {raw_c!r}. "
                f"Allowed values are strictly {sorted(VALID_CONDITIONS)}."
            )
        labels[idx_0 + 1] = raw_c

    vector_sha256 = compute_vector_sha256(labels)
    flips: dict[int, str] = {}
    directions: dict[int, str] = {}
    for i in range(1, EXPECTED_N_ROWS + 1):
        if labels[i] != s48_dict[i]:
            flips[i] = labels[i]
            directions[i] = f"{s48_dict[i]}_to_{labels[i]}"

    flip_set_hash = compute_flip_set_hash(flips, s48_dict)
    n_used_to_new = sum(1 for d in directions.values() if d == "used_to_new")
    n_new_to_used = sum(1 for d in directions.values() if d == "new_to_used")

    return {
        "path": str(path),
        "sha256": actual_sha256,
        "vector_sha256": vector_sha256,
        "baseline_sha256": baseline_sha256,
        "flip_set_hash": flip_set_hash,
        "n_flips": len(flips),
        "n_used_to_new": n_used_to_new,
        "n_new_to_used": n_new_to_used,
        "flips": flips,
        "directions": directions,
    }


def _check_not_in_registry(
    candidate_info: dict[str, Any],
    reg_data: dict[str, Any] | None,
    baseline_sha256: str,
) -> None:
    """Reject candidates that have 0 flips vs baseline or match any already-evaluated registry entry."""
    if candidate_info["n_flips"] == 0 or candidate_info["vector_sha256"] == baseline_sha256:
        raise SubmissionBuildError(
            f"Candidate (sha256={candidate_info['sha256'][:12]}...) has 0 flips vs baseline "
            f"(exact duplicate of Baseline S48 / Submission 76, vector_sha256={baseline_sha256[:12]}...)."
        )

    if not reg_data:
        return

    reg_base_sha = reg_data.get("baseline_48", {}).get("sha256")
    if reg_base_sha and reg_base_sha != baseline_sha256:
        raise SubmissionBuildError(
            f"Registry baseline_48 SHA256 ({reg_base_sha}) does not match active baseline ({baseline_sha256})."
        )

    cand_flips = candidate_info["flips"]
    cand_flip_ids = set(cand_flips.keys())

    for entry in reg_data.get("entries", []):
        sub_key = entry.get("key", "unknown")
        if candidate_info["vector_sha256"] == entry.get("sha256_vector") or (
            entry.get("sha256_lf") and candidate_info["sha256"] == entry.get("sha256_lf")
        ):
            raise SubmissionBuildError(
                f"Candidate (sha256={candidate_info['sha256'][:12]}...) is an exact duplicate of "
                f"previously evaluated submission '{sub_key}' (score={entry.get('score')})."
            )

        entry_flip_ids = {int(x) for x in entry.get("flipped_ids", [])}
        entry_flips_dict = {int(k): str(v) for k, v in entry.get("flips_dict", {}).items()}
        if entry_flip_ids and cand_flip_ids == entry_flip_ids:
            if all(cand_flips[i] == entry_flips_dict.get(i) for i in cand_flip_ids):
                raise SubmissionBuildError(
                    f"Candidate (flip_set_hash={candidate_info['flip_set_hash'][:12]}...) has the exact "
                    f"same {len(cand_flip_ids)}-flip set as previously evaluated submission '{sub_key}' "
                    f"(score={entry.get('score')})."
                )


def _is_strict_flip_subset(sub_flips: dict[int, str], sup_flips: dict[int, str]) -> bool:
    """Return True iff sub_flips is a strict subset of sup_flips with identical target directions."""
    if len(sub_flips) >= len(sup_flips):
        return False
    return all(i in sup_flips and sup_flips[i] == target for i, target in sub_flips.items())


def evaluate_and_build(
    probe_list: list[dict[str, Any]],
    s48_dict: dict[int, str],
    reg_data: dict[str, Any] | None = None,
    base_score: float = 0.94633,
    out_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Content-addressed evaluation of scored probes and strict disjoint/nested combination builder.

    Parameters
    ----------
    probe_list : list of dicts
        Each entry must contain 'file' and 'score', and optionally 'sha256' (or 'expected_sha256'),
        'base_sha256' (or 'baseline_sha256'), and 'base_id'.
    s48_dict : dict[int, str]
        Baseline reference predictions mapping ID (1..30000) -> {'new', 'used'}.
    reg_data : dict or None
        Loaded scored_registry.json dictionary used to block historical duplicates.
    base_score : float
        Public leaderboard baseline score (default 0.94633 for Submission 48).
    out_csv : Path, str, or None
        Destination path to write the combined CSV when >= 2 disjoint winning blocks are resolved.

    Returns
    -------
    dict[str, Any]
        Structured decision and artifact metadata.
    """
    norm_s48, baseline_sha256 = validate_baseline_dict(s48_dict)
    validated_base_score = validate_score(base_score, "base_score")

    if not isinstance(probe_list, list) or len(probe_list) == 0:
        raise SubmissionBuildError("probe_list must be a non-empty list of probe evaluation records.")

    evaluated_probes: list[dict[str, Any]] = []
    seen_file_shas: set[str] = set()
    seen_flip_hashes: set[str] = set()

    for idx, entry in enumerate(probe_list):
        if not isinstance(entry, dict) or "file" not in entry:
            raise SubmissionBuildError(f"Probe entry at index {idx} must be a dict containing 'file'.")

        file_path = entry["file"]
        score = validate_score(entry.get("score"), str(file_path))
        expected_sha = entry.get("expected_sha256") or entry.get("sha256")
        entry_base_sha = entry.get("base_sha256") or entry.get("baseline_sha256")
        entry_base_id = entry.get("base_id")

        cand = load_and_validate_candidate_csv(
            csv_path=file_path,
            s48_dict=norm_s48,
            baseline_sha256=baseline_sha256,
            expected_sha256=expected_sha,
            entry_base_sha256=entry_base_sha,
            entry_base_id=entry_base_id,
        )

        _check_not_in_registry(cand, reg_data, baseline_sha256)

        if cand["sha256"] in seen_file_shas or cand["flip_set_hash"] in seen_flip_hashes:
            raise SubmissionBuildError(
                f"Duplicate probe detected within input probe_list: "
                f"sha256={cand['sha256'][:12]}..., flip_set_hash={cand['flip_set_hash'][:12]}..."
            )
        seen_file_shas.add(cand["sha256"])
        seen_flip_hashes.add(cand["flip_set_hash"])

        delta = round(score - validated_base_score, 8)
        if delta > 0:
            outcome = "WIN"
        elif delta == 0:
            outcome = "TIE"
        else:
            outcome = "LOSS"

        probe_record = {
            **cand,
            "probe_index": idx,
            "score": score,
            "delta": delta,
            "outcome": outcome,
        }
        evaluated_probes.append(probe_record)

    # Step 1: Check nested subset relationships (A strict subset of B) across all evaluated probes
    superseded_flip_hashes: set[str] = set()
    rejected_superset_flip_hashes: set[str] = set()
    nested_contrasts: list[dict[str, Any]] = []
    nested_participants: set[str] = set()

    for i, p_sub in enumerate(evaluated_probes):
        for j, p_sup in enumerate(evaluated_probes):
            if i == j:
                continue
            if not _is_strict_flip_subset(p_sub["flips"], p_sup["flips"]):
                continue

            marginal_ids = sorted(set(p_sup["flips"].keys()) - set(p_sub["flips"].keys()))
            marginal_flips = {mid: p_sup["flips"][mid] for mid in marginal_ids}
            marginal_flip_set_hash = compute_flip_set_hash(marginal_flips, norm_s48)
            marginal_delta = round(p_sup["score"] - p_sub["score"], 8)

            # Forbidden Case (Scenario 5): Superset B won, but evaluated subset A lost or tied!
            # We MUST NOT extract B \ A (unprobed on its own) and MUST NOT promote B (contains refuted A).
            if p_sup["outcome"] == "WIN" and p_sub["outcome"] in {"TIE", "LOSS"}:
                raise SubmissionBuildError(
                    f"Unsafe nested contrast: superset probe (flip_set_hash={p_sup['flip_set_hash'][:12]}..., "
                    f"n_flips={p_sup['n_flips']}, score={p_sup['score']:.5f}) beat baseline, but its evaluated "
                    f"strict subset probe (flip_set_hash={p_sub['flip_set_hash'][:12]}..., n_flips={p_sub['n_flips']}, "
                    f"score={p_sub['score']:.5f} <= base={validated_base_score:.5f}) did NOT win. "
                    f"Refusing to extract the unprobed {len(marginal_ids)}-flip exclusive subset B \\ A "
                    f"(IDs={marginal_ids}, marginal_hash={marginal_flip_set_hash[:12]}...) without independent "
                    f"evaluation, and refusing to promote superset B containing non-winning subset A."
                )

            # Valid Nested Contrast requires subset A to have won (score(A) > base_score)
            if p_sub["outcome"] == "WIN":
                if (
                    p_sub["flip_set_hash"] in nested_participants
                    or p_sup["flip_set_hash"] in nested_participants
                ):
                    raise SubmissionBuildError(
                        "Ambiguous multi-level or branching nested contrast detected across probes."
                    )
                nested_participants.add(p_sub["flip_set_hash"])
                nested_participants.add(p_sup["flip_set_hash"])

                if marginal_delta > 0 and p_sup["outcome"] == "WIN":
                    # Marginal flips B \ A improved over winning subset A -> resolve {A, B} to B
                    superseded_flip_hashes.add(p_sub["flip_set_hash"])
                    resolution = "PROMOTE_SUPERSET_B"
                    resolved_hash = p_sup["flip_set_hash"]
                else:
                    # Marginal flips B \ A tied or hurt relative to A -> reject B \ A, resolve {A, B} to A
                    rejected_superset_flip_hashes.add(p_sup["flip_set_hash"])
                    resolution = "RETAIN_SUBSET_A_REJECT_MARGIN"
                    resolved_hash = p_sub["flip_set_hash"]

                nested_contrasts.append(
                    {
                        "subset_flip_set_hash": p_sub["flip_set_hash"],
                        "subset_sha256": p_sub["sha256"],
                        "subset_n_flips": p_sub["n_flips"],
                        "subset_score": p_sub["score"],
                        "superset_flip_set_hash": p_sup["flip_set_hash"],
                        "superset_sha256": p_sup["sha256"],
                        "superset_n_flips": p_sup["n_flips"],
                        "superset_score": p_sup["score"],
                        "marginal_ids": marginal_ids,
                        "marginal_n_flips": len(marginal_ids),
                        "marginal_flip_set_hash": marginal_flip_set_hash,
                        "marginal_delta": marginal_delta,
                        "resolution": resolution,
                        "resolved_flip_set_hash": resolved_hash,
                    }
                )

    # Step 2: Collect resolved winning blocks
    resolved_winners: list[dict[str, Any]] = []
    for p in evaluated_probes:
        if p["outcome"] != "WIN":
            continue
        if p["flip_set_hash"] in superseded_flip_hashes:
            continue
        if p["flip_set_hash"] in rejected_superset_flip_hashes:
            continue
        resolved_winners.append(p)

    # Step 3: Enforce strict pairwise disjointness across all resolved winning blocks,
    # and forbid unresolved partial overlap between any resolved winner and any non-winning probe.
    valid_nested_pairs = {
        (nc["subset_flip_set_hash"], nc["superset_flip_set_hash"]) for nc in nested_contrasts
    }

    for w in resolved_winners:
        for p in evaluated_probes:
            if w["flip_set_hash"] == p["flip_set_hash"]:
                continue
            pair_hashes = (w["flip_set_hash"], p["flip_set_hash"])
            rev_pair_hashes = (p["flip_set_hash"], w["flip_set_hash"])
            if pair_hashes in valid_nested_pairs or rev_pair_hashes in valid_nested_pairs:
                continue
            ids_w = set(w["flips"].keys())
            ids_p = set(p["flips"].keys())
            overlap_ids = sorted(ids_w & ids_p)
            if overlap_ids and p["outcome"] in {"TIE", "LOSS"}:
                only_w = sorted(ids_w - ids_p)
                raise SubmissionBuildError(
                    f"Winning probe (flip_set_hash={w['flip_set_hash'][:12]}..., n_flips={w['n_flips']}, "
                    f"score={w['score']:.5f}) overlaps on {len(overlap_ids)} IDs ({overlap_ids}) with "
                    f"non-winning probe (flip_set_hash={p['flip_set_hash'][:12]}..., n_flips={p['n_flips']}, "
                    f"score={p['score']:.5f}, outcome={p['outcome']}). "
                    f"Refusing to extract unprobed exclusive subset W \\ P ({len(only_w)} IDs: {only_w}) "
                    f"without independent evaluation."
                )

    for idx_a in range(len(resolved_winners)):
        for idx_b in range(idx_a + 1, len(resolved_winners)):
            w_a = resolved_winners[idx_a]
            w_b = resolved_winners[idx_b]
            ids_a = set(w_a["flips"].keys())
            ids_b = set(w_b["flips"].keys())
            overlap_ids = sorted(ids_a & ids_b)
            if overlap_ids:
                only_a = sorted(ids_a - ids_b)
                only_b = sorted(ids_b - ids_a)
                raise SubmissionBuildError(
                    f"Cannot combine partially overlapping winning blocks! "
                    f"Block 1 (flip_set_hash={w_a['flip_set_hash'][:12]}..., n_flips={w_a['n_flips']}, "
                    f"score={w_a['score']:.5f}) and Block 2 (flip_set_hash={w_b['flip_set_hash'][:12]}..., "
                    f"n_flips={w_b['n_flips']}, score={w_b['score']:.5f}) share {len(overlap_ids)} overlapping "
                    f"IDs: {overlap_ids}. "
                    f"Refusing to combine B1 U B2 (overlapping deltas are non-additive) and refusing to extract "
                    f"unvalidated difference subsets B1 \\ B2 ({len(only_a)} IDs: {only_a}) or "
                    f"B2 \\ B1 ({len(only_b)} IDs: {only_b})."
                )

    # Step 4: Handle outcomes based on count of resolved disjoint winning blocks
    if len(resolved_winners) == 0:
        return {
            "status": "NO_WINNERS",
            "output_written": False,
            "out_csv": None,
            "sha256": None,
            "vector_sha256": None,
            "baseline_sha256": baseline_sha256,
            "base_score": validated_base_score,
            "flip_set_hash": None,
            "n_flips": 0,
            "flips": {},
            "directions": {},
            "expected_score": None,
            "additive_delta_sum_prohibited": True,
            "evaluated_probes": evaluated_probes,
            "nested_contrasts": nested_contrasts,
            "resolved_winners": [],
        }

    if len(resolved_winners) == 1:
        sole = resolved_winners[0]
        return {
            "status": "SINGLE_WINNER_ALREADY_SCORED",
            "output_written": False,
            "out_csv": None,
            "sha256": sole["sha256"],
            "vector_sha256": sole["vector_sha256"],
            "baseline_sha256": baseline_sha256,
            "base_score": validated_base_score,
            "flip_set_hash": sole["flip_set_hash"],
            "n_flips": sole["n_flips"],
            "flips": dict(sole["flips"]),
            "directions": dict(sole["directions"]),
            "expected_score": None,
            "additive_delta_sum_prohibited": True,
            "evaluated_probes": evaluated_probes,
            "nested_contrasts": nested_contrasts,
            "resolved_winners": [sole],
        }

    # Step 5: >= 2 strictly disjoint winning blocks -> build disjoint union
    combined_flips: dict[int, str] = {}
    combined_directions: dict[int, str] = {}
    flip_provenance: dict[int, str] = {}

    for w in resolved_winners:
        for fid, target_cond in w["flips"].items():
            if fid in combined_flips:
                raise SubmissionBuildError(f"Internal disjointness violation on ID {fid}.")
            combined_flips[fid] = target_cond
            combined_directions[fid] = w["directions"][fid]
            flip_provenance[fid] = w["flip_set_hash"]

    combined_labels = {
        i: combined_flips.get(i, norm_s48[i]) for i in range(1, EXPECTED_N_ROWS + 1)
    }
    combined_vector_sha256 = compute_vector_sha256(combined_labels)
    combined_flip_set_hash = compute_flip_set_hash(combined_flips, norm_s48)

    # Verify the combined submission is not identical to any input probe or registry submission
    if combined_flip_set_hash in seen_flip_hashes:
        raise SubmissionBuildError(
            f"Combined flip set (flip_set_hash={combined_flip_set_hash[:12]}...) is identical to an "
            f"already evaluated input probe."
        )

    synthetic_cand_info = {
        "sha256": combined_vector_sha256,
        "vector_sha256": combined_vector_sha256,
        "flip_set_hash": combined_flip_set_hash,
        "n_flips": len(combined_flips),
        "flips": combined_flips,
    }
    _check_not_in_registry(synthetic_cand_info, reg_data, baseline_sha256)

    final_file_sha256: str | None = None
    out_csv_str: str | None = None
    output_written = False

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["ID,condition\n"]
        for i in range(1, EXPECTED_N_ROWS + 1):
            lines.append(f"{i},{combined_labels[i]}\n")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)

        # Re-validate written CSV file using both our strict loader and submission_guard
        written_info = load_and_validate_candidate_csv(
            csv_path=out_path,
            s48_dict=norm_s48,
            baseline_sha256=baseline_sha256,
        )
        if written_info["flip_set_hash"] != combined_flip_set_hash:
            raise SubmissionBuildError("Written CSV flip_set_hash mismatch after serialization.")

        if reg_data is not None:
            passed, guard_rep = check_submission_file(out_path, reg_data, norm_s48, verbose=False)
            if not passed:
                raise SubmissionBuildError(f"submission_guard rejected combined output: {guard_rep}")

        final_file_sha256 = written_info["sha256"]
        out_csv_str = str(out_path)
        output_written = True

    return {
        "status": "COMBINED_DISJOINT_UNION",
        "output_written": output_written,
        "out_csv": out_csv_str,
        "sha256": final_file_sha256,
        "vector_sha256": combined_vector_sha256,
        "baseline_sha256": baseline_sha256,
        "base_score": validated_base_score,
        "flip_set_hash": combined_flip_set_hash,
        "n_flips": len(combined_flips),
        "n_used_to_new": sum(1 for d in combined_directions.values() if d == "used_to_new"),
        "n_new_to_used": sum(1 for d in combined_directions.values() if d == "new_to_used"),
        "flips": dict(sorted(combined_flips.items())),
        "directions": dict(sorted(combined_directions.items())),
        "flip_provenance": dict(sorted(flip_provenance.items())),
        "expected_score": None,
        "additive_delta_sum_prohibited": True,
        "evaluated_probes": evaluated_probes,
        "nested_contrasts": nested_contrasts,
        "resolved_winners": resolved_winners,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content-Addressed 5th Slot Decision Engine & Disjoint Combiner"
    )
    parser.add_argument(
        "--scores-json",
        type=str,
        required=True,
        help="Path to JSON file: [{'file': 'path/to/cand.csv', 'sha256': '...', 'score': 0.94645}, ...]",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="submissions/20261030/submission_final_combined.csv",
        help="Path to output combined submission CSV (written only when >= 2 disjoint winners exist)",
    )
    parser.add_argument(
        "--base-score",
        type=float,
        default=0.94633,
        help="Baseline public score of Submission 48",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_json)
    if not scores_path.exists():
        sys.exit(f"ERROR: Scores JSON file not found: {scores_path}")

    with open(scores_path, "r", encoding="utf-8") as f:
        probe_list = json.load(f)

    with open(S48_JSON, "r", encoding="utf-8") as f:
        s48_dict = {int(k): v for k, v in json.load(f).items()}

    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    try:
        result = evaluate_and_build(
            probe_list=probe_list,
            s48_dict=s48_dict,
            reg_data=reg_data,
            base_score=args.base_score,
            out_csv=args.out_csv,
        )
    except SubmissionBuildError as exc:
        sys.exit(f"SUBMISSION BUILD BLOCKED: {exc}")

    print("=" * 80)
    print(
        f"CONTENT-ADDRESSED EVALUATION COMPLETE | BASELINE SHA256: {result['baseline_sha256'][:16]}... "
        f"(Score: {result['base_score']:.5f})"
    )
    print("=" * 80)
    for p in result["evaluated_probes"]:
        print(
            f"  Probe #{p['probe_index']}: sha256={p['sha256'][:12]}... | "
            f"flip_set_hash={p['flip_set_hash'][:12]}... | flips={p['n_flips']} "
            f"(u->n={p['n_used_to_new']}, n->u={p['n_new_to_used']}) | "
            f"score={p['score']:.5f} (delta={p['delta']:+.5f}) -> {p['outcome']}"
        )

    if result["nested_contrasts"]:
        print("\nNested Contrasts Resolved:")
        for nc in result["nested_contrasts"]:
            print(
                f"  subset={nc['subset_flip_set_hash'][:12]}... ({nc['subset_n_flips']} flips, "
                f"score={nc['subset_score']:.5f}) < superset={nc['superset_flip_set_hash'][:12]}... "
                f"({nc['superset_n_flips']} flips, score={nc['superset_score']:.5f}) | "
                f"marginal_delta={nc['marginal_delta']:+.5f} -> {nc['resolution']}"
            )

    print(f"\nFINAL STATUS: {result['status']}")
    if result["status"] == "NO_WINNERS":
        print("No probe beat baseline S48. Retain Submission 48; no new CSV emitted.")
    elif result["status"] == "SINGLE_WINNER_ALREADY_SCORED":
        sole = result["resolved_winners"][0]
        print(
            f"Single winning block (flip_set_hash={sole['flip_set_hash'][:12]}..., "
            f"sha256={sole['sha256'][:12]}..., score={sole['score']:.5f}) is already scored on Kaggle."
        )
        print("Refusing to emit a duplicate CSV for the 5th slot (prevents Submission 76 error).")
    elif result["status"] == "COMBINED_DISJOINT_UNION":
        print(
            f"Combined {len(result['resolved_winners'])} strictly disjoint winning blocks -> "
            f"{result['n_flips']} total flips (flip_set_hash={result['flip_set_hash'][:16]}...)."
        )
        print("NOTE: Public deltas are NOT summed; expected_score is intentionally None.")
        if result["output_written"]:
            print(f"Output CSV written & verified: {result['out_csv']} (SHA256: {result['sha256']})")


if __name__ == "__main__":
    main()
