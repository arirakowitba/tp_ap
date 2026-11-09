#!/usr/bin/env python3
"""Fail closed when a candidate/submission builder depends on blocked artifacts.

This is a static provenance gate, not a model validator. It prevents accidental promotion of
known-invalid dependency families and explicitly blocked portfolio namespaces. Forensic analysis
may preserve those files, but they cannot pass a promotion workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FORBIDDEN = {
    "bayesian_solver_103vec": "underidentified per-ID solver",
    "bayesian_transductive_item_effects_103vec": "score-derived item effects",
    "transductive_deep_models_103vec": "model trained with test pseudo-labels",
    "transductive_test_probabilities_103vec": "probabilities from blocked pseudo-label model",
    "bayesian_solver_108vec": "underidentified per-ID solver with no prequential validation",
    "bayesian_transductive_item_effects_108vec": "score-derived item effects from a rank-deficient system",
    "transductive_deep_models_108vec": "model family trained with test pseudo-labels",
    "portfolio_20261109_podium_attack": "portfolio namespace blocked by CX-0132",
    "map_effect_vs_d1d2": "non-identifiable item effect",
    "expected_gain_vs_s4": "uncalibrated score-derived item gain",
    "portfolio_20261108_podium_target_09490": "portfolio namespace blocked by CX-0128/CX-0129/CX-0131",
    "generate_slot2_pure_used_top6.py": "manual per-ID flip generator blocked by CX-0134",
    "generate_slot3_pure_used_top12.py": "manual per-ID flip generator blocked by CX-0134",
    "generate_slot4_podium_attack_top15.py": "manual per-ID flip generator blocked by CX-0134",
    "generate_slot5_podium_climax_top14.py": "manual per-ID flip generator blocked by CX-0134",
    "submission_20261109_01_surgical_rank5_capture_top5.csv": "manual per-ID candidate blocked by CX-0134",
    "submission_20261109_02_pure_used_capture_top6.csv": "manual per-ID candidate blocked by CX-0134",
    "submission_20261109_03_pure_used_expansion_top12.csv": "manual per-ID candidate blocked by CX-0134",
    "submission_20261109_04_podium_attack_top15.csv": "manual per-ID candidate blocked by CX-0134",
    "submission_20261109_05_podium_climax_top14.csv": "manual per-ID candidate blocked by CX-0134",
}

CANDIDATE_HINTS = (
    "submission",
    "portfolio",
    "candidate",
    "promote",
    "podium",
)


def audit_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    low = text.lower()
    path_low = path.as_posix().lower()
    candidate_context = any(h in low or h in path_low for h in CANDIDATE_HINTS)
    hits = [
        {"token": token, "reason": reason}
        for token, reason in FORBIDDEN.items()
        if token.lower() in low or token.lower() in path_low
    ]
    return {
        "path": path.as_posix(),
        "candidate_context": candidate_context,
        "forbidden_dependencies": hits,
        "blocked": bool(candidate_context and hits),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    reports = []
    missing = []
    for path in args.paths:
        if not path.is_file():
            missing.append(path.as_posix())
            continue
        reports.append(audit_file(path))

    payload = {
        "status": "BLOCKED" if missing or any(r["blocked"] for r in reports) else "PASS",
        "missing": missing,
        "reports": reports,
        "policy": "CX-0128/CX-0129/CX-0131/CX-0132/CX-0134",
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 2 if payload["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
