"""Audit observed probes, infer cohort effects and export a conditional candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ap2026q2_iterative_submissions import (
    choose_combination, cohort_confidence, infer_cohorts, public_correct, validate_baseline,
)
from ap2026q2_submission_review import displayed_score

DAY = ROOT / "submissions/20261020"
OUT = ROOT / "results/revision_submissions"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads((OUT / "feedback_iterativo_20261020.json").read_text(encoding="utf8"))
    manifest = json.loads((DAY / "pruebas_17_a_20_manifiesto.json").read_text(encoding="utf8"))
    manifest16 = json.loads((DAY / "prueba_16_manifiesto.json").read_text(encoding="utf8"))
    base_path = ROOT / record["baseline"]["file"]
    assert sha(base_path) == record["baseline"]["sha256"]
    baseline = pd.read_csv(base_path)
    validate_baseline(baseline)
    audit16_path = ROOT / record["groups_source"]
    assert sha(audit16_path) == manifest16["audit_sha256"]
    audit16 = pd.read_csv(audit16_path)
    split = np.where(audit16.full70k_seed_votes_alternative_out_of_7.ge(4), "A", "B")
    groups = {name: audit16.loc[split == name, ["ID", "label_in_16"]].rename(
        columns={"label_in_16": "alternative_label"}) for name in ("A", "B")}
    for name in ("C", "D", "E"):
        path = DAY / manifest["audits"][name]["file"]
        assert sha(path) == manifest["audits"][name]["sha256"]
        groups[name] = pd.read_csv(path)
    assert [len(groups[g]) for g in "ABCDE"] == [32, 32, 64, 64, 128]
    assert len(set().union(*(set(frame.ID) for frame in groups.values()))) == 320
    scored = []
    reference_count = public_correct(record["baseline"]["score"])
    for number, included in record["experiments"].items():
        item = ({"file": manifest16["submission"], "sha256": manifest16["submission_sha256"]}
                if number == "16" else manifest["submissions"][number])
        path = DAY / item["file"]
        assert sha(path) == item["sha256"]
        actual = pd.read_csv(path)
        validate_baseline(actual)
        wanted = pd.concat([groups[g][["ID", "alternative_label"]] for g in included])
        assert wanted.ID.is_unique
        changed = actual.condition.ne(baseline.condition)
        assert set(actual.loc[changed, "ID"]) == set(wanted.ID)
        assert actual.set_index("ID").loc[wanted.ID, "condition"].to_list() == wanted.alternative_label.to_list()
        count = public_correct(record["scores"][number])
        scored.append({"submission": number, "score_observed": record["scores"][number],
                       "public_correct_assuming_21000": count,
                       "public_gain_vs_15": count - reference_count, "flips": len(wanted)})
    summary, joint = infer_cohorts({g: len(v) for g, v in groups.items()},
        record["experiments"], record["scores"], record["baseline"]["score"])
    assert len(joint) == 1
    gain = {g: int(joint.iloc[0][g]) for g in groups}
    assert gain == dict(A=2, B=-7, C=-18, D=-17, E=-15)
    rows = []
    for g, n in ((g, len(v)) for g, v in groups.items()):
        rows.append({"cohort": g, "n": n, "public_gain_identified": gain[g],
                     **cohort_confidence(n, gain[g])})
    confidence = pd.DataFrame(rows)
    confidence.to_csv(OUT / "cohortes_16_a_20_posterior.csv", index=False)
    sensitivity = []
    for a, b in ((1., 1.), (2., 2.), (4., 4.)):
        for g in groups:
            sensitivity.append({"cohort": g, "beta_a": a, "beta_b": b,
                                **cohort_confidence(len(groups[g]), gain[g], a=a, b=b)})
    pd.DataFrame(sensitivity).to_csv(OUT / "cohortes_16_a_20_sensibilidad.csv", index=False)
    ranking, candidate = choose_combination(summary, joint, groups, baseline)
    assert ranking.iloc[0].cohorts == "A" and int(ranking.iloc[0].flips) == 32
    path = DAY / "submission_consolidada_A32_sobre15_no_enviada.csv"
    candidate.to_csv(path, index=False)
    assert pd.read_csv(path).equals(candidate)
    inferred_count = reference_count + gain["A"]
    result = {
        "source": "Five scores 16-20 explicitly supplied by the user after submission.",
        "assumptions": {"public_rows": 21000, "test_rows": 30000, "display": "truncate",
                        "fixed_public_mask": True, "prior": "Beta(2,2)",
                        "transfer_limitation": "Cohort exchangeability is unverified on hidden rows. Marginal mask priors are used independently across cohorts."},
        "baseline": record["baseline"], "observed_submissions": scored,
        "identified_public_gains": gain,
        "cohort_posterior": confidence.to_dict(orient="records"),
        "conditional_candidate": {"file": path.relative_to(ROOT).as_posix(),
            "sha256": sha(path), "cohorts": ["A"], "flips": 32,
            "public_correct_inferred": inferred_count,
            "public_score_inferred": f"{displayed_score(inferred_count, 21000, 'truncate'):.5f}",
            "public_score_observed": None, "status": "not_submitted",
            "expected_hidden_gain_beta22": float(confidence.set_index("cohort").loc["A", "expected_hidden_gain"]),
            "IDs": groups["A"].ID.astype(int).tolist()},
        "remaining_unknown": "Individual labels and public membership of all rows; E1-E4 separate effects; actual hidden performance.",
        "next_round_design_only": {"status": "not_generated", "anchor_A_gain": 2,
            "known_E_total_gain": -15,
            "experiments": {"21": ["A", "E1", "E2"], "22": ["A", "E1", "E3"],
                            "23": ["A", "E1", "E4"], "24": ["A", "F_new_independent"],
                            "25": ["consolidate_after_21_to_24"]},
            "E1_formula": "(g21+g22+g23 - 3*a - e)/2",
            "F_requirement": "Choose IDs only after auditing available fresh candidates and independent model support."}
    }
    (OUT / "resultados_cruzados_16_a_20.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    (path.with_suffix(".json")).write_text(json.dumps(result["conditional_candidate"], indent=2) + "\n", encoding="utf8")
    manifest["scores_observed_16_to_20"] = record["scores"]
    manifest["result_analysis"] = "results/revision_submissions/resultados_cruzados_16_a_20.json"
    (DAY / "pruebas_17_a_20_manifiesto.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    manifest16["score_16"] = record["scores"]["16"]
    (DAY / "prueba_16_manifiesto.json").write_text(
        json.dumps(manifest16, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({"public_gains": gain, "candidate": result["conditional_candidate"],
                      "posterior": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
