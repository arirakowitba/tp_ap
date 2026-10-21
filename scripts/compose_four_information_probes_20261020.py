"""Build four crossed, audited probes from the scored 15 and fresh CV models.

Never interpret a leaderboard cohort sum as a row-level ground truth.  Run
after grouped CV; scores for 16 through 20 are supplied later by the user.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parents[1]
DAY = STAGE / "submissions" / "20261020"
PRIOR = STAGE / "submissions" / "20261019"
OUT = STAGE / "results" / "new_catboost_20261020"
EXTRA = STAGE / "results" / "new_models_20261020"
BASE15 = PRIOR / "submission_15_top16_Q4_sobre_mejor11.csv"
EXPECTED15 = "ce425f4f88720ae74e1a415f92b50f75279ae3d26c1deae278c3c7bb13c55891"
EXP = {"17": ("A", "C"), "18": ("B", "C"),
       "19": ("B", "D"), "20": ("C", "D", "E")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(pool: pd.DataFrame, n: int, score: pd.Series,
           group: str, used_ids: set[int], used_sellers: set[str],
           min_rank: float = .0) -> pd.DataFrame:
    """Balanced target classes, distinct sellers, ≤4 per category; exact n."""
    ranked = pool.assign(selection_score=score).sort_values(
        ["selection_score", "ID"], ascending=[False, True])
    target = {"new": n // 2, "used": n // 2}
    selected = []
    sellers = set(used_sellers)
    categories: dict[str, int] = {}
    for row in ranked.itertuples():
        if (row.ID in used_ids or row.seller_id in sellers or
            target[row.historical_alternative_label] == 0 or
            categories.get(row.category_id, 0) >= 4 or
            row.rank_not_calibrated < min_rank):
            continue
        selected.append(int(row.ID))
        sellers.add(row.seller_id)
        target[row.historical_alternative_label] -= 1
        categories[row.category_id] = categories.get(row.category_id, 0) + 1
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError(f"{group}: selected {len(selected)}/{n}; loosen a "
                         "documented threshold or retrain before emitting files")
    used_ids.update(selected)
    used_sellers.update(sellers)
    result = ranked[ranked.ID.isin(selected)].copy().sort_values("ID")
    result["cohort"] = group
    assert result.seller_id.nunique() == n and result.category_id.value_counts().max() <= 4
    assert result.historical_alternative_label.value_counts().to_dict() == target.fromkeys(target, n // 2)
    return result


def main() -> None:
    assert sha(BASE15) == EXPECTED15
    # Scored artifacts are evidence: never rewrite them with newly trained models.
    feedback_path = STAGE / "results/revision_submissions/feedback_iterativo_20261020.json"
    if feedback_path.exists():
        feedback = json.loads(feedback_path.read_text(encoding="utf8"))
        observed = {k: v for k, v in feedback["scores"].items() if v is not None}
        if observed:
            frozen = json.loads((DAY / "pruebas_17_a_20_manifiesto.json").read_text(encoding="utf8"))
            for item in (*frozen["submissions"].values(), *frozen["audits"].values()):
                assert sha(DAY / item["file"]) == item["sha256"], "scored artifact changed"
            print(json.dumps({"frozen_scored_round": True, "observed_scores": observed,
                              "action": "Run analyze_crossed_results_20261020.py; design future probes under new filenames."}))
            return
    base = pd.read_csv(BASE15)
    assert list(base.columns) == ["ID", "condition"] and len(base) == 30000
    assert np.array_equal(base.ID.to_numpy(), np.arange(1, 30001))
    audit16 = pd.read_csv(DAY / "cohorte_16_auditoria.csv")
    probe16 = pd.read_csv(DAY / "submission_16_frescos_consenso64_sobre15.csv")
    m16 = json.loads((DAY / "prueba_16_manifiesto.json").read_text(encoding="utf8"))
    assert sha(DAY / m16["submission"]) == m16["submission_sha256"]
    assert sha(DAY / m16["audit"]) == m16["audit_sha256"]
    assert set(audit16.ID) == set(base.loc[base.condition.ne(probe16.condition), "ID"])
    audit16["cohort"] = np.where(audit16.full70k_seed_votes_alternative_out_of_7.ge(4), "A", "B")
    assert audit16.cohort.value_counts().to_dict() == {"A": 32, "B": 32}
    candidates = pd.read_csv(OUT / "future_candidate_ranking_not_calibrated.csv")
    assert len(candidates) == 1448 and candidates.ID.is_unique
    assert candidates.historical_control_label.eq(base.set_index("ID").loc[candidates.ID, "condition"].to_numpy()).all()
    raw = pd.read_pickle(PROJECT / "experiment_outputs_warranty_text_20261017" / "raw_text.pkl")
    assert len(raw) == 100000
    rows = candidates.ID.to_numpy(int) - 1
    candidates["seller_id"] = raw.seller.iloc[70000 + rows].astype(str).to_numpy()
    candidates["category_id"] = raw.category.iloc[70000 + rows].astype(str).to_numpy()
    assert not set(candidates.ID) & set(audit16.ID)
    extra_models = {}
    for name in ("text_linear", "svd_lgbm"):
        file = EXTRA / f"{name}_grouped_cv_predictions.npz"
        if file.exists():
            arr = np.load(file)
            assert arr["test"].shape == (30000,) and arr["oof"].shape == (70000,)
            p = arr["test"][rows]
            candidates[f"{name}_p_alt"] = np.where(
                candidates.historical_alternative_label.eq("used"), p, 1 - p)
            extra_models[name] = file.relative_to(STAGE).as_posix()
    if len(extra_models) not in (0, 2):
        raise ValueError("both complementary models must complete CV before inclusion")
    # Rank weights are prioritization choices, never calibrated label probabilities.
    linear = candidates.get("text_linear_p_alt", pd.Series(.5, index=candidates.index))
    lgbm = candidates.get("svd_lgbm_p_alt", pd.Series(.5, index=candidates.index))
    old = candidates.old_catboost_p_alt
    depth = candidates.v3_depthwise_p_alt
    symmetric = candidates.v3_symmetric_p_alt
    votes = candidates.full70k_seed_votes_alternative_out_of_7 / 7
    high_new = .32 * depth + .21 * symmetric + .20 * old + .11 * linear + .10 * lgbm + .06 * votes
    legacy_disagreement = (.27 * votes + .28 * old + .20 * depth +
                           .09 * symmetric + .08 * lgbm + .08 * linear +
                           .10 * (old - depth).clip(lower=0))
    broad_rank = .34 * candidates.rank_not_calibrated + .24 * depth + .16 * old + .13 * linear + .13 * lgbm
    # Distinct, high signal C/D, then a larger E partitioned for tomorrow's probes.
    selected_ids = set(map(int, audit16.ID))
    selected_sellers = set(map(str, audit16.seller_id))
    c = choose(candidates, 64, high_new, "C", selected_ids, selected_sellers, min_rank=.30)
    d = choose(candidates.loc[candidates.full70k_seed_votes_alternative_out_of_7.ge(2)],
               64, legacy_disagreement, "D", selected_ids, selected_sellers, min_rank=.30)
    remaining = candidates[~candidates.ID.isin(c.ID) & ~candidates.ID.isin(d.ID)]
    # E is four separately auditable groups. Preserve variety for follow-up
    # experiments if the aggregate E score raises a new question.
    e_scores = [
        ("E1_new_model", .48 * depth - .20 * old + .17 * linear + .15 * lgbm),
        ("E2_old_model", .47 * old - .23 * depth + .16 * votes + .14 * lgbm),
        ("E3_disagreement", .50 * (depth - old).abs() + .30 * (linear - lgbm).abs() + .20 * broad_rank),
        ("E4_extension", broad_rank),
    ]
    subgroups = []
    for label, metric in e_scores:
        part = choose(remaining, 32, metric, label, selected_ids, selected_sellers, min_rank=.25)
        part["subgroup"] = label
        part["cohort"] = "E"
        subgroups.append(part)
    e = pd.concat(subgroups, ignore_index=True).sort_values("ID")
    c["subgroup"] = "C"; d["subgroup"] = "D"
    groups = {"A":audit16[audit16.cohort.eq("A")].copy(),
              "B":audit16[audit16.cohort.eq("B")].copy(),
              "C":c,"D":d,"E":e}
    assert len(set().union(*[set(x.ID) for x in groups.values()])) == 320
    for name in ("C","D","E"):
        groups[name]["alternative_label"] = groups[name]["historical_alternative_label"]
    audit_columns = ["ID","item_id","historical_control_label","historical_alternative_label","alternative_label",
                     "cohort","subgroup","selection_score","seller_id","category_id",
                     "old_cross_family_support","support_six","rank_not_calibrated",
                     "full70k_seed_votes_alternative_out_of_7","old_catboost_p_alt",
                     "v3_depthwise_p_alt","v3_symmetric_p_alt"] + [f"{n}_p_alt" for n in extra_models]
    for name in ("C","D","E"):
        groups[name][audit_columns].to_csv(DAY / f"cohorte_{name}_auditoria.csv", index=False)
    # A/B already changed in the pending 16; all new probes use the 15 as control.
    changes = {"A":dict(zip(groups["A"].ID, groups["A"].label_in_16)),
               "B":dict(zip(groups["B"].ID, groups["B"].label_in_16))}
    for name in ("C","D","E"):
        changes[name] = dict(zip(groups[name].ID, groups[name].historical_alternative_label))
    matrix = np.array([[int(x in group) for x in ("A","B","C","D","E")]
                       for group in (("A","B"), *EXP.values())])
    assert matrix.shape == (5, 5) and np.linalg.matrix_rank(matrix) == 5
    checked = pd.read_csv(PROJECT / "verified_public_labels.csv")
    baseline = base.set_index("ID").condition
    for _, row in checked.iterrows():
        located = candidates.loc[candidates.item_id.astype(str).eq(str(row.item_id))]
        assert not len(located) or int(located.ID.iloc[0]) not in selected_ids
    manifest = {"date_argentina":"2026-10-20", "reference":"15",
                "reference_score_observed":"0.94523", "reference_sha256":EXPECTED15,
                "public_fraction_hypothesis":.70,"public_rows_hypothesis":21000,
                "scores_observed_16_to_20":None,
                "cohorts":{"A":32,"B":32,"C":64,"D":64,"E":128},
                "models_extra_used":extra_models,
                "design_matrix_order":["16","17","18","19","20"],
                "design_matrix_columns":["A","B","C","D","E"],
                "design_matrix":matrix.tolist(),
                "audits":{},"submissions":{}}
    for name in ("C","D","E"):
        p = DAY / f"cohorte_{name}_auditoria.csv"
        manifest["audits"][name] = {"file":p.name,"sha256":sha(p),
                                     "IDs":groups[name].ID.astype(int).tolist(),
                                     "subgroups":groups[name].subgroup.value_counts().to_dict()}
    for number, included in EXP.items():
        out = base.copy()
        rows_changed = {}
        for name in included:
            rows_changed.update(changes[name])
        mask = out.ID.isin(rows_changed)
        out.loc[mask,"condition"] = out.loc[mask,"ID"].map(rows_changed)
        assert list(out.columns)==["ID","condition"] and out.shape==(30000,2)
        assert set(out.loc[out.condition.ne(base.condition),"ID"]) == set(rows_changed)
        assert out.ID.equals(base.ID) and set(out.condition)=={"new","used"}
        file = DAY / f"submission_{number}_informacion_cruzada_{'_'.join(included)}_sobre15.csv"
        out.to_csv(file,index=False)
        assert pd.read_csv(file).equals(out)
        manifest["submissions"][number] = {"file":file.name,"sha256":sha(file),
            "cohorts":list(included),"flips":len(rows_changed),
            "to_used":int(out.loc[mask,"condition"].eq("used").sum()),
            "to_new":int(out.loc[mask,"condition"].eq("new").sum())}
    manifest_file=DAY / "pruebas_17_a_20_manifiesto.json"
    manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({"files":manifest["submissions"],"audits":manifest["audits"],
                      "full_rank":bool(np.linalg.matrix_rank(matrix)==5)},ensure_ascii=False))


if __name__=="__main__":
    main()
