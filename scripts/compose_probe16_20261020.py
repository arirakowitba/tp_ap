"""Build the first adaptive probe of 20 Oct from scored submission 15.

All test labels stay predictions. Public scores later identify only the net
contribution of this entire cohort, not the ground truth of individual rows.
"""

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd


STAGE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AP2026Q2_EXPERIMENT_ROOT", STAGE.parents[1]))
PRIOR = STAGE / "submissions" / "20261019"
DEST = STAGE / "submissions" / "20261020"
CONTROL = PRIOR / "submission_15_top16_Q4_sobre_mejor11.csv"
CONTROL_SHA = "ce425f4f88720ae74e1a415f92b50f75279ae3d26c1deae278c3c7bb13c55891"
MODEL_DIR = ROOT / "experiment_outputs_today_20261019"
REFERENCE = ROOT / "experiment_outputs_finalday_20261018_2110" / "final_case_ranking_NOT_calibrated.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert sha(CONTROL) == CONTROL_SHA
    base = pd.read_csv(CONTROL)
    assert base.shape == (30000, 2) and list(base.columns) == ["ID", "condition"]
    assert np.array_equal(base.ID.to_numpy(), np.arange(1, 30001))
    assert set(base.condition) == {"new", "used"}

    cases = pd.read_csv(REFERENCE)
    assert np.array_equal(cases.ID.to_numpy(), base.ID.to_numpy())
    previously_tested = pd.read_csv(PRIOR / "probes_11_13_case_audit.csv")
    assert len(previously_tested) == 176
    raw = pd.read_pickle(ROOT / "experiment_outputs_warranty_text_20261017" / "raw_text.pkl")
    assert len(raw) == 100000
    catalog = raw.iloc[70000:].reset_index(drop=True)
    # Check raw row alignment against the seller and category audited yesterday.
    pos = previously_tested.ID.to_numpy(int) - 1
    assert np.array_equal(catalog.seller.iloc[pos].astype(str), previously_tested.seller_id.astype(str))
    assert np.array_equal(catalog.category.iloc[pos].astype(str), previously_tested.category_id.astype(str))
    cases["seller_id"] = catalog.seller.astype(str).to_numpy()
    cases["category_id"] = catalog.category.astype(str).to_numpy()

    for name, col in (("robust_depthwise_v2", "new_depth_alt"),
                      ("rich_symmetric_v2", "new_symmetric_alt")):
        prediction = np.load(MODEL_DIR / f"{name}_oof.npz")["test"]
        assert prediction.shape == (30000,)
        cases[col] = np.where(cases.historical_alternative_label.eq("used"),
                              prediction, 1 - prediction)

    cv = cases.rotated_cv_probability_alternative.to_numpy(float)
    depth = cases.new_depth_alt.to_numpy(float)
    sym = cases.new_symmetric_alt.to_numpy(float)
    nn = cases.new_matched_no_public_nn_blend_probability_alternative.to_numpy(float)
    compact = cases.new_compact_probability_alternative.to_numpy(float)
    old = cases.old_reference_probability_alternative.to_numpy(float)
    cases["cross_family_support"] = sum((x > .5).astype(int) for x in
        (cv, depth, sym, compact, nn))
    # A prioritization score, NOT an estimated probability of a correct label.
    cases["conservative_rank_not_probability"] = (
        .28 * np.minimum.reduce((cv, depth, sym))
        + .22 * np.median(np.stack((cv, depth, sym)), axis=0)
        + .22 * nn + .16 * compact + .12 * old
    )

    # Only rows never evaluated in the latest rounds, with the historic
    # prediction still present in 15. This preserves every previous flip.
    pool = cases.loc[
        cases.uncertain_pool & ~cases.currently_scored & ~cases.currently_pending
        & ~cases.ID.isin(previously_tested.ID)
        & cases.historical_control_label.eq(base.condition.to_numpy())
    ].sort_values(["cross_family_support", "conservative_rank_not_probability"],
                  ascending=False)
    assert len(pool) >= 1000
    chosen = []
    chosen_sellers = set()
    per_category = {}
    per_class = {"new": 0, "used": 0}
    for row in pool.itertuples():
        target = row.historical_alternative_label
        if (per_class[target] >= 32 or row.seller_id in chosen_sellers
                or per_category.get(row.category_id, 0) >= 5):
            continue
        chosen.append(int(row.ID))
        chosen_sellers.add(row.seller_id)
        per_category[row.category_id] = per_category.get(row.category_id, 0) + 1
        per_class[target] += 1
        if len(chosen) == 64:
            break
    assert len(chosen) == len(set(chosen)) == 64
    assert per_class == {"new": 32, "used": 32}
    selected = pool.loc[pool.ID.isin(chosen)].copy().sort_values("ID")
    assert selected.seller_id.nunique() == 64
    assert selected.category_id.value_counts().max() <= 5
    assert selected.cross_family_support.ge(2).all()

    checked = pd.read_csv(ROOT / "verified_public_labels.csv")
    verify = dict(zip(checked.item_id.astype(str), checked.condition.astype(str)))
    for item, label in verify.items():
        row = cases.loc[cases.item_id.astype(str).eq(item)]
        assert len(row) == 1
        assert base.condition.iloc[int(row.ID.iloc[0]) - 1] == label
        assert int(row.ID.iloc[0]) not in chosen

    out = base.copy()
    alternative = dict(zip(selected.ID, selected.historical_alternative_label))
    mask = out.ID.isin(chosen)
    assert np.array_equal(out.loc[mask, "condition"].to_numpy(),
                          selected.historical_control_label.to_numpy())
    out.loc[mask, "condition"] = out.loc[mask, "ID"].map(alternative).to_numpy()
    changed = set(out.loc[out.condition.ne(base.condition), "ID"].to_numpy(int))
    assert changed == set(chosen)
    assert len(out) == 30000 and set(out.condition) == {"new", "used"}
    DEST.mkdir(parents=True, exist_ok=True)
    csv = DEST / "submission_16_frescos_consenso64_sobre15.csv"
    out.to_csv(csv, index=False)
    assert pd.read_csv(csv).equals(out)

    selected["label_in_15"] = selected.historical_control_label
    selected["label_in_16"] = selected.historical_alternative_label
    cols = ["ID", "item_id", "label_in_15", "label_in_16", "cross_family_support",
            "conservative_rank_not_probability", "rotated_cv_probability_alternative",
            "new_depth_alt", "new_symmetric_alt", "new_compact_probability_alternative",
            "new_matched_no_public_nn_blend_probability_alternative",
            "old_reference_probability_alternative", "full70k_seed_votes_alternative_out_of_7",
            "seller_id", "category_id"]
    audit = DEST / "cohorte_16_auditoria.csv"
    selected[cols].to_csv(audit, index=False)
    assert pd.read_csv(audit).ID.tolist() == sorted(chosen)
    manifest = {
        "date_argentina": "2026-10-20", "reference": CONTROL.name,
        "reference_sha256": sha(CONTROL), "submission": csv.name,
        "submission_sha256": sha(csv), "audit": audit.name,
        "audit_sha256": sha(audit), "rows": len(out), "flips": len(chosen),
        "to_new": per_class["new"], "to_used": per_class["used"],
        "distinct_sellers": selected.seller_id.nunique(),
        "distinct_categories": selected.category_id.nunique(),
        "support_distribution": {str(k): int(v) for k, v in
                                 selected.cross_family_support.value_counts().sort_index().items()},
        "IDs": sorted(chosen),
        "score_15": "0.94523", "score_16": None,
        "interpretation": "score_16 - score_15 identifies an aggregate public contribution only",
    }
    dest_manifest = DEST / "prueba_16_manifiesto.json"
    dest_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"submission": str(csv.resolve()), "manifest": str(dest_manifest.resolve()),
                      "audit": str(audit.resolve()), "sha256": sha(csv),
                      "flips": len(chosen), "support": manifest["support_distribution"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
