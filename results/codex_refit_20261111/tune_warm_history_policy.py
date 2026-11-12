"""Fit a warm-only probability policy and test it on separate archived rows."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_structural_blends import score, sha


def read(path, base):
    df = pd.read_csv(path, dtype={"id": str}).rename(columns={base: "base"})
    assert df.id.is_unique
    return df


def features(d):
    b = logit(d.base.clip(1e-6, 1-1e-6))
    h = logit(d.recent_history.clip(1e-6, 1-1e-6))
    return np.column_stack((b, h-b))


def blend(d, w):
    return (1-w) * d.base.to_numpy() + w * d.recent_history.to_numpy()


def main():
    archive = Path(sys.argv[1])
    a_path = Path("recent_seller_history/pairs_private.csv")
    b_path = Path("recent_seller_history_replay/pairs_private.csv")
    for p in (a_path, b_path):
        assert sha(p) == json.loads(p.with_name("report.json").read_text())["pairs_sha256"]
    a, b = read(a_path, "attributes"), read(b_path, "lgb")
    ids = set(a.id) | set(b.id)
    with zipfile.ZipFile(archive) as z, z.open("train_data.jsonlines") as f:
        raw = {str(row["id"]): (str(row["seller_id"]), str(row["category_id"]), row["condition"])
               for row in map(json.loads, f) if str(row["id"]) in ids}
    assert set(raw) == ids
    for d in (a, b):
        d["seller"], d["category"], labels = np.asarray([raw[item] for item in d.id]).T
        assert np.array_equal(labels == "used", d.y.to_numpy() == 1)
    confirm = {"exclusive": a.loc[a.fresh].copy(), "replay": b.copy()}
    assert not set(confirm["exclusive"].id) & set(confirm["replay"].id)
    # The warm fit excludes every seller in either warm evaluation cohort.
    forbidden = set().union(*(set(d.loc[d.seen, "seller"]) for d in confirm.values()))
    tune = a.loc[a.seen & ~a.fresh & ~a.id.isin(b.id) & ~a.seller.isin(forbidden)].copy()
    assert len(tune) > 1000
    assert not set(tune.seller) & forbidden
    y = tune.y.to_numpy()
    x = features(tune)
    grid = (0., .25, .5, .75, 1.)
    dev = [(w, log_loss(y, blend(tune, w))) for w in grid]
    best_w = min(dev, key=lambda item: item[1])[0]
    splits = list(StratifiedGroupKFold(4, shuffle=True, random_state=28111218).split(x, y, tune.seller))
    reg_oof = np.zeros(len(tune))
    for fit, ev in splits:
        assert not set(tune.seller.iloc[fit]) & set(tune.seller.iloc[ev])
        reg_oof[ev] = LogisticRegression(C=.1).fit(x[fit], y[fit]).predict_proba(x[ev])[:,1]
    assert np.all((reg_oof > 0) & (reg_oof < 1))
    thresholds = expit(np.array([-.12, -.06, 0., .06, .12]))
    # Select one threshold using only grouped OOF predictions. Tie goes to .5.
    t = max(thresholds, key=lambda v: (np.sum((reg_oof >= v) == y), -abs(v-.5)))
    model = LogisticRegression(C=.1).fit(x, y)
    out = Path("warm_history_policy")
    out.mkdir(exist_ok=True)
    report = {
        "scope": "Development replay from archived paired scores. Meta-fit sellers and IDs disjoint from warm evaluation; "
                 "base fits of the two panels can overlap, and prior global evaluations may have used these rows.",
        "archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
        "input_sha256": [sha(a_path), sha(b_path)], "tune_rows": len(tune),
        "tune_sellers": tune.seller.nunique(), "warm_confirm_seller_overlap_with_tune": 0,
        "selected_weight": best_w, "weight_logloss": dev, "meta_threshold": float(t),
        "meta_parameters": {"intercept": model.intercept_.tolist(), "coef": model.coef_.tolist()},
        "meta_oof_logloss": log_loss(y, reg_oof), "metrics": {}, "private_pairs_sha256": {},
    }
    for panel, d in confirm.items():
        group = d.seen.to_numpy(dtype=bool)
        actual = d.y.to_numpy()
        base = d.base.to_numpy()
        fixed = np.where(group, blend(d, .5), base)
        selected = np.where(group, blend(d, best_w), base)
        lp = model.decision_function(features(d))
        meta = np.where(group, expit(lp), base)
        # Shift only for the binary decision. Evaluate log loss on raw probabilities.
        meta_shift = np.where(group, expit(lp-logit(t)), base)
        arms = {"fixed_50": (fixed, fixed), "selected_weight": (selected, selected),
                "logistic": (meta_shift, meta)}
        report["metrics"][panel] = {}
        for name, (decision, prob) in arms.items():
            d[name] = prob
            d[name+"_hard"] = decision >= .5
            stats = {}
            for key, mask in (("all", np.ones(len(d), bool)), ("seen", group)):
                stats[key] = score(actual[mask], decision[mask], base[mask],
                                   d.seller.to_numpy()[mask], d.category.to_numpy()[mask], iterations=2500)
                stats[key]["logloss"] = log_loss(actual[mask], prob[mask])
            disagreement = (decision >= .5) != (base >= .5)
            stats["disagreement_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(d.id[disagreement])).encode()).hexdigest()
            report["metrics"][panel][name] = stats
        pp = out / f"{panel}_pairs_private.csv"
        d.drop(columns=["seller","category"]).to_csv(pp,index=False)
        report["private_pairs_sha256"][panel] = sha(pp)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("tune", len(tune), tune.seller.nunique(), "weight", best_w, "threshold", t)
    for panel, results in report["metrics"].items():
        print(panel, {n: v["seen"] for n, v in results.items()})


if __name__ == "__main__":
    main()
