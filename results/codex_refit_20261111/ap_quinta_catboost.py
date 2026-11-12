"""Test a new CatBoost refit against Line 3 and build a disjoint manual trial."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from ap_retrieval_trial import evaluate
from refit_ap_line3 import ag, causal_features, parser

OUT = Path("entrega_ap")
FROZEN_SETTINGS = dict(iterations=680, depth=7, learning_rate=.075, l2_leaf_reg=6,
                       thread_count=6, random_seed=20261111, verbose=False,
                       allow_writing_files=False)


def train(tr, te, fit, ids):
    ag.compute_causal_seller_features = causal_features([te["start_time"][ids]])
    splits = list(StratifiedGroupKFold(4, shuffle=True, random_state=61111).split(
        fit, tr["y"][fit], tr["seller_id"][fit]))
    m = ag.build_fold_matrices(fit, [ids], splits, tr, [te],
                               Path(f"line3_today/text_{len(fit)}.npz"))
    model = CatBoostClassifier(**FROZEN_SETTINGS)
    model.fit(m["X_m1_seen_causal_T"], m["y_T"])
    return model.predict_proba(m["X_m1_seen_causal_evals"][0])[:, 1], model


def blended(p, q, weight):
    a = logit(np.clip(p, 1e-5, 1 - 1e-5))
    b = logit(np.clip(q, 1e-5, 1 - 1e-5))
    return expit(weight * a + (1 - weight) * b)


def measure(y, baseline, candidate, seller, margin):
    stats = np.column_stack((np.ones(len(y)), candidate, np.ones(len(y))))
    r = evaluate(stats, baseline, y, seller, (.5, .5 + margin, 1, False))
    r["logloss_raw"] = float(log_loss(y, candidate))
    return r


def main():
    with zipfile.ZipFile(sys.argv[1]) as z:
        def read(name, is_train):
            with z.open(name + "_data.jsonlines") as f:
                return pd.DataFrame(parser.parse_item_record(json.loads(l), is_train, i)
                                    for i, l in enumerate(f))
        df, test = read("train", True), read("test", False)
    assert len(df) == 70000 and len(test) == 30000 and df.id.is_unique and test.id.is_unique
    tr, te = map(ag.precompute_item_intrinsic_arrays, (df, test))
    pairs = pd.read_csv("line3_today/pairs.csv")
    val = pairs.row.to_numpy()
    assert df.id.iloc[val].reset_index(drop=True).equals(pairs.id)
    fit = np.setdiff1d(np.arange(len(df)), val)
    cv, _ = train(tr, tr, fit, val)
    print("CatBoost holdout trained", flush=True)
    ct, model = train(tr, te, np.arange(len(df)), np.arange(len(test)))
    print("CatBoost full refit trained", flush=True)
    model.save_model(str(OUT / "ap_quinta_catboost.cbm"))
    pt = pd.read_csv("line3_today/test_probs.csv")
    assert pt.item.equals(test.id)
    np.savez(OUT / "ap_quinta_catboost_probabilities.npz", val=cv, test=ct)
    anchor = pd.read_csv(OUT / "S4_EVT154_respaldo.csv")
    assert anchor.ID.equals(pt.ID)
    used = anchor.condition.eq("used").to_numpy()
    previous = []
    for name in ("ap_20261111_01_consenso_ruteado", "ap_proxima_disjunta_104"):
        f = pd.read_csv(OUT / (name + ".csv")).set_index("ID")
        assert f.index.is_unique and set(f.index) == set(anchor.ID)
        previous.append(f.condition.reindex(anchor.ID).eq("used").to_numpy() != used)
    for i in range(2):
        data = "".join(p.read_text() for p in sorted(Path("inputs").glob(f"reject{i}.*")))
        f = pd.read_csv(io.StringIO(data)).set_index("ID")
        assert f.index.is_unique and set(f.index) == set(anchor.ID)
        previous.append(f.condition.reindex(anchor.ID).eq("used").to_numpy() != used)
    blocked = np.logical_or.reduce(previous)
    sellers = pairs.seller_id.to_numpy()
    y, base = pairs.y.to_numpy(), pairs.m1.to_numpy()
    tune = pd.util.hash_pandas_object(pairs.seller_id, index=False).to_numpy() % 2 == 0
    options = []
    for w in (.35, .65, 1.):
        pv = blended(cv, base, w)
        pp = blended(ct, pt.m1.to_numpy(), w)
        for margin in (.025, .075, .125, .175):
            mask = ((pp >= .5) != used) & ~blocked & (np.abs(pp - .5) >= margin)
            if mask.sum() < 110:
                continue
            score = measure(y[tune], base[tune], pv[tune], sellers[tune], margin)
            options.append((w, margin, score, mask, pp, pv))
    if not options:
        raise ValueError("No policy has sufficient disjoint support")
    w, margin, tuned, mask, pred, pv = max(
        options, key=lambda x: (x[2]["q05"], x[2]["delta"], -x[2]["logloss_raw"]))
    confirm = measure(y[~tune], base[~tune], pv[~tune], sellers[~tune], margin)
    out = anchor.copy()
    out.loc[mask, "condition"] = np.where(pred[mask] >= .5, "used", "new")
    path = OUT / "ap_quinta_catboost_disjunta.csv"
    out.to_csv(path, index=False, lineterminator="\n")
    saved = pd.read_csv(path)
    assert saved.equals(out) and saved.ID.is_unique and not (mask & blocked).any()
    report = dict(catboost_weight=w, minimum_margin=margin, tune=tuned, confirm_development=confirm,
                  refit_rows=len(df), test_rows=len(saved), flips=int(mask.sum()),
                  new_to_used=int((mask & ~used).sum()), seen_flips=int((mask & pt.seen).sum()),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  public_probability="NO_IDENTIFIABLE",
                  validation_options=[dict(weight=a, margin=b, tune=c, test_flips=int(d.sum()))
                                      for a,b,c,d,_,_ in options])
    (OUT / "ap_quinta_catboost_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "validation_options"}), flush=True)


if __name__ == "__main__":
    main()
