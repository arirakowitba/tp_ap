"""Train-only diagnostic of seller-history support and text disagreement.

The test labels and public leaderboard are never read. Each split has held-out
seller groups (new) and held-out rows from fit sellers (seen). Tune and confirm
are disjoint, and text vectorizers are fit only on fit rows.
"""
import csv
import json
import zipfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


def load():
    with zipfile.ZipFile("data.zip") as z, z.open("train_data.jsonlines") as f:
        rows = [json.loads(line) for line in f]
    y = np.array([r["condition"] == "new" for r in rows], dtype=int)
    seller = np.array([str(r["seller_id"]) for r in rows])
    title = [r.get("title") or "" for r in rows]
    return rows, y, seller, title


def split(seller, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(seller)
    rng.shuffle(unique)
    new_tune = set(unique[:len(unique)//10])
    new_confirm = set(unique[len(unique)//10:len(unique)//5])
    seen = ~(np.isin(seller, list(new_tune | new_confirm)))
    ids = np.where(seen)[0]
    rng.shuffle(ids)
    seen_tune, seen_confirm = ids[:len(ids)//10], ids[len(ids)//10:len(ids)//5]
    tune = np.flatnonzero(np.isin(seller, list(new_tune)))
    confirm = np.flatnonzero(np.isin(seller, list(new_confirm)))
    tune = np.r_[tune, seen_tune]
    confirm = np.r_[confirm, seen_confirm]
    fit = np.setdiff1d(np.arange(len(seller)), np.r_[tune, confirm])
    assert not set(seller[tune]).intersection(new_confirm)
    assert not set(seller[confirm]).intersection(new_tune)
    return fit, tune, confirm


def prior(seller, y, fit, idx, alpha):
    counts = defaultdict(lambda: [0, 0])
    for s, t in zip(seller[fit], y[fit]):
        counts[s][0] += int(t)
        counts[s][1] += 1
    p0 = float(np.mean(y[fit]))
    pos = np.array([counts[s][0] for s in seller[idx]])
    n = np.array([counts[s][1] for s in seller[idx]])
    return (pos + alpha*p0)/(n+alpha), n


def report(y, seller, ix, text_p, mixed_p, n, group, rule):
    old = (text_p >= .5)
    new = (mixed_p >= .5)
    d = old != new
    fix = int(np.sum(d & (new == y[ix])))
    brk = int(np.sum(d & (old == y[ix])))
    return {
        "group": group, "rule": rule, "n": len(ix),
        "text_accuracy": round(accuracy_score(y[ix], old), 6),
        "accuracy": round(accuracy_score(y[ix], new), 6),
        "fix": fix, "break": brk,
        "text_logloss": round(log_loss(y[ix], text_p), 6),
        "logloss": round(log_loss(y[ix], mixed_p), 6),
        "seen": int(np.sum(n > 0)),
    }


def main():
    rows, y, seller, title = load()
    all_result = []
    for seed in (2026, 314159):
        fit, tune, confirm = split(seller, seed)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=90000,
                              sublinear_tf=True, strip_accents="unicode")
        xfit = vec.fit_transform([title[i] for i in fit])
        model = LogisticRegression(C=4, max_iter=120, solver="liblinear")
        model.fit(xfit, y[fit])
        result = {"seed": seed, "fit_size": len(fit), "tune_size": len(tune),
                  "confirm_size": len(confirm), "vector_features": xfit.shape[1]}
        for name, ix in (("tune", tune), ("confirm", confirm)):
            xt = vec.transform([title[i] for i in ix])
            ptext = model.predict_proba(xt)[:, 1]
            pp, n = prior(seller, y, fit, ix, 3.)
            # Predeclared evidence rule: strong seller history disagrees with
            # text and seller's smoothed prior is decisive. No label tuning.
            gate = (n >= 3) & (np.abs(pp-.5) >= .30) & ((pp >= .5) != (ptext >= .5))
            pmix = ptext.copy()
            pmix[gate] = .6*pp[gate] + .4*ptext[gate]
            for seg, mask in (("all", np.ones(len(ix), bool)),
                              ("seen", n > 0), ("new", n == 0),
                              ("gated", gate)):
                if not np.any(mask):
                    continue
                result.setdefault(name, []).append(report(y, seller, ix[mask],
                      ptext[mask], pmix[mask], n[mask], seg, "seller_gate"))
            if name == "confirm":
                with open(f"seller_diagnostic_pairs_{seed}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["id", "seller_id", "label_train", "text_p",
                                "seller_prior", "seller_n_fit", "gate", "mixed_p"])
                    for j, row, p, q, k, g, m in zip(ix, [rows[i] for i in ix],
                                                       ptext, pp, n, gate, pmix):
                        w.writerow([row["id"], seller[j], y[j], p, q, k, int(g), m])
        all_result.append(result)
    with open("seller_diagnostic_results.json", "w") as f:
        json.dump(all_result, f, indent=2)
    print(json.dumps(all_result, indent=2))


if __name__ == "__main__":
    main()
