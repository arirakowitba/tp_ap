"""Train-only title-neighbor experiment and one manual submission candidate."""
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


OUT = Path("entrega_ap")
POLICIES = [(s, p, n, a) for s in (.62, .76, .88) for p in (.8, .93)
            for n in (1, 2) for a in (False, True)]


def normal(title):
    value = unicodedata.normalize("NFKD", str(title or "").casefold())
    return " ".join(re.findall(r"[a-z0-9]+", value))


def read_data(path):
    with zipfile.ZipFile(path) as z:
        def load(name, labeled):
            with z.open(name + "_data.jsonlines") as f:
                rows = []
                for line in f:
                    r = json.loads(line)
                    y = int(r["condition"] == "used") if labeled else -1
                    rows.append((str(r["id"]), str(r["seller_id"]), str(r["category_id"]),
                                 normal(r.get("title")), int(r.get("start_time") or 0), y))
            return pd.DataFrame(rows, columns=["item", "seller", "cat", "title", "stamp", "y"])
        return load("train", True), load("test", False)


def retrieve(train, query):
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                            dtype=np.float32, sublinear_tf=True)
    xt = tfidf.fit_transform(train.title)
    xq = tfidf.transform(query.title)
    seller = train.seller.to_numpy()
    labels = train.y.to_numpy()
    stamp = train.stamp.to_numpy()
    query_seller = query.seller.to_numpy()
    query_stamp = query.stamp.to_numpy()
    groups = train.groupby("cat", sort=False).indices
    stats = np.zeros((len(query), 3))
    for cat, queries in query.groupby("cat", sort=False).indices.items():
        fit = groups.get(cat)
        if fit is None:
            continue
        db = xt[fit].T.tocsc()
        for begin in range(0, len(queries), 96):
            q = queries[begin:begin + 96]
            scores = (xq[q] @ db).tocsr()
            for j, row in enumerate(q):
                sl = slice(scores.indptr[j], scores.indptr[j + 1])
                sim = scores.data[sl]
                ids = fit[scores.indices[sl]]
                keep = (sim >= .55) & ((seller[ids] != query_seller[row]) |
                                        (stamp[ids] < query_stamp[row]))
                sim, ids = sim[keep], ids[keep]
                by_seller = set()
                weights = []
                ys = []
                for k in np.argsort(-sim, kind="stable"):
                    key = seller[ids[k]]
                    if key in by_seller:
                        continue
                    by_seller.add(key)
                    weights.append(float(sim[k]))
                    ys.append(labels[ids[k]])
                    if len(weights) == 8:
                        break
                if weights:
                    w = np.array(weights)
                    near = w >= w[0] - .12
                    stats[row] = [w[0], np.average(np.array(ys)[near], weights=w[near] ** 8), near.sum()]
    return stats


def select(stats, p_model, policy):
    sim, purity, n = stats.T
    min_sim, min_purity, min_n, agree = policy
    cls = purity >= .5
    ok = (sim >= min_sim) & (np.maximum(purity, 1 - purity) >= min_purity) & (n >= min_n)
    if agree:
        ok &= cls == (p_model >= .5)
    return ok, cls


def evaluate(stats, model, y, group, policy):
    eligible, cls = select(stats, model, policy)
    flip = eligible & (cls != (model >= .5))
    d = np.where(flip, np.where(cls == y, 1, -1), 0)
    per_seller = pd.Series(d).groupby(group).sum().to_numpy()
    rng = np.random.default_rng(11011)
    draws = [per_seller[rng.integers(0, len(per_seller), len(per_seller))].sum()
             for _ in range(600)]
    return {"selected": int(eligible.sum()), "selected_accuracy": float((cls[eligible] == y[eligible]).mean())
            if eligible.any() else None, "flips": int(flip.sum()), "delta": int(d.sum()),
            "fixes": int((d > 0).sum()), "breaks": int((d < 0).sum()), "q05": float(np.quantile(draws, .05))}


def main():
    import sys
    tr, te = read_data(sys.argv[1])
    assert len(tr) == 70000 and len(te) == 30000
    assert tr.item.is_unique and te.item.is_unique
    pairs = pd.read_csv("line3_today/pairs.csv")
    val = tr.iloc[pairs.row].reset_index(drop=True)
    assert val.item.equals(pairs.id) and val.y.equals(pairs.y)
    fit = tr.drop(index=pairs.row).reset_index(drop=True)
    dv = retrieve(fit, val)
    print("validation retrieval done", flush=True)
    base, y = pairs.m1.to_numpy(), val.y.to_numpy()
    tune = pd.util.hash_pandas_object(val.seller, index=False).to_numpy() % 2 == 0
    sels = val.seller.to_numpy()
    dt = retrieve(tr, te)
    ref = pd.read_csv("line3_today/test_probs.csv")
    assert ref.item.equals(te.item)
    a = pd.read_csv(OUT / "S4_EVT154_respaldo.csv")
    assert a.ID.equals(ref.ID)
    anchor = a.condition.eq("used").to_numpy()
    excluded = np.zeros(len(a), dtype=bool)
    for name in ("ap_20261111_01_consenso_ruteado", "ap_proxima_disjunta_104"):
        f = pd.read_csv(OUT / (name + ".csv"))
        assert f.ID.equals(a.ID)
        excluded |= f.condition.ne(a.condition).to_numpy()
    for i in range(2):
        data = "".join(p.read_text() for p in sorted(Path("inputs").glob(f"reject{i}.*")))
        f = pd.read_csv(io.StringIO(data))
        assert f.ID.equals(a.ID)
        excluded |= f.condition.ne(a.condition).to_numpy()
    results = []
    for policy in POLICIES:
        if policy[-1]:
            continue
        ok, cls = select(dt, ref.m1.to_numpy(), policy)
        k = int((ok & (cls != anchor) & ~excluded).sum())
        if k < 90:
            continue
        r = evaluate(dv[tune], base[tune], y[tune], sels[tune], policy)
        results.append((policy, r, k))
    if not results:
        raise ValueError("No retrieval rule has enough non-overlapping changes")
    frozen, tuned, _ = max(results, key=lambda x: (x[1]["q05"], x[1]["delta"], -x[1]["flips"]))
    confirm = evaluate(dv[~tune], base[~tune], y[~tune], sels[~tune], frozen)
    print("frozen", frozen, "tune", tuned, "confirm", confirm, flush=True)
    ok, cls = select(dt, ref.m1.to_numpy(), frozen)
    mask = ok & (cls != anchor) & ~excluded
    a.loc[mask, "condition"] = np.where(cls[mask], "used", "new")
    path = OUT / "ap_quinta_vecinos_train.csv"
    a.to_csv(path, index=False, lineterminator="\n")
    check = pd.read_csv(path)
    assert check.equals(a) and check.ID.is_unique and check.condition.isin(["new", "used"]).all()
    report = {"policy": frozen, "tune": tuned, "confirm_development": confirm,
              "feasible_tune_grid": results,
              "test_flips": int(mask.sum()), "new_to_used": int((mask & cls).sum()),
              "seen_flips": int((mask & ref.seen.to_numpy()).sum()),
              "line3_agrees_with_flip": int((mask & ((ref.m1 >= .5).to_numpy() == cls)).sum()),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "public_probability": "NO_IDENTIFIABLE"}
    (OUT / "ap_quinta_vecinos_train_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
