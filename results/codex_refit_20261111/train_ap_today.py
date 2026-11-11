"""Frozen train-only comparison and full-data refit for AP-2026-Q-2."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold


def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_data(archive):
    records = []
    with zipfile.ZipFile(archive) as z:
        for name, n, train in [("train", 70000, True), ("test", 30000, False)]:
            with z.open(name + "_data.jsonlines") as f:
                count = 0
                for line in f:
                    r = json.loads(line)
                    attrs = [a for a in r.get("attributes") or [] if isinstance(a, dict)]
                    title = str(r.get("title") or "")
                    warranty = str(r.get("warranty") or "")
                    shipping = r.get("shipping") or {}
                    state = ((r.get("seller_address") or {}).get("state") or {}).get("id")
                    text = " ".join([title, "WARRANTY", warranty, "TAGS",
                                     *map(str, r.get("tags") or []), "ATTR",
                                     *(str(a.get("id") or "") + ":" + str(a.get("value_name") or "") for a in attrs)])
                    if train:
                        assert r["condition"] in ("new", "used")
                    fields = dict(item=str(r["id"]), seller=str(r["seller_id"]),
                                  category=str(r["category_id"]), y=int(r["condition"] == "used") if train else -1,
                                  time=float(r["start_time"]), title=title, text=text)
                    for k in ("price", "base_price", "original_price", "initial_quantity", "available_quantity", "sold_quantity"):
                        fields[k] = np.log1p(max(float(r.get(k) or 0), 0))
                    for k in ("attributes", "pictures", "variations", "tags", "non_mercado_pago_payment_methods",
                              "descriptions", "deal_ids", "sub_status"):
                        fields[k] = len(r.get(k) or [])
                    for k in ("automatic_relist", "accepts_mercadopago", "official_store_id", "video_id", "parent_item_id"):
                        fields[k] = int(bool(r.get(k)))
                    for k in ("free_shipping", "local_pick_up"):
                        fields[k] = int(bool(shipping.get(k)))
                    fields.update(title_len=len(title), title_words=len(title.split()), warranty_len=len(warranty),
                                  price_gap=fields["price"] - fields["base_price"],
                                  sell_rate=fields["sold_quantity"] - fields["initial_quantity"],
                                  listing=str(r.get("listing_type_id") or ""), buying=str(r.get("buying_mode") or ""),
                                  currency=str(r.get("currency_id") or ""), state=str(state or ""),
                                  ship_mode=str(shipping.get("mode") or ""))
                    records.append(fields)
                    count += 1
                assert count == n
    df = pd.DataFrame(records)
    assert df.item.is_unique
    return df


def matrices(df):
    word = HashingVectorizer(n_features=2**17, ngram_range=(1, 2), alternate_sign=False, dtype=np.float32)
    char = HashingVectorizer(n_features=2**17, ngram_range=(3, 5), analyzer="char_wb",
                             alternate_sign=False, dtype=np.float32)
    return [hstack([word.transform(df[col]), char.transform(df[col])], format="csr") for col in ("title", "text")]


def text_predict(matrix, y, fit, predict):
    positive = np.asarray(matrix[fit[y[fit] == 1]].sum(axis=0)).ravel() + 1
    negative = np.asarray(matrix[fit[y[fit] == 0]].sum(axis=0)).ravel() + 1
    ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
    model = LogisticRegression(solver="liblinear", C=2., max_iter=120, random_state=11)
    model.fit(matrix[fit].multiply(ratio), y[fit])
    return model.predict_proba(matrix[predict].multiply(ratio))[:, 1]


def seller_history(df, fit, requested):
    history = {}
    for seller, group in df.iloc[fit].groupby("seller", sort=False):
        block = group.groupby("time", sort=True).y.agg(["count", "sum"])
        history[seller] = (block.index.to_numpy(), np.r_[0, block["count"].cumsum()], np.r_[0, block["sum"].cumsum()])
    count = np.zeros(len(requested))
    total = count.copy()
    for j, (seller, stamp) in enumerate(df.iloc[requested][["seller", "time"]].itertuples(index=False, name=None)):
        if seller in history:
            stamps, counts, totals = history[seller]
            k = np.searchsorted(stamps, stamp, side="left")
            count[j], total[j] = counts[k], totals[k]
    return np.column_stack([np.log1p(count), (total + 3) / (count + 6)]), count > 0


def fit_predict(df, mats, fit, predict, tag):
    y = df.y.to_numpy()
    idx = np.r_[fit, predict]
    fit_n = len(fit)
    inner = list(GroupKFold(3).split(fit, y[fit], df.seller.iloc[fit]))
    text = []
    for m, matrix in enumerate(mats):
        p = np.full(len(idx), np.nan)
        for j, (a, b) in enumerate(inner):
            p[b] = text_predict(matrix, y, fit[a], fit[b])
            print(tag, "text", m, j, flush=True)
        p[fit_n:] = text_predict(matrix, y, fit, predict)
        assert np.isfinite(p).all()
        text.append(p)

    cats = ["category", "listing", "buying", "currency", "state", "ship_mode"]
    raw = df.drop(columns=["item", "seller", "title", "text", "y", *cats]).to_numpy()[idx]
    cat_codes = []
    for key in cats:
        vocab = {v: j + 1 for j, v in enumerate(df.iloc[fit][key].value_counts().head(200).index)}
        cat_codes.append(df[key].iloc[idx].map(vocab).fillna(0).to_numpy())
    relative = df.iloc[idx].price - df.iloc[idx].category.map(df.iloc[fit].groupby("category").price.median())
    h, seen = seller_history(df, fit, idx)
    features = np.column_stack([raw, relative, h, text[0], text[1], *cat_codes])
    cat_mask = np.arange(features.shape[1]) >= features.shape[1] - len(cats)
    out = {}
    for name, use_enriched in (("baseline", False), ("enriched", True)):
        local = features.copy()
        if not use_enriched:
            local[:, -len(cats) - 1] = 0
        model = HistGradientBoostingClassifier(max_iter=330, learning_rate=.075, max_leaf_nodes=39,
                min_samples_leaf=24, l2_regularization=6, categorical_features=cat_mask,
                early_stopping=False, random_state=11)
        model.fit(local[:fit_n], y[fit])
        out[name] = model.predict_proba(local[fit_n:])[:, 1]
        print(tag, "model", name, flush=True)
    return out, seen[fit_n:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    t0 = time.monotonic()
    df = load_data(args.archive)
    mats = matrices(df)
    print("inputs validated and encoded", round(time.monotonic() - t0), flush=True)
    train = np.arange(70000)
    test = np.arange(70000, 100000)
    rng = np.random.default_rng(20261111)
    groups = np.unique(df.seller.iloc[train])
    rng.shuffle(groups)
    cold = np.isin(df.seller.iloc[train], groups[:len(groups) // 5])
    eligible = df.iloc[train[~cold]]
    last = eligible.groupby("seller").time.transform("max")
    first = eligible.groupby("seller").time.transform("min")
    # Hold out all latest-time ties, so training history is strictly prior.
    warm = eligible.index[(last == eligible.time) & (last > first)].to_numpy()
    val = np.sort(np.r_[train[cold], warm])
    fit = np.setdiff1d(train, val)
    assert not set(df.seller.iloc[train[cold]]) & set(df.seller.iloc[fit])
    assert len(np.unique(val)) == len(val)
    p, seen = fit_predict(df, mats, fit, val, "development")
    pairs = df.iloc[val][["item", "seller", "category", "y"]].reset_index(drop=True)
    pairs = pairs.assign(seen=seen, **p)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    for mode, sub in pairs.groupby("seen"):
        a = (sub.baseline >= .5) == sub.y
        b = (sub.enriched >= .5) == sub.y
        print("evaluation", mode, len(sub), a.sum(), b.sum(), log_loss(sub.y, sub.baseline),
              log_loss(sub.y, sub.enriched), flush=True)
    p, seen = fit_predict(df, mats, train, test, "refit")
    pd.DataFrame({"ID": np.arange(1, 30001), "item": df.item.iloc[test].to_numpy(),
                  "seen": seen, **p}).to_csv(args.out / "test_probs.csv", index=False)
    report = {"archive_sha256": hash_file(args.archive), "pairs_sha256": hash_file(args.out / "pairs.csv"),
              "test_probabilities_sha256": hash_file(args.out / "test_probs.csv"),
              "code_sha256": hash_file(Path(__file__)), "n_fit_validation": len(fit),
              "seconds": time.monotonic() - t0}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
