"""Bag the validated full70k native-text + nested-NB CatBoost recipe.

Reuses only the fold-safe nested text probabilities produced by
refit_full_native_nb.py.  It never reads test labels and never uploads.
"""
import argparse, gc, hashlib, json, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from deep_reference import extract_deep_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--seeds", default="20261106,20261108")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with zipfile.ZipFile(args.zip) as z:
        train = [json.loads(x) for x in z.open("train_data.jsonlines")]
        test = [json.loads(x) for x in z.open("test_data.jsonlines")]
    assert len(train) == 70000 and len(test) == 30000
    rows = train + test
    ntr = len(train)
    fit = np.arange(ntr)
    tst = np.arange(ntr, len(rows))
    y = np.asarray([int(r["condition"] == "used") for r in train])
    test_ids = [str(r["id"]) for r in test]

    af, cat, _, _, _ = extract_deep_features(rows)
    df = af[[c for c in af if not c.startswith(("freq_", "seller_", "sell_mean_", "rel_price_cat"))]].copy()
    for col in ["seller_id", "category_id", "city"]:
        counts = cat.iloc[fit][col].value_counts()
        df["fit_freq_" + col] = cat[col].map(counts).fillna(0).astype(float)
    med = df.iloc[fit].groupby(cat.iloc[fit].category_id).log_price.median()
    df["fit_rel_price"] = df.log_price - cat.category_id.map(med).fillna(float(df.iloc[fit].log_price.median()))

    text = np.asarray([
        str(r.get("title") or "") + " WARRANTY " + str(r.get("warranty") or "") + " ATTR " +
        " ".join(str(a.get("value_name") or a.get("name") or "") for a in r.get("attributes") or [] if isinstance(a, dict))
        for r in rows
    ], dtype=object)
    cats = ["category_id", "listing_type_id", "buying_mode", "currency_id", "status", "seller_id", "state", "ship_mode"]
    for key in cats[:6]:
        df[key] = [str(r.get(key) or "MISSING") for r in rows]
    df["state"] = [str(((r.get("seller_address") or {}).get("state") or {}).get("id") or "MISSING") for r in rows]
    df["ship_mode"] = [str((r.get("shipping") or {}).get("mode") or "MISSING") for r in rows]
    del rows, train, test, af
    gc.collect()

    ptext = np.load(Path(args.base) / "nested_text_probabilities.npy")
    assert ptext.shape == (100000,) and np.isfinite(ptext).all()
    all_nested = [np.load(Path(args.base) / "nested_nb_test_probs.npy")]
    all_native = [np.load(Path(args.base) / "native_text_test_probs.npy")]
    run_meta = []

    for seed in seeds:
        for name, data, text_features in [
            ("nested_nb", df.assign(nested_nb_text=ptext), []),
            ("native_text", df.assign(native_text=text), ["native_text"]),
        ]:
            start = time.time()
            model = CatBoostClassifier(
                iterations=900, depth=6, learning_rate=.06, l2_leaf_reg=6,
                random_seed=seed, thread_count=6, verbose=False,
                loss_function="Logloss", allow_writing_files=False,
            )
            model.fit(data.iloc[fit], y, cat_features=cats, text_features=text_features)
            pred = model.predict_proba(data.iloc[tst])[:, 1]
            np.save(out / f"{name}_seed{seed}.npy", pred)
            model.save_model(str(out / f"{name}_seed{seed}.cbm"))
            (all_nested if name == "nested_nb" else all_native).append(pred)
            run_meta.append({"seed": seed, "model": name, "seconds": time.time() - start})
            print("FIT_DONE", seed, name, run_meta[-1]["seconds"], flush=True)

    nested = np.mean(all_nested, axis=0)
    native = np.mean(all_native, axis=0)
    blend = .5 * nested + .5 * native
    np.save(out / "nested_bag3.npy", nested)
    np.save(out / "native_bag3.npy", native)
    np.save(out / "blend_bag3.npy", blend)
    pd.DataFrame({
        "ID": np.arange(1, 30001), "item_id": test_ids,
        "p_nested_bag3": nested, "p_native_bag3": native, "p_blend_bag3": blend,
    }).to_csv(out / "test_probabilities_bag3.csv", index=False)
    pd.DataFrame({
        "ID": np.arange(1, 30001),
        "condition": np.where(blend >= .5, "used", "new"),
    }).to_csv(out / "candidate_blend_bag3_threshold050.csv", index=False)
    manifest = {
        "recipe": "mean over seeds 20261104/06/08 of .5 native-text + .5 nested-NB CatBoost",
        "threshold": .5,
        "seeds": [20261104] + seeds,
        "source_zip_sha256": sha(args.zip),
        "probabilities_sha256": sha(out / "test_probabilities_bag3.csv"),
        "candidate_sha256": sha(out / "candidate_blend_bag3_threshold050.csv"),
        "test_labels_used": False,
        "runs": run_meta,
        "seconds": time.time() - t0,
        "warning": "Unsubmitted; requires bridge/backtest before promotion.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
