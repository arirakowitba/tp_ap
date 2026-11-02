import base64
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

HERE = Path(__file__).parent
REPO_DATA = HERE.parent / "results" / "intensive_20261030"
ROOT = REPO_DATA if REPO_DATA.exists() else HERE
NAMES = [
    "paired_external_virgin_confirm_20261101.csv.gz",
    "paired_external_nested_outer_777001.csv.gz",
    "paired_external_nested_outer_777002.csv.gz",
]
TEST_SEEN = 0.71623


def load(name):
    if (ROOT / name).exists():
        raw = (ROOT / name).read_bytes()
    else:
        raw = base64.b64decode((ROOT / (name + ".b64")).read_bytes())
        (ROOT / name).write_bytes(raw)
    return pd.read_csv(ROOT / name), hashlib.sha256(raw).hexdigest()


def evaluate(df, base, cand, seed=40):
    y = df.y_true.to_numpy()
    delta = (df[cand].to_numpy() == y).astype(int) - (df[base].to_numpy() == y).astype(int)
    by_seen = {int(k): delta[df.is_seen.to_numpy() == k] for k in (0, 1)}
    mean_seen = by_seen[1].mean()
    mean_new = by_seen[0].mean()
    transported = 30000 * (TEST_SEEN * mean_seen + (1 - TEST_SEEN) * mean_new)
    # Cluster bootstrap: resample sellers, preserving all of each seller's rows.
    g = pd.DataFrame({"seller": df.seller_id, "seen": df.is_seen, "delta": delta}).groupby(["seen", "seller"]).delta.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(2000):
        rates = {}
        for segment in (0, 1):
            t = g.loc[segment].to_numpy()
            sample = t[rng.integers(0, len(t), len(t))]
            rates[segment] = sample[:, 0].sum() / sample[:, 1].sum()
        vals.append(30000 * (TEST_SEEN * rates[1] + (1 - TEST_SEEN) * rates[0]))
    return {
        "base": base, "cand": cand, "n": len(df), "seen_pct": df.is_seen.mean(),
        "net": int(delta.sum()), "seen_net": int(by_seen[1].sum()), "new_net": int(by_seen[0].sum()),
        "transported_30k": transported, "transported_ci95": np.quantile(vals, [.025, .975]).tolist(),
    }


def main():
    cuts = []
    for name in NAMES:
        df, sha = load(name)
        cuts.append(df)
        print(name, "sha256", sha, "rows", len(df), "sellers", df.seller_id.nunique())
        for base, cand in [
            ("pred_cb_single_050", "pred_blend_bag3_050"),
            ("pred_cb_single_tuned", "pred_blend_bag3_tuned"),
        ]:
            print(evaluate(df, base, cand))
        print("logloss CB", log_loss(df.y_true, df.p_cb_single),
              "blend bag3", log_loss(df.y_true, df.p_blend_bag3))
    for i in range(3):
        for j in range(i + 1, 3):
            x = set(cuts[i].item_id)
            y = set(cuts[j].item_id)
            print("overlap", i, j, "rows", len(x & y), "sellers", len(set(cuts[i].seller_id) & set(cuts[j].seller_id)))


if __name__ == "__main__":
    main()
