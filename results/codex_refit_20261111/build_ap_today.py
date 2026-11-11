"""Build reproducible AP candidates and audit paired validation and score feasibility."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from train_ap_today import hash_file

N_PUBLIC = 20998
ANCHOR_HITS = 19899
NEEDED = 50
WEIGHTS = (.15, .25, .35, .50, .65, 1.)


def read_chunks(stem):
    parts = sorted(Path("inputs").glob(stem + ".[0-9][0-9][0-9]"))
    assert parts, stem
    data = "".join(p.read_text() for p in parts)
    return pd.read_csv(io.StringIO(data))


def logits(p):
    return logit(np.clip(p, 1e-5, 1 - 1e-5))


def target_hash(frame):
    payload = "\n".join(f"{i}:{label}" for i, label in frame.itertuples(index=False, name=None))
    return hashlib.sha256(payload.encode()).hexdigest()


def mix(base, challenger, weight):
    return expit((1 - weight) * logits(base) + weight * logits(challenger))


def paired(y, base, candidate, seller, category, seed=1126):
    y, base, candidate = map(np.asarray, (y, base, candidate))
    good = (candidate >= .5) == y
    b = (base >= .5) == y
    d = good.astype(int) - b.astype(int)
    rng = np.random.default_rng(seed)
    seller = pd.factorize(seller)[0]
    category = pd.factorize(category)[0]
    keep = d != 0
    ids, cats, ds = seller[keep], category[keep], d[keep]
    boot = []
    for _ in range(1500):
        seller_w = rng.poisson(1, seller.max() + 1)
        cat_w = rng.poisson(1, category.max() + 1)
        boot.append(np.dot(ds, seller_w[ids] * cat_w[cats]))
    return dict(n=len(y), hits=int(good.sum()), baseline_hits=int(b.sum()),
                delta=int(d.sum()), fixes=int((d > 0).sum()), breaks=int((d < 0).sum()),
                q05=float(np.quantile(boot, .05)), q95=float(np.quantile(boot, .95)),
                logloss=log_loss(y, candidate), baseline_logloss=log_loss(y, base))


def eval_arms(f, y, base, challenger, seller, category):
    by_weight = {}
    for name, part in f.groupby("seen"):
        by_weight[str(name)] = {}
        for w in WEIGHTS:
            by_weight[str(name)][str(w)] = paired(part[y], part[base], mix(part[base], part[challenger], w),
                                                 part[seller], part[category])
    return by_weight


def sample_tail(k, assumed_precision):
    # Conditional on exactly this many beneficial flips across 30,000 rows,
    # plus uniform public sampling. Count beneficial and harmful jointly.
    fixes = round(k * assumed_precision)
    n = np.arange(max(0, N_PUBLIC - (30000 - k)), min(k, N_PUBLIC) + 1)
    draw_p = hypergeom.pmf(n, 30000, k, N_PUBLIC)
    minimum_fixes = (n + NEEDED + 1) // 2
    return float(np.dot(draw_p, hypergeom.sf(minimum_fixes - 1, k, fixes, n)))


def record_candidate(name, mask, a, out, seen):
    df = a.copy()
    df.loc[mask, "condition"] = df.loc[mask, "condition"].map({"used": "new", "new": "used"})
    assert df.ID.equals(a.ID) and df.condition.isin(["new", "used"]).all()
    p = out / f"{name}.csv"
    df.to_csv(p, index=False, lineterminator="\n")
    saved = pd.read_csv(p)
    assert saved.equals(df)
    k = int(mask.sum())
    return dict(file=p.name, sha256=hash_file(p), id_target_sha256=target_hash(df),
                flips=k, seen_flips=int((mask & seen).sum()),
                new_to_used=int((mask & (a.condition == "new")).sum()),
                max_possible_gain=k,
                needed_precision_at_mean_public_support=(1 + NEEDED / (k * N_PUBLIC / 30000)) / 2,
                conditional_tail={str(x): sample_tail(k, x) for x in [.50, .53, .55, .57, .60, .65]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refit", type=Path)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    anchor = read_chunks("s4")
    assert anchor.ID.tolist() == list(range(1, 30001)) and anchor.condition.isin(["new", "used"]).all()
    anchor_content = b"".join(p.read_bytes() for p in sorted(Path("inputs").glob("s4.[0-9][0-9][0-9]")))
    (args.out / "S4_EVT154_respaldo.csv").write_bytes(anchor_content)
    anchor_hash = hash_file(args.out / "S4_EVT154_respaldo.csv")
    assert hashlib.sha256(anchor_content.replace(b"\n", b"\r\n")).hexdigest() == (
        "bb73f1ff1b5fdc76fd8880d4cc146adf46a0558138aec4546e17508cfcf5183f")
    assert target_hash(anchor) == "c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd"
    f3 = read_chunks("l3test")
    f4 = read_chunks("l4test")
    pod = read_chunks("podium")
    local = pd.read_csv("model_today/test_probs.csv")
    for f in [f3, f4, pod, local]:
        assert f.ID.equals(anchor.ID)
    assert (f4.row.to_numpy() + 1 == anchor.ID).all()
    seen = f3.fit_seller_fold_frac > 0
    a_used = anchor.condition == "used"
    pod_used = pod.p_optimal_blend >= .5
    ag = np.where(seen, f3.p_m1_seen_causal_test, f4.p_l4_new_seller_blend)
    local_challenger = local.enriched.to_numpy()
    all_p = {"ag_routed": ag, "local_refit": local_challenger,
             "ag_local_mix": mix(ag, local_challenger, .30)}

    s3 = read_chunks("l3seen")
    s4 = read_chunks("l4seen")
    assert len(s3) == len(s4) == 32092
    assert not s3.duplicated(["outer_fold", "row"]).any()
    assert not s4.duplicated(["outer_fold", "row"]).any()
    sg = s3.merge(s4[["outer_fold", "row", "p_l4_new_seller_blend"]], on=["outer_fold", "row"], validate="one_to_one")
    p_cols = ["p_m0_seen_val", "p_m1_seen_val", "p_l4_new_seller_blend"]
    assert sg.groupby("row").y_true.nunique().max() == 1
    ss = sg.groupby(["row", "seller_id", "category_id", "y_true"], as_index=False)[p_cols].mean()
    unseen = read_chunks("l3oof")
    assert len(unseen) == 70000 and unseen.row.is_unique and unseen.id.is_unique
    n_seen = len(ss)
    assert n_seen == 8023 and s3.groupby("row").size().eq(4).all()
    proxy = pd.concat([
        ss.rename(columns={"p_m0_seen_val": "base", "p_m1_seen_val": "challenger"}).assign(seen=True),
        unseen.rename(columns={"p_m0_unseen": "base", "p_m1_unseen": "challenger"}).assign(seen=False)])
    report = dict(anchor_sha256=anchor_hash, public_size_assumed=N_PUBLIC, additional_hits_for_exact_95000=NEEDED,
                  anchor_vs_podium_argmax_flips=int((pod_used != a_used).sum()), ag_seen_unique=n_seen,
                  ag_seen_replay_rows=len(s3), ag_test_seen=int(seen.sum()),
                  ag_proxy=eval_arms(proxy, "y_true", "base", "challenger", "seller_id", "category_id"),
                  local_proxy=eval_arms(pd.read_csv("model_today/pairs.csv"), "y", "baseline", "enriched", "seller", "category"))
    if args.refit and (args.refit / "test_probs.csv").exists():
        ref = pd.read_csv(args.refit / "test_probs.csv")
        assert ref.ID.equals(anchor.ID) and ref.item.equals(local.item)
        assert ref.seen.equals(seen)
        strict = np.where(seen, ref.m1_seen_causal, f4.p_l4_new_seller_blend)
        all_p.update(strict_refit_routed=strict, strict_full=ref.m1.to_numpy(),
                     strict_ag_mix=mix(strict, ag, .40), strict_local_mix=mix(ref.m1, local.enriched, .25))
        ref_pairs = pd.read_csv(args.refit / "pairs.csv")
        assert ref_pairs.id.is_unique
        report["strict_refit_proxy"] = eval_arms(ref_pairs, "y", "m0", "m1", "seller_id", "category_id")
        loc_pairs = pd.read_csv("model_today/pairs.csv").rename(columns={"item": "id"})
        both = ref_pairs.merge(loc_pairs[["id", "y", "baseline", "enriched"]], on=["id", "y"], validate="one_to_one")
        assert len(both) == len(ref_pairs) == len(loc_pairs)
        report["refit_paired_vs_local"] = {}
        for key, sub in both.groupby("seen"):
            report["refit_paired_vs_local"][str(key)] = {}
            for w in (0., .25, .5):
                p = mix(sub.m1, sub.enriched, w)
                report["refit_paired_vs_local"][str(key)][str(w)] = paired(
                    sub.y, sub.enriched, p, sub.seller_id, sub.category_id)
    rows = {}
    for name, p in all_p.items():
        for margin in (0., .05, .10, .15, .20):
            change = ((p >= .5) != a_used) & (np.abs(p - .5) >= margin)
            rows[f"{name}_{margin:.2f}"] = dict(flips=int(change.sum()), n2u=int((change & ~a_used).sum()),
                                              seen=int((change & seen).sum()))
    report["grid_test_counts"] = rows

    if args.refit and (args.refit / "test_probs.csv").exists():
        policies = [
            ("ap_20261111_01_consenso_ruteado", "strict_ag_mix", .10),
            ("ap_20261111_02_refit_ruteado", "strict_full", 0.),
            ("ap_20261111_03_especialistas_ag", "ag_routed", 0.)]
    else:
        policies = [("ap_20261111_01_consenso_ruteado", "ag_local_mix", .10),
                    ("ap_20261111_02_especialistas_ag", "ag_routed", 0.)]
    report["candidates"] = {}
    masks = {}
    for name, model, margin in policies:
        p = all_p[model]
        mask = ((p >= .5) != a_used) & (np.abs(p - .5) >= margin)
        assert mask.sum() >= NEEDED
        masks[name] = mask
        report["candidates"][name] = record_candidate(name, mask, anchor, args.out, seen) | {
            "model": model, "minimum_model_margin": margin}
    report["candidate_overlap"] = {a + " vs " + b: dict(
        intersection=int((x & yy).sum()), symmetric_difference=int((x != yy).sum()))
        for i, (a, x) in enumerate(masks.items()) for b, yy in list(masks.items())[i + 1:]}
    historical = {}
    for p in sorted(Path("inputs").glob("history_*.json")):
        if p.exists():
            for r in json.loads(p.read_text()):
                if r.get("valid"):
                    historical[r["id"]] = r
    report["historical_vector_count"] = len(historical)
    report["historical_s1_s5"] = [v for key, v in historical.items() if key.startswith("S")]
    bins = {}
    for lo, hi in [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 30000)]:
        cohort = [r["score"] for r in historical.values() if lo <= r["flips"] < hi and r["score"] is not None]
        if cohort:
            bins[f"{lo}-{hi}"] = dict(n=len(cohort), median=float(np.median(cohort)), max=max(cohort))
    report["historical_vs_s4_by_flip_distance"] = bins
    (args.out / "auditoria_ap_20261111.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("anchor_vs_podium_argmax_flips", "ag_seen_unique", "ag_test_seen",
                    "grid_test_counts", "candidates", "candidate_overlap", "historical_vector_count",
                    "historical_vs_s4_by_flip_distance")}, indent=2))


if __name__ == "__main__":
    main()
