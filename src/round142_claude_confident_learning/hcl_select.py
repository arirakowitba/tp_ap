"""H-CL selection and confirm evaluation.
Step 1 (tune only): pick ONE config by summed tune accuracy over the 3 cuts (tie-break: summed tune logloss),
        write selection_lock.json with a hash of the tune-only inputs BEFORE any confirm metric is computed.
Step 2: open confirm once for the locked config vs control (same seed), paired, seen/new, logloss, seller bootstrap.
usage: python hcl_select.py OUTDIR AG_DIR"""
import sys, json, hashlib, time, numpy as np, pandas as pd
from pathlib import Path
out, ag = Path(sys.argv[1]), Path(sys.argv[2]); seeds = [2026, 314159, 20261030]
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
D = {s: np.load(out / f"seed{s}" / "probs.npz") for s in seeds}
Y = {}
for s in seeds:
    tu = pd.read_csv(ag / f"preds_A_hist_seed{s}_tune.csv.gz"); co = pd.read_csv(ag / f"preds_A_hist_seed{s}_confirm.csv.gz")
    assert (tu.row_idx.values == D[s]["tune_rows"]).all() and (co.row_idx.values == D[s]["confirm_rows"]).all()
    Y[s] = dict(tune=tu.y_used.values, conf=co.y_used.values, seen=co.is_seen.values == 1, seller=co.seller_id.astype(str).values)
configs = list(D[seeds[0]]["configs"])
# ---- step 1: tune only
tune_tab = {}
for i, c in enumerate(configs):
    accs = [float(((D[s]["tune"][i] >= .5) == Y[s]["tune"]).mean()) for s in seeds]
    lls = [ll(Y[s]["tune"], D[s]["tune"][i]) for s in seeds]
    tune_tab[c] = dict(acc_by_cut=accs, acc_sum=sum(accs), ll_sum=sum(lls))
ctrl = configs.index("control")
beats = {c: sum(tune_tab[c]["acc_by_cut"][k] > tune_tab["control"]["acc_by_cut"][k] for k in range(3)) for c in configs}
best = max(configs, key=lambda c: (round(tune_tab[c]["acc_sum"], 10), -tune_tab[c]["ll_sum"]))
lock = dict(selected=best, rule="max sum tune acc over 3 cuts, tie-break min sum tune logloss; extend to CatBoost only if selected beats control on tune in >=2/3 cuts",
            beats_control_on_tune_cuts=beats[best], tune_table=tune_tab, locked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tune_inputs_sha256=hashlib.sha256(b"".join(D[s]["tune"].tobytes() for s in seeds)).hexdigest())
(out / "selection_lock.json").write_text(json.dumps(lock, indent=2)); print("LOCKED", best, "beats control on tune in", beats[best], "cuts", flush=True)
# ---- step 2: confirm, once
j = configs.index(best); res = {}
for s in seeds:
    y, seen, sel = Y[s]["conf"], Y[s]["seen"], Y[s]["seller"]
    a, b = D[s]["confirm"][ctrl], D[s]["confirm"][j]; ap, bp = a >= .5, b >= .5
    fx, br = (bp == y) & (ap != y), (bp != y) & (ap == y)
    d = pd.Series(((bp == y).astype(int) - (ap == y).astype(int))).groupby(sel).sum().values
    rng = np.random.default_rng(42); bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)])
    res[s] = dict(control=dict(acc=float((ap == y).mean()), ll=ll(y, a), acc_seen=float((ap == y)[seen].mean()), acc_new=float((ap == y)[~seen].mean())),
                  selected=dict(acc=float((bp == y).mean()), ll=ll(y, b), acc_seen=float((bp == y)[seen].mean()), acc_new=float((bp == y)[~seen].mean()),
                                ll_seen=ll(y[seen], b[seen]), ll_new=ll(y[~seen], b[~seen])),
                  control_ll_seen=ll(y[seen], a[seen]), control_ll_new=ll(y[~seen], a[~seen]),
                  fixes=int(fx.sum()), breaks=int(br.sum()), net=int(fx.sum() - br.sum()),
                  net_seen=int(fx[seen].sum() - br[seen].sum()), net_new=int(fx[~seen].sum() - br[~seen].sum()),
                  seller_boot_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_net_pos=float((bs > 0).mean()))
(out / "confirm_result.json").write_text(json.dumps(dict(selected=best, by_cut=res), indent=2)); print(json.dumps(res, indent=2))
