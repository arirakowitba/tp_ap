"""B10: re-run the frozen 10-Oct ensemble (selected_ensemble.json -> submission_optimized_ensemble.csv, public .94428)
without changing its logic, to (a) refit 70k->test and measure fidelity to the historical CSV and
(b) evaluate it on Antigravity's fit/tune/confirm cuts (same row IDs as A_hist) training on fit+tune.

usage: python b10_eval.py MAIN_WORKTREE OUTDIR full|split SEED THREADS
  MAIN_WORKTREE: checkout of origin/main (provides src/ and optimization/)
  split mode reads the A_hist tune/confirm row ids from OUTDIR/../ag/preds_A_hist_seed{SEED}_{tune,confirm}.csv.gz
Uses AP2026Q2_DATA_DIR for the original JSONL. Never reads test labels (there are none) nor public scores."""
import sys, os, json, time, shutil, hashlib
from pathlib import Path
main_wt, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 1
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = str(threads)
sys.path.insert(0, str(main_wt / "src"))
import numpy as np, pandas as pd
import catboost, lightgbm, xgboost, sklearn
import confirm_and_submit as cs
from optimize_leaderboard import enrich, data_folder
from sklearn.metrics import accuracy_score, log_loss

FITONLY = os.environ.get("B10_DEV", "fit+tune") == "fit"
tag = "full" if mode == "full" else (f"seed{seed}_fit" if FITONLY else f"seed{seed}")
od = out / tag; od.mkdir(parents=True, exist_ok=True)
# frozen config copied (read-only use) so outputs never touch the shared optimization/ folder
for f in ["selected_ensemble.json", "metrics.csv", "lgb_tpe_2_params.json", "cat_enriched_d9_params.json", "cat_native_text_d7_params.json"]:
    shutil.copy(main_wt / "optimization" / f, od / f)
cs.OUT = od
_Cat, _LGB = cs.CatBoostClassifier, cs.LGBMClassifier
class Cat(_Cat):
    def __init__(self, **kw): kw["thread_count"] = threads; super().__init__(**kw)
class LGB(_LGB):
    def __init__(self, **kw): kw["n_jobs"] = threads; super().__init__(**kw)
cs.CatBoostClassifier, cs.LGBMClassifier = Cat, LGB

t0 = time.time()
selected = json.loads((od / "selected_ensemble.json").read_text())
folder = data_folder()
raw = pd.read_json(folder / "train_data.jsonlines", lines=True)
y = (raw.condition == "used").astype("int8").to_numpy()
groups = raw.parent_item_id.astype("string").fillna("__ROW__" + raw.id.astype(str)).to_numpy()
env = dict(catboost=catboost.__version__, lightgbm=lightgbm.__version__, xgboost=xgboost.__version__, sklearn=sklearn.__version__,
           numpy=np.__version__, pandas=pd.__version__, python=sys.version.split()[0], threads=threads)
if mode == "full":
    full, ids = enrich(raw); del raw
    test = pd.read_json(folder / "test_data.jsonlines", lines=True); xt, _ = enrich(test, ids); del test
    p = cs.fit_models(full, y, groups, xt, selected, "full_train")
    np.save(od / "test_p.npy", p)
    sub = pd.DataFrame({"ID": np.arange(1, len(p) + 1), "condition": np.where(p >= selected["threshold"], "used", "new")})
    sub.to_csv(od / "submission_b10_full.csv", index=False)
    hist = pd.read_csv(main_wt / "submissions/submission_optimized_ensemble.csv").sort_values("ID")
    agree = float((hist.condition.values == sub.condition.values).mean())
    rep = dict(mode="full", agree_with_hist_94428=agree, n_diff=int((hist.condition.values != sub.condition.values).sum()),
               sha256=hashlib.sha256((od / "submission_b10_full.csv").read_bytes()).hexdigest(), elapsed_sec=round(time.time() - t0), env=env)
else:
    ag = out.parent / "ag"
    tu = pd.read_csv(ag / f"preds_A_hist_seed{seed}_tune.csv.gz"); co = pd.read_csv(ag / f"preds_A_hist_seed{seed}_confirm.csv.gz")
    assert (raw.id.values[tu.row_idx] == tu.item_id.values).all() and (raw.id.values[co.row_idx] == co.item_id.values).all()
    assert (y[co.row_idx] == co.y_used.values).all()
    conf = co.row_idx.to_numpy(); dev = np.setdiff1d(np.arange(len(raw)), np.r_[conf, tu.row_idx.to_numpy()] if FITONLY else conf)
    assert len(np.intersect1d(conf, tu.row_idx)) == 0
    x, ids = enrich(raw.iloc[dev].reset_index(drop=True)); z, _ = enrich(raw.iloc[conf].reset_index(drop=True), ids); del raw
    p = cs.fit_models(x, y[dev], groups[dev], z, selected, f"split_seed{seed}", y[conf])
    pd.DataFrame({"row_idx": conf, "item_id": co.item_id.values, "seller_id": co.seller_id.values, "y_used": y[conf],
                  "is_seen": co.is_seen.values, "p_b10": p, "p_A_hist": co.p_used.values, "thr_A_hist": co.threshold.values}).to_csv(od / f"paired_b10_vs_A_hist_seed{seed}.csv.gz", index=False)
    yy = y[conf]; b = p >= .5; a = co.pred.values.astype(bool)
    rep = dict(mode="split", seed=seed, n_dev=len(dev), n_confirm=len(conf), b10_acc=float(accuracy_score(yy, b)), b10_ll=float(log_loss(yy, p)),
               A_hist_acc=float(accuracy_score(yy, a)), fixes=int(((b == yy) & (a != yy)).sum()), breaks=int(((b != yy) & (a == yy)).sum()),
               elapsed_sec=round(time.time() - t0), env=env, note=("dev = fit only (70k minus tune minus confirm), same training rows as A_hist/CB_Bag3" if FITONLY else "dev = 70k minus confirm (fit+tune)") + "; iterations and threshold 0.5 frozen from historical selected_ensemble.json/metrics.csv")
(od / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2), flush=True)
