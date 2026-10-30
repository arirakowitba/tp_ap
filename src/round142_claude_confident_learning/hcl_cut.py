"""H-CL (predeclared in CL-0006 §4): confident-learning down-weighting of probably-mislabeled FIT rows,
measured on the lgb_tpe_2 member of the 10-Oct ensemble (B10), per seller-group cut, fit-only training.

For one cut: builds the B10 stack features exactly as confirm_and_submit.fit_models does (TF-IDF/SVD on fit,
OOF target rates and OOF text teacher inside fit), then
  1) inner 5-fold StratifiedKFold (by rows, seed 7) LightGBM with the frozen lgb_tpe_2 params -> OOF p on fit,
  2) q_i = P(given label) ; flag rows with q_i < t,
  3) refit lgb_tpe_2 on fit with weight w for flagged rows, for the fixed grid t in {.05,.10,.20}, w in {0,.25,.5},
     plus control w=1 (same seed), and save tune AND confirm probabilities for every config.
Selection is done later by hcl_select.py using TUNE ONLY; this script never computes confirm metrics.
usage: python hcl_cut.py MAIN_WORKTREE AG_DIR OUTDIR SEED THREADS"""
import sys, os, json, time, hashlib
from pathlib import Path
main_wt, ag, out, seed, threads = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = str(threads)
sys.path.insert(0, str(main_wt / "src"))
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from optimize_leaderboard import enrich, numeric_text, data_folder
from optimize_stack import group_rates, text_predict

t0 = time.time(); od = out / f"seed{seed}"; od.mkdir(parents=True, exist_ok=True)
raw = pd.read_json(data_folder() / "train_data.jsonlines", lines=True)
y = (raw.condition == "used").astype("int8").to_numpy()
groups = raw.parent_item_id.astype("string").fillna("__ROW__" + raw.id.astype(str)).to_numpy()
tu = pd.read_csv(ag / f"preds_A_hist_seed{seed}_tune.csv.gz"); co = pd.read_csv(ag / f"preds_A_hist_seed{seed}_confirm.csv.gz")
assert (raw.id.values[tu.row_idx] == tu.item_id.values).all() and (raw.id.values[co.row_idx] == co.item_id.values).all()
tune, conf = tu.row_idx.to_numpy(), co.row_idx.to_numpy()
fit = np.setdiff1d(np.arange(len(raw)), np.r_[tune, conf]); ev = np.r_[tune, conf]
x, ids = enrich(raw.iloc[fit].reset_index(drop=True)); z, _ = enrich(raw.iloc[ev].reset_index(drop=True), ids)
yf, gf = y[fit], groups[fit]
both = pd.concat([x, z], ignore_index=True); it = np.arange(len(x)); iv = np.arange(len(x), len(both))
tr, va, cats, _, _ = numeric_text(both, it, iv); del both
st, sv = group_rates(tr, va, yf, gf)
oof = np.zeros(len(yf))
for tt, vv in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=2026).split(np.zeros(len(yf)), yf, gf):
    oof[vv] = text_predict(x.iloc[tt], yf[tt], x.iloc[vv])
st["text_oof_probability"] = oof; sv["text_oof_probability"] = text_predict(x, yf, z)
for c in cats:
    dt = pd.CategoricalDtype(categories=pd.Index(st[c].unique())); st[c] = st[c].astype(dt); sv[c] = sv[c].astype(dt)
print("features ready", st.shape, round(time.time() - t0), "s", flush=True)

params = json.loads((main_wt / "optimization/lgb_tpe_2_params.json").read_text()); params["n_estimators"] = 430; params["n_jobs"] = threads
# 1) inner OOF for noise detection (fit rows only)
p_in = np.zeros(len(yf))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(yf)), yf):
    m = LGBMClassifier(**params).fit(st.iloc[a], yf[a]); p_in[b] = m.predict_proba(st.iloc[b])[:, 1]
q = np.where(yf == 1, p_in, 1 - p_in)
print("inner OOF acc", round(float(((p_in >= .5) == yf).mean()), 5), "flag counts", {t: int((q < t).sum()) for t in (.05, .10, .20)}, flush=True)
# 2-3) grid incl. control; same random_state everywhere
grid = [("control", 1.0, 1.0)] + [(f"t{t:.2f}_w{w:.2f}", t, w) for t in (.05, .10, .20) for w in (0.0, .25, .5)]
P = {}
for name, t, w in grid:
    sw = np.where(q < t, w, 1.0) if name != "control" else np.ones(len(yf))
    m = LGBMClassifier(**params).fit(st, yf, sample_weight=sw)
    P[name] = m.predict_proba(sv)[:, 1]; print(name, "done", round(time.time() - t0), "s", flush=True)
nt = len(tune)
np.savez_compressed(od / "probs.npz", configs=np.array([g[0] for g in grid]), tune_rows=tune, confirm_rows=conf,
                    tune=np.vstack([P[g[0]][:nt] for g in grid]), confirm=np.vstack([P[g[0]][nt:] for g in grid]),
                    q_fit=q, fit_rows=fit)
rep = dict(seed=seed, n_fit=len(fit), n_tune=nt, n_confirm=len(conf), inner_oof_acc=float(((p_in >= .5) == yf).mean()),
           flagged={str(t): int((q < t).sum()) for t in (.05, .10, .20)}, elapsed_sec=round(time.time() - t0), threads=threads,
           note="confirm probabilities saved but not evaluated here; selection uses tune only (hcl_select.py)")
(od / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep), flush=True)
