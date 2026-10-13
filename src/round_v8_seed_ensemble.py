"""Resume a second-seed TPE model against the frozen grouped OOF baseline."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/round_v8_seed_ensemble"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p >= .5)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7)))}

def main():
    with (ROOT / "data/ap_2026_q_2/raw/train_data.jsonlines").open(encoding="utf8") as f:
        y_all = np.asarray([int(json.loads(line)["condition"] == "used") for line in f], dtype=np.int8)
    t = np.asarray(pd.read_pickle(ROOT / "results/round_v5/cache_base/matrices.pkl")["t"])
    y = y_all[t]
    params = json.loads((ROOT / "results/round_v6/tpe_feature_best_params.json").read_text(encoding="utf8"))
    params.update(n_estimators=1013, n_jobs=2, verbosity=-1, random_state=SEED)
    baseline = np.load(ROOT / "results/round_v6/tpe_feature_best_oof.npy")
    rows=[]
    for k in range(3):
        fold = pd.read_pickle(ROOT / f"results/round_v6/fold_cache/seed42_fold{k}.pkl")
        pred_path = OUT / f"seed{SEED}_fold{k}.npy"
        if pred_path.exists():
            candidate = np.load(pred_path)
        else:
            model = LGBMClassifier(**params)
            started = time.time()
            model.fit(fold["train"], y[fold["train_rows"]])
            candidate = model.predict_proba(fold["valid"])[:, 1].astype(np.float32)
            np.save(pred_path, candidate)
            (OUT / f"seed{SEED}_fold{k}.json").write_text(json.dumps({"fold":k,"seconds":time.time()-started,"metrics":metrics(y[fold["valid_rows"]],candidate)},indent=2),encoding="utf8")
            print("FOLD_SAVED",k,"seconds",round(time.time()-started,1),flush=True)
        iv=fold["valid_rows"]
        rows.append({"fold":k,"baseline":metrics(y[iv],baseline[iv]),"candidate":metrics(y[iv],candidate),"blend_50":metrics(y[iv],.5*(baseline[iv]+candidate))})
        print("FOLD_RESULT",json.dumps(rows[-1]),flush=True)
    report={"seed":SEED,"folds":rows,"baseline_oof":metrics(y,baseline)}
    for name in ["candidate","blend_50"]:
        p=np.empty_like(baseline)
        for k in range(3):
            fold=pd.read_pickle(ROOT/f"results/round_v6/fold_cache/seed42_fold{k}.pkl")
            p[fold["valid_rows"]]=np.load(OUT/f"seed{SEED}_fold{k}.npy") if name=="candidate" else .5*(baseline[fold["valid_rows"]]+np.load(OUT/f"seed{SEED}_fold{k}.npy"))
        report[name+"_oof"]=metrics(y,p)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("SUMMARY",json.dumps(report),flush=True)

if __name__ == "__main__": main()
