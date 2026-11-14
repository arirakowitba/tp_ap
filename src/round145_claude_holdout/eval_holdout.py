"""Evalúa predicciones YA CONGELADAS: accuracy OOF (56k, 5 folds) y accuracy holdout (14k) usando `te`.
usage: python eval_holdout.py DATA_DIR LABELS_CSV OUT_JSON name1 [name2 ...]
`te` está en el orden de test_data.jsonlines; ID de Kaggle = posición 1..N, así que la fila ID-1 de te."""
import sys, json, numpy as np, pandas as pd
D, LAB, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3]
P = pd.read_pickle(D + "feats.pkl"); y = P["y"]
L = pd.read_csv(LAB).sort_values("ID"); yh = (L.condition.values == "used").astype(int); idx = L.ID.values - 1
def ll(yy, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
res = json.load(open(OUT)) if __import__("os").path.exists(OUT) else {}
for n in sys.argv[4:]:
    d = np.load(D + f"pred_{n}.npz"); oof, te = d["oof"], d["te"][idx]
    r = {"oof_acc": float(((oof > .5) == y).mean()), "oof_ll": ll(y, oof), "ho_acc": float(((te > .5) == yh).mean()), "ho_ll": ll(yh, te),
         "oof_pos_rate": float((oof > .5).mean()), "ho_pred_pos_rate": float((te > .5).mean()), "ho_true_pos_rate": float(yh.mean())}
    r["gap_pp"] = 100 * (r["oof_acc"] - r["ho_acc"]); res[n] = r
    print(f"{n:28s} OOF {r['oof_acc']:.5f} ll {r['oof_ll']:.4f} | HO {r['ho_acc']:.5f} ll {r['ho_ll']:.4f} | gap {r['gap_pp']:+.2f} pp | pos oof {r['oof_pos_rate']:.4f} ho {r['ho_pred_pos_rate']:.4f} true {r['ho_true_pos_rate']:.4f}")
json.dump(res, open(OUT, "w"), indent=1)
