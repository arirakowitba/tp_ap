"""Compare predeclared seller gate to round13 strong held-out predictions.

Requires original compact Google Sheet XLSX export, original data.zip, and
round13 checkpoint. Never uses public scores as labels.
"""
import csv
import json
import zipfile

import numpy as np
import openpyxl
from sklearn.metrics import log_loss


def load_original():
    with zipfile.ZipFile("data.zip") as z, z.open("train_data.jsonlines") as f:
        rows = [json.loads(line) for line in f]
    ids = np.array([r["id"] for r in rows])
    seller = np.array([str(r["seller_id"]) for r in rows])
    y = np.array([r["condition"] == "used" for r in rows], dtype=np.int8)
    sheet = openpyxl.load_workbook("original_compact_100k.xlsx",
                                  read_only=True, data_only=True).active
    count = 0
    for i, row in enumerate(sheet.iter_rows(min_row=2, max_row=70001,
                                            values_only=True)):
        assert int(row[0]) == i
        assert row[1] == "train"
        assert int(row[4]) == int(seller[i])
        assert int(row[3]) == int(y[i])
        count += 1
    assert count == len(rows) == 70000
    return ids, seller, y


def main():
    ids, seller, y = load_original()
    part = np.load("round13_unpack/round13/partition.npz")
    fit = part["fit"]
    npos, ntotal = {}, {}
    for s, label in zip(seller[fit], y[fit]):
        npos[s] = npos.get(s, 0) + int(label)
        ntotal[s] = ntotal.get(s, 0) + 1
    results = []
    for name in ("hist", "lgbm", "cat"):
        z = np.load(f"round13_unpack/round13/{name}.npz")
        ix, p = z["rows"], z["p"]
        assert np.array_equal(z["y"], y[ix])
        seen = np.isin(seller[ix], seller[fit])
        assert np.array_equal(z["seen"], seen)
        count = np.array([ntotal.get(s, 0) for s in seller[ix]])
        prior = np.array([(npos.get(s, 0)+3*y[fit].mean())/(ntotal.get(s,0)+3)
                          for s in seller[ix]])
        gate = (count >= 3) & (np.abs(prior-.5) >= .30) & ((prior >= .5) != (p >= .5))
        q = p.copy()
        q[gate] = .6*prior[gate] + .4*p[gate]
        a, b = p >= .5, q >= .5
        fix = int(np.sum((a != y[ix]) & (b == y[ix])))
        brk = int(np.sum((a == y[ix]) & (b != y[ix])))
        delta = (b == y[ix]).astype(int) - (a == y[ix]).astype(int)
        unique = np.unique(seller[ix])
        sums = np.array([delta[seller[ix] == s].sum() for s in unique])
        rng = np.random.default_rng(29)
        boots = sums[rng.integers(len(sums), size=(2000, len(sums)))].sum(axis=1)
        result = dict(model=name,n=len(ix),seen=int(seen.sum()),gated=int(gate.sum()),
                      fix=fix,breaks=brk,base_accuracy=float(np.mean(a==y[ix])),
                      gate_accuracy=float(np.mean(b==y[ix])),
                      base_logloss=float(log_loss(y[ix],p)),
                      gate_logloss=float(log_loss(y[ix],q)),
                      bootstrap_net_ci95=np.percentile(boots,[2.5,97.5]).tolist())
        results.append(result)
        with open(f"round13_{name}_gate_pairs.csv", "w", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["id","seller_id","label_train_used","seen",
                             "base_p_used","seller_prior_used","seller_n_fit",
                             "gate","gate_p_used"])
            for j, pj, sj, nj, gj, qj, vj in zip(ix,p,prior,count,gate,q,seen):
                writer.writerow([ids[j],seller[j],y[j],int(vj),pj,sj,nj,int(gj),qj])
    with open("round13_strong_gate_results.json","w") as f:
        json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
