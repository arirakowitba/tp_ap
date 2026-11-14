"""Split "tipo test" DENTRO de las 56k de train del holdout (para elegir configuraciones sin mirar el holdout externo).
Toma 20% aleatorio (semilla 145) como test interno sin `condition`, mismo formato que test_data.jsonlines.
usage: python make_inner_split.py SRC_TRAIN_JSONL OUT_DIR"""
import sys, os, json, numpy as np
src, out = sys.argv[1], sys.argv[2].rstrip("/") + "/"; os.makedirs(out, exist_ok=True)
lines = open(src).read().splitlines(); n = len(lines)
rng = np.random.RandomState(145); te = np.zeros(n, bool); te[rng.choice(n, n // 5, replace=False)] = True
with open(out + "train_data.jsonlines", "w") as ftr, open(out + "test_data.jsonlines", "w") as fte, open(out + "inner_labels.csv", "w") as fl:
    fl.write("ID,condition,src_row\n"); i = 0
    for r, line in enumerate(lines):
        if not te[r]: ftr.write(line + "\n"); continue
        d = json.loads(line); c = d.pop("condition"); i += 1
        fte.write(json.dumps(d, ensure_ascii=False) + "\n"); fl.write(f"{i},{c},{r}\n")
print("train", (~te).sum(), "test", te.sum())
