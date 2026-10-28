import json, sys, hashlib, numpy as np, csv
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

out = sys.argv[1]
files = sys.argv[2:]
up = dn = w = None
n = 0
for f in files:
    d = np.load(f)
    rows = d["rows"]
    ns_ = float(d["ns"])
    if up is None:
        up, dn = d["up"] / ns_, d["dn"] / ns_; qs = list(d["q"])
    else:
        up += d["up"] / ns_; dn += d["dn"] / ns_; qs += list(d["q"])
    n += 1
up /= n; dn /= n
qbar = float(np.mean(qs))
priv = 1 - up - dn
w = up + priv * qbar                # P(48 wrong)
pub_gain = up - dn                  # expected public net (rows) if flipped
gain = pub_gain + priv * (2 * qbar - 1)   # expected net over the full 30k test
s48 = {int(k): v for k, v in json.load(open("s48.json")).items()}

# per-chain agreement check
per = []
for f in files:
    d = np.load(f); m = int(d["n"])
    per.append((d["up"] - d["dn"]) / float(d["ns"]))
corr = np.corrcoef(per)[0, 1] if len(per) > 1 else float("nan")

sel = np.where(gain > 0)[0]
order = sel[np.argsort(-gain[sel])]
print(f"samples={n}  chain corr={corr:.3f}")
print(f"qbar={qbar:.3f}  rows with E[total gain]>0: {len(sel)}  expected net total: {gain[sel].sum():+.2f}  public: {pub_gain[sel].sum():+.2f}")
print(f"rows with P(48 wrong)>0.5: {(w > 0.5).sum()}")
with open("posterior_por_id.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["ID", "pred_48", "P_publico_y_48_mal", "P_publico_y_48_bien", "ganancia_esperada_publica", "ganancia_esperada_total", "P_48_mal"])
    for r in np.argsort(-gain):
        wr.writerow([int(rows[r]), s48[int(rows[r])], round(up[r], 4), round(dn[r], 4), round(pub_gain[r], 4), round(gain[r], 4), round(w[r], 4)])

best = dict(s48)
for r in order:
    i = int(rows[r]); best[i] = "new" if best[i] == "used" else "used"
txt = "ID,condition\n" + "".join(f"{i},{best[i]}\n" for i in range(1, 30001))
open(out, "w", newline="").write(txt)
k48 = 19873
exp = k48 + pub_gain[sel].sum()
print(f"flips vs 48: {len(order)}  expected public ≈ {exp/21000:.5f} (48 = {k48/21000:.5f}); expected total-test gain ≈ {gain[sel].sum()/30000:+.5f}")
print("sha256", hashlib.sha256(txt.encode()).hexdigest())
print("top:", [(int(rows[r]), round(gain[r], 2)) for r in order[:30]])
