import subprocess, hashlib, io, sys, json, numpy as np, pandas as pd
refs = subprocess.run(["git","for-each-ref","--format=%(refname)","refs/remotes"],capture_output=True,text=True).stdout.split()
# index of all blobs by path across refs
paths = {}
for r in refs:
    for p in subprocess.run(["git","ls-tree","-r","--name-only",r],capture_output=True,text=True).stdout.split("\n"):
        if p.endswith(".csv"): paths.setdefault(p.split("/")[-1], []).append((r, p))
led = pd.read_csv(io.StringIO(subprocess.run(["git","show","origin/antigravity/overnight-recovery-coordination-20261030:results/exhaustive_evidence_and_aggregate_system_ag0154/canonical_reconciled_ledger_ag0154.csv"],capture_output=True,text=True).stdout))
led = led[led.public_score.notna()].reset_index(drop=True)
V, meta = [], []
for _, row in led.iterrows():
    name = str(row.csv_path).replace("\\","/").split("/")[-1]; got = None
    for r, p in paths.get(name, []):
        b = subprocess.run(["git","show",f"{r}:{p}"],capture_output=True).stdout.replace(b"\r\n", b"\n")
        if hashlib.sha256(b).hexdigest() == row.sha256_lf: got = b; break
    if got is None: meta.append((row.event_id, name, "MISSING")); continue
    d = pd.read_csv(io.BytesIO(got)); assert d.ID.tolist() == list(range(1,30001))
    V.append((d.condition.values == "used").astype(np.int8)); meta.append((row.event_id, name, "OK", row.public_score, row.public_hits_n20998))
ok = [m for m in meta if m[2]=="OK"]
np.save("/home/claude/work/r145/vectors.npy", np.array(V)); json.dump(meta, open("/home/claude/work/r145/vectors_meta.json","w"))
print(len(ok), "ok of", len(meta)); print([m for m in meta if m[2]!="OK"][:20])
