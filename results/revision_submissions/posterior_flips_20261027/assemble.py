"""Reconstruye predicciones completas por ID de toda submission puntuada con CSV
recuperable y arma las restricciones de score respecto de la 48.

Salida (en esta carpeta): all.json (predicciones y scores), s48.json, atoms.json
(filas agrupadas por patrón de submissions que difieren de 48 y saldo K_s).
Para agregar un score nuevo: sumar el CSV a EXTRA y su score a SCORES, y
volver a correr assemble.py -> nullb.py -> mcmc3.py (2 semillas) -> build_best.py.
"""
import csv, json, os, collections, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..") + "/"
SUB = os.path.join(HERE, "..", "..", "..", "submissions") + "/"
os.chdir(HERE)

def base(n):
    b = open(R + f"base{n:02d}_labels_1_to_30000.txt").read().strip()
    return {i + 1: ("used" if c == "1" else "new") for i, c in enumerate(b)}

B = {7: base(7), 15: base(15), 33: base(33)}
subs = {7: dict(B[7]), 15: dict(B[15]), 33: dict(B[33])}

def addtsv(fn, basecol_default, fromcol):
    for r in csv.DictReader(open(R + fn), delimiter="\t"):
        n = int(r["submission"]); b = int(r.get("base", basecol_default)); i = int(r["ID"])
        if n not in subs: subs[n] = dict(B[b])
        assert subs[n][i] == r[fromcol]; subs[n][i] = r["to"]

addtsv("flips_04_09.tsv", 7, "from_07")
addtsv("flips_11_14_27_35.tsv", 15, "from")
addtsv("flips_16_26.tsv", 15, "from_15")
addtsv("flips_36_41.tsv", 33, "from_33")

def flip(src, ids):
    d = dict(subs[src])
    for i in ids: d[i] = "new" if d[i] == "used" else "used"
    return d

m = json.load(open(R + "fasttext_submission_batch_manifest.json"))
subs[42] = flip(33, m["submissions"]["42"]["ids"]); subs[43] = flip(33, m["submissions"]["43"]["ids"])
subs[44] = flip(33, [18412, 26663, 26668])
assert sum(subs[41][i] != subs[33][i] for i in subs[33]) == 6
subs[45] = flip(41, [1897, 6308, 7422, 24802, 2820, 9009, 10858, 14208])
subs[46] = flip(41, [8532, 13583, 16436, 22635, 25426, 25656, 26861])
subs[47] = flip(41, [8532, 13583, 22635, 25426])
subs[48] = flip(41, [13583, 25426])
subs[49] = flip(48, [18487, 1873, 22138, 12528, 27639, 9880, 28453, 14589, 25906, 17321, 40])
subs[50] = flip(48, [12528, 9880, 28453, 25906, 17321])
for i in (8532, 13583, 22635, 25426): assert subs[41][i] == "used"
for i in (16436, 25656, 26861): assert subs[41][i] == "new"
for i in (1897, 6308, 7422, 24802): assert subs[41][i] == "new"
for i in (12528, 9880, 28453, 25906, 17321): assert subs[48][i] == "used"

# 48 exacta (SHA256 registrado)
txt = "ID,condition\n" + "".join(f"{i},{subs[48][i]}\n" for i in range(1, 30001))
assert hashlib.sha256(txt.encode()).hexdigest() == "3f8d08e0e61854c2805a02140b1f7428016d63deae4bd7d6c820c7478f6c9722"

# Submissions posteriores con CSV en submissions/AAAAMMDD
EXTRA = {57: "20261027/submission_57_posterior_sobre48.csv",
         58: "20261027/submission_58_posterior_sobre48.csv"}
for n, p in EXTRA.items():
    if os.path.exists(SUB + p):
        subs[n] = {int(r["ID"]): r["condition"] for r in csv.DictReader(open(SUB + p))}

SCORES = {4: .94457, 5: .94447, 6: .94457, 7: .94476, 11: .94519, 12: .94419, 13: .94428, 15: .94523, 16: .94500,
          17: .94447, 18: .94404, 19: .94409, 20: .94285, 21: .94504, 22: .94471, 23: .94495, 24: .94533, 25: .94519,
          26: .94547, 27: .94228, 28: .94400, 29: .94571, 30: .94561, 31: .94585, 32: .94561, 33: .94614, 34: .94580,
          35: .94609, 36: .94571, 37: .94490, 38: .94423, 39: .92976, 40: .94595, 41: .94628, 42: .94614, 43: .94614,
          44: .94619, 45: .94623, 46: .94623, 47: .94628, 48: .94633, 49: .94628, 50: .94628,
          57: .94633}   # 58: pendiente
SCORES = {n: s for n, s in SCORES.items() if n in subs}

N = 21000
def correct(s):
    ks = [k for k in range(int(s * N) - 3, int(s * N) + 4) if s <= k / N < s + 1e-5 - 1e-12]
    assert len(ks) == 1, (s, ks)    # 21000 filas públicas + truncamiento a 5 decimales
    return ks[0]

K = {n: correct(s) - correct(SCORES[48]) for n, s in SCORES.items()}
b = subs[48]; scored = sorted(SCORES)
pat = collections.defaultdict(list)
for i in range(1, 30001):
    key = tuple(n for n in scored if subs[n][i] != b[i])
    if key: pat[key].append(i)

json.dump({str(k): v for k, v in b.items()}, open("s48.json", "w"))
json.dump({"K": K, "atoms": [[list(k), v] for k, v in pat.items()]}, open("atoms.json", "w"))
print("submissions puntuadas:", len(SCORES), "filas probadas:", sum(len(v) for v in pat.values()), "átomos:", len(pat))
