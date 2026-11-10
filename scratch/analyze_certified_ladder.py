import pandas as pd

df = pd.read_csv('results/certified_five_pillar_candidates.csv')

print(f"Total certified candidates: {len(df)}")
u2n = (df['rec'] == 'new').sum()
n2u = (df['rec'] == 'used').sum()
print(f"Direction distribution: used -> new: {u2n} ({u2n/len(df)*100:.1f}%), new -> used: {n2u} ({n2u/len(df)*100:.1f}%)")

df['cum_gain'] = df['expected_gain'].cumsum()

print("\n--- CERTIFIED PROGRESSION LADDER ---")
print(f"{'K Flips':8s} | {'Exp Gain':10s} | {'Proj Hits':12s} | {'Proj Score':12s} | {'Target Milestone'}")
print("-" * 75)

milestones = [
    (8, "Octeto Quirúrgico Puro (Top 5 Consolidado)"),
    (11, "+ 0km Reales y Moneda 1905 (Top 5+)"),
    (25, "Consenso Multimodelo Fuerte (Ataque a Top 4, 0.9488+)"),
    (50, "Asalto al Podio #3 (Superar a Gonzalo Bühler 0.94895)"),
    (75, "Rumbo a 0.95000 (0.94975+)"),
    (100, "ROMPIENDO LA BARRERA DE 0.95000 (0.95030+)"),
    (110, "ASALTO AL PUESTO #2 (Superar a Nicolás Caveda 0.95095)"),
    (150, "CANDIDATO AL PUESTO #1 (0.95150+)"),
    (199, "MAXIMUM CONVICTION PORTFOLIO (0.95250+)")
]

for k, desc in milestones:
    gain = df.iloc[k - 1]['cum_gain']
    proj_hits = 19898 + gain
    proj_score = proj_hits / 20998.0
    u2n_k = (df.iloc[:k]['rec'] == 'new').sum()
    n2u_k = (df.iloc[:k]['rec'] == 'used').sum()
    print(f"Top {k:3d}   | +{gain:6.2f}    | {proj_hits:7.1f}/20998 | {proj_score:10.5f} | {desc} [U->N: {u2n_k}, N->U: {n2u_k}]")
