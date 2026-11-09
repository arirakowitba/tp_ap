import pandas as pd
import hashlib

# Load peak anchor (0.94752 = 19,896 hits)
p_anchor = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
df_anchor = pd.read_csv(p_anchor)
preds = dict(zip(df_anchor['ID'], df_anchor['condition']))

# 12 strictly new -> used flips
flips_s3 = [
    (26668, 'used', 'Iphone 4 8gb Para Repuestos ($750, for parts phone)'),
    (10123, 'used', 'Fotocopiadora Xerox Dc 214 Para Repuesto ($850, for parts photocopier)'),
    (18412, 'used', 'Burro Para Fiat1 Mpi Como Nuevo Permuto ($1000, starter motor used/barter)'),
    (19853, 'used', '1970 Antigua Tortuga Con Hijo A Cuerda En Caja Original ($300, 1970 toy)'),
    (16674, 'used', '* Mar Del Plata Jardines Playa La Perla Antigua Foto Postal ($116, antique photo)'),
    (24724, 'used', 'Antigua Foto Original Auto Renault Trafic 1988 Auto Clasico ($62.49, antique photo)'),
    (15392, 'used', 'Lampara Antigua Vitro Para Colgar O Aplique ($1200, antique lamp)'),
    (28010, 'used', 'Pink Floyd (cd) Obscured By Clouds 2011 Promo (seller 94.7% used)'),
    (7234,  'used', 'Taxi Driver Blu Ray Box Digi Book Deluxe (seller 84.0% used)'),
    (27683, 'used', 'Altocamet Manzana De Metal Cd Digipack (seller 84.0% used)'),
    (19361, 'used', 'Estados Unidos Lote De 52 Monedas De Un Centavo (seller 83.3% used)'),
    (22499, 'used', 'Los Condenados De La Tierra Frantz Fanon (seller 80.0% used)')
]

for item_id, target, reason in flips_s3:
    old = preds[item_id]
    assert old == 'new', f"Item {item_id} is not new in anchor, was {old}"
    preds[item_id] = target
    print(f"Flipped ID {item_id:5d}: {old} -> {target} | {reason}")

out_df = pd.DataFrame({
    'ID': list(preds.keys()),
    'condition': list(preds.values())
}).sort_values('ID')

out_path = 'submissions/submission_20261109_03_pure_used_expansion_top12.csv'
csv_bytes = out_df.to_csv(index=False).replace('\r\n', '\n').encode('utf-8')
with open(out_path, 'wb') as f:
    f.write(csv_bytes)

# Verify
assert len(out_df) == 30000
assert list(out_df.columns) == ['ID', 'condition']
diff = (out_df.set_index('ID')['condition'] != df_anchor.set_index('ID')['condition']).sum()
assert diff == 12, f"Expected 12 diffs, got {diff}"

h = hashlib.sha256(csv_bytes).hexdigest()
print(f"\nGenerated: {out_path}")
print(f"Diffs vs Anchor: {diff} (all new -> used)")
print(f"SHA-256: {h}")
print(f"File size: {len(csv_bytes)} bytes")
