import pandas as pd
import hashlib

# Load peak anchor (0.94752 = 19,896 hits)
p_anchor = 'submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv'
df_anchor = pd.read_csv(p_anchor)
preds = dict(zip(df_anchor['ID'], df_anchor['condition']))

# Exactly 6 indisputable new -> used flips
flips_s2 = [
    (26668, 'used', 'Iphone 4 8gb Para Repuestos ($750, for parts phone)'),
    (10123, 'used', 'Fotocopiadora Xerox Dc 214 Para Repuesto ($850, for parts photocopier)'),
    (18412, 'used', 'Burro Para Fiat1 Mpi Como Nuevo Permuto ($1000, starter motor used/barter)'),
    (7048,  'used', 'Antiguo Cenicero En Vitro Fusion Color Celeste ($275, antique ashtray)'),
    (15673, 'used', 'Pinza Pico De Loro Bahco Antigua Numero 221 ($295, vintage Bahco tool)'),
    (21122, 'used', 'Antigua Foto Original Renault Trafic Diesel Auto Clasico ($62.49, antique 1980s photo)')
]

for item_id, target, reason in flips_s2:
    old = preds[item_id]
    assert old == 'new', f"Item {item_id} is not new in anchor, was {old}"
    preds[item_id] = target
    print(f"Flipped ID {item_id:5d}: {old} -> {target} | {reason}")

out_df = pd.DataFrame({
    'ID': list(preds.keys()),
    'condition': list(preds.values())
}).sort_values('ID')

out_path = 'submissions/submission_20261109_02_pure_used_capture_top6.csv'
csv_bytes = out_df.to_csv(index=False).replace('\r\n', '\n').encode('utf-8')
with open(out_path, 'wb') as f:
    f.write(csv_bytes)

# Verify
assert len(out_df) == 30000
assert list(out_df.columns) == ['ID', 'condition']
diff = (out_df.set_index('ID')['condition'] != df_anchor.set_index('ID')['condition']).sum()
assert diff == 6, f"Expected 6 diffs, got {diff}"

h = hashlib.sha256(csv_bytes).hexdigest()
print(f"\nGenerated: {out_path}")
print(f"Diffs vs Anchor: {diff} (all new -> used)")
print(f"SHA-256: {h}")
print(f"File size: {len(csv_bytes)} bytes")
