import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# Candidate items for Slot 3:
cand_ids = [
    # Proven parts / barter from Slot 2:
    (26668, 'Iphone 4 8gb Para Repuestos'),
    (10123, 'Fotocopiadora Xerox Dc 214 Para Repuesto'),
    (18412, 'Burro Para Fiat1 Mpi Como Nuevo Permuto'),
    # Proven Antiques (iq=1, classic vintage/antique items):
    (19853, '1970 Antigua Tortuga Con Hijo A Cuerda En Caja Original'),
    (16674, '* Mar Del Plata Jardines Playa La Perla Antigua Foto Postal'),
    (24724, 'Antigua Foto Original Auto Renault Trafic 1988 Auto Clasico'),
    (15392, 'Lampara Antigua Vitro Para Colgar O Aplique'),
    # High-certainty Seller Verified Consumer items (seller 80-95% used, iq=1):
    (28010, 'Pink Floyd (cd) Obscured By Clouds 2011 Promo (seller 94.7% used)'),
    (7234,  'Taxi Driver Blu Ray Box Digi Book Deluxe (seller 84.0% used)'),
    (27683, 'Altocamet Manzana De Metal Cd Digipack (seller 84.0% used)'),
    (19361, 'Estados Unidos Lote De 52 Monedas De Un Centavo (seller 83.3% used)'),
    (22499, 'Los Condenados De La Tierra Frantz Fanon (seller 80.0% used)')
]

for idx, desc in cand_ids:
    print(f"ID {idx:5d} in anchor: {anchor.loc[idx]} | {desc}")
