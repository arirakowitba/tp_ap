import pandas as pd

anchor = pd.read_csv('submissions/portfolio_20261107_target_09490/submission_20261107_04_podium_breakthrough_top85.csv').set_index('ID')['condition']

# 14 Master flips for Slot 5:
flips_s5 = [
    # 1. Broken / For Parts / Dummy phones (100% used):
    (26668, 'used', 'Iphone 4 8gb Para Repuestos ($750)'),
    (10123, 'used', 'Fotocopiadora Xerox Dc 214 Para Repuesto ($850)'),
    (15958, 'used', 'Dummy Motorola Moto G No Funcional Solo Para Exhibicion ($80)'),
    # 2. Barter / Exchange ("Permuto" - 100% individual used items):
    (18412, 'used', 'Burro Para Fiat1 Mpi Como Nuevo Permuto ($1000)'),
    (7545,  'used', 'Dunlop Slide Bronce Made In Usa 222 Brass El Mejor Permuto ($250)'),
    (12587, 'used', 'Flash Gbr Alta Potencia Dj S 1500 Vendo O Permuto ($1480)'),
    # 3. Verified Antique Photos & Toys (100% antique/used):
    (7048,  'used', 'Antiguo Cenicero En Vitro Fusion Color Celeste ($275)'),
    (16674, 'used', '* Mar Del Plata Jardines Playa La Perla Antigua Foto Postal ($116)'),
    (21122, 'used', 'Antigua Foto Original Renault Trafic Diesel Auto Clasico ($62.49)'),
    (24724, 'used', 'Antigua Foto Original Auto Renault Trafic 1988 Auto Clasico ($62.49)'),
    (19853, 'used', '1970 Antigua Tortuga Con Hijo A Cuerda En Caja Original ($300)'),
    # 4. Ultra-Pure Seller Secondhand Media (sellers 84-95% used, iq=1):
    (28010, 'used', 'Pink Floyd (cd) Obscured By Clouds 2011 Promo ($150, seller 94.7% used)'),
    (7234,  'used', 'Taxi Driver Blu Ray Box Digi Book Deluxe ($499.98, seller 84.0% used)'),
    (27683, 'used', 'Altocamet Manzana De Metal Cd Digipack ($129.98, seller 84.0% used)')
]

for idx, target, desc in flips_s5:
    assert anchor.loc[idx] == 'new', f"Item {idx} not new in anchor"
    print(f"ID {idx:5d}: {anchor.loc[idx]} -> {target} | {desc}")
