# Auditoría independiente de Option A en tres confirm seller-group

Código: `scripts/audit_optionA_paired.py`. Ejecutado sobre los tres CSV.gz pareados de `results/intensive_20261030`, que contienen etiquetas reales del train. Hashes de los gzip verificados: 42d893124e483062a5fed1ef34eecc02405ba747391722e4d4441b8fe5ebd034, 5249b16ecdb655cd13c40be6f2627e350abdb103c6ae0357810d5d4401c5506e y 7ace3dee84527dd84354299ac7d1ba5e82987a5ec110a49a77a43342ffd453d7. Bootstrap de 2.000 remuestreos de vendedores por segmento. Reponderación hipotética 30k con 71.623% vistos, según el metadata existente; no valida identidad de test público.

| Confirm | n | % vistos | Bag3 vs CB @.5 neto (vistos/nuevos) | Efecto reponderado 30k [IC95%] | Bag3 vs CB umbral tune neto (vistos/nuevos) | Efecto reponderado 30k [IC95%] | Δ logloss bag3−CB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disjoint_3seed | 9592 | 43.8 | 0 (0/0) | 0 [−67,+67] | +15 (−3/+18) | +13 [−61,+83] | −.00385 |
| outer_777001 | 12803 | 30.0 | +9 (+11/−2) | +60 [+6,+119] | +4 (+10/−6) | +50 [−1,+103] | −.00257 |
| outer_777002 | 13259 | 28.8 | +36 (+7/+29) | +66 [+2,+133] | +57 (+9/+48) | +94 [+33,+157] | −.00414 |

Los cortes comparten filas: 1734 entre 0/1, 1782 entre 0/2 y 2504 entre 1/2; vendedores comunes 1811/1821/1968 respectivamente. No sumar netos ni tratar tres intervalos como réplicas independientes. El escenario tune cambia a la vez modelo y umbral, mientras @.5 aísla diferencia con umbral común.

**Límite fundamental:** la base de esta tabla es CatBoost del mismo pipeline, no la predicción histórica D1+D2. La mejora pareada sobre CatBoost y de log loss no predijo el resultado público de Option A (.93695 vs D1+D2 .94652). La diferencia de mezcla vistos/nuevos (confirm 29–44% vistos, test 71.6% vistos) y la distinta calidad de base impiden promover Bag3 por estos resultados. Los intervalos reponderados requieren invariancia del efecto dentro de cada estrato, que no está demostrada. Se requiere una comparación D1+D2 equivalente en train o un baseline fuerte más cercano, una evaluación realmente congelada y auditoría del refit/umbral.

No se ejecutó entrenamiento en este host: los JSON originales no están disponibles aquí. El análisis sí se recalculó fila a fila desde tres artefactos de train y se compartió para que Antigravity contraste su siguiente entrenamiento. Ningún score público se usó como etiqueta.
