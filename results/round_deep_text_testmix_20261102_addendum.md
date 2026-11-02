# Cortes testlike 20261113 y 20261114 — CatBoost profundo + texto OOF

Train original 70k SHA256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`; código `scripts/train_deep_text_20261101.py`, polaridad used=1. OOF de texto dentro de fit con grupos vendedor, selección de iteraciones y umbrales en tune, confirm medido después. Pares y reportes crudos Library `libfile_df35da553980819195c3301622eb0348`, ZIP SHA256 `d071380ec0b42c9f434479bbf6fd44382aaefac4b10a2348427568e0bdd9ef22`.

| Seed | fit/tune/confirm | Base acc/LL | Texto acc/LL | Neto | IC bootstrap vendedor | Vistos/nuevos |
|---|---:|---:|---:|---:|---:|---:|
| 20261113 | 45099/12624/12277 | .949825/.136343 | .952839/.127485 | +37 | [8,68] | +8/+29 |
| 20261114 | 44384/12881/12735 | .946996/.138538 | .950609/.128943 | +46 | [11,85] | +6/+40 |

Hashes de pares: 13 `b4d94e76d6e52b97f74b87274832200ba2d2c90ba6f0f1d794d1a15bec1104ba`, 14 `f07783c928684ffb83331d9d14430bb9bf5a0de08140411f685a38c7b2452c7f`. Estas repeticiones comparten datos de desarrollo con cortes previos, no validan contra un clon pareado de D1+D2 y no justifican promoción de CSV completo. El mayor neto en ambos viene de vendedores nuevos; no atribuir resultados públicos por segmento a partir del score agregado. D1+D2 .94652 sigue ancla pública.
