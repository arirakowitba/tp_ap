# Extensión: validación con historial de vendedor garantizado y refit 70k

Datos y código: ZIP SHA256 `3e0b032195766e681a411a1d1ea06fe906940dd025a7317adfc5302791bbe886`, `scripts/train_deep_text_20261101.py`, `scripts/train_text_condition_20261101.py`, `scripts/refit_deep_text_20261101.py`. Los cortes 20261104/05 usan seller-group original; 20261106–08 fueron iteraciones de diseño de mezcla, no confirm independientes. Para 20261109 se garantizó al menos una fila de cada vendedor visto en fit y se dejó fuera por vendedor a los nuevos. Todos los modelos de texto hicieron vocabulario e IDF dentro de cada fold GroupKFold por vendedor. CatBoost e hiperparámetros fijos depth6, lr .05; iteración óptima y umbral seleccionados en tune. Se trata de desarrollo, ya que arquitectura y partición fueron refinadas tras revisar confirm previos.

| Corte | Confirm n | % vendedor visto | Acc base → texto | Neto tune thresholds | Neto umbral común .5 | LL base → texto | IC95% seller bootstrap neto | Vistos / nuevos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261104 | 15442 | 34.0% | .91983 → .92948 | +149 | +135 | .20300 → .18457 | [+90,+217] | +9 / +140 |
| 20261105 | 16754 | 31.1% | .92438 → .93267 | +139 | +131 | .19725 → .17585 | [+82,+203] | +1 / +138 |
| 20261106 | 12977 | 50.9% | .92610 → .93018 | +53 | +30 | .18284 → .17324 | [+11,+98] | +6 / +47 |
| 20261107 | 11349 | 59.3% | .93312 → .93814 | +57 | +75 | .17287 → .15965 | [+20,+94] | +7 / +50 |
| 20261108 | 11028 | 60.6% | .93480 → .93816 | +37 | (ver pares) | .16705 → .15775 | [+4,+70] | −5 / +42 |
| 20261109 | 12357 | **72.8%** | **.94432 → .94877** | **+55** | **+97** | **.15392 → .14139** | **[+13,+101]** | **+19 / +36** |

La ganancia en el corte con mezcla test es real frente al CatBoost tabular **del mismo pipeline**, pero no frente a la predicción pública D1+D2. Las pruebas previas de modelos completos en público cayeron; por ello no extrapolar directamente .94877 a Kaggle. El histórico de 18 holdouts ya cubre 95.72% del train, por lo que no queda un confirm prístino dentro de 70k. Los cortes 20261104/05 se solapan en 3541 IDs y 2957 sellers. Tampoco usar scores públicos como etiquetas.

Un refit full 70k del modelo de texto se ejecutó con 1246 rondas (selección tune del corte 20261109), umbral .545, ID de test 1..30000 en orden original, 30k IDs originales únicos. Artefacto exploratorio local: SHA256 `4093ec0c7b70f8d018b34e6cc4ce85f91b7afffdbf77423dad64b3cee3893cdf`, 660 diferencias frente a D1+D2 (257 sellers vistos y 403 nuevos); probabilidades SHA256 `91eeb98c2c3fbe88336d31f62b8783a74c81fbc9d8dec5e7f11a1f1a45b7ab49`. Este candidato no está promovido ni subido a Kaggle. Un control automático bloqueó publicar sus 30k filas de predicción en GitHub por riesgo de divulgar datos derivados; se mantuvo local. Los pares de train y reportes JSON están en Library `libfile_381d6d89afb08191b4926bb3e4d755aa`.

Siguiente gate: evaluación pareada contra B10/H-AVG/HistGB+FastText en exactamente los mismos IDs y una política congelada distinta del corte que la seleccionó; luego calibración por seller visto/nuevo. Proponer al usuario una sonda pública del refit sólo como exploración, preservando D1+D2 .94652. No se garantiza .94700 ni .95404.
