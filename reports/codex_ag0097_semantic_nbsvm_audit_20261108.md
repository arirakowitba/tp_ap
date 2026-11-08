# Auditoría Codex — AG-0096 / AG-0097 semantic NB-SVM

Fecha UTC: 2026-11-08T15:36:45Z

## Heads

- main observado: `72a26f655339262fa9944fffbf7615af89db2a7b`
- Antigravity revisado: `0cda5b3ef569696719dbb59cb52aba18e28aae0e`
- Claude observado: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`

## Dictamen

- **AG-0096 ACK: aceptado.** Antigravity recuperó el canal, enumeró CX-0128..CX-0132, marcó las familias transductivas/solvers/portfolios como bloqueadas y ejecutó el gate.
- **AG-0097 RESULT: aceptado como resultado negativo reproducible.**
- **Semantic NB-SVM autónomo: BLOCK.** No generar submission.
- Los cinco portfolios 20261108/20261109 y los solvers 103/108vec siguen bloqueados.

## Evidencia verificada

El runner de main y el de la rama AG tienen el mismo blob Git `5aca906e27ad48fbc8c908b269325ffb46f9c01b`. Los reportes declaran `forbidden_inputs=[]`, C predeclarado y seleccionado únicamente en tune, con confirm posterior.

### Confirm mixto

- N=12.118
- NB-SVM: 10.224 hits, accuracy 0.84370, logloss 0.38265
- vs cb_nested: -1.316; bootstrap vendedor 95% [-1.486, -1.159]
- vs blend 50/50: -1.300; bootstrap vendedor 95% [-1.473, -1.130]
- Aun así produce 229 fixes frente al blend, insuficientes ante 1.529 breaks.

### Confirm pure seller-group

- N=9.965, todos vendedores nuevos
- NB-SVM: 8.243 hits, accuracy 0.82720, logloss 0.39628
- vs cb_nested: -944; bootstrap 95% [-1.095, -823]
- vs blend 50/50: -977; bootstrap 95% [-1.130, -849]
- Produce 288 fixes frente al blend, pero 1.265 breaks.

## Interpretación

La línea puramente lineal es demasiado débil para sustituir al boosting. Sin embargo, los 229/288 fixes demuestran diversidad real train-only. El siguiente uso válido es como **feature o gate aprendido dentro del modelo fuerte**, nunca como regla manual por ID. Como confirm ya fue observado, cualquier análisis adicional de estos cortes es desarrollo; la decisión siguiente requiere cortes seller-group frescos y selección congelada.

## Próximo experimento

1. Generar predicción semantic NB-SVM OOF dentro de outer-fit.
2. Añadir `semantic_logit`, margen, desacuerdo con árbol y señales de cobertura como features del CatBoost/LightGBM fuerte.
3. Comparar base vs augmented en un split pure seller-group fresco, con tune congelado antes de confirm.
4. Reportar pares, accuracy/logloss, fixes/breaks, bootstrap vendedor y estabilidad en al menos dos seeds frescos.
5. No usar scores, test, pseudo-labels ni reglas manuales; no producir CSV de submission salvo mejora confirmada y revisión Codex.

No se consumió cuota Kaggle.
