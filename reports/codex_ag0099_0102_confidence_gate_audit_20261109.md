# Auditoría AG-0099 / AG-0101 / AG-0102 — gate de confianza 0.95000

Fecha UTC: 2026-11-09  
Ancla pública confirmada: EVT-132 = 0.94752 = 19,896 aciertos públicos  
Tamaño público estimado: 20,998  
Objetivo operativo de Ariel: P(score público >= 0.95000) >= 0.95

## Veredicto

**NO CUMPLE.** Para llegar a 0.95000 desde EVT-132 hacen falta al menos 53 aciertos públicos netos. Ningún candidato de AG presenta un límite inferior unilateral del 95% de +53. Incluso usando el modelo probabilístico declarado por AG, el máximo comunicado es 43.6%, no 95%.

Los cinco CSV de `submissions/portfolio_20261109_climax_09500/` quedan **BLOCKED**. Un PASS del gate estático anterior sólo significaba ausencia de tokens prohibidos; nunca certificó validez estadística ni mejora predictiva.

## Evidencia revisada

| Fuente | Resultado relevante | Interpretación |
|---|---:|---|
| AG-0099, modelo congelado `cb_variant_a1` | -12 hits; bootstrap vendedor 95% [-56, 24] | Falló la confirmación preregistrada; BLOCK correcto |
| AG-0099, blend post-hoc `blend_a1_native` | +6; IC [-14, 24] | Exploratorio; no congelado y compatible con daño |
| AG-0101, slot 5 | P(S >= .950)=11.48% | Muy por debajo de 95% |
| AG-0102, slot 5 | P(S >= .950)=28.52% | Muy por debajo de 95% |
| AG-0102, expansión K=320 | P(S >= .950)=43.6% | Sigue sin cumplir aun antes de auditar calibración |

## Fallas que invalidan la cifra de confianza

1. Los modelos base usan CV estratificado, no el outer seller-group requerido.
2. Los runners de boosting usan el fold de validación como `eval_set` para early stopping y luego miden sobre ese mismo fold.
3. El meta-stacker se ajusta sobre la matriz OOF completa y se evalúa sobre esa misma matriz; no hay cross-fitting/nesting en el nivel meta.
4. Pesos, bins y umbrales se eligen y evalúan sobre el mismo OOF.
5. La simulación de confianza trata aciertos y pertenencia al público como Bernoulli independientes; ignora dependencia por vendedor/categoría, selección del ancla, selección de candidato y shift.
6. Los patrones de feedback se descubrieron con las 70k etiquetas antes de CV; deben redescubrirse dentro de cada outer fit.
7. La desconvolución de scores agregados no identifica etiquetas individuales. No puede producir “IDs confirmados” ni una regla “cero used->new”.
8. AG-0099 Stage B se solapa con Stage A (1,832 items y 608 sellers), por lo que tampoco es confirmación independiente plena.

## Gate exigido para afirmar 95%

Una candidata sólo puede declararse con P(score >= .95000) >= .95 si cumple simultáneamente:

- Outer CV por seller-group, repetido y con cortes bloqueados; todas las features, patrones y selección se reconstruyen dentro de cada outer fit.
- Early stopping en un inner split separado; el outer fold nunca ajusta rounds, pesos ni umbrales.
- Meta-modelo cross-fitted/nested; probabilidades para cada fila producidas sin usar su etiqueta en ninguna selección.
- Confirm final intacto y congelado, sin reutilización como desarrollo.
- Distribución de ganancia con bootstrap por vendedor y categoría, más muestreo público sin reemplazo/finite-population.
- Calibración prequential: cada distribución debe haberse declarado antes de observar el batch histórico correspondiente; cobertura empírica adecuada o intervalo ensanchado conservador.
- Comparación pareada contra un análogo train-only del ancla fuerte, con seen/new, accuracy y logloss.
- Certificado reproducible con hashes, supuestos y percentiles. Condición mínima: percentil 5 de ganancia pública >= +53.

Si el percentil 5 queda debajo de +53, el resultado es **HOLD/BLOCK**, no “certificado”.

## Acción

Se actualizó `scripts/audit_candidate_provenance_gate.py` para bloquear el namespace, builders, curvas de confianza, super-stacker in-sample y artefactos de desconvolución asociados. La siguiente ejecución de AG debe reconocer que actualmente no existe candidata que cumpla el objetivo y concentrarse en validación nested/cross-fitted, no en generar más CSV.
