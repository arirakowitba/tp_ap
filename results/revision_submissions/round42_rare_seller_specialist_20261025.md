# Ronda 42 — especialista de vendedores poco frecuentes

Fecha: 2026-10-25. Sin submission a Kaggle.

## Hipótesis y protocolo

El test contiene 8.513 avisos de vendedores ausentes de train. Se evaluó una segunda etapa HistGB especializada en vendedores con **hasta cinco avisos en el fit**; el modelo conserva las señales tabulares, dos predictores de texto OOF agrupados por vendedor y los 40 vectores FastText de los 100.000 textos originales. Se anuló el peso de los vendedores más frecuentes **solo en el ajuste de la segunda etapa**. Se comparó con HistGB + FastText de ronda 39 en dos particiones disjuntas por vendedor previamente guardadas y en el corte por filas de ronda 40, desagregado en vendedores vistos/nuevos. Umbral fijo `p(used) >= 0.475`. Mezcla principal fijada antes de ver resultados: 75% base + 25% especialista; la mezcla 50/50 y el especialista puro son diagnósticos. Ninguna etiqueta de test o score público se usó como etiqueta individual.

En todo train, 36.524/70.000 avisos pertenecen a vendedores con hasta cinco publicaciones; su tasa de `used` es 42,33%, frente a 46,30% de train completo. Esta diferencia no implica que sea igual en el público.

## Resultados pareados contra HistGB + FastText

| Partición | Segmento | n | Especialista puro neto | Mezcla 25% fix/break/net | IC 95% neto de mezcla por vendedor |
|---|---:|---:|---:|---:|---:|
| 39002 | tune | 13.787 | −63 | 35/33/**+2** | `[−15,+19]` |
| 39002 | confirm | 14.533 | −49 | 32/34/**−2** | `[−18,+14]` |
| 39011 | tune | 14.335 | −24 | 49/28/**+21** | `[+5,+37]` |
| 39011 | confirm | 14.490 | −49 | 48/37/**+11** | `[−10,+33]` |
| Corte por filas | Vistos | 18.606 | −91 | 33/40/**−7** | `[−25,+11]` |
| Corte por filas | Nuevos | 9.394 | −1 | 19/16/**+3** | `[−9,+14]` |

El especialista puro pierde claramente. La mezcla 25% es cercana a neutra en la primera partición y más favorable en la segunda, pero el confirm segundo y la medición por segmento incluyen cero. En vendedores nuevos del corte por filas, el *log loss* sí pasa de **0,259810 a 0,257091**, con solo +3 decisiones correctas netas. En vistos la mezcla empeora tanto aciertos como *log loss* (0,151457 → 0,152992). Las direcciones de cambio tampoco son estables: para `used→new`, los netos de los cuatro segmentos agrupados son −2, −4, +7, +15; para `new→used`, +4, +2, +14, −4.

## Decisión y límites

No promover la mezcla ni generar submission. El entrenamiento con subconjunto estricto puede aportar alguna mejora probabilística para vendedores nuevos, pero todavía no demuestra un aumento confiable de aciertos. Estos cortes comparten datos con otros ensayos; los IC por vendedor no corrigen la selección acumulada de hipótesis. La regla FastText `used→new` de ronda 39 continúa siendo la candidata acotada para revisión, sin extrapolar su validación a la política exacta de la submission 33.

## Reproducción

`round42_rare_seller_specialist.py` genera `round42_rare_seller_pairs_{first,second,row}.npz` y `round42_rare_seller_report.json`; `round42_bootstrap.py` genera IC por vendedor con 2.000 réplicas, semilla 42042. El checkpoint conserva los datos originales, textos, vectores, predicciones base pareadas, fuentes comunes, código y hashes. Archivo original `data.zip` SHA256 `f8e78574061cbd75197896fa1cd08a0c78af3cb5d73e7e2597df4ab43b82d574`. Ejecución: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python work20/round42_rare_seller_specialist.py`, luego `python work20/round42_bootstrap.py`.
