# Ronda 143 — Ablation de rasgos léxicos finos de condición y precio relativo (29-10-2026)

Código ejecutable: [`src/revision_submissions/enriched_features_experiment_20261029.py`](../../src/revision_submissions/enriched_features_experiment_20261029.py).
Resultados JSON: [`results/revision_submissions/round143_enriched_features_results.json`](round143_enriched_features_results.json).
Pares guardados: `paired_enriched_seed_2026.csv` y `paired_enriched_seed_314159.csv` (12.905 y 12.843 filas con IDs explícitos, etiquetas verdaderas y probabilidades de baseline vs enriquecido).

## 1. Hipótesis y motivación

Se probó si la adición de rasgos manuales específicos del dominio de comercio electrónico en español podía resolver las clasificaciones dudosas de CatBoost (donde CatBoost se equivoca pero el texto acierta, ~680 filas en OOF):
- Diccionario léxico fino de condición:
  - Señales explícitas de usado: `como nuevo`, `igual a nuevo`, `poco uso`, `impecable`, `para repuesto`, `desarme`, `no funciona`, `con detalles`, `reacondicionado`, `sin garantía`, `garantía de prueba`.
  - Señales explícitas de nuevo: `sellado de fábrica`, `en blíster`, `0km`, `en caja original`, `sin uso`, `factura A/B`, `garantía oficial/de fábrica`.
  - Conteo acumulativo y puntuación neta de indicios (`net_cue_score`).
- Normalización de precio por categoría: z-score $z = (\log(p) - \mu_{\text{cat}}) / \sigma_{\text{cat}}$ y desviación respecto de la mediana, calculadas estrictamente sobre fit.
- Indicadores de confianza: tienda oficial (`official_store_id`) y participación en promociones/ofertas (`deal_ids`).

## 2. Comparación pareada estricta sobre confirm (mismos IDs que Ronda 142)

| Semilla | Modelo | Acc Total | Log Loss | Fixes / Breaks | Neto Confirm | IC 95% Bootstrap Vendedor | Acc Vistos | Acc Nuevos |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 | CatBoost Baseline | **0.90988** | **0.22834** | — | — | — | 0.93133 | **0.90058** |
| 2026 | CatBoost Enriquecido | 0.90492 | 0.23839 | 133 / 197 | **-64** | **[-139.0, -2.0]** | 0.93082 | 0.89369 |
| 314159 | CatBoost Baseline | 0.89940 | **0.24365** | — | — | — | **0.93200** | 0.88551 |
| 314159 | CatBoost Enriquecido | 0.89940 | 0.24429 | 142 / 142 | **+0** | **[-49.0, +60.0]** | 0.93017 | 0.88629 |

## 3. Diagnóstico y decisión

1. **El enriquecimiento léxico crudo perjudica la generalización:** En la semilla 2026, la adición de estas 15 columnas binarias provocó una pérdida neta de 64 aciertos con un intervalo bootstrap 95% enteramente negativo `[-139.0, -2.0]`. En la semilla 314159, el neto fue exactamente cero (+0, 142 fixes vs 142 breaks).
2. **Causa del fallo:** La probabilidad continua `sparse_p` (derivada de 107.000 n-gramas TF-IDF con regularización L2 en GroupKFold) ya contiene una estimación más suave y calibrada de la probabilidad léxica. Al agregar variables binarias aisladas ("repuesto", "como nuevo", etc.), los árboles de decisión de CatBoost crean puntos de corte tempranos de alta varianza que disparan falsos positivos (por ejemplo, manuales de repuestos, artículos con la frase "no es repuesto" o "sin detalles"), degradando la predicción en vendedores nuevos (donde la exactitud cayó de 0.90058 a 0.89369).
3. **Decisión:** Descartar definitivamente la inclusión directa de indicadores léxicos manuales en el árbol y registrar el resultado negativo.
