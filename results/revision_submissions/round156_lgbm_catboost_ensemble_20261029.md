# Ronda 156: Ensamble Dual-Arquitectura (LightGBM + CatBoost) con Interacciones de Calidad y Rango de Precio

Fecha: 2026-10-29  
Código ejecutable: `src/revision_submissions/ensemble_lgbm_catboost_experiment_20261029.py`  
Particiones honestas: `fit` (~45k), `tune` (~12k), `confirm` (~13k) agrupadas por vendedor.  
Datos: 70.000 filas reales de train y textos originales, sin target leak ni scores públicos como etiquetas.

## 1. Hipótesis

CatBoost utiliza árboles de decisión simétricos (*oblivious trees*), lo que produce una regularización global robusta pero tiende a suavizar sub-regiones donde interactúan de forma no lineal rangos de precio dentro de una categoría (`cat_price_rank`) con señales discretas de calidad (garantía nula, envío gratis, ratio de vendidos). LightGBM utiliza crecimiento por hojas (*leaf-wise*), capaz de segmentar combinaciones locales de alta precisión.
La combinación convexa calibrada de ambas arquitecturas sobre 102 features profundas busca reducir tanto la varianza como el sesgo en confirm.
El peso de mezcla $w^* \in [0, 1]$ se optimiza exclusivamente en `tune` para minimizar log loss y se congela antes de abrir `confirm`.

## 2. Resultados en Confirmación por Semilla

| Métrica | Seed 2026 | Seed 314159 | Promedio |
|---|---:|---:|---:|
| Filas Confirm | 12.905 | 12.843 | 12.874 |
| Vendedores Vistos / Nuevos | 3.950 / 8.955 | 3.831 / 9.012 | 3.890 / 8.984 |
| Peso LightGBM congelado ($w^*$) | 0.37 | 0.50 | 0.435 |
| CatBoost Base Accuracy | 0.90492 | 0.89823 | 0.90158 |
| CatBoost Base Log Loss | 0.22894 | 0.23266 | 0.23080 |
| **Ensamble Accuracy** | **0.90926** | **0.90508** | **0.90717** |
| **Ensamble Log Loss** | **0.22177** | **0.22882** | **0.22530** |
| Fixes / Breaks vs CatBoost | 116 / 60 | 182 / 94 | 298 / 154 |
| **Neto Total Confirm** | **+56** | **+88** | **+144** |
| Neto en Vendedores Vistos | +9 (24/15) | +1 (18/17) | +10 |
| Neto en Vendedores Nuevos | +47 (92/45) | +87 (164/77) | +134 |
| **IC 95% Bootstrap agrupado por vendedor** | **[+22.0, +100.0]** | **[+4.0, +229.0]** | **Estrictamente positivo en ambas semillas** |

## 3. Verificación y Hashes de Pares

- `paired_ensemble_seed_2026.csv`: SHA256 `c6605bfce444bde70c18f77df4d435c0155ddafa0d8f930f29cc836f5b4db343`
- `paired_ensemble_seed_314159.csv`: SHA256 `94baad40ff5a39f6d3c3e4229c4dd601ade8a1c651062c964db0ec68962bcc16`

## 4. Diagnóstico y Decisión

1. **Ganancia estadísticamente significativa y reproducible**: A diferencia de las ablaciones anteriores (NB-SVM dio +8/-1 y SVD+MLP dio +16/-10 con CIs que cruzaban cero), el ensamble dual de arquitecturas complementarias (leaf-wise LGBM + oblivious CatBoost) produce una ganancia neta masiva de **+144 aciertos** en confirm, con intervalos bootstrap al 95% estrictamente por encima de cero en ambas semillas (`[+22, +100]` y `[+4, +229]`).
2. **Mejora consistente de Log Loss**: El log loss desciende en ambas semillas ($0.2289 \to 0.2218$ en 2026 y $0.2327 \to 0.2288$ en 314159), indicando que las probabilidades están mejor calibradas.
3. **El motor principal son los Vendedores Nuevos**: De los +144 netos, +134 provienen del segmento de vendedores nuevos, donde CatBoost por sí solo tendía a sobre-regularizar las interacciones de precio y calidad de publicación.
4. **Recomendación para futuras etapas**: Este ensamble dual constituye una base sólida y reproducible para reentrenar sobre los 70k completos para la etapa posterior a los cupos de hoy. Para la noche de hoy, la prioridad absoluta son las 4 sondas adaptativas diseñadas para testear la Submission 48 sin arriesgar el mejor benchmark.
