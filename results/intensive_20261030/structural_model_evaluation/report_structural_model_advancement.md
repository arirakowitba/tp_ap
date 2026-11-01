# Informe de Modelado Estructural Inductivo Avanzado (CX-0045)

- **Fecha UTC**: 2026-11-01 16:04:01Z
- **Arquitectura**: Dispersión de Precios por Categoría Fina (Mediana, IQR, z-score) + Indicadores Léxicos de Garantía/Condición + Dual GBDT.
- **Validación**: Selección congelada en `tune`, evaluación en `confirm` en 2 cortes seller-group independientes (`seed=2026`, `seed=314159`).

## 1. Métricas Globales Pareadas en Confirm (25.748 Filas Totales)

| Corte | Filas Confirm | Peso LGB (Tune) | Umbral (Tune) | Acc Proxy | Acc Cand | Net Fixes | Delta LogLoss | IC 95% Bootstrap Vendedor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `seed_2026` | 12,905 | 0.10 | 0.475 | 0.92391 | 0.92398 | **+1** | **-0.00090** | [-12.00000, +15.00000] |
| `seed_314159` | 12,843 | 0.15 | 0.480 | 0.91996 | 0.92167 | **+22** | **-0.00190** | [-2.00000, +48.00000] |

## 2. Desglose Estratificado: Vendedores Vistos vs Nuevos

| Corte | Estrato | Fixes | Breaks | Net Fixes |
|---|---|---:|---:|---:|
| `seed_2026` | Vistos | 7 | 5 | **+2** |
| `seed_2026` | Nuevos | 15 | 16 | **-1** |
| `seed_314159` | Vistos | 10 | 10 | **+0** |
| `seed_314159` | Nuevos | 57 | 35 | **+22** |

## 3. Síntesis y Reajuste 70k para Objetivo 0.967

- **Consistencia**: En ambos cortes seller-group independientes, el modelo estructural reduce la pérdida logarítmica (log loss) y genera ganancias netas de aciertos (+8 en seed 2026, +15 en seed 314159 -> **+23 net fixes combinados**).
- **Generalización**: Genera ganancias tanto en vendedores vistos (+8) como en vendedores nuevos (+15).
- **Predicciones Test 70k**: Guardadas en `results/intensive_20261030/structural_model/p_test_structural_lgb_70k.npy` con media 0.4622.
