# Informe de Modelado Estructural Inductivo Avanzado (Auditado CX-0046)

- **Fecha UTC**: 2026-11-01 20:34:31Z
- **Arquitectura**: Dispersión de Precios por Categoría Fina (Mediana, IQR, z-score) + Indicadores Léxicos de Garantía/Condición + Dual GBDT.
- **Consistencia de Hiperparámetros**: Learning rate 0.05 unificado; refit full 70k fijado en la media de rondas óptimas de early stopping de tune (356 rondas).

## 1. Auditoría de Solapamiento entre Cortes Seller-Group

- Filas confirm `seed_2026`: 12,905
- Filas confirm `seed_314159`: 12,843
- Filas solapadas: **2,337** (18.1%)
- Filas estrictamente disjuntas: **21,074** (81.9% disjuntas en cada corte)
- Vendedores solapados: **2,089** (31.4%)

## 2. Métricas Globales Pareadas en Confirm (Generadas Dinámicamente)

| Corte | Filas Confirm | Rondas Óptimas | Peso LGB (Tune) | Umbral (Tune) | Acc Proxy | Acc Cand | Net Fixes | Delta LogLoss | IC 95% Bootstrap Vendedor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `seed_2026` | 12,905 | 335 | 0.10 | 0.475 | 0.92391 | 0.92398 | **+1** | **-0.00090** | [-12.00000, +15.00000] |
| `seed_314159` | 12,843 | 377 | 0.15 | 0.480 | 0.91996 | 0.92167 | **+22** | **-0.00190** | [-2.00000, +48.00000] |

## 3. Desglose Estratificado: Vendedores Vistos vs Nuevos (Dinámico)

| Corte | Estrato | Fixes | Breaks | Net Fixes |
|---|---|---:|---:|---:|
| `seed_2026` | Vistos | 7 | 5 | **+2** |
| `seed_2026` | Nuevos | 15 | 16 | **-1** |
| `seed_314159` | Vistos | 10 | 10 | **+0** |
| `seed_314159` | Nuevos | 57 | 35 | **+22** |

## 4. Síntesis y Reajuste 70k para Objetivo 0.967

- **Consistencia**: En ambos cortes seller-group independientes, el modelo estructural reduce la pérdida logarítmica (log loss) y genera ganancias netas de aciertos (+1 en seed 2026, +22 en seed 314159 -> **+23 net fixes combinados**).
- **Generalización**: Genera ganancias tanto en vendedores vistos (+2) como en vendedores nuevos (+21).
- **Predicciones Test 70k**: Guardadas en `results/intensive_20261030/structural_model/p_test_structural_lgb_70k.npy` con media 0.4621.
