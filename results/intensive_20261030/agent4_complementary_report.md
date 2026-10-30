# Agente 4 — Auditoría de Investigación Complementaria Acotada (MLP Disperso)

**Fecha:** 2026-10-30 00:28:09  
**Tiempo de ejecución:** 662.69s  
**Estado:** **DESCARTADO POR CRITERIO DE PARADA EN TUNE — CÓMPUTO LIBERADO PARA BOOSTING**

---

## 1. Verificación de Alineación por ID
- Longitud `fit`: `45,330` | `tune`: `11,765` | `confirm`: `12,905`.
- Coincidencia exacta de índices y orden de `item_id` en `confirm`: **`True`** (`['MLA4083767684', 'MLA7773254020', 'MLA5584337811']` ... `['MLA7866677118', 'MLA2944079554', 'MLA9327405176']`).

## 2. Descomposición Justa: Aporte del Modelo vs. Cambio de Umbral (Confirm, Semilla 2026)
El reporte piloto anterior informó `+7` aciertos netos comparando `CatBoost (th=0.515)` contra `0.90 CB + 0.10 MLP (th=0.500)`. Al separar umbral y modelo en los mismos 12,905 IDs:

| Comparación | Umbral Base | Umbral Cand | Acc Base | Acc Cand | Fixes | Breaks | Neto | IC 95% Vendedor | LL Base | LL Cand | $\Delta$ LL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Comparación mixta del piloto** (`CB@0.515` vs `Blend@0.500`) | 0.515 | 0.500 | 0.92491 | 0.92546 | 42 | 35 | **+7** | [-13.0, +30.0] | 0.18773 | 0.19355 | **+0.00582** |
| **2. Sólo cambio de umbral en CB** (`CB@0.515` vs `CB@0.500`, sin MLP) | 0.515 | 0.500 | 0.92491 | 0.92445 | 14 | 20 | **-6** | [-22.0, +8.0] | 0.18773 | 0.18773 | `0.00000` |
| **3. Umbral común `0.500`** (`CB@0.500` vs `Blend@0.500`) | 0.500 | 0.500 | 0.92445 | 0.92546 | 40 | 27 | **+13** | [-10.0, +42.0] | 0.18773 | 0.19355 | **+0.00582** |
| **4. Umbral común `0.515`** (`CB@0.515` vs `Blend@0.515`) | 0.515 | 0.515 | 0.92491 | 0.92553 | 41 | 33 | **+8** | [-13.0, +32.0] | 0.18773 | 0.19355 | **+0.00582** |

## 3. Ajuste Inductivo de Vocabulario e IDF por Fold (OOF en `fit`)
- **Teacher lineal `txt_sparse_p` (referencia):** Acc = `0.81240`, Log Loss = `0.41851`.
- **MLP disperso (vocabulario/IDF global en `fit`):** Acc = `0.80243`, Log Loss = `0.46680`.
- **MLP disperso (vocabulario/IDF estrictamente inductivo por fold):** Acc = `0.80430`, Log Loss = `0.43689`.
- Al ajustar el vocabulario e IDF estrictamente dentro de cada fold, el MLP disperso sigue muy por debajo del modelo lineal regularizado (`txt_sparse_p`) tanto en accuracy (~6 puntos menos) como en calibración (log loss ~0.41 vs ~0.29).

## 4. Criterio de Parada en `tune`
- **MLP individual en `tune`:** Accuracy = `0.82329` (< 0.860 requerido), Log Loss = `0.40537` (> 0.320 máximo).
- **Mezcla 0.90 CB + 0.10 MLP en `tune`:** Empeora el Log Loss de `0.18077` a `0.18837` (`+0.00353`). El peso óptimo en `tune` para minimizar log loss es exactamente **`w_mlp = 0.000`**.
- **Decisión:** Se descarta formalmente la línea MLP disperso y se libera todo el presupuesto de cómputo para la línea principal **CatBoost + LightGBM** y su validación externa.
