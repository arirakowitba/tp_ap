# AG-0185-RESULT-FRENTE1 — Evaluación Pareada del Modelo Amplio Multi-Familia (LightGBM + CatBoost + XGBoost)

- **ID**: `AG-0185-RESULT-FRENTE1`
- **Fecha UTC**: `2026-11-13T23:52:00Z`
- **De**: Antigravity (`AG`)
- **Para**: Codex (`CX`) / Claude (`CL`) / Ariel
- **In_reply_to / Complementa**: `CX-0185`, `AG-0184-ACK`, `AG-0187-RESULT-FRENTE3`
- **Rama / HEADs observados**:
  - `origin/main`: `386f8298fcce7b702d6ce7bc98cd19a8ec8db0df`
  - `antigravity/overnight-recovery-coordination-20261030`: `14f3ef0`
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
  - `origin/codex/ap-refit-candidates-20261111`: `b1b1609813b714082a0851416a390472da018389`
- **Prioridad / Tipo**: `P0` / `RESULT`
- **Estado formal**: `DEVELOPMENT_ONLY` (`0` llamadas a Kaggle API, `0` consumo de cuota, probabilidades públicas `= "NO_IDENTIFIABLE"` por Gate C).

---

## 0. Resumen Negativo y Modelos Degradados Primero (Gate H)

1. **Degradación de Modelos Individuales No-LGBM en Accuracy Pura**:
   - `CatBoost Native` (con 10 columnas categóricas nativas) quedó **`-48` aciertos netos** frente a `LightGBM Enriched` en el panel completo (`Fixes: 186, Breaks: 234`, `Acc: 0.93907 vs 0.94092`). En `seen_bracketed` perdió `-24` hits y en `cold_new` perdió `-24` hits (`Q05 = -84.0`).
   - `XGBoost Enriched` (árboles histograma con regularización $L_2=2.0$) quedó **`-29` aciertos netos** frente al baseline (`Fixes: 153, Breaks: 182`, `Acc: 0.93980 vs 0.94092`). En `seen_bracketed` perdió `-14` hits y en `cold_new` perdió `-15` hits (`Q05 = -62.0`).
   - **Conclusión negativa**: Ninguno de los retadores individuales supera de forma aislada a `LightGBM Enriched`. Se descarta formalmente el reemplazo directo de `LightGBM` por un árbol alternativo único.

2. **Cota de Transferencia hacia la Meta +50**:
   - El ensamble amplio multi-familia (`Broad Blend`) mejora la calibración (`logloss` baja de `0.15905` a `0.15536`) y aporta **`+7` aciertos netos en vendedores nuevos**, pero empata en el global (`-3` hits). Por sí solo, **el Frente 1 no alcanza la meta de `+50` aciertos**.

---

## 1. Experimento Pareado y Especificación Técnica

- **Script Reproducible**: [`scripts/run_frentes_experiment.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/run_frentes_experiment.py)
- **Partición Evaluada**: Partición canónica certificada [`frente3_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/data/splits/frente3_manifest.json) (SHA `9c7a6a9a1305ec3f...`, `fit=44,054`, `val=25,946`).
- **Cache Text/SVD OOF**: `data/splits/text_9c7a6a9a1305ec3f.npz` (`17.1 MB`, SHA ptx1 `7792396ebbc1...`).
- **Baseline**: `LightGBM Enriched` (`m1`, 900 árboles, lr=0.035, leaves=48, min_child=35, colsample=0.75, reg_lambda=1.0).
- **Retador 1 (`CatBoost Native`)**: `CatBoostClassifier` (900 iteraciones, depth=7, lr=0.05, l2=5.0, 10 variables categóricas nativas).
- **Retador 2 (`XGBoost Enriched`)**: `XGBClassifier` (750 estimadores, depth=6, lr=0.035, colsample=0.75, subsample=0.85, reg_lambda=2.0).
- **Ensamble (`Broad Blend`)**: `0.45 * p_lgb + 0.35 * p_cb + 0.20 * p_xgb`.

---

## 2. Resultados de Validación Pareada (Bootstrap Cluster 2,000 Iteraciones)

| Modelo / Candidato | Régimen | N | Hits | Aciertos Base | Net Hits | Fixes | Breaks | Acc Candidato | Acc Baseline | Logloss | Delta Logloss | Q05 / Q95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **CatBoost Native** | `all` | 25,946 | 24,365 | 24,413 | **`-48`** | 186 | 234 | 0.93907 | 0.94092 | 0.15803 | `-0.00103` | `[-84.0, -12.0]` |
| | `seen_bracketed` | 13,102 | 12,587 | 12,611 | **`-24`** | 44 | 68 | 0.96069 | 0.96252 | 0.10567 | `-0.00609` | `[-40.1, -6.0]` |
| | `cold_new` | 12,844 | 11,778 | 11,802 | **`-24`** | 142 | 166 | 0.91700 | 0.91887 | 0.21143 | `+0.00414` | `[-56.0, +6.0]` |
| **XGBoost Enriched** | `all` | 25,946 | 24,384 | 24,413 | **`-29`** | 153 | 182 | 0.93980 | 0.94092 | 0.15863 | `-0.00042` | `[-62.0, +2.0]` |
| | `seen_bracketed` | 13,102 | 12,597 | 12,611 | **`-14`** | 42 | 56 | 0.96146 | 0.96252 | 0.10909 | `-0.00268` | `[-30.0, +2.0]` |
| | `cold_new` | 12,844 | 11,787 | 11,802 | **`-15`** | 111 | 126 | 0.91770 | 0.91887 | 0.20917 | `+0.00188` | `[-42.0, +14.0]` |
| **Broad Blend** | `all` | 25,946 | 24,410 | 24,413 | **`-3`** | 85 | 88 | 0.94080 | 0.94092 | **`0.15536`** | **`-0.00370`** | `[-24.0, +18.0]` |
| | `seen_bracketed` | 13,102 | 12,601 | 12,611 | **`-10`** | 20 | 30 | 0.96176 | 0.96252 | 0.10585 | `-0.00592` | `[-22.0, +2.0]` |
| | `cold_new` | 12,844 | 11,809 | 11,802 | **`+7`** | 65 | 58 | **`0.91942`** | 0.91887 | **`0.20586`** | **`-0.00143`** | `[-10.1, +26.0]` |

---

## 3. Hallazgos Científicos y Conclusión

1. **Mejora Consistente en Calibración y Probabilidades**:
   - Aunque `CatBoost` y `XGBoost` tienen menor exactitud puntual como clasificadores binarios independientes, sus probabilidades son notablemente complementarias: el blend multi-familia reduce la entropía cruzada (`logloss`) de forma global (`-0.00370`) y en ambos regímenes.
2. **Ganancia Positiva en Vendedores Nuevos (`cold_new`)**:
   - El blend logra una mejora neta de **`+7` aciertos** (`Q95 = +26.0`) en el régimen de vendedores nuevos, donde `LightGBM` suele sobreajustar en categorías pequeñas.
3. **Decisión**: Conservar `Broad Blend` como base de probabilidad para el ruteo del Frente 2. No publicar CSV a Kaggle.
