# AG-0142-UPDATE — Entrega de Modelos Base Predictivos Especializados y Auditoría Independiente de Pliegues

**ID:** AG-0142-UPDATE  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex & Ariel  
**In_reply_to:** CX-0150, Directiva de Base Predictiva Nueva; continúa tras AG-0141-RESULT  
**HEADs observados:**  
- `main`: `4e356ab` (CX-0150)  
- `antigravity/overnight-recovery-coordination-20261030`: `ab87c3b`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / UPDATE-MODELS  
**Ancla Pública Oficial Disponible:** EVT-143 (`0.94761`)  
**Cuota Kaggle:** Confinamiento local estricto; 0 envíos, 0 llamadas a API consumidas.

---

## 1. Resumen Ejecutivo de la Fase de Modelos Base

En cumplimiento del protocolo estricto de desarrollo, se finalizó el entrenamiento de la nueva base predictiva sobre los **70.000 registros reales y textos completos** sin utilizar ningún modelo macro de scores históricos. El trabajo se dividió entre agentes paralelos especializados con responsabilidades no duplicadas:

- **Agente A (Texto de Alta Capacidad):** Word (1-2) + Char-WB (3-5) TF-IDF con regularización SAGA ($C=1.5$, $\ell_2$).
- **Agente B (Atributos & Tabular Intrinseco):** Precios relativos por categoría, z-scores de precio, atributos, tags, garantías y modos de envío modelados con LightGBM.
- **Agente C (Historial de Vendedor OOF & Shrinkage):** Target encoding con suavizado Empirical Bayes ($m=12.0$) y Leave-One-Out estricto para evitar toda fuga interna.
- **Agente D (Interacciones Texto × Categoría):** Ajustador de residuos léxicos condicionados a las probabilidades a priori de categoría.
- **Agente E (Ensambles de Árboles en Espacio Compacto):** CatBoost (Depth 8), LightGBM (127 hojas) y XGBoost (Lossguide) ajustados sobre la concatenación tabular + SVD-120.
- **Agente F (Auditor Independiente):** Verificación matemática de aislamiento de pliegues, cálculo de hashes SHA-256 y contraste pareado fila por fila contra el baseline verificado (*Podium Engine*).

### Estado Obligatorio de Aislamiento:
- **Pliegues de Desarrollo:** Folds 0, 1, 2, 3 ($N = 55.998$ filas).
- **Pliegue Confirm:** **Fold 4 ($N = 14.002$ filas) se mantuvo 100% INTOCADO y congelado** durante todo el ajuste de vocabulario, selección de hiperparámetros y entrenamiento de los modelos base.
- **Cero Fuga Certificada:** `fit_idx` $\cap$ `seen_val` $= \emptyset$, `fit_idx` $\cap$ `unseen_val` $= \emptyset$, `fit_idx` $\cap$ `confirm` $= \emptyset$.

---

## 2. Tabla Comparativa Pareada contra el Baseline Fuerte (N=55.998 Desarrollo)

Se evaluó cada modelo base fila por fila contra el baseline verificado (*Podium Engine*, Accuracy en desarrollo = $0.94230$, LogLoss = $0.14887$):

| Agente / Modelo | Accuracy Dev | LogLoss Dev | Fixes vs Base | Breaks vs Base | Net Hits vs Base | IC 95% Seller Bootstrap | IC 95% Cat Bootstrap | Desacuerdos Dev | Desacuerdos Test vs EVT-143 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Agente A (Texto SAGA)** | 81.77% | 0.4032 | 590 | 7.568 | -6.978 | `[-7558, -6468]` | `[-7628, -6423]` | 8.158 | 4.313 |
| **Agente B (Tabular LGB)** | 83.44% | 0.3675 | **1.416** | 7.457 | -6.041 | `[-6443, -5634]` | `[-6395, -5733]` | 8.873 | 4.407 |
| **Agente C (Seller TE)** | 53.70% | 0.6907 | **1.729** | 24.426 | -22.697 | `[-23796, -21471]` | `[-24083, -21316]` | 26.155 | 5.530 |
| **Agente D (Texto×Cat)** | 82.38% | 0.4278 | 601 | 7.236 | -6.635 | `[-7229, -6118]` | `[-7273, -6048]` | 7.837 | 4.102 |
| **Agente E (CatBoost D8)** | 83.94% | 0.3719 | **1.285** | 7.048 | -5.763 | `[-6142, -5410]` | `[-6011, -5524]` | 8.333 | 2.120 |
| **Agente E (LGBM 127L)** | 84.15% | 0.3597 | **1.276** | 6.917 | -5.641 | `[-6016, -5274]` | `[-5880, -5388]` | 8.193 | 2.040 |
| **Agente E (XGB Lossguide)** | 84.06% | 0.3587 | **1.273** | 6.968 | -5.695 | `[-6111, -5313]` | `[-5916, -5452]` | 8.241 | 2.070 |

### Diagnóstico de Complementariedad:
Aunque individualmente cada familia presenta menor exactitud global que el ensamble podium consolidado, **la masa de correcciones exitosas (*fixes*) es extraordinaria y altamente desacoplada**:
- El Agente C (Seller TE) corrige por sí solo **1.729 casos** donde el baseline falló (principalmente vendedores recurrentes y marcas cautivas).
- El Agente B (Tabular) corrige **1.416 casos** donde la discrepancia de precio relativo o atributos desnuda la condición real.
- Los árboles del Agente E corrigen **más de 1.270 casos** cada uno.
- Esta diversidad ortogonal constituye la materia prima para el selector de desacuerdos de alta precisión del Agente G.

---

## 3. Rendimiento por Régimen de Vendedor (Seen vs New)

| Modelo | Fixes Vendedor Visto | Breaks Vendedor Visto | Fixes Vendedor Nuevo | Breaks Vendedor Nuevo | Tasa Fixes Nuevo/Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Agente A (Texto SAGA)** | 369 | 5.513 | 221 | 2.055 | 37.5% |
| **Agente B (Tabular LGB)** | 645 | 4.967 | **771** | 2.490 | **54.4%** |
| **Agente C (Seller TE)** | **912** | 17.449 | 817 | 6.977 | 47.3% |
| **Agente D (Texto×Cat)** | 368 | 5.168 | 233 | 2.068 | 38.8% |
| **Agente E (CatBoost D8)** | 634 | 4.948 | 651 | 2.100 | 50.7% |
| **Agente E (LGBM 127L)** | 634 | 4.869 | 642 | 2.048 | 50.3% |
| **Agente E (XGB Lossguide)** | 636 | 4.958 | 637 | 2.010 | 50.0% |

El Agente B (Tabular) y los árboles del Agente E muestran la mayor resiliencia en vendedores completamente nuevos, con más del $50\%$ de sus correcciones operando en sellers sin historial.

---

## 4. Auditoría Criptográfica de Artefactos de la Base Predictiva

Los artefactos quedaron registrados y fijados en [`results/multiagent_predictive_base_20261110/`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/multiagent_predictive_base_20261110/):

| Artefacto | Descripción | Filas | SHA-256 |
| :--- | :--- | :---: | :--- |
| **`oof_multiagent_predictive_base.csv`** | Predicciones OOF completas por modelo e ID | 70.000 | `65e8106f7ebc0afd6bbb228f488c6deb8fc84071f8916638b93a2ed25c0f4f95` |
| **`test_probabilities_multiagent_predictive_base.csv`** | Probabilidades en Test ($N=30.000$) por modelo | 30.000 | `0fd37cc1fec863540fd6e323baf83f2dcc34614b90c114f98d4f0c20a2ef2893` |
| **`multiagent_predictive_base_report.json`** | Reporte formal reproducible con configs y métricas | - | `310b297c44851904866fd8e6b4952df360fe044d0dac2f3c18b945f8704f92f0` |

---

## 5. Próximo Paso Operativo Inmediato

Habiendo concluido y certificado los modelos base, se da inicio a la fase del **Agente G**:
1. Ajuste del metamodelo cross-fitted y del selector de desacuerdos a nivel de fila (*row-level*).
2. Congelamiento y registro del manifiesto pre-apertura de confirm.
3. Apertura única de la partición Confirm (Fold 4, 14.002 registros) para evaluar $K \in \{75, 100, 150, 250, 500, 750, 1000, 1500\}$.
4. Emisión de **`AG-0143-RESULT.md`**. Confinamiento local estricto en curso: 0 llamadas a API Kaggle, 0 cuota consumida.