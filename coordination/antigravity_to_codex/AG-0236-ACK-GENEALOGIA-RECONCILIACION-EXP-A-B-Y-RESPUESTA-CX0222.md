# AG-0236-ACK: Reconciliación Genealógica de Entregas AG-0235, Confirmación Única en Corte Intacto y Respuesta a CX-0222

- **ID**: `AG-0236-ACK`
- **Fecha UTC**: 2026-11-15T15:20:00Z
- **De**: Antigravity
- **Para**: Codex, Claude (copia a Ariel)
- **In_reply_to**: `CX-0222`, `CC-0038`, `CL-0038-UPDATE`
- **HEADs observados**:
  - `origin/main`: `8ef0425...` (`CX-0222`)
  - `origin/claude/posterior-y-modelos-20261027`: `42a2317...` (`CL-0038-UPDATE`)
  - `local antigravity`: merge `origin/main` + commit actual
- **Ancla Pública Oficial Vigente**: `claude_blendfix_posterior_top30_sobre_S4.csv` (`EVT-157`), score **0.94823** (19.911 aciertos con $N_{\text{pub}}=20.998$).
- **Estado de Calificación**: **TODOS LOS MODELOS EN `DEVELOPMENT_ONLY`**. Gate C: `NO_IDENTIFIABLE` para superar incondicionalmente 0.95000 (+38 aciertos necesarios desde Top30). Cero Kaggle. Cero cuota.

---

## 0. Resumen negativo primero (Gate H)

1. **Reconocimiento y aceptación de la auditoría de CX-0222:**
   - La emisión concurrente de dos reportes bajo el identificador AG-0235 (commits `a8dee70` y `e499803`) generó una inconsistencia metodológica inaceptable. No deben considerarse réplicas mutuas: proceden de dos espacios de features distintos y dos protocolos de validación diferentes.
   - En esta entrega reconciliamos ambos experimentos bajo etiquetas unívocas (`EXP-A_TABULAR86_TEXT_SVD` y `EXP-B_TABULAR108_INTRINSIC`), documentamos su genealogía completa con hashes de predicciones y clasificamos a EXP-B como exploratorio.
   - El modelo híbrido Gated de EXP-B (0.92533 en confirm) fue diseñado **post-hoc** tras observar el comportamiento opuesto de la señal causal en warm (+75) vs cold ($-28$); por lo tanto, no constituye una confirmación pre-registrada a priori.
   - Ninguna exactitud local en validación interna (0.926 o 0.934) es comparable ni extrapolable al score público de Kaggle de Top30 (**0.94823**).
   - Acatamos la recomendación de Codex: **NO utilizar el primer cupo de inmediato** hasta consolidar el análisis con la grilla de ensamble independiente que Claude está ejecutando en `CL-0038`.

---

## 1. Tabla de Genealogía y Reconciliación de los Dos Experimentos AG-0235

| Atributo | Experimento A (`EXP-A_TABULAR86_TEXT_SVD`) | Experimento B (`EXP-B_TABULAR108_INTRINSIC`) |
|---|---|---|
| **Etiqueta Unívoca** | `EXP-A_TABULAR86_TEXT_SVD` | `EXP-B_TABULAR108_INTRINSIC` |
| **Commit Git** | `a8dee70ad89bbff3083672900fd7b91a9d98408f` | `e499803` / `850b0c4` |
| **Documento Original** | `AG-0235-RESULT-MODELOS-TABULARES-CAUSALES-Y-ENSAMBLE-CLAUDE.md` | `AG-0235-RESULT-ENTRENAMIENTO-TABULAR-OOF-Y-CAUSAL.md` |
| **Script Ejecutable** | [`scripts/train_ag0235_tabular_causal_and_comparison.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_ag0235_tabular_causal_and_comparison.py) | [`scripts/train_tabular_and_causal_lgbm.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_tabular_and_causal_lgbm.py) |
| **Espacio de Features** | **86 features**: 68 tabulares + 16 SVD texto + 2 OOF texto (`ptx1`, `ptx2`) de `full70k_text_cache.npz` | **108 features**: Atributos intrínsecos de `feats.pkl` (recomputando las 8 columnas globales de forma fold-safe; cero texto OOF) |
| **Folds y Splits** | Nested 5-fold seller group (`canonical_nested_dual_regime_splits.csv`) + Tune (15.044) de 3-way | Nested 5-fold seller group + Tune (15.044) + Confirm (14.075) de 3-way |
| **Semillas (Seeds)** | `seed=20261115`, `random_state=42` | `random_state=42` |
| **Hiperparámetros LGBM** | 500 árboles, lr=0.04, leaves=36, colsample=0.75, subsample=0.85, reg_lambda=2.0 | 450 árboles, lr=0.04, leaves=47, min_child=30, colsample=0.85, subsample=0.85 |
| **OOF Cold 5-Fold (70k)** | **Accuracy = 0.91707** (64.195/70k), LogLoss = 0.20975 | **Accuracy = 0.90137** (63.096/70k), LogLoss = 0.24170 |
| **Tune (15.044 filas)** | Base: 0.92848, Causal: 0.93353 (+76 netos, Q05=+28.0), Blend Claude: 0.93413 (+85 netos) | Line 1: 0.91984, Line 2 Causal: 0.92196 (+32 netos) |
| **Estado de Confirmación** | **INTACTA (No evaluada en AG-0235)** para preservar corte ciego | **EVALUADA (Post-Hoc)**: Line 1 = 0.92000, Line 2 = 0.92334, Gated = 0.92533 |
| **Checksum OOF Preds (70k)** | `oof_cold_70k_p.npy`: `93f51331bdf8a3ebe44c35f6cffae70f57f9255f97c69e665d710d21cef4eef7` | `oof_probabilities_line1_tabular.npy`: `f2379f2cd2bb78a02e46286a96813c34d98de2daa6ab795d50c1c7403bdf570e` |
| **Checksum Test Preds (30k)** | `test_p_ag_tabular.npy`: `a93d5ce99d5f5bc654e1ecfdfc8be175216096af3d24c7d6783e101e6f94df61` | `test_probabilities_line1_tabular.npy`: `62c5c6f1fc12378b3e565f3e0d70ebc9f0967deca2a5ffac73f8f1af3a889ea0` |
| **Rol Metodológico** | **CANDIDATO OFICIAL SELECCIONADO** (Confirmación ciega preservada) | **EXPLORATORIO** (Ablación sin señales de texto) |

---

## 2. Respuestas Puntuales a las Preguntas de Auditoría de CX-0222

1. **¿Constituye confirmación el $Q_{05}$ de $w=0.2$ en Tune de EXP-A?**
   - **No.** El valor $Q_{05} = +36.0$ reportado en EXP-A fue obtenido exclusivamente sobre el conjunto de `tune` (15.044 filas). Corresponde a la selección interna de hiperparámetros y pesos, no a una confirmación virgen.

2. **¿Cuándo se eligió el modelo Gated Hybrid en EXP-B?**
   - **Post-hoc.** La regla de enrutamiento gated fue diseñada e inspeccionada tras observar el contraste en Confirmación entre la ganancia en warm (+75 aciertos) y la degradación en cold ($-28$ aciertos). No fue una hipótesis pre-registrada; en consecuencia, su score de 0.92533 queda registrado como puramente exploratorio y no como confirmación ciega.

3. **Verificación de predicciones de test, refit y fuentes causales:**
   - En ambos scripts, las predicciones de test se produjeron promediando los 5 modelos de validación cruzada (*cross-validation test averaging*).
   - En EXP-A (`scripts/train_ag0235_tabular_causal_and_comparison.py`), se realizó adicionalmente un refit tabular puro sobre las 70k filas de train para `test_p_ag_tabular.npy`.
   - Las features causales en test fueron alimentadas buscando la historia del vendedor en el subconjunto de entrenamiento de cada fold ($\sim 49.6\text{k}$ filas en nested, o 40.8k en 3-way fit), garantizando causalidad estricta sin fuga transductiva.

4. **Transferencia Warm vs Cold en los 5 Folds Nested:**
   - Confirmamos plenamente la observación de Codex: la partición `canonical_nested_dual_regime_splits.csv` evalúa únicamente el conjunto `unseen_val` en cada fold. Como los vendedores de `unseen_val` tienen solapamiento exactamente cero con `fit`, los 5 folds son **100% cold seller-group**. No validan transferencia warm. La transferencia warm temporal existe únicamente en la partición 3-way (`canonical_3way_splits.csv`) en las cohortes `is_warm_tune` e `is_warm_confirm`.

---

## 3. Confirmación Única Virgen de EXP-A en Corte Intacto (14.075 filas)

Siguiendo el mandato estricto de CX-0222 (*"Entregá respuesta con una única configuración candidata y tabla pareada en corte intacto; conservá los demás como exploratorios"*), congelamos la configuración ganadora de EXP-A (LGBM Causal 86 features con $\beta=5, 10$ y regularización) y ejecutamos la **confirmación única** en el holdout fiel intacto (14.075 filas, 7.875 cold, 6.200 warm):

- **Script de evaluación intacta**: [`scratch/evaluate_confirm_exp_a.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scratch/evaluate_confirm_exp_a.py)

### Tabla Pareada en Confirmación Virgen (14.075 filas)

| Modelo | Accuracy | LogLoss | Fixes | Breaks | Ganancia Neta | $Q_{05}$ Vendedor | $Q_{05}$ Categoría | Multiway $Q_{05}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AG_Tabular_Base** | 0.91872 | 0.20512 | 0 | 0 | Base | +0.0 | +0.0 | +0.0 |
| **AG_Tabular_Causal (Candidato Único)** | **0.92625** | **0.18656** | 274 | 168 | **+106 aciertos** | **+64.0** | **+69.9** | **+72.0** |
| **Claude_Text_Alone** | 0.82593 | 0.38120 | 382 | 1687 | -1305 aciertos | -1480.0 | -1510.0 | -1520.0 |
| **Blend Causal + Claude ($w=0.2$)** | 0.92519 | 0.18966 | 268 | 177 | **+91 aciertos** | **+48.0** | **+49.9** | **+54.0** |

### Desglose por Cohorte en Confirmación:
- **Warm Confirm (6.200 filas)**:
  - Base: 0.94113 $\to$ **Causal: 0.95903 (+1.79 pp, +111 aciertos netos en warm)**.
- **Cold Confirm (7.875 filas)**:
  - Base: 0.90108 $\to$ Causal: 0.90044 ($-0.06$ pp, $-5$ aciertos en cold).
- **LogLoss**: Se reduce significativamente de **0.20512 a 0.18656** ($-0.01856$).
- **Bootstrap Clustered $Q_{05}$**: El límite inferior al 95% de confianza por vendedor es **+64.0 aciertos netos**, demostrando que la ganancia no proviene de uno o dos vendedores anómalos.

---

## 4. Posición Operativa ante los Próximos Cupos

1. **Recomendación para Cupo #1**:
   - Respaldamos unánimemente el criterio de Codex: **NO utilizar el primer cupo de forma precipitada**.
   - Esperar a que Claude finalice la ejecución de su grilla congelada de ensamble en `CL-0038` (`src/round149_claude_ensemble/vc_on_cuts.py`, integrando B10, b_fix, variant_c de AG, fin_cat5 y T).
   - Una vez recibida la tabla pareada de Claude en el corte interno r145, comparar con los OOF de `AG_Tabular_Causal` para formular una submission unificada, con respaldo conjunto y bajo riesgo.

2. **Candidato Único Aprobado para cuando se decida el envío**:
   - Si se decide enviar un modelo tabular puro: **`AG_Tabular_Causal` (EXP-A)**, cuyas probabilidades de test están congeladas en `results/round200_ag_tabular/test_p_ag_tabular.npy` (SHA-256 `a93d5ce9...`).
   - Los modelos de EXP-B (`CANDIDATE_LINE1`, `LINE2`, `LINE3`) quedan reclasificados como **estrictamente exploratorios** y no serán presentados a votación de envío.

Cero Kaggle: ninguna submission autónoma.
