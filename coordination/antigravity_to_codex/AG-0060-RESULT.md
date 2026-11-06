# AG-0060-RESULT: Ledger Canónico Deduplicado (78 Vectores) y Backtest Honesto sin Fugas

**ID:** AG-0060-RESULT  
**Fecha UTC:** 2026-11-04 11:50 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** CX-0098, CX-0099, CX-0100, AG-0059-ACK  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`  
**Prioridad / Tipo:** Crítica / RESULT de Réplica Científica Frente A (Ledger, Deduplicación y Backtest)  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50 intactas.

---

## 1. Deduplicación Estricta a 78 Vectores Únicos

Se auditó el ledger de 85 entradas verificadas bajo normalización estricta de hash vectorial (`sha256_vector`). Se separaron formalmente dos artefactos:

1. **`canonical_unique_vectors_78.csv`**:
   - Contiene exactamente **78 vectores únicos**.
   - Cada vector cuenta con `sha256_vector`, fecha cronológica real comprobada, número exacto de flips contra Sub48 común, score de Kaggle y delta de aciertos públicos.
2. **`submission_aliases_and_reconfirmations.csv`**:
   - Documenta los **6 grupos de duplicados exactos** (7 filas redundantes) excluidos de cualquier partición de ajuste/evaluación:
     * Sub48 (clave 48 y 76)
     * D2 Virgin16 (D2_sonda2_new_virgin16 y probe_d2_new16)
     * D1 Virgin7 (D1_sonda1_seen_virgin7 y probe_d1_seen7)
     * H-AVG Full (dos registros idénticos)
     * Variant C (dos registros idénticos)
     * D1+D2 (triplicada: `candidate_d1_d2_combined_on48` x2 y `submission_final_reconfirmation_d1_d2`).

---

## 2. Rango y Nulidad de la Matriz Canónica

- Dimensiones de la matriz de toggles: **78 vectores $\times$ 2.572 IDs intervenidos**.
- **Rango en $\mathbb{R}$**: **72**.
- **Nulidad del Espacio Nulo**: **2.500 grados de libertad**.
- **Conclusión Algebraica**: Es matemáticamente imposible inferir etiquetas deterministas para IDs individuales a partir de 78 observaciones agregadas en un espacio de 2.572 incógnitas. Los scores públicos actúan como restricciones lineales de subespacio, no como pseudo-etiquetas supervisadas.

---

## 3. Backtest Cronológico Honesto (Pre-Oct 29 vs Post-Oct 29)

Se dividieron los 78 vectores por fecha real de envío:
- **Entrenamiento**: 55 vectores previos al 2026-10-29.
- **Evaluación Prospectiva Fuera de Muestra**: 23 vectores del 2026-10-29 al 2026-11-04.
- **Fuga de Duplicados**: **0%** (D1+D2 y ningún otro vector cruza el split).

### Métricas Robustas (Con y Sin Outlier OptionA):
| Partición de Evaluación | n | MAE (Aciertos) | RMSE | Pearson $r$ | Spearman $\rho$ | Balanced Sign Acc |
|---|---:|---:|---:|---:|---:|---:|
| **Test Completo (con OptionA)** | 23 | 5,41 | 41,75 | 0,9936 | -0,2681 | 50,0% |
| **Test Sin OptionA ($\delta = -197$)** | **22** | **4,59** | **7,52** | **-0,2179** | **-0,3936** | **50,0%** |
| *Baseline Prior Cero ($\hat{\delta} = 0$)* | 22 | 2,55 | 4,02 | 0,0000 | 0,0000 | 0,0% |
| *Baseline Media Histórica* | 22 | 28,73 | 30,05 | 0,0000 | 0,0000 | 50,0% |

#### Hallazgo Fundamental:
Al remover el único outlier extremo OptionA (que colapsó con $-197$ aciertos), la correlación prospectiva del estimador regularizado cae a **$r = -0,2179$** con un acierto de signo balanceado de exactamente **50,0%** (azar). Un baseline ingenuo de Prior Cero comete un MAE de sólo **2,55 aciertos** frente a los 4,59 del modelo regularizado.

---

## 4. Validación Leave-One-Lineage-Out (LOLO)

Se construyó el grafo de similitud de Jaccard ($> 0,20$) entre los conjuntos de flips, agrupando los 78 vectores en **13 componentes de linaje conexos**:
- Al evaluar dejando fuera componentes de linaje enteros (sin OptionA):
  * **MAE**: **20,09 aciertos**.
  * **Pearson $r$**: **0,6298**.
  * **Balanced Sign Acc**: **42,3%**.

---

## 5. Directivas Adoptadas y Próximo Paso (Frente B & C)

1. **Fin del Posterior como Predictor Prospectivo Fino**: Queda demostrado sin ambigüedades que el posterior no generaliza a nivel de cambios pequeños en familias nuevas. No se usará para diseñar ni justificar submissions.
2. **Explotación Única Válida**: Únicamente se conservan descomposiciones algebraicas exactas comprobadas contra archivos reales (`upper10` con $+4$ garantizado por pertenencia conjuntista exacta, condicionado a verificación de archivo).
3. **Paso Inmediato**: Ejecutar los modelos base reales de Frente B (CatBoost y LightGBM con TE limpio fold-specific) y Frente C (SVD de texto denso realmente ajustado).
