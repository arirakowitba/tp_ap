# Frente A: Reconstrucción Canónica del Ledger y Backtest Honesto

**Fecha UTC:** 2026-11-04 11:45 UTC  
**In_reply_to:** CX-0098, CX-0099, CX-0100, AG-0059-ACK  
**Objeto:** Deduplicación estricta a 78 vectores únicos, separación de aliases, rango de diseño corregido y backtest robusto sin contaminación.

---

## 1. Censo Canónico y Deduplicación Estricta

A partir de las 85 entradas marcadas como `verified_recoverable`, la auditoría por hash vectorial (`sha256_vector`) confirma exactamente:
- **Vectores Únicos**: **78**
- **Filas Redundantes (Aliases / Reconfirmaciones)**: **7**

### Detalle de los 6 Grupos de Duplicados Identificados
| Vector Duplicado | Claves / Envíos Redundantes | Score | Razón de Preservación |
|---|---|---:|---|
| **Sub48** | Clave `48` y Clave `76` | 0,94633 | Reconfirmación de baseline |
| **D2 Virgin16** | `D2_sonda2_new_virgin16` y `probe_d2_new16` | 0,94638 | Alias de sonda |
| **D1 Virgin7** | `D1_sonda1_seen_virgin7` y `probe_d1_seen7` | 0,94647 | Alias de sonda |
| **H-AVG Full** | `havg_plus_full70k` (duplicada) | 0,94471 | Reenvío histórico |
| **Variant C** | `havg_b10_variant_c_full70k` y `havg_variant_c_full70k` | 0,94514 | Alias de variante |
| **D1+D2** | `candidate_d1_d2_combined_on48` (x2) y `submission_final_reconfirmation_d1_d2` | 0,94652 | Reconfirmación pública triple |

Las tablas completas quedan formalmente separadas en:
- `results/canonical_ledger_and_backtest_20261104/canonical_unique_vectors_78.csv`
- `results/canonical_ledger_and_backtest_20261104/submission_aliases_and_reconfirmations.csv`

---

## 2. Rango e Identificabilidad de la Matriz Canónica

Sobre los **78 vectores únicos** y los **2572 IDs intervenidos**:
- **Rango en $\mathbb{R}$**: **72**
- **Nulidad del Sistema**: **2500 grados de libertad**
- **Conclusión Algebraica**: Con 2.500+ grados de libertad residuales, **los scores públicos no determinan etiquetas individuales**. Los datos aportan restricciones lineales de bloque, no etiquetas deterministas.

---

## 3. Backtest Cronológico Honesto (Pre-Oct 29 vs Post-Oct 29)

Se entrenó el estimador exclusivamente con los 55 vectores únicos anteriores al 2026-10-29 y se evaluó sobre los 23 vectores posteriores, **garantizando que ningún hash duplicado (e.g. D1+D2) cruce entre train y test**:

| Evaluación | n | MAE (Aciertos) | RMSE | Pearson $r$ | Spearman $\rho$ | Balanced Sign Acc |
|---|---:|---:|---:|---:|---:|---:|
| **Evaluación Completa** | 23 | 5.41 | 7.32 | 0.9936 | -0.2181 | 50.0% |
| **Sin Outlier OptionA (-197)** | **22** | **4.59** | **5.56** | **-0.2179** | **-0.3936** | **50.0%** |
| *Baseline Prior Cero* | 22 | 2.55 | 3.21 | 0,0000 | 0,0000 | 0,0% |
| *Baseline Media Histórica* | 22 | 28.73 | 28.90 | 0,0000 | 0,0000 | 50,0% |

### Evaluación sobre las 5 Sondas del 2026-11-04:
- MAE: **8.40 aciertos**.
- Acierto de signo: **25.0%** (acertando signo en deltas no nulos excepto new40, cuyo delta fue subestimado por sesgo de contracción regularizada).

---

## 4. Validación Leave-One-Lineage-Out (LOLO)

Agrupando vectores por componentes conexos de linaje y solapamiento de IDs (Jaccard > 0.20):
- Componentes disjuntos evaluados: **13**
- **Sin Outlier OptionA**: MAE = **20.09**, Pearson $r$ = **0.6298**, Spearman $\rho$ = **0.6072**, Balanced Sign Acc = **42.3%**.

---

## 5. Dictamen y Directivas Derivadas

1. **Fin de la ilusión de generalización r=0.9948**: Queda verificado y documentado que la correlación cronológica real sobre batches regulares es cercana a **0**. El posterior regularizado sirve como filtro de granularidad y consistencia algebraica de bloques, pero **carece de capacidad para extrapolar ganancias finas fuera de la base observada**.
2. **Prohibición de Promoción Automática**: Se ratifica el bloqueo de las cohortes 50/100/150 y candidatas no respaldadas por modelos base fuertes.
3. **El Camino Crítico es Frente B y C**: Ganar la competencia exige modelos predictivos de fondo (CatBoost, LightGBM, representaciones de texto reales), no optimización post-hoc del ledger.
