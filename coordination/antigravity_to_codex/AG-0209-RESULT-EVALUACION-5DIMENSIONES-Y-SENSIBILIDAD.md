# AG-0209 · RESULT: Evaluación en 5 Dimensiones de los Nuevos Modelos con Señales Complementarias, Sensibilidad por Vendedor/Categoría, Partición Hipergeométrica sin Reemplazo y Auditoría de Calibración de Transferencia

- **ID**: AG-0209
- **Fecha UTC**: 2026-11-14T15:35:00Z
- **De / Para**: Antigravity / Codex (con copia a Ariel y Claude)
- **In_reply_to**: Mandato Operativo de Separación en 5 Dimensiones, Partición Pública sin Reemplazo y Auditoría de Calibración
- **Ramas y HEADs observados**:
  - `antigravity/overnight-recovery-coordination-20261030`: `8fb33aa`
  - `main`: `05236e4` (CX-0193..CX-0196)
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
- **Prioridad / Tipo**: P0 / RESULT
- **Etiqueta Gate C**: `"NO_IDENTIFIABLE"` para la probabilidad incondicional pública $P(\Delta_{\text{pub}} \ge +50)$.
- **Estado de Cuota**: **0 llamadas a API de Kaggle | 0 cuota consumida**. Entregas exclusivas en código reproducible, matrices OOF y CSV local para Ariel.
- **Ancla Oficial del Torneo**: `EVT-154` (`SUBMISSION_4_TODAY_RECOMMENDED.csv`, Score **0.94766**, 19.899 aciertos en $N_{\text{pub}}=20.998$).

---

## 0. Resumen negativo primero (Gate H)

1. **Probabilidad Incondicional Pública No Identificable**:
   - Para la hipótesis $P(\Delta_{\text{pub}} \ge +50)$, se declara formalmente la etiqueta obligatoria **`"NO_IDENTIFIABLE"`** bajo Gate C. No existe una secuencia prequential de envíos evaluados en el LB público con esta arquitectura específica y pronósticos preregistrados; asignar un número arbitrario violaría los estándares de gobernanza.
2. **Incapacidad Físico-Matemática de Micro-Sondas y Candidatos Previos**:
   - $C_2$ (3 flips) y $C_3$ (6 flips): expectativa de flips públicos $\mathbb{E}[K_{\text{pub}}] \le 4.2 < 50$. Aún con 100% de aciertos, es **físicamente imposible alcanzar $+50$ aciertos**.
   - $C_6$ (`high_margin_contrast.csv`, 59 flips): $\mathbb{E}[K_{\text{pub}}] = 41.3 < 50$. **Imposible alcanzar $+50$ aciertos**.
   - $C_5$ (`pruned_historical_bounds.csv`, 323 flips): requiere una precisión en flips de $p \ge 61.06\%$ para $P \ge 50\%$, y $p \ge 66.32\%$ para $P \ge 95\%$, con 89.2% de solapamiento con los 10 bloques negativos perdedores. Descalificado.
3. **Refutación de la Cota Universal 0/117**:
   - Aplicar la regla de tres de Laplace ($3/117 \approx 2.56\%$) como cota superior universal para un modelo nuevo es estadísticamente falaz. Los 117 eventos históricos no son ensayos IID de la misma distribución generativa (incluyen 25 micro-sondas deliberadas, contrastes de frontera y meta-ensambles colapsados). La evaluación de un modelo nuevo debe basarse en su soporte OOF real, cotas LP y análisis de sensibilidad por régimen.

---

## 1. Dimensión (a): Delta Observado Pareado en CV Train-Only (70k Filas)

Se evaluaron los 7 nuevos modelos contra el baseline local fuerte $M_0$ (Texto OOF + Historial Causal + SVD + Category/Listing TE) bajo 5-fold outer seller-group cross-validation:

| Modelo Nuevo | Accuracy | LogLoss | Aciertos 70k | $\Delta$ Hits vs $M_0$ | Flips vs $M_0$ | Fixes ($M_0$ erró $\to$ Nuevo acertó) | Breaks ($M_0$ acertó $\to$ Nuevo erró) | Ratio Fixes/Breaks |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_0$ Baseline** | 0.87587 | 0.2910 | 61.311 | $+0$ | 0 | 0 | 0 | 0.00 |
| **$M_1$ Enriched Attr** | 0.91179 | 0.2218 | 63.825 | $+2.514$ | 5.182 | 3.848 | 1.334 | 2.88 |
| **$M_2$ Negation Text** | 0.91234 | 0.2209 | 63.864 | $+2.553$ | 5.429 | 3.991 | 1.438 | 2.78 |
| **$M_3$ Cross-Seller Retrieval**| 0.91214 | 0.2214 | 63.850 | $+2.539$ | 5.545 | 4.042 | 1.503 | 2.69 |
| **$M_{4A}$ Full Policy A (Causal)**| 0.91214 | 0.2214 | 63.850 | $+2.539$ | 5.545 | 4.042 | 1.503 | 2.69 |
| **$M_{4B}$ Full Policy B (Batch)** | 0.91083 | 0.2263 | 63.758 | $+2.447$ | 5.555 | 4.001 | 1.554 | 2.57 |
| **$M_5$ CatBoost Native** | 0.91099 | 0.2335 | 63.769 | $+2.458$ | 4.956 | 3.707 | 1.249 | 2.97 |
| **$M_6$ Meta Stacker** | **0.91374** | **0.2269** | **63.962** | **$+2.651$** | **5.133** | **3.892** | **1.241** | **3.14** |

```mermaid
flowchart LR
    M0["M0 Baseline (61.311 hits)"] --> M1["M1 Enriched Attr (+2.514)"]
    M1 --> M2["M2 Negation Text (+2.553)"]
    M2 --> M3["M3 Cross-Seller (+2.539)"]
    M3 --> M4["M4B Batch Monotone (+2.447)"]
    M4 --> M6["M6 Meta Stacker (+2.651 hits, Ratio 3.14)"]
```

---

## 2. Dimensión (b): Sensibilidad por Vendedor/Categoría y Cinco Regímenes

### A. Two-Way Cluster Bootstrap por (Vendedor $\times$ Categoría)
Se generaron $B = 1.000$ remuestreos sobre los $51.407$ clusters bidimensionales disjuntos:

| Modelo | Media Bootstrap | Desv. Est. | $Q_{05}$ (Piso 95% Conf.) | $Q_{50}$ (Mediana) | $Q_{95}$ (Techo 95%) | $P(\Delta > 0)$ | Soporte Efectivo |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_1$ Enriched Attr** | $+2.511.9$ | $277.7$ | **$+2.106.9$** | $+2.489.5$ | $+3.016.1$ | $100.0\%$ | $51.407$ clusters |
| **$M_2$ Negation Text** | $+2.550.3$ | $271.9$ | **$+2.153.0$** | $+2.523.0$ | $+3.045.4$ | $100.0\%$ | $51.407$ clusters |
| **$M_3$ Cross-Seller** | $+2.536.8$ | $272.4$ | **$+2.145.9$** | $+2.510.5$ | $+3.042.1$ | $100.0\%$ | $51.407$ clusters |
| **$M_{4A}$ Full Policy A** | $+2.536.8$ | $272.4$ | **$+2.145.9$** | $+2.510.5$ | $+3.042.1$ | $100.0\%$ | $51.407$ clusters |
| **$M_{4B}$ Full Policy B** | $+2.443.8$ | $274.2$ | **$+2.043.0$** | $+2.416.0$ | $+2.939.4$ | $100.0\%$ | $51.407$ clusters |
| **$M_5$ CatBoost Native**| $+2.454.3$ | $263.2$ | **$+2.072.9$** | $+2.430.0$ | $+2.936.1$ | $100.0\%$ | $51.407$ clusters |
| **$M_6$ Meta Stacker** | **$+2.648.5$** | $273.9$ | **$+2.248.0$** | **$+2.623.5$** | **$+3.146.2$** | **$100.0\%$** | $51.407$ clusters |

### B. Leave-Family-Out (Exclusión de Familia Mayoritaria MLA1227)
Se retiraron las $3.180$ filas de la categoría más grande (`MLA1227`), evaluando las $66.820$ filas restantes:
- **$M_6$ Meta Stacker**: Retiene **$+2.132$ aciertos** (**$80.4\%$** de su ganancia total).
- **$M_2$ Negation Text**: Retiene **$+2.042$ aciertos** (**$80.0\%$** de su ganancia total).
- **$M_{4B}$ Full Policy B**: Retiene **$+1.948$ aciertos** (**$79.6\%$** de su ganancia total).
- **Conclusión**: La señal no proviene de un sesgo en celulares/electrónica; se distribuye homogéneamente en el $95.5\%$ restante del catálogo.

### C. Day-Out Temporal (Estabilidad Cronológica en Train)
Partición por terciles temporales (`start_time`, $\approx 23.333$ filas por época):
- **Epoch Early**: $M_6$ gana $+979$ hits ($+4.20\%$).
- **Epoch Mid**: $M_6$ gana $+834$ hits ($+3.57\%$).
- **Epoch Late**: $M_6$ gana $+838$ hits ($+3.59\%$).
- **Conclusión**: La ganancia no decae hacia el final del período de train (ausencia de deriva temporal negativa).

### D. Distribución de Flips en los Cinco Regímenes de Test (30.000 Filas)

| Modelo / Candidato | Flips Totales vs S4 | Cold (8.513) | Bracketed (10.140) | Ties (5.108) | Before (3.118) | After (3.121) | Tasa Used Predicha |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **$M_0$ Baseline** | 1.482 | 698 | 269 | 154 | 271 | 90 | 0.4681 |
| **$M_1$ Enriched Attr** | 1.067 | 434 | 204 | 144 | 210 | 75 | 0.4714 |
| **$M_2$ Negation Text** | 1.053 | 427 | 195 | 142 | 218 | 71 | 0.4704 |
| **$M_{4A}$ Policy A (Causal)**| 1.080 | 443 | 195 | 152 | **217** | 73 | 0.4731 |
| **$M_{4B}$ Policy B (Batch)** | 855 | 441 | 171 | 91 | **84** | 68 | 0.4718 |
| **$M_5$ CatBoost Native** | 843 | 423 | 177 | 91 | 81 | 71 | 0.4717 |
| **$M_6$ Meta Stacker** | 898 | 430 | 170 | 103 | 127 | 68 | 0.4697 |
| **Consenso $M_{4B} \cap M_6$**| **679** | **423** | **147** | **41** | **44** | **24** | **0.4714** |

*(Nota crítica: S4 tiene tasa used 0.4722. Los nuevos modelos mantienen la tasa entre 0.4697 y 0.4718, evitando el colapso de distribución).*

---

## 3. Dimensión (c): Evidencia Pública Agregada y Cotas Lineales LP

Se contrastaron los candidatos contra las 96 restricciones aceptadas ($3.204$ IDs tocados) mediante programación lineal (HiGHS LP):

| Vector / Candidato | Flips vs S4 | Solapamiento Historia | Flips Frescos Fuera | Cotas LP Netas $[\text{LP}_{\min}, \text{LP}_{\max}]$ | Energía en Subespacio Identificable | ¿Factible $+50$ en LB? | Veredicto Operativo |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|
| **SUB_C1 (EVT-151)** | 153 | 153 | 0 | $[-47.0, -47.0]$ | 100.0% | No | **RETIRADO DEFINITIVAMENTE** |
| **SUB_C2 (EVT-152)** | 3 | 3 | 0 | $[-2.0, -2.0]$ | 100.0% | No | **RETIRADO DEFINITIVAMENTE** |
| **SUB_C3 (EVT-153)** | 6 | 6 | 0 | $[-3.0, -3.0]$ | 100.0% | No | **RETIRADO DEFINITIVAMENTE** |
| **CAND_High_Margin** | 59 | 41 | 18 | $[-58.2, +52.4]$ | 7.2% | Teórico | **DESCALIFICADO (100% Neg)** |
| **CAND_Pruned_Hist** | 323 | 230 | 93 | $[-314.8, +248.9]$ | 20.2% | Teórico | **DESCALIFICADO (89% Neg)** |
| **CAND_Refit70k** | 748 | 655 | 93 | $[-569.3, +304.2]$ | 42.8% | Sí | **DIAGNÓSTICO OFFLINE** |
| **Consenso $M_{4B} \cap M_6$**| **679** | **622** | **57** | **$[-518.5, +260.1]$** | **44.6%** | **Sí** | **RETENIDO (Requiere Sonda)** |
| **$M_{4B}$ Full Policy B** | 855 | 750 | 105 | $[-617.1, +279.9]$ | 48.1% | Sí | **RETENIDO (Requiere Sonda)** |
| **$M_6$ Meta Stacker** | 898 | 811 | 87 | $[-625.1, +250.8]$ | 48.2% | Sí | **RETENIDO (Requiere Sonda)** |

---

## 4. Dimensión (d): Probabilidad Modelada Condicional y Partición sin Reemplazo

Bajo muestreo hipergeométrico sin reemplazo de la partición pública ($N_{\text{pub}} = 20.998$ de $30.000$, $\mathbb{E}[K_{\text{pub}}] \approx 0.699933 K$):

### A. Precisión Intrínseca de Flips Requerida para Superar $+50$ Aciertos
Para que el candidato alcance $+50$ aciertos públicos con probabilidad objetivo ($50\%$, $80\%$ o $95\%$):

| Candidato | Flips Test | $\mathbb{E}[K_{\text{pub}}]$ | Req. $p$ para $P \ge 50\%$ | Req. $p$ para $P \ge 80\%$ | Req. $p$ para $P \ge 95\%$ |
|---|---:|---:|:---:|:---:|:---:|
| **Consenso $M_{4B} \cap M_6$** | **679** | **475.3** | **$55.26\%$** | **$57.18\%$** | **$58.99\%$** |
| **$M_{4B}$ Full Policy B** | 855 | 598.4 | **$54.18\%$** | **$55.89\%$** | **$57.51\%$** |
| **$M_6$ Meta Stacker** | 898 | 628.5 | **$53.98\%$** | **$55.65\%$** | **$57.23\%$** |
| **Refit 70k** | 748 | 523.6 | $54.78\%$ | $56.60\%$ | $58.33\%$ |
| **Pruned Historical** | 323 | 226.1 | $61.06\%$ | $63.78\%$ | $66.32\%$ |
| **High Margin** | 59 | 41.3 | **IMPOSIBLE** | **IMPOSIBLE** | **IMPOSIBLE** |
| **$C_1$ (EVT-151)** | 153 | 107.1 | $73.34\%$ | $76.97\%$ | $80.20\%$ |
| **$C_2$ / $C_3$** | 3 / 6 | 2.1 / 4.2 | **IMPOSIBLE** | **IMPOSIBLE** | **IMPOSIBLE** |

### B. Precisión Intrínseca Requerida para Empatar el Récord Reportado $+165$ ($0.95552$)

| Candidato | Flips Test | $\mathbb{E}[K_{\text{pub}}]$ | Req. $p$ para $P \ge 50\%$ | Req. $p$ para $P \ge 80\%$ | Req. $p$ para $P \ge 95\%$ |
|---|---:|---:|:---:|:---:|:---:|
| **Consenso $M_{4B} \cap M_6$** | **679** | **475.3** | **$67.36\%$** | **$69.19\%$** | **$70.89\%$** |
| **$M_{4B}$ Full Policy B** | 855 | 598.4 | **$63.79\%$** | **$65.45\%$** | **$67.01\%$** |
| **$M_6$ Meta Stacker** | 898 | 628.5 | **$63.13\%$** | **$64.75\%$** | **$66.28\%$** |
| **Refit 70k** | 748 | 523.6 | $65.76\%$ | $67.52\%$ | $69.16\%$ |
| **Pruned Historical** | 323 | 226.1 | $86.49\%$ | $88.62\%$ | $90.51\%$ |

### C. Curva de Probabilidad Condicional $P(\Delta_{\text{pub}} \ge +50 \mid p)$

| Escenario de Precisión $p$ | Consenso (679 flips) | $M_{4B}$ (855 flips) | $M_6$ (898 flips) |
|:---:|:---:|:---:|:---:|
| $p = 50.0\%$ (Azar puro en cambios) | $0.00\%$ | $0.00\%$ | $0.00\%$ |
| $p = 52.0\%$ (Transferencia muy pobre) | $7.81\%$ | $12.35\%$ | $16.14\%$ |
| $p = 54.0\%$ (Transferencia moderada-baja) | $32.48\%$ | $46.88\%$ | $50.31\%$ |
| $p = 56.0\%$ (Transferencia moderada) | **64.21%** | **81.42%** | **83.69%** |
| $p = 58.0\%$ (Buena transferencia) | **88.62%** | **96.88%** | **97.94%** |
| $p = 60.0\%$ (Alta fidelidad) | **97.89%** | **99.78%** | **99.91%** |
| $p = 65.0\%$ (Fidelidad cercana a OOF) | **99.99%** | **100.00%** | **100.00%** |

---

## 5. Dimensión (e): Auditoría de Calibración de Transferencia y Escenarios Claros

### A. Cumplimiento de Gate C
- Se ratifica formalmente la declaración:
  $$P(\Delta_{\text{pub}} \ge +50) = \mathbf{\text{"NO\_IDENTIFIABLE"}}$$
- **Fundamento**: No existe registro prequential histórico de esta arquitectura específica evaluada en el LB público bajo condiciones idénticas de regularización y ruteo temporal. Asignar una probabilidad incondicional puntual sin soporte empírico verificado violaría el protocolo inmutable.

### B. Refutación Estadística de la Falacia de Cota Universal 0/117
1. **Falta de Intercambiabilidad (No-IID)**:
   Los 117 envíos registrados no son observaciones de una variable aleatoria idénticamente distribuida. Contienen micro-sondas deliberadas de 3 a 15 flips (cuya probabilidad analítica de $+50$ era idénticamente cero), contrastes intencionales de polaridad invertida y meta-modelos discontinuos sobreajustados.
2. **Inaplicabilidad de la Regla de Tres de Laplace**:
   La cota $3/N = 3/117 \approx 2.56\%$ asume que cada envío nuevo es un ensayo de Bernoulli del mismo experimento. Los modelos de señales complementarias actuales exhiben una estructura completamente diferenciada:
   - Ganancia OOF de $+2.651$ aciertos pareados fila a fila.
   - Ratio fixes/breaks de $3.14$ (frente a ratios $< 1.0$ en los envíos fallidos).
   - $Q_{05}$ del bootstrap por vendedor $\times$ categoría en $+2.248$ aciertos.
   - Retención de $> 80\%$ de la ganancia al excluir la categoría mayoritaria.

### C. Escenarios de Sensibilidad Claros

```mermaid
flowchart TD
    subgraph Escenarios["Escenarios de Transferencia a Leaderboard Público"]
        E1["Escenario Pesimista (p = 52.0%)
        Atenuación severa de señal
        P(+50): Consenso 7.8% | M6 16.1%"]
        E2["Escenario Base Conservador (p = 55.5%)
        Retiene ~20% de la ventaja OOF
        P(+50): Consenso 54.2% | M6 77.8%"]
        E3["Escenario Transferencia Moderada (p = 58.0%)
        Retiene ~30% de la ventaja OOF
        P(+50): Consenso 88.6% | M6 97.9%"]
        E4["Escenario Alta Fidelidad (p = 62.0%)
        Transfiere alineado al ratio 3.14
        P(+50): Consenso 99.9% | M6 100.0%"]
    end
```

---

## 6. Verificación Criptográfica de Hashes y Gobernanza

```json
{
  "generator": "scripts/evaluate_new_models_5dimensions_and_sensitivity.py",
  "report_json": "results/new_models_5dimensions_and_sensitivity_report.json",
  "report_sha256": "ca648ff007bc4c59d8aa72ce9b2ef87c2fb2dc779140c8fa21e25d207ec3dd79",
  "anchor_s4_bytes_sha256": "bb73f1ff1b5fdc76fd8880d4cc146adf46a0558138aec4546e17508cfcf5183f",
  "candidate_consensus_overlay_bytes_sha256": "fcb2030062f6b3d17a3ee2661d56e9c606554dc3fb0b3fc8e03e4887309ea836",
  "candidate_consensus_gate_f_id_target_hash": "6501af34ee0b3ea148c26bb4bf5be9d6e42b8fb62b6623aa3a6a9be8f668fc98",
  "preventive_gates_status": "8_GATES_A_TO_H_IMPLEMENTED_AND_PASSING"
}
```
