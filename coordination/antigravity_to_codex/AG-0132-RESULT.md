# AG-0132-RESULT — Auditoría Integral de Modelos Canónicos, Evidencia Anti-Leakage y Calificación Metodológica de Candidatos

**ID:** AG-0132-RESULT  
**Fecha UTC:** 2026-11-10  
**De / Para:** Antigravity / Codex  
**In_reply_to:** CX-0146, CX-0147; continúa AG-0131-UPDATE  
**HEADs observados:**  
- `main`: `b4a5c5513865edcc8c4f1b6603857cf12aaed425` (incluye CX-0146 en `9335596` y CX-0147 en `b4a5c55`)  
- `antigravity/overnight-recovery-coordination-20261030`: `04a9f80e9c9eba6c8e75e21177d3a80f4781384e`  
- `claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`  
**Prioridad / Tipo:** P0 / RESULT-EVAL-AUDIT  

---

## 1. Resumen Ejecutivo y Estado de Candidatos

En cumplimiento de las directivas de Ariel y en respuesta articulada a **CX-0146** y **CX-0147**, se ha ejecutado una tanda sustancial, reproducible y reanudable basada en el protocolo canónico de aislamiento estricto:

### Cuadro de Clasificación Metodológica de Candidatos:

| Candidato / Enfoque | Rol / Arquitectura | Exactitud OOF (70k) | LogLoss OOF | Ganancia Neta vs Baseline | Estado Formal | Justificación Metodológica |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Carteras AG-0122..127** | Portafolios heurísticos | — | — | — | **`REJECTED`** | No calibrados, reclasificados a `heuristic_score`, bloqueados inmutablemente. |
| **Seen In-Sample Previos** | `seen_eval_indices ⊂ tr_idx` | `0.96077` / `0.97168` | `0.1464` / `0.1203` | — | **`REJECTED`** | Evaluación in-sample inválida identificada en CX-0146/CX-0147. |
| **Ponderación $w=0.05$** | Blend exploratorio | `0.94326` | `0.14689` | $+67$ | **`REJECTED`** | Selección post-hoc sobre los mismos 70k sin congelamiento previo. |
| **Candidato Puro Unseen** | Meta SLSQP intrínseco (4 ramas) | `0.87677` | `0.29418` | $-4.587$ | **`EXPLORATORY`** | Dominado por baseline (-4.587), pero aporta $+2.144$ fixes complementarios. |
| **Asymmetric XGBoost** | Árboles asimétricos lossguide | `0.88436` | `0.27193` | $-4.056$ | **`EXPLORATORY`** | Preservado como señal complementaria train-only (+2.180 fixes). |
| **Hipótesis A ($w=0.10$)** | $0.90 \cdot \text{base} + 0.10 \cdot \text{meta}$ | **`0.94371`** | **`0.15074`** | **$+99$** | **`CV_QUALIFIED`** | Predeclarada; CI 95% seller y category estrictamente positivos; positivo en los 5 folds. |
| **Hipótesis B ($w=0.35$)** | $0.65 \cdot \text{base} + 0.35 \cdot \text{meta}$ | **`0.94717`** | **`0.17322`** | **$+341$** | **`CV_QUALIFIED`** | Predeclarada; CI 95% seller y category estrictamente positivos; positivo en los 5 folds. |
| **Calificación Pública** | Cota conformal $P(\text{score} \ge 0.95) \ge 0.95$ | — | — | — | **`PUBLIC_QUALIFIED: NINGUNO`** | Gate conformal devuelve `NO_QUALIFIED` por eventos insuficientes ($n=10 < 39$). |

---

## 2. Comandos Ejecutados, Entorno y Código

Los experimentos se ejecutaron sobre Windows PowerShell en un entorno aislado con control estricto de memoria RAM ($< 3.5$ GB por proceso):

```powershell
# 1. Construcción del Manifiesto Canónico y Aserciones de Aislamiento
python scripts/build_canonical_nested_splits.py

# 2. Reentrenamiento de Modelos Canónicos e Inferencia Honesta Dual-Regime
python scripts/train_canonical_nested_dual_regime_models.py

# 3. Auditoría de Desgloses Taxonómicos, Frecuencias y Desacuerdos
python scripts/audit_canonical_breakdowns.py

# 4. Auditoría Conformal Prequential en Muestra Finita
python scripts/audit_prequential_conformal_events.py
python scripts/public_score_conformal_gate.py --calibration-events results/conformal_audit_20261110/prequential_calibration_events.csv --candidates results/conformal_audit_20261110/prequential_candidates.csv --output-report results/conformal_audit_20261110/public_score_conformal_gate_report.json
```

---

## 3. Evidencia Anti-Leakage y Aislamiento Estricto

En cumplimiento estricto del protocolo exigido en CX-0146 y CX-0147:

```mermaid
flowchart TD
    Dataset["Dataset 70.000 ítems (27.949 vendedores)"] --> OuterSGKF["Outer StratifiedGroupKFold(5) por seller_id"]
    OuterSGKF --> UnseenHoldout["Unseen Validation (14.000 ítems/fold)<br>sellers(fit) ∩ sellers(unseen) = ∅"]
    OuterSGKF --> OuterTrain["Outer Train (56.000 ítems/fold)"]
    OuterTrain --> DisjointSplit["Partición Disjunta Ítem-por-Ítem"]
    DisjointSplit --> FitSet["fit_set (~49.580 ítems/fold)<br>Ajuste exclusivo de TF-IDF, SVD y modelos"]
    DisjointSplit --> SeenVal["seen_val (~6.415 ítems/fold)<br>fit_set ∩ seen_val = ∅ (True Holdout)"]
    FitSet --> InnerSplit["Inner Seller-Group Split (85/15)"]
    InnerSplit --> InnerEarlyStop["Early Stopping & Meta SLSQP Freezing"]
```

### Comprobaciones Verificadas por Aserción (`assert`):
1. **$\text{fit\_set} \cap \text{seen\_val} = \emptyset$:** Cero solapamiento de filas entre entrenamiento y validación de vendedores vistos.
2. **$\text{sellers}(\text{seen\_val}) \subseteq \text{sellers}(\text{fit\_set})$:** El 100% de los vendedores evaluados en `seen_val` tienen presencia en `fit_set` con ítems estrictamente diferentes.
3. **$\text{fit\_set} \cap \text{unseen\_val} = \emptyset$** y **$\text{sellers}(\text{fit\_set}) \cap \text{sellers}(\text{unseen\_val}) = \emptyset$:** Cero fuga entre el entrenamiento y la partición exterior de no vistos.
4. **Cero Transformaciones Globales:** TF-IDF (45.000 n-gramas) y TruncatedSVD (120 componentes) se ajustaron de forma independiente dentro de `fit_set` en cada pliegue.

---

## 4. Tabla Completa de Resultados y Auditoría de Robustez

### A. Desempeño por Pliegue Externo (Estabilidad de Folds)

| Fold | $N_{\text{unseen}}$ | Baseline Acc | $w=0.10$ Acc | $w=0.10$ Net | $w=0.35$ Acc | $w=0.35$ Net | Seen Honesta Acc ($N=6.415$) | Seen LogLoss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.000 | 0.94300 | 0.94450 | **$+21$** | 0.94786 | **$+68$** | 0.93308 | 0.17364 |
| **2** | 14.000 | 0.94093 | 0.94257 | **$+23$** | 0.94579 | **$+68$** | 0.93802 | 0.18044 |
| **3** | 14.000 | 0.94264 | 0.94450 | **$+26$** | 0.94807 | **$+76$** | 0.94154 | 0.16622 |
| **4** | 13.998 | 0.94256 | 0.94335 | **$+11$** | 0.94706 | **$+63$** | 0.93698 | 0.17417 |
| **5** | 14.002 | 0.94237 | 0.94365 | **$+18$** | 0.94708 | **$+66$** | 0.93707 | 0.16998 |
| **Global** | **70.000** | **0.94230** | **0.94371** | **$+99$** | **0.94717** | **$+341$** | **`0.93734`** | **`0.17289`** |

> [!NOTE]
> Las hipótesis $w=0.10$ y $w=0.35$ superan al baseline en **el 100% de los pliegues independientes** (5 de 5), descartando que la ganancia provenga de una anomalía en una partición aislada.

### B. Desglose por Buckets de Frecuencia de Vendedor

| Bucket de Frecuencia | Población ($N$) | % Dataset | Baseline Acc | $w=0.10$ Acc | $w=0.10$ Net | $w=0.35$ Acc | $w=0.35$ Net |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 ítem (Vendedores Aislados)** | 19.926 | 28.5% | 0.89767 | 0.89998 | **$+46$** | 0.90675 | **$+181$** |
| **2 a 5 ítems (Pequeños)** | 16.598 | 23.7% | 0.94632 | 0.94740 | **$+18$** | 0.95120 | **$+81$** |
| **6 a 20 ítems (Medianos)** | 14.834 | 21.2% | 0.96131 | 0.96259 | **$+19$** | 0.96447 | **$+47$** |
| **>20 ítems (Grandes)** | 18.642 | 26.6% | 0.97130 | 0.97216 | **$+16$** | 0.97302 | **$+32$** |

*Origen de la ganancia:* La mayor proporción de aciertos rescatados ($+181$ de $+341$ en $w=0.35$) se concentra en los vendedores con 1 solo ítem, donde las señales intrínsecas (texto y árboles) compensan la ausencia total de historia del vendedor en el baseline.

### C. Desglose Taxonómico: Top 20 Categorías vs Long Tail

| Segmento Taxonómico | Población ($N$) | % Dataset | Baseline Acc | $w=0.10$ Net | $w=0.35$ Net |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top 20 Categorías de Alto Volumen** | 10.350 | 14.8% | 0.95536 | $+8$ | $+21$ |
| **Long Tail (723 Categorías Restantes)** | 59.650 | 85.2% | 0.94003 | **$+91$** | **$+320$** |

*Hallazgo:* El $93.8\%$ de los fixes netos ($+320$ aciertos) proviene de la cola larga de 723 categorías, demostrando la alta capacidad de generalización léxica y tabular del ensamble.

---

## 5. Auditoría de Conjuntos de Desacuerdo (Discordance Sets)

| Modelo / Ensamble | Tamaño Desacuerdo | % del Total | Fixes (+) | Breaks (-) | Ganancia Neta | Precisión de Giro | SHA-256 Vector de IDs en Desacuerdo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CatBoost D8** | 10.116 | 14.45% | 2.274 | 7.842 | -5.568 | 22.48% | `35171449cc4bbe393845b46ca4f7a22e85a6a378d3840778cbe4eec74b88fc7c` |
| **LightGBM 127L** | 9.852 | 14.07% | 2.269 | 7.583 | -5.314 | 23.03% | `7e8bf5cb5865a3684a2cbaf6b1e6e06b9b1d283626fa50529dcf4e122b10a26e` |
| **XGBoost Lossguide** | 9.887 | 14.12% | 2.270 | 7.617 | -5.347 | 22.96% | `8b7a87126ea2e1fdfe58cfef6df8ee87cb5b26335165c3bbec6696b92a2a0ee6` |
| **Text SAGA** | 10.290 | 14.70% | 721 | 9.569 | -8.848 | 7.01% | `772f42a999ce2364c39dc53e8d9753e8d8d7b7adadcb22bc8c962b9a1e0b57e7` |
| **Blend $w=0.10$** | **267** | **0.38%** | **183** | **84** | **$+99$** | **`68.54%`** | `2ab1d88235e01bdc75338ec7f9a263c97db0e26b177813a48e787dff9e0f10c6` |
| **Blend $w=0.35$** | **1.143** | **1.63%** | **742** | **401** | **$+341$** | **`64.92%`** | `887e92263c8474ff1d0a514d7a8e2da759530febeec17698059e0a0dffbb3f72` |

### Matriz de Similitud Jaccard de Desacuerdos entre Modelos:
- LightGBM vs XGBoost: **`0.7645`** (alta correlación entre familias de árboles).
- CatBoost vs LightGBM: **`0.6965`**.
- Árboles vs Text SAGA: **`0.1651` a `0.1779`** (ortogonalidad confirmada entre la señal densa SVD y la señal sparse n-gramas).
- Blend $w=0.10$ vs Blend $w=0.35$: **`0.2336`** (el conjunto de giros de $w=0.10$ está contenido casi en su totalidad dentro de $w=0.35$).

---

## 6. Dictamen Conformal Prequential en Muestra Finita

Al ejecutar [`scripts/public_score_conformal_gate.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/public_score_conformal_gate.py):
- Se evaluaron $m=2$ hipótesis simultáneas ($w=0.10$ y $w=0.35$) con cobertura individual $1 - \frac{0.05}{2} = 97.5\%$.
- Eventos prequential comparables verificados: $n = 10$.
- Eventos mínimos requeridos para cota al 95%: $n \ge \lceil 1 / 0.025 \rceil - 1 = \mathbf{39 \text{ eventos}}$ ($n \ge 19$ para un solo candidato).
- **Dictamen:** `status: NO_QUALIFIED` (código de salida `2`).
- **Consecuencia Directa:** Ningún modelo pasa a estado `PUBLIC_QUALIFIED`. Cero submissions enviadas a Kaggle.

---

## 7. Próximos Experimentos Ordenados por Valor Esperado y Costo Computacional

Para avanzar hacia la superación de `0.95000` respetando el régimen nested canónico:

1. **Shrinkage Empírico de Bayes Jerárquico por Categoría (P0 — Costo Bajo, ~10 min):**
   - Regularizar las predicciones dudosas ($0.40 < p < 0.60$) contrayendo la probabilidad hacia el prior empírico de la hoja taxonómica y de la categoría padre.
2. **Subword Character Embeddings FastText (P1 — Costo Medio, ~25 min):**
   - Incorporar representaciones de sub-palabras de 128 dimensiones para capturar patrones morfológicos en títulos abreviados sin inflar el vocabulario léxico sparse.
3. **Calibración Isotónica Cruzada (Cross-Calibrated Isotonic Regression) (P1 — Costo Bajo, ~5 min):**
   - Calibrar monótonamente las probabilidades OOF dentro de los inner pliegues para corregir desviaciones de polaridad en rangos intermedios.
4. **Acumulación de Evidencia Prequential (P2 — Costo Nulo de Cómputo, Requiere Ventana Kaggle Futura):**
   - Congelar formalmente las predicciones antes de cualquier apertura futura de cuota para acumular eventos $n$ válidos hasta alcanzar la masa crítica conformal ($n \ge 19$).

---

## 8. Manifiesto de Artefactos Producidos e Integridad Criptográfica

| Archivo / Artefacto | Tamaño | SHA-256 |
| :--- | :---: | :--- |
| `scripts/build_canonical_nested_splits.py` | 7.9 KB | `2f98e6ad799fcfbdad4bf3427ec565ee5da232709ff33a17e0e7a2cf1620a2e3` |
| `scripts/train_canonical_nested_dual_regime_models.py` | 27.2 KB | `0be116dc2cbb29c0f86a2f8b5a153202eb868b209d73fc1a9f99d9be877477ea` |
| `scripts/audit_canonical_breakdowns.py` | 8.8 KB | `a492f231e5f385c9772db1dd7dc85d263a2a0eb52317dc00609b55231cb6336e` |
| `data/splits/canonical_nested_dual_regime_splits.csv` | 3.5 MB | `91ee8fd2dc9e16ac52c5df19286e71d9bbfae9cede4a6394d342e698ed76c7fb` |
| `results/canonical_nested_models_20261110/oof_canonical_nested_dual_regime.csv` | 8.9 MB | `ff120cf8672202620725608850d6419292f2f5b2c407e753f5e73159ada2a7b2` |
| `results/canonical_nested_models_20261110/seen_val_honest_oof.csv` | 2.9 MB | `e1b561071f657d64a0be25de6b86e20bb6f570eeea8bcb325e78ca1b6d0b3eb5` |
| `results/canonical_nested_models_20261110/test_probabilities_canonical_nested.csv` | 3.7 MB | `c1696137c226bf9d6dff4707b69f06b3bdeb22f472c8e7cdeb95d5facf7c4cb8` |
| `results/canonical_nested_models_20261110/paired_canonical_evaluation_report.json` | 5.7 KB | `95c1438e0117f786a9ca4fc0473e4a28d178265e81954138118e1657c74dae3d` |
| `results/canonical_nested_models_20261110/canonical_breakdown_audit_report.json` | 6.8 KB | `8270518040182ea4628a2b0a52137a294bfc745237d77ac62967fe46be2b6060` |
| `results/conformal_audit_20261110/public_score_conformal_gate_report.json` | 2.4 KB | `05077ea418293c683b58c734b46c927f12e8417c805ba0a2491a13eeb4a5e3c8` |
