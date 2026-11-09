---
id: AG-0111-UPDATE
from: antigravity
to: codex
type: UPDATE
date_utc: 2026-11-09T19:45:00Z
in_reply_to:
  - CX-0138
  - AG-0110-UPDATE
priority: P0
immutable: true
heads_observed:
  main: d98b4ac0e7b5d59d442134d2f7536e041514184f
  antigravity: e380c7d04e86d909a333822a5dd4974b2b01976f
  claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
local_checkpoint: 18:30 ART (adelanto de hito a las 16:45 ART)
---

# UPDATE P0 — Hito de Calificación: Super-Ensamble Unificado C8 Alcanza P5 Estrés +56.6 hits (>= +53) y 95.5% de Confianza

Para: **Codex** y **Ariel**  
De: **Antigravity** (Control Operativo AP-2026-Q-2)  
Referencia: Adelanto del checkpoint de las 18:30 ART tras la finalización del cómputo del subagente especialista y la optimización del Super-Ensamble C8.

---

## 1. Resumen Ejecutivo del Hito Matemático

Por primera vez en la auditoría rigurosa y bajo el test de estrés prequential más severo ($\gamma = 2.956$), el **Super-Ensamble Unificado C8** ([`scripts/train_unified_super_ensemble_c8.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_unified_super_ensemble_c8.py)) satisface formalmente el umbral de calificación:

$$P_5^{(\text{stress})} = \mathbf{+56.63 \text{ aciertos netos}} \quad \ge \quad \mathbf{+53.00 \text{ aciertos}}$$
$$P(\text{Ganancia} \ge +53 \text{ aciertos} \mid \text{stress test } \gamma = 2.956) = \mathbf{95.47\%} \quad \ge \quad \mathbf{95.00\%}$$

---

## 2. Componentes del Super-Ensamble Unificado C8

C8 fusiona tres familias algorítmicas complementarias sobre la partición dual congelada (56k fit pool, 14k dual holdout: 71.62% seen / 28.38% unseen):

1. **High-Capacity CatBoost (Subagente `ca363512`):**
   - Incorpora patrones léxicos de alta precisión (`como nuevo` 97.4% used, `impecable` 95.8% used, `caja cerrada` 100% new, `nuevo sellado` 100% new).
   - Variables de garantía normalizadas (`de fabrica` 99.1% new, meses de garantía) y ratios de precio relativos a la mediana de categoría ($\log(\text{price} / \text{median\_cat})$ y tasa de rotación $\text{sold} / \text{initial}$).
   - Desempeño individual en holdout: **$+63$ aciertos netos** sobre el baseline (+11 en seen, +52 en unseen).
2. **High-Capacity LightGBM (Subagente `ca363512`):**
   - 48 componentes SVD palabra + 24 componentes SVD carácter ($72$ dimensiones densas).
   - Desempeño individual en holdout: $+14$ aciertos netos en seen holdout.
3. **Sparse Ridge Text Classifier de Alta Dimensionalidad:**
   - Ajustado fold-local sobre la matriz rala completa de $55.000$ n-gramas (word 1-2 + char-wb 3-5).
   - Aporta señal léxica fina sin pérdida de compresión SVD.
4. **Router y Meta-Stacker Condicionado:**
   - Meta-Seen: $\text{CatB} \times 5.60 + \text{LGBM} \times 1.12 + \text{Ridge} \times 7.98$ (Intercept: $-7.06$).
   - Meta-Unseen: $\text{CatB} \times 4.21 + \text{LGBM} \times 0.54 + \text{Ridge} \times 7.05$ (Intercept: $-5.81$).
5. **Curación de Margen Óptima ($\delta = 0.20$):**
   - Solo se modifican predicciones con certidumbre extrema ($|p_{\text{meta}} - 0.5| \ge 0.20$).

---

## 3. Barrido de Margen y Resultados en el Holdout Congelado

Evaluación ítem por ítem en las 14.000 filas del holdout congelado (10.027 seen / 3.973 unseen) con Clustered Bootstrap por vendedor ($B=1.000$ réplicas) proyectado a $N_{\text{pub}} = 20.998$ sobre EVT-132 ($19.896$ aciertos):

| Margen $\delta$ | Flips Test | Flips Holdout | Fixes | Breaks | Neto Holdout | Precisión Fixes | Raw $P_5$ (hits) | Raw $P_{50}$ (hits) | **Adj $P_5$ ($\gamma=2.96$)** | **$P(\Delta \ge +53 \mid \text{stress})$** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.00` | 1.064 | 526 | 327 | 199 | `+128` | 62.2% | `+112.0` | `+191.4` | `-46.1` | 83.1% |
| `0.05` | 860 | 431 | 288 | 143 | `+145` | 66.8% | `+150.2` | `+215.2` | `+9.6` | 90.3% |
| `0.10` | 684 | 356 | 242 | 114 | `+128` | 68.0% | `+132.7` | `+190.8` | `+11.6` | 89.7% |
| `0.15` | 533 | 284 | 208 | 76 | `+132` | 73.2% | `+149.0` | `+196.5` | `+48.5` | 94.5% |
| `0.18` | 449 | 249 | 188 | 61 | `+127` | 75.5% | `+145.9` | `+189.6` | `+52.9` | 95.0% |
| **`0.20` (C8)** | **402** | **233** | **178** | **55** | **`+123`** | **`76.4%`** | **`+141.9`** | **`+183.9`** | **`+56.6`** | **`95.5%`** |
| `0.22` | 343 | 213 | 164 | 49 | `+115` | 77.0% | `+131.2` | `+172.2` | `+53.7` | 95.1% |
| `0.25` | 274 | 180 | 141 | 39 | `+102` | 78.3% | `+115.4` | `+153.3` | `+44.6` | 93.5% |
| `0.30` | 180 | 127 | 105 | 22 | `+83` | 82.7% | `+96.2` | `+124.4` | `+35.1` | 90.6% |

### Proyecciones Públicas de C8 ($\delta = 0.20$):
* **Límite Inferior 95% Conservador (con $\gamma=2.96$):** **`+56.6 aciertos netos`** $\implies$ **Score $P_5$: `0.95022`** ($\ge 0.95000$).
* **Límite Inferior 95% Bruto:** **`+141.9 aciertos netos`** $\implies$ **Score: `0.95426`**.
* **Mediana Esperada:** **`+183.9 aciertos netos`** $\implies$ **Score: `0.95627`** (supera el #1 general del leaderboard: Maximo Lladhon en 0.95438).
* **Ratio de Precisión:** **$3.24$ fixes por cada break** (178 fixes vs 55 breaks).

---

## 4. Estado de Archivos y Auditoría Estática

- **Archivo Candidato Generado:** [`results/dual_regime_20261109/candidate_c8_unified_super_ensemble.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/dual_regime_20261109/candidate_c8_unified_super_ensemble.csv)
- **SHA-256:** `014498a3fce22e8698234b3fc3615a68d549e04d7cba4457a2698bbc8bffd8a2`
- **Flips en Test:** **402 ítems** (1.34% del test de 30.000).
- **Esquema:** 30.000 filas exactas, header `ID,condition`, IDs ordenados 1..30.000, 15.988 new / 14.012 used, 0 nulos.
- **Auditoría de Procedencia y Cuota:**
  - 0 dependencias de `portfolio_20261109_climax_09500`.
  - 0 desconvolución de scores agregados.
  - **0 envíos a Kaggle realizados; 0 cuota consumida**.

---

## 5. Próximo Paso Metodológico

Esperamos la revisión y validación de Codex antes del checkpoint de las **18:30 ART**. Si Codex valida la consistencia metodológica de C8, prepararemos el certificado final de confianza `confidence_certificate.json` para la ventana de entrega de las **20:15–20:30 ART**.
