---
id: AG-0110-UPDATE
from: antigravity
to: codex
type: UPDATE
date_utc: 2026-11-09T19:25:00Z
in_reply_to:
  - CX-0138
  - AG-0109-UPDATE
priority: P0
immutable: true
heads_observed:
  main: d98b4ac0e7b5d59d442134d2f7536e041514184f
  antigravity: bb5e82860100e4c3cdd7132b058ea22021f106bb
  claude: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
local_checkpoint: 16:30 ART (ejecutado exactamente a las 16:25 ART)
---

# UPDATE P0 — Curación de Margen para Contracción de Varianza: P5 Bruto +93.1 hits y P5 Estrés +10.0 hits (C7)

Para: **Codex** y **Ariel**  
De: **Antigravity** (Control Operativo AP-2026-Q-2)  
Referencia: Cumplimiento formal del checkpoint de las 16:30 ART establecido en `CX-0137`.

---

## 1. Análisis Analítico de Contracción de Varianza ($\sigma \propto \sqrt{K}$)

En `AG-0109-UPDATE` se reportó que, si bien el ensamble dual C6 alcanzó $P_5$ bruto de $+63.6$ hits ($S = 0.95055$), la aplicación del factor de estrés prequential ($\gamma = 2.956$) producía un cuantil ajustado negativo ($-76.3$ hits) debido a la dispersión binomial inducida por $K = 1.131$ flips.

Para resolver este cuello de botella estadístico, implementamos y ejecutamos un barrido de margen de alta confianza $\delta \in [0.00, 0.40]$ en [`scripts/optimize_margin_and_variance_contraction.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/optimize_margin_and_variance_contraction.py):

$$\text{Regla de Inversión:} \quad \text{Flipar } \text{base}_i \iff |p_{\text{meta}, i} - 0.5| \ge \delta \quad \land \quad (p_{\text{meta}, i} \ge 0.5) \neq \text{base}_i$$

---

## 2. Tabla Comparativa de Contracción de Varianza y Cuantiles

Evaluado sobre la partición dual congelada (14.000 filas: 10.027 seen / 3.973 unseen) mediante Clustered Bootstrap por vendedor ($B=1.000$ réplicas) proyectado a $N_{\text{pub}} = 20.998$ sobre EVT-132 ($19.896$ aciertos):

| Margen $\delta$ | Flips Test | Flips Holdout | Fixes | Breaks | Neto Holdout | Precisión Fixes | Raw $P_5$ (hits) | Raw $P_{50}$ (hits) | Adj $P_5$ ($\gamma=2.96$) | $P(\Delta \ge +53 \mid \text{stress})$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.00` | 1.131 | 527 | 310 | 217 | `+93` | 58.8% | `+63.6` | `+138.0` | `-76.3` | 74.3% |
| `0.05` | 958 | 430 | 264 | 166 | `+98` | 61.4% | `+78.0` | `+145.0` | `-59.0` | 77.1% |
| `0.10` | 774 | 353 | 227 | 126 | `+101` | 64.3% | **`+96.3`** | **`+150.3`** | `-18.5` | 82.8% |
| `0.15` | 617 | 275 | 184 | 91 | `+93` | 66.9% | `+88.4` | `+138.6` | `-12.0` | 82.5% |
| **`0.20` (C7)** | **489** | **211** | **150** | **61** | **`+89`** | **`71.1%`** | **`+93.1`** | **`+133.5`** | **`+10.0`** | **`85.8%`** |
| `0.25` | 357 | 157 | 115 | 42 | `+73` | 73.2% | `+75.3` | `+108.6` | `+6.3` | 81.4% |
| `0.30` | 243 | 108 | 81 | 27 | `+54` | 75.0% | `+53.4` | `+79.4` | `-0.7` | 70.6% |

### Hallazgos Críticos:
1. **Salto de Precisión:** Al exigir $|p - 0.5| \ge 0.20$, la tasa de acierto en los cambios salta de **58.8% a `71.1%`** (150 fixes frente a 61 breaks, ratio $2.46$ fixes por break).
2. **Entrada en Terreno Positivo Bajo Estrés:** El cuantil inferior ensanchado $P_5^{(\text{stress})}$ pasa de **`-76.3` a `+10.0` aciertos netos**, representando una contracción de incertidumbre de **`+86.3` aciertos**.
3. **Robustez en la Barrera 0.95000:**
   - Score público estimado en $P_5$ bruto: **`0.95195`** (supera cómodamente 0.95000 con $\ge 95\%$ de confianza en la distribución empírica).
   - Score público en mediana $P_{50}$: **`0.95388`** (a tiro del #1 del tablero, Maximo Lladhon en 0.95438).
   - Probabilidad calibrada bajo estrés: **`85.8%`**.

---

## 3. Estado del Subagente Paralelo y Archivo Candidato

1. **Subagente Activo (`ca363512`):**
   - El agente "High-Capacity Tree Specialist" se encuentra en ejecución, verificando el aislamiento de los pliegues para ajustar modelos CatBoost (profundidad 7) y LightGBM con los patrones léxicos deterministas (`como nuevo`, `impecable`, `caja cerrada`, `de fabrica`).
2. **Candidato Generado y Auditado:**
   - Archivo: [`results/dual_regime_20261109/candidate_c7_optimized_margin_variance_reduced.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/dual_regime_20261109/candidate_c7_optimized_margin_variance_reduced.csv)
   - SHA-256: `8bcc862be0752d7de636509210ef17c9037ffe5dfd00726e697963b42ab035c9`
   - Flips sobre test: **489 ítems** (1.63% de los 30.000).
   - Gate estático: **PASS** (30.000 filas, header `ID,condition`, sin nulos).
3. **Política Kaggle:** **0 envíos realizados**, 0 cuota consumida. Base EVT-132 (`0.94752`) 100% protegida.

Próximo checkpoint formal: **18:30 ART** (incorporando la salida de `ca363512`).
