# AG-0218-RESULT: REPRODUCCIÓN HISTORIAL TEMPORAL WARM, EVALUACIÓN POR 5 REGÍMENES Y CERO ARRASTRE COLD

**Fecha:** 2026-11-14  
**Emisor:** Antigravity  
**Destinatario:** Codex / Ariel / Claude  
**Branch:** `antigravity/overnight-recovery-coordination-20261030`  
**Conformidad:** Gate H (Resumen Negativo Primero), Gate B/E (Cortes Train-Only y Bootstrap Multiway), Gate C (Cero Cuota / P=NO_IDENTIFIABLE).

---

## 0. Resumen negativo primero

1. **Degradación Severa y Estadísticamente Significativa de Batch en Vendedores Nuevos (`new_seller`):**
   - En el corte de validación interna ($N=13.574$ registros cold), el modelo Batch respecto al Baseline Fuerte arroja un saldo de **$-32$ aciertos netos** (149 fixes vs 181 breaks), con un deterioro de LogLoss de $0.32504 \to 0.32897$.
   - El bootstrap multiway por vendedor $\times$ categoría ($B=2.000$) arroja un percentil 95 estrictamente negativo: **$Q_{95} = -4.0 < 0$** (Mediana $-32.0$, $Q_{05} = -62.0$, $P(\Delta > 0) = 3.1\%$).
   - Conclusión metodológica: la señal batch de vendedor es tóxica si se aplica de forma homogénea a publicaciones de vendedores nuevos; requiere routing condicional estricto.

2. **Discrepancia de Integridad Criptográfica del Runner Local de CX-0200:**
   - Hash reportado en `results/codex_refit_20261111/late_ap_nov13_report.json`: `37d5f87c8801d0f50fe737c355c709e992769493ebfeefbe8c3a164b638927a4`.
   - Hash del runner físico en disco (`results/codex_refit_20261111/late_batch_seller.py` en git `780a375`): `d4359a1b149bf0cb92a2a0a38dfc6fcfb9875fec6ec26f1fc98a72b2c8c4c794`.
   - En cumplimiento de directivas de gobernanza, el resultado $+46 / -22$ fue clasificado como hipótesis no verificada hasta su reproducción en código y particiones nativas limpias.

3. **Inviabilidad Física de Umbrales Agresivos ($\tau \ge 0.90$) para Alcanzar $+50$ Públicos:**
   - La partición pública contiene $\approx 30\%$ del test ($\approx 9.000$ IDs). Para lograr $+50$ aciertos netos se necesitan al menos 50 flips en público.
   - En $\tau = 0.90$ solo se generan 140 flips warm en test ($\approx 42.0$ flips estimados en público): **soporte físicamente insuficiente**.
   - En $\tau = 0.92$ (101 flips, $\approx 30.3$ en público) y $\tau = 0.95$ (53 flips, $\approx 15.9$ en público), la ganancia máxima teórica con precisión perfecta es inferior a $+30$ y $+16$ respectivamente.

4. **Declaración Obligatoria Gate C (Incertidumbre Pública):**
   - La probabilidad incondicional de superar a S4 por $\ge +50$ aciertos públicos y el cuantil público $Q_{05}$ son formalmente **`NO_IDENTIFIABLE`** al carecer de calibración prequential secuencial sobre el split no observado.

---

## 1. Auditoría y Reproducción de la Hipótesis CX-0200

Se analizó la hipótesis empírica formulada en CX-0200:
- Vistos: $N=13.102$, Fixes=$129$, Breaks=$83 \implies \mathbf{+46}$ netos.
- Nuevos: $N=12.844$, Fixes=$117$, Breaks=$139 \implies \mathbf{-22}$ netos.
- Total: $N=25.946 \implies \mathbf{+24}$ netos.

**Diagnóstico Estructural:** La validación cruzada por grupos de vendedor (`GroupKFold` sobre `seller_id`) impone $0\%$ de vendedores vistos en validación. Si bien evalúa la generalización cold, oculta totalmente la capacidad predictiva intra-vendedor, que en el conjunto de test real abarca **21.487 publicaciones (71.6% del test)**. Para solventar este desajuste sin fugas hacia la evaluación final, se implementó un diseño de partición temporal train-only con control de los cinco regímenes.

---

## 2. Diseño de Cortes Train-Only Temporales (Warm Real vs Cold Disjoint)

Se trabajó exclusivamente sobre los 55.925 registros de desarrollo (`role != 'confirm'`). El conjunto `confirm` (14.075 filas) se mantuvo **estrictamente congelado y virgen**.

- **Inner Fit:** 30.219 registros (54.0% de dev).
- **Inner Validation:** 25.706 registros (46.0% de dev), particionados en:
  - `new_seller (Cold Disjoint)`: 13.574 registros (52.8%)
  - `bracketed (Warm Interleaved)`: 8.008 registros (31.2%)
  - `after_train (Warm Causal Posterior)`: 3.387 registros (13.2%)
  - `before_train (Warm Anterior)`: 456 registros (1.8%)
  - `same_timestamp (Warm Empate)`: 281 registros (1.1%)
  - **Subtotal Warm en Validación:** 12.132 registros (47.2%).

---

## 3. Resultados Experimentales Pareados por Régimen y Bootstrap Multiway

Se compararon frente al **Baseline Fuerte** (Texto OOF subwords + SVD 16 + Atributos + Causal Seller TE):
1. **Batch Model (Política B):** Incorpora historial batch intra-vendedor e interacción vendedor $\times$ categoría.
2. **Regime-Aware Routed Model:** Preserva el Baseline Fuerte en `new_seller (Cold)` y activa el Modelo Batch en los 4 regímenes Warm.
3. **Causal Model (Política A):** Incorpora encoding causal jerárquico de categoría ($t < t_i$).

### Tabla de Métricas Pareadas en Validación ($N=25.706$)

| Régimen en Validación | $N$ | Modelo | Fixes | Breaks | Net $\Delta$ | LogLoss (Cand vs Base) | Boot Multiway $Q_{05}$ | Mediana | $Q_{95}$ | $P(\Delta > 0)$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`new_seller (Cold)`** | 13.574 | Batch (Política B) | 149 | 181 | **$-32$** | 0.3290 vs 0.3250 | $-62.0$ | $-32.0$ | **$-4.0$** | 3.1% |
| | | **Regime-Aware Routed** | 0 | 0 | **$\pm 0$** | 0.3250 vs 0.3250 | $\pm 0.0$ | $\pm 0.0$ | $\pm 0.0$ | - |
| | | Causal (Política A) | 311 | 246 | $+65$ | 0.3187 vs 0.3250 | $+24.0$ | $+66.0$ | $+110.0$ | 99.8% |
| **`bracketed (Warm)`** | 8.008 | **Batch / Routed** | 119 | 78 | **$+41$** | 0.1605 vs 0.1771 | $+10.0$ | $+42.0$ | $+72.0$ | 98.5% |
| **`same_timestamp (Warm)`** | 281 | **Batch / Routed** | 29 | 10 | **$+19$** | 0.2776 vs 0.3548 | $+8.0$ | $+20.0$ | $+30.0$ | 100.0% |
| **`before_train (Warm)`** | 456 | **Batch / Routed** | 27 | 9 | **$+18$** | 0.1891 vs 0.2668 | $+8.0$ | $+18.0$ | $+28.0$ | 99.9% |
| **`after_train (Warm)`** | 3.387 | **Batch / Routed** | 30 | 21 | **$+9$** | 0.2259 vs 0.2307 | $-2.0$ | $+8.0$ | $+20.0$ | 93.6% |
| **Subtotal All Warm** | 12.132 | **Batch / Routed** | 205 | 118 | **$+87$** | 0.1826 vs 0.1995 | **$+52.0$** | **$+86.0$** | **$+124.0$** | **100.0%** |
| **Total Global** | 25.706 | Batch (Política B) | 354 | 299 | $+55$ | 0.2599 vs 0.2658 | $+6.0$ | $+56.0$ | $+102.0$ | 98.4% |
| | | **Regime-Aware Routed** | 205 | 118 | **$+87$** | 0.2578 vs 0.2658 | **$+50.0$** | **$+86.0$** | **$+124.0$** | **100.0%** |

### Conclusiones de la Evaluación:
- **Confirmación de la Hipótesis:** La señal batch de vendedor es consistentemente ganadora en publicaciones de vendedores vistos ($+87$ net fixes sobre $12.132$ filas warm, con $Q_{05} = +52.0$), pero consistentemente dañina en vendedores fríos ($-32$ net fixes, $Q_{95} = -4.0$).
- **Eficacia del Routing:** Al conmutar a `Regime-Aware Routed`, la pérdida en frío se extirpa por completo ($\Delta = 0$), reteniendo el $100\%$ de la ganancia warm ($+87$ net fixes, $Q_{05} = +50.0$).

---

## 4. Soporte de Flips en Test y Precisión Requerida para $+50$

El conjunto de test oficial consta de 30.000 IDs:
- **`new_seller (Cold)`:** 8.513 IDs (28.4%).
- **Regímenes Warm:** 21.487 IDs (71.6%):
  - `bracketed`: 10.140 IDs
  - `same_timestamp`: 5.108 IDs
  - `before_train`: 3.118 IDs
  - `after_train`: 3.121 IDs

Para garantizar **cero arrastre de la pérdida cold**, todo overlay warm construido sobre **S4** (`SUBMISSION_4_TODAY_RECOMMENDED.csv`) impone **flips cold = 0** de forma determinista:

$$\text{Overlay}_{\tau}(i) = \begin{cases} \text{S4}(i) & \text{si } i \in \text{new\_seller (Cold)} \\ \text{used} & \text{si } i \in \text{Warm} \land P(\text{batch}) \ge \tau \\ \text{new} & \text{si } i \in \text{Warm} \land P(\text{batch}) \le 1 - \tau \\ \text{S4}(i) & \text{en otro caso} \end{cases}$$

### Desglose de Flips vs S4 en Test Set (30.000 IDs)

$$\text{Condición para Ganancia Pública } \ge +50: \quad k_{\text{pub}} \cdot (2p - 1) \ge 50 \implies p \ge 0.5 + \frac{25}{k_{\text{pub}}}, \quad k_{\text{pub}} \approx 0.30 \cdot k_{\text{warm}}$$

| Umbral $\tau$ | Flips Warm | Flips Cold | Flips $N \to U$ | Flips $U \to N$ | `bracketed` | `same_time` | `before` | `after` | $k_{\text{pub}}$ Estimado | Precisión $p$ Necesaria para $+50$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | **921** | 0 | 594 | 327 | 374 | 148 | 196 | 203 | 276.3 | **59.0%** |
| **0.60** | **676** | 0 | 446 | 230 | 278 | 108 | 145 | 145 | 202.8 | **62.3%** |
| **0.70** | **484** | 0 | 311 | 173 | 192 | 79 | 107 | 106 | 145.2 | **67.2%** |
| **0.75** | **392** | 0 | 255 | 137 | 158 | 65 | 87 | 82 | 117.6 | **71.3%** |
| **0.80** | **304** | 0 | 192 | 112 | 128 | 52 | 61 | 63 | 91.2 | **77.4%** |
| **0.85** | **225** | 0 | 130 | 95 | 93 | 36 | 49 | 47 | 67.5 | **87.0%** |
| **0.90** | **140** | 0 | 63 | 77 | 64 | 19 | 33 | 24 | 42.0 | *Insuficiente ($<50$)* |
| **0.92** | **101** | 0 | 42 | 59 | 45 | 13 | 23 | 20 | 30.3 | *Insuficiente ($<50$)* |
| **0.95** | **53** | 0 | 15 | 38 | 25 | 7 | 13 | 8 | 15.9 | *Insuficiente ($<50$)* |

---

## 5. División de Roles y No Duplicación con Claude

- **Claude:** Asume el modelado exclusivo de la señal **cold (`new_seller`)**, explorando arquitecturas textuales profundas, embeddings y regularización transductiva.
- **Antigravity:** Asume con exclusividad la señal de **historial temporal intra-vendedor en regímenes cálidos**.
- **Cero Duplicación:** Antigravity no reentrenará ni generará variantes sobre la partición cold, preservando intactas las predicciones cold de S4 o del consenso oficial. Cero llamadas a API de Kaggle.

---

## 6. Inventario de Hashes Inmutables y Artefactos

- **Script Runner:** `scripts/train_warm_temporal_history_and_regimes.py`
- **Reporte de Validación JSON:** `results/warm_history_temporal_validation_report.json`  
  SHA-256: `5e8dc97eb4b4e7fe8ecf635ff20ddc9bf10e4708764041d8e137184a56a6549c`
- **Archivo de Pares Privados:** `results/warm_history_pairs_private.csv`  
  SHA-256: `096a43c1da03dfaafb49cd915c6552088dd580a63ffa2be7984fe5d3c91d5165`

### Hashes Canónicos ID/Target (30.000 IDs de Test)
- **Ancla S4 (`condition_s4`):**  
  `c0497e08e453037fa88f96efcc89771c73b820f76b345f1a9109364828dd93dd` (Flips: 0)
- **Overlay Warm $\tau = 0.80$ (`condition_overlay_tau80`):**  
  `d89f6d2d03dcb4556f5dc438b1d95d83db05cfe7fd2b00780d60cb385ff4f9fe` (Flips: 304, Cold Flips: 0)
- **Overlay Warm $\tau = 0.70$ (`condition_overlay_tau70`):**  
  `fb436a5c6df905f8e1ff5e0943000affcfbc97c851f3ae5f77a1c6b54be274ae` (Flips: 484, Cold Flips: 0)
- **Overlay Warm $\tau = 0.85$ (`condition_overlay_tau85`):**  
  `52188e018611b186e6f53777cc7509fcd8d53b9bbb7a3ca46c080a39389f9433` (Flips: 225, Cold Flips: 0)
