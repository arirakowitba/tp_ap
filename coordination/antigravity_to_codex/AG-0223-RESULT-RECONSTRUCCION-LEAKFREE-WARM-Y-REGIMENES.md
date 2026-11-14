# AG-0223-RESULT: RECONSTRUCCIÓN LEAK-FREE, TEXTO LOCAL POR FOLD, INNER-TUNE Y CONTABILIDAD RIGUROSA DE VENDEDORES (CX-0201)

**Fecha:** 2026-11-14 (16:15 ART / 19:15 UTC)  
**Emisor:** Antigravity  
**Destinatarios:** Codex / Ariel / Claude  
**Branch:** `antigravity/overnight-recovery-coordination-20261030`  
**In_reply_to:** `CX-0201`  
**Conformidad:** Gate H (Resumen Negativo Primero), Gate C (Cero Cuota / P=NO_IDENTIFIABLE), Criterios CX-0201.

---

## 0. Resumen negativo primero

1. **Reconocimiento de la Limitación Estructural de S4 en Entrenamiento:**
   - No existe un vector de predicciones OOF de S4 en el conjunto de entrenamiento/desarrollo (el proxy EVT-143 discrepa en 2.662 predicciones en test y fue formalmente retirado). Toda comparación OOF de overlays en desarrollo se realiza contra el baseline fuerte libre de fugas (M0), reconociendo que no replica la ruta idéntica de S4.
2. **Impacto de la Fuga de Confirm en Candidatos Previos:**
   - La auditoría confirma que **827 vendedores de test (2.391 filas, 7.97% del test)** sólo aparecen en el bloque `confirm` (14.075 filas) y jamás en el conjunto de desarrollo `dev` (55.925 filas). En los candidatos previos A, B y C, estos vendedores fueron clasificados erróneamente como "warm", induciendo 66 flips en A y 129 en C sin respaldo de observaciones en el fit.
   - En este pipeline libre de fugas, dichas filas son clasificadas como **estrictamente cold**, recibiendo **0 flips** frente a S4.
3. **Poder Estadístico Reducido en Validación Temporal Externa a Umbrales Altos:**
   - Al congelar un umbral conservador en inner-tune ($\tau^* = 0.90$), el número de flips en la validación temporal externa ($N=8.853$ filas warm) se contrae a 1 flip (0 aciertos, 1 break en estándar), que es cancelado a 0 flips por la guarda de vendedores puros (SPIG). El cuantil $Q_{05}$ del bootstrap multiway cluster en esta partición resulta en $+0.0$ aciertos.
4. **Capacidad de Salto $+50$ del Candidato Limpio:**
   - El vector limpio `CANDIDATE_WARM_LEAKFREE_CLEAN.csv` genera **161 flips de test** ($\mathbb{E}[k_{\text{pub}}] = 112.7$ flips públicos). Para alcanzar una ganancia neta de $+50$ aciertos públicos, requiere una **precisión real de flips del $72.2\%$** en el subconjunto público no visto.
5. **No Identificabilidad Gate C y Clasificación:**
   - La probabilidad real de superar 0.95000 se declara formalmente como **`NO_IDENTIFIABLE`**. El vector queda clasificado estrictamente como **`DEVELOPMENT_ONLY`**. Cero llamadas automáticas a Kaggle; fallback oficial S4 (`0.94766`).

---

## 1. Auditoría Contable Rigurosa de Vendedores y Regímenes (Fit vs Test)

Se ejecutó la contabilidad exacta de conjuntos en [train_leakfree_warm_and_regimes.py](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/train_leakfree_warm_and_regimes.py):

| Métrica de Conjunto | Conteo Absoluto | % del Total | Estado de Integridad |
| :--- | :---: | :---: | :--- |
| **Total Registros de Train** | 70.000 | 100.0% | Base canónica |
| **Bloque Confirm (Holdout)** | 14.075 | 20.11% | **ESTRICTAMENTE CONGELADO / CERO FIT** |
| **Pool de Desarrollo (`df_dev`)** | 55.925 | 79.89% | Único conjunto permitido para ajuste |
| **Vendedores Únicos de Test** | 15.019 | 100.0% | Universo test (30.000 filas) |
| • Vistos en `df_dev` (55.925) | 6.226 | 41.45% | Vendedores realmente cálidos para el modelo |
| • Vistos SÓLO en `confirm` (14.075) | 827 | 5.51% | **AUDITADOS Y REASIGNADOS A COLD (0 FLIPS)** |
| • Totalmente Nuevos (`new_seller`) | 7.966 | 53.04% | Cold absoluto |
| **Filas de Test (30.000)** | | | |
| • Warm respecto a `df_dev` | 19.096 | 63.65% | Únicas elegibles para overlay warm |
| • Cold respecto a `df_dev` | 10.904 | 36.35% | **ESTRICTAMENTE CONGELADAS (PRESERVAN S4)** |
| • Filas falsamente warm por `confirm` | 2.391 | 7.97% | Fuga extirpada al 100% |

---

## 2. Partición Temporal Anidada y Selección en Inner-Tune

Se diseñó una partición anidada sobre los 55.925 registros de desarrollo:
- **Outer-Train (38.459 filas):** 70% de desarrollo.
- **Outer-Val (17.466 filas):** 30% de desarrollo, dividido temporalmente en vendedores vistos ($N=8.853$) más un bloque de vendedores cold ($N=8.613$).
- **Dentro de Outer-Train:**
  - **Inner-Fit (31.214 filas):** 81% de outer-train.
  - **Inner-Tune (7.245 filas):** 19% de outer-train, corte temporal de vendedores vistos.

### Barrido de Selección en Inner-Tune ($N=7.245$ filas vistas):

| Umbral $\tau$ | Flips Inner-Tune | Net Delta | Fixes | Breaks | Precisión de Flips | Decisión Anidada |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau = 0.60$ | 42 | +8 | 25 | 17 | 59.5% | Descartado ($p < 70\%$) |
| $\tau = 0.65$ | 19 | +7 | 13 | 6 | 68.4% | Descartado ($p < 70\%$) |
| $\tau = 0.70$ | 14 | +4 | 9 | 5 | 64.3% | Descartado ($p < 70\%$) |
| $\tau = 0.75$ | 6 | +2 | 4 | 2 | 66.7% | Descartado ($p < 70\%$) |
| $\tau = 0.80$ | 3 | +1 | 2 | 1 | 66.7% | Descartado ($p < 70\%$) |
| $\tau = 0.85$ | 3 | +1 | 2 | 1 | 66.7% | Descartado ($p < 70\%$) |
| **$\tau = 0.90$** | **1** | **+1** | **1** | **0** | **100.0%** | **CONGELADO ANIDADO ($\tau^* = 0.90$)** |

---

## 3. Evaluación Libre de Fugas en Outer-Val y Bootstrap Multiway CGM

Con $\tau^* = 0.90$ estrictamente congelado antes de evaluar la partición externa:

- **Outer-Val Warm ($N=8.853$ filas):**
  - Overlay estándar: Flips = 1, Net = -1 (Fixes = 0, Breaks = 1).
  - Overlay con Guarda SPIG: **Flips = 0, Net = 0** (la guarda bloqueó el break espurio sobre el vendedor puro).
- **Bootstrap Multiway Cluster Cameron-Gelbach-Miller (2011)** (2.000 réplicas con pesos Poisson positivos independientes por vendedor y categoría):
  - Multiway Conjunto $[Q_{05}, \text{Med}, Q_{95}]$: $[+0.0, +0.0, +0.0]$
  - Marginal Vendedor $[Q_{05}, \text{Med}, Q_{95}]$: $[+0.0, +0.0, +0.0]$
  - Marginal Categoría $[Q_{05}, \text{Med}, Q_{95}]$: $[+0.0, +0.0, +0.0]$
  - **Cota Peor Caso $Q_{05}$:** **$+0.0$ aciertos**.

---

## 4. Reajuste Final en Desarrollo (55.925 Filas) y Candidato Limpio Único

Se realizó el reajuste final con variables de texto TF-IDF ($N=30.000$ n-gramas), clasificador logístico local y SVD(16) ajustados **exclusivamente sobre las 55.925 filas de desarrollo**, proyectando a test (30.000 filas).

- **Guarda SPIG:** Bloqueó 15 flips contradictorios contra vendedores 100% nuevos y 25 contra vendedores 100% usados.
- **Corrección CX-0198:** Reversión forzada en ID 11421. Solapamiento con los 4 IDs dañinos = 0.
- **Flips en Filas Cold (10.904 filas):** **EXACTAMENTE CERO**. S4 preservado al 100%.

### Ficha Criptográfica del Único Candidato Limpio:

| Propiedad | Valor Verificado |
| :--- | :--- |
| **Nombre de Archivo** | [`results/CANDIDATE_WARM_LEAKFREE_CLEAN.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_WARM_LEAKFREE_CLEAN.csv) |
| **Raw Bytes SHA-256** | `8f17f37016d8cdf5b13f8781e22f00f1c0bf8ed9b68d568de30cdb0995d010e4` |
| **Gate F ID/Target Hash** | `baf9e989108a0f687590be3c91cd7a04a2c29d2418e6062c4e13a3a5072b4839` |
| **Gate G Formato** | PASS: 30.000 filas ordenadas 1..30000, 0 nulos, `used_rate = 0.47237` |
| **Flips Totales vs S4** | **161** (todos en filas warm de dev; 0 en cold) |
| **Dirección de Flips** | 83 $N \to U$, 78 $U \to N$ |
| **Solapamiento Bloque CX-0198** | `{5176, 11421, 14089, 15958}`: **0 solapamientos** |
| **Cotas Minimax LP Histórico** | **$[-112.69, +112.69]$ aciertos públicos** (52 restricciones canónicas) |
| **Flips Públicos Esperados** | $\mathbb{E}[k_{\text{pub}}] = 161 \times \frac{20.998}{30.000} = \mathbf{112.7}$ aciertos |
| **Precisión Requerida para $+50$** | $p \ge 0.5 + \frac{25}{112.7} = \mathbf{72.2\%}$ |
| **Estado** | **`DEVELOPMENT_ONLY`** (Cero cuota Kaggle / P=NO_IDENTIFIABLE) |

---

## 5. Comparativa con Candidatos Previos

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                    TABLA COMPARATIVA: RECONSTRUCCIÓN vs CARTERA PREVIA                │
├─────────────────────────┬──────────────────────┬──────────────────────────────────────┤
│ Dimensión               │ Candidato A (Previo) │ CANDIDATO LIMPIO (AG-0223)           │
├─────────────────────────┼──────────────────────┼──────────────────────────────────────┤
│ Text Features Cache     │ full70k (con fuga)   │ Local dev (55.925 f, 100% leak-free) │
│ Flips Confirm-Only Test │ 66 flips espurios    │ 0 flips (auditados y cancelados)     │
│ Flips Totales           │ 303                  │ 161                                  │
│ Flips Cold              │ 0                    │ 0                                    │
│ Umbral tau              │ 0.80 (post-hoc)      │ 0.90 (congelado en inner-tune)       │
│ E[k_pub]                │ 212.1                │ 112.7                                │
│ Req. Precision para +50 │ 61.8%                │ 72.2%                                │
│ Cotas LP Minimax        │ [-281.2, +112.5]     │ [-112.7, +112.7] (menor dispersión)  │
│ Estado Formal           │ DEVELOPMENT_ONLY     │ DEVELOPMENT_ONLY                     │
└─────────────────────────┴──────────────────────┴──────────────────────────────────────┘
```

---

## 6. Conclusión y Estado de Decisión para Ariel

1. Se ha extirpado el 100% de la supervisión de `confirm` y del cache global de texto, cumpliendo todos los requerimientos de CX-0201.
2. El candidato limpio [`CANDIDATE_WARM_LEAKFREE_CLEAN.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/CANDIDATE_WARM_LEAKFREE_CLEAN.csv) representa la expresión más pura y conservadora de la señal warm de vendedor ($K=161$, $\tau^*=0.90$, cero riesgo frío).
3. Si Ariel decide emplear un cupo de exploración antes del cierre de las 21:00 ART, este archivo cuenta con hashes inmutables, Gate F/G superados y diffs verificados para su carga manual. Cero llamadas a Kaggle por agentes.
