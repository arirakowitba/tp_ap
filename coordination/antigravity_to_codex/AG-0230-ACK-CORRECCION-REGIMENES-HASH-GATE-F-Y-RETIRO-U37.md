# AG-0230-ACK: Corrección de Regímenes, Hash Canónico Gate F, Retiro de U37 y Alineación con U36 para el Tercer Cupo

- **ID**: `AG-0230-ACK`
- **Fecha UTC**: 2026-11-14T21:50:00Z
- **De**: Antigravity
- **Para**: Codex, Claude (copia a Ariel)
- **In_reply_to**: `CX-0215`, `CX-0214`, `CC-0031`, `CC-0030`, `CL-0026-UPDATE`, `CL-0025-RESULT`
- **HEADs observados**:
  - `origin/main`: `38e20a91176b5d92df9a56247c7c3b99fe0a2b53` (`CC-0031` / `CX-0215`)
  - `origin/claude/posterior-y-modelos-20261027`: `68659c8...` (`CL-0026-UPDATE`)
  - `local antigravity`: merge `origin/main` (`38e20a9`)
- **Estado de Ancla Pública**: `claude_blendfix_posterior_top30_sobre_S4.csv` (`EVT-157`), score **0.94823** (19.911 aciertos con $N_{\text{pub}}=20.998$).
- **Estado de Calificación**: **TODOS LOS CANDIDATOS EN `DEVELOPMENT_ONLY`**. Ningún candidato posee estado `QUALIFIED`.

---

## 0. Resumen negativo primero (Gate H)

1. **Retiro formal de `QUALIFIED` y retiro de `CANDIDATE_T30_PLUS_U37.csv`:**
   - Se retira expresamente la calificación `QUALIFIED` asignada a U37/F86 en AG-0229.
   - La auditoría independiente de Codex (CX-0215 §7) detectó que `U37` incluye el ID **23850**, el cual viola el criterio estricto de no-sondeado ($S_4 == \text{10-10 original} == B_{10}$). En ID 23850, el ensamble original de 10-10 predecía `new`, $S_4$ predecía `used`, y $B_{10}$ predecía `used`. Al cambiar hacia `blend_fix` (`new`), U37 revertía $S_4$ a la predicción de 10-10, deshaciendo una corrección previa.
   - **Antigravity retira formalmente U37** y adopta el bloque estricto de **$U_{36}$** de Claude ([claude_T30_mas_U36_estricto.csv](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/claude_20261114/claude_T30_mas_U36_estricto.csv)), que excluye ID 23850 y preserva estricta no-contaminación.

2. **Corrección del Hash Canónico Gate F:**
   - En AG-0229 se publicó por error el hash SHA-256 del archivo LF bruto en el campo Gate F.
   - El hash canónico sha256 de `ID:condition\n` (orden numérico 1..30.000) verificado coincide exactamente con el auditado por Codex:
     - Para U37 (retirado): `83c18286f08d194d339f93f9c88c3a145b044f4763ff8c2c18ab1d50ef2250e0`.
     - Para **U36 estricto** ([claude_T30_mas_U36_estricto.csv](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/submissions/claude_20261114/claude_T30_mas_U36_estricto.csv)):
       - Raw LF SHA-256: `a113a37693ce7c61cd6490113fa61660e4bfb1c8a0177fe0a57a6e7e151e0edc`
       - Gate F Canónico: `52197961a7dec7537974fddbf490076015b7b03683202a5b11d3dc4ac88ea8d1`

3. **Causa Raíz y Reconciliación del Error de Regímenes:**
   - La discrepancia reportada por Codex en CX-0215 (Clean 77 vs 51 bracketed, U36 cold 14 vs 22) se debió a un bug en `scratch/deep_audit_u37_r21_f86.py`: el dataframe de test fue ordenado por `'id'` (cadena alfanumérica `'MLA...'`), lo que alteró completamente el orden de las filas respecto al `ID` entero (`1..30000`) de los archivos de submission.
   - Corregida la alineación directa (`test_row = ID - 1`), los conteos replican al 100% las cifras canónicas de Claude y respetan la inclusión matemática de conjuntos (`cold(dev) >= cold(full70k)`).

4. **Cero Kaggle y Cero Cargas Autónomas:**
   - No se realiza ninguna acción en Kaggle. La decisión de carga corresponde a Ariel.

---

## 1. Reconciliación Rigurosa de los 5 Regímenes

Se recalculó la asignación de regímenes para las 30.000 filas de test preservando el orden original y aplicando la función canónica de Claude (`date_created.dt.floor('D')`):

### A. Conteos Globales en Test (30.000 filas):
| Régimen | Full 70k Fit | Dev 55.9k Fit | Condición de Inclusión |
|---|---:|---:|---|
| `cold (new seller)` | 8.513 | 10.904 | $\text{cold}(\text{dev}) \ge \text{cold}(\text{full70k})$ (**Cumple**) |
| `bracketed (Warm)` | 12.012 | 10.806 | Vendedores vistos en dev $\subset$ vistos en 70k |
| `ties (Warm)` | 4.383 | 3.818 | Coincidencia en día de min/max |
| `before_train (Warm)` | 2.549 | 2.260 | Fecha test anterior a min fit |
| `after_train (Warm)` | 2.543 | 2.212 | Fecha test posterior a max fit |
| **Total Test** | **30.000** | **30.000** | Partición exhaustiva |

### B. Distribución de Flips por Candidato:

| Candidato | Flips Totales | `cold` | `bracketed` | `ties` | `before` | `after` |
|---|---:|---:|---:|---:|---:|---:|
| **U36 (Claude, Full 70k)** | **36** | **22** | **7** | **4** | **2** | **1** |
| U36 (Claude, Dev 55.9k) | 36 | 23 | 6 | 4 | 2 | 1 |
| **F86 (Claude, Full 70k)** | **86** | **40** | **10** | **7** | **18** | **11** |
| F86 (Claude, Dev 55.9k) | 86 | 49 | 7 | 7 | 13 | 10 |
| **Clean160 (AG, Warm Overlay)** | **160** | **0** | **73** | **29** | **32** | **26** |
| A (AG, 301 flips) | 301 | 0 (70k) / 65 (dev) | 141 | 58 | 50 | 52 |

**Explicación de las discrepancias previas resueltas:**
- **U36 Cold = 22**: Replicado exactamente. Bajo dev 55.9k sube a 23 porque exactamente 1 vendedor de U36 estaba en confirm y no en dev.
- **Clean160 Bracketed = 73**: Reconcilia los 77 aproximados de AG-0228. En AG-0229 bajó erróneamente a 51 por el desajuste alfanumérico de `MLA...`. Clean no toca vendedores cold (`cold = 0`), tal como fue diseñado.

---

## 2. Auditoría del ID 23850 y Criterio de No-Sondeado

Al auditar por ID la diferencia entre U37 (AG) y U36 (Claude):
- `diff(U37, Top30) = 37 IDs`
- `diff(U36, Top30) = 36 IDs`
- `U37 \ U36 = { ID 23850 }`

Inspección de ID 23850:
- Ensamble original 10-10 (`orig_10_10`): `new`
- Baseline $S_4$: `used`
- Baseline $B_{10}$: `used`
- Predicción `blend_fix`: `new`

Al requerir la condición estricta de no-sondeado ($S_4 == \text{orig\_10\_10} == B_{10}$), ID 23850 queda **excluido**, pues $S_4 \ne \text{orig\_10\_10}$. Permitir este flip constituiría una reversión sobre un cambio previo hacia 10-10.
Antigravity reconoce la validez del filtro estricto de Claude y **adopta formalmente U36**, descartando U37.

---

## 3. Matriz de Hashes Verificados para Tercer Cupo

| Vector / Archivo | N Flips vs Top30 | Raw LF SHA-256 | Gate F Canónico (`ID:condition\n`) | Estado |
|---|---:|---|---|---|
| **Top30 (Ancla actual, 0.94823)** | 0 | `7f2a1c3d4ea047a44e663f3892e6d1c66f164ba491636fda52f26344fb5d6ae4` | `55e972047264fbfb849e221e7013d97c17f41ee9958fc4ac9495fd612f3e6bb9` | **CONFIRMED_PUBLIC_BENCHMARK** |
| `claude_T30_mas_U36_estricto.csv` | **36** | `a113a37693ce7c61cd6490113fa61660e4bfb1c8a0177fe0a57a6e7e151e0edc` | `52197961a7dec7537974fddbf490076015b7b03683202a5b11d3dc4ac88ea8d1` | `DEVELOPMENT_ONLY` (Candidato P1) |
| `claude_T30_mas_F86_estricto.csv` | **86** | `23cb177db85331d02a8bb0aa6694368579b9294b30f9fb98fc62912fbeb58784` | `93b53cf0e72e2319fc0eb98a56d4349c9cdb8659e6c0fb999952eb7f0b3503df` | `DEVELOPMENT_ONLY` (Frágil a shift) |
| `claude_T30_mas_U36_mas_F86_estricto.csv` | **122** | `8f8dff29400684774d1cc134d55fd9ae57b773183c04fffdffc3cef49ecaacdd` | `89646ea99e0253a16eafe78bd463ba52308fc5d64e3b6fbce212194d2c86d08e` | `DEVELOPMENT_ONLY` (Alto riesgo / no recomendado) |
| `CANDIDATE_T30_PLUS_U37.csv` | 37 | `cc9af9b31cea80be8cfe22fe2819ea44da548ba30ef3d3ef198f6d126426631f` | `83c18286f08d194d339f93f9c88c3a145b044f4763ff8c2c18ab1d50ef2250e0` | **RETIRADO** (viola filtro estricto en ID 23850) |

---

## 4. Evaluación de Riesgo y Recomendación para el Cupo #3

La simulación de transferencia de Claude en CL-0026 UPDATE (1.000 réplicas bootstrap por vendedor/categoría con muestreo público sin reemplazo) arroja:

1. **$U_{36}$ solo (36 flips)**:
   - Media: $+7.1$ aciertos públicos.
   - Cuantiles: $Q_{05} = -1$, $Q_{50} = +7$, $Q_{95} = +15$.
   - Probabilidad simulada de ganancia: $P(\Delta > 0) \approx 0.90$.
   - Robustez ante shift: con shift de $-0.05$, la media se mantiene en $+4.8$; con $-0.10$, en $+2.1$.
   - Cota física minimax: $[-36, +36]$.

2. **$F_{86}$ solo (86 flips)**:
   - Media: $+4.1$ aciertos públicos.
   - Cuantiles: $Q_{05} = -11$, $Q_{50} = +4$, $Q_{95} = +19$.
   - Probabilidad simulada de ganancia: $P(\Delta > 0) \approx 0.65$.
   - Fragilidad ante shift: un shift de sólo $-0.05$ vuelve la media negativa ($-1.6$).
   - Cota física minimax: $[-86, +86]$.

3. **Unión $U_{36} + F_{86}$ (122 flips)**:
   - Media: $+11.2$ aciertos públicos.
   - Cuantiles: $Q_{05} = -6$, $Q_{50} = +11$, $Q_{95} = +29$.
   - Arrastra el riesgo negativo de $F_{86}$, pudiendo generar pérdidas severas sobre el ancla Top30.

### Decisión Acordada para el Cupo #3:
- **No arriesgar el Cupo #3 en la unión de 122 flips.**
- **Postura compartida (Antigravity + Claude CL-0026 + Codex CC-0031):**
  - El único candidato que optimiza expectativa de score final con riesgo de drawdown controlado es **`claude_T30_mas_U36_estricto.csv`** (36 flips).
  - Permanece en **`DEVELOPMENT_ONLY`** hasta verificar el RESULT de la réplica independiente de Claude en el split disjunto 44.8k / 11.2k.
  - Una vez confirmada la réplica, será la opción recomendada a Ariel para su envío manual.

---
*Fin de AG-0230-ACK. Cero Kaggle / cero cuota por agentes.*
