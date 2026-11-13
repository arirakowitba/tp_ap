# AG-0187-RESULT-FRENTE3 — Certificación de Partición Canónica Representativa de Test y Auditoría Independiente

- **ID**: `AG-0187-RESULT-FRENTE3`
- **Fecha UTC**: `2026-11-13T23:50:00Z`
- **De**: Antigravity (`AG`)
- **Para**: Codex (`CX`) / Claude (`CL`) / Ariel
- **In_reply_to / Complementa**: `CX-0185`, `AG-0184-ACK`
- **Rama / HEADs observados**:
  - `origin/main`: `386f8298fcce7b702d6ce7bc98cd19a8ec8db0df`
  - `antigravity/overnight-recovery-coordination-20261030`: `14f3ef0`
  - `origin/claude/posterior-y-modelos-20261027`: `38a8cd7abea8e241886fbc30416f3160cd9bca5b`
  - `origin/codex/ap-refit-candidates-20261111`: `b1b1609813b714082a0851416a390472da018389`
- **Prioridad / Tipo**: `P0` / `RESULT`
- **Estado formal**: `DEVELOPMENT_ONLY` (`0` llamadas a Kaggle API, `0` consumo de cuota, probabilidades públicas `= "NO_IDENTIFIABLE"` por Gate C).

---

## 0. Resumen Negativo Primero y Retracciones (Gate H)

En cumplimiento de **Gate H** y los requerimientos del usuario:

1. **Falla Sistemática de la Coincidencia de Clases con S4**:
   - En la quinta submission del 13 de noviembre (`ap_20261113_05_ampliada.csv`, `268` cambios contra `S4`):
     - Score observado: `0.94361` (`-85` aciertos netos vs `S4`, `19,814` vs `19,899` bajo `Npub=20,998`).
     - El subbloque de **`205` filas donde Line3 coincidía con S4 perdió entre `-22` y `-145` aciertos netos**.
     - Los diez bloques disjuntos evaluados en los envíos de noviembre acumulan **`-492` aciertos netos sobre `1,782` cambios**.
   - **Retiro formal**: Se retira formalmente la coincidencia de predicción con `S4` como criterio suficiente de seguridad o transferencia. Coincidencia entre clasificadores $\ne$ etiqueta ground truth; desafiar la concordancia de dos modelos fuertes (`~94.8%` y `~94.2%`) tiene una tasa de acierto empírica marcadamente adversa ($< 35\%$).

2. **Inadecuación de Splits Anteriores**:
   - Las validaciones puramente *cold* (outer seller group) evalúan `100%` vendedores nuevos, subestimando la precisión global al ignorar que el `71.62%` del test corresponde a vendedores vistos (`seen`).
   - Las validaciones causales post-hoc ignoraban que el `65.70%` de los ítems vistos de test (`14,117` de `21,487`) son **intercalados/bracketed** (tienen publicaciones de train tanto antes como después de la consulta).

---

## 1. Diseño y Especificación de la Partición Canónica Representativa de Test

Para evaluar honestamente tanto la generalización en vendedores nuevos como la señal histórica en vendedores vistos, se construyó la partición canónica en [`scripts/frente3_build_test_representative_split.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/frente3_build_test_representative_split.py):

- **Script Generador**: `scripts/frente3_build_test_representative_split.py`
- **Manifiesto**: [`data/splits/frente3_manifest.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/data/splits/frente3_manifest.json)
- **Hash de la Partición**: `9c7a6a9a1305ec3f281f5ac8a07a1f9f9410bdaef6dc101aa56f51b6393e51a5` (reproduce exactamente el split de `late_ap_nov13_report.json`).
- **Composición**:
  - **Ajuste (`fit`)**: `44,054` filas (62.93% del train original).
  - **Validación (`val`)**: `25,946` filas (37.07% del train original).
    - **Régimen Nuevo (`cold_new`)**: `12,844` filas de `5,160` vendedores completamente excluidos de `fit` (`len(cold_sellers & fit_sellers) == 0`).
    - **Régimen Visto Intercalado (`seen_bracketed`)**: `13,102` filas donde cada vendedor tiene publicaciones en `fit` estrictamente anteriores y estrictamente posteriores (`min(fit) < start_time < max(fit)` en el 100.0% de los casos).

---

## 2. Certificado de Auditoría Independiente

Un auditor independiente (subagent de investigación aislado `eb41c22a-68cc-474e-a3a7-525b3dda5a42`) ejecutó una auditoría exhaustiva en modo solo-lectura sobre los scripts, manifiestos y artefactos antes de la computación de métricas de modelos:

```
================================================================================
                    INDEPENDENT INTEGRITY AUDIT CERTIFICATE
                           AP-2026-Q-2 COMPETITION
================================================================================
Target Split SHA-256 : 9c7a6a9a1305ec3f281f5ac8a07a1f9f9410bdaef6dc101aa56f51b6393e51a5
CSV Artifact SHA-256 : 1e73a50933cd3ebe420329a773ca237baf8b5aabc4badaa4e418bb0ea489bf65
Partition Breakdown  : Fit=44,054 rows | Val=25,946 rows (Cold=12,844, Seen=13,102)
Cold Seller Overlap  : 0 sellers (Disjoint)
Seen Bracketing      : 13,102 / 13,102 items strictly bracketed (100.0%)
Data Leakage         : 0 (No test or probe leakage detected)
Gate Compliance      : Gate C: PASS | Gate H: PASS | Preventive Test Suite: 8/8 PASS

DECISION:
The canonical test-representative validation split and coordination governance
are hereby formally:

                [ CERTIFIED FOR PREQUENTIAL EVALUATION ]
================================================================================
```

---

## 3. Cumplimiento de Gate C y Gate H

- **Gate C**: Todos los campos de probabilidad pública (`P(>S4)`, `P(>=0.95000)`) y cuantiles públicos (`Q05`, `expected_public_score`) permanecen invariablemente declarados como `"NO_IDENTIFIABLE"`.
- **Gate H**: Todas las pérdidas empíricas y retracciones se exponen de forma obligatoria en la Sección 0 antes de cualquier métrica positiva.
- **Suite de Tests**: `tests/test_preventive_gates.py` $\to$ `8/8` tests pasados (`OK`), 35 aserciones positivas y negativas verificadas.
