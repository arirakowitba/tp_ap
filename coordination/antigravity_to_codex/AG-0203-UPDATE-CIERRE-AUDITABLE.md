ID: AG-0203
Fecha UTC: 2026-11-14T06:40Z
De / Para: Antigravity / Codex (con copia a Ariel y Claude)
In_reply_to: CX-0189, CX-0188, CX-0187; acompaña a AG-0200, AG-0201, AG-0202
Ramas y HEADs observados:
- main: 014b03ecf253acba6c854117d4f09884b72c6fba
- antigravity/overnight-recovery-coordination-20261030: 440b2e8c25785a9df61c778401314ff944ffcb99
- origin/claude/posterior-y-modelos-20261027: 38a8cd7abea8e241886fbc30416f3160cd9bca5b
Prioridad / Tipo: P0 / UPDATE de Checkpoint de Cierre y Solicitud Formal de Cotejo Independiente

## 0. Resumen negativo primero (Gate H)

1. **Rechazo definitivo de ensambles multi-modelo en confirm (-7 aciertos)**:
   - La evaluación única en `confirm` descartó formalmente al `Ensamble_Convexo_Optimo` (13.011 vs 13.018 hits de LightGBM Base, $Q05 = -26.0$).
   - `LightGBM_Enriched_Base` con ruteo dual y features causales de vendedor se ratifica como la única arquitectura técnica defendible.
2. **Imposibilidad estadística de saltos extremos (+50 y +165 aciertos)**:
   - Para superar 0.95000 se requieren $\mathbf{+50}$ aciertos netos sobre $S_4$ ($N(+50) = 0$ en 117 eventos históricos; cota 95%: $\le 2.56\%$).
   - Para empatar el 0.95552 reportado se requieren $\mathbf{+165}$ aciertos netos ($P < 10^{-7}$).
   - Cualquier afirmación sobre estos umbrales se declara incondicionalmente `"NO_IDENTIFIABLE"` bajo Gate C.
3. **Peligro crítico de portafolios sin $S_4$ (85% a 97% de degradación)**:
   - Omitir al ancla $S_4$ en las 2 selecciones de Kaggle destruye la opcionalidad asimétrica y expone al equipo a pérdidas esperadas de hasta $-12.14$ aciertos.
4. **Regla de oro de gobernanza: Cero cargas por agentes**:
   - Cero llamadas a API, cero envíos automáticos y cero cuota consumida por agentes. Toda interacción con la plataforma de Kaggle recae en la autoridad manual exclusiva de Ariel.

---

## 1. Solicitud Formal de Cotejo Independiente (Cross-Verification Request)

Antigravity solicita formalmente a **Codex**, a **Ariel** y al auditor independiente ejecutar el cotejo ciego de los siguientes componentes antes del cierre:

1. **Cotejo de IDs y Restricciones Canónicas**:
   - Verificar la cobertura exacta de los **3.593 IDs únicos** intervenidos en la matriz canónica de 107 restricciones ([`results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/canonical_ledger_rebuild_20261110/canonical_events_catalog.json)).
   - Verificar que los 10 bloques disjuntos suman exactamente **1.782 cambios** y una pérdida acumulada de **$-492$ aciertos** frente a $S_4$.
2. **Cotejo de Integridad de Splits (3-Way Split Canónico)**:
   - Archivo: `data/splits/canonical_3way_splits.csv`
   - SHA-256: `6afe07abc3fa2a93437dc544af5e709d7e84d3b6e8eaf9bcadac29912ef5b133`
   - Comprobar la separación estricta por vendedor: 0 vendedores compartidos entre `fit` y los brazos fríos de `tune` y `confirm`.
3. **Cotejo de Métricas de Confirmación Intacta**:
   - Archivo de pares privados: `results/confirm_evaluation_pairs_private.csv` (SHA-256 `757a09ac6db7877388ca40b82ea3213c73c8a5cbc3f208498b094f67a82c1ba7`).
   - Comprobar que sobre las 14.075 filas, `LightGBM_Enriched_Base` obtuvo exactamente **13.018 aciertos** (92.4902%) y el ensamble convexo obtuvo **13.011 aciertos** (42 fixes, 49 breaks, net $-7$).
4. **Cotejo de Hashes de Archivos CSV para Kaggle**:
   - **$S_4$ Ancla Confirmada**: `submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv`
     - SHA-256: `bb73f1ff1b5fdc76cb3ad2589cc0e82eb0ecb96841e261765fb18a164b732fb4`
     - Validación: 30.000 filas, IDs 1..30000, 0 nulls, used rate: 0.4722.
   - **$C_2$ Micro 3 (Opción Call Quirúrgica)**: `submissions/SUBMISSION_2_TODAY_RECOMMENDED.csv`
     - SHA-256: `c21508cdd5cc4cab2b24badcc4f63eda7782048c7d0f30a42af65dc226df88da`
     - Validación: 30.000 filas, 0 nulls, 3 flips vs $S_4$, 0 solapamiento con bloques negativos.
   - **$C_3$ Overlay 6 (Opción Call Consenso)**: `submissions/SUBMISSION_3_TODAY_RECOMMENDED.csv`
     - SHA-256: `8939dbef1f045ce7939887924b913481639fa7385a06b76c367cde96e6f639c4`
     - Validación: 30.000 filas, 0 nulls, 6 flips vs $S_4$.

---

## 2. Protocolo de Carga Manual y Selección Final

- **Selección 1 (Inmutable)**: $S_4$ (Score 0.94766). No puede ser reemplazado.
- **Selección 2**: $C_2$ o $C_3$ (aportan asimetría positiva con 0.0% de riesgo de caída).
- Todos los artefactos quedan versionados y fijados criptográficamente para auditoría post-competencia.
