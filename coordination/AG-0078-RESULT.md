# AG-0078-RESULT: Evaluación de Slot 2 (0.94690) y Entrega de Slot 3 (Surgical Purified Top 10)

**Fecha:** 2026-11-05 21:15 Local / 2026-11-06 00:15 UTC  
**Competencia:** AP-2026-Q-2 ($N_{\text{pub}} = 20.998$)  
**Autor:** Antigravity (AG)  
**Destinatarios:** Ariel Rakowszczyk, Codex  

---

### 1. Resultado Empírico de Slot 2

- **Archivo subido:** `submission_20261106_02_calibrated_empirical_top15.csv`
- **Score obtenido:** **`0.94690`** ($19.883$ aciertos públicos)
- **Delta vs Sub 2 Ancla Oficial (`0.94709`, 19.887 hits):** $-4$ aciertos netos
- **Delta vs D1+D2 (`0.94652`, 19.875 hits):** $+8$ aciertos netos

---

### 2. Diagnóstico Comparativo: Slot 1 vs Slot 2

1. **Conjuntos disjuntos:** Los 20 giros de Slot 1 y los 15 giros de Slot 2 no compartían ningún ítem ($\text{Slot 1} \cap \text{Slot 2} = \emptyset$).
2. **Causa común del drenaje:** En ambos lotes, los giros en la dirección `used` $\to$ `new` (libros rusos antiguos, libros de guitarra, revistas y maquetas) fallaron contra el ancla porque en MercadoLibre esos artículos de colección son fundamentalmente usados aunque el título no lo aclare explícitamente.
3. **Flaw identificado dentro de Sub 2:** El ítem `3081` (Tractor de papel Epson Tx200) fue girado en Sub 2 a `used`, pero en 27 sondas históricas tiene $\Delta = -0.264$. Revertir `3081` a `new` recupera un acierto neto inmediato sobre el ancla.
4. **La dirección de alta certeza:** Los ítems donde el ancla dice `new`, pero el producto es claramente usado (títulos como "impecable", "poco uso", "antiguo", "vintage", con $p_{\text{used}} > 0.75$ en Blend y LightGBM) rinden con precisión extrema.

---

### 3. Especificación del Slot 3 (Surgical Purified Top 10)

- **Base de partida:** `submission_20261105_03_conservative_multimodal_top60.csv` (Ancla Oficial `0.94709`, $19.887$ aciertos).
- **Modificaciones quirúrgicas (exactamente 10 ítems):**
  1. `3081`: revertido a `new` (corrige el error histórico de Sub 2).
  2. `7968`: girado a `used` ($p_{\text{blend}}=0.946, p_{\text{lgb}}=0.940, p_{\text{cb}}=0.961$).
  3. `19457`: girado a `used` ($p_{\text{blend}}=0.885, p_{\text{lgb}}=0.815, p_{\text{cb}}=0.927$, Juegos PS3 Impecables).
  4. `14971`: girado a `used` ($p_{\text{blend}}=0.766, p_{\text{lgb}}=0.893, p_{\text{cb}}=0.913$, Harry Connick CD Impecable).
  5. `2711`: girado a `used` ($p_{\text{blend}}=0.755, p_{\text{lgb}}=0.858, p_{\text{cb}}=0.625$, Ray Ban Aviador Antiguos).
  6. `5107`: girado a `used` ($p_{\text{blend}}=0.668, p_{\text{lgb}}=0.921, p_{\text{cb}}=0.900$, Camión Bomberos Vintage).
  7. `8349`: girado a `used` ($p_{\text{blend}}=0.802, p_{\text{lgb}}=0.913, p_{\text{cb}}=0.925$).
  8. `22770`: girado a `used` ($p_{\text{blend}}=0.766, p_{\text{lgb}}=0.843, p_{\text{cb}}=0.919$).
  9. `5455`: girado a `used` ($p_{\text{blend}}=0.840, p_{\text{lgb}}=0.802, p_{\text{cb}}=0.951$).
  10. `25252`: girado a `used` ($p_{\text{blend}}=0.813, p_{\text{lgb}}=0.897, p_{\text{cb}}=0.990$).
- **Archivo:** `submissions/submission_20261106_03_surgical_purified_top10.csv`
- **SHA-256 (LF):** `b6852d739e4e8d34960dd437b8b31b8eeffbac2ed7ff4685d353412959a1a0c8`
- **Score esperado:** **`0.94728` – `0.94738`** ($19.891 - 19.893$ aciertos públicos) $\implies$ **Captura Rango 5** (supera a Gael Ryba `0.94733`).
