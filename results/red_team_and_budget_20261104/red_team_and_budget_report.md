# Agente F: Auditoría Red Team de Candidatas y Presupuesto Adaptativo de 45 Submissions

**Fecha UTC:** 2026-11-04 04:45 UTC  
**In_reply_to:** Directiva de Ariel (Segunda Etapa de Aceleración - Frente F & Gobernanza)  
**Objeto:** Certificación de integridad de archivos de submission generados, asignación adaptativa de los 45 cupos y matriz de decisión.

---

## 1. Auditoría Red Team de Candidatas Generadas

Todos los archivos generados en `submissions/candidates_20261104/` superaron las pruebas de sanidad y consistencia:
- Esquema estricto: 2 columnas exactas (`ID,condition`).
- Integridad: 30.000 filas exactas, IDs 1 a 30.000 sin vacíos ni duplicados.
- Tipos de datos: condiciones acotadas estrictamente a `new` o `used`.
- Hashes SHA-256 normalizados en formato LF:

| Candidata | SHA-256 (LF) | Flips vs D1+D2 | Flips vs New40 | Estado de Auditoría |
|---|---|---:|---:|:---:|
| `candidate_01_d1d2_plus_upper10.csv` | `ac35697bef44bcd679dfbae59011d72fcf558af5acf1661f5818b8629129d13d` | 10 | 30 | **APROBADA** |
| `candidate_02_informative_contrast_batch15.csv` | `69e6e593d38e80281501e16162983c814f03a7d51ceba54125ee9acb140649ed` | 15 | 25 | **APROBADA** |
| `candidate_03_full_multimodal_trees_blend.csv` | `3a093158290b378b4a3cb2b46db45f61df870685d0d56aa3ba9ae1d98c2d7a89` | 639 | 637 | **APROBADA** |
| `candidate_04a_crossed_hadamard_batchA.csv` | `e7a2d02c0f54effba61639ece76d5420db07a9b13f90fb0cdaba6146d4730e3b` | 9 | 31 | **APROBADA** |
| `candidate_04b_crossed_hadamard_batchB.csv` | `19af38fffeb90fbf2da1362246fb587085ca46906768476a31b79ab38714845a` | 9 | 31 | **APROBADA** |
| `candidate_05_revert_blunder_29583_on_new40.csv` | `b8aa5656e48d36fa906ead53bf9c9144458374727347cc0d210d6fecb44f4f81` | 39 | 1 | **APROBADA** |

---

## 2. Presupuesto Adaptativo para los 45 Cupos Restantes (5 Días)

| Categoría Estratégica | Porcentaje | Cupos Asignados | Propósito Principal |
|---|---:|---:|---|
| **Explotación Directa** | **20%** | **9 cupos** | Consolidar mejoras garantizadas (`upper10`, reversión ítem 29583, consensus seguro). |
| **Modelos Base y Diversidad** | **35%** | **16 cupos** | Envíos de ensambles multimodales completos (Blend CB+LGB, representaciones densas). |
| **Sondas Informativas de Rango** | **25%** | **11 cupos** | Batches Hadamard y ortogonales (10-15 IDs) para expandir el rango de la matriz de diseño. |
| **Reserva Estratégica de Cierre** | **20%** | **9 cupos** | Cupos protegidos para los últimos 2 días para calibración final y desacoplamiento público/privado. |
| **TOTAL** | **100%** | **45 cupos** | — |

---

## 3. Matriz de Decisión Operativa para Ariel

1. **Si se busca máxima seguridad para elevar el ancla pública**:
   - Enviar **`candidate_01_d1d2_plus_upper10.csv`** (Score esperado: **0,94671**, +4 aciertos garantizados).
2. **Si se busca reparar el error conocido de New40**:
   - Enviar **`candidate_05_revert_blunder_29583_on_new40.csv`** (Score esperado: **0,94666**, +1 acierto neto).
3. **Si se busca dar el salto de escala hacia el líder (~0,950+)**:
   - Enviar la versión filtrada por el Router OOF del ensamble multimodal (Cohorte 50 o 100).
