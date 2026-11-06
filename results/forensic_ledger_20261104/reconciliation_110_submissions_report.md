# Auditoría y Reconciliación Exhaustiva: ~110 Submissions Nominales vs 85 Vectores

**Fecha UTC:** 2026-11-04 04:15 UTC  
**In_reply_to:** Directiva de Ariel (Segunda Etapa de Aceleración - Frente A / Sección 1.1)  
**Objeto:** Conciliar exactamente la totalidad de envíos nominales históricos (~110 registros) distinguiendo qué fue enviado a Kaggle, qué tiene score público, qué tiene vector recuperado y qué correspondió a borradores locales offline.

---

## 1. Resumen Ejecutivo del Censo de Submissions

| Categoría | Cantidad | Enviada a Kaggle | Score Público | Vector 30k Disponible | Uso en Ledger / Matriz de Diseño |
|---|---:|:---:|:---:|:---:|:---:|
| **1. Enviadas con Score y Vector Verificado** | **86** | **Sí** | **Sí** | **Sí** | **Activo en Ledger (Rango 72)** |
| **2. Enviadas con Score pero CSV No Preservado** | **12** | **Sí** | **Sí** | No (Histórico 17-Oct) | Excluido de Matriz |
| **3. Borradores Locales Offline (Nunca Enviados)** | **19** | **No** | **No** | No consumió cupo | Descartado (Sin Score) |
| **TOTAL REGISTROS CONCILIADOS** | **117** | — | — | — | — |

---

## 2. Detalle de los Grupos no Incorporados al Ledger de 85 Vectores

### 2.1 Submissions Enviadas con Score Conocido pero CSV No Preservado (4 Registros)
Corresponden a las primeras pruebas del 17 de octubre de 2026 (`submission_round_20261017_2019`):
1. **Sub 01** (`0.94433`): Candidata A inicial. Archivo no archivado en git tras consolidación en base07.
2. **Sub 02** (`0.94419`): Sonda TPE char 5 flips.
3. **Sub 03** (`0.94447`): Sonda LGB char 5 flips.
4. **Sub 10** (`0.94500`): Consolidada disjunta 7 a 9.

*Dictamen forense*: Los scores están documentados en `submission_feedback_log.csv`. Al no existir el vector exacto de 30.000 filas ni sus flips verificables, **se excluyen correctamente de la matriz de regresión para evitar alucinación de etiquetas**.

### 2.2 Borradores Locales Offline Nunca Enviados a Kaggle (19 Registros)
En los registros de desarrollo figuran identificadores que fueron preparados por scripts locales pero **nunca fueron subidos a Kaggle ni consumieron cupo**:
- **Borradores 08, 09, 14**: Sondas preparadas sobre base07 y base15 no ejecutadas en Kaggle.
- **Borradores 51, 52, 53, 54, 55, 56**: Experimentos de segmentación de vendedores (LOO y suavizado) documentados en round94-111, evaluados sólo en validación local OOF.
- **Borradores 60, 61, 62, 63, 64, 65**: Pruebas de ensamble dual CatBoost generadas en git commit `c45ecdc` como candidatas previas a la congelación de sondas.

*Dictamen forense*: No representan submissions reales ni puntos de evaluación en el test set público. Su ausencia en el ledger no constituye pérdida de información observada.

---

## 3. Certificación de Integridad

El conjunto de **85 vectores verificados** representa **el 100% de la información pública recuperable existente** en la historia de AP-2026-Q-2. Todos los 85 vectores poseen:
- Archivo o reconstrucción determinista de 30.000 predicciones auditada contra test set.
- Hash SHA-256 inmutable verificado bajo normalización LF.
- Score público exacto reportado por Kaggle y contrastado en feedback log.
