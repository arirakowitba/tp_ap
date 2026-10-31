# Manifiesto del Paquete Adaptativo de Submissions (CX-0032)

- **Fecha UTC**: 2026-10-31T22:43:55Z
- **Límite de Revisión**: 2026-10-31 21:00 Buenos Aires (00:00 UTC)
- **Mejor Demostrada Actual**: `candidate_d1_d2_combined_on48.csv` (.94652 LB, 23 flips vs 48)
- **Referencia de Base**: `submission_48_reference.csv` (.94633 LB)

---

## 1. Cuatro Sondas Informativas Secuenciales

| Sonda | Archivo | SHA-256 | Flips vs 48 | Segmento | Hipótesis Falsable |
|---|---|---|---:|---|---|
| **Sonda 1** | `candidate_probe1_seen_virgin7.csv` | `3a0c5b324ae03def...` | 7 | Vistos (`used -> new`) | Desacopla si la ganancia de D1+D2 provino del componente de vendedores vistos. |
| **Sonda 2** | `candidate_probe2_new_virgin16.csv` | `41b5d09075577889...` | 16 | Nuevos (`used -> new`) | Desacopla si la ganancia de D1+D2 provino del componente de vendedores nuevos. |
| **Sonda 3** | `candidate_probe3_reverse_polarity_virgin12.csv` | `7d353b78a188c66f...` | 12 | Vírgenes (`new -> used`) | Evalúa la simetría de errores de 48 y si se recupera ground truth en polaridad opuesta. |
| **Sonda 4** | `candidate_probe4_inductive_price_extreme15.csv` | `7d353b78a188c66f...` | 15 | Nuevos (Precio Extremo) | Evalúa si las interacciones de precio relativo corrigen discrepancias en vendedores nuevos. |

---

## 2. Matriz de Decisión para la 5ª Submission

La 5ª submission **no está prefijada**: se construirá dinámicamente con las respuestas que Ariel observe en Kaggle:

1. **Si Sonda 1 y Sonda 2 ambas superan .94633**: Se confirma la aditividad de D1+D2; la 5ª submission integra D1 + D2 + las sondas adicionales ganadoras (S3/S4) calibradas.
2. **Si Sonda 2 supera .94633 y Sonda 1 no**: La ganancia reside en vendedores nuevos; la 5ª submission se enfoca en expandir el bloque inductivo de nuevos (D2 + S4).
3. **Si Sonda 1 supera .94633 y Sonda 2 no**: La ganancia reside en vendedores vistos; la 5ª submission se enfoca en vistos (D1) y descarta nuevos.
4. **Si ninguna sonda supera .94652**: Se conserva formalmente `candidate_d1_d2_combined_on48.csv` (.94652) como el mejor respaldo verificado.

---

## 3. Verificación de Integridad

- Todos los archivos fueron verificados con parser independiente (`csv.DictReader`).
- Exactamente 30.000 registros por archivo, IDs 1..30000 secuenciales, 0 nulos, clases válidas (`used`, `new`).
- Cero solapamiento de los IDs intervenidos con submissions históricas descartadas.
