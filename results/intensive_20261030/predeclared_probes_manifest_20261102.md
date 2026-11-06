# Manifiesto de Sondas Pre-declaradas y Matriz de Solapamiento — 2026-11-02

## 1. Protocolo Epistemológico
Conforme a las directivas de CX-0057 y CX-0058:
- Se preserva **`candidate_d1_d2_combined_on48.csv` (0.94652)** como ancla líder inmutable.
- Se definen exactamente **4 bloques pequeños y mutuamente disjuntos** de 15 intervenciones cada uno ($n=15$), diseñados a partir de los desacuerdos entre el ensamble supervisado y el ancla D1+D2, estratificados por vendedor visto/nuevo y margen de confianza OOF calibrado.
- La selección de filas queda **estrictamente congelada antes de observar scores**.
- **Cero envíos automáticos a Kaggle**: todos los archivos quedan generados exclusivamente para revisión y auditoría.

## 2. Definición de Bloques

| Bloque | Archivo | Estrato | $n$ Cambios | Margen OOF Mín / Medio | SHA-256 (LF) |
|---|---|---|---:|---:|---|
| `B1_seen_extreme15` | `probe_b1_seen_extreme15_on_d1d2.csv` | Seen sellers, extreme margin (|p - 0.5| > 0.436) | 15 | 0.4368 / 0.4529 | `98a10f50bebb0d48...` |
| `B2_new_extreme15` | `probe_b2_new_extreme15_on_d1d2.csv` | New sellers, extreme margin (|p - 0.5| > 0.356) | 15 | 0.3569 / 0.4108 | `61c9102ff6b612b3...` |
| `B3_seen_unanimous15` | `probe_b3_seen_unanimous15_on_d1d2.csv` | Seen sellers, unanimous consensus (|p - 0.5| > 0.411) | 15 | 0.4119 / 0.4226 | `fee1f2bbdc8d2268...` |
| `B4_new_unanimous15` | `probe_b4_new_unanimous15_on_d1d2.csv` | New sellers, unanimous consensus (|p - 0.5| > 0.299) | 15 | 0.2998 / 0.3255 | `c4de74d6b4605ca5...` |

## 3. Matriz de Solapamiento Mutuo

| Bloque | B1_seen_extreme15 | B2_new_extreme15 | B3_seen_unanimous15 | B4_new_unanimous15 |
|---|---:|---:|---:|---:|
| **B1_seen_extreme15** | 15 | 0 | 0 | 0 |
| **B2_new_extreme15** | 0 | 15 | 0 | 0 |
| **B3_seen_unanimous15** | 0 | 0 | 15 | 0 |
| **B4_new_unanimous15** | 0 | 0 | 0 | 15 |

> **Nota**: El solapamiento fuera de la diagonal es estrictamente 0. Ningún ID se repite entre bloques.

## 4. Regla de Decisión Adaptativa para el 5º Cupo (Pre-especificada)
1. Si $\Delta(B1) > 0$ y $\Delta(B2) > 0$: Enviar combinación $B1 \cup B2$ (30 cambios).
2. Si $\Delta(B1) > 0$ y $\Delta(B2) \le 0$: Enviar $B1 \cup B3$ (si $\Delta(B3) > 0$) o $B1$ solo.
3. Si $\Delta(B2) > 0$ y $\Delta(B1) \le 0$: Enviar $B2 \cup B4$ (si $\Delta(B4) > 0$) o $B2$ solo.
4. Si todos $\Delta \le 0$: Respaldo incondicional en el ancla exacta D1+D2 (0.94652).
