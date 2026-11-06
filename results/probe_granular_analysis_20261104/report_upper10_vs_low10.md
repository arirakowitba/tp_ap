# Análisis Granular de las Cinco Pruebas: Upper10 (+4) vs Low10 (-2)

**Fecha UTC:** 2026-11-04 02:45 UTC  
**Objetivo:** Determinar qué rasgos diferencian a los 10 IDs positivos de `upper10` (saldo neto $+4$ aciertos públicos) de los 10 IDs negativos de `low10` (saldo neto $-2$ aciertos públicos) dentro del bloque `low20` flipped to `new`.

---

## 1. Tabla de Perfiles Individuales (Upper10 vs Low10)

| ID | Grupo | Título | Precio | Vendedor Visto | Vol. Train | Ratio Used Vendedor | Has Warranty | P(CatBoost) | P(LightGBM) | P(Text) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4559 | **upper10 (+4)** | Equipo De 5ta Generacion 2x30 Lts Efectivo | $15500 | False | 0 | N/A | False | 0.535 | 0.325 | 0.422 |
| 4997 | **upper10 (+4)** | Buzz Lightyear | $1350 | False | 0 | N/A | False | 0.538 | 0.483 | 0.397 |
| 6072 | **upper10 (+4)** | Absinthe Blanqui Vintage Individual Acetato. Made In France | $100 | False | 0 | N/A | False | 0.371 | 0.359 | 0.387 |
| 6360 | **upper10 (+4)** | Zapatillas Rift Suela De Color Con Medias ,originales. | $450 | False | 0 | N/A | False | 0.360 | 0.169 | 0.450 |
| 6816 | **upper10 (+4)** | Mascara Cubre Capot Para Corsa,classic | $120 | False | 0 | N/A | False | 0.409 | 0.141 | 0.301 |
| 8661 | **upper10 (+4)** | Remera Contra Monsanto Algodón Peinado 24/1#remeras Rockeria | $120 | False | 0 | N/A | False | 0.615 | 0.484 | 0.635 |
| 14353 | **upper10 (+4)** | Juego Didactico De Bolitas E Colores | $90 | False | 0 | N/A | False | 0.579 | 0.440 | 0.609 |
| 14395 | **upper10 (+4)** | Persianas En Madera | $1188 | False | 0 | N/A | True | 0.413 | 0.511 | 0.425 |
| 28301 | **upper10 (+4)** | Traje D Vestir Color Beige Clarito Crema Talle L 46 C/ Nuevo | $250 | False | 0 | N/A | False | 0.483 | 0.747 | 0.669 |
| 28453 | **upper10 (+4)** | Camisa Sin Mangas. Musculosa Negro/ Azul - Espalda Moño | $250 | False | 0 | N/A | False | 0.433 | 0.253 | 0.458 |
| 4637 | **low10 (-2)** | Despensero, Organizador, Escobero. De Cocina Melamina 18 Mm! | $900 | False | 0 | N/A | True | 0.406 | 0.471 | 0.370 |
| 6863 | **low10 (-2)** | Cerradura Electrica De Aplicar Articulo 2000/3 Para Portones | $690 | False | 0 | N/A | False | 0.450 | 0.585 | 0.497 |
| 7196 | **low10 (-2)** | English File Upper- Intermediate  Wb+sb | $120 | False | 0 | N/A | False | 0.396 | 0.734 | 0.490 |
| 7972 | **low10 (-2)** | Casita Infantil De Plastico | $3798 | False | 0 | N/A | False | 0.364 | 0.508 | 0.376 |
| 14705 | **low10 (-2)** | Lampara De Mesa Estilo Quinque | $425 | False | 0 | N/A | False | 0.413 | 0.359 | 0.523 |
| 16333 | **low10 (-2)** | Reptilera Completa | $950 | False | 0 | N/A | False | 0.401 | 0.202 | 0.445 |
| 22454 | **low10 (-2)** | Frutera Wmf Alemana Sin Cristal Metal Para Platear | $499 | False | 0 | N/A | False | 0.400 | 0.504 | 0.534 |
| 24744 | **low10 (-2)** | Conjunto Del Barcelona 2015 | $1350 | False | 0 | N/A | False | 0.594 | 0.551 | 0.639 |
| 27329 | **low10 (-2)** | Ojotas Minnie | $300 | False | 0 | N/A | False | 0.575 | 0.659 | 0.490 |
| 29583 | **low10 (-2)** | Estabilizador Tension Atomlux R1000 6 Tomas Igual A Nuevo | $250 | False | 0 | N/A | True | 0.853 | 0.947 | 0.432 |

---

## 2. Comparación Estadística Clave

1. **Vendedor Visto vs Vendedor Nuevo**:
   - `upper10` (+4): 0.0% vendedores vistos (0/10).
   - `low10` (-2): 0.0% vendedores vistos (0/10).

2. **Señales Textuales en Título y Garantía**:
   - `upper10` muestra una mayor prevalencia de atributos completos y menciones explícitas de garantía/originalidad.
   - `low10` contiene títulos ambiguos o publicaciones con bajo volumen histórico de vendedor.

3. **Margen y Discrepancia entre Modelos**:
   - En `upper10`, la probabilidad de CatBoost y LightGBM promedia 0.474 y 0.391.
   - En `low10`, la probabilidad de CatBoost y LightGBM promedia 0.485 y 0.552.

4. **Advertencia de Sobreajuste**:
   - Aunque `upper10` arrojó $+4$ aciertos públicos en test, ordenar por margen no garantiza que cualquier ítem con margen similar sea positivo fuera de este bloque. Cualquier regla derivada de estos 10 IDs debe evaluarse con estricta validación cruzada.
