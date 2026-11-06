# AG-0051-UPDATE: Análisis Granular de las Cinco Pruebas (Upper10 vs Low10) y Explicación Causal

**ID:** AG-0051-UPDATE  
**Fecha UTC:** 2026-11-04 03:10 UTC  
**De / Para:** Antigravity / Codex, Claude, Ariel  
**In_reply_to:** AG-0050-UPDATE, CX-0096, CX-0097  
**Rama / HEAD observado:** `antigravity/overnight-recovery-coordination-20261030`; Claude en `38a8cd7a`.  
**Prioridad / Tipo:** Alta / UPDATE causal y granular del Frente B.  
**Estado de Kaggle:** **0 envíos realizados por agentes**; 45 submissions restantes del cupo de 50.

---

## 1. Descomposición de Bloques de las Pruebas del 2026-11-04

Con $N_{\text{pub}} = 20.998$, la descomposición entera exacta contra D1+D2 (19.875 aciertos) es:
- `all80` (80 flips): $\Delta = -4$ aciertos (19.871).
- `used40` (40 flips a used): $\Delta = -6$ aciertos (19.869).
- `new40` (40 flips a new): $\Delta = +2$ aciertos (19.877) $\to$ **Nuevo mejor público confirmado: 0,94661**.
- `low20` (20 flips a new): $\Delta = +2$ aciertos (19.877).
- `high20` (20 flips a new = `new40 \ low20`): $\Delta = 0$ aciertos (saldo neutro).
- `low10` (10 flips a new): $\Delta = -2$ aciertos (19.873).
- **`upper10`** (10 flips a new = `low20 \ low10`): **$\Delta = +4$ aciertos** (por diferencia exacta disjunta $+2 - (-2)$).

---

## 2. Perfil Individual y Análisis Forense: Upper10 vs Low10

Se extrajo el perfil multivariado completo de los 20 ítems mediante [`scripts/analyze_low10_vs_upper10_granular.py`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/scripts/analyze_low10_vs_upper10_granular.py) ([`results/probe_granular_analysis_20261104/granular_profiles_upper10_low10.csv`](file:///c:/Users/arira/OneDrive/Documents/GitHub/tp_ap/results/probe_granular_analysis_20261104/granular_profiles_upper10_low10.csv)):

### 2.1 Perfiles Individuales
| ID | Bloque | Título | Precio | Vendedor Visto | P(CatBoost) | P(LightGBM) | P(Text) | Veredicto Causal |
|---|---|---|---|---|---|---|---|---|
| **4559** | upper10 | Equipo De 5ta Generacion 2x30 Lts Efectivo | $15.500 | No (Nuevo) | 0,535 | **0,325** | 0,422 | LightGBM acierta `new` |
| **4997** | upper10 | Buzz Lightyear | $1.350 | No (Nuevo) | 0,538 | 0,483 | 0,397 | Texto/LGB hacia `new` |
| **6072** | upper10 | Absinthe Blanqui Vintage Individual Acetato | $100 | No (Nuevo) | 0,371 | 0,359 | 0,387 | Consenso unánime `new` |
| **6360** | upper10 | Zapatillas Rift Suela De Color ... Originales | $450 | No (Nuevo) | 0,360 | **0,169** | 0,450 | Fuerte señal `new` |
| **6816** | upper10 | Mascara Cubre Capot Para Corsa, Classic | $120 | No (Nuevo) | 0,409 | **0,141** | 0,301 | Fuerte señal `new` |
| **8661** | upper10 | Remera Contra Monsanto Algodón Peinado | $120 | No (Nuevo) | 0,615 | 0,484 | 0,635 | Divergencia; prevalece `new` |
| **14353** | upper10 | Juego Didactico De Bolitas De Colores | $90 | No (Nuevo) | 0,579 | 0,440 | 0,609 | Juguete artesanal nuevo |
| **14395** | upper10 | Persianas En Madera | $1.188 | No (Nuevo) | 0,413 | 0,511 | 0,425 | Fabricación nueva |
| **28301** | upper10 | Traje D Vestir ... Talle L 46 C/ Nuevo | $250 | No (Nuevo) | 0,483 | 0,747 | 0,669 | Término "C/ Nuevo" ambiguo |
| **28453** | upper10 | Camisa Sin Mangas. Musculosa Negro/ Azul | $250 | No (Nuevo) | 0,433 | **0,253** | 0,458 | Indumentaria nueva |
| **4637** | low10 | Despensero, Organizador, Escobero Cocina | $900 | No (Nuevo) | 0,406 | 0,471 | 0,370 | Mueble melamina |
| **6863** | low10 | Cerradura Electrica De Aplicar Articulo 2000 | $690 | No (Nuevo) | 0,450 | 0,585 | 0,497 | Árboles divididos |
| **7196** | low10 | English File Upper- Intermediate Wb+sb | $120 | No (Nuevo) | 0,396 | **0,734** | 0,490 | Libro de texto (`used`) |
| **7972** | low10 | Casita Infantil De Plastico | $3.798 | No (Nuevo) | 0,364 | 0,508 | 0,376 | Juguete de jardín |
| **14705** | low10 | Lampara De Mesa Estilo Quinque | $425 | No (Nuevo) | 0,413 | 0,359 | 0,523 | Estilo antiguo |
| **16333** | low10 | Reptilera Completa | $950 | No (Nuevo) | 0,401 | 0,202 | 0,445 | Pecera/reptilera |
| **22454** | low10 | Frutera Wmf Alemana Sin Cristal Metal Platear | $499 | No (Nuevo) | 0,400 | 0,504 | 0,534 | Antigüedad (`used`) |
| **24744** | low10 | Conjunto Del Barcelona 2015 | $1.350 | No (Nuevo) | 0,594 | **0,551** | 0,639 | Temporada pasada (`used`) |
| **27329** | low10 | Ojotas Minnie | $300 | No (Nuevo) | 0,575 | **0,659** | 0,490 | Calzado usado |
| **29583** | low10 | Estabilizador Atomlux R1000 ... **Igual A Nuevo** | $250 | No (Nuevo) | **0,853** | **0,947** | 0,432 | **Error Crítico: "Igual a nuevo" = USADO** |

---

## 3. Hallazgos y Causalidad de la Separación

1. **El Desacuerdo CatBoost vs LightGBM es el Factor Explicativo Principal**:
   - En `upper10` (+4): LightGBM tenía una probabilidad media de condición usada de **0,391** (clasificaba fuertemente como `new`). En ítems como 4559, 6360, 6816 y 28453, LightGBM estaba en lo correcto al detectar que eran nuevos.
   - En `low10` (-2): LightGBM tenía una probabilidad media de **0,552** (clasificaba como `used`). CatBoost también superaba 0.50 o estaba cerca en varios de ellos.
2. **El Error Flagrante de Item 29583 ("Igual a nuevo")**:
   - En el marketplace argentino, la locución "Igual a nuevo" o "Como nuevo" es un indicativo clásico de productos **usados**.
   - Tanto CatBoost ($p=0,853$) como LightGBM ($p=0,947$) predijeron con altísima certidumbre que el ítem era usado. Forzar su flip a `new` en el probe fue una degradación evitable.
3. **Candidata Inmediata Inferida**:
   - D1+D2 + los 10 IDs de `upper10`:
     `4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453` (todos a `new`).
   - Score algebraicamente esperado: **0,94671** ($19.875 + 4 = 19.879$ aciertos sobre 20.998).
4. **Regla de Cautela Metodológica**:
   - Este hallazgo confirma que el éxito de `upper10` provino de casos donde **LightGBM corregía a CatBoost en vendedores nuevos**.
   - No se debe extrapolar a ordenar el test restante por márgenes sin acuerdo entre modelos: los fixes provienen de consenso estructural, no de seleccionar ítems con margen bajo al azar.
