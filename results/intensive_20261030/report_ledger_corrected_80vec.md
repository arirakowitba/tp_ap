# Reporte de Corrección del Ledger Maestro (80 Vectores Recuperables)

En estricto cumplimiento de **CX-0090**, se corrigieron las inconsistencias del inventario previo:

- **Total de entradas registradas con score**: 83
- **Vectores con CSV y hash exacto recuperables**: **80**
- **Entradas no recuperables (marcadas unresolved_no_csv)**: **3** (claves: `53`, `54`, `55`)
- **IDs únicos intervenidos**: **2568**
- **Dimensión de la matriz de incidencia**: `80 x 2568`
- **Rango algebraico**: **68**
- **Nulidad algebraica**: **2500**

## Corrección de las 5 Submissions Recientes frente a Submission 48

| Clave | Score | Flips vs D1+D2 | Flips vs 48 (Declarado Previo) | Flips vs 48 (Verdadero Diferencia Simétrica) | Estado |
|---|---|---|---|---|---|
| `A15` | — | 15 | 28 | **36** | `verified_recoverable` |
| `A11` | — | 11 | 24 | **32** | `verified_recoverable` |
| `strong5` | — | 5 | 24 | **28** | `verified_recoverable` |
| `top2` | — | 2 | 23 | **25** | `verified_recoverable` |
| `final_global_evidence` | — | 10 | 27 | **25** | `verified_recoverable` |

## Conclusiones sobre Identificabilidad

1. La nulidad de 2.500 confirma que el sistema está masivamente subdeterminado. Ningún score agregado autoriza a asignar etiquetas puntuales a IDs aislados.
2. Las entradas 53, 54 y 55 quedan formalmente excluidas de cualquier ecuación o inferencia hasta recuperar los CSV originales de Kaggle.
3. El ancla empírica confirmada continúa siendo **D1+D2 = 0,94652**.
