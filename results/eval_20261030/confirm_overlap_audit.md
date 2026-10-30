# Auditoría de Solapamiento Histórico de Particiones (Confirm y Tune)

**Fecha:** 2026-10-30  
**Objetivo Metodológico:** Cuantificar el grado de solapamiento entre las particiones históricas de evaluación y la semilla predeclarada `20261030`.  
**Conclusión Crítica:** Ninguna partición de confirm evaluada sobre las 70.000 etiquetas de train es "virgen". Todas reutilizan filas y vendedores previamente examinados por otros experimentos.

## Tabla de Solapamiento con `R151_seed20261030_confirm` (N = 12881 filas, 6586 vendedores)

| Partición Histórica | N Filas | N Vendedores | Solapamiento Filas | % Filas | Solapamiento Vendedores | % Vendedores |
|---|---|---|---|---|---|---|
| `R151_seed2026_tune` | 11765 | 6533 | **2122** | 16.5% | 1998 | 30.3% |
| `R151_seed2026_confirm` | 12905 | 6649 | **2393** | 18.6% | 2073 | 31.5% |
| `R157_seed2026_tune` | 11765 | 6547 | **2105** | 16.3% | 1999 | 30.4% |
| `R157_seed2026_confirm` | 12906 | 6656 | **2382** | 18.5% | 2070 | 31.4% |
| `R151_seed314159_tune` | 12473 | 6573 | **2147** | 16.7% | 1983 | 30.1% |
| `R151_seed314159_confirm` | 12843 | 6642 | **2249** | 17.5% | 2020 | 30.7% |
| `R157_seed314159_tune` | 12473 | 6602 | **2137** | 16.6% | 2016 | 30.6% |
| `R157_seed314159_confirm` | 12844 | 6614 | **2293** | 17.8% | 2037 | 30.9% |
| `R151_seed20261030_tune` | 13451 | 6573 | **0** | 0.0% | 937 | 14.2% |
| `R151_seed20261030_confirm` | 12881 | 6586 | **12881** | 100.0% | 6586 | 100.0% |

### Implicaciones Metodológicas:
1. **Reutilización de datos:** Al haber corrido decenas de rondas sobre el mismo conjunto de 70.000 filas, variar la semilla de división de vendedores cambia la asignación pseudoaleatoria, pero el 18% a 22% de las filas de confirm coinciden por puro azar de muestreo (10% tune + 10% confirm).
2. **Protocolo:** Para evitar sesgo de selección retrospectiva, cualquier nueva configuración debe optimizarse y congelarse exclusivamente en `fit/tune`, y evaluarse en `confirm` **una única vez**.
