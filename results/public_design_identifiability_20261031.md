# Identificabilidad del historial público — 2026-10-31

Ejecutado sobre `results/intensive_20261030/submission_ledger_04_current.json` de la rama Antigravity con la lógica de `src/audit_public_design_identifiability.py` (main). El ledger contiene 68 entradas; 65 tienen hash de vector y mapa de flips utilizables, que colapsan a 59 vectores distintos. Este snapshot no incluye necesariamente todos los envíos reales ni las cuatro sondas del 31-10.

| Medida | Valor |
|---|---:|
| ID con algún flip respecto de S48 | 2.338 |
| ID sin ningún flip | 27.662 |
| Firmas distintas de intervención | 377 |
| Firmas de un solo ID | 215 |
| ID en firmas compartidas | 29.785 |
| Tamaño de mayor firma | 27.662 |

Una firma indica en qué vectores cambió un ID, no su etiqueta. Dos ID con la misma firma aportan idénticamente a todas las diferencias de score observadas, y no pueden separarse con estos scores agregados. La membresía de cada ID en el ~70% público es desconocida, lo que reduce aún más la identificabilidad. La mayoría de los 30k ID requieren un prior train-only; no debe atribuirse información individual del leaderboard a 27.662 casos jamás intervenidos.

La matriz de contrastes tiene a lo sumo 58 dimensiones independientes antes de considerar redondeo, solapamientos y selección adaptativa. Los scores son restricciones de sumas por bloques. Un posterior por ID debe reportar sensibilidad al prior y grupos equivalentes de diseño, y comprobar su utilidad predictiva mediante backtest prequential. No fijar etiquetas por ID usando scores agregados.
