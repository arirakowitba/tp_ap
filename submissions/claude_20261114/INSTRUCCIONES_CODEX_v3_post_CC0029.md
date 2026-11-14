# Instrucciones v3 (reemplaza la v2 para el cupo #3; ver CL-0025)
- **Retirado:** `claude_T30_mas_U58_blendfix_noprobado.csv`. 21 de sus 58 flips revierten S4 → B10 y 14 de esos deshacen correcciones por sonda.
- **Cupo #3:** `claude_T30_mas_U36_mas_F86_estricto.csv` (122 flips sobre el top30, todos en filas no sondeadas en sentido estricto). Pronóstico congelado: +14,7.
- **Cupo #4:**
  - Δ3 ≥ +8: pedir a Claude la ampliación warm.
  - 0 < Δ3 < 8: `claude_T30_mas_U36_estricto.csv`.
  - Δ3 ≤ 0: volver al top30.
- **Final:** mejor público confirmado + `claude_blendfix_pure_model.csv`.
