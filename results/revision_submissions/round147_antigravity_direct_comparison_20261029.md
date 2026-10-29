# Auditoría pareada de Antigravity dual-text contra CatBoost previo (29-10-2026)

Se unieron por `id` los CSV de `paired_predictions_seed_{2026,314159}.csv` (CatBoost de ronda 142 de Antigravity) y `paired_dual_text_seed_{2026,314159}.csv` (dual-text NB-SVM de ronda 144). Igualdad de `y_true` comprobada en todos los 12.905/12.843 IDs. Comparación sobre exactamente los mismos confirm, sin inferir labels de Kaggle.

| Seed | Dual vs CatBoost anterior fixes/breaks | Neto | Seen fixes/breaks | New fixes/breaks | Log loss Cat→dual | IC95 neto bootstrap vendedor |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 173/171 | **+2** | 41/28 | 132/143 (**−11**) | .228338→.226608 | [−51,+48] |
| 314159 | 199/153 | **+46** | 37/38 | 162/115 (**+47**) | .243651→.236342 | [+4,+90] |

La mejora +24/+30 reportada por Antigravity usa su baseline CatBoost reentrenado en el mismo experimento; contra el CatBoost de la ronda anterior, seed2026 queda casi neutro y pierde en nuevos. Esto no invalida la baja de log loss, pero limita la afirmación de ganancia de accuracy consistente. Los dos cortes comparten filas y no son evidencia independiente. El dual aún debe contrastarse con el mejor baseline fuerte HistGB+FastText/CatBoost de nuestro protocolo, con particiones y procedencia equivalentes, antes de proponer submission. No generar CSV para Kaggle sólo por estos números.

Cálculo determinista: parsear ambos CSV por `id`, comprobar `y_true_used==y_true`, comparar `pred_cat_used` con `pred_dual`; log loss usa `p_cat_used` y `p_dual`. Bootstrap 2.000 remuestreos de sumas de delta por `seller_id`, RNG xorshift32 seed de cada corte. Los CSV originales permanecen en `main`.
