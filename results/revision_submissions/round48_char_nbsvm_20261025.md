# Char NB-SVM en vendedores reservados

Se entrenó TF-IDF de caracteres 3–5 (máximo 180.000 términos, min_df 5), ratio NB supervisado y regresión logística C=0,3. Cada uno de dos cortes seller-group usó unas 41.000 etiquetas solo en fit. Tune y confirm no comparten vendedores con fit ni entre sí. El clasificador se mezcló con HistGB+FastText a pesos fijos 5, 10, 20 y 40%, umbral de decisión 0,475; los pares completos están en los NPZ. No se usaron test labels ni leaderboard para ajuste.

| Split | Peso | Tune: corrige/rompe/neto | Confirm: corrige/rompe/neto | Log loss confirm base→mezcla |
|---|---:|---:|---:|---:|
| 39002 | 5% | 6/8/−2 | 7/5/+2 | .210628→.213404 |
| 39011 | 5% | 7/11/−4 | 6/13/−7 | .205913→.209711 |
| 39002 | 20% | 26/37/−11 | 37/24/+13 | .210628→.234485 |
| 39011 | 20% | 38/49/−11 | 34/51/−17 | .205913→.232849 |

El peso elegido por tune sería cero en ambos cortes. Entre los desacuerdos con el baseline al 20%, la dirección `new→used` en confirm fue +5/39 y −15/43, y la dirección `used→new` +8/22 y −2/42; no hay dirección robusta. Casi todos los flips ocurren donde el baseline tiene probabilidad .4–.55. El log loss empeora siempre y el segundo confirm pierde 17 aciertos con 85 cambios.

**Decisión:** descartar este clasificador como candidato para test. La representación de caracteres supervisada simple no corrigió la incertidumbre de HistGB+FastText de forma estable. Los resultados son contra ese baseline reproducible, no contra la política exacta de la 41. El próximo intento de texto debe modelar interacciones de título/descripcion y contexto tabular con predicciones OOF agrupadas, o priorizar errores sistemáticos en segmentos antes de proponer flips.

Reproducción: `OPENBLAS_NUM_THREADS=2 python3 agent_text/char_nbsvm.py` desde la raíz compartida. Artefactos: `report.json`, dos `*_char_pairs.npz` con índice, y, seller_id y tres probabilidades.
