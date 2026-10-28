# Sesión 27–28 de octubre de 2026 — resumen (Hugo + Claude)

Branch: `claude/posterior-y-modelos-20261027`, creada desde `experiment-ledger-20261025`. Ninguna submission fue enviada por el asistente; los scores los informó Hugo.

## 1. Revisión de submissions y pruebas

- Se reconstruyó byte a byte la **48** (0,94633), mejor pública: base 33 + 6 flips de 41 + 2 de 48. SHA256 `3f8d08e0…6c9722`.
- Scores nuevos informados: **54 = 0,94457** (−37 netos sobre 204 flips), **55 = 0,94514** (−25 sobre 65). Sus IDs no están en el repositorio y no entran al modelo. Según Hugo, 51–53 no se subieron (ver la nota en la ronda 125).
- Tabla de efecto marginal de cada bloque: de ~50 pruebas solo 24, 26, 29, 30, 31→33, 41, 44 y 48 fueron positivas, y todas ya están dentro de la 48.

## 2. Posterior exacta de flips (ronda 125)

`posterior_flips_20261027/` convierte cada score en una ecuación entera sobre la 48 (21.000 filas públicas, truncamiento a 5 decimales) y muestrea la posterior por ID con un MCMC que respeta todas las ecuaciones. Detalle: [round125_posterior_flips_57_58_20261027.md](round125_posterior_flips_57_58_20261027.md).

- **57** = 48 + 20 flips de ganancia esperada positiva → **0,94633** (empate).
- **58** = 48 + 2 flips → **0,94633** (empate).
- Conclusión: la evidencia de 04–58 está agotada; en desacuerdos entre modelos y 48, 48 está mal ≈1/3 de las veces (q posterior 0,334).

## 3. Nueva familia de modelos (ronda 126)

Datos oficiales recibidos de Hugo (70.000 train / 30.000 test). CV estratificada aleatoria de 5 folds, porque test comparte fechas y 72 % de vendedores con train. Detalle completo: [../round126_modelos/README.md](../round126_modelos/README.md).

| Hito | OOF acc |
|---|---:|
| LightGBM con TF-IDF y TE de vendedor | 0,9399 |
| CatBoost profundidad 8 con `seller_id` categórico + stacks de texto | 0,9472 |
| CatBoost profundidad 5 (árboles más bajos generalizan mejor) | 0,9509 |
| Corte por logloss en vez de accuracy (más iteraciones) | 0,9511–0,9513 |
| **`ens_ll8`: promedio de 8 CatBoost profundidad 5** | **0,9519** (logloss 0,1244) |

Probado sin mejora: vecinos de precio por título, siete vistas de texto separadas, segundo nivel con suavizado por vendedor, profundidad 3 y 8–10, lr 0,02. Los stacks de texto (`sparse_all` + `nbsvm`) valen ~0,35 puntos. Los errores restantes se concentran en vendedores con un solo aviso en train; en vendedores con historial, el 80 % de los errores contradice la etiqueta mayoritaria del propio vendedor (probable ruido de etiqueta).

## 4. Tanda propuesta para el 28-10

`submissions/20261028/`: cuatro pruebas disjuntas sobre la 48 con los **493 IDs donde `ens_ll8` contradice a 48 y que nunca se probaron**, por banda de confianza del modelo (59: ≥0,85, 24 flips; 60: 0,70–0,85, 91; 61: 0,60–0,70, 140; 62: 0,50–0,60, 238). Miden directamente con qué frecuencia un modelo de 95 % OOF le gana a 48. El orden adaptativo está en [estrategia_bandas_28_10.md](../../submissions/20261028/estrategia_bandas_28_10.md). Cada score nuevo se agrega a `posterior_flips_20261027/assemble.py`.

## 5. Agentes del 28-10 (06:15–18:30)

Cuatro agentes trabajan en paralelo con el mismo protocolo (5 folds, semilla 2026) para que todo sea comparable y mezclable con `ens_ll8`. La máquina tiene 2 núcleos: A y B usan uno cada uno; C corre con prioridad baja (`nice`); D no entrena.

| Agente | Línea de trabajo | Código / registro |
|---|---|---|
| A | Información nueva en features: vendedores sin historial, campos JSON no usados, contexto de categoría | `src/round127_agente_a/`, `results/round127_agente_a/` |
| B | XGBoost/LightGBM bien ajustados, texto neuronal desde cero, ensamblado con CV anidada | `src/round127_agente_b/`, `results/round127_agente_b/` |
| C | Arquitecturas distintas a GBM (MLP con embeddings, k-NN, semi-supervisado) para diversidad del ensamble | `src/round127_agente_c/`, `results/round127_agente_c/` |
| D | Búsqueda de información pública (soluciones del desafío MLA new/used, técnicas, reglas de la competencia) | `results/round127_agente_d/` |

Regla de integridad para D: no descargar ni usar datasets etiquetados que puedan contener los ítems de test; solo registrarlos para que el equipo decida con la cátedra.

## 6. Pendientes

- Recibir scores 59–62, recalcular la posterior y armar la consolidación del día.
- Si las bandas altas dan positivo, extender a los 614 desacuerdos ya probados con la posterior como filtro.
- Mejoras de modelo con información nueva (no más semillas): vendedores sin historial, texto.
