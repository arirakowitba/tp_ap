# NB-SVM OOF dentro del CatBoost fuerte — 2026-11-03

## Hipótesis y protocolo

Se añadió una probabilidad NB-SVM OOF como única variable nueva al mismo CatBoost con variables profundas, target encodings y probabilidad TF-IDF word+char OOF. NB-SVM usa CountVectorizer binario de palabras 1–3 y caracteres 3–5, razón log-count por clase calculada sólo dentro de cada fold seller-group y regresión logística. Los umbrales se eligieron en tune por semilla antes de mirar confirm. No se usaron scores públicos ni test labels.

Input: 70.000 filas originales, SHA-256 del JSONL `aa49b8710d06f666eeb3cf65c256c8c4f90bf4dbb22c697e313a814ebf5e01cf`.

## Resultados pareados

| Seed | N confirm | Umbral base / dual | Base acc | Dual acc | Base / dual logloss | Fixes / breaks | Neto | Seen / new | Bootstrap vendedor 95% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 12.905 | .465 / .485 | .92553 | .92414 | .18574 / .18878 | 94 / 112 | -18 | +7 / -25 | [-77, +26] |
| 314159 | 12.843 | .480 / .520 | .91988 | .92260 | .19868 / .19413 | 113 / 78 | +35 | +2 / +33 | [0, +73] |
| 20261143 | 12.471 | .470 / .540 | .93200 | .93272 | .17322 / .16996 | 106 / 97 | +9 | -7 / +16 | [-22, +41] |

Hay 38.219 evaluaciones y 31.756 IDs únicos. Al deduplicar solapamientos por prioridad de semilla, el neto queda entre +8 y +18; 5.891 de 6.061 IDs solapados no cambian su delta entre cortes y su suma conjunta es apenas +4. La mejora no alcanza decenas estables y los umbrales duales varían .485–.540. Dos cortes mejoran logloss y uno empeora.

Hashes SHA-256 de pares: seed2026 `85721166144bba97902f02750f654dbf611089dc2f12d449d1383bd38fc4cda2`; seed314159 `d6f7d175ed3a8d40d827c0b21b0848108bcff606c8613c8ff5701e242cc7f942`; seed20261143 `760c902ce5835fe457caca94c6f43e83e400b7c8d125bd0d140a7ebdaf64686b`.

## Decisión

NB-SVM aporta diversidad real y una señal positiva en dos cortes, concentrada en vendedores nuevos, pero no una mejora estable suficiente para promover un refit o gastar Kaggle. Mantenerlo como componente de una cartera de modelos para estudiar una mezcla fijada en tune contra bag3; no usar el resultado de confirm para elegir gate, peso o umbral. El siguiente paso útil es evaluar una mezcla predeclarada CatBoost bag3 + NB-SVM y un NB-SVM lineal como modelo independiente, ambos frente al mismo baseline y en un confirm fresco.

Código histórico: `src/revision_submissions/deep_catboost_nbsvm_ablation_20261029.py`. No se generó submission.
