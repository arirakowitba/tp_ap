# Deep tabular CatBoost + cross-fitted NB teacher

Rama independiente con las particiones heredadas fit49k/tune7k/confirm14k. El holdout no es históricamente virgen.

- Features determinísticas deep+common en streaming: imágenes (solo metadatos), shipping, stock, ratios y formato textual; seller/category como categorías.
- Se excluyen condition y atributos cuyo id/name contiene condition antes de construir texto o features.
- Precio relativo por categoría/vendedor usa estadísticas sin target ajustadas solo en training de cada etapa.
- Teacher NB: TF-IDF palabra uni/bigram30k + caracteres3–5 30k, log-count ratio + LR C4. Cada fila train recibe predicción OOF3fold. Eval recibe teacher entrenado sobre training completo. Ningún label eval/test entra en teacher.
- CatBoost depth7,750itermax,lr.06,l2=15,random_strength.8,rsm.8,border64,1thread; early stop80 sobre tune. Umbral elegido en tune entre .475/.49/.5/.51/.525. Confirm posterior una vez, sin seleccionar pesos allí.
- Blend histórico85%+candidate15% sobre common holdout es diagnóstico fijo, no búsqueda.
- Checkpoints porfold teacher, por etapa, lock selección antesconfirm. Refit full se omite si no aporta o si ya consumió20min antes de iniciarlo; presupuesto rama25min aproximado.

Ejecutar `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u scripts/train_deep_teacher_round2.py` desde raíz del proyecto. El script reutiliza cachés locales consistentes y valida identidad de filas.
