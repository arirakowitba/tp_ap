# Rondas 128–129 — discriminante regularizado y Fourier RBF

Dos familias nuevas sobre embeddings FastText de los textos originales, train 70k etiquetado. R128: LDA/QDA con shrinkage .5/.9. R129: características aleatorias Fourier para kernel RBF (gamma .005/.02; 512 dimensiones), regresión logística. Tres cortes históricos seller-group a/b/c; fit separado de holdout, tune y confirm disjuntos dentro de cada corte. Baseline pareado HistGB+FastText de ronda39. Pesos de blend .05/.10/.20; selección congelada en tune exige neto positivo de accuracy y menor log loss.

Resultado: en ambas familias y los tres cortes, ninguna variante cumplió simultáneamente ambos criterios en tune; el baseline fue seleccionado para confirm en los seis casos. No hay ganancia validada ni predicción de test que promover. Los confirms históricos se solapan: a/b 2868, a/c 3218, b/c 3185 filas; 616 en los tres, por lo que no son réplicas independientes. Los pares guardan IDs de vendedor, etiquetas train y probabilidades de base/modelo/selección. No se usaron etiquetas de test ni scores públicos por ID. CatBoost no tiene pares comparables para estos mismos cortes; no se afirma comparación directa.

Checkpoint Library `libfile_98eccc574f508191aff2bbd1c0a35660`, SHA256 `36dff48901e549d5c165222b71ff9087ffdb5fc81027e8bf607b22e409055d2a`.
