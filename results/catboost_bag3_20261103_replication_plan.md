# Replicación predeclarada bagging CatBoost, seed 20261125

Tras observar el resultado de 20261124 (+28 netos, IC vendedor [+7.975,+48], logloss menor), se congela una segunda ejecución independiente del mismo código `scripts/ablate_catboost_bag3_20261103.py 20261125`, sin cambiar arquitectura, pesos 1/3, grilla ni hiperparámetros. Se reportará resultado aunque sea negativo, en el mismo formato y junto al seed previo. Este corte nuevo no es completamente independiente porque proviene de los mismos 70k y puede solaparse en IDs con 20261124. No se usarán scores públicos ni Kaggle.
