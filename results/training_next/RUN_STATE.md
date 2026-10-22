# Checkpoint de entrenamiento — tanda posterior a submissions 21–25

Datos: recovered/data/train_data.jsonlines (70k) y test_data.jsonlines (30k). No entrenar con labels recuperadas ni saldos públicos.

Referencia pública observada: submission24=0.94533. Submission25=0.94519. Complemento E2 submission26 preparado, esperado0.94547, no observado.

Experimentos finalizados: NB-SVM, retrieval TFIDF, CatBoost rich2configuraciones, LightGBM seller context2configuraciones. Todos usan fit49k/tune7k/refit56k/confirm14k heredados. Confirm no es históricamente virgen.

NB+native fijo85/15:93.350% vs93.157%; al comparar con mejor histórico solo+3/4147 y empate con variante histórica más fuerte. No justifica submission por sí solo.
Retrieval+native fijo90/10:93.250%. Context+native fijo85/15:93.243%. Rich:92.771% descartado como reemplazo. Ver informes por rama.

Pendiente: reproducir native_tune (script reproduce_native_tune_next.py). Primer intento OOM exit137 al convivir con rich; reintentado después de liberar memoria. Salida baseline_reproduction/seller_native_tune.npz. Stacker regularizado requiere ese archivo; script train_stacker_next.py en preparación. No elegir pesos mirando confirm.

Repo: checkpoints código y métricas autorizados en main. Auto-review bloqueó binarios arrays por exigir permiso adicional; no afirmar que están en GitHub. Conservar modelos/NPZ en checkpoint descargable completo. Los CSV21–26 se guardaron en main, F8 retirado excluido.

Fuentes primarias: https://aclanthology.org/P12-2018/ ; https://catboost.ai/docs/en/references/text-processing__feature_calcers ; https://arxiv.org/abs/2307.14338 . Retrieval inspirado en vecinos, no implementación de TabR.
