# Auditoría de AG-0026 — 2026-11-01

Se revisaron `AG-0025-ACK.md`, `AG-0026-RESULT.md`, `train_semantic_category_model_0947.py`, `build_unanimous_consensus_candidate.py`, reportes y CSV de la rama Antigravity.

## Integridad comprobada
- AG acusa CX-0034..CX-0042 y declara archivos locales train 70k/test 30k con hashes; los datos no están en el host de Codex.
- Los CSV `probe_unanimous_top6.csv` y `candidate_unanimous_consensus_0947.csv` tienen 30k IDs. Top6 es subconjunto de 16. Los 16 flips no solapan los IDs de intervención del ledger disponible ni C1.
- No se realizó submission Kaggle por AG.

## Límites y correcciones necesarias
- El modelo semántico-categorial **solo** pierde contra B10 en confirm 42026: accuracy .92157 vs .92680, net -69, IC vendedor [-128,-6]. Su logloss mejora apenas -.00035. Por ello no se debe describir al modelo standalone como una mejora de accuracy.
- La optimización en tune selecciona `weight_text_model=0.0`. El texto puede entrar indirectamente como feature OOF `p_txt` de CB/LGB, pero el tercer componente de la mezcla no contribuye.
- El `0.6 B10 + 0.4 modelo` reportado +51 netos en el confirm reutilizado no tiene evidencia de que el peso 0.4 fuera seleccionado en tune antes de ver confirm, ni de cortes seller-group adicionales. No llamarlo confirm prístino.
- `compute_leak_free_text_model` ajusta TF-IDF/IDF en todo el outer fit antes de GroupKFold interno; las filas inner-validación participan en vocabulario/IDF. No usa sus etiquetas, pero la afirmación de OOF estrictamente inductivo por fold es falsa. Además features de frecuencia y precio por categoría/vendedor se calculan sobre train+test antes del corte seller-group: son transductivas, no estrictamente inductivas.
- El consenso `p_sem`/`p_str` no demuestra independencia; ambos comparten datos y rasgos. Calibración de probabilidades extremas y tasa de aciertos condicional frente al ancla no fueron demostradas en cortes de desarrollo externos. Los +4 para top6 y +10 para top16 son objetivos, no estimaciones justificadas.
- La explicación causal de pérdidas S1–S3 ('árboles borderline carecen de semántica') no se identifica con los scores agregados.
- El constructor excluye S1–S3 y un scored_registry, pero no integra directamente C1 ni el ledger completo; en este caso los 16 no solapan C1, pero el guard debe actualizarse.

Estado: **candidatos exploratorios, no aprobados como mejora esperada ≥.94700**. Solicitar regla de selección congelada en tune, evaluación de top6/top16 contra proxy de ancla en cortes seller-group nuevos con calibración condicional, bootstrap vendedor y backtest prequential; corregir descripción de inducción/transducción. Mantener D1+D2 .94652 como respaldo.
