# Tanda2 — continuación
Ramas activas: especialista qty<=1 (2configs CatBoost), robustezfreshseed20261023 (CatBoost+NB+meta enfit49k/tune7k/confirm14k). No detener salvo memoria. Logs results/specialist_round2.log y fresh_round2.log.

Auditoría provenance.md corrige viejo test49k y umbralhistorico. Referencia preferida refit49k_without_public, umbral.505; producción production_probabilities.p_used_full70k_selected. Histórica producción híbrida base70k+auxiliarescongelados, no decir todo70k.

Meta coherent selectedC.1 tune3CV da94.3574 vs93.9715 sobre4147(+16neto), LL.150958vs.155061, p.1093 noajustado. Artefactos historical_fullaligned. Resultado prometedor, no concluyente; reutilizado. Otro metaC1fijo corrected da94.0921(+5). No escoger entreellos porconfirm.

Antes de gastar cupo candidato, validar robustez nueva partición. Submission26 sigue preparada (0.94547inferido). No hay nuevos scoresKaggle. No labelsrecuperados ni saldosindividuales enfit.

Persistir code/report en rama checkpoint/training-20261022; main última actualización rechazada porautoreview. Modelos/arrays guardar checkpointLibrary archivo existente libfile_50719e31e274819185c823b430bb1564, última versión1antes deestatanda.
