# Tanda3 — estado de reanudación
Datos y artefactos en /workspace/scratch/7bf0ae12b2f1. IdentidaddatasetSHA en training_round2/deep_teacher/data_fingerprint.json.

Finalizados: segunda partición fresh (native92.679%, NBblend92.943%, meta2logits93.243; no confundir con metas de5ramas anteriores), sparse descartadoaccuracy, specialistqty cerrado porausencia mejora tune yproceso interrumpido.

Deep teacher: strict OOF3fold porstage,113features,confirm94.2357%,new89.4786%. Artefactosfullteacher+CBMya terminados y auditados. Dropout.2:94.25%, +2vsdeep,noevidenciasuperioridad; fulltest listo. Fields título/garantía:94.3643%, LL.145954; exportfullenproceso.

TPE: optimize_deep_tpe_round3.py, SQLitepersistente6trials2threads,logdeep_tpe_round3.log; elegirsolo weightedtune yconfirmaruna vez, snapshots30s. Parent/rootbase ydropoutNOduplicar profesores. Fieldsusaexport_field_teachers_full_round3.py para cerrarfullsiinterrumpido.

Composiciónfinal pendiente: compose_final_models_round3.py espera vectoresTPE+fields. Metanew-only5ramas sobre7ktune ymeta hist+5sobre2098common; C[.01,.1,1]CV,threshold.5. No selección sobreconfirm. Histórico test producciónselected híbrida70kbase/auxcongelados; withoutpublic@.505 esreferenciacorrecta.

Calibraciónauditoría: bajarLL/BriernoaseguraECEmejor; blendfijo85native15NB mejoró+27/+37en2particionesquecomparten2909filas, NO sumarcomoindependientes.

ArchivoLibrary checkpoint_entrenamiento_20261021.zip idlibfile_50719e31e274819185c823b430bb1564 última versión3 antesdeestatanda, próximausarpreparedhelper si>50MiB. Repo rama checkpoint/training-20261022; mainactualización bloqueadaauto-review, no volveraintentar sinautorizaciónespecífica.
