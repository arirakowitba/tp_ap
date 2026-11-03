# Bag3: auditoría de solapamiento y alcance de intervención — 2026-11-03

Código: `scripts/audit_bag3_overlap_20261103.py`. Insumos privados pareados de tres cortes (Library `libfile_8ef362b0c9348191bf4cdc97b7d42437`), probabilidades test full70k (Library `libfile_df465119624c8191b9ca945a539ea4da`), ancla D1+D2 SHA `57172f209ff5018d4ceef8c57172cc731828cc78eb9cb02dedccff4a51292502`, ledger 78 de `results/historical_text_seller_full70k/submission_ledger_04_current.json`.

## Validación pareada, umbral común fijo

Los tres confirm contienen 37.762 filas, pero sólo **30.017 item_id únicos**; 7.075 aparecen en más de un corte. Para no contarlos varias veces, promedié el delta de aciertos de cada ítem entre sus apariciones y remuestreé vendedores (3.000 réplicas). No reutilicé las etiquetas confirm para elegir el umbral: .475 y .5 son umbrales de sensibilidad preexistentes.

| Umbral | Saldo por filas duplicadas | Saldo por item único (media de apariciones) | IC 95% bootstrap vendedor |
|---|---:|---:|---:|
| .475 | +29 | +21,83 | [-2,17; +45,34] |
| .5 | +26 | +17 | [-5,00; +38,50] |

El bagging conserva una señal positiva puntual, pero los intervalos de estos agregados incluyen cero. Las comparaciones por corte de +28/+14/+6 usan umbrales elegidos en tune y no constituyen tres confirmaciones independientes.

## Test sin etiquetas

Al umbral analítico .475, hay 137 diferencias bag3 vs su primer modelo. En 79 el ancla pública D1+D2 ya coincide con bag3; el componente nuevo del bagging cambiaría **58 IDs** del ancla: 26 vendedor seen, 32 new. De ellos, 44 ya aparecen en algún vector del ledger y 14 no; sólo 6 tienen margen de bag respecto de .475 >= .05. Ninguno de los 58 coincide con los cuatro efectos condicionalmente identificados por 33/40 (1383, 10965, 20534, 20806).

El recuento de desacuerdos y los scores agregados no permiten estimar el efecto público de esos 58 flips como etiquetas individuales. No se promueve ni genera una submission por esta auditoría. D1+D2 .94652 permanece ancla pública; próximo objetivo train-only es una base diversa de alcance amplio frente a la arquitectura fuerte histórica y bag3. AG-0040 sigue pendiente de corrección: seleccionó peso/umbral en confirm y comparó contra una base débil.

## Reproducción

Con los dos checkpoints privados disponibles en las rutas que lee el script:

```bash
python scripts/audit_bag3_overlap_20261103.py
```

No se consultó Kaggle ni se gastó cuota.
