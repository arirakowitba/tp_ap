# Ronda 112 — fusión logística de probabilidad base e historial del vendedor (27-10-2026)

Como arquitectura alternativa a incluir las estadísticas leave-one-out del vendedor dentro de HistGB, entrené una regresión logística con logit de HistGB+FastText, logit de la proporción de etiquetas del vendedor suavizada con 1, log conteo e interacciones. Se ajustó **solo en tune** de cada corte, usando exclusivamente etiquetas de fit para el historial del vendedor; confirm quedó fuera de ese entrenamiento. Para vendedores nuevos se preservó la base.

| Corte | Aciertos confirm vistos: base | HistGB con señal LOO | Fusión logística | Neto fusión−HistGB LOO | Log loss LOO→fusión |
|---|---:|---:|---:|---:|---:|
| 105 | 8635/9175 | 8778 | 8740 | −38 | .118881→.127799 |
| 108 | 8656/9215 | 8826 | 8800 | −26 | .113854→.117880 |
| Temporal 110 | 7439/7888 | 7562 | 7542 | −20 | .109166→.111545 |

La fusión mejora claramente la base, pero pierde en los tres confirm frente al modelo LOO ya elegido, tanto en aciertos como en log loss. Se descarta como candidata. Los cortes se solapan y no constituyen réplicas independientes. Código y predicciones pareadas: Library `libfile_1e8ad6301cc481919b8b2df2227f5041`, SHA-256 `a9082f68b68f54597443ba5d1c61e3a0a56803ed8a8763436be9b3a9b5819942`. Sin labels de test ni envío a Kaggle.