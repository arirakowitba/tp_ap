# Rondas 113–114 — historial reciente del vendedor (27-10-2026)

Probé añadir, a la señal leave-one-out de vendedor con suavizado 20, la proporción suavizada de las cinco observaciones etiquetadas más cercanas por `date_created` del mismo vendedor y su conteo. En fit se excluye la fila propia; en tune/confirm solo se usan etiquetas de fit. El control es el modelo simple de vendedor reentrenado en el mismo corte 42k/14k/14k. Vendedores nuevos conservan la base.

| Variante | Neto tune vistos vs HistGB+FastText | Neto confirm vistos vs HistGB+FastText | Log loss confirm vistos |
|---|---:|---:|---:|
| Vendedor simple | +147 | +141 | .110936 |
| Vendedor + vecinos temporales | +161 | +145 | .110757 |

La diferencia directamente pareada a favor de vecinos temporales fue +14 en tune (64 decisiones distintas) y **+4 en confirm** (68 distintas); bootstrap por vendedor del delta confirm [−12,+21]. No hay evidencia estable para sustituir la candidata 53, que además usa suavizado 1 y fue validada en varios cortes. Se descarta este agregado por ahora. Código, pares y hashes: Library `libfile_f4590115b4208191b310591d98315153`, SHA-256 `08c68145a7745e2ce85c69bd9e3e6329ebf9c38290d37cd53f52436dce06af21`. Sin test labels ni submission.