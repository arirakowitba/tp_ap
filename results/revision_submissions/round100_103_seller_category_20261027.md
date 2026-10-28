# Rondas 100–103 — historial por vendedor y categoría (27-10-2026)

Hipótesis: añadir una media leave-one-out de condición por par vendedor–categoría, suavizada con la media leave-one-out del vendedor (peso 10), podría mejorar la compuerta de vendedores vistos. La fila propia se excluye de ambas medias en fit; tune/confirm solo usan etiquetas de fit. La base y el modelo simple de vendedor se reentrenaron en los mismos dos cortes por filas, con 42k fit, 14k tune y 14k confirm.

| Corte | Variante | Neto tune vistos vs HistGB+FastText | Neto confirm vistos vs HistGB+FastText | Log loss confirm vistos |
|---|---|---:|---:|---:|
| 20261038/39 | Vendedor simple | +194 | +155 | .106881 |
| 20261038/39 | Vendedor + categoría | +193 | +168 | .105756 |
| 20261040/41 | Vendedor simple | +151 | +172 | .111369 |
| 20261040/41 | Vendedor + categoría | +158 | +171 | .111077 |

En comparación **directamente pareada** con el modelo simple, la categoría dio −1 y +7 en tune; +13 y −1 en confirm. Los bootstrap por vendedor del delta de confirm fueron [−4,+30] y [−19,+15], ambos incluyen cero. Los cortes se solapan. No promuevo la señal de categoría ni sustituyo la candidata 51: el log loss mejoró ligeramente, pero no se confirmó ganancia de aciertos. Para vendedores nuevos, la compuerta mantiene la misma predicción base. Código, pares y hashes en Library `libfile_35ac1f8d0b448191832ea0a4614b9338`, SHA-256 `16b3226a1e3b6404ddaa818ddaf56c385e2070a9cff5b0d31189c4d3aa45d84b`. No se usaron labels de test ni cuota Kaggle.