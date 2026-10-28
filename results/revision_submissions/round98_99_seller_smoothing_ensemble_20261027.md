# Rondas 98–99 — contraste adicional del suavizado y promedio (27-10-2026)

En el mismo corte de R96 (42k fit, 14k tune, 14k confirm) entrené una variante con suavizado 5, sin cambiar la regla de acceso: se usa para vendedores vistos; los nuevos retienen HistGB+FastText.

| Variante | Tune vistos, neto vs base | Confirm vistos, fixes/breaks, neto | Log loss confirm vistos |
|---|---:|---:|---:|
| R96 suavizado 20 | +168 | 260/98, +162 | .150816→.107426 |
| R98 suavizado 5 | +163 | 258/91, +167 | .150816→.106902 |
| Promedio 50/50 | +168 | neto +165 | entre ambas variantes |

En los dos cortes comparables, tune favoreció 20 por pocos aciertos y confirm favoreció 5 por pocos. Esto no identifica con solidez un mejor suavizado; la candidata 51 de R97 sigue siendo la primera prueba predefinida. El promedio es una alternativa, no una mejora confirmada.

Se reentrenó suavizado 5 en las 70k etiquetas y se generó candidata 52: base exacta 48 para vendedores nuevos y promedio de probabilidades 5/20 para vistos. Tiene 370 cambios vs 48 y solo 55 vs candidata 51; su CSV tiene SHA-256 `4a6055ca754154df0fc6f902f7e1de29c4027eb4319ca2477100b771548b45ab`, Library `libfile_10db7d43f85c8191bbff723ff3cfe2bb`. Checkpoint `libfile_8ae32f5a51d48191b9efde0893983852`, SHA-256 `90458fe3f5e290c1689efed07d4d9785d48f4623797fc35966f7e198b6b099cb`.

No hay score de 51/52 ni labels de test. La 52 solo tendría sentido después de ver el resultado agregado de 51; al diferir en 55 decisiones, enviar ambas juntas sería un uso pobre de cuota. No se ha enviado ninguna a Kaggle desde este entorno.