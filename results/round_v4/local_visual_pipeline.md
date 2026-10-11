# Embeddings visuales locales opcionales

`src/local_visual_embeddings_round4.py` deja preparada la extracción real de imágenes y videos locales del dataset. **No se ejecutó inferencia visual sobre la competencia:** no hay píxeles accesibles ni pesos CLIP instalados en esta sesión. Esta implementación no es una mejora validada del score.

## Entrada y alineación

Un CSV UTF-8 necesita cuatro columnas:

```csv
split,row_index,path,kind
train,11,assets/product_11.jpg,image
train,11,assets/product_11.mp4,video
test,40,assets/product_40.mp4,video
```

`row_index` es el índice original de base cero; `split` solamente train/test; `kind` solamente image/video. Rutas relativas se resuelven contra el directorio del CSV. Pueden existir varias imágenes distintas por producto, pero duplicar el mismo asset produce error. Los IDs de publicación/video/foto no son predictores. El script rechaza URLs remotas y jamás descarga archivos o pesos.

## Validación sin modelo

```bash
python src/local_visual_embeddings_round4.py \
  --manifest assets.csv --output results/local_visual_validation \
  --validate-only --deadline 2026-10-11T03:20:50Z
```

Pillow verifica imágenes y ffprobe inspecciona videos locales. Rutas inexistentes o archivos corruptos se registran sin presentarlos como assets válidos. El archivo `local_visual_validation_tests.json` recoge diez comprobaciones realizadas sin PyTorch, CLIP o pesos: esquema incompleto, índices negativos, tipo inválido, URL remota, duplicados, archivo inexistente, imagen corrupta, conjunto sin assets válidos validación de imagen local sin inferencia y extracción/caché de tres frames reproducibles de un video sintético local. La pequeña imagen y el video sintéticos de QA solamente comprueban formato y decodificación; no se usaron como datos de competencia ni para generar embeddings.

Repetir comprobaciones: `python src/check_local_visual_manifest.py` (requiere Pillow y ffmpeg/ffprobe, sin PyTorch o pesos).

## Modelo, dependencias y ejecución

Dependencias opcionales: PyTorch CPU, torchvision compatible, Pillow, NumPy, paquete oficial OpenAI CLIP (incluye ftfy/regex/tqdm) y ejecutables ffmpeg/ffprobe. La instalación documentada oficialmente está en https://github.com/openai/CLIP; estas dependencias no se instalan automáticamente ni se añaden al camino obligatorio del modelo tabular.

Solo se acepta el checkpoint oficial **ViT-B/32**, ya descargado por quien ejecute, cuyo SHA-256 coincide exactamente con:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

URL oficial publicada por OpenAI:

https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

Fuente del mapeo y verificación SHA: https://github.com/openai/CLIP/blob/main/clip/clip.py. Se usa `clip.load(ruta_local, device='cpu', jit=False)` y el preprocesamiento oficial. No se utilizan ONNXRuntime, API de inferencia o títulos/metadatos actuales de YouTube.

```bash
python src/local_visual_embeddings_round4.py \
  --manifest assets.csv --output results/local_visual \
  --model-path /ruta/local/ViT-B-32.pt \
  --max-assets 512 --deadline 2026-10-11T03:20:50Z
```

CPU: dos threads de PyTorch y ffmpeg. El presupuesto inicial admite 512 assets, configurable. El deadline admite timestamp Unix o ISO con zona; se controla antes de cada etapa y limita subprocess de video. Para un límite estricto que también interrumpa una operación nativa de carga/inferencia en curso, el coordinador debe ejecutar este comando con su timeout de proceso.

## Imágenes, videos y caché

Imágenes: embedding congelado de 512 dimensiones, normalizado L2. Si hay varias fotos del producto, se promedia y renormaliza. Videos: tres frames locales en 10%, 50% y 90% de la duración informada por ffprobe; extracción ffmpeg con tamaño máximo 512×512. Cada frame registra timestamp, SHA-256 y ruta. Se promedian embeddings normalizados por video y después entre videos del mismo producto. Imagen y video permanecen como dos bloques separados.

La caché usa hashes del checkpoint, del contenido real de cada imagen/frame y de la regla de preprocesamiento; no usa nombres de archivos como identidad. Se registran versión de PyTorch y hashes del código CLIP cargado. Se validan forma, valores finitos y norma de cada vector recuperado de caché.

## Salida y faltantes

- `local_visual_validation.json`: inventario y errores de assets.
- `local_visual_encoded_assets.json`: assets realmente codificados y hashes/timestamps de frames.
- `local_clip_{train,test}_{image,video}.npz`: arrays **sparse** `row_index` y `embeddings[n,512]`, solo para productos codificados.
- `local_visual_coverage.csv`: todas las 70.000/30.000 filas, flags image_missing/video_missing y conteos de assets/frames codificados.
- `local_visual_run.json`: cobertura, pesos/modelo, deadline, ejecución real y ausencia de entrenamiento/descarga.

Un faltante no recibe un vector inventado de ceros. Para combinar posteriormente con LightGBM/CatBoost, se hace join por split/row_index; las coordenadas ausentes quedan NaN y la imputación/calibración se ajusta dentro de los folds. Esta rama solamente prepara representaciones: no entrena ni evalúa un modelo predictivo. El embedding temporal promedio es un resumen de tres imágenes; no modela movimiento, audio ni toda la secuencia.
