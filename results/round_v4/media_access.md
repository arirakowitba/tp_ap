# Acceso a imágenes y videos: ronda 4

El archivo contiene enlaces de fotos y `video_id` de YouTube, no imágenes ni clips binarios adjuntos. El inventario se realizó fila por fila sin usar `condition` ni consultar publicaciones actuales.

| Partición | Filas | Con fotos | Con video_id |
|---|---:|---:|---:|
| train | 70,000 | 69,459 (99.23%) | 2,102 (3.00%) |
| test | 30,000 | 29,752 (99.17%) | 883 (2.94%) |

## Evidencia de acceso

- 16 solicitudes iniciales: cuatro páginas `embed` de YouTube y doce URLs de imágenes Mercado Libre (thumbnail/foto original, HTTP, equivalente HTTPS y `secure_url` del archivo).
- Las doce respuestas de imágenes fueron **403**, con cero firmas de imagen válidas. Se detuvo la descarga masiva de fotos.
- Las cuatro páginas `embed` devolvieron 200 HTML: eso acredita una página, **no un stream reproducible ni frames**.
- Dos comprobaciones adicionales de páginas `embed`, limitadas a 256 KiB, no expusieron `streamingData` ni `playabilityStatus`. Estado real de reproducción: no verificado.
- Por indicación explícita del coordinador, se hizo un seguimiento acotado de dos IDs: dos páginas `watch` y dos thumbnails públicos del mismo video. Las páginas fueron 200 HTML sin datos de streaming dentro del límite; los thumbnails fueron 200 `text/html` de 195 bytes idénticos, con firma de imagen inválida.
- Se verificaron además cuatro variantes en el CDN actual `http2.mlstatic.com` de dos IDs de fotos originales, siguiendo la migración de dominios documentada por Mercado Libre. Todas respondieron **403**; no se prosiguió con descargas masivas.
- Total de esta sesión: **26 solicitudes acotadas** (16 iniciales + 6 verificaciones de dos videos + 4 comprobaciones de migración del mismo asset); cero imágenes descargadas, cero frames, cero modelos visuales o temporales entrenados. No se usaron credenciales, proxies adicionales ni cambios de identidad para superar bloqueos.

Los registros separados conservan HTTP, MIME, bytes, hashes de muestras y duración: `media_access_probes.json`, `video_player_checks.json`, `video_asset_followup.json`, `media_cdn_migration_checks.json`. No se guardaron títulos, descripciones, engagement ni etiquetas externas de YouTube.

## Metadatos utilizables

`media_features_train.csv` y `media_features_test.csv` contienen **50 variables numéricas más row_index**, alineadas con las 70.000/30.000 filas originales. El extractor `src/multimodal_access_round4.py` permite regenerarlos; `media_feature_validation.json` verifica alineación secuencial, cantidad de filas y ausencia de infinitos.

Variables: cantidad de fotos; presencia de thumbnail/video; proporción de URLs seguras y quality no vacío; dimensiones `size` y `max_size` con media, mínimo, máximo y dispersión; área y relación ancho/alto; proporción de fotos verticales/horizontales/cuadradas/grandes; expansión de resolución original frente a la mostrada. Dimensiones faltantes quedan NaN. No se incluyen IDs de imágenes, IDs de video ni fechas extraídas de URLs como predictores.

Estas variables describen **metadatos multimedia**, no embeddings del contenido visual. El baseline ya tenía cantidad de fotos y algunas dimensiones medias/máximas; los agregados de dispersión, orientación y expansión añaden hipótesis nuevas. Las variables constantes deben descartarse usando solamente el conjunto de entrenamiento de cada split.

## Modelos preparados conceptualmente

Si los píxeles fueran accesibles, la primera opción bajo presupuesto CPU sería **CLIP ViT-B/32** congelado: PyTorch local, `encode_image`, normalización L2 y regresor/clasificador entrenado solamente en los folds correspondientes; una imagen original por producto primero. La documentación oficial muestra CPU y extracción separada de features. **SigLIP base patch16 224** es una segunda opción de representación imagen/texto, no una promesa de mayor accuracy en este dataset. El paper introduce una pérdida sigmoid por pares imagen/texto; el modelo base opera con resolución 224×224. Ninguno se ejecutó porque falta acceso verificable a los assets.

Para videos: obtener un stream público válido del ID existente, extraer frames locales con ffmpeg y agregar embeddings de CLIP por media/dispersión, dejando un flag de faltante. Un thumbnail de YouTube, si estuviera disponible, es una sola imagen auxiliar y nunca equivale a análisis temporal. Con apenas ~3% de filas con video, todo candidato necesita validación global y en ese subconjunto.

No se utiliza ONNXRuntime; el antecedente de telemetría rechazado de la ronda 3 se respeta. PyTorch u OpenVINO locales serían los backends futuros.

## Fuentes primarias consultadas

- OpenAI, código y documentación CLIP: https://github.com/openai/CLIP
- Zhai et al., Sigmoid Loss for Language Image Pre-Training: https://arxiv.org/abs/2303.15343
- Pesos de Google y model card de implementación: https://huggingface.co/google/siglip-base-patch16-224 (la tarjeta fue escrita por Hugging Face; el paper anterior es la fuente técnica primaria).
- Mercado Libre, formato histórico `video_id = YOUTUBE_ID_HERE`: https://developers.mercadolibre.com.ar/en_us/list-products
- Mercado Libre, migración documentada del dominio legacy y prefijo `D_NQ_NP_`: https://developers.mercadolibre.com.ar/es_ar/gestiona-preguntas-respuestas/actualizacion-de-dominios-de-imagenes y https://developers.mercadolibre.com.ar/es_ar/administra-areas-de-cobertura/trabajar-con-imagenes
- Google, ejemplos oficiales de URLs `i.ytimg.com/vi/ID/default.jpg`: https://developers.google.com/youtube/v3/getting-started

## Reproducción

```bash
python src/multimodal_access_round4.py --data-dir ../data --output results/round_v4 --max-probes 20 --video-followup --cdn-migration
```

La ejecución futura limita todo el set combinado a 20 solicitudes. Los 26 registros históricos incluyen dos relecturas de las páginas `embed` para comprobar reproducción y el seguimiento posterior del CDN actual; se documentan por separado y no se ocultan en el contador inicial.
