# Embeddings reales de títulos

Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, variante ONNX INT8 AVX2, 384 dimensiones. La inferencia se ejecuta localmente: el dataset no se envía a Hugging Face ni a otro servicio. Se descargan solamente pesos públicos y tokenizador.

Preparar dependencias opcionales: `pip install openvino==2026.4.1 tokenizers==0.23.3`.

Descargar y verificar la revisión congelada del modelo:

```bash
python src/download_embedding_model.py --model-dir ../models/title_minilm
```
Descargar a `models/title_minilm/` los archivos del modelo:

- `onnx/model_quint8_avx2.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `sentence_bert_config.json`
- `1_Pooling/config.json`

URL base: `https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/`.

Ejecutar desde el repo:

```bash
python src/embedding_round.py --data-dir ../data --model-dir ../models/title_minilm --output results/round_v3 --cache results/round_v3/cache_base
```

`--deadline` admite un timestamp UNIX para detener el proceso. El modelo genera embeddings sobre títulos sin acceder al target; después se usa la misma partición agrupada seed42. Se compara regresión logística sobre embeddings solos y LightGBM agregándolos a las variables existentes. Las métricas son de selección, no scores de Kaggle ni confirmaciones independientes.

Los pesos y las matrices derivadas no deben versionarse. Los CSV/JSON registran parámetros, probabilidades y resultados de cada configuración que finalizó.

La prueba visual se documenta aparte en `embedding_image_feasibility.json`: los intentos de descarga de fotografías devolvieron errores HTTP y no se entrenó ningún modelo visual.

La inferencia usa OpenVINO CPU directamente, sin importar ONNXRuntime ni instalar el paquete opcional de telemetría. Dos intentos con ONNXRuntime se bloquearon por telemetría incidental; ver `embedding_runtime_review.json`.
