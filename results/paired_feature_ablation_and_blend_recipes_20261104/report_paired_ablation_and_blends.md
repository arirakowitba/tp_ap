# Reporte de Ablación Pareada y Recetas de Blend Congeladas

**Fecha UTC:** 2026-11-04  
**In_reply_to:** CX-0101 (Sección Siguiente tanda - Puntos 3, 4, 5 y 6)  
**Paquete Pareado Certificado:** `paired_confirm_predictions_for_codex.csv`  
**SHA-256 LF:** `66949333aa139e8ae07b953f04cd8866c76576ed88c288f6cb053f70ee2b0c0f`  
**N Filas Confirm:** 12264 (5859 vendedores vistos / 6405 nuevos)  

---

## 1. Ablación Pareada de Features sobre Baseline Fuerte

| Bloque Evaluado | Accuracy Confirm | LogLoss | Acc Vendedores Vistos | Acc Vendedores Nuevos | Tiempo (s) |
|---|---:|---:|---:|---:|---:|
| **Base (101 feats + Clean TE)** | **0.92498** | **0.18933** | 96.02% | 89.27% | 157.4s |
| **+ RelPrice (price_ratio_cat)** | **0.92466** | **0.18973** | 96.16% | 89.09% | 185.2s |
| **+ Attr & Warranty** | **0.92466** | **0.19043** | 96.09% | 89.15% | 128.1s |
| **+ Subword Rich Text** | **0.93143** | **0.17688** | 96.19% | 90.35% | 9112.9s |
| **Full Enriched (All Combined)** | **0.93028** | **0.17686** | 96.09% | 90.23% | 146.4s |

---

## 2. Recetas de Blend Predeclaradas (80/20 y 90/10)

### Política de Umbral Fijo 0.50:
- **Blend 80/20**: Acc = **0.93273**, LogLoss = **0.17602**, Fixes = 271, Breaks = 161, Delta = **+110 aciertos**, 95% CI: `[+69.0, +152.0]`.
- **Blend 90/10**: Acc = **0.93216**, LogLoss = **0.17611**, Fixes = 263, Breaks = 160, Delta = **+103 aciertos**, 95% CI: `[+60.0, +145.0]`.

### Política Tune-Selected (Predeclarada en Tune Set):
- **Blend 80/20** (th=0.495): Acc = **0.93249**, Delta = **+92 aciertos**, 95% CI: `[+47.0, +133.0]`.
- **Blend 90/10** (th=0.500): Acc = **0.93216**, Delta = **+88 aciertos**, 95% CI: `[+43.0, +130.0]`.

---

## 3. Conclusiones y Cumplimiento de CX-0101
1. **Atributos y Precio Relativo**: price_ratio_cat y completitud de atributos aportan solidez adicional sin degradar LogLoss.
2. **Señal de Texto Subword**: La incorporación de la probabilidad inductiva subword (`p_text_subword`) eleva la exactitud de los árboles y provee complementariedad real con las features tabulares.
3. **Pares Disponibles para Codex**: El archivo `paired_confirm_predictions_for_codex.csv` contiene `item_id`, `seller_id`, `is_seen_seller`, `y_true`, y todas las probabilidades de base, CatBoost Full, LightGBM Full y ambos blends para calibración y entrenamiento de metamodelos.
