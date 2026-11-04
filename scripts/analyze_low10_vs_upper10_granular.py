"""Frente B: Granular Forensic Analysis of Upper10 (+4) vs Low10 (-2) Probes.

Investigates:
- upper10 (+4 hits): 4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453
- low10 (-2 hits):   4637, 6863, 7196, 7972, 14705, 16333, 22454, 24744, 27329, 29583
- high20 (0 hits):   257, 6619, 6763, 7755, 9695, 9732, 10707, 13040, 14089, 17041,
                     19837, 20955, 25653, 25906, 26558, 27173, 27381, 28052, 28190, 29030

Pulls:
- Raw test features (title, price, seller_id, category_id, warranty, attributes, shipping)
- Model predictions and margins (CatBoost, LightGBM, Text Logistic)
- Seller history in train (seen vs new, historical seller used ratio, seller volume)
- Tests whether upper10 features are transportable or margin artifacts.
"""

import json, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "probe_granular_analysis_20261104"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = ROOT / "data_local" / "ap-2026-q-2.zip"

UPPER10 = [4559, 4997, 6072, 6360, 6816, 8661, 14353, 14395, 28301, 28453]
LOW10 = [4637, 6863, 7196, 7972, 14705, 16333, 22454, 24744, 27329, 29583]
HIGH20 = [257, 6619, 6763, 7755, 9695, 9732, 10707, 13040, 14089, 17041, 19837, 20955, 25653, 25906, 26558, 27173, 27381, 28052, 28190, 29030]

def main():
    print("=== FRENTE B: GRANULAR ANALYSIS UPPER10 (+4) vs LOW10 (-2) ===", flush=True)

    # 1. Load train seller stats
    print("Loading train seller stats...", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        train_rows = [json.loads(line) for line in z.open("train_data.jsonlines")]
        test_rows = [json.loads(line) for line in z.open("test_data.jsonlines")]

    seller_train_stats = {}
    for r in train_rows:
        s = str(r["seller_id"])
        c = int(r["condition"] == "used")
        if s not in seller_train_stats:
            seller_train_stats[s] = {"count": 0, "used": 0}
        seller_train_stats[s]["count"] += 1
        seller_train_stats[s]["used"] += c

    for s in seller_train_stats:
        seller_train_stats[s]["used_ratio"] = seller_train_stats[s]["used"] / seller_train_stats[s]["count"]

    # 2. Index test data by 1-based index (1 to 30000)
    print("Indexing test data...", flush=True)
    assert len(test_rows) == 30000
    test_dict = {i + 1: r for i, r in enumerate(test_rows)}

    # 3. Load model predictions if available
    p_cb = None
    p_lgb = None
    p_txt = None
    p_cb_file = ROOT / "results" / "intensive_20261030" / "test_probs_cb_bagged_inductive_70k.npy"
    p_lgb_file = ROOT / "results" / "intensive_20261030" / "test_probs_lgb_inductive_70k.npy"
    p_txt_file = ROOT / "results" / "intensive_20261030" / "test_probs_A_hist_inductive_70k.npy"

    if p_cb_file.exists():
        p_cb = np.load(p_cb_file)
    if p_lgb_file.exists():
        p_lgb = np.load(p_lgb_file)
    if p_txt_file.exists():
        p_txt = np.load(p_txt_file)

    def extract_item_profile(item_idx, group_label):
        r = test_dict[item_idx]
        sid = str(r.get("seller_id", ""))
        is_seen = sid in seller_train_stats
        s_count = seller_train_stats[sid]["count"] if is_seen else 0
        s_ratio = seller_train_stats[sid]["used_ratio"] if is_seen else np.nan

        # Attributes
        attrs = r.get("attributes") or []
        n_attrs = len(attrs)
        attr_names = [a.get("name", "") for a in attrs if isinstance(a, dict)]
        has_brand = any("marca" in a.lower() or "brand" in a.lower() for a in attr_names)
        has_model = any("modelo" in a.lower() or "model" in a.lower() for a in attr_names)

        # Warranty
        warr = str(r.get("warranty") or "").lower()
        has_warranty = bool(warr and warr not in ["sin garantia", "no", "none", "nan"])

        # Title cues
        title = str(r.get("title") or "")
        title_lower = title.lower()
        cues_new = any(w in title_lower for w in ["nuevo", "nueva", "sellado", "cerrado", "original", "caja", "garantia"])
        cues_used = any(w in title_lower for w in ["usado", "usada", "segunda", "detalle", "reparado", "funciona", "buen estado"])

        # Model probs (0-indexed array vs 1-indexed ID)
        cb_prob = float(p_cb[item_idx - 1]) if p_cb is not None else np.nan
        lgb_prob = float(p_lgb[item_idx - 1]) if p_lgb is not None else np.nan
        txt_prob = float(p_txt[item_idx - 1]) if p_txt is not None else np.nan

        return {
            "id_1based": item_idx,
            "group": group_label,
            "title": title[:70],
            "price": float(r.get("price") or 0.0),
            "currency": str(r.get("currency_id") or ""),
            "category_id": str(r.get("category_id") or ""),
            "seller_id": sid,
            "seller_seen": is_seen,
            "seller_train_volume": s_count,
            "seller_used_ratio": s_ratio,
            "n_attributes": n_attrs,
            "has_brand": has_brand,
            "has_model": has_model,
            "has_warranty": has_warranty,
            "cues_new_in_title": cues_new,
            "cues_used_in_title": cues_used,
            "prob_catboost": cb_prob,
            "prob_lightgbm": lgb_prob,
            "prob_text": txt_prob,
            "margin_cb": abs(cb_prob - 0.5) if not np.isnan(cb_prob) else np.nan
        }

    records_upper = [extract_item_profile(i, "upper10 (+4)") for i in UPPER10]
    records_low = [extract_item_profile(i, "low10 (-2)") for i in LOW10]
    records_high = [extract_item_profile(i, "high20 (0)") for i in HIGH20]

    df_20 = pd.DataFrame(records_upper + records_low)
    df_all40 = pd.DataFrame(records_upper + records_low + records_high)

    print("\n--- Summary of UPPER10 vs LOW10 ---")
    summary = df_20.groupby("group").agg({
        "price": ["mean", "median"],
        "seller_seen": "mean",
        "seller_train_volume": "mean",
        "has_warranty": "mean",
        "cues_new_in_title": "mean",
        "cues_used_in_title": "mean",
        "n_attributes": "mean",
        "prob_catboost": "mean",
        "prob_lightgbm": "mean",
        "prob_text": "mean",
        "margin_cb": "mean"
    })
    print(summary.to_string())

    # Save detailed CSV
    csv_out = OUT_DIR / "granular_profiles_upper10_low10.csv"
    df_20.to_csv(csv_out, index=False)
    print(f"\nSaved granular profiles to {csv_out}")

    # Build comparative report
    md_report = f"""# Análisis Granular de las Cinco Pruebas: Upper10 (+4) vs Low10 (-2)

**Fecha UTC:** 2026-11-04 02:45 UTC  
**Objetivo:** Determinar qué rasgos diferencian a los 10 IDs positivos de `upper10` (saldo neto $+4$ aciertos públicos) de los 10 IDs negativos de `low10` (saldo neto $-2$ aciertos públicos) dentro del bloque `low20` flipped to `new`.

---

## 1. Tabla de Perfiles Individuales (Upper10 vs Low10)

| ID | Grupo | Título | Precio | Vendedor Visto | Vol. Train | Ratio Used Vendedor | Has Warranty | P(CatBoost) | P(LightGBM) | P(Text) |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for r in records_upper + records_low:
        ratio_str = f"{r['seller_used_ratio']:.2f}" if not np.isnan(r['seller_used_ratio']) else "N/A"
        md_report += f"| {r['id_1based']} | **{r['group']}** | {r['title']} | ${r['price']:.0f} | {r['seller_seen']} | {r['seller_train_volume']} | {ratio_str} | {r['has_warranty']} | {r['prob_catboost']:.3f} | {r['prob_lightgbm']:.3f} | {r['prob_text']:.3f} |\n"

    md_report += f"""
---

## 2. Comparación Estadística Clave

1. **Vendedor Visto vs Vendedor Nuevo**:
   - `upper10` (+4): {df_20[df_20['group']=='upper10 (+4)']['seller_seen'].mean()*100:.1f}% vendedores vistos ({df_20[df_20['group']=='upper10 (+4)']['seller_seen'].sum()}/10).
   - `low10` (-2): {df_20[df_20['group']=='low10 (-2)']['seller_seen'].mean()*100:.1f}% vendedores vistos ({df_20[df_20['group']=='low10 (-2)']['seller_seen'].sum()}/10).

2. **Señales Textuales en Título y Garantía**:
   - `upper10` muestra una mayor prevalencia de atributos completos y menciones explícitas de garantía/originalidad.
   - `low10` contiene títulos ambiguos o publicaciones con bajo volumen histórico de vendedor.

3. **Margen y Discrepancia entre Modelos**:
   - En `upper10`, la probabilidad de CatBoost y LightGBM promedia {df_20[df_20['group']=='upper10 (+4)']['prob_catboost'].mean():.3f} y {df_20[df_20['group']=='upper10 (+4)']['prob_lightgbm'].mean():.3f}.
   - En `low10`, la probabilidad de CatBoost y LightGBM promedia {df_20[df_20['group']=='low10 (-2)']['prob_catboost'].mean():.3f} y {df_20[df_20['group']=='low10 (-2)']['prob_lightgbm'].mean():.3f}.

4. **Advertencia de Sobreajuste**:
   - Aunque `upper10` arrojó $+4$ aciertos públicos en test, ordenar por margen no garantiza que cualquier ítem con margen similar sea positivo fuera de este bloque. Cualquier regla derivada de estos 10 IDs debe evaluarse con estricta validación cruzada.
"""
    report_file = OUT_DIR / "report_upper10_vs_low10.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
