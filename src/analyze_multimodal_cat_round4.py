"""Report the matched CatBoost media ablation using selection predictions only."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from catboost import CatBoostClassifier


def run(out):
    out = Path(out)
    rows, paired = [], []
    for path in sorted(out.glob("r4_cat_*.json")):
        result = json.loads(path.read_text())
        rows.append(dict(name=result["name"], accuracy=result["accuracy"],
                         iterations=result["params"]["iterations"],
                         seconds=result["seconds"], media_metadata=result["media_metadata"]))
        model = CatBoostClassifier()
        model.load_model(str(out / (result["name"] + ".cbm")))
        importance = np.asarray(model.get_feature_importance())
        pd.DataFrame({"feature": model.feature_names_, "importance": importance}).sort_values(
            "importance", ascending=False).to_csv(out / (result["name"] + "_importance.csv"), index=False)
    for config in ("enriched_d9", "native_text_d7"):
        a, b = (out / f"r4_cat_{config}_{suffix}_predictions.csv" for suffix in ("original", "media"))
        if not a.exists() or not b.exists():
            continue
        base, media = pd.read_csv(a), pd.read_csv(b)
        np.testing.assert_array_equal(base.row_index, media.row_index)
        np.testing.assert_array_equal(base.y, media.y)
        base_ok = (base.p >= .5) == base.y
        media_ok = (media.p >= .5) == media.y
        gained, lost = int((~base_ok & media_ok).sum()), int((base_ok & ~media_ok).sum())
        total = gained + lost
        paired.append(dict(config=config, corrected_errors=gained, introduced_errors=lost,
                           net_correct=gained-lost, delta_accuracy=float(media_ok.mean()-base_ok.mean()),
                           paired_exact_pvalue=float(binomtest(gained, total, .5).pvalue) if total else 1.))
    pd.DataFrame(paired).to_csv(out / "cat_round4_paired_diagnostics.csv", index=False)
    table = pd.DataFrame(rows)
    report = "# CatBoost: texto y metadatos de imágenes y videos\n\n"
    report += "Selección congelada: 49.000 filas de entrenamiento y 21.000 de evaluación, agrupadas por familia (semilla 42). Umbral fijo 0,5. Cada representación conserva los atributos y la proyección de texto SVD96 de los mejores CatBoost originales. No se consultó la partición de confirmación.\n\n"
    report += "La variante native agrega el procesamiento de texto propio de CatBoost. La variante media agrega estadísticas de las dimensiones y orientación de las imágenes, cantidad de fotos y presencia de video, tomadas exclusivamente del archivo original. **No se utilizaron píxeles ni fotogramas.**\n\n"
    report += "| Modelo | Metadatos media | Accuracy selección | Árboles congelados | Segundos |\n|---|---|---:|---:|---:|\n"
    for row in table.itertuples():
        report += f"| {row.name} | {'Sí' if row.media_metadata else 'No'} | {100*row.accuracy:.4f}% | {row.iterations} | {row.seconds:.1f} |\n"
    report += "\nComparación emparejada a umbral idéntico:\n\n| Representación | Errores corregidos | Errores nuevos | Saldo | Δ accuracy (pp) | p exacto |\n|---|---:|---:|---:|---:|---:|\n"
    for row in paired:
        report += f"| {row['config']} | {row['corrected_errors']} | {row['introduced_errors']} | {row['net_correct']} | {100*row['delta_accuracy']:.4f} | {row['paired_exact_pvalue']:.4f} |\n"
    report += "\nEl p emparejado es descriptivo: las configuraciones se seleccionaron en esta misma partición. La mejora debe verificarse con la configuración congelada en la partición reservada; estos resultados no representan el puntaje público de Kaggle.\n"
    (out / "cat_round4_report.md").write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
