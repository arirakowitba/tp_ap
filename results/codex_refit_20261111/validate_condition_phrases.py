"""Paired LightGBM ablation of condition phrases and normalized warranty lengths."""
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile

import numpy as np

from validate_variant_inventory import METADATA, PICTURES, load, run, sha

PHRASES = {
    "unopened": r"sin (?:abrir|estrenar|desembalar)|(?:caja|embalaje) (?:cerrad[oa]|sellad[oa])",
    "like_new": r"(?:como|casi) nuev[oa]s?|igual a nuev[oa]|impecable|excelente estado",
    "wear": r"poco uso|horas? de uso|mes(?:es)? de uso|marcas? de uso|desgaste|rayad[oa]s?",
    "fault": r"no funciona|para reparar|para repuestos|(?:pantalla|vidrio) rot[oa]|sin funcionar",
    "refurbished": r"reacondicionad[oa]s?|refurbished|remanufacturad[oa]s?|recertificad[oa]s?",
    "warranty_missing": r"sin garantia|no (?:tiene|posee|ofrece|incluye) garantia",
    "original_packaging": r"(?:caja|embalaje) original|con su caja|sin caja",
    "original_accessories": r"accesorios originales|con accesorios|sin accesorios|solo el equipo",
    "spare_part": r"repuesto para|compatible con|funda para|estuche para|cargador para",
    "unused": r"sin uso|nunca (?:se uso|usad[oa])|no usad[oa]s?",
}
PATTERNS = [re.compile(r"\b(?:" + phrase + r")\b") for phrase in PHRASES.values()]
DURATION = re.compile(r"\b(\d{1,3})\s*(dias?|mes(?:es)?|anos?)\b")


def normalize(text):
    plain = unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def phrase_features(df):
    result = []
    for title, subtitle, warranty in df[["title", "subtitle", "warranty_str"]].itertuples(index=False, name=None):
        fields = [normalize(x) for x in (title, subtitle, warranty)]
        row = [float(bool(pattern.search(field))) for field in fields for pattern in PATTERNS]
        durations = [int(n) * (365 if unit.startswith("ano") else 30 if unit.startswith("mes") else 1)
                     for n, unit in DURATION.findall(fields[2])]
        row += [np.log1p(max(durations)) if durations else np.nan, float(len(durations))]
        result.append(row)
    return np.asarray(result, dtype=np.float32)


def main():
    archive = Path(sys.argv[1])
    paths = list(map(Path, sys.argv[5:]))
    assert len(paths) == 8
    sources = (Path("recent_seller_history"), Path("structural_heldout_confirm"))
    for source in sources:
        assert sha(archive) == json.loads((source / "report.json").read_text())["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        df, extra = load(z, "train")
    assert len(df) == 70000 and df.id.is_unique
    features = phrase_features(df)
    extra = extra[:, sorted(METADATA + PICTURES)]
    destination = "condition_phrase_evaluation"
    out = {
        "scope": "Train-only replay of two archived panels, fixed parameters and threshold. "
                 "The exclusive warm-heavy subset and CX-0177 have disjoint IDs. No public forecast.",
        "archive_sha256": sha(archive), "runner_sha256": sha(Path(__file__)),
        "shared_runner_sha256": sha(Path(__file__).with_name("validate_variant_inventory.py")),
        "phrase_coverage": {field: int(np.any(features[:, i*len(PATTERNS):(i+1)*len(PATTERNS)] != 0,
                                              axis=1).sum())
                            for i, field in enumerate(("title", "subtitle", "warranty"))},
        "panels": {source.name: run(df, extra, features, source, paths if i == 0 else paths[:-1],
                                   i == 0, destination) for i, source in enumerate(sources)},
    }
    for panel in out["panels"].values():
        panel["metrics"].pop("variations")
        panel["metrics"].pop("fresh_variations")
    (Path(destination) / "report.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
