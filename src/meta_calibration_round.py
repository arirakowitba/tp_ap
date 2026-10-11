"""Fit calibration meta-models on seed42 holdout predictions and evaluate on seed731 confirmation.

This uses only persisted out-of-fold/holdout probabilities, not raw rows. It must not be
interpreted as a new feature model or as an independent test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "optimization" / "holdout_predictions.csv"
CONFIRMATION_PATH = ROOT / "results" / "round_v3" / "prior_confirmation_predictions.csv"
OUT = ROOT / "results" / "round_v4"


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _best_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.35, 0.651, 0.001)
    scores = np.array([accuracy_score(y, probability >= q) for q in thresholds])
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "logloss": float(log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7))),
    }


def run() -> pd.DataFrame:
    selection = pd.read_csv(SELECTION_PATH)
    confirmation = pd.read_csv(CONFIRMATION_PATH)
    for frame, name in ((selection, "selection"), (confirmation, "confirmation")):
        required = {"y_used", "probability_used"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} predictions must contain {sorted(required)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} predictions contain null labels/probabilities")
        if not frame["probability_used"].between(0, 1).all():
            raise ValueError(f"{name} probabilities must be in [0, 1]")

    ys = selection["y_used"].to_numpy(dtype=int)
    ps = selection["probability_used"].to_numpy(dtype=float)
    yc = confirmation["y_used"].to_numpy(dtype=int)
    pc = confirmation["probability_used"].to_numpy(dtype=float)

    platt = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    platt.fit(_logit(ps), ys)
    ps_platt = platt.predict_proba(_logit(ps))[:, 1]
    pc_platt = platt.predict_proba(_logit(pc))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(ps, ys)
    ps_iso = isotonic.predict(ps)
    pc_iso = isotonic.predict(pc)

    variants = [
        ("raw_fixed_050", ps, pc, 0.5),
        ("raw_threshold_selected_seed42", ps, pc, _best_threshold(ys, ps)[0]),
    ]
    for name, psel, pconf in (
        ("platt", ps_platt, pc_platt),
        ("isotonic", ps_iso, pc_iso),
    ):
        variants.append((f"{name}_threshold_selected_seed42", psel, pconf, _best_threshold(ys, psel)[0]))

    records = []
    for name, psel, pconf, threshold in variants:
        s = _metrics(ys, psel, threshold)
        c = _metrics(yc, pconf, threshold)
        records.append({
            "candidate": name,
            "threshold_selected_on_seed42": threshold,
            **{f"selection_{key}": value for key, value in s.items()},
            **{f"confirmation_{key}": value for key, value in c.items()},
        })

    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT / "meta_calibration_metrics.csv", index=False)
    metadata = {
        "selection_rows": len(selection),
        "confirmation_rows": len(confirmation),
        "selection_source": "optimization/holdout_predictions.csv",
        "confirmation_source": "results/round_v3/prior_confirmation_predictions.csv",
        "calibration_fit": "seed42 holdout only",
        "threshold_selection": "seed42 holdout only",
        "confirmation_used_for_tuning": False,
        "independent_test_caveat": "Both partitions reuse the same competition train set and prior model family.",
        "promoted": False,
        "goal_95_reached": False,
    }
    (OUT / "meta_calibration_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
