#!/usr/bin/env python3
"""Tune-locked router for testing whether a challenger adds selective value.

The rule is selected only on tune pairs, then evaluated once on confirm pairs.
It never uses public scores or test pseudo-labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pairs(path: Path, y_col: str, id_col: str, base_col: str, challenger_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {id_col, y_col, base_col, challenger_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate IDs")
    y = df[y_col].to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path}: {y_col} must be binary 0/1")
    for col in (base_col, challenger_col):
        p = df[col].to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"{path}: invalid probabilities in {col}")
    return df


def mask_for(df: pd.DataFrame, base_col: str, challenger_col: str, rule: dict) -> np.ndarray:
    pb = df[base_col].to_numpy(float)
    pc = df[challenger_col].to_numpy(float)
    b = pb >= 0.5
    c = pc >= 0.5
    mask = b != c
    mask &= np.abs(pc - 0.5) >= rule["challenger_margin_min"]
    mask &= np.abs(pb - 0.5) <= rule["baseline_margin_max"]
    direction = rule["direction"]
    if direction == "0_to_1":
        mask &= (~b) & c
    elif direction == "1_to_0":
        mask &= b & (~c)
    return mask


def evaluate(df: pd.DataFrame, y_col: str, base_col: str, challenger_col: str, mask: np.ndarray) -> dict:
    y = df[y_col].to_numpy(int)
    b = df[base_col].to_numpy(float) >= 0.5
    c = df[challenger_col].to_numpy(float) >= 0.5
    routed = b.copy()
    routed[mask] = c[mask]
    delta = (routed == y).astype(int) - (b == y).astype(int)
    n = int(mask.sum())
    fixes = int((delta == 1).sum())
    breaks = int((delta == -1).sum())
    return {
        "n_routed": n,
        "fixes": fixes,
        "breaks": breaks,
        "net": fixes - breaks,
        "baseline_hits": int((b == y).sum()),
        "routed_hits": int((routed == y).sum()),
        "baseline_accuracy": float((b == y).mean()),
        "routed_accuracy": float((routed == y).mean()),
        "net_lcb_normal95": float((fixes - breaks) - 1.96 * math.sqrt(max(n, 1))),
    }


def seller_bootstrap(df: pd.DataFrame, y_col: str, base_col: str, challenger_col: str,
                     mask: np.ndarray, seller_col: str, seed: int, reps: int) -> list[float] | None:
    if seller_col not in df.columns:
        return None
    y = df[y_col].to_numpy(int)
    b = df[base_col].to_numpy(float) >= 0.5
    c = df[challenger_col].to_numpy(float) >= 0.5
    routed = b.copy()
    routed[mask] = c[mask]
    delta = (routed == y).astype(int) - (b == y).astype(int)
    by_seller = pd.Series(delta).groupby(df[seller_col].astype(str)).sum().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for i in range(reps):
        boot[i] = by_seller[rng.integers(0, len(by_seller), len(by_seller))].sum()
    return np.quantile(boot, [0.025, 0.975]).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True, type=Path)
    ap.add_argument("--confirm", required=True, type=Path)
    ap.add_argument("--base-col", required=True)
    ap.add_argument("--challenger-col", required=True)
    ap.add_argument("--y-col", default="y_true")
    ap.add_argument("--id-col", default="item_id")
    ap.add_argument("--seller-col", default="seller_id")
    ap.add_argument("--min-routed", type=int, default=20)
    ap.add_argument("--bootstrap-reps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    tune = load_pairs(args.tune, args.y_col, args.id_col, args.base_col, args.challenger_col)
    confirm = load_pairs(args.confirm, args.y_col, args.id_col, args.base_col, args.challenger_col)

    candidates = []
    for cm in np.arange(0.0, 0.401, 0.05):
        for bm in np.arange(0.05, 0.501, 0.05):
            for direction in ("both", "0_to_1", "1_to_0"):
                rule = {
                    "challenger_margin_min": round(float(cm), 4),
                    "baseline_margin_max": round(float(bm), 4),
                    "direction": direction,
                }
                mask = mask_for(tune, args.base_col, args.challenger_col, rule)
                metrics = evaluate(tune, args.y_col, args.base_col, args.challenger_col, mask)
                if metrics["n_routed"] >= args.min_routed:
                    candidates.append((metrics["net_lcb_normal95"], metrics["net"], -metrics["n_routed"], rule, metrics))

    no_op = {
        "challenger_margin_min": 1.0,
        "baseline_margin_max": 0.0,
        "direction": "both",
    }
    selected = max(candidates, key=lambda x: x[:3]) if candidates else None
    if selected is None or selected[0] <= 0:
        rule = no_op
        tune_metrics = evaluate(
            tune, args.y_col, args.base_col, args.challenger_col,
            mask_for(tune, args.base_col, args.challenger_col, no_op),
        )
        selection_note = "No candidate had positive conservative tune LCB; selected no-op."
    else:
        _, _, _, rule, tune_metrics = selected
        selection_note = "Selected by maximum tune-only normal-approximation 95% lower bound."

    confirm_mask = mask_for(confirm, args.base_col, args.challenger_col, rule)
    confirm_metrics = evaluate(confirm, args.y_col, args.base_col, args.challenger_col, confirm_mask)
    confirm_metrics["seller_bootstrap_net_ci95"] = seller_bootstrap(
        confirm, args.y_col, args.base_col, args.challenger_col, confirm_mask,
        args.seller_col, args.seed, args.bootstrap_reps,
    )

    result = {
        "contract": "tune-selected once, confirm-evaluated once; train-only labels",
        "selection_note": selection_note,
        "selected_rule": rule,
        "tune": tune_metrics,
        "confirm": confirm_metrics,
        "inputs": {
            "tune_path": str(args.tune),
            "tune_sha256": sha256(args.tune),
            "confirm_path": str(args.confirm),
            "confirm_sha256": sha256(args.confirm),
            "base_col": args.base_col,
            "challenger_col": args.challenger_col,
        },
        "limitations": [
            "Grid search still creates selection multiplicity; positive confirm requires replication.",
            "A no-op selection is a valid result and blocks routing.",
            "This script never establishes public-test transport.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
