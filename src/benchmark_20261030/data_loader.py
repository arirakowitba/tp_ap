"""Data Loader and Hash Verification Module.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "train_data.jsonlines"
TEST_PATH = ROOT / "data" / "ap_2026_q_2" / "raw" / "test_data.jsonlines"

def compute_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def load_train_dataset():
    t0 = time.time()
    rows = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            
    ids = np.array([str(r["id"]) for r in rows])
    sellers = np.array([str(r["seller_id"]) for r in rows])
    y_used = np.array([1 if r.get("condition") == "used" else 0 for r in rows], dtype=int)
    
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    
    # Text combo: title + warranty + attributes
    combos = []
    for r in rows:
        parts = [r.get("title") or ""]
        if r.get("warranty"):
            parts.append(str(r["warranty"]))
        attrs = r.get("attributes")
        if isinstance(attrs, list):
            for a in attrs:
                v = a.get("value_name")
                if v:
                    parts.append(str(v))
        combos.append(" ".join(parts))
        
    print(f"Loaded {len(rows)} train records in {time.time()-t0:.2f}s", flush=True)
    return rows, ids, sellers, y_used, titles, warranties, combos

def load_test_dataset():
    t0 = time.time()
    rows = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            
    test_ids = np.arange(1, len(rows) + 1)
    item_ids = [str(r["id"]) for r in rows]
    sellers = np.array([str(r["seller_id"]) for r in rows])
    titles = [r.get("title") or "" for r in rows]
    warranties = [r.get("warranty") or "" for r in rows]
    
    combos = []
    for r in rows:
        parts = [r.get("title") or ""]
        if r.get("warranty"):
            parts.append(str(r["warranty"]))
        attrs = r.get("attributes")
        if isinstance(attrs, list):
            for a in attrs:
                v = a.get("value_name")
                if v:
                    parts.append(str(v))
        combos.append(" ".join(parts))
        
    print(f"Loaded {len(rows)} test records in {time.time()-t0:.2f}s", flush=True)
    return rows, test_ids, item_ids, sellers, titles, warranties, combos
