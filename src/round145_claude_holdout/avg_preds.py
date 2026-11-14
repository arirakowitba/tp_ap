"""Promedio simple (sin pesos elegidos: nada que seleccionar) de predicciones. usage: python avg_preds.py DATA_DIR OUT a b [c ...]"""
import sys, numpy as np
D, out, names = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3:]
P = [np.load(D + f"pred_{n}.npz") for n in names]
np.savez(D + f"pred_{out}.npz", oof=np.mean([p["oof"] for p in P], 0), te=np.mean([p["te"] for p in P], 0)); print("saved", out, names)
