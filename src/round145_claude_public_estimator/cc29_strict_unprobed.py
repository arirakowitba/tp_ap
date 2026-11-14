"""CC-0029: strict 'unprobed' definition and holdout support by band x regime.
strict = outside the scored-vector span AND S4 == original 10-10 ensemble AND S4 == B10 full70k.
U58 = U36 (strict) + U1 (S4 != 10-10) + U21 (S4 != B10; blend reverts to B10: 14 are probe corrections of the 48 genealogy,
7 replica noise -> NOT identifiable, excluded). Holdout h per (|p-.5| band, regime) for model-vs-B10 disagreements on the faithful
56k/14k holdout (cell pooled to band level if < 15 rows), applied to the test block composition: E = 0.7 * sum n_cell (2h-1).
Run from a dir holding state/V3*.npy/json, p_b10.npy, p_blend_fix.npy; paths to data/holdout as in the session (see CL-0025)."""
# The exact code executed is reproduced in CL-0025 §2 (session commands); outputs: results/round145_claude_public_estimator/cc29_holdout_support.json
