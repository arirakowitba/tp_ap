# Deep teacher TPE optimization

This branch reuses immutable deep features and the parent NB teacher OOF caches. It does not fit another teacher or use public/test labels.

The persistent Optuna study uses six serial trials (three startup trials, then TPE), seed 20261024, and two threads. All CatBoost fits use model seed 20261024; the enqueued control shares the parent's hyperparameters but is not an exact reproduction of its seed20261023 or 750-tree cap. The cap is1000 trees with80-round early stopping on tune logloss. Tune thresholds are0.475/0.49/0.5/0.51/0.525. The search objective is71.6% seen-seller accuracy plus28.4% new-seller accuracy. Completed-trial ties use tune logloss before confirmation.

`selection_lock.json` and the standard `tune.npz` are written before loading/evaluating the confirmation predictions. `confirm.npz` and `test.npz` use keys `rows,p,y,seen` and `p`, respectively. `tune.npz` uses `rows,p`.

Run `python -u tp_ap/scripts/optimize_deep_tpe_round3.py`. The SQLite study, per-trial snapshots, JSON and NPZ files allow continuation. A stale RUNNING trial is resumed with its same Optuna parameters and CatBoost snapshot. The current stage is in `progress.json`. The search closes after six completed trials or the search budget; confirmation and optional full refit are subsequently performed without choosing weights on confirmation.

The full refit may be retained for ensemble utility even if its standalone accuracy is up to0.2 percentage points below the parent. This is a compute-allocation gate, not evidence of improvement. Existing row splits and previous model searches were reused; this is exploratory optimization, not an independently verified public-score gain.
