"""Shared, leakage-aware utilities for the AP 2026 Q2 condition notebooks."""

from __future__ import annotations

import ast
import json
import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

TARGET = "condition"
SEED = 42
FEATURES_VERSION = "safe_listing_features_v2"
COMPETITION = "ap-2026-q-2"


def find_repo_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "ap2026q2_common.py").exists():
            return candidate
    return here


def _looks_like_competition_folder(folder: Path) -> bool:
    """Return whether ``folder`` directly contains both competition JSONL files."""
    return (
        (folder / "train_data.jsonlines").exists()
        and (folder / "test_data.jsonlines").exists()
    )


def project_data_dir() -> Path:
    override = os.getenv("AP2026Q2_DATA_DIR")
    if override:
        return Path(override).expanduser()

    drive = Path("/content/drive/MyDrive/Analisis_Predictivo/ap_2026_q_2")
    if drive.exists():
        return drive

    # Notebook kernels are independent processes, so an environment variable
    # set in 00_carga_datos.ipynb does not automatically reach notebook 01 or
    # the model notebooks. Detect the standard local OneDrive data folder too.
    local_candidates = [
        Path.home() / "OneDrive" / "Documents" / "TP 3 AP",
        Path.home() / "OneDrive" / "Documents" / "GitHub" / "tp_ap" / "data" / "ap_2026_q_2",
    ]
    for candidate in local_candidates:
        if _looks_like_competition_folder(candidate):
            return candidate
        if _looks_like_competition_folder(candidate / "raw"):
            return candidate

    return find_repo_root() / "data" / "ap_2026_q_2"


def processed_dir() -> Path:
    cache_override = os.getenv("AP2026Q2_CACHE_DIR")
    if cache_override:
        return Path(cache_override).expanduser()
    return project_data_dir() / "processed"


def _find_competition_files(base_dir: Path) -> tuple[Path | None, Path | None]:
    """Find the two JSONL files in either the configured folder or ``raw``."""
    raw_dir = base_dir / "raw"
    search_dirs = [raw_dir, base_dir] if base_dir != raw_dir else [raw_dir]
    train_path = test_path = None
    for folder in search_dirs:
        if not folder.exists():
            continue
        train_path = train_path or next(folder.rglob("train_data.jsonlines"), None)
        test_path = test_path or next(folder.rglob("test_data.jsonlines"), None)
    return train_path, test_path


def _kaggle_auth_hint() -> str:
    """Return actionable authentication instructions without exposing credentials."""
    return (
        "No hay datos locales y Kaggle no está autenticado. Opciones:\n"
        "1) Si ya tienes los JSONL descargados, configura "
        "AP2026Q2_DATA_DIR apuntando a la carpeta que contiene "
        "train_data.jsonlines y test_data.jsonlines.\n"
        "2) En PowerShell, para esta sesión: "
        '$env:KAGGLE_API_TOKEN = "<tu token de Kaggle>" ' 
        "y vuelve a ejecutar la celda.\n"
        "3) En Colab, crea un Secret llamado KAGGLE_API_TOKEN y habilita el acceso. "
        "También puedes usar ~/.kaggle/kaggle.json o las variables KAGGLE_USERNAME y KAGGLE_KEY."
    )


def download_competition(force: bool = False) -> Path:
    """Download via KaggleHub, preferring files already in the persistent cache."""
    base_dir = project_data_dir()
    raw_dir = base_dir / "raw"
    if not force:
        train_path, test_path = _find_competition_files(base_dir)
        if train_path is not None and test_path is not None:
            print("Reutilizo train/test existentes:", train_path.parent)
            return train_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("KAGGLE_API_TOKEN")
    try:
        from google.colab import userdata

        token = userdata.get("KAGGLE_API_TOKEN") or token
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            "No pude leer el Secret KAGGLE_API_TOKEN de Colab. Verifica su nombre y permisos."
        ) from exc
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token

    has_legacy_credentials = bool(
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not token and not has_legacy_credentials and not kaggle_json.exists():
        raise RuntimeError(_kaggle_auth_hint())

    try:
        import kagglehub

        folder = Path(
            kagglehub.competition_download(
                COMPETITION, output_dir=str(raw_dir), force_download=force
            )
        )
    except Exception as exc:
        message = str(exc).lower()
        if "unauthenticated" in message or "authenticate" in message or "401" in message:
            raise RuntimeError(_kaggle_auth_hint()) from exc
        raise
    for required in ("train_data.jsonlines", "test_data.jsonlines"):
        if next(folder.rglob(required), None) is None and next(raw_dir.rglob(required), None) is None:
            raise FileNotFoundError(f"KaggleHub no dejó {required} dentro de {raw_dir}")
    print("Datos disponibles en:", raw_dir)
    return raw_dir


def raw_paths() -> tuple[Path, Path, Path | None]:
    base_dir = project_data_dir()
    raw_dir = base_dir / "raw"
    # Also accept AP2026Q2_DATA_DIR pointing directly at the folder holding
    # the Kaggle files; useful for local validation without copying 300 MB.
    search_dirs = [raw_dir, base_dir] if base_dir != raw_dir else [raw_dir]
    train_path = test_path = sample_path = None
    for folder in search_dirs:
        if not folder.exists():
            continue
        train_path = train_path or next(folder.rglob("train_data.jsonlines"), None)
        test_path = test_path or next(folder.rglob("test_data.jsonlines"), None)
        sample_path = sample_path or next(folder.rglob("sample_submission.csv"), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"No encuentro train_data.jsonlines/test_data.jsonlines en {raw_dir}. "
            "Ejecuta primero 00_carga_datos.ipynb."
        )
    return train_path, test_path, sample_path


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_path, test_path, sample_path = raw_paths()
    train = pd.read_json(train_path, lines=True)
    test = pd.read_json(test_path, lines=True)
    sample = pd.read_csv(sample_path, dtype=str) if sample_path else None
    if TARGET not in train or TARGET in test:
        raise ValueError("condition debe existir solo en train.")
    if sample is None:
        print("Aviso: Kaggle no dejó sample_submission.csv; las celdas de submission pedirán el template oficial.")
    return train, test, sample


def _parse_nested(value):
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None
    return None


def _nested_get(value, *keys):
    item = _parse_nested(value)
    for key in keys:
        if not isinstance(item, dict):
            return np.nan
        item = item.get(key)
    return item if isinstance(item, (str, int, float, bool)) else np.nan


def _nested_count(value):
    item = _parse_nested(value)
    return len(item) if isinstance(item, (list, tuple, dict)) else 0


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic feature creation; all learned transforms stay in model pipelines."""
    result = pd.DataFrame(index=frame.index)
    categorical = [
        "category_id", "listing_type_id", "buying_mode", "currency_id",
        "accepts_mercadopago", "automatic_relist",
    ]
    for col in categorical:
        if col in frame:
            result[col] = frame[col].astype("string").fillna("__MISSING__")

    title = frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    result["title_text"] = title
    result["title_chars"] = title.str.len().astype("float32")
    result["title_words"] = title.str.split().str.len().astype("float32")
    result["title_mentions_used"] = title.str.contains(
        r"\busad[oa]s?\b|segunda mano", regex=True
    ).astype("int8")
    result["title_mentions_new"] = title.str.contains(
        r"\bnuev[oa]s?\b|sin uso|a estrenar", regex=True
    ).astype("int8")

    for col in ("price", "base_price", "initial_quantity"):
        if col in frame:
            values = pd.to_numeric(frame[col], errors="coerce").clip(lower=0)
            result[f"log_{col}"] = np.log1p(values).astype("float32")
    if {"price", "base_price"}.issubset(frame.columns):
        price = pd.to_numeric(frame["price"], errors="coerce")
        base = pd.to_numeric(frame["base_price"], errors="coerce").replace(0, np.nan)
        result["price_base_ratio"] = (price / base).replace([np.inf, -np.inf], np.nan).astype("float32")

    for col in ("warranty", "parent_item_id", "original_price", "video_id"):
        if col in frame:
            result[f"{col}_missing"] = frame[col].isna().astype("int8")
    for col, name in (
        ("attributes", "n_attributes"), ("variations", "n_variations"),
        ("pictures", "n_pictures"), ("tags", "n_tags"),
        ("descriptions", "n_description_refs"),
        ("non_mercado_pago_payment_methods", "n_other_payment_methods"),
    ):
        if col in frame:
            result[name] = frame[col].map(_nested_count).astype("int16")

    if "shipping" in frame:
        result["shipping_mode"] = frame["shipping"].map(
            lambda value: _nested_get(value, "mode")
        ).astype("string").fillna("__MISSING__")
        for col in ("free_shipping", "local_pick_up"):
            result[col] = frame["shipping"].map(
                lambda value: _nested_get(value, col)
            ).astype("string").fillna("__MISSING__")
    if "seller_address" in frame:
        for col, path in (
            ("seller_state", ("state", "id")), ("seller_city", ("city", "id"))
        ):
            result[col] = frame["seller_address"].map(
                lambda value, path=path: _nested_get(value, *path)
            ).astype("string").fillna("__MISSING__")

    if "date_created" in frame:
        created = pd.to_datetime(frame["date_created"], errors="coerce", utc=True)
        result["created_month"] = created.dt.month.astype("Int8").astype("string").fillna("__MISSING__")
        result["created_hour"] = created.dt.hour.astype("Int8").astype("string").fillna("__MISSING__")
        result["created_weekday"] = created.dt.dayofweek.astype("Int8").astype("string").fillna("__MISSING__")
    return result


def prepare_feature_cache(force: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    out_dir = processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file, test_file, target_file, ids_file, meta_file = (
        out_dir / "train_features.pkl", out_dir / "test_features.pkl",
        out_dir / "target.pkl", out_dir / "test_ids.pkl", out_dir / "feature_metadata.json",
    )
    if not force and all(path.exists() for path in (train_file, test_file, target_file, ids_file, meta_file)):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("features_version") == FEATURES_VERSION:
            print("Reutilizo cache de features:", out_dir)
            return (
                pd.read_pickle(train_file), pd.read_pickle(target_file),
                pd.read_pickle(test_file), pd.read_pickle(ids_file),
            )
    train, test, _ = load_raw_data()
    if "id" not in test:
        raise KeyError("No encuentro id en test.")
    y = train[TARGET].astype(str).reset_index(drop=True)
    test_ids = test["id"].astype("string").reset_index(drop=True)
    x_train = build_features(train).reset_index(drop=True)
    x_test = build_features(test).reset_index(drop=True)
    x_train.to_pickle(train_file)
    x_test.to_pickle(test_file)
    y.to_pickle(target_file)
    test_ids.to_pickle(ids_file)
    meta_file.write_text(json.dumps({
        "features_version": FEATURES_VERSION,
        "train_rows": len(x_train), "test_rows": len(x_test),
        "target": TARGET, "random_seed": SEED,
    }, indent=2), encoding="utf-8")
    return x_train, y, x_test, test_ids


def load_feature_data(force: bool = False):
    return prepare_feature_cache(force=force)


def holdout_indices(y: pd.Series, test_size: float = 0.20, seed: int = SEED):
    indices = np.arange(len(y))
    return train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)


def _tabular_preprocessor(frame: pd.DataFrame, min_frequency: int = 10):
    cat_cols = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [col for col in frame.columns if col not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler(with_mean=False)),
        ]), num_cols))
    if cat_cols:
        transformers.append(("categorical", OneHotEncoder(
            handle_unknown="ignore", min_frequency=min_frequency, dtype=np.float32
        ), cat_cols))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def _sparse_linear_inputs(train_x: pd.DataFrame, eval_x: pd.DataFrame):
    train_titles = train_x["title_text"].fillna("").astype(str)
    eval_titles = eval_x["title_text"].fillna("").astype(str)
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=60000,
        strip_accents="unicode", sublinear_tf=True, dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=3, max_features=60000,
        sublinear_tf=True, dtype=np.float32,
    )
    a = word.fit_transform(train_titles)
    b = char.fit_transform(train_titles)
    av = word.transform(eval_titles)
    bv = char.transform(eval_titles)
    train_tab = train_x.drop(columns=["title_text"])
    eval_tab = eval_x.drop(columns=["title_text"])
    prep = _tabular_preprocessor(train_tab)
    c = prep.fit_transform(train_tab)
    cv = prep.transform(eval_tab)
    x_train = hstack([a, b, c], format="csr", dtype=np.float32)
    x_eval = hstack([av, bv, cv], format="csr", dtype=np.float32)
    return x_train, x_eval


def _tabular_tree_inputs(train_x: pd.DataFrame, eval_x: pd.DataFrame):
    train_tab = train_x.drop(columns=["title_text"])
    eval_tab = eval_x.drop(columns=["title_text"])
    prep = _tabular_preprocessor(train_tab, min_frequency=10)
    return prep.fit_transform(train_tab), prep.transform(eval_tab)


def _make_model(model_key: str, params: dict | None = None):
    params = dict(params or {})
    if model_key == "baseline_majority":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(**{"strategy": "most_frequent", **params})
    if model_key == "logistic_tfidf_tabular":
        return LogisticRegression(**{"C": 2.0, "solver": "liblinear", "max_iter": 1000, **params})
    if model_key == "svm_tfidf_tabular":
        return LinearSVC(**{"C": 0.5, "dual": "auto", "max_iter": 5000, "tol": 1e-3, **params})
    if model_key == "naive_bayes_title":
        return MultinomialNB(**{"alpha": 0.3, **params})
    if model_key == "random_forest_tabular":
        defaults = {"n_estimators": 350, "min_samples_leaf": 2, "max_features": "sqrt",
                    "n_jobs": -1 if os.name != "nt" else 1, "random_state": SEED}
        return RandomForestClassifier(**{**defaults, **params})
    if model_key == "xgboost_tabular":
        from xgboost import XGBClassifier
        defaults = {"n_estimators": 450, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0,
                    "objective": "binary:logistic", "eval_metric": "logloss",
                    "tree_method": "hist", "n_jobs": -1, "random_state": SEED}
        return XGBClassifier(**{**defaults, **params})
    if model_key == "lightgbm_tabular":
        from lightgbm import LGBMClassifier
        defaults = {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 31,
                    "min_child_samples": 30, "colsample_bytree": 0.85, "subsample": 0.85,
                    "reg_lambda": 2.0, "verbosity": -1, "random_state": SEED, "n_jobs": -1}
        return LGBMClassifier(**{**defaults, **params})
    if model_key == "catboost_tabular":
        from catboost import CatBoostClassifier
        defaults = {"iterations": 700, "depth": 7, "learning_rate": 0.04,
                    "loss_function": "Logloss", "eval_metric": "Accuracy", "l2_leaf_reg": 6,
                    "random_seed": SEED, "verbose": False, "thread_count": -1,
                    "allow_writing_files": False}
        return CatBoostClassifier(**{**defaults, **params})
    if model_key == "mlp_tabular":
        defaults = {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 1e-4,
                    "batch_size": 256, "early_stopping": True, "n_iter_no_change": 8,
                    "max_iter": 100, "random_state": SEED}
        return MLPClassifier(**{**defaults, **params})
    if model_key == "mlp_tfidf_svd_tabular":
        defaults = {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 1e-3,
                    "batch_size": 256, "early_stopping": True, "n_iter_no_change": 8,
                    "max_iter": 60, "random_state": SEED}
        return MLPClassifier(**{**defaults, **params})
    raise KeyError(f"Modelo desconocido: {model_key}")


def _fit_predict(model_key, train_x, y_train, eval_x, params=None):
    model = _make_model(model_key, params)
    if model_key in ("logistic_tfidf_tabular", "svm_tfidf_tabular"):
        fit_x, predict_x = _sparse_linear_inputs(train_x, eval_x)
        model.fit(fit_x, y_train)
        predictions = model.predict(predict_x)
        return model, predictions
    if model_key == "naive_bayes_title":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=120000,
            strip_accents="unicode", sublinear_tf=True,
        )
        fit_x = vectorizer.fit_transform(train_x["title_text"].fillna("").astype(str))
        predict_x = vectorizer.transform(eval_x["title_text"].fillna("").astype(str))
        model.fit(fit_x, y_train)
        return (vectorizer, model), model.predict(predict_x)
    if model_key in ("random_forest_tabular", "xgboost_tabular"):
        fit_x, predict_x = _tabular_tree_inputs(train_x, eval_x)
        y_values = np.asarray(y_train).astype(str)
        if model_key == "xgboost_tabular":
            labels = sorted(np.unique(y_values))
            label_to_int = {label: i for i, label in enumerate(labels)}
            encoded = np.array([label_to_int[value] for value in y_values])
            model.fit(fit_x, encoded)
            numeric_predictions = model.predict(predict_x).astype(int)
            predictions = np.array(labels)[numeric_predictions]
            return (model, labels), predictions
        model.fit(fit_x, y_values)
        return model, model.predict(predict_x)
    if model_key in ("lightgbm_tabular", "catboost_tabular"):
        train_tab = train_x.drop(columns=["title_text"]).copy()
        eval_tab = eval_x.drop(columns=["title_text"]).copy()
        cat_cols = train_tab.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        for col in cat_cols:
            joined = pd.concat([train_tab[col], eval_tab[col]], ignore_index=True).astype("string").fillna("__MISSING__")
            levels = pd.Index(joined.unique())
            train_tab[col] = pd.Categorical(train_tab[col].astype("string").fillna("__MISSING__"), categories=levels)
            eval_tab[col] = pd.Categorical(eval_tab[col].astype("string").fillna("__MISSING__"), categories=levels)
        if model_key == "catboost_tabular":
            train_cb, eval_cb = train_tab.copy(), eval_tab.copy()
            for col in cat_cols:
                train_cb[col] = train_cb[col].astype("string").fillna("__MISSING__").astype(str)
                eval_cb[col] = eval_cb[col].astype("string").fillna("__MISSING__").astype(str)
            model.fit(train_cb, np.asarray(y_train).astype(str), cat_features=cat_cols, verbose=False)
            return model, model.predict(eval_cb).reshape(-1)
        model.fit(train_tab, np.asarray(y_train).astype(str), categorical_feature=cat_cols)
        return model, model.predict(eval_tab)
    if model_key == "mlp_tabular":
        train_tab = train_x.drop(columns=["title_text"])
        eval_tab = eval_x.drop(columns=["title_text"])
        prep = _tabular_preprocessor(train_tab, min_frequency=20)
        fit_x = prep.fit_transform(train_tab)
        predict_x = prep.transform(eval_tab)
        labels = np.unique(np.asarray(y_train).astype(str))
        label_to_int = {label: index for index, label in enumerate(labels)}
        encoded = np.asarray([label_to_int[str(label)] for label in y_train])
        model.fit(fit_x, encoded)
        predictions = labels[model.predict(predict_x).astype(int)]
        return (prep, model, labels), predictions
    if model_key == "mlp_tfidf_svd_tabular":
        from sklearn.decomposition import TruncatedSVD
        word = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=40000,
            strip_accents="unicode", sublinear_tf=True, dtype=np.float32,
        )
        text_train = word.fit_transform(train_x["title_text"].fillna("").astype(str))
        text_eval = word.transform(eval_x["title_text"].fillna("").astype(str))
        svd = TruncatedSVD(n_components=64, n_iter=5, random_state=SEED)
        reduced_train = svd.fit_transform(text_train).astype(np.float32)
        reduced_eval = svd.transform(text_eval).astype(np.float32)
        prep = _tabular_preprocessor(train_x.drop(columns=["title_text"]), min_frequency=20)
        tab_train = prep.fit_transform(train_x.drop(columns=["title_text"]))
        tab_eval = prep.transform(eval_x.drop(columns=["title_text"]))
        fit_x = hstack([csr_matrix(reduced_train), tab_train], format="csr", dtype=np.float32)
        predict_x = hstack([csr_matrix(reduced_eval), tab_eval], format="csr", dtype=np.float32)
        labels = np.unique(np.asarray(y_train).astype(str))
        label_to_int = {label: index for index, label in enumerate(labels)}
        encoded = np.asarray([label_to_int[str(label)] for label in y_train])
        model.fit(fit_x, encoded)
        predictions = labels[model.predict(predict_x).astype(int)]
        return (word, svd, prep, model, labels), predictions
    model.fit(train_x, y_train)
    return model, model.predict(eval_x)


def score_predictions(y_true, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "f1_used": float(f1_score(y_true, predictions, pos_label="used")),
    }


def evaluate_holdout(model_key: str, params: dict | None = None) -> dict:
    x_train, y, _, _ = load_feature_data()
    train_idx, valid_idx = holdout_indices(y)
    start = time.perf_counter()
    fitted, predictions = _fit_predict(
        model_key, x_train.iloc[train_idx], y.iloc[train_idx], x_train.iloc[valid_idx], params
    )
    result = score_predictions(y.iloc[valid_idx], predictions)
    result.update({
        "model": model_key, "seed": SEED, "split": "stratified_random_80_20",
        "features_version": FEATURES_VERSION, "seconds": time.perf_counter() - start,
        "n_train": len(train_idx), "n_validation": len(valid_idx),
    })
    return {"metrics": result, "model": fitted, "valid_idx": valid_idx, "predictions": predictions}


def cross_validate_model(model_key: str, n_splits: int = 3, params: dict | None = None) -> pd.DataFrame:
    x_train, y, _, _ = load_feature_data()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y), start=1):
        _, predictions = _fit_predict(
            model_key, x_train.iloc[fit_idx], y.iloc[fit_idx], x_train.iloc[val_idx], params
        )
        rows.append({"model": model_key, "fold": fold, **score_predictions(y.iloc[val_idx], predictions)})
    return pd.DataFrame(rows)


def historical_scores(model_key: str | None = None) -> pd.DataFrame:
    path = find_repo_root() / "results" / "historical_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame if model_key is None else frame.loc[frame["model_key"].eq(model_key)].copy()


def append_run(metrics: dict) -> Path:
    path = processed_dir() / "experiment_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([metrics])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)
    return path


def fit_full_and_submit(model_key: str, params: dict | None = None) -> Path:
    x_train, y, x_test, test_ids = load_feature_data()
    fitted, predictions = _fit_predict(model_key, x_train, y, x_test, params)
    return write_submission(model_key, predictions, test_ids)


def create_submission_template_from_test(path: str | Path | None = None) -> Path:
    """Create an inferred Kaggle template when the private competition exposes none.

    The competition target is ``condition`` and the raw test column is ``id``;
    Kaggle's validator expects the submission identifier header ``ID``. This
    function preserves the exact test ID values and order, but the template
    remains inferred rather than official.
    """
    _, test_path, _ = raw_paths()
    test = pd.read_json(test_path, lines=True)
    if "id" not in test:
        raise KeyError("No encuentro la columna id en test para crear el template.")
    ids = test["id"].astype("string").reset_index(drop=True)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Los IDs de test estan vacios o duplicados.")
    output_path = Path(path) if path is not None else project_data_dir() / "sample_submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": ids, TARGET: ""}).to_csv(output_path, index=False)
    print("Template inferido desde test:", output_path, "| filas:", len(ids))
    return output_path


def write_submission(model_key: str, predictions, test_ids: pd.Series | None = None) -> Path:
    _, test_path, sample_path = raw_paths()
    if sample_path is None:
        data_dir = project_data_dir()
        raise FileNotFoundError(
            "No se genero submission para evitar inventar IDs: falta el "
            f"sample_submission.csv oficial de Kaggle en {data_dir} "
            f"o {data_dir / 'raw'}. Descargalo desde la pestana Data de la competencia "
            "y dejalo con ese nombre."
        )
    sample = pd.read_csv(sample_path, dtype=str)
    id_col = next((col for col in sample.columns if str(col).strip().lower() == "id"), None)
    target_col = next((col for col in sample.columns if str(col).strip().lower() == TARGET), None)
    if id_col is None or target_col is None:
        raise ValueError(f"Encabezados inesperados en sample_submission: {sample.columns.tolist()}")
    if test_ids is None:
        test = pd.read_json(test_path, lines=True)
        if "id" not in test:
            raise KeyError("No encuentro id en test.")
        test_ids = test["id"].astype("string").reset_index(drop=True)
    official_ids = sample[id_col].astype("string").reset_index(drop=True)
    test_ids = test_ids.astype("string").reset_index(drop=True)
    predictions = np.asarray(predictions).astype(str)
    if len(sample) != len(predictions) or len(test_ids) != len(predictions):
        raise ValueError("Filas de template, test y predicciones no coinciden.")
    if official_ids.isna().any() or test_ids.isna().any():
        raise ValueError("El template o test contiene IDs vacios.")
    if official_ids.duplicated().any() or test_ids.duplicated().any():
        raise ValueError("IDs duplicados en sample_submission o en test.")
    if set(official_ids) != set(test_ids):
        missing_test = len(set(test_ids) - set(official_ids))
        extra_sample = len(set(official_ids) - set(test_ids))
        raise ValueError(
            f"Los IDs del template no coinciden con test (faltan {missing_test}; "
            f"sobran {extra_sample}). Usa el sample_submission oficial de esta competencia."
        )
    pred_by_id = pd.Series(predictions, index=test_ids)
    predictions = official_ids.map(pred_by_id).to_numpy()
    # Preserve the official template's exact header, ID spelling and order.
    output = sample.copy()
    output[target_col] = predictions
    if output[target_col].isna().any() or not set(output[target_col].unique()).issubset({"new", "used"}):
        raise ValueError("Predicciones condition invalidas o vacias.")
    out_dir = find_repo_root() / "outputs" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"submission_{model_key}.csv"
    output.to_csv(path, index=False)
    check = pd.read_csv(path, dtype=str)
    if check.columns.tolist() != sample.columns.tolist() or len(check) != len(sample):
        raise ValueError("El CSV exportado no respeta el formato de Kaggle.")
    if not check[id_col].astype("string").reset_index(drop=True).equals(official_ids):
        raise ValueError("Los IDs cambiaron al exportar.")
    print("Submission lista:", path, "| filas:", len(check), "| columnas:", check.columns.tolist())
    return path
