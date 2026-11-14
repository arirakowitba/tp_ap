"""AP-2026-Q-2: Product Identity, Specification and Condition Representation Extractor.

Extracts structured features addressing structural blindspots of LightGBM Enriched:
1. Product identity specifications:
   - Part/accessory vs main product (repuestos, fundas, pantallas, tapas, opticas, etc.)
   - Year/vintage detection (<=2012) and continuous normalized year
   - Vintage x Part interaction (distinguishes new aftermarket parts from used vintage products)
   - Vintage x Main interaction (strong used product cue)
   - Alphanumeric capacity/storage (GB/MB/TB)
   - Measurements/dimensions (cm, mm, kg, pulgadas, etc.)
   - Model code alphanumeric patterns (e.g. S7, A30, i7, etc.)
   - Pack/kit/lote/combo indicators
   - Collectibles / vintage publications / tomos / revistas
2. Condition cue representation:
   - Explicit defect/used cues (para repuesto, no funciona, roto, detalles de uso)
   - Explicit new cues (sellado, en caja, nuevo, a estrenar, garantia)
   - Net condition cue balance
3. Price x Specification interactions (using strictly fit-computed category median price):
   - Relative price x part
   - Relative price x main product
   - Relative price x vintage
   - Relative price x defect
4. Leak-free Cross-Seller Product Token Target Encoding:
   - Leave-one-out smoothed target encoding inside fit
   - Out-of-sample lookup on eval sets (tune / confirm)
"""
import re
import numpy as np
import pandas as pd

PART_REGEX = re.compile(
    r"\b(?:repuesto|repuestos|accesorio|accesorios|funda|fundas|cargador|cargadores|"
    r"bateria|baterias|cable|cables|tapa|tapas|modulo|modulos|vidrio|pantalla|pantallas|"
    r"paragolpe|optica|espejo|bomba|filtro|filtros|embrague|amortiguador|amortiguadores|"
    r"pastillas?|disco|discos|correa|correas|bujia|bujias|soporte|soportes|adaptador|"
    r"adaptadores|carcasa|calcomania|stickers?|faro|faros|radiador|guardabarro|"
    r"guardabarros|biela|piston|arbol de leva|valvula|valvulas)\b",
    re.IGNORECASE,
)

DEFECT_REGEX = re.compile(
    r"\b(?:para repuesto|para despiece|para desarme|no funciona|a reparar|con detalle|"
    r"con detalles|pantalla rota|bloqueado|sin probar|roto|rota|detalles de uso|"
    r"desgaste|marcas de uso|usado|usada|segunda mano|leido|subrayado|hojas amarillas|despiece)\b",
    re.IGNORECASE,
)

NEW_CUE_REGEX = re.compile(
    r"\b(?:nuevo|nueva|sellado|sellada|en caja|cerrada|cerrado|en blister|"
    r"con etiqueta|sin abrir|0km|a estrenar|factura|garantia oficial)\b",
    re.IGNORECASE,
)

YEAR_REGEX = re.compile(r"\b(19\d\d|200\d|201[0-5])\b")
CAP_REGEX = re.compile(r"\b(?:\d+)\s*(?:gb|mb|tb)\b", re.IGNORECASE)
DIM_REGEX = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:cm|mm|m|kg|g|gr|ml|l|lt|litros?|oz|pulgadas?|pulg)\b|\b\d+x\d+\b|\b\d+[\'\"”]\b)",
    re.IGNORECASE,
)
MODEL_CODE_REGEX = re.compile(r"\b(?:[a-z]\d{1,4}|\d{1,4}[a-z]|[a-z]{1,3}-\d{2,4})\b", re.IGNORECASE)
PACK_REGEX = re.compile(r"\b(?:pack|kit|lote|combo|x\s*\d+|\d+\s*(?:unidades|piezas|u\b|unid))\b", re.IGNORECASE)
COLLECTIBLE_REGEX = re.compile(r"\b(?:coleccion|colección|antiguo|antigua|retro|vintage|inundacion|serie completa|fasciculo|tomo|revista)\b", re.IGNORECASE)


def extract_regex_features(df: pd.DataFrame) -> dict:
    """Vectorized string feature extraction on full DataFrame."""
    titles = df["title"].astype(str).str.lower()
    attrs = df["attr_text"].astype(str).str.lower()
    combined = titles + " " + attrs

    is_part = combined.str.contains(PART_REGEX).to_numpy().astype(np.float32)
    has_defect = combined.str.contains(DEFECT_REGEX).to_numpy().astype(np.float32)
    has_new_cue = combined.str.contains(NEW_CUE_REGEX).to_numpy().astype(np.float32)
    cue_balance = has_defect - has_new_cue

    # Years
    year_matches = combined.apply(lambda s: [int(y) for y in YEAR_REGEX.findall(s)])
    has_vintage = year_matches.apply(lambda l: any(y <= 2012 for y in l)).to_numpy().astype(np.float32)
    min_year = year_matches.apply(lambda l: min(l) if len(l) > 0 else 0).to_numpy()
    year_num = np.where(min_year > 0, (min_year - 2000.0) / 10.0, 0.0).astype(np.float32)

    # Vintage x Part interactions
    vintage_x_part = (has_vintage * is_part).astype(np.float32)
    vintage_x_main = (has_vintage * (1.0 - is_part)).astype(np.float32)

    # Specs
    has_cap = combined.str.contains(CAP_REGEX).to_numpy().astype(np.float32)
    has_dim = combined.str.contains(DIM_REGEX).to_numpy().astype(np.float32)
    has_model_code = combined.str.contains(MODEL_CODE_REGEX).to_numpy().astype(np.float32)
    has_pack = combined.str.contains(PACK_REGEX).to_numpy().astype(np.float32)
    has_collectible = combined.str.contains(COLLECTIBLE_REGEX).to_numpy().astype(np.float32)

    return {
        "is_part": is_part,
        "has_defect": has_defect,
        "has_new_cue": has_new_cue,
        "cue_balance": cue_balance,
        "has_vintage": has_vintage,
        "year_num": year_num,
        "vintage_x_part": vintage_x_part,
        "vintage_x_main": vintage_x_main,
        "has_cap": has_cap,
        "has_dim": has_dim,
        "has_model_code": has_model_code,
        "has_pack": has_pack,
        "has_collectible": has_collectible,
    }


def compute_identity_feature_matrices(df: pd.DataFrame, fit_idx: np.ndarray, eval_indices_list: list) -> tuple:
    """Build leak-free feature matrices for fit and all eval splits."""
    extracted = extract_regex_features(df)
    n_all = len(df)

    # Price interactions strictly using fit medians
    fit_df = df.iloc[fit_idx]
    cat_medians = fit_df.groupby("category_id")["price"].median()
    overall_median = fit_df["price"].median()

    all_cat_medians = df["category_id"].map(cat_medians).fillna(overall_median).to_numpy()
    rel_price = np.log1p(df["price"].to_numpy() / (all_cat_medians + 1.0)).astype(np.float32)

    price_x_part = (rel_price * extracted["is_part"]).astype(np.float32)
    price_x_main = (rel_price * (1.0 - extracted["is_part"])).astype(np.float32)
    price_x_vintage = (rel_price * extracted["has_vintage"]).astype(np.float32)
    price_x_defect = (rel_price * extracted["has_defect"]).astype(np.float32)
    price_x_pack = (rel_price * extracted["has_pack"]).astype(np.float32)

    # Cross-seller token target encoding
    # Clean product token key: category_id + "_" + first alpha token >= 3 chars
    def get_token_key(title, cat):
        tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z]{3,}\b", str(title))]
        tok = tokens[0] if tokens else "none"
        return f"{cat}_{tok}"

    prod_keys = np.array([get_token_key(t, c) for t, c in zip(df["title"], df["category_id"])])

    # Fit table
    fit_keys = prod_keys[fit_idx]
    fit_y = df["y"].iloc[fit_idx].to_numpy()
    fit_prior = float(fit_y.mean())

    key_series = pd.Series(fit_y, index=fit_keys)
    table_cnt = key_series.groupby(level=0).size()
    table_sum = key_series.groupby(level=0).sum()

    # Fit LOO
    fit_cnts = table_cnt.reindex(fit_keys).fillna(0).to_numpy() - 1
    fit_sums = table_sum.reindex(fit_keys).fillna(0).to_numpy() - fit_y
    fit_te = ((fit_sums + 5.0 * fit_prior) / (fit_cnts + 5.0)).astype(np.float32)
    fit_log_cnt = np.log1p(np.maximum(0, fit_cnts)).astype(np.float32)

    # Base feature array for full dataset
    feat_matrix_all = np.column_stack([
        extracted["is_part"],
        extracted["has_defect"],
        extracted["has_new_cue"],
        extracted["cue_balance"],
        extracted["has_vintage"],
        extracted["year_num"],
        extracted["vintage_x_part"],
        extracted["vintage_x_main"],
        extracted["has_cap"],
        extracted["has_dim"],
        extracted["has_model_code"],
        extracted["has_pack"],
        extracted["has_collectible"],
        rel_price,
        price_x_part,
        price_x_main,
        price_x_vintage,
        price_x_defect,
        price_x_pack,
    ])

    X_id_fit = np.column_stack([
        feat_matrix_all[fit_idx],
        fit_te,
        fit_log_cnt,
    ]).astype(np.float32)

    X_id_evals = []
    for eval_idx in eval_indices_list:
        ev_keys = prod_keys[eval_idx]
        ev_cnts = table_cnt.reindex(ev_keys).fillna(0).to_numpy()
        ev_sums = table_sum.reindex(ev_keys).fillna(0).to_numpy()
        ev_te = ((ev_sums + 5.0 * fit_prior) / (ev_cnts + 5.0)).astype(np.float32)
        ev_log_cnt = np.log1p(ev_cnts).astype(np.float32)

        X_id_eval = np.column_stack([
            feat_matrix_all[eval_idx],
            ev_te,
            ev_log_cnt,
        ]).astype(np.float32)
        X_id_evals.append(X_id_eval)

    feature_names = [
        "is_part", "has_defect", "has_new_cue", "cue_balance",
        "has_vintage", "year_num", "vintage_x_part", "vintage_x_main",
        "has_cap", "has_dim", "has_model_code", "has_pack", "has_collectible",
        "rel_price", "price_x_part", "price_x_main", "price_x_vintage", "price_x_defect", "price_x_pack",
        "prod_token_te", "prod_token_log_cnt"
    ]

    return X_id_fit, X_id_evals, feature_names
