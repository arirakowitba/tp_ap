"""Line 5 — Frozen Simple Ensembles, CX-0164 Corrected Rerun vs Diagnostic Comparison,
and Confirm Evaluation Across Unseen-Seller & Seen-Seller Regimes (T98).

Mandatory Rules (CX-0164 + User Prompt):
1. Separates and compares:
   - Diagnostic T97 Podium Recipe (`T97_Policy_A = 58,003` hits on unseen_val, `T97_LGB = 58,333`, `T97_CB = 56,874`);
   - Clean `CX-0164` Corrected Podium Baseline (`Podium_Clean_OOF_A`, `Podium_Clean_OOF_B`, `Podium_Clean_OOF_C`,
     built on the exact same 9-feature CatBoost `2a` + 82-feature LightGBM `m0` + Podium Text `m0` where `p_text_oof`
     and `TE` are strictly 4-fold seller-grouped OOF inside `fit_idx`);
   - Frozen Simple Ensembles (`E1_Equal_Average`, `E2_Convex_Prob_Blend`, `E3_Logit_Stack`, `E4_Rank_Average`)
     combining the 4 competitive enriched specialist lines (`Line 1 M3`, `Line 2 2B`, `Line 3 M1`, `Line 4 L4`).
2. Every ensemble weight/rule is selected strictly inside each outer fold's 4-fold inner OOF (`fit_idx`),
   written to `freeze_line5_outer{k}.json` BEFORE evaluating `seen_val_idx` (`~6,418` rows/fold, `32,092` total)
   and `unseen_val_idx` (`14,000` rows/fold, `70,000` total).
3. Evaluates BOTH regimes separately (`unseen_val` = 70,000 rows; `seen_val` = 32,092 rows) and never blends
   seen/new via post-hoc validation weights. For test predictions (`30,000` rows), applies a strictly fold-local
   deterministic regime router (`is_seen_in_fit = test_seller_id in set(fit_sellers)`), where seen test sellers
   use the inner-selected seen policy and unseen test sellers use the inner-selected unseen policy.
4. Computes accuracy, logloss, fixes, breaks, per-fold delta, and 2,000-replicate two-way cluster bootstrap
   (`seller_id x category_id`, seed `20261111`).
5. Generates test disagreements ONLY as `ID` and model probabilities (`test_disagreements_vs_s4_probs_only.csv`),
   NEVER as inferred labels. Status of all outputs is `DEVELOPMENT_ONLY` (`submission_generated = false`,
   `kaggle_api_used = false`, `public_probability_claims = "NO_IDENTIFIABLE"`).
"""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "results" / "t98_line5_simple_ensembles_20261111"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def fit_convex_weights(preds_list: list, y_true: np.ndarray, bounds=None, x0=None) -> list:
    k = len(preds_list)
    P = np.column_stack([clip_prob(p) for p in preds_list])
    if x0 is None:
        x0 = np.full(k, 1.0 / k)
    if bounds is None:
        bounds = [(0.0, 1.0) for _ in range(k)]

    def obj(w):
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        if s <= 0:
            return 1e9
        w = w / s
        return float(log_loss(y_true, clip_prob(P @ w)))

    res = minimize(obj, x0=x0, bounds=bounds, method="SLSQP", constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w_opt = np.maximum(0.0, res.x)
    w_opt = w_opt / w_opt.sum()
    return [float(x) for x in w_opt]


def fit_logit_stack(preds_list: list, y_true: np.ndarray, C: float = 1.0, non_negative: bool = True) -> dict:
    Z = np.column_stack([logit(clip_prob(p)) for p in preds_list])
    if non_negative:
        # L-BFGS-B optimization of regularized logistic loss with non-negative coefficients
        k = len(preds_list)

        def loss_fn(params):
            w = params[:k]
            b = params[k]
            z = Z @ w + b
            p = expit(np.clip(z, -30.0, 30.0))
            nll = -np.sum(y_true * np.log(clip_prob(p)) + (1.0 - y_true) * np.log(1.0 - clip_prob(p)))
            reg = 0.5 * (1.0 / C) * np.sum(w ** 2)
            return float((nll + reg) / len(y_true))

        x0 = np.append(np.full(k, 1.0 / k), 0.0)
        bounds = [(0.0, 5.0) for _ in range(k)] + [(-5.0, 5.0)]
        res = minimize(loss_fn, x0=x0, bounds=bounds, method="L-BFGS-B")
        coef = [float(x) for x in res.x[:k]]
        intercept = float(res.x[k])
    else:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=500, random_state=42)
        clf.fit(Z, y_true)
        coef = [float(x) for x in clf.coef_[0]]
        intercept = float(clf.intercept_[0])
    return {"coef": coef, "intercept": intercept, "C": C, "non_negative": non_negative}


def apply_logit_stack(preds_list: list, stk: dict) -> np.ndarray:
    Z = np.column_stack([logit(clip_prob(p)) for p in preds_list])
    w = np.asarray(stk["coef"], dtype=np.float64)
    b = float(stk["intercept"])
    return expit(np.clip(Z @ w + b, -30.0, 30.0))


def rank_average(preds_list: list) -> np.ndarray:
    n = len(preds_list[0])
    r_sum = np.zeros(n, dtype=np.float64)
    for p in preds_list:
        r_sum += rankdata(p, method="average") / (n + 1.0)
    return r_sum / len(preds_list)


def two_way_cluster_bootstrap(
    y: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    seller_codes: np.ndarray,
    cat_codes: np.ndarray,
    n_sellers: int,
    n_cats: int,
    n_boot: int = 2000,
    seed: int = 20261111,
) -> dict:
    rng = np.random.default_rng(seed)
    hc = ((p_cand >= 0.5).astype(int) == y).astype(np.float64)
    hb = ((p_base >= 0.5).astype(int) == y).astype(np.float64)
    dh = hc - hb

    pc = clip_prob(p_cand)
    pb = clip_prob(p_base)
    llc = -(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
    llb = -(y * np.log(pb) + (1.0 - y) * np.log(1.0 - pb))
    dl = llc - llb

    n_rows = len(y)
    boot_dh = np.empty(n_boot, dtype=np.float64)
    boot_dl = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        ws = rng.poisson(1.0, size=n_sellers)
        wc = rng.poisson(1.0, size=n_cats)
        w = ws[seller_codes] * wc[cat_codes]
        w_sum = w.sum()
        if w_sum == 0:
            continue
        boot_dh[b] = n_rows * np.dot(w, dh) / w_sum
        boot_dl[b] = np.dot(w, dl) / w_sum

    fixes = int(((hc == 1) & (hb == 0)).sum())
    breaks = int(((hc == 0) & (hb == 1)).sum())
    return {
        "delta_hits": int(fixes - breaks),
        "fixes": fixes,
        "breaks": breaks,
        "boot_delta_hits_median": round(float(np.median(boot_dh)), 2),
        "boot_delta_hits_q05": round(float(np.quantile(boot_dh, 0.05)), 2),
        "boot_delta_hits_q95": round(float(np.quantile(boot_dh, 0.95)), 2),
        "boot_frac_gt0": round(float((boot_dh > 0).mean()), 4),
        "delta_logloss": round(float(llc.mean() - llb.mean()), 6),
        "boot_delta_logloss_q05": round(float(np.quantile(boot_dl, 0.05)), 6),
        "boot_delta_logloss_q95": round(float(np.quantile(boot_dl, 0.95)), 6),
    }


def evaluate_paired_across_folds(
    y: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    fold_ids: np.ndarray,
    sellers: np.ndarray,
    cats: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261111,
) -> dict:
    _, seller_codes = np.unique(sellers, return_inverse=True)
    _, cat_codes = np.unique(cats, return_inverse=True)
    res = two_way_cluster_bootstrap(
        y, p_cand, p_base, seller_codes, cat_codes, int(seller_codes.max() + 1), int(cat_codes.max() + 1), n_boot, seed
    )
    fold_deltas = []
    for k in sorted(np.unique(fold_ids)):
        m = fold_ids == k
        hc = ((p_cand[m] >= 0.5).astype(int) == y[m]).astype(int)
        hb = ((p_base[m] >= 0.5).astype(int) == y[m]).astype(int)
        fold_deltas.append(int((hc - hb).sum()))
    res["per_fold_deltas"] = fold_deltas
    res["folds_positive"] = int(sum(d > 0 for d in fold_deltas))
    res["folds_zero"] = int(sum(d == 0 for d in fold_deltas))
    res["worst_fold_delta"] = int(min(fold_deltas))
    res["cand_hits"] = int(((p_cand >= 0.5).astype(int) == y).sum())
    res["base_hits"] = int(((p_base >= 0.5).astype(int) == y).sum())
    res["cand_acc"] = round(float(accuracy_score(y, (p_cand >= 0.5).astype(int))), 6)
    res["base_acc"] = round(float(accuracy_score(y, (p_base >= 0.5).astype(int))), 6)
    res["cand_logloss"] = round(float(log_loss(y, clip_prob(p_cand))), 6)
    res["base_logloss"] = round(float(log_loss(y, clip_prob(p_base))), 6)
    return res


def main():
    t0 = time.time()
    print("Loading frozen splits and T97 reference predictions...")
    df_splits = pd.read_csv(SPLIT_DIR / "t98_dual_regime_outer_seller_group_splits.csv")
    df_t97 = pd.read_csv(ROOT / "results/seller_group_nested_podium_20261111/paired_outer_val_predictions.csv")
    df_s4 = pd.read_csv(ROOT / "submissions/SUBMISSION_4_TODAY_RECOMMENDED.csv")

    y_all = df_splits["y"].values.astype(int)
    sellers_all = df_splits["seller_id"].astype(str).values
    cats_all = df_splits["category_id"].astype(str).values
    ids_all = df_splits["id"].astype(str).values
    outer_folds = df_splits["outer_fold"].values.astype(int)

    # Containers for 70k unseen_val OOF
    unseen_preds = {
        "T97_Policy_A": df_t97["p_A_podium_recipe"].values,
        "T97_Policy_B": df_t97["p_B_dual_50_50"].values,
        "T97_Policy_C": df_t97["p_C_logistic_stack"].values,
        "T97_LGB": df_t97["p_lgb"].values,
        "T97_CB": df_t97["p_cb"].values,
        "T97_Text": df_t97["p_text"].values,
        # CX-0164 Corrected Clean Podium Baseline (exact same base model structures, leak-free OOF stack features)
        "CX0164_Clean_LGB_M0": np.zeros(70000),
        "CX0164_Clean_CB_2A": np.zeros(70000),
        "CX0164_Clean_Text_M0": np.zeros(70000),
        "CX0164_Clean_Policy_A": np.zeros(70000),
        "CX0164_Clean_Policy_B": np.zeros(70000),
        "CX0164_Clean_Policy_C": np.zeros(70000),
        # Enriched Standalone Specialists (Lines 1-4)
        "Line1_Text_Subword_M3": np.zeros(70000),
        "Line2_CatBoost_Native": np.zeros(70000),
        "Line3_LightGBM_Enriched": np.zeros(70000),
        "Line4_NewSeller_Specialist": np.zeros(70000),
        # Line 5 Simple Ensembles on Enriched Specialists (frozen on inner OOF)
        "Line5_E1_Equal_4Way": np.zeros(70000),
        "Line5_E2_Convex_4Way": np.zeros(70000),
        "Line5_E3_Logit_Stack_4Way": np.zeros(70000),
        "Line5_E4_Equal_3Tree_Spec": np.zeros(70000),
        "Line5_E5_Selected_Confirm": np.zeros(70000),
    }

    seen_records = []
    test_fold_preds = {
        "Line5_Unseen_Policy_Test": np.zeros((5, 30000)),
        "Line5_Seen_Policy_Test": np.zeros((5, 30000)),
        "Line5_Regime_Routed_Test": np.zeros((5, 30000)),
        "CX0164_Clean_Policy_A_Test": np.zeros((5, 30000)),
    }
    frozen_policies = []
    per_fold_rows = []

    for k in range(5):
        role_k = df_splits[f"role_fold_{k}"].values
        inner_k = df_splits[f"inner_fold_{k}"].values
        fit_idx = np.where(role_k == "fit")[0]
        seen_val_idx = np.where(role_k == "seen_val")[0]
        unseen_val_idx = np.where(role_k == "unseen_val")[0]

        z1 = np.load(ROOT / f"results/t98_line1_text_subword_20261111/fold_{k}_predictions.npz")
        z2 = np.load(ROOT / f"results/t98_line2_catboost_native_20261111/fold_{k}_predictions.npz")
        z3 = np.load(ROOT / f"results/t98_line3_lightgbm_enriched_20261111/fold_{k}_predictions.npz")
        z4 = np.load(ROOT / f"results/t98_line4_new_seller_specialist_20261111/fold_{k}_predictions.npz")

        assert np.array_equal(fit_idx, z1["fit_idx"]) and np.array_equal(fit_idx, z2["fit_idx"])
        assert np.array_equal(fit_idx, z3["fit_idx"]) and np.array_equal(fit_idx, z4["fit_idx"])
        y_fit = y_all[fit_idx]
        inner_f = inner_k[fit_idx]

        # ---------------------------------------------------------------------
        # PART A: CX-0164 Clean Podium Rerun (M0 LGB + 2A CB + M0 Text)
        # ---------------------------------------------------------------------
        c_lgb_fit = z3["p_m0_fit_oof"]
        c_cb_fit = z2["p_2a_fit_oof"]
        c_txt_fit = z1["p_m0_fit_oof"]

        # Policy A (Podium bounds [0.05,0.80], [0.05,0.80], [0.05,0.50]) on clean inner-OOF
        wA_clean = fit_convex_weights(
            [c_lgb_fit, c_cb_fit, c_txt_fit],
            y_fit,
            bounds=[(0.05, 0.80), (0.05, 0.80), (0.05, 0.50)],
            x0=[0.45, 0.40, 0.15],
        )
        # Policy C (LogisticRegression(C=1.0) on logits) on clean inner-OOF
        stkC_clean = fit_logit_stack([c_lgb_fit, c_cb_fit, c_txt_fit], y_fit, C=1.0, non_negative=False)

        # Evaluate Clean Podium on unseen_val
        c_lgb_un = z3["p_m0_unseen_val"]
        c_cb_un = z2["p_2a_unseen_val"]
        c_txt_un = z1["p_m0_unseen_val"]
        unseen_preds["CX0164_Clean_LGB_M0"][unseen_val_idx] = c_lgb_un
        unseen_preds["CX0164_Clean_CB_2A"][unseen_val_idx] = c_cb_un
        unseen_preds["CX0164_Clean_Text_M0"][unseen_val_idx] = c_txt_un
        unseen_preds["CX0164_Clean_Policy_A"][unseen_val_idx] = (
            wA_clean[0] * c_lgb_un + wA_clean[1] * c_cb_un + wA_clean[2] * c_txt_un
        )
        unseen_preds["CX0164_Clean_Policy_B"][unseen_val_idx] = 0.5 * c_lgb_un + 0.5 * c_cb_un
        unseen_preds["CX0164_Clean_Policy_C"][unseen_val_idx] = apply_logit_stack(
            [c_lgb_un, c_cb_un, c_txt_un], stkC_clean
        )

        # Evaluate Clean Podium on seen_val (using causal prior seller_te for LGB M0 and native seller for CB 2A)
        c_lgb_sn = z3["p_m0_seen_val"]
        c_cb_sn = z2["p_2a_seen_val"]
        c_txt_sn = z1["p_m0_seen_val"]
        p_clean_A_sn = wA_clean[0] * c_lgb_sn + wA_clean[1] * c_cb_sn + wA_clean[2] * c_txt_sn
        p_clean_B_sn = 0.5 * c_lgb_sn + 0.5 * c_cb_sn
        p_clean_C_sn = apply_logit_stack([c_lgb_sn, c_cb_sn, c_txt_sn], stkC_clean)

        # ---------------------------------------------------------------------
        # PART B: Line 5 Simple Ensembles of Enriched Specialists (Lines 1-4)
        # ---------------------------------------------------------------------
        # Inner-OOF predictions on fit_idx (100% seller-disjoint 4 inner folds):
        s1_fit = z1["p_m3_fit_oof"]          # Line 1: Enriched Text + Subword NB-SVM Blend
        s2_fit = z2["p_cb_fit_oof"]          # Line 2: Enriched Native CatBoost Blend
        s3_fit = z3["p_m1_fit_oof"]          # Line 3: Enriched LightGBM
        s4_fit = z4["p_l4_fit_oof"]          # Line 4: New-Seller Intrinsic Specialist Blend

        # Fit Convex & Logit-Stack weights strictly on fit_idx inner-OOF (via 4-fold cross-val inside fit_idx to select best rule!)
        e1_oof = 0.25 * (s1_fit + s2_fit + s3_fit + s4_fit)
        e4_oof = (s2_fit + s3_fit + s4_fit) / 3.0
        e2_oof = np.zeros(len(fit_idx))
        e3_oof = np.zeros(len(fit_idx))

        for j in range(4):
            m_tr = inner_f != j
            m_va = inner_f == j
            wj = fit_convex_weights([s1_fit[m_tr], s2_fit[m_tr], s3_fit[m_tr], s4_fit[m_tr]], y_fit[m_tr])
            e2_oof[m_va] = wj[0] * s1_fit[m_va] + wj[1] * s2_fit[m_va] + wj[2] * s3_fit[m_va] + wj[3] * s4_fit[m_va]
            stkj = fit_logit_stack([s1_fit[m_tr], s2_fit[m_tr], s3_fit[m_tr], s4_fit[m_tr]], y_fit[m_tr], C=0.5, non_negative=True)
            e3_oof[m_va] = apply_logit_stack([s1_fit[m_va], s2_fit[m_va], s3_fit[m_va], s4_fit[m_va]], stkj)

        # Full inner-OOF weights for outer application
        w_e2_unseen = fit_convex_weights([s1_fit, s2_fit, s3_fit, s4_fit], y_fit)
        stk_e3_unseen = fit_logit_stack([s1_fit, s2_fit, s3_fit, s4_fit], y_fit, C=0.5, non_negative=True)

        cand_inner_scores = {}
        for cname, carr in [
            ("E1_Equal_4Way", e1_oof),
            ("E2_Convex_4Way", e2_oof),
            ("E3_Logit_Stack_4Way", e3_oof),
            ("E4_Equal_3Tree_Spec", e4_oof),
        ]:
            cand_inner_scores[cname] = {
                "hits": int(((carr >= 0.5) == y_fit).sum()),
                "acc": round(float(accuracy_score(y_fit, carr >= 0.5)), 6),
                "logloss": round(float(log_loss(y_fit, clip_prob(carr))), 6),
            }

        # Select frozen confirm rule for unseen regime by lowest inner-OOF logloss
        best_unseen_rule = min(cand_inner_scores.keys(), key=lambda c: (cand_inner_scores[c]["logloss"], -cand_inner_scores[c]["hits"]))

        # Seen-seller regime inner selection: use causal-seen items inside fit_idx
        # Note: in fit_idx, items of multi-item sellers that have >=1 earlier item in fit_idx
        df_fit_sub = df_splits.iloc[fit_idx].sort_values(["seller_id", "start_time", "row"])
        df_fit_sub["prior_cnt"] = df_fit_sub.groupby("seller_id").cumcount()
        causal_seen_rows_global = set(df_fit_sub.loc[df_fit_sub["prior_cnt"] >= 1, "row"].values)
        seen_mask_in_fit = np.array([r in causal_seen_rows_global for r in fit_idx], dtype=bool)

        # On seen_val, Line 3 M1 (causal seller_te) + Line 2 2B (with causal seller) + Line 4 + Line 1 M3:
        # Since z2["p_2b_seen_fit_oof"] has causal seller stats on seen_mask_in_fit:
        s2_seen_fit = z2["p_2b_seen_fit_oof"]
        w_seen_convex = fit_convex_weights(
            [s1_fit[seen_mask_in_fit], s2_seen_fit[seen_mask_in_fit], s3_fit[seen_mask_in_fit], s4_fit[seen_mask_in_fit]],
            y_fit[seen_mask_in_fit],
        )
        # Note: in fit_idx inner-OOF, s3_fit was trained in unseen seller-group mode, whereas on seen_val
        # both Line 2 (`p_cb_seen_val`) and Line 3 (`p_m1_seen_val`) have causal seller history.
        # Therefore, a simple preregistered fixed seen-regime ensemble `0.45*Line3_Seen + 0.35*Line2_Seen + 0.15*Line4 + 0.05*Line1`
        # and equal average `0.50*Line3_Seen + 0.50*Line2_Seen` are also evaluated without touching seen_val!
        w_seen_prereg = [0.05, 0.35, 0.45, 0.15]  # [Line1_M3, Line2_Seen, Line3_Seen, Line4]

        freeze_obj = {
            "outer_fold": k + 1,
            "clean_podium_wA": wA_clean,
            "clean_podium_stkC": stkC_clean,
            "inner_oof_candidates_unseen": cand_inner_scores,
            "selected_unseen_confirm_rule": best_unseen_rule,
            "w_e2_unseen_L1_L2_L3_L4": w_e2_unseen,
            "stk_e3_unseen_L1_L2_L3_L4": stk_e3_unseen,
            "w_seen_inner_convex_L1_L2_L3_L4": w_seen_convex,
            "w_seen_prereg_L1_L2_L3_L4": w_seen_prereg,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (OUT_DIR / f"freeze_line5_outer{k}.json").write_text(json.dumps(freeze_obj, indent=2), encoding="utf-8")
        frozen_policies.append(freeze_obj)

        # Apply frozen models to unseen_val_idx
        s1_un = z1["p_m3_unseen_val"]
        s2_un = z2["p_cb_unseen_val"]
        s3_un = z3["p_m1_unseen_val"]
        s4_un = z4["p_l4_unseen_val"]

        unseen_preds["Line1_Text_Subword_M3"][unseen_val_idx] = s1_un
        unseen_preds["Line2_CatBoost_Native"][unseen_val_idx] = s2_un
        unseen_preds["Line3_LightGBM_Enriched"][unseen_val_idx] = s3_un
        unseen_preds["Line4_NewSeller_Specialist"][unseen_val_idx] = s4_un

        p_e1_un = 0.25 * (s1_un + s2_un + s3_un + s4_un)
        p_e2_un = w_e2_unseen[0] * s1_un + w_e2_unseen[1] * s2_un + w_e2_unseen[2] * s3_un + w_e2_unseen[3] * s4_un
        p_e3_un = apply_logit_stack([s1_un, s2_un, s3_un, s4_un], stk_e3_unseen)
        p_e4_un = (s2_un + s3_un + s4_un) / 3.0

        unseen_map_k = {
            "E1_Equal_4Way": p_e1_un,
            "E2_Convex_4Way": p_e2_un,
            "E3_Logit_Stack_4Way": p_e3_un,
            "E4_Equal_3Tree_Spec": p_e4_un,
        }
        p_e5_un = unseen_map_k[best_unseen_rule]

        unseen_preds["Line5_E1_Equal_4Way"][unseen_val_idx] = p_e1_un
        unseen_preds["Line5_E2_Convex_4Way"][unseen_val_idx] = p_e2_un
        unseen_preds["Line5_E3_Logit_Stack_4Way"][unseen_val_idx] = p_e3_un
        unseen_preds["Line5_E4_Equal_3Tree_Spec"][unseen_val_idx] = p_e4_un
        unseen_preds["Line5_E5_Selected_Confirm"][unseen_val_idx] = p_e5_un

        # Apply frozen models to seen_val_idx
        s1_sn = z1["p_m3_seen_val"]
        s2_sn = z2["p_cb_seen_val"]
        s3_sn = z3["p_m1_seen_val"]
        s4_sn = z4["p_l4_seen_val"]

        p_sn_equal4 = 0.25 * (s1_sn + s2_sn + s3_sn + s4_sn)
        p_sn_dual_tree = 0.50 * s2_sn + 0.50 * s3_sn
        p_sn_convex_in = w_seen_convex[0] * s1_sn + w_seen_convex[1] * s2_sn + w_seen_convex[2] * s3_sn + w_seen_convex[3] * s4_sn
        p_sn_prereg = w_seen_prereg[0] * s1_sn + w_seen_prereg[1] * s2_sn + w_seen_prereg[2] * s3_sn + w_seen_prereg[3] * s4_sn

        seen_df_k = pd.DataFrame({
            "outer_fold": k + 1,
            "row": seen_val_idx,
            "id": ids_all[seen_val_idx],
            "seller_id": sellers_all[seen_val_idx],
            "category_id": cats_all[seen_val_idx],
            "y_true": y_all[seen_val_idx],
            "p_CX0164_Clean_LGB_M0": c_lgb_sn,
            "p_CX0164_Clean_CB_2A": c_cb_sn,
            "p_CX0164_Clean_Text_M0": c_txt_sn,
            "p_CX0164_Clean_Policy_A": p_clean_A_sn,
            "p_CX0164_Clean_Policy_B": p_clean_B_sn,
            "p_CX0164_Clean_Policy_C": p_clean_C_sn,
            "p_Line1_Text_Subword_M3": s1_sn,
            "p_Line2_CatBoost_Native": s2_sn,
            "p_Line3_LightGBM_Enriched": s3_sn,
            "p_Line4_NewSeller_Specialist": s4_sn,
            "p_Line5_Seen_Equal_4Way": p_sn_equal4,
            "p_Line5_Seen_Dual_LGB_CB": p_sn_dual_tree,
            "p_Line5_Seen_Inner_Convex": p_sn_convex_in,
            "p_Line5_Seen_Prereg_Blend": p_sn_prereg,
        })
        seen_records.append(seen_df_k)

        # Test predictions for fold k (strictly fold-local routing by whether test seller is in fit_idx)
        s1_te = z1["p_m3_test"]
        s2_te_unseen = z2["p_2b_unseen_test"]
        s2_te_seen = z2["p_cb_test"]
        s3_te_unseen = z3["p_m1_unseen_test"]
        s3_te_seen = z3["p_m1_seen_causal_test"]
        s4_te = z4["p_l4_test"]
        test_in_fit = z3["test_in_fit_seller"].astype(bool)

        p_te_unseen_pol = w_e2_unseen[0] * s1_te + w_e2_unseen[1] * s2_te_unseen + w_e2_unseen[2] * s3_te_unseen + w_e2_unseen[3] * s4_te
        p_te_seen_pol = w_seen_prereg[0] * s1_te + w_seen_prereg[1] * s2_te_seen + w_seen_prereg[2] * s3_te_seen + w_seen_prereg[3] * s4_te
        p_te_routed = np.where(test_in_fit, p_te_seen_pol, p_te_unseen_pol)

        test_fold_preds["Line5_Unseen_Policy_Test"][k] = p_te_unseen_pol
        test_fold_preds["Line5_Seen_Policy_Test"][k] = p_te_seen_pol
        test_fold_preds["Line5_Regime_Routed_Test"][k] = p_te_routed
        test_fold_preds["CX0164_Clean_Policy_A_Test"][k] = (
            wA_clean[0] * z3["p_m0_test"] + wA_clean[1] * z2["p_2a_test"] + wA_clean[2] * z1["p_m0_test"]
        )

        y_un = y_all[unseen_val_idx]
        y_sn = y_all[seen_val_idx]
        per_fold_rows.append({
            "outer_fold": k + 1,
            "selected_unseen_rule": best_unseen_rule,
            "unseen_hits_T97_PolicyA": int(((unseen_preds["T97_Policy_A"][unseen_val_idx] >= 0.5) == y_un).sum()),
            "unseen_hits_CX0164_Clean_A": int(((unseen_preds["CX0164_Clean_Policy_A"][unseen_val_idx] >= 0.5) == y_un).sum()),
            "unseen_hits_CX0164_Clean_B": int(((unseen_preds["CX0164_Clean_Policy_B"][unseen_val_idx] >= 0.5) == y_un).sum()),
            "unseen_hits_CX0164_Clean_C": int(((unseen_preds["CX0164_Clean_Policy_C"][unseen_val_idx] >= 0.5) == y_un).sum()),
            "unseen_hits_Line1_M3": int(((s1_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line2_CB": int(((s2_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line3_LGB": int(((s3_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line4_NewSeller": int(((s4_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line5_E1_Equal4": int(((p_e1_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line5_E2_Convex4": int(((p_e2_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line5_E3_Logit4": int(((p_e3_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line5_E4_Equal3Tree": int(((p_e4_un >= 0.5) == y_un).sum()),
            "unseen_hits_Line5_E5_Confirm": int(((p_e5_un >= 0.5) == y_un).sum()),
            "seen_hits_CX0164_Clean_A": int(((p_clean_A_sn >= 0.5) == y_sn).sum()),
            "seen_hits_CX0164_Clean_B": int(((p_clean_B_sn >= 0.5) == y_sn).sum()),
            "seen_hits_Line2_CB": int(((s2_sn >= 0.5) == y_sn).sum()),
            "seen_hits_Line3_LGB": int(((s3_sn >= 0.5) == y_sn).sum()),
            "seen_hits_Line4_NewSeller": int(((s4_sn >= 0.5) == y_sn).sum()),
            "seen_hits_Line5_Dual_LGB_CB": int(((p_sn_dual_tree >= 0.5) == y_sn).sum()),
            "seen_hits_Line5_Inner_Convex": int(((p_sn_convex_in >= 0.5) == y_sn).sum()),
            "seen_hits_Line5_Prereg_Blend": int(((p_sn_prereg >= 0.5) == y_sn).sum()),
        })

    df_per_fold = pd.DataFrame(per_fold_rows)
    df_per_fold.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

    df_seen_all = pd.concat(seen_records, ignore_index=True)
    df_seen_all.to_csv(OUT_DIR / "oof_seen_all_folds.csv", index=False)

    df_unseen_out = pd.DataFrame({
        "row": np.arange(70000),
        "id": ids_all,
        "seller_id": sellers_all,
        "category_id": cats_all,
        "outer_fold": outer_folds + 1,
        "y_true": y_all,
        **unseen_preds,
    })
    df_unseen_out.to_csv(OUT_DIR / "oof_unseen_70k.csv", index=False)

    # Save test probabilities ONLY (zero inferred labels)
    df_test_out = pd.DataFrame({
        "ID": df_s4["ID"].values,
        "p_CX0164_Clean_Policy_A": test_fold_preds["CX0164_Clean_Policy_A_Test"].mean(axis=0),
        "p_Line5_Unseen_Policy": test_fold_preds["Line5_Unseen_Policy_Test"].mean(axis=0),
        "p_Line5_Seen_Policy": test_fold_preds["Line5_Seen_Policy_Test"].mean(axis=0),
        "p_Line5_Regime_Routed": test_fold_preds["Line5_Regime_Routed_Test"].mean(axis=0),
    })
    df_test_out.to_csv(OUT_DIR / "test_predictions_30k.csv", index=False)

    s4_used = (df_s4["condition"].astype(str).str.lower() == "used").values
    disagree_mask = (df_test_out["p_Line5_Regime_Routed"].values >= 0.5) != s4_used
    df_disagree = df_test_out.loc[disagree_mask].copy()
    df_disagree.to_csv(OUT_DIR / "test_disagreements_vs_s4_probs_only.csv", index=False)

    print("Running 2,000-replicate seller x category bootstrap across Unseen (70k) and Seen (32,092)...")
    unseen_summary = {}
    for kname, arr in unseen_preds.items():
        unseen_summary[kname] = {
            "hits": int(((arr >= 0.5) == y_all).sum()),
            "accuracy": round(float(accuracy_score(y_all, arr >= 0.5)), 6),
            "logloss": round(float(log_loss(y_all, clip_prob(arr))), 6),
            "roc_auc": round(float(roc_auc_score(y_all, clip_prob(arr))), 6),
        }

    y_sn_all = df_seen_all["y_true"].values
    sn_sellers = df_seen_all["seller_id"].astype(str).values
    sn_cats = df_seen_all["category_id"].astype(str).values
    sn_folds = df_seen_all["outer_fold"].values

    seen_cols = [c for c in df_seen_all.columns if c.startswith("p_")]
    seen_summary = {}
    for c in seen_cols:
        arr = df_seen_all[c].values
        seen_summary[c[2:]] = {
            "hits": int(((arr >= 0.5) == y_sn_all).sum()),
            "accuracy": round(float(accuracy_score(y_sn_all, arr >= 0.5)), 6),
            "logloss": round(float(log_loss(y_sn_all, clip_prob(arr))), 6),
            "roc_auc": round(float(roc_auc_score(y_sn_all, clip_prob(arr))), 6),
        }

    paired_unseen = {
        # 1. CX-0164 Clean Podium Rerun vs Diagnostic T97
        "CX0164_Clean_Policy_A_vs_T97_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["CX0164_Clean_Policy_A"], unseen_preds["T97_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "CX0164_Clean_Policy_B_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["CX0164_Clean_Policy_B"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "CX0164_Clean_Policy_C_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["CX0164_Clean_Policy_C"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        # 2. Enriched Specialists & Line 5 Ensembles vs CX0164_Clean_Policy_A
        "Line1_M3_vs_CX0164_Clean_Text_M0": evaluate_paired_across_folds(
            y_all, unseen_preds["Line1_Text_Subword_M3"], unseen_preds["CX0164_Clean_Text_M0"], outer_folds, sellers_all, cats_all
        ),
        "Line2_CB_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["Line2_CatBoost_Native"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "Line3_LGB_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["Line3_LightGBM_Enriched"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "Line4_NewSeller_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["Line4_NewSeller_Specialist"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "Line5_E5_Confirm_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["Line5_E5_Selected_Confirm"], unseen_preds["CX0164_Clean_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "Line5_E5_Confirm_vs_T97_Policy_A": evaluate_paired_across_folds(
            y_all, unseen_preds["Line5_E5_Selected_Confirm"], unseen_preds["T97_Policy_A"], outer_folds, sellers_all, cats_all
        ),
        "Line5_E5_Confirm_vs_Line4_NewSeller": evaluate_paired_across_folds(
            y_all, unseen_preds["Line5_E5_Selected_Confirm"], unseen_preds["Line4_NewSeller_Specialist"], outer_folds, sellers_all, cats_all
        ),
        "Line5_E4_Equal3Tree_vs_Line4_NewSeller": evaluate_paired_across_folds(
            y_all, unseen_preds["Line5_E4_Equal_3Tree_Spec"], unseen_preds["Line4_NewSeller_Specialist"], outer_folds, sellers_all, cats_all
        ),
    }

    paired_seen = {
        "CX0164_Clean_Policy_B_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_CX0164_Clean_Policy_B"].values, df_seen_all["p_CX0164_Clean_Policy_A"].values, sn_folds, sn_sellers, sn_cats
        ),
        "CX0164_Clean_Policy_C_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_CX0164_Clean_Policy_C"].values, df_seen_all["p_CX0164_Clean_Policy_A"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line3_LGB_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line3_LightGBM_Enriched"].values, df_seen_all["p_CX0164_Clean_Policy_A"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line2_CB_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line2_CatBoost_Native"].values, df_seen_all["p_CX0164_Clean_Policy_A"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line5_Seen_Prereg_Blend_vs_CX0164_Clean_Policy_A": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line5_Seen_Prereg_Blend"].values, df_seen_all["p_CX0164_Clean_Policy_A"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line5_Seen_Dual_LGB_CB_vs_Line3_LGB": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line5_Seen_Dual_LGB_CB"].values, df_seen_all["p_Line3_LightGBM_Enriched"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line5_Seen_Prereg_Blend_vs_Line3_LGB": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line5_Seen_Prereg_Blend"].values, df_seen_all["p_Line3_LightGBM_Enriched"].values, sn_folds, sn_sellers, sn_cats
        ),
        "Line5_Seen_Inner_Convex_vs_Line3_LGB": evaluate_paired_across_folds(
            y_sn_all, df_seen_all["p_Line5_Seen_Inner_Convex"].values, df_seen_all["p_Line3_LightGBM_Enriched"].values, sn_folds, sn_sellers, sn_cats
        ),
    }

    manifest = {
        "task": "T98_Line5_Frozen_Simple_Ensembles_and_CX0164_Clean_Rerun",
        "status": "DEVELOPMENT_ONLY",
        "submission_generated": False,
        "kaggle_api_used": False,
        "public_probability_claims": "NO_IDENTIFIABLE",
        "script": "scripts/t98_line5_simple_ensembles.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "unseen_standalone_summary_70k": unseen_summary,
        "seen_standalone_summary_32092": seen_summary,
        "paired_comparisons_unseen_70k": paired_unseen,
        "paired_comparisons_seen_32092": paired_seen,
        "test_disagreements_vs_s4_count": int(disagree_mask.sum()),
        "frozen_policies": frozen_policies,
        "output_hashes": {
            "oof_unseen_70k.csv": sha256_file(OUT_DIR / "oof_unseen_70k.csv"),
            "oof_seen_all_folds.csv": sha256_file(OUT_DIR / "oof_seen_all_folds.csv"),
            "per_fold_metrics.csv": sha256_file(OUT_DIR / "per_fold_metrics.csv"),
            "test_predictions_30k.csv": sha256_file(OUT_DIR / "test_predictions_30k.csv"),
            "test_disagreements_vs_s4_probs_only.csv": sha256_file(OUT_DIR / "test_disagreements_vs_s4_probs_only.csv"),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {manifest['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
