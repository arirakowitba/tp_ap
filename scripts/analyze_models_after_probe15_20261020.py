"""Audit grouped-CV model gains and public cohort constraints without test labels."""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import beta
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiment_outputs_hour_20261017_2200" / "rotated_cv"
RECENT = ROOT / "experiment_outputs_today_20261019"
MODEL = STAGE / "results" / "new_catboost_20261020"
PRIOR = STAGE / "submissions" / "20261019"
TODAY = STAGE / "submissions" / "20261020"


def sample_posterior(n, public_net, seed):
    """Beta(1,1) mixture over an unknown 70%-public member count."""
    m = np.arange(n+1, dtype=int)
    good = (np.abs(public_net) <= m) & (((m+public_net) % 2) == 0)
    m = m[good]
    k = (m+public_net)//2
    choose_n_m = gammaln(n+1)-gammaln(m+1)-gammaln(n-m+1)
    choose_m_k = gammaln(m+1)-gammaln(k+1)-gammaln(m-k+1)
    ll = choose_n_m + m*np.log(.7) + (n-m)*np.log(.3)
    ll += choose_m_k + betaln(1+k,1+m-k)
    weights = np.exp(ll-logsumexp(ll))
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(m), 150000, p=weights)
    draws = beta.rvs(1+k[choice], 1+m[choice]-k[choice], random_state=rng)
    return {"expected_public_rows": float((m*weights).sum()),
            "posterior_mean_p_alt": float(draws.mean()),
            "posterior_90pct_interval": [float(v) for v in np.quantile(draws, [.05,.95])],
            "caveat": "Exchangeability and 70% random public allocation are assumptions."}


def main():
    y = (pd.read_pickle(ROOT / "notebook_work/ap_split_build/validation_cache/target.pkl")
         .astype(str).to_numpy() == "used")
    oldo = np.load(OLD / "RichDepthwise_oof_predictions.npz")
    oldp = np.load(OLD / "RichPlain_oof_predictions.npz")
    old_oof = (oldo["oof"]+oldp["oof"])/2
    old_test = (oldo["test"]+oldp["test"])/2
    recent = np.load(MODEL / "rich_text_depthwise_v3_grouped_cv_predictions.npz")
    new_oof, new_test = recent["oof"], recent["test"]
    sym = np.load(MODEL / "rich_text_symmetric_v3_grouped_cv_predictions.npz")
    assert len(y) == len(old_oof) == len(new_oof) == 70000
    assert len(new_test) == len(old_test) == len(sym["test"]) == 30000
    old_label, new_label = old_oof >= .5, new_oof >= .5
    new_wins = int(((new_label == y) & (old_label != y)).sum())
    old_wins = int(((old_label == y) & (new_label != y)).sum())
    assert new_wins == 305 and old_wins == 265
    metrics = {}
    for weight in (0, .25, .4, .6, .75, 1):
        prob = (1-weight)*old_oof + weight*new_oof
        metrics[str(weight)] = {"accuracy": float(accuracy_score(y, prob >= .5)),
                                "correct": int(((prob >= .5) == y).sum()),
                                "log_loss": float(log_loss(y, prob)),
                                "changed_predictions_vs_old": int(((prob >= .5) != old_label).sum())}

    audit = pd.read_csv(PRIOR / "probes_11_13_case_audit.csv")
    base15 = pd.read_csv(PRIOR / "submission_15_top16_Q4_sobre_mejor11.csv")
    base11 = pd.read_csv(PRIOR / "submission_11_consenso_cv32.csv")
    first = pd.read_csv(TODAY / "cohorte_16_auditoria.csv")
    cases = pd.read_csv(ROOT / "experiment_outputs_finalday_20261018_2110"
                        / "final_case_ranking_NOT_calibrated.csv")
    assert base15.shape == (30000,2) and list(base15.columns)==["ID","condition"]
    i=cases.ID.to_numpy(int)-1
    cases["old_catboost_p_alt"] = np.where(cases.historical_alternative_label.eq("used"), old_test[i], 1-old_test[i])
    cases["v3_depthwise_p_alt"] = np.where(cases.historical_alternative_label.eq("used"), new_test[i], 1-new_test[i])
    cases["v3_symmetric_p_alt"] = np.where(cases.historical_alternative_label.eq("used"), sym["test"][i], 1-sym["test"][i])
    for name, col in (("robust_depthwise_v2", "depth_v2_alt"),
                      ("rich_symmetric_v2", "sym_v2_alt")):
        p = np.load(RECENT / f"{name}_oof.npz")["test"]
        cases[col] = np.where(cases.historical_alternative_label.eq("used"),p[i],1-p[i])
    features = ["rotated_cv_probability_alternative", "depth_v2_alt", "sym_v2_alt",
                "new_compact_probability_alternative",
                "new_matched_no_public_nn_blend_probability_alternative"]
    cases["old_cross_family_support"] = sum((cases[col] > .5).astype(int) for col in features)
    cases["v3_supports_alt"] = cases.v3_depthwise_p_alt > .5
    cases["support_six"] = cases.old_cross_family_support + cases.v3_supports_alt.astype(int)
    cases["rank_not_calibrated"] = .55*cases.ranking_final_NOT_probability + .25*cases.v3_depthwise_p_alt + .20*cases.old_catboost_p_alt
    excluded = set(audit.ID) | set(first.ID)
    eligible = cases.loc[cases.uncertain_pool & ~cases.currently_scored & ~cases.currently_pending
                         & ~cases.ID.isin(excluded)
                         & cases.historical_control_label.eq(base15.condition.to_numpy())].copy()
    assert len(eligible)==1448
    cols = ["ID", "item_id", "historical_control_label", "historical_alternative_label",
            "old_cross_family_support", "support_six", "old_catboost_p_alt",
            "v3_depthwise_p_alt", "v3_symmetric_p_alt", "full70k_seed_votes_alternative_out_of_7",
            "rank_not_calibrated"]
    ranked = eligible.sort_values(["support_six", "rank_not_calibrated"], ascending=False)[cols]
    ranked.to_csv(MODEL / "future_candidate_ranking_not_calibrated.csv", index=False)
    records = {}
    for title, mask, net in (
        ("P32",audit.question_cohort.eq("P_strong_cv"),4),
        ("Q144",audit.question_cohort.eq("Q_broad_new"),-19),
        ("Q4_48",audit.question_cohort.eq("Q_broad_new") & audit.cross_family_support.eq(4),None)):
        q=audit.loc[mask]
        row=cases.set_index("ID").loc[q.ID]
        records[title] = {"size":len(q),"public_net_observed":net,
                          "old_p_alt_mean":float(row.old_catboost_p_alt.mean()),
                          "new_v3_p_alt_mean":float(row.v3_depthwise_p_alt.mean()),
                          "v3_support_count":int(row.v3_supports_alt.sum())}
        if net is not None: records[title]["cohort_posterior"] = sample_posterior(len(q),net,20261020+len(q))
    changed15 = set(base15.loc[base15.condition.ne(base11.condition),"ID"])
    q15 = cases.loc[cases.ID.isin(changed15)]
    assert len(q15)==16
    records["Q4_15_top16"] = {"size":16,"old_p_alt_mean":float(q15.old_catboost_p_alt.mean()),
                                "new_v3_p_alt_mean":float(q15.v3_depthwise_p_alt.mean()),
                                "public_net_observed":None}
    report = {"cv_models":metrics,"paired_disagreements":{"v3_correct_old_wrong":new_wins,
              "old_correct_v3_wrong":old_wins},"remaining_candidates":len(eligible),
              "remaining_support_five":{str(k):int(v) for k,v in
                  eligible.old_cross_family_support.value_counts().sort_index().items()},
              "remaining_v3_supports_alt":int(eligible.v3_supports_alt.sum()),
              "cohorts":records,"source_15_sha256":"ce425f4f88720ae74e1a415f92b50f75279ae3d26c1deae278c3c7bb13c55891",
              "score_15":None,
              "calibration_limitation":"CV reused; public split unknown row membership; prior probe selection affects calibration."}
    (MODEL / "audit_cv_public.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({"pair":report["paired_disagreements"],"metrics":metrics,
                      "cohorts":records,"remaining":len(eligible)},ensure_ascii=False))


if __name__ == "__main__":
    main()
