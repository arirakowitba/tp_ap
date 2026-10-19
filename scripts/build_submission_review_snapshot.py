"""Consolidate the experiment workspace into a small, versioned review snapshot."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
def records(frame):
    return json.loads(frame.to_json(orient="records",force_ascii=False))
def main(experiment_root,repo_root):
    root=Path(experiment_root).resolve();dest=Path(repo_root).resolve()/"results/revision_submissions"
    dest.mkdir(parents=True,exist_ok=True);out=root/"experiment_outputs_finalday_20261018_2110"
    history=pd.read_csv(root/"submission_feedback_log.csv")
    H=pd.read_csv(root/"submission_bayes_verified_only.csv",dtype={"ID":str})
    best=pd.read_csv(out/"CONTROL_OBSERVADO_94476.csv",dtype={"ID":str})
    scored=np.zeros(30000,bool);all_rows=[];source_files=[];contrasts={}
    for i,row in enumerate(history.itertuples()):
        file=root/row.submission;frame=pd.read_csv(file,dtype={"ID":str})
        assert frame.ID.equals(H.ID) and len(frame)==30000
        flips=np.flatnonzero(frame.condition.to_numpy()!=H.condition.to_numpy())
        has_score=pd.notna(row.public_accuracy_pct)
        if has_score:scored[flips]=True
        short=Path(row.submission).name
        contrasts[short]=flips.tolist()
        item={"record_order":i+1,"name":short,"path":row.submission,
              "public_score":None if not has_score else float(row.public_accuracy_pct)/100,
              "status":"observed" if has_score else "pending",
              "flips_vs_verified_control":flips.tolist(),"n_flips_vs_verified_control":len(flips),
              "n_changed_vs_observed07":int((frame.condition!=best.condition).sum()),
              "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"notes":str(row.notes)}
        all_rows.append(item);source_files.append({"path":row.submission,"sha256":item["sha256"]})
    plan=np.load(out/"plan_75/provisional_case_measurement_matrix.npz")
    pool=pd.read_csv(out/"plan_75/case_universe_and_cohorts.csv")
    ranks=pd.read_csv(out/"final_case_ranking_NOT_calibrated.csv")
    q=pd.read_pickle(root/"notebook_work/ap_split_build/validation_cache/test_ids.pkl").astype(str).to_numpy()
    locks=pd.read_csv(root/"verified_public_labels.csv",dtype={"item_id":str})
    lookup={s:i for i,s in enumerate(q)};known=np.array([lookup[s] for s in locks.item_id])
    a=ranks.old_reference_probability_alternative.to_numpy()
    votes=ranks.full70k_seed_votes_alternative_out_of_7.to_numpy()
    probability_doubt=(a>.2)&(a<.8);seed_doubt=(votes>0)&(votes<7)
    union=probability_doubt|seed_doubt;union[known]=False
    assert set(np.flatnonzero(union))==set(pool.row_zero_based)
    columns=["ID","old_reference_probability_alternative","full70k_seed_votes_alternative_out_of_7",
       "full70k_seed_probability_sd","old_nn31_opposes_original_flip","cohort_id",
       "ranking_final_NOT_probability","new_compact_probability_alternative",
       "new_public_nn_blend_probability_alternative"]
    pool["currently_scored"]=scored[pool.row_zero_based];columns.append("currently_scored")
    case=pool[columns].copy()
    for col in case.select_dtypes(include="float"):case[col]=case[col].round(7)
    constraints=dict(np.load(out/"public_joint/public_score_constraints.npz"))
    joint={k:constraints[k].tolist() for k in ["patterns","delta","atom_counts","atom_of_tested_row","tested_rows","known_alternative","integer_bounds"]}
    atoms=pd.read_csv(out/"public_joint/joint_atom_posteriors.csv")
    cases=pd.read_csv(out/"submissions/submission_10_cases.csv").fillna("")
    predictive=dict(np.load(out/"submissions/predictive_laws_NOT_guaranteed.npz"))
    hist,bins=np.histogram(a,bins=np.linspace(0,1,51))
    snap={
      "schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),
      "competition":"ap-2026-q-2","snapshot_label":"2026-10-18 · corte anterior al score de la submission 10",
      "dataset":{"train":70000,"test":30000,"historical_holdout":21000,"verified_labels":len(known)},
      "scope":{"public_N_officially_confirmed":False,"score_display_hypotheses":[{"N":21000,"mode":"truncate"},{"N":20998,"mode":"round"}],
       "metrics_in_training_refits_are_validation":False,"private_test_labels_available":False},
      "history":all_rows,"source_files":source_files,
      "counts":{"test_total":30000,"scored_submissions":int(history.public_accuracy_pct.notna().sum()),
       "pending_submissions":int(history.public_accuracy_pct.isna().sum()),
       "unique_tested_ids":int(scored.sum()),"raw_ambiguous_union":int((probability_doubt|seed_doubt).sum()),
       "ambiguous_unverified":int(union.sum()),"ambiguous_scored":int((union&scored).sum()),
       "ambiguous_unscored":int((union&~scored).sum()),"verified_excluded_from_pool":int(((probability_doubt|seed_doubt)&np.isin(np.arange(30000),known)).sum())},
      "doubt_definition":{"probability_interval":[.2,.8],"seed_votes_disagreement":[1,6],"models_share_training_data":True},
      "doubt_overlap":[{"probability_doubt":bool(x),"seed_disagreement":bool(y),"rows":int(((probability_doubt==x)&(seed_doubt==y)).sum())} for x in [False,True] for y in [False,True]],
      "all_alt_probability_histogram":{"edges":bins.tolist(),"counts":hist.tolist()},
      "case_columns":columns,"case_values":json.loads(case.to_json(orient="values",force_ascii=False)),
      "joint_system":joint,"joint_atoms":records(atoms),
      "plan_manifest":json.loads((out/"plan_75/design_manifest.json").read_text()),
      "plan_verification":json.loads((out/"plan_75/design_verification.json").read_text()),
      "plan_schedule":records(pd.read_csv(out/"plan_75/adaptive_schedule_75.csv")),
      "plan_cases":{"ids":plan["row_ids"].tolist(),"cohort_ids":plan["cohort_ids"].tolist()},
      "plan_membership_local_indices":[np.flatnonzero(row).tolist() for row in plan["matrix"]],
      "submission10_manifest":json.loads((out/"submissions/manifest.json").read_text()),
      "submission10_cases":records(cases[["ID","batch_block","old_condition","new_condition","title","warranty","old_reference_probability_alternative","full70k_seed_votes_alternative_out_of_7","ranking_final_NOT_probability"]]),
      "predictive_sensitivity":records(pd.read_csv(out/"submissions/predictive_sensitivity_NOT_guaranteed.csv")),
      "predictive_laws":{k:v.tolist() for k,v in predictive.items()},
      "reference_holdout":{"accuracy":.943952380952381,"log_loss":.1459216189798784,"correct":19823,"rows":21000},
      "latest_training":records(pd.read_csv(out/"training_models_summary.csv")),
      "latest_holdout":records(pd.read_csv(out/"candidate_holdout_summary.csv")),
      "scalable_manifest":json.loads((out/"scalable_evidence/manifest.json").read_text()),
      "control_score_identity":json.loads((out/"confirmed_feedback.json").read_text()),
      "pending_score_warning":"08, 09 y 10 no se tratan como resultados observados. No se infiere un score individual a partir del aspecto de una captura.",
    }
    file=dest/"snapshot_20261018.json";file.write_text(json.dumps(snap,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    history.to_csv(dest/"submission_feedback_log.csv",index=False)
    audit={"files_verified":len(all_rows),"snapshot_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"counts":snap["counts"],"bytes":file.stat().st_size}
    (dest/"snapshot_manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print("SNAPSHOT READY",json.dumps(audit),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--experiment-root",required=True);p.add_argument("--repo-root",required=True);a=p.parse_args();main(a.experiment_root,a.repo_root)

