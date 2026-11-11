import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

evt = pd.read_csv(ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
y_evt = evt["condition"].values
s1 = pd.read_csv(ROOT / "submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv")
diff_s1 = set(np.where(s1["condition"] != y_evt)[0])

test_rep = pd.read_csv(ROOT / "results/multiagent_predictive_base_repaired/test_probabilities_multiagent_predictive_base_repaired.csv")
p_base_pod = test_rep["p_baseline_podium"].values
y_base_pod = np.where(p_base_pod >= 0.5, "used", "new")
p_stack = np.load(ROOT / "results/meta_stacking_ensemble/test_probabilities.npy")
y_stack = np.where(p_stack >= 0.5, "used", "new")
consensus_mask = (y_evt == y_stack) & (y_evt == y_base_pod)

agent_cols = ["p_agent_a_text", "p_agent_b_tab", "p_agent_c_seller", "p_agent_d_inter", "p_agent_e_cb", "p_agent_e_lgb", "p_agent_e_xgb"]
probs = test_rep[agent_cols].values
mean_p = probs.mean(axis=1)

with open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_raw = [json.loads(line) for line in f]

u_cand = np.where((y_evt == "used") & consensus_mask & (mean_p < 0.35))[0]
n_cand = np.where((y_evt == "new") & consensus_mask & (mean_p > 0.65))[0]

print(f"=== U2N candidates ({len(u_cand)} items: EVT=used -> new, agent_mean < 0.35) ===")
for idx in u_cand:
    in_s1 = "YES" if idx in diff_s1 else "no"
    t = test_raw[idx].get("title", "")
    pr = test_raw[idx].get("price", 0.0)
    print(f"ID {idx+1:5d}: mean_p={mean_p[idx]:.3f}, in_s1={in_s1}, ${pr:8.1f} | {t}")

print(f"\n=== N2U candidates ({len(n_cand)} items: EVT=new -> used, agent_mean > 0.65) ===")
for idx in n_cand:
    in_s1 = "YES" if idx in diff_s1 else "no"
    t = test_raw[idx].get("title", "")
    pr = test_raw[idx].get("price", 0.0)
    print(f"ID {idx+1:5d}: mean_p={mean_p[idx]:.3f}, in_s1={in_s1}, ${pr:8.1f} | {t}")
