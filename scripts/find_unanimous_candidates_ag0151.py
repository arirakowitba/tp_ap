import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

# Load test predictions and metadata
with open(ROOT / "data/ap_2026_q_2/raw/test_data.jsonlines", encoding="utf-8") as f:
    test_raw = [json.loads(line) for line in f]

evt = pd.read_csv(ROOT / "submissions/portfolio_20261109_podium_ladder/submission_slot5_surgical_quad_rank6_breakthrough.csv")
y_evt = evt["condition"].values

s1 = pd.read_csv(ROOT / "submissions/SUBMISSION_1_TODAY_RECOMMENDED.csv")
diff_s1 = set(np.where(s1["condition"] != y_evt)[0])

test_rep = pd.read_csv(ROOT / "results/multiagent_predictive_base_repaired/test_probabilities_multiagent_predictive_base_repaired.csv")
p_stack = np.load(ROOT / "results/meta_stacking_ensemble/test_probabilities.npy")
p_base_pod = test_rep["p_baseline_podium"].values
y_base_pod = np.where(p_base_pod >= 0.5, "used", "new")
y_stack = np.where(p_stack >= 0.5, "used", "new")

# Agreement mask
consensus_mask = (y_evt == y_stack) & (y_evt == y_base_pod)

agent_cols = ["p_agent_a_text", "p_agent_b_tab", "p_agent_c_seller", "p_agent_d_inter", "p_agent_e_cb", "p_agent_e_lgb", "p_agent_e_xgb"]
probs = test_rep[agent_cols].values
mean_p = probs.mean(axis=1)
std_p = probs.std(axis=1)

# Find items where EVT says 'used', but ALL 7 agents say 'new' with high certainty (p < 0.2)
cand_u2n = []
for i in range(30000):
    if i in diff_s1 or not consensus_mask[i]:
        continue
    if y_evt[i] == "used":
        # All 7 agents predict new
        if np.all(probs[i] < 0.35) and mean_p[i] < 0.25 and p_stack[i] < 0.35:
            title = test_raw[i].get("title", "")
            price = float(test_raw[i].get("price") or 0.0)
            cand_u2n.append((i, mean_p[i], std_p[i], price, title))

# Find items where EVT says 'new', but ALL 7 agents say 'used' with high certainty (p > 0.8)
cand_n2u = []
for i in range(30000):
    if i in diff_s1 or not consensus_mask[i]:
        continue
    if y_evt[i] == "new":
        # All 7 agents predict used
        if np.all(probs[i] > 0.65) and mean_p[i] > 0.75 and p_stack[i] > 0.65:
            title = test_raw[i].get("title", "")
            price = float(test_raw[i].get("price") or 0.0)
            cand_n2u.append((i, mean_p[i], std_p[i], price, title))

print(f"U2N unanimous candidates (EVT=used, all agents < 0.35): {len(cand_u2n)}")
for idx, mp, sp, pr, t in sorted(cand_u2n, key=lambda x: x[1])[:10]:
    print(f"  ID {idx+1}: mean_p={mp:.3f}, std={sp:.3f}, ${pr:.1f} | {t}")

print(f"\nN2U unanimous candidates (EVT=new, all agents > 0.65): {len(cand_n2u)}")
for idx, mp, sp, pr, t in sorted(cand_n2u, key=lambda x: -x[1])[:10]:
    print(f"  ID {idx+1}: mean_p={mp:.3f}, std={sp:.3f}, ${pr:.1f} | {t}")
