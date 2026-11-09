import hashlib
import json
import subprocess
import pandas as pd
from pathlib import Path

ROOT = Path(".")

anchor_path = ROOT / "submissions" / "portfolio_20261107_target_09490" / "submission_20261107_04_podium_breakthrough_top85.csv"
df_anc = pd.read_csv(anchor_path).rename(columns={'ID': 'id', 'condition': 'target'})
df_anc['id'] = df_anc['id'].astype(str)
anc_map = dict(zip(df_anc['id'], df_anc['target']))

subs = [
    {
        "proposed_event_id": "EVT-134",
        "name": "submission_20261109_01_surgical_rank5_capture_top5.csv",
        "score": 0.94733,
        "public_hits": 19892,
        "path": "submissions/submission_20261109_01_surgical_rank5_capture_top5.csv",
        "vector_commit": "b50a5e371788a4e6777d7e968b70e847238dafd9"
    },
    {
        "proposed_event_id": "EVT-135",
        "name": "submission_20261109_02_pure_used_capture_top6.csv",
        "score": 0.94752,
        "public_hits": 19896,
        "path": "submissions/submission_20261109_02_pure_used_capture_top6.csv",
        "vector_commit": "3fa6c87"
    },
    {
        "proposed_event_id": "EVT-136",
        "name": "submission_20261109_03_pure_used_expansion_top12.csv",
        "score": 0.94747,
        "public_hits": 19895,
        "path": "submissions/submission_20261109_03_pure_used_expansion_top12.csv",
        "vector_commit": "c421bef"
    },
    {
        "proposed_event_id": "EVT-137",
        "name": "submission_20261109_04_podium_attack_top15.csv",
        "score": 0.94733,
        "public_hits": 19892,
        "path": "submissions/submission_20261109_04_podium_attack_top15.csv",
        "vector_commit": "4ae1b03"
    },
    {
        "proposed_event_id": "EVT-138",
        "name": "submission_20261109_05_podium_climax_top14.csv",
        "score": 0.94738,
        "public_hits": 19893,
        "path": "submissions/submission_20261109_05_podium_climax_top14.csv",
        "vector_commit": "6155654"
    }
]

out_events = []
for s in subs:
    p = ROOT / s["path"]
    df = pd.read_csv(p)
    col_id = [c for c in df.columns if c.lower() in ('id', 'item_id')][0]
    col_target = [c for c in df.columns if c.lower() in ('condition', 'target')][0]
    df = df.rename(columns={col_id: 'id', col_target: 'target'})
    df['id'] = df['id'].astype(str)
    
    # Normalized lines: "id,target\n" sorted by int(id)
    df_sorted = df.sort_values(by='id', key=lambda col: col.astype(int))
    normalized_text = "\n".join([f"{r['id']},{r['target']}" for _, r in df_sorted.iterrows()]) + "\n"
    norm_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    # Flips vs anchor
    cur_map = dict(zip(df['id'], df['target']))
    flips = [k for k in cur_map if cur_map[k] != anc_map[k]]
    
    # Find full commit SHA
    try:
        full_commit = subprocess.check_output(["git", "rev-parse", s["vector_commit"]], text=True).strip()
    except Exception:
        full_commit = s["vector_commit"]
        
    out_events.append({
        "proposed_event_id": s["proposed_event_id"],
        "name": s["name"],
        "score": s["score"],
        "public_hits": s["public_hits"],
        "vector_commit": full_commit,
        "normalized_vector_sha256": norm_hash,
        "n_rows": len(df),
        "n_flips_vs_prior_anchor": len(flips),
        "direction": "mixed" if s["proposed_event_id"] == "EVT-134" else "new_to_used"
    })

delta_data = {
    "schema_version": 1,
    "generated_at_utc": "2026-11-08",
    "base_catalog": {
        "path": "results/canonical_ledger_rebuild_20261107/canonical_events_catalog.json",
        "counts": {
            "total_events": 133,
            "scored_with_vector": 106,
            "scored_no_vector": 7,
            "error_no_score": 2,
            "draft_offline": 18,
            "unique_scored_vector_hashes": 94
        }
    },
    "delta_source": {
        "antigravity_head": "6155654b197bf5cfff8d1a7cb196e9554d67f82f",
        "binding_limitation": "Scores are associated by filename and commit chronology; no Kaggle-exported upload SHA is available. Do not claim cryptographic upload binding."
    },
    "events": out_events
}

delta_dir = ROOT / "results" / "canonical_ledger_delta_20261108"
delta_dir.mkdir(parents=True, exist_ok=True)
with open(delta_dir / "events_134_138.json", "w", encoding="utf-8") as f:
    json.dump(delta_data, f, indent=2)
print("Saved events_134_138.json successfully!")
for e in out_events:
    print(f"{e['proposed_event_id']} | {e['name'][:40]:40s} | score={e['score']} (hits={e['public_hits']}) | flips={e['n_flips_vs_prior_anchor']} | sha={e['normalized_vector_sha256'][:16]}... | commit={e['vector_commit'][:7]}")
