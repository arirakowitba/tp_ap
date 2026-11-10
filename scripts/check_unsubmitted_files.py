"""Check which files in submissions/ have been submitted and their scores."""
import pandas as pd
from pathlib import Path

df_events = pd.read_csv("results/canonical_ledger_rebuild_20261108/canonical_events_catalog.csv")
event_map = {}
for _, r in df_events.iterrows():
    fn = str(r["display_name"]).strip()
    event_map[fn] = (r["event_id"], r["public_score"], r["public_hits"])

sub_files = list(Path("submissions").glob("*.csv"))
print(f"Total CSVs in submissions/: {len(sub_files)}")

for sf in sub_files:
    info = event_map.get(sf.name)
    if info:
        print(f"SUBMITTED: {sf.name:60s} | {info[0]} | Score: {info[1]:.5f} | Hits: {info[2]}")
    else:
        print(f"NOT SUBMITTED: {sf.name:56s}")
