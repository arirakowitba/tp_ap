"""Compare the top 5 submissions in history to see what made them score high."""
import pandas as pd
from pathlib import Path

df_events = pd.read_csv("results/canonical_ledger_rebuild_20261108/canonical_events_catalog.csv")
top_events = df_events.sort_values(by="public_score", ascending=False).head(8)

print("=== TOP HISTORICAL SUBMISSIONS ===")
for _, r in top_events.iterrows():
    print(f"Event: {r['event_id']:8s} | Score: {r['public_score']:.5f} | Hits: {r['public_hits']} | Name: {r['display_name']}")
