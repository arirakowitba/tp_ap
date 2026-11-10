import json
import glob

with open('results/canonical_ledger_rebuild_20261108/canonical_events_catalog.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

scored = [e for e in events if e.get('public_score') is not None]
print(f"Total events: {len(events)}, scored: {len(scored)}")
for e in scored[-10:]:
    eid = e.get('event_id', 'UNKNOWN')
    score = e.get('public_score')
    hits = e.get('public_hits')
    name = e.get('display_name')
    print(f"{eid:10s} | Score: {score:.5f} | Hits: {hits} | {name}")
