import json

with open('results/canonical_ledger_rebuild_20261108/canonical_events_catalog.json', 'r', encoding='utf-8') as f:
    events = json.load(f)

for e in events:
    name = e.get('display_name', '')
    if any(k in name.lower() for k in ['isolate', 'sonda', 'probe', 'cohort']):
        eid = e.get('event_id')
        score = e.get('public_score')
        hits = e.get('public_hits')
        print(f"{eid}: score={score} ({hits}) | {name}")
