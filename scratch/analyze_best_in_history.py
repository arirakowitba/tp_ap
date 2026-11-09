import json
import pandas as pd
from pathlib import Path

# Load all scored submissions from canonical ledger
ledger_path = 'results/canonical_ledger_rebuild_20261107/canonical_submission_ledger_20261107.json'
with open(ledger_path) as f:
    ledger = json.load(f)

print(f"Total entries in ledger: {len(ledger)}")
scored = [e for e in ledger if e.get('score') is not None]
print(f"Total scored entries: {len(scored)}")

df_scored = pd.DataFrame(scored)
df_scored = df_scored.sort_values(by='score', ascending=False)
print("\nTop 15 submissions in history:")
print(df_scored[['key', 'score', 'hits', 'name']].head(15).to_string())
