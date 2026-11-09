with open('scripts/rebuild_canonical_submission_ledger_20261107.py', encoding='utf-8') as f:
    text = f.read()

for line in text.splitlines()[:25]:
    print(line)
