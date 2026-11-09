import json
from pathlib import Path
import pandas as pd

ROOT = Path('.')
models = list(ROOT.glob('results/**/*.csv'))
reports = list(ROOT.glob('results/**/*.md'))

print(f"Total CSVs in results: {len(models)}")
for r in reports:
    try:
        t = r.read_text(encoding='utf-8')
        for line in t.splitlines():
            if 'accuracy' in line.lower() or 'oof' in line.lower() or 'logloss' in line.lower():
                if any(k in line for k in ['0.94', '0.93', '0.92', '0.91']):
                    print(f"{r.name[:45]}: {line.strip()[:80]}")
    except:
        pass
