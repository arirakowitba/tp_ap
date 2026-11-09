with open('scripts/rebuild_canonical_submission_ledger_20261107.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace OUT_DIR
content = content.replace('canonical_ledger_rebuild_20261107', 'canonical_ledger_rebuild_20261108')
content = content.replace('133 Events / 106 Scored Vectors', '138 Events / 111 Scored Vectors')
content = content.replace('Target: 133', 'Target: 138')
content = content.replace('Target: 94', 'Target: 99')

# Replace delta loading to load both
old_delta = """    # 4. Incorporate the 5 submissions of 2026-11-07 (EVT-129 to EVT-133) strictly from Git Pinned Commits
    delta_path = ROOT / "results" / "canonical_ledger_delta_20261107" / "events_129_133.json"
    with open(delta_path, encoding="utf-8") as f:
        delta_data = json.load(f)

    for delta_evt in delta_data["events"]:"""

new_delta = """    # 4. Incorporate submissions of 2026-11-07 and 2026-11-08 strictly from Git Pinned Commits
    delta_paths = [
        ROOT / "results" / "canonical_ledger_delta_20261107" / "events_129_133.json",
        ROOT / "results" / "canonical_ledger_delta_20261108" / "events_134_138.json"
    ]
    all_delta_events = []
    for dp in delta_paths:
        with open(dp, encoding="utf-8") as f:
            all_delta_events.extend(json.load(f)["events"])

    for delta_evt in all_delta_events:"""

assert old_delta in content, "old_delta not found in content"
content = content.replace(old_delta, new_delta)

with open('scripts/rebuild_canonical_submission_ledger_20261108.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully generated scripts/rebuild_canonical_submission_ledger_20261108.py")
