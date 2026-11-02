"""Read append-only progress only; no model/checkpoint reads or scoring."""
import json
from pathlib import Path
stage=Path(__file__).resolve().parent
p=stage/'run/events.jsonl'
with p.open('rb')as f:
    f.seek(max(0,p.stat().st_size-1048576))
    lines=f.read().decode('utf-8',errors='ignore').splitlines()
for line in reversed(lines):
    try:r=json.loads(line)
    except ValueError:continue
    if r['kind']=='update_complete':
        print(json.dumps(dict(update=r['update'],new_complete=r['new_complete_updates'],attempt=r['attempt'],model_seconds=r['model_seconds'],
            observation3000_closed=(stage/'run/milestones/observation-3000.json').exists(),
            observation6000_closed=(stage/'run/milestones/observation-6000.json').exists(),model_work_closed=(stage/'model-work.json').exists())))
        break
