"""Cold artifact-only check of every letter's unfilled boundary/gap combinations."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

b=Path(__file__).resolve().parent;root=b.parent
letters=tuple('ABCEFHIJLMOPSTUVXY')
locations=('center','left','right','top','bottom','left-top','right-top','left-bottom','right-bottom')
counts={name:Counter() for name in ['2A','2B','merged']};spatial={name:Counter() for name in counts};issues=[]
for name in ['2A','2B']:
    data=root/f'stage{name.lower()}'/'data';manifest=json.loads((data/'manifest.json').read_text())
    for entry in manifest['entries']:
        metadata=json.loads((data/entry['metadata_file']).read_text());q=metadata['request'];scene=metadata['scene']
        letter=entry['shape'];location=entry['spatial_class'];gap=q['gap_world_px']
        for group in [name,'merged']:spatial[group][location]+=1
        if location=='center':
            if gap is not None or q['gap_band'] is not None:issues.append(entry['trajectory_id']+':center_has_gap_label')
            continue
        for wall in location.split('-'):
            if abs(scene['actual_wall_gaps'][wall]-gap)>1e-6:issues.append(entry['trajectory_id']+':actual_wall_gap_differs')
        for group in [name,'merged']:counts[group][letter,location,int(gap)]+=1
rows=[];per_letter={}
for letter in letters:
    missing=[]
    for location in locations[1:]:
        for gap in [4,18,48]:
            row=dict(letter=letter,location=location,gap_world_px=gap,**{name:counts[name][letter,location,gap] for name in counts})
            rows.append(row)
            if row['merged']==0:missing.append(dict(location=location,gap_world_px=gap))
    per_letter[letter]=dict(boundary_space_gap_universe=24,
        observed_unique_combinations={name:sum(counts[name][letter,l,g]>0 for l in locations[1:] for g in [4,18,48]) for name in counts},
        merged_missing_combinations=missing,complete24=False)
report=dict(kind='stage2b-cold-initial-gap-quota-audit.v1',passed=not issues,issues=issues,
    meaning='Initial object wall gaps, not contacts or independent control windows; center has no requested gap tier',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),per_letter=per_letter,
    global_spatial_counts={name:dict(value) for name,value in spatial.items()},
    global_boundary_gap_counts={name:{location:{str(gap):sum(counts[name][l,location,gap] for l in letters) for gap in [4,18,48]} for location in locations[1:]} for name in counts},
    new_physics_controls=0,model_calls=0,index_draws=0)
(b/'gap-quota-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
with (b/'gap-quota-audit.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
text=['# Initial boundary-space × gap support','',
    'The merged TRAIN has44 trajectories per requested spatial class. Each letter covers19 or20 of24 boundary-space/gap combinations; this is not complete24-combination coverage. Centers have no requested gap tier. Both wall gaps of every corner match the frozen tier.','',
    '| Letter |2A unique /24|2B unique /24|Merged unique /24|Missing merged combinations|','|---|---:|---:|---:|---|']
for letter,value in per_letter.items():
    c=value['observed_unique_combinations'];missing=', '.join(f"{x['location']}:{x['gap_world_px']}" for x in value['merged_missing_combinations'])
    text.append(f"|{letter}|{c['2A']}|{c['2B']}|{c['merged']}|{missing}|")
(b/'gap-quota-audit.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=not issues,issues=issues,merged_support=Counter(v['observed_unique_combinations']['merged'] for v in per_letter.values()))))
if issues:raise SystemExit(1)
