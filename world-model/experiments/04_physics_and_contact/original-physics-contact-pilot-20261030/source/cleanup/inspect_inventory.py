import collections
import json
from pathlib import Path

d = Path(__file__).resolve().parent
data = json.loads((d/'inventory-base.json').read_text(encoding='utf-8'))
files = data['files']
print('CHECKPOINTS')
print(json.dumps([r for r in files if r['suffix'] in {'.pt','.pth','.ckpt','.zip'}], indent=2))
groups = collections.defaultdict(lambda: [0,0,[]])
for row in files:
    if row['suffix'] in {'.npz','.npy','.pkl','.pickle'}:
        rel = Path(row['relative_path'])
        key = '/'.join(rel.parts[:4]) if len(rel.parts)>4 else '/'.join(rel.parts[:-1])
        v=groups[key];v[0]+=1;v[1]+=row['bytes']
        if len(v[2])<3:v[2].append(rel.name)
print('ARRAY_GROUPS')
print(json.dumps(dict(sorted(groups.items(),key=lambda kv:kv[1][1],reverse=True)), indent=2))
sizes=collections.Counter(r['suffix'] for r in files)
print('FILE_EXT_COUNTS')
print(json.dumps(sizes,indent=2))
