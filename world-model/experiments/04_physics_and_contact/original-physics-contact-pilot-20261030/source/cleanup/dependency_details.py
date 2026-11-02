import collections
import json
from pathlib import Path
import numpy as np

d=Path(__file__).resolve().parent
repo=d.parents[2]
base=json.loads((d/'inventory-base.json').read_text(encoding='utf-8'))
groups=collections.defaultdict(lambda: [0,0,[]])
for r in base['files']:
    p=Path(r['relative_path'])
    if r['suffix'] in {'.gz','.mkv'}:
        key='/'.join(p.parts[:3])
        g=groups[key];g[0]+=1;g[1]+=r['bytes']
        if len(g[2])<2:g[2].append(p.name)
print('RGB_PACKS',json.dumps(dict(groups),indent=2))
names = [
 'native-rgb-supervised-20261029/r2c1r1/run/evaluation/cohort-train_visual.npz',
 'native-rgb-supervised-20261029/r2c1r1/run/evaluation/cohort-val_motion.npz',
 'native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/evaluation/epoch1/natural-batch00-step0.npz',
 'native-rgb-supervised-20261029/r3a/trajectories/R3A-train-B-000-a0.npz',
]
for name in names:
    p=repo/'artifacts'/name
    if p.exists():
        with np.load(p,allow_pickle=False) as z:
            print(name,[(key,z[key].shape,str(z[key].dtype)) for key in z.files])
print('R3A_TRAIN/VAL_FILES')
split=collections.defaultdict(lambda:[0,0,[]])
for r in base['files']:
    rel=r['relative_path'].replace('\\','/')
    if '/r3a/trajectories/' in rel:
        p=Path(rel);kind='train' if p.name.startswith('R3A-train-') else 'seen_val' if p.name.startswith('R3A-seen_val-') else 'other'
        value=split[kind];value[0]+=1;value[1]+=r['bytes']
        if len(value[2])<5:value[2].append(p.name)
print(json.dumps(dict(split),indent=2))
