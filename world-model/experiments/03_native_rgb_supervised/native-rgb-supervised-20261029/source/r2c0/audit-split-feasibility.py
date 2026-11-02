"""Independent saved-state feature constraints for a whole-family split."""
import sys,itertools,json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,atomic_json,utc
OUT=EXPERIMENT/'r2c0';d=read(EXPERIMENT/'r2b/report/r2b-report.json')
families=defaultdict(dict)
for row in d['populations']['h1_windows']:
 f=families[row['letter']].setdefault(row['family_id'],dict(translation=0,rotation=0,positive=0,negative=0,H4eligible_translation=0,H4eligible_rotation=0))
 trans=row['translation_image96_px']>=2;pos=row['signed_rotation_rad']>=__import__('math').radians(5);neg=row['signed_rotation_rad']<=-__import__('math').radians(5)
 for key,flag in [('translation',trans),('rotation',pos or neg),('positive',pos),('negative',neg),('H4eligible_translation',trans and row['anchor']<=20),('H4eligible_rotation',(pos or neg) and row['anchor']<=20)]:f[key]+=int(flag)
result={}
for letter,fs in families.items():
 feasible=[]
 for trans,rot in itertools.permutations(sorted(fs),2):
  if not fs[trans]['translation'] or not fs[rot]['rotation']:continue
  train=[f for fid,f in fs.items() if fid not in (trans,rot)]
  if not all(sum(f[k] for f in train)>0 for k in ['translation','rotation','positive','negative','H4eligible_translation','H4eligible_rotation']):continue
  feasible.append(dict(val_translation_family=trans,val_rotation_family=rot))
 result[letter]=dict(source_families=len(fs),features_by_family=fs,feasible_ordered_val_pairs=feasible,feasible_ordered_val_pair_count=len(feasible))
atomic_json(OUT/'independent-split-feasibility.json',dict(kind='R2C0-independent-split-constraint-audit.v1',at_utc=utc(),letters=result,all_letters_feasible=all(v['feasible_ordered_val_pair_count']>0 for v in result.values()),models_or_physics=0))
print(json.dumps({k:v['feasible_ordered_val_pair_count'] for k,v in result.items()}))
