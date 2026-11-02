"""Whole-source duplicate disclosure only; no model/simulator, no split selection."""
import sys,json,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc
OUT=EXPERIMENT/'r2c0';DATA=EXPERIMENT/'r2b'
rows=[r for r in read(DATA/'index.json')['slots'] if r['status']=='completed']
items=[];statehash=defaultdict(list);actionhash=defaultdict(list);rgbhash=defaultdict(list)
frames=defaultdict(list);stateframes=defaultdict(list);exactpose=defaultdict(list)
def hasharray(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
for r in rows:
 m=read(DATA/r['metadata_file'])
 with np.load(DATA/r['trajectory_file'],allow_pickle=False) as z:
  s=z['states'].copy();a=z['actions'].copy();rgb=z['rgb'].copy()
 tid=r['slot_id'];statehash[hasharray(s)].append(tid);actionhash[hasharray(a)].append(tid);rgbhash[hasharray(rgb)].append(tid)
 exactpose[hasharray(s[0])].append(tid)
 for j,image in enumerate(rgb):frames[hasharray(image)].append([tid,j])
 for j,state in enumerate(s):stateframes[hasharray(state)].append([tid,j])
 items.append(dict(id=tid,letter=r['letter'],states=s,actions=a))
near=[];mindists=[]
for i,left in enumerate(items):
 for right in items[i+1:]:
  if left['letter']!=right['letter']:continue
  x=left['states'][0];y=right['states'][0]
  pusher=float(np.linalg.norm(x[:2]-y[:2]));origin=float(np.linalg.norm(x[2:4]-y[2:4]));angle=abs(math.degrees(math.atan2(math.sin(x[4]-y[4]),math.cos(x[4]-y[4]))))
  entry=dict(left=left['id'],right=right['id'],letter=left['letter'],initial_pusher_world_px=pusher,initial_object_origin_world_px=origin,initial_angle_deg=angle,action_RMS_world_px=float(np.sqrt(np.mean((left['actions']-right['actions'])**2))))
  if pusher<=15 and origin<=12 and angle<=5:near.append(entry)
  delta=left['states'][:,None,:]-right['states'][None,:,:]
  pd=np.linalg.norm(delta[:,:,:2],axis=2);od=np.linalg.norm(delta[:,:,2:4],axis=2)
  ad=np.abs(np.degrees(np.arctan2(np.sin(delta[:,:,4]),np.cos(delta[:,:,4]))))
  hits=np.argwhere((pd<=15)&(od<=12)&(ad<=5))
  if len(hits):mindists.append(dict(left=left['id'],right=right['id'],letter=left['letter'],matching_frame_pairs=len(hits),indices=hits.tolist()))
def dupegroups(mapping,whole=False):
 return [dict(sha256=k,items=v) for k,v in mapping.items() if len(v)>1 and (whole or len({x[0] for x in v})>1)]
result=dict(kind='R2C0-all-source-duplicates.v1',at_utc=utc(),source='R2B53 only',trajectories=53,raw_frames=1325,
 exact_trajectory_state_duplicates=dupegroups(statehash,True),exact_trajectory_action_duplicates=dupegroups(actionhash,True),exact_trajectory_RGB_duplicates=dupegroups(rgbhash,True),exact_initial_state_duplicates=dupegroups(exactpose,True),exact_cross_trajectory_RGB_frame_duplicates=dupegroups(frames),exact_cross_trajectory_state_frame_duplicates=dupegroups(stateframes),
 near_duplicate_rule='Same letter, pusher<=15worldpx,object-origin<=12worldpx,wrapped angle<=5deg; disclosure only. Shared wall/letter/background or high background pixel fraction is not sufficient to call a full scene duplicate.',near_initial_scene_pairs=near,near_any_saved_frame_pairs=mindists,
 caveat='Near-repeat audit does not establish unbiased natural sampling; six seen letters, shared source scene families and short trajectories are disclosed. This audit cannot modify the frozen split or add data.',new_physics_controls=0,new_model_calls=0)
atomic_json(OUT/'source-duplicate-audit.json',result)
print(json.dumps({k:len(v) for k,v in result.items() if isinstance(v,list)}))
