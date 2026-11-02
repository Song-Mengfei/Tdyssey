from pathlib import Path
import gzip,json,math,sys
import numpy as np
import cv2
from shapely.geometry import Point,Polygon
repo=Path(__file__).resolve().parents[4];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_env import shape_geometry
from tdyssey.envs.official import official_shape
d=Path(__file__).resolve().parent;geometry=shape_geometry('B')
holes=[Polygon(r)for r in geometry.interiors]
out=[]
for name in ['B-corner_tl','B-corner_tr']:
 original=json.load(gzip.open(d/f'{name}-original-substeps.json.gz','rt'))
 rows=[r for c in original for r in c['substeps']]
 counts=dict(solid=0,hole=0,outside=0,overlap_gt1=0,material_overlap_gt2=0)
 for row in rows:
  state=np.asarray(row['post_state']);delta=state[:2]-state[2:4];a=state[4]
  local=Point(math.cos(a)*delta[0]+math.sin(a)*delta[1],-math.sin(a)*delta[0]+math.cos(a)*delta[1])
  region='solid'if geometry.contains(local)else'hole'if any(h.contains(local)for h in holes)else'outside'
  counts[region]+=1;counts['overlap_gt1']+=max(row['penetrations'])>1;counts['material_overlap_gt2']+=max(row['penetrations'])>2
 counterpart=repo/'artifacts/original-physics-contact-pilot-20261030/physics-regression/v4'/f'{name}-candidate.npz'
 with np.load(d/f'{name}-original.npz')as z:oldactions=z['actions'].copy();oldrgb0=z['rgb'][0].copy()
 with np.load(counterpart)as z:
  action_equal=bool(np.array_equal(oldactions,z['actions']));rgb_equal=bool(np.array_equal(oldrgb0,z['rgb'][0]));newrows=json.loads(z['substep_metadata_json'].item())
 cap=cv2.VideoCapture(str(d/f'{name}-before-after-10Hz.mp4'));decoded=0;first_counts=None
 while True:
  ok,frame=cap.read()
  if not ok:break
  if decoded==0:first_counts=[int((np.min(frame[90:474,8:392],axis=2)<220).sum()),int((np.min(frame[90:474,408:792],axis=2)<220).sum())]
  decoded+=1
 cap.release()
 assert action_equal and rgb_equal and decoded==25 and min(first_counts)>0
 out.append(dict(name=name,initial_RGB_bitwise_match=rgb_equal,requested_actions_bitwise_match=action_equal,
  original_completed_integration_not_geometry_pass=True,original_microsteps=len(rows),original_region_counts=counts,
  v6_microsteps=len(newrows),v6_illegal_regions=sum(r['pusher_region']!='outside'for r in newrows),v6_overlap_gt1=sum(max(r['penetrations'])>1 for r in newrows),
  actual_video_decoded_frames=decoded,first_decoded_nonwhite_pixels_left_right=first_counts))
result=dict(geometry_hash=official_shape('B').geometry_hash,rows=out,new_physics_controls=0,new_model_calls=0,optimizer_updates=0)
(d/'geometry-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
