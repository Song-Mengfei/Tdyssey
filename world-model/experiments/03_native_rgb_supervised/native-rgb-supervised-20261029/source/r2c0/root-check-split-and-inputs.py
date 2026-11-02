"""Independent saved-data and explicit raw-array interface audit; CPU only."""
import sys,json,math,hashlib,inspect
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc
from tdyssey.native_rgb_r2c0_data import R2C0Dataset,choose_split,read_source_catalogue
from tdyssey.native_rgb_r2c0_evaluation import rollout_raw,normalize_actions
OUT=EXPERIMENT/'r2c0';DATA=OUT/'data';m=read(DATA/'manifest.json');split=read(DATA/'split-plan.json')
assert m['counts']==dict(trajectories=53,train=41,seen_val=12,H1=1113,H4=954,train_H1=861,val_H1=252,train_H4=738,val_H4=216)
assert choose_split(read_source_catalogue(EXPERIMENT/'r2b'))==split
entries={r['trajectory_id']:r for r in m['entries']}
train={r['family_id'] for r in m['entries'] if r['split']=='train'};val={r['family_id'] for r in m['entries'] if r['split']=='seen_val'}
assert len(train)==41 and len(val)==12 and not train&val
feasible=read(OUT/'independent-split-feasibility.json')['letters']
for row in split['selections']:
 p=dict(val_translation_family=entries[row['translation_family']]['family_id'],val_rotation_family=entries[row['rotation_family']]['family_id'])
 assert p in feasible[row['letter']]['feasible_ordered_val_pairs']
 assert all(row['train_kept_support'].values())
arrays_cache={};source=EXPERIMENT/'r2b';green=red=0
for e in m['entries']:
 assert e['protocol_sha256']=='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
 assert sha(source/e['file'])==e['sha256'] and sha(source/e['metadata_file'])==e['metadata_sha256']
 with np.load(source/e['file'],allow_pickle=False) as z:arrays_cache[e['trajectory_id']]={k:z[k].copy() for k in ['rgb','states','actions']}
 rgb=arrays_cache[e['trajectory_id']]['rgb'].astype(np.int16)
 red+=int(((rgb[:,0]>150)&(rgb[:,0]>rgb[:,1]+70)&(rgb[:,0]>rgb[:,2]+70)).sum())
 green+=int(((rgb[:,1]>100)&(rgb[:,1]>rgb[:,0]+30)&(rgb[:,1]>rgb[:,2]+30)).sum())
checks=0
for name,N in [('train',861),('seen_val',252)]:
 ds=R2C0Dataset(DATA/'manifest.json',split=name,cache_bytes=1024**2);assert len(ds)==N
 for i,row in enumerate(ds.window_rows):
  item=ds[i];a=row['anchor'];src=arrays_cache[row['trajectory_id']]
  assert np.array_equal(item['image'],src['rgb'][a-3:a+2].astype(np.float32)/np.float32(255))
  original=src['actions'][a-3:a+1].astype(np.float32)/511*2-1
  assert np.array_equal(item['action'],original)
  assert np.array_equal(item['action'],normalize_actions(src['actions'][a-3:a+1]))
  assert item['image'].shape==(5,3,96,96) and item['action'].shape==(4,2)
  assert row['moving']==(row['translation_image96_px']>=2 or row['abs_rotation_deg']>=5)
  checks+=1
 ds.close()
assert checks==1113 and green==0 and red==0
e=next(e for e in m['entries'] if e['split']=='train');z=arrays_cache[e['trajectory_id']];anchor=7
h=z['rgb'][anchor-3:anchor+1][None].astype(np.float32)/255;actions=z['actions'][anchor-3:anchor+4][None]
calls=[]
def fake(history,normalized_actions,**ctx):
 step=ctx['step'];assert history.shape==(1,4,3,96,96)
 assert np.array_equal(normalized_actions,normalize_actions(actions[:,step:step+4]))
 if step:assert np.all(history[:,-1]==np.float64(1.125+step-1))
 calls.append(history.copy());return np.full((1,3,96,96),1.125+step,dtype=np.float64)
mock=rollout_raw(fake,h,actions,window_ids=['mock:7'],cohort='root_mock_only')
assert not mock['hard_stop'] and len(calls)==4 and np.all(mock['h1']==1.125) and np.all(mock['h4']==4.125)
assert mock['status_h4']==['finite_out_of_range'] and 'truth' not in inspect.signature(rollout_raw).parameters
assert 'torch' not in sys.modules
prior=read(OUT/'input-identities.json')
for r in prior['files']:assert sha(r['path'])==r['sha256']
result=dict(kind='R2C0-root-data-and-interface-audit.v1',at_utc=utc(),passed=True,all1113_saved_H1_model_arrays_exact=True,H1_H4_counts=m['counts'],split_deterministic_recomputed=True,split_disjoint_whole_families=True,all_training_motion_and_direction_constraints=True,network_RGB_is_saved_uint8_div255=True,normalization_bitwise_original=True,all_labels_excluded_from_network_inputs_by_array_bridge=True,RGB_green_pixels=green,RGB_red_pixels=red,all_R2B_and_parent_review_inputs_unchanged=True,finite_out_of_range_mock_raw_feedback_exact=True,real_model_calls=0,simulator_instances=0,physics_controls=0,torch_imported=False)
atomic_json(OUT/'root-data-interface-audit.json',result);print(json.dumps(result))
