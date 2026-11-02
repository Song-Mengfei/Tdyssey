"""Lazy original HID GPU statistics + qualified HMF CPU measurements."""
import importlib.util,sys
from functools import lru_cache
from authorization import require_permit

@lru_cache(maxsize=8)
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def score_native_batch(permit,label,batch,mode,gpu_prediction,cpu_prediction,truth_arrays):
    c=require_permit(permit)
    if not permit.scoring_parity_passed:raise RuntimeError('Saved-only field/qualification parity not yet passed')
    if gpu_prediction is None:raise RuntimeError('No silent CPU basic-statistics fallback')
    scorer=load_module('_loss_screen_original_hid_metrics',c['hid_metrics_path'])
    record=dict(section='A',label=label,batch_id=batch['batch_id'],mode=mode,rows=batch['rows'])
    result=scorer.score_batch(permit.root,c,record,cpu_prediction,gpu=gpu_prediction,truth_arrays=truth_arrays)
    for row in result['rows']:
        row['B_holes_status']=row['visible_topology_status'] if row['letter']=='B' else 'not_applicable'
        row['C_opening_status']=row['visible_topology_status'] if row['letter']=='C' else 'not_applicable'
    return result

def truth_series(row,arrays,truth_features):
    a=int(row['anchor'])
    return dict(initial_last=arrays['rgb'][a],targets=arrays['rgb'][a+1:a+17],truth_features=truth_features,
                recorded_rotation_deg=row['recorded_rotation_deg'],target_motion=row['target_motion'])

