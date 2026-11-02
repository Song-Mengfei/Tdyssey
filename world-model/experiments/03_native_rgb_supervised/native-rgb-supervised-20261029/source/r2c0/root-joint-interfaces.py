"""Joint real saved-array/order + injected mock-rollout audit; no model/env."""
import sys,json,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc
from tdyssey.native_rgb_r2c0_data import R2C0Dataset,metadata_only_sampler_factory
from tdyssey.native_rgb_r2c0_interfaces import PreparedInputs,evaluate_prepared_cohort
from tdyssey.native_rgb_r2c0_evaluation import rollout_raw
from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
OUT=EXPERIMENT/'r2c0';D=OUT/'data';E=OUT/'evaluation'
ds=R2C0Dataset(D/'manifest.json');order=metadata_only_sampler_factory(ds)
frozen=read(D/'draws4800.json')['indices'];assert order.take(4800)==frozen
restored=metadata_only_sampler_factory(ds);restored.load_state_dict(read(D/'order-state480.json'))
assert restored.take(4320)==frozen[480:]
cohort=load_eval_cohort(D/'manifest.json',E/'indices.json','val_natural')
def make():
 o=metadata_only_sampler_factory(ds)
 return PreparedInputs(ds,o,dataset_identity=ds.identity(),sampling_identity=o.identity(),evaluation_identity=cohort['identity'])
p=make();assert list(p.begin_update())==frozen[:16]
one=p.next_microbatch();assert set(one['model_inputs'])=={'image','action'}
assert one['model_inputs']['image'].shape==(4,5,3,96,96) and one['model_inputs']['action'].shape==(4,4,2)
saved=p.state_dict();q=make();q.load_state_dict(saved)
for _ in range(3):
 a=p.next_microbatch();b=q.next_microbatch();assert a['indices']==b['indices']
 assert np.array_equal(a['model_inputs']['image'],b['model_inputs']['image']) and np.array_equal(a['model_inputs']['action'],b['model_inputs']['action'])
assert p.pending is q.pending is None
assert list(p.begin_update())==list(q.begin_update())==frozen[16:32]
mockcalls=[]
def fake(images,actions,**context):
 assert images.shape[1:]==(4,3,96,96) and actions.shape[1:]==(4,2)
 assert set(context)=={'seed','cohort','step','window_ids'}
 mockcalls.append((images.copy(),actions.copy()));return images[:,-1].copy()
kwargs=dict(cohort='val_natural',expected_identity=cohort['identity'])
try:evaluate_prepared_cohort(rollout_raw,fake,cohort,execution_guard=lambda _:False,**kwargs)
except PermissionError:pass
else:raise AssertionError('False execution guard must reject callback')
assert not mockcalls
r=evaluate_prepared_cohort(rollout_raw,fake,cohort,execution_guard=lambda _:True,**kwargs)
first=[(a.copy(),b.copy()) for a,b in mockcalls];mockcalls.clear()
changed=dict(cohort,truth_h1=np.full_like(cohort['truth_h1'],13),truth_h4=np.full_like(cohort['truth_h4'],71))
r2=evaluate_prepared_cohort(rollout_raw,fake,changed,execution_guard=lambda _:True,**kwargs)
assert len(first)==len(mockcalls)==12
assert all(np.array_equal(a,x) and np.array_equal(b,y) for (a,b),(x,y) in zip(first,mockcalls))
assert np.array_equal(r['h1'],r2['h1']) and np.array_equal(r['h4'],r2['h4']) and not r['hard_stop']
assert 'torch' not in sys.modules
assert sha(EXPERIMENT/'native-rgb-cost-ledger.json')=='2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6'
result=dict(kind='R2C0-root-joint-interface-audit.v1',at_utc=utc(),passed=True,
 full4800_stream_matches_frozen=True,state480_continuation_matches4320=True,
 real16draw_micro4_arrays_exact=True,mid_microbatch_data_only_restore_exact=True,
 optimizer_gradient_resume_tested=False,network_inputs_only_image_action=True,
 evaluation_false_guard_prevents_callback=True,changed_future_truth_cannot_change_callback_inputs_or_outputs=True,
 mock_callbacks=24,actual_model_calls=0,environment_constructions=0,physics_controls=0,torch_imported=False,
 data_manifest_sha256=sha(D/'manifest.json'),eval_indices_sha256=sha(E/'indices.json'),eval_protocol_sha256=sha(E/'protocol.json'),
 source_files={n:sha(ROOT/n) for n in ['tdyssey/native_rgb_r2c0_data.py','tdyssey/native_rgb_r2c0_interfaces.py','tdyssey/native_rgb_r2c0_evaluation.py','scripts/native_rgb_r2c0_evaluation.py']})
atomic_json(OUT/'root-joint-interface-audit.json',result);ds.close();print(json.dumps(result))
