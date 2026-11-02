"""Read-only evidence audit and synthetic serializer reproduction; no saved pixels decoded."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
E=ROOT/'artifacts/gpc-checkpoint500-offline-diagnosis-20261026'
def read(p): return json.loads(p.read_bytes())
def digest(b): return hashlib.sha256(b).hexdigest()
def check(base,d):
    p=base/d['path']; data=p.read_bytes()
    assert len(data)==d['bytes'] and digest(data)==d['sha256'],str(p)
start=time.monotonic()
d=read(E/'delivery.json'); inv=read(E/'evidence-inventory.json')
check(ROOT,d['evidence_inventory'])
for item in d['tracked_files']: check(ROOT,item)
for item in inv['entries']: check(E,item)
source=read(E/'source-identities.json')
for item in source['local']:check(ROOT,item)
upstream=ROOT.parent/'Tdyssey/third_party/gpc_code'
for item in source['upstream']:check(upstream,item)
pre=read(E/'predecode-code-bindings.json')
for item in pre['current_sources']:check(ROOT,item)
manifest=read(E/'fixed-samples.json')
assert digest((E/'fixed-samples.json').read_bytes())==d['sample_manifest_sha256']
rows=[read(p) for p in (ROOT/'artifacts/gpc-checkpoint500-evaluation-20261026/measurements').glob('*.json')]
assert len(rows)==3996
expected={}
for r in rows:
    cell=(r['shape'],r['stratum'],r['contact_by_h4']); key=(r['anchor'],r['candidate'])
    expected[cell]=min(key,expected.get(cell,key))
observed={(b['shape'],b['stratum'],b['contact_by_h4']):(b['anchor'],b['candidate']) for b in manifest['branches']}
assert observed==expected and len(observed)==8
for branch in manifest['branches']:
    assert {(r['role'],r['mode'],r['horizon'],r['seed']) for r in branch['predictions']}=={
        (role,mode,h,5478) for role in ('frozen','adapted') for mode,h in (('AR',1),('AR',4),('TF',4))}
schedule=read(ROOT/'artifacts/wm-sampling-b-only-20261026/training-schedule.json')['entries']
chosen=[]
for shape in ('T','L'):
    for contact in (False,True):
        selected=[];seen=set()
        for r in schedule:
            key=(r['trajectory_id'],r['anchor'])
            if r['shape']==shape and r['contact']==contact and key not in seen:
                seen.add(key);selected.append(r['cursor'])
                if len(selected)==2:break
        chosen.extend(selected)
assert chosen==[r['cursor'] for r in manifest['train_windows']]
assert len(list((E/'pixels').glob('*.png')))==32 and not (E/'pixels/pixels.json').exists()
receipts=[read(E/f'command-{n}-receipt.json') for n in (1,2,3)]
assert [r['status'] for r in receipts]==['complete','complete','failed']
assert not (E/'command-4-receipt.json').exists()
assert 'Object of type int64 is not JSON serializable' in (E/'command-3.stderr.txt').read_text()

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','pymunk','pygame','mujoco','gym','gymnasium','jax','tensorflow'):
            raise AssertionError('Forbidden real runtime '+fullname)
sys.meta_path.insert(0,BlockRuntime())
sys.path.insert(0,str(ROOT.parent/'Tdyssey/.venv/Lib/site-packages'))
import numpy as np
spec=importlib.util.spec_from_file_location('readonly_pixel_helper',ROOT/'scripts/gpc_checkpoint500_offline_pixels.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
alpha=np.zeros((96,96),dtype=float);alpha[10:18,10:18]=1.;alpha[60:80,60:80]=.1
bounds=(slice(np.int64(8),np.int64(20)),slice(np.int64(8),np.int64(20)))
fake=SimpleNamespace(CONFIG={'shape_gate':{'minimum_object_seed_pixels':12},'chromatic':{'component_seed_alpha':.5}},
    continuous_maps=lambda rgb:(alpha,np.zeros_like(alpha),'red'),_region=lambda a,n:(bounds,64/104))
raw=np.full((3,96,96),.5,dtype=np.float32)
metrics,_=helper.pixel_diagnostics(raw,fake)
assert [type(x).__name__ for x in metrics['roi']]==['int64']*4
failure=None
try: helper._json({'branches':[{'predictions':[{'metrics':metrics}]}]})
except TypeError as error: failure=str(error)
assert failure and 'int64' in failure
fixed=dict(metrics,roi=[int(v) for v in metrics['roi']])
decoded=json.loads(helper._json({'branches':[{'predictions':[{'metrics':fixed}]}]}))
assert decoded['branches'][0]['predictions'][0]['metrics']['roi']==[8,20,8,20]
assert all(k=='roi' or fixed[k]==v for k,v in metrics.items())
fake._region=lambda a,n:(None,0.)
empty,_=helper.pixel_diagnostics(raw,fake);assert json.loads(helper._json(empty))['roi'] is None
assert 'torch' not in sys.modules
result=dict(kind='root_offline_partial_review.v1',commit=d['commit'],
    verified=dict(tracked=len(d['tracked_files']),evidence=len(inv['entries']),local_sources=len(source['local']),
                  upstream_sources=len(source['upstream']),predecode_sources=len(pre['current_sources'])),
    sample_sha256=d['sample_manifest_sha256'],eight_branch_selection_verified=True,train_cursors=chosen,
    command_seconds=sum(r['elapsed_seconds'] for r in receipts),commands_spent=3,stopped_on_first_numerical_error=True,
    command1_interpreter_peak='unknown; redirector only, not retroactively certified',
    serializer_reproduction=dict(original_error=failure,carrier='four NumPy int64 ROI bounds',
        synthetic_explicit_int_projection_json_passed=True,other_values_unchanged=True,no_roi_json_passed=True,
        limit='Demonstrates concrete carrier; does not recover exact lost in-memory report or prove no other carrier.'),
    viewed_existing_pngs=['IL00-away-comparison.png','IT01-toward-comparison.png','IL00-away-segmentation.png'],
    visual_observation='Recognizable object/pusher with pink background and weak outside-ROI alpha in viewed panels. T contact AR H4 local errors remain. No scientific acceptance reversal.',
    contract_review='Inspected actual dataset/rollout, pinned native objective excerpt and source-AST arithmetic tests; no definite production-contract bug established. No native neural test performed.',
    decision='Accept partial diagnosis and honest stop; authorize separately only minimal offline serialization repair and fixed-sample re-export, no new model/data/threshold work.',
    operations=dict(saved_NPY_NPZ_decodes=0,held_content_opens=0,weight_opens=0,real_Torch=0,models=0,predictions=0,physics=0),
    seconds=time.monotonic()-start)
with (HERE/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
