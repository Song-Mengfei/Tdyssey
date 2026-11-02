"""Freeze the already approved four-case diagnostic. No physical construction."""
from pathlib import Path
import hashlib
import json
import sys
import uuid
import importlib.metadata
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(ROOT))
from tdyssey.direct_target_collection import read,write,sha
from tdyssey.native_rgb_original_physics import PROTOCOL_SHA256
from tdyssey.original_contact_pilot import storage_snapshot
IDS=['DT-O-01-0-b0','DT-T-00-0-b0','DT-L-05-0-b0','DT-C-06-1-b0'];CAPS=[8,8,8,11]

def main():
    if (ROOT/'runtime-config.json').exists():raise FileExistsError('Already frozen; no replacement')
    review=REPO/'artifacts/direct-target-diagnosis-review-20261031';old=REPO/'artifacts/direct-target-acceptance-20261031'
    approval=read(ROOT/'APPROVAL.json')
    for rel,expected in approval['source_specification_sha256'].items():
        if sha(review/rel)!=expected:raise ValueError('Approved specification changed: '+rel)
    assert approval['max_attempts']==4 and approval['total_control_cap']==35 and approval['total_regular_substep_cap']==350
    assert [x['id'] for x in approval['cases']]==IDS and [x['control_cap'] for x in approval['cases']]==CAPS
    oldcfg=read(old/'runtime-config.json');bindings={Path(x['path']).resolve():x['sha256'] for x in oldcfg['bindings']}
    for path,expected in bindings.items():
        if sha(path)!=expected:raise ValueError('Old source/evidence changed: '+str(path))
    preservation=[]
    # Bind all old120 raw+metadata files, not merely the selected four.
    for p in sorted([*(old/'collection/raw').glob('*.npz'),*(old/'collection/attempts').glob('*.json')],key=str):
        preservation.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    assert len(preservation)==240
    write(ROOT/'OLD_RECORDS_BASELINE.json',dict(files=preservation,old_records=120,old_labels_modified=False),exclusive=True)
    roots=read(old/'storage-roots.json')['roots']
    for p in (review,ROOT):
        if str(p) not in roots:roots.append(str(p))
    write(ROOT/'storage-roots.json',dict(roots=roots,shared_cap_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2),exclusive=True)
    required=[ROOT/p for p in ('observer.py','tests_offline.py','run_diagnostic.py','report.py','audit_results.py','prepare.py',
        'APPROVAL.json','HUMAN_AUTHORIZATION.json','OLD_RECORDS_BASELINE.json','storage-roots.json')]
    required.extend(review/rel for rel in approval['source_specification_sha256'])
    required.extend([old/'runtime-config.json',old/'plan.json',old/'collection/ledger.json'])
    required.extend(Path(x['path']) for x in preservation)
    distribution=importlib.metadata.distribution('pymunk');pkg=Path(distribution.locate_file('pymunk'))
    required.extend(pkg/p for p in ('__init__.py','space.py','body.py','arbiter.py','contact_point_set.py'))
    required.extend(pkg.glob('_chipmunk*.pyd'));required.append(Path(distribution._path)/'METADATA')
    for p in required:bindings[p.resolve()]=sha(p)
    c=dict(kind='four-case-original-continuation-diagnostic.v1',run_id='D4C-20261031-'+uuid.uuid4().hex,execution_authorized=True,
        artifact_root=str(ROOT),repo=str(REPO),source_root=oldcfg['source_root'],old_root=str(old),source_plan=str(old/'plan.json'),
        source_diagnostic_review=str(review),source_diagnostic_review_hashes=approval['source_specification_sha256'],
        case_ids=IDS,control_caps=CAPS,max_attempts=4,max_controls=35,max_regular_substeps=350,
        future_controls_after_failed_control=4,initialization_substeps='Count actual originalreset_scene calls separately; no exploration',
        physics_protocol_sha256=PROTOCOL_SHA256,physics_settings=oldcfg['physics_settings'],actual_pymunk_version=distribution.version,
        observer_change=dict(AO_OW_gt1='would_stop_only_in_four_cases',AW_gt1='hard_stop',nonfinite='hard_stop',solid_closedhole='hard_stop',object_outer_envelope=[3,508]),
        original_action_domain=[0,512],action_transformation=False,state_match_atol=1e-8,action_match='exact',rgb_match='exact_including_original_partial_cutpoint',
        sampler_changes_authorized=False,production_guard_changes_authorized=False,training_eligible=False,model_calls=0,optimizer_updates=0,
        retries=0,automatic_extension=False,next_phase_released=False,new_output_cap_bytes=16*1024**2,shared_cap_bytes=3*1024**3,
        atomic_reserve_bytes=256*1024**2,storage_roots=str(ROOT/'storage-roots.json'),parent_thread_id=approval['parent_thread_id'],parent_host_id='local',
        bindings=[dict(path=str(p),sha256=v) for p,v in sorted(bindings.items(),key=lambda x:str(x[0]))])
    snapshot=storage_snapshot([*roots,ROOT])
    if snapshot['missing_roots'] or snapshot['bytes']+256*1024**2+16*1024**2>3*1024**3:raise RuntimeError('Shared budget unavailable')
    write(ROOT/'STORAGE_PREFLIGHT.json',dict(shared=snapshot,new_bytes=storage_snapshot([ROOT])['bytes'],new_cap_bytes=16*1024**2,atomic_reserve_bytes=256*1024**2),exclusive=True)
    write(ROOT/'runtime-config.json',c,exclusive=True)
    write(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],runtime_config_sha256=sha(ROOT/'runtime-config.json'),bindings=len(bindings),physical_attempts=0,model_calls=0),exclusive=True)
    print(json.dumps(dict(run_id=c['run_id'],config_sha256=sha(ROOT/'runtime-config.json'),bindings=len(bindings),physical_attempts=0,model_calls=0)))
if __name__=='__main__':main()
