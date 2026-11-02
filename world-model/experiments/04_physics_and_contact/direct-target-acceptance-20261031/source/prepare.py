"""Freeze the human-approved acceptance table. Geometry and files only."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import uuid
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from tdyssey.direct_target_policy import build_plan
from tdyssey.direct_target_collection import LIMITS, PROTOCOL_SHA256, sha, write
from tdyssey.original_contact_pilot import storage_snapshot

def main():
    root=Path(__file__).resolve().parent
    if (root/'runtime-config.json').exists():raise FileExistsError('Frozen run already exists')
    plan=build_plan()
    if any(not a['plan_valid'] for a in plan['attempts']):raise ValueError('Unexecutable pre-execution plan')
    write(root/'plan.json',plan,exclusive=True)
    oldroots=json.loads((REPO/'artifacts/strict-original-contact-pilot-20261031/storage-roots.json').read_text())
    roots=list(oldroots['roots'])
    for p in ('pipeline-review-20261031','pipeline-config-review-20261031','direct-target-sampling-review-20261031',root.name):
        path=str(REPO/'artifacts'/p)
        if path not in roots:roots.append(path)
    write(root/'storage-roots.json',dict(roots=roots,shared_cap_bytes=LIMITS['shared_bytes'],atomic_reserve_bytes=LIMITS['atomic_reserve_bytes']),exclusive=True)
    policy=json.loads((REPO/'artifacts/strict-original-contact-pilot-20261031/policies/resource-policy.json').read_text())
    policy.update(kind='direct-target-acceptance-shared-storage.v1',collection_cap_bytes=LIMITS['new_bytes'],
        training_authorized=False,actual_concurrent_collectors=1,scope='Entire new root, data plus reports/media; no deletion',
        maximum_physical_attempts=120)
    write(root/'policies/resource-policy.json',policy,exclusive=True)
    source=REPO.parent/'Tdyssey/third_party/diffusion_policy'
    # Bind the source dependency closure without importing any simulator/model.
    queue=[REPO/p for p in ('tdyssey/direct_target_collection.py','tdyssey/direct_target_policy.py',
        'tdyssey/direct_target_report.py','scripts/direct_target_collect.py','scripts/direct_target_background.py',
        'scripts/direct_target_report.py','tdyssey/__init__.py')]
    closure=set()
    while queue:
        path=queue.pop().resolve()
        if path in closure or not path.is_file():continue
        closure.add(path)
        if path.suffix!='.py':continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    candidate=base.joinpath(*(node.module or '').split('.'))
                elif node.module and node.module.startswith('tdyssey'):
                    candidate=REPO.joinpath(*node.module.split('.'))
                else:continue
                if candidate.with_suffix('.py').is_file():queue.append(candidate.with_suffix('.py'))
                elif candidate.is_dir():
                    queue.append(candidate/'__init__.py')
                    for alias in node.names:queue.append(candidate/(alias.name+'.py'))
    closure.update(source/p for p in ('diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'))
    closure.update(REPO/p for p in ('manifests/official_dp.json',
        'artifacts/direct-target-sampling-review-20261031/PLAN.txt',
        'artifacts/direct-target-sampling-review-20261031/APPROVAL.json',
        'artifacts/direct-target-sampling-review-20261031/IMPLEMENTATION_HANDOFF.txt',
        'artifacts/strict-original-contact-pilot-20261031/validation/strict-ui-original-v1/summary.json',
        'tests/test_direct_target_collection.py','tests/test_direct_target_policy.py','tests/test_direct_target_report.py'))
    closure.update(root/p for p in ('plan.json','storage-roots.json','policies/resource-policy.json','HUMAN_AUTHORIZATION.json','prepare.py'))
    config=dict(kind='direct-target-acceptance.v1',run_id='DTA-20261031-'+uuid.uuid4().hex,
        artifact_root=str(root),repo=str(REPO),source_root=str(source),plan=str(root/'plan.json'),
        resource_policy=str(root/'policies/resource-policy.json'),storage_roots=str(root/'storage-roots.json'),
        approval_evidence=str(root/'HUMAN_AUTHORIZATION.json'),stop_file=str(root/'background/STOP.json'),
        python_executable=str(REPO.parent/'Tdyssey/.venv/Scripts/python.exe'),
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',
        owner_thread_id='__LOCAL_CHAT__',
        limits=LIMITS,physics_protocol_sha256=PROTOCOL_SHA256,
        physics_settings=dict(physics_hz=100,dt=.01,iterations=10,control_hz=10,kp=100,kv=20,damping=0,pusher='KINEMATIC',radius=15),
        target_domain=[0,512],target_clipping=False,old_model_domain=[0,511],old_model_normalization_changed=False,
        training_authorized=False,model_calls=0,optimizer_updates=0,auto_retry=False,next_phase_released=False,
        initialization='reset_scene only, no hidden physics calls; actual Space.step initialization calls counted',
        branches='Full original reconstruction plus3real control replay, actual4state/RGB+3actions equality',
        plan_sha256=sha(root/'plan.json'),bindings=[dict(path=str(p),sha256=sha(p)) for p in sorted(closure,key=str)])
    before=storage_snapshot(roots)
    reserve=LIMITS['atomic_reserve_bytes']+LIMITS['new_bytes']
    if before['missing_roots'] or before['bytes']+reserve>LIMITS['shared_bytes']:
        raise RuntimeError('Shared capacity unavailable; preserve evidence, no physical attempt')
    write(root/'storage-preflight.json',dict(actual_shared=before,new_root_bytes=storage_snapshot([root])['bytes'],
        reserved_whole_newroot_and_atomic_bytes=reserve,physics_calls=0,model_calls=0,optimizer_updates=0),exclusive=True)
    write(root/'runtime-config.json',config,exclusive=True)
    write(root/'freeze-receipt.json',dict(run_id=config['run_id'],runtime_config_sha256=sha(root/'runtime-config.json'),
        plan_sha256=config['plan_sha256'],source_bindings=len(config['bindings']),physical_attempts=0,model_calls=0,optimizer_updates=0),exclusive=True)
    print(json.dumps(dict(run_id=config['run_id'],config_sha256=sha(root/'runtime-config.json'),attempts=120,families=96,
        shared_bytes=before['bytes'],physics_calls=0,model_calls=0,optimizer_updates=0)))

if __name__=='__main__':main()
