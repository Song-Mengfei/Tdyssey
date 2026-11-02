"""Freeze this single authorized workflow. No simulator/model construction."""
from pathlib import Path
import json
import sys
import uuid

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(REPO)]
from common import atomic_json, check_storage, read, sha, utc

def main():
    target = ROOT/'runtime-config.json'
    if target.exists():
        raise FileExistsError('Do not replace a frozen run')
    from collect import build_plan
    seed = 547810315
    plan = build_plan(seed)
    atomic_json(ROOT/'collection-plan.json', plan, exclusive=True)
    old = read(REPO/'artifacts/direct-target-four-case-diagnostic-20261031/storage-roots.json')
    old_roots = old.get('roots', old.get('shared_storage_roots', []))
    roots = sorted(set(old_roots + [str(ROOT), str(REPO/'artifacts/direct-target-diagnosis-review-20261031'),
        str(REPO/'artifacts/contact-event-revision-review-20261031')]))
    checkpoint = REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
    expected = '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
    if sha(checkpoint) != expected:
        raise ValueError('Latest complete checkpoint identity mismatch')
    c = dict(kind='accelerated-rgb-one-epoch.v1', run_id='AR1-20261031-'+uuid.uuid4().hex,
        artifact_root=str(ROOT), repo=str(REPO), seed=seed, epochs=1,
        python_executable=str(REPO.parent/'Tdyssey/.venv/Scripts/python.exe'),
        source_root=str(REPO.parent/'Tdyssey/third_party/gpc_code'),
        physics_source_root=str(REPO.parent/'Tdyssey/third_party/diffusion_policy'),
        collection_plan=str(ROOT/'collection-plan.json'),
        parent_thread_id='__LOCAL_CHAT__', parent_host_id='local',
        owner_thread_id='__LOCAL_CHAT__',
        limits=dict(attempts=2000, controls_per_attempt=51, controls=102000, regular_substeps=1020000),
        initial_families=1536, reserve_families=464, reserve_policy='Do not fill unused budget; only same-strategy coverage-deficit supplement if implemented and explicitly accounted.',
        collection_workers=2, training_authorized=True, auto_retry=False, auto_next_epoch=False,
        physics_settings=dict(physics_hz=100, dt=.01, iterations=10, control_hz=10,
                              kp=100, kv=20, damping=0, pusher='KINEMATIC', radius=15),
        target_domain=[0,512], target_clipping=False, model_domain=[0,511],
        model_action_policy='Preserve original normalization; exclude incompatible windows, never clip raw requests.',
        letters=list('TLOCXB'), held_out_letters=list('DGKNQRWZ'),
        shared_storage_roots=roots, shared_cap_bytes=10_000_000_000,
        global_cap_bytes=10_000_000_000, atomic_reserve_bytes=268435456,
        starting_checkpoint=str(checkpoint), resume_checkpoint=str(checkpoint),
        starting_checkpoint_sha256=expected, device='cuda', max_cuda_bytes=6*1024**3,
        microbatch=4, effective_batch=16, lr=1e-5,
        description='New original-physics data phase, full unique pool once; not exact same-distribution continuation.',
        callback_authorized=True, callback_requires_running_app=True, server_authorized=False,
        frozen_at_utc=utc())
    paths = {p.resolve() for p in ROOT.glob('*.py')}
    paths |= {p.resolve() for p in (REPO/'tdyssey').glob('*.py')}
    paths |= {p.resolve() for p in (REPO.parent/'Tdyssey/third_party/gpc_code').rglob('*.py')}
    paths |= {Path(b['path']).resolve() for b in read(REPO/'artifacts/direct-target-acceptance-20261031/runtime-config.json')['bindings']}
    paths |= {ROOT/'HUMAN_AUTHORIZATION.json', ROOT/'PARENT_HANDOFF.txt', ROOT/'OWNER_MEMORY.txt',
              ROOT/'collection-plan.json', checkpoint,
              REPO/'artifacts/contact-event-revision-review-20261031/PROPOSAL.txt',
              REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/background/runtime-config.json'}
    c['bindings'] = [dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
    atomic_json(target,c,exclusive=True)
    atomic_json(ROOT/'storage-preflight.json',check_storage(ROOT,c,full=True))
    digest=sha(target)
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,
        bindings=len(c['bindings']),physical_calls=0,model_calls=0,at_utc=utc()))
    atomic_json(ROOT/'PHASE_STATE.json',dict(run_id=c['run_id'],config_sha256=digest,phase='ready_to_launch',
        callback_authorized=True,parent_thread_id=c['parent_thread_id'],auto_next_epoch=False,
        memory=str(ROOT/'OWNER_MEMORY.txt')))
    print(json.dumps(dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']))))

if __name__=='__main__':
    main()
