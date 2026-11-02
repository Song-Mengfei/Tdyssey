"""Freeze approved saved-data continuation and biased diagnostic inputs; CPU only."""
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
AR1=ROOT.parent/'accelerated-rgb-run-20261031'
sys.path[:0]=[str(ROOT),str(REPO)]
from common import atomic_json,read,sha,check_storage,utc

def main():
    target=ROOT/'runtime-config.json'
    if target.exists():raise FileExistsError('Do not replace a frozen run identity')
    plan=REPO/'artifacts/oc-training-review-20261031/ANALYSIS_AND_NEXT_STEP.txt'
    if sha(plan)!='e2a84aca0c38a24f5bfd4a72a0c0b68b906e429f32b2be501b9dac4fe023c3f0':raise ValueError('Approved plan changed')
    parent=AR1/'training/last.pt'
    expected='492e9badaa3e622254c829e1ea44cad6ad8c4c4636feb15c7764ec07e3745f11'
    preflight=read(ROOT/'RESTORE_PREFLIGHT.json')
    if sha(parent)!=expected or not preflight.get('passed',preflight.get('full_state_readable',False)):
        raise ValueError('Parent checkpoint or actual CPU restore preflight failed')
    uid=uuid.uuid4().hex
    old=read(AR1/'runtime-config.json')
    c=dict(kind='oc-two-epoch-continuation.v1',run_id='OC23-20261031-'+uid,
        artifact_root=str(ROOT),repo=str(REPO),dataset_root=str(AR1),data_root=str(AR1),
        source_root=old['source_root'],python_executable=old['python_executable'],
        starting_checkpoint=str(parent),starting_checkpoint_sha256=expected,expected_parent_lineage=9433,
        epochs=[2,3],planned_samples_per_epoch=30935,epoch_order_seeds={'2':20261033,'3':20261034},
        limits=dict(new_physical_attempts=0,integrator_calls=0,additional_epochs=2,samples_per_epoch=30935,additional_draws=61870,optimizer_batches=3868),
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        worker_lifecycle='windows-task-scheduler',windows_task_name='Tdyssey-OC23-'+uid[:8],
        shared_storage_roots=sorted(set(old['shared_storage_roots']+[str(ROOT),str(REPO/'artifacts/oc-training-review-20261031')])),
        shared_cap_bytes=10_000_000_000,atomic_reserve_bytes=268435456,max_cuda_bytes=6*1024**3,
        device='cuda',microbatch=4,effective_batch=16,learning_rate=1e-5,formal_sampler_steps=3,
        diagnostic_max_families_per_split_letter=8,diagnostic_letters=['O','C'],
        epoch1_denoise16_case_limit=8,new_physics_authorized=False,training_authorized=True,
        automatic_retry=False,auto_next_epoch=False,server_authorized=False,
        callback_authorized=True,callback_requires_running_app=True,frozen_at_utc=utc())
    import diagnostics
    from engine import load_ar1,freeze_orders
    diagnostic=diagnostics.freeze(ROOT,c)
    train=read(AR1/'selection/train-index.json')
    _,orders=freeze_orders(ROOT,c,train,load_ar1())
    paths={Path(x['path']).resolve() for x in old['bindings']}
    paths|={p.resolve() for p in ROOT.glob('*.py')}
    paths|={ROOT/n for n in ('HUMAN_AUTHORIZATION.json','READ_THREAD_EVIDENCE.json','PARENT_HANDOFF.txt','OWNER_MEMORY.txt',
        'RESTORE_PREFLIGHT.json','diagnostic-spec.json','loss-spec.json','epoch-orders.json')}
    paths|={plan,parent,AR1/'runtime-config.json',AR1/'freeze-receipt.json',AR1/'selection/train-index.json',
        AR1/'selection/seen-val-index.json',AR1/'selection/summary.json',AR1/'evaluation-spec.json',
        AR1/'evaluation/after/metrics.json',AR1/'evaluation/before/metrics.json',AR1/'training/result.json'}
    spec=read(ROOT/'diagnostic-spec.json')
    for row in spec['rows']:
        paths|={Path(row['raw_path']),Path(row['metadata_path'])}
    for record in read(AR1/'evaluation/after/metrics.json')['raw_files']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Completed epoch1 raw prediction changed')
        paths.add(Path(record['path']))
    c['bindings']=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
    atomic_json(target,c,exclusive=True)
    storage=check_storage(ROOT,c,full=True);atomic_json(ROOT/'storage-preflight.json',storage)
    digest=sha(target)
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),
        diagnostic=diagnostic,orders={e:{k:v for k,v in row.items() if k!='batches'} for e,row in orders.items()},
        model_calls=0,optimizer_updates=0,physics_calls=0,at_utc=utc()))
    atomic_json(ROOT/'PHASE_STATE.json',dict(run_id=c['run_id'],config_sha256=digest,phase='ready_to_launch',
        status='ready_to_launch',authorized_epochs=[2,3],new_physics=0,owner_memory=str(ROOT/'OWNER_MEMORY.txt'),
        parent_thread_id=c['parent_thread_id'],callback_authorized=True,auto_next_epoch=False))
    print(dict(run_id=c['run_id'],config_sha256=digest,windows_task_name=c['windows_task_name'],
        bindings=len(c['bindings']),diagnostic=diagnostic,shared_bytes=storage['shared_bytes_estimate']))

if __name__=='__main__':main()
