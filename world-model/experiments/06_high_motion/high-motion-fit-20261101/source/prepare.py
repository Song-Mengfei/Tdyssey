"""Freeze one approved finite high-motion fit. Stdlib preparation; no model load."""
from pathlib import Path
import uuid
import sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(REPO),str(REPO/'scripts')]
from common import read,sha,atomic_json,check_storage,utc
PLAN_SHA='138d641c88cafe9a82e3c805aa2bf539219684135615e551fa8d862562b9e0d0'
PARENT_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'

def main():
    if (ROOT/'runtime-config.json').exists():raise FileExistsError('Frozen identity already exists')
    if sha(ROOT/'PLAN.txt')!=PLAN_SHA:raise ValueError('Approved PLAN differs')
    human=read(ROOT/'READ_THREAD_EVIDENCE.json')
    if human['threadId']!='__LOCAL_CHAT__' or human['turnId']!='__LOCAL_CHAT__' or human['userMessage']['id']!='__LOCAL_CHAT__' or human['userMessage']['type']!='userMessage':raise ValueError('Direct actual human evidence differs')
    oldroot=REPO/'artifacts/region-weighted-cd-20261101'
    if sha(oldroot/'runtime-config.json')!='6ae814d58c7af48b6c2cc6e63181a6f254281453b2f1d7a1c63f387725142c46':raise ValueError('Original recipe identity differs')
    old=read(oldroot/'runtime-config.json')
    for b in old['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Reused original source/data binding differs: '+b['path'])
    index=read(ROOT/'selection/index.json')
    uid=uuid.uuid4().hex
    c=dict(kind='high-motion-training-fit.v1',run_id='HMF-20261101-'+uid,artifact_root=str(ROOT),repo=str(REPO),
        dataset_root=old['dataset_root'],data_root=old['data_root'],source_root=old['source_root'],python_executable=old['python_executable'],
        approved_plan=str(ROOT/'PLAN.txt'),approved_plan_sha256=PLAN_SHA,starting_checkpoint=old['starting_checkpoint'],
        starting_checkpoint_sha256=PARENT_SHA,expected_parent_lineage=11367,arms=['fit'],selection_index=str(ROOT/'selection/index.json'),
        order_path=str(ROOT/'selection/order.json'),selection_seed=2026110107,
        max_successful_updates=3000,max_supervised_exposures=48000,max_attempts=3000,measurement_batches=32,
        milestones=[0,250,500,1000,2000,3000],checkpoints=[0,250,500,1000,2000,3000],
        limits=dict(new_physical_attempts=0,integrator_calls=0,train_clips=48,seen_val_clips=24,reference_clips=3,max_successful_updates=3000,max_exposures=48000),
        shared_storage_roots=sorted(set(old['shared_storage_roots']+[str(ROOT),str(ROOT.parent/'high-motion-fit-proposal-20261101')])),
        shared_cap_bytes=10000000000,atomic_reserve_bytes=268435456,max_cuda_bytes=6*1024**3,device='cuda',
        microbatch=4,effective_batch=16,learning_rate=1e-5,gradient_clip=1.,formal_sampler_steps=3,
        full_training_state_retention=['latest','best-fit','final'],training_authorized=True,new_physics_authorized=False,
        history_policy='original50percentclean50percentweakrealhistory',loss='originalnativeequalpixelresidualMSE',
        generated_history=False,region_weighting=False,automatic_retry=False,auto_next_epoch=False,server_authorized=False,
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        worker_lifecycle='windows-task-scheduler',windows_task_name='Tdyssey-HMF-'+uid[:8],
        callback_authorized=True,callback_requires_running_app=True,frozen_at_utc=utc())
    import evaluation
    evaluation_spec=evaluation.freeze(ROOT,c)
    paths={Path(b['path']).resolve() for b in old['bindings']}
    paths|={p.resolve() for p in ROOT.glob('*.py')}
    paths|={ROOT/n for n in ['PLAN.txt','USER_AUTHORIZATION.json','READ_THREAD_EVIDENCE.json','OWNER_MEMORY.txt','HEARTBEAT_POLICY.txt','launch.ps1','VERIFICATION.json']}
    paths|={p.resolve() for p in (ROOT/'selection').rglob('*') if p.is_file()}
    paths|={p.resolve() for p in ROOT.glob('*spec.json')}
    paths|={p.resolve() for p in ROOT.glob('*PREFLIGHT.json')}
    paths|={p.resolve() for p in ROOT.glob('*-standard.json')}
    paths|={p.resolve() for p in ROOT.glob('*-calibration.json')}
    paths|={p.resolve() for p in ROOT.glob('prelaunch-*.json')}
    paths|={oldroot/'runtime-config.json',oldroot/'freeze-receipt.json',Path(c['starting_checkpoint'])}
    if isinstance(evaluation_spec,dict):
        for b in evaluation_spec.get('bindings',[]):paths.add(Path(b['path'] if isinstance(b,dict) else b).resolve())
    missing=[str(p)for p in paths if not p.is_file()]
    if missing:raise ValueError('Required immutable input missing: '+repr(missing))
    c['bindings']=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(paths)]
    storage=check_storage(ROOT,c,reserve_bytes=1800000000,full=True)
    atomic_json(ROOT/'storage-preflight.json',storage)
    atomic_json(ROOT/'runtime-config.json',c,exclusive=True)
    digest=sha(ROOT/'runtime-config.json')
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),
        selection_index_sha256=sha(ROOT/'selection/index.json'),order_sha256=sha(ROOT/'selection/order.json'),
        selection_frozen_before_model=True,model_calls=0,optimizer_updates=0,physics_calls=0,at_utc=utc()),exclusive=True)
    atomic_json(ROOT/'PHASE_STATE.json',dict(run_id=c['run_id'],config_sha256=digest,status='ready_to_launch',phase='ready_to_launch',
        callback_authorized=True,owner_memory=str(ROOT/'OWNER_MEMORY.txt'),parent_thread_id=c['parent_thread_id'],auto_next_epoch=False))
    print(dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),task_name=c['windows_task_name'],shared_bytes=storage['shared_bytes_estimate']))

if __name__=='__main__':main()

