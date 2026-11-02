"""Freeze the exact authorized A/B and full source/cache identities; CPU only."""
from pathlib import Path
import sys
import uuid
import time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OC=ROOT.parent/'oc-continuation-20261031'
AR1=ROOT.parent/'accelerated-rgb-run-20261031'
sys.path[:0]=[str(ROOT),str(REPO)]
from common import atomic_json,read,sha,check_storage,utc
PLAN_SHA='1da99ad12b29f5687cc6d6c39d28858c8818dc2bb679b49da7f067e6ab3361f8'
PARENT_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'

def preflight(path):
    import torch
    torch.set_num_threads(2);began=time.perf_counter()
    if sha(path)!=PARENT_SHA:raise ValueError('Exact common epoch2 checkpoint differs')
    p=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    required={'model','optimizer','scaler','global_rng','conditioning_rngs','order','progress','identity','legacy_scheduler_provenance'}
    if p['kind']!='oc_native_rgb_complete.v1' or not required.issubset(p):raise ValueError('Full epoch2 parent required')
    q=p['progress']
    if q['epoch']!=2 or q['lineage_updates']!=11367 or q['unique_samples_exposed']!=30935 or q['batches_consumed']!=1934 or not q['epoch_complete'] or not q['complete_optimizer_boundary'] or q['pending_attempt'] is not None:raise ValueError('Epoch2 full optimizer boundary differs')
    if any(abs(g['lr']-1e-5)>1e-14 for g in p['optimizer']['param_groups']):raise ValueError('Held native LR changed')
    if any(not bool(torch.isfinite(t).all()) for t in p['model'].values() if torch.is_floating_point(t)):raise ValueError('Nonfinite common model')
    result=dict(passed=True,path=str(path),sha256=PARENT_SHA,bytes=path.stat().st_size,kind=p['kind'],progress=q,
        model_tensors=len(p['model']),optimizer_state_entries=len(p['optimizer']['state']),scaler=p['scaler'],
        global_rng_keys=sorted(p['global_rng']),conditioning_policy_sha256=p['conditioning_rngs']['policy_sha256'],
        cpu_only=True,models_constructed=0,physics_calls=0,wall_seconds=time.perf_counter()-began,at_utc=utc())
    atomic_json(ROOT/'RESTORE_PREFLIGHT.json',result);return result

def main():
    target=ROOT/'runtime-config.json'
    if target.exists():raise FileExistsError('Frozen run cannot be replaced')
    plan=ROOT.parent/'generated-history-four-step-review-20261101/PLAN.txt'
    if sha(plan)!=PLAN_SHA:raise ValueError('Approved plan changed')
    evidence=read(ROOT/'READ_THREAD_EVIDENCE.json')
    if evidence['thread_id']!='__LOCAL_CHAT__' or evidence['user_message']['type']!='userMessage':raise ValueError('Direct human evidence missing')
    parent=OC/'training/epoch2.pt';preflight(parent)
    if sha(OC/'runtime-config.json')!='613529d5e71bc6bdf76d15d36d0c3e555a400dafe052ed2ab2e842829ba7ea48':raise ValueError('Historical OC frozen config identity changed')
    uid=uuid.uuid4().hex;old=read(OC/'runtime-config.json')
    for binding in old['bindings']:
        if sha(binding['path'])!=binding['sha256']:raise ValueError('Original baseline binding changed: '+binding['path'])
    c=dict(kind='generated-history-four-step-ab.v1',run_id='GHAB4-20261101-'+uid,
        artifact_root=str(ROOT),repo=str(REPO),dataset_root=str(AR1),data_root=str(AR1),oc_root=str(OC),
        source_root=old['source_root'],python_executable=old['python_executable'],
        approved_plan=str(plan),approved_plan_sha256=PLAN_SHA,starting_checkpoint=str(parent),starting_checkpoint_sha256=PARENT_SHA,expected_parent_lineage=11367,
        arms=['A','B'],planned_segments_per_arm=7568,planned_targets_per_arm=30272,planned_batches_per_arm=1892,measurement_batches=32,
        pool_index=str(ROOT/'pool-index.json'),order_path=str(ROOT/'order.json'),
        limits=dict(new_physical_attempts=0,integrator_calls=0,segments_per_arm=7568,targets_per_arm=30272,optimizer_batches_per_arm=1892,total_exposures=60544,total_optimizer_batches=3784),
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        worker_lifecycle='windows-task-scheduler',windows_task_name='Tdyssey-GHAB4-'+uid[:8],
        shared_storage_roots=sorted(set(old['shared_storage_roots']+[str(ROOT),str(plan.parent)])),
        shared_cap_bytes=10_000_000_000,atomic_reserve_bytes=268435456,max_cuda_bytes=6*1024**3,
        device='cuda',microbatch=4,effective_batch=16,learning_rate=1e-5,formal_sampler_steps=3,generation_seed=2026110104,
        generated_prefix_steps=3,supervised_steps=4,gradient_clip=1.,
        new_physics_authorized=False,training_authorized=True,automatic_retry=False,auto_next_epoch=False,server_authorized=False,
        callback_authorized=True,callback_requires_running_app=True,frozen_at_utc=utc())
    import pool
    if not (ROOT/'pool-index.json').exists():pool.build(ROOT,c)
    index=read(ROOT/'pool-index.json');order=read(ROOT/'order.json')
    if len(index['segments'])!=7568 or len(order['batches'])!=1892:raise ValueError('Approved pool/order differ')
    # Evaluation.freeze only copies saved identities/specs. No inference here.
    import evaluation
    eval_freeze=evaluation.freeze(ROOT,c) if hasattr(evaluation,'freeze') else {}
    paths={Path(x['path']).resolve() for x in old['bindings']}
    paths|={p.resolve() for p in ROOT.glob('*.py')}
    paths.add(ROOT/'launch.ps1')
    if isinstance(eval_freeze,dict):
        for value in eval_freeze.get('bindings',[]):
            paths.add(Path(value['path'] if isinstance(value,dict) else value).resolve())
    paths|={ROOT/n for n in ('USER_AUTHORIZATION.json','READ_THREAD_EVIDENCE.json','PARENT_HANDOFF.txt','OWNER_MEMORY.txt','HEARTBEAT_POLICY.txt','RESTORE_PREFLIGHT.json','pool-index.json','order.json','baseline-reuse.json','VERIFICATION.json','SOURCE_CHECK.json','offline-final-checks.log')}
    paths|={Path(r['path']) for r in index['sources']+index['cache']}
    paths|={plan,pool.PREVIEW,parent,OC/'runtime-config.json',OC/'freeze-receipt.json',OC/'diagnostic-spec.json',
        AR1/'runtime-config.json',AR1/'selection/train-index.json',AR1/'selection/seen-val-index.json',AR1/'selection/summary.json',AR1/'evaluation-spec.json'}
    for p in (OC/'evaluation/epoch2/metrics.json',):
        paths.add(p)
        for r in read(p)['raw_files']:
            if sha(r['path'])!=r['sha256']:raise ValueError('Baseline saved prediction changed')
            paths.add(Path(r['path']))
    paths|={p.resolve() for p in ROOT.glob('*spec.json')}
    c['bindings']=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
    atomic_json(target,c,exclusive=True);digest=sha(target)
    storage=check_storage(ROOT,c,full=True);atomic_json(ROOT/'storage-preflight.json',storage)
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),
        pool=index['summary'],order_sha256=sha(ROOT/'order.json'),model_calls=0,optimizer_updates=0,physics_calls=0,at_utc=utc()),exclusive=True)
    atomic_json(ROOT/'PHASE_STATE.json',dict(run_id=c['run_id'],config_sha256=digest,phase='ready_to_launch',status='ready_to_launch',
        owner_memory=str(ROOT/'OWNER_MEMORY.txt'),parent_thread_id=c['parent_thread_id'],callback_authorized=True,auto_next_epoch=False))
    print(dict(run_id=c['run_id'],config_sha256=digest,windows_task_name=c['windows_task_name'],bindings=len(c['bindings']),shared_bytes=storage['shared_bytes_estimate']))

if __name__=='__main__':main()
