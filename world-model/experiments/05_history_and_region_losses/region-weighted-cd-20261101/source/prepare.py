"""Freeze bounded O/B attribution followed by one approved region-weighted C/D."""
from pathlib import Path
import sys
import time
import random
import uuid
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
AB=ROOT.parent/'generated-history-ab-20261101'
OC=ROOT.parent/'oc-continuation-20261031'
AR1=ROOT.parent/'accelerated-rgb-run-20261031'
sys.path[:0]=[str(ROOT),str(REPO)]
from common import atomic_json,read,sha,check_storage,utc
PLAN_SHA='27d9bc5a4ed62255dba4cd9ae6914d17945a6759f25e56f6c57639cb2c3d7efb'
PARENT_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'

def preflight():
    import torch
    torch.set_num_threads(2);began=time.perf_counter();path=OC/'training/epoch2.pt'
    if sha(path)!=PARENT_SHA:raise ValueError('Exact common epoch2 differs')
    p=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    required={'model','optimizer','scaler','global_rng','conditioning_rngs','order','progress','identity','legacy_scheduler_provenance'}
    if p['kind']!='oc_native_rgb_complete.v1' or not required.issubset(p):raise ValueError('Full epoch2 parent required')
    q=p['progress']
    if q['epoch']!=2 or q['lineage_updates']!=11367 or q['unique_samples_exposed']!=30935 or q['batches_consumed']!=1934 or not q['epoch_complete'] or not q['complete_optimizer_boundary'] or q['pending_attempt'] is not None:raise ValueError('Wrong complete epoch2 boundary')
    if any(abs(g['lr']-1e-5)>1e-14 for g in p['optimizer']['param_groups']):raise ValueError('Parent held LR changed')
    if any(not bool(torch.isfinite(t).all()) for t in p['model'].values() if torch.is_floating_point(t)):raise ValueError('Nonfinite common model')
    result=dict(passed=True,path=str(path),sha256=PARENT_SHA,bytes=path.stat().st_size,kind=p['kind'],progress=q,
        model_tensors=len(p['model']),optimizer_state_entries=len(p['optimizer']['state']),scaler=p['scaler'],
        global_rng_keys=sorted(p['global_rng']),conditioning_policy_sha256=p['conditioning_rngs']['policy_sha256'],
        cpu_only=True,models_constructed=0,physics_calls=0,wall_seconds=time.perf_counter()-began,at_utc=utc())
    atomic_json(ROOT/'RESTORE_PREFLIGHT.json',result)

def assignment():
    order=read(ROOT/'order.json');seed=2026110152;rng=random.Random(seed)
    chosen=[sorted(rng.sample(range(4),2)) for _ in order['batches']]
    value=dict(seed=seed,generated_indices=chosen,order_sha256=sha(ROOT/'order.json'),
        unit='Exactly two distinct local segment slots per batch4; independent pre-run Python RNG, never prediction-dependent',
        batches=1892,generated_segments=3784,real_segments=3784,fully_true_target_frames=18920,generated_containing_target_frames=11352,
        full_true_fraction=.625,not_clean_weak_policy=True)
    p=ROOT/'assignment.json'
    if p.exists():
        if read(p)!=value:raise ValueError('Frozen D assignment differs')
    else:atomic_json(p,value,exclusive=True)
    return value

def main():
    if (ROOT/'runtime-config.json').exists():raise FileExistsError('No replacing frozen identity')
    plan=ROOT.parent/'generated-history-next-step-review-20261101/ANALYSIS_AND_PLAN.txt'
    if sha(plan)!=PLAN_SHA:raise ValueError('Approved C/D plan changed')
    ev=read(ROOT/'READ_THREAD_EVIDENCE.json')
    if ev['thread_id']!='__LOCAL_CHAT__' or ev['user_message']['type']!='userMessage':raise ValueError('Direct human evidence missing')
    if sha(AB/'runtime-config.json')!='56927d96995d2582263ba730ab227ab923554610128d7ce32a75f8dbf9ededef':raise ValueError('Old A/B identity changed')
    old=read(AB/'runtime-config.json')
    for b in old['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Frozen old source/data changed: '+b['path'])
    for name in ('pool-index.json','order.json'):
        if sha(ROOT/name)!=sha(AB/name):raise ValueError('Exact pool/order must be reused')
    preflight();allocation=assignment();uid=uuid.uuid4().hex
    c=dict(kind='region-weighted-four-step-cd.v1',run_id='RWCD-20261101-'+uid,artifact_root=str(ROOT),repo=str(REPO),
        dataset_root=str(AR1),data_root=str(AR1),oc_root=str(OC),ab_root=str(AB),source_root=old['source_root'],python_executable=old['python_executable'],
        approved_plan=str(plan),approved_plan_sha256=PLAN_SHA,starting_checkpoint=str(OC/'training/epoch2.pt'),starting_checkpoint_sha256=PARENT_SHA,
        expected_parent_lineage=11367,arms=['C','D'],planned_segments_per_arm=7568,planned_targets_per_arm=30272,planned_batches_per_arm=1892,measurement_batches=32,
        pool_index=str(ROOT/'pool-index.json'),order_path=str(ROOT/'order.json'),existing_cache_root=str(AB/'cache'),assignment_path=str(ROOT/'assignment.json'),
        limits=dict(new_physical_attempts=0,integrator_calls=0,diagnostic_predicted_frames=768,segments_per_arm=7568,targets_per_arm=30272,
            optimizer_batches_per_arm=1892,total_exposures=60544,total_optimizer_batches=3784),
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        worker_lifecycle='windows-task-scheduler',windows_task_name='Tdyssey-RWCD-'+uid[:8],
        shared_storage_roots=sorted(set(old['shared_storage_roots']+[str(ROOT),str(plan.parent)])),
        shared_cap_bytes=10_000_000_000,atomic_reserve_bytes=268435456,max_cuda_bytes=6*1024**3,device='cuda',microbatch=4,effective_batch=16,
        learning_rate=1e-5,formal_sampler_steps=3,generation_seed=2026110104,loss_coefficients=[.5,.25,.25],gradient_clip=1.,
        generated_prefix_steps=3,supervised_steps=4,D_generated_slots_per_batch=2,D_fully_true_target_fraction=.625,
        diagnostic_models={'baseline':dict(path=str(OC/'training/epoch2.pt'),sha256=PARENT_SHA),
            'A':dict(path=str(AB/'training/A/final.pt'),sha256='b4d7dd0fb58f535e7129c65bda025fdcfe2d09ce2c00835fab955d3c6cc5a2a5'),
            'B':dict(path=str(AB/'training/B/final.pt'),sha256='3b10eaf84f3bd13104a4667338621dee855efc85b0e3ce2cbbe06e93cca9efba')},
        diagnosis_before_training=True,interface_fault_stops_training=True,new_physics_authorized=False,training_authorized=True,
        automatic_retry=False,auto_next_epoch=False,server_authorized=False,callback_authorized=True,callback_requires_running_app=True,frozen_at_utc=utc())
    import masks,evaluation,diagnosis
    atomic_json(ROOT/'loss-spec.json',masks.specification())
    eval_spec=evaluation.freeze(ROOT,c);diag_spec=diagnosis.freeze(ROOT,c)
    paths={Path(x['path']).resolve() for x in old['bindings']}
    paths|={p.resolve() for p in ROOT.glob('*.py')}
    paths|={ROOT/n for n in ('launch.ps1','USER_AUTHORIZATION.json','READ_THREAD_EVIDENCE.json','PARENT_HANDOFF.txt','OWNER_MEMORY.txt','HEARTBEAT_POLICY.txt',
        'RESTORE_PREFLIGHT.json','MASK_PREFLIGHT.json','pool-index.json','order.json','assignment.json','loss-spec.json','VERIFICATION.json','PRELAUNCH_CORRECTION.json','offline-final-checks.log','diagnosis-tests-after-isolation-fix.log')}
    paths|={plan,plan.parent/'SAVED_ANALYSIS.json',plan.parent/'O_DIRECTION_ANALYSIS.json',masks.CALIBRATION,AB/'runtime-config.json',AB/'freeze-receipt.json'}
    for value in (eval_spec,diag_spec):
        if isinstance(value,dict):
            for b in value.get('bindings',[]):paths.add(Path(b['path'] if isinstance(b,dict) else b).resolve())
    paths|={Path(v['path']) for v in c['diagnostic_models'].values()}
    paths|={p.resolve() for p in ROOT.glob('*spec.json')}
    c['bindings']=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths) if p.is_file()]
    atomic_json(ROOT/'runtime-config.json',c,exclusive=True);digest=sha(ROOT/'runtime-config.json')
    storage=check_storage(ROOT,c,reserve_bytes=1_000_000_000,full=True);atomic_json(ROOT/'storage-preflight.json',storage)
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),
        pool=read(ROOT/'pool-index.json')['summary'],assignment={k:v for k,v in allocation.items() if k!='generated_indices'},
        model_calls=0,optimizer_updates=0,physics_calls=0,at_utc=utc()),exclusive=True)
    atomic_json(ROOT/'PHASE_STATE.json',dict(run_id=c['run_id'],config_sha256=digest,status='ready_to_launch',phase='ready_to_launch',
        owner_memory=str(ROOT/'OWNER_MEMORY.txt'),parent_thread_id=c['parent_thread_id'],auto_next_epoch=False,callback_authorized=True))
    print(dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),task_name=c['windows_task_name'],shared_bytes=storage['shared_bytes_estimate']))

if __name__=='__main__':main()
