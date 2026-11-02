"""Single low-priority serial preparation. No Denoiser/model call."""
from pathlib import Path
import sys,os,time,uuid,json,ast,tempfile
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(REPO),str(REPO/'scripts')]
from resource_limits import constrain_cpu
CPU=constrain_cpu()
from common import read,sha,atomic_json,snapshot,initialize_storage
import numpy as np
SOURCE_REPORT_SHA='c47ac9014d6061c88bba0288b01ff5cd49b6182f6977faf39074c38572404772'
SOURCE_HMF_SHA='872013d0b17e7d3e52933312a1b74993beb0ed6fa2fd04ee4054e0c1f7220cc1'

def stat_receipt(bindings):
    out={}
    for b in bindings:
        p=Path(b['path']).resolve();s=p.stat()
        out[str(p)]=dict(sha256=b['sha256'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    return out

def checks(c,receipt):
    from data import DiagnosticData
    import engine
    ds=DiagnosticData(ROOT,dict(c,_validated_source_stats=stat_receipt(receipt['bindings'])))
    A=ds.A_spec;B=ds.B_spec
    compared=0
    # Exact saved input proof parity, no model sampler.
    for batch in A['batches']:
        for mode in A['input_modes']:
            inputs=ds.A_inputs(batch,mode);history=inputs['initial_histories'];actions=inputs['absolute_actions']
            expected=next(v for v in A['expected_input_evidence']if v['mode']==mode and v['batch_id']==batch['batch_id'])
            from importlib import util
            if compared==0:
                p=ROOT.parent/'generated-history-ab-20261101/evaluation.py';spec=util.spec_from_file_location('_diag_inputs_reference',p);mod=util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
            proof=mod._inputs_evidence(batch['rows'],inputs,batch['seeds'])
            if any(proof[k]!=expected[k] for k in proof):raise ValueError('A original input proof differs')
            compared+=1
    for batch in B['batches']:
        originals=ds.B_inputs(batch,'recorded')
        for action in B['actions']:
            v=ds.B_inputs(batch,action)
            if not np.array_equal(v['initial_histories'],originals['initial_histories'])or not np.array_equal(v['absolute_actions'][:,:3],originals['absolute_actions'][:,:3]):raise ValueError('B changed prior history/actions')
            for i,row in enumerate(batch['rows']):
                if not np.array_equal(v['absolute_actions'][i,3],np.asarray(row['actions'][action])):raise ValueError('B target was altered/clipped')
    # Boundary unit check uses mock list arguments, never a torch/model forward.
    checks_dir=ROOT/'checks';checks_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checks_dir)as temporary:
        p=Path(temporary).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Fixture cleanup outside newroot')
        (p/'inference').mkdir();budget=engine.Budget(p,dict(run_id='CPU_mock_only',_runtime_config_sha256='not_a_run'))
        budget.counts.update(reserved_predictions=3668,reserved_image_forwards=11004)
        budget.sections['A'].update(reserved_predictions=3456);budget.sections['B'].update(reserved_predictions=212)
        row=dict(section='B',label='mock',batch_id=0,mode='real_history')
        budget.begin(row,0,1,4)
        for i in range(3):budget.before_forward(None,([None]*4,));budget.after_forward(None,([None]*4,),None)
        budget.complete();before=budget.state()
        rejected=False
        try:budget.begin(row,0,1,4)
        except RuntimeError:rejected=True
        if not rejected or budget.counts['reserved_predictions']!=3672:raise AssertionError('Hardcap failed')
        budget.close()
    # Six old saved images: qualified/unknown examples by letter, not new predictions.
    baseline=read(ROOT.parent/'high-motion-fit-20261101/evaluation/0/primary/metrics.json')
    samples=[];refs=[]
    for letter in ('O','C','B'):
        for qualified in (True,False):
            choices=[x for x in baseline['rows']if x['letter']==letter and bool(x['metrics'].get('legacy_shape_assessable'))==qualified]
            if not choices:continue
            row=choices[0]
            ref=next(x for x in baseline['raw_files']if x['sample_id']==row['sample_id']and x['input_mode']==row['input_mode'])
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Small saved parity raw changed')
            with np.load(ref['path'],allow_pickle=False)as z:prediction=z['raw'][row['horizon']-1].copy()
            with np.load(row.get('raw_path',ref['raw_path']),allow_pickle=False)as z:
                a=int(ref['anchor']);truth=z['rgb'][a+row['horizon']].copy();last=z['rgb'][a].copy()
            samples.append(dict(sample_id=row['sample_id']+':'+row['input_mode']+':'+str(row['horizon']),letter=letter,prediction=prediction,truth=truth,last=last))
            refs.append(dict(path=ref['path'],sha256=ref['sha256'],bytes=Path(ref['path']).stat().st_size))
    if not any(x['metrics'].get('legacy_shape_assessable') for x in baseline['rows']):raise RuntimeError('No existing qualified reference available')
    from resource_limits import torch_limits
    tr=torch_limits()
    import torch,metrics
    if not torch.cuda.is_available():raise RuntimeError('CUDA statistics qualification unavailable; no CPU model fallback')
    began=time.perf_counter();parity=metrics.check_saved_samples(samples,torch=torch,device=torch.device('cuda'))
    if not parity['passed']:raise ValueError('Original scoring qualification/parity differs: '+json.dumps(parity))
    ds.cache.clear()
    record=dict(passed=True,CPU=CPU,torch_limits=tr,A_original_input_groups_compared=compared,B_history_past3_and_raw_actions_exact=True,
        budget_boundary_mock_only=True,model_calls=0,model_loads=0,denoiser_forwards=0,new_predictions=0,physical_steps=0,
        existing_saved_GPU_statistics_validation_images=len(samples),statistics_validation_wall_seconds=time.perf_counter()-began,scoring_parity=parity,
        GPU_name=torch.cuda.get_device_name(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),reference_files=refs,
        scope='Only necessary small saved numeric qualification; not model warmup or a throughput benchmark.')
    atomic_json(ROOT/'VERIFICATION.json',record);return record,refs

def main():
    if (ROOT/'runtime-config.json').exists():raise FileExistsError('Do not replace frozen runtime')
    auth=read(ROOT/'USER_AUTHORIZATION.json')
    if auth['threadId']!='__LOCAL_CHAT__'or auth['userMessage']['id']!='__LOCAL_CHAT__':raise ValueError('Direct human authority differs')
    if sha(ROOT/'SOURCE_ANALYSIS_REPORT.txt')!=SOURCE_REPORT_SHA:raise ValueError('Approved section09 source differs')
    hmf=ROOT.parent/'high-motion-fit-20261101'
    if sha(hmf/'runtime-config.json')!=SOURCE_HMF_SHA:raise ValueError('Source HMF identity differs')
    old=read(hmf/'runtime-config.json')
    if not read(hmf/'CLOSEOUT.json')['actual_final_callback_sent']:raise ValueError('Old HMF must stay closed')
    uid=uuid.uuid4().hex
    c=dict(kind='hmf-inference-diagnostics.v1',run_id='HID-20261102-'+uid,artifact_root=str(ROOT),repo=str(REPO),hmf_root=str(hmf),data_root=old['data_root'],
        source_root=old['source_root'],python_executable=old['python_executable'],source_HMF_config_sha256=SOURCE_HMF_SHA,approved_source_report_sha256=SOURCE_REPORT_SHA,
        limits=dict(max_predictions=3672,max_image_denoiser_forwards=11016,A_predictions=3456,B_predictions=216,optimizer_updates=0,backward_calls=0,new_physical_steps=0),
        max_predictions=3672,max_image_denoiser_forwards=11016,formal_sampler_steps=3,precision='CUDA_FP32',device='cuda',
        B_selection_seed=2026110200,B_noise_seeds=[2026110201,2026110202,2026110203],diagnostic_cache_bytes=32*1024**2,
        shared_cap_bytes=10000000000,atomic_reserve_bytes=268435456,new_root_cap_bytes=500000000,max_cuda_bytes=6*1024**3,
        shared_storage_roots=sorted(set(old['shared_storage_roots']+[str(ROOT),str(ROOT.parent/'retrospective-analysis-20261101')])),
        parent_thread_id='__LOCAL_CHAT__',parent_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        worker_lifecycle='windows-task-scheduler-direct-single-process',windows_task_name='Tdyssey-HID-'+uid[:8],
        training_authorized=False,optimizer_authorized=False,new_physics_authorized=False,inference_authorized=True,automatic_retry=False,automatic_next_work=False,
        callback_authorized=True,callback_requires_running_app=True,CPU_policy='Only own BelowNormal single numeric process; all numerical threads1; no parallel audits/pools/CPUvideo/CPUmodel',frozen_at_utc=None)
    from specification import freeze
    specification=freeze(ROOT,c);c['checkpoints']=read(ROOT/'checkpoint-map.json')
    verify,extra=checks(c,specification)
    bindings={b['path']:b for b in specification['bindings']+extra}
    paths={p.resolve()for p in ROOT.glob('*.py')}|{p.resolve()for p in ROOT.glob('*spec.json')}|{ROOT/n for n in ['checkpoint-map.json','specification-receipt.json','PLAN.txt','PARENT_HANDOFF.txt','OWNER_MEMORY.txt','USER_AUTHORIZATION.json','SOURCE_ANALYSIS_REPORT.txt','HEARTBEAT_POLICY.txt','launch.ps1','VERIFICATION.json']}
    paths|={REPO/n for n in ['tdyssey/gpc_world_model.py','tdyssey/native_rgb_evaluation.py','tdyssey/native_rgb_r3a_data.py','tdyssey/envs/official.py','scripts/native_rgb_stage3_score.py','scripts/direct_target_background.py','manifests/upstream.json']}
    paths|={ROOT.parent/n for n in ['high-motion-fit-20261101/evaluation.py','region-weighted-cd-20261101/evaluation.py','generated-history-ab-20261101/evaluation.py','accelerated-rgb-run-20261031/train_eval.py']}
    paths|={p.resolve()for p in (Path(c['source_root'])/'gpc_rank_evaluation/diffusion').glob('*.py')}
    for p in sorted(paths):
        if not p.is_file():raise FileNotFoundError(p)
        if str(p)not in bindings:bindings[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    c['bindings']=list(bindings.values());c['frozen_at_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    resource=initialize_storage(ROOT,c);atomic_json(ROOT/'resource-preflight.json',resource)
    atomic_json(ROOT/'runtime-config.json',c,exclusive=True);digest=sha(ROOT/'runtime-config.json')
    atomic_json(ROOT/'freeze-receipt.json',dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),scoped_references_only=True,
        max_predictions=3672,max_image_denoiser_forwards=11016,model_calls_during_prepare=0,GPU_saved_statistics_images=verify['existing_saved_GPU_statistics_validation_images'],physics_calls=0,
        CPU=CPU,selected_B_families=[r['family_id']for r in read(ROOT/'B-spec.json')['rows']]),exclusive=True)
    print(dict(run_id=c['run_id'],config_sha256=digest,bindings=len(c['bindings']),task_name=c['windows_task_name'],VERIFICATION_passed=verify['passed'],resource=resource))

if __name__=='__main__':main()

