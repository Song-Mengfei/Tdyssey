"""C/D evaluation of existing cases; old epoch2/A/B results are references.

Import/freeze/reuse are saved-only. Runtime C/D calls belong to the single owner
worker. Original native sampler math is unchanged; initial noise is observed and
checked against an independent same-seed/batch CUDA-generator replay.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
AB=ROOT.parent/'generated-history-ab-20261101'
_OLD=None
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    from common import atomic_json
    atomic_json(path,value)


def frozen_json(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Frozen evaluation metadata would change: '+str(path))
    else:write(path,value)


def old():
    global _OLD
    if _OLD is None:
        spec=importlib.util.spec_from_file_location('_cd_frozen_ab_evaluation',AB/'evaluation.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);_OLD=module
    return _OLD


def _mean(metrics,mode,split,horizon,key):
    rows=[row for row in metrics['rows'] if row.get('method')!='copy_last' and row.get('split')==split and row['horizon']==horizon]
    if len(rows)!=16:raise ValueError('Original motion-validation denominator changed')
    values=[row['metrics'].get(key) for row in rows]
    if any(value is None or not np.isfinite(value) for value in values):raise ValueError('Missing original strict support signal')
    return float(np.mean(values))


def freeze(root,config):
    root=Path(root);helper=old();result=read(AB/'background/training-result.json')
    if not result.get('complete'):raise ValueError('Original A/B result is incomplete')
    paths=[AB/'background/training-result.json',AB/'evaluation.py',AB/'report.py']
    formal=read(AB/'formal-spec.json');diag=read(AB/'diagnostic-spec.json')
    helper._spec_check(formal,formal=True);helper._spec_check(diag,formal=False)
    for name,value in (('formal-spec.json',formal),('diagnostic-spec.json',diag)):
        target=root/name
        if target.exists() and read(target)!=value:raise ValueError('Frozen existing evaluation cases would change')
        if not target.exists():write(target,value)
        paths.append(AB/name)
    refs={}
    for label in ('baseline','A','B'):
        value=deepcopy(result['evaluations'][label]);refs[label]=value
        for info,spec in [(value['formal'],formal)]+[(value['diagnostics'][mode],diag) for mode in ('real_history','autoregressive')]:
            path=Path(info['metrics']);metrics=read(path);paths.append(path)
            helper._check_existing_mode(metrics,spec)
            rows={(row['sample_id'],int(row['horizon'])) for row in metrics['rows'] if row.get('method')!='copy_last'}
            if rows!={(row['sample_id'],h) for row in spec['rows'] for h in range(1,17)}:raise ValueError('Original full sixteen-frame score support changed')
            paths.extend(Path(row['path']) for row in metrics['raw_files'])
        paths.append(Path(value['identity_evidence']))
    A_real=read(refs['A']['diagnostics']['real_history']['metrics']);A_ar=read(refs['A']['diagnostics']['autoregressive']['metrics'])
    start_real=read(refs['baseline']['diagnostics']['real_history']['metrics'])
    limits=dict(kind='cd-predeclared-support-signals.v1',source='Original saved A full precision; no rounded display thresholds',
        motion_validation_families=16,
        H1_IoU_min=_mean(A_real,'real_history','seen_val',1,'raw_mask_iou'),
        H1_position_max_imagepx=_mean(A_real,'real_history','seen_val',1,'object_position_error_image_px'),
        H16_AR_position_max_imagepx=_mean(A_ar,'autoregressive','seen_val',16,'object_position_error_image_px'),
        epoch2_H1_position_reference_imagepx=_mean(start_real,'real_history','seen_val',1,'object_position_error_image_px'),
        comparisons='Strict >=/<= saved float64 means. No posthoc tolerance or relaxation. Unknown is not pass.',
        sources=[dict(path=str(path),sha256=sha(path)) for path in (Path(refs['A']['diagnostics']['real_history']['metrics']),Path(refs['A']['diagnostics']['autoregressive']['metrics']),Path(refs['baseline']['diagnostics']['real_history']['metrics']))])
    frozen_json(root/'support-limits.json',limits)
    frozen_json(root/'reused-evaluations.json',dict(source_result=str((AB/'background/training-result.json').resolve()),source_result_sha256=sha(AB/'background/training-result.json'),evaluations=refs,
        original_initial_noise='Historical epsilon tensors were not separately stored. Original batch/slot/seeds/source remain frozen; new C/D records actual epsilon and independent replay matches.'))
    paths.extend([root/'support-limits.json',root/'reused-evaluations.json',AB/'evaluation.py',
        REPO/'artifacts/accelerated-rgb-run-20261031/train_eval.py',REPO/'artifacts/accelerated-rgb-run-20261031/selection.py',
        REPO/'scripts/native_rgb_stage3_score.py',REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json'])
    paths.extend(Path(row['raw_path']) for spec in (formal,diag) for row in spec['rows'])
    return dict(formal_spec=str(root/'formal-spec.json'),diagnostic_spec=str(root/'diagnostic-spec.json'),support_limits=str(root/'support-limits.json'),
        bindings=[dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(set(paths))],new_model_calls=0,new_physics_calls=0)


def run_baseline(root,config,ctx=None):
    """Return all three old references; never loads a checkpoint or model."""
    root=Path(root);source=read(root/'reused-evaluations.json')
    if sha(source['source_result'])!=source['source_result_sha256']:raise ValueError('Old A/B source result changed')
    result=deepcopy(source['evaluations'])
    for value in result.values():
        value['reused']=True;value['new_model_calls']=0
        for info in [value['formal'],*value['diagnostics'].values()]:
            info['source_wall_seconds']=info.get('wall_seconds');info['wall_seconds']=0.;info['reused']=True;info['sampler_batch_calls']=0
    write(root/'evaluation/reused-results.json',result)
    return result


@contextmanager
def capture_noise(torch,expected,sink):
    """Observe the native single draw; return its exact tensor unchanged."""
    original=torch.randn;calls=[0]
    def observed(*args,**kwargs):
        value=original(*args,**kwargs);calls[0]+=1
        if tuple(value.shape)!=tuple(expected.shape) or not torch.equal(value,expected):
            raise RuntimeError('Actual initial epsilon differs from original-seed batch replay')
        array=value.detach().float().cpu().contiguous().numpy()
        expected_array=expected.detach().float().cpu().contiguous().numpy()
        sink(dict(actual_initial_epsilon_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            replayed_seed_batch_epsilon_sha256=hashlib.sha256(expected_array.tobytes()).hexdigest(),
            per_case_initial_epsilon_sha256=[hashlib.sha256(row.tobytes()).hexdigest() for row in array],
            shape=list(array.shape),dtype=str(array.dtype),actual_matches_independent_replay=True))
        return value
    torch.randn=observed
    try:
        yield
        if calls[0]!=1:raise RuntimeError('Native sampler did not make exactly one initial randn draw')
    finally:torch.randn=original


def native_rollout(ctx,inputs,seeds):
    import torch
    device=next(ctx.model.parameters()).device;index=device.index if device.index is not None else torch.cuda.current_device()
    history=torch.as_tensor(inputs['initial_histories'].copy(),device=device,dtype=torch.float32)
    absolute=inputs['absolute_actions'];normalized=absolute.astype(np.float32)/np.float32(255.5)-np.float32(1)
    if not np.isfinite(absolute).all() or (absolute<0).any() or (absolute>511).any():raise ValueError('Original action interface changed')
    actions=torch.as_tensor(normalized,device=device);real=None;noise=[];outputs=[]
    if 'real_histories' in inputs:real=torch.as_tensor(inputs['real_histories'].copy(),device=device,dtype=torch.float32)
    sampler=ctx.source.DiffusionSampler(ctx.model,ctx.source.DiffusionSamplerConfig(**SAMPLER))
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for step,seed in enumerate(seeds):
            if real is not None:history=real[:,step]
            # Independent generator replay does not consume the actual sampler's
            # default RNG. The original same-B4 seed/shape is retained.
            generator=torch.Generator(device=device).manual_seed(int(seed))
            expected=torch.randn(len(history),3,96,96,device=device,dtype=torch.float32,generator=generator)
            receipt=[]
            with torch.random.fork_rng(devices=[index]):
                torch.random.default_generator.manual_seed(int(seed));torch.cuda.default_generators[index].manual_seed(int(seed))
                with capture_noise(torch,expected,receipt.append):
                    raw,_=sampler.sample(history.contiguous(),actions[:,step:step+4].contiguous())
            noise.append(dict(step=step+1,seed=int(seed),batch_size=len(history),**receipt[0]))
            raw=raw.detach().float();outputs.append(raw)
            if real is None:history=torch.cat((history[:,1:],raw[:,None]),dim=1)
        result=torch.stack(outputs,dim=1).cpu().numpy()
    if result.dtype!=np.float32 or result.shape!=(len(history),16,3,96,96) or not np.isfinite(result).all():raise FloatingPointError('Invalid raw C/D prediction')
    return result,noise


def _group(root,config,ctx,label,spec,mode,store,formal=False):
    helper=old();score=helper._ar1();root=Path(root);folder=root/'evaluation'/label/('formal' if formal else 'diagnostics')
    if not formal:folder=folder/mode
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('No repeated C/D evaluation')
    calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    rows=[];files=[];proofs=[];noise_proofs=[];started=time.monotonic()
    for bi,batch in enumerate(spec['batches']):
        inputs=helper.prepare_inputs(store,batch['rows'],mode)
        proofs.append(helper._inputs_evidence(batch['rows'],inputs,batch['seeds']))
        predictions,noise=native_rollout(ctx,inputs,batch['seeds'])
        noise_proofs.append(dict(batch=bi,sample_ids=[row['sample_id'] for row in batch['rows']],steps=noise))
        for slot,row in enumerate(batch['rows']):
            path=folder/f'raw-{bi:03d}-{slot}.npz';score.raw_npz(path,config,root,raw=predictions[slot])
            files.append(dict(**row,path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,seeds=batch['seeds'],
                initial_epsilon_sha256=[item['per_case_initial_epsilon_sha256'][slot] for item in noise]))
            for h in range(1,17):rows.append(helper._scored_row(score,row,store,predictions[slot,h-1],h,label,mode,calibration))
        write(folder/'progress.json',dict(completed_batches=bi+1,total_batches=len(spec['batches']),sampler_batch_calls=(bi+1)*16))
    payload=dict(complete=True,label=label,method=label,input_mode=mode,rows=rows,raw_files=files,input_evidence=proofs,noise_evidence=noise_proofs,
        sampler=SAMPLER,model_identity=deepcopy(ctx.checkpoint_identity),cases=len(spec['rows']),independent_families=len(spec['rows']),
        true_future_feedback=mode=='real_history',true_future_leakage_into_own_step=False,attribution_only=mode=='real_history',raw_feedback_unclipped=mode=='autoregressive',
        horizons=list(range(1,17)),highlight_horizons=[1,4,8,16],sampler_batch_calls=len(spec['batches'])*16,denoiser_batch_forwards=len(spec['batches'])*48,
        independent_epsilon_replay_draws=len(spec['batches'])*16,epsilon_recording='Actual original native randn tensor observed unchanged; independent same-seed same-B4 CUDA-generator replay matches exactly. Old historical tensors were not stored.',
        physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-started)
    write(folder/'metrics.json',payload)
    return dict(metrics=str((folder/'metrics.json').resolve()),raw_files=files,wall_seconds=payload['wall_seconds'],sampler_batch_calls=payload['sampler_batch_calls'])


def run_arm(root,config,ctx,label):
    if label not in ('C','D'):raise ValueError('Only approved C/D new model evaluation')
    helper=old();root=Path(root);formal=read(root/'formal-spec.json');diag=read(root/'diagnostic-spec.json')
    helper._spec_check(formal,formal=True);helper._spec_check(diag,formal=False)
    if not hasattr(ctx,'checkpoint_identity') or not ctx.checkpoint_identity.get('sha256'):raise ValueError('Explicit final evaluated checkpoint identity required')
    store=helper._ar1().RawStore(max_bytes=48*1024**2);mode=ctx.model.training;ctx.model.eval();before=helper._model_sha(ctx.model);start=time.monotonic()
    try:
        f=_group(root,config,ctx,label,formal,'autoregressive',store,formal=True)
        d={name:_group(root,config,ctx,label,diag,name,store) for name in ('real_history','autoregressive')}
        after=helper._model_sha(ctx.model)
        if before!=after:raise RuntimeError('Evaluation mutated model')
        proof=dict(label=label,checkpoint=ctx.checkpoint_identity,model_state_sha256_before=before,model_state_sha256_after=after,model_state_unchanged=True,
            formal_spec_sha256=sha(root/'formal-spec.json'),diagnostic_spec_sha256=sha(root/'diagnostic-spec.json'),
            same_original_batch_composition_actions_seeds=True,actual_epsilon_and_independent_replay_recorded=True,
            physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-start)
        path=root/'evaluation'/label/'identity-evidence.json';write(path,proof)
        result=dict(label=label,formal=f,diagnostics=d,identity_evidence=str(path.resolve()),wall_seconds=proof['wall_seconds'])
        write(root/'evaluation'/label/'result.json',result);return result
    finally:ctx.model.train(mode);store.cache.clear()
