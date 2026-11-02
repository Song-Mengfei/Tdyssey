"""Paired frozen evaluation for one generated-history A/B experiment.

Imports and freeze perform saved-data work only. Runtime model calls belong to
the one authorized background worker. No physics, new cases or new sampler.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
AR1=ROOT.parent/'accelerated-rgb-run-20261031'
OC=ROOT.parent/'oc-continuation-20261031'
START_SHA='c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.)
_AR1_HELPER=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    from common import atomic_json
    atomic_json(path,value)
def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def _ar1():
    global _AR1_HELPER
    if _AR1_HELPER is None:
        for folder in (REPO,AR1):
            if str(folder) not in sys.path:sys.path.append(str(folder))
        name='_gh4_frozen_ar1_evaluation';spec=importlib.util.spec_from_file_location(name,AR1/'train_eval.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        _AR1_HELPER=module
    return _AR1_HELPER

def _bindings(paths):
    return [dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(Path(x) for x in paths))]

def _spec_check(spec,*,formal):
    rows=spec['rows'];expected=48 if formal else 32
    if len(rows)!=expected or len({r['family_id'] for r in rows})!=expected:
        raise ValueError('Frozen evaluation family denominator changed')
    if formal:
        if Counter(r['letter'] for r in rows)!=Counter(dict.fromkeys('TLOCXB',8)) or any(r['split']!='seen_val' for r in rows):
            raise ValueError('Formal48 split/letter scope changed')
    elif Counter((r['split'],r['letter']) for r in rows)!=Counter({(s,l):8 for s in ('train','seen_val') for l in ('O','C')}):
        raise ValueError('Motion32 split/letter scope changed')
    flat=[r['sample_id'] for b in spec['batches'] for r in b['rows']]
    if flat!=[r['sample_id'] for r in rows] or any(len(b['seeds'])!=16 for b in spec['batches']):
        raise ValueError('Original batch order/noise pairing changed')
    if spec.get('sampler')!=SAMPLER:raise ValueError('Only original native3 sampler is approved')

def freeze(root,config):
    """Copy only original formal48 and motion32 identities; no new selection."""
    root=Path(root);original_formal=AR1/'evaluation-spec.json';original_diag=OC/'diagnostic-spec.json'
    formal=read(original_formal);old_diag=read(original_diag)
    diag={k:deepcopy(old_diag[k]) for k in ('rows','batches','counts','seed','sampler','selection','scope','replanning','real_history','autoregressive')}
    diag.update(kind='gh4-existing-motion32-spec.v1',source_spec_path=str(original_diag.resolve()),source_spec_sha256=sha(original_diag))
    _spec_check(formal,formal=True);_spec_check(diag,formal=False)
    paths=[original_formal,original_diag,OC/'training/epoch2.pt',OC/'training/result.json',
           OC/'evaluation/epoch2/metrics.json',REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json',
           AR1/'train_eval.py',AR1/'selection.py',REPO/'scripts/native_rgb_stage3_score.py',
           REPO/'tdyssey/native_rgb_evaluation.py',REPO/'tdyssey/native_rgb_training.py']
    existing=read(OC/'evaluation/epoch2/metrics.json');oldresult=read(OC/'training/result.json')
    proof=next(x for x in oldresult['epochs'] if int(x['epoch'])==2)
    if proof['checkpoint']['sha256']!=START_SHA or sha(OC/'training/epoch2.pt')!=START_SHA or Path(proof['formal_metrics']).resolve()!=(OC/'evaluation/epoch2/metrics.json').resolve():
        raise ValueError('Reused formal result is not from approved common epoch2 checkpoint')
    source_model_rows=[r for r in existing['rows'] if r['method']!='copy_last']
    if {(r['sample_id'],r['horizon']) for r in source_model_rows}!={(r['sample_id'],h) for r in formal['rows'] for h in (1,4,8,16)}:
        raise ValueError('Existing formal epoch2 endpoint identities changed')
    rawrefs={r['sample_id']:r for r in existing['raw_files']}
    for spec in (formal,diag):
        for row in spec['rows']:
            p=Path(row['raw_path']);paths.append(p)
            meta=Path(row.get('metadata_path',str(p.with_suffix('.json'))))
            if meta.exists():paths.append(meta)
            with np.load(p,allow_pickle=False) as z:
                a=int(row['anchor']);actions=z['passed_actions']
                if not np.array_equal(actions,z['requested_actions']) or not z['control_valid'][a-3:a+16].all():
                    raise ValueError('Evaluation input crosses changed/unclean timeline')
                if a<3 or a+16>=len(z['rgb']) or z['rgb'].shape[1:]!=(3,96,96):raise ValueError('Incomplete four-history/H16 RGB')
                if not np.isfinite(actions[a-3:a+16]).all() or (actions[a-3:a+16]<0).any() or (actions[a-3:a+16]>511).any():
                    raise ValueError('Unchanged model action domain exceeded')
            if sha(p)!=row['raw_sha256']:raise ValueError('Frozen evaluation truth changed')
    for b in formal['batches']:
        for row in b['rows']:
            x=rawrefs[row['sample_id']];p=Path(x['path']);paths.append(p)
            if x['seeds']!=b['seeds'] or sha(p)!=x['sha256']:raise ValueError('Existing baseline raw/noise identity changed')
    for name,value in [('formal-spec.json',formal),('diagnostic-spec.json',diag)]:
        path=root/name
        if path.exists():
            if read(path)!=value:raise RuntimeError('Evaluation spec would change')
        else:write(path,value)
    available={}
    for mode in ('real_history','autoregressive'):
        p=OC/f'diagnostics/epoch2/{mode}/metrics.json';available[mode]=p.exists()
        if p.exists():
            value=read(p);_check_existing_mode(value,diag)
            paths.append(p);paths += [Path(x['path']) for x in value['raw_files']]
    baseline=dict(kind='gh4-baseline-reuse.v1',common_checkpoint_path=str((OC/'training/epoch2.pt').resolve()),
        common_checkpoint_sha256=START_SHA,formal_metrics_path=str((OC/'evaluation/epoch2/metrics.json').resolve()),
        formal_metrics_sha256=sha(OC/'evaluation/epoch2/metrics.json'),formal_reused=True,
        formal_missing_steps_saved_only_scoring=True,motion_diagnostic_existing=available,
        new_physics_calls=0,new_model_calls_during_freeze=0)
    path=root/'baseline-reuse.json'
    if path.exists() and read(path)!=baseline:raise RuntimeError('Baseline reuse identity changed')
    if not path.exists():write(path,baseline)
    return dict(formal_spec=str(root/'formal-spec.json'),diagnostic_spec=str(root/'diagnostic-spec.json'),
        baseline_reuse=str(path),formal48=48,motion32=32,bindings=_bindings(paths))

def _check_existing_mode(metrics,spec):
    rows={x['sample_id']:x for x in metrics['raw_files']}
    if set(rows)!={x['sample_id'] for x in spec['rows']}:raise ValueError('Existing diagnostic cases differ')
    for b in spec['batches']:
        for row in b['rows']:
            ref=rows[row['sample_id']]
            if ref['seeds']!=b['seeds'] or sha(ref['path'])!=ref['sha256']:
                raise ValueError('Existing diagnostic noise/raw differs')

def prepare_inputs(store,rows,mode):
    """No model work. AR input contains only initial history and known actions."""
    initial=[];actions=[];true_histories=[]
    for row in rows:
        d=store.arrays(row);a=int(row['anchor'])
        if not d['control_valid'][a-3:a+16].all():raise ValueError('Unclean prediction window')
        initial.append(d['rgb'][a-3:a+1].astype(np.float32)/np.float32(255))
        actions.append(d['passed_actions'][a-3:a+16].copy())
        if mode=='real_history':
            # Each slice ends before its own target a+k+1.
            true_histories.append(np.stack([d['rgb'][a+k-3:a+k+1].astype(np.float32)/np.float32(255) for k in range(16)]))
    x=dict(initial_histories=np.stack(initial),absolute_actions=np.stack(actions))
    if mode=='real_history':x['real_histories']=np.stack(true_histories)
    elif mode!='autoregressive':raise ValueError('Unknown input mode')
    return x

def callback_rollout(sample,initial_histories,absolute_actions,seeds,*,real_histories=None):
    """Offline-testable action/history contract; raw feedback is never repaired."""
    history=np.asarray(initial_histories,np.float32).copy();actions=np.asarray(absolute_actions)
    if history.shape[1:]!=(4,3,96,96) or actions.shape!=(len(history),19,2) or len(seeds)!=16:raise ValueError('H16 input shape mismatch')
    if not np.isfinite(actions).all() or (actions<0).any() or (actions>511).any():raise ValueError('Unchanged model action domain exceeded')
    if real_histories is not None and np.asarray(real_histories).shape!=(len(history),16,4,3,96,96):raise ValueError('True history shape mismatch')
    normalized=actions.astype(np.float32)/np.float32(255.5)-np.float32(1)
    outputs=[]
    for k in range(16):
        if real_histories is not None:history=np.asarray(real_histories[:,k],np.float32).copy()
        raw=np.asarray(sample(history.copy(),normalized[:,k:k+4].copy(),seed=int(seeds[k])))
        if raw.dtype!=np.float32 or raw.shape!=(len(history),3,96,96) or not np.isfinite(raw).all():raise ValueError('Native raw prediction contract')
        outputs.append(raw.copy())
        if real_histories is None:history=np.concatenate((history[:,1:],raw[:,None]),axis=1)
    return np.stack(outputs,axis=1)

def native_rollout(ctx,inputs,seeds):
    """Same native sampler/noise convention; keep recurrent histories on GPU."""
    import torch
    device=next(ctx.model.parameters()).device
    history=torch.as_tensor(inputs['initial_histories'].copy(),device=device,dtype=torch.float32)
    absolute=inputs['absolute_actions']
    if not np.isfinite(absolute).all() or (absolute<0).any() or (absolute>511).any():raise ValueError('Model action contract changed')
    # CPU float32 expression exactly matches the previous formal input contract.
    normalized=absolute.astype(np.float32)/np.float32(255.5)-np.float32(1)
    actions=torch.as_tensor(normalized,device=device);real=None
    if 'real_histories' in inputs:real=torch.as_tensor(inputs['real_histories'].copy(),device=device,dtype=torch.float32)
    index=device.index if device.index is not None else torch.cuda.current_device()
    sampler=ctx.source.DiffusionSampler(ctx.model,ctx.source.DiffusionSamplerConfig(**SAMPLER));out=[]
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for k,seed in enumerate(seeds):
            if real is not None:history=real[:,k]
            with torch.random.fork_rng(devices=[index]):
                torch.random.default_generator.manual_seed(int(seed));torch.cuda.default_generators[index].manual_seed(int(seed))
                raw,_=sampler.sample(history.contiguous(),actions[:,k:k+4].contiguous())
            raw=raw.detach().float();out.append(raw)
            if real is None:history=torch.cat((history[:,1:],raw[:,None]),dim=1)
        result=torch.stack(out,dim=1).cpu().numpy()
    if result.dtype!=np.float32 or result.shape!=(len(history),16,3,96,96) or not np.isfinite(result).all():raise FloatingPointError('Incomplete/nonfinite raw evaluation sequence')
    return result

def _annotations(metrics):
    value=deepcopy(metrics)
    value.update(visible_angle_assessable=False,physical_rotation_assessable=False,underrotation_assessable=False,
        underrotation_deg=None,absolute_underrotation_deg=None,
        angle_unknown_reason='Visible angular registration is uncalibrated for signed rotation under symmetry/occlusion/deformation; numeric legacy angle is diagnostic only.',
        shape_assessable=bool(value.get('legacy_shape_assessable',False)))
    return value

def _scored_row(helper,row,store,prediction,step,label,mode,calibration):
    d=store.arrays(row);a=int(row['anchor']);h=int(step)
    from selection import centers
    xy=centers(row['letter'],d['states'][[a,a+h]])
    translation=float(np.linalg.norm(xy[1]-xy[0])*96/512)
    angle=math.degrees(float(d['states'][a+h,4]-d['states'][a,4]))
    threshold={1:(1,3),4:(2,5),8:(4,10),16:(8,20)}.get(h)
    return dict(sample_id=row['sample_id'],family_id=row['family_id'],split=row['split'],letter=row['letter'],
        task=row['task'],anchor=a,step=h,horizon=h,method=label,input_mode=mode,
        truth_translation_image_px=translation,truth_signed_rotation_deg=angle,
        obvious_motion=None if threshold is None else bool(translation>=threshold[0] or abs(angle)>=threshold[1]),
        metrics=_annotations(helper._metric(prediction,d['rgb'][a+h],d['rgb'][a],row['letter'],calibration)))

def _model_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def _inputs_evidence(rows,inputs,seeds):
    result=dict(sample_ids=[x['sample_id'] for x in rows],family_ids=[x['family_id'] for x in rows],
        seeds=list(seeds),initial_history_sha256=array_sha(inputs['initial_histories']),
        absolute_actions_sha256=array_sha(inputs['absolute_actions']),
        normalized_actions_sha256=array_sha(inputs['absolute_actions'].astype(np.float32)/np.float32(255.5)-np.float32(1)),
        shape_history=list(inputs['initial_histories'].shape),shape_actions=list(inputs['absolute_actions'].shape))
    if 'real_histories' in inputs:result['per_target_true_histories_sha256']=array_sha(inputs['real_histories'])
    return result

def _evaluate_group(root,config,ctx,label,spec,mode,store,*,formal=False):
    helper=_ar1();root=Path(root)
    folder=root/'evaluation'/label/'formal' if formal else root/'evaluation'/label/'diagnostics'/mode
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('No repeated evaluation phase: '+str(folder))
    calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    rows=[];files=[];inputs_proofs=[];begin=time.monotonic()
    for bi,b in enumerate(spec['batches']):
        inputs=prepare_inputs(store,b['rows'],mode);inputs_proofs.append(_inputs_evidence(b['rows'],inputs,b['seeds']))
        predictions=native_rollout(ctx,inputs,b['seeds'])
        for j,row in enumerate(b['rows']):
            path=folder/f'raw-{bi:03d}-{j}.npz';helper.raw_npz(path,config,root,raw=predictions[j])
            files.append(dict(**row,path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,seeds=b['seeds']))
            for step in range(1,17):rows.append(_scored_row(helper,row,store,predictions[j,step-1],step,label,mode,calibration))
        write(folder/'progress.json',dict(label=label,mode=mode,completed_batches=bi+1,total_batches=len(spec['batches']),sampler_batch_calls=(bi+1)*16))
    payload=dict(complete=True,label=label,method=label,input_mode=mode,rows=rows,raw_files=files,input_evidence=inputs_proofs,
        sampler=SAMPLER,model_identity=deepcopy(ctx.checkpoint_identity),
        paired_original_batch_noise=True,true_future_feedback=mode=='real_history',
        true_future_leakage_into_own_step=False,attribution_only=mode=='real_history',
        raw_feedback_unclipped=mode=='autoregressive',horizons=list(range(1,17)),highlight_horizons=[1,4,8,16],
        cases=len(spec['rows']),independent_families=len({x['family_id'] for x in spec['rows']}),
        sampler_batch_calls=len(spec['batches'])*16,denoiser_batch_forwards=len(spec['batches'])*48,
        physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-begin,
        limits=['Native sampler unchanged, formal3 denoising only. No displayed/quantized frames fed back.',
                'True-history t+16 is one-step attribution, not closed-loop H16.',
                'Numeric visible angle/corner counts cannot establish signed underrotation or rigid shape preservation; unknown is not a pass.'])
    write(folder/'metrics.json',payload)
    return dict(metrics=str((folder/'metrics.json').resolve()),raw_files=files,wall_seconds=payload['wall_seconds'],sampler_batch_calls=payload['sampler_batch_calls'])

def _reuse_formal(root,config,ctx,spec,store):
    root=Path(root);folder=root/'evaluation/baseline/formal';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('Baseline formal assembly already complete')
    helper=_ar1();oldpath=OC/'evaluation/epoch2/metrics.json';original=read(oldpath)
    if ctx.checkpoint_identity.get('sha256')!=START_SHA:raise ValueError('Baseline model is not approved epoch2')
    originals={(x['sample_id'],int(x['horizon'])):x for x in original['rows'] if x['method']!='copy_last'}
    raw={x['sample_id']:x for x in original['raw_files']};calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    scored=[];files=[];reused=0;newscore=0;begin=time.monotonic()
    for b in spec['batches']:
        for row in b['rows']:
            x=raw[row['sample_id']]
            if x['seeds']!=b['seeds'] or sha(x['path'])!=x['sha256']:raise ValueError('Reused baseline prediction identity changed')
            with np.load(x['path'],allow_pickle=False) as z:pred=z['raw']
            if pred.shape!=(16,3,96,96) or pred.dtype!=np.float32 or not np.isfinite(pred).all():raise ValueError('Baseline raw contract changed')
            files.append(dict(**row,**{k:x[k] for k in ('path','sha256','seeds')},reused=True))
            for h in range(1,17):
                old=originals.get((row['sample_id'],h))
                if old is not None:
                    value=deepcopy(old);value.update(method='baseline',input_mode='autoregressive',step=h,split=row['split'],anchor=row['anchor'])
                    value['metrics']=_annotations(old['metrics']);reused+=1
                else:value=_scored_row(helper,row,store,pred[h-1],h,'baseline','autoregressive',calibration);newscore+=1
                scored.append(value)
    payload=dict(complete=True,label='baseline',method='baseline',input_mode='autoregressive',rows=scored,raw_files=files,
        sampler=SAMPLER,model_identity=ctx.checkpoint_identity,reused_formal_metrics=str(oldpath.resolve()),
        reused_formal_metrics_sha256=sha(oldpath),reused_endpoint_scores=reused,new_saved_only_endpoint_scores=newscore,
        new_model_calls=0,sampler_batch_calls=0,denoiser_batch_forwards=0,true_future_feedback=False,
        raw_feedback_unclipped=True,physical_steps=0,optimizer_updates=0,cases=48,
        horizons=list(range(1,17)),highlight_horizons=[1,4,8,16],wall_seconds=time.monotonic()-begin)
    # Original four copy-last endpoints are a reference, never a substitute A arm.
    by_id={r['sample_id']:r for r in spec['rows']}
    for old in original['rows']:
        if old['method']=='copy_last':
            value=deepcopy(old);value.update(step=int(old['horizon']),input_mode='copy_last',split=by_id[old['sample_id']]['split'])
            value['metrics']=_annotations(old['metrics']);payload['rows'].append(value)
    payload['reused_copy_last_endpoint_scores']=sum(x['method']=='copy_last' for x in payload['rows'])
    write(folder/'metrics.json',payload)
    return dict(metrics=str((folder/'metrics.json').resolve()),raw_files=files,reused=True,wall_seconds=payload['wall_seconds'],sampler_batch_calls=0)

def _run(root,config,ctx,label):
    if label not in ('baseline','A','B'):raise ValueError('Only approved common-start/A/B comparisons')
    if not hasattr(ctx,'checkpoint_identity') or not {'path','sha256'}.issubset(ctx.checkpoint_identity):raise ValueError('Explicit evaluated checkpoint identity required')
    root=Path(root);helper=_ar1();store=helper.RawStore(max_bytes=48*1024**2)
    formal=read(root/'formal-spec.json');diag=read(root/'diagnostic-spec.json')
    _spec_check(formal,formal=True);_spec_check(diag,formal=False)
    oldmode=ctx.model.training;ctx.model.eval();before=_model_sha(ctx.model);begin=time.monotonic()
    try:
        formal_result=_reuse_formal(root,config,ctx,formal,store) if label=='baseline' else _evaluate_group(root,config,ctx,label,formal,'autoregressive',store,formal=True)
        diagnostic_results={};reuse=read(root/'baseline-reuse.json')
        for mode in ('real_history','autoregressive'):
            if label=='baseline' and reuse['motion_diagnostic_existing'][mode]:
                path=OC/f'diagnostics/epoch2/{mode}/metrics.json';value=read(path);_check_existing_mode(value,diag)
                diagnostic_results[mode]=dict(metrics=str(path.resolve()),raw_files=value['raw_files'],reused=True,sampler_batch_calls=0)
            else:diagnostic_results[mode]=_evaluate_group(root,config,ctx,label,diag,mode,store)
        after=_model_sha(ctx.model)
        if before!=after:raise RuntimeError('Evaluation changed model state')
        proof=dict(label=label,checkpoint=ctx.checkpoint_identity,model_state_sha256_before=before,model_state_sha256_after=after,
            model_state_unchanged=True,formal_spec_sha256=sha(root/'formal-spec.json'),diagnostic_spec_sha256=sha(root/'diagnostic-spec.json'),
            same_original_batch_composition_actions_seeds=True,training_rng_preservation='Engine protected callback captures/restores global, supervisory, generation RNGs and mode; this module does not use supervisory/generation generators.',
            baseline_formal_original_runtime_tensors_not_separately_saved=label=='baseline',
            source_sampler=SAMPLER,history_storage='Native float32 recurrent history remains on GPU; only completed16frame outputs copied to CPU; no extra quantization/clip.',
            physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-begin,loader=dict(cache_bytes=store.bytes,loads=store.loads))
        path=root/'evaluation'/label/'identity-evidence.json';write(path,proof)
        result=dict(label=label,formal=formal_result,diagnostics=diagnostic_results,identity_evidence=str(path.resolve()),wall_seconds=proof['wall_seconds'])
        write(root/'evaluation'/label/'result.json',result);return result
    finally:
        ctx.model.train(oldmode);store.cache.clear()

def run_baseline(root,config,ctx):return _run(root,config,ctx,'baseline')
def run_arm(root,config,ctx,label):return _run(root,config,ctx,label)
