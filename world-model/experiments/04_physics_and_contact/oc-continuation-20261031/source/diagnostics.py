"""Frozen moving O/C attribution groups; no import-time model or simulator work.

RGB/action histories alone cross the prediction boundary. Saved states/contacts
select deliberately motion-biased diagnostic examples, never natural test data.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
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
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.)
SEED=2026103107
_AR1=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value):
    from common import atomic_json
    atomic_json(path,value)

def load_ar1():
    global _AR1
    if _AR1 is None:
        if str(REPO) not in sys.path:sys.path.append(str(REPO))
        # Keep new-stage common first; old selection has saved geometry only.
        if str(AR1) not in sys.path:sys.path.append(str(AR1))
        name='_oc_frozen_ar1_train_eval';spec=importlib.util.spec_from_file_location(name,AR1/'train_eval.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);_AR1=module
    return _AR1

def _committed(meta,a,h=16):
    return any(int(s['start_control'])<=a and a+h<=int(s['end_control']) for s in meta.get('segments',[]))

def select_moving_rows(records,train_rows,*,per_group=8):
    """Saved-only selection: strongest actual *first target* contact rotation.

    Train anchors must belong to the existing unique H1 pool. Seen-val anchors
    may be any clean mathematical H16 window of those preassigned families;
    crossing a recorded replan is disclosed, not called precommitted open-loop.
    """
    membership={(r['family_id'],int(r['anchor'])) for r in train_rows}
    candidates=defaultdict(list);support=defaultdict(lambda:dict(families=0,candidates=0))
    for record in records:
        meta=read(record['metadata_path']);letter=meta['letter'];split=meta['split']
        if letter not in ('O','C') or split not in ('train','seen_val'):continue
        p=Path(record['raw_path'])
        if sha(p)!=record['raw_sha256'] or sha(record['metadata_path'])!=record['metadata_sha256']:
            raise ValueError('Frozen source changed before diagnostics selection')
        with np.load(p,allow_pickle=False) as z:
            states=z['states'];actions=z['passed_actions'];valid=z['control_valid'];contact=z['contact_positive']
            if not np.array_equal(actions,z['requested_actions']):raise ValueError('Unmatched requested/passed actions')
        clean=int(meta['clean_control_count']);found=[]
        for a in range(3,clean-16+1):
            if split=='train' and (meta['family_id'],a) not in membership:continue
            if not valid[a-3:a+16].all() or (actions[a-3:a+16]<0).any() or (actions[a-3:a+16]>511).any():continue
            step_angle=math.degrees(float(states[a+1,4]-states[a,4]))
            total_angle=math.degrees(float(states[a+16,4]-states[a,4]))
            if not bool(contact[a]) or abs(step_angle)<.5:continue
            row=dict(sample_id=f'{meta["family_id"]}:{meta["attempt_id"]}:a{a}:H16:motion-diagnostic',
                raw_path=str(p.resolve()),raw_sha256=record['raw_sha256'],metadata_path=record['metadata_path'],
                metadata_sha256=record['metadata_sha256'],family_id=meta['family_id'],attempt_id=meta['attempt_id'],
                split=split,letter=letter,task=meta['task'],anchor=a,horizon=16,
                positive_contact=True,contact_positive=True,truth_h1_signed_rotation_deg=step_angle,
                truth_h16_signed_rotation_deg=total_angle,event='contact_and_rotation',
                committed_future=_committed(meta,a),feedback_recorded=not _committed(meta,a))
            found.append(row)
        key=(split,letter)
        if found:
            support[key]['families']+=1;support[key]['candidates']+=len(found)
            # One best moving anchor per family, then at most eight families.
            found.sort(key=lambda r:(-abs(r['truth_h1_signed_rotation_deg']),-abs(r['truth_h16_signed_rotation_deg']),r['sample_id']))
            candidates[key].append(found[0])
    chosen=[];counts={}
    for split in ('train','seen_val'):
        for letter in ('O','C'):
            key=(split,letter);values=candidates[key]
            values.sort(key=lambda r:(-abs(r['truth_h1_signed_rotation_deg']),-abs(r['truth_h16_signed_rotation_deg']),r['sample_id']))
            selected=values[:per_group];chosen.extend(selected)
            counts[f'{split}:{letter}']=dict(selected=len(selected),missing_full_h16=max(0,per_group-len(selected)),**support[key])
    if len({r['family_id'] for r in chosen})!=len(chosen):raise ValueError('Diagnostic family duplicated')
    return chosen,counts

def freeze(root,config):
    root=Path(root);ar1=Path(config.get('data_root',str(AR1)))
    train=read(ar1/'selection/train-index.json');selection=read(ar1/'selection/summary.json')
    rows,counts=select_moving_rows(selection['source_records'],train)
    batches=[]
    for split in ('train','seen_val'):
        for letter in ('O','C'):
            group=[r for r in rows if r['split']==split and r['letter']==letter]
            for start in range(0,len(group),4):
                batch=group[start:start+4];ids=[r['sample_id'] for r in batch]
                seeds=[int(digest(dict(domain='oc-moving-paired-real-auto.v1',seed=SEED,ids=ids,step=i))[:8],16) for i in range(16)]
                batches.append(dict(rows=batch,seeds=seeds))
    oldspec=read(ar1/'evaluation-spec.json');oldmetrics=read(ar1/'evaluation/after/metrics.json')
    strong={r['sample_id'] for r in oldmetrics['rows'] if r['method']=='after' and r['horizon']==16 and r['letter'] in ('O','C') and r['obvious_motion']}
    raw_by_id={r['sample_id']:r for r in oldmetrics['raw_files']};denoise=[]
    for bi,b in enumerate(oldspec['batches']):
        for slot,row in enumerate(b['rows']):
            if row['sample_id'] not in strong:continue
            oldraw=raw_by_id[row['sample_id']]
            if sha(oldraw['path'])!=oldraw['sha256']:raise ValueError('Frozen epoch1 output changed')
            denoise.append(dict(row=row,original_batch_index=bi,original_batch_size=len(b['rows']),
                original_slot=slot,seeds=b['seeds'],reused3=oldraw))
    if len(denoise)!=8 or {l:sum(x['row']['letter']==l for x in denoise) for l in ('O','C')}!={'O':4,'C':4}:
        raise ValueError('Approved original strong-motion O4/C4 identity changed')
    spec=dict(kind='oc-frozen-motion-diagnostics.v1',run_id=config.get('run_id'),rows=rows,batches=batches,
        counts=counts,seed=SEED,sampler=SAMPLER,formal_spec_path=str((ar1/'evaluation-spec.json').resolve()),
        formal_spec_sha256=sha(ar1/'evaluation-spec.json'),denoise16_cases=denoise,
        selection='Existing clean H16; actual positive AO impulse at anchor control and absolute first-step physical rotation>=.5deg. Per family maximize absolute H1 rotation then H16 rotation; choose at most8 families per train/seen-val O/C. Predictions never consulted.',
        scope='Deliberately motion-biased attribution group, not natural distribution or new independent test set. Train results are fitting diagnostics, never validation generalization.',
        replanning='feedback_recorded=true means recorded future actions cross real replanning; no future true RGB enters autoregressive inference.',
        real_history='At target a+k+1 use only true RGB[a+k-3:a+k+1] and actions[a+k-3:a+k+1]. Attribution, not a closed-loop H16 forecast.',
        autoregressive='Only initial4 true frames; raw model float32 feedback for16 steps; no truth/padding/shape repairs.',
        denoise16_scope='Only epoch1 existing O4/C4 strong-motion seen-val. Original3 outputs reused. Source sampler unchanged except num_steps=16; original batch-noise slot injected without other sibling model calls. B1/B4 kernel rounding may differ.')
    loss=dict(kind='oc-fixed-sample-fixed-noise-residual.v1',run_id=config.get('run_id'),rows=rows,
        batches=[dict(rows=b['rows'],seed_index=i) for i,b in enumerate(batches)],
        seeds=dict(branch_seed=3026103107,target_seed=4026103107,weak_seed=5026103107),
        objective='Existing native residual MSE, same50/50 clean/weak policy; independent generators reset to these seeds at every checkpoint. No optimizer/gradient.',
        limitation='Selected moving samples only; loss is not natural average or proof of global convergence.')
    for name,value in [('diagnostic-spec.json',spec),('loss-spec.json',loss)]:
        path=root/name
        if path.exists():
            if read(path)!=value:raise RuntimeError('Frozen diagnostic spec would change')
        else:write(path,value)
    return dict(diagnostic_spec=str(root/'diagnostic-spec.json'),loss_spec=str(root/'loss-spec.json'),counts=counts,selected=len(rows))

def predict_modes(sample,frames,actions,seeds,*,mode):
    """Pure callback contract tested without Torch. Future target never input."""
    frames=np.asarray(frames);actions=np.asarray(actions)
    if frames.ndim!=5 or frames.shape[1:]!=(20,3,96,96) or actions.shape!=(len(frames),19,2):raise ValueError('Expected20 real frames and19 aligned actions')
    if len(seeds)!=16:raise ValueError('Sixteen paired seeds required')
    if actions.min()<0 or actions.max()>511 or not np.isfinite(actions).all():raise ValueError('Original normalized model domain violated')
    if mode not in ('real_history','autoregressive'):raise ValueError('Unknown diagnostic mode')
    normalized=actions.astype(np.float32)/np.float32(255.5)-np.float32(1.)
    history=frames[:,:4].astype(np.float32)/np.float32(255.);out=[]
    for k in range(16):
        if mode=='real_history':history=frames[:,k:k+4].astype(np.float32)/np.float32(255.)
        raw=np.asarray(sample(history.copy(),normalized[:,k:k+4].copy(),seed=int(seeds[k])))
        if raw.dtype!=np.float32 or raw.shape!=(len(frames),3,96,96) or not np.isfinite(raw).all():raise FloatingPointError('Raw prediction shape/dtype/finite contract')
        out.append(raw.copy())
        if mode=='autoregressive':history=np.concatenate([history[:,1:],raw[:,None]],axis=1)
    return np.stack(out,axis=1)

def _score_sequence(ar1,raw,row,store,calibration,method):
    d=store.arrays(row);a=int(row['anchor']);last=d['rgb'][a];result=[]
    for k in range(16):
        result.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],split=row['split'],letter=row['letter'],
            task=row['task'],anchor=a,step=k+1,horizon=k+1,method=method,
            truth_signed_rotation_deg=math.degrees(float(d['states'][a+k+1,4]-d['states'][a,4])),
            metrics=ar1._metric(raw[k],d['rgb'][a+k+1],last,row['letter'],calibration)))
    return result

def evaluate_modes(root,config,ctx,epoch):
    from tdyssey.native_rgb_r2c1r1_training import torch_native_sample
    ar1=load_ar1();root=Path(root);spec=read(root/'diagnostic-spec.json')
    calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    sample=torch_native_sample(ctx.model,ctx.source,SAMPLER);results={};oldmode=ctx.model.training;ctx.model.eval()
    try:
        for mode in ('real_history','autoregressive'):
            folder=root/'diagnostics'/f'epoch{epoch}'/mode;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'metrics.json').exists():raise RuntimeError('No repeated diagnostic inference')
            begin=time.monotonic();scored=[];files=[]
            for bi,b in enumerate(spec['batches']):
                frames=[];actions=[]
                for row in b['rows']:
                    d=ctx.store.arrays(row);a=int(row['anchor'])
                    if not d['control_valid'][a-3:a+16].all():raise ValueError('Diagnostic crosses suspicious controls')
                    frames.append(d['rgb'][a-3:a+17]);actions.append(d['passed_actions'][a-3:a+16])
                predictions=predict_modes(sample,np.stack(frames),np.stack(actions),b['seeds'],mode=mode)
                for j,row in enumerate(b['rows']):
                    path=folder/f'raw-{bi:03d}-{j}.npz';ar1.raw_npz(path,config,root,raw=predictions[j])
                    files.append(dict(**row,path=str(path.resolve()),sha256=sha(path),seeds=b['seeds']))
                    scored+=_score_sequence(ar1,predictions[j],row,ctx.store,calibration,mode)
                write(folder/'progress.json',dict(completed_batches=bi+1,total_batches=len(spec['batches']),model_batch_calls=(bi+1)*16))
            payload=dict(complete=True,epoch=epoch,method=mode,rows=scored,raw_files=files,
                sampler=SAMPLER,paired_noise=True,true_future_feedback=mode=='real_history',
                true_future_leakage_into_own_step=False,raw_autoregressive_feedback=mode=='autoregressive',
                attribution_only=mode=='real_history',model_batch_calls=len(spec['batches'])*16,
                physical_steps=0,wall_seconds=time.monotonic()-begin)
            write(folder/'metrics.json',payload);results[mode]=dict(metrics=str((folder/'metrics.json').resolve()),raw_files=files)
    finally:ctx.model.train(oldmode)
    return results

def fixed_loss(root,config,ctx,epoch):
    import torch
    from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,make_conditioning_plan,conditioning_residual_loss
    ar1=load_ar1();root=Path(root);spec=read(root/'loss-spec.json');path=root/'diagnostics'/f'epoch{epoch}'/'fixed-loss.json'
    if path.exists():raise RuntimeError('Fixed loss already recorded; no silent repetition')
    rng=make_conditioning_rngs(**spec['seeds'],device=next(ctx.model.parameters()).device)
    rows=[];total=defaultdict(lambda:dict(sum=0.,samples=0));oldmode=ctx.model.training;ctx.model.eval();begin=time.monotonic()
    try:
        with torch.no_grad():
            for bi,b in enumerate(spec['batches']):
                arrays=ctx.store.batch(b['rows']);batch={k:torch.as_tensor(v,device='cuda') for k,v in arrays.items()}
                plan=make_conditioning_plan(batch,rng)
                with torch.autocast('cuda',dtype=torch.float16,enabled=True):loss,details=conditioning_residual_loss(ctx.model,batch,noise_plan=plan)
                n=len(b['rows']);split=b['rows'][0]['split'];letter=b['rows'][0]['letter']
                if any(r['split']!=split or r['letter']!=letter for r in b['rows']):raise ValueError('Fixed loss batches must be split/letter homogeneous')
                for key in (split,split+':'+letter):total[key]['sum']+=float(loss)*n;total[key]['samples']+=n
                rows.append(dict(batch=bi,sample_ids=[r['sample_id'] for r in b['rows']],split=split,letter=letter,
                    loss=float(loss),samples=n,history_branch=details['history_branch'],noise_hashes=plan.tensor_hashes(),
                    rng_before=plan.rng_before,rng_after=plan.rng_after))
        noise_identity=digest([{k:v for k,v in r.items() if k not in ('loss',)} for r in rows])
        if epoch>1:
            previous=read(root/'diagnostics/epoch1/fixed-loss.json')
            if previous['noise_and_sample_identity']!=noise_identity:raise RuntimeError('Fixed residual sample/noise changed between checkpoints')
        result=dict(epoch=epoch,batches=rows,means={k:dict(mean=v['sum']/v['samples'],samples=v['samples']) for k,v in total.items()},
            noise_and_sample_identity=noise_identity,training_generators_used=False,seeds=spec['seeds'],
            objective=spec['objective'],physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-begin)
        write(path,result);return str(path.resolve())
    finally:ctx.model.train(oldmode)

@contextmanager
def inject_first_noise(torch,*,batch_size,slot,shape=(3,96,96),sink=None):
    """Reuse original batch slot without forwarding unapproved sibling cases."""
    original=torch.randn;calls=[0]
    def replacement(*args,**kwargs):
        requested=tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list)) else tuple(args)
        if requested!=(1,*shape):raise RuntimeError('Unexpected sampler random tensor request')
        if calls[0]:raise RuntimeError('Unexpected additional sampler noise; churn/conditioning must remain0')
        calls[0]+=1
        full=original(int(batch_size),*shape,**kwargs);selected=full[int(slot):int(slot)+1]
        if len(selected)!=1:raise RuntimeError('Invalid original batch noise slot')
        if sink is not None:sink(selected)
        return selected
    torch.randn=replacement
    try:
        yield
        if calls[0]!=1:raise RuntimeError('Native sampler did not consume its single initial noise draw')
    finally:torch.randn=original

def denoise16(root,config,ctx):
    import torch
    ar1=load_ar1();root=Path(root);spec=read(root/'diagnostic-spec.json')
    folder=root/'diagnostics/epoch1/denoise16';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('16-step diagnostic already recorded')
    calibration=read(REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    settings=dict(SAMPLER,num_steps_denoising=16);sampler=ctx.source.DiffusionSampler(ctx.model,ctx.source.DiffusionSamplerConfig(**settings))
    device=next(ctx.model.parameters()).device;device_index=device.index if device.index is not None else torch.cuda.current_device()
    files=[];scored=[];reuse=[];begin=time.monotonic();oldmode=ctx.model.training;ctx.model.eval()
    try:
        for ci,case in enumerate(spec['denoise16_cases']):
            row=case['row'];d=ctx.store.arrays(row);a=int(row['anchor']);noise_hashes=[]
            def sample(history,actions,*,seed):
                def sink(x):
                    array=x.detach().float().cpu().numpy();noise_hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
                with torch.random.fork_rng(devices=[device_index]),torch.inference_mode():
                    torch.random.default_generator.manual_seed(seed);torch.cuda.default_generators[device_index].manual_seed(seed)
                    h=torch.as_tensor(history.copy(),device=device);act=torch.as_tensor(actions.copy(),device=device)
                    with inject_first_noise(torch,batch_size=case['original_batch_size'],slot=case['original_slot'],sink=sink):
                        result,_=sampler.sample(h,act)
                    return result.detach().float().cpu().numpy()
            raw=ar1.rollout_raw(sample,d['rgb'][None,a-3:a+1].astype(np.float32)/np.float32(255),
                                d['passed_actions'][None,a-3:a+16],seeds=case['seeds'])[0]
            path=folder/f'raw-{ci:02d}.npz';ar1.raw_npz(path,config,root,raw=raw)
            record=dict(**row,path=str(path.resolve()),sha256=sha(path),seeds=case['seeds'],
                original_batch_index=case['original_batch_index'],original_batch_size=case['original_batch_size'],
                original_slot=case['original_slot'],initial_noise_sha256=noise_hashes)
            files.append(record);scored+=_score_sequence(ar1,raw,row,ctx.store,calibration,'denoise16')
            reuse.append(dict(row=row,original3=case['reused3'],initial_seed_and_slot=record))
            write(folder/'progress.json',dict(completed_cases=ci+1,total_cases=8,extra_sibling_model_calls=0))
        result=dict(complete=True,epoch=1,sampler=settings,rows=scored,raw_files=files,reused3_refs=reuse,
            approved_cases=8,extra_sibling_model_calls=0,sampler_calls=8*16,denoiser_forwards=8*16*16,
            initial_noise_pairing='For each original seed draw original Bx3x96x96 then inject original slot into B1 sampler; source math/config otherwise unchanged.',
            numerical_limit='Original3 epsilon was not separately saved: it is reconstructed from frozen seed/batch-shape/slot and unchanged Torch execution contract. Model B1 versus original B4 can have floating kernel rounding differences; this is not a bitwise guarantee of all network intermediates, so not every output difference can be attributed exclusively to denoising count.',
            physical_steps=0,optimizer_updates=0,wall_seconds=time.monotonic()-begin)
        write(folder/'metrics.json',result);return dict(metrics=str((folder/'metrics.json').resolve()),raw_files=files,reused3_refs=reuse)
    finally:ctx.model.train(oldmode)

def run_epoch(root,config,ctx,epoch):
    """Engine owns full training RNG/mode snapshot+restore around this callback."""
    if epoch not in (1,2,3):raise ValueError('Only approved checkpoints1/2/3')
    result=dict(epoch=epoch,fixed_loss=fixed_loss(root,config,ctx,epoch))
    if epoch in (1,3):result.update(evaluate_modes(root,config,ctx,epoch))
    if epoch==1:result['denoise16']=denoise16(root,config,ctx)
    write(Path(root)/'diagnostics'/f'epoch{epoch}'/'result.json',result)
    return result
