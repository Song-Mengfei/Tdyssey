"""Authorized 768-frame O/B attribution before C/D; no import-time model work.

The sole background worker supplies one model and a weights-only loader. Original
four complete B4 batches, native3 sampler and seed slots are retained verbatim.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
OLD=ROOT.parent/'generated-history-ab-20261101'
CAL=REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json'
CHECKPOINTS={
 'baseline':(ROOT.parent/'oc-continuation-20261031/training/epoch2.pt','c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187'),
 'A':(OLD/'training/A/final.pt','b4d7dd0fb58f535e7129c65bda025fdcfe2d09ce2c00835fab955d3c6cc5a2a5'),
 'B':(OLD/'training/B/final.pt','3b10eaf84f3bd13104a4667338621dee855efc85b0e3ce2cbbe06e93cca9efba')}
_EVAL=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    from common import atomic_json
    atomic_json(path,value)
def ev():
    global _EVAL
    if _EVAL is None:
        name='_cd_original_ghab_evaluation';spec=importlib.util.spec_from_file_location(name,OLD/'evaluation.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);_EVAL=module
    return _EVAL
def immutable(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError('Frozen diagnosis object would change: '+str(path))
    else:write(path,value)

def check_saved_timeline(row):
    """Contract evidence, not a claim that physics/model dynamics are correct."""
    path=Path(row['raw_path']);a=int(row['anchor'])
    if sha(path)!=row['raw_sha256']:raise ValueError('Frozen real RGB changed')
    with np.load(path,allow_pickle=False) as z:
        rgb=z['rgb'];actions=z['passed_actions'];requested=z['requested_actions']
        if rgb.dtype!=np.uint8 or rgb.shape[1:]!=(3,96,96) or len(rgb)!=len(actions)+1:raise ValueError('Native RGB/action timeline failed')
        if a<3 or a+16>=len(rgb) or not z['control_valid'][a-3:a+16].all():raise ValueError('Incomplete clean H16 timeline')
        if not np.array_equal(actions,requested):raise ValueError('Requested versus passed XY differs')
        if actions.shape!=(len(rgb)-1,2):raise ValueError('Action XY shape failed')
        window=actions[a-3:a+16]
        if not np.isfinite(window).all() or (window<0).any() or (window>511).any():raise ValueError('Unchanged model action range failed')
        for key in ('times','control_time_seconds'):
            if key not in z or not np.allclose(np.diff(z[key][a-3:a+17]),.1,rtol=0,atol=1e-10):raise ValueError('Real10Hz time alignment failed')
        if not np.array_equal(np.diff(z['frame_indices'][a-3:a+17]),np.ones(19,dtype=int)):raise ValueError('Noncontiguous frame index')
        if len(set(z['segment_ids'][a-3:a+17].tolist()))!=1:raise ValueError('Evaluation crosses a segment boundary')
        if not np.isfinite(z['states'][a-3:a+17]).all():raise ValueError('Nonfinite real state')
    meta_path=path.with_suffix('.json');meta=read(meta_path)
    if not meta['request_equals_passed'] or meta['family_id']!=row['family_id'] or meta['split']!='seen_val' or meta['letter']!=row['letter']:
        raise ValueError('Real raw/metadata label identity failed')
    return dict(sample_id=row['sample_id'],previous_frame_indices=[[a+k-3,a+k] for k in range(16)],
        target_frame_indices=list(range(a+1,a+17)),action_indices=[[a+k-3,a+k] for k in range(16)],
        action_axes='original X then Y; no swap/mirror/sign change',normalization='float32(a)/float32(255.5)-float32(1)',
        RGB_normalization='uint8 float32 / float32(255)',own_target_and_later_RGB_absent_from_own_history=True,
        known_future_actions_allowed=True,request_equals_passed=True,control_time_seconds=.1)

def freeze(root,c):
    """Only saved arrays/hash work, including thresholds before new inference."""
    import direction
    root=Path(root);source=read(OLD/'formal-spec.json');old=ev();old._spec_check(source,formal=True)
    chosen=[]
    for index,batch in enumerate(source['batches']):
        letters={r['letter'] for r in batch['rows']}
        if letters&{'O','B'}:
            if len(batch['rows'])!=4 or not letters.issubset({'O','B'}) or len(letters)!=1:raise ValueError('O/B cases are not complete original B4 groups')
            b=deepcopy(batch);b['original_batch_index']=index;chosen.append(b)
    rows=[r for b in chosen for r in b['rows']]
    if Counter(r['letter'] for r in rows)!=Counter(O=8,B=8) or [b['original_batch_index'] for b in chosen]!=[0,1,6,7]:raise ValueError('Original formal O/B denominator/batch layout changed')
    config_models=c.get('diagnostic_models',{})
    for label,(path,expected) in CHECKPOINTS.items():
        entry=config_models.get(label,dict(path=str(path),sha256=expected))
        if Path(entry['path']).resolve()!=path.resolve() or entry['sha256']!=expected or sha(path)!=expected:raise ValueError('Approved diagnostic checkpoint differs')
    paths=[OLD/'formal-spec.json',OLD/'evaluation.py',CAL,*[v[0] for v in CHECKPOINTS.values()]]
    store=old._ar1().RawStore(max_bytes=48*1024**2);proofs=[];old_inputs={}
    old_metrics={label:read(OLD/f'evaluation/{label}/formal/metrics.json') for label in CHECKPOINTS}
    for label in CHECKPOINTS:paths.append(OLD/f'evaluation/{label}/formal/metrics.json')
    for row in rows:
        proofs.append(check_saved_timeline(row));paths +=[Path(row['raw_path']),Path(row['raw_path']).with_suffix('.json')]
    for b in chosen:
        inputs=old.prepare_inputs(store,b['rows'],'real_history');proof=old._inputs_evidence(b['rows'],inputs,b['seeds'])
        for label in ('A','B'):
            prior=old_metrics[label]['input_evidence'][b['original_batch_index']]
            for key in ('sample_ids','family_ids','seeds','initial_history_sha256','absolute_actions_sha256','normalized_actions_sha256','shape_history','shape_actions'):
                if proof[key]!=prior[key]:raise ValueError('Original AR runtime input proof differs: '+key)
        for label,metrics in old_metrics.items():
            refs={x['sample_id']:x for x in metrics['raw_files']}
            for row in b['rows']:
                ref=refs[row['sample_id']]
                if ref['seeds']!=b['seeds'] or sha(ref['path'])!=ref['sha256']:raise ValueError('Original AR prediction/noise binding differs')
                paths.append(Path(ref['path']))
        old_inputs[str(b['original_batch_index'])]=proof
    protocol=direction.protocol()
    spec=dict(kind='cd-original-formal16-real-history-diagnosis.v1',rows=rows,batches=chosen,sampler=source['sampler'],
        original_formal_spec=str((OLD/'formal-spec.json').resolve()),original_formal_spec_sha256=sha(OLD/'formal-spec.json'),
        checkpoints={k:dict(path=str(p.resolve()),sha256=h) for k,(p,h) in CHECKPOINTS.items()},
        prediction_frames_per_checkpoint=256,total_prediction_frames=768,optimizer_updates=0,new_physical_steps=0,
        direction_protocol=protocol,original_initial_and_action_input_proofs=old_inputs,
        original_noise_layout='Four complete original B4 batches, same original indices0,1,6,7 and16perbatch seeds. No repacking or sibling predictions.',
        old_actual_epsilon_not_historically_saved=True,
        interface_gate='Stop only if actual input/action/timing/normalization/noise contract or engineering integrity fails; model wrong direction alone is not an interface fault.')
    immutable(root/'diagnosis-spec.json',spec);immutable(root/'direction-protocol.json',protocol)
    immutable(root/'diagnosis-input-contract.json',dict(passed=True,rows=proofs,original_batch_inputs=old_inputs,
        interface_scope='Saved data/runtime-input construction and sampler packaging; does not prove all physical contact dynamics correct.'))
    # Calibrate every true O sequence before new model output exists. No threshold tuning.
    truth_cal=[];cal=read(CAL)
    for row in rows:
        if row['letter']!='O':continue
        d=store.arrays(row);a=row['anchor'];truth=d['rgb'][a:a+17]
        vals=direction._o_label(truth,d['states'][a:a+17,4],truth[1:].astype(np.float32)/255.,'autoregressive',cal,protocol)
        truth_cal.append(dict(sample_id=row['sample_id'],frames=[dict(horizon=f['horizon'],
            local=f['local_direction']['truth_axis_calibration_error_deg'],cumulative=f['cumulative_direction']['truth_axis_calibration_error_deg']) for f in vals['frames']]))
    immutable(root/'direction-truth-calibration.json',dict(protocol_sha256=sha(root/'direction-protocol.json'),
        true_RGB_only=True,prediction_calls=0,rows=truth_cal,thresholds_not_fitted_to_new_predictions=True))
    store.cache.clear()
    paths +=[root/'diagnosis-spec.json',root/'direction-protocol.json',root/'diagnosis-input-contract.json',root/'direction-truth-calibration.json']
    bindings=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))]
    return dict(spec=str((root/'diagnosis-spec.json').resolve()),contract=str((root/'diagnosis-input-contract.json').resolve()),
        direction_protocol=str((root/'direction-protocol.json').resolve()),cases=16,predicted_frames=768,original_batch_size=4,bindings=bindings)

def tensor_bytes_sha(tensor):
    a=tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float32).tobytes()).hexdigest()

@contextmanager
def capture_initial_noise(torch,seeds,shape,records,*,replay=True):
    """Observe native randn without replacing epsilon; restore even on errors.

    Fresh independent Generator reconstruction consumes no training/default RNG.
    One draw per native sample is asserted (native3, churn0, s_cond0).
    """
    original=torch.randn;calls=0
    def observed(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs)
        if calls>=len(seeds) or tuple(result.shape)!=tuple(shape):raise ValueError('Unexpected native initial-noise draw/shape')
        seed=int(seeds[calls]);calls+=1
        value=dict(step=calls,original_seed=seed,batch_size=int(shape[0]),shape=list(result.shape),
            actual_initial_epsilon_sha256=tensor_bytes_sha(result),
            per_case_initial_epsilon_sha256=[tensor_bytes_sha(result[j]) for j in range(shape[0])],
            hash_definition='contiguous float32 raw bytes; before sigma scaling',native_return_unmodified=True)
        if replay:
            generator=torch.Generator(device=result.device).manual_seed(seed)
            regenerated=original(tuple(shape),device=result.device,dtype=result.dtype,generator=generator)
            value.update(independent_seed_and_original_B4_layout_reconstruction_equal=bool(torch.equal(result,regenerated)),
                reconstruction_epsilon_sha256=tensor_bytes_sha(regenerated))
            if not value['independent_seed_and_original_B4_layout_reconstruction_equal']:raise ValueError('Native epsilon differs from original seed/B4 reconstruction')
        records.append(value)
        return result
    torch.randn=observed
    try:
        yield
        if calls!=len(seeds):raise ValueError('Missing native initial-noise draws')
    finally:torch.randn=original

def _inference_model(root,c,ctx,label,spec,store):
    import torch
    helper=ev();ar1=helper._ar1();folder=root/'diagnosis'/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('Diagnostic inference cannot rerun completed checkpoint')
    checkpoint=spec['checkpoints'][label];ctx.load_checkpoint(checkpoint['path'],checkpoint['sha256'],label)
    if ctx.checkpoint_identity['sha256']!=checkpoint['sha256']:raise ValueError('Runtime checkpoint identity mismatch')
    ctx.model.eval();state_before=helper._model_sha(ctx.model);scored=[];raw_files=[];proofs=[];noise=[];parity=[]
    prior=read(OLD/f'evaluation/{label}/formal/metrics.json');prior_refs={r['sample_id']:r for r in prior['raw_files']}
    begin=time.monotonic();cal=read(CAL)
    for bi,batch in enumerate(spec['batches']):
        inputs=helper.prepare_inputs(store,batch['rows'],'real_history')
        proof=helper._inputs_evidence(batch['rows'],inputs,batch['seeds']);frozen=spec['original_initial_and_action_input_proofs'][str(batch['original_batch_index'])]
        if proof!=frozen:raise ValueError('Runtime real-history/action tensors differ from frozen contract')
        proof['original_batch_index']=batch['original_batch_index'];proofs.append(proof)
        draws=[]
        with capture_initial_noise(torch,batch['seeds'],(4,3,96,96),draws):
            predictions=helper.native_rollout(ctx,inputs,batch['seeds'])
        write(folder/'progress.json',dict(completed_batches=bi+1,total_batches=4,predicted_frames=(bi+1)*64,
            new_batch_predictions_completed_before_saved_scoring=True))
        noise.append(dict(original_batch_index=batch['original_batch_index'],sample_ids=[r['sample_id'] for r in batch['rows']],draws=draws))
        for slot,row in enumerate(batch['rows']):
            path=folder/f'raw-{bi:03d}-{slot}.npz';ar1.raw_npz(path,c,root,raw=predictions[slot])
            raw_files.append(dict(**row,path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,seeds=batch['seeds'],original_batch_index=batch['original_batch_index'],original_batch_slot=slot))
            with np.load(prior_refs[row['sample_id']]['path'],allow_pickle=False) as z:old_h1=z['raw'][0]
            error=float(np.max(np.abs(old_h1-predictions[slot,0])))
            parity.append(dict(sample_id=row['sample_id'],bitwise_equal=bool(np.array_equal(old_h1,predictions[slot,0])),
                max_absolute_difference=error,same_initial_history_action_model_seed_B4_slot=True,
                interpretation='Numerical parity is separate from input contract. Historical epsilon tensor was not archived; originalseed/B4 reconstruction and unchanged sampler are evidence.'))
            for h in range(1,17):scored.append(helper._scored_row(ar1,row,store,predictions[slot,h-1],h,label,'real_history',cal))
        write(folder/'progress.json',dict(completed_batches=bi+1,total_batches=4,predicted_frames=(bi+1)*64))
    after=helper._model_sha(ctx.model)
    if after!=state_before:raise ValueError('Zero-update diagnosis changed model weights')
    parity_summary=dict(cases=len(parity),bitwise_equal_cases=sum(r['bitwise_equal'] for r in parity),
        maximum_absolute_difference=max(r['max_absolute_difference'] for r in parity),
        nonidentical_cases=[r for r in parity if not r['bitwise_equal']],
        interpretation='Input/model/seed/B4 identities are checked separately. No tolerance or causal explanation is assigned to output discrepancies.')
    value=dict(complete=True,label=label,input_mode='real_history',rows=scored,raw_files=raw_files,input_evidence=proofs,
        sampler=helper.SAMPLER,model_identity=deepcopy(ctx.checkpoint_identity),initial_epsilon=noise,original_AR_H1_parity=parity,
        original_AR_H1_parity_summary=parity_summary,
        actual_predicted_frames=256,sampler_batch_calls=64,denoiser_batch_forwards=192,image_denoiser_forwards=768,
        optimizer_updates=0,physical_steps=0,model_sha_before=state_before,model_sha_after=after,model_unchanged=True,
        future_target_RGB_passed_to_model=False,true_history_targets_are_independent_single_steps=True,
        wall_seconds=time.monotonic()-begin,old_AR_actual_epsilon_historically_archived=False)
    write(folder/'metrics.json',value)
    return dict(metrics=str((folder/'metrics.json').resolve()),actual_predicted_frames=256,wall_seconds=value['wall_seconds'],H1_parity=parity_summary)

def _media(root,c,spec,metrics):
    """Every formal O/B case: true, three AR, three independent RH outputs."""
    import cv2
    import imageio_ffmpeg
    import os
    import subprocess
    from common import check_storage
    folder=root/'diagnosis/media';folder.mkdir(parents=True,exist_ok=True);records=[]
    refs={label:{r['sample_id']:r for r in read(path)['raw_files']} for label,path in metrics.items()}
    for row in spec['rows']:
        with np.load(row['raw_path'],allow_pickle=False) as z:
            a=row['anchor'];truth=z['rgb'][a+1:a+17].astype(np.float32)/255.
        columns=[('REAL TARGET',truth)]
        for label,rawrefs in refs.items():
            with np.load(rawrefs[row['sample_id']]['path'],allow_pickle=False) as z:columns.append((label,z['raw'].copy()))
        frames=[];width=len(columns)*192
        for k in range(16):
            frame=np.full((250,width,3),255,np.uint8)
            cv2.putText(frame,f'{row["family_id"]} a{a} | target +{k+1} ({(k+1)/10:.1f}s)',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(15,15,15),1)
            for j,(label,images) in enumerate(columns):
                cv2.putText(frame,label,(j*192+4,35),cv2.FONT_HERSHEY_SIMPLEX,.36,(15,15,15),1)
                im=np.rint(np.clip(images[k].transpose(1,2,0),0,1)*255).astype(np.uint8)
                frame[54:246,j*192:(j+1)*192]=cv2.resize(im,(192,192),interpolation=cv2.INTER_NEAREST)
            frames.append(frame)
        stem=row['family_id']+'-a'+str(a);board=folder/(stem+'-H1-H4-H8-H16.png')
        ok,buf=cv2.imencode('.png',cv2.cvtColor(np.concatenate([frames[h-1] for h in (1,4,8,16)],axis=0),cv2.COLOR_RGB2BGR))
        if not ok:raise RuntimeError('Diagnostic board encoding failed')
        check_storage(root,c,reserve_bytes=len(buf));board.write_bytes(buf.tobytes())
        allboard=folder/(stem+'-all16.png');ok,buf=cv2.imencode('.png',cv2.cvtColor(np.concatenate(frames,axis=0),cv2.COLOR_RGB2BGR))
        if not ok:raise RuntimeError('Diagnostic all16 encoding failed')
        check_storage(root,c,reserve_bytes=len(buf));allboard.write_bytes(buf.tobytes())
        video=folder/(stem+'-10fps.mp4');check_storage(root,c,reserve_bytes=4*1024**2)
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x250','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        proc=subprocess.run(cmd,input=b''.join(x.tobytes() for x in frames),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
        if proc.returncode:raise RuntimeError('Saved video encoder failure '+proc.stderr.decode(errors='replace')[-200:])
        capture=cv2.VideoCapture(str(video));count=0;fps=float(capture.get(cv2.CAP_PROP_FPS))
        while capture.read()[0]:count+=1
        capture.release()
        if count!=16 or abs(fps-10)>1e-3:raise ValueError('Saved video frame/fps verification failed')
        records.append(dict(sample_id=row['sample_id'],letter=row['letter'],board=str(board.resolve()),all16_board=str(allboard.resolve()),video=str(video.resolve()),fps=fps,frames=count,
            real_history_display_warning='RH columns are independent single-step outputs using real preceding history, not a generated physical trajectory.'))
    return records

def run(root,c,ctx):
    """Called once by the unique protected worker, before any C/D update."""
    import direction
    root=Path(root);begin=time.monotonic();spec=read(root/'diagnosis-spec.json');store=ev()._ar1().RawStore(max_bytes=48*1024**2)
    result=dict(completed=False,interface_contract_passed=False,interface_fault_proven=False,
        actual_predicted_frames=0,optimizer_updates=0,physical_steps=0,metrics_by_model={})
    try:
        if direction.protocol()!=read(root/'direction-protocol.json'):raise ValueError('Frozen direction protocol differs')
        for row in spec['rows']:check_saved_timeline(row)
        for label in ('baseline','A','B'):
            result['metrics_by_model'][label]=_inference_model(root,c,ctx,label,spec,store)
            result['actual_predicted_frames']+=256
            write(root/'diagnosis/runtime-progress.json',result)
        result['original_AR_H1_parity']={label:info['H1_parity'] for label,info in result['metrics_by_model'].items()}
        metrics={}
        for label in ('baseline','A','B'):
            metrics[label+'_AR16']=str((OLD/f'evaluation/{label}/formal/metrics.json').resolve())
        for label in ('baseline','A','B'):metrics[label+'_RH1']=result['metrics_by_model'][label]['metrics']
        analysis=direction.analyze_saved(spec,metrics,read(CAL));analysis_path=root/'diagnosis/direction-and-holes.json';write(analysis_path,analysis)
        result['diagnostic_highlights']=dict(O_display=[r for r in analysis['o_direction_rows'] if 'AC-O-01-08:' in r['sample_id']],
            B_display=[r for r in analysis['b_hole_rows'] if 'AC-B-00-08:' in r['sample_id']],support_signals=analysis['support_signals'])
        result.update(interface_contract_passed=True,interface_fault_proven=False,
            contract_scope='Same frozen originalXY/time/normalization/native sampler B4-seed slots. No demonstrated interface corruption. Does not prove correct learned contact dynamics.',
            original_actual_epsilon_not_historically_saved=True,new_initial_epsilon_observed_and_original_B4_reconstructed=True,
            model_failure_does_not_imply_interface_failure=True,analysis_path=str(analysis_path.resolve()))
        # Persist gate promptly; media is saved-only and cannot change its conclusion.
        write(root/'diagnosis/interface-gate.json',dict(interface_contract_passed=True,interface_fault_proven=False,actual_predicted_frames=768,analysis_path=result['analysis_path']))
        result['media']=_media(root,c,spec,metrics)
        lines=['# Same-case O/B diagnosis','',
            '768 independent true-history predicted targets completed: original16cases ×16targets ×3 checkpoints. Original AR16 reused. Zero optimizer updates and physical steps.',
            '','Interface contract passed: no demonstrated XY/sign/time/normalization/input-label corruption. A wrong model rotation is not evidence of an interface fault.',
            '','RH direction compares each previous REAL frame with its independent prediction. AR local and cumulative directions are separate. Old formal angular unknowns remain unchanged.',
            '','Historical AR epsilon tensors were not archived. New actual B4 epsilon and slots are hashed and checked against independent same-seed/B4 reconstruction; original sample order/action hashes match old A/B runtime records.',
            '','Initial-target output parity with original AR (independent evidence; numerical discrepancy is not silently called an action fault):','']
        for label,parity in result['original_AR_H1_parity'].items():
            lines.append('- '+label+': '+str(parity['bitwise_equal_cases'])+'/16 bitwise equal; max absolute difference '+repr(parity['maximum_absolute_difference'])+'; nonidentical IDs '+json.dumps([x['sample_id'] for x in parity['nonidentical_cases']]))
        lines += [
            '','Independent visible support (unknown never passes):','']
        for label,signals in analysis['support_signals'].items():lines.append('- '+label+': '+json.dumps(signals,ensure_ascii=False))
        lines+=['','All case counts/directions, local/cumulative tables, holes and uncertainty: '+str(analysis_path.resolve()),'','Saved comparisons (RH is not a physical rollout):','']
        for item in result['media']:lines.append('- '+item['sample_id']+' | '+item['board']+' | '+item['video'])
        report=root/'diagnosis/REPORT.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        result.update(completed=True,report_path=str(report.resolve()),automatic_continuation='Approved C/D may continue after engine restores complete common epoch2 state; no new approval required.')
    except Exception as error:
        result['actual_predicted_frames']=sum(read(p).get('predicted_frames',0) for p in (root/'diagnosis').glob('*/progress.json'))
        result.update(error=repr(error),failure_stage='diagnosis',interface_contract_passed=False,
            interface_fault_proven=False,engineering_failure_or_contract_violation=True,
            stop_training=True,limits='A failure of this check is not itself proof that physics/XY was wrong. Preserve the specific exception and all completed outputs; no automatic retry.')
        write(root/'diagnosis/failure.json',result)
    finally:
        store.cache.clear();result['wall_seconds']=time.monotonic()-begin
        write(root/'diagnosis/result.json',result);write(root/'diagnosis/summary.json',result);write(root/'diagnosis-result.json',result)
    return result
