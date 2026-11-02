"""Frozen high-motion fitting evaluation. Import/freeze are saved-only.

Only the unique worker calls run_checkpoint. Native3 sampler, raw float feedback,
unchanged actions, explicit batch slots and observed/replayed noise are retained.
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
from scipy import ndimage

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
CD=ROOT.parent/'region-weighted-cd-20261101'
CAL=ROOT.parent/'native-rgb-20261028/frozen-renderer-calibration.json'
SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.)
HORIZONS=(1,4,8,16)
_HELPER=None

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    from common import atomic_json
    atomic_json(p,v)
def immutable(p,v):
    if Path(p).exists():
        if read(p)!=v:raise ValueError('Frozen evaluation object would change: '+str(p))
    else:write(p,v)
def helper():
    global _HELPER
    if _HELPER is None:
        spec=importlib.util.spec_from_file_location('_hm_saved_CD_eval',CD/'evaluation.py')
        value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value);_HELPER=value
    return _HELPER
def palette(image,cal):
    from tdyssey.native_rgb_evaluation import palette_masks
    return palette_masks(image,cal)
def stable_seed(key,base=2026110107):return (int(hashlib.sha256(key.encode()).hexdigest()[:8],16)+base)%2**32

def standard():
    return dict(kind='high-motion-fit-visible-standard.v1',frozen_before_predictions=True,
        H1_strong_IoU_min=.95,H1_translation_position_max_imagepx=.3,copy_error_reduction_min=.5,
        AR_IoU_min=.90,AR_position_max_imagepx=1.,case_IoU_floor=.85,case_fraction_min=.9,
        blue_position_mean_max_imagepx=1.,angle_error_max_deg=5.,axis_ratio_min=1.2,
        true_axis_calibration_max_deg=5.,axis_jump_max_deg=45.,dual_axis_agreement_max_deg=10.,
        visible_rotation_guard_deg=5.,hole_min_pixels=2,
        strong_definition='true image-centroid displacement>=1px OR abs saved physical rotation>=3deg',
        dominant_definition='strong target: translation/1 >= abs(rotation)/3 => translation; otherwise rotation. Threshold-normalized classification, not a physical unit equivalence.',
        shape_definition='Original frozen renderer-calibrated visible topology/area/aligned-contour tolerances; independent same-frame hole/opening/components and blue checks. Occlusion/clipping unavailable remains unknown, never pass.',
        angle_definition='PCA of gray and visible white convex-deficit regions, modulo180 and time-unwrapped; each real sequence calibrated against saved true angles first. Independent RH compares previous REAL frame to one prediction; AR compares its own trajectory. A visible-axis appendix, not a rigid-pose certificate.',
        uncertainty='Unknown does not pass. Nonrotating targets have direction not_required, not false correctness. Existing formal numeric angular registrations do not override this diagnostic.',
        extra_noise_seeds=[2026110171,2026110172],primary_noise_seed=2026110170,
        primary_failure_layout='Three old failure cases are a new fixed B3 batch in this experiment; no numerical claim of pairing with historical CD B4 outputs.')

def features(image,cal):
    """Two image-derived axes; no poses/letter IDs enter model inputs."""
    import cv2
    from tdyssey.native_rgb_evaluation import _topology
    gray,blue=palette(image,cal);y,x=np.nonzero(gray);hull=np.zeros_like(gray,dtype=np.uint8)
    if len(x)>=3:cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((x,y)).astype(np.int32)),1)
    void=hull.astype(bool)&~gray&~blue
    def axis(mask):
        yy,xx=np.nonzero(mask)
        if len(xx)<3:return None
        e,v=np.linalg.eigh(np.cov(np.column_stack((xx,yy)),rowvar=False,bias=True))
        return dict(angle=float(np.degrees(np.arctan2(v[1,-1],v[0,-1]))),ratio=float(e[-1]/max(e[0],1e-10)))
    topology=_topology(gray,cal['minimum_component_pixels'])
    # Enclosed white regions use the same background4 connectivity as old score.
    labels,n=ndimage.label(~np.pad(gray,1),structure=ndimage.generate_binary_structure(2,1));sizes=np.bincount(labels.ravel());outside=labels[0,0]
    holes=[labels[1:-1,1:-1]==i for i in range(1,n+1) if i!=outside and sizes[i]>=2]
    enclosed=np.logical_or.reduce(holes) if holes else np.zeros_like(gray)
    cropped=bool(gray[0].any() or gray[-1].any() or gray[:,0].any() or gray[:,-1].any())
    return dict(gray=axis(gray),void=axis(void),gray_area=int(gray.sum()),blue_area=int(blue.sum()),
        holes=topology['holes'],components=topology['components'],tiny_fragment_pixels=topology['tiny_fragment_pixels'],
        blue_in_enclosed_region=bool((enclosed&blue).any()),cropped=cropped,
        possible_gray_occlusion=bool((ndimage.binary_dilation(gray)&blue).any()),
        white_deficit_pixels=int(void.sum()))

def axis_delta(first,second):return float((second-first+90)%180-90)
def pair_axes(a,b,p):
    values={};reasons=[]
    for key in ('gray','void'):
        aa,bb=a[key],b[key];values[key]=None if aa is None or bb is None else axis_delta(aa['angle'],bb['angle'])
        if aa is None or bb is None:reasons.append(key+'_axis_missing')
        elif min(aa['ratio'],bb['ratio'])<p['axis_ratio_min']:reasons.append(key+'_near_circular')
        elif abs(values[key])>p['axis_jump_max_deg']:reasons.append(key+'_axis_jump')
    if a['cropped'] or b['cropped']:reasons.append('viewport_crop')
    if a['blue_in_enclosed_region'] or b['blue_in_enclosed_region']:reasons.append('blue_in_hole')
    if a['components']!=1 or b['components']!=1:reasons.append('gray_fragment_or_missing')
    return values,reasons

def rotation_assessment(truth_features,recorded_deg,pred_features,mode,step,p):
    """RH never accumulates independent outputs into a predicted trajectory."""
    k=step-1;local=mode=='real_history';expected=float(recorded_deg[k+1]-recorded_deg[k] if local else recorded_deg[k+1])
    truth_pairs=[pair_axes(truth_features[i],truth_features[i+1],p) for i in range(k+1)]
    if local:
        gt,gt_reasons=truth_pairs[-1];prediction,pred_reasons=pair_axes(truth_features[k],pred_features[k],p)
        calibration={q:None if gt[q] is None else abs(gt[q]-expected) for q in gt}
    else:
        prediction={q:0. for q in ('gray','void')};gt={q:0. for q in prediction};gt_reasons=[];pred_reasons=[];calibration={q:0. for q in prediction}
        for i,(td,tr) in enumerate(truth_pairs):
            pd,pr=pair_axes(truth_features[0] if i==0 else pred_features[i-1],pred_features[i],p)
            gt_reasons+=tr;pred_reasons+=pr
            for q in prediction:
                gt[q]=None if gt[q] is None or td[q] is None else gt[q]+td[q]
                prediction[q]=None if prediction[q] is None or pd[q] is None else prediction[q]+pd[q]
                calibration[q]=None if gt[q] is None or calibration[q] is None else max(calibration[q],abs(gt[q]-float(recorded_deg[i+1])))
    reasons=gt_reasons+pred_reasons
    for q,value in calibration.items():
        if value is None or value>p['true_axis_calibration_max_deg']:reasons.append(q+'_GT_calibration_failed')
    values=list(prediction.values())
    if all(v is not None for v in values) and abs(values[0]-values[1])>p['dual_axis_agreement_max_deg']:reasons.append('dual_axis_disagreement')
    required=abs(expected)>=3.
    if required and abs(expected)<p['visible_rotation_guard_deg']:reasons.append('true_rotation_below_visible_guard')
    if required and any(gt[q] is None or gt[q]*np.sign(expected)<p['visible_rotation_guard_deg'] for q in gt):
        reasons.append('GT_visible_response_not_resolved')
    error=None if any(v is None for v in values) else max(abs(v-expected) for v in values)
    status='not_required' if not required else 'unknown' if reasons else 'pass' if all(v*np.sign(expected)>=p['visible_rotation_guard_deg'] for v in values) and error<=p['angle_error_max_deg'] else 'fail'
    classification='unknown'
    if not reasons and required:
        classification='correct' if all(v*np.sign(expected)>=p['visible_rotation_guard_deg'] for v in values) else 'opposite' if all(v*np.sign(expected)<=-p['visible_rotation_guard_deg'] for v in values) else 'no_resolved_rotation'
    return dict(status=status,required=required,classification=classification,truth_signed_rotation_deg=expected,
        visible_signed_rotation_deg=prediction,max_visible_angle_error_deg=error,GT_calibration_error_deg=calibration,
        unknown_reasons=sorted(set(reasons)),input_mode=mode,physical_rigid_pose_proven=False)

def _motion(row,data):
    from tdyssey.envs.official import official_shape
    a=int(row['anchor']);states=data['states'][a:a+17];local=np.asarray(official_shape(row['letter']).centroid,np.float64)
    co,si=np.cos(states[:,4]),np.sin(states[:,4]);xy=states[:,2:4]+np.column_stack((co*local[0]-si*local[1],si*local[0]+co*local[1]))
    turns=np.degrees(np.unwrap(states[:,4])-states[0,4]);out=[]
    for k in range(16):
        t=float(np.linalg.norm(xy[k+1]-xy[k])*96/512);rot=float(turns[k+1]-turns[k]);strong=t>=1 or abs(rot)>=3
        out.append(dict(step=k+1,translation_imagepx=t,signed_rotation_deg=rot,strong=bool(strong),dominance='transition' if not strong else 'translation' if t>=abs(rot)/3 else 'rotation'))
    return out,turns.tolist()

def freeze(root,c):
    root=Path(root);index_path=root/'selection/index.json';index=read(index_path);std=standard()
    groups=[('train',index['train48']),('seen_val',index['seen_val24']),('original_failure',index['failures3'])]
    panel={r.get('sample_id',r.get('window_id',r.get('segment_id'))) for r in index['fixed_train_panel12']}
    old=helper().old();store=old._ar1().RawStore(max_bytes=48*1024**2);batches=[];rows=[];cal=read(CAL);GT=[];paths=[index_path,CAL,CD/'evaluation.py',CD/'direction.py',ROOT.parent/'generated-history-ab-20261101/evaluation.py',ROOT.parent/'accelerated-rgb-run-20261031/train_eval.py',REPO/'tdyssey/envs/official.py',REPO/'tdyssey/native_rgb_evaluation.py',REPO/'scripts/native_rgb_stage3_score.py']
    for role,items in groups:
        expected=48 if role=='train' else 24 if role=='seen_val' else 3
        if len(items)!=expected:raise ValueError('Approved evaluation case denominator changed')
        # Selection has already frozen stable order, with its chosen panel first.
        # Preserve that order and its slots rather than sorting the cases again.
        ordered=items
        normalized=[]
        for item in ordered:
            row=deepcopy(item);row['sample_id']=row.get('sample_id',row.get('window_id',row.get('segment_id')));row['selection_role']=role
            if not row['sample_id']:raise ValueError('Stable case identity missing')
            d=store.arrays(row);a=int(row['anchor'])
            if a<3 or a+16>=len(d['rgb']) or not d['control_valid'][a-3:a+16].all():raise ValueError('Incomplete/unclean evaluation trajectory')
            if (d['passed_actions'][a-3:a+16]<0).any() or (d['passed_actions'][a-3:a+16]>511).any():raise ValueError('Model contract exceeded')
            motion,angles=_motion(row,d);row['target_motion']=motion;row['recorded_rotation_deg']=angles
            f=[features(im,cal) for im in d['rgb'][a:a+17]]
            # Ground-truth self-check across actual saved rotations, not a model output.
            checks=[rotation_assessment(f,angles,f[1:],'real_history',h,std) for h in range(1,17)]
            checks_ar=[rotation_assessment(f,angles,f[1:],'autoregressive',h,std) for h in range(1,17)]
            GT.append(dict(sample_id=row['sample_id'],letter=row['letter'],truth_features=f,recorded_rotation_deg=angles,
                local_GT_to_GT=checks,cumulative_GT_to_GT=checks_ar,model_outputs_used=False))
            normalized.append(row);rows.append(row);paths.append(Path(row['raw_path']))
            meta=Path(row['raw_path']).with_suffix('.json')
            if meta.exists():paths.append(meta)
        # Each role/letter is packed separately. Failure case3 is deliberately B3.
        chunks=[normalized] if role=='original_failure' else [[x for x in normalized if x['letter']==l] for l in ('O','C','B')]
        for chunk in chunks:
            for begin in range(0,len(chunk),4):
                rr=chunk[begin:begin+4];bid=len(batches)
                batches.append(dict(batch_id=bid,selection_role=role,rows=rr,seeds=[stable_seed(f'{bid}:{h}',std['primary_noise_seed']) for h in range(16)],
                    extra_seeds=[[stable_seed(f'{bid}:{h}',seed) for h in range(16)] for seed in std['extra_noise_seeds']]))
    # Panel exactly comprises three entire training B4 groups; never repack.
    panel_batches=[b['batch_id'] for b in batches if b['selection_role']=='train' and all(r['sample_id'] in panel for r in b['rows'])]
    selected={r['sample_id'] for b in batches if b['batch_id'] in panel_batches for r in b['rows']}
    if selected!=panel or len(selected)!=12 or len(panel_batches)!=3:raise ValueError('Fixed panel must preserve three original B4 batches')
    if len({r['family_id'] for r in rows})!=75:raise ValueError('Evaluation family identities overlap roles')
    spec=dict(kind='high-motion-fit-evaluation-spec.v1',rows=rows,batches=batches,panel_batch_ids=panel_batches,
        train_batch_ids=[b['batch_id'] for b in batches if b['selection_role']=='train'],sampler=SAMPLER,
        full_cases=75,train_cases=48,seen_val_cases=24,original_failures=3,fixed_panel_cases=12,
        selection_index_sha256=sha(index_path),primary_noise_seed=std['primary_noise_seed'],extra_noise_seeds=std['extra_noise_seeds'],
        batch_layout='train48=12B4; seen_val24=6B4; failures3=1B3. Panel3 original B4 retained unchanged.',
        historical_failure_comparison='New B3 noise/layout paired only within this experiment; old CD B4 outputs are historical, never substituted as this baseline.',
        scopes=dict(full=75,panel=12,train_full=48))
    immutable(root/'evaluation-spec.json',spec);immutable(root/'fit-standard.json',std)
    immutable(root/'GT-measurement-calibration.json',dict(kind='saved-GT-to-GT-calibration.v1',palette_sha256=sha(CAL),standard_sha256=sha(root/'fit-standard.json'),
        rows=GT,physics_steps=0,model_calls=0,shape_standard='Existing renderer GT-to-GT frozen calibration preserved, no new rendering. Same-frame visible masks and actual occlusion recorded above.'))
    paths +=[root/'evaluation-spec.json',root/'fit-standard.json',root/'GT-measurement-calibration.json'];store.cache.clear()
    paths +=[root/n for n in ('prelaunch-GT-calibration-v1.json','GT-calibration-correction-receipt.json') if (root/n).exists()]
    return dict(spec=str((root/'evaluation-spec.json').resolve()),standard=str((root/'fit-standard.json').resolve()),GT_calibration=str((root/'GT-measurement-calibration.json').resolve()),
        bindings=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths))])

def _scored(row,prediction,truth,last,pred_features,truth_features,step,mode,cal,std):
    m=helper().old()._ar1()._metric(prediction,truth,last,row['letter'],cal)
    rot=rotation_assessment(truth_features,row['recorded_rotation_deg'],pred_features,mode,step,std)
    pf,tf=pred_features[step-1],truth_features[step]
    topology_unknown=pf['cropped'] or tf['cropped'] or pf['blue_in_enclosed_region'] or tf['blue_in_enclosed_region']
    topology_pass=pf['holes']==tf['holes'] and pf['components']==tf['components'] and pf['tiny_fragment_pixels']<=tf['tiny_fragment_pixels']
    topology='unknown' if topology_unknown else 'pass' if topology_pass else 'fail'
    shape='unknown' if m.get('legacy_shape_deformation_failure') is None else 'fail' if m['legacy_shape_deformation_failure'] else 'pass'
    if topology=='fail':shape='fail'
    elif topology=='unknown' and shape!='fail':shape='unknown'
    motion=row['target_motion'][step-1]
    return dict(sample_id=row['sample_id'],family_id=row['family_id'],letter=row['letter'],split=row['split'],selection_role=row['selection_role'],
        anchor=row['anchor'],horizon=step,input_mode=mode,strong_H1=motion['strong'],dominance=motion['dominance'],true_H1_motion=motion,
        metrics=m,rotation=rot,visible_shape_status=shape,visible_topology_status=topology,
        B_holes_status=topology if row['letter']=='B' else 'not_applicable',C_opening_status=topology if row['letter']=='C' else 'not_applicable',
        precise_corner_or_rigid_geometry_proven=False)

def _stats(values):
    good=np.array([float(v) for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(values),available=len(good),unknown=len(values)-len(good),mean=float(good.mean()) if len(good) else None,
        median=float(np.median(good)) if len(good) else None,P10=float(np.percentile(good,10)) if len(good) else None,
        P90=float(np.percentile(good,90)) if len(good) else None,worst=float(good.max()) if len(good) else None)

def assess(rows,copy_rows,std,role='train'):
    """Strict PLAN8 checks; unknowns and missing dominance groups do not pass."""
    rr=[x for x in rows if x['selection_role']==role];copies={(x['sample_id'],x['input_mode'],x['horizon']):x for x in copy_rows};signals=[]
    def signal(letter,name,values,comparison,threshold):
        stat=_stats(values);v=stat['mean'];passed=None if v is None or stat['unknown'] else v>=threshold if comparison=='>=' else v<=threshold
        signals.append(dict(letter=letter,name=name,comparison=comparison,threshold=threshold,stats=stat,passed=passed))
    for letter in ('O','C','B'):
        strong=[x for x in rr if x['letter']==letter and x['input_mode']=='real_history' and x['strong_H1']]
        signal(letter,'H1_strong_IoU',[x['metrics']['raw_mask_iou'] for x in strong],'>=',std['H1_strong_IoU_min'])
        for dominance,key in [('translation','object_position_error_image_px'),('rotation','raw_contour_error_image_px')]:
            selected=[x for x in strong if x['dominance']==dominance]
            if dominance=='translation':signal(letter,'H1_translation_position',[x['metrics'][key] for x in selected],'<=',std['H1_translation_position_max_imagepx'])
            now=_stats([x['metrics'][key] for x in selected]);baseline=_stats([copies[(x['sample_id'],x['input_mode'],x['horizon'])]['metrics'][key] for x in selected])
            reduction=None if not selected or now['unknown'] or baseline['unknown'] or not baseline['mean'] else 1-now['mean']/baseline['mean']
            signals.append(dict(letter=letter,name='H1_'+dominance+'_copy_reduction',comparison='>=',threshold=.5,value=reduction,n=len(selected),passed=None if reduction is None else reduction>=.5))
        for h in HORIZONS:
            selected=[x for x in rr if x['letter']==letter and x['input_mode']=='autoregressive' and x['horizon']==h]
            if h>1:
                signal(letter,f'AR{h}_IoU',[x['metrics']['raw_mask_iou'] for x in selected],'>=',std['AR_IoU_min'])
                signal(letter,f'AR{h}_position',[x['metrics']['object_position_error_image_px'] for x in selected],'<=',std['AR_position_max_imagepx'])
            known=all(x['metrics'].get('raw_mask_iou') is not None and np.isfinite(x['metrics']['raw_mask_iou']) for x in selected)
            fraction=None if not selected or not known else sum(x['metrics']['raw_mask_iou']>=std['case_IoU_floor'] for x in selected)/len(selected)
            signals.append(dict(letter=letter,name=f'H{h}_case_fraction_IoU85',value=fraction,n=len(selected),
                descriptive_only=True,gate=False,passed=None,scope='Per-letter descriptive; PLAN90percent applies to all training cases combined.'))
        for mode in ('real_history','autoregressive'):
            selected=[x for x in rr if x['letter']==letter and x['input_mode']==mode]
            signal(letter,mode+'_blue_position',[x['metrics']['pusher_position_error_image_px'] for x in selected],'<=',std['blue_position_mean_max_imagepx'])
            signals[-1]['per_horizon']={str(h):_stats([x['metrics']['pusher_position_error_image_px'] for x in selected if x['horizon']==h]) for h in HORIZONS}
            for name,values in [('blue_present',['fail' if 'empty_pusher' in x['metrics'].get('failure_reasons',[]) else 'pass' for x in selected]),
                                ('visible_shape',[x['visible_shape_status'] for x in selected]),
                                ('topology',[x['visible_topology_status'] for x in selected]),
                                ('direction_and_angle',[x['rotation']['status'] for x in selected if x['rotation']['required']])]:
                counts=dict(Counter(values));passed=bool(values) and all(v=='pass' for v in values)
                signals.append(dict(letter=letter,name=mode+'_'+name,denominator=len(values),counts=counts,passed=passed,
                    unknown_not_pass=True))
    for h in HORIZONS:
        selected=[x for x in rr if x['input_mode']=='autoregressive' and x['horizon']==h]
        values=[x['metrics'].get('raw_mask_iou') for x in selected];known=all(v is not None and np.isfinite(v) for v in values)
        fraction=None if not selected or not known else sum(v>=std['case_IoU_floor'] for v in values)/len(values)
        signals.append(dict(letter='ALL_OCB',name=f'H{h}_case_fraction_IoU85',value=fraction,n=len(values),comparison='>=',threshold=.9,passed=None if fraction is None else fraction>=.9))
    gates=[x for x in signals if x.get('gate',True)]
    return dict(role=role,case_denominator=len({x['sample_id'] for x in rr}),signals=signals,all_passed=bool(rr) and all(x['passed'] is True for x in gates),
        unknown_signals=sum(x['passed'] is None or x.get('counts',{}).get('unknown',0)>0 for x in gates),
        scientific_scope='Training fitting only' if role=='train' else 'Original unseen-to-this-fit trajectories; not training fitting')

def _run_set(root,c,ctx,label,spec,batches,noise_set,store,cal,std):
    old=helper().old();folder=root/'evaluation'/label/noise_set;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():raise RuntimeError('No repeated checkpoint/noise evaluation')
    rows=[];copies=[];rawfiles=[];proofs=[];noiseproof=[];begin=time.monotonic()
    calibration={x['sample_id']:x for x in read(root/'GT-measurement-calibration.json')['rows']}
    for mode in ('real_history','autoregressive'):
        for batch in batches:
            bid=batch['batch_id'];rr=batch['rows'];seeds=batch['seeds'] if noise_set=='primary' else batch['extra_seeds'][int(noise_set[-1])-1]
            inputs=old.prepare_inputs(store,rr,mode);proof=old._inputs_evidence(rr,inputs,seeds);proofs.append(dict(mode=mode,batch_id=bid,**proof))
            predictions,noise=helper().native_rollout(ctx,inputs,seeds);noiseproof.append(dict(mode=mode,batch_id=bid,steps=noise))
            for slot,row in enumerate(rr):
                path=folder/f'{mode}-b{bid:02d}-s{slot}.npz';old._ar1().raw_npz(path,c,root,raw=predictions[slot])
                rawfiles.append(dict(**row,path=str(path.resolve()),sha256=sha(path),seeds=seeds,batch_id=bid,batch_slot=slot,input_mode=mode,
                    initial_epsilon_sha256=[d['per_case_initial_epsilon_sha256'][slot] for d in noise]))
                d=store.arrays(row);a=int(row['anchor']);truth=d['rgb'][a+1:a+17];tf=calibration[row['sample_id']]['truth_features'];pf=[features(im,cal) for im in predictions[slot]]
                for h in range(1,17):rows.append(_scored(row,predictions[slot,h-1],truth[h-1],d['rgb'][a],pf,tf,h,mode,cal,std))
                cache=root/'copy-reference'/(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:20]+'-'+mode+'.json')
                if cache.exists():
                    reference=read(cache)
                    if reference['sample_id']!=row['sample_id'] or reference['raw_sha256']!=row['raw_sha256'] or reference['standard_sha256']!=sha(root/'fit-standard.json'):raise ValueError('Copy-reference identity changed')
                    copies.extend(reference['rows'])
                else:
                    copyraw=d['rgb'][a:a+16] if mode=='real_history' else np.repeat(d['rgb'][a:a+1],16,axis=0)
                    cf=[features(im,cal) for im in copyraw]
                    cr=[_scored(row,copyraw[h-1],truth[h-1],d['rgb'][a],cf,tf,h,mode,cal,std) for h in range(1,17)]
                    write(cache,dict(sample_id=row['sample_id'],raw_sha256=row['raw_sha256'],standard_sha256=sha(root/'fit-standard.json'),rows=cr));copies.extend(cr)
            write(folder/'progress.json',dict(mode=mode,batch_id=bid,completed_raw_files=len(rawfiles),predicted_frames=len(rawfiles)*16))
    support=assess(rows,copies,std,'train');val=assess(rows,copies,std,'seen_val') if any(x['selection_role']=='seen_val' for x in rows) else None
    payload=dict(complete=True,label=label,noise_set=noise_set,rows=rows,copy_rows=copies,raw_files=rawfiles,input_evidence=proofs,noise_evidence=noiseproof,
        checkpoint_identity=deepcopy(ctx.checkpoint_identity),sampler=SAMPLER,training_fit_support=support,seen_val_support=val,
        horizons=list(range(1,17)),predicted_frames=len(rawfiles)*16,sampler_batch_calls=len(batches)*32,denoiser_batch_forwards=len(batches)*96,
        image_denoiser_forwards=len(rawfiles)*48,optimizer_updates=0,physical_steps=0,wall_seconds=time.monotonic()-begin,
        label_separation='All own-step targets excluded from input history. Real-history outputs are independent steps; AR feedback is original unmodified native float with no future truth.',
        limits=['No output shape repair, pose input or new physics. Copy reference receives real previous frame for RH, initial anchor only for AR.',
            'Old historical failures use this new experiment B3/noise; do not claim numerical pairing with CD original B4.',
            'Reliable image-axis diagnostics and visible shape tests are not a rigid collision/pose proof. Unknown never passes.'])
    write(folder/'metrics.json',payload)
    return dict(metrics=str((folder/'metrics.json').resolve()),support=support,predicted_frames=payload['predicted_frames'],sampler_batch_calls=payload['sampler_batch_calls'],wall_seconds=payload['wall_seconds'])

def run_checkpoint(root,c,ctx,label,scope='full',extra_seeds=False):
    root=Path(root);label=str(label);spec=read(root/'evaluation-spec.json');std=read(root/'fit-standard.json')
    if std!=standard() or spec['sampler']!=SAMPLER:raise ValueError('Frozen standards/sampler changed')
    if scope not in ('full','panel','train_full'):raise ValueError('Unapproved evaluation scope')
    if extra_seeds and scope!='full':raise ValueError('Extra noise belongs only to0/final full evaluation')
    ids=None if scope=='full' else set(spec['panel_batch_ids'] if scope=='panel' else spec['train_batch_ids'])
    batches=[b for b in spec['batches'] if ids is None or b['batch_id'] in ids]
    helper_old=helper().old();store=helper_old._ar1().RawStore(max_bytes=48*1024**2);cal=read(CAL)
    before=helper_old._model_sha(ctx.model);mode=ctx.model.training;ctx.model.eval();begin=time.monotonic()
    try:
        primary=_run_set(root,c,ctx,label,spec,batches,'primary',store,cal,std);extras=[]
        if extra_seeds:
            panel=[b for b in spec['batches'] if b['batch_id'] in spec['panel_batch_ids']]
            for number in (1,2):extras.append(_run_set(root,c,ctx,label,spec,panel,'extra'+str(number),store,cal,std))
        after=helper_old._model_sha(ctx.model)
        if before!=after:raise ValueError('Evaluation changed model state')
        value=read(primary['metrics']);panel_ids={x['sample_id'] for b in spec['batches'] if b['batch_id'] in spec['panel_batch_ids'] for x in b['rows']}
        fit_rows=[x for x in value['rows'] if x['sample_id'] in panel_ids and x['input_mode']=='autoregressive' and x['horizon'] in HORIZONS]
        fit_score=float(np.mean([1-x['metrics']['raw_mask_iou']+x['metrics']['object_position_error_image_px']/10 for x in fit_rows]))
        result=dict(label=label,scope=scope,primary=primary,metrics=primary['metrics'],extras=extras,
            panel_all_passed=primary['support']['all_passed'] if scope=='panel' else False,
            full_train_all_passed=primary['support']['all_passed'] if scope in ('full','train_full') else False,
            fit_score=fit_score,panel_fit_score=fit_score,fit_score_scope='Same fixed12 training-panel highlighted-AR mean(1-IoU + position/10), across every scope; ranking only, not an acceptance score.',
            full_train_unknown_count=primary['support']['unknown_signals'],
            successful_updates=int(getattr(ctx,'successful_updates',0)),exposure_count=int(getattr(ctx,'exposure_count',0)),
            actual_predicted_frames=primary['predicted_frames']+sum(x['predicted_frames'] for x in extras),
            sampler_batch_calls=primary['sampler_batch_calls']+sum(x['sampler_batch_calls'] for x in extras),
            model_state_sha256_before=before,model_state_sha256_after=after,model_state_unchanged=True,
            checkpoint_identity=deepcopy(ctx.checkpoint_identity),wall_seconds=time.monotonic()-begin,
            training_rng_preservation='Engine protected captures/restores global, supervision RNGs, explicit generators, model mode and progress; eval native seeds are scoped.',physical_steps=0,optimizer_updates=0)
        write(root/'evaluation'/label/'result.json',result);return result
    finally:ctx.model.train(mode);store.cache.clear()
