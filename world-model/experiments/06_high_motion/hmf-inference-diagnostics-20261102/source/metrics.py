"""A/B diagnostic scoring, used only by the one BelowNormal worker.

No import-time scoring, hashing or model work. CUDA basic statistics preserve
the original palette and FP64 measurement formulas. CPU remaining fields retain
the frozen HMF contour/topology/axis/shape definitions. No model is called here.
"""
from __future__ import annotations
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
HMF=ROOT.parent/'high-motion-fit-20261101'
CAL=ROOT.parent/'native-rgb-20261028/frozen-renderer-calibration.json'
SCALE=512/96
DIAGONAL=96*math.sqrt(2)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
@lru_cache(maxsize=1)
def hmf():
    if str(REPO) not in sys.path:sys.path.append(str(REPO))
    spec=importlib.util.spec_from_file_location('_inference_diag_original_hmf',HMF/'evaluation.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
@lru_cache(maxsize=1)
def calibration():return read(CAL)
@lru_cache(maxsize=1)
def standard():return read(HMF/'fit-standard.json')

def _rgb64(a):
    a=np.asarray(a)
    if a.dtype==np.uint8:return a.astype(np.float64)/255.
    return a.astype(np.float64,copy=False)
def _uint8(td,key):
    name=key+'_uint8'
    if name in td:return np.asarray(td[name])
    value=np.asarray(td[key])
    if value.dtype==np.uint8:return value
    # Compatibility only: real RGB was stored as uint8. This recovers those
    # original bytes; it never quantizes predictions or changes model feedback.
    return np.rint(value*255.).astype(np.uint8)

def _numpy_basic(prediction,truth,last,cal):
    from tdyssey import native_rgb_evaluation as old
    p,t,previous=_rgb64(prediction),_rgb64(truth),_rgb64(last)
    gray,blue=old.palette_masks(prediction,cal);tg,tb=old.palette_masks(truth,cal);lg,lb=old.palette_masks(last,cal)
    def dist(a,b):
        x,y=old._centroid(a),old._centroid(b)
        return DIAGONAL if x is None or y is None else float(np.linalg.norm(x-y))
    return dict(gray=gray,blue=blue,target_gray=tg,target_blue=tb,last_gray=lg,last_blue=lb,
        raw_mask_iou=float(old._iou(gray,tg) or 0),object_position_error_image_px=dist(gray,tg),
        pusher_mask_iou=float(old._iou(blue,tb) or 0),pusher_position_error_image_px=dist(blue,tb),
        object_drift_from_last_error=1-float(old._iou(gray,lg) or 0),object_drift_from_last_image_px=dist(gray,lg),
        rgb_full_mae=float(np.abs(p-t).mean()),rgb_full_mse=float(((p-t)**2).mean()))

def gpu_basic(gpu,truth,last,cal):
    """One existing prediction tensor; no denoiser/model invocation.

    Palette complement is evaluated in float32 BEFORE multiplication by the
    float64 prototype, exactly as numpy palette_masks does. Truth photometric
    differences use original uint8/255 float64, not rounded float32 targets.
    """
    import torch
    device=gpu.device;shape=tuple(gpu.shape);p=gpu.reshape(-1,3,96,96).float()
    t=np.asarray(truth).reshape(-1,3,96,96);l=np.asarray(last).reshape(-1,3,96,96)
    def rgb32(a):return a.astype(np.float32)/np.float32(255) if a.dtype==np.uint8 else a.astype(np.float32,copy=False)
    tf=torch.as_tensor(rgb32(t),device=device);lf=torch.as_tensor(rgb32(l),device=device)
    prototype=torch.tensor(cal['object_rgb'],device=device,dtype=torch.float64);direction=1-prototype
    def masks(a):
        pixels=a.permute(0,2,3,1);alpha=((1-pixels).double()*direction).sum(-1)/(direction*direction).sum()
        residual=torch.linalg.vector_norm(pixels.double()-(1-alpha[...,None]*direction),dim=-1)
        gray=(alpha>=cal['palette']['minimum_alpha'])&(alpha<=cal['palette']['maximum_alpha'])&(residual<=cal['palette']['colour_residual'])
        red,green,blue=a.unbind(1)
        bm=(blue>.60)&(red<.45)&(green<.65)&((blue-torch.maximum(red,green))>.25)
        return gray,bm
    gray,blue=masks(p);tg,tb=masks(tf);lg,lb=masks(lf)
    yy,xx=torch.meshgrid(torch.arange(96,device=device,dtype=torch.float64),torch.arange(96,device=device,dtype=torch.float64),indexing='ij')
    def center(mask):
        n=mask.sum((-2,-1));den=n.clamp(min=1).double()
        return torch.stack(((mask*xx).sum((-2,-1))/den,(mask*yy).sum((-2,-1))/den),-1),n
    def distance(a,b):
        x,n=center(a);y,m=center(b);d=torch.linalg.vector_norm(x-y,dim=-1)
        return torch.where((n>0)&(m>0),d,torch.full_like(d,DIAGONAL))
    def iou(a,b):
        union=(a|b).sum((-2,-1));return (a&b).sum((-2,-1)).double()/union.clamp(min=1).double()
    target64=torch.as_tensor(_rgb64(t),device=device);diff=p.double()-target64
    values=dict(raw_mask_iou=iou(gray,tg),object_position_error_image_px=distance(gray,tg),
        pusher_mask_iou=iou(blue,tb),pusher_position_error_image_px=distance(blue,tb),
        object_drift_from_last_error=1-iou(gray,lg),object_drift_from_last_image_px=distance(gray,lg),
        rgb_full_mae=diff.abs().mean((1,2,3)),rgb_full_mse=(diff*diff).mean((1,2,3)))
    mask_values=dict(gray=gray,blue=blue,target_gray=tg,target_blue=tb,last_gray=lg,last_blue=lb)
    cpu_masks={k:v.cpu().numpy() for k,v in mask_values.items()};cpu_values={k:v.cpu().numpy() for k,v in values.items()}
    count=len(p)
    return [{**{k:v[i] for k,v in cpu_masks.items()},**{k:float(v[i]) for k,v in cpu_values.items()}} for i in range(count)]

def _features(gray,blue,topology):
    """Same HMF features, reusing the exact masks/topology already computed."""
    import cv2
    from scipy import ndimage
    y,x=np.nonzero(gray);hull=np.zeros_like(gray,dtype=np.uint8)
    if len(x)>=3:cv2.fillConvexPoly(hull,cv2.convexHull(np.column_stack((x,y)).astype(np.int32)),1)
    void=hull.astype(bool)&~gray&~blue
    def axis(mask):
        yy,xx=np.nonzero(mask)
        if len(xx)<3:return None
        e,v=np.linalg.eigh(np.cov(np.column_stack((xx,yy)),rowvar=False,bias=True))
        return dict(angle=float(np.degrees(np.arctan2(v[1,-1],v[0,-1]))),ratio=float(e[-1]/max(e[0],1e-10)))
    labels,n=ndimage.label(~np.pad(gray,1),structure=ndimage.generate_binary_structure(2,1));sizes=np.bincount(labels.ravel());outside=labels[0,0]
    holes=[labels[1:-1,1:-1]==i for i in range(1,n+1) if i!=outside and sizes[i]>=2]
    enclosed=np.logical_or.reduce(holes) if holes else np.zeros_like(gray)
    return dict(gray=axis(gray),void=axis(void),gray_area=int(gray.sum()),blue_area=int(blue.sum()),holes=topology['holes'],
        components=topology['components'],tiny_fragment_pixels=topology['tiny_fragment_pixels'],blue_in_enclosed_region=bool((enclosed&blue).any()),
        cropped=bool(gray[0].any() or gray[-1].any() or gray[:,0].any() or gray[:,-1].any()),
        possible_gray_occlusion=bool((ndimage.binary_dilation(gray)&blue).any()),white_deficit_pixels=int(void.sum()))

def remaining(prediction,truth,last,letter,basic,cal):
    """Exact original remaining score; skip only already-unqualified legacy fit."""
    from scipy import ndimage
    from tdyssey import native_rgb_evaluation as old
    gray,blue,tg,tb,lg,lb=(basic[k] for k in ('gray','blue','target_gray','target_blue','last_gray','last_blue'))
    topology=old._topology(gray,cal['minimum_component_pixels']);target_topology=old._topology(tg,cal['minimum_component_pixels'])
    contour=old._contour_distance(gray,tg)
    structure=np.ones((3,3),bool);p,t=_rgb64(prediction),_rgb64(truth)
    object_roi=ndimage.binary_dilation(lg|tg,structure=structure,iterations=2)&~ndimage.binary_dilation(lb|tb,structure=structure,iterations=1)
    blue_roi=ndimage.binary_dilation(lb|tb,structure=structure,iterations=2);combined=ndimage.binary_dilation(lg|tg|lb|tb,structure=structure,iterations=2)
    def mae(roi):return 1. if not roi.any() else float(np.abs(p-t)[:,roi].sum()/(3*roi.sum()))
    def border(mask):return bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    occluded=bool((ndimage.binary_dilation(gray)&blue).any() or (ndimage.binary_dilation(tg)&tb).any())
    clipping=border(gray) or border(tg);failures=[]
    if not gray.any():failures.append('empty_object')
    if not blue.any():failures.append('empty_pusher')
    m={k:v for k,v in basic.items() if not isinstance(v,np.ndarray)}
    m.update(hard_generation_failure=bool(failures),failure_reasons=failures,prediction_status='ok',truth_issue=None if tg.any() and tb.any() and lg.any() else 'empty_truth_object_pusher_or_last_object',
        raw_object_error=1-m['raw_mask_iou'],raw_contour_error_image_px=DIAGONAL if contour is None else contour/SCALE,
        raw_contour_error_world_px=DIAGONAL*SCALE if contour is None else contour,raw_contour_unavailable=contour is None,
        object_foreground_rgb_mae=mae(object_roi),pusher_foreground_rgb_mae=mae(blue_roi),combined_foreground_rgb_mae=mae(combined),
        object_roi_pixels=int(object_roi.sum()),pusher_roi_pixels=int(blue_roi.sum()),combined_roi_pixels=int(combined.sum()),pusher_error=1-m['pusher_mask_iou'],
        predicted_components=topology['components'],truth_components=target_topology['components'],predicted_holes=topology['holes'],truth_holes=target_topology['holes'],
        extra_fragments=max(0,topology['components']-target_topology['components']),missing_components=max(0,target_topology['components']-topology['components']),
        holes_difference=abs(topology['holes']-target_topology['holes']),tiny_fragment_pixels=topology['tiny_fragment_pixels'],
        area_fraction_error=float(abs(int(gray.sum())-int(tg.sum()))/max(1,int(tg.sum()))),
        prediction_object_border_touch=border(gray),truth_object_border_touch=border(tg),prediction_pusher_border_touch=border(blue),truth_pusher_border_touch=border(tb),
        object_outside_viewport_assessable=False,occlusion_ambiguous=occluded,legacy_shape_deformation_failure=None,legacy_shape_assessable=False,
        legacy_shape_unknown_reason=None,legacy_score_error=None,aligned_mask_iou=None,denominator_retained=True)
    for name in ('pusher_position_error','object_position_error','object_drift_from_last'):m[name+'_world_px']=m[name+'_image_px']*SCALE
    # These are the EXACT original score_rgb pre-registration exclusion rules.
    if occluded or clipping:
        m.update(legacy_shape_unknown_reason='contact_occlusion_or_viewport_clipping',legacy_numeric_status='not_computed_due_to_prior_unknown',
            legacy_numeric_skip_reason='Original score_rgb pre-registration pusher-adjacency/viewport exclusion; qualified shape remains unknown.')
    elif failures:
        m.update(legacy_shape_unknown_reason='legacy_score_exception',legacy_numeric_status='not_computed_due_to_prior_unknown',
            legacy_numeric_skip_reason='Original score_rgb rejects empty object/pusher; primary failure penalties retained.')
    else:
        # Qualified shape still uses the exact original CPU registration and
        # calibration; no alternate transform, tolerance or approximate score.
        previous=old.score_rgb(prediction,truth,cal,letter=letter)
        m.update(legacy=previous,aligned_mask_iou=previous['aligned_mask_iou'],legacy_shape_deformation_failure=previous['shape_deformation_failure'],
            legacy_shape_assessable=previous['shape_measurement_available'],legacy_numeric_status='original_CPU_reference_computed')
    m['motion_vector_error_image_px']=m['object_position_error_image_px'];m['visible_angle_error_deg']=m.get('legacy',{}).get('angle_error_deg')
    m['predicted_raw_corner_counts']=None;m['truth_raw_corner_counts']=None
    if letter in 'OC' and not failures:
        import cv2
        def corners(mask):
            cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            return [len(cv2.approxPolyDP(c,.75,True)) for c in cs if cv2.contourArea(c)>=2]
        m['predicted_raw_corner_counts']=corners(gray);m['truth_raw_corner_counts']=corners(tg)
    m['angle_scope']='Visible-mask angular alignment only; symmetry, occlusion, deformation can make it unavailable/ambiguous'
    return m,_features(gray,blue,topology)

def _center(mask):
    y,x=np.nonzero(mask);return None if len(x)==0 else np.array([x.mean(),y.mean()],np.float64)
def response_from_history(prediction,last,basic,pf,cal):
    """Visible response only; no counterfactual target or rigid-pose assertion."""
    gray,blue,lg,lb=(basic[k] for k in ('gray','blue','last_gray','last_blue'))
    def vector(a,b):
        x,y=_center(a),_center(b);return None if x is None or y is None else (x-y).tolist()
    return dict(object_visible_delta_xy_imagepx=vector(gray,lg),object_visible_motion_imagepx=basic['object_drift_from_last_image_px'],
        gray_mask_change_from_real_previous=basic['object_drift_from_last_error'],blue_visible_delta_xy_imagepx=vector(blue,lb),
        predicted_gray_pixels=int(gray.sum()),predicted_blue_pixels=int(blue.sum()),visible_axes=pf,
        object_motion_scope='Visible gray centroid, not physical body pose. Occlusion or shape change can move it.',physical_correctness_proven=False)

def score_batch(root,c,batch_record,raw_arrays,*,gpu=None,truth_arrays=None):
    """Called once per counted batch; no raw archives are written or hashed."""
    if truth_arrays is None:raise ValueError('Pass shared data.truth_arrays; scorer does not independently reread raw archives')
    raw=np.asarray(raw_arrays);rr=batch_record['rows'];section=batch_record['section'];mode=batch_record.get('mode','real_history')
    if section=='B':mode='real_history'
    if raw.dtype!=np.float32 or raw.ndim!=5 or raw.shape[0]!=len(rr) or raw.shape[2:]!=(3,96,96):raise ValueError('Native prediction tensor contract')
    if raw.shape[1]!=(16 if section=='A' else 1):raise ValueError('Unapproved prediction time denominator')
    cal=calibration();std=standard();begin=time.perf_counter();targets=[];lasts=[]
    action=batch_record.get('action_type','recorded');has_GT=section=='A' or action=='recorded'
    for td in truth_arrays:
        previous=_uint8(td,'initial_last');lasts.append(np.repeat(previous[None],raw.shape[1],axis=0))
        targets.append(_uint8(td,'targets') if has_GT else np.repeat(previous[None],raw.shape[1],axis=0))
    gpu_seconds=0.
    if gpu is not None:
        import torch
        torch.cuda.synchronize(gpu.device);started=time.perf_counter();all_basic=gpu_basic(gpu,np.stack(targets),np.stack(lasts),cal);torch.cuda.synchronize(gpu.device);gpu_seconds=time.perf_counter()-started
    else:
        all_basic=[_numpy_basic(raw[j,k],targets[j][k],lasts[j][k],cal) for j in range(len(rr)) for k in range(raw.shape[1])]
    result=[];cpu_started=time.perf_counter();skip_count=0;reference_count=0
    for j,(row,td) in enumerate(zip(rr,truth_arrays)):
        basics=all_basic[j*raw.shape[1]:(j+1)*raw.shape[1]];scored=[];pred_features=[]
        for k,basic in enumerate(basics):
            if has_GT:
                m,pf=remaining(raw[j,k],targets[j][k],lasts[j][k],row['letter'],basic,cal);scored.append(m)
                skip_count+=m['legacy_numeric_status']=='not_computed_due_to_prior_unknown';reference_count+=m['legacy_numeric_status']=='original_CPU_reference_computed'
            else:
                from tdyssey import native_rgb_evaluation as old
                top=old._topology(basic['gray'],cal['minimum_component_pixels']);pf=_features(basic['gray'],basic['blue'],top);scored.append(None)
            pred_features.append(pf)
        for k,(m,pf,basic) in enumerate(zip(scored,pred_features,basics)):
            # B features/angles were sliced to its one true target by data.py.
            rotation=hmf().rotation_assessment(td['truth_features'],td['recorded_rotation_deg'],pred_features,mode,k+1,std) if has_GT else None
            shape=topology=None
            if has_GT:
                tf=td['truth_features'][k+1];unknown=pf['cropped'] or tf['cropped'] or pf['blue_in_enclosed_region'] or tf['blue_in_enclosed_region']
                matches=pf['holes']==tf['holes'] and pf['components']==tf['components'] and pf['tiny_fragment_pixels']<=tf['tiny_fragment_pixels']
                topology='unknown' if unknown else 'pass' if matches else 'fail';shape='unknown' if m['legacy_shape_deformation_failure'] is None else 'fail' if m['legacy_shape_deformation_failure'] else 'pass'
                if topology=='fail':shape='fail'
                elif topology=='unknown' and shape!='fail':shape='unknown'
            label=str(batch_record.get('label',batch_record.get('checkpoint',{}).get('label','unknown')))
            item=dict(section=section,checkpoint=label,checkpoint_identity=batch_record.get('checkpoint'),sample_id=row['sample_id'],original_sample_id=row.get('original_sample_id',row['sample_id']),
                family_id=row['family_id'],letter=row['letter'],split=row['split'],selection_role=row.get('selection_role',row.get('B_role')),
                anchor=row['anchor'],horizon=k+1,input_mode=mode,batch_id=batch_record['batch_id'],batch_slot=j,
                action_type=action,noise_id=batch_record.get('noise_id'),noise_name=batch_record.get('noise_name'),GT_scored=has_GT,counterfactual_GT_available=False,
                parent_phase=row.get('parent_phase'),recorded_hold_identical=row.get('recorded_hold_identical',False),
                prediction_raw_ref=next((x for x in batch_record.get('raw_files',[]) if x['sample_id']==row['sample_id']),None),
                metrics=m,rotation=rotation,visible_shape_status=shape,visible_topology_status=topology,
                predicted_visible_features=pf,response_from_history=response_from_history(raw[j,k],lasts[j][k],basic,pf,cal),
                precise_corner_or_rigid_geometry_proven=False)
            if has_GT:
                motion=td['target_motion'][k];item.update(strong_H1=motion['strong'],dominance=motion['dominance'],true_H1_motion=motion)
                a,b=_center(basic['target_gray']),_center(basic['last_gray'])
                item['truth_visible_motion_xy_imagepx']=None if a is None or b is None else (a-b).tolist()
                if section=='B':
                    pv=item['response_from_history']['object_visible_delta_xy_imagepx'];tv=item['truth_visible_motion_xy_imagepx']
                    pn=None if pv is None else float(np.linalg.norm(pv));tn=None if tv is None else float(np.linalg.norm(tv))
                    item['recorded_motion_comparison']=dict(predicted_visible_magnitude_imagepx=pn,truth_visible_magnitude_imagepx=tn,
                        visible_magnitude_ratio=None if pn is None or not tn else pn/tn,
                        visible_direction_cosine=None if not pn or not tn else float(np.dot(pv,tv)/(pn*tn)),
                        truth_translation_below_one_imagepx=tn is None or tn<1.,
                        scope='Grayvisiblecentroid relative tosame realpreviousframe. Not physicalbodypose; smalltranslation ratios may be unstable.')
            else:item['counterfactual_accuracy_status']='No same-state counterfactual GT; sensitivity only. No target-error or correct-torque claim.'
            result.append(item)
    return dict(rows=result,scoring=dict(wall_seconds=time.perf_counter()-begin,gpu_basic_wall_seconds=gpu_seconds,
        cpu_remaining_wall_seconds=time.perf_counter()-cpu_started,legacy_numeric_skipped_prior_unknown=skip_count,original_CPU_qualified_registration_frames=reference_count,
        CPU_execution='Single calling worker thread. No multiprocessing, model, new inference or repeated archive hash.'),
        qualification_limits=['Original HMF visible shape/axis/unknown definitions retained. Unqualified legacy numeric attachments explicitly skipped, not approximated.',
            'Czero-hole topology is not a certificate for opening/corners. Blue response and gray object response are separate.',
            'RH t+16 is an independent real-history target, never a generated16step trajectory.'])

def pair_statistics(first,second,cal=None,*,gpu=None,features_first=None,features_second=None):
    """Action/noise pair distance: these are responses, not GT accuracies."""
    cal=calibration() if cal is None else cal
    basic=gpu_basic(gpu,np.asarray(second)[None],np.asarray(second)[None],cal)[0] if gpu is not None else _numpy_basic(first,second,second,cal)
    from tdyssey import native_rgb_evaluation as old
    gray,other,blue,other_blue=(basic[k] for k in ('gray','target_gray','blue','target_blue'))
    contour=old._contour_distance(gray,other);region=(gray|other)&~(blue|other_blue)
    pixel=None if not region.any() else float(np.abs(_rgb64(first)-_rgb64(second))[:,region].mean())
    result=dict(object_centroid_change_imagepx=basic['object_position_error_image_px'] if gray.any() and other.any() else None,object_mask_change=1-basic['raw_mask_iou'] if (gray|other).any() else None,
        object_contour_change_imagepx=None if contour is None else contour/SCALE,
        object_visible_region_rgb_mae_excluding_blue=pixel,raw_full_rgb_mae=basic['rgb_full_mae'],
        blue_centroid_change_imagepx=basic['pusher_position_error_image_px'] if blue.any() and other_blue.any() else None,blue_mask_change=1-basic['pusher_mask_iou'] if (blue|other_blue).any() else None,
        gray_presence_pixels=[int(gray.sum()),int(other.sum())],blue_presence_pixels=[int(blue.sum()),int(other_blue.sum())])
    pa=features_first if features_first is not None else _features(gray,blue,old._topology(gray,cal['minimum_component_pixels']))
    pb=features_second if features_second is not None else _features(other,other_blue,old._topology(other,cal['minimum_component_pixels']))
    axes,reasons=hmf().pair_axes(pb,pa,standard());vals=list(axes.values())
    if all(x is not None for x in vals) and abs(vals[0]-vals[1])>standard()['dual_axis_agreement_max_deg']:reasons.append('dual_axis_disagreement')
    result.update(apparent_axis_change_deg=axes,axis_pair_assessable=not reasons,axis_unknown_reasons=sorted(set(reasons)),
        physical_rotation_or_counterfactual_correctness_proven=False)
    return result

def check_saved_samples(samples,*,torch=None,device=None):
    """ROOT ONLY: execute on a few existing saved images, not model outputs.

    Caller chooses and loads the small frozen sample list once in its single
    BelowNormal process. This function neither hashes archives nor calls models.
    It is deliberately not executed during source preparation by subagents.
    """
    from scripts.native_rgb_stage3_score import score_image
    from tdyssey import native_rgb_evaluation as old
    cal=calibration();checks=[]
    primary=('raw_mask_iou','object_position_error_image_px','pusher_position_error_image_px','rgb_full_mae','rgb_full_mse','raw_contour_error_image_px','predicted_holes','truth_holes','predicted_components','truth_components')
    for sample in samples:
        pred,truth,last=(sample[k] for k in ('prediction','truth','last'))
        reference=score_image(pred,truth,last,cal,letter=sample['letter'])
        if torch is not None:
            tensor=torch.as_tensor(np.asarray(pred)[None],device=device,dtype=torch.float32)
            basic=gpu_basic(tensor,np.asarray(truth)[None],np.asarray(last)[None],cal)[0]
        else:basic=_numpy_basic(pred,truth,last,cal)
        cpu=_numpy_basic(pred,truth,last,cal);mask_equal=all(np.array_equal(basic[k],cpu[k]) for k in ('gray','blue','target_gray','target_blue','last_gray','last_blue'))
        actual,feat=remaining(pred,truth,last,sample['letter'],basic,cal)
        differences={k:abs(float(actual[k])-float(reference[k])) for k in primary}
        features_equal=feat==hmf().features(pred,cal)
        qualification_equal=actual['legacy_shape_assessable']==reference['legacy_shape_assessable'] and actual['legacy_shape_deformation_failure']==reference['legacy_shape_deformation_failure']
        ok=mask_equal and features_equal and qualification_equal and max(differences.values())<=1e-12
        checks.append(dict(sample_id=sample.get('sample_id'),passed=ok,masks_bitwise_equal=mask_equal,features_equal=features_equal,
            qualification_equal=qualification_equal,primary_max_abs_difference=max(differences.values()),differences=differences,
            prior_unknown_numeric_skip=actual['legacy_numeric_status']=='not_computed_due_to_prior_unknown'))
    return dict(passed=bool(checks) and all(x['passed'] for x in checks),samples=checks,model_calls=0,
        scope='Small existing savedraw parity check;1e-12 checks arithmetic implementation, not scientific threshold relaxation.')

def checks_design():
    return ['ROOT ONLY: existing O/C/B savedraw samples, including original pre-registration qualified and unknown frames, through check_saved_samples.',
        'Require exact gray/blue masks and original features/shape qualification; main scalar differences<=1e-12. Failure stops inference; no altered scientific thresholds.',
        'Static Bcontract: onlyrecorded rows receive GTmetrics/rotation;hold/alternative metrics and accuracy statuses stayNone; no invented counterfactual labels.',
        'Source/count contract: no model call in metrics/report; only one caller, no thread/process pools or rawarchive rewrite/hash.']
