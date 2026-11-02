"""Frozen independent visible-direction/topology diagnostics; saved RGB only.

This appendix does not replace the original formally unknown rotation metrics.
Real-history predictions are independent one-step transitions from the previous
REAL frame. Only autoregressive outputs form a predicted cumulative trajectory.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy import ndimage

REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.append(str(REPO))
from tdyssey.native_rgb_evaluation import palette_masks

def protocol():
    return dict(kind='cd-visible-direction-protocol.v1',clockwise_positive_image_y_down=True,
        axis_ratio_min=1.2,truth_axis_calibration_max_deg=5.,axis_jump_max_deg=45.,
        gray_hole_delta_agreement_max_deg=10.,resolved_rotation_floor_deg=5.,
        calibration_margin_deg=1.,hole_min_pixels=2,fragment_min_pixels=3,
        fragment_fraction_max=.05,gray_foreground_connectivity=8,white_background_connectivity=4,
        definition='PCA gray and single enclosed-hole axes, modulo180 continuity. Both must agree. Real-history uses previous REAL frame to independent prediction; AR reports local and cumulative separately.',
        frozen_before_new_predictions=True,
        limits=['Conservative new visible-axis appendix, not calibrated rigid pose or a change to old formal unknown.',
          'Tiny true or predicted changes, near-circular masks, axis jumps, hole occlusion, clipping, fragmentation and failed true-image calibration remain unknown.',
          'Hole count/area and topology candidates do not establish correct shape, rigidity, collision or the physical cause of a lost hole.'])

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _axis(mask):
    y,x=np.nonzero(mask)
    if len(x)<3:return None
    eigen,vector=np.linalg.eigh(np.cov(np.column_stack((x,y)),rowvar=False,bias=True))
    return dict(angle=float(np.degrees(np.arctan2(vector[1,-1],vector[0,-1]))),
                ratio=float(eigen[-1]/max(eigen[0],1e-10)),pixels=int(len(x)))

def measure(image,calibration,p=None):
    p=protocol() if p is None else p
    gray,blue=palette_masks(image,calibration)
    labels,n=ndimage.label(~np.pad(gray,1),structure=ndimage.generate_binary_structure(2,1))
    sizes=np.bincount(labels.ravel());outside=labels[0,0]
    ids=[i for i in range(1,n+1) if i!=outside and sizes[i]>=p['hole_min_pixels']]
    masks=[labels[1:-1,1:-1]==i for i in ids]
    union=np.logical_or.reduce(masks) if masks else np.zeros_like(gray)
    gl,gn=ndimage.label(gray,structure=ndimage.generate_binary_structure(2,2))
    gs=sorted(np.bincount(gl.ravel())[1:].tolist(),reverse=True)
    area=int(gray.sum());small=sum(gs[1:]) if gs else 0
    fragmented=bool(small>=p['fragment_min_pixels'] and small>area*p['fragment_fraction_max'])
    cropped=bool(gray[0].any() or gray[-1].any() or gray[:,0].any() or gray[:,-1].any())
    occluded=bool((union&blue).any())
    return dict(gray=_axis(gray),hole=_axis(masks[0]) if len(masks)==1 else None,
        hole_count=len(ids),hole_areas=sorted([int(m.sum()) for m in masks],reverse=True),
        enclosed_area=int(union.sum()),gray_area=area,blue_area=int(blue.sum()),
        gray_components=gn,significant_fragmentation=fragmented,cropped=cropped,blue_in_hole=occluded)

def _delta(a,b):return float((b-a+90.)%180.-90.)
def _region_reasons(m,region,p):
    reasons=[];a=m[region]
    if a is None:reasons.append(region+'_axis_missing')
    elif a['ratio']<p['axis_ratio_min']:reasons.append(region+'_near_circular')
    if m['cropped']:reasons.append('viewport_crop')
    if m['significant_fragmentation']:reasons.append('gray_fragmentation')
    if m['blue_in_hole']:reasons.append('blue_occludes_enclosed_region')
    if m['hole_count']!=1:reasons.append('O_single_hole_unavailable')
    return reasons

def _pair_delta(left,right,p):
    values={};reasons=[]
    for region in ('gray','hole'):
        reasons+=_region_reasons(left,region,p)+_region_reasons(right,region,p)
        a,b=left[region],right[region]
        values[region]=None if a is None or b is None else _delta(a['angle'],b['angle'])
        if values[region] is not None and abs(values[region])>p['axis_jump_max_deg']:
            reasons.append(region+'_axis_jump')
    return values,sorted(set(reasons))

def _classify(predicted,truth_delta,calibration_errors,reasons,p):
    reasons=list(reasons)
    finite=[abs(x) for x in calibration_errors.values() if x is not None]
    if len(finite)!=2 or max(finite)>p['truth_axis_calibration_max_deg']:
        reasons.append('truth_image_axis_calibration_failed')
    guard=max(p['resolved_rotation_floor_deg'],(max(finite) if finite else 0.)+p['calibration_margin_deg'])
    if abs(truth_delta)<guard:reasons.append('true_rotation_below_visible_guard')
    vals=[predicted[x] for x in ('gray','hole')]
    if any(x is None for x in vals):reasons.append('predicted_axis_unavailable')
    elif abs(vals[0]-vals[1])>p['gray_hole_delta_agreement_max_deg']:
        reasons.append('gray_hole_direction_disagreement')
    status='unknown'
    if not reasons:
        if all(abs(v)<guard for v in vals):status='no_resolved_rotation'
        elif all(v*np.sign(truth_delta)>=guard for v in vals):status='correct'
        elif all(v*np.sign(truth_delta)<=-guard for v in vals):status='opposite'
        else:reasons.append('predicted_direction_below_or_across_guard')
    return dict(classification=status,known=status in ('correct','opposite'),signed_delta_deg=predicted,
        truth_signed_delta_deg=float(truth_delta),visible_guard_deg=float(guard),
        truth_axis_calibration_error_deg=calibration_errors,unknown_reasons=sorted(set(reasons)),
        physical_rigid_rotation_assessable=False)

def _o_label(truth,angles,pred,mode,calibration,p):
    tm=[measure(im,calibration,p) for im in truth]
    pm=[measure(im,calibration,p) for im in pred]
    true_angles=np.degrees(np.unwrap(np.asarray(angles,dtype=float)));true_angles-=true_angles[0]
    frames=[];t_cum={'gray':0.,'hole':0.};p_cum={'gray':0.,'hole':0.}
    t_reasons=[];p_reasons=[];max_errors={'gray':0.,'hole':0.}
    for k in range(16):
        td,tr=_pair_delta(tm[k],tm[k+1],p)
        previous=tm[k] if mode=='real_history' else (tm[0] if k==0 else pm[k-1])
        pd,pr=_pair_delta(previous,pm[k],p)
        truth_step=float(true_angles[k+1]-true_angles[k])
        local_error={r:None if td[r] is None else td[r]-truth_step for r in td}
        local=_classify(pd,truth_step,local_error,tr+pr,p)
        t_reasons+=tr;p_reasons+=pr
        for r in t_cum:
            t_cum[r]=None if td[r] is None or t_cum[r] is None else t_cum[r]+td[r]
            err=None if t_cum[r] is None else t_cum[r]-true_angles[k+1]
            max_errors[r]=None if err is None or max_errors[r] is None else max(max_errors[r],abs(err))
        cumulative=None
        if mode!='real_history':
            for r in p_cum:p_cum[r]=None if pd[r] is None or p_cum[r] is None else p_cum[r]+pd[r]
            cumulative=_classify(dict(p_cum),float(true_angles[k+1]),dict(max_errors),t_reasons+p_reasons,p)
        frames.append(dict(horizon=k+1,truth_local_rotation_deg=truth_step,
            truth_cumulative_rotation_deg=float(true_angles[k+1]),local_direction=local,
            cumulative_direction=cumulative,prediction_axes=pm[k],truth_previous_axes=tm[k],truth_target_axes=tm[k+1]))
    key='local_direction' if mode=='real_history' else 'cumulative_direction'
    selected=frames[-1][key]
    return dict(**selected,input_mode=mode,summary_definition='H16 independent local transition' if mode=='real_history' else 'H16 cumulative AR trajectory',
        first_opposite=next((f['horizon'] for f in frames if f[key]['classification']=='opposite'),None),
        first_local_opposite=next((f['horizon'] for f in frames if f['local_direction']['classification']=='opposite'),None),
        classifications=dict(Counter(f[key]['classification'] for f in frames)),frames=frames)

def _b_label(truth,pred,calibration,p):
    frames=[]
    for k,(gt,image) in enumerate(zip(truth[1:],pred),1):
        t=measure(gt,calibration,p);v=measure(image,calibration,p)
        reasons=[]
        for name,m in [('truth',t),('prediction',v)]:
            if m['cropped']:reasons.append(name+'_viewport_crop')
            if m['blue_in_hole']:reasons.append(name+'_blue_in_hole')
            if m['significant_fragmentation']:reasons.append(name+'_gray_fragmentation')
            if m['gray_area']==0:reasons.append(name+'_gray_missing')
        if t['hole_count']!=2:reasons.append('truth_two_visible_holes_unavailable')
        known=not reasons;mismatch=known and v['hole_count']!=t['hole_count']
        candidate=None
        if mismatch:
            da=v['enclosed_area']-t['enclosed_area'];dg=v['gray_area']-t['gray_area']
            if da<0 and dg>0:candidate='filling_candidate_not_causal_proof'
            elif da>0 and dg<0:candidate='wall_loss_or_merge_candidate_not_causal_proof'
            else:candidate='hole_loss_cause_ambiguous'
        frames.append(dict(horizon=k,known=known,predicted_holes=v['hole_count'],truth_holes=t['hole_count'],
            mismatch=bool(mismatch),unknown_reasons=reasons,prediction=v,truth=t,topology_candidate=candidate))
    unknown=[f['horizon'] for f in frames if not f['known']]
    return dict(hole_count_assessable=not unknown,counts16=[f['predicted_holes'] for f in frames],
        truth_counts16=[f['truth_holes'] for f in frames],unknownframes=unknown,
        first_mismatch=next((f['horizon'] for f in frames if f['mismatch']),None),
        holes_at_H16=frames[-1]['predicted_holes'],H16_assessable=frames[-1]['known'],
        frames=frames,rigid_shape_preservation_proven=False)

def analyze_saved(spec,metrics_by_label,calibration):
    """Analyze every original O/B case; caller persists returned JSON.

    ``metrics_by_label`` values may be metrics paths, result {'metrics':path}, or
    already-read original metrics dictionaries. No model/Torch imports or calls.
    """
    spec=read(spec) if isinstance(spec,(str,Path)) else spec
    calibration=read(calibration) if isinstance(calibration,(str,Path)) else calibration
    models={};refs={}
    for label,value in metrics_by_label.items():
        if isinstance(value,dict) and 'metrics' in value and 'raw_files' not in value:value=value['metrics']
        data=read(value) if isinstance(value,(str,Path)) else value
        models[label]=data;refs[label]={x['sample_id']:x for x in data['raw_files']}
    output=dict(kind='cd-independent-visible-direction-and-holes.v1',protocol=protocol(),
        o_direction_rows=[],b_hole_rows=[],model_calls=0,physics_steps=0,
        official_rotation_metrics_unchanged=True)
    for row in spec['rows']:
        if row['letter'] not in ('O','B'):continue
        with np.load(row['raw_path'],allow_pickle=False) as z:
            a=int(row['anchor']);truth=z['rgb'][a:a+17].copy();angles=z['states'][a:a+17,4].copy()
        if len(truth)!=17:raise ValueError('Incomplete truth timeline')
        result={k:row[k] for k in ('sample_id','family_id','letter','anchor','raw_path')};result['labels']={}
        for label,metrics in models.items():
            rf=refs[label][row['sample_id']]
            if sha(rf['path'])!=rf['sha256']:raise ValueError('Prediction raw hash changed')
            with np.load(rf['path'],allow_pickle=False) as z:pred=z['raw'].copy()
            if pred.shape!=(16,3,96,96) or pred.dtype!=np.float32 or not np.isfinite(pred).all():raise ValueError('Prediction raw contract failed')
            mode=metrics.get('input_mode','autoregressive')
            result['labels'][label]=_o_label(truth,angles,pred,mode,calibration,output['protocol']) if row['letter']=='O' else _b_label(truth,pred,calibration,output['protocol'])
            result['labels'][label]['prediction_raw_path']=rf['path']
        output['o_direction_rows' if row['letter']=='O' else 'b_hole_rows'].append(result)
    output['counts']={'O_cases':len(output['o_direction_rows']),'B_cases':len(output['b_hole_rows']),'labels':list(models)}
    output['support_signals']={label:support_signals(output,label) for label in models}
    return output

def support_signals(analysis,label):
    """Frozen strict visible-support signals. Unknown never counts as success."""
    os=[r for r in analysis['o_direction_rows'] if 'AC-O-01-08:' in r['sample_id']]
    bs=[r for r in analysis['b_hole_rows'] if 'AC-B-00-08:' in r['sample_id']]
    o=os[0]['labels'].get(label,{}) if len(os)==1 else {}
    b=bs[0]['labels'].get(label,{}) if len(bs)==1 else {}
    o_pass=bool(o.get('input_mode')=='autoregressive' and o.get('known') and o.get('classification')=='correct' and o.get('truth_signed_delta_deg',0)>0)
    b_show=bool(b.get('H16_assessable') and b.get('holes_at_H16')==2)
    others=[r['labels'].get(label,{}) for r in analysis['b_hole_rows'] if 'AC-B-00-08:' not in r['sample_id']]
    other_pass=bool(len(others)==7 and all(x.get('hole_count_assessable') and x.get('first_mismatch') is None and x.get('counts16')==[2]*16 for x in others))
    return dict(O_display_known_correct_clockwise=o_pass,O_display_classification=o.get('classification','unknown'),
        O_display_mode=o.get('input_mode','unknown'),B_display_H16_two_visible_holes=b_show,
        B_other7_no_hole_mismatch_or_unknown=other_pass,B_visible_hole_support_passed=b_show and other_pass,
        rigid_shape_or_exact_physical_angle_proven=False)
