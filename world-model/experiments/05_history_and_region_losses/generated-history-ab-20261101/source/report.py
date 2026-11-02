"""Saved-only paired A/B report: identical supervision is not equal compute.

Reads frozen evaluation metrics/raw files only. No Torch, simulator, renderer,
model construction or prediction scoring. Numeric angles remain unknown unless
the metric producer explicitly declares them assessable.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HIGHLIGHTS = (1,4,8,16)
LETTERS = tuple('TLOCXB')
LABELS = ('baseline','A','B')
TIE_EPSILON = 1e-8
# direction +1 means larger is favorable; -1 means smaller is favorable.
FIELDS = {'raw_mask_iou':1, 'raw_contour_error_image_px':-1,
          'object_position_error_image_px':-1, 'motion_vector_error_image_px':-1,
          'pusher_position_error_image_px':-1, 'pusher_mask_iou':1,
          'holes_difference':-1, 'missing_components':-1, 'extra_fragments':-1,
          'aligned_mask_iou':1, 'visible_angle_error_deg':-1,
          'absolute_underrotation_deg':-1}
CORE = ('raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def guard(root,config,extra=0):
    from common import check_storage
    return check_storage(root,config,reserve_bytes=int(extra))


def write(path,value,root,config):
    from common import replace_atomic
    payload=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    guard(root,config,len(payload));path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    replace_atomic(temporary,path)


def metric(row,key):
    m=row.get('metrics',{})
    if key=='visible_angle_error_deg' and m.get('angle_assessable') is not True:
        return None
    if key=='absolute_underrotation_deg' and m.get('underrotation_assessable') is not True:
        return None
    if key=='aligned_mask_iou' and m.get('legacy_shape_assessable') is not True:
        return None
    value=m.get(key)
    return float(value) if isinstance(value,(int,float)) and math.isfinite(float(value)) else None


def distribution(values,total=None):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    total=len(values) if total is None else total
    result=dict(available=len(values),unknown=total-len(values))
    if values:
        result.update(zip(('mean','median','P10','P90','min','max'),map(float,
            (np.mean(values),np.median(values),np.quantile(values,.1),np.quantile(values,.9),min(values),max(values)))))
    else:
        result.update(dict.fromkeys(('mean','median','P10','P90','min','max')))
    return result


def summarize(rows):
    n=len(rows);failures=Counter(reason for row in rows for reason in row.get('metrics',{}).get('failure_reasons',[]))
    return dict(denominator=n,families=len({row.get('family_id',row['sample_id']) for row in rows}),
        metrics={key:distribution([metric(row,key) for row in rows],n) for key in FIELDS},
        shape_assessable=sum(row.get('metrics',{}).get('legacy_shape_assessable') is True for row in rows),
        shape_unknown=sum(row.get('metrics',{}).get('legacy_shape_assessable') is not True for row in rows),
        angle_assessable=sum(metric(row,'visible_angle_error_deg') is not None for row in rows),
        angle_unknown=sum(metric(row,'visible_angle_error_deg') is None for row in rows),
        angle_unknown_reasons=dict(Counter(row.get('metrics',{}).get('angle_unknown_reason') or 'not_explicitly_assessable' for row in rows if metric(row,'visible_angle_error_deg') is None)),
        shape_unknown_reasons=dict(Counter(row.get('metrics',{}).get('legacy_shape_unknown_reason') or 'not_assessable' for row in rows if row.get('metrics',{}).get('legacy_shape_assessable') is not True)),
        numeric_angle_diagnostic=distribution([row.get('metrics',{}).get('visible_angle_error_deg') for row in rows],n),
        blue_missing=failures.get('empty_pusher',0),object_missing=failures.get('empty_object',0),
        hard_generation_failures=sum(bool(row.get('metrics',{}).get('hard_generation_failure')) for row in rows),
        visible_hole_count_mismatches=sum(row.get('metrics',{}).get('holes_difference') not in (None,0) for row in rows),
        failure_reasons=dict(failures))


def compare_pair(left,right):
    """Metric-specific signed changes, with explicit numerical tie tolerance."""
    out={};directions=[]
    for key,direction in FIELDS.items():
        before,after=metric(left,key),metric(right,key)
        if before is None or after is None:
            out[key]=dict(status='unknown',delta=None);continue
        delta=after-before;favorable=delta*direction
        status='tied' if abs(delta)<=TIE_EPSILON else 'improved' if favorable>0 else 'regressed'
        out[key]=dict(status=status,delta=delta,before=before,after=after)
        if key in CORE and status!='tied':directions.append(status)
    statuses=set(directions)
    image_status=('unknown' if any(out[key]['status']=='unknown' for key in CORE) else
                  'mixed' if len(statuses)>1 else next(iter(statuses)) if statuses else 'tied')
    return dict(image_metric_direction=image_status,metrics=out)


def compare_summary(pairs):
    result=dict(denominator=len(pairs),image_metric_direction=dict(Counter(row['image_metric_direction'] for row in pairs)),metrics={})
    for key in FIELDS:
        values=[row['metrics'][key] for row in pairs]
        counts=Counter(value['status'] for value in values)
        result['metrics'][key]=dict(**{status:counts[status] for status in ('improved','tied','regressed','unknown')},
                                   delta=distribution([value['delta'] for value in values],len(values)))
    return result


def _path(value,root):
    if isinstance(value,dict):
        value=value.get('metrics',value.get('path'))
    if not isinstance(value,str):raise ValueError('Missing saved evaluation metrics path')
    path=Path(value)
    return path if path.is_absolute() else root/path


def _rows(value):
    return [row for row in value.get('rows',[]) if row.get('method')!='copy_last']


def _index(value):
    rows={}
    for row in _rows(value):
        key=(row['sample_id'],int(row['horizon']))
        if key in rows:raise ValueError('Repeated saved case/horizon')
        rows[key]=row
    return rows


def check_pairing(bundle):
    indexed={label:_index(value) for label,value in bundle.items()}
    reference=set(indexed['baseline'])
    if any(set(indexed[label])!=reference for label in LABELS):
        raise ValueError('A/B/start case and horizon identities differ')
    if reference and {key[1] for key in reference}!=set(range(1,17)):
        raise ValueError('Saved-only metrics do not cover all sixteen target times')
    raw={label:{row['sample_id']:row for row in value.get('raw_files',[])} for label,value in bundle.items()}
    if any(set(raw[label])!=set(raw['baseline']) for label in LABELS):
        raise ValueError('A/B/start raw prediction cases differ')
    for sid in raw['baseline']:
        if {h for case,h in reference if case==sid}!=set(range(1,17)):
            raise ValueError('A saved case lacks one or more of its sixteen target times')
    for sid in raw['baseline']:
        seeds=[raw[label][sid].get('seeds') for label in LABELS]
        if seeds[0] is None or any(value!=seeds[0] for value in seeds[1:]):
            raise ValueError('A/B/start prediction-noise seeds differ')
    verified_raw=0
    for label in LABELS:
        for record in raw[label].values():
            if not record.get('sha256') or sha(record['path'])!=record['sha256']:
                raise ValueError('A saved raw prediction hash differs')
            verified_raw+=1
    if any(list(raw[label])!=list(raw['baseline']) for label in LABELS):
        raise ValueError('A/B/start prediction case order differs')
    proofs={label:bundle[label].get('input_evidence') for label in LABELS}
    if not proofs['A'] or not proofs['B']:
        raise ValueError('A/B runtime history/action input evidence is missing')
    required={'sample_ids','family_ids','seeds','initial_history_sha256',
              'absolute_actions_sha256','normalized_actions_sha256','shape_history','shape_actions'}
    for label in LABELS:
        if proofs[label]:
            if any(not required.issubset(item) for item in proofs[label]):
                raise ValueError('Incomplete runtime history/action evidence')
            if [sid for item in proofs[label] for sid in item['sample_ids']]!=list(raw[label]):
                raise ValueError('Runtime input evidence case order differs from raw records')
    if proofs['A']!=proofs['B']:
        raise ValueError('A/B actual initial history, actions or noise input evidence differs')
    baseline_runtime=bool(proofs['baseline'])
    if baseline_runtime:
        if proofs['baseline']!=proofs['A']:
            raise ValueError('Common-start runtime input evidence differs')
    elif not bundle['baseline'].get('reused_formal_metrics'):
        raise ValueError('Missing common-start runtime input proof without an explicit reused-formal source')
    return indexed,dict(case_horizons=len(reference),cases=len(raw['baseline']),
        full_sixteen_times=True,same_recorded_seeds=True,same_case_order=True,
        raw_prediction_files_hash_verified=verified_raw,
        A_B_runtime_initial_history_actions_and_noise_match=True,
        baseline_runtime_input_evidence_available=baseline_runtime,
        baseline_input_evidence_scope='Matched runtime input hashes' if baseline_runtime else
            'Reused formal output: frozen original spec, raw hashes, case order and seeds; original epoch2 runtime input tensors were not separately saved')


def _raw(value,sample_id):
    record=next((row for row in value.get('raw_files',[]) if row['sample_id']==sample_id),None)
    if record is None:raise ValueError('Missing saved prediction case '+sample_id)
    path=Path(record['path'])
    if record.get('sha256') and sha(path)!=record['sha256']:raise ValueError('Saved prediction changed '+str(path))
    with np.load(path,allow_pickle=False) as archive:raw=archive['raw'].copy()
    if raw.shape!=(16,3,96,96) or not np.isfinite(raw).all():raise ValueError('Incomplete/nonfinite saved prediction')
    return raw,dict(path=str(path),sha256=sha(path))


def _truth(row):
    path=Path(row['raw_path'])
    if row.get('raw_sha256') and sha(path)!=row['raw_sha256']:raise ValueError('Frozen truth source changed')
    a=int(row['anchor'])
    with np.load(path,allow_pickle=False) as archive:
        rgb=archive['rgb'][a+1:a+17].copy();valid=archive['control_valid'][a-3:a+16]
    if rgb.shape!=(16,3,96,96) or a<3 or not valid.all():raise ValueError('Missing clean real H16 truth')
    return rgb.astype(np.float32)/np.float32(255)


def _frames(sample_id,columns):
    import cv2
    frames=[]
    for step in range(16):
        frame=np.full((250,len(columns)*192,3),255,np.uint8)
        cv2.putText(frame,f'{sample_id} | t+{step+1}={(step+1)/10:.1f}s',(4,13),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1)
        for col,(label,images) in enumerate(columns):
            for line,part in enumerate(label.split('|')[:2]):
                cv2.putText(frame,part,(col*192+4,30+line*13),cv2.FONT_HERSHEY_SIMPLEX,.38,(20,20,20),1)
            display=np.rint(np.clip(images[step].transpose(1,2,0),0,1)*255).astype(np.uint8)
            frame[54:246,col*192:(col+1)*192]=cv2.resize(display,(192,192),interpolation=cv2.INTER_NEAREST)
        frames.append(frame)
    return frames


def media(root,config,name,row,bundle,*,mode,scope):
    import cv2
    import imageio_ffmpeg
    folder=root/'report';truth=_truth(row);columns=[('SAVED TRUTH',truth)];inputs=[]
    for label,title in (('baseline','COMMON EPOCH2 START'),('A','A / TRUE HISTORY TRAIN'),('B','B / GENERATED HISTORY')):
        raw,source=_raw(bundle[label],row['sample_id']);inputs.append(source)
        title+='|AR16' if mode=='autoregressive' else '|real-history single step'
        columns.append((title,raw))
    frames=_frames(row['sample_id'],columns);records={}
    for suffix,selected in (('H1-H4-H8-H16',[frames[h-1] for h in HIGHLIGHTS]),('all-H1-H16',frames)):
        board=np.concatenate(selected,axis=0);ok,png=cv2.imencode('.png',cv2.cvtColor(board,cv2.COLOR_RGB2BGR))
        if not ok:raise RuntimeError('Comparison board encoding failed')
        path=folder/f'{name}-{suffix}.png';guard(root,config,len(png));path.write_bytes(png.tobytes())
        records['board' if suffix.startswith('H1-') else 'all16_board']=dict(path=str(path),sha256=sha(path))
    path=folder/f'{name}-10fps.mp4';guard(root,config,4*1024**2)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','768x250','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    kw={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
    result=subprocess.run(cmd,input=b''.join(frame.tobytes() for frame in frames),stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
    if result.returncode:raise RuntimeError('Saved comparison video failed '+result.stderr.decode(errors='replace')[-300:])
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while cap.read()[0]:count+=1
    cap.release()
    if count!=16 or abs(fps-10)>1e-3:raise RuntimeError('Comparison video timeline differs from true10Hz')
    records['video']=dict(path=str(path),sha256=sha(path),frames=count,fps=fps)
    return dict(scope=scope,mode=mode,split=row.get('split','seen_val'),letter=row['letter'],
                sample_id=row['sample_id'],sources=inputs,display_quantization_only=True,**records)


def _subsets(rows,strong,static):
    return {'ALL':rows,**{letter:[row for row in rows if row['letter']==letter] for letter in LETTERS},
            'O_strong4':[row for row in rows if row['letter']=='O' and row['sample_id'] in strong],
            'C_strong4':[row for row in rows if row['letter']=='C' and row['sample_id'] in strong],
            'OC_strong8':[row for row in rows if row['sample_id'] in strong],
            'frozen_bitstatic':[row for row in rows if row['sample_id'] in static]}


def _case_record(scope,mode,split,key,left,right,comparison):
    details=compare_pair(left,right)
    observations={}
    for label,row in ((comparison.split('_vs_')[1],left),(comparison.split('_vs_')[0],right)):
        m=row.get('metrics',{})
        observations[label]={name:m.get(name) for name in ('predicted_holes','truth_holes','missing_components','extra_fragments','failure_reasons','legacy_shape_assessable','legacy_shape_unknown_reason','angle_assessable','visible_angle_assessable','physical_rotation_assessable','angle_unknown_reason','underrotation_assessable','underrotation_deg','occlusion_ambiguous')}
    return dict(scope=scope,mode=mode,split=split,sample_id=key[0],horizon=key[1],
                letter=right['letter'],family_id=right.get('family_id'),comparison=comparison,
                observations=observations,**details)


def _format(group,key,stat='mean',digits=3):
    value=group['metrics'][key].get(stat)
    return 'unknown' if value is None else f'{value:.{digits}f}'


def _table(groups):
    columns=['Arm','H','n','IoU mean / median / P10','Position mean / median / P90 px','Contour mean / median / P90 px','Shape unknown','Angle unknown','Blue missing']
    rows=[]
    for g in groups:
        trip=lambda key,tail:' / '.join(_format(g,key,x) for x in ('mean','median',tail))
        rows.append([g['arm'],str(g['horizon']),str(g['denominator']),trip('raw_mask_iou','P10'),
                     trip('object_position_error_image_px','P90'),trip('raw_contour_error_image_px','P90'),
                     str(g['shape_unknown']),str(g['angle_unknown']),str(g['blue_missing'])])
    return columns,rows


def _comparison_table(groups):
    columns=['Comparison','H','n','IoU better / tied / worse / unknown','Position better / tied / worse / unknown','Contour better / tied / worse / unknown','Mixed image directions']
    rows=[]
    for g in groups:
        counts=lambda key:' / '.join(str(g['metrics'][key][x]) for x in ('improved','tied','regressed','unknown'))
        rows.append([g['comparison'],str(g['horizon']),str(g['denominator']),counts('raw_mask_iou'),counts('object_position_error_image_px'),counts('raw_contour_error_image_px'),str(g['image_metric_direction'].get('mixed',0))])
    return columns,rows


def _markdown_table(columns,rows):
    return ['| '+' | '.join(columns)+' |','|'+'|'.join('---' for _ in columns)+'|']+['| '+' | '.join(row)+' |' for row in rows]


def _html_table(columns,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def generate_report(root:Path,config:dict,engine_result:dict)->dict:
    root=Path(root).resolve();folder=root/'report';folder.mkdir(parents=True,exist_ok=True)
    ar1=Path(config.get('data_root',REPO/'artifacts/accelerated-rgb-run-20261031'))
    oc=Path(config.get('oc_root',REPO/'artifacts/oc-continuation-20261031'))
    evaluations=engine_result['evaluations'];inputs=[];formal={};diagnostic={};evaluation_accounting={}
    for arm in LABELS:
        value=evaluations[arm];path=_path(value['formal'],root);formal[arm]=read(path);inputs.append(dict(path=str(path),sha256=sha(path)))
        diagnostic[arm]={}
        for mode in ('real_history','autoregressive'):
            path=_path(value['diagnostics'][mode],root);diagnostic[arm][mode]=read(path);inputs.append(dict(path=str(path),sha256=sha(path)))
        evaluation_accounting[arm]={}
        for name,payload,result in [('formal',formal[arm],value['formal'])]+[(mode,diagnostic[arm][mode],value['diagnostics'][mode]) for mode in ('real_history','autoregressive')]:
            evaluation_accounting[arm][name]=dict(reused=bool(result.get('reused')),cases=len(payload.get('raw_files',[])),
                sampler_batch_calls=result.get('sampler_batch_calls',payload.get('sampler_batch_calls')),
                original_metric_denoiser_batch_forwards=payload.get('denoiser_batch_forwards'),
                wall_seconds=result.get('wall_seconds'),physical_steps=payload.get('physical_steps',0))
        if value.get('identity_evidence'):
            path=Path(value['identity_evidence']);inputs.append(dict(path=str(path),sha256=sha(path)))
        for payload in [formal[arm],*diagnostic[arm].values()]:
            inputs.extend(dict(path=record['path'],sha256=record['sha256']) for record in payload.get('raw_files',[]))
    reused=formal['baseline'].get('reused_formal_metrics')
    if reused:
        expected=formal['baseline'].get('reused_formal_metrics_sha256')
        if not expected or sha(reused)!=expected:raise ValueError('Reused common-start formal metrics changed')
        inputs.append(dict(path=str(reused),sha256=expected))
    formal_index,formal_pairing=check_pairing(formal)
    diag_indices={};diag_pairing={}
    for mode in ('real_history','autoregressive'):
        diag_indices[mode],diag_pairing[mode]=check_pairing({arm:diagnostic[arm][mode] for arm in LABELS})
    specpath=ar1/'evaluation-spec.json';spec=read(specpath);formal_cases={row['sample_id']:row for row in spec['rows']};inputs.append(dict(path=str(specpath),sha256=sha(specpath)))
    dpath=Path(config.get('diagnostic_spec',oc/'diagnostic-spec.json'));dspec=read(dpath);diag_cases={row['sample_id']:row for row in dspec['rows']};inputs.append(dict(path=str(dpath),sha256=sha(dpath)))
    strong={row['sample_id'] for row in _rows(formal['baseline']) if row['letter'] in 'OC' and row['horizon']==16 and row.get('obvious_motion')}
    static={sid for sid,row in formal_cases.items() if row.get('bitstatic_future')}
    groups=[];case_pairs=[];comparison_groups=[]
    for arm in LABELS:
        for h in range(1,17):
            rows=[row for row in formal_index[arm].values() if row['horizon']==h]
            for subset,selected in _subsets(rows,strong,static).items():
                groups.append(dict(scope='formal48',mode='autoregressive',split='seen_val',subset=subset,arm=arm,horizon=h,**summarize(selected)))
        for mode in ('real_history','autoregressive'):
            for split in ('train','seen_val'):
                for h in range(1,17):
                    rows=[row for row in diag_indices[mode][arm].values() if row['horizon']==h and row.get('split')==split]
                    for letter in ('ALL_OC','O','C'):
                        selected=rows if letter=='ALL_OC' else [row for row in rows if row['letter']==letter]
                        groups.append(dict(scope='diagnostic32',mode=mode,split=split,subset=letter,arm=arm,horizon=h,**summarize(selected)))
    copy_rows=[row for row in formal['baseline'].get('rows',[]) if row.get('method')=='copy_last']
    copy_path=None
    if not copy_rows:
        copy_path=oc/'evaluation/epoch2/metrics.json'
        if copy_path.exists():
            copy_rows=[row for row in read(copy_path).get('rows',[]) if row.get('method')=='copy_last']
            inputs.append(dict(path=str(copy_path),sha256=sha(copy_path)))
    for h in sorted({int(row['horizon']) for row in copy_rows}):
        rows=[row for row in copy_rows if row['horizon']==h]
        for subset,values in _subsets(rows,strong,static).items():
            groups.append(dict(scope='formal48',mode='autoregressive',split='seen_val',subset=subset,
                               arm='copy_last_reference',horizon=h,**summarize(values)))
    for comparison,left_arm,right_arm in (('B_vs_A','A','B'),('A_vs_baseline','baseline','A'),('B_vs_baseline','baseline','B')):
        local=[]
        for key,right in formal_index[right_arm].items():
            pair=_case_record('formal48','autoregressive','seen_val',key,formal_index[left_arm][key],right,comparison);local.append(pair);case_pairs.append(pair)
        for h in range(1,17):
            selected=[pair for pair in local if pair['horizon']==h]
            for subset,values in _subsets(selected,strong,static).items():
                comparison_groups.append(dict(scope='formal48',mode='autoregressive',split='seen_val',subset=subset,horizon=h,comparison=comparison,**compare_summary(values)))
        for mode in ('real_history','autoregressive'):
            local=[]
            for key,right in diag_indices[mode][right_arm].items():
                pair=_case_record('diagnostic32',mode,right['split'],key,diag_indices[mode][left_arm][key],right,comparison);local.append(pair);case_pairs.append(pair)
            for split in ('train','seen_val'):
                for h in range(1,17):
                    for letter in ('ALL_OC','O','C'):
                        values=[pair for pair in local if pair['split']==split and pair['horizon']==h and (letter=='ALL_OC' or pair['letter']==letter)]
                        comparison_groups.append(dict(scope='diagnostic32',mode=mode,split=split,subset=letter,horizon=h,comparison=comparison,**compare_summary(values)))
    media_records=[]
    for letter in LETTERS:
        candidates=sorted(sid for sid,row in formal_cases.items() if row['letter']==letter)
        preferred=[sid for sid in candidates if sid in strong] if letter in 'OC' else []
        if candidates:
            sid=(preferred or candidates)[0];record=media(root,config,f'formal-{letter}',formal_cases[sid],formal,mode='autoregressive',scope='formal48');record['selection']='First frozen strong-motion ID for O/C; first original case ID otherwise; predictions not consulted';media_records.append(record)
    for split in ('train','seen_val'):
        for letter in ('O','C'):
            rows=[row for row in diag_cases.values() if row['split']==split and row['letter']==letter]
            rows.sort(key=lambda row:(-abs(row.get('truth_h16_signed_rotation_deg',0)),row['sample_id']))
            if not rows:continue
            row=rows[0]
            for mode in ('real_history','autoregressive'):
                bundle={arm:diagnostic[arm][mode] for arm in LABELS}
                record=media(root,config,f'diagnostic-{split}-{letter}-{mode}',row,bundle,mode=mode,scope='diagnostic32');record['selection']='Maximum absolute recorded H16 physical rotation within the unchanged frozen split/letter group; predictions not consulted';media_records.append(record)
    for record in media_records:inputs.extend(record['sources'])
    limitations=[
        'The primary method comparison is B versus A. Beating only the shared epoch2 start cannot establish generated-history benefit.',
        'A/B each receive the same30272 supervised targets,7568 segments and1892 optimizer batches. B additionally generates22704 online historical images (68112 image-level denoiser forwards at3steps); this is not equal compute.',
        'Only one A/B run is approved. The shared epoch2 start was selected using development evidence; these are exposed seen letters and seen-val families, not an untouched final test.',
        'Copy-last reference scores are reused where actual saved scores exist; absent horizons are not invented. They are an additional retention/reference baseline, never a replacement for the trained A/B contrast.',
        'All image statistics are from raw saved outputs. Display clipping/quantization is viewing only and is never fed back or used to repair shape.',
        'Raw IoU, contour and centroid errors mix pose and shape. Hole counts describe visible masks; zero visible-hole difference cannot establish hole preservation under occlusion.',
        'Numeric angular registration is not angle assessability. Angle statistics and paired angle changes require an explicit angle_assessable=true; all other numeric angles remain an unvalidated diagnostic and unknown for conclusions.',
        'Signed underrotation is not inferred from absolute visible-mask angular registration. The dedicated underrotation measurement remains unknown unless explicitly declared assessable.',
        'Shape-aligned IoU is summarized only for explicitly assessable cases. Occlusion, clipping, symmetry and deformation unknowns do not count as passing.',
        'Real-history images at t+16 are separate one-step predictions with four true prior frames for that target. They are not sixteen-step closed-loop forecasts. AR16 starts from four true frames then feeds back raw predictions.',
        'The32movement cases are deliberately motion-biased and preserve separate train/seen-val groups; late target frames can have smaller motion than the chosen initial transition.',
        'Each per-case metric is classified by its signed change with1e-8 numerical tie tolerance. These signs are not a statistical significance or physical-validity test. Mixed image direction uses only IoU/centroid/contour, not an overall model-success label.',
        'Four-step generated-history training still retains one real frame at its fourth loss. Its truncated no-grad generated history is not full trajectory backpropagation and does not guarantee H16.',
        'No new physical data, new loss, architecture, server execution, second A/B pass or unapproved extra training is included.' ]
    accounting={key:engine_result.get(key) for key in ('complete','arms','throughput','restore_proofs','rng_preservation','physical_steps','supervised_targets_total','optimizer_batches_total','generation_counts','wall_seconds')}
    summary=dict(kind='generated-history-paired-ab-report.v1',run_id=config.get('run_id'),
        primary_comparison='B_vs_A',formal_pairing=formal_pairing,diagnostic_pairing=diag_pairing,
        groups=groups,comparison_groups=comparison_groups,media=media_records,training_accounting=accounting,
        evaluation_accounting=evaluation_accounting,
        strong_OC_ids=sorted(strong),frozen_bitstatic_case_ids=sorted(static),
        tie_tolerance=TIE_EPSILON,metric_directions=FIELDS,limits=limitations,
        saved_only=True,report_model_calls=0,report_physics_calls=0)
    write(folder/'case-comparisons.json',dict(rows=case_pairs,tie_tolerance=TIE_EPSILON,primary_comparison='B_vs_A'),root,config)
    write(folder/'summary.json',summary,root,config)
    # Human report emphasizes matched A/B, with full1..16 and per-case data linked.
    selected=lambda **filters:[g for g in groups if all(g.get(k)==v for k,v in filters.items()) and g['horizon'] in HIGHLIGHTS]
    comparisons=lambda **filters:[g for g in comparison_groups if all(g.get(k)==v for k,v in filters.items()) and g['horizon'] in HIGHLIGHTS]
    lines=['# Four-step generated-history A/B: saved paired results','',
        'Common start: OC new-data epoch2. A uses true history for all four targets; B feeds back its current detached predictions for the next losses. The primary contrast is B versus A.','',
        f'Supervised targets total: {engine_result.get("supervised_targets_total")}; optimizer batches total: {engine_result.get("optimizer_batches_total")}; physical steps: {engine_result.get("physical_steps")}.',
        'Supervision is matched. B has additional generation computation; compare actual timing/accounting in summary.json.','',
        '## Formal48: all six letters','']
    lines+=_markdown_table(*_table(selected(scope='formal48',subset='ALL')))
    lines+=['','## Primary paired comparison: B versus A','']+_markdown_table(*_comparison_table(comparisons(scope='formal48',subset='ALL',comparison='B_vs_A')))
    for subset in ('O_strong4','C_strong4','B','T','L','X','frozen_bitstatic'):
        lines+=['',f'## {subset}','']+_markdown_table(*_table(selected(scope='formal48',subset=subset)))
        lines+=['']+_markdown_table(*_comparison_table(comparisons(scope='formal48',subset=subset,comparison='B_vs_A')))
    for split in ('train','seen_val'):
        for mode in ('real_history','autoregressive'):
            lines+=['',f'## Motion diagnostic32: {split} / {mode}','']+_markdown_table(*_table(selected(scope='diagnostic32',split=split,mode=mode,subset='ALL_OC')))
            lines+=['']+_markdown_table(*_comparison_table(comparisons(scope='diagnostic32',split=split,mode=mode,subset='ALL_OC',comparison='B_vs_A')))
    lines+=['','## Interpretation limits','']+['- '+text for text in limitations]
    lines+=['','## Same-case raw comparisons','']
    for item in media_records:
        lines.append(f'- [{item["scope"]} {item["split"]} {item["letter"]} {item["mode"]}]({item["board"]["path"]}) · [all16]({item["all16_board"]["path"]}) · [10fps video]({item["video"]["path"]})')
    lines+=['',f'[All1..16 grouped metrics and paired counts]({folder/"summary.json"}) · [Every paired case/metric]({folder/"case-comparisons.json"})']
    markdown='\n'.join(lines)+'\n';guard(root,config,len(markdown.encode()));(root/'REPORT.md').write_text(markdown,encoding='utf-8')
    chunks=['<!doctype html><meta charset="utf-8"><title>Generated-history A/B</title>',
        '<style>body{font:15px system-ui;max-width:1250px;margin:24px auto;padding:0 20px}td,th{border:1px solid #ccc;padding:5px;font-size:12px}table{border-collapse:collapse}.table{overflow-x:auto}img,video{max-width:100%}details{margin:18px 0}p,li{line-height:1.5}</style>',
        '<h1>Four-step generated-history A/B</h1><p>Primary comparison: B versus A. Shared epoch2 start, same supervised targets/order/noise. B spends additional generation compute.</p>',
        '<h2>Formal48: all letters</h2>',_html_table(*_table(selected(scope='formal48',subset='ALL'))),
        '<h2>Primary B versus A: all paired cases retained</h2>',_html_table(*_comparison_table(comparisons(scope='formal48',subset='ALL',comparison='B_vs_A')))]
    for subset in ('O_strong4','C_strong4','B','T','L','X','frozen_bitstatic'):
        chunks+=['<details><summary>'+html.escape(subset)+'</summary>',_html_table(*_table(selected(scope='formal48',subset=subset))),_html_table(*_comparison_table(comparisons(scope='formal48',subset=subset,comparison='B_vs_A'))),'</details>']
    for split in ('train','seen_val'):
        for mode in ('real_history','autoregressive'):
            chunks+=[f'<h2>Motion diagnostic: {split} / {mode}</h2>',_html_table(*_table(selected(scope='diagnostic32',split=split,mode=mode,subset='ALL_OC'))),_html_table(*_comparison_table(comparisons(scope='diagnostic32',split=split,mode=mode,subset='ALL_OC',comparison='B_vs_A')))]
    # Literal per-case H16 tables remain usable when opened directly from disk.
    for scope in ('formal48','diagnostic32'):
        rows=[]
        for case in case_pairs:
            if case['scope']==scope and case['horizon']==16 and case['comparison']=='B_vs_A':
                values=[case['sample_id'],case['split'],case['mode'],case['image_metric_direction']]
                for key in CORE:
                    value=case['metrics'][key];values.append(value['status']+' ('+('unknown' if value['delta'] is None else f'{value["delta"]:+.4f}')+')')
                rows.append(values)
        chunks+=[f'<details><summary>{scope}: every H16 B/A case</summary>',_html_table(['Case','Split','Input','Image directions','IoU delta','Position delta','Contour delta'],rows),'</details>']
    chunks+=['<h2>Interpret these comparisons carefully</h2><ul>']+['<li>'+html.escape(x)+'</li>' for x in limitations]+['</ul>']
    for item in media_records:
        chunks+=[f'<h2>{item["scope"]} / {item["split"]} / {item["letter"]} / {item["mode"]}</h2>',
                 '<p>'+html.escape(item['sample_id'])+' — '+html.escape(item.get('selection',''))+'</p>',
                 f'<img src="{Path(item["board"]["path"]).name}">',
                 f'<video controls preload="metadata" src="{Path(item["video"]["path"]).name}"></video>',
                 f'<details><summary>All sixteen individual target times</summary><img src="{Path(item["all16_board"]["path"]).name}"></details>']
    chunks+=['<p><a href="summary.json">All1..16 groups, median and tails, paired denominators</a> · <a href="case-comparisons.json">Every paired case and metric</a></p>']
    page='\n'.join(chunks);guard(root,config,len(page.encode()));(folder/'index.html').write_text(page,encoding='utf-8')
    manifest=dict(kind='generated-history-ab-report-manifest.v1',inputs=inputs,
        outputs=[dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size) for path in sorted(folder.iterdir()) if path.is_file() and path.name!='final-manifest.json'],
        report=dict(path=str(root/'REPORT.md'),sha256=sha(root/'REPORT.md')),
        report_model_calls=0,report_physics_calls=0)
    write(folder/'final-manifest.json',manifest,root,config)
    return dict(report=str(root/'REPORT.md'),index=str(folder/'index.html'),summary=str(folder/'summary.json'),
                cases=str(folder/'case-comparisons.json'),manifest=str(folder/'final-manifest.json'),media=media_records,
                report_model_calls=0,report_physics_calls=0)
