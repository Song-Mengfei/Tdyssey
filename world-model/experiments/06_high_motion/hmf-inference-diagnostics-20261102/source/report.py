"""Small saved-result A/B report, called by the sole BelowNormal worker.

No model, optimizer, physics, archive hashing, video encoding or new sampling.
Old0/final/train12 metrics and media are references, never rescored here.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import html
import json
import math
import os
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
HMF=ROOT.parent/'high-motion-fit-20261101'
LABELS=('0','250','500','1000','2000','final')
H=(1,4,8,16)
KEYS=('raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px','pusher_position_error_image_px')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    from common import atomic_json
    atomic_json(p,v)
def guard(root,c,n):
    from common import check_storage
    check_storage(root,c,reserve_bytes=int(n))
def stats(values):
    a=np.array([float(v) for v in values if v is not None and math.isfinite(v)],float)
    return dict(denominator=len(values),available=len(a),unknown=len(values)-len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,P10=float(np.percentile(a,10)) if len(a) else None,
        P90=float(np.percentile(a,90)) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)
def _label(record):
    if 'label' in record:return str(record['label'])
    value=record.get('checkpoint',{});return str(value.get('label')) if isinstance(value,dict) else str(value)
def _item(entry):return read(entry.get('score_path',entry.get('metrics')))

def _load_A(root,aspec,entries,bspec):
    """Exactly39 comparable cases/stage: existingtrain12 +27 A cases."""
    oldspec=read(HMF/'evaluation-spec.json');panel={r['sample_id'] for b in oldspec['batches'] if b['batch_id'] in oldspec['panel_batch_ids'] for r in b['rows']}
    a_ids={r['sample_id'] for r in aspec['rows']};wanted=panel|a_ids
    rows={};refs={};copies={};sources=[];B_copy={}
    for label in LABELS:
        path=Path(aspec['existing_HMF_metrics'][label]);value=read(path)
        rows[label]=[dict(r,checkpoint=label,score_origin='reused_original_HMF') for r in value['rows'] if r['sample_id'] in wanted]
        refs[label]={(r['sample_id'],r['input_mode']):dict(r,array_key='raw',raw_index=None) for r in value['raw_files'] if r['sample_id'] in wanted}
        if label=='0':
            copies[label]=[r for r in value['copy_rows'] if r['sample_id'] in wanted]
            bykey={(r['sample_id'],r['input_mode'],r['horizon']):r for r in value['copy_rows']}
            B_copy={r['sample_id']:bykey[(r['original_sample_id'],'real_history',int(r['parent_phase'])+1)] for r in bspec['rows']}
        sources.append(dict(path=str(path),kind='Existing metrics reference; no model call, rescoring or archive rehash.'))
    score_cost=[]
    for entry in entries:
        record=entry['record'];label=_label(record);value=_item(entry);score_cost.append(value['scoring'])
        rows[label].extend(dict(r,score_origin='newly_filled_missing_midpoint') for r in value['rows'])
        for rf in record['raw_files']:
            refs[label][(rf['sample_id'],record['mode'])]=dict(rf,array_key=rf.get('array_key','raw'))
    for label in LABELS:
        index={(r['sample_id'],r['input_mode'],r['horizon']):r for r in rows[label]}
        expected={(sid,mode,h) for sid in wanted for mode in ('real_history','autoregressive') for h in range(1,17)}
        if set(index)!=expected or len(index)!=len(rows[label]):raise ValueError('A incomplete/duplicate timeline or unexpectedtrain48 scope')
    rows['copy']=[dict(r,checkpoint='copy',score_origin='reused_original_HMF_copy_reference') for r in copies['0']]
    return rows,refs,copies,panel,sources,score_cost,B_copy

def _A_groups(rows,panel):
    result=[]
    for label in ('copy',*LABELS):
        for role in ('fixed_train12','seen_val','original_failure'):
            for letter in ('O','C','B','ALL'):
                for mode in ('real_history','autoregressive'):
                    rr=[r for r in rows[label] if ((r['sample_id'] in panel) if role=='fixed_train12' else r['selection_role']==role) and r['input_mode']==mode and (letter=='ALL' or r['letter']==letter)]
                    subsets=[('H'+str(h),[r for r in rr if r['horizon']==h]) for h in range(1,17)]
                    if mode=='real_history':subsets +=[('strong_H1',[r for r in rr if r['strong_H1']]),('transition_H1',[r for r in rr if not r['strong_H1']])]
                    for subset,qq in subsets:
                        if not qq:continue
                        result.append(dict(checkpoint=label,role=role,letter=letter,mode=mode,subset=subset,
                            families=len({r['family_id'] for r in qq}),targets=len(qq),metrics={k:stats([r['metrics'].get(k) for r in qq]) for k in KEYS},
                            rotation=dict(Counter(r['rotation']['status'] for r in qq if r['rotation']['required'])),
                            direction=dict(Counter(r['rotation']['classification'] for r in qq if r['rotation']['required'])),
                            visible_shape=dict(Counter(r['visible_shape_status'] for r in qq)),topology=dict(Counter(r['visible_topology_status'] for r in qq)),
                            blue_missing=sum('empty_pusher' in r['metrics'].get('failure_reasons',[]) for r in qq)))
    return result

def _A_regressions(rows):
    indexes={label:{(r['sample_id'],r['input_mode'],r['horizon']):r for r in rr} for label,rr in rows.items()};out=[]
    pairs=[(a,b) for a,b in zip(LABELS,LABELS[1:])]+[('0',label) for label in LABELS[1:]]
    for before,after in pairs:
        for key,old in indexes[before].items():
            new=indexes[after][key];fields={}
            for metric in KEYS:
                a,b=old['metrics'].get(metric),new['metrics'].get(metric);delta=None if a is None or b is None else b-a
                status='unknown' if delta is None else 'tied' if delta==0 else 'improved' if (delta>0 if metric=='raw_mask_iou' else delta<0) else 'regressed'
                fields[metric]=dict(before=a,after=b,delta=delta,status=status)
            out.append(dict(before_checkpoint=before,after_checkpoint=after,sample_id=old['sample_id'],family_id=old['family_id'],letter=old['letter'],
                role=old['selection_role'],mode=old['input_mode'],horizon=old['horizon'],metrics=fields,after_rotation=new['rotation'],after_shape=new['visible_shape_status']))
    return out

def _raw(ref):
    with np.load(ref['path'],allow_pickle=False) as z:
        value=z[ref.get('array_key','raw')]
        return value.copy() if ref.get('raw_index') is None else value[int(ref['raw_index'])].copy()

def _truth(row):
    with np.load(row['raw_path'],allow_pickle=False) as z:
        a=int(row['anchor']);return z['rgb'][a:a+17].copy()

def _font(size=13):
    from PIL import ImageFont
    path=Path('C:/Windows/Fonts/msyh.ttc')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()

def _board(path,header,columns,horizons,captions=None):
    """Display quantization only. Never returned to model/scoring paths."""
    from PIL import Image,ImageDraw
    width=len(columns)*144;height=30+len(horizons)*174;im=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(im);font=_font()
    draw.text((3,2),header,font=font,fill='black')
    for line,h in enumerate(horizons):
        y=30+line*174
        for j,(label,images) in enumerate(columns):
            draw.text((j*144+3,y),label+f' H{h}',font=font,fill='black')
            a=np.rint(np.clip(images[h-1].transpose(1,2,0),0,1)*255).astype(np.uint8)
            im.paste(Image.fromarray(a).resize((144,144),Image.Resampling.NEAREST),(j*144,y+24))
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path,compress_level=1)

def _A_media(root,c,aspec,refs):
    """Six GT-frozen highlights, all27 per-case outcomes remain in JSON."""
    folder=root/'report/media';folder.mkdir(parents=True,exist_ok=True);records=[]
    chosen=[]
    for role in ('seen_val','original_failure'):
        for letter in ('O','C','B'):
            chosen.append(next(r for r in aspec['rows'] if r['selection_role']==role and r['letter']==letter))
    for row in chosen:
        truth=_truth(row).astype(np.float32)/np.float32(255)
        for mode in ('real_history','autoregressive'):
            copy=truth[:-1] if mode=='real_history' else np.repeat(truth[:1],16,axis=0)
            columns=[('GT',truth[1:]),('COPY',copy)]+[(label,_raw(refs[label][(row['sample_id'],mode)])) for label in LABELS]
            path=folder/(row['family_id']+'-'+mode+'-H1-H4-H8-H16.png');guard(root,c,3*1024**2)
            header='原seen-val未参加HMF拟合' if row['selection_role']=='seen_val' else '原失败展示例，未参加HMF拟合'
            header+=' | '+row['sample_id']+' | '+('RH独立单步，非闭环' if mode=='real_history' else 'AR原始自身历史16步')
            _board(path,header,columns,H);records.append(dict(sample_id=row['sample_id'],role=row['selection_role'],letter=row['letter'],mode=mode,path=str(path.resolve()),selection='First frozen originalcase perletter/role, not chosen by new predictionquality.'))
    return records

def _A_curves(root,c,groups):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'report/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=[]
    for role in ('seen_val','original_failure'):
        for mode in ('real_history','autoregressive'):
            fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
            for col,letter in enumerate(('O','C','B')):
                for label in ('copy',*LABELS):
                    rr={int(g['subset'][1:]):g for g in groups if g['role']==role and g['mode']==mode and g['letter']==letter and g['checkpoint']==label and g['subset'].startswith('H')}
                    for ax,key in ((axes[0,col],'raw_mask_iou'),(axes[1,col],'object_position_error_image_px')):
                        ax.plot(range(1,17),[rr[h]['metrics'][key]['mean'] for h in range(1,17)],label=label);ax.grid(alpha=.2)
                axes[0,col].set_title(letter+' | '+role);axes[0,col].set_ylabel('grayIoU');axes[1,col].set_ylabel('visiblecentroid error(px)');axes[1,col].set_xlabel('target horizon /10Hz')
            axes[0,0].legend(fontsize=8);fig.suptitle(role+' / '+mode+(' (independent one-step targets)' if mode=='real_history' else ' (rawautoregressive rollout)'));fig.tight_layout()
            path=root/'report'/f'A-{role}-{mode}-all16-curves.png';guard(root,c,3*1024**2);fig.savefig(path,dpi=115);plt.close(fig)
            result.append(dict(role=role,mode=mode,path=str(path.resolve())))
    return result

def _B_load(entries):
    """Only216new images; one bounded read per new batch, no archive hash."""
    images={};scored={};records={};cost=[]
    for entry in entries:
        record=entry['record'];label=_label(record);noise=record['noise_id'];action=record['action_type'];value=_item(entry);cost.append(value['scoring'])
        prediction=entry['prediction']
        with np.load(prediction['path'],allow_pickle=False) as z:raw=z[prediction.get('array_key','raw')].copy()
        for slot,row in enumerate(record['rows']):
            key=(label,row['sample_id'],noise,action)
            if key in images:raise ValueError('Duplicate sensitivitycell')
            images[key]=raw[slot,0];records[key]=row
        for row in value['rows']:scored[(label,row['sample_id'],noise,action)]=row
    if len(images)!=216 or set(images)!=set(scored):raise ValueError('Missing sensitivitypredictions/scores')
    return images,scored,records,cost

def _B_sensitivity(root,c,bspec,entries,copy_baseline):
    import metrics
    images,scored,records,cost=_B_load(entries);pair_rows=[];summaries=[];media=[];begin=time.perf_counter()
    import torch
    device=torch.device('cuda')
    actions=list(bspec['actions']);noises=sorted({k[2] for k in images});labels=list(bspec['checkpoint_labels'])
    numerical=('object_centroid_change_imagepx','object_mask_change','object_contour_change_imagepx','object_visible_region_rgb_mae_excluding_blue','blue_centroid_change_imagepx','blue_mask_change')
    for case in bspec['rows']:
        sid=case['sample_id'];identical=bool(case.get('recorded_hold_identical',False))
        for label in labels:
            def pair(kind,left,right):
                a,b=images[left],images[right];gpu=torch.as_tensor(a[None],device=device,dtype=torch.float32)
                value=metrics.pair_statistics(a,b,gpu=gpu,features_first=scored[left]['predicted_visible_features'],features_second=scored[right]['predicted_visible_features'])
                out=dict(case_id=sid,letter=case['letter'],checkpoint=label,kind=kind,left=dict(noise_id=left[2],action=left[3]),right=dict(noise_id=right[2],action=right[3]),metrics=value)
                pair_rows.append(out);return out
            noise_pairs={}
            for action in actions:
                noise_pairs[action]=[pair('same_action_different_noise',(label,sid,a,action),(label,sid,b,action)) for a,b in combinations(noises,2)]
            for a,b in combinations(actions,2):
                effects=[pair('same_noise_different_action',(label,sid,n,a),(label,sid,n,b)) for n in noises]
                is_identical=identical and {a,b}=={'recorded','hold'}
                measures={}
                for key in numerical:
                    effect=stats([x['metrics'][key] for x in effects]);noiseA=stats([x['metrics'][key] for x in noise_pairs[a]]);noiseB=stats([x['metrics'][key] for x in noise_pairs[b]])
                    scale=None if noiseA['mean'] is None or noiseB['mean'] is None else (noiseA['mean']+noiseB['mean'])/2
                    ratio=None if not scale or effect['mean'] is None else effect['mean']/scale
                    measures[key]=dict(action_effect=effect,same_action_noise_left=noiseA,same_action_noise_right=noiseB,
                        noise_reference_mean=scale,action_to_noise_ratio=ratio,ratio_undefined_when_noise_zero=scale==0)
                summaries.append(dict(case_id=sid,letter=case['letter'],checkpoint=label,action_pair=[a,b],recorded_hold_identical=identical,
                    identical_action_control=is_identical,eligible_as_weak_action_sensitivity_evidence=not is_identical,metrics=measures,
                    interpretation='Magnitude of visible object/blue response versus noise, not counterfactualGT accuracy or correctphysics.'))
        # All three noises and both models, withGT explicitly restricted torecorded.
        from PIL import Image,ImageDraw
        # Read only this savedstrip, not a simulator/re-render. Original anchor
        # is the selected currenthistoryendpoint, not the parenttrajectorystart.
        with np.load(case['raw_path'],allow_pickle=False) as z:a=int(case['anchor']);previous=z['rgb'][a].astype(np.float32)/255.;target=z['rgb'][a+1].astype(np.float32)/255.
        width=5*144;height=35+len(labels)*len(noises)*178;canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas);font=_font()
        title=case['letter']+' '+sid+' | GT仅适用于recorded；hold/other无反事实真值'
        if identical:title+=' | hold=recorded退化对照'
        draw.text((3,2),title,font=font,fill='black')
        line=0
        for label in labels:
            for noise in noises:
                yy=35+line*178;line+=1
                columns=[('realprevious',previous),('GTrecordedonly',target)]+[(action,images[(label,sid,noise,action)]) for action in actions]
                for col,(name,img) in enumerate(columns):
                    draw.text((col*144+3,yy),label+' n'+str(noise)+' '+name,font=font,fill='black')
                    a=np.rint(np.clip(img.transpose(1,2,0),0,1)*255).astype(np.uint8);canvas.paste(Image.fromarray(a).resize((144,144),Image.Resampling.NEAREST),(col*144,yy+25))
        path=root/'report/media'/('B-'+case['letter']+'-'+str(case.get('parent_phase',0))+'-'+str(bspec['rows'].index(case))+'.png');guard(root,c,3*1024**2);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path,compress_level=1)
        media.append(dict(case_id=sid,letter=case['letter'],path=str(path.resolve()),all_models_actions_noises_displayed=True,recorded_hold_identical=identical))
    original_accuracy=[x for key,x in scored.items() if key[3]=='recorded']
    for item in original_accuracy:
        reference=copy_baseline[item['sample_id']];item['copy_reference']=reference
        item['copy_reference_origin']='OriginalHMF0 savedreal-historycopy row at the same parentphase; no rescoring.'
        item['copy_comparison']={key:dict(prediction=item['metrics'].get(key),copy=reference['metrics'].get(key)) for key in KEYS}
    if any(x['metrics'] is not None for key,x in scored.items() if key[3]!='recorded'):raise ValueError('Counterfactual actions unexpectedly received GT accuracy scores')
    return dict(rows=summaries,raw_pairs=pair_rows,recorded_GT_accuracy=original_accuracy,media=media,
        same_action_hold_identical_cases=[r['sample_id'] for r in bspec['rows'] if r.get('recorded_hold_identical')],
        scoring_cost=cost,aggregation_and_static_board_wall_seconds=time.perf_counter()-begin,
        limits=['Onlyrecordedaction has same-stateGT. Hold/other images are sensitivity responses, not validated counterfactualphysics.',
            'Visible graycentroid can move through deformation/occlusion; qualified axes remain visible patterns, not a rigidpose/torque proof.',
            'Blue rod changes are separate and do not alone demonstrate object action use.',
            'When hold equals recorded, a zero response is a degenerate identical-actioncontrol, not evidence of weak conditioning.',
            'Three frozen noise draws describe these12cases, not an exhaustive causalidentification or statisticalpower study.'])

def _table(columns,rows):return ['| '+' | '.join(columns)+' |','|'+'|'.join('---' for _ in columns)+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows]
def generate_report(root,c,result):
    root=Path(root);folder=root/'report';folder.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    aspec=read(root/'A-spec.json');bspec=read(root/'B-spec.json');rows,refs,copies,panel,sources,A_cost,B_copy=_load_A(root,aspec,result['A_batches'],bspec)
    A_groups=_A_groups(rows,panel);regressions=_A_regressions(rows);write(folder/'A-all16-curves.json',A_groups);write(folder/'A-per-case-regressions.json',regressions)
    A_media=_A_media(root,c,aspec,refs);A_curves=_A_curves(root,c,A_groups)
    B=_B_sensitivity(root,c,bspec,result['B_batches'],B_copy);write(folder/'B-action-noise-comparisons.json',B)
    accounting={k:v for k,v in result.items() if k not in ('A_batches','B_batches')}
    summary=dict(kind='HMF-inference-diagnostics-A+B-report.v1',run_id=c.get('run_id'),A=dict(groups=A_groups,media=A_media,curves=A_curves,scoring_cost=A_cost,
        scopes=dict(fixed_train12='Existing12 at allcheckpoints. No newmiddletrain48.',seen_val='Sameoriginal24 families at all6checkpoints.',original_failure='Sameoriginal3 cases, originalB3slots at all6checkpoints.'),
        reused_sources=sources,new_missing_midpoint_predictions=3456,old_scores_not_rescored=True),B=B,accounting=accounting,
        report_wall_seconds=time.perf_counter()-begin,report_model_calls=0,report_physics_calls=0,new_video_encoders=0,
        limits=['OnlyA+B approved. No training, optimizer/backward, collection, physics, UI model switch or section10continuation.',
            'New midpoint validation was filled now, not measured during the original HMFtraining.',
            'Original scientific definitions/thresholds/unknowns retained. Skiponly originalprior-unqualified legacy numericattachments, with explicitnotcomputedreason.',
            'OriginalHMF support.stats.worst is maximum even forIoU. Thisreport uses min/P10/max explicitly and nevercalls thatfield worstIoU.',
            'BothRH andAR show all16targets. RH rereads realprecedinghistory and is not a closed-loop16step prediction.',
            'No old492raw rehash or full-library rescoring. Existingvideos are onlylinks, not re-encoded.'])
    write(folder/'summary.json',summary)
    table=[]
    for role in ('seen_val','original_failure','fixed_train12'):
        for letter in ('O','C','B'):
            for mode in ('real_history','autoregressive'):
                for h in H:
                    line=[role,letter,mode,'H'+str(h)]
                    for label in ('copy',*LABELS):
                        g=next(x for x in A_groups if x['role']==role and x['letter']==letter and x['mode']==mode and x['subset']=='H'+str(h) and x['checkpoint']==label)
                        a,b=g['metrics']['raw_mask_iou']['mean'],g['metrics']['object_position_error_image_px']['mean'];line.append(f'{a:.5f}/{b:.5f}')
                    table.append(line)
    lines=['# HMF：补中期验证与动作敏感性','',
        '只执行获批的A+B推理诊断。没有继续训练、改变物理或扩采；本报告不批准后续操作。',
        '','A：原seen-val24与原失败3例补250/500/1000/2000，0/final和中间固定train12原指标直接引用。所有16时刻及逐例退步见JSON；下表突出H1/4/8/16。数值为 grayIoU / 可见质心误差px。',
        '',*_table(['分母','字母','历史','时距','copy',*LABELS],table),
        '','B：同12真实历史、两模型、三动作、三噪声。仅recorded有同状态GT；hold/alternative只谈敏感性。物体响应与蓝杆响应分列，并与同动作换噪声的波动比较；hold=recorded时不拿零差异判断作用弱。',
        '','精确动作效应/噪声比、原动作GT准确率和unknown：[B-action-noise-comparisons.json](report/B-action-noise-comparisons.json)。',
        '','全16曲线：[A-all16-curves.json](report/A-all16-curves.json)；逐例退步：[A-per-case-regressions.json](report/A-per-case-regressions.json)。',
        '','固定图和原媒体索引：[index.html](report/index.html)。同一原始图像仅作显示量化，不修复/回灌。',
        '','调用数、额外/失败成本、单线程/BelowNormal实际回执与耗时见summary.accounting及原执行结果，不把计划数冒充完成数。','',*['- '+x for x in summary['limits']],*['- '+x for x in B['limits']]]
    report=root/'REPORT.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    elements=[]
    for x in [*A_curves,*A_media,*B['media']]:
        path=Path(x['path']);relative=path.relative_to(folder).as_posix();title=html.escape(x.get('case_id',x.get('sample_id',x.get('role','')))+' '+x.get('mode',''))
        elements.append('<details><summary>'+title+'</summary><img loading="lazy" style="max-width:100%" src="'+html.escape(relative)+'"></details>')
    page='<!doctype html><meta charset="utf-8"><title>HMF inference diagnostics</title><style>body{font:15px system-ui;margin:24px}details{margin:16px 0}</style><h1>补中期验证 + 动作敏感性</h1><p>无续训。RH不是闭环；替代动作无反事实GT。原始raw/JSON保留，不新编码视频。</p><p><a href="../REPORT.md">完整报告</a> | <a href="'+html.escape(str((HMF/'report/index.html').resolve()))+'">原HMF媒体（复用，不重编码）</a></p>'+''.join(elements)
    (folder/'index.html').write_text(page,encoding='utf-8')
    # Engine saved eachnewraw SHA once. Report deliberately doesnotrehash those
    # archives or oldbaseline; final root manifest can reuse executionreceipts.
    manifest=dict(kind='A+B-report-artifact-index.v1',complete=True,report=str(report.resolve()),summary=str((folder/'summary.json').resolve()),
        files=[str(p.resolve()) for p in (folder/'A-all16-curves.json',folder/'A-per-case-regressions.json',folder/'B-action-noise-comparisons.json',folder/'summary.json',folder/'index.html')],
        static_images=[x['path'] for x in [*A_curves,*A_media,*B['media']]],new_model_calls=0,new_video_encoders=0,
        prediction_archive_hash_source='Use engine once-only prediction receipts; no repeat hashing inreport.')
    write(folder/'final-manifest.json',manifest)
    return dict(complete=True,report=str(report.resolve()),summary=str((folder/'summary.json').resolve()),index=str((folder/'index.html').resolve()),
        manifest=str((folder/'final-manifest.json').resolve()),A_curves=str((folder/'A-all16-curves.json').resolve()),B_comparisons=str((folder/'B-action-noise-comparisons.json').resolve()),
        report_model_calls=0,new_video_encoders=0,wall_seconds=time.perf_counter()-begin)
