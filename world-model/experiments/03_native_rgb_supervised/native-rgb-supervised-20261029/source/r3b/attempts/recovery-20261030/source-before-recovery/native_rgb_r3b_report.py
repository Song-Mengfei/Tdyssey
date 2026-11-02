"""Local paired R3B report from frozen raw predictions; no model or physics."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

METRICS=('object_roi_mae','gray_iou','gray_contour_error_px','gray_visible_centroid_error_px',
    'gray_centroid_error_px','rotation_error_deg','gray_drift_from_last_px',
    'blue_pusher_iou','blue_pusher_centroid_error_px','blue_pusher_roi_mae')
HIGHER_BETTER={'gray_iou','blue_pusher_iou'}
METHODS=('copy_last','t0','epoch1')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    from .native_rgb_training import atomic_json
    atomic_json(path,value)
def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='')as stream:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(stream,fieldnames=fields or['no_rows']);writer.writeheader();writer.writerows(rows)
def finite(value):return isinstance(value,(int,float))and math.isfinite(value)
def mean(values):return sum(values)/len(values)if values else None
def fmt(value,digits=4):return 'N/A'if value is None else f'{value:.{digits}f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows])


def summarize(rows):
    n=len(rows);shapes=Counter(r['shape_status']for r in rows)
    result=dict(planned_denominator=n,completed=sum(r['status']=='completed'for r in rows),
        shape_pass=shapes['pass'],shape_fail=shapes['fail'],shape_unknown=shapes['unknown'],
        shape_pass_fraction_full_denominator=shapes['pass']/n if n else None,
        shape_unknown_reasons=dict(Counter(str(r['metrics'].get('shape_unknown_reason'))for r in rows if r['shape_status']=='unknown')),
        physical_measurable=sum(r['metrics'].get('physical_measurement_status')=='ok'for r in rows),metrics={})
    for name in METRICS:
        values=[r['metrics'].get(name)for r in rows if finite(r['metrics'].get(name))]
        result['metrics'][name]=dict(mean_available=mean(values),available=len(values),unavailable=n-len(values),full_planned_denominator=n)
    return result


def paired(rows,baseline,metric):
    by={method:{r['window_id']:r for r in rows if r['method']==method}for method in(baseline,'epoch1')}
    keys=sorted(set(by[baseline])|set(by['epoch1']));pairs=[];wins=ties=losses=0
    for key in keys:
        left=by[baseline].get(key);right=by['epoch1'].get(key)
        a=None if left is None else left['metrics'].get(metric);b=None if right is None else right['metrics'].get(metric)
        if not(finite(a)and finite(b)):continue
        delta=b-a;better=delta>1e-12 if metric in HIGHER_BETTER else delta< -1e-12
        tie=abs(delta)<=1e-12
        wins+=int(better);ties+=int(tie);losses+=int(not better and not tie);pairs.append((a,b,delta))
    return dict(full_planned_denominator=len(keys),paired_available=len(pairs),paired_unavailable=len(keys)-len(pairs),
        mean_baseline_available=mean([a for a,b,d in pairs]),mean_epoch1_available=mean([b for a,b,d in pairs]),
        mean_delta_available=mean([d for a,b,d in pairs]),wins=wins,ties=ties,losses=losses,
        better_direction='higher'if metric in HIGHER_BETTER else'lower')


def _selection(rows,cohort,horizon,subgroup):
    selected=[r for r in rows if cohort in r['cohorts']and r['horizon']==horizon]
    if subgroup.startswith('letter_'):selected=[r for r in selected if r['letter']==subgroup[-1]]
    elif subgroup=='obvious_motion':selected=[r for r in selected if r['obvious_motion']]
    elif subgroup=='small_motion':selected=[r for r in selected if not r['obvious_motion']and not r['nearstatic_aux']]
    elif subgroup=='nearstatic_aux':selected=[r for r in selected if r['nearstatic_aux']]
    return selected


def build_numeric(t0,epoch1,spec):
    if not t0.get('evaluation_complete')or not epoch1.get('evaluation_complete'):raise ValueError('Full paired evaluations must complete before a model-quality report')
    rows={}
    for payload in(t0,epoch1):
        for row in payload['rows']:
            key=(row['window_id'],row['horizon'],row['method'])
            if key in rows:
                if row['method']!='copy_last'or rows[key]['metrics']!=row['metrics']:
                    raise ValueError('Repeated method/window metrics differ; no silent pairing')
            else:rows[key]=row
    rows=list(rows.values())
    for cohort,expected in(('natural',144),('stress',16),('old4',4)):
        for horizon in(1,4):
            for method in METHODS:
                actual=[r for r in rows if cohort in r['cohorts']and r['horizon']==horizon and r['method']==method]
                if len(actual)!=expected or len({r['window_id']for r in actual})!=expected:
                    raise ValueError(f'Full {cohort}/{method}/H{horizon} denominator differs: {len(actual)}')
    summaries=[];pairs=[];transitions=[]
    subgroups=('all',*[f'letter_{l}'for l in'TLOCXB'],'obvious_motion','small_motion','nearstatic_aux')
    for cohort in('natural','stress','old4'):
        for horizon in(1,4):
            for subgroup in subgroups:
                subset=_selection(rows,cohort,horizon,subgroup)
                for method in METHODS:
                    summaries.append(dict(cohort=cohort,horizon=horizon,subgroup=subgroup,method=method,
                        **summarize([r for r in subset if r['method']==method])))
                for baseline in('t0','copy_last'):
                    for metric in METRICS:pairs.append(dict(cohort=cohort,horizon=horizon,subgroup=subgroup,
                        baseline=baseline,metric=metric,**paired(subset,baseline,metric)))
                    before={r['window_id']:r['shape_status']for r in subset if r['method']==baseline}
                    after={r['window_id']:r['shape_status']for r in subset if r['method']=='epoch1'}
                    counts=Counter((before.get(k,'unknown'),after.get(k,'unknown'))for k in set(before)|set(after))
                    transitions.append(dict(cohort=cohort,horizon=horizon,subgroup=subgroup,baseline=baseline,
                        full_planned_denominator=len(set(before)|set(after)),counts={a+'->'+b:n for(a,b),n in counts.items()}))
    return rows,dict(summaries=summaries,paired_metrics=pairs,shape_transitions=transitions,
        natural_planned_windows=144,stress_planned_windows=16,old4_planned_windows=4,
        natural_selection=spec['natural_selection'],old4_is_regression_not_independent_validation=True)


def _display(raw):
    import numpy as np
    value=np.asarray(raw)
    if value.shape!=(3,96,96)or not np.isfinite(value).all():return None,dict(displayable=False,reason='missing/nonfinite/raw-shape')
    if value.dtype==np.uint8:return value.transpose(1,2,0).copy(),dict(displayable=True,display_clipped=False)
    if value.dtype!=np.float32:raise ValueError('Stored predictions must remain float32')
    clipped=bool((value<0).any()or(value>1).any())
    return np.rint(np.clip(value,0,1)*255).astype(np.uint8).transpose(1,2,0),dict(displayable=True,display_clipped=clipped,
        raw_min=float(value.min()),raw_max=float(value.max()),raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _font(size):
    from PIL import ImageFont
    for path in('C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def make_panels(stage,out,spec,rows,status_cb=None):
    import numpy as np
    from PIL import Image,ImageDraw
    from .native_rgb_r3b_evaluation import load_case_images,CALIBRATION
    from .native_rgb_evaluation import palette_masks
    calibration=read(CALIBRATION);look={(r['window_id'],r['horizon'],r['method']):r for r in rows}
    panel_ids=list(dict.fromkeys([*spec['report_panel_ids']['natural_O'],*spec['report_panel_ids']['natural_C'],*spec['report_panel_ids']['old4']]))
    records=[]
    for number,key in enumerate(panel_ids):
        before=load_case_images(stage,'t0',key);after=load_case_images(stage,'epoch1',key);meta=before['window']
        for horizon in(1,4):
            truth=before[f'truth_h{horizon}'];last=before['last'];methods=[truth,last,before[f'prediction_h{horizon}'],after[f'prediction_h{horizon}']]
            arrays=[];notes=[]
            for raw in methods:
                if raw is None:arrays.append(None);notes.append(dict(displayable=False,reason='unavailable'))
                else:image,note=_display(raw);arrays.append(image);notes.append(note)
            labels=['GT','Copy last','6000 (t0)','Epoch1'];scale=3;width=4*(96*scale+12)+12;height=96*scale+104
            canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
            draw.text((12,5),f'{key} | H{horizon} | '+','.join(meta['cohorts']),fill='black',font=_font(16))
            for i,(image,label)in enumerate(zip(arrays,labels)):
                x=12+i*(96*scale+12);draw.text((x,30),label,fill='black',font=_font(16))
                if image is None:draw.text((x+15,120),'N/A: unobservable',fill='red',font=_font(14))
                else:canvas.paste(Image.fromarray(np.repeat(np.repeat(image,scale,axis=0),scale,axis=1)),(x,54))
                if i:
                    record=look[(key,horizon,METHODS[i-1])];metric=record['metrics']
                    draw.text((x,346),f"ROI {fmt(metric.get('object_roi_mae'))}; IoU {fmt(metric.get('gray_iou'),3)}",fill='black',font=_font(12))
                    draw.text((x,364),'Shape: '+record['shape_status']+(' | display clipped'if notes[i].get('display_clipped')else''),fill='red'if record['shape_status']!='pass'else'black',font=_font(12))
            path=out/f'panel-{number:02d}-H{horizon}.png';canvas.save(path,compress_level=9)
            item=dict(window_id=key,letter=meta['letter'],cohorts=meta['cohorts'],horizon=horizon,file=path.name,
                sha256=sha(path),bytes=path.stat().st_size,source_RGB_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
                source_raw_refs=dict(t0=before['raw_references'],epoch1=after['raw_references']),display_notes=notes,
                pixel_policy='Same coordinates, original RGB; integer nearest display only. No registration, smoothing, warp or shape repair.')
            if meta['letter']in'OC':
                mask,_=palette_masks(truth,calibration);lastmask,_=palette_masks(last,calibration);ys,xs=np.nonzero(mask|lastmask)
                if len(xs):x0=max(0,int(xs.min())-4);x1=min(96,int(xs.max())+5);y0=max(0,int(ys.min())-4);y1=min(96,int(ys.max())+5)
                else:x0=y0=0;x1=y1=96
                zoomscale=6;cw=(x1-x0)*zoomscale;ch=(y1-y0)*zoomscale
                zoom=Image.new('RGB',(4*(cw+12)+12,ch+82),'white');zd=ImageDraw.Draw(zoom)
                zd.text((10,4),f'{key} H{horizon} | O/C GT-defined identical integer crop; no image alignment',fill='black',font=_font(14))
                for i,(image,label)in enumerate(zip(arrays,labels)):
                    x=12+i*(cw+12);zd.text((x,30),label,fill='black',font=_font(14))
                    if image is not None:zoom.paste(Image.fromarray(np.repeat(np.repeat(image[y0:y1,x0:x1],zoomscale,axis=0),zoomscale,axis=1)),(x,54))
                    else:zd.text((x+5,80),'N/A',fill='red',font=_font(14))
                zp=out/f'corners-{number:02d}-H{horizon}.png';zoom.save(zp,compress_level=9)
                item.update(closeup_file=zp.name,closeup_sha256=sha(zp),closeup_bytes=zp.stat().st_size,
                    closeup_crop=[x0,y0,x1,y1],integer_nearest_scale=zoomscale)
            records.append(item)
        if status_cb:status_cb('report_panel_saved',dict(completed_windows=number+1,planned_windows=len(panel_ids),window_id=key))
    return records


def build_report(stage,*,physics_source=None,status_cb=None):
    stage=Path(stage).resolve();out=stage/'report';out.mkdir(parents=True,exist_ok=True)
    spec=read(stage/'evaluation/spec.json');t0=read(stage/'evaluation/t0/metrics.json');epoch1=read(stage/'evaluation/epoch1/metrics.json')
    if t0['spec_sha256']!=sha(stage/'evaluation/spec.json')or epoch1['spec_sha256']!=t0['spec_sha256']:
        raise ValueError('Paired evaluation spec differs')
    rows,result=build_numeric(t0,epoch1,spec)
    pipeline=read(stage/'background/pipeline-state.json');progress=pipeline.get('formal_progress',{})
    if progress.get('stage_attempts')!=1499 or progress.get('formal_draws')!=23984 or not progress.get('epoch_complete'):
        raise ValueError('Actual complete first epoch and all23984 consumed targets required')
    summaries=result['summaries'];lookup={(q['cohort'],q['horizon'],q['subgroup'],q['method']):q for q in summaries}
    def metric(cohort,horizon,subgroup,method,name):return lookup[cohort,horizon,subgroup,method]['metrics'][name]
    panels=make_panels(stage,out,spec,rows,status_cb)
    result.update(kind='R3B-first-epoch-paired-report.v1',training_progress=progress,model_accounting=pipeline.get('model_accounting'),
        engineering_probe=pipeline.get('engineering_probe'),spec_sha256=t0['spec_sha256'],panels=panels,
        source_metrics=[dict(path=str(p),sha256=sha(p))for p in(stage/'evaluation/t0/metrics.json',stage/'evaluation/epoch1/metrics.json')],
        model_calls_in_report=0,physics_steps_in_report=0,next_epoch_authorized=False,awaiting_explicit_user_approval=True,
        shape_pass_meaning='No detected failure under frozen tests; unknown/occlusion/unmeasurable is never a pass, and a pass does not prove general rigidity.')
    write(out/'summary.json',result);csv_write(out/'paired-metrics.csv',result['paired_metrics'])
    flat=[]
    for q in summaries:
        row={k:v for k,v in q.items()if k not in('metrics','shape_unknown_reasons')}
        for name,value in q['metrics'].items():
            for field,v in value.items():row[name+'_'+field]=v
        row['shape_unknown_reasons']=json.dumps(q['shape_unknown_reasons'],ensure_ascii=False);flat.append(row)
    csv_write(out/'summary.csv',flat)
    shape_rows=[dict(window_id=r['window_id'],cohorts=','.join(r['cohorts']),letter=r['letter'],horizon=r['horizon'],method=r['method'],
        status=r['status'],shape_status=r['shape_status'],shape_unknown_reason=r['metrics'].get('shape_unknown_reason'),
        physical_status=r['metrics'].get('physical_measurement_status'),physical_reason=r['metrics'].get('physical_measurement_reason'),
        predicted_holes=r['metrics'].get('predicted_holes'),truth_holes=r['metrics'].get('truth_holes'),
        extra_fragments=r['metrics'].get('extra_fragments'),missing_components=r['metrics'].get('missing_components'),
        raw_corner_counts=json.dumps(r['metrics'].get('predicted_raw_corner_counts')))for r in rows]
    csv_write(out/'shape-and-observability.csv',shape_rows)
    natural=[]
    for subgroup in('all',*[f'letter_{l}'for l in'TLOCXB']):
        values=[lookup['natural',4,subgroup,m]for m in METHODS]
        natural.append([subgroup.removeprefix('letter_'),values[0]['planned_denominator'],*[fmt(q['metrics']['object_roi_mae']['mean_available'])for q in values],
            ' / '.join(fmt(q['metrics']['gray_iou']['mean_available'],3)for q in values[1:]),
            ' -> '.join(f"{q['shape_pass']}/{q['shape_fail']}/{q['shape_unknown']}"for q in values[1:])])
    h1=lookup['natural',1,'all','epoch1'];h4=lookup['natural',4,'all','epoch1'];base=lookup['natural',4,'all','t0'];copy=lookup['natural',4,'all','copy_last']
    e=h4['metrics']['object_roi_mae']['mean_available'];b=base['metrics']['object_roi_mae']['mean_available'];c=copy['metrics']['object_roi_mae']['mean_available']
    quality=('本轮自然 H4 灰物体 ROI 误差低于同窗口 6000 和 copy-last。'if finite(e)and finite(b)and finite(c)and e<b and e<c else'本轮自然 H4 灰物体 ROI 误差未同时低于同窗口 6000 和 copy-last。')
    oc=[]
    for letter in'OC':
        before=lookup['natural',4,'letter_'+letter,'t0'];after=lookup['natural',4,'letter_'+letter,'epoch1']
        oc.append(f"{letter}: ROI {fmt(before['metrics']['object_roi_mae']['mean_available'])}→{fmt(after['metrics']['object_roi_mae']['mean_available'])}，灰 IoU {fmt(before['metrics']['gray_iou']['mean_available'],3)}→{fmt(after['metrics']['gray_iou']['mean_available'],3)}；形状未检出失败/失败/未知 {before['shape_pass']}/{before['shape_fail']}/{before['shape_unknown']}→{after['shape_pass']}/{after['shape_fail']}/{after['shape_unknown']}（分母24）。")
    near=[]
    for horizon in(1,4):
        qs=[lookup['natural',horizon,'nearstatic_aux',m]for m in METHODS]
        near.append([f'H{horizon}',qs[0]['planned_denominator'],*[fmt(q['metrics']['gray_drift_from_last_px']['mean_available'])for q in qs]])
    report=f'''# R3B 完整首轮与配对评测

**已完成本次批准的首轮 1,499 批、23,984 个不同目标窗口；没有进入第二轮。**
正式更新成功 {progress.get('stage_successes')}，AMP skip {progress.get('AMP_skips')}；旧模型 lineage6000 与本阶段 probe/正式计数分开。
仅使用原 R3A，CP1/CP2未混入，八个 heldout 字母未访问。LR保持1e-5、clean/weak各50%micro、micro4×4有效16。

{quality} 这是一轮固定窗口结果，不能据此宣称收敛、形状刚性已解决、模型无效或继续训练无用。

自然验证每个144seen-val family预先均匀抽取一个合法H4anchor，H1/H4同anchor；16stress和旧4回归单列。
H4仅反馈原始预测。表中误差越低越好、IoU越高越好；形状为未检出失败/失败/未知，未知保留完整分母。

{table(['自然H4','N','Copy ROI','6000 ROI','Epoch1 ROI','6000/Epoch1灰IoU','6000→Epoch1形状P/F/U'],natural)}

{' '.join(oc)}

自然H1 epoch1 ROI误差 {fmt(h1['metrics']['object_roi_mae']['mean_available'])}，灰IoU {fmt(h1['metrics']['gray_iou']['mean_available'],3)}。
H4 epoch1物理位姿可测 {h4['physical_measurable']}/{h4['planned_denominator']}；转角误差只在可测且按形状固有对称周期的样本上计算。
遮挡、越界和不可测不会算通过。棱角数/孔洞/碎片/面积/轮廓与蓝推杆逐项见[完整表](report/summary.csv)、[形状与可测性](report/shape-and-observability.csv)。

{table(['自然近静止组','N','Copy drift px','6000 drift px','Epoch1 drift px'],near)}

O/C图和旧O/B/C/X回归均为GT/copy-last/6000/epoch1同坐标并排。近景只作同一GT定义整数crop和最近邻放大；
不配准、不平滑、不修边。预览如需限幅会标注，原float32和评分保持原值。

原6000 checkpoint只读保留；本阶段单份滚动完整last.pt保留Adam/scaler、实际sampler cursor及三条conditioning RNG。
仅报告本轮改善/退步及剩余形状问题，不自动续训或更改采样。后续实施必须等待用户明确审批。

[O/C与旧四例图](report/index.html) · [配对逐指标胜/平/负与有效分母](report/paired-metrics.csv) ·
[机器结果](result.json) · [原始float32索引t0](evaluation/t0/index.json) · [epoch1索引](evaluation/epoch1/index.json)
'''
    (stage/'REPORT.md').write_text(report,encoding='utf-8')
    details=['# R3B detailed fixed-cohort summaries','',quality,'']
    for cohort in('natural','stress','old4'):
        details+=['## '+cohort,'',table(['H','method','N','ROI MAE','gray IoU','contour px','centroid px available/N','angle deg available/N','shape P/F/U','blue pusher MAE'],[
            [h,m,q['planned_denominator'],fmt(q['metrics']['object_roi_mae']['mean_available']),fmt(q['metrics']['gray_iou']['mean_available'],3),
             fmt(q['metrics']['gray_contour_error_px']['mean_available']),
             fmt(q['metrics']['gray_centroid_error_px']['mean_available'])+f" ({q['metrics']['gray_centroid_error_px']['available']}/{q['planned_denominator']})",
             fmt(q['metrics']['rotation_error_deg']['mean_available'])+f" ({q['metrics']['rotation_error_deg']['available']}/{q['planned_denominator']})",
             f"{q['shape_pass']}/{q['shape_fail']}/{q['shape_unknown']}",fmt(q['metrics']['blue_pusher_roi_mae']['mean_available'])]
            for h in(1,4)for m in METHODS for q in[lookup[cohort,h,'all',m]]]),'']
    details+=['Shape pass means no detected failure under the frozen tests, not proven rigidity. Unknown and metric-unavailable counts are explicit. Stress and old4 do not enter natural means. See summary.csv for each letter, motion and nearstatic subgroup.']
    (out/'DETAILS.md').write_text('\n'.join(details)+'\n',encoding='utf-8')
    cards=[]
    for p in panels:
        cards.append(f'<section><h2>{html.escape(p["window_id"])} H{p["horizon"]}</h2><img src="{p["file"]}" alt="Same-coordinate raw prediction comparison">'+
            (f'<h3>O/C integer nearest closeup</h3><img src="{p["closeup_file"]}">'if'closeup_file'in p else'')+'</section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R3B paired O/C and regressions</title><style>body{font:15px system-ui;margin:24px;background:#eee}section{background:white;padding:16px;margin:16px 0}img{max-width:100%;image-rendering:pixelated}</style><h1>R3B fixed paired results</h1><p>GT / copy last / original6000 / epoch1. Natural panels were frozen before training. No registration, smoothing or shape repair.</p><p><a href="DETAILS.md">Detailed metrics</a> · <a href="summary.csv">All groups</a> · <a href="shape-and-observability.csv">Shape/unknown</a></p>'+''.join(cards),encoding='utf-8')
    final=dict(kind='R3B-first-epoch-result.v1',complete=True,training_progress=progress,
        natural_H1_epoch1=h1,natural_H4_copy_last=copy,natural_H4_t0=base,natural_H4_epoch1=h4,
        interpretation=quality,rigidity_proven=False,report_path=str(stage/'REPORT.md'),gallery_path=str(out/'index.html'),
        detailed_summary_path=str(out/'summary.json'),next_epoch_authorized=False,awaiting_explicit_user_approval=True,
        checkpoint_path=str(stage/'run/last.pt'),checkpoint_sha256=sha(stage/'run/last.pt'),
        callback_required=True,callback_sent_by_reporter=False,model_calls_in_report=0,physics_steps_in_report=0)
    write(stage/'result.json',final)
    write(out/'inventory.json',dict(kind='R3B-report-inventory.v1',files=[dict(file=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.iterdir())if p.is_file()and p.name!='inventory.json'],
        pixels_repaired=False,model_calls=0,physics_steps=0))
    if status_cb:status_cb('report_complete',dict(report_path=str(stage/'REPORT.md'),gallery_path=str(out/'index.html'),natural144=True))
    return dict(report_path=str(stage/'REPORT.md'),result_path=str(stage/'result.json'),gallery_path=str(out/'index.html'),
        natural_H4_object_roi_mae=e,natural_H4_shape=dict(pass_count=h4['shape_pass'],fail_count=h4['shape_fail'],unknown_count=h4['shape_unknown'],denominator=144),
        no_model_or_physics_in_report=True,next_epoch_started=False)
