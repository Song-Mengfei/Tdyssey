"""Saved-only final report. Safe hashed names, original gates, no new inference."""
from common import *
from collections import Counter
from safe_names import reserve_panel
import html,io
LABELS=('E2','BASE','SNR','GDL','FFL')

def summarize(rows):
    import numpy as np
    def mean(k):
        v=[r['metrics'].get(k) for r in rows]
        return None if not v or any(x is None for x in v) else float(np.mean(v))
    return dict(n=len(rows),IoU=mean('raw_mask_iou'),position=mean('object_position_error_image_px'),pusher_position=mean('pusher_position_error_image_px'),
        topology=dict(Counter(x['visible_topology_status'] for x in rows)),shape=dict(Counter(x['visible_shape_status'] for x in rows)),
        rotation=dict(Counter(x['rotation']['status'] for x in rows)),precise_corners_proven=False)

def generate():
    import numpy as np
    from PIL import Image,ImageDraw
    c=read(ROOT/'frozen-config.json');runs={l:read(c['E2_metrics'] if l=='E2' else ROOT/f'evaluation/{l}/metrics.json') for l in LABELS}
    groups=[]
    for label,v in runs.items():
        for role in ('train','seen_val','original_failure'):
            for letter in ('O','C','B'):
                for mode in ('real_history','autoregressive'):
                    rr=[x for x in v['rows'] if x['selection_role']==role and x['letter']==letter and x['input_mode']==mode]
                    for h in range(1,17):groups.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,**summarize([x for x in rr if x['horizon']==h])))
                    if mode=='real_history':groups.append(dict(label=label,role=role,letter=letter,mode=mode,horizon='strong_H1',**summarize([x for x in rr if x['strong_H1']])))
    write(ROOT/'report/grouped-all16.json',groups)
    lookup={(x['label'],x['role'],x['letter'],x['mode'],x['horizon']):x for x in groups};checks=[]
    for arm in ('SNR','GDL','FFL'):
        for letter in ('O','C','B'):
            a=lookup['BASE','seen_val',letter,'autoregressive',4];b=lookup[arm,'seen_val',letter,'autoregressive',4]
            ra=lookup['BASE','seen_val',letter,'real_history','strong_H1'];rb=lookup[arm,'seen_val',letter,'real_history','strong_H1']
            conditions=dict(AR4_IoU=b['IoU']>a['IoU'],AR4_position=b['position']<=a['position'],RH_strong_IoU=rb['IoU']>=ra['IoU'],RH_strong_position=rb['position']<=ra['position'])
            topology=[dict(horizon=h,BASE=lookup['BASE','seen_val',letter,'autoregressive',h]['topology'],candidate=lookup[arm,'seen_val',letter,'autoregressive',h]['topology']) for h in (1,4,8,16)]
            conditions['topology_fail_not_increased']=all(x['candidate'].get('fail',0)<=x['BASE'].get('fail',0) for x in topology)
            checks.append(dict(arm=arm,letter=letter,numeric_directional_support=all(conditions.values()),conditions=conditions,topology=topology,
                               original_absolute_acceptance=runs[arm]['support']['seen_val']['all_passed']))
    write(ROOT/'report/comparison-checks.json',checks)
    pairing=all(v['evidence']==runs['E2']['evidence'] for v in runs.values());write(ROOT/'report/evaluation-pairing.json',dict(passed=pairing))
    if not pairing:raise RuntimeError('Evaluation pairing mismatch')
    key=lambda x:(x['sample_id'],x['input_mode'],x['horizon']);regressions=[]
    for arm in ('SNR','GDL','FFL'):
        for reference in ('E2','BASE'):
            index={key(x):x for x in runs[reference]['rows']}
            for row in runs[arm]['rows']:
                old=index[key(row)];a=old['metrics']['raw_mask_iou'];b=row['metrics']['raw_mask_iou']
                regressions.append(dict(arm=arm,reference=reference,sample_id=row['sample_id'],letter=row['letter'],role=row['selection_role'],mode=row['input_mode'],horizon=row['horizon'],
                    IoU_before=a,IoU_after=b,position_before=old['metrics']['object_position_error_image_px'],position_after=row['metrics']['object_position_error_image_px'],
                    IoU_status='unknown' if a is None or b is None else 'improved' if b>a else 'regressed' if b<a else 'tied'))
    write(ROOT/'report/per-case-comparisons.json',regressions)
    spec=read(ROOT/'evaluation-spec.json');cases=[x for b in spec['batches'] for x in b['rows']]
    chosen=[next(x for x in cases if x['selection_role']=='train' and x['letter']==l) for l in ('O','C','B')]+[x for x in cases if x['selection_role']=='original_failure']
    panels=[];mapping={};io_out=output_io()
    for row in chosen:
        sid=row['sample_id'];a=int(row['anchor'])
        with np.load(row['raw_path'],allow_pickle=False) as z:truth=z['rgb'][a+1:a+17].copy()
        for mode in ('real_history','autoregressive'):
            frames={'GT':truth}
            for label in LABELS:
                rec=next(x for x in runs[label]['raw_files'] if x['sample_id']==sid and x.get('mode',x.get('input_mode'))==mode)
                with np.load(rec['path'],allow_pickle=False) as z:frames[label]=z['raw'].copy()
            canvas=Image.new('RGB',(900,1362),'white');d=ImageDraw.Draw(canvas);d.text((10,8),row['selection_role']+' '+row['letter']+' '+sid+' | '+mode,fill='black')
            for j,h in enumerate((1,4,8,16)):d.text((116+194*j,32),'H'+str(h),fill='black')
            for k,(label,data) in enumerate(frames.items()):
                y=54+218*k;d.text((6,y+80),label,fill='black')
                for j,h in enumerate((1,4,8,16)):
                    array=data[h-1]
                    if array.dtype!=np.uint8:array=(np.clip(array,0,1)*255).astype(np.uint8)
                    image=Image.fromarray(array.transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST);canvas.paste(image,(110+194*j,y))
            path=reserve_panel(io_out,mapping,sid,mode);buf=io.BytesIO();canvas.save(buf,format='PNG');write_bytes(path,buf.getvalue());panels.append(dict(path=str(path),sample_id=sid,role=row['selection_role'],letter=row['letter'],mode=mode))
    notes='Frozen literature-driven GPC adaptation: native BASE / nominal effective-SNR gamma5 / image gradient magnitude difference / complex focal frequency. One coefficient, one seed,128 attempts per arm. Exploratory reused seen-val; not an independent confirmatory test. Engineering usability, directional support and original absolute gates are separate. RH uses real history each step; AR uses only raw self-feedback. Unknown is not pass; no rigid-corner/physical-causality certificate. No model inference or new video in this report.'
    tables=[]
    for role in ('train','seen_val','original_failure'):
        for mode in ('real_history','autoregressive'):
            lines=[role+' / '+mode,'letter horizon | '+' | '.join(LABELS)]
            for letter in ('O','C','B'):
                for h in (1,4,8,16):
                    values=[lookup[l,role,letter,mode,h] for l in LABELS]
                    lines.append(letter+' H'+str(h)+' | '+' | '.join(f"{x['IoU']:.6f}/{x['position']:.6f}" for x in values))
            tables.append('\n'.join(lines))
    summary=dict(per_arm_support={a:all(x['numeric_directional_support'] for x in checks if x['arm']==a) for a in ('SNR','GDL','FFL')},
        original_absolute={l:{r:runs[l]['support'][r]['all_passed'] for r in ('train','seen_val','original_failure')} for l in LABELS},panels=panels,
        notes=notes,all16_groups=str(ROOT/'report/grouped-all16.json'),per_case=str(ROOT/'report/per-case-comparisons.json'))
    write(ROOT/'report/summary.json',summary)
    body='<!doctype html><meta charset="utf-8"><title>Loss literature screen</title><style>body{font:16px sans-serif;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Loss literature screen</h1><p>'+html.escape(notes)+'</p><pre>'+html.escape(json.dumps(summary['per_arm_support'],indent=2))+'</pre>'
    for table in tables:body+='<pre>'+html.escape(table)+'</pre>'
    for panel in panels:body+='<h2>'+html.escape(panel['role']+' '+panel['letter']+' '+panel['mode'])+'</h2><img src="'+html.escape(Path(panel['path']).name)+'">'
    write_bytes(ROOT/'report/index.html',body.encode('utf-8'));write_bytes(ROOT/'REPORT.txt',(notes+'\n\n'+'\n\n'.join(tables)+'\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)).encode('utf-8'))
    manifest=dict(complete=True,report=str(ROOT/'REPORT.txt'),index=str(ROOT/'report/index.html'),summary=str(ROOT/'report/summary.json'),panels=panels,model_calls=0,new_video_encoders=0)
    write(ROOT/'report/final-manifest.json',manifest);return manifest

