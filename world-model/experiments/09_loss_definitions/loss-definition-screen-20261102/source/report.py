"""Saved-result tables and honest same-case panels; zero new model calls."""
from common import *
from collections import Counter
import html,io
ARMS=('G_ONLY','NATIVE_G','NATIVE_T')
def summarize(rows):
    import math,statistics
    def mean(k):
        v=[r['metrics'].get(k) for r in rows]
        return None if not v or any(x is None or not math.isfinite(x) for x in v) else statistics.mean(v)
    return dict(n=len(rows),IoU=mean('raw_mask_iou'),position=mean('object_position_error_image_px'),pusher_position=mean('pusher_position_error_image_px'),contour=mean('raw_contour_error_image_px'),
        topology=dict(Counter(x['visible_topology_status'] for x in rows)),shape=dict(Counter(x['visible_shape_status'] for x in rows)),rotation=dict(Counter(x['rotation']['status'] for x in rows)),precise_corners_proven=False)
def generate():
    import numpy as np
    from PIL import Image,ImageDraw
    from safe_names import reserve_panel
    c=read(ROOT/'frozen-config.json');stages=read(ROOT/'model-stages-complete.json');executed=stages['executed_arms'];labels=['E2','BASE']+executed
    runs={l:read(c['E2_metrics'] if l=='E2' else c['BASE_metrics'] if l=='BASE' else ROOT/f'evaluation/{l}/metrics.json') for l in labels}
    groups=[]
    for label,value in runs.items():
        for role in ['train','seen_val','original_failure']:
            for letter in ['O','C','B']:
                for mode in ['real_history','autoregressive']:
                    rr=[x for x in value['rows'] if x['selection_role']==role and x['letter']==letter and x['input_mode']==mode]
                    for h in range(1,17):groups.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,**summarize([x for x in rr if x['horizon']==h])))
                    if mode=='real_history':groups.append(dict(label=label,role=role,letter=letter,mode=mode,horizon='strong_H1',**summarize([x for x in rr if x['strong_H1']])))
    write(ROOT/'report/grouped-all16.json',groups);lookup={(x['label'],x['role'],x['letter'],x['mode'],x['horizon']):x for x in groups};checks=[]
    def compare(a,b,operator):return None if a is None or b is None else operator(a,b)
    for arm in executed:
        for letter in ['O','C','B']:
            a=lookup['BASE','seen_val',letter,'autoregressive',4];b=lookup[arm,'seen_val',letter,'autoregressive',4];ra=lookup['BASE','seen_val',letter,'real_history','strong_H1'];rb=lookup[arm,'seen_val',letter,'real_history','strong_H1']
            conditions=dict(AR4_IoU=compare(b['IoU'],a['IoU'],lambda x,y:x>y),AR4_position=compare(b['position'],a['position'],lambda x,y:x<=y),RH_strong_IoU=compare(rb['IoU'],ra['IoU'],lambda x,y:x>=y),RH_strong_position=compare(rb['position'],ra['position'],lambda x,y:x<=y))
            topology=[dict(horizon=h,BASE=lookup['BASE','seen_val',letter,'autoregressive',h]['topology'],candidate=lookup[arm,'seen_val',letter,'autoregressive',h]['topology']) for h in [1,4,8,16]]
            conditions['topology_fail_not_increased']=all(x['candidate'].get('fail',0)<=x['BASE'].get('fail',0) for x in topology)
            checks.append(dict(arm=arm,letter=letter,numeric_directional_support=all(x is True for x in conditions.values()),conditions=conditions,topology=topology,original_absolute_acceptance=runs[arm]['support']['seen_val']['all_passed']))
    write(ROOT/'report/comparison-checks.json',checks)
    pairing=all(v['evidence']==runs['E2']['evidence'] for v in runs.values());write(ROOT/'report/evaluation-pairing.json',dict(passed=pairing))
    if not pairing:raise RuntimeError('Evaluation pairing mismatch')
    key=lambda x:(x['sample_id'],x['input_mode'],x['horizon']);changes=[]
    for arm in executed:
        for reference in ['E2','BASE']:
            index={key(x):x for x in runs[reference]['rows']}
            for row in runs[arm]['rows']:
                old=index[key(row)];a=old['metrics']['raw_mask_iou'];b=row['metrics']['raw_mask_iou']
                changes.append(dict(arm=arm,reference=reference,sample_id=row['sample_id'],letter=row['letter'],role=row['selection_role'],mode=row['input_mode'],horizon=row['horizon'],IoU_before=a,IoU_after=b,
                    position_before=old['metrics']['object_position_error_image_px'],position_after=row['metrics']['object_position_error_image_px'],IoU_status='unknown' if a is None or b is None else 'improved' if b>a else 'regressed' if b<a else 'tied',
                    topology_before=old['visible_topology_status'],topology_after=row['visible_topology_status'],shape_before=old['visible_shape_status'],shape_after=row['visible_shape_status']))
    write(ROOT/'report/per-case-comparisons.json',changes)
    spec=read(ROOT/'evaluation-spec.json');cases=[x for b in spec['batches'] for x in b['rows']];chosen=[next(x for x in cases if x['selection_role']=='train' and x['letter']==l) for l in ['O','C','B']]+[x for x in cases if x['selection_role']=='original_failure']
    panels=[];mapping={};out=output_io()
    for row in chosen:
        sid=row['sample_id'];a=int(row['anchor'])
        with np.load(row['raw_path'],allow_pickle=False) as z:truth=z['rgb'][a+1:a+17].copy()
        for mode in ['real_history','autoregressive']:
            frames={'GT':truth}
            for label in labels:
                rec=next(x for x in runs[label]['raw_files'] if x['sample_id']==sid and x.get('mode',x.get('input_mode'))==mode)
                with np.load(rec['path'],allow_pickle=False) as z:frames[label]=z['raw'].copy()
            canvas=Image.new('RGB',(900,54+218*len(frames)),'white');d=ImageDraw.Draw(canvas);d.text((10,8),row['selection_role']+' '+row['letter']+' '+sid+' | '+mode,fill='black')
            for j,h in enumerate([1,4,8,16]):d.text((116+194*j,32),'H'+str(h),fill='black')
            for k,(label,data) in enumerate(frames.items()):
                y=54+218*k;d.text((6,y+80),label,fill='black')
                for j,h in enumerate([1,4,8,16]):
                    arr=data[h-1]
                    if arr.dtype!=np.uint8:arr=(np.clip(arr,0,1)*255).astype(np.uint8)
                    canvas.paste(Image.fromarray(arr.transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST),(110+194*j,y))
            path=reserve_panel(out,mapping,sid,mode);buf=io.BytesIO();canvas.save(buf,format='PNG');write_bytes(path,buf.getvalue());panels.append(dict(path=str(path),sample_id=sid,role=row['selection_role'],letter=row['letter'],mode=mode,width=900,height=canvas.height))
    notes='Error-definition screen, not another noise-weight screen. G_ONLY replaces native score objective by cG*squaredDice of RGB-derived gray/blue occupancy; NATIVE_G adds it; NATIVE_T adds16 fixed spatial-projection W1 plus relative mass. D is continuous noisy-GT denoising reconstruction from the same forward, not pure-noise rollout training. cG/cT are first16train-only complete-parameter-gradient unit conversions, fixed thereafter. E2 and BASE128 reused read-only. Native data/split/order/noise/precision and3-denoise inference/old evaluation/gates unchanged. A single-seed128step exploratory screen is not convergence/generalization, rigidity/topology, contact or penetration proof. RH gives actual history each single step; AR16 is raw self-feedback without future truth or output repair. Unknown is not pass. G_ONLY may impair color/background/sampling and is not silently anchored by native MSE.'
    tables=[];fmt=lambda v:'unknown' if v is None else f'{v:.6f}'
    for role in ['train','seen_val','original_failure']:
        for mode in ['real_history','autoregressive']:
            lines=[role+' / '+mode,'letter horizon | '+' | '.join(labels)]
            for letter in ['O','C','B']:
                for h in [1,4,8,16]:lines.append(letter+' H'+str(h)+' | '+' | '.join(fmt(lookup[l,role,letter,mode,h]['IoU'])+'/'+fmt(lookup[l,role,letter,mode,h]['position']) for l in labels))
            tables.append('\n'.join(lines))
    diagnosis=read(ROOT/'diagnosis/summary.json');risk=[]
    for arm in executed:
        rr=runs[arm]['auxiliary_RGB_mask_diagnostics']
        for role in ['train','seen_val','original_failure']:
            for letter in ['O','C','B']:
                for mode in ['real_history','autoregressive']:
                    for h in [1,4,8,16]:
                        selected=[r for r in rr if r['role']==role and r['letter']==letter and r['mode']==mode and r['horizon']==h]
                        risk.append(dict(arm=arm,role=role,letter=letter,mode=mode,horizon=h,n=len(selected),gray_empty=sum(x['gray_empty'] for x in selected),blue_empty=sum(x['blue_empty'] for x in selected),background_MSE=None if not selected else sum(x['background_MSE'] for x in selected)/len(selected),RGB_outside_01_fraction=None if not selected else sum(x['RGB_outside_01_fraction'] for x in selected)/len(selected),u_strict_outside_fraction=None if not selected else sum(x['u_strict_outside_fraction'] for x in selected)/len(selected),u_exact_boundary_fraction=None if not selected else sum(x['u_exact_boundary_fraction'] for x in selected)/len(selected)))
    write(ROOT/'report/RGB-sampling-risks.json',risk)
    summary=dict(executed_arms=executed,skipped=stages['skipped'],per_arm_support={a:all(x['numeric_directional_support'] for x in checks if x['arm']==a) for a in executed},original_absolute={l:{r:runs[l]['support'][r]['all_passed'] for r in ['train','seen_val','original_failure']} for l in labels},
        coefficients=read(ROOT/'coefficients.json'),panels=panels,notes=notes,all16_groups=str(ROOT/'report/grouped-all16.json'),per_case=str(ROOT/'report/per-case-comparisons.json'),diagnosis=str(ROOT/'diagnosis/summary.json'),RGB_risks=str(ROOT/'report/RGB-sampling-risks.json'))
    write(ROOT/'report/summary.json',summary)
    body='<!doctype html><meta charset="utf-8"><title>Loss definition screen</title><style>body{font:16px sans-serif;margin:30px}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Loss definition screen</h1><p>'+html.escape(notes)+'</p><pre>'+html.escape(json.dumps(summary['per_arm_support'],indent=2))+'</pre>'
    for table in tables:body+='<pre>'+html.escape(table)+'</pre>'
    for panel in panels:body+='<h2>'+html.escape(panel['role']+' '+panel['letter']+' '+panel['mode'])+'</h2><img src="'+html.escape(Path(panel['path']).name)+'">'
    body+='<h2>GT-only error-definition responses, not model predictions</h2><pre>'+html.escape(json.dumps(diagnosis['definitions'],indent=2))+'</pre>'
    for panel in diagnosis['panels']:body+='<p>'+html.escape(panel['sample_id'])+'</p><img src="../diagnosis/'+Path(panel['path']).name+'">'
    write_bytes(ROOT/'report/index.html',body.encode('utf-8'));write_bytes(ROOT/'REPORT.txt',(notes+'\n\n'+'\n\n'.join(tables)+'\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)).encode('utf-8'))
    manifest=dict(complete=True,report=str(ROOT/'REPORT.txt'),index=str(ROOT/'report/index.html'),summary=str(ROOT/'report/summary.json'),panels=panels,diagnostic_panels=diagnosis['panels'],model_calls=0,new_video_encoders=0)
    write(ROOT/'report/final-manifest.json',manifest);return manifest
