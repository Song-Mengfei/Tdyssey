"""Saved-only report. No additional model calls, search, physics, or video."""
from common import *
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import html,io
LABELS=('E2','A128','A256','B128','B256')
def average(rows,key):
    vals=[r['metrics'].get(key) for r in rows]
    if not vals or any(x is None or not np.isfinite(x) for x in vals):return None
    return float(np.mean(vals))
def summarize(rows):
    return dict(n=len(rows),IoU=average(rows,'raw_mask_iou'),position=average(rows,'object_position_error_image_px'),pusher_position=average(rows,'pusher_position_error_image_px'),
        topology=dict(Counter(r['visible_topology_status'] for r in rows)),shape=dict(Counter(r['visible_shape_status'] for r in rows)),rotation=dict(Counter(r['rotation']['status'] for r in rows)),
        rotation_direction=dict(Counter(r['rotation']['classification'] for r in rows)),precise_corners_proven=False)
def fmt(x):return 'unknown' if x is None else f'{x:.4f}'
def image_frame(x):
    if x.dtype!=np.uint8:x=(np.clip(x,0,1)*255).astype(np.uint8)
    return Image.fromarray(x.transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST)
def generate():
    runs={l:read(ROOT/f'evaluation/{l}/metrics.json') for l in LABELS};cfg=read(ROOT/'config.json');spec=read(ROOT/'evaluation-spec.json');groups=[]
    for l,v in runs.items():
        for role in ('train','seen_val','original_failure'):
            for letter in ('O','C','B'):
                for mode in ('real_history','autoregressive'):
                    rr=[r for r in v['rows'] if r['selection_role']==role and r['letter']==letter and r['input_mode']==mode]
                    for h in range(1,17):groups.append(dict(label=l,role=role,letter=letter,mode=mode,horizon=h,**summarize([r for r in rr if r['horizon']==h])))
                    if mode=='real_history':groups.append(dict(label=l,role=role,letter=letter,mode=mode,horizon='strong_H1',**summarize([r for r in rr if r['strong_H1']])))
    write(ROOT/'report/grouped-all16.json',groups)
    lookup={(x['label'],x['role'],x['letter'],x['mode'],x['horizon']):x for x in groups}
    checks=[]
    for letter in ('O','C','B'):
        a=lookup['A256','seen_val',letter,'autoregressive',4];b=lookup['B256','seen_val',letter,'autoregressive',4]
        ar_iou=None if a['IoU'] is None or b['IoU'] is None else b['IoU']>a['IoU']
        ar_pos=None if a['position'] is None or b['position'] is None else b['position']<=a['position']
        ar=a;br=b;a=lookup['A256','seen_val',letter,'real_history','strong_H1'];b=lookup['B256','seen_val',letter,'real_history','strong_H1']
        rh_iou=None if a['IoU'] is None or b['IoU'] is None else b['IoU']>=a['IoU'];rh_pos=None if a['position'] is None or b['position'] is None else b['position']<=a['position']
        topology=[]
        for h in (1,4,8,16):
            aa=lookup['A256','seen_val',letter,'autoregressive',h]['topology'];bb=lookup['B256','seen_val',letter,'autoregressive',h]['topology']
            topology.append(dict(h=h,A=aa,B=bb,no_more_failures=bb.get('fail',0)<=aa.get('fail',0),unknown_is_not_pass=True))
        checks.append(dict(letter=letter,AR4_IoU_improved=ar_iou,AR4_position_not_worse=ar_pos,RH_strong_IoU_not_worse=rh_iou,RH_strong_position_not_worse=rh_pos,topology=topology,
            numeric_directional_support=all(x is True for x in (ar_iou,ar_pos,rh_iou,rh_pos)) and all(x['no_more_failures'] for x in topology),
            absolute_acceptance=runs['B256']['support']['seen_val']['all_passed'],RH_A=a,RH_B=b,AR4_A=ar,AR4_B=br))
    write(ROOT/'report/comparison-checks.json',checks)
    # Same inputs, seeds and raw layout must be equal for all checkpoint evaluations.
    proofs=[runs[l]['evidence'] for l in LABELS]
    paired=all(p==proofs[0] for p in proofs[1:]);write(ROOT/'report/evaluation-pairing.json',dict(passed=paired,checkpoint_labels=LABELS,inputs_batch_slots_seed_and_actual_epsilon_equal=True if paired else False))
    if not paired:raise RuntimeError('Evaluation input/noise pairing changed')
    cases=[r for b in spec['batches'] for r in b['rows']];panels=[]
    chosen=[next(r for r in cases if r['selection_role']=='train' and r['letter']==letter) for letter in ('O','C','B')]+[r for r in cases if r['selection_role']=='original_failure']
    for row in chosen:
        sid=row['sample_id'];a=int(row['anchor'])
        with np.load(row['raw_path'],allow_pickle=False) as f:truth=f['rgb'][a+1:a+17].copy()
        for mode in ('real_history','autoregressive'):
            frames={'GT':truth}
            for l in LABELS:
                record=next(r for r in runs[l]['raw_files'] if r['sample_id']==sid and r['mode']==mode)
                with np.load(record['path'],allow_pickle=False) as f:frames[l]=f['raw'].copy()
            canvas=Image.new('RGB',(900,6*218+54),'white');draw=ImageDraw.Draw(canvas);draw.text((10,8),f"{row['selection_role']} {row['letter']} {sid} | {mode}",fill='black')
            for j,h in enumerate((1,4,8,16)):draw.text((116+194*j,32),f'H{h}',fill='black')
            for k,(l,fr) in enumerate(frames.items()):
                y=54+218*k;draw.text((6,y+80),l,fill='black')
                for j,h in enumerate((1,4,8,16)):
                    canvas.paste(image_frame(fr[h-1]),(110+194*j,y))
                    if l!='GT':
                        m=next(r for r in runs[l]['rows'] if r['sample_id']==sid and r['input_mode']==mode and r['horizon']==h)
                        draw.text((110+194*j,y+194),'IoU '+fmt(m['metrics']['raw_mask_iou'])+' pos '+fmt(m['metrics']['object_position_error_image_px']),fill='black')
            buffer=io.BytesIO();canvas.save(buffer,format='PNG');path=ROOT/f'report/{sid}-{mode}.png';write_bytes(path,buffer.getvalue());panels.append(dict(path=str(path),case=sid,mode=mode,letter=row['letter'],role=row['selection_role']))
    tables=[]
    for role in ('train','seen_val','original_failure'):
        table=['<h2>'+role+' — AR IoU / visible-centroid error (imagepx)</h2><table><tr><th>Letter / horizon</th>'+''.join('<th>'+l+'</th>' for l in LABELS)+'</tr>']
        for letter in ('O','C','B'):
            for h in (1,4,8,16):
                cells=[lookup[l,role,letter,'autoregressive',h] for l in LABELS]
                table.append('<tr><td>'+letter+' H'+str(h)+'</td>'+''.join('<td>'+fmt(x['IoU'])+' / '+fmt(x['position'])+'</td>' for x in cells)+'</tr>')
        table.append('</table>');tables.append(''.join(table))
    support=all(x['numeric_directional_support'] for x in checks)
    absolute=runs['B256']['support']['seen_val']['all_passed']
    text='''本轮完成了有限的 loss 对照。A为原生单步残差MSE，B额外加入4倍四步原始RGB递推MSE；推理采样器未改。
量化的反向使用明确的STE近似，跨步梯度已实测；不是真实离散量化的解析梯度。
两臂256次更新和4096次原单步曝光相同；B另有1024次未来帧监督与6144次额外图像前向（含重算），不等总计算。
固定train12/seen-val24/旧失败3分开。RH每次用真实历史，不能叫16步闭环；AR仅回馈自身raw浮点图像。
本轮只有一个初始化/噪声协议与一个系数、有限更新，不证明最优loss或所有旧问题已修复。
棱角、方向、拓扑unknown保持unknown，不能用IoU改善替代刚体形状验收。背景主导梯度仍未作为本轮已证明原因。'''
    body='<!doctype html><meta charset="utf-8"><title>Loss AR4 A/B</title><style>body{font:16px sans-serif;max-width:1150px;margin:30px auto}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>四步 loss 对照</h1><pre>'+html.escape(text)+f'\n三字母预设数值支持条件全部满足: {support}\n原绝对seen-val验收: {absolute}</pre>'+''.join(tables)
    for p in panels:body+='<h2>'+html.escape(p['role']+' '+p['letter']+' '+p['mode'])+'</h2><img src="'+html.escape(Path(p['path']).name)+'">'
    body+='<p>全16步指标: <a href="grouped-all16.json">grouped-all16.json</a>；逐项支持与unknown: <a href="comparison-checks.json">comparison-checks.json</a>。逐帧raw和逐案例评分在evaluation目录。</p>'
    write_bytes(ROOT/'report/index.html',body.encode())
    summary=dict(numeric_directional_support_all_letters=support,original_seen_val_acceptance=absolute,per_letter=checks,panels=panels,HTML=str(ROOT/'report/index.html'))
    write(ROOT/'report/summary.json',summary)
    lines=[text,'\n预设数值支持全部满足: '+str(support),'原seen-val绝对验收: '+str(absolute),'\n最终 seen-val AR4/AR16 (IoU / visible centroid error px), A -> B:']
    for letter in ('O','C','B'):
        for h in (4,16):
            aa=lookup['A256','seen_val',letter,'autoregressive',h];bb=lookup['B256','seen_val',letter,'autoregressive',h]
            lines.append(f"{letter} H{h}: {fmt(aa['IoU'])}/{fmt(aa['position'])} -> {fmt(bb['IoU'])}/{fmt(bb['position'])}")
    lines+=['\n完整公式/预算/证据: PLAN.txt, config.json, pairing-proof.json, preflight.json, completion.json, report/index.html。','本轮结束后实际回传独立分析聊天，waiting_user_review，不续训或改部署。']
    write_bytes(ROOT/'REPORT.txt','\n'.join(lines).encode());return dict(summary=str(ROOT/'report/summary.json'),report=str(ROOT/'REPORT.txt'),HTML=str(ROOT/'report/index.html'))
