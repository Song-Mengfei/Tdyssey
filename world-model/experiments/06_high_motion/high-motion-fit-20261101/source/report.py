"""Saved-only fit report, complete case index, paired panels and true10fps video.

No model imports/calls, physics or new predictions. Display quantization never
feeds the model. Training-fit and original seen-val identities stay separate.
"""
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
H=(1,4,8,16)
KEYS=('raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px','pusher_position_error_image_px')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    from common import atomic_json
    atomic_json(p,v)
def guard(root,c,n):
    from common import check_storage
    check_storage(root,c,reserve_bytes=n)
def stats(values):
    a=np.array([float(v) for v in values if v is not None and np.isfinite(v)],float)
    return dict(n=len(values),available=len(a),unknown=len(values)-len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,P10=float(np.percentile(a,10)) if len(a) else None,
        P90=float(np.percentile(a,90)) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)

def groups(payload):
    result=[]
    for role in ('train','seen_val','original_failure'):
        for mode in ('real_history','autoregressive'):
            for letter in ('O','C','B','ALL'):
                rr=[r for r in payload['rows'] if r['selection_role']==role and r['input_mode']==mode and (letter=='ALL' or r['letter']==letter)]
                selections=[('strong_H1',[r for r in rr if r['strong_H1']]),('transition_H1',[r for r in rr if not r['strong_H1']])]
                selections+=[('H'+str(h),[r for r in rr if r['horizon']==h]) for h in H]
                for subset,selected in selections:
                    if not selected:continue
                    result.append(dict(role=role,mode=mode,letter=letter,subset=subset,targets=len(selected),families=len({r['family_id'] for r in selected}),
                        metrics={key:stats([r['metrics'].get(key) for r in selected]) for key in KEYS},
                        visible_shape=dict(Counter(r['visible_shape_status'] for r in selected)),
                        topology=dict(Counter(r['visible_topology_status'] for r in selected)),
                        rotation=dict(Counter(r['rotation']['status'] for r in selected if r['rotation']['required'])),
                        direction=dict(Counter(r['rotation']['classification'] for r in selected if r['rotation']['required'])),
                        blue_missing=sum('empty_pusher' in r['metrics'].get('failure_reasons',[]) for r in selected)))
    return result

def check_pairing(payloads):
    """Hashes and epsilon/slot observations, not just same numeric seed."""
    source=payloads['0'];base={(r['sample_id'],r['input_mode']):r for r in source['raw_files']}
    inputs={(p['mode'],p['batch_id']):p for p in source['input_evidence']};noise={(p['mode'],p['batch_id']):p for p in source['noise_evidence']};checked=0
    for label,value in payloads.items():
        if not value['complete']:raise ValueError('Incomplete evaluation metrics')
        for rf in value['raw_files']:
            if sha(rf['path'])!=rf['sha256']:raise ValueError('Raw prediction changed')
            original=base[(rf['sample_id'],rf['input_mode'])]
            for k in ('family_id','raw_path','raw_sha256','anchor','batch_id','batch_slot','seeds','initial_epsilon_sha256'):
                if rf[k]!=original[k]:raise ValueError('Checkpoint pair identity differs: '+label+' '+k)
            checked+=1
        for proof in value['input_evidence']:
            if proof!=inputs[(proof['mode'],proof['batch_id'])]:raise ValueError('History/action input hashes differ')
        for proof in value['noise_evidence']:
            if proof!=noise[(proof['mode'],proof['batch_id'])]:raise ValueError('Actual epsilon differs between checkpoints')
    return dict(passed=True,raw_files_verified=checked,same_case_actions_history_batch_slot_seeds_and_actual_epsilon=True,
        limits='Old CD B4 failure outputs are not reused. This experiment uses its own frozen B3 baseline/final pairing.')

def _raw(payload,sid,mode):
    ref=next(x for x in payload['raw_files'] if x['sample_id']==sid and x['input_mode']==mode)
    with np.load(ref['path'],allow_pickle=False) as z:return z['raw'].copy()

def _canvas(row,step,columns,canvas_columns=8):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    width=canvas_columns*160;im=Image.new('RGB',(width,220),'white');draw=ImageDraw.Draw(im)
    font_path=Path('C:/Windows/Fonts/msyh.ttc');font=ImageFont.truetype(str(font_path),13) if font_path.exists() else ImageFont.load_default()
    identity='训练中见过的具体轨迹' if row['selection_role']=='train' else '未参加本轮训练的原 seen-val 轨迹'
    if row['selection_role']=='original_failure':identity+=' / 原失败展示例'
    draw.text((4,2),identity+' | '+row['family_id']+f' a{row["anchor"]} | t+{step+1}={(step+1)/10:.1f}s',font=font,fill='black')
    for j,(label,images) in enumerate(columns):
        draw.text((j*160+3,23),label,font=font,fill='black')
        if images is None:
            draw.text((j*160+3,60),'未在此检查点评测',font=font,fill='gray');continue
        a=np.rint(np.clip(images[step].transpose(1,2,0),0,1)*255).astype(np.uint8)
        im.paste(Image.fromarray(cv2.resize(a,(160,160),interpolation=cv2.INTER_NEAREST)),(j*160,54))
    return np.asarray(im)

def _encode(path,frames,width):
    import imageio_ffmpeg
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{width}x220','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,**({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
    try:
        for frame in frames:process.stdin.write(frame.tobytes())
        process.stdin.close();error=process.stderr.read();code=process.wait()
        if code:raise RuntimeError('Report video encoding failed '+error.decode(errors='replace')[-300:])
    finally:
        if process.poll() is None:process.kill()
    import cv2
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=0
    while cap.read()[0]:count+=1
    cap.release()
    if abs(fps-10)>1e-3:raise ValueError('Report video10Hz identity failed')
    return count

def media(root,c,spec,payloads,extras):
    """All75 cases, no attractive-case reselection. Six concatenated videos."""
    import cv2
    folder=root/'report/media';folder.mkdir(parents=True,exist_ok=True)
    labels=[x for x in ('0','250','500','1000','2000','final') if x in payloads]
    available={label:{(r['sample_id'],r['input_mode']) for r in p['raw_files']} for label,p in payloads.items()}
    result=[];video_records=[]
    for role in ('train','seen_val','original_failure'):
        rr=[r for r in spec['rows'] if r['selection_role']==role]
        for mode in ('real_history','autoregressive'):
            video=folder/f'{role}-{mode}-10fps.mp4';records=[]
            guard(root,c,32*1024**2)
            def frames():
                for index,row in enumerate(rr):
                    with np.load(row['raw_path'],allow_pickle=False) as z:
                        a=int(row['anchor']);truth=z['rgb'][a+1:a+17].astype(np.float32)/255.
                        copy=z['rgb'][a:a+16].astype(np.float32)/255. if mode=='real_history' else np.repeat(z['rgb'][a:a+1].astype(np.float32)/255.,16,axis=0)
                    columns=[('GT',truth),('COPY previous GT' if mode=='real_history' else 'COPY initial',copy)]
                    for label in labels:
                        display_label=label+'@'+str(payloads[label].get('actual_successful_updates','unknown'))
                        columns.append((display_label+(' RH1' if mode=='real_history' else ' AR16'),_raw(payloads[label],row['sample_id'],mode) if (row['sample_id'],mode) in available[label] else None))
                    generated=[_canvas(row,k,columns) for k in range(16)]
                    stem=row['family_id']+'-'+hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8]+'-'+mode
                    board=folder/(stem+'-H1-H4-H8-H16.png');ok,buf=cv2.imencode('.png',cv2.cvtColor(np.concatenate([generated[h-1] for h in H],axis=0),cv2.COLOR_RGB2BGR))
                    if not ok:raise RuntimeError('Saved board encoding failed')
                    guard(root,c,len(buf));board.write_bytes(buf.tobytes())
                    records.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],letter=row['letter'],role=role,mode=mode,
                        board=str(board.resolve()),board_sha256=sha(board),video=str(video.resolve()),start_seconds=index*1.6,end_seconds=(index+1)*1.6,
                        checkpoints=labels,selection='Every frozen case; not selected by predicted quality.',
                        RH_warning='RH outputs are independent one-step targets; watching them sequentially does not establish learned cumulative motion.' if mode=='real_history' else None))
                    yield from generated
            count=_encode(video,frames(),8*160)
            if count!=len(rr)*16:raise ValueError('Video case/frame denominator mismatch')
            result.extend(records);video_records.append(dict(path=str(video.resolve()),sha256=sha(video),role=role,mode=mode,frames=count,fps=10,cases=len(rr)))
    # Two extra predeclared seeds: fixed12, before/final paired. Metrics remain
    # separate, and every output is displayed rather than keeping a best seed.
    if extras:
        panel=[r for b in spec['batches'] if b['batch_id'] in spec['panel_batch_ids'] for r in b['rows']]
        for mode in ('real_history','autoregressive'):
            video=folder/f'fixed12-three-noises-{mode}-10fps.mp4';records=[];guard(root,c,16*1024**2)
            def frames_extra():
                for index,row in enumerate(panel):
                    with np.load(row['raw_path'],allow_pickle=False) as z:a=int(row['anchor']);truth=z['rgb'][a+1:a+17].astype(np.float32)/255.
                    columns=[('GT',truth)]
                    for seed in ('primary','extra1','extra2'):
                        for label in ('0','final'):
                            value=payloads[label] if seed=='primary' else extras[label][seed]
                            columns.append((label+'@'+str(payloads[label].get('actual_successful_updates','unknown'))+' '+seed,_raw(value,row['sample_id'],mode)))
                    generated=[_canvas(row,k,columns) for k in range(16)]
                    board=folder/(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:12]+'-all-seeds-'+mode+'.png')
                    ok,buf=cv2.imencode('.png',cv2.cvtColor(np.concatenate([generated[h-1] for h in H],axis=0),cv2.COLOR_RGB2BGR))
                    if not ok:raise RuntimeError('Noise-check board encoding failed')
                    guard(root,c,len(buf));board.write_bytes(buf.tobytes())
                    records.append(dict(sample_id=row['sample_id'],mode=mode,board=str(board.resolve()),video=str(video.resolve()),start_seconds=index*1.6,noise_sets=['primary','extra1','extra2'],all_noise_outputs_retained=True))
                    yield from generated
            count=_encode(video,frames_extra(),8*160)
            if count!=192:raise ValueError('Noise12 montage denominator mismatch')
            result.extend(records);video_records.append(dict(path=str(video.resolve()),sha256=sha(video),role='fixed12_extra_noise',mode=mode,frames=count,fps=10,cases=12))
    return result,video_records

def _table(columns,rows):
    return ['| '+' | '.join(columns)+' |','|'+'|'.join('---' for _ in columns)+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows]

def curves(root,c,spec,payloads):
    """Compare the same fixed12 cases at every checkpoint, not48 against12."""
    os.environ.setdefault('MPLCONFIGDIR',str(root/'report/mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids={r['sample_id'] for b in spec['batches'] if b['batch_id'] in spec['panel_batch_ids'] for r in b['rows']}
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,letter in enumerate(('O','C','B')):
        for label in ('0','250','500','1000','2000','final'):
            if label not in payloads:continue
            rows=[r for r in payloads[label]['rows'] if r['sample_id'] in ids and r['letter']==letter and r['input_mode']=='autoregressive']
            for axis,key in [(axes[0,col],'raw_mask_iou'),(axes[1,col],'object_position_error_image_px')]:
                value=[np.mean([r['metrics'][key] for r in rows if r['horizon']==h]) for h in range(1,17)]
                axis.plot(range(1,17),value,label=label+'@'+str(payloads[label]['actual_successful_updates']))
                axis.grid(alpha=.2)
        axes[0,col].set_title(letter+' / same4 fit-trained trajectories');axes[0,col].set_ylabel('gray IoU');axes[1,col].set_ylabel('position error (image px)');axes[1,col].set_xlabel('AR horizon (10Hz)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed12 training panel, raw recursive prediction / no future truth');fig.tight_layout()
    path=root/'report/fixed12-motion-curves.png';guard(root,c,4*1024**2);fig.savefig(path,dpi=130);plt.close(fig)
    return dict(path=str(path.resolve()),sha256=sha(path),case_scope='Same12 fixed training cases at every checkpoint; four per letter.')

def generate_report(root,c,training_result):
    root=Path(root);folder=root/'report';folder.mkdir(parents=True,exist_ok=True);begin=time.monotonic()
    results=training_result['evaluations'];payloads={label:read(v['metrics']) for label,v in results.items()}
    for label,p in payloads.items():p['actual_successful_updates']=int(results[label]['successful_updates'])
    if '0' not in payloads or 'final' not in payloads:raise ValueError('Complete baseline/final evaluations required')
    pairing=check_pairing(payloads);extra={}
    for label in ('0','final'):
        extra[label]={read(x['metrics'])['noise_set']:read(x['metrics']) for x in results[label].get('extras',[])}
    if set(extra['0'])!={'extra1','extra2'} or set(extra['final'])!={'extra1','extra2'}:raise ValueError('Both fixed extra noise sets are required before/final')
    for seed in ('extra1','extra2'):check_pairing({'0':extra['0'][seed],'final':extra['final'][seed]})
    spec=read(root/'evaluation-spec.json');std=read(root/'fit-standard.json')
    allgroups={label:groups(p) for label,p in payloads.items()};extra_groups={label:{seed:groups(p) for seed,p in vv.items()} for label,vv in extra.items()}
    final=payloads['final'];baseline=payloads['0'];old={(x['sample_id'],x['input_mode'],x['horizon']):x for x in baseline['rows']};cases=[]
    for row in final['rows']:
        if row['horizon'] not in H:continue
        original=old[(row['sample_id'],row['input_mode'],row['horizon'])]
        cases.append(dict(sample_id=row['sample_id'],letter=row['letter'],role=row['selection_role'],mode=row['input_mode'],horizon=row['horizon'],
            metrics={key:dict(before=original['metrics'].get(key),after=row['metrics'].get(key)) for key in KEYS},rotation=row['rotation'],shape=row['visible_shape_status'],topology=row['visible_topology_status']))
    worst=[]
    for role in ('train','seen_val','original_failure'):
        for letter in ('O','C','B'):
            rr=[x for x in final['rows'] if x['selection_role']==role and x['letter']==letter and x['input_mode']=='autoregressive' and x['horizon']==16]
            if rr:
                for key,fn in [('raw_mask_iou',min),('object_position_error_image_px',max)]:
                    x=fn(rr,key=lambda x:x['metrics'][key]);worst.append(dict(role=role,letter=letter,metric=key,sample_id=x['sample_id'],value=x['metrics'][key],included_in_full_case_media=True))
    support=dict(training=final['training_fit_support'],seen_val=final['seen_val_support'],
        full_training_fit_passed=final['training_fit_support']['all_passed'],
        seen_val_support_passed=bool(final['seen_val_support'] and final['seen_val_support']['all_passed']),
        principal_question='Can this one model fit these48 concrete training trajectories?',
        repeated_fit_is_not_new_data_or_generalization=True,automatic_next_stage=False)
    support['fixed12_extra_noise_support']={seed:value['training_fit_support'] for seed,value in extra['final'].items()}
    support['fixed12_extra_noise_all_passed']=all(v['all_passed'] for v in support['fixed12_extra_noise_support'].values())
    write(folder/'support-signals.json',support);write(folder/'case-comparisons.json',cases)
    display_payloads=dict(payloads)
    for label,p in payloads.items():
        if label.endswith('-train_full'):display_payloads[label.removesuffix('-train_full')]=p
    media_records,videos=media(root,c,spec,display_payloads,extra)
    motion_curves=curves(root,c,spec,display_payloads)
    accounting={k:v for k,v in training_result.items() if k!='evaluations'}
    summary=dict(kind='high-motion-fit-saved-report.v1',run_id=c.get('run_id'),fit_support=support,groups=allgroups,
        extra_noise_groups=extra_groups,pairing=pairing,worst_cases=worst,media=media_records,videos=videos,motion_curves=motion_curves,
        checkpoint_identity={label:p['checkpoint_identity'] for label,p in payloads.items()},
        actual_training_accounting=accounting,evaluation_accounting={label:{k:results[label].get(k) for k in ('scope','actual_predicted_frames','sampler_batch_calls','wall_seconds','successful_updates','exposure_count','fit_score')} for label in results},
        standard=std,report_wall_seconds=time.monotonic()-begin,saved_only=True,new_model_calls=0,new_physics_steps=0,
        limits=['Primary conclusion is fitting48 repeatedly trained trajectories. The24 original seen-val and3 old failure cases were not fit-trained and stay separate.',
          'Intermediate checks use fixed12 training cases; approved conditional train_full checks, if present, are separately recorded and shown. Missing middle-stage cells say not evaluated, never fabricated.',
          'RH panels contain independent targets conditioned on real precedingfourframes; they are not generated16step physical trajectories.',
          'Direction/angle or shape unavailable under symmetry/occlusion/deformation remains unknown, not success. Gray centroid, corner counts and hole count alone do not establish rigidity or contact correctness.',
          'No additional training, hyperparameter change or data collection is approved by this report.'])
    write(folder/'summary.json',summary)
    lines=['# 现有高运动轨迹拟合验证','',
      '主结论：'+('48段训练轨迹达到全部冻结拟合条件。' if support['full_training_fit_passed'] else '48段训练轨迹尚未达到全部冻结拟合条件；失败与不可判定均保留。'),
      '','本轮反复拟合固定48段、最多768个唯一H1目标；24段原seen-val和3个原失败例没有参加本轮训练，单列报告。训练loss或技术完成不等于真实运动/形状通过。',
      '','完整训练状态、实际曝光和成功更新见 summary.json / 原训练结果。最佳fit评分仅用于同固定12案例排序，不是验收通过。',
      '',json.dumps({k:v for k,v in training_result.items() if k in ('progress','exposures','stop_reason','wall_seconds')},ensure_ascii=False),
      '','以下H1为全部明显运动目标的独立真实历史预测；AR4/8/16仅起点四真帧，原float递归，无未来真图、修形或伪造标签。','']
    table=[]
    for label in ('0','250','500','1000','2000','final'):
        if label not in allgroups:continue
        for g in allgroups[label]:
            if g['letter']!='ALL' or not ((g['mode']=='real_history' and g['subset'] in ('strong_H1','transition_H1')) or (g['mode']=='autoregressive' and g['subset'] in ('H1','H4','H8','H16'))):continue
            fmt=lambda key:('/'.join('unknown' if g['metrics'][key][x] is None else f'{g["metrics"][key][x]:.4f}' for x in ('mean','median','P90')))
            table.append([label+'@'+str(payloads[label]['actual_successful_updates']),g['role'],g['mode']+' '+g['subset'],g['targets'],fmt('raw_mask_iou'),fmt('object_position_error_image_px'),json.dumps(g['rotation']),json.dumps(g['visible_shape'])])
    lines+=_table(['检查点','轨迹身份','预测范围','目标数','IoU均/中位/P90','位置px均/中位/P90','旋转pass/fail/unknown','可见形状'],table)
    lines+=['','所有冻结标准逐项、每字母/主导运动/逐H、unknown分母： [support-signals.json](report/support-signals.json)。',
      '','全部逐例对比： [case-comparisons.json](report/case-comparisons.json)。所有75案例图/真实10fps视频索引：[index.html](report/index.html)。',
      '','同固定12训练轨迹的递推误差曲线： ![motion curves](report/fixed12-motion-curves.png)',
      '','最差案例没有从报告移除：','']
    for x in worst:lines.append('- '+json.dumps(x,ensure_ascii=False))
    lines+=['','两额外预冻结noise的12个固定案例全部保留，起点/最终配对；不挑最好的随机输出。','',*['- '+x for x in summary['limits']]]
    report=root/'REPORT.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    htmlrows=[]
    for item in media_records:
        board=Path(item['board']).relative_to(folder).as_posix();video=Path(item['video']).relative_to(folder).as_posix()
        title=html.escape(item['sample_id']+' | '+item.get('role','fixed12 all3noises')+' | '+item['mode'])
        htmlrows.append(f'<details><summary>{title}</summary><p>视频定位 {item["start_seconds"]:.1f}s；RH只作独立单步观察。</p><img loading="lazy" src="{html.escape(board)}"><video controls preload="none" src="{html.escape(video)}#t={item["start_seconds"]:.1f}"></video></details>')
    page='<!doctype html><meta charset="utf-8"><title>High-motion fit</title><style>body{font:15px system-ui;margin:24px}img,video{max-width:100%}details{margin:16px 0}</style><h1>训练拟合与原验证轨迹分开</h1><p>全部冻结案例均保留。原始预测只作显示量化，不反馈修复；RH不是闭环16步。</p><p>训练拟合全部通过：'+str(support['full_training_fit_passed'])+'；原seen-val全部支持：'+str(support['seen_val_support_passed'])+'</p><a href="../REPORT.md">完整报告</a>'+''.join(htmlrows)
    (folder/'index.html').write_text(page,encoding='utf-8')
    paths=[report,folder/'index.html',folder/'summary.json',folder/'support-signals.json',folder/'case-comparisons.json',Path(motion_curves['path'])]
    paths +=[Path(v['path']) for v in videos]+[Path(m['board']) for m in media_records]
    manifest=dict(complete=True,saved_only=True,files=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
        no_model_calls=True,no_physics_calls=True,media_all_frozen_cases=True)
    write(folder/'final-manifest.json',manifest)
    return dict(complete=True,report=str(report.resolve()),summary=str((folder/'summary.json').resolve()),index=str((folder/'index.html').resolve()),
        manifest=str((folder/'final-manifest.json').resolve()),support=str((folder/'support-signals.json').resolve()),full_training_fit_passed=support['full_training_fit_passed'],
        seen_val_support_passed=support['seen_val_support_passed'],new_model_calls=0,new_physics_steps=0,wall_seconds=time.monotonic()-begin)
