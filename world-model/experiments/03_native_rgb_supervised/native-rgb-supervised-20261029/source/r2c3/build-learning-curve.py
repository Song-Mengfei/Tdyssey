"""Artifact-only cold learning curve. No model/checkpoint load/endpoint scoring.

--point reads one immutable closed observation while training continues.
--final requires the whole model-work clock closed before loading the matrix.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import sys

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[2]
sys.path.insert(0,str(ROOT))
from scripts import native_rgb_r2c1_report as frozen
from scripts.native_rgb_r2c2_report import method_summary
from tdyssey.native_rgb_r2c3_evaluation_io import load_referenced_predictions

POINTS=(300,1000,3000,6000)
SETTINGS=(3,16)
COHORTS=frozen.COHORTS
METHODS=('copy_last',)+tuple(f'u{u}_s{s}' for u in POINTS for s in SETTINGS)
read,write,sha,fmt=frozen.read,frozen.write,frozen.sha,frozen.fmt

def method(u,s):return f'u{u}_s{s}'
def finite_mean(rows,key):return frozen.finite_metric_summary(rows,key)


def closed_point(stage,u):
    p=stage/'run/milestones'/f'observation-{u}.json';obs=read(p)
    if obs.get('closed') is not True or obs.get('lineage_update')!=u:
        raise RuntimeError('Only immutable completed observation markers may be read')
    fixed=read(stage/'run'/f'fixed-loss-{u}.json')
    if fixed!=obs['fixed_loss'] or not fixed['fixed_noise_across_observations'] or not fixed['training_rng_restored']:
        raise ValueError('Fixed-loss file is not bound to its immutable observation')
    _,inventory=frozen.closed_inventory(p,Path(obs['evaluation_root']))
    return obs,inventory


def metric_rows_for(stage,u,s,cohort,prepared,calibration,inventory):
    # C3 stores endpoints as references into the exact float32 four-step files.
    folder=stage/'run/evaluation'/f'update{u}'/f'steps{s}'
    result,steps,source=load_referenced_predictions(folder,cohort,prepared,calibration)
    rows,raw,metric_source=frozen.load_metrics(stage/'run/evaluation',f'update{u}/steps{s}',cohort,
        prepared,result,steps,inventory,calibration,'trained')
    label=method(u,s)
    return ([dict(r,method=label,lineage_update=u,num_steps_denoising=s)for r in rows],
        [dict(r,method=label,cohort=cohort,lineage_update=u,num_steps_denoising=s)for r in raw],
        (result,steps),dict(source,metrics=metric_source))


def prepare_inputs(config):
    protocol=read(Path(config['evaluation_indices']).parent/'protocol.json')
    if sha(config['calibration'])!=protocol['score']['calibration_file_sha256']:
        raise ValueError('Frozen calibration differs')
    for p,h in ((ROOT/protocol['score']['metrics_source'],protocol['score']['metrics_source_sha256']),
        (ROOT/'tdyssey/native_rgb_evaluation.py',protocol['score']['legacy_evaluation_source_sha256'])):
        if sha(p)!=h:raise ValueError('Frozen score math source differs')
    return {name:frozen.load_eval_cohort(config['data_manifest'],config['evaluation_indices'],name)for name in COHORTS},read(config['calibration'])


def point_summary(stage,u):
    if u==6000 and (not(stage/'model-work.json').exists()or read(stage/'model-work.json').get('closed')is not True):
        raise RuntimeError('Wait for whole model-work closure before reading the6000 final point')
    config=read(ROOT/'configs/native_rgb_r2c3.json');prepared,calibration=prepare_inputs(config)
    observation,relative_inventory=closed_point(stage,u)
    inventory={f'update{u}/'+key:dict(value,file=f'update{u}/'+key)for key,value in relative_inventory.items()}
    result=dict(kind='R2C3-closed-observation-short-summary.v1',lineage_update=u,
        observation_path=str(stage/'run/milestones'/f'observation-{u}.json'),
        observation_sha256=sha(stage/'run/milestones'/f'observation-{u}.json'),
        fixed_loss=observation['fixed_loss'],settings=observation['settings'],groups={},
        whole_training_closed=(read(stage/'model-work.json').get('closed')is True)if(stage/'model-work.json').exists()else False,
        training_continues_without_waiting=observation.get('continuing_without_waiting',u<6000),
        own_model_calls=0,own_checkpoint_loads=0,own_endpoint_scoring_calls=0)
    for cohort in COHORTS:
        rows=[]
        for s in SETTINGS:rows.extend(metric_rows_for(stage,u,s,cohort,prepared[cohort],calibration,inventory)[0])
        result['groups'][cohort]={}
        for h in (1,4):
            subset=[r for r in rows if r['horizon']==h]
            result['groups'][cohort][f'H{h}']={str(s):method_summary([r for r in subset if r['method']==method(u,s)])for s in SETTINGS}
            if cohort=='val_natural':
                for name,predicate in (('nonobvious',lambda r:not r['H1_obvious_motion']),('nearstatic',lambda r:r['H1_nearstatic_aux'])):
                    result['groups'][cohort][f'H{h}_{name}']={str(s):method_summary([r for r in subset if predicate(r) and r['method']==method(u,s)])for s in SETTINGS}
    return result


def groups_for(rows):
    pairs=[(method(u,16),method(u,3))for u in POINTS]
    pairs.extend((method(u,s),'copy_last')for u in POINTS for s in SETTINGS)
    for s in SETTINGS:
        pairs.extend((method(u,s),method(previous,s))for previous,u in zip(POINTS,POINTS[1:]))
        pairs.extend((method(u,s),method(300,s))for u in POINTS[2:])
    return dict(methods={m:method_summary([r for r in rows if r['method']==m])for m in METHODS},
        pairs={f'{a}_vs_{b}':frozen.paired(rows,a,b)for a,b in pairs})


def build(stage):
    import numpy as np
    config=read(ROOT/'configs/native_rgb_r2c3.json')
    work,inventory=frozen.closed_inventory(stage/'model-work.json',stage/'run/evaluation')
    if work.get('closed')is not True:raise RuntimeError('Wait for whole model stage to close')
    old1=Path(config['reuse_r1_root']);old2=Path(config['reuse_r2c2_root'])
    _,inv1=frozen.closed_inventory(old1/'model-work.json',old1/'run/evaluation')
    _,inv2=frozen.closed_inventory(old2/'model-work.json',old2/'run/evaluation')
    prepared,calibration=prepare_inputs(config);all_rows=[];all_steps=[];arrays={};sources={};groups={}
    for cohort in COHORTS:
        p=prepared[cohort];n=len(p['window_ids']);copy=p['histories'][:,-1].copy()
        hold=dict(planned_windows=n,window_ids=p['window_ids'],h1=copy,h4=copy,status_h1=['ok']*n,status_h4=['ok']*n)
        copy_steps=[[copy[i]]*4 for i in range(n)]
        rows,steps,src=frozen.load_metrics(old1/'run/evaluation','copy_last',cohort,p,hold,copy_steps,inv1,calibration,'copy_last')
        sources[cohort]={'copy_last':src};arrays[cohort]=dict(prepared=p,methods={'copy_last':(hold,copy_steps)})
        rows=[dict(r,lineage_update=None,num_steps_denoising=None)for r in rows]
        all_steps.extend(dict(r,cohort=cohort,lineage_update=None,num_steps_denoising=None)for r in steps)
        for u in POINTS:
            for s in SETTINGS:
                label=method(u,s)
                if u==300:
                    old,inv,folder=(old1,inv1,'final')if s==3 else(old2,inv2,'steps16')
                    result,raw,source=frozen.load_predictions(old/'run/evaluation',folder,cohort,p,inv,calibration)
                    scored,raw_rows,metric_source=frozen.load_metrics(old/'run/evaluation',folder,cohort,p,result,raw,inv,calibration,'trained')
                    scored=[dict(r,method=label,lineage_update=u,num_steps_denoising=s)for r in scored]
                    raw_rows=[dict(r,method=label,cohort=cohort,lineage_update=u,num_steps_denoising=s)for r in raw_rows]
                    arrays[cohort]['methods'][label]=(result,raw);sources[cohort][label]=dict(source,metrics=metric_source,reused=True)
                else:
                    scored,raw_rows,arr,source=metric_rows_for(stage,u,s,cohort,p,calibration,inventory)
                    arrays[cohort]['methods'][label]=arr;sources[cohort][label]=dict(source,reused=False)
                rows.extend(scored);all_steps.extend(raw_rows)
        all_rows.extend(rows);groups[cohort]={}
        for h in (1,4):
            endpoint=[r for r in rows if r['horizon']==h]
            subsets={'all':endpoint,'H1_obvious_motion':[r for r in endpoint if r['H1_obvious_motion']],
                'H1_small_or_nearstatic':[r for r in endpoint if not r['H1_obvious_motion']],
                'H1_nearstatic_aux':[r for r in endpoint if r['H1_nearstatic_aux']]}
            groups[cohort][f'H{h}']={name:groups_for(values)for name,values in subsets.items()}
    losses={};obs={}
    for u in POINTS[1:]:
        observation,_=closed_point(stage,u);losses[str(u)]=observation['fixed_loss'];obs[str(u)]=observation
    reference=None
    for u,value in losses.items():
        identity=[{k:r[k]for k in('cohort','batch','window_ids','seed','noise_generator_before_sha256','noise_generator_after_sha256')}for r in value['rows']]
        if reference is None:reference=identity
        if identity!=reference or len(value['rows'])!=9:raise ValueError('Fixed-loss windows/noise/order differ across actual observations')
    events=[json.loads(line)for line in(stage/'run/events.jsonl').read_text(encoding='utf-8').splitlines()]
    updates=[r for r in events if r['kind']=='update_complete'];skips=[r for r in events if r['kind']=='AMP_skipped_update']
    draw_file=stage/'run/actual-new-draw-indices.npz'
    with np.load(draw_file,allow_pickle=False)as archive:draws=archive['indices'].copy()
    pool=read(Path(config['data_manifest']).parent/'pools.json')['windows']
    if draws.ndim!=1 or len(draws)!=work['training_result']['progress']['r2c3']['actual_draws'] or any(not 0<=int(i)<len(pool)for i in draws):
        raise ValueError('Actual new-draw archive/counters/window references differ')
    selected=[pool[int(i)]for i in draws];repeats=Counter(int(i)for i in draws)
    exposure=dict(new_actual_draws=len(draws),lineage_total_draws=work['training_result']['progress']['samples'],
        unique_windows=len(repeats),independent_families=len({r['family_id']for r in selected}),
        maximum_window_repeat=max(repeats.values())if repeats else 0,
        letter_group_counts=dict(Counter(r['letter']+('/moving'if r['moving']else'/other')for r in selected)),
        category_counts=dict(Counter(r['category']for r in selected)),draw_file=str(draw_file),draw_file_sha256=sha(draw_file),
        meaning='Repeated saved TRAIN exposures, not new independent data or physics controls')
    report=dict(kind='R2C3-full-native-learning-curve.v1',lineage_points=list(POINTS),sampler_settings=list(SETTINGS),
        methods=list(METHODS),rows=all_rows,raw_step_rows=all_steps,cohorts=groups,prediction_sources=sources,
        endpoint_rows=len(all_rows),expected_endpoint_rows=648,new_expected_rows=432,reused_expected_rows=216,
        frozen_windows=36,missing_endpoint_metrics=sum(r.get('metrics_not_performed',False)for r in all_rows),
        fixed_loss=losses,fixed_loss300_measured=False,closed_observations=obs,model_work=work,
        model_work_sha256=sha(stage/'model-work.json'),config_sha256=sha(ROOT/'configs/native_rgb_r2c3.json'),
        own_endpoint_scoring_calls=0,own_model_calls=0,own_checkpoint_loads=0,own_physics_controls=0,
        endpoint_metrics_recomputed_cold=False,time_policy='record_only_v2',user_set_model_time_budget=False,
        midstep_failures={m:{str(k):sum(r['hard_failure']is True for r in all_steps if r['method']==m and r['step']==k)for k in range(1,5)}for m in METHODS},
        training_updates_logged=len(updates),AMP_skips_logged=len(skips),actual_exposures=exposure,
        attribution='6000 is an observation point, not convergence/capacity/failure proof. Same6 seen letters and fixed windows; train/val trends and sampler factors are separate.',
        limitations=['TRAIN展示窗不是泛化；natural/motion来自同12 seen轨迹并部分时间重叠。',
            'unknown不是通过；可评估未检出失败也不等于刚性保证。','copy/300结果直接复用；无新增300推理或fixed loss。',
            '所有raw四步含H2/H3保留，端点/路径失败分开，有限越界原值反馈和评分。','耗时仅统计，无旧时间门禁；服务器完整训练永远由用户手动执行。'])
    parent_provenance=stage/'parent-view-1000/provenance.json'
    if parent_provenance.exists():
        parent=read(parent_provenance)
        if parent.get('model_calls')!=0 or parent.get('score_recomputations')!=0 or parent.get('sampler_steps')!=3:
            raise ValueError('Expected the parent cold visual review of1000/steps3')
        report['parent_visual_review1000']=dict(provenance=str(parent_provenance),provenance_sha256=sha(parent_provenance),
            gallery=str(stage/'parent-view-1000/index.html'),unknown_summary=parent['unknown_summary'],
            rule='prediction or truth gray mask binary-dilated one image pixel intersects its blue pusher mask',
            rule_scope='possible occlusion/contact adjacency proxy, not proof of actual hidden body',
            direct_human_readable_boundary='At1000/3steps H4 all24unknown are occlusion_ambiguous and none object-border; not24 confirmed occlusions or24 shape passes',
            parent_visual_notes='At1000 B two holes recovered and X forks partly improve; O remains rounded and C visibly deformed',
            no_parent_images_regenerated=True)
    parent3000=stage/'parent-view-3000/provenance.json'
    if parent3000.exists():
        parent=read(parent3000)
        if parent.get('model_calls')!=0 or parent.get('new_scores')!=0 or parent.get('sampler_steps')!=3:
            raise ValueError('Expected the parent cold visual review of3000/steps3')
        report['parent_visual_review3000']=dict(provenance=str(parent3000),provenance_sha256=sha(parent3000),
            gallery=str(stage/'parent-view-3000/index.html'),H4_summary=parent['H4_summary'],
            selection='Same O/B/C/X four focus windows as1000; all36H4 included',
            parent_visual_notes='At3000 B outer boundary is squarer and X forks improve; O still lacks right-angle corners, C remains visibly deformed',
            no_parent_images_regenerated=True)
    if len(all_rows)!=648:raise ValueError('All648 endpoint slots must remain visible')
    return report,arrays,updates,skips


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    return plt


def curves(report,updates,skips,out):
    plt=plotting();pages=[]
    fig,axes=plt.subplots(4,3,figsize=(15,15),constrained_layout=True)
    specs=[('H1','raw_mask_iou','H1 gray IoU ↑'),('H4','raw_mask_iou','H4 gray IoU ↑'),
        ('H4','raw_contour_error_image_px','H4 contour error px ↓'),('H4','object_position_error_image_px','H4 centroid error px ↓')]
    for row,(h,key,label)in enumerate(specs):
        for col,cohort in enumerate(COHORTS):
            group=report['cohorts'][cohort][h]['all'];ax=axes[row,col]
            for s in SETTINGS:ax.plot(POINTS,[group['methods'][method(u,s)]['gray_object'][key]['mean']for u in POINTS],'-o',label=f'{s} denoising steps')
            base=group['methods']['copy_last']['gray_object'][key]['mean']
            if base is not None:ax.axhline(base,color='gray',ls='--',label='copy LAST')
            ax.set(title=cohort+' · '+label,xlabel='Cumulative complete updates');ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xticks(POINTS)
    fig.suptitle('Same-window native RGB learning curves · full denominators · 300 reused',fontsize=15)
    path=out/'learning-curves.png';fig.savefig(path,dpi=130);plt.close(fig);pages.append(path.name)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for col,(h,key,label,area)in enumerate([('H1','raw_object_error','near-static H1 gray error ↓','gray_object'),
        ('H4','raw_object_error','near-static H4 gray error ↓','gray_object'),('H4','object_drift_from_last_image_px','near-static H4 LAST drift px ↓','gray_drift_from_last')]):
        g=report['cohorts']['val_natural'][h]['H1_nearstatic_aux']
        for s in SETTINGS:axes[0,col].plot(POINTS,[g['methods'][method(u,s)][area][key]['mean']for u in POINTS],'-o',label=f'{s}steps')
        axes[0,col].set(title=label,xlabel='Complete updates');axes[0,col].grid(alpha=.25);axes[0,col].legend();axes[0,col].set_xticks(POINTS)
    for col,(outcome,title)in enumerate([('failure','All36 H4 gated shape failure counts'),('unknown','All36 H4 unknown proxy (not pass)')]):
        for s in SETTINGS:
            values=[sum(report['cohorts'][c]['H4']['all']['methods'][method(u,s)]['shape_outcomes'].get(outcome,0)for c in COHORTS)for u in POINTS]
            axes[1,col].plot(POINTS,values,'-o',label=f'{s}steps')
        axes[1,col].set(title=title,xlabel='Complete updates');axes[1,col].grid(alpha=.25);axes[1,col].legend();axes[1,col].set_xticks(POINTS)
    for s in SETTINGS:
        axes[1,2].plot(POINTS,[report['cohorts']['val_motion']['H4']['all']['methods'][method(u,s)]['topology']['tiny_fragment_pixels']['mean']for u in POINTS],'-o',label=f'{s}steps')
    axes[1,2].set(title='motion H4 tiny-fragment pixels (raw diagnostic)',xlabel='Complete updates');axes[1,2].grid(alpha=.25);axes[1,2].legend();axes[1,2].set_xticks(POINTS)
    path=out/'static-and-shape-curves.png';fig.savefig(path,dpi=130);plt.close(fig);pages.append(path.name)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    x=[r['update']for r in updates];loss=[r['loss']for r in updates]
    axes[0,0].plot(x,loss,lw=.4,alpha=.35,label='actual training batches')
    for start in range(0,len(updates),100):
        chunk=updates[start:start+100]
        if chunk:axes[0,0].plot(sum(r['update']for r in chunk)/len(chunk),sum(r['loss']for r in chunk)/len(chunk),'o',ms=3,color='black')
    axes[0,0].set(title='Actual training loss (batch/noise vary)',xlabel='Complete update',yscale='log');axes[0,0].legend()
    for c in COHORTS:axes[0,1].plot(POINTS[1:],[report['fixed_loss'][str(u)]['cohort_means'][c]for u in POINTS[1:]],'-o',label=c)
    axes[0,1].set(title='Fixed-window same noise-plan H1 loss · no300 measurement',xlabel='Complete update');axes[0,1].legend();axes[0,1].set_xticks(POINTS[1:])
    axes[1,0].plot(x,[r['optimizer_lr_used']for r in updates]);axes[1,0].set(title='Actual LR used: L(u−1), hold eta_min after3000',xlabel='Complete update')
    if skips:axes[1,1].scatter([r['completed_updates']for r in skips],[r['scale_after']for r in skips],label='actual AMP skip lower scale')
    else:axes[1,1].text(.5,.5,'No AMP skips recorded',ha='center',va='center',transform=axes[1,1].transAxes)
    axes[1,1].set(title='AMP skips: charged attempts, not completed updates',xlabel='Complete update at skip',ylabel='Scale after')
    for ax in axes.flat:ax.grid(alpha=.25)
    path=out/'loss-and-lr-curves.png';fig.savefig(path,dpi=130);plt.close(fig);pages.append(path.name)
    return [dict(file=name,sha256=sha(out/name),bytes=(out/name).stat().st_size)for name in pages]


def galleries(report,arrays,out):
    import numpy as np
    plt=plotting();out.mkdir(parents=True,exist_ok=True);pages=[]
    score={(r['cohort'],r['horizon'],r['method'],r['window_id']):r for r in report['rows']}
    for s in SETTINGS:
        for cohort in COHORTS:
            p=arrays[cohort]['prepared'];methods=arrays[cohort]['methods']
            for h in (1,4):
                for start in range(0,12,3):
                    fig=plt.figure(figsize=(18,12),facecolor='white')
                    fig.text(.5,.969,f'R2C3 · 原生{s}次去噪 · {cohort} · H{h} ({h*.1:.1f}秒)',ha='center',va='top',fontsize=15)
                    fig.text(.5,.942,'全部固定窗 / 真值、复制LAST、累计300/1000/3000/6000更新 / 300直接复用 / 原RGB全帧',ha='center',va='top',fontsize=10)
                    ids=[]
                    for local,i in enumerate(range(start,start+3)):
                        header=.90-.295*local;bottom=.65-.295*local;identifier=p['window_ids'][i];meta=p['metadata'][i];ids.append(identifier)
                        near=' / 近静止辅助'if meta['translation_image96_px']<.1 and meta['abs_rotation_deg']<.5 else''
                        fig.text(.025,header,f"{meta['letter']}  {identifier}  |  {'明显运动'if meta['moving']else'非明显运动'}{near}",fontsize=10,va='bottom')
                        columns=[(p[f'truth_h{h}'][i],'真值RGB',None),(methods['copy_last'][0][f'h{h}'][i],'复制LAST','copy_last')]
                        columns.extend((methods[method(u,s)][0][f'h{h}'][i],f'累计{u}更新',method(u,s))for u in POINTS)
                        for j,(raw,label,m)in enumerate(columns):
                            left=.016+j*.164;ax=fig.add_axes([left,bottom,.152,.20]);ax.axis('off')
                            fig.text(left+.076,header-.025,label,ha='center',va='bottom',fontsize=10)
                            image,note=frozen.display_encoding(raw.astype(np.float64)/255 if getattr(raw,'dtype',None)==np.uint8 else raw)
                            if image is None:ax.text(.5,.5,note,ha='center',va='center',wrap=True,transform=ax.transAxes,fontsize=9,color='crimson')
                            else:ax.imshow(image,interpolation='nearest')
                            if m is not None:
                                q=score[(cohort,h,m,identifier)];shape={'failure':'失败','unknown':'未知','assessable_no_detected_failure':'未检出失败'}.get(q['shape_outcome'],q['shape_outcome'])
                                fig.text(left,bottom-.010,f"1−IoU={fmt(q.get('raw_object_error'))}  质心={fmt(q.get('object_position_error_image_px'))}px",va='top',fontsize=7.4)
                                failure='未评'if q['endpoint_hard_generation_failure']is None else str(int(q['endpoint_hard_generation_failure']))+'/'+str(int(q['any_path_step_hard_failure']))
                                fig.text(left,bottom-.025,f'形状={shape} 硬失败端点/路径={failure}'+(' 显示限幅'if note else''),va='top',fontsize=7.4,color='crimson'if note else'black')
                    name=f'longitudinal-steps{s}-{cohort}-H{h}-{start//3+1:02d}.png';path=out/name
                    fig.savefig(path,dpi=140);plt.close(fig)
                    pages.append(dict(file=name,sha256=sha(path),bytes=path.stat().st_size,setting=s,cohort=cohort,horizon=h,window_ids=ids,columns=['truth','copy_last',*POINTS],full96px_viewport=True))
    if len(pages)!=48:raise ValueError('Expected48 complete longitudinal gallery pages')
    for s in SETTINGS:
        for c in COHORTS:
            for h in(1,4):
                ids=[w for p in pages if p['setting']==s and p['cohort']==c and p['horizon']==h for w in p['window_ids']]
                if ids!=arrays[c]['prepared']['window_ids']:raise ValueError('Gallery window order/denominator differs')
    return pages


def markdown(r):
    lines=['# R2C3 原生GPC训练长度 × 采样次数学习曲线','',
        '同一300更新完整状态恢复，数据/80-20分布/原生网络与损失保持。300的3/16次结果直接引用旧阶段；本阶段只实际测1000/3000/6000。copy-last不重复推理或评分。所有36固定窗、失败和unknown保留；模型时间仅统计。','',
        '## 全组灰物体指标','',
        '|组|H|完整更新|去噪次数|IoU|灰ROI RGB MAE|轮廓误差px|质心对truth误差px|质心对LAST漂移px|形状结果|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in COHORTS:
        for h in(1,4):
            for m in METHODS:
                g=r['cohorts'][c][f'H{h}']['all']['methods'][m];row=next(x for x in r['rows']if x['cohort']==c and x['horizon']==h and x['method']==m)
                values=[g['gray_object'][k]['mean']for k in('raw_mask_iou','object_foreground_rgb_mae','raw_contour_error_image_px','object_position_error_image_px')]
                values.append(g['gray_drift_from_last']['object_drift_from_last_image_px']['mean'])
                lines.append(f"|{c}|{h}|{row['lineage_update']or'—'}|{row['num_steps_denoising']or'copy'}|"+'|'.join(fmt(v,6)for v in values)+f"|{g['shape_outcomes']}|")
    lines.extend(['','## natural非明显8窗与严格近静止4窗','',
        '|子组|H|方法|N|灰1−IoU|灰ROI MAE|轮廓px|质心truth误差px|LAST漂移px|形状结果|','|---|---:|---|---:|---:|---:|---:|---:|---:|---|'])
    for subset in('H1_small_or_nearstatic','H1_nearstatic_aux'):
        for h in(1,4):
            for m in METHODS:
                g=r['cohorts']['val_natural'][f'H{h}'][subset]['methods'][m]
                values=[g['gray_object'][k]['mean']for k in('raw_object_error','object_foreground_rgb_mae','raw_contour_error_image_px','object_position_error_image_px')]
                values.append(g['gray_drift_from_last']['object_drift_from_last_image_px']['mean'])
                lines.append(f"|{subset}|{h}|{m}|{g['denominator']}|"+'|'.join(fmt(v,6)for v in values)+f"|{g['shape_outcomes']}|")
    lines.extend(['','## 同权重横向、同采样器纵向逐窗比较','',
        '下列为H4全组灰(1−IoU)逐窗赢/平/输；全部H1/H4、每窗delta、natural8/4子组见paired-windows.csv。负相对改善为退步。','',
        '|组|比较|赢/平/输|相对灰误差改善|','|---|---|---|---:|'])
    for c in COHORTS:
        for name,p in r['cohorts'][c]['H4']['all']['pairs'].items():
            o=p['endpoint_outcomes'];v=p['relative_gray_error_reduction']
            lines.append(f"|{c}|{name}|{o.get('win',0)}/{o.get('tie',0)}/{o.get('loss',0)}|{'未评/零参考'if v is None else format(v,'.2%')}|")
    lines.extend(['','## 固定窗固定噪声H1残差loss','',
        '只在1000/3000/6000实测，同一9个batch/36窗及相同noise计划控制条件。核对每batch seed/IDs/shape和generator before/after状态SHA，并使用冻结的同一噪声生成代码；没有保存每个sigma/epsilon张量数值digest，因此不声称独立逐字节核对实际noise tensors。没有测300，曲线不补造300基线；训练batch loss因实际抽样/噪声不同另列。','',
        '|累计完整更新|TRAIN展示|natural seen-val|motion seen-val|','|---:|---:|---:|---:|'])
    for u in POINTS[1:]:lines.append(f'|{u}|'+ '|'.join(fmt(r['fixed_loss'][str(u)]['cohort_means'][c],8)for c in COHORTS)+'|')
    lines.extend(['','## 形状、孔洞与碎片','',
        'shape unknown保留为未知；可评估未检出失败不是刚性证明。以下固定全36窗，同时列可判定分母和unknown。不同训练点的可判定集合会变化，不能把failure/可判定集合的条件比例变化直接叫保形通过率改善，也不能把unknown从分母删除。所有方法H1/H4孔洞/组件/碎片原始计数和分组汇总见JSON与endpoint-scores.csv。蓝推杆指标独立列在pusher-summary.csv，不能替代灰物体。','',
        '|更新|去噪次数|H|固定全窗N|failure|unknown|可判定N|可判定未检出失败（非刚性证明）|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for u in POINTS:
        for s in SETTINGS:
            for h in(1,4):
                total=Counter()
                for c in COHORTS:total.update(r['cohorts'][c][f'H{h}']['all']['methods'][method(u,s)]['shape_outcomes'])
                unknown=total.get('unknown',0)
                lines.append(f"|{u}|{s}|{h}|36|{total.get('failure',0)}|{unknown}|{36-unknown}|{total.get('assessable_no_detected_failure',0)}|")
    lines.extend(['',
        'unknown的occlusion_ambiguous是一个代理规则：真值或预测灰mask膨胀1图像像素后与各自蓝推杆mask相交，就标为可能接触/遮挡歧义。它不证明物体实际被遮住；可见的轮廓畸变也不能因此算通过。父对1000/原生3次H4逐行核对：24个unknown全部由该代理触发，prediction/truth object-border均False，并非24个实际确认遮挡。评分口径保持冻结不改。','',
        '父1000目检发现B双孔恢复、X部分分叉改善，同时O仍圆、C仍明显变形。已有角点/全36窗图库直接引用 [父1000视觉核对](../parent-view-1000/index.html)，不重复生成那11张图；本报告48页仅是规定的全部训练点纵向矩阵。','',
        '父3000在相同O/B/C/X四窗、固定3次的300/1000/3000对照中观察到B外边更方、X分叉继续改善，O仍缺直角、C仍明显变形。直接引用 [父3000视觉核对](../parent-view-3000/index.html) 及其provenance；不重复生成那11张图，也不以四个focus窗替代全36窗统计。','',
        '## 证据范围与下一步','',
        '横向同update的3vs16变化支持采样器因素；纵向同setting的300→1000→3000→6000变化支持300时训练长度不足的因素，两者可以同时影响。TRAIN改善而val不改善或退步是过拟合迹象，不能用TRAIN单独宣称泛化。nearstatic漂移、形状failure/unknown和局部窗退步须和总体均值一起看。','',
        '6000仅是本次观察点，不自动宣称收敛、失败、模型能力上限或继续训练无用。若固定sampler的3000→6000仍持续改善，应建议父审进一步观察10000+；若指标分歧或变化很小，应说明证据不足，不能唯一归因为数据、架构或采样。新阶段最终由父对话安排，实施端不自动续训。','',
        '只对当前6个seen字母和固定窗口成立。natural/motion是同12条刻意含运动轨迹且时间窗部分重叠，不能当独立测试或8未见字母泛化。原RGB四步反馈未修色修形、未截幅评分；PNG显示编码限幅时标注。所有36窗H1/H4在48页图库中全量展示。','',
        '用户另外指定的O/B/C/X目视主图严格为真实未来/1000/3000/6000四列，固定3次去噪，H1/H4分开；O/C有2行4列大细节。见 [四列棱角与孔洞对照](focus-gallery/index.html) 及逐项人工判断focus-gallery/VISUAL_REVIEW.md。16次结果独立保留于完整科学矩阵，不混入指定主图。','',
        '真实模型耗时、恢复/检查点/AMP与调用信息见model-work.json及RUNTIME.md；本脚本0模型、0checkpoint load、0冷评分、0physics。通用check_repo历史CRLF/LF问题保持原样，不声称所有仓库检查全绿。',''])
    return '\n'.join(lines)


def csv_file(path,rows):
    fields=sorted({k for row in rows for k in row})
    with path.open('w',encoding='utf-8-sig',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:frozen.canonical(v)if isinstance(v,(dict,list))else v for k,v in row.items()})


def runtime_markdown(report):
    work=report['model_work'];training=work['training_result'];progress=training['progress'];local=progress['r2c3']
    budget=work['budget'];ledger=budget['ledger']
    lines=['# R2C3 实际运行与费用（时间只统计）','',
        f"真实新阶段 attempts={local['attempts']}、complete={local['completed_updates']}、AMP skips={local['AMP_skips']}，模型血缘累计{progress['updates']}更新（从300恢复），新实际draw={local['actual_draws']}、血缘samples={progress['samples']}。另一个旧实验250更新不混入这条6000血缘。",'',
        f"整个模型阶段{budget['elapsed_model_work_seconds']:.3f}s，record_only_v2，没有时间硬帽。全局账本累计{ledger['charged_seconds']:.3f}s、{ledger['charged_updates']}尝试/{ledger['completed_updates']}完整更新；历史费用不退不重置，active_stage={ledger['active_stage']}。",'',
        '|累计更新|去噪|sampler尝试/完成|denoiser尝试/完成|raw帧|端点评分|sample纯耗时s|CPU评分s|setting总s|peak reserved B|',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for u in POINTS[1:]:
        for s in SETTINGS:
            d=training['observations'][str(u)][str(s)]
            lines.append(f"|{u}|{s}|{d['sampler_batch_attempts']}/{d['sampler_batch_completed']}|{d['denoiser_forward_attempts']}/{d['denoiser_forward_completed']}|{d['raw_frames_saved']}|{d['endpoint_rows_scored']}|{d['sampler_seconds']:.3f}|{d['CPU_scoring_seconds']:.3f}|{d['evaluation_and_scoring_seconds']:.3f}|{d['peak_cuda_reserved']}|")
    lines.extend(['','300_3的forward未在R1单独实际计数，不用36×3理论值冒充观测；300_16的实际记录来自C2。旧R1 final总评估耗时与本表纯sample耗时不是同口径，不能直接给异口径加速比。','',
        f"训练forward尝试/完成={training['training_forward_calls']}/{training['training_forward_completed_calls']}，backward microbatches={training['training_backward_completed_microbatches']}，实际optimizer应用={training['optimizer_step_applications_observed']}；固定loss forwards={training['fixed_loss_forward_calls']}。完整恢复加载次数={training['resume_checkpoint_loads']}；最终真实full reload verified={training['final_checkpoint_reload_verified']}。参数相对300变化tensor数={training.get('parameter_tensors_changed_from_300')}。",'',
        f"GPU全阶段peak reserved={training['peak_cuda_reserved']}B、allocated={training['peak_cuda_allocated']}B。普通finite AMP overflow降scale继续，保留已消费draw/噪声/队列，attempt收费但不涨complete/LR；连续16才按失稳停，真实skip事件与scaler见AMP-skips.csv。",'',
        '## 持久检查点','',
        '|种类|更新|字节|SHA256|路径|','|---|---:|---:|---|---|'])
    checkpoint=training['checkpoint'];lines.append(f"|最终完整可恢复|{checkpoint['updates']}|{checkpoint['bytes']}|{checkpoint['sha256']}|{checkpoint['path']}|")
    for m in training['milestones']:lines.append(f"|model-only，不是resume payload|{m['updates']}|{m['bytes']}|{m['sha256']}|{m['path']}|")
    lines.extend(['','1000/3000观察marker中的full_payload_at_observation是当时rolling last.pt的历史SHA引用，路径后来继续原子更新；不能把旧SHA误当最终文件应保持不变。独立model-only 1000/3000保留，6000保留最终完整状态。每100成功完整边界保存；不另存重复300完整权重。','',
        'LR使用真实恢复的原CosineAnnealingLR(T_max3000,eta_min1e-5)；更新u使用L(u−1)，成功后设置L(u)。到3000后保持eta_min，不再调用原cosine.step导致反弹；truecompleted/phase另持久。实际每步LR在training-updates.csv，失败/AMP skip不推进。','',
        f"实际新曝光{report['actual_exposures']['new_actual_draws']}、血缘总曝光{report['actual_exposures']['lineage_total_draws']}，独立TRAIN families={report['actual_exposures']['independent_families']}，unique窗={report['actual_exposures']['unique_windows']}，本阶段单窗最高重复={report['actual_exposures']['maximum_window_repeat']}。这是保存TRAIN的重复曝光，非新独立数据；完整80/20每字母/类别计数及draw archive SHA见actual-exposure-summary.json。",'',
        '完整原生RGB训练与四步raw-feedback接口不变；0新physics/采样数据/未见字母/阻尼比较/服务器。耗时无论超过任何旧试验数字都只是统计，不要求用户补时间。6000仅观察点，下一阶段由父对话安排，服务器完整训练永远由用户手动执行。',''])
    return '\n'.join(lines)


def science_summary(report):
    result=dict(kind='R2C3-cold-curated-scientific-summary.v1',groups={},nearstatic={},
        endpoint_rows=648,missing=report['missing_endpoint_metrics'],fixed_loss=report['fixed_loss'],
        no_new300_evaluation=True,no300_fixedloss_measurement=True,claims_convergence=False,claims_capacity_limit=False,
        claims_unseen_generalization=False,update6000_is_observation_only=True,own_endpoint_scores=0)
    if 'parent_visual_review1000'in report:result['parent_visual_review1000']=report['parent_visual_review1000']
    if 'parent_visual_review3000'in report:result['parent_visual_review3000']=report['parent_visual_review3000']
    lines=['# 实际学习曲线观察','']
    for c in COHORTS:
        result['groups'][c]={}
        for h in (1,4):
            g=report['cohorts'][c][f'H{h}']['all'];result['groups'][c][f'H{h}']={}
            for s in SETTINGS:
                values=[]
                for u in POINTS:
                    m=g['methods'][method(u,s)]
                    values.append(dict(update=u,IoU=m['gray_object']['raw_mask_iou']['mean'],gray_error=m['gray_object']['raw_object_error']['mean'],
                        ROI_MAE=m['gray_object']['object_foreground_rgb_mae']['mean'],contour_px=m['gray_object']['raw_contour_error_image_px']['mean'],
                        centroid_truth_px=m['gray_object']['object_position_error_image_px']['mean'],LAST_drift_px=m['gray_drift_from_last']['object_drift_from_last_image_px']['mean'],
                        shape=m['shape_outcomes'],endpoint_failures=m['endpoint_hard_failures'],path_failures=m['path_hard_failures']))
                result['groups'][c][f'H{h}'][str(s)]=values
                lines.append(f"- {c} H{h} / {s}次：IoU "+'→'.join(fmt(v['IoU'],6)for v in values)+f"；6000vs3000逐窗{g['pairs'][method(6000,s)+'_vs_'+method(3000,s)]['endpoint_outcomes']}；6000形状{values[-1]['shape']}。")
    for h in (1,4):
        g=report['cohorts']['val_natural'][f'H{h}']['H1_nearstatic_aux'];result['nearstatic'][f'H{h}']={}
        for s in SETTINGS:
            values=[dict(update=u,gray_error=g['methods'][method(u,s)]['gray_object']['raw_object_error']['mean'],LAST_drift_px=g['methods'][method(u,s)]['gray_drift_from_last']['object_drift_from_last_image_px']['mean'],
                vs_copy=g['pairs'][method(u,s)+'_vs_copy_last']['endpoint_outcomes'])for u in POINTS]
            result['nearstatic'][f'H{h}'][str(s)]=values
            lines.append(f"- nearstatic4 H{h}/{s}次：灰error "+'→'.join(fmt(v['gray_error'],6)for v in values)+'；LAST漂移 '+ '→'.join(fmt(v['LAST_drift_px'],6)for v in values)+f"px；6000对copy逐窗{values[-1]['vs_copy']}。")
            for previous,current in zip(values,values[1:]):
                if (all(v is not None for v in(previous['gray_error'],current['gray_error'],previous['LAST_drift_px'],current['LAST_drift_px']))
                        and current['gray_error']<previous['gray_error'] and current['LAST_drift_px']>previous['LAST_drift_px']):
                    lines.append(f"  - 近静止折中：{previous['update']}→{current['update']}的H{h}/{s}次灰IoU误差改善，但相对LAST质心漂移增大；二者单列，不能仅凭重叠/轮廓改善宣布静止保持通过。")
    continued=[];overfit=[]
    for s in SETTINGS:
        changes={c:(result['groups'][c]['H4'][str(s)][-2]['gray_error'],result['groups'][c]['H4'][str(s)][-1]['gray_error'])for c in COHORTS}
        if all(b is not None and a is not None and b<a for a,b in(changes['val_natural'],changes['val_motion'])):continued.append(s)
        if changes['train_visual'][1]<changes['train_visual'][0] and all(b>=a for a,b in(changes['val_natural'],changes['val_motion'])):overfit.append(s)
    result['both_seenval_H4_improve_3000_to6000_settings']=continued
    result['TRAIN_improves_both_seenval_do_not_settings']=overfit
    lines.extend(['','300→6000纵向变化与同权重3vs16横向变化必须分开归因；两因素可同时影响，窗口重复评估限制外推。'])
    if continued:lines.append(f'在{continued}次固定sampler下，两seen-val H4在3000→6000的均值数值继续改善，建议父审幅度、逐窗与形状后进一步观察10000+；若变化微小或指标分歧则证据仍不足，6000不作为收敛上限。')
    else:lines.append('当前没有固定sampler同时让两seen-val H4在3000→6000均改善；应结合分组、形状和静止漂移判断证据，不据此宣布训练无用或模型能力上限。')
    if overfit:lines.append(f'{overfit}次设置出现TRAIN H4改善而两个seen-val H4不改善的走向分歧，可作需继续核验的过拟合线索，不是已过拟合结论；不能仅以TRAIN拟合宣布泛化。')
    lines.append('全部逐窗、孔洞/碎片、形状unknown和中间失败均保留。下一步由父对话安排；实施端不会自动续训。')
    return result,'\n'.join(lines)+'\n'


def focus_gallery(report,arrays,out):
    """Human-requested four-column saved-RGB focus, never a statistical sample."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=True);p=arrays['val_motion']['prepared'];methods=arrays['val_motion']['methods']
    ids=['R2B-new-25-O-left-bottom:12','R2B-new-48-B-top:5','R2B-new-28-C-left:6','R2B-new-38-X-right:13']
    by_id={wid:i for i,wid in enumerate(p['window_ids'])}
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
    title_font,subtitle_font,row_font,small_font=font(33),font(21),font(23),font(18)
    config=read(ROOT/'configs/native_rgb_r2c3.json');manifest_path=Path(config['data_manifest']);manifest=read(manifest_path)
    entries={e['trajectory_id']:e for e in manifest['entries']};records=[];figures=[]

    def crop_for(images):
        if any(frozen.numeric_status(a)!='ok'for a in images):return (0,0,96,96),'full96 because an array status is uncertain'
        mask=np.logical_or.reduce([(a<.98).any(axis=0)for a in images]);yy,xx=np.nonzero(mask)
        if not len(xx):return(0,0,96,96),'full96 because valid foreground bbox is unavailable'
        size=min(96,max(int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))+12)
        left=max(0,min(96-size,int((xx.min()+xx.max()+1-size)//2)))
        top=max(0,min(96-size,int((yy.min()+yy.max()+1-size)//2)))
        box=(left,top,left+size,top+size)
        outside=mask.copy();outside[top:top+size,left:left+size]=False
        if outside.any():return(0,0,96,96),'full96 because common crop would omit a foreground pixel'
        return box,'union of all4 images any-channel<.98 bbox, fixed6px margin, square/clamped to viewport; all thresholded foreground retained'

    def source_record(u,identifier,h):
        folder=STAGE/'run/evaluation'/f'update{u}'/'steps3';side=read(folder/'cohort-val_motion.json')
        item=next(x for x in side['raw_step_files']if identifier in x['window_ids']and x['step']==h-1)
        rawfile=folder/item['file']
        if sha(rawfile)!=item['sha256']:raise ValueError('Focus raw file SHA changed')
        return dict(training_updates=u,raw_file=str(rawfile),raw_file_sha256=item['sha256'],
            raw_batch_row=item['window_ids'].index(identifier),window_ids=item['window_ids'],step=item['step'],
            paired_seed=item['seed'],raw_dtype=item['raw_dtype'],raw_shape=item['raw_shape'],
            input_history_sha256=item['input_history_sha256'],normalized_actions_sha256=item['normalized_actions_sha256'])

    def figure(selected,h,detail):
        width,header,row_h,tile_target=(2560,178,660,552)if detail else(1920,165,405,330)
        footer=70;canvas=Image.new('RGB',(width,header+row_h*len(selected)+footer),'white');draw=ImageDraw.Draw(canvas)
        draw.text((28,15),('O / C 放大细节'if detail else'O / B / C / X 固定窗口')+f' · H{h} ({h*.1:.1f}秒)',font=title_font,fill='#182436')
        draw.text((28,62),'四列严格为真实未来 / 训练1000 / 训练3000 / 训练6000；全部原生3次去噪',font=subtitle_font,fill='#384656')
        draw.text((28,96),'共同坐标crop、同整数倍最近邻；无旋转/平移对齐，无锐化/平滑/修形；四个目视样本不作泛化统计。',font=small_font,fill='#536171')
        cell=width//4;xs=[cell*j for j in range(4)]
        for x,label in zip(xs,['真实未来','训练 1,000 更新','训练 3,000 更新','训练 6,000 更新']):
            draw.text((x+34,132),label,font=row_font,fill='#182436')
        rowrecords=[]
        for row,identifier in enumerate(selected):
            i=by_id[identifier];meta=p['metadata'][i];y=header+row*row_h
            truth=p[f'truth_h{h}'][i].astype(np.float32)/np.float32(255)
            images=[truth]+[methods[method(u,3)][1][i][h-1]for u in(1000,3000,6000)]
            if any(a is None or np.asarray(a).shape!=(3,96,96)for a in images):raise ValueError('Focus raw frame is absent/structurally invalid')
            box,reason=crop_for(images);size=box[2]-box[0];zoom=max(1,tile_target//size)
            entry=entries[meta['trajectory_id']];truthfile=(manifest_path.parent/manifest['source_root']/entry['file']).resolve()
            columns=[dict(role='truth',trajectory_file=str(truthfile),trajectory_file_sha256=entry['sha256'],frame_index=meta['anchor']+h,
                uint8_source_frame_sha256=hashlib.sha256(p[f'truth_h{h}'][i].tobytes()).hexdigest())]+[source_record(u,identifier,h)for u in(1000,3000,6000)]
            seeds=[x['paired_seed']for x in columns[1:]]
            if len(set(seeds))!=1:raise ValueError('Focus must use the same paired sampler seed across training points')
            interest='外直边 / 转角 / O内孔'if meta['letter']=='O'else'外直边 / 转角 / C开口两端'if meta['letter']=='C'else'外轮廓 / 孔洞或分叉'
            draw.line((25,y,width-25,y),fill='#d9e1e9',width=1)
            draw.text((28,y+9),f"{meta['letter']}  {identifier}  |  {interest}",font=row_font,fill='#25384b')
            for x,a,column in zip(xs,images,columns):
                encoded,note=frozen.display_encoding(a)
                if encoded is None:raise ValueError('Focus invalid raw cannot be silently rendered')
                tile=Image.fromarray(encoded).crop(box).resize((size*zoom,size*zoom),Image.Resampling.NEAREST)
                px=x+(cell-tile.width)//2;py=y+47+(tile_target-tile.height)//2
                canvas.paste(tile,(px,py))
                column.update(numeric_status=frozen.numeric_status(a),raw_value_sha256=frozen._numeric_digest(a)['sha256'],
                    encoded_uint8_sha256=hashlib.sha256(encoded.tobytes()).hexdigest(),tile_origin_canvas=[px,py])
                if note:draw.text((x+28,y+row_h-28),'仅PNG编码限幅；原始输出未改',font=small_font,fill='#b9283a')
            rowrecords.append(dict(window_id=identifier,letter=meta['letter'],cohort='val_motion',horizon=h,
                all_columns_same_crop_xyxy=list(box),crop_side_pixels=size,common_integer_zoom=zoom,
                crop_reason=reason,rotation_alignment=False,translation_alignment=False,
                sharpening=False,smoothing=False,shape_repair=False,columns=columns))
        draw.text((28,canvas.height-57),'目视判断外直边、转角、O内孔、C开口两端；IoU、阈值与unknown都不能代替角点观察。',font=subtitle_font,fill='#384656')
        draw.text((28,canvas.height-29),'共同crop由四列全部非白范围决定，不以真值位置独自裁切；若数组状态/范围不确定就保留全96像素画面。',font=small_font,fill='#536171')
        filename=('OC-detail'if detail else'focus-four')+f'-H{h}.png';path=out/filename;canvas.save(path,optimize=True)
        page=dict(file=filename,sha256=sha(path),bytes=path.stat().st_size,dimensions=list(canvas.size),horizon=h,
            selected_window_ids=selected,columns=['truth','trained1000_steps3','trained3000_steps3','trained6000_steps3'],rows=rowrecords)
        figures.append(page)
        return page

    for h in(1,4):figure(ids,h,False)
    for h in(1,4):figure([ids[0],ids[2]],h,True)
    references=[dict(file=str(STAGE/f'parent-view-{u}/render.py'),sha256=sha(STAGE/f'parent-view-{u}/render.py'))for u in(1000,3000)]
    provenance=dict(kind='R2C3-direct-human-four-column-focus-gallery.v1',closed_model_work_sha256=sha(STAGE/'model-work.json'),
        model_calls=0,checkpoint_loads=0,endpoint_scoring_calls=0,physics_controls=0,training_updates=[1000,3000,6000],sampler_steps=3,
        selection='Exact four predeclared val_motion O/B/C/X windows requested by human; not a statistical sample',focus_ids=ids,
        display_encoding='frozen display_encoding: rint(clip(raw,0,1)*255).astype(uint8), PNG-only; original floats/feedback/scores unchanged',
        parent_render_algorithm_references=references,figures=figures,visual_review_pending=True,
        manual_OC_judgments={f'H{h}':{'O':None,'C':None}for h in(1,4)},
        numerical_or_unknown_is_not_a_visual_corner_judgment=True)
    write(out/'provenance.json',provenance)
    links=''.join(f'<h2>{p["file"]}</h2><a href="{p["file"]}"><img src="{p["file"]}" alt="四列同窗原RGB对照"></a>'for p in figures)
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1000/3000/6000棱角与孔洞</title><style>body{font:18px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1450px;margin:24px auto;color:#182436}img{max-width:100%;height:auto}a{color:#175f9c}</style><h1>固定3次去噪：1000 → 3000 → 6000</h1><p>主图严格四列：真实未来、训练1000、训练3000、训练6000。4个val_motion固定窗口O/B/C/X与之前相同；H1=0.1秒、H4=0.4秒分图。16次去噪只在完整科学矩阵中，不混入本页。</p><p>O/C细节每行所有列使用同一union bbox与6像素固定margin、同一整数最近邻倍数，保留位置/角度/轮廓误差，无对齐、锐化、平滑或修形。点击图片看原分辨率。全部raw SHA、step、paired seed、crop与zoom见<a href="provenance.json">provenance</a>；角点目检见<a href="VISUAL_REVIEW.md">独立文字判断</a>。</p>'+links+'<p>这是4个指定目视样本，不代表泛化统计；全部36窗和3/16矩阵见<a href="../media/index.html">完整48页图库</a>。</p></html>',encoding='utf-8')
    return dict(index=str(out/'index.html'),provenance=str(out/'provenance.json'),provenance_sha256=sha(out/'provenance.json'),figures=[{k:v for k,v in p.items()if k!='rows'}for p in figures],visual_review_pending=True)


def finalize(stage):
    report,arrays,updates,skips=build(stage);out=stage/'report';out.mkdir(exist_ok=True)
    write(out/'actual-exposure-summary.json',report['actual_exposures'])
    report['curve_figures']=curves(report,updates,skips,out)
    pages=galleries(report,arrays,out/'media');report['gallery_pages']=pages;report['gallery_rows']=144
    report['human_requested_focus_gallery']=focus_gallery(report,arrays,out/'focus-gallery')
    summary,science=science_summary(report)
    write(out/'SCIENCE_SUMMARY.json',summary);(out/'SCIENCE_SUMMARY.md').write_text(science,encoding='utf-8')
    (out/'RUNTIME.md').write_text(runtime_markdown(report),encoding='utf-8')
    write(out/'r2c3-learning-curve.json',report);(out/'REPORT.md').write_text(markdown(report),encoding='utf-8')
    csv_file(out/'endpoint-scores.csv',report['rows']);csv_file(out/'all-raw-steps.csv',report['raw_step_rows'])
    csv_file(out/'training-updates.csv',updates);csv_file(out/'AMP-skips.csv',skips)
    paired=[];pusher=[]
    for c in COHORTS:
        for h in(1,4):
            for subset,g in report['cohorts'][c][f'H{h}'].items():
                for name,p in g['pairs'].items():paired.extend(dict(r,cohort=c,horizon=h,subset=subset,pair=name)for r in p['windows'])
                for m,summary in g['methods'].items():pusher.append(dict(cohort=c,horizon=h,subset=subset,method=m,denominator=summary['denominator'],**{k:v['mean']for k,v in summary['blue_pusher'].items()}))
    csv_file(out/'paired-windows.csv',paired);csv_file(out/'pusher-summary.csv',pusher)
    links=''.join(f'<li><a href="{p["file"]}">{p["setting"]}次 · {p["cohort"]} · H{p["horizon"]} · {i+1}</a></li>'for i,p in enumerate(pages))
    (out/'media/index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C3全部纵向原RGB</title><style>body{font:17px sans-serif;max-width:1300px;margin:2em auto;line-height:1.6}img{max-width:100%}</style><h1>全部36窗H1/H4 · 3/16次去噪分开</h1><p>48页144行，每行真值/copy/300/1000/3000/6000，全96px最近邻显示，无裁图无选择好坏。原float反馈/评分未变。</p><ul>'+links+'</ul>'+''.join(f'<h2>{p["setting"]}次 · {p["cohort"]} H{p["horizon"]}</h2><img loading="lazy" src="{p["file"]}" alt="完整固定窗纵向对照">'for p in pages),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C3学习曲线</title><style>body{font:17px sans-serif;max-width:1200px;margin:2em auto;line-height:1.7}img{max-width:100%}</style><h1>R2C3 原生训练长度 × 采样次数</h1><p>300旧结果复用；同一完整状态连续训练至1000/3000/6000，只在这些新点测3/16。648端点保留全部分母；固定loss无300测量。6000是观察点，非收敛或能力上限。</p><ul><li><a href="focus-gallery/index.html">用户指定四列对照与O/C放大细节</a></li><li><a href="SCIENCE_SUMMARY.md">实际趋势与静止/形状限制摘要</a></li><li><a href="REPORT.md">完整分组、横纵比较与边界</a></li><li><a href="RUNTIME.md">实际运行、恢复、checkpoint与耗时</a></li><li><a href="media/index.html">全部48页图库</a></li><li><a href="r2c3-learning-curve.json">完整机器报告</a></li><li><a href="endpoint-scores.csv">648端点</a></li><li><a href="paired-windows.csv">全部逐窗胜负</a></li><li><a href="pusher-summary.csv">推杆独立结果</a></li></ul>'+''.join(f'<h2>{p["file"]}</h2><img src="{p["file"]}">'for p in report['curve_figures']),encoding='utf-8')
    from PIL import Image
    for p in pages:
        with Image.open(out/'media'/p['file'])as image:image.verify()
    write(out/'page-QA.json',dict(kind='R2C3-full-gallery-cold-QA.v1',pages=pages,page_count=48,gallery_rows=144,
        all_window_order_verified=True,png_integrity_verified=True,full96px_viewport=True,finalized=False,
        representative_visual_review_pending=True,human_requested_focus_gallery=report['human_requested_focus_gallery'],model_calls=0,checkpoint_loads=0,endpoint_scoring_calls=0))
    gallery_bytes=sum(p.stat().st_size for p in out.rglob('*.png')if p.is_file())
    numerical_bytes=sum(p.stat().st_size for p in out.rglob('*')if p.is_file()and p.suffix not in('.png',))
    if gallery_bytes>32*1024**2 or numerical_bytes>25*1024**2:raise RuntimeError('Report byte scope exceeded; do not drop windows silently')
    print(json.dumps(dict(endpoint_rows=648,new432_reused216=True,pages=48,gallery_bytes=gallery_bytes,numerical_bytes=numerical_bytes,own_model_calls=0,cold_scores=0,visual_review_pending=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--final',action='store_true');g.add_argument('--point',type=int,choices=POINTS[1:]);args=p.parse_args()
    if args.point:
        result=point_summary(STAGE,args.point);out=STAGE/'report';out.mkdir(exist_ok=True)
        write(out/f'point-{args.point}-summary.json',result)
        print(json.dumps(dict(point=args.point,fixed_loss=result['fixed_loss']['cohort_means'],motion_H4={s:result['groups']['val_motion']['H4'][str(s)]['gray_object']['raw_mask_iou']['mean']for s in SETTINGS},nearstatic_H4={s:result['groups']['val_natural']['H4_nearstatic'][str(s)]['gray_drift_from_last']['object_drift_from_last_image_px']['mean']for s in SETTINGS},own_model_calls=0,cold_scores=0)))
    else:finalize(STAGE)


if __name__=='__main__':main()
