"""Artifact-only R2C1 abort display: no scoring, inference, reload or training."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/native_rgb_r2c1_report.py').is_file())
sys.path.insert(0,str(ROOT))
from scripts import native_rgb_r2c1_report as frozen


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    stage=ROOT/'artifacts/native-rgb-supervised-20261029/r2c1';out=stage/'report'
    assert frozen.sha(Path(frozen.__file__))=='0cab0bec9400296871d4ddd5a881a92eba24f5b697c19ef2ddf9c027b0c55b34'
    work=frozen.read(stage/'model-work.json');tr=work['training_result']
    assert work['closed'] is True and work['status']=='stopped_on_error'
    assert all(tr.get(k)==0 for k in ('model_calls','native_sampler_batch_attempts','native_sampler_batch_completed','training_forward_calls','training_forward_completed_calls'))
    original=frozen.read(out/'r2c1-report.json')
    # Frozen build only loads already-budgeted metrics/explicit placeholders;
    # its cold path never calls the scorer, including for copy-last.
    report,arrays=frozen.build_report(stage/'model-work.json',stage/'run/evaluation',
        stage.parent/'r2c0/data/manifest.json',stage.parent/'r2c0/evaluation',
        ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
    assert report['endpoint_metrics_recomputed_cold'] is False
    assert all(r.get('metrics_not_performed') and r['raw_object_error'] is None for r in report['rows'])
    for page in original['gallery_pages']:assert frozen.sha(out/'media'/page['file'])==page['sha256']
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    media=out/'media-readable';media.mkdir(exist_ok=True);pages=[];missing=[]
    for page_kind in ('main','original_T_task_baseline'):
        method='trained' if page_kind=='main' else 'original_GPC'
        for cohort in frozen.COHORTS:
            prepared=arrays[cohort]['prepared'];result=arrays[cohort]['methods'][method][0]
            for i,identifier in enumerate(prepared['window_ids']):
                for step in range(1,5):
                    if page_kind=='main':pass
                    missing.append(dict(method=method,cohort=cohort,window_id=identifier,future_step=step,
                        status='not_attempted_engineering_abort',raw_output_present=False,
                        reason=work['failure']['reason']))
            for horizon in (1,4):
                assert all('not_attempted' in status or 'missing' in status for status in result[f'status_h{horizon}'])
                assert np.isnan(result[f'h{horizon}']).all()
                for start in (0,6):
                    fig,axes=plt.subplots(6,3,figsize=(9.3,13.5));fig.subplots_adjust(top=.88,bottom=.035,left=.035,right=.98,wspace=.12,hspace=.85)
                    context='主图：真值 / 仅视觉copy-last / 未执行预测' if page_kind=='main' else '原始T任务GPC独立基线：未执行（模型调用0）'
                    fig.suptitle(f'R2C1 工程中止，尚无学习或预测结果\n{context}\n{cohort} · H{horizon}（{horizon*.1:.1f}秒）· 所有固定窗口保留',fontsize=11,y=.99)
                    ids=[]
                    for local,i in enumerate(range(start,start+6)):
                        meta=prepared['metadata'][i];ids.append(meta['window_id'])
                        x=meta['translation_image96_px'] if horizon==1 else meta['h4_translation_image96_px']
                        angle=meta['signed_rotation_deg'] if horizon==1 else meta['h4_signed_rotation_deg']
                        title=f"{meta['letter']} {meta['trajectory_id']} a={meta['anchor']} | 真幅度 Δ={x:.3f}px θ={angle:+.2f}°\n灰/蓝/轮廓/形状均未评价；unknown不表示已观察到遮挡或形变"
                        axes[local,0].text(0,1.14,title,transform=axes[local,0].transAxes,fontsize=6.5,ha='left',va='bottom')
                        axes[local,0].imshow(np.moveaxis(prepared[f'truth_h{horizon}'][i],0,-1),interpolation='nearest')
                        copy=np.rint(prepared['histories'][i,-1]*255).astype(np.uint8)
                        axes[local,1].imshow(np.moveaxis(copy,0,-1),interpolation='nearest')
                        axes[local,2].text(.5,.5,'未执行预测\nNaN仅为预分配占位\n不是模型非有限输出',ha='center',va='center',transform=axes[local,2].transAxes,fontsize=8,color='crimson')
                        for j,label in enumerate(('真实保存RGB','仅视觉复制（未评分）','训练后预测：未执行' if method=='trained' else '原始GPC：未执行')):
                            axes[local,j].axis('off');axes[local,j].set_title(label,fontsize=8)
                    name=f'{page_kind}-{cohort}-H{horizon}-{start//6+1:02d}.png';path=media/name
                    fig.savefig(path,dpi=120,facecolor='white');plt.close(fig)
                    pages.append(dict(file='media-readable/'+name,sha256=frozen.sha(path),bytes=path.stat().st_size,
                        page_kind=page_kind,cohort=cohort,horizon=horizon,window_ids=ids))
    assert len(missing)==288 and len(pages)==24
    for kind in ('main','original_T_task_baseline'):
        for cohort in frozen.COHORTS:
            for horizon in (1,4):
                observed=[wid for p in pages if p['page_kind']==kind and p['cohort']==cohort and p['horizon']==horizon for wid in p['window_ids']]
                assert observed==arrays[cohort]['prepared']['window_ids']
    with (out/'missing-raw-predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(missing[0]));writer.writeheader();writer.writerows(missing)
    absence=dict(kind='R2C1-pre-evaluation-engineering-abort-missingness.v1',closed=True,status=work['status'],failure=work['failure'],
        model_work_sha256=frozen.sha(stage/'model-work.json'),new_charged_model_seconds=work['budget']['elapsed_model_work_seconds'],
        optimizer_attempts=0,training_forward_calls=0,native_sampler_batch_calls=0,
        raw_model_images=dict(original_GPC=dict(planned=144,generated=0,missing=144),trained=dict(planned=144,generated=0,missing=144)),
        model_endpoint_images=dict(original_GPC=dict(planned=72,generated=0,missing=72),trained=dict(planned=72,generated=0,missing=72)),
        endpoint_metric_rows=dict(planned_all_three_methods=216,computed=0,missing=216,primary_values='null/None, never zero-filled'),
        copy_last='Visual copy of stored LAST only; no copy-last scoring was performed or backfilled',
        missing_nan='Preallocated placeholder, not observed model nonfinite output',unknown='Unperformed assessment, not observed occlusion/deformation',
        midstep_failure_counts='Zero observed events because no generation occurred; NOT evidence of zero failures/success',
        original_frozen_report_sha256=frozen.sha(out/'r2c1-report.json'),readable_pages=pages,
        missing_raw_table=dict(file='missing-raw-predictions.csv',sha256=frozen.sha(out/'missing-raw-predictions.csv'),rows=288),
        gallery_rows=dict(main=72,original_T_task_baseline=72),new_scoring_calls=0,new_model_calls=0,new_checkpoint_loads=0,new_physics_controls=0)
    frozen.write(out/'abort-missingness.json',absence)
    links=''.join(f'<li><a href="../{p["file"]}">{p["page_kind"]} · {p["cohort"]} · H{p["horizon"]} · 第{i+1}页</a></li>' for i,p in enumerate(pages))
    html='<!doctype html><meta charset="utf-8"><title>R2C1工程中止：全部缺测窗口</title><h1>尚无模型预测或训练结果</h1><p>初始化t0检查点保存后的日志调用抛出TypeError。本次0次模型预测、0次训练forward、0optimizer attempts。NaN仅为预分配占位，不是实际模型非有限输出。复制上一帧只作视觉参考，全部评分未执行、未后补。</p><p>原计划288张四步模型输出全部缺失；36窗H1/H4分母完整保留。未评价不能视为失败0或保形通过。</p><ul>'+links+'</ul><p><a href="../abort-missingness.json">缺测/费用事实</a> · <a href="../missing-raw-predictions.csv">288行原计划输出缺失表</a> · <a href="../media/index.html">首次通用显示原件（NaN标签不适合本次缺测语境，已保留审计）</a></p>'
    (media/'index.html').write_text(html,encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C1缺测报告</title><h1>R2C1训练前工程中止</h1><p>0模型预测、0训练、0attempt。原数组是未执行占位；评分全部缺测，不能据此判断模型学习或泛化。</p><p><a href="media-readable/index.html">打开全部24页缺测展示（主72行/独立基线72行）</a></p><p><a href="abort-missingness.json">缺测说明</a> · <a href="missing-raw-predictions.csv">288行原计划缺失表</a> · <a href="r2c1-report.json">冻结脚本的原始未评分报告</a></p>',encoding='utf-8')
    record=dict(kind='R2C1-artifact-only-abort-display-fix.v1',frozen_source_unchanged=True,frozen_source_sha256=frozen.sha(Path(frozen.__file__)),
        original_report_sha256=frozen.sha(out/'r2c1-report.json'),original_pages_preserved=24,
        original_pages_verified_sha=True,helper_sha256=frozen.sha(Path(__file__)),
        reason='Generic encoder called not_attempted NaN placeholders nonfinite, and first-row text overlapped headline; only display repaired',
        metrics_unchanged=True,cold_metrics_recomputed=False,new_model_calls=0,new_scoring_calls=0,new_physics_controls=0,
        new_pages=24,new_page_rows=dict(main=72,original_T_task_baseline=72),absence_sha256=frozen.sha(out/'abort-missingness.json'),
        readable_index_sha256=frozen.sha(media/'index.html'))
    frozen.write(out/'display-repair-record.json',record)
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert size<30*1024**2
    print(json.dumps(dict(readable_pages=24,raw_missing_rows=288,missing_metrics=216,report_bytes=size,new_scoring_calls=0,new_model_calls=0)))


if __name__=='__main__':main()
