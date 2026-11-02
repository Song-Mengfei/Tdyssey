"""R2C1R1 artifact-only gallery layout; no metric recomputation."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/native_rgb_r2c1_report.py').is_file())
sys.path.insert(0,str(ROOT))
from scripts import native_rgb_r2c1_report as frozen
from scripts.native_rgb_r2c1_report import COHORTS,sha,display_encoding,fmt
def render_readable(report,arrays,out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():plt.rcParams["font.family"]=FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    out=Path(out);out.mkdir(parents=True,exist_ok=True);pages=[]
    score_by_key={(r["cohort"],r["horizon"],r["method"],r["window_id"]):r for r in report["rows"]}
    for page_kind in ("main","original_T_task_baseline"):
        for cohort in COHORTS:
            prepared=arrays[cohort]["prepared"];methods=arrays[cohort]["methods"]
            for horizon in (1,4):
                truth=prepared[f"truth_h{horizon}"]
                method="trained" if page_kind=="main" else "original_GPC"
                for start in range(0,12,6):
                    fig,axes=plt.subplots(6,3,figsize=(9.3,13.5));fig.subplots_adjust(top=.84,bottom=.025,left=.035,right=.98,wspace=.12,hspace=1.10)
                    headline="真值 / 复制最后输入 / 训练后原RGB" if page_kind=="main" else "原始T任务GPC独立基线（未微调，错误颜色参照）"
                    fig.suptitle(f"R2C1R1 · {headline}\n{cohort} · H{horizon} ({horizon*.1:.1f}秒) · 全部固定窗/失败保留",fontsize=12)
                    ids=[]
                    for local,i in enumerate(range(start,start+6)):
                        row=prepared["metadata"][i];identifier=prepared["window_ids"][i];ids.append(identifier)
                        pred=methods[method][0][f"h{horizon}"][i];copy=methods["copy_last"][0][f"h{horizon}"][i]
                        quality=score_by_key[(cohort,horizon,method,identifier)];hold=score_by_key[(cohort,horizon,"copy_last",identifier)]
                        title=(f"{row['letter']} {row['trajectory_id']} a={row['anchor']} | 灰1−IoU copy={fmt(hold['raw_object_error'])} 当前={fmt(quality['raw_object_error'])}\n"
                            f"端点硬失败={quality['endpoint_hard_generation_failure']} 路径失败={quality['any_path_step_hard_failure']} 形状={quality['shape_outcome']}")
                        axes[local,0].text(0,1.45,title,transform=axes[local,0].transAxes,fontsize=6.5,ha="left",va="bottom")
                        for j,(raw,label) in enumerate(((truth[i],"真实RGB"),(copy,"复制最后输入"),(pred,"训练后预测" if method=="trained" else "原始T任务GPC"))):
                            value=raw.astype(np.float64)/255 if getattr(raw,"dtype",None)==np.uint8 else raw
                            image,note=display_encoding(value);ax=axes[local,j];ax.axis("off");ax.set_title(label,fontsize=9)
                            if image is None:ax.text(.5,.5,note,ha="center",va="center",wrap=True,transform=ax.transAxes,fontsize=8,color="crimson")
                            else:ax.imshow(image,interpolation="nearest")
                            if note:ax.text(.0,-.08,note,transform=ax.transAxes,fontsize=6,color="crimson",ha="left",va="top",wrap=True)
                    name=f"{page_kind}-{cohort}-H{horizon}-{start//6+1:02d}.png";path=out/name
                    fig.savefig(path,dpi=150,facecolor="white");plt.close(fig)
                    pages.append(dict(file=name,sha256=sha(path),bytes=path.stat().st_size,page_kind=page_kind,cohort=cohort,horizon=horizon,window_ids=ids))
    return pages

def main():
 stage=ROOT/'artifacts/native-rgb-supervised-20261029/r2c1r1';out=stage/'report'
 assert sha(Path(frozen.__file__))=='0cab0bec9400296871d4ddd5a881a92eba24f5b697c19ef2ddf9c027b0c55b34'
 report,arrays=frozen.build_report(stage/'model-work.json',stage/'run/evaluation',stage.parent/'r2c0/data/manifest.json',stage.parent/'r2c0/evaluation',ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
 assert report['endpoint_metrics_recomputed_cold'] is False
 pages=render_readable(report,arrays,out/'media-readable')
 assert len(pages)==24
 for kind in ('main','original_T_task_baseline'):
  for cohort in COHORTS:
   for horizon in (1,4):
    ids=[w for p in pages if p['page_kind']==kind and p['cohort']==cohort and p['horizon']==horizon for w in p['window_ids']]
    assert ids==arrays[cohort]['prepared']['window_ids']
 original=frozen.read(out/'r2c1-report.json')
 for p in original['gallery_pages']:assert sha(out/'media'/p['file'])==p['sha256']
 links=''.join(f'<li><a href="{p["file"]}">{p["page_kind"]} · {p["cohort"]} · H{p["horizon"]} · {i+1}</a></li>' for i,p in enumerate(pages))
 (out/'media-readable/index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C1R1全部36窗</title><h1>R2C1R1完整预测图库</h1><p>主图真值/复制最后输入/训练后；原始T任务GPC单独baseline页。有限越界仅PNG显示编码截幅，原float和评分无裁剪、修色、修形。</p><ul>'+links+'</ul><p>全部36窗H1/H4，主72行、独立baseline72行。所有失败保留；原始模型H2/H3失败不因H4掩盖。</p>',encoding='utf-8')
 record=dict(kind='R2C1R1-layout-only-readable-gallery.v1',original_source_sha256=sha(Path(frozen.__file__)),original_report_sha256=sha(out/'r2c1-report.json'),helper_sha256=sha(Path(__file__)),pages=pages,all_window_ids_order_verified=True,original24pages_preserved=True,score_math_changed=False,raw_arrays_changed=False,display_encoding_changed=False,cold_metric_recomputations=0,model_calls=0,checkpoint_loads=0,environment_constructions=0,physics_controls=0)
 frozen.write(out/'layout-repair-record.json',record)
 total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert total<30*1024**2
 print(json.dumps(dict(readable_pages=len(pages),bytes=total,cold_metric_recomputations=0,model_calls=0)))
if __name__=='__main__':main()
