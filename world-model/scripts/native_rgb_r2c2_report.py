"""Cold fixed-weight sampler-step comparison; reads already-scored raw outputs.

This module never scores endpoints, loads a checkpoint, calls a model, constructs
an environment or steps physics. The original R1 loader verifies bound metric
input identities before aggregation. Reused three-step/copy results stay in R1.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import native_rgb_r2c1_report as frozen

COHORTS=frozen.COHORTS
METHODS=("copy_last","trained3","trained8","trained16")
GRAY_KEYS=frozen.OBJECT_METRICS+("object_position_error_image_px",)
DRIFT_KEYS=("object_drift_from_last_error","object_drift_from_last_image_px")
TOPOLOGY_KEYS=("predicted_components","truth_components","predicted_holes","truth_holes",
    "extra_fragments","missing_components","holes_difference","tiny_fragment_pixels","area_fraction_error")
LABELS={"copy_last":"复制 LAST","trained3":"训练后 · 3次去噪","trained8":"训练后 · 8次去噪","trained16":"训练后 · 16次去噪"}
read,write,sha,fmt=frozen.read,frozen.write,frozen.sha,frozen.fmt


def method_summary(rows):
    """Same full-denominator means, with already-measured position/topology."""
    result=frozen.summary(rows)
    result["gray_object"]={k:frozen.finite_metric_summary(rows,k) for k in GRAY_KEYS}
    result["topology"]={k:frozen.finite_metric_summary(rows,k) for k in TOPOLOGY_KEYS}
    result["topology_assessable_count"]=sum(r.get("legacy_shape_assessable") is True for r in rows)
    result["topology_unknown_count"]=sum(r.get("shape_outcome")=="unknown" for r in rows)
    # Descriptive raw mask counters never override the established unknown gate.
    result["topology_observed_rows"]=[dict(window_id=r["window_id"],shape_outcome=r["shape_outcome"],
        legacy_shape_assessable=r.get("legacy_shape_assessable"),
        **{k:r.get(k) for k in TOPOLOGY_KEYS}) for r in rows]
    return result


def compare_group(rows):
    return dict(methods={m:method_summary([r for r in rows if r["method"]==m]) for m in METHODS},
        pairs={f"{m}_vs_{ref}":frozen.paired(rows,m,ref)
            for m,ref in (("trained3","copy_last"),("trained8","copy_last"),("trained16","copy_last"),
                          ("trained8","trained3"),("trained16","trained3"),("trained16","trained8"))})


def _verify_protocol(prepared_root,calibration_path):
    protocol=read(prepared_root/"protocol.json")
    expected=((Path(calibration_path),protocol["score"]["calibration_file_sha256"]),
        (ROOT/protocol["score"]["metrics_source"],protocol["score"]["metrics_source_sha256"]),
        (ROOT/"tdyssey/native_rgb_evaluation.py",protocol["score"]["legacy_evaluation_source_sha256"]))
    for path,digest in expected:
        if sha(path)!=digest:raise ValueError("Frozen calibration/endpoint metric source changed")
    return protocol


def build_report(model_work,evaluation_root,reuse_model_work,reuse_evaluation_root,
                 manifest_path,prepared_root,calibration_path):
    import numpy as np
    work,inventory=frozen.closed_inventory(model_work,evaluation_root)
    previous,previous_inventory=frozen.closed_inventory(reuse_model_work,reuse_evaluation_root)
    prepared_root=Path(prepared_root).resolve()
    _verify_protocol(prepared_root,calibration_path)
    calibration=read(calibration_path);audit=read(prepared_root/"preparation-audit.json")
    all_rows=[];all_steps=[];cohorts={};arrays={};sources={}
    for cohort in COHORTS:
        prepared=frozen.load_eval_cohort(manifest_path,prepared_root/"indices.json",cohort)
        if prepared["identity"]!=audit["cohort_identities"][cohort] or len(prepared["window_ids"])!=12:
            raise ValueError("Frozen12/12/12 evaluation identity differs")
        n=len(prepared["window_ids"]);copy=prepared["histories"][:,-1].copy()
        hold=dict(planned_windows=n,window_ids=prepared["window_ids"],h1=copy,h4=copy,
            status_h1=["ok"]*n,status_h4=["ok"]*n)
        methods={"copy_last":(hold,[[copy[i]]*4 for i in range(n)])}
        sources[cohort]={};rows=[]
        settings=(("copy_last",reuse_evaluation_root,"copy_last",previous_inventory,"copy_last"),
                  ("trained3",reuse_evaluation_root,"final",previous_inventory,"trained"),
                  ("trained8",evaluation_root,"steps8",inventory,"trained"),
                  ("trained16",evaluation_root,"steps16",inventory,"trained"))
        for method,root,stage,bound,metric_method in settings:
            if method=="copy_last":result,steps=methods[method];source={"raw_source":"saved LAST copied identically, not re-inferred"}
            else:
                result,steps,source=frozen.load_predictions(root,stage,cohort,prepared,bound,calibration)
                methods[method]=(result,steps)
            scored,step_rows,metric_source=frozen.load_metrics(root,stage,cohort,prepared,result,steps,bound,calibration,metric_method)
            # Labels differ only after the frozen scorer/input identity checks.
            renamed=[dict(r,method=method) for r in scored]
            renamed_steps=[dict(r,method=method,cohort=cohort) for r in step_rows]
            rows.extend(renamed);all_steps.extend(renamed_steps)
            sources[cohort][method]=dict(source,budgeted_metrics=metric_source,
                evaluation_root=str(Path(root).resolve()),stage=stage,
                reused_from_R1=method in ("copy_last","trained3"))
        arrays[cohort]=dict(prepared=prepared,methods=methods);all_rows.extend(rows)
        cohorts[cohort]={}
        for horizon in (1,4):
            endpoint=[r for r in rows if r["horizon"]==horizon]
            subsets={"all":endpoint,"H1_obvious_motion":[r for r in endpoint if r["H1_obvious_motion"]],
                "H1_small_or_nearstatic":[r for r in endpoint if not r["H1_obvious_motion"]],
                "H1_nearstatic_aux":[r for r in endpoint if r["H1_nearstatic_aux"]]}
            cohorts[cohort][f"H{horizon}"]={name:compare_group(values) for name,values in subsets.items()}
    report=dict(kind="R2C2-fixed-weight-sampler-steps-same-window-report.v1",
        model_work_sha256=sha(model_work),model_work=work,
        reuse_model_work_sha256=sha(reuse_model_work),reuse_model_work=previous,
        prediction_inventory_count=len(inventory),reused_inventory_count=len(previous_inventory),
        data_manifest_sha256=sha(manifest_path),indices_sha256=sha(prepared_root/"indices.json"),
        protocol_sha256=sha(prepared_root/"protocol.json"),calibration_sha256=sha(calibration_path),
        frozen_windows=36,methods=list(METHODS),endpoint_rows=len(all_rows),expected_endpoint_rows=288,
        new_expected_endpoint_rows=144,reused_expected_endpoint_rows=144,
        cohorts=cohorts,rows=all_rows,raw_step_rows=all_steps,prediction_sources=sources,
        midstep_failures={m:{str(k):sum(r["hard_failure"] is True for r in all_steps if r["method"]==m and r["step"]==k)
            for k in range(1,5)} for m in METHODS},
        missing_endpoint_metrics=sum(r.get("metrics_not_performed",False) for r in all_rows),
        limits=["Same fixed36 windows, all failures and unknowns retained; no finite-only primary mean.",
            "Fixed300-update checkpoint, no optimizer or parameter updates; only native num_steps_denoising varies.",
            "Three-step and copy-last outputs/metrics are reused directly from closed R1, not inferred/scored again.",
            "Endpoint/scoring math is unchanged; cold reporting never fills unperformed metrics.",
            "Raw float32 feedback/score values are unchanged; PNG display encoding only may saturate with a label.",
            "Shape unknown from occlusion/viewport remains unknown; no-detected-failure is not proof of rigidity.",
            "Natural/motion overlap in the same12 seen trajectories; this is not independent or unseen-letter testing.",
            "Elapsed model time is record_only_v2 statistics, never a human time budget or automatic stop."],
        endpoint_metrics_recomputed_cold=False,own_model_calls=0,own_checkpoint_loads=0,
        own_environment_constructions=0,own_physics_controls=0,torch_imported="torch" in sys.modules)
    if len(all_rows)!=288:raise ValueError("Report endpoint denominator must remain288")
    return report,arrays


def galleries(report,arrays,out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():plt.rcParams["font.family"]=FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    out=Path(out);out.mkdir(parents=True,exist_ok=True);pages=[]
    score={(r["cohort"],r["horizon"],r["method"],r["window_id"]):r for r in report["rows"]}
    for cohort in COHORTS:
        prepared=arrays[cohort]["prepared"];methods=arrays[cohort]["methods"]
        for horizon in (1,4):
            truth=prepared[f"truth_h{horizon}"]
            for start in range(0,12,3):
                fig=plt.figure(figsize=(15,12),facecolor="white")
                fig.text(.5,.969,f"R2C2 固定权重 · {cohort} · H{horizon} ({horizon*.1:.1f}秒)",ha="center",va="top",fontsize=15)
                fig.text(.5,.942,"全部固定窗口 / 原始RGB全帧 / 3次直接复用 / 只改变原生去噪次数",ha="center",va="top",fontsize=10)
                ids=[]
                for local,i in enumerate(range(start,start+3)):
                    header=.90-.295*local;bottom=.65-.295*local
                    identifier=prepared["window_ids"][i];meta=prepared["metadata"][i];ids.append(identifier)
                    motion="明显运动" if meta["moving"] else "非明显运动"
                    near=" / 近静止辅助" if meta["translation_image96_px"]<.1 and meta["abs_rotation_deg"]<.5 else ""
                    fig.text(.035,header,f"{meta['letter']}  {identifier}  |  {motion}{near}",fontsize=10,va="bottom")
                    columns=[(truth[i],"真值 RGB",None)]+[(methods[m][0][f"h{horizon}"][i],LABELS[m],m) for m in METHODS]
                    for j,(raw,label,method) in enumerate(columns):
                        left=.025+j*.195;ax=fig.add_axes([left,bottom,.18,.20]);ax.axis("off")
                        fig.text(left+.09,header-.025,label,ha="center",va="bottom",fontsize=10)
                        value=raw.astype(np.float64)/255 if getattr(raw,"dtype",None)==np.uint8 else raw
                        image,note=frozen.display_encoding(value)
                        if image is None:ax.text(.5,.5,note,ha="center",va="center",wrap=True,transform=ax.transAxes,fontsize=9,color="crimson")
                        else:ax.imshow(image,interpolation="nearest")
                        if method is not None:
                            q=score[(cohort,horizon,method,identifier)]
                            shape={"failure":"失败","unknown":"未知","assessable_no_detected_failure":"未检出失败"}.get(q["shape_outcome"],q["shape_outcome"])
                            fig.text(left,bottom-.010,f"1−IoU={fmt(q.get('raw_object_error'))}  质心误差={fmt(q.get('object_position_error_image_px'))}px",va="top",fontsize=7.5)
                            failure="未评" if q["endpoint_hard_generation_failure"] is None else str(int(q["endpoint_hard_generation_failure"]))+'/'+str(int(q["any_path_step_hard_failure"]))
                            fig.text(left,bottom-.025,f"形状={shape}  硬失败端点/路径={failure}"+("  显示限幅" if note else ""),va="top",fontsize=7.5,color="crimson" if note else "black")
                        elif note:fig.text(left,bottom-.010,note,va="top",fontsize=7,color="crimson")
                name=f"compare-{cohort}-H{horizon}-{start//3+1:02d}.png";path=out/name
                fig.savefig(path,dpi=150,facecolor="white");plt.close(fig)
                pages.append(dict(file=name,sha256=sha(path),bytes=path.stat().st_size,cohort=cohort,
                    horizon=horizon,window_ids=ids,columns=["truth",*METHODS],full_96px_viewport=True,
                    display_encoding="frozen bounded PNG conversion; no scoring/feedback changes"))
    return pages


def _pair_text(p):
    o=p["endpoint_outcomes"];im=p["relative_gray_error_reduction"]
    return f"{o.get('win',0)}/{o.get('tie',0)}/{o.get('loss',0)}"+(f" ({im:.2%})" if im is not None else " (未评/零参考)")


def scientific_observations(report):
    """Describe measured tradeoffs without assuming a larger step count wins."""
    lines=[]
    motion=report['cohorts']['val_motion']['H4']['all']
    static=report['cohorts']['val_natural']['H4']['H1_nearstatic_aux']
    for m in ('trained8','trained16'):
        base_motion=motion['methods']['trained3']['gray_object']['raw_object_error']['mean']
        new_motion=motion['methods'][m]['gray_object']['raw_object_error']['mean']
        base_static=static['methods']['trained3']['gray_object']['raw_object_error']['mean']
        new_static=static['methods'][m]['gray_object']['raw_object_error']['mean']
        shape=[]
        for method in ('trained3',m):
            totals=Counter()
            for c in COHORTS:totals.update(report['cohorts'][c]['H4']['all']['methods'][method]['shape_outcomes'])
            shape.append(f"失败{totals.get('failure',0)}/未知{totals.get('unknown',0)}")
        lines.append(f"- {m}对3次：motion H4灰error {fmt(base_motion,6)}→{fmt(new_motion,6)}，逐窗{_pair_text(motion['pairs'][m+'_vs_trained3'])}；"
            f"nearstatic H4 {fmt(base_static,6)}→{fmt(new_static,6)}，逐窗{_pair_text(static['pairs'][m+'_vs_trained3'])}；全部H4形状{shape[0]}→{shape[1]}。")
    if report['missing_endpoint_metrics']:
        lines.append('存在未执行评分，现阶段只报告实际已完成证据；建议先由父对话审阅异常，不把缺测填作改善或擅自重试。')
    else:
        improved=[]
        for m in ('trained8','trained16'):
            old_m=motion['methods']['trained3']['gray_object']['raw_object_error']['mean']
            new_m=motion['methods'][m]['gray_object']['raw_object_error']['mean']
            old_s=static['methods']['trained3']['gray_object']['raw_object_error']['mean']
            new_s=static['methods'][m]['gray_object']['raw_object_error']['mean']
            if all(v is not None for v in (old_m,new_m,old_s,new_s)) and new_m<old_m and new_s<old_s:improved.append(m)
        if improved:
            lines.append('在当前固定窗口，'+ '/'.join(improved)+'同时降低motion与nearstatic H4平均灰误差；建议父审进一步核对形状failure/unknown和逐窗退步后决定是否采用，不能据此宣称刚性或泛化通过。')
        else:
            lines.append('当前结果没有一个新增设置同时降低motion与nearstatic H4平均灰误差；增加去噪次数尚不能作为同时解决运动与静止保持的证据。建议把本阶段的实际代价和形状/漂移结果交父审，再决定训练或数据方向，勿自动改抽样或续训。')
    return lines


def markdown(report):
    lines=["# R2C2：固定权重，只改变原生采样器去噪次数","",
        "300更新的同一模型，copy-last与3次输出/评分直接引用R2C1R1，新增8/16次推理。没有继续训练、换seed、修色修形或重复保存检查点。耗时仅统计，record_only_v2持续有效，用户未设模型时间预算。","",
        "## 全分母灰物体结果","",
        "每行12窗。IoU越高越好；MAE、轮廓/质心误差和漂移越低越好。位置与轮廓均为96×96图像像素。C/3/8/16依次为copy-last、同权重原生3/8/16次去噪。","",
        "|组|H|指标|C / 3 / 8 / 16|","|---|---:|---|---|"]
    metric_labels={"raw_mask_iou":"灰IoU","object_foreground_rgb_mae":"灰ROI RGB MAE",
        "raw_contour_error_image_px":"轮廓误差px","object_position_error_image_px":"质心对truth误差px"}
    for cohort in COHORTS:
        for horizon in (1,4):
            g=report["cohorts"][cohort][f"H{horizon}"]["all"]
            for key,label in metric_labels.items():
                values=[g["methods"][m]["gray_object"][key]["mean"] for m in METHODS]
                lines.append(f"|{cohort}|{horizon}|{label}|"+' / '.join(fmt(v,6) for v in values)+'|')
            values=[g["methods"][m]["gray_drift_from_last"]["object_drift_from_last_image_px"]["mean"] for m in METHODS]
            lines.append(f"|{cohort}|{horizon}|质心对LAST漂移px|"+' / '.join(fmt(v,6) for v in values)+'|')
    lines.extend(["","## 逐窗胜负与非明显运动/近静止","",
        "下表为灰(1−IoU)逐窗赢/平/输及相对均值改善；负百分比为退步。完整每窗差值保留在JSON和paired-windows.csv，不以总体均值掩盖失败。","",
        "|组/子组|H|每方法N|3 vs C|8 vs C|16 vs C|8 vs 3|16 vs 3|","|---|---:|---:|---|---|---|---|---|"])
    group_specs=[(c,"all") for c in COHORTS]+[("val_natural",s) for s in ("H1_small_or_nearstatic","H1_nearstatic_aux")]
    for cohort,subset in group_specs:
        for horizon in (1,4):
            g=report["cohorts"][cohort][f"H{horizon}"][subset]
            names=("trained3_vs_copy_last","trained8_vs_copy_last","trained16_vs_copy_last","trained8_vs_trained3","trained16_vs_trained3")
            lines.append(f"|{cohort}/{subset}|{horizon}|{g['methods']['copy_last']['denominator']}|"+'|'.join(_pair_text(g["pairs"][p]) for p in names)+'|')
    lines.extend(["","非明显运动沿用H1未达2图像px平移或5°旋转的固定分组；严格近静止辅助为H1<0.1px且<0.5°，不是精确静止。该分组不改变抽样/验收阈值。","",
        "|natural子组|H|指标|C / 3 / 8 / 16|","|---|---:|---|---|"])
    for subset in ('H1_small_or_nearstatic','H1_nearstatic_aux'):
        for horizon in (1,4):
            g=report['cohorts']['val_natural'][f'H{horizon}'][subset]
            for key,label in {'raw_object_error':'灰1−IoU',**metric_labels}.items():
                lines.append(f"|{subset}|{horizon}|{label}|"+' / '.join(fmt(g['methods'][m]['gray_object'][key]['mean'],6) for m in METHODS)+'|')
            lines.append(f"|{subset}|{horizon}|对LAST质心漂移px|"+' / '.join(fmt(g['methods'][m]['gray_drift_from_last']['object_drift_from_last_image_px']['mean'],6) for m in METHODS)+'|')
    lines.extend(['','下列每窗保留非明显运动的全部8窗，近静止4窗明确标记；胜负详见上表及paired-windows.csv。','',
        '|natural窗口|H|近静止辅助|灰1−IoU C / 3 / 8 / 16|对LAST质心漂移px C / 3 / 8 / 16|',
        '|---|---:|---|---|---|'])
    for horizon in (1,4):
        window_rows=[r for r in report['rows'] if r['cohort']=='val_natural' and r['horizon']==horizon and not r['H1_obvious_motion']]
        ids=list(dict.fromkeys(r['window_id'] for r in window_rows))
        for identifier in ids:
            by_method={r['method']:r for r in window_rows if r['window_id']==identifier}
            lines.append(f"|{identifier}|{horizon}|{by_method['copy_last']['H1_nearstatic_aux']}|"+' / '.join(fmt(by_method[m].get('raw_object_error'),6) for m in METHODS)+'|'+
                ' / '.join(fmt(by_method[m].get('object_drift_from_last_image_px'),6) for m in METHODS)+'|')
    lines.extend(["",
        "## 形状、孔洞、碎片与四步失败","",
        "|组|H|方法|端点/路径硬失败|形状计数|孔洞差均值|多余碎片均值|tiny碎片像素均值|","|---|---:|---|---|---|---:|---:|---:|"])
    for cohort in COHORTS:
        for horizon in (1,4):
            for method in METHODS:
                m=report["cohorts"][cohort][f"H{horizon}"]["all"]["methods"][method]
                lines.append(f"|{cohort}|{horizon}|{method}|{m['endpoint_hard_failures']}/{m['path_hard_failures']}|{m['shape_outcomes']}|{fmt(m['topology']['holes_difference']['mean'])}|{fmt(m['topology']['extra_fragments']['mean'])}|{fmt(m['topology']['tiny_fragment_pixels']['mean'])}|")
    lines.extend(["","遮挡、边界或不可评估继续记unknown，不能当通过；可评估未检出失败也不证明刚性。孔洞/碎片原始计数不推翻unknown门控。原四步硬失败计数（step1/2/3/4）："])
    for m in METHODS:lines.append(f"- {m}: {report['midstep_failures'][m]}")
    lines.extend(["","## 蓝推杆单列","","|组|H|方法|IoU|ROI RGB MAE|质心误差px|","|---|---:|---|---:|---:|---:|"])
    for cohort in COHORTS:
        for horizon in (1,4):
            for method in METHODS:
                m=report["cohorts"][cohort][f"H{horizon}"]["all"]["methods"][method]["blue_pusher"]
                lines.append(f"|{cohort}|{horizon}|{method}|{fmt(m['pusher_mask_iou']['mean'],6)}|{fmt(m['pusher_foreground_rgb_mae']['mean'],6)}|{fmt(m['pusher_position_error_image_px']['mean'],6)}|")
    lines.extend(["","## 证据范围与下一步","",
        "增加生成迭代是否改善运动、静止变化和棱角，必须分别看上述同窗误差、近静止逐窗结果以及形状failure/unknown，不能只看更平滑的图或更低总体均值。所有36窗H1/H4真值/C/3/8/16均展示在media/index.html。原float评分/回灌未裁剪，有限越界保留为失败；PNG编码限幅时标注。","",
        "只对6个seen字母和当前窗口成立。natural/motion来自同12条刻意含运动的轨迹，部分窗重叠；不是24独立轨迹、8未见字母泛化或无偏自然测试。当前诊断不自动改变抽样或续训，下一阶段由父对话依据运动/静止/形状共同安排。","",
        "完整运行耗时、真实denoiser forward计数/每设置调用耗时/显存由model-work.json保留；本报告冷态没有模型、checkpoint重载或补评分。已知通用check_repo的旧CRLF/LF发布hash差异不在本阶段修写。","",
        *scientific_observations(report),""])
    return '\n'.join(lines)


def self_test():
    """Pure aggregation/layout contracts; no data, inference or score calls."""
    assert METHODS==("copy_last","trained3","trained8","trained16")
    assert len(GRAY_KEYS)==5 and "object_position_error_image_px" in GRAY_KEYS
    assert "holes_difference" in TOPOLOGY_KEYS and "tiny_fragment_pixels" in TOPOLOGY_KEYS
    sample=[dict(window_id="x",method=m,horizon=1,raw_object_error=v,all_steps_and_endpoint_valid=True,
        endpoint_hard_generation_failure=False,any_path_step_hard_failure=False) for m,v in zip(METHODS,(.5,.4,.3,.6))]
    assert frozen.paired(sample,"trained8","trained3")["endpoint_outcomes"]=={"win":1}
    assert frozen.paired(sample,"trained16","trained3")["endpoint_outcomes"]=={"loss":1}
    assert frozen.finite_metric_summary([{"x":1},{"x":None}],"x")["mean"] is None
    return dict(passed=True,model_calls=0,checkpoint_loads=0,scoring_calls=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    for name in ("model-work","evaluation-root","reuse-model-work","reuse-evaluation-root","data-manifest","prepared-root","calibration","out"):
        p.add_argument("--"+name,type=Path)
    args=p.parse_args(argv)
    if args.self_test:print(json.dumps(self_test()));return 0
    if any(getattr(args,n.replace('-','_')) is None for n in ("model-work","evaluation-root","reuse-model-work","reuse-evaluation-root","data-manifest","prepared-root","calibration","out")):
        p.error("All paths required unless --self-test")
    report,arrays=build_report(args.model_work,args.evaluation_root,args.reuse_model_work,args.reuse_evaluation_root,
        args.data_manifest,args.prepared_root,args.calibration)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    pages=galleries(report,arrays,out/"media");report["gallery_pages"]=pages
    if len(pages)!=24:raise ValueError("Expected24 complete comparison pages")
    for cohort in COHORTS:
        for horizon in (1,4):
            ids=[w for page in pages if page['cohort']==cohort and page['horizon']==horizon for w in page['window_ids']]
            if ids!=arrays[cohort]['prepared']['window_ids']:raise ValueError("Gallery window order/denominator differs")
    report['gallery_rows']=sum(len(page['window_ids']) for page in pages)
    write(out/"r2c2-report.json",report);(out/"REPORT.md").write_text(markdown(report),encoding="utf-8")
    paired_rows=[]
    for c in COHORTS:
        for h in (1,4):
            for subset,group in report['cohorts'][c][f'H{h}'].items():
                for pair,value in group['pairs'].items():
                    paired_rows.extend(dict(row,cohort=c,horizon=h,subset=subset,pair=pair) for row in value['windows'])
    for filename,rows in (("endpoint-scores.csv",report["rows"]),("all-raw-steps.csv",report["raw_step_rows"]),("paired-windows.csv",paired_rows)):
        fields=sorted({key for row in rows for key in row})
        with (out/filename).open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({key:frozen.canonical(v) if isinstance(v,(dict,list)) else v for key,v in row.items()})
    links=''.join(f'<li><a href="{html.escape(page["file"])}">{page["cohort"]} · H{page["horizon"]} · {i+1}</a></li>' for i,page in enumerate(pages))
    (out/'media/index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C2全部同窗对照</title><style>body{font:17px sans-serif;max-width:1100px;margin:2em auto;line-height:1.6}img{max-width:100%}</style><h1>固定权重，原生3/8/16次去噪</h1><p>全部36窗H1/H4，共72行；每行真值、copy-last、训练后3/8/16。全96px画面最近邻显示，无裁图。有限越界仅PNG编码限幅且标注，原float/评分/回灌未变。</p><ul>'+links+'</ul>'+''.join(f'<h2>{page["cohort"]} H{page["horizon"]}</h2><img src="{page["file"]}" alt="全部固定窗口">' for page in pages),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2C2固定权重采样步数诊断</title><style>body{font:17px sans-serif;max-width:1000px;margin:2em auto;line-height:1.7}</style><h1>R2C2 固定权重采样步数诊断</h1><p>只改变原生去噪次数3/8/16，3和copy-last复用旧结果。耗时只统计，无训练或权重更新。</p><ul><li><a href="REPORT.md">完整结果、逐组结论与限制</a></li><li><a href="media/index.html">全部24页 / 36窗H1与H4对照</a></li><li><a href="r2c2-report.json">完整机器报告</a></li><li><a href="endpoint-scores.csv">288行端点数据（新144/复用144）</a></li><li><a href="paired-windows.csv">全部逐窗胜负与近静止分组</a></li><li><a href="all-raw-steps.csv">四步路径和中间失败</a></li></ul><p>形状未知不是通过；natural/motion来自同seen轨迹，不能声称独立未见泛化。原始float和运行内评分未冷重算。</p>',encoding='utf-8')
    from PIL import Image
    dimensions=[]
    for page in pages:
        with Image.open(out/'media'/page['file']) as image:
            image.verify()
        with Image.open(out/'media'/page['file']) as image:dimensions.append(list(image.size))
    qa=dict(kind='R2C2-complete-gallery-integrity-QA.v1',pages=pages,page_count=24,gallery_rows=72,
        methods_per_row=['truth',*METHODS],all_window_order_verified=True,png_integrity_verified=True,
        dimensions=dimensions,representative_visual_review_pending=True,finalized=False,
        cold_metric_recomputations=0,model_calls=0,checkpoint_loads=0,environment_constructions=0)
    write(out/'page-QA.json',qa)
    size=sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if size>20*1024**2:raise RuntimeError('Report/media exceed20MiB; no windows may be silently dropped')
    print(json.dumps(dict(endpoint_rows=len(report['rows']),windows=36,pages=24,gallery_rows=72,bytes=size,
        cold_metric_recomputations=0,own_model_calls=0,representative_visual_review_pending=True)))
    return 0


if __name__=='__main__':raise SystemExit(main())
