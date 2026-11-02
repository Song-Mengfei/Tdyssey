"""Freeze R2C0 evaluation indices and saved-RGB truth previews; no predictions."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c0_data import R2C0Dataset, LETTERS
from tdyssey.native_rgb_r2c0_evaluation import normalize_actions

DOMAINS=dict(train_visual="r2c0.train-visual-v1",val_natural="r2c0.val-natural-v1",val_motion="r2c0.val-motion-v1")
OTHER_PREVIEW_DOMAIN="r2c0.train-other-preview-v1"
CALIBRATION=ROOT/"artifacts/native-rgb-20261028/frozen-renderer-calibration.json"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def rank(cohort,trajectory,anchor,role=None):
    return canonical_hash(dict(domain=DOMAINS[cohort],seed=5478,trajectory_id=trajectory,anchor=int(anchor),role=role))


def selected_row(dataset,row,cohort,role,rank_sha,exposures=0):
    byid={entry["trajectory_id"]:i for i,entry in enumerate(dataset.entries)}
    entry=dataset.entries[byid[row["trajectory_id"]]];anchor=int(row["anchor"])
    return dict(deepcopy(row),window_id=f"{entry['trajectory_id']}:{anchor}",source_h1_window_id=row["window_id"],
        cohort=cohort,selection_role=role,selection_sha256=rank_sha,entry_index=byid[entry["trajectory_id"]],
        npz_sha256=entry["sha256"],metadata_sha256=entry["metadata_sha256"],first480_exposures=exposures,
        history_state_indices=list(range(anchor-3,anchor+1)),h1_target_state_index=anchor+1,h4_target_state_index=anchor+4,
        absolute_action_indices=list(range(anchor-3,anchor+4)),horizon4_legal=True)


def choose_indices(train,val,first480):
    """Metadata-only deterministic choices, before any model work."""
    if first480["seed"]!=5478 or len(first480["indices"])!=480:raise ValueError("Frozen first480 exposure list required")
    exposure=Counter(first480["window_ids"])
    if first480["window_ids"]!=[train.window_rows[i]["window_id"] for i in first480["indices"]]:raise ValueError("First480 indices/window IDs differ")
    result={name:[] for name in DOMAINS};missing=[];used=set()
    train_h4=set(train.h4_windows);val_h4=set(val.h4_windows)
    train_byid={e["trajectory_id"]:i for i,e in enumerate(train.entries)}
    for letter in LETTERS:
        for role,flag in (("translation","translation"),("rotation","rotation")):
            candidates=[row for row in train.window_rows if row["letter"]==letter and row[flag]
                and exposure[row["window_id"]]>0 and (train_byid[row["trajectory_id"]],row["anchor"]) in train_h4
                and (row["trajectory_id"],row["anchor"]) not in used]
            if not candidates:
                missing.append(dict(cohort="train_visual",letter=letter,role=role,reason="no distinct first480-exposed H4legal role window"));continue
            winner=min(candidates,key=lambda row:rank("train_visual",row["trajectory_id"],row["anchor"],role))
            used.add((winner["trajectory_id"],winner["anchor"]))
            result["train_visual"].append(selected_row(train,winner,"train_visual",role,
                rank("train_visual",winner["trajectory_id"],winner["anchor"],role),exposure[winner["window_id"]]))
    for entry_index,entry in enumerate(val.entries):
        candidates=[row for row in val.window_rows if row["trajectory_id"]==entry["trajectory_id"] and (entry_index,row["anchor"]) in val_h4]
        if not candidates:raise ValueError("Frozen val trajectory has no legal H4 history")
        winner=min(candidates,key=lambda row:rank("val_natural",row["trajectory_id"],row["anchor"]))
        result["val_natural"].append(selected_row(val,winner,"val_natural","unfiltered_hash_anchor",rank("val_natural",winner["trajectory_id"],winner["anchor"])))
        candidates=[row for row in candidates if row["moving"] and row["anchor"]!=winner["anchor"]]
        if not candidates:
            missing.append(dict(cohort="val_motion",letter=entry["letter"],trajectory_id=entry["trajectory_id"],reason="no distinct H1obvious-motion H4legal anchor after natural exclusion"));continue
        motion=min(candidates,key=lambda row:rank("val_motion",row["trajectory_id"],row["anchor"]))
        result["val_motion"].append(selected_row(val,motion,"val_motion","H1_obvious_motion",rank("val_motion",motion["trajectory_id"],motion["anchor"])))
    ids=[(row["trajectory_id"],row["anchor"]) for cohort in result.values() for row in cohort]
    if len(ids)!=len(set(ids)):raise ValueError("Evaluation windows duplicate across roles/cohorts")
    return result,missing


def protocol():
    return dict(kind="R2C0-same-window-native-RGB-evaluation.v1",seed=5478,
        scope="preparation only; zero current model/inference/training authorization",domains=DOMAINS,
        cohorts=dict(train_visual="12 unique TRAIN, translation then rotation per letter; minhash among frozenfirst480-exposed fullH4legal windows, prior role excluded",
            val_natural="1 minhash H4legal anchor per frozenvaltrajectory, no motion filter; trajectories deliberately contain motion, so not unbiased natural test",
            val_motion="at most1 H1obvious-motion H4legal anchor pervaltrajectory after natural exclusion; missing support retained"),
        obvious_motion="H1 area-centroid displacement>=2 image96px OR abs raw unwrapped rotation>=5deg; small_or_static is only a historical variable name for the complement, display near-static/small motion, never exact static; no old2worldpx main criterion",
        timeline=dict(history="states/images a-3..a,4 real observed RGB",h1_actions="actual a-3..a,4 actions",h4_audit_action_prefix="a-3..a+3,7 actual actions for audit, NEVER all7 in one GPC call",
            h1_truth="storedRGB[a+1]",h4_truth="storedRGB[a+4]",validity="all7 history/future transitions valid; no cross-trajectory windows"),
        model_callback=dict(entry="tdyssey.native_rgb_r2c0_evaluation.rollout_raw",history_shape="[N,4,3,96,96]",
            actions_shape="[N,4,2] already normalized",normalization="absolute.astype(float32)/511*2-1 exactly once; callback goes directly to normalized native sampler, NEVER old absolute-action wrapper or inverse/renormalize",
            feedback="original generated rawRGB only; finite out-of-range unchanged; no future truth/state/pose/template/environment render/repair",
            pairing="same cohort/order/batchshape and domain-separated perbatch+step seed for original/final; no claim of independent perwindow diffusion noise",
            hard_stop="nonfinite/wrongshape/callback failure stopscohort, preserves rawfailure and every frozen denominator; no retries"),
        methods=["copy_last","original_published_GPC","trained_final"],methods_currently_executed=[],
        h1_reuses_first_h4_call=True,training_result_available=False,
        score=dict(entry="tdyssey.native_rgb_r2c0_evaluation.score_endpoints",metrics_source="scripts/native_rgb_stage3_score.py",
            metrics_source_sha256=sha(ROOT/"scripts/native_rgb_stage3_score.py"),
            legacy_evaluation_source_sha256=sha(ROOT/"tdyssey/native_rgb_evaluation.py"),
            calibration_file=str(CALIBRATION.relative_to(ROOT)),calibration_file_sha256=sha(CALIBRATION),
            object="unaligned visible-gray 1-IoU,gray fixedLAST|TRUE ROI RGBMAE,raw contour distance; blue pusher metrics separate",
            small_or_static="report gray prediction-vs-LAST drift separately for H1non-obvious windows, with actual truth amplitudes; not a requirement to beat zero-error copylast",
            hard_failures="status/missing/wrongshape/nonfinite/out-of-range/emptygray/emptyblue retain denominator and existing penalties; rawvalues saved unmodified",
            shape="occlusion/border/unassessable remains unknown; failFalse alone is never a shape/rigidity guarantee",
            acceptance="no new pass target declared here; paired errors/wins/hardfail/unknown per cohort and horizon, TRAIN fit is not unseen-shape transfer"),
        display="truth preview only from saveduint8RGB; white/gray/blue unaltered, action charts/text outside image panels; no predictions created")


def load_eval_cohort(manifest_path,indices_path,cohort):
    """Prepare saved arrays explicitly; no training order or predictor is touched."""
    import numpy as np
    indices_path=Path(indices_path).resolve();indices=read(indices_path)
    if cohort not in DOMAINS:raise ValueError("Unknown explicit evaluation cohort")
    if sha(manifest_path)!=indices["data_identity"]["data_manifest_sha256"]:raise ValueError("Frozen data manifest differs")
    split="train" if cohort=="train_visual" else "seen_val"
    ds=R2C0Dataset(manifest_path,split=split,cache_bytes=1024*1024)
    histories=[];actions=[];h1=[];h4=[];audit=[]
    for row in indices["cohorts"][cohort]:
        entry,arrays=ds.trajectory(int(row["entry_index"]));a=int(row["anchor"])
        if (entry["trajectory_id"]!=row["trajectory_id"] or entry["sha256"]!=row["npz_sha256"]
                or entry["split"]!=row["split"] or (int(row["entry_index"]),a) not in set(ds.h4_windows)):
            raise ValueError("Evaluation split/trajectory/anchor/source identity differs")
        frames=arrays["rgb"];history=frames[a-3:a+1].astype(np.float32)/np.float32(255)
        absolute=arrays["actions"][a-3:a+4].copy()
        item=ds[int(row["dataset_index"])]
        if not np.array_equal(item["image"][:4],history) or not np.array_equal(item["action"],normalize_actions(absolute[:4])):
            raise ValueError("Actual dataset image/action model interface alignment differs")
        selected_frames=list(range(a-3,a+1))+[a+1,a+4]
        hashes=[hashlib.sha256(frames[t].tobytes()).hexdigest() for t in selected_frames]
        if hashes!=[entry["source_frame_sha256"][t] for t in selected_frames]:raise ValueError("Saved rawRGB frame bytes differ")
        histories.append(history);actions.append(absolute);h1.append(frames[a+1].copy());h4.append(frames[a+4].copy())
        audit.append(dict(window_id=row["window_id"],frame_indices=selected_frames,frame_sha256=hashes,
            action_indices=list(range(a-3,a+4)),actions_absolute=absolute.tolist(),
            model_h1_actions_normalized=normalize_actions(absolute[:4]).tolist(),dataset_image_action_bitwise_equal=True))
    ids=[row["window_id"] for row in indices["cohorts"][cohort]]
    identity=dict(indices["data_identity"],indices_file_sha256=sha(indices_path),protocol_file_sha256=sha(indices_path.with_name("protocol.json")),
        cohort=cohort,window_ids_sha256=canonical_hash(ids))
    ds.close()
    n=len(ids)
    return dict(histories=np.stack(histories) if n else np.empty((0,4,3,96,96),np.float32),
        absolute_actions=np.stack(actions) if n else np.empty((0,7,2)),
        truth_h1=np.stack(h1) if n else np.empty((0,3,96,96),np.uint8),truth_h4=np.stack(h4) if n else np.empty((0,3,96,96),np.uint8),
        window_ids=ids,metadata=deepcopy(indices["cohorts"][cohort]),identity=identity,alignment_audit=audit,
        predictor_calls=0,render_calls=0,physics_controls=0,training_draws_consumed=0)


def prepare_other_examples(manifest_path):
    """Display-only examples of existing first480other exposures, not eval rows."""
    import numpy as np
    manifest_path=Path(manifest_path).resolve();ds=R2C0Dataset(manifest_path,"train")
    draws=read(manifest_path.with_name("draws480.json"));exposure=Counter(draws["window_ids"])
    h4=set(ds.h4_windows);byid={e["trajectory_id"]:i for i,e in enumerate(ds.entries)}
    h4meta={r["window_id"].replace(":H4",""):r for r in read(manifest_path.with_name("h4-windows.json"))}
    metadata={e["trajectory_id"]:read(ds.source_root/e["metadata_file"]) for e in ds.entries}
    eligible=[r for r in ds.window_rows if not r["moving"] and exposure[r["window_id"]]>0 and (byid[r["trajectory_id"]],r["anchor"]) in h4]
    tags=("nearstatic_AO","nearstatic_noAO","small","planned_release","actual_AO_loss","wallstall")
    def matches(row,tag):
        near=row["translation_image96_px"]<.1 and row["abs_rotation_deg"]<.5
        if tag=="nearstatic_AO":return near and row["agent_object_contact"] is True
        if tag=="nearstatic_noAO":return near and row["agent_object_contact"] is False
        if tag=="small":return not near
        if tag=="planned_release":return 14<=row["anchor"]<=19
        if tag=="wallstall":return row["boundary_process"]=="observed_wall_contact_stall"
        records=metadata[row["trajectory_id"]]["control_records"];a=row["anchor"]
        return records[a]["agent_object_contact_substeps"]==0 and records[a-1]["agent_object_contact_substeps"]>0
    def order(row,tag):return canonical_hash(dict(domain=OTHER_PREVIEW_DOMAIN,seed=5478,tag=tag,trajectory_id=row["trajectory_id"],anchor=row["anchor"]))
    chosen=[];missing=[];used=set();counts={}
    for tag in tags:
        candidates=[r for r in eligible if matches(r,tag)];counts[tag]=len(candidates)
        candidates=[r for r in candidates if (r["trajectory_id"],r["anchor"]) not in used]
        if not candidates:missing.append(dict(tag=tag,reason="no distinct first480exposed other H4legal window"));continue
        winner=min(candidates,key=lambda r:order(r,tag));used.add((winner["trajectory_id"],winner["anchor"]))
        row=selected_row(ds,winner,"train_other_preview",tag,order(winner,tag),exposure[winner["window_id"]])
        future=h4meta[row["window_id"]];row.update(h4_translation_image96_px=future["translation_image96_px"],h4_signed_rotation_deg=future["signed_rotation_deg"])
        chosen.append(row)
    histories=[];actions=[];truth1=[];truth4=[];alignment=[]
    for row in chosen:
        entry,arrays=ds.trajectory(row["entry_index"]);a=row["anchor"];frames=arrays["rgb"]
        indices=list(range(a-3,a+1))+[a+1,a+4];hashes=[hashlib.sha256(frames[t].tobytes()).hexdigest() for t in indices]
        if hashes!=[entry["source_frame_sha256"][t] for t in indices]:raise ValueError("Otherpreview savedframe hash differs")
        histories.append(frames[a-3:a+1].astype(np.float32)/np.float32(255));actions.append(arrays["actions"][a-3:a+4].copy())
        truth1.append(frames[a+1].copy());truth4.append(frames[a+4].copy());alignment.append(dict(window_id=row["window_id"],frame_indices=indices,frame_sha256=hashes))
    ds.close()
    sidecar=dict(kind="R2C0-first480-TRAIN-other-preview-only.v1",evaluation_cohort=False,selection_domain=OTHER_PREVIEW_DOMAIN,seed=5478,
        data_manifest_sha256=sha(manifest_path),first480_draws_sha256=sha(manifest_path.with_name("draws480.json")),
        near_static_rule="<.1imagepx and<.5deg audit-only display tag, not exact stationary or a change to80/20drawpolicy",
        tag_order=list(tags),eligible_unique_counts=counts,rows=chosen,missing_support=missing,alignment=alignment,
        additional_training_draws=0,additional_evaluation_windows=0,indices_changed=False)
    prepared=dict(metadata=chosen,histories=np.stack(histories),absolute_actions=np.stack(actions),truth_h1=np.stack(truth1),truth_h4=np.stack(truth4)) if chosen else None
    return sidecar,prepared


def previews(manifest_path,indices_path,out,supplemental=None):
    """Scientific panels of stored pixels only; annotations never enter RGB."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fontpath=Path("C:/Windows/Fonts/msyh.ttc")
    if fontpath.exists():plt.rcParams["font.family"]=FontProperties(fname=str(fontpath)).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    output=[]
    prepared_cohorts={name:load_eval_cohort(manifest_path,indices_path,name) for name in DOMAINS}
    if supplemental is not None:prepared_cohorts["train_other_preview"]=supplemental
    for cohort,prepared in prepared_cohorts.items():
        metadata=prepared["metadata"]
        for start in range(0,len(metadata),6):
            selected=metadata[start:start+6];fig,axes=plt.subplots(len(selected),7,figsize=(14,2.15*len(selected)),squeeze=False,
                gridspec_kw=dict(width_ratios=[1,1,1,1,1,1,1.9]))
            fig.subplots_adjust(left=.025,right=.99,top=.89,bottom=.035,hspace=.85,wspace=.1)
            title="TRAIN抽样示例（非评估cohort）" if cohort=="train_other_preview" else cohort
            fig.suptitle(f"R2C0 原始真RGB｜{title}｜0.1秒半衰期（已选）｜无预测/无重渲染",fontsize=13,y=.985)
            for local,row in enumerate(selected):
                i=start+local;a=row["anchor"]
                frames=list(prepared["histories"][i])+[prepared["truth_h1"][i]/255,prepared["truth_h4"][i]/255]
                labels=[f"历史 t−{3-j}" if j<3 else "历史 t" for j in range(4)]+["真 H1 t+1","真 H4 t+4"]
                for j,(frame,label) in enumerate(zip(frames,labels)):
                    axes[local,j].imshow(np.moveaxis(frame,0,-1),interpolation="nearest",vmin=0,vmax=1)
                    axes[local,j].set_title(label,fontsize=9);axes[local,j].axis("off")
                category={"translation_only":"明显平移","rotation_only":"明显旋转","both":"平移+旋转","other":"近静止/小运动"}[row["category"]]
                exposure=f"首480实际暴露{row['first480_exposures']}次" if row["split"]=="train" else "val冻结评估窗，不参与训练draw"
                if cohort=="train_other_preview":exposure+=f" | {row['selection_role']}"
                boundary={"interior":"内部","along_near_boundary_geometry":"沿边（几何）","approaching_boundary_geometry":"靠近边界（几何）",
                    "leaving_boundary_geometry":"离开边界（几何）","observed_wall_contact_stall":"真实墙接触小运动","boundary_other":"其它边界过程"}.get(row["boundary_process"],row["boundary_process"])
                def value(x):return f"{x:+.2e}" if abs(x)<.001 else f"{x:+.3f}"
                annotation=(f"{row['split']} | {row['letter']} {row['family_id']} | a={a} | {category} | {exposure}\n"
                    f"真幅度 H1 Δ={value(row['translation_image96_px'])}px θ={value(row['signed_rotation_deg'])}°；"
                    f"H4 Δ={value(row['h4_translation_image96_px'])}px θ={value(row['h4_signed_rotation_deg'])}° | {boundary}")
                axes[local,0].text(0,1.18,annotation,transform=axes[local,0].transAxes,fontsize=7,ha="left",va="bottom",linespacing=1.4)
                ax=axes[local,6];ax.axis("off");target=prepared["absolute_actions"][i]
                ax.text(.02,.98,"绝对目标及增量方向（图外）",transform=ax.transAxes,fontsize=7,va="top")
                for j,xy in enumerate(target):
                    y=.82-j*.105
                    ax.text(.28,y,f"t{j-3:+d}: ({xy[0]:.1f}, {xy[1]:.1f})",transform=ax.transAxes,fontsize=7,va="center")
                    delta=target[j]-target[j-1] if j else np.zeros(2);length=float(np.linalg.norm(delta))
                    if length>1e-9:ax.annotate("",xy=(.13+.06*delta[0]/length,y-.06*delta[1]/length),xytext=(.13,y),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=.9,color="tab:blue"))
                    else:ax.plot([.13],[y],".",transform=ax.transAxes,color="tab:blue",markersize=3)
                ax.text(.02,.015,"箭头非物体运动；全部7个动作",transform=ax.transAxes,fontsize=6,va="bottom")
            filename=f"truth-{cohort}-{start//6+1:02d}.png";path=out/filename
            fig.savefig(path,dpi=125,facecolor="white");plt.close(fig)
            output.append(dict(file=filename,sha256=sha(path),bytes=path.stat().st_size,window_ids=[r["window_id"] for r in selected]))
    return output


def prepare(manifest_path,out):
    manifest_path=Path(manifest_path).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    train=R2C0Dataset(manifest_path,"train");val=R2C0Dataset(manifest_path,"seen_val")
    draws_path=manifest_path.with_name("draws480.json");draws=read(draws_path)
    cohorts,missing=choose_indices(train,val,draws)
    identity=dict(data_manifest_sha256=sha(manifest_path),split_plan_sha256=sha(manifest_path.with_name("split-plan.json")),
        h1_catalogue_sha256=sha(manifest_path.with_name("h1-windows.json")),h4_catalogue_sha256=sha(manifest_path.with_name("h4-windows.json")),first480_draws_sha256=sha(draws_path))
    h4={row["window_id"].replace(":H4",""):row for row in read(manifest_path.with_name("h4-windows.json"))}
    for rows in cohorts.values():
        for row in rows:
            future=h4[row["window_id"]];row.update(h4_translation_image96_px=future["translation_image96_px"],h4_signed_rotation_deg=future["signed_rotation_deg"])
    train.close();val.close()
    write(out/"protocol.json",protocol())
    indices=dict(kind="R2C0-frozen-evaluation-indices.v1",seed=5478,data_identity=identity,domains=DOMAINS,cohorts=cohorts,missing_support=missing,
        val_scope="Anchor-natural within deliberately motion-containing seen-val families; not an unbiased natural or unseen-shape test",
        model_work_authorized=False,predictions_created=False)
    write(out/"indices.json",indices)
    prepared={name:load_eval_cohort(manifest_path,out/"indices.json",name) for name in DOMAINS}
    rows=[row for cohort in cohorts.values() for row in cohort]
    overlap=[]
    for natural in cohorts["val_natural"]:
        for motion in cohorts["val_motion"]:
            if natural["trajectory_id"]==motion["trajectory_id"]:
                a,b=natural["anchor"],motion["anchor"]
                overlap.append(dict(natural=natural["window_id"],motion=motion["window_id"],
                    shared_history_or_future_states=len(set(range(a-3,a+5))&set(range(b-3,b+5))),
                    shared_future_controls=len(set(range(a,a+4))&set(range(b,b+4)))))
    other_sidecar,other_prepared=prepare_other_examples(manifest_path)
    write(out/"train-other-preview.json",other_sidecar)
    images=previews(manifest_path,out/"indices.json",out,other_prepared)
    audit=dict(kind="R2C0-evaluation-preparation-audit.v1",indices_sha256=sha(out/"indices.json"),protocol_sha256=sha(out/"protocol.json"),
        counts={name:len(values) for name,values in cohorts.items()},missing_support=missing,
        cohort_identities={name:value["identity"] for name,value in prepared.items()},
        alignment={name:value["alignment_audit"] for name,value in prepared.items()},previews=images,
        coverage=dict(obvious_translation=sum(row["translation"] for row in rows),positive_rotation_ge5=sum(row["signed_rotation_deg"]>=5 for row in rows),
            negative_rotation_le_minus5=sum(row["signed_rotation_deg"]<=-5 for row in rows),small_or_static=sum(not row["moving"] for row in rows),
            boundary=sum(row["boundary_process"]!="interior" for row in rows)),
        temporal_overlap_within_val_cohorts=overlap,cohort_scores_must_remain_separate=True,
        all_model_input_histories_actions_match_dataset=True,all_selected_frame_hashes_match_source=True,
        train_visual_windows_unique=len({r["window_id"] for r in cohorts["train_visual"]})==len(cohorts["train_visual"]),
        natural_motion_window_duplicates=False,predictor_calls=0,model_calls=0,physics_controls=0,environment_constructions=0,
        environment_render_calls=0,training_draws_consumed=0,torch_imported="torch" in sys.modules,
        output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()))
    write(out/"preparation-audit.json",audit)
    report=["# R2C0 评估准备（尚未运行模型）","",f"固定TRAIN可视{len(cohorts['train_visual'])}、val natural{len(cohorts['val_natural'])}、val motion{len(cohorts['val_motion'])}窗。缺层{len(missing)}项。",
        "","TRAIN可视窗均在首480实际draw出现且完整H4合法；同letter先平移后旋转，排除重复窗。natural每个已冻结val轨迹仅hash抽anchor，不筛运动；val轨迹本身是刻意含运动的诊断切分。motion与natural按trajectory+anchor去重，仍可能共享同轨迹历史/未来，统计分开。",
        "","4历史RGB来自存储原帧；每步模型只收4动作，H4七动作前缀仅审计。原GPC/训练后同窗同batch/seed；H1复用H4第一步。有限越界原值回灌并保留硬失败，非finite/shape错误硬停；无未来truth/state/pose/重渲染/修复。",
        "","本次只有真RGB预览和CPU接口检查，无预测、训练或300/1200执行许可。原物理及模型账本未动。"]
    (out/"README.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    if sum(p.stat().st_size for p in out.rglob("*") if p.is_file())>10*1024**2:raise RuntimeError("Evaluation/previews exceed10MiBallocation")
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-manifest",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();result=prepare(args.data_manifest,args.out)
    print(json.dumps(dict(counts=result["counts"],missing=len(result["missing_support"]),model_calls=0,render_calls=0,coverage=result["coverage"])))
    return 0


if __name__=="__main__":raise SystemExit(main())
