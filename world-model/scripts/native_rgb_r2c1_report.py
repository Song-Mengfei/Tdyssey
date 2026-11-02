"""Cold R2C1 raw-prediction scoring and complete truth/baseline/final galleries.

No model, Torch, simulator, environment rendering or training is invoked here.
Frozen R2C0 endpoint metric math is called with an explicit calibration object.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2c0_evaluation import load_eval_cohort
from tdyssey.native_rgb_r2c0_evaluation import score_endpoints
from tdyssey.native_rgb_evaluation import palette_masks

COHORTS=("train_visual","val_natural","val_motion")
METHODS=("copy_last","original_GPC","trained")
OBJECT_METRICS=("raw_object_error","raw_mask_iou","object_foreground_rgb_mae","raw_contour_error_image_px")
PUSHER_METRICS=("pusher_error","pusher_mask_iou","pusher_foreground_rgb_mae","pusher_position_error_image_px")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):raise ValueError("Prediction path escapes evaluation root")
    return path
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def fmt(value,digits=3):return "未评" if value is None else format(value,f".{digits}f")


def numeric_status(raw):
    import numpy as np
    if raw is None:return "missing"
    raw=np.asarray(raw)
    if raw.shape!=(3,96,96) or not np.issubdtype(raw.dtype,np.floating):return "wrong_shape_or_dtype"
    if not np.isfinite(raw).all():return "nonfinite"
    return "finite_out_of_range" if raw.min()<0 or raw.max()>1 else "ok"


def step_quality(raw,calibration):
    """A midstep diagnostic only; does not change frozen endpoint score math."""
    import numpy as np
    status=numeric_status(raw);reasons=[] if status=="ok" else [status]
    result=dict(numeric_status=status,hard_failure=status!="ok",failure_reasons=reasons,
        raw_min=None,raw_max=None,raw_dtype=None if raw is None else str(np.asarray(raw).dtype))
    if raw is not None and np.issubdtype(np.asarray(raw).dtype,np.number):
        finite=np.asarray(raw)[np.isfinite(raw)]
        if len(finite):result.update(raw_min=float(finite.min()),raw_max=float(finite.max()))
    if status=="ok":
        gray,blue=palette_masks(raw,calibration)
        if not gray.any():reasons.append("empty_gray")
        if not blue.any():reasons.append("empty_blue")
        result.update(hard_failure=bool(reasons),gray_visible_pixels=int(gray.sum()),blue_visible_pixels=int(blue.sum()))
    return result


def display_encoding(raw):
    """Display-only copy, never fed back into scoring or any model."""
    import numpy as np
    status=numeric_status(raw)
    if status in ("missing","wrong_shape_or_dtype","nonfinite"):
        return None,"无效/未生成："+status
    # PNG needs a bounded integer encoding. No original array is modified.
    encoded=np.rint(np.clip(np.asarray(raw),0,1)*255).astype(np.uint8)
    note="" if status=="ok" else f"越界硬失败；仅显示编码截幅 [{float(np.min(raw)):.4g},{float(np.max(raw)):.4g}]"
    return np.moveaxis(encoded,0,-1),note


def closed_inventory(model_work,evaluation_root):
    work=read(model_work)
    if work.get("closed") is not True:raise RuntimeError("Wait for root runclosed; refusing hot predictions")
    inventory={}
    for row in work.get("evaluation_inventory",[]):
        name=row["file"]
        if name in inventory:raise ValueError("Duplicate prediction inventory path")
        path=safe(evaluation_root,name)
        if not path.is_file() or sha(path)!=row["sha256"] or ("bytes" in row and path.stat().st_size!=row["bytes"]):
            raise ValueError("Closed prediction inventory SHA/size differs")
        inventory[name]=row
    return work,inventory


def load_predictions(evaluation_root,stage,cohort,prepared,inventory,calibration):
    import numpy as np
    root=Path(evaluation_root).resolve();folder=root/stage;name=f"cohort-{cohort}"
    ids=prepared["window_ids"];n=len(ids)
    absent=dict(window_ids=ids,planned_windows=n,h1=np.full((n,3,96,96),np.nan),h4=np.full((n,3,96,96),np.nan),
        status_h1=["missing_file"]*n,status_h4=["missing_file"]*n,step_records=[],hard_stop=True)
    raw_steps=[[None]*4 for _ in ids]
    for extension in ("npz","json"):
        relative=f"{stage}/{name}.{extension}"
        if relative not in inventory:
            if (root/relative).exists():raise ValueError("Prediction exists but is not bound by closed inventory")
            return absent,raw_steps,dict(files_missing=True,raw_steps_expected=4*n,raw_steps_present=0)
    sidecar=read(folder/(name+".json"))
    if sidecar.get("cohort_identity")!=prepared["identity"] or sidecar.get("window_ids")!=ids:
        raise ValueError("Prediction cohort/data/indices/protocol identity differs")
    with np.load(folder/(name+".npz"),allow_pickle=False) as z:
        if list(z["window_ids"])!=ids:raise ValueError("Prediction NPZ window order differs")
        result={key:z[key].copy() for key in ("h1","h4")}
        result.update(window_ids=ids,planned_windows=n,status_h1=[str(v) for v in z["status_h1"]],status_h4=[str(v) for v in z["status_h4"]])
    if any(result[key].shape!=(n,3,96,96) for key in ("h1","h4")) or any(len(result[k])!=n for k in ("status_h1","status_h4")):
        raise ValueError("Endpoint arrays/status denominators differ")
    result.update(step_records=sidecar.get("step_records",[]),hard_stop=sidecar.get("hard_stop",False))
    positions={identifier:i for i,identifier in enumerate(ids)};seen=set();raw_files=[]
    for row in sidecar.get("raw_step_files",[]):
        path=safe(folder,row["file"]);step=int(row["step"])
        if not 0<=step<4 or sha(path)!=row["sha256"]:raise ValueError("Raw step SHA or step index differs")
        rel=path.relative_to(root).as_posix()
        if rel in inventory and inventory[rel]["sha256"]!=row["sha256"]:raise ValueError("Raw-step inventory mismatch")
        with np.load(path,allow_pickle=False) as z:
            raw=z["raw"].copy();batch_ids=[str(v) for v in z["window_ids"]]
        if row["window_ids"]!=batch_ids or len(batch_ids)!=len(set(batch_ids)) or set(batch_ids)-set(ids):raise ValueError("Raw step window membership differs")
        if raw.shape!=(len(batch_ids),3,96,96):
            for identifier in batch_ids:
                key=(identifier,step)
                if key in seen:raise ValueError("Duplicate raw step")
                seen.add(key);raw_steps[positions[identifier]][step]=raw.copy()
        else:
            for j,identifier in enumerate(batch_ids):
                key=(identifier,step)
                if key in seen:raise ValueError("Duplicate raw step")
                seen.add(key);raw_steps[positions[identifier]][step]=raw[j].copy()
        raw_files.append(dict(file=rel,sha256=row["sha256"],step=step,window_ids=batch_ids,raw_dtype=str(raw.dtype),raw_shape=list(raw.shape)))
    errors=[]
    for i,identifier in enumerate(ids):
        for step,key in ((0,"h1"),(3,"h4")):
            raw=raw_steps[i][step]
            if raw is not None and raw.shape==(3,96,96) and not np.array_equal(raw,result[key][i],equal_nan=True):
                errors.append(dict(window_id=identifier,endpoint=key,reason="raw step and endpoint values differ"))
    if errors:raise ValueError("H1/H4 aliases differ from original step values")
    return result,raw_steps,dict(files_missing=False,raw_steps_expected=4*n,raw_steps_present=len(seen),
        evaluation_complete=sidecar.get("evaluation_complete"),raw_files=raw_files,step_records=sidecar.get("step_records",[]))


def score_method(prepared,result,raw_steps,calibration,method):
    endpoint=score_endpoints(result,prepared["truth_h1"],prepared["truth_h4"],prepared["histories"][:,-1],prepared["metadata"],calibration)
    step_rows=[];path={}
    for i,identifier in enumerate(prepared["window_ids"]):
        quality=[]
        for step in range(4):
            item=dict(window_id=identifier,method=method,step=step+1,**step_quality(raw_steps[i][step],calibration))
            step_rows.append(item);quality.append(item)
        path[identifier]=quality
    rows=[]
    for row in endpoint:
        i=prepared["window_ids"].index(row["window_id"]);meta=prepared["metadata"][i];h=row["horizon"]
        failures=[q for q in path[row["window_id"]][:h] if q["hard_failure"]]
        rows.append(dict(row,method=method,cohort=meta["cohort"],letter=meta["letter"],family_id=meta["family_id"],
            anchor=meta["anchor"],H1_truth_translation_image_px=meta["translation_image96_px"],H1_truth_signed_rotation_deg=meta["signed_rotation_deg"],
            H4_truth_translation_image_px=meta["h4_translation_image96_px"],H4_truth_signed_rotation_deg=meta["h4_signed_rotation_deg"],
            H1_obvious_motion=bool(meta["moving"]),H1_nearstatic_aux=bool(meta["translation_image96_px"]<.1 and meta["abs_rotation_deg"]<.5),
            endpoint_hard_generation_failure=bool(row["hard_generation_failure"]),
            any_path_step_hard_failure=bool(failures),path_step_failure_count=len(failures),
            path_failure_steps=[q["step"] for q in failures],path_failure_reasons=[q["failure_reasons"] for q in failures],
            all_steps_and_endpoint_valid=not failures and not row["hard_generation_failure"]))
    return rows,step_rows


def _numeric_digest(array):
    """Value identity tolerates losslessfloat32/float64 endpoint containers."""
    import numpy as np
    a=np.asarray(array)
    if np.issubdtype(a.dtype,np.floating):
        a=a.astype(np.float64,copy=True);a[np.isnan(a)]=np.nan
    return dict(shape=list(a.shape),sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())


def metrics_input_identity(prepared,result,raw_steps):
    return dict(window_ids=list(prepared["window_ids"]),cohort_identity=prepared["identity"],
        h1=_numeric_digest(result["h1"]),h4=_numeric_digest(result["h4"]),
        status_h1=list(result["status_h1"]),status_h4=list(result["status_h4"]),
        truth_h1=_numeric_digest(prepared["truth_h1"]),truth_h4=_numeric_digest(prepared["truth_h4"]),
        last=_numeric_digest(prepared["histories"][:,-1]),
        raw_steps=[[None if raw is None else _numeric_digest(raw) for raw in steps] for steps in raw_steps])


def evaluate_saved_predictions(prepared,result,raw_steps,calibration,*,method,cohort):
    """Budgeted CPU scoring API: runner must call inside its active model clock.

    No model/inference is performed. The returned JSON envelope is later read
    by the cold report; cold reporting never fills unperformed metric work.
    """
    if method not in METHODS or cohort not in COHORTS or result["window_ids"]!=prepared["window_ids"]:
        raise ValueError("Explicit fixedmethod/cohort/windoworder required")
    n=len(prepared["window_ids"])
    if len(raw_steps)!=n or any(len(steps)!=4 for steps in raw_steps):raise ValueError("Exactly4step slots perwindow includingmissing")
    rows,step_rows=score_method(prepared,result,raw_steps,calibration,method)
    return dict(kind="R2C1-budgeted-endpoint-and-path-metrics.v1",evaluation_complete=True,scoring_complete=True,method=method,cohort=cohort,
        cohort_identity=prepared["identity"],window_ids=list(prepared["window_ids"]),
        input_identity=metrics_input_identity(prepared,result,raw_steps),
        calibration_canonical_sha256=hashlib.sha256(canonical(calibration).encode()).hexdigest(),
        scorer_source_sha256=sha(Path(__file__)),rows=rows,raw_step_rows=step_rows,
        endpoint_math="unchangedR2C0score_endpoints",requires_runner_active_budget=True,model_calls=0)


def metrics_placeholder(prepared,*,method,cohort,reason="not_attempted"):
    """Initialize an unperformed metric slot; never a fabricated score."""
    if method not in METHODS or cohort not in COHORTS:raise ValueError("Unknown fixedmetric cohort/method")
    return dict(kind="R2C1-budgeted-endpoint-and-path-metrics.v1",evaluation_complete=False,scoring_complete=False,
        method=method,cohort=cohort,cohort_identity=prepared["identity"],window_ids=list(prepared["window_ids"]),missing_reason=reason)


def missing_metrics(prepared,method):
    """Missing budgeted evaluation is unknown, never an invented measurement."""
    rows=[];steps=[]
    for i,identifier in enumerate(prepared["window_ids"]):
        meta=prepared["metadata"][i]
        for horizon in (1,4):
            row=dict(window_id=identifier,horizon=horizon,method=method,cohort=meta["cohort"],letter=meta["letter"],family_id=meta["family_id"],anchor=meta["anchor"],
                H1_truth_translation_image_px=meta["translation_image96_px"],H1_truth_signed_rotation_deg=meta["signed_rotation_deg"],
                H4_truth_translation_image_px=meta["h4_translation_image96_px"],H4_truth_signed_rotation_deg=meta["h4_signed_rotation_deg"],
                H1_obvious_motion=bool(meta["moving"]),H1_nearstatic_aux=meta["translation_image96_px"]<.1 and meta["abs_rotation_deg"]<.5,
                endpoint_hard_generation_failure=None,any_path_step_hard_failure=None,all_steps_and_endpoint_valid=False,
                native_numeric_status="budgeted_metrics_missing",shape_outcome="unknown",metrics_not_performed=True,
                path_failure_steps=[],path_failure_reasons=[])
            for key in OBJECT_METRICS+PUSHER_METRICS+("object_drift_from_last_error","object_drift_from_last_image_px"):row[key]=None
            rows.append(row)
        for step in range(1,5):steps.append(dict(window_id=identifier,method=method,cohort=meta["cohort"],step=step,
            hard_failure=None,numeric_status="budgeted_metrics_missing",failure_reasons=["unperformed_budgeted_evaluation"]))
    return rows,steps


def load_metrics(evaluation_root,stage,cohort,prepared,result,raw_steps,inventory,calibration,method):
    relative=f"{stage}/cohort-{cohort}.metrics.json"
    if relative not in inventory:
        if safe(evaluation_root,relative).exists():raise ValueError("Metrics not bound byclosedinventory")
        return (*missing_metrics(prepared,method),dict(file=relative,missing=True))
    path=safe(evaluation_root,relative);value=read(path)
    if (value.get("kind")=="R2C1-budgeted-endpoint-and-path-metrics.v1" and value.get("evaluation_complete") is False
            and value.get("method")==method and value.get("cohort")==cohort and value.get("cohort_identity")==prepared["identity"]
            and value.get("window_ids")==prepared["window_ids"]):
        return (*missing_metrics(prepared,method),dict(file=relative,sha256=sha(path),missing=True,missing_reason=value.get("missing_reason")))
    if (value.get("kind")!="R2C1-budgeted-endpoint-and-path-metrics.v1" or value.get("method")!=method or value.get("cohort")!=cohort
            or value.get("cohort_identity")!=prepared["identity"] or value.get("window_ids")!=prepared["window_ids"]
            or value.get("input_identity")!=metrics_input_identity(prepared,result,raw_steps)
            or value.get("calibration_canonical_sha256")!=hashlib.sha256(canonical(calibration).encode()).hexdigest()
            or value.get("scorer_source_sha256")!=sha(Path(__file__))):
        raise ValueError("Budgeted metrics input/source/calibration identity differs")
    if len(value["rows"])!=2*len(prepared["window_ids"]) or len(value["raw_step_rows"])!=4*len(prepared["window_ids"]):raise ValueError("Budgetedmetrics denominators differ")
    return value["rows"],value["raw_step_rows"],dict(file=relative,sha256=sha(path),missing=False)


def finite_metric_summary(rows,key):
    values=[row.get(key) for row in rows]
    # Primary measurements must retain finite failure penalties for every row.
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return dict(denominator=len(values),mean=None,unavailable=sum(v is None or not math.isfinite(float(v)) for v in values))
    return dict(denominator=len(values),mean=sum(values)/len(values) if values else None,
        min=min(values) if values else None,max=max(values) if values else None,unavailable=0)


def summary(rows):
    return dict(denominator=len(rows),endpoint_hard_failures=sum(r["endpoint_hard_generation_failure"] is True for r in rows),
        path_hard_failures=sum(r["any_path_step_hard_failure"] is True for r in rows),
        metrics_not_performed=sum(r.get("metrics_not_performed",False) for r in rows),
        missing_or_unattempted=sum(any(tag in r["native_numeric_status"] for tag in ("missing","not_attempted","callback","before_H4")) for r in rows),
        shape_outcomes=dict(Counter(r["shape_outcome"] for r in rows)),
        gray_object={k:finite_metric_summary(rows,k) for k in OBJECT_METRICS},
        blue_pusher={k:finite_metric_summary(rows,k) for k in PUSHER_METRICS},
        gray_drift_from_last={k:finite_metric_summary(rows,k) for k in ("object_drift_from_last_error","object_drift_from_last_image_px")})


def paired(rows,method,reference):
    if len({r["horizon"] for r in rows})>1:raise ValueError("Paired comparison requires one explicit horizon")
    a={r["window_id"]:r for r in rows if r["method"]==method};b={r["window_id"]:r for r in rows if r["method"]==reference}
    if set(a)!=set(b):raise ValueError("Paired method denominators differ")
    details=[]
    for key in a:
        left,right=a[key],b[key]
        delta=None if left["raw_object_error"] is None or right["raw_object_error"] is None else left["raw_object_error"]-right["raw_object_error"]
        outcome="not_evaluated" if delta is None else "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        details.append(dict(window_id=key,method=method,reference=reference,gray_object_error_delta=delta,outcome=outcome,
            all_step_valid_win=outcome=="win" and left["all_steps_and_endpoint_valid"],
            method_endpoint_hard_failure=left["endpoint_hard_generation_failure"],method_path_hard_failure=left["any_path_step_hard_failure"]))
    mean_a=sum(v["raw_object_error"] for v in a.values())/len(a) if a and all(v["raw_object_error"] is not None for v in a.values()) else None
    mean_b=sum(v["raw_object_error"] for v in b.values())/len(b) if b and all(v["raw_object_error"] is not None for v in b.values()) else None
    return dict(denominator=len(details),endpoint_outcomes=dict(Counter(r["outcome"] for r in details)),
        all_step_valid_wins=sum(r["all_step_valid_win"] for r in details),method_gray_mean=mean_a,reference_gray_mean=mean_b,
        relative_gray_error_reduction=None if mean_b in (None,0) or mean_a is None else (mean_b-mean_a)/mean_b,
        relative_improvement_undefined_zero_reference=mean_b==0,windows=details)


def build_report(model_work,evaluation_root,manifest_path,prepared_root,calibration_path):
    import numpy as np
    work,inventory=closed_inventory(model_work,evaluation_root)
    prepared_root=Path(prepared_root).resolve();protocol=read(prepared_root/"protocol.json")
    for path,expected in ((Path(calibration_path),protocol["score"]["calibration_file_sha256"]),
        (ROOT/protocol["score"]["metrics_source"],protocol["score"]["metrics_source_sha256"]),
        (ROOT/"tdyssey/native_rgb_evaluation.py",protocol["score"]["legacy_evaluation_source_sha256"])):
        if sha(path)!=expected:raise ValueError("Frozen calibration/endpoint metric source changed")
    calibration=read(calibration_path);audit=read(prepared_root/"preparation-audit.json")
    all_rows=[];all_steps=[];cohorts={};arrays={};sources={}
    for cohort in COHORTS:
        prepared=load_eval_cohort(manifest_path,prepared_root/"indices.json",cohort)
        if prepared["identity"]!=audit["cohort_identities"][cohort] or len(prepared["window_ids"])!=12:
            raise ValueError("Frozen12/12/12 evaluation identity differs")
        n=len(prepared["window_ids"]);copy=prepared["histories"][:,-1].copy()
        hold=dict(planned_windows=n,window_ids=prepared["window_ids"],h1=copy,h4=copy,status_h1=["ok"]*n,status_h4=["ok"]*n)
        methods={"copy_last":(hold,[[copy[i]]*4 for i in range(n)])};sources[cohort]={}
        for stage,method in (("baseline","original_GPC"),("final","trained")):
            result,steps,source=load_predictions(evaluation_root,stage,cohort,prepared,inventory,calibration)
            methods[method]=(result,steps);sources[cohort][method]=source
        rows=[];arrays[cohort]=dict(prepared=prepared,methods=methods)
        for method,(result,steps) in methods.items():
            stage={"copy_last":"copy_last","original_GPC":"baseline","trained":"final"}[method]
            scored,step_rows,metric_source=load_metrics(evaluation_root,stage,cohort,prepared,result,steps,inventory,calibration,method)
            sources[cohort].setdefault(method,{})["budgeted_metrics"]=metric_source
            rows.extend(scored);all_steps.extend(dict(row,cohort=cohort) for row in step_rows)
        all_rows.extend(rows);cohorts[cohort]={}
        for horizon in (1,4):
            endpoint=[r for r in rows if r["horizon"]==horizon]
            subsets={"all":endpoint,"H1_obvious_motion":[r for r in endpoint if r["H1_obvious_motion"]],
                "H1_small_or_nearstatic":[r for r in endpoint if not r["H1_obvious_motion"]],
                "H1_nearstatic_aux":[r for r in endpoint if r["H1_nearstatic_aux"]]}
            cohorts[cohort][f"H{horizon}"]={group:dict(methods={m:summary([r for r in values if r["method"]==m]) for m in METHODS},
                trained_vs_copy_last=paired(values,"trained","copy_last"),trained_vs_original_GPC=paired(values,"trained","original_GPC")) for group,values in subsets.items()}
    report=dict(kind="R2C1-complete-same-window-native-RGB-report.v1",model_work_sha256=sha(model_work),model_work=work,
        prediction_inventory_count=len(inventory),data_manifest_sha256=sha(manifest_path),indices_sha256=sha(prepared_root/"indices.json"),
        protocol_sha256=sha(prepared_root/"protocol.json"),preparation_audit_sha256=sha(prepared_root/"preparation-audit.json"),calibration_sha256=sha(calibration_path),
        frozen_windows=36,endpoint_rows=216,cohorts=cohorts,rows=all_rows,raw_step_rows=all_steps,prediction_sources=sources,
        midstep_failures={m:{str(k):sum(r["hard_failure"] is True for r in all_steps if r["method"]==m and r["step"]==k) for k in range(1,5)} for m in METHODS},
        limits=["All36 frozenwindows and failures retained;12percohort/horizon, no finite-only primary means.",
            "Endpoint metrics use frozen math unchanged; path/midstep failures and all-step-valid wins remain separate and cannot be erased by a good H4endpoint.",
            "Visible occlusion/viewport/unassessable shape stays unknown; assessable_no_detected_failure is not a rigid-shape guarantee.",
            "Natural anchors lie within motion-enriched seen-letter trajectories; valnatural/motion overlap within trajectories; not unseen-letter generalization.",
            "Display-only PNG saturation when finiteout-of-range is labelled; rawfloat scoring/feedback remains unchanged, no repair/recolor of model output."],
        endpoint_metrics_recomputed_cold=False,own_model_calls=0,own_environment_constructions=0,own_physics_controls=0,torch_imported="torch" in sys.modules)
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
    score_by_key={(r["cohort"],r["horizon"],r["method"],r["window_id"]):r for r in report["rows"]}
    for page_kind in ("main","original_T_task_baseline"):
        for cohort in COHORTS:
            prepared=arrays[cohort]["prepared"];methods=arrays[cohort]["methods"]
            for horizon in (1,4):
                truth=prepared[f"truth_h{horizon}"]
                method="trained" if page_kind=="main" else "original_GPC"
                for start in range(0,12,6):
                    fig,axes=plt.subplots(6,3,figsize=(9.3,13.5));fig.subplots_adjust(top=.92,bottom=.025,left=.035,right=.98,wspace=.12,hspace=.70)
                    headline="真值 / 复制最后输入 / 训练后原RGB" if page_kind=="main" else "原始T任务GPC独立基线（未微调，错误颜色参照）"
                    fig.suptitle(f"{headline}\n{cohort} · H{horizon} ({horizon*.1:.1f}秒) · 全部固定窗/失败保留",fontsize=12)
                    ids=[]
                    for local,i in enumerate(range(start,start+6)):
                        row=prepared["metadata"][i];identifier=prepared["window_ids"][i];ids.append(identifier)
                        pred=methods[method][0][f"h{horizon}"][i];copy=methods["copy_last"][0][f"h{horizon}"][i]
                        quality=score_by_key[(cohort,horizon,method,identifier)];hold=score_by_key[(cohort,horizon,"copy_last",identifier)]
                        title=(f"{row['letter']} {row['trajectory_id']} a={row['anchor']} | 灰1−IoU copy={fmt(hold['raw_object_error'])} 当前={fmt(quality['raw_object_error'])}\n"
                            f"端点硬失败={quality['endpoint_hard_generation_failure']} 路径失败={quality['any_path_step_hard_failure']} 形状={quality['shape_outcome']}")
                        axes[local,0].text(0,1.3,title,transform=axes[local,0].transAxes,fontsize=7,ha="left",va="bottom")
                        for j,(raw,label) in enumerate(((truth[i],"真实RGB"),(copy,"复制最后输入"),(pred,"训练后预测" if method=="trained" else "原始T任务GPC"))):
                            value=raw.astype(np.float64)/255 if getattr(raw,"dtype",None)==np.uint8 else raw
                            image,note=display_encoding(value);ax=axes[local,j];ax.axis("off");ax.set_title(label,fontsize=9)
                            if image is None:ax.text(.5,.5,note,ha="center",va="center",wrap=True,transform=ax.transAxes,fontsize=8,color="crimson")
                            else:ax.imshow(image,interpolation="nearest")
                            if note:ax.text(.0,-.08,note,transform=ax.transAxes,fontsize=6,color="crimson",ha="left",va="top",wrap=True)
                    name=f"{page_kind}-{cohort}-H{horizon}-{start//6+1:02d}.png";path=out/name
                    fig.savefig(path,dpi=120,facecolor="white");plt.close(fig)
                    pages.append(dict(file=name,sha256=sha(path),bytes=path.stat().st_size,page_kind=page_kind,cohort=cohort,horizon=horizon,window_ids=ids))
    return pages


def markdown(report):
    lines=["# R2C1 同窗原RGB评估","","灰色物体为主；蓝推杆指标独立。端点失败、途中失败、缺测及形状unknown不剔除。","",
        "|Cohort|H|N|copy灰error|原GPC灰error|训练后灰error|对copy胜/平/负|相对改善|训练后端点/路径失败|形状unknown|",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|"]
    for cohort in COHORTS:
        for horizon in (1,4):
            group=report["cohorts"][cohort][f"H{horizon}"]["all"];m=group["methods"];p=group["trained_vs_copy_last"];o=p["endpoint_outcomes"]
            means=[m[k]["gray_object"]["raw_object_error"]["mean"] for k in METHODS]
            improvement=p["relative_gray_error_reduction"]
            lines.append(f"|{cohort}|{horizon}|12|{fmt(means[0],6)}|{fmt(means[1],6)}|{fmt(means[2],6)}|{o.get('win',0)}/{o.get('tie',0)}/{o.get('loss',0)} (未评{o.get('not_evaluated',0)})|{'undefined' if improvement is None else format(improvement,'.2%')}|{m['trained']['endpoint_hard_failures']}/{m['trained']['path_hard_failures']}|{m['trained']['shape_outcomes'].get('unknown',0)}|")
    lines.extend(["","端点数学沿用冻结协议；H2/H3失败另列，all-step-valid胜数不能被H4恢复掩盖。有限越界原值评估；PNG仅编码截幅时明确标注，未修色/修形。",
        "","near-static/small漂移、灰ROI RGB/IoU/原轮廓及蓝推杆分别见JSON与CSV。TRAIN拟合并非泛化，val均6seen字母且轨迹刻意含运动，natural与motion仍可能同轨迹时间重叠。",
        "","所有36窗的H1/H4都在media/index.html；主图不混入原始T任务GPC，原始基线独立页面。",""])
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("model-work","evaluation-root","data-manifest","prepared-root","calibration","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args(argv)
    report,arrays=build_report(args.model_work,args.evaluation_root,args.data_manifest,args.prepared_root,args.calibration)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    pages=galleries(report,arrays,out/"media");report["gallery_pages"]=pages
    report["gallery_denominators"]={kind:sum(len(p["window_ids"]) for p in pages if p["page_kind"]==kind) for kind in ("main","original_T_task_baseline")}
    write(out/"r2c1-report.json",report);(out/"REPORT.md").write_text(markdown(report),encoding="utf-8")
    for filename,rows in (("endpoint-scores.csv",report["rows"]),("all-raw-steps.csv",report["raw_step_rows"])):
        fields=sorted({key for row in rows for key in row})
        with (out/filename).open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({key:canonical(value) if isinstance(value,(dict,list)) else value for key,value in row.items()})
    links="\n".join(f'<li><a href="{r["file"]}">{r["page_kind"]} · {r["cohort"]} · H{r["horizon"]} · {i+1}</a></li>' for i,r in enumerate(pages))
    (out/"media/index.html").write_text('<!doctype html><meta charset="utf-8"><title>R2C1全部预测</title><h1>全部36窗H1/H4</h1><p>主图：真值/复制最后输入/训练后。原始T任务GPC独立基线页。失败和缺测保留，有限越界的显示截幅不改变原float评估。</p><ul>'+links+'</ul>',encoding="utf-8")
    size=sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    if size>30*1024**2:raise RuntimeError("Report/media exceed30MiB; no windows may be silently dropped")
    print(json.dumps(dict(endpoint_rows=len(report["rows"]),windows=36,pages=len(pages),gallery_rows=report["gallery_denominators"],bytes=size,own_model_calls=0)))
    return 0


if __name__=="__main__":raise SystemExit(main())
