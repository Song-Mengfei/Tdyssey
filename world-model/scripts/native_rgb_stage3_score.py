"""CPU-only scoring of all frozen stage3 raw RGB windows, including failures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey import native_rgb_evaluation as legacy

DEFAULT_PROTOCOL=ROOT/"configs/native_rgb_stage3_score.json"
METHODS=("hold","frozen","fitted")
ARRAYS=("last","true","hold","frozen","fitted")
DIAGONAL=96*math.sqrt(2)
SCALE=512/96


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(legacy._json_value(value),stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def load_protocol(path=DEFAULT_PROTOCOL,calibration_path=None):
    protocol=read_json(path)
    if (protocol["version"]!="stage3-raw-rgb-score-v1" or protocol["cohorts"]!={"train":18,"development":36}
            or protocol["image_shape"]!=[3,96,96] or protocol["methods_scored"]!=list(METHODS)
            or protocol["train_dynamic_target"]["mean_object_error_relative_reduction_vs_hold_gte"]!=.20
            or protocol["train_dynamic_target"]["strict_window_wins_vs_hold_gte"]!=12):
        raise ValueError("Scoring protocol differs from the predeclared stage3 diagnostic")
    path=Path(calibration_path or ROOT/protocol["calibration"]["path"])
    calibration=legacy.validate_calibration(path)
    if (sha(path)!=protocol["calibration"]["file_sha256"] or calibration["sha256"]!=protocol["calibration"]["internal_sha256"]
            or sha(legacy.__file__)!=protocol["calibration"]["legacy_evaluation_source_sha256"]):
        raise ValueError("Frozen palette/renderer/legacy scoring identity changed")
    return protocol,calibration,path


def _rgb(image):
    import numpy as np
    image=np.asarray(image)
    if image.shape!=(3,96,96):raise ValueError("wrong_shape")
    if image.dtype==np.uint8:values=image.astype(np.float64)/255
    elif np.issubdtype(image.dtype,np.floating):values=image.astype(np.float64,copy=False)
    else:raise ValueError("unsupported_dtype")
    if not np.isfinite(values).all():raise ValueError("nonfinite")
    if values.min()<0 or values.max()>1:raise ValueError("out_of_range")
    return values


def _border(mask):return bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())


def _centroid_distance(first,second):
    import numpy as np
    a,b=legacy._centroid(first),legacy._centroid(second)
    return DIAGONAL if a is None or b is None else float(np.linalg.norm(a-b))


def _roi_mae(prediction,truth,roi):
    import numpy as np
    return 1. if not roi.any() else float(np.abs(prediction-truth)[:,roi].sum()/(3*roi.sum()))


def _failure(reason,status="ok",truth_issue=None):
    return dict(hard_generation_failure=True,failure_reasons=[reason],prediction_status=str(status),truth_issue=truth_issue,
        raw_mask_iou=0.,raw_object_error=1.,raw_contour_error_image_px=DIAGONAL,raw_contour_error_world_px=DIAGONAL*SCALE,
        object_foreground_rgb_mae=1.,combined_foreground_rgb_mae=1.,pusher_foreground_rgb_mae=1.,
        pusher_mask_iou=0.,pusher_error=1.,pusher_position_error_image_px=DIAGONAL,pusher_position_error_world_px=DIAGONAL*SCALE,
        object_position_error_image_px=DIAGONAL,object_position_error_world_px=DIAGONAL*SCALE,
        object_drift_from_last_error=1.,object_drift_from_last_image_px=DIAGONAL,object_drift_from_last_world_px=DIAGONAL*SCALE,
        rgb_full_mae=1.,rgb_full_mse=1.,predicted_components=None,predicted_holes=None,extra_fragments=None,
        missing_components=None,tiny_fragment_pixels=None,area_fraction_error=None,aligned_mask_iou=None,
        legacy_shape_deformation_failure=None,legacy_shape_assessable=False,legacy_shape_unknown_reason=reason,
        legacy_score_error=None,object_outside_viewport_assessable=False,denominator_retained=True)


def score_image(prediction,truth,last,calibration,*,letter,status="ok",truth_full_mask=None,last_full_mask=None):
    """Primary measurements never depend on legacy score success or alignment."""
    import numpy as np
    from scipy import ndimage
    try:target_rgb,last_rgb=_rgb(truth),_rgb(last)
    except Exception as error:return _failure("invalid_truth_or_last",status,f"{type(error).__name__}: {error}")
    if str(status).strip().lower()!="ok":return _failure("inference_status_failure",status)
    try:rgb=_rgb(prediction)
    except Exception as error:return _failure(str(error),status)
    predicted,blue=legacy.palette_masks(rgb,calibration)
    target,target_blue=legacy.palette_masks(target_rgb,calibration)
    prior,prior_blue=legacy.palette_masks(last_rgb,calibration)
    structure=np.ones((3,3),dtype=bool)
    object_roi=ndimage.binary_dilation(prior|target,structure=structure,iterations=2)&~ndimage.binary_dilation(prior_blue|target_blue,structure=structure,iterations=1)
    pusher_roi=ndimage.binary_dilation(prior_blue|target_blue,structure=structure,iterations=2)
    foreground_roi=ndimage.binary_dilation(prior|target|prior_blue|target_blue,structure=structure,iterations=2)
    true_object_ok=bool(target.any());true_pusher_ok=bool(target_blue.any());prior_ok=bool(prior.any())
    failures=[]
    if not predicted.any():failures.append("empty_object")
    if not blue.any():failures.append("empty_pusher")
    object_iou=legacy._iou(predicted,target) if true_object_ok else 0.
    pusher_iou=legacy._iou(blue,target_blue) if true_pusher_ok else 0.
    contour=legacy._contour_distance(predicted,target)
    predicted_topology=legacy._topology(predicted,calibration["minimum_component_pixels"])
    true_topology=legacy._topology(target,calibration["minimum_component_pixels"])
    row=dict(hard_generation_failure=bool(failures),failure_reasons=failures,prediction_status=str(status),
        truth_issue=None if true_object_ok and true_pusher_ok and prior_ok else "empty_truth_object_pusher_or_last_object",
        raw_mask_iou=float(object_iou or 0),raw_object_error=1-float(object_iou or 0),
        raw_contour_error_image_px=DIAGONAL if contour is None else contour/SCALE,
        raw_contour_error_world_px=DIAGONAL*SCALE if contour is None else contour,
        raw_contour_unavailable=contour is None,object_foreground_rgb_mae=_roi_mae(rgb,target_rgb,object_roi),
        combined_foreground_rgb_mae=_roi_mae(rgb,target_rgb,foreground_roi),pusher_foreground_rgb_mae=_roi_mae(rgb,target_rgb,pusher_roi),
        object_roi_pixels=int(object_roi.sum()),pusher_roi_pixels=int(pusher_roi.sum()),combined_roi_pixels=int(foreground_roi.sum()),
        pusher_mask_iou=float(pusher_iou or 0),pusher_error=1-float(pusher_iou or 0),
        pusher_position_error_image_px=_centroid_distance(blue,target_blue),
        object_position_error_image_px=_centroid_distance(predicted,target),
        object_drift_from_last_error=1-float(legacy._iou(predicted,prior) or 0),
        object_drift_from_last_image_px=_centroid_distance(predicted,prior),
        rgb_full_mae=float(np.abs(rgb-target_rgb).mean()),rgb_full_mse=float(((rgb-target_rgb)**2).mean()),
        predicted_components=predicted_topology["components"],truth_components=true_topology["components"],
        predicted_holes=predicted_topology["holes"],truth_holes=true_topology["holes"],
        extra_fragments=max(0,predicted_topology["components"]-true_topology["components"]),
        missing_components=max(0,true_topology["components"]-predicted_topology["components"]),
        holes_difference=abs(predicted_topology["holes"]-true_topology["holes"]),tiny_fragment_pixels=predicted_topology["tiny_fragment_pixels"],
        area_fraction_error=float(abs(int(predicted.sum())-int(target.sum()))/max(1,int(target.sum()))),
        prediction_object_border_touch=_border(predicted),truth_object_border_touch=_border(target),
        prediction_pusher_border_touch=_border(blue),truth_pusher_border_touch=_border(target_blue),
        object_outside_viewport_assessable=False,
        occlusion_ambiguous=bool((ndimage.binary_dilation(predicted)&blue).any() or (ndimage.binary_dilation(target)&target_blue).any()),
        legacy_shape_deformation_failure=None,legacy_shape_assessable=False,legacy_shape_unknown_reason=None,legacy_score_error=None,
        aligned_mask_iou=None,denominator_retained=True)
    for name in ("pusher_position_error","object_position_error","object_drift_from_last"):
        row[name+"_world_px"]=row[name+"_image_px"]*SCALE
    for name,value in (("true",truth_full_mask),("last",last_full_mask)):
        if value is None:continue
        full=np.asarray(value)
        if full.shape!=(96,96) or not np.isfinite(full).all() or full.min()<0 or full.max()>1:
            row[name+"_full_mask_issue"]="invalid_supplied_full_mask";continue
        full=full>=calibration["palette"]["minimum_alpha"]
        visible=target if name=="true" else prior
        row[name+"_visible_fraction"]=float((visible&full).sum()/max(1,full.sum()))
        row[name+"_full_mask_border_touch"]=_border(full)
        row[name+"_full_mask_components"]=legacy._topology(full,calibration["minimum_component_pixels"])
        if name=="true":row["predicted_visible_vs_true_full_mask_iou"]=legacy._iou(predicted,full)
    try:
        previous=legacy.score_rgb(rgb,target_rgb,calibration,letter=letter,truth_full_mask=truth_full_mask)
        row["legacy"]=previous
        row["aligned_mask_iou"]=previous["aligned_mask_iou"]
        row["legacy_shape_deformation_failure"]=previous["shape_deformation_failure"]
        row["legacy_shape_assessable"]=previous["shape_measurement_available"]
        if not previous["shape_measurement_available"]:
            row["legacy_shape_unknown_reason"]="contact_occlusion_or_viewport_clipping"
    except Exception as error:
        row["legacy_score_error"]=f"{type(error).__name__}: {error}"
        row["legacy_shape_unknown_reason"]="legacy_score_exception"
        if predicted.any() and target.any():
            try:row.update(legacy.align_masks_for_score(predicted,target,angle_step=calibration["angle_grid_degrees"]))
            except Exception as alignment_error:row["alignment_error"]=f"{type(alignment_error).__name__}: {alignment_error}"
    return row


def _windows(value):
    value=read_json(value) if isinstance(value,(str,Path)) else value
    if isinstance(value,dict):value=value["windows"]
    return list(value)


def _labels(window):
    h1=window.get("h1",window)
    motion=h1.get("motion_class",window.get("motion_class","unknown"))
    boundary=h1.get("actual_boundary_le32")
    gap=h1.get("actual_boundary_min_world_px")
    if gap is None:
        values=[h1.get(key) for key in ("min_wall_gap_before","min_wall_gap_after")]
        if all(value is not None for value in values):gap=min(values)
    if boundary is None and gap is not None:boundary=gap<=32
    band=h1.get("actual_boundary_gap_band")
    if band is None and gap is not None:band="negative" if gap<0 else "0-8" if gap<=8 else "8-32" if gap<=32 else "32-64" if gap<=64 else ">64"
    return dict(motion_class=motion,actual_boundary=boundary,actual_boundary_gap_band=band or "unknown")


def _summary(rows):
    metrics=("raw_object_error","raw_mask_iou","raw_contour_error_image_px","raw_contour_error_world_px", "object_foreground_rgb_mae",
             "pusher_error","pusher_position_error_image_px","pusher_foreground_rgb_mae","combined_foreground_rgb_mae",
             "object_drift_from_last_error","object_drift_from_last_image_px")
    n=len(rows)
    means={name:sum(row[name] for row in rows)/n if n else None for name in metrics}
    shape=[row for row in rows if row["legacy_shape_assessable"]]
    aligned=[row["aligned_mask_iou"] for row in rows if row.get("aligned_mask_iou") is not None]
    return dict(denominator=n,means_all_rows=means,hard_generation_failures=sum(row["hard_generation_failure"] for row in rows),
        failure_reasons=dict(Counter(reason for row in rows for reason in row["failure_reasons"])),
        invalid_truth_rows=sum(row.get("truth_issue") is not None for row in rows),
        legacy_deformation=dict(assessable=len(shape),failed=sum(row["legacy_shape_deformation_failure"] is True for row in shape),
            unknown=n-len(shape),total=n),
        observed_extra_fragment_rows=sum((row.get("extra_fragments") or 0)>0 for row in rows),
        aligned_shape_auxiliary=dict(mean_visible_aligned_iou=sum(aligned)/len(aligned) if aligned else None,available=len(aligned),total=n,
            qualification="Auxiliary visible-mask alignment; unavailable rows are not passed and this finite-subset mean is not the fit criterion"))


def comparison(rows,method="fitted",baseline="hold"):
    n=len(rows);left=[row["methods"][method] for row in rows];right=[row["methods"][baseline] for row in rows]
    a=sum(row["raw_object_error"] for row in left)/n if n else None
    b=sum(row["raw_object_error"] for row in right)/n if n else None
    return dict(denominator=n,method=method,baseline=baseline,mean_object_error=a,baseline_mean_object_error=b,
        relative_reduction=None if not b else (b-a)/b,
        strict_wins=sum(x["raw_object_error"]<y["raw_object_error"]-1e-12 for x,y in zip(left,right)),
        ties=sum(abs(x["raw_object_error"]-y["raw_object_error"])<=1e-12 for x,y in zip(left,right)),
        method_hard_failures=sum(row["hard_generation_failure"] for row in left),baseline_hard_failures=sum(row["hard_generation_failure"] for row in right))


def train_target(rows,protocol_valid=True):
    values=comparison(rows)
    truth_valid=all(row["methods"]["fitted"].get("truth_issue") is None for row in rows)
    values.update(required_relative_reduction=.20,required_strict_wins=12,required_denominator=18,
        protocol_valid=bool(protocol_valid),truth_valid=truth_valid,hold_valid=values["baseline_hard_failures"]==0,
        passed=bool(len(rows)==18 and protocol_valid and truth_valid and values["relative_reduction"] is not None
                    and values["relative_reduction"]>=.20 and values["strict_wins"]>=12
                    and values["method_hard_failures"]==0 and values["baseline_hard_failures"]==0),
        qualification="18-window microfit diagnostic only. Shape failures/unknown and development evidence remain separate.")
    return values


def score_cohort(npz_path,windows,cohort,calibration):
    import numpy as np
    expected=18 if cohort=="train" else 36
    windows=_windows(windows)
    if len(windows)!=expected:raise ValueError("Frozen cohort metadata must retain its18/36 original windows")
    try:
        with np.load(npz_path,allow_pickle=False) as source:arrays={name:source[name] for name in source.files}
        io_error=None
    except Exception as error:arrays={};io_error=f"{type(error).__name__}: {error}"
    issues=[];rows=[]
    for key in ("last","true","hold"):
        if key not in arrays or arrays[key].shape!=(expected,3,96,96):issues.append(dict(issue="missing_or_wrong_shape_reference_array",key=key))
    for index,window in enumerate(windows):
        letter=window.get("letter",window.get("shape"))
        if letter not in legacy.TRAIN_LETTERS:raise ValueError("No held-out letters allowed in stage3")
        def image(key):
            array=arrays.get(key)
            return array[index] if array is not None and array.ndim and index<len(array) else None
        if image("hold") is not None and image("last") is not None and not np.array_equal(image("hold"),image("last"),equal_nan=True):
            issues.append(dict(index=index,issue="hold_not_exact_last"))
        measurements={}
        for method in METHODS:
            status_array=arrays.get(method+"_status")
            status=str(status_array[index]) if status_array is not None and status_array.ndim and index<len(status_array) else "ok"
            if image(method) is None:status="missing_prediction_array_or_row"
            measurements[method]=score_image(image(method),image("true"),image("last"),calibration,letter=letter,status=status,
                truth_full_mask=image("true_full_mask"),last_full_mask=image("last_full_mask"))
        rows.append(dict(index=index,window_id=window.get("window_id",f"{window['trajectory_id']}:{window['anchor']}"),
                         letter=letter,metadata=window,labels=_labels(window),methods=measurements))
    metadata_motion=dict(Counter(row["labels"]["motion_class"] for row in rows))
    if cohort=="development" and metadata_motion!={"moving":31,"near_static":4,"intermediate":1}:issues.append(dict(issue="development_H1_strata_differ_from_frozen31_4_1",actual=metadata_motion))
    if cohort=="train" and Counter(row["letter"] for row in rows)!=Counter(legacy.TRAIN_LETTERS):issues.append(dict(issue="train_not_one_per_fixed18letters"))
    summaries={method:_summary([row["methods"][method] for row in rows]) for method in METHODS}
    groups={}
    for name,key in (("H1_motion",lambda row:row["labels"]["motion_class"]),("letter",lambda row:row["letter"]),
                     ("actual_boundary",lambda row:str(row["labels"]["actual_boundary"])),("actual_boundary_gap_band",lambda row:row["labels"]["actual_boundary_gap_band"])):
        groups[name]={}
        for label in sorted({key(row) for row in rows}):
            selected=[row for row in rows if key(row)==label]
            groups[name][label]=dict(methods={method:_summary([row["methods"][method] for row in selected]) for method in METHODS},
                fitted_vs_hold=comparison(selected),fitted_vs_frozen=comparison(selected,"fitted","frozen"),
                static_interpretation="Drift and shape only; no beat-zero-hold requirement" if name=="H1_motion" and label=="near_static" else None)
    exists=Path(npz_path).is_file()
    result=dict(cohort=cohort,denominator=expected,raw_npz_path=str(Path(npz_path).resolve()),raw_npz_sha256=sha(npz_path) if exists else None,
        raw_npz_bytes=Path(npz_path).stat().st_size if exists else None,raw_io_error=io_error,rows=rows,methods=summaries,groups=groups,
        protocol_issues=issues,fitted_vs_hold=comparison(rows),fitted_vs_frozen=comparison(rows,"fitted","frozen"))
    if cohort=="train":result["dynamic_fit_target"]=train_target(rows,not issues and io_error is None)
    return result,arrays


def _gallery(output,cohort,rows,arrays):
    import numpy as np
    from PIL import Image,ImageDraw
    output=Path(output);output.mkdir(parents=True,exist_ok=False);pages=[]
    columns=ARRAYS
    for start in range(0,len(rows),6):
        current=rows[start:start+6];canvas=Image.new("RGB",(960,len(current)*232),"white");draw=ImageDraw.Draw(canvas)
        order=[]
        for local,row in enumerate(current):
            y=local*232;index=row["index"];order.append(row["window_id"])
            draw.text((4,y),f"{cohort} #{index} {row['letter']} {row['window_id']} {row['labels']['motion_class']} boundary={row['labels']['actual_boundary']}",fill="black")
            for column,name in enumerate(columns):
                draw.text((column*192+4,y+15),name,fill="black")
                try:
                    value=_rgb(arrays[name][index]);tile=Image.fromarray(np.rint(value.transpose(1,2,0)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST)
                except Exception:
                    tile=Image.new("RGB",(192,192),"#fff0f0");ImageDraw.Draw(tile).text((8,80),"INVALID RAW / MISSING",fill="black")
                canvas.paste(tile,(column*192,y+32))
                if name in METHODS:
                    metric=row["methods"][name]
                    draw.text((column*192+4,y+218),f"IoU={metric['raw_mask_iou']:.3f} fail={metric['hard_generation_failure']}",fill="black")
        filename=f"{cohort}-{start//6+1:02d}.png";canvas.save(output/filename)
        pages.append(dict(file=filename,window_ids=order,original_order_indices=[row["index"] for row in current]))
    write_json(output/"pages.json",dict(columns=list(columns),pages=pages,all_windows_shown=True,raw_arrays_unchanged=True))
    return pages


def build_report(train_npz,dev_npz,train_windows,dev_windows,output,*,calibration_path=None,protocol_path=DEFAULT_PROTOCOL):
    output=Path(output)
    if output.exists():raise FileExistsError("Do not overwrite stage3 model evidence")
    protocol,calibration,calibration_path=load_protocol(protocol_path,calibration_path)
    train,train_arrays=score_cohort(train_npz,train_windows,"train",calibration)
    dev,dev_arrays=score_cohort(dev_npz,dev_windows,"development",calibration)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(kind="stage3-raw-rgb-score-report.v1",protocol_sha256=sha(protocol_path),source_sha256=sha(__file__),
        calibration_internal_sha256=calibration["sha256"],calibration_file_sha256=sha(calibration_path),
        train=train,development=dev,primary="Unaligned gray object1-IoU, all18/36 retained",
        raw_array_files_retained=Path(train_npz).is_file() and Path(dev_npz).is_file(),
        static_dev_does_not_require_winning_over_hold=True,registration_auxiliary_only=True,model_calls=0,optimizer_updates=0,physics_controls=0)
    write_json(output/"score-report.json",report)
    from scripts.native_rgb_stage2a_coverage import _write_csv
    _write_csv(output/"per-window.csv",[dict(cohort=cohort,index=row["index"],window_id=row["window_id"],letter=row["letter"],
        **row["labels"],method=method,**{key:value for key,value in result.items() if key!="legacy"})
        for cohort,value in (("train",train),("development",dev)) for row in value["rows"] for method,result in row["methods"].items()])
    _gallery(output/"gallery-train", "train",train["rows"],train_arrays)
    _gallery(output/"gallery-development", "development",dev["rows"],dev_arrays)
    lines=["# Stage3 raw native RGB evidence","",f"TRAIN18 dynamic fitting diagnostic passed: {train['dynamic_fit_target']['passed']}.",
        "This is a bounded18-sample microfit diagnostic, not generalization or full acceptance. All18/36 failure rows remain in primary means.","",
        "| Cohort | Method | N | Mean gray1-IoU | Gray foreground RGB MAE | Raw contour(imagepx) | Hard failures | Legacy shape failures/assessable;unknown |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    for name,value in (("train",train),("development",dev)):
        for method,summary in value["methods"].items():
            m=summary["means_all_rows"];s=summary["legacy_deformation"]
            lines.append(f"| {name} | {method} | {summary['denominator']} | {m['raw_object_error']} | {m['object_foreground_rgb_mae']} | {m['raw_contour_error_image_px']} | {summary['hard_generation_failures']} | {s['failed']}/{s['assessable']}; unknown{s['unknown']} |")
    lines += ["",f"TRAIN target details: {train['dynamic_fit_target']}",
        "Object/pusher metrics are separate. Gray RGB ROI is fixed from last/true masks, excluding fixed blue-pusher regions; no prediction-dependent ROI. Actual RGB arrays remain unchanged.",
        "Development overall,31moving/4near-static/1intermediate,letters and actual boundary groups are in JSON. Moving uses pairedhold/frozen comparisons; static uses drift/shape, without requiring a win over zero-error hold.",
        "All windows are shown in frozen supplied order: TRAIN3 pages/development6 pages,5 columns last/true/hold/frozen/fitted. INVALID placeholders indicate unrenderable raw arrays, never repaired predictions.",
        "Legacy alignment is auxiliary visible-mask registration:10degree global,1degree local andhalf-imagepixel translation grid. This grid does not establish pose accuracy. Occluded/clipped shape measurements remain unknown, not passed.",
        "Original NPZ source paths/hashes, per-window failures, visible components/holes/area/fragments, legacydeformation assessability and optionalunoccluded truth-mask diagnostics are retained in JSON/CSV."]
    with (output/"SCORE_REPORT.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    write_json(output/"files.json",[dict(file=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.rglob("*")) if p.is_file()])
    return report


def calibration_check(output,*,physics_root,protocol_path=DEFAULT_PROTOCOL,calibration_path=None):
    """18 seen canonical truth renders using existing thresholds; never fit them."""
    import numpy as np
    from tdyssey.native_rgb_env import make_env,CONTROLLER,PROTOCOL_SHA256
    output=Path(output)
    if output.exists():raise FileExistsError("Preserve the pre-model palette check")
    protocol,calibration,calibration_path=load_protocol(protocol_path,calibration_path)
    images=[];rows=[]
    for letter in legacy.TRAIN_LETTERS:
        env=make_env(physics_root,letter,CONTROLLER);env.reset_scene()
        state=np.zeros(15);state[:5]=(40,40,256,256,0);env.set_render_state(state)
        image=env.render_clean();gray,blue=legacy.palette_masks(image,calibration)
        if env.control_step!=0 or not gray.any() or not blue.any():raise RuntimeError("Truthpalette/protocol check failed")
        reference=calibration["per_letter"][letter]
        if not reference["area_min"]<=int(gray.sum())<=reference["area_max"]:raise RuntimeError("Canonical truth area differs from frozen renderer calibration")
        rows.append(dict(letter=letter,object_pixels=int(gray.sum()),pusher_pixels=int(blue.sum()),topology=legacy._topology(gray,2),physics_controls=env.control_step))
        images.append(image)
    output.mkdir(parents=True,exist_ok=False)
    with (output/"canonical-seen18-truth.npz").open("xb") as stream:np.savez_compressed(stream,rgb=np.asarray(images),letters=np.asarray(legacy.TRAIN_LETTERS))
    result=dict(kind="stage3-pre-model-seen18-palette-check.v1",passed=True,letters=list(legacy.TRAIN_LETTERS),rows=rows,
        protocol_sha256=sha(protocol_path),source_sha256=sha(__file__),renderer_source_sha256=legacy._renderer_source_identity(),
        calibration_file_sha256=sha(calibration_path),calibration_internal_sha256=calibration["sha256"],physics_protocol_sha256=PROTOCOL_SHA256,
        thresholds_modified=False,model_outputs_used=False,physics_controls=0,model_calls=0)
    write_json(output/"calibration-check.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("calibration-check");check.add_argument("--physics-root",required=True);check.add_argument("--output",required=True)
    score=sub.add_parser("score")
    for name in ("train-npz","dev-npz","train-windows","dev-windows","output"):score.add_argument("--"+name,required=True)
    for command in (check,score):command.add_argument("--protocol",default=str(DEFAULT_PROTOCOL));command.add_argument("--calibration")
    args=parser.parse_args(argv)
    if args.command=="calibration-check":result=calibration_check(args.output,physics_root=args.physics_root,protocol_path=args.protocol,calibration_path=args.calibration);print(json.dumps(dict(passed=result["passed"],letters=len(result["letters"]),physics_controls=0)))
    else:
        result=build_report(args.train_npz,args.dev_npz,args.train_windows,args.dev_windows,args.output,calibration_path=args.calibration,protocol_path=args.protocol)
        print(json.dumps(result["train"]["dynamic_fit_target"]))
    return 0


if __name__=="__main__":raise SystemExit(main())
