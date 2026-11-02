"""3C CPU reference preparation and old129/final comparison; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.native_rgb_stage3_score import _rgb,_summary,comparison


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path,value):
    from tdyssey.native_rgb_evaluation import _json_value
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(_json_value(value),stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def prepare(stage):
    stage=Path(stage).resolve();parent=stage.parent/"stage3b"
    stage.mkdir(parents=True,exist_ok=True)
    copies=stage/"cpu-reference-copies";copies.mkdir(exist_ok=False)
    json_names=("frozen-windows.json","train18-windows.json","dev36-windows.json","paired-batches.json")
    refs=[]
    for name in json_names:
        original=parent/name;copied=copies/name
        with copied.open("xb") as stream:stream.write(original.read_bytes())
        if sha(copied)!=sha(original):raise RuntimeError("Private small JSON copy differs")
        refs.append(dict(role=name,source=str(original),sha256=sha(original),bytes=original.stat().st_size,
                         private_copy=copied.relative_to(stage).as_posix(),byte_identical=True))
    for name in ("truth/train18.npz","truth/dev36.npz","predictions/frozen-train18.npz","predictions/frozen-dev36.npz",
                 "predictions/fitted-train18.npz","predictions/fitted-dev36.npz","predictions/train18.npz","predictions/dev36.npz",
                 "score/score-report.json","model-work.json"):
        original=parent/name
        refs.append(dict(role=name,source=str(original),sha256=sha(original),bytes=original.stat().st_size,reference_only=True))
    work=read(parent/"model-work.json");checkpoint=work["training_result"]["last"]
    if checkpoint["updates"]!=129:raise ValueError("Continuation parent must be complete update129")
    result=dict(kind="stage3c-CPU-reused-input-and-score-plan.v1",source_script_sha256=sha(__file__),references=refs,
        parent_checkpoint=checkpoint,checkpoint_identity_from_closed_parent_metadata_only=True,weights_read_or_loaded=False,
        expected_parent_updates=129,expected_parent_order_cursor=2064,expected_max_final_updates=250,local_attempt_cap=121,
        score_source=dict(path=str(ROOT/"scripts/native_rgb_stage3_score.py"),sha256=sha(ROOT/"scripts/native_rgb_stage3_score.py")),
        protocol=dict(path=str(ROOT/"configs/native_rgb_stage3_score.json"),sha256=sha(ROOT/"configs/native_rgb_stage3_score.json")),
        calibration=dict(path=str(ROOT/"artifacts/native-rgb-20261028/frozen-renderer-calibration.json"),
                         sha256=sha(ROOT/"artifacts/native-rgb-20261028/frozen-renderer-calibration.json")),
        score_plan="After root cold completion: run original score CLI exactly once for final18TRAIN/36DEV. New composed NPZ reuses original3B frozen arrays/truth; no baseline or update129 inference.",
        comparison_plan="Read existing3B score for129; read final original-protocol score. Paired per-window rows and original metric summaries for hold/frozen/update129/final, same full18/36 denominators. Only final-vs-hold20percent+12wins is the registered fit target.",
        gallery=dict(cohort="train",windows=18,order="Original frozen18 order; all windows, never outcome-ranked",rows_per_page=6,pages=3,
                     columns=["true","hold(last)","frozen","update129","final"],display_scale=2,resize="nearest-neighbor display only",
                     invalid="Explicit placeholder for invalid/unavailable raw RGB; original arrays retained without clipping"),
        models_or_Torch_loaded=False,model_calls=0,physics_controls=0,old_files_modified=False)
    write(stage/"cpu-reference-plan.json",result)
    print(json.dumps(dict(plan_sha256=sha(stage/"cpu-reference-plan.json"),reference_files=len(refs),private_small_json_copies=len(json_names),models_loaded=False)))
    return result


def finish(stage):
    from PIL import Image,ImageDraw
    from scripts.native_rgb_stage2a_coverage import _write_csv
    stage=Path(stage).resolve();parent=stage.parent/"stage3b";plan=read(stage/"cpu-reference-plan.json")
    work=read(stage/"model-work.json")
    if work["budget"]["closed"] is not True:raise RuntimeError("Wait for root cold completion")
    if sha(ROOT/"scripts/native_rgb_stage3_score.py")!=plan["score_source"]["sha256"] or sha(ROOT/"configs/native_rgb_stage3_score.json")!=plan["protocol"]["sha256"]:
        raise RuntimeError("Old scoring source/protocol changed")
    for row in plan["references"]:
        if sha(row["source"])!=row["sha256"]:raise RuntimeError("Parent129/reference evidence changed")
        if row.get("private_copy"):
            if sha(stage/row["private_copy"])!=row["sha256"] or sha(stage/row["role"])!=row["sha256"]:
                raise RuntimeError("Small canonical/private input copy differs from the fixed129 source")
    before=read(parent/"score/score-report.json");after=read(stage/"score/score-report.json")
    if before["protocol_sha256"]!=after["protocol_sha256"] or before["source_sha256"]!=after["source_sha256"]:
        raise RuntimeError("Comparison must use the same original score")
    output=stage/"comparison129-final";output.mkdir(exist_ok=False)
    paired_rows=[];summary={};comparison_rows={};input_checks=[]
    final_update=work["training_result"]["progress"]["updates"]
    for cohort,npz_name in (("train","train18.npz"),("development","dev36.npz")):
        old=before[cohort]["rows"];new=after[cohort]["rows"]
        if [r["window_id"] for r in old]!=[r["window_id"] for r in new]:raise RuntimeError("Frozen comparison row order differs")
        with np.load(parent/"predictions"/npz_name,allow_pickle=False) as f:old_arrays={k:f[k] for k in f.files}
        with np.load(stage/"predictions"/npz_name,allow_pickle=False) as f:new_arrays={k:f[k] for k in f.files}
        for key in ("last","true","hold","last_full_mask","true_full_mask","frozen","frozen_status","window_ids"):
            equal=bool(old_arrays[key].dtype==new_arrays[key].dtype and old_arrays[key].shape==new_arrays[key].shape
                       and np.ascontiguousarray(old_arrays[key]).tobytes()==np.ascontiguousarray(new_arrays[key]).tobytes())
            input_checks.append(dict(cohort=cohort,key=key,raw_array_bytes_identical=equal))
            if not equal:raise RuntimeError("Original truth/baseline was changed instead of reused")
        merged=[]
        for a,b in zip(old,new):
            methods=dict(hold=b["methods"]["hold"],frozen=b["methods"]["frozen"],update129=a["methods"]["fitted"],final=b["methods"]["fitted"])
            row=dict(index=b["index"],window_id=b["window_id"],letter=b["letter"],labels=b["labels"],methods=methods)
            merged.append(row)
            for method,m in methods.items():
                paired_rows.append(dict(cohort=cohort,index=b["index"],window_id=b["window_id"],letter=b["letter"],**b["labels"],method=method,
                    raw_object_error=m["raw_object_error"],raw_mask_iou=m["raw_mask_iou"],raw_contour_error_image_px=m["raw_contour_error_image_px"],
                    object_foreground_rgb_mae=m["object_foreground_rgb_mae"],pusher_error=m["pusher_error"],
                    pusher_position_error_image_px=m["pusher_position_error_image_px"],object_position_error_image_px=m["object_position_error_image_px"],
                    predicted_holes=m.get("predicted_holes"),truth_holes=m.get("truth_holes"),extra_fragments=m.get("extra_fragments"),
                    tiny_fragment_pixels=m.get("tiny_fragment_pixels"),area_fraction_error=m.get("area_fraction_error"),
                    legacy_deformation_failure=m["legacy_shape_deformation_failure"],legacy_shape_assessable=m["legacy_shape_assessable"],
                    hard_generation_failure=m["hard_generation_failure"],prediction_status=m["prediction_status"]))
        comparison_rows[cohort]=merged
        summary[cohort]=dict(methods={name:_summary([row["methods"][name] for row in merged]) for name in ("hold","frozen","update129","final")},
            final_vs_hold=comparison(merged,"final","hold"),final_vs_update129=comparison(merged,"final","update129"),
            H1_motion={label:dict(final_vs_hold=comparison([r for r in merged if r["labels"]["motion_class"]==label],"final","hold"),
                                 final_vs_update129=comparison([r for r in merged if r["labels"]["motion_class"]==label],"final","update129"))
                       for label in sorted({r["labels"]["motion_class"] for r in merged})})
        if cohort=="train":
            pages=[];columns=("true","hold(last)","frozen","update129","final")
            for start in range(0,18,6):
                current=merged[start:start+6];canvas=Image.new("RGB",(960,len(current)*232),"white");draw=ImageDraw.Draw(canvas)
                for local,row in enumerate(current):
                    index=row["index"];y=local*232
                    draw.text((4,y),f"TRAIN #{index} {row['letter']} {row['window_id']} update129 vs update{final_update}",fill="black")
                    images=(old_arrays["true"][index],old_arrays["hold"][index],old_arrays["frozen"][index],old_arrays["fitted"][index],new_arrays["fitted"][index])
                    for column,(label,image) in enumerate(zip(columns,images)):
                        draw.text((column*192+4,y+15),label,fill="black")
                        try:
                            values=_rgb(image);tile=Image.fromarray(np.rint(values.transpose(1,2,0)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST)
                        except Exception:
                            tile=Image.new("RGB",(192,192),"#fff0f0");ImageDraw.Draw(tile).text((8,80),"INVALID RAW / MISSING",fill="black")
                        canvas.paste(tile,(column*192,y+32))
                        method="hold" if label=="hold(last)" else label
                        if method in row["methods"]:
                            m=row["methods"][method];draw.text((column*192+4,y+218),f"IoU={m['raw_mask_iou']:.3f} fail={m['hard_generation_failure']}",fill="black")
                name=f"train129-final-{start//6+1:02d}.png";canvas.save(output/name)
                pages.append(dict(file=name,window_ids=[r["window_id"] for r in current],frozen_order_indices=[r["index"] for r in current]))
            write(output/"gallery-pages.json",dict(columns=list(columns),pages=pages,all18_shown=True,raw_arrays_unchanged=True))
    result=dict(kind="stage3c-original-metric-129-to-final-comparison.v1",parent_updates=129,final_updates=final_update,
        old_score_sha256=sha(parent/"score/score-report.json"),new_score_sha256=sha(stage/"score/score-report.json"),
        score_source_sha256=after["source_sha256"],protocol_sha256=after["protocol_sha256"],reference_plan_sha256=sha(stage/"cpu-reference-plan.json"),
        input_array_identity_checks=input_checks,summary=summary,registered_final_fit_target=after["train"]["dynamic_fit_target"],
        qualification="129-to-final differences reuse the original per-window metrics and do not create a new acceptance criterion. Final versushold20percent+12/18 remains unchanged. All18/36 fixed windows and failures remain.",
        inference_calls_for_old_baseline_or129=0,models_or_Torch_loaded=False,physics_controls=0,new_scoring_or_thresholds=False)
    write(output/"comparison.json",result);_write_csv(output/"all54-method-comparison.csv",paired_rows)
    write(output/"files.json",[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()])
    print(json.dumps(dict(final_updates=final_update,registered_target=after["train"]["dynamic_fit_target"],all18_comparison_pages=3)))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","finish"));parser.add_argument("--stage",required=True)
    args=parser.parse_args();prepare(args.stage) if args.command=="prepare" else finish(args.stage)
