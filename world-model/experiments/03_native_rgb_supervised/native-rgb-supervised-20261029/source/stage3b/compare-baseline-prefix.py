"""CPU-only old3 versus cold3B baseline prefix identity/raw-array comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(stage):
    value=read(stage/"source-identity.json")
    errors=[]
    for row in value["files"]:
        path=(stage/row["snapshot"]).resolve()
        if not path.is_relative_to(stage) or not path.is_file() or sha(path)!=row["sha256"]:
            errors.append(row["snapshot"])
    return value,errors


def ranges(array):
    finite=np.isfinite(array)
    return dict(shape=list(array.shape),dtype=str(array.dtype),contiguous_bytes_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        finite_values=int(finite.sum()),total_values=int(array.size),min=float(array[finite].min()) if finite.any() else None,
        max=float(array[finite].max()) if finite.any() else None,below_zero=int((array<0).sum()),above_one=int((array>1).sum()))


def compare(old_stage,new_stage,output):
    old_stage,new_stage=Path(old_stage).resolve(),Path(new_stage).resolve()
    output=Path(output)
    if output.exists():raise FileExistsError("Preserve each comparison result")
    work=read(new_stage/"model-work.json")
    if work.get("budget",{}).get("closed") is not True:raise RuntimeError("Wait for root's cold-completion notification; no active model data reads")
    plan=read(new_stage/"baseline-compare-plan.json")
    copied=[]
    for row in plan["identical_input_files"]:
        name=row["relative_path"];a,b=old_stage/name,new_stage/name
        old_hash=sha(a) if a.is_file() else None;new_hash=sha(b) if b.is_file() else None
        copied.append(dict(file=name,old_sha256=old_hash,new_sha256=new_hash,expected_old_sha256=row["old_sha256"],
                           byte_identical=old_hash==new_hash==row["old_sha256"]))
    old_id,old_snapshot_errors=identity(old_stage);new_id,new_snapshot_errors=identity(new_stage)
    old_sources={row["source"]:row["sha256"] for row in old_id["files"]}
    new_sources={row["source"]:row["sha256"] for row in new_id["files"]}
    common=[dict(source=name,old_sha256=old_sources[name],new_sha256=new_sources[name],identical=old_sources[name]==new_sources[name])
            for name in sorted(old_sources.keys()&new_sources.keys())]
    old_asset,new_asset=old_id["original_asset"],new_id["original_asset"]
    asset_identical=old_asset["sha256"]==new_asset["sha256"]==plan["original_published_checkpoint_sha256"]
    old_batch=read(old_stage/"paired-batches.json")["cohorts"]["train18"][0]
    new_batch=read(new_stage/"paired-batches.json")["cohorts"]["train18"][0]
    positions=old_batch["positions"]
    if len(positions)!=4:raise ValueError("The reviewed repeated prefix is exactly the first batch of4")
    old_path=old_stage/"predictions/frozen-train18.npz";new_path=new_stage/"predictions/frozen-train18.npz"
    with np.load(old_path,allow_pickle=False) as f:old={key:f[key] for key in f.files}
    with np.load(new_path,allow_pickle=False) as f:new={key:f[key] for key in f.files}
    rows=[]
    for position in positions:
        a,b=old["prediction"][position],new["prediction"][position]
        left,right=ranges(a),ranges(b)
        same_shape=a.shape==b.shape
        all_finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        if same_shape:
            both=np.isfinite(a)&np.isfinite(b)
            max_finite=float(np.abs(a.astype(np.float64)-b.astype(np.float64))[both].max()) if both.any() else None
            numeric_equal=bool(np.array_equal(a,b,equal_nan=True))
        else:both=np.zeros(0,bool);max_finite=None;numeric_equal=False
        rows.append(dict(position=position,old_window_id=str(old["window_ids"][position]),new_window_id=str(new["window_ids"][position]),
            window_id_identical=str(old["window_ids"][position])==str(new["window_ids"][position]),old=left,new=right,
            old_status=str(old["status"][position]),new_status=str(new["status"][position]),
            status_identical=str(old["status"][position])==str(new["status"][position]),numeric_array_equal=numeric_equal,
            byte_identical=bool(a.dtype==b.dtype and same_shape and left["contiguous_bytes_sha256"]==right["contiguous_bytes_sha256"]),
            all_values_finite=all_finite,max_abs_difference_all_values=max_finite if all_finite and same_shape else None,
            finite_overlap_values=int(both.sum()),max_abs_difference_finite_overlap_auxiliary=max_finite))
    old_prefix=np.ascontiguousarray(old["prediction"][positions]);new_prefix=np.ascontiguousarray(new["prediction"][positions])
    input_identity_ok=all(row["byte_identical"] for row in copied) and not old_snapshot_errors and not new_snapshot_errors and all(row["identical"] for row in common) and asset_identical and old_batch==new_batch
    all_raw_identical=all(row["byte_identical"] and row["window_id_identical"] for row in rows)
    result=dict(kind="stage3-versus-stage3b-first-frozen-batch-comparison.v1",plan_sha256=sha(new_stage/"baseline-compare-plan.json"),
        script_sha256=sha(__file__),old_stage=str(old_stage),new_stage=str(new_stage),repeated_batch=old_batch,
        new_batch=new_batch,batch_seed_members_shape_identical=old_batch==new_batch,input_files=copied,
        common_frozen_source_files=common,old_snapshot_errors=old_snapshot_errors,new_snapshot_errors=new_snapshot_errors,
        original_published_asset=dict(old=old_asset,new=new_asset,metadata_identity_identical=asset_identical,weights_deserialized=False),
        input_source_seed_identity_consistent=input_identity_ok,rows=rows,
        old_raw_npz_sha256=sha(old_path),new_raw_npz_sha256=sha(new_path),
        old_first4_contiguous_sha256=hashlib.sha256(old_prefix.tobytes()).hexdigest(),
        new_first4_contiguous_sha256=hashlib.sha256(new_prefix.tobytes()).hexdigest(),
        all_four_raw_bytes_identical=all_raw_identical,statuses_identical=all(row["status_identical"] for row in rows),
        review_required=not(input_identity_ok and all_raw_identical and all(row["status_identical"] for row in rows)),
        qualification="Whole NPZ hashes differ legitimately because3B has a full baseline instead of14 missing old rows. First4 equality uses raw numerical arrays/dtype/shape, never file compression bytes. Finite-overlap difference is auxiliary if any value is nonfinite. Differences must be reviewed, not hidden by choosing a preferred baseline.",
        original_arrays_modified=False,models_loaded=False,model_calls=0,physics_controls=0,new_score_or_threshold=False)
    with output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({key:result[key] for key in ("input_source_seed_identity_consistent","all_four_raw_bytes_identical","statuses_identical","review_required")}))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-stage",required=True);parser.add_argument("--new-stage",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();compare(args.old_stage,args.new_stage,args.output)
