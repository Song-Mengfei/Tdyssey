"""Manual cold-only four-window CPU audit; not a collector or training launcher."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ROOT=STAGE.parent
REPO=ROOT.parents[1]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def cold_guards(plan):
    if (STAGE/"COLLECTION_RUNNING.lock").exists():raise RuntimeError("Collection remains active; do not inspect hot validation")
    required=(STAGE/"data/manifest.json",STAGE/"physics-audit.json",STAGE/"resource-ledger.json")
    if not all(path.is_file() for path in required):raise RuntimeError("Root must finish final manifest and physical audit first")
    for path,expected in plan["runtime_sources"].items():
        if digest(REPO/path)!=expected:raise RuntimeError("Frozen runtime source changed: "+path)
    if digest(ROOT/"stage2b/report/merged-train-manifest.json")!=plan["train_view_sha256"]:raise RuntimeError("Frozen TRAIN view changed")
    if digest(STAGE/"frozen-requests.json")!=plan["validation_requests_sha256"]:raise RuntimeError("Frozen validation requests changed")
    if digest(STAGE/"source-identity.json")!=plan["collector_source_identity_sha256"]:raise RuntimeError("Collector source identity changed")
    manifest,audit,ledger=map(read,required)
    if (manifest.get("failure") is not None or manifest.get("stage")!="2C" or manifest.get("split")!="seen_val"
            or len(manifest["entries"])!=162 or manifest.get("actual_controls")!=10368):raise RuntimeError("Validation fixed batch incomplete/failed")
    if (audit.get("manifest_sha256",audit.get("dataset_manifest_sha256"))!=digest(required[0]) or audit.get("passed") is not True
            or audit.get("physics_collection_gate_passed") is not True or audit.get("invalid_controls")!=0
            or audit.get("controls")!=10368 or audit.get("audited_substeps")!=103680
            or audit.get("valid_h1_candidates")!=9882 or audit.get("valid_h4_candidates")!=9396
            or audit.get("threshold_world_px")!=2 or audit.get("threshold_relaxed") is not False):raise RuntimeError("Current fixed physical gate not passed")
    if (ledger.get("failure") is not None or ledger.get("charged_controls")!=10368 or ledger.get("completed_controls")!=10368
            or len(ledger["trajectories"])!=162 or any(row.get("status")!="completed" or row.get("pending_control") is not None for row in ledger["trajectories"])):
        raise RuntimeError("Cold ledger does not confirm all162 completed slots")
    return manifest,audit


def execute():
    started=time.monotonic();plan=read(HERE/"planned-cpu-samples.json");cold_guards(plan)
    for name in ("cpu-samples.json","dataset-validation.json","dataset-validation.md"):
        if (HERE/name).exists():raise FileExistsError("Preserve existing CPU evidence: "+name)
    if (STAGE/"joint-data").exists():raise FileExistsError("Preserve existing joint reference directory")
    sys.path.insert(0,str(REPO))
    from tdyssey.native_rgb_reference import build_joint_reference,validate_cpu_samples
    from tdyssey.native_rgb_data import NativeRGBDataset,verify_dataset
    from tdyssey.native_rgb_training import data_identity
    built=build_joint_reference(ROOT);joint=Path(built["root"])
    identity=data_identity(joint)  # pure data/file-hash function; never construct a model or optimizer
    manifest,audit=verify_dataset(joint)
    declared=manifest["splits"];split_sets={name:set(values) for name,values in declared.items()}
    if {name:len(values) for name,values in split_sets.items()}!=plan["expected_split_trajectories"]:raise RuntimeError("Joint split counts differ")
    if split_sets["train"]&split_sets["seen_val"] or split_sets["unseen_test"]:raise RuntimeError("Split contamination")
    expected_split_hash=hashlib.sha256(json.dumps(declared,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if identity["split_sha256"]!=expected_split_hash or identity["manifest_sha256"]!=digest(joint/"manifest.json"):raise RuntimeError("Training pure-data identity does not bind joint view")
    catalogs={}
    for split in ("train","seen_val"):
        dataset=NativeRGBDataset(joint,split,cache_bytes=1048576)
        catalogs[split]=dict(h1=len(dataset.windows),h4=len(dataset.h4_windows),trajectories=len(dataset.entries),
            cache_cap_bytes=dataset.cache_bytes,initial_cache_bytes=dataset._image_bytes)
        if len(dataset.windows)!=plan["expected_h1_catalogs"][split] or len(dataset.h4_windows)!=plan["expected_h4_catalogs"][split]:raise RuntimeError("H1/H4 recorded catalogs differ")
        if dataset._image_bytes!=0:raise RuntimeError("Dataset eagerly built an image cache")
    samples=validate_cpu_samples(joint,HERE/"cpu-samples.json",plan["selections"])
    for split,batch in samples["batches"].items():
        if batch["image"]!=plan["expected_cpu_batch_images"] or batch["action"]!=plan["expected_cpu_batch_actions"]:raise RuntimeError("Original GPC batch layout differs")
    if len(samples["samples"])!=4 or any(sample["cache_bytes"]>1048576 for sample in samples["samples"]):raise RuntimeError("CPU sample/cache budget exceeded")
    result=dict(kind="stage2c-joint-dataset-cpu-validation.v1",passed=True,joint_root=str(joint),
        planned_samples_sha256=digest(HERE/"planned-cpu-samples.json"),execution_script_sha256=digest(__file__),
        data_identity=identity,split_counts={name:len(values) for name,values in split_sets.items()},split_disjoint=True,
        manifest_sha256=digest(joint/"manifest.json"),audit_sha256=digest(joint/"physics-audit.json"),
        identity_audit_sha256=digest(joint/"identity-audit.json"),catalogs=catalogs,cpu_samples=samples,
        all_npz_metadata_snapshots_and_source_audits_verified=True,npz_files_copied=0,
        models_loaded=0,denoiser_calls=0,optimizer_updates=0,training_iterators_constructed=0,index_draws=0,physics_controls=0,
        torch_imported="torch" in sys.modules,wall_seconds=time.monotonic()-started)
    write(HERE/"dataset-validation.json",result)
    (HERE/"dataset-validation.md").write_text(
        "# Stage2C joint reference and four CPU samples\n\n"
        "Passed: source/split/hash-bound396 TRAIN +162 seen_val;0 unseen/probe entries. Original NPZ files were referenced, never copied.\n\n"
        f"H1/H4 catalogs: {catalogs}.\n\n"
        f"CPU batches: {samples['batches']}. Images/actions keep originalfloat32[0,1]/[-1,1], five CHW RGB frames and four real actions.\n\n"
        "Only four predeclared direct Dataset items were read/rendered, uint8LRU≤1MiB/dataset. Pure training data_identity accepts the canonical joint manifest/audit/split hash.\n\n"
        "No physics control, model/denoiser/weight, optimizer, training iterator or index draw. This is data compatibility evidence, not model performance or training authorization.\n",encoding="utf-8")
    print(json.dumps(dict(passed=True,joint_root=str(joint),catalogs=catalogs,batches=samples["batches"],wall_seconds=result["wall_seconds"])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-after-root-cold-confirmation",action="store_true",required=True)
    parser.parse_args()
    try:execute()
    except BaseException as error:
        evidence=dict(kind="stage2c-joint-dataset-cpu-error.v1",error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            execution_script_sha256=digest(__file__),frozen_source_edits=0,physics_controls=0,model_calls=0)
        if not (HERE/"execution-error.json").exists():write(HERE/"execution-error.json",evidence)
        raise


if __name__=="__main__":main()
