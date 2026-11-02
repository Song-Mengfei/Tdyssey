"""Single-use fixed-weight native denoising-step comparison. Root launches once."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c2_inference import (
    CHECKPOINT_SHA,SAMPLER,ExistingInferenceLedger,infer,read,sha,atomic_json,
    storage_for_config,verify_closed_prior)


def config_contract(config):
    if config.get("execution_authorized") is not True or config.get("is_proposal") is not False:
        raise PermissionError("Explicit current R2C2 authorization required")
    if config.get("time_policy")!="record_only_v2" or config.get("cap_time",False) is not None:
        raise ValueError("Model time is statistics only; no hard time limits")
    if config.get("new_settings")!=[8,16] or config.get("sampler")!=SAMPLER:
        raise ValueError("Only native3 reference and8/16 num_steps variable authorized")
    if config.get("seed")!=5478 or config.get("evaluation_batch")!=4:
        raise ValueError("Frozen seed5478/batch4 required")
    if config.get("checkpoint_sha256")!=CHECKPOINT_SHA or config.get("checkpoint_bytes")!=105300718:
        raise ValueError("Only trusted R1 complete300-update checkpoint")
    if Path(config["repo_root"]).resolve()!=ROOT or Path(config["output"]).resolve()!=ROOT/"artifacts/native-rgb-supervised-20261029/r2c2":
        raise ValueError("Unique authorized R2C2 output root required")
    if config.get("stage_artifact_cap_bytes")!=128*1024**2 or config.get("max_cuda_bytes")!=6*1024**3:
        raise ValueError("R2C2 persistent128MiB/CUDA6GiB guards required")
    return True


def preflight(config_path):
    path=Path(config_path).resolve();config=read(path);config_contract(config)
    if Path(config["config_file"]).resolve()!=path:raise ValueError("Config self identity differs")
    root=Path(config["output"]);state=read(config["stage_state"])
    if state.get("current_stage")!="R2C2" or state.get("r2c2_execution_authorized") is not True or state.get("training_authorized") is not False:
        raise PermissionError("Current stage-state must authorize R2C2 inference and forbid training")
    for key,digest_key in (("parent_authorization","parent_authorization_sha256"),("time_policy_file","time_policy_sha256"),("parent_acceptance","parent_acceptance_sha256")):
        if sha(config[key])!=config[digest_key]:raise ValueError("Authorization/time correction/parent audit changed: "+key)
    for filename,expected in config["input_hashes"].items():
        if sha(filename)!=expected:raise ValueError("Frozen R2C2 input changed: "+filename)
    checkpoint=Path(config["checkpoint"])
    if checkpoint.stat().st_size!=config["checkpoint_bytes"] or sha(checkpoint)!=CHECKPOINT_SHA:
        raise ValueError("Trusted checkpoint size/SHA differs")
    sources=read(config["source_identity"])
    for row in sources["files"]:
        if sha(row["source"])!=row["sha256"] or sha(root/row["snapshot"])!=row["sha256"]:
            raise ValueError("Frozen stage source/snapshot differs: "+row["source"])
    if not any(Path(row["source"]).resolve()==path for row in sources["files"]):raise ValueError("Config absent from source freeze")
    verify_closed_prior(config)
    if (root/"run").exists() or (root/"run-claim.json").exists():raise FileExistsError("R2C2 is single-use; no replay or automatic retry")
    storage=storage_for_config(config).check(force=True)
    return config,dict(config_sha256=sha(path),source_identity_sha256=sha(config["source_identity"]),
        prior_ledger_sha256=sha(config["ledger_path"]),storage=storage,Torch_imported="torch" in sys.modules,model_calls=0)


def run(config_path,*,execute=False):
    if execute is not True:raise PermissionError("Explicit --execute required")
    config,checked=preflight(config_path);root=Path(config["output"])
    with (root/"run-claim.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(kind="R2C2-one-use-inference-claim.v1",pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),**checked),stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    ledger=stage=None;result=None;failure=None
    try:
        ledger=ExistingInferenceLedger(config["ledger_path"],config)
        stage=ledger.start("R2C2-fixed300-native-sampler8-16-diagnostic")
        result=infer(config,stage=stage,storage=storage_for_config(config),execute=True)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    finally:
        if stage is not None:
            final=stage.finish("complete" if failure is None and result is not None and result["failure"] is None else "stopped_with_evidence")
            accounting=dict(closed=True,time_policy="record_only_v2",elapsed_model_work_seconds=final["stages"][-1]["observed_seconds"],
                model_time_hard_limit=None,ledger=final)
        else:accounting=dict(closed=True,model_stage_started=False,time_policy="record_only_v2")
        work=dict(kind="R2C2-one-run-fixed-weight-model-work.v1",closed=True,status=(result or {}).get("status","entry_failed"),
            failure=failure or (result or {}).get("failure"),preflight=checked,budget=accounting,inference_result=result,
            evaluation_inventory=(result or {}).get("evaluation_inventory",[]),run_was_once_claimed=True,automatic_retry=False,
            new_physics_controls=0,environment_constructions=0,optimizer_attempts=0,completed_updates=0)
        atomic_json(root/"model-work.json",work)
    print(json.dumps(dict(status=work["status"],failure=work["failure"],model_work_closed=True,model_seconds_statistics=accounting.get("elapsed_model_work_seconds")),ensure_ascii=False))
    return 0 if work["failure"] is None else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.check:
        if args.execute:raise ValueError("Choose CPU check or one actual execution")
        _,checked=preflight(args.config);print(json.dumps(checked,ensure_ascii=False));return 0
    return run(args.config,execute=args.execute)


if __name__=="__main__":raise SystemExit(main())
