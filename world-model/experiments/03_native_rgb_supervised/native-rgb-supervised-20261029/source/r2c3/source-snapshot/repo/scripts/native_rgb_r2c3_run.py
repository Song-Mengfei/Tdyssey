"""One explicitly authorized complete-state native training continuation."""
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
from tdyssey.native_rgb_r2c3_training import train,read,sha,atomic_json,RESUME_SHA,R1_CONFIG_SHA


def config_contract(config):
    if config.get("execution_authorized") is not True or config.get("training_authorized") is not True or config.get("is_proposal") is not False:
        raise PermissionError("Current explicit C3 continuation authorization required")
    if config.get("time_policy")!="record_only_v2" or config.get("cap_time",False) is not None:
        raise ValueError("No model time budget/cutoff is authorized")
    if config.get("targets")!=[1000,3000,6000] or config.get("target_lineage_updates")!=6000 or config.get("resume_lineage_updates")!=300 or config.get("target_new_complete_updates")!=5700:
        raise ValueError("C3 must continue300 to1000/3000/6000,5700 new complete updates")
    if config.get("evaluation_settings")!=[3,16] or config.get("fixed_loss_checkpoints")!=[1000,3000,6000]:
        raise ValueError("Only scheduled3/16 matrix and no repeated300-point evaluation")
    if (config.get("seed"),config.get("effective_batch"),config.get("microbatch"),config.get("amp"))!=(5478,16,4,True):raise ValueError("Original seed/batch/AMP required")
    if config.get("optimizer")!={"lr":.0001,"betas":[.9,.999],"eps":1e-8,"weight_decay":.01}:raise ValueError("Original AdamW settings required")
    if config.get("sampler")!={"num_steps_denoising":3,"sigma_min":.002,"sigma_max":5,"rho":7,"order":1,"s_churn":0,"s_cond":0}:raise ValueError("Original sampler math required")
    if config.get("resume_checkpoint_sha256")!=RESUME_SHA or config.get("resume_checkpoint_bytes")!=105300718:raise ValueError("Only actual R1 full300 checkpoint")
    if config.get("stage_artifact_cap_bytes")!=400*1024**2 or config.get("max_cuda_bytes")!=6*1024**3:raise ValueError("Explicit400MiB artifact/6GiB CUDA guards required")
    if config.get("stop_on_amp_skip") is not False or config.get("max_consecutive_overflows")!=16 or config.get("checkpoint_interval_updates")!=100:raise ValueError("Normal AMPskip continues;16consecutive guards,100-success checkpoints required")
    if Path(config["repo_root"]).resolve()!=ROOT or Path(config["output"]).resolve()!=ROOT/"artifacts/native-rgb-supervised-20261029/r2c3":raise ValueError("Unique C3 output required")
    return True


def preflight(config_path):
    path=Path(config_path).resolve();config=read(path);config_contract(config);root=Path(config["output"])
    if Path(config["config_file"]).resolve()!=path:raise ValueError("Config self path differs")
    state=read(config["stage_state"])
    if state.get("current_stage")!="R2C3" or state.get("r2c3_execution_authorized") is not True or state.get("training_authorized") is not True:raise PermissionError("Stage-state must authorize C3 continuation")
    for filename,expected in config["input_hashes"].items():
        if sha(filename)!=expected:raise ValueError("Frozen C3 input changed: "+filename)
    if sha(config["resume_checkpoint"])!=RESUME_SHA or Path(config["resume_checkpoint"]).stat().st_size!=105300718 or sha(config["parent_resume_config"])!=R1_CONFIG_SHA:raise ValueError("Parent checkpoint/config differs")
    for key,digest_key in (("parent_authorization","parent_authorization_sha256"),("time_policy_file","time_policy_sha256")):
        if sha(config[key])!=config[digest_key]:raise ValueError("Authorization/time policy differs")
    sources=read(config["source_identity"])
    for row in sources["files"]:
        if sha(row["source"])!=row["sha256"] or sha(root/row["snapshot"])!=row["sha256"]:raise ValueError("Frozen C3 source/snapshot changed: "+row["source"])
    if not any(Path(row["source"]).resolve()==path for row in sources["files"]):raise ValueError("Config absent from source freeze")
    if sha(config["ledger_path"])!=config["prior_ledger_sha256"] or sha(config["prior_ledger_copy"])!=config["prior_ledger_sha256"]:raise ValueError("Closed C2 ledger prior/copy differs")
    prior=read(config["ledger_path"])
    if prior!=read(config["prior_ledger_copy"]) or prior.get("active_stage") is not None or prior.get("max_seconds") is not None or prior.get("charged_updates")!=550 or prior.get("completed_updates")!=550:raise ValueError("Actual closed550 v2 history must be preserved")
    if (root/"run").exists() or (root/"run-claim.json").exists():raise FileExistsError("C3 is single-use, no automatic retry")
    from tdyssey.native_rgb_r2c3_continuation import storage_for_config
    storage=storage_for_config(config).check(force=True)
    return config,dict(config_sha256=sha(path),source_identity_sha256=sha(config["source_identity"]),prior_ledger_sha256=sha(config["ledger_path"]),storage=storage,Torch_imported="torch" in sys.modules,model_calls=0)


def run(config_path,*,execute=False):
    if execute is not True:raise PermissionError("Explicit --execute required")
    config,checked=preflight(config_path);root=Path(config["output"])
    with (root/"run-claim.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(kind="R2C3-one-use-continuation-claim.v1",pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),**checked),stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    from tdyssey.native_rgb_r2c3_continuation import create_ledger,make_budget
    ledger=stage=budget=None;result=failure=None
    try:
        ledger=create_ledger(config);stage=ledger.start("R2C3-resume300-to1000-3000-6000")
        budget=make_budget(stage,config);result=train(config,work_budget=budget,execute=True)
    except BaseException as exc:failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    finally:
        status="complete" if failure is None and result is not None and result["failure"] is None else "stopped_with_evidence"
        if budget is not None:budget.finish(status);accounting=budget.report()
        elif stage is not None:stage.finish(status);accounting=dict(closed=True,ledger=stage.ledger.snapshot())
        else:accounting=dict(closed=True,model_stage_started=False)
        work=dict(kind="R2C3-one-run-longitudinal-model-work.v1",closed=True,status=(result or {}).get("status","entry_failed"),failure=failure or (result or {}).get("failure"),
            preflight=checked,budget=accounting,training_result=result,evaluation_inventory=(result or {}).get("evaluation_inventory",[]),run_was_once_claimed=True,
            automatic_retry=False,new_physics_controls=0,environment_constructions=0)
        atomic_json(root/"model-work.json",work)
    print(json.dumps(dict(status=work["status"],failure=work["failure"],model_work_closed=True),ensure_ascii=False))
    return 0 if work["failure"] is None else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);parser.add_argument("--execute",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.check:
        if args.execute:raise ValueError("Choose pure check or unique execution")
        _,checked=preflight(args.config);print(json.dumps(checked,ensure_ascii=False));return 0
    return run(args.config,execute=args.execute)


if __name__=="__main__":raise SystemExit(main())
