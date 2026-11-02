"""Single-use R2C1R1 entry. CPU preflight is separate; root owns the only launch."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c1r1_training import sha,atomic_json,train


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def config_contract(config):
    if config.get("execution_authorized") is not True or config.get("is_proposal") is not False or config.get("resume") is not None:
        raise PermissionError("R2C1R1 needs current explicit authorization and original initialization, never old resume")
    if config["budget"]!={"max_attempts":300,"max_model_seconds":1172.641,"latest_new_attempt_seconds":872.641,"tail_reserve_seconds":300.}:
        raise ValueError("Only the authorized300/1172.641/872.641/300 caps")
    if config["seed"]!=5478 or config["effective_batch"]!=16 or config["microbatch"]!=4 or config["amp"] is not True:
        raise ValueError("Fixed native seed5478/effective16/micro4/AMP")
    if config["optimizer"]!={"lr":.0001,"betas":[.9,.999],"eps":1e-8,"weight_decay":.01}:
        raise ValueError("Original native AdamW settings required")
    if config["sampler"]!={"num_steps_denoising":3,"sigma_min":.002,"sigma_max":5,"rho":7,"order":1,"s_churn":0,"s_cond":0}:
        raise ValueError("Original three-step native sampler required")
    if config["init_checkpoint_sha256"]!="4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892":
        raise ValueError("Only original published denoiser asset")
    if not math.isfinite(config["native_call_reserve_seconds"]) or config["native_call_reserve_seconds"]<1:
        raise ValueError("Nonzero native-call time reserve required")
    if Path(config["repo_root"]).resolve()!=ROOT or Path(config["output"]).resolve()!=ROOT/"artifacts/native-rgb-supervised-20261029/r2c1r1":
        raise ValueError("R2C1R1 output must be its unique authorized stage root")
    return True


def preflight(config_path):
    config_path=Path(config_path).resolve();config=read(config_path);config_contract(config)
    if Path(config["config_file"]).resolve()!=config_path:raise ValueError("Config self path differs")
    stage=Path(config["output"]).resolve();state=read(config["stage_state"])
    if state["current_stage"]!="R2C1R1" or state.get("r2c1r1_execution_authorized") is not True:
        raise PermissionError("Current stage-state does not authorize R2C1R1")
    for path,expected in config["input_hashes"].items():
        if sha(path)!=expected:raise ValueError("Frozen R2C0 source/input changed: "+path)
    if Path(config["init_checkpoint"]).stat().st_size!=35043916 or sha(config["init_checkpoint"])!=config["init_checkpoint_sha256"]:
        raise ValueError("Original asset size/hash differs")
    source=read(config["source_identity"])
    for row in source["files"]:
        if sha(row["source"])!=row["sha256"] or sha(stage/row["snapshot"])!=row["sha256"]:
            raise ValueError("Frozen source/snapshot changed: "+row["source"])
    if not any(Path(row["source"]).resolve()==config_path for row in source["files"]):raise ValueError("Config not included in source freeze")
    from tdyssey.native_rgb_r2c1r1_budget import validate_authorization,verify_closed_prior
    validate_authorization(config)
    verify_closed_prior(config)
    if (stage/"run").exists() or (stage/"run-claim.json").exists():raise FileExistsError("R2C1R1 is single-use; no replay or automatic restart")
    from tdyssey.native_rgb_r2c1r1_budget import storage_for_config
    storage=storage_for_config(config).check(force=True)
    return config,dict(config_sha256=sha(config_path),source_identity_sha256=sha(config["source_identity"]),prior_ledger_sha256=sha(config["ledger_path"]),storage=storage,Torch_imported="torch" in sys.modules,model_calls=0)


def run(config_path,*,execute=False):
    if execute is not True:raise PermissionError("Explicit --execute required; preflight alone never loads a model")
    config,checked=preflight(config_path);stage_root=Path(config["output"])
    claim=dict(kind="R2C1R1-one-use-claim.v1",pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat(),**checked)
    with (stage_root/"run-claim.json").open("x",encoding="utf-8") as stream:
        json.dump(claim,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    from tdyssey.native_rgb_training import CostLedger
    from tdyssey.native_rgb_r2c1r1_budget import make_budget
    budget=stage=None;result=None;failure=None
    try:
        ledger=CostLedger(config["ledger_path"],3000,5400.)
        stage=ledger.start("R2C1R1-selected0.1s-learning-diagnostic")
        budget=make_budget(stage,config)
        result=train(config,work_budget=budget,execute=True)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc))
    finally:
        if budget is not None:
            budget.finish("bounded_complete" if failure is None and result is not None and result["failure"] is None else "stopped_with_evidence")
            budget_report=budget.report()
        elif stage is not None:
            stage.finish("budget_setup_failed_no_model")
            budget_report=dict(closed=True,ledger=stage.ledger.snapshot())
        else:budget_report=dict(closed=True,model_stage_started=False)
        record=dict(kind="R2C1R1-one-run-model-work.v1",closed=True,status=(result or {}).get("status","entry_failed"),failure=failure or (result or {}).get("failure"),
            preflight=checked,budget=budget_report,training_result=result,evaluation_inventory=(result or {}).get("evaluation_inventory",[]),
            run_was_once_claimed=True,automatic_retry=False,new_physics_controls=0,environment_constructions=0)
        atomic_json(stage_root/"model-work.json",record)
    print(json.dumps(dict(status=record["status"],failure=record["failure"],charged_ledger=budget_report.get("ledger",{}).get("charged_updates"),model_work_closed=True),ensure_ascii=False))
    return 0 if record["failure"] is None else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True);parser.add_argument("--execute",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.check:
        if args.execute:raise ValueError("Choose CPU preflight or unique execution")
        _,report=preflight(args.config);print(json.dumps(report,ensure_ascii=False));return 0
    return run(args.config,execute=args.execute)


if __name__=="__main__":raise SystemExit(main())
