"""Read-only compatibility plan; real execution needs separate human authority.

Default mode imports no model/decoder, creates no directory and claims no ledger.
--execute additionally requires a reviewed clean commit, authorization identity,
and the native guardian's actual-PID/nonce/Job Object handoff.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.wm_compat_identity import ATTEMPT_ID, REGISTRY, AUTHORIZATION_ID, validate_execution_identity
FROZEN_INVENTORY=ROOT/"artifacts/wm-interface-v2-audit/after.json"
FROZEN_INVENTORY_SHA256="f1e2325e87df1d7dd8ef513da1ee15061ed8787ba0b6ba731d43f9d68dea5871"
FROZEN_RUN_NAME="frozen-wm-validation-v1-20261024"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen_assets(assets, run_dir, checkpoint, inventory_path=FROZEN_INVENTORY):
    """Compare saved payloads to the previously sealed byte inventory."""
    content=Path(inventory_path).read_bytes()
    if hashlib.sha256(content).hexdigest()!=FROZEN_INVENTORY_SHA256:
        raise ValueError("Frozen input provenance inventory changed")
    rows=json.loads(content)["files"]
    expected={PureWindowsPath(row["path"]).name:row for row in rows
              if PureWindowsPath(row["path"]).parent.name==FROZEN_RUN_NAME}
    run_dir,checkpoint=Path(run_dir).resolve(),Path(checkpoint).resolve()
    identities={}
    for value in assets:
        path=Path(value).resolve()
        actual=digest(path)
        if path==checkpoint:
            from tdyssey.wm_compat_budget import EXACT_CHECKPOINT
            valid=path.stat().st_size==35043916 and actual==EXACT_CHECKPOINT
        else:
            row=expected.get(path.name) if path.parent==run_dir else None
            valid=row is not None and path.stat().st_size==row["bytes"] and actual==row["sha256"]
        if not valid:
            raise ValueError("Saved input differs from the sealed inventory: "+path.name)
        identities[str(path)]=actual
    identities[str(Path(inventory_path).resolve())]=FROZEN_INVENTORY_SHA256
    return identities


def read_only_plan():
    from tdyssey.wm_compat_budget import canonical_plan, DEFAULT_LIMITS
    plan=canonical_plan()
    return dict(mode="read_only_plan",attempt_id=ATTEMPT_ID,registry=REGISTRY,
        execution_authorized=False,development_only=True,
        real_measurement_rows=dict(planned=1072,available=1056,original_missing=16),
        planned_forecast_rows=len(plan["calls"]),available_forecast_rows=2268,missing_forecast_rows=36,
        planned_limits=DEFAULT_LIMITS,forecast_images_executed=0,model_loads=0,
        real_operations=0,ledger_created=False,transform_executed=False,
        warning="Engineering implementation does not authorize an experiment.")


def build_binding(args, handoff):
    from tdyssey.wm_compat_budget import canonical_plan, EXACT_CHECKPOINT
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("Real evaluation requires a clean reviewed source checkout")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if args.reviewed_commit!=head:
        raise ValueError("Reviewed commit must exactly match current HEAD")
    subprocess.run(["git","diff","--exit-code","994c6ff","--","tdyssey/wm_image_measurement_v2.py",
                    "tdyssey/wm_measurement_contract.py","configs/wm_image_measurement_v2.json"],cwd=ROOT,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    source_paths=[*(ROOT/"tdyssey").rglob("*.py"),ROOT/"scripts/validate_wm_compat.py",
        ROOT/"scripts/run_guarded_wm_compat.py",ROOT/"scripts/guarded_c_child.py",
        ROOT/"configs/wm_image_measurement_v2.json",ROOT/"manifests/gpc_world_model.json"]
    spec=json.loads((ROOT/"manifests/gpc_world_model.json").read_text())
    upstream_head=subprocess.check_output(["git","-C",str(args.gpc_root),"rev-parse","HEAD"],text=True).strip()
    if upstream_head!=spec["source_commit"]:
        raise ValueError("GPC source revision differs from manifest")
    for name,expected in spec["source_sha256"].items():
        path=args.gpc_root/name
        if digest(path)!=expected:
            raise ValueError("GPC source differs: "+name)
        source_paths.append(path)
    if args.checkpoint.stat().st_size!=35043916 or digest(args.checkpoint)!=EXACT_CHECKPOINT:
        raise ValueError("Only the exact phase-two checkpoint is allowed")
    assets=[args.run_dir/"report.json",args.checkpoint]
    assets.extend(args.run_dir/f"cal-{shape}-{i:02d}.npz" for shape in ("L","T") for i in range(12))
    for branch in canonical_plan()["branches"]:
        if branch["status"]=="missing":
            continue
        prefix=branch["anchor"]+"-"+branch["candidate"]
        assets.append(args.run_dir/(prefix+"-physical.npz"))
        assets.extend(args.run_dir/(prefix+f"-noise-{seed}.npz") for seed in (5478,5479,5480))
    from tdyssey.wm_compat_evaluation import REPORT_SHA256
    from tdyssey.wm_compat_transform import T1_SHA256
    if digest(assets[0])!=REPORT_SHA256 or len(assets)!=278:
        raise ValueError("Frozen input inventory differs")
    source={str(path.resolve()):digest(path) for path in sorted(source_paths)}
    asset=verify_frozen_assets(sorted(assets),args.run_dir,args.checkpoint)
    launch_seal=handoff["launch_seal"]
    asset[launch_seal["path"]]=launch_seal["sha256"]
    binding=dict(attempt_id=ATTEMPT_ID,protocol_sha256=digest(ROOT/"docs/GPC_INPUT_COMPATIBILITY_PROPOSAL.md"),
        code_sha256=source,asset_sha256=asset,checkpoint_sha256=EXACT_CHECKPOINT,
        reviewed_commit=head,authorization_id=args.authorization_id,
        t1_spec_sha256=T1_SHA256,
        launch_seal=launch_seal,
        guardian_started_monotonic=handoff["started_monotonic"],
        guardian_deadline_monotonic=handoff["deadline_monotonic"])
    binding["code_sha256"][str((ROOT/"docs/GPC_INPUT_COMPATIBILITY_PROPOSAL.md").resolve())]=binding["protocol_sha256"]
    return binding


def binding_checker(binding):
    paths={**binding["code_sha256"],**binding["asset_sha256"]}
    identities={name:(Path(name).stat().st_size,Path(name).stat().st_mtime_ns) for name in paths}
    def check(*, full=False):
        for name,expected in paths.items():
            path=Path(name)
            if (path.stat().st_size,path.stat().st_mtime_ns)!=identities[name] or (full and digest(path)!=expected):
                raise ValueError("Bound source or asset changed: "+name)
    return check


def execute(args, handoff):
    from tdyssey.wm_compat_budget import CompatBudget
    from tdyssey.wm_compat_guard import WorkerHeartbeat, FORBIDDEN_COUNTERS, from_handoff
    zero=dict(phase="setup",model_instances=0,model_loads=0,cuda_allocated_bytes=0,cuda_reserved_bytes=0,
              **{name:0 for name in FORBIDDEN_COUNTERS})
    budget=None
    with WorkerHeartbeat(zero,handoff=handoff) as heartbeat:
        try:
            binding=build_binding(args,handoff)
            verify=binding_checker(binding)
            remaining=math.floor(handoff["deadline_monotonic"]-time.monotonic())
            if remaining<=0:
                raise TimeoutError("Original guardian deadline expired during setup")
            budget=CompatBudget.create(ROOT,args.ledger_dir,args.output_dir/"results",binding,
                limits={"seconds":remaining},telemetry_dirs=(args.output_dir/"guardian",),
                storage=from_handoff(handoff))
            from tdyssey.wm_compat_engine import ChunkSink, TorchBackend
            from tdyssey.wm_compat_evaluation import load_saved_inputs,run_evaluation,save_summary
            from tdyssey.wm_image_measurement_v2 import decode
            sink=ChunkSink(budget)
            sink.record("binding.json",binding,maximum=128*1024)
            heartbeat.update(phase="measurement")
            inputs=load_saved_inputs(args.run_dir)
            if any(name in sys.modules for name in ("torch","pymunk","pygame","diffusion_policy")):
                raise RuntimeError("Forbidden runtime imported before M gate")
            result=run_evaluation(inputs,budget=budget,
                backend_factory=lambda:TorchBackend(args.gpc_root,args.checkpoint),measure=decode,
                sink=sink,heartbeat=heartbeat,binding_check=verify)
            verify(full=True)
            if any(name in sys.modules for name in ("pymunk","pygame","diffusion_policy")):
                raise RuntimeError("Unexpected physical/policy runtime import")
            heartbeat.update(phase="export")
            save_summary(result,sink,budget,branches=inputs["branches"])
            heartbeat.update(phase="complete")
            print(json.dumps(dict(status=result["budget"]["status"],counts=result["budget"]["counts"],
                                  stop_reason=result["budget"]["stop_reason"])))
            return 0
        except BaseException as exc:
            if budget is not None:
                try:
                    budget.stop(type(exc).__name__+": "+str(exc))
                except BaseException:
                    pass
            try:
                heartbeat.update(phase="error",error=type(exc).__name__+": "+str(exc))
            except BaseException as secondary:
                exc.add_note("Secondary error-status publication failure: "+type(secondary).__name__+": "+str(secondary))
            raise


def validate_handoff_paths(args, handoff):
    """Bind writes to exactly the roots monitored by the native guardian."""
    for name in ("output_dir", "ledger_dir"):
        guardian_path=handoff.get(name)
        if guardian_path is None or getattr(args,name).resolve()!=Path(guardian_path).resolve():
            raise ValueError("Worker "+name+" differs from the guardian handoff")


def validate_execution_authority(args, handoff):
    """Require the production launch claim in addition to native ownership."""
    seal=handoff.get("launch_seal")
    validate_execution_identity(ROOT,args.output_dir,args.ledger_dir,args.authorization_id)
    fixed=ROOT/"artifacts"/REGISTRY/"launcher-attempt.json"
    if (not isinstance(seal,dict) or Path(seal.get("path","")).resolve()!=fixed.resolve()
            or seal.get("attempt_id")!=ATTEMPT_ID or args.authorization_id!=AUTHORIZATION_ID
            or seal.get("authorization_id")!=args.authorization_id
            or seal.get("reviewed_commit")!=args.reviewed_commit):
        raise ValueError("Execution requires the matching fixed permanent launcher seal")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--authorization-id",help="Identity of separately granted human execution authority")
    parser.add_argument("--reviewed-commit",help="Exact clean commit reviewed for that future run")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--ledger-dir",type=Path)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/frozen-wm-validation-v1-20261024")
    parser.add_argument("--gpc-root",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(read_only_plan(),indent=2,sort_keys=True))
        return 0
    if not args.authorization_id or not args.reviewed_commit or any(
            getattr(args,name) is None for name in ("output_dir","ledger_dir","gpc_root","checkpoint")):
        parser.error("Execution requires separate authorization, exact reviewed commit and explicit new paths")
    from tdyssey.wm_compat_guard import verify_guardian
    handoff=verify_guardian()
    for name in ("output_dir","ledger_dir","run_dir","gpc_root","checkpoint"):
        setattr(args,name,getattr(args,name).resolve())
    validate_handoff_paths(args,handoff)
    validate_execution_authority(args,handoff)
    return execute(args,handoff)


if __name__=="__main__":
    raise SystemExit(main())
