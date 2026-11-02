"""Offline L diagnosis using saved records/RGB only; never launch an experiment.

Write into a fresh derived directory. No model, simulator, policy, optimizer,
network or formal-attempt entry point is part of this command.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write_json(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:stream.write(data)
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/wm-input-compat-eval-v3-20261025")
    parser.add_argument("--physical-dir",type=Path,default=ROOT/"artifacts/frozen-wm-validation-v1-20261024")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--replay-inputs",action="store_true",help="Reconstruct saved windows and compare hashes; no predictor")
    parser.add_argument("--figures",action="store_true",help="Read existing RGB/NPY and export fixed-rule diagnostic plates")
    parser.add_argument("--residual-selection",type=Path,help="Frozen selection JSON for residual localization without object refits")
    args=parser.parse_args(argv)
    destination=args.output_dir.resolve()
    run,physical=args.run_dir.resolve(),args.physical_dir.resolve()
    if any(destination==root or destination.is_relative_to(root) for root in (run,physical)):
        parser.error("Derived diagnostics must be outside the immutable input roots")
    destination.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    from tdyssey.wm_l_diagnostics import analyze_run,save_artifacts
    evidence=save_artifacts(analyze_run(run),destination)
    if args.replay_inputs:
        from tdyssey.wm_l_replay_diagnostics import audit_saved_conditioning
        replay=audit_saved_conditioning(run,physical)
        evidence["conditioning_replay"]=write_json(destination/"conditioning-replay.json",replay)
        if not replay["passed"]:
            raise ValueError("Saved input reconstruction failed; inspect preserved evidence")
    if args.figures:
        from tdyssey.wm_l_diagnostic_figures import export_case_figures
        manifest=export_case_figures(run,physical,destination)
        evidence["visual_cases"]=manifest
    if args.residual_selection:
        from tdyssey.wm_l_measurement_diagnostics import audit_selected_residuals
        residuals=audit_selected_residuals(run,args.residual_selection)
        evidence["residual_localization"]=write_json(destination/"residual-localization.json",residuals)
    forbidden=[name for name in ("torch","pymunk","pygame","diffusion_policy") if name in sys.modules]
    if forbidden:
        raise RuntimeError("Forbidden runtime imported during offline analysis: "+repr(forbidden))
    sources=[ROOT/"scripts/diagnose_wm_l.py",*(ROOT/"tdyssey").glob("wm_l_*.py"),
        ROOT/"tdyssey/wm_compat_engine.py",ROOT/"tdyssey/wm_image_measurement_v2.py",
        ROOT/"configs/wm_image_measurement_v2.json"]
    manifest=dict(kind="offline_L_diagnosis.v1",created_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run),physical_dir=str(physical),wall_seconds=time.monotonic()-start,evidence=evidence,
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        new_model_loads=0,new_model_predictions=0,new_physical_operations=0,policy_samples=0,optimizer_updates=0,
        forbidden_runtime_modules_imported=forbidden,
        interpretation="Offline development diagnosis, not a new M/T/L attempt or independent validation")
    written=write_json(destination/"manifest.json",manifest)
    print(json.dumps(dict(completed=True,manifest=written,wall_seconds=manifest["wall_seconds"])))
    return 0


if __name__=="__main__":raise SystemExit(main())
