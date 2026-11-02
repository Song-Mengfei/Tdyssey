"""Plan finite raw T/L SFT; --execute uses the external process/resource guard."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.raw_world_model_training import plan_training,export_boundaries,train


def _profile(args):
    from tdyssey.official_world_model import initial_profile,model_profile
    if args.checkpoint is None:raise ValueError("Explicit --checkpoint is required for execution")
    if args.initial:return initial_profile(args.checkpoint,args.checkpoint_sha256)
    if args.checkpoint_sha256 is None or args.sidecar is None:
        raise ValueError("Trained initialization requires checkpoint SHA256 and sidecar")
    return model_profile(args.checkpoint,args.checkpoint_sha256,sidecar=args.sidecar,
                         expected_sidecar_sha256=args.sidecar_sha256)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",type=Path,nargs="+",required=True,help="Finished official T and L dataset roots")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,help="Even unique-window count; default uses all balanced windows once")
    parser.add_argument("--gpc-root",type=Path,help="Explicit pinned upstream source for execution")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--initial",action="store_true",help="Explicit released initialization, for training only")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--sidecar",type=Path)
    parser.add_argument("--sidecar-sha256")
    parser.add_argument("--export-steps",type=int,nargs="*",default=[])
    parser.add_argument("--seconds",type=float,default=1200,help="Total guarded wall/monotonic deadline, max1800")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--_plan-sha256",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args._worker:
        from tdyssey.raw_training_guard import worker_gate
        with worker_gate() as control:
            plan=plan_training(args.datasets,args.steps)
            if plan["plan_sha256"]!=args._plan_sha256:raise ValueError("Worker data/order changed after parent planning")
            result=train(plan,args.gpc_root,_profile(args),args.output,execute=True,check=control.check,export_steps=args.export_steps)
            print(json.dumps(dict(status=result["status"],counts=result["counts"])),file=sys.stderr)
        return result
    try:
        plan=plan_training(args.datasets,args.steps);exports=export_boundaries(plan["steps"],args.export_steps)
        if not 10<args.seconds<=1800:raise ValueError("Choose a finite deadline above10 and at most1800 seconds")
        if not args.execute:
            summary={key:plan[key] for key in ("kind","steps","seed","available_windows","unused_windows",
                "selected_shapes","whole_episode_splits","selected_split","epochs","resampling","scene_identity",
                "action_training","objective","list_sha256","plan_sha256")}
            summary.update(mode="read_only",export_steps=exports,output=str(args.output),training_executed=False)
            print(json.dumps(summary,indent=2));return summary
        if args.gpc_root is None:raise ValueError("Execution needs --gpc-root")
        _profile(args)
        from tdyssey.raw_training_guard import run_guarded
        original=list(sys.argv[1:] if argv is None else argv)
        command=[str(Path(__file__).resolve()),*original,"--_worker","--_plan-sha256",plan["plan_sha256"]]
        return run_guarded(dict(output_dir=str(args.output),max_updates=plan["steps"],wall_seconds=args.seconds),command,execute=True)
    except (ValueError,PermissionError,FileNotFoundError) as error:parser.error(str(error))


if __name__=="__main__":
    result=main()
    if result.get("status") in ("failed","stopped"):raise SystemExit(1)
