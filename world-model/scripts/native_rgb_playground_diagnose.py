"""Inference-only finite R3B playground diagnostic entry; no default execution."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_playground_diagnostic_runner import make_plan,write,preflight,execute

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--check',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    for key,value in(('SDL_VIDEODRIVER','dummy'),('SDL_AUDIODRIVER','dummy'),('PYGAME_HIDE_SUPPORT_PROMPT','1'),('PYTHONDONTWRITEBYTECODE','1')):os.environ.setdefault(key,value)
    if args.plan:
        if args.execute or args.check:parser.error('Plan creation does not execute')
        if args.plan.exists():raise FileExistsError('Preserve existing frozen plan')
        write(args.plan,make_plan());print(json.dumps(dict(plan=str(args.plan),Torch_imported='torch'in sys.modules,physics_steps=0)));return 0
    if args.config is None:parser.error('--config required for check/execute')
    if args.check:
        config,plan=preflight(args.config);print(json.dumps(dict(passed=True,run_id=config['run_id'],planned_cases=len(plan['cases']),Torch_imported='torch'in sys.modules,physics_steps=0)));return 0
    if not args.execute:parser.error('Explicit --execute required')
    return execute(args.config,resume=args.resume)

if __name__=='__main__':raise SystemExit(main())
