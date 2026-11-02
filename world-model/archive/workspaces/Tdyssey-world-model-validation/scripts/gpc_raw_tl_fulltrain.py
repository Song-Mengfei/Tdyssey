"""Read-only plan unless the exact one-run raw T/L authority is supplied."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey import gpc_raw_fulltrain_entry as entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',type=Path,default=entry.CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--authorization-id',default='')
    parser.add_argument('--reviewed-commit',default='')
    parser.add_argument('--authorized-config-sha256',default='')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--gpc-root',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            from tdyssey.gpc_raw_fulltrain_worker import run_worker
            return run_worker(args.worker)
        prepared = entry.plan(args.config)
        if not args.execute:
            print(json.dumps({k:prepared[k] for k in ('kind','ready','gaps','config_sha256','inputs_sha256','plan_identities')},indent=2))
            return 0
        binding = entry.verify_execution(prepared,authorization_id=args.authorization_id,
            reviewed_commit=args.reviewed_commit,authorized_config_sha256=args.authorized_config_sha256)
        if args.gpc_root is None or args.checkpoint is None: raise PermissionError('Explicit pinned source/initial checkpoint required')
        from tdyssey.gpc_adaptation_guard import supervise
        report = supervise(prepared,binding,python_executable=args.python,gpc_root=args.gpc_root,
            checkpoint=args.checkpoint,script_path=Path(__file__),config_path=args.config,raw_fulltrain=True)
        print(json.dumps(report,indent=2))
        return 0 if report['status'] == 'complete' else 2
    except Exception as error:
        print(json.dumps(dict(status='refused_or_stopped',reason=type(error).__name__+': '+str(error))))
        return 2


if __name__ == '__main__': raise SystemExit(main())
