"""One-shot detached calibration-v2 launcher. Never launches formal collection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from common import (KIND, actual_python_pe, atomic_json, child_environment,
                    exclusive_json, in_job, utc, validate_config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    config,paths,by_id=validate_config(args.config,args.config_sha,require_release=not args.check_only,mode='calibration-v2')
    background=Path(config['artifact_root'])/'background'
    runtime=background/'calibration-v2'
    evidence=dict(kind=KIND,mode='calibration-v2',run_id=config['run_id'],config_sha256=args.config_sha,
        launcher_pid=os.getpid(),launcher_is_process_in_job=in_job(),interpreter_pe=actual_python_pe(),
        venv_launcher=config['python_executable'],at_utc=utc(),hidden=True,
        actual_physics_steps=0,model_calls=0,optimizer_updates=0)
    if args.check_only:
        print(json.dumps(dict(evidence,passed=True,files_written=0,processes_started=0,
                              planned_calibration_attempts=18,formal_released=False),ensure_ascii=False))
        return 0
    for name in ('launch-claim.json','worker.lock.json','worker-startup.json','completion.json'):
        if (runtime/name).exists():
            raise RuntimeError('Existing calibration attempt requires owner audit; never auto retry: '+name)
    exclusive_json(runtime/'launch-claim.json',evidence)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    command=[actual_python_pe(),'-X','utf8','-B',str(background/'calibration-worker.py'),
        '--config',str(args.config.resolve()),'--config-sha',args.config_sha,'--launcher-pid',str(os.getpid())]
    try:
        with (runtime/'worker-stdout.log').open('ab') as out,(runtime/'worker-stderr.log').open('ab') as err:
            child=subprocess.Popen(command,cwd=config['repo'],env=child_environment(config),stdin=subprocess.DEVNULL,
                stdout=out,stderr=err,close_fds=True,startupinfo=startup,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
        evidence.update(spawn_succeeded=True,worker_pid=child.pid,command=command,
            creation_flags=['DETACHED_PROCESS','CREATE_NEW_PROCESS_GROUP','CREATE_BREAKAWAY_FROM_JOB'],
            note='Worker must prove actual parent exit and IsProcessInJob=false before first calibration control')
        atomic_json(runtime/'launcher-receipt.json',evidence)
        print(json.dumps(evidence,ensure_ascii=False),flush=True)
        return 0
    except Exception as error:
        evidence.update(spawn_succeeded=False,error=repr(error))
        atomic_json(runtime/'launcher-receipt.json',evidence)
        raise


if __name__=='__main__':
    raise SystemExit(main())
