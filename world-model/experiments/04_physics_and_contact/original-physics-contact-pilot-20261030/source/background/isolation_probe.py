"""Optional detached engineering probe. Never starts a collector or imports it.

Root may run this through the same hidden Explorer broker planned for launcher.py.
Parent exits immediately after spawning one fixed standard-library-only child.
The child waits for the actual parent exit, proves it is outside a Windows Job,
and writes a receipt under pilot/background/isolation-probes/<unique-label>/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from common import (actual_python_pe, atomic_json, child_environment, exclusive_json,
                    in_job, utc, wait_launcher_exit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--python-executable',type=Path,required=True)
    parser.add_argument('--probe-child',action='store_true')
    parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.label):
        parser.error('Unique safe label required')
    python=args.python_executable.resolve(strict=True)
    if python.name.lower()!='python.exe' or python.parent.name!='Scripts' or not (python.parent.parent/'pyvenv.cfg').is_file():
        parser.error('Use an existing Windows venv')
    destination=Path(__file__).resolve().parent/'isolation-probes'/args.label
    if args.probe_child:
        if not args.parent_pid or args.parent_pid==os.getpid():
            parser.error('Actual probe parent PID required')
        result=dict(kind='tdyssey-zero-physics-detached-isolation-probe.v1',label=args.label,
            pid=os.getpid(),observed_parent_pid=os.getppid(),at_utc=utc(),is_process_in_job=in_job(),
            interpreter_pe=actual_python_pe(),sys_executable=sys.executable,sys_prefix=sys.prefix,
            expected_venv=str(python.parent.parent),actual_physics_steps=0,model_calls=0,optimizer_updates=0,
            collection_processes_started=0)
        try:
            result['parent_exit']=wait_launcher_exit(args.parent_pid)
            # A small independent lifetime after parent exit makes the boundary visible.
            time.sleep(1)
            result['after_parent_exit_is_process_in_job']=in_job()
            result['passed']=(not result['is_process_in_job'] and not result['after_parent_exit_is_process_in_job']
                              and Path(sys.prefix).resolve()==python.parent.parent)
        except Exception as error:
            result.update(passed=False,error=repr(error))
        result['finished_at_utc']=utc()
        exclusive_json(destination/'child-result.json',result)
        return 0 if result['passed'] else 1
    if destination.exists():
        raise RuntimeError('Existing probe label; preserve receipt and use a new reviewed label')
    destination.mkdir(parents=True,exist_ok=False)
    evidence=dict(kind='tdyssey-zero-physics-detached-isolation-probe.v1',label=args.label,
        parent_pid=os.getpid(),parent_is_process_in_job=in_job(),actual_physics_steps=0,
        model_calls=0,optimizer_updates=0,collection_processes_started=0,at_utc=utc())
    exclusive_json(destination/'parent-claim.json',evidence)
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    command=[actual_python_pe(),'-X','utf8','-B',str(Path(__file__).resolve()),'--label',args.label,
             '--python-executable',str(python),'--probe-child','--parent-pid',str(os.getpid())]
    try:
        with (destination/'stdout.log').open('ab') as out,(destination/'stderr.log').open('ab') as err:
            child=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parent),
                env=child_environment(dict(python_executable=str(python))),stdin=subprocess.DEVNULL,
                stdout=out,stderr=err,close_fds=True,startupinfo=startup,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
        evidence.update(spawn_succeeded=True,child_pid=child.pid,command=command)
    except Exception as error:
        evidence.update(spawn_succeeded=False,error=repr(error))
        atomic_json(destination/'parent-receipt.json',evidence)
        raise
    atomic_json(destination/'parent-receipt.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
