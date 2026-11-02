"""Bounded hidden independent launcher for one authorized recovery subtask."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tdyssey/native_rgb_r3a_data.py').is_file())
BASE=ROOT/'artifacts/native-rgb-supervised-20261029/r3b'
HELPERS=BASE.parent/'r3a/background-worker.py'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True,type=Path);args=parser.parse_args()
    api=runpy.run_path(str(HELPERS));read,write,sha,utc,job=(api[k]for k in('read','atomic_json','sha','utc','process_job_state'))
    spec=read(args.spec);output=Path(spec['output']).resolve();entry=Path(spec['entry']).resolve()
    if not output.is_relative_to(BASE/'attempts')or not entry.is_relative_to(ROOT):raise ValueError('Out-of-scope launch path')
    if spec.get('execution_authorized')is not True:raise PermissionError('Explicit launch authorization missing')
    for b in spec['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Frozen launch source/config changed: '+b['path'])
    venv=ROOT.parent/'Tdyssey/.venv/Scripts/python.exe';base_python=Path(api['python_process_image']()).resolve()
    if Path(sys.executable).resolve()!=venv.resolve():raise ValueError('Existing venv required')
    output.mkdir(parents=True,exist_ok=True)
    with(output/'launcher-once-claim.json').open('x',encoding='utf-8')as stream:
        json.dump(dict(pid=os.getpid(),at_utc=utc(),run_id=spec['run_id']),stream);stream.flush();os.fsync(stream.fileno())
    command=[str(venv),'-X','utf8','-B','-u',str(entry)]+[str(s).replace('@LAUNCHER_PID@',str(os.getpid()))for s in spec['args']]
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
    receipt=dict(kind='R3B-recovery-independent-launch.v1',run_id=spec['run_id'],at_utc=utc(),launcher_pid=os.getpid(),
        launcher_ppid=os.getppid(),launcher_job=job(),actual_process_executable=str(base_python),venv_python_identity=str(venv),
        output=str(output),entry=str(entry),command=command,cwd=str(ROOT),creation_flags=flags,creation_flags_hex=hex(flags),
        spec_path=str(args.spec.resolve()),spec_sha256=sha(args.spec),launcher_sha256=sha(__file__),entry_sha256=sha(entry),
        hidden=True,close_fds=True,shell=False,automatic_restart=False,launch_succeeded=False)
    write(output/'launch-attempt.json',receipt);child=None
    try:
        info=subprocess.STARTUPINFO();info.dwFlags|=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
        env=os.environ.copy();env['__PYVENV_LAUNCHER__']=str(venv);env['PYTHONDONTWRITEBYTECODE']='1'
        with(output/'worker.stdout.log').open('ab',buffering=0)as stdout,(output/'worker.stderr.log').open('ab',buffering=0)as stderr:
            child=subprocess.Popen(command,executable=str(base_python),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                close_fds=True,shell=False,creationflags=flags,startupinfo=info,env=env)
        receipt['worker_pid']=child.pid;write(output/'launch-attempt.json',receipt)
        deadline=time.monotonic()+120
        while not(output/'startup-proof.json').is_file():
            if child.poll()is not None:raise RuntimeError('Worker exited before startup proof: '+str(child.returncode))
            if time.monotonic()>=deadline:raise RuntimeError('Startup proof deadline exceeded')
            time.sleep(.05)
        proof=read(output/'startup-proof.json');live=job(child.pid)
        if proof['pid']!=child.pid or live['is_process_in_job']or not live['process_active']:raise RuntimeError('Worker not independent/active')
        if Path(proof['actual_process_executable']).resolve()!=base_python:raise RuntimeError('Unexpected actual Python PE')
        if Path(proof['prefix']).resolve()!=venv.parent.parent.resolve():raise RuntimeError('Existing venv identity lost')
        if proof.get('run_id')!=spec['run_id']:raise RuntimeError('Worker run_id differs')
        receipt.update(launch_succeeded=True,independent_job_verified=True,worker_job_query=live,ready_at_utc=utc(),launcher_exit_intent=0)
        write(output/'launcher-receipt.json',receipt);return 0
    except BaseException as exc:
        receipt.update(error=repr(exc),worker_pid=None if child is None else child.pid,actual_child_exitcode=None if child is None else child.poll(),launcher_exit_intent=2,no_job_bound_fallback=True)
        write(output/'launcher-receipt.json',receipt);return 2

if __name__=='__main__':raise SystemExit(main())
