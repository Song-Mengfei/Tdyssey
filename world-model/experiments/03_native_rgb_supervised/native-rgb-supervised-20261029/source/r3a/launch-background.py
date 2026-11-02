"""Launch one job-independent hidden Windows R3A worker, or inspect its local proof.

No automatic restart, GPU training, physics in dummy mode, or offline chat callback.
Breakaway denial is recorded and never silently downgraded to a job-bound child.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid

BASE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dummy',action='store_true');mode.add_argument('--collect',action='store_true');mode.add_argument('--inspect',action='store_true')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--config',type=Path)
    args=p.parse_args();directory=args.run_dir.resolve()
    if not directory.is_relative_to(BASE):raise ValueError('All launcher output must remain inside R3A artifacts')
    worker=BASE/'background-worker.py';api=runpy.run_path(str(worker))
    read=api['read'];write=api['atomic_json'];sha=api['sha'];utc=api['utc'];job=api['process_job_state']
    if args.inspect:
        result=dict(run_dir=str(directory),at_utc=utc())
        for name in('launcher-receipt.json','worker-startup-proof.json','heartbeat.json','status.json','completion.json','failure.json'):
            if(directory/name).exists():result[name]=read(directory/name)
        if'receipt'in result:raise AssertionError('Unexpected legacy receipt key')
        if 'launcher-receipt.json'in result:
            try:result['live_worker_job_query']=job(result['launcher-receipt.json']['worker_pid'])
            except OSError as exc:result['live_worker_job_query']=dict(active_unknown_or_already_exited=True,error=str(exc))
        print(json.dumps(result,ensure_ascii=False));return 0
    if os.name!='nt':raise RuntimeError('Detached launch is implemented only for Windows')
    if args.collect and args.config is None:raise ValueError('Actual collection needs an explicit frozen v3 --config')
    if args.dummy and args.config is not None:raise ValueError('Dummy smoke test must not accept a collector config')
    collector=ROOT/'scripts/native_rgb_r3a_collect_v3.py'
    config=None if args.config is None else args.config.resolve()
    if config is not None:
        value=read(config)
        if(value.get('execution_authorized')is not True or value.get('training_authorized',False)is not False
                or Path(value['output']).resolve()!=BASE):raise ValueError('Only actual R3A collection scope accepted')
        if not collector.is_file():raise FileNotFoundError('Audited v3 continuation collector is not ready')
    directory.mkdir(parents=True,exist_ok=False)
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB
    base_python=Path(api['python_process_image']()).resolve()
    launch_id=uuid.uuid4().hex;selected_mode='dummy'if args.dummy else'collect'
    command=[sys.executable,'-u',str(worker),'--run-dir',str(directory),'--launch-id',launch_id,
        '--launcher-pid',str(os.getpid()),'--creation-flags',str(flags),'--worker-sha256',sha(worker),'--mode',selected_mode]
    command+=['--base-python',str(base_python)]
    if config is not None:command+=['--config',str(config),'--config-sha256',sha(config),'--collector-sha256',sha(collector)]
    receipt=dict(kind='R3A-detached-launcher-receipt.v1',launch_id=launch_id,at_utc=utc(),mode=selected_mode,
        launcher_pid=os.getpid(),launcher_ppid=os.getppid(),launcher_job_state=job(),
        creation_flags=flags,creation_flags_hex=hex(flags),
        flags=['DETACHED_PROCESS','CREATE_NEW_PROCESS_GROUP','CREATE_BREAKAWAY_FROM_JOB'],
        close_fds=True,stdin='DEVNULL',shell=False,hidden_window=True,
        run_dir=str(directory),command=command,launcher_source_sha256=sha(__file__),worker_source_sha256=sha(worker),
        actual_python_process_executable=str(base_python),venv_python_identity=sys.executable,
        venv_startup='Direct base executable with __PYVENV_LAUNCHER__; preserves existing venv prefix/site without the Job-creating venv redirector',
        stdout_log=str(directory/'worker.stdout.log'),stderr_log=str(directory/'worker.stderr.log'),
        GPU_training_started=False,automatic_callback=False,OS_reboot_resume_implemented=False)
    write(directory/'launch-attempt.json',receipt)
    child=None
    try:
        info=subprocess.STARTUPINFO();info.dwFlags|=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
        env=os.environ.copy();env['__PYVENV_LAUNCHER__']=sys.executable
        with (directory/'worker.stdout.log').open('ab',buffering=0)as output,(directory/'worker.stderr.log').open('ab',buffering=0)as error:
            child=subprocess.Popen(command,executable=str(base_python),cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=error,
                close_fds=True,shell=False,creationflags=flags,startupinfo=info,env=env)
        receipt['worker_pid']=child.pid
        deadline=time.monotonic()+15.
        while not(directory/'worker-startup-proof.json').exists():
            if child.poll()is not None:raise RuntimeError(f'Worker exited before startup proof: {child.returncode}')
            if time.monotonic()>=deadline:raise RuntimeError('No independent startup proof; no release granted')
            time.sleep(.05)
        proof=read(directory/'worker-startup-proof.json')
        if proof.get('passed')is not True or proof.get('is_process_in_job')is not False or proof.get('worker_pid')!=child.pid:
            raise RuntimeError('Worker failed its independent Windows Job/source proof; no collector release')
        live=job(child.pid)
        if live['is_process_in_job']or not live['process_active']:raise RuntimeError('Live worker is not independent and running')
        receipt.update(launch_succeeded=True,independent_job_verified=True,live_worker_job_query=live,
            ready_at_utc=utc(),startup_proof_file=str(directory/'worker-startup-proof.json'),
            limitation='Survives launcher exit and Windows Job teardown if no external process explicitly kills it; no reboot/sleep continuation.')
        write(directory/'launcher-receipt.json',receipt)
        write(directory/'launcher-release.json',dict(at_utc=utc(),launch_id=launch_id,worker_pid=child.pid,mode=selected_mode))
        print(json.dumps(receipt,ensure_ascii=False));return 0
    except Exception as exc:
        receipt.update(launch_succeeded=False,error=f'{type(exc).__name__}: {exc}',
            worker_pid=None if child is None else child.pid,release_granted=False,
            no_job_bound_fallback_attempted=True)
        write(directory/'launcher-receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
