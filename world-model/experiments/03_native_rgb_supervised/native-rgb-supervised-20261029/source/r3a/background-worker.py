"""One independent Windows worker; collect once or run a zero-physics smoke test.

No restart on OS reboot, model work, scheduler, network or automatic chat callback.
The actual collector owns its existing claims, accounting and resume verification.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[3]
REQUIRED_FLAGS=0x00000008|0x00000200|0x01000000


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8')as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    deadline=time.monotonic()+2.
    while True:
        try:os.replace(temporary,path);return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.025)


def process_job_state(pid=None):
    """Ask Windows about actual job membership; unknown is an error, never false."""
    if os.name!='nt':raise RuntimeError('This launcher requires Windows')
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=wintypes.HANDLE
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];api.IsProcessInJob.restype=wintypes.BOOL
    api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];api.GetExitCodeProcess.restype=wintypes.BOOL
    api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
    handle=api.GetCurrentProcess()if pid is None else api.OpenProcess(0x1000,False,int(pid))
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        inside=wintypes.BOOL();code=wintypes.DWORD()
        if not api.IsProcessInJob(handle,None,ctypes.byref(inside)):raise ctypes.WinError(ctypes.get_last_error())
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=os.getpid()if pid is None else int(pid),is_process_in_job=bool(inside.value),
            process_active=code.value==259,exit_code=None if code.value==259 else int(code.value))
    finally:
        if pid is not None:api.CloseHandle(handle)


def python_process_image():
    """Actual PE image, unlike sys.executable/_base_executable venv aliases."""
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    api.GetModuleFileNameW.restype=wintypes.DWORD
    buffer=ctypes.create_unicode_buffer(32768)
    length=api.GetModuleFileNameW(None,buffer,len(buffer))
    if not length or length>=len(buffer):raise ctypes.WinError(ctypes.get_last_error())
    return str(Path(buffer.value).resolve())


def collection_snapshot(config):
    if config is None:return dict(physics_controls=0,collector_invoked=False)
    base=Path(config['output']);value={}
    for filename in('resource-ledger.json','index.json','manifest.json'):
        try:
            record=read(base/filename)
            if filename=='resource-ledger.json':value['ledger']={k:record.get(k)for k in(
                'successful_scenes','attempted_scenes','failed_scenes','charged_controls','completed_controls',
                'engine_step_calls','engine_unknown_advance_substeps','reserved_not_started_controls','pending_control','closed')}
            elif filename=='index.json':value['index']={k:record.get(k)for k in('status','collection_closed','gate_passed')}
            else:value['manifest']=dict(successful_entries=len(record['entries']),collection_closed=record['collection_closed'])
        except Exception as exc:value[filename+'_read_error']=f'{type(exc).__name__}: {exc}'
    return value


def run(args):
    directory=Path(args.run_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    if not directory.is_relative_to(ROOT/'artifacts/native-rgb-supervised-20261029/r3a'):
        raise ValueError('Worker status/log directory must stay inside R3A artifacts')
    status_path=directory/'status.json';started=time.monotonic();config=None;child=None;counter=0
    proof=dict(kind='R3A-detached-worker-startup-proof.v1',at_utc=utc(),worker_pid=os.getpid(),
        worker_ppid=os.getppid(),launcher_pid=args.launcher_pid,launch_id=args.launch_id,mode=args.mode,
        creation_flags_requested=args.creation_flags,creation_flags_hex=hex(args.creation_flags),
        worker_source_sha256=sha(__file__),worker_file=str(Path(__file__).resolve()),
        python_executable=sys.executable,model_calls=0,checkpoint_loads=0,automatic_callback=False,
        actual_python_process_executable=python_process_image(),
        OS_reboot_resume_implemented=False)
    try:
        proof.update(process_job_state())
        if args.creation_flags&REQUIRED_FLAGS!=REQUIRED_FLAGS:raise RuntimeError('Required detached/newgroup/breakaway flags absent')
        if proof['is_process_in_job']:raise RuntimeError('Worker is still inside a Windows Job; refuse collection')
        if sha(__file__)!=args.worker_sha256:raise ValueError('Worker source SHA changed between launcher and child')
        if Path(python_process_image()).resolve()!=Path(args.base_python).resolve():raise ValueError('Underlying Python process executable differs')
        if args.mode=='collect':
            if not args.config or not args.config_sha256 or not args.collector_sha256:raise ValueError('Real collection requires explicit frozen config and v3 source identities')
            if sha(args.config)!=args.config_sha256:raise ValueError('Config changed after launch')
            config=read(args.config)
            if (config.get('execution_authorized')is not True or config.get('training_authorized',False)is not False
                    or Path(config['output']).resolve()!=ROOT/'artifacts/native-rgb-supervised-20261029/r3a'):
                raise ValueError('Only authorized R3A collection, no GPU/training scope')
            collector=ROOT/'scripts/native_rgb_r3a_collect_v3.py'
            if sha(collector)!=args.collector_sha256:raise ValueError('Frozen v3 collector source changed after launch')
            proof.update(config_file=str(Path(args.config).resolve()),config_sha256=args.config_sha256,
                collector_file=str(collector),collector_sha256=args.collector_sha256)
        proof['passed']=True;atomic_json(directory/'worker-startup-proof.json',proof)
        atomic_json(status_path,dict(state='waiting_launcher_release',at_utc=utc(),worker_pid=os.getpid(),mode=args.mode))
        deadline=time.monotonic()+30.
        while not(directory/'launcher-release.json').exists():
            if time.monotonic()>=deadline:raise RuntimeError('Launcher startup handshake was not released; no collector started')
            time.sleep(.05)
        release=read(directory/'launcher-release.json')
        if release.get('launch_id')!=args.launch_id or release.get('worker_pid')!=os.getpid()or release.get('mode')!=args.mode:
            raise ValueError('Launcher release does not match this worker')
        if args.mode=='collect':
            command=[sys.executable,'-u',str(collector),'--config',str(Path(args.config).resolve()),'--continue-unfinished','--execute']
            info=subprocess.STARTUPINFO();info.dwFlags|=subprocess.STARTF_USESHOWWINDOW;info.wShowWindow=0
            env=os.environ.copy();env.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',
                __PYVENV_LAUNCHER__=sys.executable)
            with (directory/'collector.stdout.log').open('ab',buffering=0)as output,(directory/'collector.stderr.log').open('ab',buffering=0)as error:
                child=subprocess.Popen(command,executable=args.base_python,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=error,
                    close_fds=True,shell=False,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,
                    startupinfo=info,env=env)
            atomic_json(directory/'collector-start.json',dict(at_utc=utc(),command=command,collector_pid=child.pid,
                worker_pid=os.getpid(),actual_python_process_executable=args.base_python,
                job_state=process_job_state(child.pid),restart_count=0))
        running_started=time.monotonic();interval=15. if args.mode=='dummy'else 10.;next_heartbeat=running_started+interval
        def publish(final=False):
            state=dict(kind='R3A-background-heartbeat.v1',at_utc=utc(),worker_pid=os.getpid(),worker_ppid=os.getppid(),
                launcher_pid=args.launcher_pid,launch_id=args.launch_id,mode=args.mode,sequence=counter,
                elapsed_seconds=time.monotonic()-running_started,collector_pid=None if child is None else child.pid,
                collector_exit_code=None if child is None else child.poll(),worker_is_process_in_job=process_job_state()['is_process_in_job'],
                whole_worker_closed=final,local_snapshot=collection_snapshot(config))
            for path in(directory/'heartbeat.json',status_path):
                try:atomic_json(path,dict(state,state='closed'if final else'running'))
                except Exception:traceback.print_exc()
            return state
        publish()
        while True:
            now=time.monotonic()
            if now>=next_heartbeat:
                counter+=1;publish();next_heartbeat+=interval
            if args.mode=='dummy':
                if counter>=2:break
            elif child.poll()is not None:break
            time.sleep(min(.2,max(.01,next_heartbeat-time.monotonic())))
        code=0 if child is None else child.wait();final=publish(final=True)
        snapshot=final['local_snapshot']
        closed_ok=(args.mode=='dummy'or(code==0 and snapshot.get('manifest',{}).get('successful_entries')==912
            and snapshot.get('manifest',{}).get('collection_closed')is True))
        completion=dict(kind='R3A-background-worker-completion.v1',at_utc=utc(),worker_pid=os.getpid(),mode=args.mode,
            worker_exit_code=0 if closed_ok else 2,collector_exit_code=None if child is None else code,
            completed=closed_ok,heartbeat_sequence=counter,elapsed_seconds=time.monotonic()-started,
            local_snapshot=snapshot,model_calls=0,checkpoint_loads=0,GPU_training_started=False,
            callback_attempted=False,callback_note='Offline worker saves local evidence; reopen Codex to review/report.',
            OS_reboot_resume_implemented=False)
        atomic_json(directory/'completion.json',completion);print(json.dumps(completion),flush=True)
        return completion['worker_exit_code']
    except BaseException as exc:
        failure=dict(state='worker_failed',at_utc=utc(),worker_pid=os.getpid(),mode=args.mode,
            launch_id=args.launch_id,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
            collector_pid=None if child is None else child.pid,
            collector_exit_code=None if child is None else child.poll(),collector_restarted=False,
            actual_collector_may_still_be_running=child is not None and child.poll()is None)
        if not(directory/'worker-startup-proof.json').exists():
            proof.update(passed=False,error=failure['error']);atomic_json(directory/'worker-startup-proof.json',proof)
        atomic_json(status_path,failure);atomic_json(directory/'failure.json',failure)
        traceback.print_exc();return 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--launch-id',required=True)
    p.add_argument('--launcher-pid',required=True,type=int);p.add_argument('--creation-flags',required=True,type=int)
    p.add_argument('--worker-sha256',required=True);p.add_argument('--mode',required=True,choices=('dummy','collect'))
    p.add_argument('--base-python',required=True)
    p.add_argument('--config');p.add_argument('--config-sha256');p.add_argument('--collector-sha256')
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
