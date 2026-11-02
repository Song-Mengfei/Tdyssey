"""Detached owner for one finite collection and saved-only report. No next stage."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v,exclusive=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p if exclusive else p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('x' if exclusive else 'w',encoding='utf8') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if not exclusive:temp.replace(p)
def actual_python():
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];k.GetModuleFileNameW.restype=wintypes.DWORD
    buf=ctypes.create_unicode_buffer(32768)
    if not k.GetModuleFileNameW(None,buf,len(buf)):raise ctypes.WinError(ctypes.get_last_error())
    return buf.value
def in_job(handle=None):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)];k.IsProcessInJob.restype=wintypes.BOOL
    result=wintypes.BOOL()
    if not k.IsProcessInJob(handle or k.GetCurrentProcess(),None,ctypes.byref(result)):raise ctypes.WinError(ctypes.get_last_error())
    return bool(result.value)
def wait_launcher_exit(pid):
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(0x100000,False,pid)
    if not h:
        if ctypes.get_last_error()==87:return dict(pid=pid,exited=True,evidence='OpenProcess ERROR_INVALID_PARAMETER')
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if k.WaitForSingleObject(h,30000)!=0:raise RuntimeError('Launcher still alive; no physical attempt started')
        return dict(pid=pid,exited=True,evidence='WaitForSingleObject WAIT_OBJECT_0')
    finally:k.CloseHandle(h)
def environment(c):
    env=os.environ.copy();env.update(__PYVENV_LAUNCHER__=c['python_executable'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',PYGAME_HIDE_SUPPORT_PROMPT='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    return env
def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen configuration identity mismatch')
    c=read(path)
    if c['kind']!='direct-target-acceptance.v1' or c['training_authorized'] or c['auto_retry'] or c['next_phase_released']:
        raise ValueError('Unapproved scope')
    if c['limits']!=dict(attempts=120,controls_per_attempt=64,controls=7680,regular_substeps=76800,new_bytes=268435456,shared_bytes=3221225472,atomic_reserve_bytes=268435456):
        raise ValueError('Unified physical/storage budget mismatch')
    if c['model_calls']!=0 or c['optimizer_updates']!=0:raise ValueError('No model/training release')
    for binding in c['bindings']:
        if sha(binding['path'])!=binding['sha256']:raise ValueError('Frozen binding changed: '+binding['path'])
    if not Path(path).resolve().is_relative_to(Path(c['artifact_root']).resolve()):raise ValueError('Configuration outside new run root')
    return c
def hidden_info():
    si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
    return si
def child_run(command,c,runtime,phase,digest):
    proc=None
    try:
        with (runtime/(phase+'-stdout.log')).open('ab') as out,(runtime/(phase+'-stderr.log')).open('ab') as err:
            proc=subprocess.Popen(command,cwd=c['repo'],env=environment(c),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                close_fds=True,startupinfo=hidden_info(),creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        child=dict(pid=proc.pid,run_id=c['run_id'],config_sha256=digest,command=command,is_process_in_job=in_job(int(proc._handle)),started_at_utc=utc())
        write(runtime/(phase+'-process.json'),child,exclusive=True)
        if child['is_process_in_job']:
            proc.terminate();proc.wait(timeout=10);raise RuntimeError('Owned child is in a Windows Job')
        while True:
            code=proc.poll()
            row=dict(run_id=c['run_id'],config_sha256=digest,state=phase if code is None else phase+'_exited',
                worker_pid=os.getpid(),child_pid=proc.pid,child_exit_code=code,updated_at_utc=utc(),model_calls=0,optimizer_updates=0,next_phase_released=False)
            write(runtime/'status.json',row);write(runtime/'heartbeat.json',row)
            if code is not None:return code
            time.sleep(10)
    except BaseException:
        if proc is not None and proc.poll() is None:
            if phase=='collecting':
                write(c['stop_file'],dict(run_id=c['run_id'],reason='Owner failure; save owned child partial',at_utc=utc()))
                # Only this frozen child is stopped; keep owner alive until durable partial closes.
                while proc.poll() is None:time.sleep(2)
            else:
                proc.terminate();proc.wait(timeout=20)
        raise
def worker(a,c,runtime):
    start=time.perf_counter();failure=None;collection=None;report_exit=None
    startup=dict(run_id=c['run_id'],config_sha256=a.config_sha,pid=os.getpid(),actual_interpreter=actual_python(),
        is_process_in_job=in_job(),started_at_utc=utc(),model_calls=0,optimizer_updates=0)
    try:
        startup['launcher_exit']=wait_launcher_exit(a.launcher_pid)
        write(runtime/'worker-startup.json',startup,exclusive=True)
        if startup['is_process_in_job']:raise RuntimeError('Background owner must actually leave Windows Job')
        validate(a.config,a.config_sha)
        command=[actual_python(),'-X','utf8','-B',str(Path(c['repo'])/'scripts/direct_target_collect.py'),
                 '--config',str(Path(a.config).resolve()),'--config-sha',a.config_sha]
        collect_exit=child_run(command,c,runtime,'collecting',a.config_sha)
        path=Path(c['artifact_root'])/'collection/completion.json'
        if path.exists():
            collection=read(path)
            if collection['run_id']!=c['run_id'] or collection['config_sha256']!=a.config_sha:
                raise RuntimeError('Actual collector completion identity mismatch')
        if collect_exit!=0 or collection is None:
            failure='Collector failed, exit='+str(collect_exit)
        # Reporting consumes only saved truth, including partial/failed attempts.
        validate(a.config,a.config_sha)
        report_exit=child_run([actual_python(),'-X','utf8','-B',str(Path(c['repo'])/'scripts/direct_target_report.py'),
                               '--root',c['artifact_root']],c,runtime,'reporting_saved_only',a.config_sha)
        if report_exit!=0:failure=(failure+'; ' if failure else '')+'Saved-only report failed, exit='+str(report_exit)
    except BaseException as exc:
        failure=repr(exc);write(runtime/'worker-error.json',dict(error=failure,traceback=traceback.format_exc(),at_utc=utc()))
    terminal=dict(run_id=c['run_id'],config_sha256=a.config_sha,state='failed_waiting_review' if failure else 'waiting_callback',
        failed=failure is not None,failure=failure,collection=collection,report_exit_code=report_exit,
        whole_phase_wall_seconds=time.perf_counter()-start,finished_at_utc=utc(),model_calls=0,optimizer_updates=0,
        training_authorized=False,next_phase_released=False,automatic_retry=False,callback_sent=False,
        report=str(Path(c['artifact_root'])/'REPORT.md'),summary=str(Path(c['artifact_root'])/'report/summary.json'))
    write(runtime/'completion.json',terminal);write(runtime/'status.json',terminal)
    write(Path(c['artifact_root'])/'PHASE_STATE.json',dict(**terminal,status='waiting_callback',parent_thread_id=c['parent_thread_id']))
    return 1 if failure else 0
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('check','launch','worker'))
    p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--launcher-pid',type=int)
    a=p.parse_args();c=validate(a.config,a.config_sha);runtime=Path(c['artifact_root'])/'background';runtime.mkdir(parents=True,exist_ok=True)
    if a.mode=='check':print(json.dumps(dict(passed=True,physics_calls=0,model_calls=0,run_id=c['run_id'])));return 0
    if a.mode=='worker':return worker(a,c,runtime)
    for name in ('launch-claim.json','worker-startup.json','completion.json'):
        if (runtime/name).exists():raise FileExistsError('No automatic restart of '+name)
    write(runtime/'launch-claim.json',dict(run_id=c['run_id'],pid=os.getpid(),at_utc=utc()),exclusive=True)
    command=[actual_python(),'-X','utf8','-B',str(Path(__file__).resolve()),'worker','--config',str(Path(a.config).resolve()),
             '--config-sha',a.config_sha,'--launcher-pid',str(os.getpid())]
    with (runtime/'worker-stdout.log').open('ab') as out,(runtime/'worker-stderr.log').open('ab') as err:
        child=subprocess.Popen(command,cwd=c['repo'],env=environment(c),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,startupinfo=hidden_info(),creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    write(runtime/'launcher-receipt.json',dict(run_id=c['run_id'],config_sha256=a.config_sha,launcher_pid=os.getpid(),worker_pid=child.pid,
        launcher_is_process_in_job=in_job(),command=command,hidden=True,at_utc=utc()),exclusive=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
