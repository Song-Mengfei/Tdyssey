"""The already-proven service-supervised lifecycle, for exactly epochs2/3."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT),str(REPO),str(REPO/'scripts')]
from common import atomic_json,check_storage,read,sha,utc
import direct_target_background as detached

def service_lifecycle(c):
    if c.get('worker_lifecycle')!='windows-task-scheduler' or os.environ.get('TDYSSEY_SERVICE_RUN_ID')!=c['run_id']:
        return None
    proof=read(ROOT/'background/service-entry.json')
    if proof.get('run_id')!=c['run_id'] or proof.get('task_name')!=c['windows_task_name']:
        raise RuntimeError('Service launcher identity mismatch')
    import ctypes
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.OpenProcess(0x100000,False,int(proof['pid']))
    if not h:raise RuntimeError('Service supervisor is not alive')
    try:
        if k.WaitForSingleObject(h,0)!=0x102:raise RuntimeError('Service supervisor exited')
    finally:k.CloseHandle(h)
    return dict(task_name=proof['task_name'],supervisor_pid=proof['pid'],supervisor_alive=True,
        mechanism='Windows Task Scheduler service supervises this run; actual Job membership retained.')

def validate(path,digest):
    if sha(path)!=digest:raise ValueError('Frozen runtime config SHA mismatch')
    c=read(path)
    if c['kind']!='oc-two-epoch-continuation.v1' or c['epochs']!=[2,3]:raise ValueError('Approved epoch scope changed')
    if c['limits']!=dict(new_physical_attempts=0,integrator_calls=0,additional_epochs=2,samples_per_epoch=30935,additional_draws=61870,optimizer_batches=3868):
        raise ValueError('Approved exposure/zero-physics scope changed')
    if Path(c['artifact_root']).resolve()!=ROOT or c['shared_cap_bytes']!=10_000_000_000:
        raise ValueError('Run root/shared cap changed')
    if c['starting_checkpoint_sha256']!='492e9badaa3e622254c829e1ea44cad6ad8c4c4636feb15c7764ec07e3745f11':
        raise ValueError('Wrong epoch1 checkpoint')
    for b in c['bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('Frozen binding changed: '+b['path'])
    return c

def environment(c,phase):
    e=os.environ.copy()
    e.update(__PYVENV_LAUNCHER__=c['python_executable'],PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYGAME_HIDE_SUPPORT_PROMPT='1',SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if phase=='reporting':e['CUDA_VISIBLE_DEVICES']=''
    return e

def status(c,digest,phase,**extra):
    s=dict(run_id=c['run_id'],config_sha256=digest,phase=phase,worker_pid=os.getpid(),updated_at_utc=utc(),**extra)
    atomic_json(ROOT/'background/status.json',s);atomic_json(ROOT/'background/heartbeat.json',s)
    atomic_json(ROOT/'PHASE_STATE.json',dict(**s,status=phase,parent_thread_id=c['parent_thread_id'],
        owner_memory=str(ROOT/'OWNER_MEMORY.txt'),callback_authorized=True,callback_requires_running_app=True,
        authorized_epochs=[2,3],new_physics=0,auto_next_epoch=False))

def child_phase(c,a,name):
    bg=ROOT/'background'
    cmd=[detached.actual_python(),'-X','utf8','-B',str(Path(__file__)),name,'--config',str(Path(a.config).resolve()),'--config-sha',a.config_sha]
    with (bg/(name+'-stdout.log')).open('ab') as out,(bg/(name+'-stderr.log')).open('ab') as err:
        p=subprocess.Popen(cmd,cwd=REPO,env=environment(c,name),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
            close_fds=True,startupinfo=detached.hidden_info(),creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    proof=dict(run_id=c['run_id'],config_sha256=a.config_sha,pid=p.pid,command=cmd,
        actual_is_process_in_job=detached.in_job(int(p._handle)),service_lifecycle=service_lifecycle(c),at_utc=utc())
    atomic_json(bg/(name+'-process.json'),proof,exclusive=True)
    if proof['actual_is_process_in_job'] and not proof['service_lifecycle']:
        p.terminate();p.wait(timeout=15);raise RuntimeError('No independent service supervisor')
    try:
        while True:
            code=p.poll();status(c,a.config_sha,name,child_pid=p.pid,child_exit_code=code)
            if code is not None:
                if code:raise RuntimeError(f'{name} exit{code}; inspect its stderr/failure evidence')
                return read(bg/(name+'-result.json'))
            time.sleep(15)
    except BaseException:
        if p.poll() is None:p.terminate();p.wait(timeout=20)
        raise

def run(c,a):
    began=time.perf_counter();results={};failure=None
    try:
        proof=dict(run_id=c['run_id'],config_sha256=a.config_sha,pid=os.getpid(),actual_interpreter=detached.actual_python(),
            actual_is_process_in_job=detached.in_job(),launcher_exit=detached.wait_launcher_exit(a.launcher_pid),
            service_lifecycle=service_lifecycle(c),started_at_utc=utc())
        atomic_json(ROOT/'background/worker-startup.json',proof,exclusive=True)
        if proof['actual_is_process_in_job'] and not proof['service_lifecycle']:raise RuntimeError('Service ownership proof missing')
        for name in ('training','reporting'):
            validate(a.config,a.config_sha);results[name]=child_phase(c,a,name)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic_json(ROOT/'background/failure.json',failure)
    result=dict(run_id=c['run_id'],config_sha256=a.config_sha,completed=failure is None,failed=failure is not None,
        failure=failure,results=results,new_physical_attempts=0,integrator_calls=0,wall_seconds=time.perf_counter()-began,
        finished_at_utc=utc(),state='waiting_callback',callback_sent=False,auto_retry=False,auto_next_epoch=False)
    atomic_json(ROOT/'background/completion.json',result);status(c,a.config_sha,'waiting_callback',failed=failure is not None)
    if failure:
        (ROOT/'FAILED.md').write_text('# O/C continuation stopped\n\n'+failure['message']+'\n\nSee background/failure.json and training partial evidence. No automatic retry or new method.\n',encoding='utf-8')
    return int(failure is not None)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('check','worker','training','reporting'))
    p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--launcher-pid',type=int)
    a=p.parse_args();c=validate(a.config,a.config_sha);(ROOT/'background').mkdir(exist_ok=True)
    if a.mode=='check':print(dict(passed=True,model_calls=0,physics_calls=0));return 0
    if a.mode=='worker':return run(c,a)
    check_storage(ROOT,c,full=True)
    if a.mode=='training':
        from engine import run as train
        result=train(ROOT,c)
    else:
        from report import generate_report
        result=generate_report(ROOT,c,read(ROOT/'background/training-result.json'))
    atomic_json(ROOT/'background'/(a.mode+'-result.json'),result)
    atomic_json(ROOT/'background'/(a.mode+'-storage.json'),check_storage(ROOT,c,full=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
