"""CPU-only observability for the Windows R3B data-process startup boundary.

Only this diagnostic process installs hooks. Production renderer/data behavior is
called unchanged. No Torch or CUDA imports occur in this module.
"""
from __future__ import annotations
import atexit
import ctypes
from ctypes import wintypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

CHILDREN=[]
HANDLES=[]
HOOK_RESTORE=[]
EVENT_LOCK=threading.RLock()

def utc():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def actual_pe():
    if os.name!='nt':return str(Path(sys.executable).resolve())
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD]
    api.GetModuleFileNameW.restype=wintypes.DWORD
    value=ctypes.create_unicode_buffer(32768)
    if not api.GetModuleFileNameW(None,value,len(value)):raise ctypes.WinError(ctypes.get_last_error())
    return str(Path(value.value).resolve())

def job():
    if os.name!='nt':return dict(supported=False,is_in_job=None)
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=wintypes.HANDLE
    api.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
    api.IsProcessInJob.restype=wintypes.BOOL
    value=wintypes.BOOL()
    if not api.IsProcessInJob(api.GetCurrentProcess(),None,ctypes.byref(value)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(supported=True,is_in_job=bool(value.value))

def identity():
    return dict(pid=os.getpid(),ppid=os.getppid(),actual_process_executable=actual_pe(),
        executable=sys.executable,base_executable=getattr(sys,'_base_executable',None),
        prefix=sys.prefix,base_prefix=sys.base_prefix,cwd=str(Path.cwd().resolve()),
        sys_path=list(sys.path),job=job(),torch_imported='torch'in sys.modules,
        python_configuration={k:os.environ.get(k)for k in('__PYVENV_LAUNCHER__','PYTHONHOME','PYTHONPATH','PYTHONEXECUTABLE')})

def output():
    value=os.environ.get('TDYSSEY_LOADER_DIAGNOSTIC_DIR')
    if not value:raise RuntimeError('Explicit diagnostic output required')
    return Path(value)

def event(step,**value):
    with EVENT_LOCK:
        root=output();root.mkdir(parents=True,exist_ok=True)
        with(root/f'bootstrap-{os.getpid()}.jsonl').open('a',encoding='utf-8')as stream:
            stream.write(json.dumps(dict(at_utc=utc(),pid=os.getpid(),step=step,**value),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()

def install_early_child_logging():
    """Called by a diagnostic-only sitecustomize, before spawn unpickling."""
    if str(os.getpid())==os.environ.get('TDYSSEY_LOADER_DIAGNOSTIC_PARENT_PID'):return
    root=output();root.mkdir(parents=True,exist_ok=True)
    stderr=(root/f'child-{os.getpid()}.stderr.log').open('a',encoding='utf-8',buffering=1)
    stdout=(root/f'child-{os.getpid()}.stdout.log').open('a',encoding='utf-8',buffering=1)
    HANDLES.extend([stderr,stdout]);sys.stderr=stderr;sys.stdout=stdout
    try:faulthandler.enable(file=stderr,all_threads=True)
    except (OSError,RuntimeError):pass
    original=sys.excepthook
    def failed(kind,value,tb):
        event('unhandled_python_exception',exception_type=kind.__name__,reason=str(value),traceback=''.join(traceback.format_exception(kind,value,tb)))
        original(kind,value,tb)
    sys.excepthook=failed
    event('sitecustomize_entered',identity=identity())
    atexit.register(lambda:event('python_atexit'))

def install_site_hook(root):
    root=Path(root).resolve();directory=root/'site-hook';directory.mkdir(exist_ok=True)
    helper=Path(__file__).resolve()
    source=('import importlib.util\n'
        f'_td_spec=importlib.util.spec_from_file_location("tdyssey_loader_early",{str(helper)!r})\n'
        '_td_module=importlib.util.module_from_spec(_td_spec)\n'
        '_td_spec.loader.exec_module(_td_module)\n'
        '_td_module.install_early_child_logging()\n')
    (directory/'sitecustomize.py').write_text(source,encoding='utf-8')
    prior=os.environ.get('PYTHONPATH','')
    os.environ['PYTHONPATH']=str(directory)+os.pathsep+str(helper.parents[1])+(os.pathsep+prior if prior else'')
    return dict(path=str(directory/'sitecustomize.py'),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        role='Diagnostic logging only; no executable selection/data/renderer changes')

def install_spawn_observation():
    """Observe default spawn arguments and exitcodes without replacing them."""
    import multiprocessing.process
    original_start=multiprocessing.process.BaseProcess.start
    def start(self,*args,**kwargs):
        event('process_start_requested',name=self.name)
        try:result=original_start(self,*args,**kwargs)
        except BaseException as exc:
            event('process_start_exception',exception_type=type(exc).__name__,reason=str(exc));raise
        CHILDREN.append(self);event('process_start_returned',name=self.name,child_pid=self.pid)
        return result
    multiprocessing.process.BaseProcess.start=start
    HOOK_RESTORE.append(lambda:setattr(multiprocessing.process.BaseProcess,'start',original_start))
    original_terminate=multiprocessing.process.BaseProcess.terminate
    def terminate(self,*args,**kwargs):
        event('process_termination_requested',child_pid=self.pid,name=self.name)
        return original_terminate(self,*args,**kwargs)
    multiprocessing.process.BaseProcess.terminate=terminate
    HOOK_RESTORE.append(lambda:setattr(multiprocessing.process.BaseProcess,'terminate',original_terminate))
    if os.name=='nt':
        import _winapi
        original_create=_winapi.CreateProcess
        def create(application,command,*args):
            env=args[4] if len(args)>4 else None
            event('CreateProcess_requested',application=application,command=command,inherit_handles=args[2],
                creation_flags=args[3],cwd=args[5],
                explicit_python_environment=None if env is None else{k:env.get(k)for k in('__PYVENV_LAUNCHER__','PYTHONHOME','PYTHONPATH')})
            try:result=original_create(application,command,*args)
            except BaseException as exc:
                event('CreateProcess_exception',exception_type=type(exc).__name__,reason=str(exc));raise
            event('CreateProcess_returned',child_pid=result[2],child_tid=result[3]);return result
        _winapi.CreateProcess=create
        HOOK_RESTORE.append(lambda:setattr(_winapi,'CreateProcess',original_create))

def child_exitcodes():
    rows=[]
    for child in CHILDREN:
        try:rows.append(dict(pid=child.pid,name=child.name,exitcode=child.exitcode,alive=child.is_alive()))
        except (ValueError,AssertionError)as exc:rows.append(dict(pid=child.pid,name=child.name,error=repr(exc)))
    return rows

def diagnostic_initializer(kwargs):
    event('data_initializer_entered',identity=identity())
    try:
        from . import native_rgb_r3a_data as data
        event('data_module_imported',module_file=data.__file__)
        data._initialize_worker(kwargs)
        event('data_initializer_complete',torch_imported='torch'in sys.modules)
    except BaseException as exc:
        event('data_initializer_failed',exception_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc());raise

def diagnostic_batch(indices):
    event('batch_entered',indices=list(indices))
    try:
        from . import native_rgb_r3a_data as data
        result=data._worker_batch(indices)
        result['worker_identity']=identity()
        event('batch_complete',indices=list(indices),statistics=result['worker_statistics'])
        return result
    except BaseException as exc:
        event('batch_failed',exception_type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc());raise

def bootstrap_initializer():
    event('bootstrap_initializer_entered',identity=identity())
    (output()/f'bootstrap-ready-{os.getpid()}.json').write_text('{}',encoding='utf-8')
    deadline=time.monotonic()+30
    while len(list(output().glob('bootstrap-ready-*.json')))<2:
        if time.monotonic()>deadline:raise TimeoutError('Two bootstrap workers did not initialize')
        time.sleep(.05)
    event('bootstrap_initializer_complete')

def bootstrap_task(index):
    event('bootstrap_task',index=index);time.sleep(.2)
    return dict(index=index,identity=identity())

def parent_has_exited(pid):
    if not pid:return False
    if os.name!='nt':raise RuntimeError('Windows diagnostic only')
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];api.GetExitCodeProcess.restype=wintypes.BOOL
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.OpenProcess(0x1000,False,int(pid))
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return True
        raise ctypes.WinError(error)
    try:
        code=wintypes.DWORD()
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value!=259
    finally:api.CloseHandle(handle)
