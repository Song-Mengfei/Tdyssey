"""Limits affect this process only; no global environment or other process edits."""
import os
import ctypes
from ctypes import wintypes
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')

def constrain_cpu():
    for key in THREAD_ENV:os.environ[key]='1'
    os.environ['OMP_WAIT_POLICY']='PASSIVE'
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.GetCurrentProcess.restype=wintypes.HANDLE
    k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL
    k.GetPriorityClass.argtypes=[wintypes.HANDLE];k.GetPriorityClass.restype=wintypes.DWORD
    handle=k.GetCurrentProcess()
    if not k.SetPriorityClass(handle,0x4000):raise ctypes.WinError(ctypes.get_last_error())
    priority=int(k.GetPriorityClass(handle))
    if priority!=0x4000:raise RuntimeError('Own BelowNormal priority was not applied')
    import cv2
    cv2.setNumThreads(1)
    if cv2.getNumThreads()!=1:raise RuntimeError('OpenCVCPUthreadlimitdiffers')
    return dict(pid=os.getpid(),priority_class=priority,priority='BelowNormal',opencv_threads=cv2.getNumThreads(),thread_environment={key:os.environ[key]for key in THREAD_ENV},
        other_process_mutations=0,system_environment_mutations=0,cpu_model_fallback=False)

def torch_limits():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:raise RuntimeError('TorchCPUthread limits differ')
    return dict(torch_intra_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())

