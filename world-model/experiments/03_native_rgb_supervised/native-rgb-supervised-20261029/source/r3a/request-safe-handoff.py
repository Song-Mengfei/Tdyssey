"""Request a terminal-independent handoff without killing an active engine call.

The frozen v2 has no cooperative stop flag. A short deny-delete read lease makes
its robust atomic ledger writer raise at a durable write boundary. The lease is
released immediately when the existing exception handler saves failure evidence,
so its final ledger write can complete. The old process must be observed exited
and its exact state cold-audited before any replacement collector is launched.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

stage=Path(__file__).resolve().parent;repo=stage.parents[2];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json


def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    if sys.platform!='win32':raise RuntimeError('Windows-only audited file-sharing handoff')
    out=stage/'background';out.mkdir(exist_ok=True)
    receipt=out/'handoff-request.json'
    if receipt.exists():raise FileExistsError('Do not repeat a handoff lease blindly')
    deadline=time.monotonic()+30;pending=None
    while time.monotonic()<deadline:
        ledger=read(stage/'resource-ledger.json');pending=ledger['pending_control']
        if ledger['closed']:raise RuntimeError('Collector already closed; inspect instead of inducing a stop')
        if pending and 0<=pending['control_index']<=40 and pending['trajectory_id'].endswith('-a0'):break
        time.sleep(.01)
    else:raise RuntimeError('No safe early-primary-attempt control boundary observed; original run remains active')
    target=stage/'resource-ledger.json';halt=stage/'halt.json'
    old_halt_mtime=halt.stat().st_mtime_ns if halt.exists() else None
    meta=stage/'trajectories'/(pending['trajectory_id']+'.json')
    old_meta_mtime=meta.stat().st_mtime_ns if meta.exists() else None
    value=dict(kind='R3A-user-requested-safe-background-handoff.v1',requested_at_utc=utc(),
        authorization='Human explicitly requested continuing collection after stopping this chat and closing Codex.',
        previous_owner_pid=read(stage/'remaining-once-claim.json')['pid'],
        observed_early_primary_control=pending,charged_controls_before_lease=ledger['charged_controls'],
        method='Deny-delete read lease on ledger target, release at saved PermissionError metadata/halt; no process kill or engine injection.',
        no_source_or_physics_change=True,physical_steps_by_this_helper=0)
    atomic_json(receipt,value)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(target),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    observed=None;began=time.monotonic()
    try:
        while time.monotonic()-began<10:
            if halt.exists()and halt.stat().st_mtime_ns!=old_halt_mtime:
                item=read(halt)
                if 'PermissionError'in item.get('traceback',''):
                    observed=dict(file='halt.json',kind='outer durable-write exception');break
            if meta.exists()and meta.stat().st_mtime_ns!=old_meta_mtime:
                item=read(meta);failure=item.get('failure',{})
                if failure.get('type')=='PermissionError':
                    observed=dict(file=meta.relative_to(stage).as_posix(),kind='saved failed-attempt write-boundary exception');break
            time.sleep(.01)
    finally:
        kernel.CloseHandle(handle)
        value.update(lease_released_at_utc=utc(),lease_seconds=time.monotonic()-began,observed_saved_exception=observed)
        atomic_json(receipt,value)
    if not observed:raise RuntimeError('Lease released without a proved handled stop; do not launch another collector')
    print(json.dumps(dict(lease_released=True,seconds=value['lease_seconds'],observed=observed,
        previous_owner_pid=value['previous_owner_pid'],next='Observe old process exit and cold audit, then explicitly freeze/activate a continuation receipt.')))


if __name__=='__main__':main()
