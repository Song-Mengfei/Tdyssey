"""Finite foreground orchestration only: worker exit, one saved-only audit, exit."""
import ctypes,sys,subprocess,traceback
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from common import read,sha,write,write_bytes,utc,output_io
def main():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    c=read(ROOT/'frozen-config.json');identity=sha(ROOT/'frozen-config.json');note=read(ROOT/'owner-notification.json')
    assert identity==note['config_sha256'] and c['run_id']==note['run_id']
    assert (ROOT/'USER_AUTHORIZATION.json').is_file() and not (ROOT/'execution-claim.json').exists()
    write(ROOT/'active-launch-receipt.json',dict(run_id=c['run_id'],config_sha256=identity,launcher_sha256=sha(__file__),hosting='one active finite exec session; no periodic monitor or scheduled task',command=[sys.executable,'-X','utf8','-B',str(ROOT/'worker.py'),'--config-sha',identity],at_utc=utc()))
    child=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'worker.py'),'--config-sha',identity],cwd=c['repo'])
    write(ROOT/'worker-exit-receipt.json',dict(run_id=c['run_id'],actual_exit_code=child.returncode,at_utc=utc()))
    if child.returncode:return child.returncode
    done=read(ROOT/'completion.json');assert done['complete'] and not done['failed']
    assert not (ROOT/'audit-claim.json').exists() and not (ROOT/'audit-result.json').exists() and not (ROOT/'saved-audit.json').exists()
    audit=ROOT/'postrun_audit.py';assert sha(audit)==note['audit_sha256']
    write(ROOT/'audit-claim.json',dict(run_id=c['run_id'],config_sha256=identity,script=str(audit),script_sha256=sha(audit),worker_actual_exit_code=child.returncode,saved_only=True,new_model_calls=0,at_utc=utc()))
    checked=subprocess.run([sys.executable,'-X','utf8','-B',str(audit)],cwd=c['repo'],capture_output=True)
    write_bytes(ROOT/'audit-stdout.txt',checked.stdout);write_bytes(ROOT/'audit-stderr.txt',checked.stderr)
    write(ROOT/'audit-result.json',dict(run_id=c['run_id'],actual_exit_code=checked.returncode,stdout=str(ROOT/'audit-stdout.txt'),stderr=str(ROOT/'audit-stderr.txt'),new_model_calls=0,at_utc=utc()))
    print('Finite worker exit='+str(child.returncode)+'; saved-only audit exit='+str(checked.returncode),flush=True)
    return checked.returncode
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        write(ROOT/'launcher-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at_utc=utc()),emergency=True);raise
