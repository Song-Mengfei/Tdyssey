"""Repair a sandbox-killed pre-start launch; never retry a physical attempt."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import worker
from common import atomic_json, read, utc

def main():
    digest='a08cdb4575710cef5d7d65e8369b4eb88ab1b1b121a7f8bc33dd3551f2eb775f'
    c=worker.validate(ROOT/'runtime-config.json',digest)
    bg=ROOT/'background'
    for p in (bg/'worker-startup.json',bg/'completion.json',ROOT/'collection/execution-claim.json',
              ROOT/'training/claim.json',bg/'detached-relaunch-receipt.json'):
        if p.exists():raise RuntimeError('Refuse repeat of an actually claimed stage: '+str(p))
    # The first launch did not reach the owner startup guard or any phase claim.
    atomic_json(bg/'initial-launch-nonstart.json',dict(run_id=c['run_id'],at_utc=utc(),
        original_launcher=read(bg/'launcher-receipt.json'),original_worker_alive=False,
        diagnostic_check_child_inherited_windows_job=True,diagnostic_physical_calls=0,
        diagnostic_model_calls=0,actual_physical_attempts=0,
        reason='Sandbox kept breakaway child in Windows Job; shell completion removed child before owner startup.'))
    cmd=[worker.detached.actual_python(),'-X','utf8','-B',str(ROOT/'worker.py'),'worker',
         '--config',str(ROOT/'runtime-config.json'),'--config-sha',digest,
         '--launcher-pid',str(os.getpid())]
    with (bg/'worker-stdout.log').open('ab') as out,(bg/'worker-stderr.log').open('ab') as err:
        child=subprocess.Popen(cmd,cwd=worker.REPO,env=worker.env(c,'owner'),stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,close_fds=True,startupinfo=worker.detached.hidden_info(),
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    in_job=worker.detached.in_job(int(child._handle))
    receipt=dict(run_id=c['run_id'],config_sha256=digest,launcher_pid=os.getpid(),worker_pid=child.pid,
        worker_is_process_in_job=in_job,hidden=True,command=cmd,at_utc=utc(),
        scope='Launch repair before any physical attempt; all frozen sources/config unchanged.')
    atomic_json(bg/'detached-relaunch-receipt.json',receipt,exclusive=True)
    if in_job:
        child.terminate();child.wait(timeout=10)
        raise RuntimeError('No detached owner possible; child stopped before stage execution')
    print(receipt)

if __name__=='__main__':main()
