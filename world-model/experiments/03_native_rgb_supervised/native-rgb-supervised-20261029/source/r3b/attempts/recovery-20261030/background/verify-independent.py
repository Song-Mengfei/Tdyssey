"""Read-only observation of the actual running pipeline after launcher exit."""
import json
from pathlib import Path
import runpy
import time

BG=Path(__file__).resolve().parent
A=BG.parent
BASE=A.parents[2]
api=runpy.run_path(str(BASE/'r3a/background-worker.py'))
read,write,job=(api[k]for k in('read','atomic_json','process_job_state'))
receipt=read(BG/'launcher-receipt.json')
post=read(A/'diagnostics/post-model-loader/verification.json')
assert post['passed']and len(post['verified_worker_pids'])==2
def snapshot():
    value=dict(at_utc=api['utc'](),status=read(BG/'status.json'),heartbeat=read(BG/'heartbeat.json'),
        worker=job(receipt['worker_pid']),loaders=[job(p)for p in post['verified_worker_pids']])
    try:value['launcher']=job(receipt['launcher_pid']);value['launcher_exited']=not value['launcher']['process_active']
    except OSError as exc:
        value['launcher']=dict(pid=receipt['launcher_pid'],winerror=exc.winerror)
        value['launcher_exited']=exc.winerror==87
    return value
first=snapshot();write(BG/'independence-snapshot1.json',first)
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    time.sleep(1);current=read(BG/'status.json')
    if current['closed']or(current['progress_updates']>first['status']['progress_updates']and current['heartbeat_sequence']>first['heartbeat']['heartbeat_sequence']):break
second=snapshot();write(BG/'independence-snapshot2.json',second)
active=[second['worker'],*second['loaders']]
progress=second['status']['progress_updates']>first['status']['progress_updates']
heartbeat=second['heartbeat']['heartbeat_sequence']>first['heartbeat']['heartbeat_sequence']
probe=read(A/'run/engineering-probe.json')
proof=dict(kind='R3B-recovery-observed-real-independent-pipeline.v1',run_id=receipt['run_id'],
    config_sha256=read(BG/'startup-proof.json')['config_sha256'],at_utc=api['utc'](),
    worker_pid=receipt['worker_pid'],loader_pids=post['verified_worker_pids'],launcher_pid=receipt['launcher_pid'],
    launcher_exited=second['launcher_exited'],all_real_processes_job_false=all(not p['is_process_in_job']for p in active),
    all_real_processes_active=all(p['process_active']for p in active),progress_advanced=progress,heartbeat_advanced=heartbeat,
    phase_before=first['status']['phase'],phase_after=second['status']['phase'],
    progress_updates_before=first['status']['progress_updates'],progress_updates_after=second['status']['progress_updates'],
    heartbeat_before=first['heartbeat']['heartbeat_sequence'],heartbeat_after=second['heartbeat']['heartbeat_sequence'],
    restored_GPU_two_loader_validation_passed=post['passed'],actual_GPU_engineering_probe_passed=probe['passed'],
    engineering_attempts=probe['engineering_attempts'],engineering_successes=probe['engineering_successes'],
    engineering_draws=probe['engineering_draws'],formal=second['status']['formal'],
    snapshot_paths=[str(BG/'independence-snapshot1.json'),str(BG/'independence-snapshot2.json')])
proof['passed']=first['launcher_exited']and proof['launcher_exited']and proof['all_real_processes_job_false']and proof['all_real_processes_active']and progress and heartbeat and probe['passed']and not second['status']['closed']
write(BG/'independence-proof.json',proof)
print(json.dumps(proof,ensure_ascii=False))
raise SystemExit(0 if proof['passed']else 2)
