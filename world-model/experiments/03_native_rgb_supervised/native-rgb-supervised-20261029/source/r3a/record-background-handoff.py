"""Persist verified independent progress and the user-requested offline handoff."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import runpy
import sys

stage=Path(__file__).resolve().parent;repo=stage.parents[2];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    directory=stage/'background/run';launch=read(directory/'launcher-receipt.json')
    proof=read(directory/'worker-startup-proof.json');child=read(directory/'collector-start.json');heartbeat=read(directory/'heartbeat.json')
    if not launch['launch_succeeded']or not proof['passed']or proof['is_process_in_job']or child['job_state']['is_process_in_job']:
        raise ValueError('Actual worker and collector must both be independent')
    if heartbeat['sequence']<1 or heartbeat['worker_is_process_in_job']or heartbeat['local_snapshot']['manifest']['successful_entries']<=587:
        raise ValueError('Actual progress after the launcher exit must be observed')
    job=runpy.run_path(str(stage/'background-worker.py'))['process_job_state']
    launcher_gone=False
    try:launcher_query=job(launch['launcher_pid']);launcher_gone=not launcher_query['process_active']
    except OSError as exc:
        if getattr(exc,'winerror',None)!=87:raise
        launcher_query=dict(pid=launch['launcher_pid'],exited=True,Windows_OpenProcess_error=87);launcher_gone=True
    if not launcher_gone:raise ValueError('Do not claim a completed launcher handoff while the launcher remains active')
    live_worker=job(proof['worker_pid']);live_collector=job(child['collector_pid'])
    if any(v['is_process_in_job']or not v['process_active']for v in (live_worker,live_collector)):raise ValueError('Live independent processes must still run')
    value=dict(kind='R3A-actual-independent-background-handoff-complete.v1',verified_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,launcher_exited=launcher_gone,launcher_query=launcher_query,
        live_worker=live_worker,live_collector=live_collector,heartbeat_sequence=heartbeat['sequence'],
        observed_successful_scenes=heartbeat['local_snapshot']['manifest']['successful_entries'],
        preserved_successful_scenes_before_handoff=587,successful_scene_target=912,
        source_identity='source-identity-v3.json',source_identity_sha256=sha(stage/'source-identity-v3.json'),
        config_sha256=sha(repo/'configs/native_rgb_r3a.json'),
        independent_startup_proof_sha256=sha(directory/'worker-startup-proof.json'),
        collector_start_sha256=sha(directory/'collector-start.json'),
        actual_progress_after_launcher_exit=True,same_original_physics=True,velocity_half_life_seconds=.1,
        OS_reboot_resume=False,requires_computer_awake_and_user_signed_in=True,
        closing_Codex_lifecycle_independent=True,actually_closed_Codex_during_test=False,
        offline_scope='Finite remaining R3A collection only; no GPU/model work, no blind restart and no offline MCP callback.',
        reopen_action='Read background/run/completion.json and live status before any action. Finish CPU data/integrity/video/report/proposal checks and actually callback parent under standing authorization; no duplicate collector.')
    atomic_json(stage/'background/handoff-success.json',value)
    state_path=stage.parent/'stage-state.json';state=read(state_path)
    state.update(status='background_running',r3a_current_source_identity='r3a/source-identity-v3.json',
        r3a_independent_background=True,r3a_background_worker_pid=proof['worker_pid'],r3a_background_collector_pid=child['collector_pid'],
        r3a_background_directory='r3a/background/run',r3a_handoff_success_file='r3a/background/handoff-success.json',
        r3a_complete=False,r3b_execution_authorized=False,r3b_training_authorized=False,collection_complete=False,
        next_action=value['reopen_action'])
    state['callback']['status']='pending_full_R3A_after_offline_collection_and_reopen'
    atomic_json(state_path,state)
    prefix='''# R3A collection continues independently after Codex closes (2026-10-30)

The human requested stopping this chat and closing Codex while collection
continues. The finite collector is now an independent Windows background process.
Both worker and collector have actual IsProcessInJob=false; the launcher has
exited and subsequent heartbeats show new successful trajectories. It is safe
to close Codex. Keep the computer awake and the user signed in. No reboot or
logoff restart has been configured, and the worker does not retry failed runs.

Actual startup and progress evidence is in
artifacts/native-rgb-supervised-20261029/r3a/background/handoff-success.json.
Production status, heartbeats, logs and completion.json are in background/run/.
The worker continues the frozen plan toward912 successes, preserving587 already
completed trajectories and all old costs. A deliberate durable-write handoff
stopped X074a0 after1 complete control; its partial remains and frozena1 is used.
No successful scene is recollected. The selected velocity half-life stays0.1s.

On reopening, FIRST inspect background/run/status.json,completion.json and the
collector owner. Do not launch another collector. If912 are successfully closed,
finish the prepared CPU integrity/data/throughput/video/report/proposal checks,
then actively send the full evidence to the planning chat using standing human
authorization and save the receipt. No GPU,R3B,heldout shapes or server work is
authorized here. Offline collection cannot invoke the Codex callback tool; the
mainR3A completion/callback remains pending. See RESUME_AFTER_BACKGROUND.md.
Earlier sections below are historical.

'''
    for path in [repo/'docs/STATUS.md',stage.parent/'SUPERVISION.md']:
        text=path.read_text(encoding='utf-8')
        if not text.startswith('# R3A collection continues independently'):path.write_text(prefix+text,encoding='utf-8')
    print(json.dumps(dict(passed=True,launcher_exited=True,worker_pid=proof['worker_pid'],collector_pid=child['collector_pid'],
        successful_scenes=value['observed_successful_scenes'],heartbeat_sequence=value['heartbeat_sequence'])))


if __name__=='__main__':main()
