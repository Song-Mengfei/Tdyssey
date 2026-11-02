"""One detached, finite collection -> selection -> one epoch -> paired predictions."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(REPO), str(REPO / 'scripts')]
from common import atomic_json, check_storage, read, sha, utc
import direct_target_background as detached

def validate(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen runtime config SHA mismatch')
    c = read(path)
    if c['kind'] != 'accelerated-rgb-one-epoch.v1' or c['epochs'] != 1:
        raise ValueError('Scope identity mismatch')
    if c['limits'] != dict(attempts=2000, controls_per_attempt=51, controls=102000,
                           regular_substeps=1020000):
        raise ValueError('Physical budget mismatch')
    if Path(c['artifact_root']).resolve() != ROOT or c['shared_cap_bytes'] != 10_000_000_000:
        raise ValueError('Run root/storage identity mismatch')
    for item in c['bindings']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])
    return c

def env(c, phase):
    value = os.environ.copy()
    value.update(__PYVENV_LAUNCHER__=c['python_executable'], PYTHONUNBUFFERED='1',
                 PYTHONDONTWRITEBYTECODE='1', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
                 PYGAME_HIDE_SUPPORT_PROMPT='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1')
    if phase in ('collecting', 'selecting'):
        value['CUDA_VISIBLE_DEVICES'] = ''
    return value

def status(c, digest, phase, **extra):
    row = dict(run_id=c['run_id'], config_sha256=digest, phase=phase,
               worker_pid=os.getpid(), updated_at_utc=utc(), **extra)
    atomic_json(ROOT/'background/status.json', row)
    atomic_json(ROOT/'background/heartbeat.json', row)
    atomic_json(ROOT/'PHASE_STATE.json', dict(**row, parent_thread_id=c['parent_thread_id'],
        callback_authorized=True, callback_requires_running_app=True,
        workflow='collect -> select -> one complete epoch -> paired predictions',
        auto_next_epoch=False))

def child_phase(c, a, phase):
    cmd = [detached.actual_python(), '-X', 'utf8', '-B', str(Path(__file__)), phase,
           '--config', str(Path(a.config).resolve()), '--config-sha', a.config_sha]
    bg = ROOT/'background'
    with (bg/(phase+'-stdout.log')).open('ab') as out, (bg/(phase+'-stderr.log')).open('ab') as err:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env(c, phase), stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, close_fds=True, startupinfo=detached.hidden_info(),
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    proof = dict(run_id=c['run_id'], config_sha256=a.config_sha, pid=proc.pid,
                 is_process_in_job=detached.in_job(int(proc._handle)), command=cmd, at_utc=utc())
    atomic_json(bg/(phase+'-process.json'), proof, exclusive=True)
    if proof['is_process_in_job']:
        proc.terminate(); proc.wait(timeout=10)
        raise RuntimeError('Phase child has not escaped Windows Job')
    try:
        while True:
            code = proc.poll()
            status(c, a.config_sha, phase, child_pid=proc.pid, child_exit_code=code)
            if code is not None:
                if code != 0:
                    raise RuntimeError(f'{phase} failed with exit {code}; inspect phase stderr and result')
                return read(bg/(phase+'-result.json'))
            time.sleep(15)
    except BaseException:
        if proc.poll() is None:
            if phase == 'collecting':
                atomic_json(bg/'STOP.json',dict(run_id=c['run_id'],reason='Owner exception; preserve finite in-flight attempts',at_utc=utc()))
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.terminate(); proc.wait(timeout=15)
            else:
                proc.terminate(); proc.wait(timeout=15)
        raise

def phase(c, a):
    check_storage(ROOT, c, full=True)
    if a.mode == 'collecting':
        from collect import collect
        result = collect(ROOT, c)
        if result.get('failed') or result.get('graceful_stop'):
            raise RuntimeError('Collection stopped; no automatic training after engineering failure or stop: '+str(result.get('failure')))
    elif a.mode == 'selecting':
        from selection import select
        result = select(ROOT, c)
        if not result.get('ready_for_training', result.get('training_ready', True)):
            raise RuntimeError('Selection has no meaningful clean new motion: '+str(result))
    else:
        from train_eval import train_and_evaluate
        result = train_and_evaluate(ROOT, c)
    atomic_json(ROOT/'background'/(a.mode+'-result.json'), result)
    atomic_json(ROOT/'background'/(a.mode+'-storage.json'), check_storage(ROOT, c, full=True))

def run(c, a):
    begin = time.perf_counter(); results = {}; failure = None
    try:
        proof = dict(run_id=c['run_id'], config_sha256=a.config_sha, pid=os.getpid(),
            actual_interpreter=detached.actual_python(), is_process_in_job=detached.in_job(),
            launcher_exit=detached.wait_launcher_exit(a.launcher_pid), started_at_utc=utc())
        atomic_json(ROOT/'background/worker-startup.json', proof, exclusive=True)
        if proof['is_process_in_job']:
            raise RuntimeError('Owner has not escaped Windows Job')
        for name in ('collecting', 'selecting', 'training_and_evaluating'):
            validate(a.config, a.config_sha)
            results[name] = child_phase(c, a, name)
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        atomic_json(ROOT/'background/failure.json', failure)
    terminal = dict(run_id=c['run_id'], config_sha256=a.config_sha,
        completed=failure is None, failed=failure is not None, failure=failure,
        results=results, wall_seconds=time.perf_counter()-begin, finished_at_utc=utc(),
        state='waiting_callback', callback_sent=False, auto_retry=False, auto_next_epoch=False)
    atomic_json(ROOT/'background/completion.json', terminal)
    status(c, a.config_sha, 'waiting_callback', failed=failure is not None)
    if failure or not (ROOT/'REPORT.md').exists():
        (ROOT/'REPORT.md').write_text('# Accelerated RGB run\n\n'
            + ('FAILED: '+failure['message'] if failure else 'Collection, selection, one full new-data epoch and paired evaluation completed.')
            + '\n\nSee [terminal receipt](background/completion.json), '
            '[selection](selection/summary.json), [training and prediction result](training/summary.json).\n'
            'No automatic second epoch or new experiment. All media are raw model predictions or labeled saved truth.\n', encoding='utf-8')
    return 1 if failure else 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('check','launch','worker','collecting','selecting','training_and_evaluating'))
    p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
    p.add_argument('--launcher-pid', type=int)
    a = p.parse_args(); c = validate(a.config, a.config_sha)
    bg = ROOT/'background'; bg.mkdir(exist_ok=True)
    if a.mode == 'check':
        print(dict(passed=True, physical_calls=0, model_calls=0)); return 0
    if a.mode == 'worker':
        return run(c, a)
    if a.mode != 'launch':
        phase(c, a); return 0
    for name in ('launch-claim.json','worker-startup.json','completion.json'):
        if (bg/name).exists():
            raise FileExistsError('No automatic repeat: '+name)
    check_storage(ROOT, c, full=True)
    atomic_json(bg/'launch-claim.json', dict(run_id=c['run_id'], at_utc=utc()), exclusive=True)
    cmd = [detached.actual_python(), '-X','utf8','-B',str(Path(__file__)), 'worker',
           '--config', str(Path(a.config).resolve()), '--config-sha',a.config_sha,
           '--launcher-pid', str(os.getpid())]
    with (bg/'worker-stdout.log').open('ab') as out, (bg/'worker-stderr.log').open('ab') as err:
        child = subprocess.Popen(cmd, cwd=REPO, env=env(c,'owner'), stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,close_fds=True,startupinfo=detached.hidden_info(),
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB)
    receipt = dict(run_id=c['run_id'], config_sha256=a.config_sha, worker_pid=child.pid,
        launcher_pid=os.getpid(), launcher_is_process_in_job=detached.in_job(), hidden=True,
        command=cmd, at_utc=utc())
    atomic_json(bg/'launcher-receipt.json',receipt,exclusive=True)
    print(receipt)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
