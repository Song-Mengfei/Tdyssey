"""One hidden Windows worker using the previously verified R3A breakaway method."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

BACKGROUND = Path(__file__).resolve().parent
ROOT = BACKGROUND.parents[3]
WORKER = ROOT / 'scripts/native_rgb_r3b_background.py'
CONFIG = BACKGROUND / 'runtime-config.json'
R3A_HELPERS = BACKGROUND.parents[1] / 'r3a/background-worker.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    api = runpy.run_path(str(R3A_HELPERS))
    read, write, sha, utc, job = (api[k] for k in ('read', 'atomic_json', 'sha', 'utc', 'process_job_state'))
    if args.inspect:
        result = dict(at_utc=utc())
        for name in ('launcher-receipt.json', 'startup-proof.json', 'heartbeat.json', 'status.json', 'completion.json', 'failure.json'):
            if (BACKGROUND / name).exists(): result[name] = read(BACKGROUND / name)
        receipt = result.get('launcher-receipt.json', {})
        for label, pid in (('worker', receipt.get('worker_pid')), ('launcher', receipt.get('launcher_pid'))):
            if pid:
                try: result[label + '_live_query'] = job(pid)
                except OSError as exc: result[label + '_live_query'] = dict(pid=pid, query_error=str(exc), winerror=exc.winerror)
        print(json.dumps(result, ensure_ascii=False)); return 0
    worker_api = runpy.run_path(str(WORKER))
    config, checked = worker_api['preflight'](CONFIG, False)
    base_python = Path(api['python_process_image']()).resolve()
    expected_venv = ROOT.parent / 'Tdyssey/.venv/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected_venv.resolve():
        raise ValueError('Use the existing project virtual environment')
    if args.check:
        print(json.dumps(dict(passed=True, preflight=checked, launcher_job=job(), actual_process_executable=str(base_python),
            venv_python=sys.executable, Torch_imported='torch' in sys.modules))); return 0
    if os.name != 'nt': raise RuntimeError('Windows only')
    # Exclusive durable claim: never launch duplicates or automatically retry.
    claim = BACKGROUND / 'launcher-once-claim.json'
    with claim.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), at_utc=utc(), run_id=config['run_id']), stream)
        stream.flush(); os.fsync(stream.fileno())
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
    command = [sys.executable, '-X', 'utf8', '-B', '-u', str(WORKER), '--config', str(CONFIG), '--execute']
    receipt = dict(kind='R3B-independent-launch-receipt.v1', at_utc=utc(), run_id=config['run_id'],
        launcher_pid=os.getpid(), launcher_ppid=os.getppid(), launcher_job=job(),
        worker_pid=None, command=command, cwd=str(ROOT), creation_flags=flags, creation_flags_hex=hex(flags),
        actual_process_executable=str(base_python), venv_python_identity=sys.executable,
        launcher_sha256=sha(__file__), worker_sha256=sha(WORKER), config_sha256=sha(CONFIG),
        previous_verified_method=str(R3A_HELPERS), previous_method_sha256=sha(R3A_HELPERS),
        hidden_window=True, shell=False, close_fds=True, stdin='DEVNULL',
        stdout_log=str(BACKGROUND/'worker.stdout.log'), stderr_log=str(BACKGROUND/'worker.stderr.log'),
        launch_succeeded=False, independent_job_verified=False, automatic_restart=False,
        reboot_resume=False, offline_MCP_callback=False)
    write(BACKGROUND/'launch-attempt.json', receipt)
    child = None
    try:
        info = subprocess.STARTUPINFO(); info.dwFlags |= subprocess.STARTF_USESHOWWINDOW; info.wShowWindow = 0
        env = os.environ.copy(); env['__PYVENV_LAUNCHER__'] = sys.executable; env['PYTHONDONTWRITEBYTECODE']='1'
        with (BACKGROUND/'worker.stdout.log').open('ab', buffering=0) as output, (BACKGROUND/'worker.stderr.log').open('ab', buffering=0) as error:
            child = subprocess.Popen(command, executable=str(base_python), cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=output, stderr=error, close_fds=True, shell=False, creationflags=flags, startupinfo=info, env=env)
        receipt['worker_pid'] = child.pid
        write(BACKGROUND/'launch-attempt.json', receipt)
        # The worker first verifies the full saved library/storage inventory.
        deadline = time.monotonic() + 120.
        proof_path = BACKGROUND/'startup-proof.json'
        while not proof_path.exists():
            if child.poll() is not None: raise RuntimeError('Worker exited before startup proof: '+str(child.returncode))
            if time.monotonic() >= deadline: raise RuntimeError('Startup proof missing; no job-bound fallback')
            time.sleep(.05)
        proof = read(proof_path)
        if proof['pid'] != child.pid or proof['job']['is_in_job'] is not False:
            raise RuntimeError('Actual worker independence proof failed')
        if proof['preflight']['config_sha256'] != receipt['config_sha256'] or proof['working_directory'] != str(ROOT):
            raise RuntimeError('Worker configuration/cwd differs')
        if Path(proof['prefix']).resolve() != (ROOT.parent/'Tdyssey/.venv').resolve():
            raise RuntimeError('Worker did not retain the existing virtual environment')
        if Path(proof['actual_process_executable']).resolve() != base_python:
            raise RuntimeError('Worker actual Python executable differs')
        live = job(child.pid)
        if live['is_process_in_job'] or not live['process_active']:
            raise RuntimeError('Worker is not job independent and active')
        receipt.update(launch_succeeded=True, independent_job_verified=True, ready_at_utc=utc(),
            worker_job_query=live, startup_proof=str(proof_path), launcher_exit_intent=0)
        write(BACKGROUND/'launcher-receipt.json', receipt)
        return 0
    except BaseException as exc:
        receipt.update(error=repr(exc), worker_pid=None if child is None else child.pid,
            no_job_bound_fallback_attempted=True, launcher_exit_intent=2)
        write(BACKGROUND/'launcher-receipt.json', receipt)
        return 2


if __name__ == '__main__': raise SystemExit(main())
