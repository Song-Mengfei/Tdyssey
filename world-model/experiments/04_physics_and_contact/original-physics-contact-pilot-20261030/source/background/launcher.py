"""Launch exactly one finite worker outside Codex's Windows Job and then exit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import (KIND, actual_python_pe, atomic_json, child_environment,
                    exclusive_json, in_job, utc, validate_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config, paths, by_id = validate_config(args.config, args.config_sha, require_release=not args.check_only)
    background = Path(config['artifact_root'])/'background'
    pe = actual_python_pe()
    evidence = dict(kind=KIND, run_id=config['run_id'], config_sha256=args.config_sha,
        launcher_pid=os.getpid(), launcher_is_process_in_job=in_job(), interpreter_pe=pe,
        venv_launcher=config['python_executable'], at_utc=utc(),
        creation_flags=['DETACHED_PROCESS', 'CREATE_NEW_PROCESS_GROUP', 'CREATE_BREAKAWAY_FROM_JOB'],
        hidden=True, check_only=args.check_only)
    if args.check_only:
        print(json.dumps(dict(evidence, passed=True, processes_started=0, actual_physics_steps=0,
                             model_calls=0, optimizer_updates=0, files_written=0), ensure_ascii=False))
        return 0
    for name in ('worker.lock.json', 'completion.json', 'worker-startup.json', 'launch-claim.json'):
        if (background/name).exists():
            raise RuntimeError('Existing background identity requires owner audit; no automatic resume: '+name)
    # Never use a fallback lacking BREAKAWAY_FROM_JOB. An OS rejection is a failed launch,
    # not permission to leave a worker attached to the Codex process tree.
    exclusive_json(background/'launch-claim.json', evidence)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    command = [pe, '-X', 'utf8', '-B', str(background/'worker.py'), '--config', str(args.config.resolve()),
               '--config-sha', args.config_sha, '--launcher-pid', str(os.getpid())]
    try:
        with (background/'worker-stdout.log').open('ab') as out, (background/'worker-stderr.log').open('ab') as err:
            process = subprocess.Popen(command, cwd=config['repo'], env=child_environment(config),
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True,
                startupinfo=startup, creationflags=flags)
        evidence.update(worker_pid=process.pid, spawn_succeeded=True, command=command,
                        worker_exit_code_at_launch=process.poll(),
                        note='Worker must independently prove IsProcessInJob=false and wait for this launcher to exit before first control')
        atomic_json(background/'launcher-receipt.json', evidence)
        print(json.dumps(evidence, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        evidence.update(spawn_succeeded=False, error=repr(error), actual_physics_steps=0)
        atomic_json(background/'launcher-receipt.json', evidence)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
