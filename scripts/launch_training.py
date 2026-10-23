"""Plan or sequentially run forward-policy comparisons using the current Python.

Planning uses only the standard library. Execution never installs dependencies,
downloads assets, chooses a GPU, or changes the existing training configurations.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.tracking import add_tracking_arguments


def build_jobs(base, seeds, objectives, scopes):
    jobs = []
    for objective in objectives:
        for scope in scopes:
            for seed in seeds:
                config = ForwardConfig(**(base | {"seed": seed, "objective": objective, "trainable": scope}))
                name = f"{objective}-{scope}-seed{seed}"
                jobs.append({"name": name, "config": asdict(config)})
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError("Repeated seeds/methods would overwrite a run")
    return jobs


def write_manifest(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True, help='New comparison directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--objectives', choices=('nft', 'advantage_flow'), nargs='+')
    parser.add_argument('--scopes', choices=('lora', 'action_network', 'head'), nargs='+')
    parser.add_argument('--source-root', type=Path, default=ROOT/'third_party/diffusion_policy')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/official-dp/ema_policy.pt')
    parser.add_argument('--execute', action='store_true', help='Without this flag, print plan and write nothing')
    parser.add_argument('--keep-going', action='store_true', help='Continue other jobs after a failed job')
    add_tracking_arguments(parser)
    args = parser.parse_args()
    try:
        payload = args.config.read_bytes()
        base = json.loads(payload)
        jobs = build_jobs(base, args.seeds, args.objectives or [base.get('objective','nft')],
                          args.scopes or [base.get('trainable','lora')])
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    root = args.output_root.resolve()
    for job in jobs:
        command = [sys.executable, '-B', str(ROOT/'scripts/train_forward_policy.py'),
            '--config', str(root/(job['name']+'.json')), '--output', str(root/job['name']),
            '--source-root', str(args.source_root.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
            '--tracking', args.tracking, '--wandb-project', args.wandb_project,
            '--run-group', args.run_group or root.name, '--run-name', job['name'],
            '--log-every', str(args.log_every)]
        if args.wandb_entity:
            command.extend(['--wandb-entity', args.wandb_entity])
        job.update(command=command, status='planned')
    manifest = {'schema_version': 1, 'status': 'planned', 'jobs': jobs,
        'base_config_sha256': hashlib.sha256(payload).hexdigest(),
        'max_training_controls_all_jobs': sum(j['config']['max_train_controls'] for j in jobs),
        'max_training_seconds_all_jobs': sum(j['config']['seconds'] for j in jobs),
        'run_output_caps_sum_bytes': sum(j['config']['max_bytes'] for j in jobs),
        'note': 'Evaluation is additional to training-control caps; W&B spool is separate from run caps.'}
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.execute:
        return 0
    # Refuse any existing comparison, including an unfinished earlier attempt.
    root.mkdir(parents=True, exist_ok=False)
    manifest['status'] = 'running'
    write_manifest(root/'comparison.json', manifest)
    for job in jobs:
        write_manifest(root/(job['name']+'.json'), job['config'])
        job['status'] = 'running'
        write_manifest(root/'comparison.json', manifest)
        process = subprocess.Popen(job['command'], cwd=ROOT)
        def interrupted(signum, frame):
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            raise KeyboardInterrupt()
        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            returncode = process.wait()
            job.update(returncode=returncode, status='complete' if returncode == 0 else 'failed')
        except KeyboardInterrupt:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            job.update(status='interrupted')
            manifest['status'] = 'interrupted'
            write_manifest(root/'comparison.json', manifest)
            return 130
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        write_manifest(root/'comparison.json', manifest)
        if job['status'] != 'complete' and not args.keep_going:
            break
    failed = any(j['status'] != 'complete' for j in jobs)
    manifest['status'] = 'failed' if failed else 'complete'
    write_manifest(root/'comparison.json', manifest)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
