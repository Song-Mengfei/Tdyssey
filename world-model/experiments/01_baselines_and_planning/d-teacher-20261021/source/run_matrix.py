"""Predeclared bounded D-teacher development matrix; no policy or training."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv/Scripts/python.exe'
SCENES = [100000, 100100, 100200, 100300]
SEARCH_SEEDS = [42, 43]
CONCURRENCY = 3


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    output = BASE / 'matrix-v1'
    output.mkdir(exist_ok=False)
    sources = ['tdyssey/simulator_search.py', 'tdyssey/simulator_planning.py',
               'scripts/evaluate_simulator_search.py', 'tdyssey/envs/official.py',
               'tdyssey/geometry.py', 'tdyssey/state.py', 'tdyssey/evaluation.py',
               'tdyssey/official.py', 'manifests/official_dp.json', 'pyproject.toml', 'uv.lock']
    signatures = {}
    for name in sources:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        signatures[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    protocol = {
        'kind': 'D_teacher_development_matrix', 'version': 1,
        'requested_scene_seeds': SCENES, 'search_seeds': SEARCH_SEEDS,
        'scene_selection': 'predeclared spaced requests; only existing geometric reset rejection',
        'known_development_scene': 100000,
        'steps': 300, 'seconds_per_run': 600, 'horizon': 80,
        'population': 12, 'generations': 3, 'elite_count': 3,
        'max_evaluations_per_plan': 40, 'execute_steps': 8, 'settle_steps': 4,
        'maximum_concurrent_runs': CONCURRENCY,
        'training': False, 'checkpoint_loaded': False,
        'success_rule': 'native coverage > 0.95; stop immediately',
        'source_sha256': signatures,
        'scope': 'privileged known-shape simulator action search; development, not held-out generalization',
        'all_runs': [], 'status': 'running',
    }
    for scene in SCENES:
        for seed in SEARCH_SEEDS:
            directory = output / f'scene-{scene}-search-{seed}'
            command = [str(PYTHON), '-B', 'scripts/evaluate_simulator_search.py',
                       '--seed', str(scene), '--search-seed', str(seed),
                       '--steps', '300', '--seconds', '600', '--horizon', '80',
                       '--population', '12', '--generations', '3', '--elite-count', '3',
                       '--max-evaluations', '40', '--execute-steps', '8', '--settle-steps', '4',
                       '--output', str(directory)]
            protocol['all_runs'].append({'scene_request': scene, 'search_seed': seed,
                                         'output': str(directory), 'command': command, 'status': 'queued'})
    save(output / 'protocol.json', protocol)
    active = []
    queued = list(protocol['all_runs'])
    beginning = time.monotonic()
    while queued or active:
        while queued and len(active) < CONCURRENCY:
            # Every process imports the exact same frozen implementation.
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                   for name, digest in signatures.items()):
                raise RuntimeError('Frozen matrix sources changed before process launch')
            row = queued.pop(0)
            logfile = Path(row['output']).with_suffix('.log')
            stream = logfile.open('w', encoding='utf-8')
            process = subprocess.Popen(row['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            row.update(status='running', pid=process.pid, started_elapsed=time.monotonic()-beginning)
            active.append((process, stream, row, time.monotonic()))
            print(f"START scene={row['scene_request']} search={row['search_seed']} pid={process.pid}", flush=True)
        changed = False
        for process, stream, row, started in list(active):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            row.update(exit_code=code, process_seconds=time.monotonic()-started,
                       status='process_complete' if code == 0 else 'process_failed')
            report_path = Path(row['output']) / 'report.json'
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                row['result'] = {key: report.get(key) for key in (
                    'status', 'scene_seed', 'success', 'stop_reason', 'max_coverage',
                    'max_iou', 'final_metrics', 'counts', 'search_counts', 'evaluation_seconds',
                    'torch_loaded', 'checkpoint_loaded', 'training', 'error')}
            active.remove((process, stream, row, started))
            print(f"FINISH scene={row['scene_request']} search={row['search_seed']} "
                  f"exit={code} result={json.dumps(row.get('result', {}))}", flush=True)
            changed = True
        save(output / 'protocol.json', protocol)
        if not changed and active:
            time.sleep(2)
    protocol['status'] = 'complete'
    protocol['elapsed_seconds'] = time.monotonic()-beginning
    save(output / 'protocol.json', protocol)
    print(f"MATRIX COMPLETE {protocol['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
