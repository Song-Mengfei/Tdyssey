"""Read-only training preflight: pinned assets, GPU kernels, disk and packages."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.policy import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--backend', choices=('cu124', 'cu128'), required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT/'third_party/diffusion_policy')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/official-dp/ema_policy.pt')
    parser.add_argument('--output-root', type=Path, required=True, help='Existing disk/directory to check; no writes')
    parser.add_argument('--tracking', action='store_true', help='Require the optional W&B extra')
    args = parser.parse_args()
    config = ForwardConfig(**json.loads(args.config.read_text()))
    doctor = [sys.executable, '-B', str(ROOT/'scripts/doctor.py'), '--runtime', '--require-cuda',
              '--dp', '--backend', args.backend]
    if args.tracking:
        doctor.append('--tracking')
    result = subprocess.run(doctor, capture_output=True, text=True, check=False)
    report = {'doctor': json.loads(result.stdout), 'config': asdict(config), 'errors': [],
              'lock_sha256': hash_file(ROOT/'uv.lock'), 'assets': {}}
    if result.returncode:
        report['errors'].append('Environment doctor failed; see doctor.errors')
    manifest = json.loads((ROOT/'manifests/official_dp.json').read_text())
    for relative, expected in manifest['source_sha256'].items():
        file = args.source_root/relative
        valid = file.is_file() and hash_file(file) == expected
        report['assets'][relative] = valid
        if not valid:
            report['errors'].append(f'Pinned upstream file missing or changed: {relative}')
    compact = manifest['compact']
    valid = (args.checkpoint.is_file() and args.checkpoint.stat().st_size == compact['bytes']
             and hash_file(args.checkpoint) == compact['sha256'])
    report['assets']['base_checkpoint_verified'] = valid
    if not valid:
        report['errors'].append('Copy or fetch the verified compact EMA checkpoint')
    if not args.output_root.is_dir():
        report['errors'].append('--output-root must be an existing directory on the training disk')
    else:
        free = shutil.disk_usage(args.output_root).free
        report['free_bytes'] = free
        if free < config.max_bytes+64*1024**2:
            report['errors'].append('Free space is below the configured output cap plus 64 MiB headroom')
    report['next_check'] = 'Actual train_forward_policy.py smoke run required before larger jobs; no training executed here'
    print(json.dumps(report, indent=2))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
