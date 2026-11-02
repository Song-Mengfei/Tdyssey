"""Explicit strict-original scene planning and separately gated collection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('plan', 'check', 'collect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026103017)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--phase', choices=('calibration-v1', 'calibration-v2', 'formal', 'supplement'))
    parser.add_argument('--physics-gate', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--storage-roots-json', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--resource-policy', type=Path)
    parser.add_argument('--letters')
    parser.add_argument('--run-label')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--runtime-config-sha')
    parser.add_argument('--require-detached', action='store_true')
    args = parser.parse_args()
    from tdyssey.strict_original_contact_pilot import build_plan, collect, verify_gate
    if args.command == 'plan':
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/'plan.json'
        if path.exists():
            raise FileExistsError('Frozen plan already exists; no overwrite')
        plan = build_plan(args.seed)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps(dict(plan=str(path.resolve()), physical_controls=0, model_calls=0)))
        return
    if any(value is None for value in (args.plan, args.physics_gate, args.policy, args.phase)):
        parser.error('--plan --physics-gate --policy --phase are required')
    if args.command == 'check':
        plan, gate, rules = verify_gate(args.plan, args.physics_gate, args.policy, args.phase)
        print(json.dumps(dict(passed=True, phase=args.phase, protocol_sha256=gate['protocol_sha256'],
            scenes=len(plan['slots']), physical_controls=0, model_calls=0)))
        return
    if any(value is None for value in (args.source_root, args.storage_roots_json, args.resource_policy, args.run_label)):
        parser.error('collect requires --source-root --storage-roots-json --resource-policy --run-label')
    roots = json.loads(args.storage_roots_json.read_text(encoding='utf-8'))['roots']
    result = collect(args.plan, args.output, args.phase, args.physics_gate, args.source_root, roots,
        policy_path=args.policy, resource_policy=args.resource_policy, letters=args.letters,
        run_label=args.run_label, stop_file=args.stop_file, run_id=args.run_id,
        runtime_config_sha256=args.runtime_config_sha, require_detached=args.require_detached)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
