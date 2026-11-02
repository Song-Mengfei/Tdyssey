"""Gated policy-v2 planner selection over the unchanged v1 original executor."""
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
    parser.add_argument('command', choices=('check', 'collect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=('calibration-v2', 'formal', 'supplement'), required=True)
    parser.add_argument('--physics-gate', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--storage-roots-json', type=Path)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--resource-policy', type=Path)
    parser.add_argument('--letters')
    parser.add_argument('--run-label')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--runtime-config-sha')
    parser.add_argument('--require-detached', action='store_true')
    args = parser.parse_args()
    from tdyssey.strict_original_target_policy_v2 import install_frozen_policy
    from tdyssey import strict_original_contact_pilot as collector
    from tdyssey.original_contact_pilot import file_sha
    gate = json.loads(args.physics_gate.read_text(encoding='utf-8'))
    if gate.get('bindings', {}).get(str(Path(__file__).resolve())) != file_sha(__file__):
        raise ValueError('Actual v2 launcher CLI is not frozen in phase gate')
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    if (policy.get('cli_source') != str(Path(__file__).resolve())
            or policy.get('cli_source_sha256') != file_sha(__file__)):
        raise ValueError('Actual v2 CLI differs from the frozen policy implementation')
    implementation = install_frozen_policy(args.policy, args.physics_gate)
    plan, gate, rules = collector.verify_gate(args.plan, args.physics_gate, args.policy, args.phase)
    if args.command == 'check':
        print(json.dumps(dict(passed=True, phase=args.phase, protocol_sha256=gate['protocol_sha256'],
            scenes=len(plan['slots']), physical_controls=0, model_calls=0, policy_implementation=implementation)))
        return
    if any(value is None for value in (args.source_root, args.storage_roots_json, args.resource_policy, args.run_label)):
        parser.error('collect requires --source-root --storage-roots-json --resource-policy --run-label')
    roots = json.loads(args.storage_roots_json.read_text(encoding='utf-8'))['roots']
    result = collector.collect(args.plan, args.output, args.phase, args.physics_gate, args.source_root, roots,
        policy_path=args.policy, resource_policy=args.resource_policy, letters=args.letters,
        run_label=args.run_label, stop_file=args.stop_file, run_id=args.run_id,
        runtime_config_sha256=args.runtime_config_sha, require_detached=args.require_detached)
    print(json.dumps(dict(result=result, policy_implementation=implementation), ensure_ascii=False))


if __name__ == '__main__':
    main()
