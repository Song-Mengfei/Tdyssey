"""Explicit plan/collect/report entry point; collection is physics-gated."""
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
    parser.add_argument('command', choices=('plan', 'collect', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026103017)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--phase', choices=('calibration-v1', 'calibration-v2', 'formal', 'supplement'))
    parser.add_argument('--physics-gate', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--storage-roots-json', type=Path)
    parser.add_argument('--policy-revision', type=Path)
    parser.add_argument('--letters')
    parser.add_argument('--run-label')
    parser.add_argument('--resource-policy', type=Path)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--videos', action='store_true')
    args = parser.parse_args()
    from tdyssey.original_contact_pilot import build_plan, write_json, collect
    if args.command == 'plan':
        destination = args.output/'plan.json'
        if destination.exists():
            raise FileExistsError('Frozen plan exists; never replace it after observing physics')
        plan = build_plan(args.seed)
        write_json(destination, plan)
        write_json(args.output/'rules.json', plan['rules'])
        print(json.dumps(dict(path=str(destination.resolve()), slots=len(plan['slots']),
                              static_rejections=len(plan['static_rejections']),
                              rules_sha256=plan['rules_sha256'], physics_steps=0,
                              optimizer_updates=0)))
    elif args.command == 'collect':
        if any(v is None for v in (args.plan, args.phase, args.physics_gate, args.source_root, args.storage_roots_json)):
            parser.error('Collection requires --plan --phase --physics-gate --source-root --storage-roots-json')
        resource = json.loads(args.storage_roots_json.read_text(encoding='utf-8'))
        roots = resource if isinstance(resource, list) else resource['roots']
        completion = collect(args.plan, args.output, args.phase, args.physics_gate, args.source_root,
                roots, rules_path=args.policy_revision,
                letters=args.letters, run_label=args.run_label,
                resource_policy=args.resource_policy, stop_file=args.stop_file)
        print(json.dumps(completion))
    else:
        from tdyssey.original_contact_pilot_report import build_report
        if args.resource_policy is None:
            print(json.dumps(build_report(args.output, videos=args.videos)))
        else:
            if args.storage_roots_json is None:
                parser.error('A resource-governed report requires --storage-roots-json')
            from tdyssey.original_contact_pilot_resources import ResourceReservation
            resource = json.loads(args.storage_roots_json.read_text(encoding='utf-8'))
            roots = resource if isinstance(resource, list) else resource['roots']
            with ResourceReservation(args.output, roots, args.resource_policy,
                    'report:'+str(os.getpid()), role='postprocess') as reservation:
                print(json.dumps(build_report(args.output, videos=args.videos, resource_reservation=reservation)))


if __name__ == '__main__':
    main()
