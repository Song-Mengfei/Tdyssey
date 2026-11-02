"""Read saved identities/current calibration only; no subprocess or real physics."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from common import CHECKPOINT, PHYSICS, PROTOCOL, atomic_json, read_json, sha, utc
from worker import merge_ledgers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, help='Optional new engineering receipt strictly under this background directory')
    parser.add_argument('--tests',action='store_true',help='Also run stdlib synthetic checks and syntax checks; no subprocess')
    args=parser.parse_args()
    background=Path(__file__).resolve().parent
    pilot=background.parent
    repo=pilot.parents[1]
    collection=pilot/'collection'
    plan=read_json(collection/'plan.json')
    ledger=read_json(collection/'ledger.json')
    checkpoint=repo/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
    physics=repo/'tdyssey/native_rgb_contact_safe_physics.py'
    config=dict(collection=str(collection),plan=dict(sha256=sha(collection/'plan.json')),
                formal_slot_ids=[s['slot_id'] for s in plan['slots'] if s['phase']=='formal'])
    merged=merge_ledgers(config,{s['slot_id']:s for s in plan['slots']},[collection/'ledger.json'])
    cli=(repo/'scripts/original_contact_pilot.py').read_text(encoding='utf-8')
    report=dict(kind='tdyssey-background-zero-work-engineering-preflight.v1',at_utc=utc(),
        actual_physics_steps_executed=0, actual_subprocesses_started=0, model_calls=0, optimizer_updates=0,
        formal_release_created=False, formal_worker_started=False,
        existing_saved_calibration=dict(phase_attempts=merged['phase_attempts'],phase_complete=merged['phase_complete'],
            controls=merged['actual_controls'],valid_controls=merged['actual_valid_controls'],
            substeps=merged['actual_substeps'],physics_seconds=merged['actual_physics_seconds'],
            interpretation='Existing historical saved data only; these are not preflight execution costs or model results'),
        checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),expected_sha256=CHECKPOINT),
        physics=dict(path=str(physics),sha256=sha(physics),expected_sha256=PHYSICS,protocol_sha256=PROTOCOL),
        cli_contract_flags={flag:flag in cli for flag in ('--run-label','--resource-policy','--stop-file','--letters')},
        background_source_bindings=[dict(path=str(background/name),sha256=sha(background/name)) for name in
            ('common.py','worker.py','launcher.py','calibration-worker.py','calibration-launcher.py','isolation_probe.py')],
        no_model_or_physics_imports=not ('torch' in sys.modules or 'numpy' in sys.modules or any(s.startswith('tdyssey') for s in sys.modules)))
    if args.tests:
        import ast
        import unittest
        import test_worker
        syntax=[]
        for path in background.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path));syntax.append(path.name)
        result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_worker))
        report['synthetic_checks']=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            passed=result.wasSuccessful(),syntax_files=syntax,actual_physics_steps=0,subprocesses_started=0,
            interpretation='Synthetic engineering fixtures only; these are not collected trajectories or model results')
        report['no_model_or_physics_imports']=not ('torch' in sys.modules or 'numpy' in sys.modules or any(s.startswith('tdyssey') for s in sys.modules))
    report['read_only_identity_checks_passed']=(report['checkpoint']['sha256']==CHECKPOINT and report['physics']['sha256']==PHYSICS and report['no_model_or_physics_imports'])
    if args.receipt:
        receipt=args.receipt.resolve()
        if receipt.parent != background or receipt.exists():
            raise ValueError('Receipt must be a new file directly under this background directory')
        atomic_json(receipt,report)
    import json
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['read_only_identity_checks_passed'] and report.get('synthetic_checks',{}).get('passed',True) else 1


if __name__=='__main__':
    raise SystemExit(main())
