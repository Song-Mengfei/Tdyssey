"""Root-reviewed, one-time cal2 gate/config freeze. Never starts physics/processes."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
REPO = P.parents[1]
SOURCE = REPO.parent / 'Tdyssey/third_party/diffusion_policy'
PLAN_SHA = '008cee370814e150f8d5e7711bd1412ee405c962ccfa309c5f501f344a528ff1'
PHYSICS_SHA = '768118e17ba69df4cca57c7041eaf84ecfd760466c1315451e134673585120bf'
MODEL_SHA = '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
PROTOCOL = '2ef7a9562c7451b502e23a5da08e5110349d25c8c2cfac37886bb990086b3d69'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def exclusive(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def binding(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': sha(p)}


def local_dependencies():
    """Conservative local AST import closure; no imports or simulator execution."""
    pending = ['tdyssey.original_contact_pilot', 'tdyssey.native_rgb_contact_safe_physics']
    found = set()
    while pending:
        name = pending.pop()
        path = REPO.joinpath(*name.split('.')).with_suffix('.py')
        if not path.is_file():
            path = REPO.joinpath(*name.split('.')) / '__init__.py'
        if not path.is_file() or path in found:
            continue
        found.add(path)
        package = name.split('.') if path.name == '__init__.py' else name.split('.')[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package) - node.level + 1]
                    names = ['.'.join(prefix + ([node.module] if node.module else []))]
                    if not node.module:
                        names += ['.'.join(prefix + [n.name]) for n in node.names]
                elif node.module:
                    names = [node.module]
            pending.extend(n for n in names if n == 'tdyssey' or n.startswith('tdyssey.'))
    return found


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--policy', type=Path, required=True)
    a.add_argument('--checks', type=Path, required=True)
    args = a.parse_args()
    runtime = P / 'background/calibration-v2'
    gate_path = P / 'calibration-v2-execution-gate.json'
    config_path = runtime / 'runtime-config.json'
    assert not gate_path.exists() and not config_path.exists(), 'Never overwrite a frozen release'
    checks = read(args.checks)
    assert checks.get('passed') is True and not checks['failures'] and not checks['errors'], 'Actual focused checks must pass'
    assert sha(checks['actual_output_log']) == checks['actual_output_log_sha256']
    assert all(sha(f) == expected for f, expected in checks['source_sha256'].items()), 'Tested sources changed'
    assert sha(args.policy) == checks['calibration_policy_sha256']
    assert isinstance(read(args.policy).get('changes'), dict), 'Explicit finite sampling revision required'
    assert sha(P / 'collection/plan.json') == PLAN_SHA
    assert sha(REPO / 'tdyssey/native_rgb_contact_safe_physics.py') == PHYSICS_SHA
    old_gate = read(P / 'physics-gate.json')
    assert old_gate['passed'] and old_gate['protocol_sha256'] == PROTOCOL
    for k in ('independent_audit', 'actual_browser_check', 'CPU_interface_check', 'actual_contact_frame_proof'):
        path = Path(old_gate[k])
        assert sha(path) == old_gate['bindings'][str(path)]
    assert read(P / 'collection/calibration-v1-completion.json')['completed'] is True
    cal1 = read(P / 'collection/revisions/calibration-v1-ledger-frozen.json')
    assert len(cal1['attempts']) == 18 and cal1['actual_controls'] == 1152
    assert all(a['phase'] == 'calibration-v1' and a['status'] == 'completed' and a['actual_controls'] == 64 for a in cal1['attempts'])
    checkpoint = REPO / 'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt'
    assert sha(checkpoint) == MODEL_SHA
    resources = P / 'policies/resource-policy-v2.json'
    assert sha(resources) == checks['resource_policy_sha256']
    r = read(resources)
    assert r['shared_cap_bytes'] == 3 * 1024**3 and r['atomic_reserve_bytes'] == 256 * 1024**2
    assert r['postprocess_reserve_bytes'] == 128 * 1024**2
    sources = local_dependencies() | {
        REPO / 'scripts/original_contact_pilot.py', Path(__file__),
        REPO / 'manifests/official_dp.json',
        SOURCE / 'diffusion_policy/env/pusht/pusht_env.py',
        SOURCE / 'diffusion_policy/env/pusht/pymunk_override.py',
    }
    sources |= {P / ('background/' + f) for f in (
        'common.py', 'worker.py', 'calibration-worker.py', 'calibration-launcher.py')}
    bindings = [binding(f) for f in sorted(sources)]
    evidence = [P / 'physics-gate.json', args.policy, args.checks, resources,
                P / 'collection/plan.json', P / 'collection/rules.json',
                P / 'collection/calibration-v1-completion.json',
                P / 'collection/calibration-v1-review-summary.json',
                P / 'collection/route-diagnosis.json']
    evidence += [Path(old_gate[k]) for k in (
        'independent_audit', 'actual_browser_check', 'CPU_interface_check', 'actual_contact_frame_proof')]
    gate_bindings = {b['path']: b['sha256'] for b in bindings + [binding(f) for f in evidence]}
    gate = dict(kind='contact-v6-reused-physics-and-new-cal2-execution-gate.v1', passed=True,
        released_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='One original frozen18-slot cal2 sampling revision only',
        training_authorized=False, model_calls=0, optimizer_updates=0,
        protocol_sha256=PROTOCOL, physics_source_sha256=PHYSICS_SHA,
        checkpoint_sha256=MODEL_SHA, previous_calibration_gate=binding(P/'physics-gate.json'),
        bindings=gate_bindings, physical_regression_reused_unchanged=True,
        sampling_policy_outcome_verified=False,
        limitation='This gate releases finite cal2 execution, not formal data, training or a claim of improved model prediction.')
    exclusive(gate_path, gate)
    notification = read(runtime / 'notification.json')
    plan = read(P / 'collection/plan.json')
    slots = [s['slot_id'] for s in plan['slots'] if s['phase'] == 'calibration-v2']
    assert len(slots) == 18
    config = dict(kind='tdyssey-contact-background.v1', mode='calibration-v2',
        run_id=notification['run_id'], run_label='calibration-v2', released=True,
        repo=str(REPO), artifact_root=str(P), collection=str(P/'collection'),
        python_executable=str(REPO.parent/'Tdyssey/.venv/Scripts/python.exe'),
        source_root=str(SOURCE), heartbeat_seconds=30, auto_retry=False,
        training_authorized=False, model_calls=0, optimizer_updates=0,
        source_bindings=bindings, calibration_slot_ids=slots)
    files = dict(plan=P/'collection/plan.json', physics_gate=gate_path,
        checkpoint=checkpoint, storage_roots=P/'storage-roots.json',
        resource_policy=resources, calibration_policy=args.policy,
        calibration_ledger=P/'collection/revisions/calibration-v1-ledger-frozen.json',
        collector_script=REPO/'scripts/original_contact_pilot.py',
        notification=runtime/'notification.json')
    config.update({k: binding(v) for k, v in files.items()})
    exclusive(config_path, config)
    config_sha = sha(config_path)
    with (runtime/'runtime-config.sha256').open('x', encoding='ascii') as f:
        f.write(config_sha + '\n')
    print(json.dumps(dict(config=str(config_path), config_sha256=config_sha,
        gate=str(gate_path), gate_sha256=sha(gate_path), source_bindings=len(bindings),
        physics_controls=0, model_calls=0, processes_started=0)))


if __name__ == '__main__':
    main()
