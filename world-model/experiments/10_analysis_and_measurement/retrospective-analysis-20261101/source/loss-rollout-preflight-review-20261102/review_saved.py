"""Independent read-only code/ledger review; no Torch import or model calls."""
import ast
import contextlib
import hashlib
import json
import pathlib
from types import SimpleNamespace

P = pathlib.Path(__file__).resolve().parent
RUN = P.parent.parent / 'loss-rollout-ab-20261102'
def read(name):
    return json.loads((RUN/name).read_text(encoding='utf-8-sig'))
c = read('config.json')
completed = read('completion.json')
freeze = read('freeze.json')
config_sha = hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest()
assert config_sha == freeze['config_sha256'] == 'a6ff528c87c125e36d8fddfa466d3c7f90cd0f6379fd4baea8774a510a2917a7'
assert completed['failed'] and not completed['complete']
ledger = completed['ledger']
images = sum(x['images_attempted'] for x in ledger['counts'].values())
assert images == 33
assert sum(ledger['successful_updates'].values()) == 0
assert not ledger['attempts'] and not ledger['target_exposures']
assert all(x['exact'] and x['max_abs'] == 0 for x in read('preflight-forward.json'))

# Confirm the five frozen execution sources, not all data/checkpoints.
source_checks=[]
for binding in c['bindings']:
    path=pathlib.Path(binding['path'])
    if path.parent == RUN and path.suffix == '.py':
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == binding['sha256']
        source_checks.append(dict(path=str(path),sha256=digest))

# Exercise the exact class proposed in the patch without importing the worker.
patch=(RUN/'proposed-repair.patch').read_text(encoding='utf-8')
lines=[]; started=False
for line in patch.splitlines():
    if line.startswith('+class ReusablePhase:'):started=True
    if started:
        if line.startswith(' class Ledger:'):break
        if line.startswith('+') and not line.startswith('+++'):lines.append(line[1:])
source='\n'.join(lines)
module=ast.parse(source)
assert len(module.body)==1 and isinstance(module.body[0],ast.ClassDef) and module.body[0].name=='ReusablePhase'
namespace={}
exec(compile(module,'proposed ReusablePhase only','exec'),namespace)
ReusablePhase=namespace['ReusablePhase']

@contextlib.contextmanager
def old_phase(ledger,key):
    old=ledger.scope;ledger.scope=key
    try:yield
    finally:ledger.scope=old

scope=SimpleNamespace(scope='outer')
old=old_phase(scope,'recompute')
with old:assert scope.scope=='recompute'
try:
    with old:pass
except AttributeError as exc:
    reproduced=str(exc)
else:raise AssertionError('Old once-only context unexpectedly reusable')
assert scope.scope=='outer'
new=ReusablePhase(scope,'recompute')
for _ in range(3):
    with new:assert scope.scope=='recompute'
    assert scope.scope=='outer' and not new.stack
with new:
    with new:assert scope.scope=='recompute' and len(new.stack)==2
    assert scope.scope=='recompute' and len(new.stack)==1
assert scope.scope=='outer' and not new.stack
other=ReusablePhase(scope,'nested')
with new:
    with other:assert scope.scope=='nested'
    assert scope.scope=='recompute'
assert scope.scope=='outer'
try:
    with new:raise ValueError('exception propagation check')
except ValueError:pass
else:raise AssertionError('Exception was suppressed')
assert scope.scope=='outer' and not new.stack

caps=c['caps']
audit=dict(config_sha256=config_sha,source_checks=source_checks,
           source_and_trace_support_root_cause=True,original_error_reproduced=reproduced,
           exact_proposed_class_stdlib_tests_passed=True,full_Torch_preflight_passed=False,
           actual_image_forwards=images,actual_optimizer_attempts=0,actual_optimizer_updates=0,
           original_caps=caps,remaining_original_preflight=caps['preflight_image_forwards']-images,
           full_recheck_expected_forwards=45,
           full_recheck_expected_cumulative_preflight=images+45,
           expected_over_original_preflight_cap=images+45-caps['preflight_image_forwards'],
           one_retry_with_unchanged_64_cap_cumulative_preflight_max=images+64,
           one_retry_with_unchanged_caps_cumulative_all_image_forwards_max=images+caps['all_image_forwards'],
           A_native_targets=4096,B_native_targets=4096,B_extra_auxiliary_targets=1024,
           A_training_image_forwards=4096,B_training_image_forwards_including_recompute=10240,
           equal_total_supervision=False,equal_total_compute=False,
           actual_worker_seconds=completed['wall_seconds'],own_CPU_seconds=completed['own_CPU_seconds'],
           repair_applied=False,models_imported_or_run_by_review=0)
(P/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_checks','original_caps']},ensure_ascii=False,indent=2))
