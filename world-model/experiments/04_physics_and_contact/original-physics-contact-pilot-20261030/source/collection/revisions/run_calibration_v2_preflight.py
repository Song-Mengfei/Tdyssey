"""Actual pure tests and source receipts; never collect data or start workers."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent
COLLECTION = OUT.parent
modules = ('tests.test_original_contact_pilot_execution',
           'tests.test_original_contact_pilot', 'tests.test_original_contact_pilot_resources')
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
def flatten(value):
    for item in value:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()
names = list(flatten(suite))
stream = io.StringIO()
with redirect_stdout(stream), redirect_stderr(stream):
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
log = OUT/'calibration-v2-preflight-tests.log'
log.write_text(stream.getvalue(), encoding='utf-8')
files = ('tdyssey/original_contact_pilot.py', 'scripts/original_contact_pilot.py',
         'tdyssey/original_contact_pilot_report.py', 'tdyssey/original_contact_pilot_resources.py',
         'tdyssey/native_rgb_contact_safe_physics.py',
         'tests/test_original_contact_pilot_execution.py', 'tests/test_original_contact_pilot.py',
         'tests/test_original_contact_pilot_resources.py')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources = {str((REPO/name).resolve()): sha(REPO/name) for name in files}
policy = COLLECTION/'policy-revision-v2.json'
receipt = dict(kind='actual-calibration-v2-pure-preflight.v1',
    checked_at_utc=datetime.now(timezone.utc).isoformat(), passed=result.wasSuccessful(),
    tests_run=result.testsRun, test_names=names,
    failures=[dict(test=t.id(), traceback=trace) for t, trace in result.failures],
    errors=[dict(test=t.id(), traceback=trace) for t, trace in result.errors],
    skipped=[dict(test=t.id(), reason=reason) for t, reason in result.skipped],
    actual_output_log=str(log), actual_output_log_sha256=sha(log),
    sources=sources, source_sha256=sources,
    calibration_policy=str(policy), calibration_policy_sha256=sha(policy),
    resource_policy=str(COLLECTION.parent/'policies/resource-policy-v2.json'),
    resource_policy_sha256=sha(COLLECTION.parent/'policies/resource-policy-v2.json'),
    physics_steps=0, model_calls=0, optimizer_updates=0,
    evidence_scope='Synthetic state/geometry/control logic, temporary filesystem, actual OS locking/process IO and tiny saved-frame CPU video only. No simulator engine, learned model or production worker executes.',
    earlier_fixture_failure='Two initial lifecycle fixtures failed WinError5 atomic replace in the sandbox system Temp; moved those isolated fixtures under checked workspace artifacts and reran. No real data or physics occurred.',
    calibration_v1_source_sha256='5018cb25e716ddf1b4331263f14225aff6b7d7e50dd64b0a3b59c0229210a233',
    calibration_v1_preserved=True, calibration_v2_started=False)
path = OUT/'calibration-v2-preflight.json'
path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(passed=receipt['passed'], tests_run=result.testsRun,
                     failures=len(result.failures), errors=len(result.errors),
                     receipt=str(path), sources=sources,
                     calibration_policy=str(policy))))
raise SystemExit(0 if result.wasSuccessful() else 1)
