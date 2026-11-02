"""Exercise staged UI with deterministic fixtures and actual CPU physics only."""
from __future__ import annotations
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
OUTPUT = Path(__file__).resolve().parent
PILOT = OUTPUT.parent
REPO = PILOT.parent.parent
STAGED = PILOT / 'ui-next'
APP = REPO / 'apps/r3b-playground'
sys.path[:0] = [str(STAGED), str(REPO)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

SOURCE_BEFORE = {name: sha(STAGED/name) for name in ('server.py', 'capture.py', 'app.js', 'index.html', 'test_playground.py')}
LIVE_BEFORE = {name: sha(APP/name) for name in SOURCE_BEFORE}
import server
server.APP = APP
server.REPO = REPO
server.WORKSPACE = REPO.parent
server.GPC = REPO.parent/'Tdyssey/third_party/gpc_code'
server.PHYSICS = REPO.parent/'Tdyssey/third_party/diffusion_policy'
server.RUN = REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030'
server.CHECKPOINT = server.RUN/'run/last.pt'
server.RUNTIME = REPO/'artifacts/r3b-playground'
server.DIAGNOSTICS = REPO/'artifacts/r3b-playground-diagnostics-20261030'
from tdyssey import native_rgb_contact_safe_physics as physical
assert physical.PROTOCOL_SHA256 == 'e7d6f4db25a912bac0e07e95d83277010e78ac65ca8b6c02ebccd6c123a0aaeb'
PHYSICAL_SOURCE_BEFORE = sha(physical.__file__)

cost = dict(physics_attempted_controls=0, physics_completed_controls=0,
            physics_failed_controls=0, executed_microsteps=0,
            attempted_microstep_rows=0, simulated_seconds=0.,
            model_calls=0, GPU_calls=0, optimizer_updates=0,
            deterministic_spy_calls=0, neural_model_constructed=False)
lock = threading.Lock()
original_step = physical.ContactSafePhysicsEnv.step

def observed_step(self, *args, **kwargs):
    with lock:
        cost['physics_attempted_controls'] += 1
    completed = False
    try:
        result = original_step(self, *args, **kwargs)
        completed = bool(result[1].get('valid', False))
        return result
    finally:
        with lock:
            cost['physics_completed_controls' if completed else 'physics_failed_controls'] += 1
            rows = self.last_substeps
            cost['attempted_microstep_rows'] += len(rows)
            cost['executed_microsteps'] += sum(bool(r.get('executed', False)) for r in rows)
            cost['simulated_seconds'] += sum(r['dt'] for r in rows if r.get('executed', False))

physical.ContactSafePhysicsEnv.step = observed_step
spec = importlib.util.spec_from_file_location('staged_cpu_tests', STAGED/'test_playground.py')
tests = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tests
spec.loader.exec_module(tests)
# Original fixtures use the live file's repo-relative parent for their temporary
# directory. Only this module-global location changes; tested source stays staged.
tests.__file__ = str(APP/'test_playground.py')
original_spy = tests.Spy.__call__
def observed_spy(self, *args, **kwargs):
    with lock:
        cost['deterministic_spy_calls'] += 1
    return original_spy(self, *args, **kwargs)
tests.Spy.__call__ = observed_spy

class CommandContractTests(unittest.TestCase):
    def test_requested_command_and_actual_endpoint_are_distinct_and_saved(self):
        import numpy as np
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        request = np.asarray(app.target) + [20., 0.]
        state = app.step(app.session, app.revision, request, 1)
        record = app.capture.records[-1]
        metadata = record['metadata']
        info = metadata['control_infos'][0]
        np.testing.assert_array_equal(metadata['requested_target'], request)
        np.testing.assert_array_equal(info['requested_action'], request)
        np.testing.assert_array_equal(state['frames'][0]['requested_target'], request)
        np.testing.assert_array_equal(info['executed_target'], app.env.state()[:2])
        self.assertGreater(float(np.linalg.norm(np.asarray(info['executed_target'])-request)), .001)
        self.assertEqual(state['metrics']['physics_substeps'], len(info['substeps']))
        self.assertAlmostEqual(state['metrics']['contact_seconds'], info['agent_object_contact_seconds'])
        self.assertAlmostEqual(sum(row['dt'] for row in info['substeps']), .1)
        self.assertTrue(all('executed_velocity' in row and 'unconstrained_velocity' in row and 'intervention' in row for row in info['substeps']))
        self.assertIn('actual endpoints/velocities', metadata['action_contract'])

    def test_source_labels_use_true_contact_time_and_request_language(self):
        js = (STAGED/'app.js').read_text(encoding='utf-8')
        html = (STAGED/'index.html').read_text(encoding='utf-8')
        self.assertIn("metric('contact',1000*(d.metrics.contact_seconds||0),0,'/ 100 ms')", js)
        self.assertNotIn("metric('contact',d.metrics.contact_substeps,0,'/ 10')", js)
        self.assertIn('本批请求目标', js)
        self.assertIn('接触受阻时实际移动会受限', html)
        self.assertIn('与原始实现的数值求解设置不同', html)

suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromModule(tests),
    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),
])
started = time.perf_counter()
stream = io.StringIO()
result = None
exception = None
try:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
except BaseException as exc:
    exception = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
finally:
    physical.ContactSafePhysicsEnv.step = original_step
    tests.Spy.__call__ = original_spy
    (OUTPUT/'CPU-check-output.txt').write_text(stream.getvalue(), encoding='utf-8')
    after = {name: sha(STAGED/name) for name in SOURCE_BEFORE}
    live_after = {name: sha(APP/name) for name in LIVE_BEFORE}
    report = dict(
        kind='staged-contact-safe-playground-CPU-check.v1',
        completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        passed=bool(result is not None and result.wasSuccessful() and exception is None),
        tests_run=0 if result is None else result.testsRun,
        failures=[] if result is None else [dict(test=str(test), traceback=tb) for test,tb in result.failures],
        errors=[] if result is None else [dict(test=str(test), traceback=tb) for test,tb in result.errors],
        exception=exception, wall_seconds=time.perf_counter()-started, cost=cost,
        staged_source_SHA256_before=SOURCE_BEFORE, staged_source_SHA256_after=after,
        staged_sources_unchanged=after==SOURCE_BEFORE,
        live_sources_unchanged=live_after==LIVE_BEFORE,
        actual_server_source=server.__file__, staged_test_source=str(STAGED/'test_playground.py'),
        physical_source=str(physical.__file__), physical_source_SHA256_before=PHYSICAL_SOURCE_BEFORE,
        physical_source_SHA256_after=sha(physical.__file__), protocol_SHA256=physical.PROTOCOL_SHA256,
        torch_imported='torch' in sys.modules,
        browser_used=False, actual_live_service_operated=False,
        note='CPU environment controls include initialization warmups, restores and fixture resets. Deterministic Spy/custom arrays are not neural inference. Existing nine interface plus two handler-concurrency checks, two contract checks; no active server/browser or actual model used.',
        open_source_issue='restore_session resets before rejecting mismatched environment protocol; reported to parent, not modified by this checker.',
    )
    write(OUTPUT/'CPU-check.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('staged_source_SHA256_before','staged_source_SHA256_after')}, ensure_ascii=False, indent=2))
if exception or not report['passed']:
    raise SystemExit(1)
