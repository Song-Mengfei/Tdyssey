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
STAGED = OUTPUT / 'v6-source-under-test'
APP = REPO / 'apps/r3b-playground'
sys.path[:0] = [str(STAGED), str(REPO)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

SOURCE_BEFORE = {name: sha(STAGED/name) for name in ('server.py', 'capture.py', 'app.js', 'index.html', 'test_playground.py', 'README.txt')}
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
assert physical.PROTOCOL_SHA256.startswith('2ef7a956')
assert sha(physical.__file__) == '768118e17ba69df4cca57c7041eaf84ecfd760466c1315451e134673585120bf'
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


FIXED_TICK_EVIDENCE = {}
class FixedTickStateTests(unittest.TestCase):
    def test_direct_restore_rejects_missing_or_non_v6_ticks_before_body_mutation(self):
        import copy
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        saved = capture.physics_snapshot(app.env)
        self.assertEqual(app.env.initial_identity()['protocol']['integration_step_tick_choices'], [1])
        for value in ['missing', None, 0, 2, 3, 4, 8, True, 1.0]:
            bad = copy.deepcopy(saved)
            if value == 'missing':
                bad['controller'].pop('integration_microstep_ticks')
            else:
                bad['controller']['integration_microstep_ticks'] = value
            bad['snapshot']['object_body']['pose']['position'] = (40., 60.)
            before = capture.physics_snapshot(app.env)
            before_calls = cost['physics_attempted_controls']
            with self.assertRaisesRegex(ValueError, 'integration-controller'):
                capture.restore_physics(app.env, bad)
            self.assertEqual(capture.physics_snapshot(app.env), before)
            self.assertEqual(cost['physics_attempted_controls'], before_calls)
        FIXED_TICK_EVIDENCE['direct_invalid_rejection'] = dict(
            allowed_ticks=[1], checked_invalid=['missing', None, 0, 2, 3, 4, 8, True, 1.0],
            before_body_write=True, added_controls=0)

    def test_integrated_restore_rejects_missing_and_old_tick_choices_before_reset(self):
        import base64
        import copy
        import numpy as np
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        exported = app.export_session()
        metadata, arrays = capture.unpack(base64.b64decode(exported['archive_base64']))
        before = dict(env=id(app.env), session=app.session, steps=app.steps,
                      revision=app.revision, total=app.total_controls,
                      state=app.env.state().copy(), history=np.stack(app.history).copy(),
                      past=np.stack(app.past).copy(), control=app.env.control_step,
                      ticks=app.env._last_micro_ticks,
                      fixture_calls=cost['deterministic_spy_calls'],
                      physical_calls=cost['physics_attempted_controls'])
        for value in ['missing', None, 2, 4, 8, True, 1.0]:
            bad = copy.deepcopy(metadata)
            if value == 'missing':
                bad['physical_state']['controller'].pop('integration_microstep_ticks')
            else:
                bad['physical_state']['controller']['integration_microstep_ticks'] = value
            bad['physical_state']['snapshot']['object_body']['pose']['position'] = (40., 60.)
            malformed = capture.session_export(bad, arrays)
            with self.assertRaisesRegex(ValueError, 'current session was preserved'):
                app.restore_session(malformed)
            self.assertEqual(id(app.env), before['env'])
            self.assertEqual(app.session, before['session'])
            self.assertEqual(app.steps, before['steps'])
            self.assertEqual(app.revision, before['revision'])
            self.assertEqual(app.total_controls, before['total'])
            self.assertEqual(app.env.control_step, before['control'])
            self.assertEqual(app.env._last_micro_ticks, before['ticks'])
            np.testing.assert_array_equal(app.env.state(), before['state'])
            np.testing.assert_array_equal(np.stack(app.history), before['history'])
            np.testing.assert_array_equal(np.stack(app.past), before['past'])
            self.assertEqual(cost['deterministic_spy_calls'], before['fixture_calls'])
            self.assertEqual(cost['physics_attempted_controls'], before['physical_calls'])
        FIXED_TICK_EVIDENCE['integrated_invalid_rejection'] = dict(
            checked_invalid=['missing', None, 2, 4, 8, True, 1.0],
            before_reset=True, body_history_controller_session_preserved=True,
            added_controls=0, added_spy_calls=0)

    def test_actual_fixed_200_steps_public_state_roundtrip_and_limits(self):
        import base64
        import numpy as np
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        app.step(app.session, app.revision, [511., 0.], 1)
        rows = app.info['substeps']
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(row['dt'] == .0005 for row in rows))
        self.assertAlmostEqual(sum(row['dt'] for row in rows), .1)
        self.assertEqual(app.env._last_micro_ticks, 1)
        exported = app.export_session()
        self.assertFalse(exported['physics_state_complete'])
        self.assertTrue(exported['public_physics_state_complete'])
        self.assertFalse(exported['native_solver_and_bias_state_complete'])
        self.assertFalse(exported['solver_cache_serialized'])
        metadata, _ = capture.unpack(base64.b64decode(exported['archive_base64']))
        self.assertEqual(metadata['physical_state']['controller']['integration_microstep_ticks'], 1)
        self.assertIn('hidden position-bias velocities', metadata['physical_state']['restore_limitation'])
        saved_state = app.env.state().copy()
        saved_history = np.stack(app.history).copy()
        app.reset('C', 'center')
        self.assertEqual(app.env._last_micro_ticks, 1)
        app.restore_session(exported)
        self.assertEqual(app.env._last_micro_ticks, 1)
        np.testing.assert_array_equal(app.env.state(), saved_state)
        np.testing.assert_array_equal(np.stack(app.history), saved_history)
        FIXED_TICK_EVIDENCE['actual_fixed_dt_public_roundtrip'] = dict(
            fixed_request=[511., 0.], microsteps=len(rows), dt=.0005,
            physical_seconds=sum(row['dt'] for row in rows), last_ticks=1, restored_ticks=1,
            exact_public_state15=True, exact_history4=True,
            physics_state_complete=False, public_physics_state_complete=True,
            native_solver_and_bias_state_complete=False, solver_cache_serialized=False,
            future_contact_bitwise_replay_claimed=False)

suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromModule(tests),
    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(FixedTickStateTests),
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
    (OUTPUT/'CPU-v6-check-output.txt').write_text(stream.getvalue(), encoding='utf-8')
    after = {name: sha(STAGED/name) for name in SOURCE_BEFORE}
    live_after = {name: sha(APP/name) for name in LIVE_BEFORE}
    report = dict(
        kind='staged-contact-safe-playground-CPU-check.v6',
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
        note='CPU environment controls include initialization warmups, restores and fixture resets. Deterministic Spy/custom arrays are not neural inference. Existing nine interface plus two handler-concurrency checks, two command-contract checks and three fixed-v6 public-state/migration checks; no active server/browser or actual model used.',
        fixed_tick_state_evidence=FIXED_TICK_EVIDENCE,
        open_source_issue=None,
    )
    write(OUTPUT/'CPU-v6-check.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('staged_source_SHA256_before','staged_source_SHA256_after')}, ensure_ascii=False, indent=2))
if exception or not report['passed']:
    raise SystemExit(1)
