"""Versioned rerun of the staged CPU suite, preserving earlier results."""
from pathlib import Path
import hashlib
import shutil

out=Path(__file__).resolve().parent
stage=out.parent/'ui-next'
frozen=out/'v5-source-under-test'
expected={
 'server.py':'b15638eb21916a2db6a0a259e617e00796e092db698ee1a950113ae4384ccbc4',
 'capture.py':'2ceef359795a6885c97285fa89c5c1acab0392b7082a4717772d4db273c5a3cc',
 'test_playground.py':'39eff55a6aa8febb5851c8546130ee0b88ddcacb9d7a2b0d468f4aa4bfc47ac5',
}
for name,digest in expected.items():
 assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest,(name,'stage changed before snapshot')
frozen.mkdir(exist_ok=False)
for name in('server.py','capture.py','test_playground.py','app.js','index.html'):
 shutil.copyfile(stage/name,frozen/name)

code=(out/'run_cpu_check.py').read_text(encoding='utf-8')
code=code.replace("STAGED = PILOT / 'ui-next'","STAGED = OUTPUT / 'v5-source-under-test'")
code=code.replace('e7d6f4db25a912bac0e07e95d83277010e78ac65ca8b6c02ebccd6c123a0aaeb',
                  'd95d0cd5ae39ee23b86311303ad8ea58b5b832e2526534ebbb96da1770ad574a')
code=code.replace("OUTPUT/'CPU-check-output.txt'","OUTPUT/'CPU-v5-check-output.txt'")
code=code.replace("OUTPUT/'CPU-check.json'","OUTPUT/'CPU-v5-check.json'")
code=code.replace("kind='staged-contact-safe-playground-CPU-check.v1'","kind='staged-contact-safe-playground-CPU-check.v5'")
injection='''
TICK_STATE_EVIDENCE = {}
class TickStateTests(unittest.TestCase):
    def test_missing_or_invalid_ticks_reject_before_direct_body_mutation(self):
        import copy
        import numpy as np
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        saved = capture.physics_snapshot(app.env)
        invalid = ['missing', None, 0, 3, True, 8.0]
        for value in invalid:
            bad = copy.deepcopy(saved)
            if value == 'missing':
                bad['controller'].pop('integration_microstep_ticks')
            else:
                bad['controller']['integration_microstep_ticks'] = value
            # If the body-write loop ran before checking ticks, this position
            # would expose the mutation rather than accidentally restoring the
            # same value and hiding the ordering error.
            bad['snapshot']['object_body']['pose']['position'] = (40., 60.)
            before = capture.physics_snapshot(app.env)
            with self.assertRaisesRegex(ValueError, 'integration-controller'):
                capture.restore_physics(app.env, bad)
            self.assertEqual(capture.physics_snapshot(app.env), before)
        TICK_STATE_EVIDENCE['direct_missing_invalid_rejection'] = dict(
            checked_values=['missing', None, 0, 3, True, 8.0],
            before_body_write=True, additional_controls=0)

    def test_same_protocol_missing_ticks_preserves_complete_playground_session(self):
        import base64
        import copy
        import numpy as np
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        exported = app.export_session()
        metadata, arrays = capture.unpack(base64.b64decode(exported['archive_base64']))
        metadata['physical_state']['controller'].pop('integration_microstep_ticks')
        metadata['physical_state']['snapshot']['object_body']['pose']['position'] = (40., 60.)
        malformed = capture.session_export(metadata, arrays)
        before = dict(env=id(app.env), session=app.session, steps=app.steps,
                      revision=app.revision, total=app.total_controls,
                      state=app.env.state().copy(), history=np.stack(app.history).copy(),
                      past=np.stack(app.past).copy(), control=app.env.control_step,
                      ticks=app.env._last_micro_ticks,
                      fixture_calls=cost['deterministic_spy_calls'],
                      physical_calls=cost['physics_attempted_controls'])
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
        TICK_STATE_EVIDENCE['same_protocol_missing_ticks_integrated_rejection'] = dict(
            before_reset=True, body_history_controller_session_preserved=True,
            added_controls=0, added_spy_calls=0)

    def test_actual_variable_dt_last_tick_state_roundtrips(self):
        import base64
        import numpy as np
        import capture
        app = server.Playground(tests.Spy())
        self.addCleanup(app.env.close)
        app.step(app.session, app.revision, [511., 0.], 1)
        rows = app.info['substeps']
        tick = physical.PROTOCOL['integration_tick_seconds']
        observed_ticks = [int(round(row['dt']/tick)) for row in rows]
        self.assertTrue(set(observed_ticks) <= {1, 2, 4, 8})
        self.assertGreater(len(set(observed_ticks)), 1, 'Fixed preregistered fast target did not exercise real variable dt')
        saved_ticks = app.env._last_micro_ticks
        self.assertEqual(saved_ticks, observed_ticks[-1])
        exported = app.export_session()
        metadata, _ = capture.unpack(base64.b64decode(exported['archive_base64']))
        self.assertEqual(metadata['physical_state']['controller']['integration_microstep_ticks'], saved_ticks)
        saved_state = app.env.state().copy()
        saved_history = np.stack(app.history).copy()
        app.reset('C', 'center')
        reset_ticks = app.env._last_micro_ticks
        self.assertNotEqual(reset_ticks, saved_ticks, 'Fixture must change the tick state before restoring it')
        app.restore_session(exported)
        self.assertEqual(app.env._last_micro_ticks, saved_ticks)
        np.testing.assert_array_equal(app.env.state(), saved_state)
        np.testing.assert_array_equal(np.stack(app.history), saved_history)
        self.assertFalse(exported['solver_cache_serialized'])
        TICK_STATE_EVIDENCE['actual_variable_dt_roundtrip'] = dict(
            fixed_request=[511., 0.], microsteps=len(rows),
            observed_tick_choices=sorted(set(observed_ticks)),
            final_saved_ticks=saved_ticks, reset_ticks=reset_ticks,
            restored_ticks=app.env._last_micro_ticks,
            exact_state15=True, exact_history4=True,
            solver_contact_cache_serialized=False)

'''
code=code.replace('suite = unittest.TestSuite([',injection+'suite = unittest.TestSuite([')
code=code.replace('    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),',
                  '    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),\n    unittest.defaultTestLoader.loadTestsFromTestCase(TickStateTests),')
code=code.replace("        open_source_issue='restore_session resets before rejecting mismatched environment protocol; reported to parent, not modified by this checker.',",
                  "        tick_state_evidence=TICK_STATE_EVIDENCE,\n        open_source_issue=None,")
code=code.replace('Existing nine interface plus two handler-concurrency checks, two contract checks;',
                  'Existing nine interface plus two handler-concurrency checks, two command-contract checks and three integration-tick/migration-state checks;')
generated=out/'run_cpu_v5_generated.py'
generated.write_text(code,encoding='utf-8')
exec(compile(code,str(generated),'exec'),{'__file__':str(generated),'__name__':'__main__'})
