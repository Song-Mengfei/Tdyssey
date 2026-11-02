"""Fixed-step v6 staged CPU suite, with public-state/migration contracts."""
from pathlib import Path
import hashlib
import shutil

out=Path(__file__).resolve().parent
stage=out.parent/'ui-next'
frozen=out/'v6-source-under-test'
expected={
 'server.py':'a85f312eb4e674771550de6ae9fffb2bfddae843c75643cd3b1fff033a503001',
 'capture.py':'bee4252023719d7169c4a28d9294341749aa17a3fd79ccecece8cdc866fb990a',
 'test_playground.py':'e842d9d69ab84cee3eeec63e94d99f6e4a72b586d982d84503ccee0ab6e70dfd',
 'app.js':'fbc33ab97a92c24b900d314d6dd01742e100e69b5202ab0fb2caefe8a3cc88fa',
 'index.html':'d72e6c4c1936815c2659ae4364a2026f70e35f3ca501a75ded698c2f2953f5da',
}
for name,digest in expected.items():
 assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest,(name,'stage changed before snapshot')
frozen.mkdir(exist_ok=False)
for name in (*expected,'README.txt'):
 shutil.copyfile(stage/name,frozen/name)

code=(out/'run_cpu_check.py').read_text(encoding='utf-8')
code=code.replace("STAGED = PILOT / 'ui-next'","STAGED = OUTPUT / 'v6-source-under-test'")
code=code.replace("('server.py', 'capture.py', 'app.js', 'index.html', 'test_playground.py')", "('server.py', 'capture.py', 'app.js', 'index.html', 'test_playground.py', 'README.txt')")
code=code.replace("assert physical.PROTOCOL_SHA256 == 'e7d6f4db25a912bac0e07e95d83277010e78ac65ca8b6c02ebccd6c123a0aaeb'", "assert physical.PROTOCOL_SHA256.startswith('2ef7a956')\nassert sha(physical.__file__) == '768118e17ba69df4cca57c7041eaf84ecfd760466c1315451e134673585120bf'")
code=code.replace("OUTPUT/'CPU-check-output.txt'","OUTPUT/'CPU-v6-check-output.txt'")
code=code.replace("OUTPUT/'CPU-check.json'","OUTPUT/'CPU-v6-check.json'")
code=code.replace("kind='staged-contact-safe-playground-CPU-check.v1'","kind='staged-contact-safe-playground-CPU-check.v6'")
injection='''
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

'''
code=code.replace('suite = unittest.TestSuite([',injection+'suite = unittest.TestSuite([')
code=code.replace('    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),',
                  '    unittest.defaultTestLoader.loadTestsFromTestCase(CommandContractTests),\n    unittest.defaultTestLoader.loadTestsFromTestCase(FixedTickStateTests),')
code=code.replace("        open_source_issue='restore_session resets before rejecting mismatched environment protocol; reported to parent, not modified by this checker.',",
                  "        fixed_tick_state_evidence=FIXED_TICK_EVIDENCE,\n        open_source_issue=None,")
code=code.replace('Existing nine interface plus two handler-concurrency checks, two contract checks;',
                  'Existing nine interface plus two handler-concurrency checks, two command-contract checks and three fixed-v6 public-state/migration checks;')
generated=out/'run_cpu_v6_generated.py'
generated.write_text(code,encoding='utf-8')
exec(compile(code,str(generated),'exec'),{'__file__':str(generated),'__name__':'__main__'})
