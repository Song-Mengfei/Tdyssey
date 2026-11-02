"""Original-loop observation, invalid-prefix costs and requested-action tests.

Synthetic worlds only: these tests never construct an engine, use a model or
count fake controls as experimental interactions.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import strict_original_contact_pilot as pilot


class MemoryLease:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def write_bytes(self, path, payload):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)


class SimplePolicy:
    def __init__(self, *args):
        pass
    def next_action(self, index):
        return np.array([123., 234.]), dict(phase='push', mode='synthetic-no-engine')
    def observe(self, *args):
        pass


def fixture_slots():
    return [dict(slot_id=f'SYNTHETIC-{index}', phase='calibration-v1', split='calibration',
        letter='O', initial_state15=np.zeros(15).tolist(), initialization={}) for index in range(2)]


class SyntheticEnv:
    def __init__(self, invalid=False, unexpected=False):
        self.index, self.invalid, self.unexpected = 0, invalid, unexpected
        self.last_substeps = []
        self.closed = False
    def state(self):
        return np.zeros(15)
    def reset_scene(self):
        return np.zeros((3, 96, 96), np.uint8)
    def initial_identity(self):
        return dict(kind='synthetic-test-only')
    def render_clean(self):
        return np.zeros((3, 96, 96), np.uint8)
    def close(self):
        self.closed = True
    def step(self, action):
        self.index += 1
        failed = (self.invalid or self.unexpected) and self.index == 3
        count = 3 if failed else 10
        self.last_substeps = [dict(dt=.01, executed=True, post_state=self.state().tolist(),
            valid=not failed or index < count-1, contacts=[], penetrations=[0., 0., 0.]) for index in range(count)]
        if failed:
            error = pilot.OriginalStateInvalid if self.invalid else RuntimeError
            if self.invalid:
                raise error('synthetic illegal original state', dict(valid=False, substeps=self.last_substeps))
            raise error('synthetic unexpected engine failure')
        return self.render_clean(), dict(valid=True, substeps=self.last_substeps)


class StrictOriginalTest(unittest.TestCase):
    def workspace_temp(self):
        path = Path(__file__).resolve().parents[1]/'artifacts/strict-original-contact-pilot-20261031/test-fixtures'
        path.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=path)

    def execute(self, folder, environments, stop=None):
        root = Path(folder)
        for name in ('plan', 'gate', 'policy', 'resources'):
            (root/(name+'.json')).write_text('{}')
        plan = dict(slots=fixture_slots())
        gate = dict(run_id='synthetic-only-no-physics')
        with patch.object(pilot, 'verify_gate', return_value=(plan, gate, pilot.RULES)), \
             patch.object(pilot, 'ResourceReservation', MemoryLease), \
             patch.object(pilot, 'RequestedTargetPolicy', SimplePolicy), \
             patch.object(pilot, 'make_strict_original_env', side_effect=environments):
            return pilot.collect(root/'plan.json', root/'out', 'calibration-v1', root/'gate.json', root, [root],
                policy_path=root/'policy.json', resource_policy=root/'resources.json', run_label='test-run', stop_file=stop)

    def test_first_illegal_substep_aborts_after_original_integrator_without_physics_write(self):
        env = object.__new__(pilot.StrictObservedOriginalEnv)
        env.last_substeps = []
        native_calls = []
        def unchanged_observer(instance, dt):
            native_calls.append(dt)
            instance.last_substeps.append(dict(index=0, executed=True, dt=dt,
                post_state=np.zeros(15).tolist(), penetrations=[0., 1.2, 0.]))
        with patch.object(pilot.original.OriginalPhysicsEnv, '_observe_step', unchanged_observer), \
             patch.object(pilot.StrictObservedOriginalEnv, '_point_region', return_value='outside'):
            with self.assertRaises(pilot.OriginalStateInvalid) as result:
                env._observe_step(.01)
        self.assertEqual(native_calls, [.01])
        self.assertEqual(len(env.last_substeps), 1)
        self.assertTrue(env.last_substeps[0]['executed'])
        self.assertEqual(result.exception.info['first_invalid_substep'], 0)
        self.assertEqual(env.last_substeps[0]['post_state'], np.zeros(15).tolist())

    def test_center_in_solid_or_closed_hole_is_illegal_at_any_depth(self):
        for region in ('solid', 'hole-0'):
            self.assertIn('illegal_pusher_center_region', pilot.state_invalidity(np.zeros(15), [0., 0., 0.], region))
        self.assertFalse(pilot.state_invalidity(np.zeros(15), [1., .1, 0.], 'outside'))

    def test_legal_prefix_and_all_costs_retained_no_scene_replacement(self):
        with self.workspace_temp() as folder:
            environments = [SyntheticEnv(invalid=True), SyntheticEnv()]
            result = self.execute(folder, environments)
            self.assertTrue(result['completed'])
            self.assertFalse(result['all_trajectories_complete'])
            self.assertEqual(result['actual_attempts'], 2)
            self.assertEqual(result['complete_trajectories'], 1)
            self.assertEqual(result['invalid_trajectories'], 1)
            self.assertEqual(result['actual_controls'], 67)
            self.assertEqual(result['actual_valid_controls'], 66)
            self.assertEqual(result['actual_substeps'], 663)
            self.assertAlmostEqual(result['simulated_seconds'], 6.63)
            metadata_path = Path(folder)/'out/trajectories/SYNTHETIC-0/metadata.json'
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata['valid_contiguous_control_ranges'], [[0, 2]])
            self.assertFalse(metadata['training_eligible'])
            raw = pilot.saved.load_attempt(metadata_path.parent/'raw.npz')
            self.assertEqual(raw['control_valid'].tolist(), [True, True, False])
            self.assertTrue(np.array_equal(raw['requested_actions'], np.tile([123., 234.], (3, 1))))
            self.assertTrue(np.allclose(raw['control_time_seconds'], [0., .1, .2, .23]))
            self.assertEqual(raw['frame_indices'].tolist(), [0, 1, 2, 3])
            self.assertTrue(all(env.closed for env in environments))

    def test_unexpected_failure_stops_whole_run_preserving_partial(self):
        with self.workspace_temp() as folder:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected collection failure'):
                self.execute(folder, [SyntheticEnv(unexpected=True), SyntheticEnv()])
            terminal = json.loads((Path(folder)/'out/runs/test-run/completion.json').read_text())
            self.assertTrue(terminal['failed'])
            self.assertEqual(terminal['actual_attempts'], 1)
            self.assertEqual(terminal['actual_substeps'], 23)
            self.assertEqual(terminal['unstarted_slots'], ['SYNTHETIC-1'])

    def test_stop_file_prevents_any_new_scene(self):
        with self.workspace_temp() as folder:
            stop = Path(folder)/'STOP'
            stop.write_text('explicit stop')
            result = self.execute(folder, [], stop)
            self.assertTrue(result['graceful_stop'])
            self.assertEqual(result['actual_attempts'], 0)
            self.assertEqual(result['actual_substeps'], 0)

    def test_disjoint_legal_ranges_cannot_bridge_a_bad_control(self):
        self.assertEqual(pilot.valid_prefix_ranges([True, True, False, True]), [[0, 2], [3, 4]])

    def test_old_v6_gate_rejected_before_factory(self):
        with self.workspace_temp() as folder:
            root = Path(folder)
            (root/'plan.json').write_text(json.dumps(dict(kind='strict-original-contact-independent-scenes.v1',
                rules=pilot.RULES, rules_sha256=pilot.saved.identity(pilot.RULES))))
            (root/'gate.json').write_text(json.dumps(dict(passed=True, training_authorized=False,
                protocol_sha256='old-modified-v6')))
            (root/'policy.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'restoration/identity gate'):
                pilot.verify_gate(root/'plan.json', root/'gate.json', root/'policy.json', 'calibration-v1')

    def test_saved_request_is_actual_planned_target_without_environment_velocity_edit(self):
        state = np.zeros(15)
        state[:4] = [100., 256., 256., 256.]
        env = SimpleNamespace(state=lambda: state.copy())
        slot = dict(letter='O', scenario='translation_nearCOM', initial_space='center', arc_start=0,
            selected_contact=dict(arc_id=0, point_local=[-40., 0.], outward_normal_local=[-1., 0.], drive_tangent_ratio=0.))
        p = pilot.RequestedTargetPolicy(env, slot)
        with patch.object(pilot.saved.PilotPolicy, 'next_action', return_value=(np.array([511., 5.]), {})):
            action, decision = p.next_action(0)
        self.assertEqual(action.tolist(), [488., 23.])
        self.assertEqual(action.tolist(), decision['requested_action'])
        self.assertFalse(decision['execution_intervention'])
        self.assertTrue(np.array_equal(state[5:], np.zeros(10)))

    def test_job_owned_collector_rejected_before_any_scene_or_output(self):
        with self.workspace_temp() as folder:
            root = Path(folder)
            plan = dict(slots=fixture_slots())
            with patch.object(pilot, 'verify_gate', return_value=(plan, {}, pilot.RULES)), \
                 patch.object(pilot, 'process_job_evidence', return_value=dict(is_process_in_job=True)), \
                 patch.object(pilot, 'make_strict_original_env') as factory:
                with self.assertRaisesRegex(RuntimeError, 'outside Windows Job'):
                    pilot.collect(root/'plan', root/'out', 'calibration-v1', root/'gate', root, [root],
                        policy_path=root/'policy', resource_policy=root/'resource', run_label='job-check',
                        run_id='synthetic-only', runtime_config_sha256='synthetic', require_detached=True)
            factory.assert_not_called()
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
