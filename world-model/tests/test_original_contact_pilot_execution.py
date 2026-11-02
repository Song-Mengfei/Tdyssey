"""Pure control logic and IO identity checks; no real engine or model calls."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
from tdyssey.original_contact_pilot import (
    PilotPolicy, execution_io_paths, normalize_letters, selected_slots,
    storage_snapshot, save_attempt, load_attempt, collect, RULES, identity,
    POLICY_EXECUTION_VERSION,
)


def contact(arc, x, outward):
    return dict(arc_id=arc, point_local=[x, 0.], outward_normal_local=[outward, 0.],
                drive_tangent_ratio=.35)


def policy():
    state = np.zeros(15)
    state[:4] = [180., 256., 256., 256.]
    environment = SimpleNamespace(state=lambda: state.copy())
    slot = dict(letter='O', scenario='translation_nearCOM', initial_space='center',
                arc_start=0, selected_contact=contact(0, -40., -1.))
    return PilotPolicy(environment, slot), state


class AcquisitionTest(unittest.TestCase):
    def test_refresh_routes_far_new_target_and_retains_departure(self):
        p, state = policy()
        p.awaiting_acquisition = False
        p.lost = 3
        chosen = contact(1, 40., 1.)
        with patch('tdyssey.original_contact_pilot.select_contact', return_value=chosen), \
             patch('tdyssey.original_contact_pilot._stage', return_value=(np.array([329., 256.]), np.array([311.5, 256.]), 33.)), \
             patch.object(p, 'route', return_value=[[180., 256.], [180., 200.], [329., 256.]]):
            action, decision = p.next_action(5)
        self.assertEqual(decision['mode'], 'free_route')
        self.assertTrue(decision['acquisition_required'])
        self.assertEqual(p.contact['arc_id'], 1)
        self.assertEqual(p.departure_contact['arc_id'], 0)
        self.assertEqual(decision['departure_source_point_local'], [-40., 0.])
        self.assertAlmostEqual(action[0], 180.)
        self.assertLess(action[1], 256.)

    def test_incidental_old_contact_cannot_acquire_new_target(self):
        p, state = policy()
        p.contact = contact(1, 40., 1.)
        def observed(local):
            event = dict(kind='agent_object', phase='post_solve', shapes_order=['object', 'agent'],
                         normal=[1., 0.], total_impulse=[1., 0.], points=[dict(object_local=local)])
            return dict(substeps=[dict(dt=.1, executed=True, post_state=state.tolist(), contacts=[event])])
        p.observe(state, state, observed([-40., 0.]))
        self.assertTrue(p.awaiting_acquisition)
        p.observe(state, state, observed([40., 0.]))
        self.assertFalse(p.awaiting_acquisition)
        self.assertEqual(p.departure_contact['arc_id'], 1)
        self.assertAlmostEqual(p.last_selected_contact_seconds, .1)

    def test_acquisition_does_not_refresh_while_valid_route_is_in_progress(self):
        p, state = policy()
        p.awaiting_acquisition = True
        p.lost = 12
        with patch('tdyssey.original_contact_pilot.select_contact') as refresh, \
             patch('tdyssey.original_contact_pilot._stage', return_value=(np.array([329., 256.]), np.array([311.5, 256.]), 33.)), \
             patch.object(p, 'route', return_value=[[180., 256.], [180., 200.], [329., 256.]]):
            _, decision = p.next_action(5)
        refresh.assert_not_called()
        self.assertEqual(decision['mode'], 'free_route')

    def test_old_normal_bridge_is_used_if_direct_staging_is_blocked(self):
        p, state = policy()
        start, target = np.array([180., 256.]), np.array([329., 256.])
        free = SimpleNamespace(covers=lambda line: True)
        with patch('tdyssey.original_contact_pilot.free_route', side_effect=lambda geometry, a, b: None if np.array_equal(a, start) else [list(a), list(b)]), \
             patch('tdyssey.original_contact_pilot._departure_route', side_effect=lambda geometry, a, b: b[0] < 180.), \
             patch('tdyssey.original_contact_pilot._free', return_value=free):
            route = p.route(object(), start, target)
        self.assertIsNotNone(route)
        self.assertEqual(p.last_route_kind, 'retained_contact_normal_departure_bridge')
        self.assertLess(route[1][0], start[0])
        self.assertEqual(route[-1], target.tolist())


class PartitionIOTest(unittest.TestCase):
    def _workspace_fixture(self):
        root = Path(__file__).resolve().parents[1]
        directory = (root/'artifacts/original-physics-contact-pilot-20261030/collection/revisions/cal2-acquisition-io-v2/test-tmp').resolve()
        if not directory.is_relative_to((root/'artifacts').resolve()):
            raise RuntimeError('Synthetic temporary cleanup must remain inside project artifacts')
        directory.mkdir(parents=True, exist_ok=True)
        return TemporaryDirectory(dir=directory)

    def _execution_fixture(self, folder, two_slots=False):
        root = Path(__file__).resolve().parents[1]
        frozen = json.loads((root/'artifacts/original-physics-contact-pilot-20261030/collection/plan.json').read_text())
        slot = deepcopy(next(r for r in frozen['slots'] if r['phase'] == 'calibration-v2' and r['letter'] == 'T'))
        slots = [slot]
        if two_slots:
            second = deepcopy(slot)
            second['slot_id'] += '-second'
            second['family_id'] += '-second'
            slots.append(second)
        folder = Path(folder)
        plan = folder/'plan.json'
        plan.write_text(json.dumps(dict(rules=RULES, rules_sha256=identity(RULES), slots=slots)))
        gate = folder/'gate.json'
        gate.write_text(json.dumps(dict(passed=True, training_authorized=False, protocol_sha256='synthetic-no-engine')))
        revision = folder/'policy.json'
        revision.write_text(json.dumps(dict(changes={}, policy_execution_version=POLICY_EXECUTION_VERSION)))
        resource = folder/'resources.json'
        resource.write_bytes((root/'artifacts/original-physics-contact-pilot-20261030/policies/resource-policy-v2.json').read_bytes())
        source = folder/'fake_physics_source.py'
        source.write_text('# A synthetic test double: no actual simulator\n')
        fake = SimpleNamespace(PROTOCOL_SHA256='synthetic-no-engine', __file__=str(source))
        return plan, gate, revision, resource, slot, fake

    def test_stop_file_is_graceful_before_any_scene_and_run_cannot_retry(self):
        with self._workspace_fixture() as folder:
            plan, gate, revision, resource, slot, fake = self._execution_fixture(folder)
            fake.make_contact_safe_env = unittest.mock.Mock(side_effect=AssertionError('No engine call permitted'))
            stop = Path(folder)/'STOP'
            stop.write_text('requested')
            output = Path(folder)/'collection'
            with patch('tdyssey.original_contact_pilot.importlib.import_module', return_value=fake):
                result = collect(plan, output, 'calibration-v2', gate, folder, [folder],
                    rules_path=revision, letters='T', run_label='stop-test', resource_policy=resource, stop_file=stop)
            self.assertTrue(result['graceful_stop'])
            self.assertFalse(result['completed'])
            self.assertFalse(result['failed'])
            self.assertEqual(result['actual_controls'], 0)
            self.assertEqual(result['unstarted_slots'], [slot['slot_id']])
            fake.make_contact_safe_env.assert_not_called()
            identity_doc = json.loads((output/'runs/stop-test/identity.json').read_text())
            self.assertEqual(identity_doc['letters'], ['T'])
            self.assertEqual(identity_doc['run_label'], 'stop-test')
            self.assertTrue(identity_doc['resource_policy_sha256'])
            with self.assertRaises(FileExistsError):
                collect(plan, output, 'calibration-v2', gate, folder, [folder], rules_path=revision,
                        letters='T', run_label='stop-test', resource_policy=resource, stop_file=stop)

    def test_substantive_failure_persists_partial_then_stops_before_next_slot(self):
        with self._workspace_fixture() as folder:
            plan, gate, revision, resource, slot, fake = self._execution_fixture(folder, two_slots=True)
            initial = np.asarray(slot['initial_state15'], np.float64)
            class NoPhysicsEnv:
                last_substeps = []
                def reset_scene(self, **kwargs):
                    return np.zeros((3, 96, 96), np.uint8)
                def state(self):
                    return initial.copy()
                def initial_identity(self):
                    return dict(kind='synthetic-no-engine')
                def step(self, action):
                    raise RuntimeError('synthetic failure before any real substep')
                def render_clean(self):
                    return np.zeros((3, 96, 96), np.uint8)
            fake.make_contact_safe_env = unittest.mock.Mock(return_value=NoPhysicsEnv())
            output = Path(folder)/'collection'
            with patch('tdyssey.original_contact_pilot.importlib.import_module', return_value=fake):
                with self.assertRaisesRegex(RuntimeError, 'Substantive collection failure'):
                    collect(plan, output, 'calibration-v2', gate, folder, [folder], rules_path=revision,
                            letters='T', run_label='failure-test', resource_policy=resource)
            fake.make_contact_safe_env.assert_called_once()
            result = json.loads((output/'runs/failure-test/completion.json').read_text())
            self.assertTrue(result['failed'])
            self.assertFalse(result['completed'])
            self.assertEqual(result['actual_attempts'], 1)
            self.assertEqual(result['actual_substeps'], 0)
            self.assertEqual(len(result['unstarted_slots']), 1)
            metadata = json.loads((output/'trajectories'/slot['slot_id']/'metadata.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertEqual(metadata['actual_valid_controls'], 0)
            self.assertTrue((output/'trajectories'/slot['slot_id']/'raw.npz').exists())

    def test_labels_isolate_all_mutable_run_files_and_reject_escape(self):
        with TemporaryDirectory() as folder:
            first = execution_io_paths(folder, 'formal', 'worker-a')
            second = execution_io_paths(folder, 'formal', 'worker-b')
            legacy = execution_io_paths(folder, 'formal')
            for key in ('identity', 'ledger', 'events', 'progress', 'completion'):
                self.assertNotEqual(first[key], second[key])
                self.assertTrue(first[key].is_relative_to(Path(folder).resolve()/'runs'/'worker-a'))
            self.assertEqual(legacy['ledger'], Path(folder).resolve()/'ledger.json')
            for invalid in ('../escape', 'x/y', 'con', 'lpt1', ''):
                with self.assertRaises(ValueError):
                    execution_io_paths(folder, 'formal', invalid)

    def test_compact_letter_partitions_cover_frozen_formal_once(self):
        fixture = json.loads((Path(__file__).resolve().parents[1]/'artifacts/original-physics-contact-pilot-20261030/collection/plan.json').read_text())
        left = selected_slots(fixture, 'formal', 'TLO')
        right = selected_slots(fixture, 'formal', 'CXB')
        a, b = {r['slot_id'] for r in left}, {r['slot_id'] for r in right}
        self.assertFalse(a & b)
        self.assertEqual(len(a), 180)
        self.assertEqual(len(b), 180)
        self.assertEqual(a | b, {r['slot_id'] for r in selected_slots(fixture, 'formal', None)})
        self.assertEqual(len(selected_slots(fixture, 'calibration-v2', None)), 18)
        for invalid in ('TT', 'T,L', 'AZ', ''):
            with self.assertRaises(ValueError):
                normalize_letters(invalid)

    def test_nested_root_pruning_keeps_identical_file_coverage(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'nested').mkdir()
            (root/'a').write_bytes(b'123')
            (root/'nested'/'b').write_bytes(b'12345')
            snapshot = storage_snapshot([root, root/'nested', root/'nested'/'b', root/'a'])
            self.assertEqual(snapshot['files'], 2)
            self.assertEqual(snapshot['bytes'], 8)
            self.assertEqual(snapshot['scanned_roots'], [str(root.resolve())])

    def test_reserved_save_retains_original_numeric_and_rgb_values(self):
        class Reservation:
            def __init__(self):
                self.calls = []
            def write_bytes(self, path, payload):
                self.calls.append((path, len(payload)))
                Path(path).write_bytes(payload)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'raw.npz'
            reserved = Reservation()
            states = np.zeros((2, 15), np.float64)
            states[1, 2] = np.nextafter(1., 2.)
            rgb = np.arange(2*3*96*96, dtype=np.uint8).reshape(2, 3, 96, 96)
            save_attempt(path, states, [[12., 34.]], [True], rgb,
                         [dict(valid=True, substeps=[])], [dict(phase='push')], reservation=reserved)
            actual = load_attempt(path)
            self.assertEqual(len(reserved.calls), 1)
            self.assertTrue(np.array_equal(actual['states'].view(np.uint64), states.view(np.uint64)))
            self.assertTrue(np.array_equal(actual['rgb'], rgb))
            self.assertEqual(actual['requested_actions'].tolist(), [[12., 34.]])


if __name__ == '__main__':
    unittest.main()
