"""Saved-state counterexample and synthetic contact evidence; zero dynamics."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from tdyssey import strict_original_contact_pilot as collector
from tdyssey.strict_original_target_policy_v2 import RequestedTargetPolicyV2, install_frozen_policy, POLICY_VERSION
from tdyssey.original_contact_pilot import load_attempt, file_sha
from tdyssey.native_rgb_r2b_sampling import contact_frame

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'artifacts/strict-original-contact-pilot-20261031/collection/trajectories/SOP-calibration-v1-calibration-T-000'


def saved_case(cls):
    meta = json.loads((CASE/'metadata.json').read_text())
    raw = load_attempt(CASE/'raw.npz')
    cursor = dict(state=raw['states'][0].copy())
    env = SimpleNamespace(state=lambda: cursor['state'].copy())
    return cls(env, meta['slot']), cursor, raw


def selected_evidence(policy, state, count):
    _, normal, _ = contact_frame(state, policy.contact)
    event = dict(kind='agent_object', phase='post_solve', shapes_order=['object', 'agent'],
        normal=normal.tolist(), total_impulse=[1., 0.], points=[dict(object_local=policy.contact['point_local'])])
    return dict(substeps=[dict(executed=True, dt=.01, post_state=state.tolist(),
        contacts=[deepcopy(event)] if index >= 10-count else []) for index in range(10)])


class PolicyV2Test(unittest.TestCase):
    def test_actual_saved_cal1_staging_reversal_reproduces_v1_and_is_absent_v2(self):
        v1, first, raw = saved_case(collector.RequestedTargetPolicy)
        v2, second, _ = saved_case(RequestedTargetPolicyV2)
        action1, decision1 = v1.next_action(0)
        action2, decision2 = v2.next_action(0)
        self.assertTrue(np.allclose(action1, raw['requested_actions'][0], rtol=0, atol=1e-12))
        self.assertTrue(np.array_equal(action1, action2))
        self.assertEqual(decision2['mode'], 'contact_entry_intent')
        first['state'] = second['state'] = raw['states'][1].copy()
        old_state = second['state'].copy()
        reverse_action, reverse_decision = v1.next_action(1)
        entry_action, entry_decision = v2.next_action(1)
        self.assertTrue(np.allclose(reverse_action, raw['requested_actions'][1], rtol=0, atol=1e-12))
        self.assertEqual(reverse_decision['mode'], 'free_route')
        self.assertEqual(entry_decision['mode'], 'contact_entry_intent')
        self.assertTrue(entry_decision['entry_started'])
        self.assertTrue(np.array_equal(entry_action, action2))
        self.assertTrue(np.array_equal(old_state, second['state']))
        self.assertEqual(entry_action.tolist(), entry_decision['requested_action'])
        self.assertFalse(entry_decision['execution_intervention'])

    def test_entry_stays_latched_until_actual_selected_positive_50ms_and_final_contact(self):
        policy, cursor, raw = saved_case(RequestedTargetPolicyV2)
        initial_action, _ = policy.next_action(0)
        cursor['state'] = raw['states'][1].copy()
        policy.observe(cursor['state'], cursor['state'], selected_evidence(policy, cursor['state'], 4))
        _, decision = policy.next_action(1)
        self.assertTrue(policy.awaiting_acquisition)
        self.assertEqual(decision['mode'], 'contact_entry_intent')
        policy.observe(cursor['state'], cursor['state'], selected_evidence(policy, cursor['state'], 5))
        self.assertFalse(policy.awaiting_acquisition)
        _, decision = policy.next_action(2)
        self.assertEqual(decision['mode'], 'contact_push_intent')
        self.assertFalse(decision['acquisition_required'])

    def test_new_contact_refresh_preserves_old_confirmed_departure_source(self):
        policy, cursor, raw = saved_case(RequestedTargetPolicyV2)
        old = deepcopy(policy.departure_contact)
        policy.awaiting_acquisition = False
        policy.entry_started = True
        policy.lost = 3
        fresh = deepcopy(old)
        fresh['arc_id'] += 1
        fresh['point_local'] = [float(old['point_local'][0])+50., float(old['point_local'][1])]
        with patch('tdyssey.strict_original_target_policy_v2.saved.select_contact', return_value=fresh), \
             patch('tdyssey.strict_original_target_policy_v2._stage', return_value=(np.array([400., 400.]), np.array([390., 390.]), 33.)), \
             patch.object(policy, 'route', return_value=[cursor['state'][:2].tolist(), [300., 300.]]):
            _, decision = policy.next_action(2)
        self.assertEqual(policy.contact['arc_id'], fresh['arc_id'])
        self.assertEqual(policy.departure_contact, old)
        self.assertFalse(policy.entry_started)
        self.assertTrue(decision['acquisition_required'])
        self.assertEqual(decision['departure_source_point_local'], old['point_local'])

    def test_outside_staging_routes_first_then_entry_latches_when_staging_is_reached(self):
        policy, cursor, raw = saved_case(RequestedTargetPolicyV2)
        _, normal, _ = contact_frame(cursor['state'], policy.contact)
        cursor['state'][:2] += normal*20.
        _, decision = policy.next_action(0)
        self.assertEqual(decision['mode'], 'free_route')
        self.assertFalse(policy.entry_started)
        cursor['state'] = raw['states'][0].copy()
        _, decision = policy.next_action(1)
        self.assertEqual(decision['mode'], 'contact_entry_intent')
        self.assertTrue(policy.entry_started)
        _, decision = policy.next_action(30)
        self.assertFalse(policy.entry_started)
        self.assertEqual(decision['phase'], 'withdraw')

    def test_process_local_install_requires_actual_implementation_sha_and_keeps_v1_source(self):
        fixture_root = ROOT/'artifacts/strict-original-contact-pilot-20261031/test-fixtures'
        fixture_root.mkdir(exist_ok=True)
        module = ROOT/'tdyssey/strict_original_target_policy_v2.py'
        old_collector_sha = file_sha(collector.__file__)
        with TemporaryDirectory(dir=fixture_root) as folder:
            policy = Path(folder)/'policy.json'
            gate = Path(folder)/'gate.json'
            value = dict(policy_execution_version=POLICY_VERSION,
                implementation_module='tdyssey.strict_original_target_policy_v2',
                implementation_source=str(module), implementation_source_sha256=file_sha(module))
            policy.write_text(json.dumps(value))
            gate.write_text(json.dumps(dict(bindings={str(module): file_sha(module)})))
            with patch.object(collector, 'RequestedTargetPolicy', collector.RequestedTargetPolicy), \
                 patch.object(collector, 'POLICY_VERSION', collector.POLICY_VERSION):
                evidence = install_frozen_policy(policy, gate)
                self.assertIs(collector.RequestedTargetPolicy, RequestedTargetPolicyV2)
                self.assertEqual(collector.POLICY_VERSION, POLICY_VERSION)
                self.assertEqual(evidence['implementation_source_sha256'], file_sha(module))
                value['implementation_source_sha256'] = 'wrong'
                policy.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, 'not frozen'):
                    install_frozen_policy(policy, gate)
        self.assertEqual(file_sha(collector.__file__), old_collector_sha)


if __name__ == '__main__':
    unittest.main()
