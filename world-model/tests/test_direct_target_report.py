"""Saved-only fixtures: no environment construction or physical steps."""
import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np

from tdyssey.direct_target_report import (centers, control_metrics, distribution,
    generate_report, legal_window, select_examples, validate_timeline, window_metrics)


def fixture(n=20):
    states = np.zeros((n+1, 15), np.float64)
    states[:, :4] = 256.
    telemetry = [dict(substeps=[dict(executed=True, dt=.01, pre_state=states[i].tolist(),
        post_state=states[i+1].tolist(), penetrations=[0., 0., 0.], contacts=[]) for _ in range(10)]) for i in range(n)]
    return dict(states=states, requested_actions=np.zeros((n, 2)), passed_actions=np.zeros((n, 2)),
        control_valid=np.ones(n, bool), rgb=np.zeros((n+1, 3, 96, 96), np.uint8),
        frame_indices=np.arange(n+1), segment_ids=np.zeros(n+1, np.int64),
        control_time_seconds=np.arange(n+1)*.1, telemetry=telemetry)


class DirectTargetSavedReportTests(unittest.TestCase):
    def setUp(self):
        self.attempt = dict(attempt_id='fixture', family_id='F', letter='T', scenario='translation',
                            all_actions_precommitted=True)

    def test_history_horizons_and_no_invalid_bridge(self):
        data = fixture()
        rows, _, _ = control_metrics(self.attempt, data)
        windows = window_metrics(self.attempt, data, rows)
        self.assertEqual({h: sum(w['horizon'] == h for w in windows) for h in (1, 4, 8, 16)},
                         {1: 17, 4: 14, 8: 10, 16: 2})
        self.assertEqual({h: sum(w['horizon'] == h and w['common_H16_anchor'] for w in windows) for h in (1, 4, 8, 16)},
                         {1: 2, 4: 2, 8: 2, 16: 2})
        self.assertTrue(legal_window(data, 3, 16))
        data['control_valid'][10] = False
        self.assertFalse(legal_window(data, 3, 16))
        data['control_valid'][10] = True
        data['segment_ids'][7:] = 1
        self.assertFalse(legal_window(data, 3, 16))

    def test_centroid_rotation_and_round_trip_motion(self):
        data = fixture()
        initial = centers('T', data['states'])[0]
        data['states'][1, 4] = math.pi
        rotated = centers('T', data['states'])[1]
        self.assertGreater(np.linalg.norm(rotated-initial), 80.)
        data['states'][:, 4] = 0.
        data['states'][4:12, 2] += np.arange(1, 9)
        data['states'][12:20, 2] += np.arange(7, -1, -1)
        rows, _, _ = control_metrics(self.attempt, data)
        window = next(w for w in window_metrics(self.attempt, data, rows) if w['anchor'] == 3 and w['horizon'] == 16)
        self.assertEqual(window['translation_world_px'], 0.)
        self.assertEqual(window['cumulative_translation_world_px'], 16.)
        self.assertEqual(window['maximum_translation_world_px'], 8.)

    def test_raw_rotation_does_not_remove_full_turns(self):
        data = fixture()
        data['states'][:, 4] = np.arange(21)*2*math.pi
        rows, _, _ = control_metrics(self.attempt, data)
        window = next(w for w in window_metrics(self.attempt, data, rows) if w['horizon'] == 1)
        self.assertAlmostEqual(window['rotation_deg'], 360.)
        self.assertAlmostEqual(window['symmetry_equivalent_rotation_deg'], 0.)

    def test_request_equals_passed_and_static_no_false_positive_video(self):
        data = fixture()
        self.assertEqual(validate_timeline(data), 20)
        data['passed_actions'][0, 0] = 1
        with self.assertRaises(ValueError):
            validate_timeline(data)
        rows, _, _ = control_metrics(self.attempt, data)
        windows = window_metrics(self.attempt, data, rows)
        examples = select_examples([self.attempt], rows, windows)
        self.assertFalse(any(e['available'] for e in examples))

    def test_contact_and_push_requires_contact(self):
        data = fixture(1)
        data['states'][1, 2] += 1.
        substep = dict(executed=True, dt=.01, post_state=data['states'][1].tolist(),
                       penetrations=[0., 0., 0.], contacts=[])
        data['telemetry'][0]['substeps'] = [substep]
        rows, _, _ = control_metrics(self.attempt, data)
        self.assertEqual(rows[0]['contact_and_push_seconds'], 0.)
        substep['contacts'] = [dict(phase='post_solve', kind='agent_object', total_impulse=[1., 0.],
                                   shapes_order=['agent', 'object'], points=[])]
        rows, _, _ = control_metrics(self.attempt, data)
        self.assertAlmostEqual(rows[0]['AO_positive_seconds'], .01)
        self.assertAlmostEqual(rows[0]['contact_and_push_seconds'], .01)
        substep['contacts'][0]['total_impulse'] = [0., 0.]
        rows, _, _ = control_metrics(self.attempt, data)
        self.assertEqual(rows[0]['AO_positive_seconds'], 0.)

    def test_complete_saved_only_report_and_decoded_media(self):
        # Workspace temp avoids a Windows sandbox temp-directory rename issue.
        with tempfile.TemporaryDirectory(prefix='.direct-target-report-fixture-', dir=Path.cwd()/'artifacts') as temp:
            root = Path(temp)
            (root/'collection/raw').mkdir(parents=True)
            (root/'collection/attempts').mkdir()
            (root/'policies').mkdir()
            def write(path, value):
                path.write_text(json.dumps(value), encoding='utf-8')
            policy = dict(collection_cap_bytes=256*1024**2, shared_cap_bytes=3*1024**3,
                atomic_reserve_bytes=256*1024**2, per_attempt_atomic_reserve_bytes=16*1024**2,
                maximum_concurrent_collectors=2, concurrent_attempt_reserve_bytes=32*1024**2,
                postprocess_reserve_bytes=128*1024**2, training_authorized=False)
            write(root/'policies/resource-policy.json', policy)
            write(root/'storage-roots.json', {'roots': []})
            write(root/'runtime-config.json', {'run_id': 'SYNTHETIC-SAVED-FIXTURE-NO-PHYSICS'})
            attempts, costs = [], []
            for letter in 'TLOCXB':
                attempt = dict(attempt_id='fixture-'+letter, family_id='fixture-family-'+letter,
                    letter=letter, scenario='static', all_actions_precommitted=True,
                    actions=[[0., 0.]]*20, initialization={'angle': 0.})
                attempts.append(attempt)
                data = fixture()
                payload = json.dumps(dict(telemetry=data.pop('telemetry'), decisions=[{'phase': 'hold'}]*20)).encode()
                data.update(telemetry_zlib=np.frombuffer(zlib.compress(payload), np.uint8),
                    telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(), np.uint8))
                path = root/'collection/raw'/f"{attempt['attempt_id']}.npz"
                np.savez_compressed(path, **data)
                cost = dict(attempt_id=attempt['attempt_id'], status='completed', actual_controls=20,
                    actual_valid_controls=20, actual_substeps=200, simulated_seconds=2.,
                    initialization_integrator_calls=0, wall_seconds=.1)
                costs.append(cost)
                write(root/'collection/attempts'/f"{attempt['attempt_id']}.json", dict(complete=True,
                    cost=cost, raw={'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
            write(root/'plan.json', {'attempts': attempts})
            write(root/'collection/ledger.json', dict(attempts=costs, actual_controls=120,
                actual_valid_controls=120, actual_substeps=1200, simulated_seconds=12.))
            write(root/'collection/completion.json', {'wall_seconds': .6})
            report = generate_report(root)
            self.assertEqual(report['all']['executed_controls'], 120)
            self.assertEqual(report['all']['completed'], 6)
            self.assertTrue(report['cost']['audit_control_count_matches_ledger'])
            self.assertEqual(report['media_unique_full_attempt_videos'], 2)
            self.assertEqual(len(report['unavailable']), 30)
            self.assertEqual(json.loads((root/'resource-reservations.json').read_text())['leases'][0]['status'], 'released')
            self.assertTrue((root/'report/final-manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
