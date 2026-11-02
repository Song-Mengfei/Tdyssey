"""CPU fixtures for direct targets and illegal-substep preservation; no physics."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey.direct_target_collection import DirectTargetOriginalEnv, save_raw, valid_windows
from tdyssey.native_rgb_original_physics import OriginalPhysicsEnv
from tdyssey.strict_original_contact_pilot import OriginalStateInvalid


def fixture_env():
    env = DirectTargetOriginalEnv.__new__(DirectTargetOriginalEnv)
    env._base = SimpleNamespace(control_step=0)
    env._reset = True
    env._terminal = False
    env._in_step = False
    env.last_substeps = []
    env.state = lambda: np.zeros(15, np.float64)
    env.render_clean = lambda: np.zeros((3, 96, 96), np.uint8)
    return env


class DirectTargetCollectionFixtures(unittest.TestCase):
    def test_original_endpoints_reach_upstream_unchanged(self):
        env = fixture_env()
        observed = []

        def upstream(base, target):
            observed.append(target.copy())
            env.last_substeps = [dict(executed=True, dt=.01) for _ in range(10)]

        env._upstream = SimpleNamespace(step=upstream)
        for target in ([0., 512.], [512., 0.], [511.75, .25]):
            env.step(target)
            self.assertTrue(np.array_equal(observed[-1], target))
            self.assertTrue(np.array_equal(env.last_passed_action, target))
        self.assertFalse(env._in_step)
        self.assertEqual(env.control_step, 3)

    def test_illegal_original_substep_stops_immediately_and_keeps_row(self):
        env = fixture_env()
        env._point_region = lambda: 'outside'

        def observed_original(instance, dt):
            index = len(instance.last_substeps)
            instance.last_substeps.append(dict(index=index, executed=True, dt=dt,
                post_state=np.zeros(15).tolist(), penetrations=[0., 1.01 if index == 2 else .1, 0.]))

        with patch.object(OriginalPhysicsEnv, '_observe_step', observed_original):
            with self.assertRaises(OriginalStateInvalid) as raised:
                for _ in range(10):
                    env._observe_step(.01)
        self.assertEqual(len(env.last_substeps), 3)
        self.assertEqual(raised.exception.info['first_invalid_substep'], 2)
        self.assertFalse(env.last_substeps[-1]['valid'])
        self.assertTrue(env.last_substeps[-1]['executed'])

    def test_complete_history_and_future_never_bridge_invalid_or_time_gap(self):
        mask = np.ones(64, bool)
        times = np.arange(65)*.1
        segments = np.zeros(65, np.int64)
        self.assertEqual([len(valid_windows(mask, times, segments, h)) for h in (1, 4, 8, 16)],
                         [61, 58, 54, 46])
        mask[10] = False
        for h in (1, 4, 8, 16):
            self.assertTrue(all(not (t-3 <= 10 < t+h) for t in valid_windows(mask, times, segments, h)))
        mask[:] = True
        times[20:] += .01
        self.assertTrue(all(not (t-3 < 20 <= t+16) for t in valid_windows(mask, times, segments, 16)))

    def test_saved_partial_retains_real_time_and_rejects_action_relabeling(self):
        states = np.zeros((5, 15))
        actions = np.asarray([[0., 512.], [512., 0.], [80., 80.], [300., 300.]])
        images = np.zeros((5, 3, 96, 96), np.uint8)
        infos = [dict(substeps=[dict(dt=.01, executed=True) for _ in range(n)]) for n in (10, 10, 10, 3)]

        class Lease:
            def write_bytes(self, path, data):
                Path(path).write_bytes(data)

        with TemporaryDirectory() as directory:
            path = Path(directory)/'partial.npz'
            save_raw(path, states, actions, actions.copy(), [True, True, True, False],
                     images, infos, [], Lease())
            with np.load(path, allow_pickle=False) as raw:
                self.assertEqual(raw['rgb'].shape[0], 5)
                self.assertAlmostEqual(raw['control_time_seconds'][-1], .33)
                self.assertEqual(raw['control_valid'].tolist(), [True, True, True, False])
            altered = actions.copy()
            altered[0, 1] = 511.
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                save_raw(path, states, actions, altered, [True]*4, images, infos, [], Lease())


if __name__ == '__main__':
    unittest.main()
