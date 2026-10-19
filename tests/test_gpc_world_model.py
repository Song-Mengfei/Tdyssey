"""Pretrained dynamics must align executed actions with true frame histories."""

import unittest

from tdyssey.gpc_world_model import action_windows


class GPCWorldModelTimingChecks(unittest.TestCase):
    def test_proposed_controls_enter_only_their_own_prediction_and_later(self):
        executed = [(10, 11), (20, 21), (30, 31)]
        proposed = [(40, 41), (50, 51), (60, 61)]
        windows = action_windows(executed, proposed)
        self.assertEqual(windows, [
            [(10, 11), (20, 21), (30, 31), (40, 41)],
            [(20, 21), (30, 31), (40, 41), (50, 51)],
            [(30, 31), (40, 41), (50, 51), (60, 61)],
        ])
        changed = action_windows(executed, proposed[:1] + [(99, 99)] + proposed[2:])
        self.assertEqual(changed[0], windows[0])
        self.assertNotEqual(changed[1], windows[1])
        self.assertEqual(executed, [(10, 11), (20, 21), (30, 31)])


if __name__ == "__main__":
    unittest.main()
