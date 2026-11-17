"""Source checks for privileged region weights and logged frame scores."""

import math
import unittest

from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.gpc_denoiser_training import _windows, score_frames, summarize_score_rows
from tdyssey.region_weight import (DEFAULT_REGION_WEIGHT, foreground_weights, motion_mask,
                                   pusher_mask, region_weights, window_case_region,
                                   window_region, window_useful)
from tdyssey.state import BodyState, Snapshot
from tdyssey.trajectories import Trajectory, Transition

try:
    import numpy as np
except ImportError:
    np = None


def official_state(position=(256, 256), angle=0.0, pusher=(100, 256)):
    pose = Pose(position, angle)
    return Snapshot("T", official_shape("T").geometry_hash, "a" * 64,
                    BodyState(pose), BodyState(Pose(pusher, 0)), Pose((256, 256), math.pi / 4), 0)


def world_pixel(x, y):
    return int((x * 96) / 512), int((y * 96) / 512)


class RegionWeightChecks(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("NumPy is required for region-weight checks")

    def test_t_and_pusher_cover_the_bodies_and_not_the_whole_frame(self):
        start = official_state()
        moved = official_state(position=(260, 256), pusher=(110, 256))
        region = window_region(start, moved)
        stem_x, stem_y = world_pixel(256, 280)
        pusher_x, pusher_y = world_pixel(100, 256)
        self.assertTrue(region[stem_y, stem_x])
        self.assertTrue(region[pusher_y, pusher_x])
        self.assertFalse(region[2, 2])
        self.assertLess(region.mean(), 0.35)
        self.assertGreater(region.mean(), 0.02)
        disk = pusher_mask((100, 256))
        self.assertTrue(disk[pusher_y, pusher_x])
        self.assertFalse(disk[2, 2])

    def test_region_weight_scales_one_pixel_eight_times(self):
        mask = np.zeros((96, 96), dtype=bool)
        mask[10, 20] = True
        weights = region_weights(mask, 8)
        self.assertEqual(weights[10, 20], 8)
        self.assertEqual(weights[10, 21], 1)
        self.assertEqual(weights[0, 0], 1)
        empty = region_weights(np.zeros((96, 96), dtype=bool), 8)
        self.assertTrue(np.array_equal(empty, np.ones((96, 96), dtype=np.float32)))
        self.assertEqual(DEFAULT_REGION_WEIGHT, 8)

    def test_score_frames_reports_region_motion_and_pose_labels(self):
        target = np.zeros((3, 96, 96), dtype=np.float64)
        still = np.zeros((3, 96, 96), dtype=np.float64)
        predicted = np.zeros((3, 96, 96), dtype=np.float64)
        target[:, 40, 40] = 1
        predicted[:, 40, 40] = 0
        still[:, 40, 40] = 1
        target[:, 50, 50] = 1
        region = np.zeros((96, 96), dtype=bool)
        region[40, 40] = True
        scored = score_frames(predicted, target, still, region, translation=2.5, rotation=0.1)
        self.assertEqual(scored["object_roi_mse"], 1)
        self.assertEqual(scored["object_roi_persistence"], 0)
        self.assertLess(scored["mse"], scored["object_roi_mse"])
        self.assertGreater(scored["motion_mse"], 0)
        self.assertEqual(scored["translation"], 2.5)
        self.assertEqual(scored["rotation"], 0.1)
        self.assertIn("object_centroid_error_pixels", scored)
        self.assertTrue(motion_mask(target, still)[50, 50])
        self.assertFalse(motion_mask(target, still)[40, 40])
        self.assertFalse(motion_mask(target, target)[50, 50])

    def test_window_useful_matches_the_contact_sampler_rule(self):
        self.assertTrue(window_useful(True, 0))
        self.assertTrue(window_useful(False, 2))
        self.assertFalse(window_useful(False, 0))

    def test_case_region_keeps_still_t_on_a_miss_and_covers_motion_on_a_hit(self):
        start = official_state()
        miss = official_state(pusher=(110, 256))
        hit = official_state(position=(280, 256), pusher=(110, 256))
        miss_mask = window_case_region(start, miss, moved=False, n_contacts=0)
        stem_x, stem_y = world_pixel(256, 280)
        table_x, table_y = world_pixel(20, 20)
        self.assertTrue(miss_mask[stem_y, stem_x])
        self.assertFalse(miss_mask[table_y, table_x])
        next_image = np.zeros((3, 96, 96), dtype=np.float64)
        last_image = np.zeros((3, 96, 96), dtype=np.float64)
        next_image[:, 10, 10] = 1
        useful_mask = window_case_region(start, hit, next_image, last_image, moved=True, n_contacts=0)
        new_x, new_y = world_pixel(280, 280)
        self.assertTrue(useful_mask[10, 10])
        self.assertTrue(useful_mask[new_y, new_x])
        self.assertFalse(useful_mask[table_y, table_x])
        weights = foreground_weights(miss_mask)
        self.assertEqual(weights[stem_y, stem_x], 1)
        self.assertEqual(weights[table_y, table_x], 0)
        empty = foreground_weights(np.zeros((96, 96), dtype=bool))
        self.assertTrue(np.array_equal(empty, np.ones((96, 96), dtype=np.float32)))

    def test_score_summary_splits_useful_and_miss_windows(self):
        rows = [
            {"useful": True, "motion_mse": 0.04, "motion_persistence": 0.2,
             "object_centroid_error_pixels": 2.0, "predicted_centroid_shift_pixels": 3.0,
             "actual_centroid_shift_pixels": 4.0, "mse": 0.001},
            {"useful": False, "motion_mse": None, "object_roi_mse": 0.01,
             "object_roi_persistence": 0.0, "predicted_centroid_shift_pixels": 0.5,
             "actual_centroid_shift_pixels": 0.0, "mse": 0.002},
        ]
        summary = summarize_score_rows(rows)
        self.assertEqual(summary["useful_windows"], 1)
        self.assertEqual(summary["miss_windows"], 1)
        self.assertEqual(summary["useful"]["motion_mse"], 0.04)
        self.assertEqual(summary["miss"]["object_roi_mse"], 0.01)
        self.assertEqual(summary["useful"]["predicted_centroid_shift_pixels"], 3.0)

    def test_windows_record_n_contacts_and_the_useful_flag(self):
        images = np.zeros((5, 3, 96, 96), dtype=np.float32)
        arrays = {"images": np.zeros((5, 3, 96, 96), dtype=np.uint8)}
        arrays["images"][4, :, 10, 10] = 255
        states = []
        pusher = (100.0, 256.0)
        obj = (256.0, 256.0)
        for step in range(5):
            target = None if step == 0 else (pusher[0] + 1.0, pusher[1])
            if step:
                pusher = (pusher[0] + 1.0, pusher[1])
            states.append(official_state(position=obj, pusher=pusher))
            states[-1] = Snapshot(states[-1].shape, states[-1].geometry_hash, states[-1].physics_hash,
                                  states[-1].object_body, states[-1].pusher_body, states[-1].goal,
                                  step, target)
        transitions = []
        for step in range(4):
            transitions.append(Transition(states[step], states[step + 1].pusher_target, states[step + 1],
                                          0.0, False, step == 3))
        trajectory = Trajectory("ep", "test", "1", 0, 0.1, tuple(transitions))
        record = {"step_diagnostics": [{"n_contacts": 0}, {"n_contacts": 0},
                                       {"n_contacts": 0}, {"n_contacts": 3}]}
        window = _windows(trajectory, arrays, record)[0]
        self.assertEqual(window["n_contacts"], 3)
        self.assertTrue(window["useful"])
        self.assertFalse(window["moved"])
        miss = _windows(trajectory, arrays, {"step_diagnostics": [{"n_contacts": 0}] * 4})[0]
        self.assertFalse(miss["useful"])
        self.assertEqual(miss["n_contacts"], 0)


if __name__ == "__main__":
    unittest.main()
