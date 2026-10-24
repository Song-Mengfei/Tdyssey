"""Source checks for official T angle-probe side labels and slices."""

import unittest

from tdyssey.angle_probe import (not_better_than_still, slice_stats, summarize_cases,
                                 world_side, worse_object_iou)


def case(side, contacts, distance, worse, still, missing=False):
    return {"world_side": side, "n_contacts": contacts, "distance": distance,
            "worse_object_iou": worse, "still_object_iou": still, "stable_missing": missing}


class AngleProbeChecks(unittest.TestCase):
    def test_world_side_uses_object_cog_x(self):
        self.assertEqual(world_side((100, 200), (150, 200)), "left")
        self.assertEqual(world_side((150, 10), (150, 200)), "right")
        self.assertEqual(world_side((200, 0), (150, 200)), "right")

    def test_worse_iou_is_the_lower_seed(self):
        self.assertEqual(worse_object_iou([0.9, 0.4]), 0.4)

    def test_still_frame_tie_counts_as_not_better(self):
        self.assertTrue(not_better_than_still(0.8, 0.8))
        self.assertFalse(not_better_than_still(0.81, 0.8))

    def test_summarize_splits_left_contact_and_far_pushes(self):
        rows = [
            case("left", 2, 20.0, 0.4, 0.5, missing=False),
            case("left", 0, 150.0, 0.2, 0.9, missing=True),
            case("right", 1, 20.0, 0.9, 0.3, missing=False),
            case("right", 0, 150.0, 0.85, 0.8, missing=False),
        ]
        summary = summarize_cases(rows)
        self.assertEqual(summary["all"]["count"], 4)
        self.assertEqual(summary["all"]["stable_missing"], 1)
        self.assertEqual(summary["all"]["not_better_than_still"], 2)
        self.assertEqual(summary["left"]["count"], 2)
        self.assertEqual(summary["left"]["min_worse_iou"], 0.2)
        self.assertEqual(summary["contact"]["count"], 2)
        self.assertEqual(summary["no_contact"]["count"], 2)
        self.assertEqual(summary["distance_20"]["mean_worse_iou"], 0.65)
        self.assertEqual(summary["distance_150"]["stable_missing"], 1)
        self.assertEqual(slice_stats([])["count"], 0)
