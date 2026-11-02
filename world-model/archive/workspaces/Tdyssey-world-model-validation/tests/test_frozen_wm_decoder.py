"""Synthetic-only checks; no simulator imports, reset, render or physics calls."""

import json
import math
import unittest

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from tdyssey.frozen_wm_decoder import CONFIG_SHA256, calibration_summary, decode
except ImportError:
    cv2 = None

from tdyssey.envs.official import official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.geometry import Pose


def synthetic_rgb(shape_name, pose, *, pusher=(60, 60), red=False, malformed=False):
    """Rasterize independent polygon vertices; never instantiate an environment."""
    shape = official_shape(shape_name)
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    goal = Pose((256, 256), math.pi/4)
    for polygon in shape.polygons(goal):
        draw.polygon([(round(x), round(y)) for x, y in polygon], fill=(144, 238, 144))
    color = (220, 35, 35) if red else (119, 136, 153)
    if malformed:
        # Same rough area as L/T, but no expected letter structure.
        draw.rectangle((200, 200, 275, 275), fill=color)
    else:
        for polygon in shape.polygons(pose):
            vertices = [(round(x), round(y)) for x, y in polygon]
            light = tuple(min(255, round(v*1.2)) for v in color)
            draw.polygon(vertices, fill=light)
            draw.line(vertices+[vertices[0]], fill=color, width=4)
            for x, y in vertices:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    if pusher is not None:
        x, y = pusher
        draw.ellipse((x-15, y-15, x+15, y+15), fill=(65, 105, 225))
        draw.ellipse((x-11, y-11, x+11, y+11), fill=(78, 126, 255))
    # Match the documented resize sampling, but keep independent PIL polygons.
    # No official renderer call, environment, or physical calibration is used.
    return cv2.resize(np.asarray(image), (96, 96), interpolation=cv2.INTER_LINEAR)


@unittest.skipUnless(cv2 is not None, "Installed image/scientific dependencies required; synthetic only")
class FrozenImageDecoderChecks(unittest.TestCase):
    def test_known_shape_origin_and_wrap_from_synthetic_gray_or_red(self):
        cases = [("L", Pose((240, 245), math.radians(179)), False),
                 ("T", Pose((250, 210), math.radians(-42)), True)]
        for name, pose, red in cases:
            with self.subTest(shape=name):
                prediction = decode(synthetic_rgb(name, pose, red=red), name, pose)
                self.assertTrue(prediction["valid"], prediction)
                self.assertLess(math.dist(prediction["pose"]["position"], pose.position), 4)
                self.assertLess(math.degrees(official_shape(name).angle_error(prediction["pose"]["angle"], pose.angle)), 4)
                self.assertGreater(prediction["coverage"], .90)
                self.assertLess(math.dist(prediction["pusher"], (60, 60)), 3)
                self.assertFalse(prediction["contact_proxy"])
                json.dumps(prediction, allow_nan=False)

    def test_goal_changes_score_but_cannot_guide_pose(self):
        pose = Pose((240, 245), math.radians(15))
        image = synthetic_rgb("L", pose)
        first = decode(image, "L", pose)
        other = decode(image, "L", Pose((430, 430), 0))
        self.assertTrue(first["valid"], first)
        self.assertEqual(first["pose"], other["pose"])
        self.assertEqual(first["pusher"], other["pusher"])
        self.assertGreater(first["coverage"], .9)
        self.assertEqual(other["coverage"], 0)

    def test_missing_or_wrong_shape_is_abstention_without_truth_repair(self):
        pose = Pose((240, 245), 0)
        for image in (np.full((96, 96, 3), 255, dtype=np.uint8),
                      synthetic_rgb("L", pose, pusher=None),
                      synthetic_rgb("L", pose, malformed=True)):
            prediction = decode(image, "L", pose)
            self.assertFalse(prediction["valid"], prediction)
            self.assertIsNone(prediction["coverage"])
            self.assertIsNone(prediction["pose"])
        nonfinite = np.zeros((96, 96, 3), dtype=float)
        nonfinite[0, 0, 0] = np.nan
        self.assertEqual(decode(nonfinite, "L", pose)["reason"], "invalid_rgb")

    def test_forecast_clipping_is_local_and_does_not_modify_raw_array(self):
        pose = Pose((240, 245), 0)
        raw = synthetic_rgb("L", pose).astype(np.float32)/255
        raw[0, 0] = (-.2, 1.2, .5)
        before = raw.copy()
        result = decode(np.moveaxis(raw, -1, 0), "L", pose)
        self.assertTrue(result["valid"], result)
        self.assertGreater(result["clipped_fraction"], 0)
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(result["decoder_config_sha256"], CONFIG_SHA256)

    def test_calibration_missing_rows_remain_in_gate_denominator(self):
        pose = Pose((240, 245), 0)
        valid = {"valid": True, "pose": {"position": [240, 245], "angle": 0},
                 "pusher": [60, 60], "coverage": 1.0}
        truth = {"pose": {"position": [240, 245], "angle": 0}, "coverage": 1.0, "pusher": [60, 60]}
        rows = [{"shape_name": "L", "prediction": valid, "truth": truth} for _ in range(11)]
        rows.append({"shape_name": "L", "prediction": {"valid": False}, "truth": truth})
        result = calibration_summary(rows)["L"]
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["missing"], 1)
        self.assertAlmostEqual(result["measurable_fraction"], 11/12)
        self.assertFalse(result["passed"])
        self.assertFalse(result["ranking_available"])
        self.assertFalse(result["success_precision_established"])
        # The summary consumes truth only as an error label, not a repair path.
        self.assertEqual(goal_metrics(official_shape("L"), pose, pose).coverage, 1)


if __name__ == "__main__":
    unittest.main()
