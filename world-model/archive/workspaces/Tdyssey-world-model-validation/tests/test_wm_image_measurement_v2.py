"""Mathematical raster fixtures only: no environment or learned-model calls."""

import builtins
import importlib.util
import math
import unittest
from unittest.mock import patch

from tdyssey.geometry import Pose

DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "scipy", "PIL", "cv2"))


@unittest.skipUnless(DEPS, "installed image/scientific dependencies required")
class MeasurementV2Checks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from test_frozen_wm_decoder import synthetic_rgb
        from tdyssey.wm_image_measurement_v2 import decode
        self.np, self.synthetic, self.decode = np, synthetic_rgb, decode
        self.pose = Pose((256, 256), 0)

    def test_missing_pusher_does_not_erase_object_or_coverage(self):
        result = self.decode(self.synthetic("L", self.pose, pusher=None), "L", self.pose)
        self.assertTrue(result["state"]["valid"], result)
        self.assertTrue(result["shape"]["valid"])
        self.assertTrue(result["coverage"]["valid"])
        self.assertFalse(result["pusher"]["valid"])
        self.assertFalse(result["state"]["ambiguity_assessed"])
        self.assertIsNone(result["state"]["ambiguity"])

    def test_multiple_meaningful_pushers_abstain_locally(self):
        image = self.synthetic("L", self.pose)
        image[5:18, 77:90] = image[5:18, 5:18]
        result = self.decode(image, "L", self.pose)
        self.assertFalse(result["pusher"]["valid"], result)
        self.assertEqual(result["pusher"]["reason"], "ambiguous_multiple_pushers")
        self.assertEqual(len(result["pusher"]["meaningful_components"]), 2)
        self.assertTrue(result["state"]["valid"])
        self.assertTrue(result["coverage"]["valid"])

    def test_missing_object_preserves_independent_pusher(self):
        image = self.synthetic("L", self.pose)
        image[25:70, 25:75] = 255
        result = self.decode(image, "L", self.pose)
        self.assertFalse(result["state"]["valid"])
        self.assertFalse(result["coverage"]["valid"])
        self.assertTrue(result["pusher"]["valid"], result)

    def test_complete_template_penalizes_missing_arm_and_wrong_shape(self):
        for image in (self.synthetic("L", self.pose, malformed=True), self.synthetic("L", self.pose)):
            if not self.np.array_equal(image, self.synthetic("L", self.pose, malformed=True)):
                image[50:62, 50:65] = 255  # Remove the foot from a pure array fixture.
            result = self.decode(image, "L", self.pose)
            self.assertFalse(result["shape"]["valid"], result)
            self.assertFalse(result["coverage"]["valid"])
            self.assertIn("full_soft_iou", result["shape"])
            self.assertLessEqual(result["shape"]["full_soft_iou"], result["shape"]["roi_soft_iou"]+1e-7)

    def test_goal_affects_coverage_only_and_no_true_state_is_an_input(self):
        image = self.synthetic("L", Pose((240, 245), .2))
        first = self.decode(image, "L", Pose((240, 245), .2))
        second = self.decode(image, "L", Pose((420, 420), 1.0))
        self.assertTrue(first["state"]["valid"], first)
        self.assertEqual(first["state"], second["state"])
        self.assertEqual(first["pusher"], second["pusher"])
        self.assertGreater(first["coverage"]["estimate"], second["coverage"]["estimate"])
        with self.assertRaises(TypeError):
            self.decode(image, "L", self.pose, true_pose=self.pose)

    def test_wrapped_angles_with_gray_and_red_geometry(self):
        for shape, degrees, red in (("L", 179, False), ("T", -42, True)):
            from tdyssey.wm_measurement_contract import angle_error_degrees
            pose = Pose((240, 245), math.radians(degrees))
            result = self.decode(self.synthetic(shape, pose, red=red), shape, pose)
            self.assertTrue(result["state"]["valid"], result)
            self.assertLess(math.dist(result["state"]["pose"]["position"], pose.position), 4)
            self.assertLess(angle_error_degrees(result["state"]["pose"]["angle"], pose.angle), 4)

    def test_fractional_colors_survive_when_binary_seed_is_identical(self):
        from tdyssey.wm_image_measurement_v2 import continuous_maps
        one = self.np.full((96, 96, 3), 255., dtype=float)
        two = one.copy()
        foreground = self.np.array([143., 163., 184.])
        one[40, 40] = 255*.88+foreground*.12
        two[40, 40] = 255*.84+foreground*.16
        first, _, _ = continuous_maps(one)
        second, _, _ = continuous_maps(two)
        self.np.testing.assert_array_equal(first >= .2, second >= .2)
        self.assertNotEqual(float(first[40, 40]), float(second[40, 40]))

    def test_raw_array_unchanged_and_runtime_packages_never_imported(self):
        array = self.synthetic("L", self.pose).astype(self.np.float32)/255
        array[0, 0] = [-.1, 1.1, .5]
        before = array.copy()
        array.flags.writeable = False
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "pymunk", "pygame", "diffusion_policy"}:
                raise AssertionError("Runtime import in offline measurement: " + name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            result = self.decode(array, "L", self.pose)
        self.np.testing.assert_array_equal(array, before)
        self.assertGreater(result["clipped_fraction"], 0)
        self.assertFalse(result["coverage"]["is_probability"])


if __name__ == "__main__":
    unittest.main()
