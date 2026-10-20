"""Palette rendering must preserve physics and the official policy image stream."""

from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set the installed official source path")
class AppearanceRuntimeChecks(unittest.TestCase):
    def test_auxiliary_palette_preserves_state_observations_and_tracks_moved_objects(self):
        import numpy as np
        from tdyssey.appearance import source_palette_image, prediction_metrics
        from tdyssey.envs.official import make_official_shape_env_class
        from tdyssey.geometry import Pose
        from tdyssey.official import load_official_source
        from tdyssey.state import BodyState

        for key in ("SDL_VIDEODRIVER", "SDL_AUDIODRIVER"):
            os.environ.setdefault(key, "dummy")
        env = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))(shape="T", legacy=True)
        self.addCleanup(env.close)
        env.seed(400000)
        env.reset()
        scene = replace(env.capture_state(), object_body=BodyState(Pose((250,220), .1)),
                        pusher_body=BodyState(Pose((80,80),0)))
        original = env.restore_state(scene)["image"].copy()
        before = env.capture_state()
        palette = source_palette_image(env)
        self.assertEqual(before, env.capture_state())
        np.testing.assert_array_equal(original, env._get_obs()["image"])
        self.assertFalse(np.array_equal(original,palette))
        # Rendering errors must restore colors just as successful renders do.
        with patch.object(env,"_render_frame",side_effect=RuntimeError("render failure")):
            with self.assertRaises(RuntimeError):
                source_palette_image(env)
        np.testing.assert_array_equal(original,env._get_obs()["image"])
        # A palette change alone should preserve a strong object-mask match.
        roi = np.ones((96,96),dtype=bool)
        matched = prediction_metrics(palette,original,original,roi)
        self.assertGreater(matched["object_mask_iou"],.9)
        self.assertLess(matched["object_centroid_error_pixels"],1)
        shifted = replace(scene,object_body=BodyState(Pose((300,240),.7)))
        future = env.restore_state(shifted)["image"]
        error = prediction_metrics(original,future,original,roi)
        self.assertGreater(error["object_centroid_error_pixels"],20)
        self.assertLess(error["object_mask_iou"],.5)


if __name__ == "__main__":
    unittest.main()
