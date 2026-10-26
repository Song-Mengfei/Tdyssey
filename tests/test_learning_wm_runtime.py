"""Physics and geometry checks for the privileged world model and planning teacher (scripts/learning)."""

import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set TDYSSEY_OFFICIAL_ROOT for actual simulator checks")
class LearningWorldModelRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(key, value)
        sys.path.insert(0, str(ROOT/"scripts/learning"))
        import numpy as np
        import torch
        import teacher2
        import wm2
        from tdyssey.envs.official import make_official_shape_env_class
        from tdyssey.official import load_official_source, physical_metrics, physical_state
        cls.np, cls.torch, cls.wm2 = np, torch, wm2
        cls.metrics, cls.state = staticmethod(physical_metrics), staticmethod(physical_state)
        cls.env_class = make_official_shape_env_class(load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"])))
        cls.geo = teacher2.Geometry("cpu")

    def env(self, name, seed=0):
        env = self.env_class(shape=name, legacy=True)
        self.addCleanup(env.close)
        env.seed(seed); env.reset()
        return env

    def place(self, env, k, dx=0.0, dy=0.0, dtheta=0.0):
        """Put the letter's centre of mass at the planner's goal (+ offsets). Set the angle first: pymunk
        rotates about the centre of mass, so setting it after the position moves an offset-COG letter (T)."""
        np = self.np
        gx, gy, gt = self.geo.goals[k].tolist(); cog = self.geo.cogs[k].numpy()
        th = gt + dtheta; c, s = math.cos(th), math.sin(th)
        env.block.angle = th
        env.block.position = (gx+dx - (c*cog[0]-s*cog[1]), gy+dy - (s*cog[0]+c*cog[1]))
        env.block.velocity = (0, 0); env.block.angular_velocity = 0

    def test_pusher_integration_matches_the_simulator(self):
        """wm2 integrates the kinematic pusher exactly (10 PD substeps per control), contact or not."""
        torch, np = self.torch, self.np
        model = self.wm2.WM2(self.geo.rects)
        rng = np.random.default_rng(0)
        for name in ("T", "L"):
            env = self.env(name)
            for _ in range(40):
                st = self.state(env)["pusher"]
                p = torch.tensor([st["position"]], dtype=torch.float64)
                v = torch.tensor([st["velocity"]], dtype=torch.float64)
                a = rng.uniform(40, 470, size=2)
                env.step(a)
                ps, vs = model.pusher(p, v, torch.tensor([a], dtype=torch.float64))
                after = self.state(env)["pusher"]
                self.assertLess(float((ps[0, -1]-torch.tensor(after["position"])).norm()), 1e-3)
                self.assertLess(float((vs[0, -1]-torch.tensor(after["velocity"])).norm()), 1e-2)

    def test_planner_goal_is_the_simulator_goal_and_success_needs_pixel_precision(self):
        for k, name in enumerate(self.wm2.SHAPES):
            with self.subTest(shape=name):
                env = self.env(name)
                self.place(env, k)
                self.assertGreater(self.metrics(env)["coverage"], 0.999)
                self.place(env, k, dx=3.0)                  # a 3 px offset already fails the 0.95 success rule
                self.assertLess(self.metrics(env)["coverage"], 0.95)


if __name__ == "__main__":
    unittest.main()
