"""Original T parity, actual L contact, matching, and pre-contact restore checks."""

from dataclasses import replace
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from tdyssey.envs.official import make_official_shape_env_class, matched_reset, official_shape
from tdyssey.evaluation import goal_metrics
from tdyssey.geometry import Pose
from tdyssey.official import load_official_source, physical_metrics
from tdyssey.state import BodyState


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set TDYSSEY_OFFICIAL_ROOT for actual T/L simulator checks")
class OfficialShapeRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(key, value)
        import numpy as np
        cls.np = np
        cls.original = load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"]))
        cls.adapted = make_official_shape_env_class(cls.original)

    def env(self, name="T", original=False, seed=100000):
        env = self.original(legacy=True) if original else self.adapted(shape=name, legacy=True)
        self.addCleanup(env.close)
        env.seed(seed)
        return env

    def test_all_capitals_have_matching_collision_render_reward_and_contacts(self):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from tdyssey.letters import ALL_SHAPES
        for name in ALL_SHAPES:
            with self.subTest(shape=name):
                env = self.env(name)
                env.reset()
                shape = official_shape(name)
                state = env.capture_state()
                if name != "T":
                    self.assertEqual(tuple(env.block.center_of_gravity), (0, 0))
                    self.assertEqual(env.block.mass, 1)
                    expected_moment = 3000 if name == "T_narrow" else shape.inertia_about_origin(1)
                    self.assertAlmostEqual(env.block.moment, expected_moment)
                actual = unary_union([Polygon([tuple(env.block.local_to_world(v)) for v in p.get_vertices()]) for p in env.block.shapes])
                expected = unary_union([Polygon(v) for v in shape.polygons(state.object_body.pose)])
                self.assertLess(actual.symmetric_difference(expected).area, 1e-6)
                # At the target all letters, including letters with holes, solve.
                at_goal = replace(state, object_body=BodyState(state.goal), pusher_body=BodyState(Pose((60, 60), 0)))
                obs = env.restore_state(at_goal)
                _, reward, done, _ = env.step(None)
                self.assertAlmostEqual(reward, 1)
                self.assertTrue(done)
                self.assertAlmostEqual(physical_metrics(env)["iou"], 1)
                self.assertEqual(obs["image"].shape, (3, 96, 96))
                self.assertGreater(float(obs["image"].std()), .01)
                overlap = replace(state, object_body=BodyState(Pose((256, 256), .2)),
                                  goal=Pose((266, 251), .3), pusher_body=BodyState(Pose((60, 60), 0)))
                env.restore_state(overlap)
                expected_metrics = goal_metrics(shape, overlap.object_body.pose, overlap.goal)
                measured_metrics = physical_metrics(env)
                self.assertAlmostEqual(measured_metrics["iou"], expected_metrics.iou, places=8)
                _, reward, _, _ = env.step(None)
                self.assertAlmostEqual(reward, min(expected_metrics.coverage/.95, 1), places=8)
                # Approach the leftmost occupied rectangle to test actual contact.
                rect = min(shape.components, key=lambda r: (r.x0, r.y0))
                pose = Pose((256, 256), 0)
                y = rect.center[1] + 256
                start = (rect.x0 + 256 - 20, y)
                contact_state = replace(state, object_body=BodyState(pose), pusher_body=BodyState(Pose(start, 0)))
                env.restore_state(contact_state)
                contacts = 0
                for _ in range(8):
                    _, _, _, info = env.step(self.np.array([start[0]+65, y]))
                    contacts += info["n_contacts"]
                self.assertGreater(contacts, 0)
                self.assertGreater(math.dist(env.block.position, pose.position), 1)
                env.close()

    def test_letter_openings_and_simultaneous_matched_reset(self):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from tdyssey.letters import ALL_SHAPES
        for name, holes in (("A", 1), ("B", 2), ("D", 1), ("O", 1), ("P", 1), ("R", 1)):
            union = unary_union([Polygon(r.vertices) for r in official_shape(name).components])
            self.assertEqual(union.geom_type, "Polygon")
            self.assertEqual(len(union.interiors), holes)
        envs = {name: self.env(name) for name in ALL_SHAPES}
        counts = {"reset_physics_substeps": 0}
        observations, records = matched_reset(envs, 100000, counts)
        self.assertEqual(counts["reset_physics_substeps"], len(records)*len(ALL_SHAPES))
        self.assertFalse(any(records[-1]["issues"].values()))
        ref = envs["T"].capture_state()
        for name in ALL_SHAPES:
            if name == "T":
                continue
            state = envs[name].capture_state()
            self.np.testing.assert_allclose(state.object_body.pose.position, ref.object_body.pose.transform((0,45)))
            self.np.testing.assert_allclose(state.goal.position, ref.goal.transform((0,45)))
            self.np.testing.assert_array_equal(observations[name]["agent_pos"], observations["T"]["agent_pos"])

    def test_t_images_physics_contacts_and_rewards_match_original(self):
        for seed in (100000, 100001):
            first, second = self.env(original=True, seed=seed), self.env(seed=seed)
            a, b = first.reset(), second.reset()
            self.np.testing.assert_array_equal(a["image"], b["image"])
            self.np.testing.assert_array_equal(a["agent_pos"], b["agent_pos"])
            self.assertEqual((first.block.mass, first.block.moment), (second.block.mass, second.block.moment))
            target = self.np.array(first.block.position)
            contacts = 0
            for step in range(12):
                a, b = first.step(target), second.step(target)
                for key in a[0]:
                    self.np.testing.assert_array_equal(a[0][key], b[0][key])
                # Shapely polygon iteration order can change final-roundoff bits.
                self.assertAlmostEqual(float(a[1]), float(b[1]), places=12)
                self.assertEqual(a[2], b[2])
                for key in a[3]:
                    self.np.testing.assert_array_equal(a[3][key], b[3][key])
                contacts += a[3]["n_contacts"]
                self.assertEqual(second.control_step, step + 1)
            self.assertGreater(contacts, 0)

    def test_collision_union_goal_reward_and_render_share_one_l_geometry(self):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        env = self.env("L")
        env.reset()
        state = replace(env.capture_state(), object_body=BodyState(Pose((250, 250), 0.2)),
                        pusher_body=BodyState(Pose((100, 100), 0)), goal=Pose((260, 250), 0.3))
        observation = env.restore_state(state)
        actual = unary_union([Polygon([tuple(env.block.local_to_world(v)) for v in p.get_vertices()]) for p in env.block.shapes])
        expected = unary_union([Polygon(v) for v in official_shape("L").polygons(state.object_body.pose)])
        self.assertLess(actual.symmetric_difference(expected).area, 1e-7)
        self.assertEqual(env.block.mass, 1)
        self.assertAlmostEqual(env.block.moment, official_shape("L").inertia_about_origin(1))
        self.assertTrue(all(p.body is env.block for p in env.block.shapes))
        metrics = goal_metrics(official_shape("L"), state.object_body.pose, state.goal)
        measured = physical_metrics(env)
        self.assertAlmostEqual(measured["iou"], metrics.iou)
        _, reward, _, _ = env.step(None)
        self.assertAlmostEqual(reward, min(metrics.coverage / .95, 1))
        self.assertEqual(env.control_step, 0)
        self.assertEqual(observation["image"].shape, (3, 96, 96))
        self.assertGreater(float(observation["image"].std()), .01)
        upright = replace(state, object_body=BodyState(Pose((256, 256), 0)), goal=Pose((400, 120), 0))
        image = env.restore_state(upright)["image"]
        def pixel(x, y):
            return image[:, round(y * 96 / 512), round(x * 96 / 512)]
        # Actual rendered object and target have the foot at the bottom/right.
        # Pymunk shades body colors; compare filled regions without assuming its
        # lighting multiplier. Target colors use the upstream renderer directly.
        self.assertTrue((pixel(241, 225) < .95).all())
        self.np.testing.assert_allclose(pixel(286, 286), pixel(241, 225), atol=1e-6)
        self.np.testing.assert_allclose(pixel(286, 225), [1, 1, 1], atol=1e-6)
        self.np.testing.assert_allclose(pixel(430, 150), self.np.array([144, 238, 144]) / 255, atol=1e-6)
        self.np.testing.assert_allclose(pixel(430, 95), [1, 1, 1], atol=1e-6)

    def test_pushing_l_moves_one_rigid_body(self):
        env = self.env("L")
        env.reset()
        state = replace(env.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                        pusher_body=BodyState(Pose((206, 236), 0)), goal=Pose((400, 120), 0))
        env.restore_state(state)
        contacts = 0
        for _ in range(10):
            _, _, _, info = env.step(self.np.array([286, 236]))
            contacts += info["n_contacts"]
        self.assertGreater(contacts, 0)
        self.assertGreater(math.dist(env.block.position, (256, 256)), 1)

    def test_full_state_restore_and_fresh_precontact_replay(self):
        env = self.env("L")
        env.reset()
        state = replace(env.capture_state(), object_body=BodyState(Pose((256, 256), .2), (1, 2), .3, (4, 5), .6),
                        pusher_body=BodyState(Pose((170, 230), 0)), goal=Pose((400, 120), .5),
                        control_step=7, pusher_target=(180, 230))
        env.restore_state(state)
        self.assertEqual(env.capture_state(), state)
        results = []
        for _ in range(2):
            env.restore_state(state)
            for _ in range(10):
                env.step(self.np.array([270, 230]))
            result = env.capture_state().object_body
            results.append([*result.pose.position, result.pose.angle, *result.velocity, result.angular_velocity])
        self.np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-7)

    def test_common_reset_matches_centers_goals_and_original_t_initial_image(self):
        environments = {name: self.env(name) for name in ("T", "L")}
        counters = {"reset_physics_substeps": 0}
        observations, records = matched_reset(environments, 100000, counters)
        self.assertEqual(records, [{"seed": 100000, "issues": {"T": [], "L": []}}])
        self.assertEqual(counters["reset_physics_substeps"], 2)
        a, b = (environments[name] for name in ("T", "L"))
        self.np.testing.assert_allclose(a.block.local_to_world(a.block.center_of_gravity), b.block.position, atol=1e-7)
        self.np.testing.assert_allclose(a.capture_state().goal.transform((0, 45)), b.goal_pose[:2])
        self.np.testing.assert_array_equal(observations["T"]["agent_pos"], observations["L"]["agent_pos"])
        original = self.env(original=True, seed=100000)
        self.np.testing.assert_array_equal(original.reset()["image"], observations["T"]["image"])
        # A changed mass is incompatible even when the visible L vertices match.
        saved = b.capture_state()
        b.block.mass = 2
        with self.assertRaisesRegex(ValueError, "physics"):
            b.restore_state(saved)

    def test_rejected_common_reset_counts_both_shapes_and_advances_seed(self):
        environments = {name: self.env(name) for name in ("T", "L")}
        counters = {"reset_physics_substeps": 0}
        # Force rejection for one shape, then let the next candidate pass.
        with patch("tdyssey.envs.official.reset_issues", side_effect=[["fixture_rejection"], [], [], []]):
            observations, records = matched_reset(environments, 100000, counters)
        self.assertEqual(counters["reset_physics_substeps"], 4)
        self.assertEqual([row["seed"] for row in records], [100000, 100001])
        self.assertEqual(observations.keys(), {"T", "L"})
        self.assertEqual(environments["T"].control_step, 0)
        self.assertEqual(environments["L"].control_step, 0)


if __name__ == "__main__":
    unittest.main()
