"""Source-only protocol checks; no simulation or third-party packages."""

from dataclasses import replace
import math
import unittest

from tdyssey.collection import CollectionConfig, WaypointPlay, reset_issues
from tdyssey.geometry import Pose, get_shape
from tdyssey.state import BodyState, Snapshot


def state(name="I", pose=Pose((250, 250), 0), pusher=(100, 100)):
    return Snapshot(name, get_shape(name).geometry_hash, "a" * 64,
                    BodyState(pose), BodyState(Pose(pusher, 0)), Pose((400, 380), 0), 0)


class CollectionProtocolChecks(unittest.TestCase):
    def test_run_limits_reject_excess_work_and_storage_up_front(self):
        for changes in ({"episodes": 0}, {"steps": 251}, {"episodes": 5, "steps": 250},
                        {"max_bytes": 1024}, {"max_bytes": 65 * 1024 * 1024},
                        {"max_bytes": 1024 * 1024, "episodes": 2, "steps": 100},
                        {"seed": True}, {"seed": 2**32}, {"mode": "expert"},
                        {"max_target_delta": float("nan")}, {"waypoint_steps": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CollectionConfig(**changes)

    def test_random_waypoints_are_repeatable_bounded_and_do_not_need_object_state(self):
        config = CollectionConfig()
        first, second = WaypointPlay(config, 7), WaypointPlay(config, 7)
        observation = {"agent_pos": (100, 120)}
        for step in range(50):
            action = first.action(observation, None, step)
            self.assertEqual(action, second.action(observation, None, step))
            self.assertLessEqual(math.dist(action, observation["agent_pos"]), 12 + 1e-9)
            self.assertTrue(all(25 <= v <= 486 for v in action))
            observation["agent_pos"] = action

    def test_angle_coverage_follows_object_pose_and_allows_far_targets(self):
        config = CollectionConfig(mode="angle-coverage", max_target_delta=200)
        record = state(pose=Pose((250, 250), math.pi / 2), pusher=(100, 100))
        action = WaypointPlay(config, 1).action({"agent_pos": (100, 100)}, record, 0)
        self.assertAlmostEqual(action[0], 100, places=6)
        self.assertAlmostEqual(action[1], 120, places=6)
        far = WaypointPlay(config, 1).action({"agent_pos": (200, 200)}, record, 32)
        self.assertGreater(math.dist(far, (200, 200)), 100)
        with self.assertRaisesRegex(ValueError, "30"):
            CollectionConfig(max_target_delta=31)
        with self.assertRaisesRegex(ValueError, "200"):
            CollectionConfig(mode="angle-coverage", max_target_delta=201)
        with self.assertRaisesRegex(ValueError, "200"):
            CollectionConfig(mode="contact-balanced", max_target_delta=201)

    def test_directed_mode_is_repeatable_and_uses_the_disclosed_state_prior(self):
        config = CollectionConfig(mode="object-directed")
        actions = [WaypointPlay(config, 1).action({"agent_pos": (100, 100)}, record, 0)
                   for record in (state(), state(pose=Pose((100, 300), 0)))]
        self.assertNotEqual(actions[0], actions[1])
        self.assertEqual(actions[0], WaypointPlay(config, 1).action({"agent_pos": (100, 100)}, state(), 0))

    def test_reset_validation_uses_rotated_shape_and_actual_material(self):
        shape = get_shape("L")
        pose = Pose((250, 250), math.pi / 2)
        opening = pose.transform((30, -30))
        valid = state("L", pose, opening)
        self.assertEqual(reset_issues(shape, valid), [])
        material = pose.transform(shape.components[0].center)
        overlap = replace(valid, pusher_body=BodyState(Pose(material, 0)))
        self.assertIn("pusher_object_overlap_or_insufficient_clearance", reset_issues(shape, overlap))
        outside = replace(valid, object_body=BodyState(Pose((20, 20), math.pi / 4)))
        self.assertIn("object_outside_walls", reset_issues(shape, outside))
        outside_goal = replace(valid, goal=Pose((500, 500), 0))
        self.assertIn("goal_outside_walls", reset_issues(shape, outside_goal))

    def test_invalid_pusher_and_initial_success_are_rejected(self):
        shape = get_shape("I")
        self.assertIn("pusher_outside_walls", reset_issues(shape, state(pusher=(10, 200))))
        initial_success = replace(state(), goal=state().object_body.pose)
        self.assertIn("already_near_goal", reset_issues(shape, initial_success))


if __name__ == "__main__":
    unittest.main()
