"""Source-only capital-L geometry and cross-origin scene matching checks."""

from dataclasses import replace
import math
import unittest

from tdyssey.envs.official import T_REFERENCE_COG, matching_l_state, official_shape
from tdyssey.geometry import Pose
from tdyssey.state import BodyState, Snapshot


class OfficialShapeChecks(unittest.TestCase):
    def state(self, name, pose=Pose((200, 200), 0)):
        return Snapshot(name, official_shape(name).geometry_hash, "a" * 64,
                        BodyState(pose, (1, 2), 0.3, (4, 5), 0.6),
                        BodyState(Pose((100, 100), 0)), Pose((256, 256), math.pi / 4), 7, (120, 130))

    def test_capital_l_has_longer_vertical_and_shorter_bottom_foot(self):
        shape = official_shape("L")
        self.assertEqual(shape.area, 5400)
        self.assertEqual(shape.centroid, (0, 0))
        left, bottom = shape.components
        self.assertEqual((left.x1 - left.x0, left.y1 - left.y0), (30, 120))
        self.assertEqual(bottom.y1, left.y1)
        self.assertEqual(bottom.y1 - bottom.y0, 30)
        self.assertEqual(bottom.x1 - left.x0, 90)
        self.assertEqual(left.overlap(bottom), 0)
        self.assertTrue(left.shares_edge(bottom))
        self.assertEqual(shape.symmetry_order, 1)

    def test_alphabet_is_complete_centered_connected_and_versioned(self):
        from tdyssey.letters import ALL_SHAPES, PATTERNS
        self.assertEqual(set(PATTERNS), set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - {"T", "L"})
        self.assertEqual(len(ALL_SHAPES), 27)
        hashes = set()
        for name in ALL_SHAPES:
            shape = official_shape(name)  # Construction checks edge connectivity and symmetry.
            hashes.add(shape.geometry_hash)
            self.assertGreater(shape.inertia_about_origin(1), 0)
            if name in PATTERNS:
                self.assertAlmostEqual(shape.centroid[0], 0)
                self.assertAlmostEqual(shape.centroid[1], 0)
                self.assertAlmostEqual(max(r.y1 for r in shape.components) - min(r.y0 for r in shape.components), 120)
                self.assertAlmostEqual(max(r.x1 for r in shape.components) - min(r.x0 for r in shape.components), 100)
        self.assertEqual(len(hashes), 27)
        for name in ("H", "I", "O", "S", "X", "Z"):
            self.assertEqual(official_shape(name).symmetry_order, 2)
        with self.assertRaises(ValueError):
            official_shape("a")

    def test_near_t_changes_only_top_bar_width_at_same_mass_center(self):
        near = official_shape("T_narrow")
        self.assertEqual(near.components[0].vertices, ((-54, -45), (54, -45), (54, -15), (-54, -15)))
        original_stem = official_shape("T").components[1]
        self.assertEqual(near.components[1].vertices, tuple((x, y-45) for x, y in original_stem.vertices))
        self.assertEqual(near.area, 5940)

    def test_official_t_coordinates_remain_original(self):
        shape = official_shape("T")
        self.assertEqual(shape.area, 6300)
        self.assertEqual(shape.components[0].vertices, ((-60, 0), (60, 0), (60, 30), (-60, 30)))
        self.assertEqual(T_REFERENCE_COG, (0, 45))
        self.assertNotEqual(shape.centroid, T_REFERENCE_COG)  # Preserve upstream convention.

    def test_matching_transforms_centers_and_preserves_full_motion(self):
        reference = self.state("T", Pose((200, 210), math.pi / 2))
        template = self.state("L")
        matched = matching_l_state(reference, template)
        self.assertAlmostEqual(matched.object_body.pose.position[0], 155)
        self.assertAlmostEqual(matched.object_body.pose.position[1], 210)
        self.assertEqual(matched.object_body.velocity, reference.object_body.velocity)
        self.assertEqual(matched.object_body.force, reference.object_body.force)
        self.assertEqual(matched.object_body.angular_velocity, 0.3)
        self.assertEqual(matched.object_body.torque, 0.6)
        self.assertEqual(matched.pusher_body, reference.pusher_body)
        self.assertEqual(matched.control_step, reference.control_step)
        self.assertEqual(matched.pusher_target, reference.pusher_target)
        self.assertEqual(matched.goal.position, reference.goal.transform((0, 45)))
        self.assertEqual(matched.physics_hash, template.physics_hash)
        with self.assertRaises(ValueError):
            matching_l_state(replace(reference, geometry_hash="b" * 64), template)


if __name__ == "__main__":
    unittest.main()
