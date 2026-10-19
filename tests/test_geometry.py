import math
import unittest

from tdyssey.evaluation import SuccessTracker, goal_metrics
from tdyssey.geometry import Pose, Rectangle, Shape, get_shape


class GeometryChecks(unittest.TestCase):
    def test_t_vertices_preserve_pinned_gpc_origin(self):
        shape = get_shape("T", 30)
        self.assertEqual(shape.components[0], Rectangle(-60, -45, 60, -15))
        self.assertEqual(shape.components[1], Rectangle(-15, -15, 15, 75))
        self.assertEqual(shape.area, 6300)
        self.assertAlmostEqual(shape.centroid[1], -30 / 7)
        self.assertEqual(shape.origin, "upstream_gpc")

    def test_new_shape_centroids_and_analytic_inertia(self):
        for name in ("I", "L"):
            with self.subTest(name=name):
                shape = get_shape(name)
                self.assertAlmostEqual(shape.centroid[0], 0)
                self.assertAlmostEqual(shape.centroid[1], 0)
        self.assertAlmostEqual(get_shape("I").inertia_about_origin(2), 2 * (30 ** 2 + 120 ** 2) / 12)
        self.assertAlmostEqual(get_shape("L", 1).inertia_about_origin(2), 793 / 147)

    def test_scaling_changes_area_inertia_and_hash(self):
        a, b = get_shape("L", 10), get_shape("L", 20)
        self.assertAlmostEqual(b.area / a.area, 4)
        self.assertAlmostEqual(b.inertia_about_origin(2) / a.inertia_about_origin(2), 4)
        self.assertNotEqual(a.geometry_hash, b.geometry_hash)
        self.assertEqual(a.geometry_hash, get_shape("L", 10).geometry_hash)

    def test_rejects_overlap_disconnection_and_false_symmetry(self):
        for components, order in [
            ((Rectangle(0, 0, 2, 2), Rectangle(1, 1, 3, 3)), 1),
            ((Rectangle(0, 0, 1, 1), Rectangle(1, 1, 2, 2)), 1),
            (get_shape("T").components, 2),
        ]:
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    Shape("invalid", components, order, "test")

    def test_rejects_invalid_parameters(self):
        for scale in (0, -1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                get_shape("I", scale)
        with self.assertRaises(ValueError):
            get_shape("Q")
        with self.assertRaises(ValueError):
            Pose((0, math.nan), 0)

    def test_half_turn_equivalence_is_shape_specific(self):
        self.assertAlmostEqual(get_shape("I").angle_error(math.pi, 0), 0)
        for name in ("T", "L"):
            self.assertAlmostEqual(get_shape(name).angle_error(math.pi, 0), math.pi)
        self.assertAlmostEqual(get_shape("T").angle_error(2 * math.pi, 0), 0)

    def test_identical_and_separated_shapes(self):
        for name in ("T", "I", "L"):
            shape = get_shape(name)
            actual = Pose((256, 256), 0.7)
            metrics = goal_metrics(shape, actual, actual)
            self.assertAlmostEqual(metrics.iou, 1)
            self.assertAlmostEqual(metrics.coverage, 1)
            self.assertAlmostEqual(metrics.position_error, 0)
            self.assertAlmostEqual(goal_metrics(shape, actual, Pose((1000, 1000), 0)).iou, 0)

    def test_rectangle_overlap_and_edge_touching(self):
        shape = get_shape("I", 30)
        metrics = goal_metrics(shape, Pose((15, 0), 0), Pose((0, 0), 0))
        self.assertAlmostEqual(metrics.coverage, 0.5)
        self.assertAlmostEqual(metrics.iou, 1 / 3)
        self.assertAlmostEqual(goal_metrics(shape, Pose((30, 0), 0), Pose((0, 0), 0)).iou, 0)

    def test_rotated_compound_overlap_is_symmetric_and_bounded(self):
        for name in ("T", "I", "L"):
            shape = get_shape(name)
            a, b = Pose((200, 200), 0.4), Pose((215, 180), 1.7)
            ab, ba = goal_metrics(shape, a, b), goal_metrics(shape, b, a)
            self.assertAlmostEqual(ab.iou, ba.iou)
            self.assertTrue(0 <= ab.iou <= 1)
        self.assertAlmostEqual(goal_metrics(get_shape("I"), Pose((0, 0), math.pi), Pose((0, 0), 0)).iou, 1)
        self.assertLess(goal_metrics(get_shape("T"), Pose((0, 0), math.pi), Pose((0, 0), 0)).iou, 1)

    def test_rotated_overlap_matches_independent_analytic_cases(self):
        cross = goal_metrics(get_shape("I"), Pose((0, 0), math.pi / 2), Pose((0, 0), 0))
        self.assertAlmostEqual(cross.iou, 1 / 7)  # Intersection is a 30x30 square.
        square = Shape("square", (Rectangle(-0.5, -0.5, 0.5, 0.5),), 4, "area_centroid")
        rotated = goal_metrics(square, Pose((0, 0), math.pi / 4), Pose((0, 0), 0))
        self.assertAlmostEqual(rotated.coverage, 2 * math.sqrt(2) - 2)
        self.assertAlmostEqual(rotated.iou, 1 / math.sqrt(2))
        self.assertAlmostEqual(goal_metrics(get_shape("T"), Pose((0, 0), math.pi), Pose((0, 0), 0)).iou, 3 / 11)

    def test_success_requires_consecutive_control_steps(self):
        tracker = SuccessTracker(0.9, 3)
        for score in (0.95, 0.95, 0.8, 0.95, 0.95):
            self.assertFalse(tracker.observe(score))
        self.assertTrue(tracker.observe(0.95))
        self.assertEqual(tracker.first_success_step, 6)
        self.assertTrue(tracker.observe(0.2))
        self.assertEqual(tracker.final_iou, 0.2)
        self.assertEqual(tracker.max_iou, 0.95)

    def test_success_tracker_rejects_invalid_scores(self):
        for score in (math.nan, -0.1, 1.1):
            with self.assertRaises(ValueError):
                SuccessTracker().observe(score)


if __name__ == "__main__":
    unittest.main()
