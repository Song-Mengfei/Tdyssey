"""Source-only checks for within-context feedback and experiment boundaries."""

from dataclasses import replace
import unittest

from tdyssey.forward_protocol import ForwardConfig, group_feedback


class ForwardProtocolChecks(unittest.TestCase):
    def test_group_centering_neutrality_and_positive_regression_curvature(self):
        self.assertEqual(group_feedback([0., 0., 0.], .1)["optimality"], [.5]*3)
        group = group_feedback([-.2, .1, .4], .1)
        for a, b in zip(group["optimality"], [0., .5, 1.]):
            self.assertAlmostEqual(a, b)
        self.assertTrue(all(a+group["anchor"] > 0 for a in group["advantages"]))
        shifted = group_feedback([99.8, 100.1, 100.4], .1)
        for a, b in zip(group["optimality"], shifted["optimality"]):
            self.assertAlmostEqual(a, b)

    def test_multishape_training_and_fixed_matched_evaluation_contract(self):
        config = ForwardConfig(shapes=["T", "L", "Y", "T_narrow"], eval_shapes=["T", "L", "Y"])
        self.assertEqual(config.shapes, ("T", "L", "Y", "T_narrow"))
        with self.assertRaises(ValueError):
            replace(config, shapes=("L",))  # T reference term needs T contexts.
        with self.assertRaises(ValueError):
            replace(config, eval_shapes=("L", "Y"))


if __name__ == "__main__":
    unittest.main()
