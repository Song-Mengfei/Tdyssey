"""Offline contract tests: fake values only, no decoder/model/environment calls."""

from copy import deepcopy
import json
import math
import unittest

from tdyssey.wm_measurement_contract import (
    DEVELOPMENT_BOUND, angle_error_degrees, coverage_measurement, measurement_frame,
    sequence_endpoint, termination_from_coverage, wrap_radians)


def frame(estimate, *, half_width=.03, pusher=True, state=True):
    return measurement_frame(
        state={"valid": state, "pose": {"position": [100., 200.], "angle": .1} if state else None,
               "reason": "ok" if state else "missing_object"},
        shape={"valid": state, "fit_iou": .9 if state else None, "reason": "ok"},
        pusher={"valid": pusher, "position": [50., 50.] if pusher else None,
                "reason": "ok" if pusher else "missing_pusher"},
        coverage=coverage_measurement(estimate, half_width=half_width, uncertainty_kind=DEVELOPMENT_BOUND))


class MeasurementContractChecks(unittest.TestCase):
    def test_strict_success_threshold_uses_interval_not_point_estimate(self):
        for estimate, expected in ((.95, "non_success"), (.95000000001, "success"), (.94999999999, "non_success")):
            measurement = coverage_measurement(estimate, half_width=0, uncertainty_kind=DEVELOPMENT_BOUND)
            self.assertEqual(termination_from_coverage(measurement)["status"], expected)
        self.assertEqual(frame(.97)["termination"]["status"], "unknown")
        self.assertEqual(frame(.99)["termination"]["status"], "success")
        self.assertEqual(frame(.9)["termination"]["status"], "non_success")
        touching = coverage_measurement(.96, interval=[.95, .98], uncertainty_kind=DEVELOPMENT_BOUND)
        self.assertEqual(termination_from_coverage(touching)["status"], "unknown")

    def test_valid_point_without_usable_uncertainty_is_not_a_success_probability(self):
        for options in ({}, {"interval": [.98, 1.]}, {"half_width": .03},
                        {"half_width": .03, "uncertainty_kind": "fit_residual_probability"},
                        {"interval": [1., .98], "uncertainty_kind": DEVELOPMENT_BOUND},
                        {"interval": [.99, 1.], "uncertainty_kind": DEVELOPMENT_BOUND}):
            value = coverage_measurement(.98, **options)
            self.assertTrue(value["valid"])
            self.assertEqual(value["estimate"], .98)
            self.assertEqual(termination_from_coverage(value)["status"], "unknown")
            self.assertFalse(value["is_probability"])
            json.dumps(value, allow_nan=False)
        for estimate in (None, float("nan"), float("inf"), -.1, 1.1, True):
            value = coverage_measurement(estimate, half_width=.03, uncertainty_kind=DEVELOPMENT_BOUND)
            self.assertFalse(value["valid"])
            self.assertIsNone(value["estimate"])
            self.assertEqual(termination_from_coverage(value)["status"], "unknown")
            json.dumps(value, allow_nan=False)

    def test_object_pusher_shape_and_coverage_validity_are_independent(self):
        value = frame(.99, pusher=False)
        self.assertTrue(value["state"]["valid"])
        self.assertTrue(value["shape"]["valid"])
        self.assertFalse(value["pusher"]["valid"])
        self.assertTrue(value["coverage"]["valid"])
        self.assertEqual(value["termination"]["status"], "success")
        self.assertTrue(sequence_endpoint([value])["stopped_score_valid"])

    def test_invalid_midframe_preserves_later_pose_and_raw_coverage_but_abstains_sequence(self):
        values = [frame(.4), frame(None, state=False), frame(.5), frame(.7)]
        before = deepcopy(values)
        result = sequence_endpoint(values)
        self.assertFalse(result["stopped_score_valid"])
        self.assertIsNone(result["stopped_score"])
        self.assertEqual(result["unknown_history"][0]["horizon"], 2)
        self.assertTrue(result["raw_endpoint"]["state"]["valid"])
        self.assertEqual(result["raw_endpoint_coverage"], .7)
        self.assertEqual(result["raw_endpoint"]["state"]["pose"], values[-1]["state"]["pose"])
        self.assertTrue(values[2]["state"]["valid"])
        self.assertEqual(values, before)

    def test_certain_earlier_success_absorbs_score_despite_later_missing_frame(self):
        values = [frame(.5), frame(.99), frame(None, state=False), frame(.3)]
        result = sequence_endpoint(values)
        self.assertTrue(result["stopped_score_valid"])
        self.assertEqual(result["stopped_score"], .99)
        self.assertEqual(result["stop_horizon"], 2)
        self.assertTrue(result["absorbing"])
        self.assertEqual(result["absorbing_model_state"], values[1])
        self.assertTrue(result["unknown_history"][0]["after_certain_absorption"])
        self.assertFalse(result["unknown_before_stop_or_endpoint"])
        self.assertEqual(result["raw_endpoint_coverage"], .3)
        result["absorbing_model_state"]["state"]["pose"]["position"][0] = -1
        self.assertEqual(values[1]["state"]["pose"]["position"][0], 100)

    def test_unknown_before_later_success_cannot_repair_unknown_stopping_time(self):
        result = sequence_endpoint([frame(.96), frame(.99), frame(.5)])
        self.assertFalse(result["stopped_score_valid"])
        self.assertEqual(result["first_interval_success_index"], 1)
        self.assertIsNone(result["absorbing_model_state"])
        self.assertEqual(result["raw_endpoint_coverage"], .5)
        self.assertTrue(result["unknown_before_stop_or_endpoint"])

    def test_all_non_success_frames_score_raw_endpoint_and_unjustified_success_is_unknown(self):
        values = [frame(.1), frame(.3), frame(.5)]
        result = sequence_endpoint(values)
        self.assertTrue(result["stopped_score_valid"])
        self.assertEqual(result["stopped_score"], .5)
        self.assertFalse(result["absorbing"])
        values[0]["termination"] = {"status": "success", "reason": "bad_point_threshold"}
        self.assertFalse(sequence_endpoint(values)["stopped_score_valid"])
        values[0] = {"coverage": None, "state": {"valid": False}}
        self.assertFalse(sequence_endpoint(values)["stopped_score_valid"])

    def test_producer_can_abstain_for_domain_uncertainty_and_wrapped_angles_are_correct(self):
        value = frame(.99)
        value["termination"] = {"status": "unknown", "reason": "generated_image_uncertainty_not_calibrated"}
        self.assertFalse(sequence_endpoint([value])["stopped_score_valid"])
        self.assertAlmostEqual(angle_error_degrees(math.radians(179), math.radians(-179)), 2)
        self.assertAlmostEqual(math.degrees(wrap_radians(math.radians(370))), 10)
        with self.assertRaises(ValueError):
            wrap_radians(float("nan"))
        with self.assertRaises(ValueError):
            coverage_measurement(.5, interval=[.4, .6], half_width=.03)
        with self.assertRaises(ValueError):
            sequence_endpoint([])


if __name__ == "__main__":
    unittest.main()
