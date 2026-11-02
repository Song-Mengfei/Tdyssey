"""Pure synthetic JSON comparisons; no optional runtime or physical calls."""

from copy import deepcopy
import json
import unittest

from tdyssey.policy_eval_comparison import BASE_SHA256, CASE_KEYS, compare_evaluation_reports


def group(method):
    delta = method == "delta_training"
    cases = []
    for shape, scene, seed in CASE_KEYS:
        cases.append({"shape": shape, "scene_seed": scene, "policy_seed": seed, "method": method,
            "status": "completed", "success": False, "success_step": None,
            "failure_reason": "control_cap", "abort_reason": None, "stop_reason": "control_cap",
            "counts": {"control_steps": 300},
            "final_metrics": {"coverage": .4, "iou": .2}, "peak_metrics": {"coverage": .5, "iou": .3}})
    return {"schema_version": 1, "kind": "tdyssey.policy_development_evaluation", "method": method,
        "comparison_label": "delta2048" if delta else "base", "sampler_mode": "training",
        "protocol": "official-t-l-com-v1", "max_controls": 300, "phase": "development",
        "training": False, "optimizer_steps": 0, "delta_applied": delta,
        "checkpoint_identity": {"base_sha256": BASE_SHA256, "delta_sha256": "b" * 64 if delta else None},
        "sampler_config": {"version": "tdyssey-ddpm-gaussian-v1", "std_floor": .01},
        "success_criterion": "coverage > 0.95", "success_observation": "after_physical_control",
        "matched_initial_state_only": True, "mid_contact_restore_validated": False,
        "state_fingerprint_before": "a" * 64, "state_fingerprint_after": "a" * 64,
        "status": "completed", "requested_case_count": 20, "completed_case_count": 20,
        "aborted_case_count": 0, "unstarted_case_count": 0, "success_count": 0, "success_fraction": 0., "cases": cases}


def refresh(report):
    completed = [case for case in report["cases"] if case["status"] == "completed"]
    report.update(completed_case_count=len(completed),
        aborted_case_count=sum(case["status"] == "aborted" for case in report["cases"]),
        unstarted_case_count=sum(case["status"] == "not_started" for case in report["cases"]),
        status="completed" if len(completed) == 20 else "incomplete",
        success_count=sum(case["success"] for case in completed))
    report["success_fraction"] = report["success_count"] / 20 if len(completed) == 20 else None


def success(report, index, step):
    report["cases"][index].update(success=True, success_step=step, failure_reason=None, stop_reason="success",
        counts={"control_steps": step}, final_metrics={"coverage": .96, "iou": .94},
        peak_metrics={"coverage": .96, "iou": .94})
    refresh(report)


class EvaluationComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.base, self.delta = group("base_training"), group("delta_training")

    def test_fixed_denominators_paired_success_and_conditional_success_steps(self):
        for index, step in ((0, 20), (3, 30), (10, 80)):
            success(self.base, index, step)
        for index, step in ((0, 10), (1, 15), (10, 100)):
            success(self.delta, index, step)
        original = deepcopy((self.base, self.delta))
        result = compare_evaluation_reports(self.base, self.delta)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["training_steps"], 0)
        t, l = result["per_shape"]["T"], result["per_shape"]["L"]
        self.assertEqual((t["n"], l["n"]), (10, 10))
        self.assertEqual((t["baseline"]["success_count"], t["delta"]["success_count"]), (2, 2))
        self.assertEqual(t["baseline"]["success_fraction"], .2)
        self.assertEqual(t["paired_success"], {"win": 1, "tie": 8, "loss": 1})
        self.assertEqual(t["success_difference_pp"], 0)
        self.assertEqual(t["first_success_steps"]["both_success_count"], 1)
        self.assertEqual(t["first_success_steps"]["paired_difference"], {"mean": -10., "median": -10})
        self.assertEqual(t["first_success_steps"]["paired_outcomes"], {"win": 1, "tie": 0, "loss": 0})
        self.assertEqual(l["first_success_steps"]["paired_difference"]["mean"], 20)
        self.assertEqual(l["first_success_steps"]["paired_outcomes"], {"win": 0, "tie": 0, "loss": 1})
        self.assertIsNone(result["cases"][1]["success_step_difference"])
        self.assertEqual(result["cases"][1]["success_step_comparison"], "censored_either_not_success")
        self.assertEqual(len(result["cases"]), 20)
        self.assertEqual(len(result["scenes"]), 5)
        self.assertEqual(result["scenes"][0]["per_shape"]["T"]["n"], 2)
        self.assertIn("correlated", result["limits"]["statistical_unit"])
        self.assertEqual((self.base, self.delta), original)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_metric_mean_median_paired_differences_and_explicit_tolerance(self):
        for case in self.delta["cases"]:
            change = .1 if case["shape"] == "T" else -.1
            for scope in ("final_metrics", "peak_metrics"):
                for metric in ("coverage", "iou"):
                    case[scope][metric] += change
        result = compare_evaluation_reports(self.base, self.delta)
        for metric in ("final_coverage", "final_iou", "peak_coverage", "peak_iou"):
            t, l = (result["per_shape"][s]["metrics"][metric] for s in ("T", "L"))
            self.assertAlmostEqual(t["paired_difference"]["mean"], .1)
            self.assertAlmostEqual(t["paired_difference"]["median"], .1)
            self.assertEqual(t["paired_outcomes"], {"win": 10, "tie": 0, "loss": 0})
            self.assertEqual(l["paired_outcomes"], {"win": 0, "tie": 0, "loss": 10})
        self.delta = group("delta_training")
        self.delta["cases"][0]["final_metrics"]["coverage"] += .5e-9
        self.delta["cases"][1]["final_metrics"]["coverage"] += 2e-9
        result = compare_evaluation_reports(self.base, self.delta, tolerance=1e-9)
        self.assertEqual(result["per_shape"]["T"]["metrics"]["final_coverage"]["paired_outcomes"],
                         {"win": 1, "tie": 9, "loss": 0})

    def test_incomplete_matrix_preserves_unknowns_and_withholds_all_aggregates(self):
        success(self.delta, 0, 4)
        for index, case in enumerate(self.delta["cases"]):
            if index >= 4:
                case.update(status="aborted" if index == 4 else "not_started", success=None, success_step=None,
                    abort_reason="synthetic interruption" if index == 4 else None,
                    final_metrics=None, peak_metrics=None, counts={"control_steps": 2 if index == 4 else 0})
        refresh(self.delta)
        result = compare_evaluation_reports(self.base, self.delta)
        self.assertEqual(result["status"], "incomplete")
        for summary in (result["overall"], *result["per_shape"].values()):
            self.assertIsNone(summary["baseline"]["success_fraction"])
            self.assertIsNone(summary["delta"]["success_count"])
            self.assertIsNone(summary["success_difference_pp"])
            self.assertIsNone(summary["paired_success"])
            self.assertIsNone(summary["metrics"]["final_coverage"]["paired_difference"]["mean"])
        self.assertEqual(result["per_shape"]["T"]["delta"]["observed_success_count"], 1)
        self.assertIsNone(result["cases"][0]["success_difference"])
        self.assertIsNone(result["cases"][4]["delta"]["success"])
        self.assertEqual(len(result["cases"]), 20)

    def test_order_missing_duplicates_and_shape_or_policy_seed_changes_rejected(self):
        variants = []
        missing = deepcopy(self.delta); missing["cases"].pop(); variants.append(missing)
        reordered = deepcopy(self.delta); reordered["cases"][0], reordered["cases"][1] = reordered["cases"][1], reordered["cases"][0]; variants.append(reordered)
        duplicate = deepcopy(self.delta); duplicate["cases"][1] = deepcopy(duplicate["cases"][0]); variants.append(duplicate)
        seed = deepcopy(self.delta); seed["cases"][0]["policy_seed"] = 999; variants.append(seed)
        for report in variants:
            with self.subTest(report=report["cases"][:2]), self.assertRaisesRegex(ValueError, "ordered cases"):
                compare_evaluation_reports(self.base, report)

    def test_sampler_identity_labels_protocol_and_nonfinite_metrics_fail_closed(self):
        mutations = [lambda r: r.update(method="base_original"),
            lambda r: r.update(comparison_label="base"), lambda r: r.update(sampler_mode="baseline"),
            lambda r: r["sampler_config"].update(std_floor=0),
            lambda r: r["sampler_config"].update(std_floor=.02),
            lambda r: r["checkpoint_identity"].update(base_sha256="c" * 64),
            lambda r: r["checkpoint_identity"].update(delta_sha256=None),
            lambda r: r.update(protocol="official-t-original-v1"),
            lambda r: r.update(max_controls=299),
            lambda r: r["cases"][0]["final_metrics"].update(coverage=float("nan")),
            lambda r: r.update(state_fingerprint_after="d" * 64)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                report = deepcopy(self.delta); mutate(report)
                with self.assertRaises(ValueError):
                    compare_evaluation_reports(self.base, report)
        with self.assertRaises(ValueError):
            compare_evaluation_reports(self.delta, self.base)

    def test_failure_success_steps_are_censored_and_strict_threshold_is_enforced(self):
        self.delta["cases"][0]["success_step"] = 300
        with self.assertRaisesRegex(ValueError, "censored"):
            compare_evaluation_reports(self.base, self.delta)
        self.delta = group("delta_training")
        success(self.delta, 0, 300)
        self.delta["cases"][0]["final_metrics"]["coverage"] = .95
        with self.assertRaisesRegex(ValueError, "Strict physical coverage"):
            compare_evaluation_reports(self.base, self.delta)


if __name__ == "__main__":
    unittest.main()
