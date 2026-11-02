"""Saved synthetic scalar traces only; no simulation, sampling or tensor import."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.analyze_c_pair_traces import analyze_case, analyze_pair, export, paired_case, runs
from tests.test_policy_eval_comparison import group, refresh

ROOT = Path(__file__).resolve().parents[1]


def case(values, *, initial=0., method="base_training", shape="T", scene=100000, seed=5478):
    n = len(values)
    success = values[-1] > .95
    return {"method": method, "shape": shape, "scene_seed": scene, "policy_seed": seed,
        "status": "completed", "success": success, "success_step": n if success else None,
        "stop_reason": "success" if success else ("control_cap" if n == 300 else "episode_timeout"),
        "initial_state": {"saved_initial_state_only": True},
        "initial_metrics": {"coverage": initial, "iou": initial / 2},
        "final_metrics": {"coverage": values[-1], "iou": values[-1] / 2},
        "peak_metrics": {"coverage": max([initial] + values), "iou": max([initial] + values) / 2},
        "counts": {"control_steps": n, "planning_calls": (n + 7) // 8},
        "latency": {"planning_seconds": [.01] * ((n + 7) // 8)},
        "steps": [{"coverage": v, "iou": v / 2, "action": [float(i), float(2*i)],
                   "upstream_done": i == n, "upstream_reward": min(v / .95, 1.)}
                  for i, v in enumerate(values, 1)]}


def report_pair():
    reports = [group(method) for method in ("base_training", "delta_training")]
    for report in reports:
        report["cases"] = [case([0., .1, .2], method=report["method"], shape=row["shape"],
            scene=row["scene_seed"], seed=row["policy_seed"]) for row in report["cases"]]
        refresh(report)
    return reports


class PairTraceChecks(unittest.TestCase):
    def test_initial_and_executed_indices_peak_loss_positive_and_near_segments(self):
        summary, series = analyze_case(case([0., .3, .9, .94, .2, .91, .4]))
        self.assertEqual(summary["first_positive_coverage_control"], 2)
        self.assertEqual(summary["first_positive_executed_control"], 2)
        self.assertEqual(summary["peak_control_first"], 4)
        self.assertAlmostEqual(summary["peak_final_coverage_loss"], .54)
        self.assertEqual(summary["positive_coverage_fraction"], 6 / 7)
        self.assertEqual(summary["near_success_segments"], [
            {"start_control": 3, "end_control": 4, "length": 2},
            {"start_control": 6, "end_control": 6, "length": 1}])
        self.assertEqual([p["control"] for p in series], list(range(8)))
        self.assertIsNone(series[0]["action_target_x"])
        self.assertEqual(summary["plan_start_controls"], [1])
        self.assertTrue(summary["plan_boundaries_inferred"])
        self.assertIn("not explicitly logged", summary["plan_boundary_basis"])
        self.assertAlmostEqual(summary["action_target_variation_length"], 6 * 5**.5)

    def test_initial_peak_zero_overlap_and_strict_success_threshold(self):
        initial, _ = analyze_case(case([.8, .7], initial=.9))
        self.assertEqual(initial["peak_control_first"], 0)
        self.assertEqual(initial["first_positive_coverage_control"], 0)
        self.assertEqual(initial["first_positive_executed_control"], 1)
        zero, _ = analyze_case(case([0., 0., 0.]))
        self.assertIsNone(zero["first_positive_coverage_control"])
        self.assertEqual(zero["positive_coverage_fraction"], 0)
        threshold, _ = analyze_case(case([.90, .95]))
        self.assertFalse(threshold["success"])
        self.assertEqual(threshold["near_success_controls"], 2)
        passed, _ = analyze_case(case([.95, .95001]))
        self.assertEqual(passed["success_step"], 2)
        self.assertEqual(passed["coverage_strictly_above_0_95_controls"], 1)

    def test_segments_close_at_end_and_plan_boundaries_do_not_use_physics_indexes(self):
        self.assertEqual(runs([.9, .9, 0, .9], lambda value: value >= .9), [
            {"start_control": 1, "end_control": 2, "length": 2},
            {"start_control": 4, "end_control": 4, "length": 1}])
        summary, series = analyze_case(case([.1] * 10))
        self.assertEqual(summary["plan_start_controls"], [1, 9])
        self.assertEqual([p["control"] for p in series if p["inferred_plan_start"]], [1, 9])
        self.assertIn("not explicitly logged", summary["plan_boundary_basis"])

    def test_shared_prefix_opposite_changes_and_success_censoring(self):
        a, a_series = analyze_case(case([0., .1, .2, .1]))
        b, b_series = analyze_case(case([0., .2, .1, .3], method="delta_training"))
        paired, series = paired_case(a, b, a_series, b_series)
        self.assertEqual(paired["opposite_change_control_count"], 2)
        self.assertEqual({p["control"] for p in paired["opposite_change_examples"]}, {3, 4})
        short, short_series = analyze_case(case([.2, .96]))
        long, long_series = analyze_case(case([.1, .8, .9, .94], method="delta_training"))
        paired, shared = paired_case(short, long, short_series, long_series)
        self.assertEqual(paired["shared_executed_controls"], 2)
        self.assertEqual([p["control"] for p in shared], [0, 1, 2])
        self.assertIsNone(paired["both_success_step_difference"])
        self.assertEqual(paired["success_difference"], -1)

    def test_tampered_counts_peaks_targets_and_post_success_continuation_rejected(self):
        mutations = [lambda c: c["counts"].update(control_steps=99),
            lambda c: c["final_metrics"].update(coverage=.1),
            lambda c: c["peak_metrics"].update(coverage=.9),
            lambda c: c["steps"][0].update(action=[0.]),
            lambda c: c["steps"][0].update(coverage=float("nan")),
            lambda c: c.update(stop_reason="control_cap")]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                value = case([0., .2]); mutation(value)
                with self.assertRaises(ValueError): analyze_case(value)
        with self.assertRaisesRegex(ValueError, "continued after first success"):
            continued = case([.96, .97])
            continued["success_step"] = 1
            analyze_case(continued)

    def test_all_twenty_pairs_scene_rows_export_and_no_overwrite(self):
        base, delta = report_pair()
        original = deepcopy((base, delta))
        summary = analyze_pair(base, delta)
        self.assertEqual((len(summary["case_summaries"]), len(summary["paired_cases"]), len(summary["per_scene"])), (40, 20, 10))
        self.assertEqual(len(summary["series"]), 160)
        self.assertEqual(len(summary["paired_series"]), 80)
        self.assertEqual((base, delta), original)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            inputs = [directory / name for name in ("base.json", "delta.json")]
            for path, data in zip(inputs, (base, delta)): path.write_text(json.dumps(data), encoding="utf-8")
            result = export(*inputs, directory / "derived")
            self.assertEqual(result["figures"]["status"], "not_requested")
            with (directory / "derived/cases.csv").open(newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 40)
            with self.assertRaisesRegex(ValueError, "new directory"):
                export(*inputs, directory / "derived")

    def test_import_and_help_without_optional_runtime_packages(self):
        code = "import json,sys;import scripts.analyze_c_pair_traces;print(json.dumps({k:k in sys.modules for k in ('torch','numpy','matplotlib','pymunk')}))"
        child = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT,
                               capture_output=True, text=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), {k: False for k in ("torch", "numpy", "matplotlib", "pymunk")})


if __name__ == "__main__":
    unittest.main()
