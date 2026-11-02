"""Offline scalar diagnosis: no tensors, third-party imports or simulator."""

import copy
import math
import unittest

from scripts.analyze_c_training_diagnostics import (
    aggregate_shapes, analyze_rows, canonical_hash, physical_runs,
    pool_moments, reward_examples,
)


def fixture():
    config = {"optimization": {"gamma": .99, "beta": 1., "gae_lambda": .95, "max_grad_norm": 1.}}
    rows = []
    for index in range(1, 4):
        shape = {"controls": 2, "plans": 1, "contact_controls": 0,
                 "unknown_contact_controls": 0, "nonzero_raw_reward_controls": 2,
                 "nonzero_training_reward_controls": 2, "nonzero_shaping_controls": 2,
                 "raw_reward_sum": .8, "training_reward_sum": -.008,
                 "shaping_sum": -.008, "coverage_delta_sum": 0.,
                 "absolute_coverage_delta": 0., "max_coverage": .4, "last_coverage": .4}
        moments = lambda mean, std: {"count": 2, "mean": mean, "std": std, "min": mean - std, "max": mean + std}
        statistics = {"old_values": moments(-.1, .01), "raw_advantages": moments(-.02, .02),
                      "returns": moments(-.12, .03), "advantages": moments(0., 1.)}
        passes = []
        for pindex in range(index):
            p = {"index": pindex, "actor_loss": -.001, "critic_loss": .002,
                 "actor_grad_norm": 1.5, "critic_grad_norm": .1,
                 "critic_mse": .001, "critic_explained_variance": -.2 if index < 3 else .8,
                 "actor_weight_change": {"l2": .01, "max_abs": .001, "changed_tensor_count": 6},
                 "critic_weight_change": {"l2": .02, "max_abs": .002, "changed_tensor_count": 4},
                 "after": {"sample_approx_kl": .0001, "clip_fraction": 0., "transition_count": 200}}
            passes.append(p)
            rows.append({"event": "ppo_pass", "batch": index, **copy.deepcopy(p)})
        rows.append({"event": "batch", "batch": index,
                     "collection": {"controls": 4, "plans": 2, "by_shape": {"T": copy.deepcopy(shape), "L": copy.deepcopy(shape)}},
                     "update": {"data_statistics": statistics, "passes": passes, "completed_passes": index,
                                "entry": {"max_logprob_error": 0., "sample_approx_kl": 0., "clip_fraction": 0.},
                                "frozen_unchanged": True, "frozen_digest": "same", "stop_reason": "completed"}})
    batches = [x for x in rows if x["event"] == "batch"]
    report = {"config_hash": canonical_hash(config), "experiment_id": "synthetic",
              "result": {"controls": 12, "batches": 3, "actor_steps": 6, "critic_steps": 6,
                         "diagnostics_by_shape": aggregate_shapes(batches),
                         "collector": {"by_shape": {s: {"successes": 0} for s in ("T", "L")}}}}
    return rows, report, config


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_population_moments_include_between_group_variance_and_weight(self):
        # [0, 2] and [4, 4, 4, 4]; unequal group sizes matter.
        pooled = pool_moments([{"count": 2, "mean": 1., "std": 1., "min": 0., "max": 2.},
                               {"count": 4, "mean": 4., "std": 0., "min": 4., "max": 4.}])
        self.assertEqual(pooled["count"], 6)
        self.assertEqual(pooled["mean"], 3.)
        self.assertAlmostEqual(pooled["std"], math.sqrt(7 / 3))

    def test_fresh_buffers_not_duplicated_by_repeated_passes_or_log_events(self):
        analysis = analyze_rows(*fixture())
        self.assertEqual(analysis["counts"], {"controls": 12, "plans": 6, "batches": 3,
                                               "actor_steps": 6, "critic_steps": 6})
        self.assertEqual(analysis["mixed_fresh_plan_statistics"]["returns"]["count"], 6)
        self.assertEqual(len(analysis["tables"]["passes"]), 6)
        self.assertEqual(len(analysis["tables"]["shape_batches"]), 6)
        self.assertEqual(analysis["pass_diagnostics"]["negative_explained_variance_count"], 3)
        self.assertEqual([s["controls"] for s in analysis["stages"]], [4, 4, 4])
        self.assertNotIn("returns", analysis["by_shape"]["T"])

    def test_mismatched_duplicate_pass_is_rejected(self):
        rows, report, config = fixture()
        rows[0]["critic_mse"] = 9.
        with self.assertRaisesRegex(ValueError, "Standalone and nested"):
            analyze_rows(rows, report, config)

    def test_report_config_and_physical_count_mismatches_fail_closed(self):
        for mutation in ("config", "count", "shape", "value_target"):
            rows, report, config = fixture()
            if mutation == "config":
                config["optimization"]["gamma"] = .9
            elif mutation == "count":
                report["result"]["controls"] = 13
            elif mutation == "shape":
                rows[1]["collection"]["by_shape"]["T"]["controls"] += 1
            else:
                rows[1]["update"]["data_statistics"]["returns"]["mean"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze_rows(rows, report, config)

    def test_plateau_formula_and_unknown_contacts_are_distinct(self):
        a = analyze_rows(*fixture())
        runs = a["physical_runs"]["flat_positive_coverage_no_contact"]
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["batches"], [1, 2, 3])
        self.assertEqual(runs[0]["controls"], 6)
        self.assertAlmostEqual(runs[0]["training_reward_sum"], runs[0]["expected_flat_training_sum"])
        records = copy.deepcopy(a["tables"]["shape_batches"])
        for row in records:
            if row["shape"] == "L" and row["batch"] == 2:
                row["unknown_contact_controls"] = 2
        split = [r for r in physical_runs(records, .99, 1.)["flat_positive_coverage_no_contact"] if r["shape"] == "L"]
        self.assertEqual([r["batches"] for r in split], [[1], [3]])

    def test_raw_reward_not_added_and_success_uses_terminal_zero(self):
        examples = reward_examples(.99, 1., .95)
        self.assertEqual(examples["zero_stagnation"]["reward"], 0.)
        self.assertAlmostEqual(examples["positive_stagnation"]["reward"], -.004)
        self.assertLess(examples["small_progress"]["reward"], 0.)
        self.assertGreater(examples["larger_progress"]["reward"], 0.)
        self.assertEqual(examples["success"]["next_potential"], 0.)
        self.assertEqual(examples["success"]["raw_reward_preserved"], 17.)
        self.assertAlmostEqual(examples["success"]["reward"], .06)
        telescope = examples["success_telescoping"]
        self.assertAlmostEqual(telescope["measured_discounted_total"], telescope["expected_discounted_bonus_minus_initial_potential"])

    def test_real_prefix_discount_and_true_terminal_vs_timeout_cutoff(self):
        examples = reward_examples(.99, 1., .95)
        success = examples["success_third_control"]
        self.assertEqual(success["actual_controls"], 3)
        self.assertEqual(success["bootstrap_mask"], 0)
        self.assertAlmostEqual(success["physical_discount"], .99 ** 3)
        self.assertAlmostEqual(success["value_target"], success["discounted_reward"])
        timeout, cutoff = examples["five_control_timeout"], examples["five_control_buffer_cutoff"]
        self.assertEqual(timeout, cutoff)
        self.assertEqual(timeout["continuation_mask"], 0)
        self.assertEqual(timeout["bootstrap_mask"], 1)
        self.assertAlmostEqual(timeout["physical_discount"], .99 ** 5)
        self.assertAlmostEqual(timeout["value_target"], .01 * sum(.99 ** i for i in range(5)) + .99 ** 5 * .7)

    def test_checkpoint_metadata_is_only_primitive_and_binding_checked(self):
        rows, report, config = fixture()
        metadata = {"checkpoints": [{"path": "fake.pt", "bytes": 123, "sha256": "a" * 64,
                                    "all_tensors_finite": True, "all_tensors_on_cpu": True,
                                    "optimizer_steps": {"actor": 6, "critic": 6},
                                    "selected_actor_parameters": 6, "critic_parameters": 4,
                                    "metadata": {"config_hash": report["config_hash"], "experiment_id": "other"}}]}
        with self.assertRaisesRegex(ValueError, "metadata binding"):
            analyze_rows(rows, report, config, metadata)


if __name__ == "__main__":
    unittest.main()
