"""Near-miss timing and failure-safe observation of an existing sampler."""

from types import SimpleNamespace
import unittest

from tdyssey.policy_diagnostics import capture_scheduler_steps, near_miss_facts, post_success_facts


class PolicyDiagnosticChecks(unittest.TestCase):
    def test_retention_excludes_first_success_and_records_loss_and_reentry(self):
        rows = [{"done": done, "metrics": {"coverage": coverage, "iou": coverage / (2 - coverage)}}
                for done, coverage in ((False, .8), (True, .96), (True, .97), (False, .95), (True, .98), (False, .7))]
        facts = post_success_facts(rows)
        self.assertEqual(facts["first_success_control"], 2)
        self.assertEqual(facts["post_success_controls"], 4)
        self.assertEqual(facts["first_loss_of_success_control"], 4)
        self.assertEqual(facts["post_success_success_fraction"], .5)
        self.assertEqual(facts["post_success_min_coverage"], .7)
        self.assertFalse(facts["final_success"])
        self.assertIsNone(post_success_facts(rows[:1])["post_success_success_fraction"])

    def test_peak_is_inside_chunk_and_not_success(self):
        rows = [{"metrics": {"iou": value, "coverage": coverage}, "upstream_done": False}
                for value, coverage in ((.1, .2), (.8, .9), (.8, .9), (.7, .85), (.6, .8))]
        facts = near_miss_facts({"steps": rows}, action_horizon=4)
        self.assertEqual(facts["best_control"], 2)
        self.assertEqual(facts["planned_controls"], [1, 4])
        self.assertEqual(facts["best_action_slot_1_based"], 2)
        self.assertEqual(facts["first_iou_drop_before_replan"], 4)
        self.assertFalse(facts["best_exceeds_success_threshold"])
        self.assertEqual(facts["remaining_controls_before_replan"], 2)

    def test_scheduler_is_restored_after_capture_fails(self):
        class Scheduler:
            def step(self, *args):
                raise RuntimeError("sample failed")

        scheduler = Scheduler()
        original = scheduler.step
        sample = SimpleNamespace(detach=lambda: SimpleNamespace(clone=lambda: "copied"))
        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            with capture_scheduler_steps(scheduler):
                scheduler.step(None, 99, sample)
        self.assertEqual(scheduler.step, original)
        self.assertNotIn("step", vars(scheduler))


if __name__ == "__main__":
    unittest.main()
