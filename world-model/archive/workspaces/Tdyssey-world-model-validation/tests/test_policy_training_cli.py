"""Standard-library CLI preflight checks; never start a training run or load assets."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_official_policy.py"
SPEC = importlib.util.spec_from_file_location("tdyssey_training_cli_checks", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class PolicyTrainingCLIChecks(unittest.TestCase):
    def arguments(self, *extra):
        # Parsing and validate_args never inspect these nonexistent asset paths.
        return CLI.parser().parse_args(["run", "--source-root", "unused-source",
                                         "--base-checkpoint", "unused-base.pt",
                                         "--run-dir", "unused-run", *extra])

    def test_default_reserves_two_inference_plans_within_eight_plan_cap(self):
        args = self.arguments()
        controls, plans = CLI.validate_args(args)
        self.assertEqual(controls, (24, 24))
        self.assertEqual(plans, (3, 3))
        self.assertEqual(sum(controls), 48)
        self.assertEqual(sum(plans) + 2, 8)
        for control_cap, plan_cap in zip(controls, plans):
            self.assertEqual(control_cap, plan_cap * 8)
        small_controls, small_plans = CLI.validate_args(self.arguments("--controls", "16", "--max-plans", "4"))
        self.assertEqual((small_controls, small_plans), ((8, 8), (1, 1)))

    def test_excess_or_inconsistent_budgets_fail_before_optional_runtime(self):
        for flags in (("--controls", "65"), ("--max-controls", "65"),
                      ("--max-plans", "9"), ("--max-plans", "3"),
                      ("--controls", "1"), ("--controls", "49"),
                      ("--controls", "17", "--max-plans", "4"),
                      ("--threads", "5"), ("--threads", "0"),
                      ("--episode-limit", "0"), ("--episode-limit", "100001")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                CLI.validate_args(self.arguments(*flags))

    def test_seed_ranges_and_known_development_scenes_are_excluded(self):
        for flags in (("--seed", "-1"), ("--seed", str(2**32)),
                      ("--t-scene", "100000"), ("--l-scene", "100002")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                CLI.validate_args(self.arguments(*flags))

    def test_help_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-B", str(SCRIPT), "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("resume-update", result.stdout)
        self.assertIn("Includes two nonphysical replay plans", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_invalid_flags_and_plan_capacity_do_not_create_output_without_site_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_path = Path(temporary) / "must-not-be-created"
            base = [sys.executable, "-S", "-B", str(SCRIPT), "run", "--source-root", "unused-source",
                    "--base-checkpoint", "unused-base.pt", "--run-dir", str(run_path)]
            for flags, error in ((["--controls", "49"], "reserving two replay plans"),
                                 (["--max-controls", "65"], "max-controls <= 64"),
                                 (["--max-plans", "9"], "max-plans <= 8"),
                                 (["--unknown-option"], "unrecognized arguments")):
                with self.subTest(flags=flags):
                    result = subprocess.run([*base, *flags], cwd=ROOT, capture_output=True, text=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)
                    self.assertFalse(run_path.exists())


if __name__ == "__main__":
    unittest.main()
