"""Evaluation entry gates and failure isolation; no real model or physics."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_c_policy_pair as cli


class EvaluationEntryChecks(unittest.TestCase):
    def test_default_plan_is_stdlib_and_has_only_fixed_cases(self):
        code = "import runpy,sys;sys.argv=['plan'];runpy.run_path('scripts/evaluate_c_policy_pair.py',run_name='__main__')"
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=cli.ROOT,
                                text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["runtime_started"])
        self.assertEqual(plan["training_steps"], 0)
        self.assertEqual(len(plan["cases"]), 20)
        self.assertEqual(plan["cases"], cli.ordered_cases())
        self.assertEqual(plan["binding"]["methods"], list(cli.METHODS))

    def test_execute_opt_in_precedes_asset_reads(self):
        args = cli.parser().parse_args(["run"])
        with patch.object(cli, "sha", side_effect=AssertionError("asset access")):
            with self.assertRaisesRegex(ValueError, "Runtime requires"):
                cli.prepare(args)

    def test_preservation_checks_changed_bytes_and_added_paths(self):
        with tempfile.TemporaryDirectory(dir=cli.ROOT) as directory:
            root = Path(directory)
            path = root / "original.json"
            path.write_text("old")
            inventory = {"files": {str(path): {"bytes": 3, "sha256": cli.sha(path)}},
                         "trees": {str(root): [str(path)]}}
            self.assertTrue(cli.unchanged(inventory))
            added = root / "extra.json"
            added.write_text("new")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                cli.unchanged(inventory)
            added.unlink()
            path.write_text("bad")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                cli.unchanged(inventory)

    def test_nested_output_rejected_before_delta_or_model_access(self):
        config, _, _ = cli.prepare(cli.parser().parse_args([]))
        with tempfile.TemporaryDirectory(dir=cli.ROOT / "artifacts") as directory:
            parent = Path(directory)
            training = parent / "training"
            training.mkdir()
            authority = parent / "authority" / cli.TRAINING_ID
            authority.mkdir(parents=True)
            ledger = authority / "ledger.json"
            ledger.write_text(json.dumps({"experiment_id": cli.TRAINING_ID, "config_hash": config.config_hash,
                "config": config.to_dict(), "pending": {}, "completed": {"train": {"control": 2048},
                "global": {"actor_step": 52, "critic_step": 52, "batch": 18}}}))
            args = cli.parser().parse_args(["run", "--execute", "--account-id", "unit",
                "--registry", str(parent / "eval-registry"), "--run-dir", str(training / "nested"),
                "--source-root", "unused", "--base-checkpoint", "unused", "--checkpoint", str(training / "final.pt"),
                "--training-ledger", str(ledger)])
            with patch.object(cli, "sha", side_effect=AssertionError("asset access")):
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    cli.prepare(args)

    def assert_no_torch_before_gate(self, budget):
        original_import = builtins.__import__
        imports = []
        def guarded_import(name, *args, **kwargs):
            if name == "torch":
                imports.append(name)
                raise AssertionError("Torch import before budget gate")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import), patch.object(cli, "unchanged", return_value=True):
            with self.assertRaises(RuntimeError):
                cli.evaluate_one(None, None, budget, "base_training", {})
        self.assertFalse(imports)

    def test_expired_account_rejects_before_torch(self):
        budget = Mock()
        budget.check_time.side_effect = RuntimeError("expired")
        self.assert_no_torch_before_gate(budget)
        budget.claim_evaluation_group.assert_not_called()

    def test_consumed_group_rejects_before_torch(self):
        budget = Mock()
        budget.claim_evaluation_group.side_effect = RuntimeError("consumed")
        self.assert_no_torch_before_gate(budget)

    def test_live_model_prevents_loading_next_group(self):
        args = SimpleNamespace(run_dir=Path("unused"))
        with patch.object(cli, "write_json"), patch.object(cli, "evaluate_one", return_value=({"status": "completed"}, lambda: object())) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "remains resident"):
                cli.run_pair(args, None, Mock(), {})
        self.assertEqual(evaluate.call_count, 1)

    def test_incomplete_group_does_not_start_next_or_claim_comparison(self):
        args, config = SimpleNamespace(run_dir=Path("unused"), account_id="unit"), SimpleNamespace(config_hash="a" * 64, to_dict=lambda: {})
        with patch.object(cli, "write_json"), patch.object(cli, "unchanged", return_value=True), \
                patch.object(cli, "cuda_resources", return_value={}), \
                patch.object(cli, "evaluate_one", return_value=({"status": "incomplete"}, lambda: None)) as evaluate:
            result = cli.run_pair(args, config, Mock(), {})
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["comparison"]["available"])
        self.assertEqual(result["training_steps"], 0)


if __name__ == "__main__":
    unittest.main()
