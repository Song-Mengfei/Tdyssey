"""C CLI gates are checked without loading assets or performing interaction."""

from copy import deepcopy
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_c_policy.py"
spec = importlib.util.spec_from_file_location("c_cli", ENTRY)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
from tdyssey.policy_protocol import BudgetExceeded, CConfig, ExperimentBudget

try:
    import torch
except ImportError:
    torch = None


class StorageBudget:
    """Fake storage/provenance API; never reserves physical/optimizer work."""
    def __init__(self):
        self.writes, self.groups = [], []

    def check_time(self):
        return 100.

    def check_storage(self):
        return {"current_bytes": 0, "pending_bytes": 0, "limit_bytes": 1024**2}

    def reserve_storage(self, amount, *, path):
        assert type(amount) is int and amount > 0
        self.writes.append((Path(path), amount, Path(path).stat().st_size if Path(path).exists() else 0))
        return len(self.writes)

    def mark_storage_written(self, reservation, amount):
        path, limit, previous = self.writes[reservation - 1]
        assert 0 <= amount <= limit and path.stat().st_size == previous + amount
        return {"retire_paths": ["never-delete-from-cli.pt"]}

    def snapshot(self):
        return {"reserved": {"train": {"control": 3}}, "completed": {"train": {"control": 3}}}

    def claim_evaluation_group(self, group, **metadata):
        self.groups.append((group, metadata))

    def complete_evaluation_group(self, group, digest):
        assert self.groups[-1][0] == group and len(digest) == 64

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class CLIContracts(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-S", "-B", str(ENTRY), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)

    def test_default_plan_and_help_are_stdlib_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            result = self.invoke("--run-dir", output, "--registry", output / "registry")
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["status"], "proposed_not_executed")
            self.assertFalse(plan["runtime_started"])
            self.assertEqual(plan["physical_controls"], 0)
            self.assertEqual(plan["optimizer_steps"], 0)
            self.assertFalse(output.exists())
            self.assertEqual(self.invoke("--help").returncode, 0)

    def test_every_runtime_mode_requires_opt_in_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            for mode in ("init", "train", "evaluate"):
                with self.subTest(mode=mode):
                    result = self.invoke(mode, "--run-dir", output, "--registry", output / "registry")
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("explicit --execute", result.stderr)
                    self.assertNotIn("torch", result.stderr)
                    self.assertFalse(output.exists())

    def test_bad_budget_and_method_fail_before_runtime_import_or_ledger(self):
        common = ["--execute", "--experiment-id", "unit", "--source-root", "missing", "--base-checkpoint", "missing"]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            common += ["--run-dir", output, "--registry", output / "registry"]
            for controls, batches in ((2049, 24), (128, 25), (129, 1), (0, 1)):
                result = self.invoke("train", *common, "--controls", controls, "--batches", batches)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("torch", result.stderr)
                self.assertFalse(output.exists())
            result = self.invoke("evaluate", *common, "--method", "delta_training",
                                 "--protocol", "matched_tl", "--group", "one")
            self.assertIn("requires --checkpoint", result.stderr)
            self.assertFalse(output.exists())

    def test_training_requires_explicit_cumulative_targets(self):
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        args = cli.parser().parse_args(["train", "--execute", "--experiment-id", "unit",
            "--run-dir", "unused", "--source-root", "unused", "--base-checkpoint", "unused"])
        with self.assertRaisesRegex(ValueError, "Explicit --controls"):
            cli.validate_args(args, config)
        args.controls, args.batches = 1024, 8
        cli.validate_args(args, config)
        args.mode = "evaluate"
        args.resume_checkpoint = Path("B-update.pt")
        args.method, args.protocol, args.group = "base_original", "matched_tl", "one"
        with self.assertRaisesRegex(ValueError, "only to training"):
            cli.validate_args(args, config)

    def test_configuration_is_hash_stable_and_rejects_twenty_thousand_controls(self):
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        self.assertEqual(config.config_hash, CConfig.from_dict(config.to_dict()).config_hash)
        proposed = config.to_dict()
        proposed["budgets"]["train"]["control"] = 20000
        with self.assertRaises(ValueError):
            CConfig.from_dict(proposed)

    def test_identifiers_follow_permanent_registry_rules_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "never-created"
            for identifier in ("Upper", "has.period", "con", "COM1", "nul"):
                result = self.invoke("init", "--execute", "--experiment-id", identifier,
                                     "--run-dir", output, "--registry", output / "registry")
                self.assertEqual(result.returncode, 1)
                self.assertIn("safe lowercase identifier", result.stderr)
                self.assertFalse(output.exists())

    def test_offline_compare_is_stdlib_only_and_never_starts_runtime(self):
        def report(method):
            return {"schema_version": 1, "kind": "tdyssey.policy_development_evaluation",
                    "method": method, "sampler_mode": "baseline" if method == "base_original" else "training",
                    "protocol": "official-t-l-com-v1", "max_controls": 300, "phase": "development",
                    "success_observation": "after_physical_control", "delta_applied": False,
                    "checkpoint_identity": {"base_sha256": "b" * 64, "delta_sha256": None},
                    "status": "completed", "completed_case_count": 1,
                    "cases": [{"shape": "T", "scene_seed": 100000, "policy_seed": 5478,
                               "status": "completed", "success": True, "success_step": 2,
                               "final_metrics": {"coverage": .96, "iou": .93},
                               "peak_metrics": {"coverage": .96, "iou": .93},
                               "counts": {"control_steps": 2},
                               "latency": {"planning_seconds": [.1], "total_seconds": .2}}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "base.json", root / "training.json"]
            for path, method in zip(paths, ("base_original", "base_training")):
                path.write_text(json.dumps(report(method)), encoding="utf-8")
            output = root / "never-created"
            result = self.invoke("compare", "--reports", *paths, "--run-dir", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["case_count"], 1)
            self.assertFalse(output.exists())

    def test_storage_helpers_use_increment_bytes_and_real_ledger_reopens(self):
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            root, fake = Path(temporary), StorageBudget()
            cli._write_json(root / "one.json", {"ok": True}, fake)
            cli._append_log(root / "fake.jsonl", {"row": 1}, fake)
            cli._append_log(root / "fake.jsonl", {"row": 2}, fake)
            self.assertEqual(len(fake.writes), 3)
            registry, output = root / "registry", root / "run"
            log = output / "training.jsonl"
            with ExperimentBudget(registry, "cli-storage", config, output, create=True) as budget:
                cli._append_log(log, {"row": 1}, budget)
                cli._append_log(log, {"row": 2}, budget)
                self.assertEqual(budget.snapshot()["written_files"][str(log.resolve())]["bytes"], log.stat().st_size)
            with ExperimentBudget(registry, "cli-storage", config, output) as budget:
                self.assertEqual(budget.snapshot()["pending"], {})
                self.assertEqual([json.loads(line)["row"] for line in log.read_text().splitlines()], [1, 2])
                self.assertEqual(budget.reserved_count("control"), 0)
                self.assertEqual(budget.reserved_count("actor_step"), 0)

    def test_expired_or_full_authority_blocks_dispatch_before_model_or_asset_access(self):
        for exhausted in ("check_time", "check_storage"):
            with self.subTest(exhausted=exhausted), tempfile.TemporaryDirectory() as temporary:
                budget = StorageBudget()
                setattr(budget, exhausted, Mock(side_effect=BudgetExceeded("Synthetic exhausted authority")))
                # Failure telemetry must obey the same exhausted cap.
                budget.reserve_storage = Mock(side_effect=BudgetExceeded("Synthetic exhausted authority"))
                argv = [str(ENTRY), "train", "--execute", "--experiment-id", "unit",
                        "--run-dir", str(Path(temporary) / "never-created"), "--source-root", "missing",
                        "--base-checkpoint", "missing", "--controls", "128", "--batches", "1"]
                with patch.object(sys, "argv", argv), patch("tdyssey.policy_protocol.ExperimentBudget", return_value=budget):
                    with patch.object(cli, "run_train") as train, redirect_stderr(io.StringIO()):
                        self.assertEqual(cli.main(), 1)
                    train.assert_not_called()
                self.assertFalse((Path(temporary) / "never-created").exists())

    def test_cleanup_closes_all_resources_and_failure_audit_never_checkpoints(self):
        first, second = Mock(), Mock()
        first.close.side_effect = RuntimeError("Synthetic close failure")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            cli._close_environments((first, second))
        self.assertEqual(second.close.call_count, 1)
        with redirect_stderr(io.StringIO()):
            cli._close_environments((first, second), original_error=ValueError("Original failure"))
        self.assertEqual(second.close.call_count, 2)
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            args = cli.parser().parse_args(["train", "--run-dir", temporary, "--experiment-id", "unit"])
            with redirect_stderr(io.StringIO()):
                path = cli._record_failure(args, config, StorageBudget(), RuntimeError("Synthetic runtime failure"))
            failure = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(failure["counts"]["reserved"]["train"]["control"], 3)
            self.assertFalse(failure["failure_handler_created_checkpoint"])
            self.assertFalse(failure["safe_resume_claim"])
            self.assertEqual(list(Path(temporary).glob("*.pt")), [])

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_evaluation_dispatch_loads_one_method_and_preserves_verified_identity(self):
        from types import SimpleNamespace
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        for method in ("base_original", "base_training", "delta_training"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                delta = root / "tiny-delta.pt"
                delta.write_bytes(b"synthetic-delta-file-only")
                argv = ["evaluate", "--execute", "--experiment-id", "unit", "--run-dir", str(root),
                        "--source-root", "unused", "--base-checkpoint", "unused", "--method", method,
                        "--protocol", "matched_tl", "--group", "one"]
                if method == "delta_training":
                    argv += ["--checkpoint", str(delta)]
                args = cli.parser().parse_args(argv)
                actor, official, budget = SimpleNamespace(), SimpleNamespace(), StorageBudget()
                loaded = SimpleNamespace(actor=actor, official=official)
                env_class = Mock()
                with patch.object(cli, "_runtime", return_value="cpu"), \
                        patch.object(cli, "_new_actor", return_value=(official, actor)) as base_loader, \
                        patch.object(cli, "_preview_delta") as preview, \
                        patch("tdyssey.policy_checkpoint.load_update_checkpoint", return_value=loaded) as delta_loader, \
                        patch("tdyssey.official.load_official_source", return_value=object()), \
                        patch("tdyssey.envs.official.make_official_shape_env_class", return_value=env_class), \
                        patch("tdyssey.policy_evaluation.evaluate_cases", return_value={"status": "completed", "success_fraction": .5}) as evaluate:
                    result = cli.run_evaluation(args, config, budget)
                env_class.assert_not_called()
                self.assertEqual(len(evaluate.call_args.args[3]), 20)
                self.assertEqual(evaluate.call_args.kwargs["method"], method)
                self.assertEqual(actor.evaluation_checkpoint_identity["base_sha256"], config.to_dict()["base"]["sha256"])
                self.assertEqual(evaluate.call_args.kwargs["delta_applied"], method == "delta_training")
                self.assertEqual(len(budget.groups), 1)
                if method == "delta_training":
                    base_loader.assert_not_called()
                    preview.assert_called_once()
                    delta_loader.assert_called_once()
                    self.assertFalse(delta_loader.call_args.kwargs["restore_rng"])
                    self.assertIsNotNone(actor.evaluation_checkpoint_identity["delta_sha256"])
                else:
                    base_loader.assert_called_once()
                    preview.assert_not_called()
                    delta_loader.assert_not_called()
                    self.assertIsNone(actor.evaluation_checkpoint_identity["delta_sha256"])
                self.assertEqual(json.loads(Path(result["report"]).read_text())["comparison_label"], "one")

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_delta_preview_refuses_b_wrong_identity_and_active_episode(self):
        config = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        metadata = {"kind": "tdyssey-C", "schema_version": 1, "experiment_id": "unit",
                    "config": config.to_dict(), "config_hash": config.config_hash, "resumable": True}
        payload = {"metadata": metadata, "base_identity": config.to_dict()["base"]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.pt"
            for mutation, expected in (({"kind": "B"}, "C delta"), ({"experiment_id": "other"}, "identity"),
                                       ({"resumable": False}, "cannot resume")):
                candidate = deepcopy(payload)
                candidate["metadata"].update(mutation)
                torch.save(candidate, path)
                with self.assertRaisesRegex(ValueError, expected):
                    cli._preview_delta(path, config, "unit", resumable=True)
            torch.save(payload, path)
            self.assertEqual(cli._preview_delta(path, config, "unit", resumable=True), metadata)
            for malformed in ([], {"metadata": None}, {"metadata": []}):
                torch.save(malformed, path)
                with self.assertRaisesRegex(ValueError, "C delta"):
                    cli._preview_delta(path, config)


if __name__ == "__main__":
    unittest.main()
