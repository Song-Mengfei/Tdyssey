"""Small exact-resume CPU fixtures; no Torch import, model or optimizer work."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_stage3c_continuation as continuation

ROOT = Path(__file__).resolve().parents[1]


def restore_checkpoint(*args, **kwargs):
    raise AssertionError("Original fixture globals must never be patched or called")


def source_identity():
    raise AssertionError("Original fixture globals must never be patched or called")


def save_training_checkpoint(*args, **kwargs):
    raise AssertionError("Original fixture globals must never be patched or called")


class State:
    def __init__(self, value, corrupt=False):
        self.value = deepcopy(value)
        self.corrupt = corrupt

    def state_dict(self):
        return deepcopy(self.value)

    def load_state_dict(self, value, **kwargs):
        self.value = deepcopy(value)
        if self.corrupt:
            self.value["corruption"] = True


def prior_ledger():
    return dict(kind="native_gpc_rgb_cost.v1", max_updates=3000, max_seconds=5400.,
        charged_updates=129, completed_updates=129, charged_seconds=continuation.PRIOR_SECONDS,
        active_stage=None, historical_cancelled_experiment={"attempts": 1899, "seconds": 5352.516},
        stages=[dict(token="s3", phase="stage3-visible-microfit", reserved_updates=0, completed_updates=0,
                     observed_seconds=28.3440000000046, status="failed:range"),
                dict(token="s3b", phase="stage3b-visible-microfit", reserved_updates=129, completed_updates=129,
                     observed_seconds=416.2810000000027, status="bounded_stop")])


class Fixture:
    def __init__(self, root):
        self.root = root
        self.parent_run = root / "stage3b/run"
        self.parent_run.mkdir(parents=True)
        self.checkpoint = self.parent_run / "last.pt"
        self.checkpoint.write_bytes(b"fake complete parent; no weights")
        self.output = root / "stage3c/run"
        self.output.parent.mkdir()
        self.review = self.output.parent / "restore-review.json"
        self.prior = prior_ledger()
        self.prior_copy = self.output.parent / "prior-ledger.json"
        self.prior_copy.write_text(json.dumps(self.prior), encoding="utf-8")
        self.work_file = root / "stage3b/model-work.json"
        self.work_file.write_text(json.dumps(dict(budget=dict(ledger=self.prior))), encoding="utf-8")
        self.additions = {}
        for name in continuation.SOURCE_ADDITIONS:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source: " + name, encoding="utf-8")
            self.additions[name] = continuation._sha256_file(path)
        self.old_source = dict(files={"unchanged-core.py": "oldhash"}, original_gpc_commit="fixed")
        self.parent_config = dict(lr=1e-4, max_updates=3000, microbatch=4, effective_batch=16,
                                  train_indices=list(range(18)), validation_indices=list(range(36)))
        self.identity = dict(source=self.old_source, phase="stage3b-visible-microfit", data={"split": "fixed"},
                             objective={"H1": True}, model_work_budget={"old": True})
        self.model_value = {f"p{i}": [float(i), .25] for i in range(296)}
        self.parameter_hashes = {k: continuation.state_digest(v) for k, v in self.model_value.items()}
        self.progress = dict(updates=129, samples=2064, attempted_draws=2064,
            initial_parameter_sha256={k: "published" for k in self.model_value},
            first_gradient_norms={k: .1 for k in self.model_value}, training_seconds=415.719,
            phase="stage3b-visible-microfit", stop_reason="stage_gradient_cutoff_with_reserved_tail")
        lr = 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * 129 / 3000))
        group = dict(lr=lr, initial_lr=1e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)
        old_ledger = deepcopy(self.prior)
        old_ledger.update(stages=self.prior["stages"][:1], charged_seconds=444.,
                          active_stage={"token": "s3b"})
        self.payload = dict(kind="fixture", config=self.parent_config, identity=self.identity, progress=self.progress,
            model=self.model_value, optimizer=dict(state={i: dict(step=129., exp_avg=[.1], exp_avg_sq=[.2]) for i in range(296)}, param_groups=[group]),
            scheduler=dict(T_max=3000, last_epoch=129, eta_min=1e-5, base_lrs=[1e-4], _last_lr=[lr]),
            scaler=dict(scale=128.), order=dict(cursor=2064, rng={"next": [9, 8]}),
            rng=dict(python=(1, 2), numpy=[3, 4], torch=b"rng", cuda=[b"cuda"], noise_generator=b"noise"), ledger=old_ledger)
        predictions = root / "stage3b/predictions"
        predictions.mkdir()
        (predictions / "fitted-train18.json").write_text(json.dumps(dict(parameter_sha256=self.parameter_hashes)), encoding="utf-8")
        self.parent_spec = dict(config=deepcopy(self.parent_config), identity=deepcopy(self.identity),
            progress=deepcopy(self.progress), last=dict(path=str(self.checkpoint), bytes=self.checkpoint.stat().st_size,
                sha256=continuation.PARENT_SHA256), checkpoint_sha256=continuation.PARENT_SHA256,
            model_work_sha256=continuation._sha256_file(self.work_file))
        self.delta = dict(kind="native-gpc-stage3c-exact-continuation.v1", output=str(self.output),
            stage_max_updates=121, stage_max_model_work_seconds=600., no_new_update_after_seconds=420.,
            prior_stage="3B", prior_charged_model_seconds=continuation.PRIOR_SECONDS, prior_charged_updates=129,
            prior_ledger_sha256=continuation._sha256_file(self.prior_copy), prior_model_work_file=str(self.work_file),
            parent_checkpoint=str(self.checkpoint), parent_checkpoint_sha256=continuation.PARENT_SHA256,
            continuation_start_updates=129, continuation_target_updates=250)
        self.config = dict(self.parent_config, **self.delta)
        current = deepcopy(self.prior)
        current["active_stage"] = dict(token="s3c", phase=continuation.PHASE, seconds_before=continuation.PRIOR_SECONDS,
                                       reserved_updates=0, completed_updates=0)
        self.ledger = SimpleNamespace(snapshot=lambda: deepcopy(current), max_updates=3000, max_seconds=5400.)
        self.budget = SimpleNamespace(ledger=self.ledger, stage=SimpleNamespace(token="s3c"), max_attempts=121,
            max_seconds=600., latest_update_seconds=420., initial_tail_reserve_seconds=180., elapsed=2.,
            require_operation=lambda *args: None, before_checkpoint=lambda *args: None,
            identity=lambda: dict(max_attempts=121, max_seconds=600., cutoff=420.))
        self.saved = []
        self.save_rejections = []
        self.load_count = 0
        self.corrupt = None
        self.rng = None
        fixture = self

        def train(config, output, data_root, source_root, init_checkpoint, **kwargs):
            identity = deepcopy(fixture.identity)
            identity.update(source=source_identity(), phase=kwargs["phase"], model_work_budget=kwargs["work_budget"].identity())
            model = State({"fresh": "never checkpoint this"}, fixture.corrupt == "model")
            optimizer = State({}, fixture.corrupt == "optimizer")
            scaler, scheduler, order = State({}), State({}), State({})
            try:
                progress = restore_checkpoint(kwargs["resume"], model, optimizer, scaler, scheduler,
                                              "generator", order, config, identity)
            except BaseException:
                try:
                    save_training_checkpoint(Path(output) / "last.pt", model, optimizer, scaler, scheduler,
                        "generator", order, dict(updates=0, samples=0, phase=kwargs["phase"]), config, identity)
                except RuntimeError as error:
                    fixture.save_rejections.append(str(error))
                raise
            return dict(progress=progress, identity=identity, config=config, target=kwargs["updates"])

        def load(path):
            self.load_count += 1
            return deepcopy(self.payload)

        def save(path, model, optimizer, scaler, scheduler, generator, order, progress, config, identity, **kwargs):
            self.saved.append(deepcopy(progress))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"verified complete129")
            return dict(path=str(path), updates=progress["updates"], sha256="fixture-only")

        def restore_rng(value, generator):
            self.rng = deepcopy(value)

        def atomic_json(path, value):
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        self.core = SimpleNamespace(ROOT=root, train=train, source_identity=lambda: deepcopy(self.old_source),
            load_checkpoint_payload=load, save_training_checkpoint=save, _cpu_tree=deepcopy,
            restore_rng=restore_rng, capture_rng=lambda generator: deepcopy(self.rng), atomic_json=atomic_json,
            tensor_hashes=lambda model: {k: continuation.state_digest(v) for k, v in model.value.items()})

    def run(self, **overrides):
        args = dict(parent_spec=self.parent_spec, allowed_config_changes=self.delta, source_additions=self.additions,
            review_path=self.review, resume=self.checkpoint, work_budget=self.budget, execute=True,
            phase=continuation.PHASE, updates=250, _core_module=self.core)
        args.update(overrides)
        real_sha = continuation._sha256_file
        with patch.object(continuation, "_sha256_file", side_effect=lambda path: continuation.PARENT_SHA256 if Path(path).resolve() == self.checkpoint else real_sha(path)):
            return continuation.train_continuation(self.config, self.output, "data", "gpc", "published", **args)


class ContinuationTests(unittest.TestCase):
    def test_complete_state_and_rng_restore_same_bytecode_private_globals_and_complete129_save(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            f = Fixture(Path(directory).resolve())
            untouched = f.core.train.__globals__["restore_checkpoint"]
            result = f.run()
            review = json.loads(f.review.read_text(encoding="utf-8"))
            self.assertEqual(review["protected_before_sha256"], review["protected_after_sha256"])
            self.assertTrue(review["all_protected_state_equal"])
            self.assertEqual(set(review["private_global_overrides"]), {"restore_checkpoint", "source_identity", "save_training_checkpoint"})
            self.assertEqual(result["target"], 250)
            self.assertEqual(result["progress"]["updates"], 129)
            self.assertEqual(result["progress"]["samples"], 2064)
            self.assertNotIn("stop_reason", result["progress"])
            self.assertEqual(result["progress"]["initial_parameter_sha256"], f.progress["initial_parameter_sha256"])
            self.assertEqual(result["progress"]["first_gradient_norms"], f.progress["first_gradient_norms"])
            self.assertEqual(result["progress"]["stage3c_resume_parameter_sha256"], f.parameter_hashes)
            self.assertEqual(f.saved[0]["updates"], 129)
            self.assertEqual(len(f.saved), 1)
            self.assertEqual(f.checkpoint.read_bytes(), b"fake complete parent; no weights")
            self.assertIs(f.core.train.__globals__["restore_checkpoint"], untouched)
            self.assertEqual(f.load_count, 1)

    def test_semantic_config_change_and_wrong_total_are_rejected_before_load(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            f = Fixture(Path(directory).resolve())
            f.config["lr"] = .0002
            with self.assertRaisesRegex(ValueError, "literal metadata delta"):
                f.run()
            self.assertEqual(f.load_count, 0)
            f.config["lr"] = .0001
            with self.assertRaisesRegex(ValueError, "total250"):
                f.run(updates=121)
            extra = dict(f.delta, lr=.0002)
            with self.assertRaisesRegex(ValueError, "fourteen"):
                f.run(allowed_config_changes=extra)
            self.assertFalse((f.output / "last.pt").exists())

    def test_rejected_parent_or_corrupted_restore_cannot_save_fresh_or_unverified_state(self):
        for reason in ["parent_identity", "optimizer", "model"]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
                f = Fixture(Path(directory).resolve())
                if reason == "parent_identity":
                    f.payload["identity"]["data"] = {"split": "wrong"}
                else:
                    f.corrupt = reason
                with self.assertRaises(ValueError):
                    f.run()
                self.assertEqual(f.saved, [])
                self.assertEqual(len(f.save_rejections), 1)
                self.assertFalse((f.output / "last.pt").exists())
                self.assertFalse(f.review.exists())
                self.assertEqual(f.checkpoint.read_bytes(), b"fake complete parent; no weights")

    def test_source_or_parent_boundary_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            f = Fixture(Path(directory).resolve())
            additions = dict(f.additions)
            additions["scripts/native_rgb_stage3c_run.py"] = "wrong"
            with self.assertRaisesRegex(ValueError, "source hash"):
                f.run(source_additions=additions)
            f.payload["order"]["cursor"] = 0
            with self.assertRaisesRegex(ValueError, "complete129"):
                f.run()
            self.assertEqual(f.saved, [])

    def test_existing_cost_and_failed_attempt_are_never_refunded_local121_is_not_target250(self):
        from tdyssey import native_rgb_training as training
        from tdyssey.native_rgb_stage_budget import ModelWorkBudget, UpdateLimitReached
        clock = SimpleNamespace(now=0.)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory, patch.object(training.time, "monotonic", lambda: clock.now):
            file = Path(directory) / "ledger.json"
            prior = prior_ledger()
            file.write_text(json.dumps(prior), encoding="utf-8")
            limits = continuation.validate_closed_prior_ledger(deepcopy(prior), prior)
            ledger = training.CostLedger(file, 3000, 5400.)
            budget = ModelWorkBudget(ledger.start(continuation.PHASE), **limits)
            for i in range(121):
                budget.reserve_update()
                if i < 120:
                    budget.complete_update()
            with self.assertRaises(UpdateLimitReached):
                budget.reserve_update()
            clock.now = 600.
            with self.assertRaises(training.BudgetExceeded):
                budget.heartbeat()
            budget.finish("mock_failed_attempt_remains_spent")
            value = ledger.snapshot()
            self.assertEqual(value["charged_updates"], 250)
            self.assertEqual(value["completed_updates"], 249)
            self.assertEqual(value["charged_seconds"], continuation.PRIOR_SECONDS + 600.)
            self.assertEqual(value["stages"][:2], prior["stages"])
            self.assertEqual((value["max_updates"], value["max_seconds"]), (3000, 5400.))
            self.assertIsNone(value["active_stage"])


if __name__ == "__main__":
    unittest.main()
