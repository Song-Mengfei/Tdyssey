"""Pure marker evaluation: no Torch/NumPy, model, SDK, physics or real assets.

The full state-machine fixture uses a file journal with fsync replaced by a
marker. It proves accounting/content/order, not crash durability or raw RGB I/O.
"""
import builtins
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from tdyssey import gpc_checkpoint_evaluation as engine
from tdyssey import gpc_adaptation as adaptation
from tdyssey import gpc_adaptation_runtime as runtime
from tdyssey import gpc_supervised_adaptation as objective
from tdyssey import gpc_evaluation_assets as asset_api


EVIDENCE = adaptation.ROOT/"artifacts/gpc-checkpoint-evaluation-engineering-20261026/marker-fixtures"
CANDIDATES = ("hold", "toward", "side", "away")


def cases():
    result = []
    for role, anchors in (("regression", [("R%02d" % i, "T" if i < 8 else "L", i) for i in range(16)]),
                          ("independent", [("I%s%02d" % (shape, i), shape, i) for shape in ("T", "L") for i in range(6)])):
        for anchor, shape, ordinal in anchors:
            for candidate in CANDIDATES:
                if role == "regression" and anchor == "R15" and candidate == "away":
                    continue
                score = dict(hold=.1, toward=.3, side=.2, away=.05)[candidate]
                latest = dict(pose=dict(position=[10., 20.], angle=0.), pusher=[3., 4.], coverage=.01)
                turning = role == "independent" and ordinal == 0 and candidate in ("toward", "side")
                truth = [dict(pose=dict(position=[10.+h, 20.], angle=.12 if turning else 0.),
                              pusher=[3.+h, 4.], coverage=score*h/4) for h in range(1, 5)]
                result.append(dict(role=role, anchor=anchor, shape=shape, candidate=candidate,
                    truth=truth, latest=latest, goal=dict(position=[50., 50.], angle=0.),
                    contact_by_h4=role == "independent" and ordinal == 0,
                    past_actions=[[1., 2.], [3., 4.], [5., 6.]],
                    future_actions=[[7.+h, 8.+h] for h in range(4)]))
    return result


class MemoryWriter:
    def __init__(self, directory):
        self.directory = directory; directory.mkdir()
        self.documents, self.raw_names, self.measurement_count = {}, set(), 0

    def raw(self, name, value):
        if name in self.raw_names:
            raise AssertionError("Duplicate fresh prediction output")
        self.raw_names.add(name)
        return name

    def json(self, name, value):
        if name.startswith("measurements/"):
            self.measurement_count += 1
        else:
            self.documents[name] = deepcopy(value)
        return name


class MarkerModel:
    def __init__(self, role):
        self.role, self.frozen, self.evaluating = role, False, False

    def parameters(self):
        return iter((SimpleNamespace(device="marker-cpu"),))

    def requires_grad_(self, value):
        if value is not False:
            raise AssertionError("Evaluation cannot enable gradients")
        self.frozen = True; return self

    def eval(self):
        self.evaluating = True; return self

    def train(self, *args):
        raise AssertionError("Training route forbidden")


class MarkerSampler:
    def __init__(self, model, driver):
        self.model, self.driver, self.counter = model, driver, 0

    def capture(self): return self.counter.to_bytes(8, "little")
    def restore(self, value): self.counter = int.from_bytes(value, "little")
    def seed(self, seed): self.counter = seed

    def sample(self, given, actions):
        self.driver.samples += 1
        if self.driver.fail_forecast and self.driver.samples == 1:
            raise MemoryError("marker forecast gate")
        self.counter += 1
        self.driver.action_windows.append(tuple(tuple(row) for row in actions))
        return dict(marker=self.counter)


class Driver:
    def __init__(self, directory, *, fail_forecast=False):
        self.writer = MemoryWriter(directory)
        self.fail_forecast = fail_forecast
        self.physical = cases(); self.samples = 0; self.action_windows = []
        self.loads, self.closed, self.refs, self.callbacks = [], 0, [], []
        self.dataset = SimpleNamespace(role="independent_validation",
            shape_indices=lambda shape: tuple(range(8)) if shape == "T" else tuple(range(8, 16)))

    def assets(self):
        return dict(sources=deepcopy(engine.SOURCES), source_logical_steps=500,
                    new_account_updates=0, marker_verified=True)

    def prepared(self):
        return dict(ready=True, config=engine.configuration_template(), assets=self.assets(),
                    data=dict(path="synthetic-metadata-only"), benchmark={})

    def binding(self):
        return dict(candidate_id=engine.CANDIDATE, authorization_id=engine.AUTHORIZATION,
                    config_sha256="a"*64, source_logical_steps=500, deadline_unix=3700., deadline_monotonic=3610.)

    def state_loader(self, role, assets):
        if not assets["marker_verified"]:
            raise AssertionError("Unverified marker weights")
        return dict(weight_sha256=assets["sources"][role]["sha256"])

    def factory(self, role, *, state_dict):
        if any(reference() is not None for reference in self.refs):
            raise AssertionError("More than one model was live")
        if state_dict != dict(weight_sha256=engine.SOURCES[role]["sha256"]):
            raise AssertionError("Crossed weight identity")
        self.loads.append((role, state_dict["weight_sha256"]))
        model = MarkerModel(role); self.refs.append(weakref.ref(model)); return model

    def close(self):
        if not self.fail_forecast and any(reference() is not None for reference in self.refs):
            raise AssertionError("Engine kept a sampler/model reference during release")
        self.closed += 1

    def validation_loss(self, model, inputs, *, noise_plan):
        if not model.frozen or not model.evaluating or inputs["dataset_role"] != "independent_validation":
            raise AssertionError("Loss used training mode or training data")
        return (.1 if model.role == "frozen" else .08), dict(
            noise_generator_after_sha256=hashlib.sha256(str(noise_plan["seed"]).encode()).hexdigest())

    def progress(self, phase, step, state):
        if step != 0 or state["completed_updates"] != 0 or state["spent_train_examples"] != 0:
            raise AssertionError("Source500 leaked into new account counters")
        if state["active_models"] > 1:
            raise AssertionError("Multiple active models")
        self.callbacks.append((phase, step, state["active_models"]))

    def rollout(self, case, seed, *, role, sampler, rng, budget, matched, sink, measure, progress):
        original = rng.capture(); key = (case["anchor"], case["candidate"], seed)
        pair = matched.setdefault(key, dict(entries={}, exits={}))
        rows = []
        try:
            rng.seed(seed)
            for mode, horizons in engine.MODES:
                for horizon in horizons:
                    if role == "frozen" and mode == "AR": pair["entries"][horizon] = rng.capture()
                    else: rng.restore(pair["entries"][horizon])
                    entry = rng.capture()
                    actions = (case["past_actions"]+case["future_actions"][:horizon])[-4:]
                    normalized = [[v/511*2-1 for v in row] for row in actions]
                    operation = budget.reserve_forecast(metadata=dict(role=role, case=list(key), mode=mode, horizon=horizon))
                    raw = sampler.sample(dict(mode=mode, horizon=horizon), normalized)
                    exit_state = rng.capture()
                    if role == "frozen" and mode == "AR": pair["exits"][horizon] = exit_state
                    if entry != pair["entries"][horizon] or exit_state != pair["exits"][horizon]:
                        raise AssertionError("Inference RNG was not paired")
                    truth = case["truth"][horizon-1]
                    valid = not (case["anchor"] == "R00" and case["candidate"] == "hold")
                    coverage = truth["coverage"]
                    measurement = dict(state=dict(valid=valid, pose=truth["pose"] if valid else None),
                        shape=dict(valid=valid), pusher=dict(valid=True, position=truth["pusher"]),
                        coverage=dict(valid=valid, estimate=coverage if valid else None,
                            interval=[max(0., coverage-.03), min(1., coverage+.03)] if valid else None,
                            uncertainty_kind="development_absolute_error_bound"),
                        termination=dict(status="non_success" if valid else "unknown"))
                    name = f"predictions/{role}-{case['anchor']}-{case['candidate']}-{seed}-{mode}-H{horizon}.npy"
                    row = dict(role=role, status="completed", stratum=case["role"], shape=case["shape"],
                        anchor=case["anchor"], candidate=case["candidate"], seed=seed, mode=mode, horizon=horizon,
                        truth=truth, latest=case["latest"], contact_by_h4=case["contact_by_h4"], measurement=measurement,
                        rng_entry=hashlib.sha256(entry).hexdigest(), rng_exit=hashlib.sha256(exit_state).hexdigest(),
                        action_sha256=hashlib.sha256(json.dumps(normalized).encode()).hexdigest(),
                        raw_min=-.2, raw_max=1.2, raw_path=sink.raw(name, raw))
                    rows.append(row); sink.json(name.replace("predictions/", "measurements/").replace(".npy", ".json"), row)
                    progress("evaluation", budget.snapshot()["completed_updates"])
                    budget.complete_forecast(operation)
            return rows
        finally:
            rng.restore(original)

    def run(self, **overrides):
        arguments = dict(factory=self.factory, close_model=self.close, writer=self.writer,
            make_sampler=lambda model: MarkerSampler(model, self), progress=self.progress,
            case_loader=lambda prepared: self.physical, dataset_factory=lambda data: self.dataset,
            state_loader=self.state_loader, rollout=self.rollout, clock=lambda: 100., monotonic=lambda: 10.)
        arguments.update(overrides)
        with patch.object(asset_api, "check_prepared_assets", side_effect=lambda value: None), \
             patch.object(runtime.os, "fsync", return_value=None), \
             patch.object(adaptation, "batch", side_effect=lambda dataset, indices, device:
                dict(dataset_role=dataset.role, indices=indices)), \
             patch.object(objective, "make_generator", side_effect=lambda seed, device: dict(seed=seed)), \
             patch.object(objective, "make_noise_plan", side_effect=lambda inputs, generator: generator), \
             patch.object(objective, "validation_loss", self.validation_loss):
            return engine.run_evaluation(self.prepared(), self.binding(), **arguments)


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="marker-", dir=EVIDENCE)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(EVIDENCE.resolve()))
        self.addCleanup(self.temporary.cleanup)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        original_import = builtins.__import__
        def no_runtime(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "scipy", "pygame", "pymunk", "diffusion_policy", "transformers"}:
                raise AssertionError("Real tensor/model/physics import forbidden: "+name)
            return original_import(name, *args, **kwargs)
        self.stack.enter_context(patch("builtins.__import__", side_effect=no_runtime))
        self.sentinels = []
        for module, name in ((objective, "make_optimizer"), (objective, "select_trainable_parameters"),
                (objective, "supervised_denoising_loss"), (objective, "make_real_denoiser"),
                (adaptation, "train_candidate"), (runtime, "CheckpointStore"),
                (runtime, "_restore_rng"), (runtime, "native_resource_preflight")):
            self.sentinels.append(self.stack.enter_context(patch.object(module, name,
                side_effect=AssertionError("Training/restore/native route forbidden: "+name))))

    def budget(self, name):
        return engine.EvaluationBudget.create(self.root/name,
            dict(candidate_id=engine.CANDIDATE, authorization_id=engine.AUTHORIZATION),
            case_keys=[(case["anchor"], case["candidate"]) for case in cases()],
            deadline_seconds=3600, clock=lambda: 100., monotonic=lambda: 10.)

    def test_exact_eval_configuration_has_zero_training_and_unchanged_scientific_fields(self):
        cfg = engine.configuration_template()
        self.assertEqual(engine.validate_config(cfg), cfg)
        self.assertTrue(all(key not in cfg for key in ("train", "optimizer", "objective", "trainable_prefixes")))
        self.assertEqual((cfg["limits"]["max_updates"], cfg["limits"]["max_train_examples"], cfg["limits"]["checkpoint_bytes"]), (0, 0, 0))
        self.assertEqual((cfg["limits"]["model_loads"], cfg["limits"]["forecasts"], cfg["limits"]["euler_steps"]), (2, 3996, 11988))
        for field, value in (("max_updates", 1), ("model_loads", 3), ("forecasts", 3995)):
            mutation = deepcopy(cfg); mutation["limits"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): engine.validate_config(mutation)
        mutation = deepcopy(cfg); mutation["sources"]["adapted"]["logical_update"] = 499
        with self.assertRaises(ValueError): engine.validate_config(mutation)

    def test_update_optimizer_selection_and_restore_routes_are_refused_without_spend(self):
        operations = (("reserve_update", (1,)), ("complete_update", (1,)), ("reserve_optimizer", ()),
            ("select_checkpoint", ("f"*64,)), ("restore_rng", ()), ("restore", ()),
            ("reserve_model_load", ("train_init",)))
        for ordinal, (name, arguments) in enumerate(operations):
            budget = self.budget("deny-"+str(ordinal))
            with self.subTest(operation=name), self.assertRaises(runtime.AdaptationRuntimeError):
                getattr(budget, name)(*arguments)
            state = budget.snapshot()
            self.assertEqual((state["spent_reserved_updates"], state["completed_updates"], state["spent_train_examples"]), (0, 0, 0))
            self.assertEqual(state["selection_count"], 0)
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError, "cannot resume"):
            engine.EvaluationBudget.resume(self.root/"never-open-old-ledger", {})

    def test_model_order_and_release_are_required_before_adapted_loading(self):
        first = self.budget("wrong-first")
        with self.assertRaises(runtime.AdaptationRuntimeError): first.reserve_model_load("adapted")
        concurrent = self.budget("concurrent")
        operation = concurrent.reserve_model_load("frozen"); concurrent.complete_model_load(operation)
        with self.assertRaises(runtime.AdaptationRuntimeError): concurrent.reserve_model_load("adapted")
        self.assertEqual(concurrent.snapshot()["spent_reserved_model_loads"], 1)
        partial = self.budget("partial-release")
        operation = partial.reserve_model_load("frozen"); partial.complete_model_load(operation)
        with self.assertRaisesRegex(runtime.AdaptationRuntimeError, "incomplete role"):
            partial.release_model("frozen")

    def test_complete_marker_flow_has_exact_full_counts_paired_noise_and_invalid_denominators(self):
        driver = Driver(self.root/"full")
        report = driver.run()
        state = report["budget"]
        self.assertEqual((state["spent_reserved_updates"], state["completed_updates"], state["spent_train_examples"]), (0, 0, 0))
        self.assertEqual((state["completed_eval_batches"], state["spent_eval_examples"], state["completed_forecasts"], state["completed_euler_steps"]), (32, 32, 3996, 11988))
        self.assertEqual(state["model_load_roles"], ["frozen", "adapted"])
        self.assertEqual(state["released_model_roles"], ["frozen", "adapted"])
        self.assertEqual((state["active_models"], state["selection_count"], driver.closed), (0, 0, 2))
        self.assertIsNone(state["checkpoint"])
        self.assertEqual(driver.loads, [(role, engine.SOURCES[role]["sha256"]) for role in ("frozen", "adapted")])
        self.assertEqual(len(driver.writer.raw_names), 3996)
        self.assertEqual(driver.writer.measurement_count, 3996)
        self.assertEqual(driver.samples, 3996)
        self.assertEqual(len(driver.callbacks), engine.PROGRESS_MAX)
        self.assertEqual(report["progress_publications"], 4035)
        self.assertEqual(report["source_logical_steps"], 500)
        self.assertEqual(report["old_forecasts_reused"], 0)
        self.assertEqual(report["loss_rows"]["frozen"][0]["index"], 0)
        self.assertEqual([row["noise"] for row in report["loss_rows"]["frozen"]],
                         [row["noise"] for row in report["loss_rows"]["adapted"]])
        self.assertEqual(driver.action_windows[0], tuple(tuple(value/511*2-1 for value in row)
            for row in [[1., 2.], [3., 4.], [5., 6.], [7., 8.]]))
        screens = driver.writer.documents["screens.json"]
        self.assertEqual(sum(row["total"] for row in screens["cells"]), 3996)
        self.assertEqual(sum(row["invalid_count"] for row in screens["cells"]), 36)
        self.assertEqual(screens["paired_retention"]["denominator_source"], "declared_cases")
        independent = [row for row in screens["dynamics"] if row["stratum"] == "independent"]
        self.assertTrue(independent)
        self.assertTrue(all(row["contact_physical_branches"] == 4 and row["rotation_physical_branches"] == 2
            and row["contact_better_than_persistence"] is None and row["rotation_screen"] is None for row in independent))
        journal = (driver.writer.directory/"ledger/highwater.jsonl").read_bytes().splitlines()
        self.assertEqual(len(journal), 8064)
        self.assertLessEqual((driver.writer.directory/"ledger/highwater.jsonl").stat().st_size
                            +2*engine.METADATA_SLOT, engine.JOURNAL_BUDGET)
        self.assertEqual(json.loads(journal[-1])["kind"], "evaluation_finished")
        self.assertTrue(all(not marker.called for marker in self.sentinels))

    def test_forecast_failure_retains_reservation_and_never_loads_adapted_or_resumes(self):
        driver = Driver(self.root/"failure", fail_forecast=True)
        with self.assertRaisesRegex(MemoryError, "marker forecast gate"):
            driver.run()
        state = json.loads((driver.writer.directory/"ledger/first-failure.json").read_bytes())["state"]
        self.assertEqual((state["spent_reserved_forecasts"], state["completed_forecasts"], state["spent_reserved_euler_steps"]), (1, 0, 3))
        self.assertEqual(state["completed_eval_batches"], 16)
        self.assertEqual(state["completed_updates"], 0)
        self.assertEqual(len(driver.loads), 1)
        self.assertNotIn("completion.json", driver.writer.documents)
        self.assertNotIn("screens.json", driver.writer.documents)

    def test_wrong_provenance_is_refused_before_model_state_or_case_loader(self):
        driver = Driver(self.root/"bad-source")
        prepared = driver.prepared(); prepared["assets"]["source_logical_steps"] = 499
        state_loader, case_loader = Mock(), Mock()
        with patch.object(asset_api, "check_prepared_assets", return_value=None):
            with self.assertRaisesRegex(ValueError, "provenance"):
                engine.run_evaluation(prepared, driver.binding(), factory=driver.factory, close_model=driver.close,
                    writer=driver.writer, state_loader=state_loader, case_loader=case_loader)
        state_loader.assert_not_called(); case_loader.assert_not_called()
        self.assertFalse((driver.writer.directory/"ledger").exists())

    def test_missing_physical_branch_is_not_padded_or_reused(self):
        driver = Driver(self.root/"missing"); driver.physical.pop()
        with self.assertRaisesRegex(ValueError, "63 regression"):
            driver.run()
        self.assertEqual(driver.loads, [])
        self.assertEqual(driver.samples, 0)
        self.assertFalse((driver.writer.directory/"ledger").exists())

    def test_account_deadline_is_limited_by_original_monotonic_remainder(self):
        driver = Driver(self.root/"deadline")
        binding = driver.binding(); binding["deadline_monotonic"] = 20.
        stop = Mock(side_effect=RuntimeError("marker stop before weight load"))
        with patch.object(driver, "binding", return_value=binding):
            with self.assertRaisesRegex(RuntimeError, "marker stop before weight load"):
                driver.run(memory_check=stop)
        attempt = json.loads((driver.writer.directory/"ledger/attempt.json").read_bytes())
        self.assertEqual(attempt["limits"]["deadline_seconds"], 10.)
        self.assertEqual(attempt["deadline_monotonic"], 20.)
        self.assertEqual(attempt["deadline_unix"], 110.)
        self.assertLessEqual(attempt["deadline_monotonic"], binding["deadline_monotonic"])
        self.assertEqual(driver.loads, [])

    def test_journal_admission_precedes_spend_and_failure_summary_stays_in_reserved_slot(self):
        budget = self.budget("metadata-cap")
        journal = budget.directory/"highwater.jsonl"
        original = journal.read_bytes()
        budget._journal_bytes = engine.JOURNAL_BUDGET-2*engine.METADATA_SLOT
        stat = Path.stat
        def no_old_journal_stat(path, *args, **kwargs):
            if path == journal:
                raise AssertionError("Cap refusal must precede historical journal access")
            return stat(path, *args, **kwargs)
        with patch.object(Path, "stat", no_old_journal_stat):
            with self.assertRaisesRegex(runtime.AdaptationRuntimeError, "journal metadata admission"):
                budget.reserve_model_load("frozen")
        self.assertEqual(journal.read_bytes(), original)
        state = budget.snapshot()
        self.assertEqual((state["spent_reserved_model_loads"], state["active_models"], state["completed_updates"]), (0, 0, 0))
        self.assertEqual(state["status"], "stopped")
        length = self.budget("bounded-reason")
        length.fail("\U0010ffff"*5000)
        failure = length.directory/"first-failure.json"
        self.assertLessEqual(failure.stat().st_size, engine.METADATA_SLOT)
        self.assertEqual(len(json.loads(failure.read_bytes())["reason"]), 2000)


if __name__ == "__main__":
    unittest.main()
