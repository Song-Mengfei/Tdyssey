"""Actual production loop with ordinary markers, FakeTorch and no tensor runtime."""
import ast
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from tdyssey import gpc_adaptation as production
from tdyssey import gpc_supervised_adaptation as objective


BASELINE = production.ROOT/"artifacts/gpc-gradient-lifetime-review-20261026/baseline-gpc-adaptation-3974e84.py"
BASELINE_SHA256 = "62ff0149cc94a042619dafcc3e0cd7f04d6c982df44f209131a33f9652db09f4"
BASELINE_FUNCTION_AST_SHA256 = "b0cfe9f2b0bb5f548d68844481d8ad0668bd54dd49bc16b0ad9b7d5d7063c9f4"


def function_ast_dump(function):
    # Empty type_params was added to FunctionDef in Python 3.12; it carries no
    # meaning for this nongeneric function and is absent on source-only 3.11.
    return ast.dump(function, include_attributes=False).replace(", type_params=[]", "")


def remove_gradient_deletion(function):
    class RemoveGradientDeletion(ast.NodeTransformer):
        count = 0
        def visit_Delete(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "grads":
                self.count += 1
                return None
            return node

    transform = RemoveGradientDeletion()
    return transform.visit(function), transform.count


def baseline_function():
    """Use sealed source, or reverse only the deletion and verify its old AST."""
    try:
        source = BASELINE.read_bytes()
    except FileNotFoundError:
        path = Path(production.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "train_candidate")
        function, deletions = remove_gradient_deletion(function)
        if deletions != 1:
            raise AssertionError("Fresh-checkout baseline reconstruction requires exactly one del grads")
    else:
        if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
            raise AssertionError("The sealed clean3974e84 production source changed")
        tree = ast.parse(source.decode("utf-8"), filename=str(BASELINE))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "train_candidate")
    if hashlib.sha256(function_ast_dump(function).encode("utf-8")).hexdigest() != BASELINE_FUNCTION_AST_SHA256:
        raise AssertionError("Old train_candidate AST differs from sealed clean3974e84")
    namespace = dict(vars(production))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BASELINE), "exec"), namespace)
    return namespace["train_candidate"], function


class Gradient:
    def __init__(self, value):
        self.value = value


class Parameter:
    def __init__(self, name, value, *, trainable=True):
        self.name, self.value, self.requires_grad = name, value, trainable
        self.device, self.grad = "ordinary-marker", None


class MarkerModel:
    def __init__(self, fixture):
        self.fixture = fixture
        self.items = (Parameter("first", 4.), Parameter("second", -2.),
                      Parameter("frozen", 7., trainable=False))

    def parameters(self):
        return iter(self.items)

    def train(self):
        self.fixture.events.append(("train",))


class MarkerLoss:
    def __init__(self, fixture):
        self.fixture = fixture

    def backward(self):
        fixture = self.fixture
        fixture.events.append(("backward", fixture.round))
        if fixture.failure == "missing_grad":
            return
        for index, parameter in enumerate(fixture.model.items[:2]):
            value = (3., 4.)[index]
            if fixture.failure == "nonfinite_grad" and index == 0:
                value = float("nan")
            parameter.grad = Gradient(value)
        fixture.current_refs = [weakref.ref(parameter.grad) for parameter in fixture.model.items[:2]]
        if fixture.round == 1:
            fixture.first_refs = list(fixture.current_refs)


class MarkerOptimizer:
    def __init__(self, fixture):
        self.fixture = fixture

    def zero_grad(self, *, set_to_none):
        fixture = self.fixture
        if set_to_none is not True:
            raise AssertionError("Production must retain its set_to_none=True setting")
        for parameter in fixture.model.items:
            parameter.grad = None
        fixture.events.append(("zero_grad",))
        if fixture.round == 1:
            fixture.alive_after_second_zero = [ref() is not None for ref in fixture.first_refs]

    def step(self):
        fixture = self.fixture
        fixture.events.append(("step", fixture.round))
        fixture.step_calls += 1
        for parameter, expected, reference in zip(fixture.model.items[:2], (.6, .8), fixture.current_refs):
            if parameter.grad is not reference():
                raise AssertionError("Optimizer must receive the current backward's p.grad object")
            if parameter.grad is None or not math.isclose(parameter.grad.value, expected, abs_tol=1e-12):
                raise AssertionError("Clipped p.grad was lost or changed before optimizer.step")
            parameter.value -= .1*parameter.grad.value
        if fixture.failure == "nonfinite_parameter":
            fixture.model.items[0].value = float("inf")


class MarkerBudget:
    def __init__(self, fixture):
        self.fixture = fixture
        self.attempt = dict(started_monotonic=0., started_unix=0., binding={})
        self.state = dict(completed_updates=0, spent_train_examples=0, spent_reserved_updates=0,
                          pending=None, checkpoint=None, selection_count=0, status="active")

    def snapshot(self):
        return deepcopy(self.state)

    def clock(self):
        return 0.

    def check(self):
        self.fixture.events.append(("budget.check",))

    def reserve_update(self, size):
        if self.state["pending"] is not None:
            raise AssertionError("Concurrent update reservation")
        self.state["spent_reserved_updates"] += 1
        self.state["spent_train_examples"] += size
        operation = self.state["spent_reserved_updates"]
        self.state["pending"] = operation
        self.fixture.events.append(("reserve", operation, size))
        return operation

    def complete_update(self, operation):
        if self.state["pending"] != operation:
            raise AssertionError("Completion without matching reservation")
        self.state["completed_updates"] += 1
        self.state["pending"] = None
        self.fixture.events.append(("complete", operation))

    def select_checkpoint(self, sha256):
        if self.state["completed_updates"] != 2 or sha256 != "f"*64:
            raise AssertionError("Only the final completed fake checkpoint is selectable")
        self.state["selection_count"] += 1
        self.fixture.events.append(("select", sha256))

    def fail(self, reason):
        self.state.update(status="stopped", first_reason=reason)
        self.fixture.events.append(("fail", reason))


class MarkerStore:
    def __init__(self, fixture):
        self.fixture = fixture
        self.saves = []

    def save(self, model, optimizer, *, extra_state):
        self.saves.append(dict(parameters=[parameter.value for parameter in model.items],
                               extra_state=deepcopy(extra_state)))
        self.fixture.budget.state["checkpoint"] = dict(sha256="f"*64)
        self.fixture.events.append(("save",))


class Fixture:
    def __init__(self, failure=None):
        self.failure = failure
        self.events, self.metrics, self.first_refs, self.current_refs = [], [], [], []
        self.round, self.step_calls = 0, 0
        self.alive_after_second_zero = self.alive_at_second_forward = None
        self.model = MarkerModel(self)
        self.optimizer = MarkerOptimizer(self)
        self.budget = MarkerBudget(self)
        self.store = MarkerStore(self)
        self.cfg = dict(train=dict(batch_size=1, max_examples=2, max_updates=2,
            phase_seconds=900, checkpoint_every=2, selected_update=2,
            grad_clip_norm=1., order="alternate_T_L_no_shuffle"))

    def batch(self, dataset, indices, device):
        if device != "ordinary-marker":
            raise AssertionError("Unexpected real tensor device")
        self.events.append(("batch", tuple(indices)))
        return dict(index=indices[0])

    def loss(self, model, inputs, *, generator):
        if model is not self.model or generator != "independent-marker-rng":
            raise AssertionError("Unexpected model/generator route")
        self.round += 1
        self.events.append(("forward", self.round, inputs["index"]))
        if self.round == 2:
            self.alive_at_second_forward = [ref() is not None for ref in self.first_refs]
        return MarkerLoss(self), dict(loss_denoising=.25, model_output_calls=1,
                                       inference_sampler_calls=0)

    def isfinite(self, value):
        kind = "gradient" if isinstance(value, Gradient) else "parameter"
        self.events.append(("isfinite", kind, math.isfinite(value.value)))
        # Never retain the input object in this result or in the event log.
        finite = math.isfinite(value.value)
        return SimpleNamespace(all=lambda: finite)

    def clip(self, parameters, max_norm, *, error_if_nonfinite):
        if parameters != list(self.model.items[:2]) or max_norm != 1. or error_if_nonfinite is not True:
            raise AssertionError("Production clip parameter selection/settings changed")
        if any(parameter.grad is None for parameter in parameters):
            raise AssertionError("del grads must not clear p.grad before clipping")
        if any(parameter.grad is not reference() for parameter, reference in zip(parameters, self.current_refs)):
            raise AssertionError("Clip must receive the current backward's p.grad object")
        norm = math.sqrt(sum(parameter.grad.value**2 for parameter in parameters))
        self.events.append(("clip", norm))
        for parameter in parameters:
            parameter.grad.value *= min(1., max_norm/norm)
        return norm

    def progress(self, phase, step):
        self.events.append(("progress", phase, step))
        if self.failure == "post_step_gate" and step == 2:
            raise MemoryError("fake post-step ceiling")

    def record_metrics(self, step, values):
        self.events.append(("metrics", step))
        self.metrics.append((step, deepcopy(values)))

    def run(self, function):
        torch = ModuleType("torch")
        torch.isfinite = self.isfinite
        torch.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=self.clip))
        dataset = SimpleNamespace(shape_indices=lambda shape: (10,) if shape == "T" else (11,))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"torch": torch}))
            stack.enter_context(patch.dict(function.__globals__, batch=self.batch))
            stack.enter_context(patch.object(objective, "supervised_denoising_loss", self.loss))
            return function(self.model, self.optimizer, dataset, cfg=self.cfg, budget=self.budget,
                store=self.store, generator="independent-marker-rng", progress=self.progress,
                metrics=self.record_metrics, monotonic=lambda: 0.)


class GradientLifetimeTests(unittest.TestCase):
    def test_actual_old_and_repaired_functions_have_only_the_gradient_reference_deletion_delta(self):
        _, old = baseline_function()
        current_tree = ast.parse(Path(production.__file__).read_text(encoding="utf-8"))
        current = next(node for node in current_tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "train_candidate")
        old_dump = function_ast_dump(old)
        without_deletion, count = remove_gradient_deletion(current)
        self.assertEqual(count, 1)
        self.assertEqual(function_ast_dump(without_deletion), old_dump)

    def test_two_actual_loop_iterations_release_previous_gradient_without_numeric_or_order_change(self):
        old_function, _ = baseline_function()
        old, repaired = Fixture(), Fixture()
        old_result = old.run(old_function)
        repaired_result = repaired.run(production.train_candidate)
        self.assertEqual(old.alive_after_second_zero, [True, True])
        self.assertEqual(old.alive_at_second_forward, [True, True])
        self.assertEqual(repaired.alive_after_second_zero, [False, False])
        self.assertEqual(repaired.alive_at_second_forward, [False, False])
        self.assertEqual(old.events, repaired.events)
        self.assertEqual(old.metrics, repaired.metrics)
        self.assertEqual(old.store.saves, repaired.store.saves)
        self.assertEqual(old_result, repaired_result)
        self.assertEqual(repaired.step_calls, 2)
        self.assertEqual(repaired.budget.state["spent_reserved_updates"], 2)
        self.assertEqual(repaired.budget.state["spent_train_examples"], 2)
        self.assertEqual(repaired.budget.state["completed_updates"], 2)
        self.assertEqual(repaired.budget.state["selection_count"], 1)
        self.assertEqual([values["indices"] for _, values in repaired.metrics], [[10], [11]])
        self.assertEqual([values["gradient_norm"] for _, values in repaired.metrics], [5., 5.])
        self.assertAlmostEqual(repaired.model.items[0].value, 3.88)
        self.assertAlmostEqual(repaired.model.items[1].value, -2.16)
        self.assertEqual(repaired.model.items[2].value, 7.)
        self.assertIsNone(repaired.model.items[2].grad)
        type(self).comparison_evidence = dict(
            old=dict(alive_after_second_zero=list(old.alive_after_second_zero),
                alive_at_second_forward=list(old.alive_at_second_forward),
                final_parameters=[parameter.value for parameter in old.model.items],
                spent_updates=old.budget.state["spent_reserved_updates"],
                spent_examples=old.budget.state["spent_train_examples"],
                completed_updates=old.budget.state["completed_updates"]),
            repaired=dict(alive_after_second_zero=list(repaired.alive_after_second_zero),
                alive_at_second_forward=list(repaired.alive_at_second_forward),
                final_parameters=[parameter.value for parameter in repaired.model.items],
                spent_updates=repaired.budget.state["spent_reserved_updates"],
                spent_examples=repaired.budget.state["spent_train_examples"],
                completed_updates=repaired.budget.state["completed_updates"]),
            metrics_equal=old.metrics == repaired.metrics, events_equal=old.events == repaired.events)

    def assert_failed_reservation(self, fixture, *, spent=1, completed=0):
        self.assertEqual(fixture.budget.state["status"], "stopped")
        self.assertEqual(fixture.budget.state["spent_reserved_updates"], spent)
        self.assertEqual(fixture.budget.state["spent_train_examples"], spent)
        self.assertEqual(fixture.budget.state["completed_updates"], completed)
        self.assertEqual(fixture.budget.state["pending"], spent)
        self.assertEqual(fixture.budget.state["selection_count"], 0)
        self.assertIsNone(fixture.budget.state["checkpoint"])
        self.assertEqual(fixture.store.saves, [])

    def test_missing_gradient_still_fails_after_spending_the_reservation(self):
        fixture = Fixture("missing_grad")
        with self.assertRaisesRegex(FloatingPointError, "Missing or nonfinite intended parameter gradients"):
            fixture.run(production.train_candidate)
        self.assert_failed_reservation(fixture)
        self.assertEqual(fixture.step_calls, 0)
        self.assertFalse(any(event[0] == "clip" for event in fixture.events))

    def test_nonfinite_gradient_still_fails_before_clip_or_optimizer(self):
        fixture = Fixture("nonfinite_grad")
        with self.assertRaisesRegex(FloatingPointError, "Missing or nonfinite intended parameter gradients"):
            fixture.run(production.train_candidate)
        self.assert_failed_reservation(fixture)
        self.assertEqual(fixture.step_calls, 0)
        self.assertFalse(any(event[0] == "clip" for event in fixture.events))

    def test_nonfinite_updated_parameter_still_fails_before_durable_completion(self):
        fixture = Fixture("nonfinite_parameter")
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite updated parameter"):
            fixture.run(production.train_candidate)
        self.assert_failed_reservation(fixture)
        self.assertEqual(fixture.step_calls, 1)
        self.assertFalse(any(event[0] == "progress" for event in fixture.events))
        self.assertEqual(fixture.metrics, [])

    def test_second_post_step_gate_keeps_spent_two_completed_one_and_no_checkpoint(self):
        fixture = Fixture("post_step_gate")
        with self.assertRaisesRegex(MemoryError, "fake post-step ceiling"):
            fixture.run(production.train_candidate)
        self.assert_failed_reservation(fixture, spent=2, completed=1)
        self.assertEqual(fixture.step_calls, 2)
        self.assertEqual(len(fixture.metrics), 1)
        self.assertIn("training_failed: MemoryError", fixture.budget.state["first_reason"])


if __name__ == "__main__":
    unittest.main()
