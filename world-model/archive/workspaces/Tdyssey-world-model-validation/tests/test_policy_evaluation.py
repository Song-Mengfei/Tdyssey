"""Fake evaluation only: no real simulator, checkpoint, optimizer or B ledger."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

if torch is not None:
    from tdyssey.envs.official import T_REFERENCE_COG, official_shape
    from tdyssey.geometry import Pose
    from tdyssey.policy_evaluation import (
        EvaluationCase, MATCHED_TL_PROTOCOL, ORIGINAL_T_PROTOCOL,
        bind_evaluation_identity, evaluate_cases, paired_comparison,
    )
    from tdyssey.state import BodyState, Snapshot


BASE = "b" * 64
DELTA = "d" * 64


class FakeBudget:
    def __init__(self, *, controls=1000, plans=1000, resets=1000, expire=False):
        self.limits = {"control": controls, "plan": plans, "reset": resets}
        self.events, self.expire = [], expire

    def check_time(self):
        if self.expire:
            raise RuntimeError("Development wall-time exhausted")

    def remaining(self, kind, phase):
        assert phase == "development"
        return self.limits[kind] - sum(event["kind"] == kind for event in self.events)

    def reserve(self, kind, *, phase, shape, amount):
        assert phase == "development" and shape in ("T", "L") and amount == 1
        if self.remaining(kind, phase) < amount:
            raise RuntimeError(f"Development {kind} budget exhausted")
        self.events.append({"kind": kind, "shape": shape, "completed": False})
        return len(self.events)

    def mark_completed(self, reservation):
        event = self.events[reservation - 1]
        assert not event["completed"]
        event["completed"] = True


def fake_official():
    class Official:
        def __init__(self):
            self.generator = torch.Generator().manual_seed(87)

        def observation_tensors(self, history):
            assert len(history) == 2
            return {key: torch.stack([torch.as_tensor(obs[key]) for obs in history])[None]
                    for key in ("image", "agent_pos")}

    return Official()


def fake_actor(official):
    class Network(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.noise_scheduler = SimpleNamespace(timesteps=torch.tensor([5, 4]),
                                                   num_inference_steps=2, custom_timesteps=True)
            self.kwargs = {"generator": official.generator}

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network, self.calls = Network(), []
            self.generator = torch.Generator().manual_seed(74)

        def sample(self, observations, *, generator, mode, actor_version):
            assert not torch.is_grad_enabled()
            assert all(not module.training for module in self.modules())
            assert set(observations) == {"image", "agent_pos"}
            assert observations["image"].shape == (1, 2, 3, 96, 96)
            assert observations["agent_pos"].shape == (1, 2, 2)
            self.calls.append({"mode": mode, "seed": generator.initial_seed(), "keys": tuple(observations)})
            # Deliberate global/existing-local consumption exercises restoration.
            random.random()
            np.random.random()
            torch.rand(1)
            torch.rand(1, generator=self.generator)
            torch.rand(1, generator=official.generator)
            self.network.noise_scheduler.timesteps = torch.arange(99, -1, -1)
            self.network.noise_scheduler.num_inference_steps = 100
            self.network.noise_scheduler.custom_timesteps = False
            return SimpleNamespace(actions=100 + torch.randn((1, 8, 2), generator=generator) * self.network.weight)

    return Actor()


class FakeEnvironment:
    sim_hz, control_hz, legacy = 100, 10, True

    def __init__(self, shape, profiles, *, timeout=None, fail_step=None, done_step=None):
        self.project_shape, self.profiles = shape, profiles
        self.timeout, self.fail_step, self.done_step = timeout, fail_step, done_step
        self.closed, self.steps, self.restored = False, 0, []
        self.block = SimpleNamespace(center_of_gravity=T_REFERENCE_COG if shape == "T" else (0, 0))

    def seed(self, seed):
        self.seed_value = seed
        random.random()
        np.random.random()
        torch.rand(1)

    def observation(self):
        return {"image": np.full((3, 96, 96), .2 + self.steps * .001, dtype=np.float32),
                "agent_pos": np.array([100 + self.steps, 100], dtype=np.float32)}

    def reset(self):
        self.steps = 0
        self.profile = self.profiles.get((self.project_shape, self.seed_value), (.2, .95, .951))
        self.coverage = self.profile[0]
        position = (0., 0.) if self.seed_value == 999 else (200., 200.)
        self.state = Snapshot(self.project_shape, official_shape(self.project_shape).geometry_hash,
                              "a" * 64, BodyState(Pose(position, 0)),
                              BodyState(Pose((100., 100.), 0)), Pose((350., 350.), 0), 0)
        if self.seed_value == 998:
            self.state = replace(self.state, control_step=7)
        return self.observation()

    def capture_state(self):
        return self.state

    def restore_state(self, state):
        assert state.control_step == 0
        self.restored.append(state)
        self.state = state
        return self.observation()

    def step(self, action):
        assert not torch.is_grad_enabled()
        assert not self.closed and len(action) == 2
        self.steps += 1
        if self.steps == self.fail_step:
            raise RuntimeError("Synthetic step interruption")
        self.coverage = self.profile[min(self.steps, len(self.profile) - 1)]
        self.state = replace(self.state, control_step=self.steps)
        timeout = self.steps == self.timeout
        done = self.coverage > .95 or self.steps == self.done_step or timeout
        return self.observation(), min(self.coverage / .95, 1), done, {"TimeLimit.truncated": timeout}

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self, profiles=None, **kwargs):
        self.profiles, self.kwargs, self.environments = profiles or {}, kwargs, []

    def __call__(self, shape):
        environment = FakeEnvironment(shape, self.profiles, **self.kwargs)
        self.environments.append(environment)
        return environment


def fake_metrics(env):
    return {"coverage": env.coverage, "iou": env.coverage / (2 - env.coverage)}


@unittest.skipIf(torch is None, "Installed torch/numpy required for fake evaluation")
class PolicyEvaluationChecks(unittest.TestCase):
    def run_method(self, method="base_training", *, cases=None, profiles=None, budget=None,
                   protocol=None, max_controls=3, factory=None, **callbacks):
        protocol = MATCHED_TL_PROTOCOL if protocol is None else protocol
        official = fake_official()
        actor = fake_actor(official)
        identity = {"base_sha256": BASE, "delta_sha256": DELTA if method == "delta_training" else None}
        bind_evaluation_identity(actor, **identity)
        factory = FakeFactory(profiles) if factory is None else factory
        budget = FakeBudget() if budget is None else budget
        cases = (EvaluationCase("T", 10, 20), EvaluationCase("L", 10, 20)) if cases is None else cases
        result = evaluate_cases(official, actor, factory, cases, method=method, protocol=protocol,
                                budget=budget, max_controls=max_controls, metrics_fn=fake_metrics,
                                delta_applied=method == "delta_training", checkpoint_identity=identity, **callbacks)
        return result, actor, factory, budget

    def test_case_callbacks_preserve_abort_and_unstarted_order_with_detached_data(self):
        starts, finishes = [], []
        def finished(case):
            finishes.append((case["shape"], case["status"], case["success"]))
            case["shape"] = "modified callback copy"
        report, _, factory, _ = self.run_method(budget=FakeBudget(controls=1),
            on_case_start=starts.append, on_case=finished)
        self.assertEqual([(r["shape"], r["status"]) for r in starts], [("T", "started")])
        self.assertEqual(finishes, [("T", "aborted", None), ("L", "not_started", None)])
        self.assertEqual([r["shape"] for r in report["cases"]], ["T", "L"])
        self.assertIsNone(report["success_fraction"])
        self.assertTrue(all(env.closed for env in factory.environments))

    def test_real_evaluation_only_authority_integrates_with_fake_cases_and_case_journal(self):
        from scripts.evaluate_c_policy_pair import append_case
        from tdyssey.policy_eval_budget import EvaluationBudget
        from tests.test_policy_eval_budget import BINDING
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            with EvaluationBudget(root / "registry", "fake-eval", BINDING, root / "run", create=True) as budget:
                budget.claim_evaluation_group("base_training", method="base_training",
                    protocol=MATCHED_TL_PROTOCOL, case_digest=BINDING["case_digest"])
                report, _, _, _ = self.run_method(budget=budget,
                    on_case=lambda case: append_case(root / "run/cases.jsonl", case, budget))
                self.assertEqual(report["status"], "completed")
                budget.complete_evaluation_group("base_training", "e" * 64)
                snapshot = budget.snapshot()
                self.assertEqual(snapshot["reserved"], snapshot["completed"])
                self.assertEqual(snapshot["completed"]["development"]["reset"], 4)
                self.assertEqual(snapshot["completed"]["development"]["control"], report["counts"]["control_steps"])
                self.assertEqual(set(snapshot["completed"]), {"development"})
                self.assertEqual(len((root / "run/cases.jsonl").read_text().splitlines()), 2)

    def test_three_methods_share_fixed_cases_inputs_and_honest_counts(self):
        profiles = {("L", 10): (.2, .4, .3, .35)}
        for method in ("base_original", "base_training", "delta_training"):
            with self.subTest(method=method):
                result, actor, factory, budget = self.run_method(method, profiles=profiles)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["success_fraction"], .5)
                self.assertEqual([case["success_step"] for case in result["cases"]], [2, None])
                self.assertEqual(result["cases"][1]["final_metrics"]["coverage"], .35)
                self.assertEqual(result["cases"][1]["peak_metrics"]["coverage"], .4)
                self.assertEqual(result["counts"]["control_steps"], 5)
                self.assertEqual(result["counts"]["control_physics_substeps"], 50)
                self.assertEqual(result["counts"]["reset_physics_substeps"], 4)
                self.assertEqual(result["counts"]["planning_calls"], 2)
                self.assertEqual(result["counts"]["denoising_steps"], 200)
                self.assertEqual({call["mode"] for call in actor.calls},
                                 {"baseline" if method == "base_original" else "training"})
                self.assertEqual([call["seed"] for call in actor.calls], [20, 20])
                self.assertTrue(all(env.closed for env in factory.environments))
                self.assertEqual(len({id(env) for env in factory.environments}), 4)
                self.assertEqual(sum(len(env.restored) for env in factory.environments), 2)
                self.assertTrue(all(event["completed"] for event in budget.events))
                self.assertEqual(result["state_fingerprint_before"], result["state_fingerprint_after"])
                json.dumps(result, allow_nan=False)

    def test_original_t_has_own_protocol_and_exact_300_control_short_final_prefix(self):
        result, actor, factory, _ = self.run_method("base_original", cases=(EvaluationCase("T", 10, 20),),
                                                   profiles={("T", 10): (.2,)}, protocol=ORIGINAL_T_PROTOCOL,
                                                   max_controls=300)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["success_fraction"], 0)
        self.assertEqual(result["counts"]["control_steps"], 300)
        self.assertEqual(result["counts"]["planning_calls"], 38)
        self.assertEqual(result["counts"]["reset_physics_substeps"], 1)
        self.assertEqual(len(actor.calls), 38)
        self.assertEqual(len(factory.environments), 1)
        self.assertEqual(factory.environments[0].restored, [])
        self.assertFalse(result["matched_initial_state_only"])
        self.assertEqual(result["cases"][0]["stop_reason"], "control_cap")

    def test_independent_policy_seeds_and_per_scene_denominators(self):
        cases = (EvaluationCase("T", 10, 20), EvaluationCase("T", 10, 21), EvaluationCase("T", 11, 20))
        result, actor, _, _ = self.run_method(cases=cases)
        self.assertEqual([call["seed"] for call in actor.calls], [20, 21, 20])
        actions = [case["steps"][0]["action"] for case in result["cases"]]
        self.assertEqual(actions[0], actions[2])
        self.assertNotEqual(actions[0], actions[1])
        self.assertEqual(result["per_scene"][0]["requested_case_count"], 2)
        self.assertEqual(result["per_scene"][0]["success_fraction"], 1)
        interrupted, _, _, _ = self.run_method(cases=cases, budget=FakeBudget(controls=3))
        self.assertIsNone(interrupted["per_scene"][0]["success_fraction"])
        self.assertEqual(interrupted["per_scene"][0]["completed_case_success_fraction"], 1)

    def test_actual_tiny_sampler_training_next_chain_is_unchanged(self):
        from tests.test_policy_rl_runtime import actor_fixture, observation_fixture, torch as rl_torch
        if rl_torch is None:
            self.skipTest("Installed diffusers required for the tiny sampler")
        torch.set_num_threads(4)
        official, actor = fake_official(), actor_fixture()
        training_generator = torch.Generator().manual_seed(481)
        official.generator = training_generator
        identity = {"base_sha256": BASE, "delta_sha256": None}
        bind_evaluation_identity(actor, **identity)
        training_observation = observation_fixture(1)
        expected_generator = torch.Generator().set_state(training_generator.get_state())
        expected = actor.sample(training_observation, generator=expected_generator)
        actor.train()
        report = evaluate_cases(official, actor, FakeFactory(), (EvaluationCase("T", 10, 20),),
                                method="base_training", protocol=MATCHED_TL_PROTOCOL, budget=FakeBudget(),
                                max_controls=3, metrics_fn=fake_metrics, checkpoint_identity=identity)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(actor.training)
        following = actor.sample(training_observation, generator=training_generator)
        torch.testing.assert_close(following.chain, expected.chain, rtol=0, atol=0)
        torch.testing.assert_close(training_generator.get_state(), expected_generator.get_state(), rtol=0, atol=0)

    def test_original_initial_peak_is_not_silently_counted_as_post_control_success(self):
        report, _, _, _ = self.run_method("base_original", cases=(EvaluationCase("T", 10, 20),),
                                          protocol=ORIGINAL_T_PROTOCOL, max_controls=1,
                                          profiles={("T", 10): (1., .3)})
        self.assertFalse(report["cases"][0]["success"])
        self.assertEqual(report["cases"][0]["peak_metrics"]["coverage"], 1)
        self.assertEqual(report["cases"][0]["counts"]["control_steps"], 1)
        self.assertEqual(report["success_observation"], "after_physical_control")

    def test_rng_modes_scheduler_local_generators_weights_and_training_cursor_are_unchanged(self):
        official = fake_official()
        actor = fake_actor(official)
        actor.train()
        actor.network.eval()
        identity = {"base_sha256": BASE, "delta_sha256": None}
        bind_evaluation_identity(actor, **identity)
        modes = [(module, module.training) for module in actor.modules()]
        scheduler = actor.network.noise_scheduler
        previous_timesteps = scheduler.timesteps.clone()
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        local_states = (official.generator.get_state().clone(), actor.generator.get_state().clone())
        weights = {key: value.clone() for key, value in actor.state_dict().items()}
        training_cursor = {"scene_index": 7, "episode_controls": 11, "history": ["old", "current"]}
        cursor_before = deepcopy(training_cursor)
        factory = FakeFactory()
        evaluate_cases(official, actor, factory, (EvaluationCase("T", 10, 20),), method="base_training",
                       protocol=MATCHED_TL_PROTOCOL, budget=FakeBudget(), max_controls=3,
                       metrics_fn=fake_metrics, checkpoint_identity=identity)
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        for current, previous in zip((official.generator.get_state(), actor.generator.get_state()), local_states):
            torch.testing.assert_close(current, previous, rtol=0, atol=0)
        self.assertTrue(all(module.training == mode for module, mode in modes))
        torch.testing.assert_close(scheduler.timesteps, previous_timesteps, rtol=0, atol=0)
        self.assertEqual((scheduler.num_inference_steps, scheduler.custom_timesteps), (2, True))
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        self.assertEqual(training_cursor, cursor_before)

    def test_budget_abort_reports_partial_and_unstarted_cases_without_invented_fraction(self):
        cases = tuple(EvaluationCase("T", seed, 20) for seed in (10, 11, 12))
        result, _, factory, budget = self.run_method(cases=cases, budget=FakeBudget(controls=3))
        self.assertEqual([case["status"] for case in result["cases"]], ["completed", "aborted", "not_started"])
        self.assertEqual((result["completed_case_count"], result["aborted_case_count"], result["unstarted_case_count"]), (1, 1, 1))
        self.assertIsNone(result["success_fraction"])
        self.assertEqual(result["completed_case_success_fraction"], 1)
        self.assertEqual(result["counts"]["control_steps"], 3)
        self.assertEqual(budget.remaining("control", "development"), 0)
        self.assertIn("budget exhausted", result["cases"][1]["abort_reason"])
        self.assertTrue(all(env.closed for env in factory.environments))
        expired, _, empty_factory, _ = self.run_method(cases=cases, budget=FakeBudget(expire=True))
        self.assertEqual(expired["counts"]["reset_attempts"], 0)
        self.assertEqual(empty_factory.environments, [])
        self.assertIsNone(expired["success_fraction"])

    def test_timeout_and_interrupted_step_are_distinct_from_completed_failure(self):
        timeout, _, _, _ = self.run_method(cases=(EvaluationCase("T", 10, 20),),
                                           factory=FakeFactory({("T", 10): (.2,)}, timeout=1))
        self.assertEqual(timeout["status"], "completed")
        self.assertFalse(timeout["cases"][0]["success"])
        self.assertEqual(timeout["cases"][0]["stop_reason"], "episode_timeout")
        interrupted, actor, factory, _ = self.run_method(cases=(EvaluationCase("T", 10, 20),),
                                                         factory=FakeFactory(fail_step=2))
        self.assertEqual(interrupted["status"], "incomplete")
        self.assertEqual(interrupted["counts"]["control_attempts"], 2)
        self.assertEqual(interrupted["counts"]["control_steps"], 1)
        self.assertEqual(interrupted["cases"][0]["interrupted_operation"], "control")
        self.assertIsNone(interrupted["success_fraction"])
        self.assertTrue(actor.training)
        self.assertTrue(all(env.closed for env in factory.environments))

    def test_fixed_matched_scene_rejection_has_no_retry_or_contact_restore_claim(self):
        for seed, reason in ((999, "Fixed matched scene rejected"), (998, "never a contact snapshot")):
            with self.subTest(seed=seed):
                result, _, factory, _ = self.run_method(cases=(EvaluationCase("L", seed, 20),))
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["counts"]["reset_physics_substeps"], 2)
                self.assertEqual(result["counts"]["control_steps"], 0)
                self.assertIn(reason, result["cases"][0]["abort_reason"])
                self.assertEqual([env.seed_value for env in factory.environments], [seed, seed])
                self.assertTrue(all(not env.restored for env in factory.environments))
                self.assertFalse(result["mid_contact_restore_validated"])

    def test_reusing_an_environment_instance_is_rejected_before_another_reset(self):
        shared = FakeEnvironment("T", {})
        result, _, _, _ = self.run_method("base_original", cases=(EvaluationCase("T", 10, 20),
                                                                   EvaluationCase("T", 11, 20)),
                                           protocol=ORIGINAL_T_PROTOCOL, factory=lambda shape: shared)
        self.assertEqual([case["status"] for case in result["cases"]], ["completed", "aborted"])
        self.assertEqual(result["counts"]["reset_attempts"], 1)
        self.assertIn("reused an instance", result["cases"][1]["abort_reason"])
        self.assertTrue(shared.closed)

    def test_paired_json_reports_align_exact_cases_and_reject_missing_or_incomplete_data(self):
        reports = [self.run_method(method)[0] for method in ("base_original", "base_training", "delta_training")]
        reports = json.loads(json.dumps(reports))
        paired = paired_comparison(reports)
        self.assertEqual(paired["case_count"], 2)
        self.assertEqual(len(paired["comparisons"]), 3)
        for comparison in paired["comparisons"]:
            self.assertEqual(comparison["mean_differences"]["success_difference"], 0)
            self.assertEqual(comparison["mean_differences"]["final_metrics_coverage_difference"], 0)
        for change in ("missing", "reordered", "incomplete", "base", "protocol"):
            changed = deepcopy(reports)
            if change == "missing":
                changed[-1]["cases"].pop()
            elif change == "reordered":
                changed[-1]["cases"].reverse()
            elif change == "incomplete":
                changed[-1]["status"] = "incomplete"
            elif change == "base":
                changed[-1]["checkpoint_identity"]["base_sha256"] = "e" * 64
            else:
                changed[-1]["protocol"] = ORIGINAL_T_PROTOCOL
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    paired_comparison(changed)

    def test_four_checkpoint_groups_keep_delta_stage_labels_and_reject_duplicate_identity(self):
        reports = [self.run_method(method)[0] for method in ("base_original", "base_training", "delta_training")]
        reports[0]["comparison_label"] = "base-original"
        reports[1]["comparison_label"] = "base-training"
        reports[2]["comparison_label"] = "delta-1024"
        next_stage = deepcopy(reports[2])
        next_stage["comparison_label"] = "delta-2048"
        next_stage["checkpoint_identity"]["delta_sha256"] = "e" * 64
        reports.append(next_stage)
        reports = json.loads(json.dumps(reports))
        paired = paired_comparison(reports)
        self.assertEqual(len(paired["comparisons"]), 6)
        self.assertEqual([group["comparison_label"] for group in paired["groups"]],
                         ["base-original", "base-training", "delta-1024", "delta-2048"])
        stages = next(pair for pair in paired["comparisons"]
                      if pair["reference_label"] == "delta-1024" and pair["candidate_label"] == "delta-2048")
        self.assertEqual(stages["reference_method"], "delta_training")
        self.assertEqual(stages["candidate_method"], "delta_training")
        self.assertEqual(stages["reference_checkpoint_identity"]["delta_sha256"], DELTA)
        self.assertEqual(stages["candidate_checkpoint_identity"]["delta_sha256"], "e" * 64)
        for failure in ("label", "identity"):
            changed = deepcopy(reports)
            if failure == "label":
                changed[-1]["comparison_label"] = changed[-2]["comparison_label"]
            else:
                changed[-1]["checkpoint_identity"] = deepcopy(changed[-2]["checkpoint_identity"])
            with self.subTest(failure=failure):
                with self.assertRaisesRegex(ValueError, "labels|duplicated"):
                    paired_comparison(changed)
        # group_id is an explicit compatibility fallback; method remains the
        # fallback for the original three-report API tested above.
        for report in reports:
            report["group_id"] = report.pop("comparison_label")
        self.assertEqual(len(paired_comparison(reports)["comparisons"]), 6)

    def test_method_provenance_mislabel_and_protocol_errors_fail_before_budget(self):
        official = fake_official()
        actor = fake_actor(official)
        identity = {"base_sha256": BASE, "delta_sha256": DELTA}
        bind_evaluation_identity(actor, **identity)
        budget, factory = FakeBudget(), FakeFactory()
        common = dict(official=official, actor=actor, env_factory=factory,
                      cases=(EvaluationCase("L", 10, 20),), budget=budget, metrics_fn=fake_metrics)
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate_cases(**common, method="base_training", protocol=MATCHED_TL_PROTOCOL,
                           checkpoint_identity=identity)
        with self.assertRaisesRegex(ValueError, "T only"):
            evaluate_cases(**common, method="delta_training", protocol=ORIGINAL_T_PROTOCOL,
                           checkpoint_identity=identity, delta_applied=True)
        with self.assertRaisesRegex(ValueError, "identity marker"):
            evaluate_cases(**common, method="base_training", protocol=MATCHED_TL_PROTOCOL,
                           checkpoint_identity={"base_sha256": BASE, "delta_sha256": None})
        self.assertEqual(budget.events, [])
        self.assertEqual(factory.environments, [])


if __name__ == "__main__":
    unittest.main()
