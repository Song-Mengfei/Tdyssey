"""Fake-only continuous streams: no research checkpoint, real physics or updates."""

from copy import deepcopy
import gc
import unittest
import weakref

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from tdyssey.policy_protocol import BudgetExceeded
    from tdyssey.policy_returns import BlockBoundary, plan_gae
    from tdyssey.policy_rl import SampledChain, SamplerConfig
    from tdyssey.policy_stream import ContinuousCollector, SceneCursor, SceneExhausted


_Module = nn.Module if torch is not None else object
EXCLUDED = {100000, 100001, 100002, 900001, 900002}
SCENES = {"T": [700001, 700002, 700003, 700004], "L": [800001, 800002, 800003, 800004]}


class FakeBudget:
    """In-memory C ledger protocol; durability is covered by protocol tests."""

    def __init__(self, *, controls=1000, plans=100, resets=100,
                 stop_after_controls=None, stop_after_plans=None):
        self.limits = {"control": controls, "plan": plans, "reset": resets}
        self.reserved = {"train": {kind: 0 for kind in self.limits}}
        self.completed = deepcopy(self.reserved)
        self.events = []
        self.stop_after_controls = stop_after_controls
        self.stop_after_plans = stop_after_plans

    def check_time(self):
        if ((self.stop_after_controls is not None and self.completed["train"]["control"] >= self.stop_after_controls)
                or (self.stop_after_plans is not None and self.completed["train"]["plan"] >= self.stop_after_plans)):
            raise BudgetExceeded("fake deadline")

    def remaining(self, kind, phase="train"):
        return max(0, self.limits[kind] - self.reserved[phase][kind])

    def reserve(self, kind, phase="train", shape=None, amount=1):
        self.check_time()
        if self.remaining(kind, phase) < amount:
            raise BudgetExceeded("fake operation cap")
        self.reserved[phase][kind] += amount
        self.events.append({"kind": kind, "phase": phase, "shape": shape,
                            "amount": amount, "completed": False})
        return len(self.events)

    def mark_completed(self, event_id):
        event = self.events[event_id - 1]
        assert not event["completed"]
        event["completed"] = True
        self.completed[event["phase"]][event["kind"]] += event["amount"]

    def snapshot(self):
        return deepcopy({"reserved": self.reserved, "completed": self.completed, "events": self.events})


class FakeOfficial:
    def observation_tensors(self, history):
        assert len(history) == 2
        return {key: torch.stack([torch.as_tensor(obs[key]) for obs in history])[None]
                for key in ("image", "agent_pos")}


class FakeActor(_Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget
        self.input_keys = []

    def encode(self, observations):
        self.input_keys.append(set(observations))
        assert not torch.is_grad_enabled()
        return observations["agent_pos"][:, :, 0].float()

    def sample(self, observations, *, generator, mode, actor_version):
        assert not torch.is_grad_enabled()
        event = self.budget.events[-1]
        assert event["kind"] == "plan" and not event["completed"] and event["phase"] == "train"
        assert mode == "training"
        self.input_keys.append(set(observations))
        chain = torch.randn((1, 101, 16, 2), generator=generator)
        targets = chain[:, -1] * 10 + 256
        return SampledChain({key: value.clone() for key, value in observations.items()}, chain,
                            torch.arange(99, -1, -1), torch.zeros(1, 100, 16, 2),
                            torch.ones(100), torch.zeros(1, 100), targets,
                            targets[:, 1:9].clone(), SamplerConfig(), "fake-scheduler", mode, actor_version)


class FakeCritic(_Module):
    def forward(self, features):
        return features[:, -1:] * 0.01


class FakeEnv:
    sim_hz, control_hz = 100, 10

    def __init__(self, shape, budget, *, success_after=None, fail_step=False):
        self.project_shape = shape
        self.budget = budget
        self.success_after = success_after or {}
        self.fail_step = fail_step
        self.reset_seeds = []
        self.episode_steps = self.total_steps = 0
        self.current_seed = None
        self.actions = []

    def seed(self, seed):
        self.current_seed = seed

    def _observation(self):
        return {"image": np.full((3, 96, 96), self.episode_steps / 1000, dtype=np.float32),
                "agent_pos": np.array([self.episode_steps, self.current_seed % 100], dtype=np.float64),
                "privileged_object_pose": np.array([111.0, 222.0, 0.5])}

    def reset(self):
        event = self.budget.events[-1]
        assert event["kind"] == "reset" and not event["completed"] and event["shape"] == self.project_shape
        self.reset_seeds.append(self.current_seed)
        self.episode_steps = 0
        self.coverage = 0.1
        return self._observation()

    def step(self, action):
        event = self.budget.events[-1]
        assert event["kind"] == "control" and not event["completed"] and event["shape"] == self.project_shape
        assert not torch.is_grad_enabled()
        self.episode_steps += 1
        self.total_steps += 1
        self.actions.append(np.array(action).copy())
        if self.fail_step:
            raise RuntimeError("fake physical operation failed")
        successful = self.success_after.get(self.current_seed) == self.episode_steps
        self.coverage = 0.951 if successful else 0.1 + self.episode_steps * 0.01
        return (self._observation(), min(self.coverage / 0.95, 1.0), successful,
                {"n_contacts": int(self.episode_steps % 2 == 0)})


def fake_metrics(env):
    return {"coverage": env.coverage, "iou": env.coverage / 2}


def collector_fixture(*, budget=None, scenes=None, episode_limit=300,
                      success_after=None, failed_shape=None, generator_seed=42):
    budget = budget or FakeBudget()
    environments = {shape: FakeEnv(shape, budget, success_after=success_after,
                                  fail_step=failed_shape == shape) for shape in ("T", "L")}
    generators = {shape: torch.Generator().manual_seed(generator_seed + index)
                  for index, shape in enumerate(("T", "L"))}
    actor = FakeActor(budget)
    collector = ContinuousCollector(FakeOfficial(), actor, FakeCritic(), environments,
                                    generators, SceneCursor(scenes or SCENES, EXCLUDED), budget,
                                    episode_limit=episode_limit, metrics_fn=fake_metrics)
    return collector, environments, budget, actor


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class SceneCursorChecks(unittest.TestCase):
    def test_fixed_excluded_unique_scenes_restore_and_exhaustion(self):
        cursor = SceneCursor({"T": [700001], "L": [800001]}, EXCLUDED)
        self.assertEqual(cursor.next("T"), 700001)
        with self.assertRaises(SceneExhausted):
            cursor.next("T")
        state = cursor.state_dict()
        restored = SceneCursor({"T": [700001], "L": [800001]}, EXCLUDED)
        restored.load_state_dict(state)
        self.assertEqual(restored.remaining("T"), 0)
        self.assertEqual(restored.next("L"), 800001)
        with self.assertRaisesRegex(ValueError, "rewind"):
            restored.load_state_dict(state)
        for scenes, excluded in ((SCENES, {100000}),
                                 ({"T": [900001], "L": [800001]}, EXCLUDED),
                                 ({"T": [700001, 700001], "L": [800001]}, EXCLUDED)):
            with self.subTest(scenes=scenes, excluded=excluded), self.assertRaises(ValueError):
                SceneCursor(scenes, excluded)


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class ContinuousCollectorChecks(unittest.TestCase):
    def test_batches_preserve_episodes_history_rng_and_actual_shape_ratio(self):
        collector, envs, budget, actor = collector_fixture()
        first = collector.collect_batch(max_plans=2, max_controls=13, actor_version="v0")
        self.assertEqual([fragment[0].shape for fragment in first], ["T", "L"])
        self.assertEqual([fragment[0].block.num_controls for fragment in first], [8, 5])
        self.assertTrue(all(fragment[-1].block.boundary == BlockBoundary.BUFFER_CUTOFF for fragment in first))
        old_ids = [fragment[0].block.episode_id for fragment in first]
        old_rng = {shape: collector.generators[shape].get_state().clone() for shape in ("T", "L")}
        second = collector.collect_batch(max_plans=2, max_controls=9, actor_version="v1")
        self.assertEqual([fragment[0].block.episode_id for fragment in second], old_ids)
        self.assertEqual([envs[shape].reset_seeds for shape in ("T", "L")], [[700001], [800001]])
        self.assertEqual([envs[shape].episode_steps for shape in ("T", "L")], [16, 6])
        self.assertEqual(second[0][0].sample.observations["agent_pos"][0, :, 0].tolist(), [7, 8])
        self.assertEqual(second[1][0].sample.observations["agent_pos"][0, :, 0].tolist(), [4, 5])
        self.assertTrue(all(p.sample.actor_version == "v1" for fragment in second for p in fragment))
        self.assertTrue(all(not torch.equal(old_rng[s], collector.generators[s].get_state()) for s in old_rng))
        summary = collector.summary()
        self.assertFalse(summary["resumable"])
        self.assertEqual(summary["control_shares"], {"T": 16 / 22, "L": 6 / 22})
        self.assertEqual(summary["by_shape"]["T"]["contact_controls"], 8)
        self.assertEqual(summary["by_shape"]["L"]["contact_controls"], 3)
        self.assertEqual(summary["by_shape"]["T"]["nonzero_training_reward_controls"], 16)
        self.assertTrue(all(keys == {"image", "agent_pos"} for keys in actor.input_keys))
        self.assertEqual(budget.completed["train"], {"control": 22, "plan": 4, "reset": 2})
        for fragment in second:
            self.assertEqual(len(plan_gae([p.block for p in fragment])), 1)
            for record in fragment[0].control_records:
                self.assertIn("n_contacts", record)
                self.assertAlmostEqual(record["coverage_delta"], 0.01)
                self.assertEqual(record["control_physics_substeps"], 10)
            self.assertFalse(fragment[0].sample.chain.requires_grad)
            self.assertEqual(fragment[0].sample.chain.device.type, "cpu")

    def test_round_robin_cursor_survives_single_plan_partial_batches(self):
        collector, envs, _, _ = collector_fixture()
        first = collector.collect_batch(5, 2, "v0")
        second = collector.collect_batch(5, 1, "v1")
        self.assertEqual((first[0][0].shape, second[0][0].shape), ("T", "L"))
        self.assertEqual((envs["T"].total_steps, envs["L"].total_steps), (2, 1))
        self.assertEqual(collector.summary()["last_batch"]["plan_order"], ["L"])

    def test_success_priority_timeout_and_new_episodes_have_separate_gae(self):
        collector, envs, budget, _ = collector_fixture(
            episode_limit=3, success_after={700001: 3, 800001: 3})
        fragments = collector.collect_batch(4, 24, "v0")
        self.assertEqual(len(fragments), 4)
        self.assertEqual([fragment[0].shape for fragment in fragments], ["T", "L", "T", "L"])
        self.assertEqual([fragment[0].block.boundary for fragment in fragments],
                         [BlockBoundary.SUCCESS, BlockBoundary.SUCCESS,
                          BlockBoundary.TIMEOUT, BlockBoundary.TIMEOUT])
        self.assertEqual(len({fragment[0].block.episode_id for fragment in fragments}), 4)
        self.assertTrue(collector.is_episode_boundary)
        self.assertEqual(budget.completed["train"]["control"], 12)
        self.assertEqual(envs["T"].reset_seeds, [700001, 700002])
        self.assertEqual(envs["L"].reset_seeds, [800001, 800002])
        for fragment in fragments:
            plan, = fragment
            result, = plan_gae([plan.block])
            self.assertEqual(plan.block.continuation_mask, 0)
            self.assertAlmostEqual(result.advantage, result.delta)
            self.assertEqual(plan.control_records[0]["control_index"], 1)
        success = fragments[0][0]
        self.assertIsNone(success.block.final_observation_value)
        self.assertEqual([r["first_success"] for r in success.control_records], [False, False, True])
        timeout = fragments[2][0]
        self.assertAlmostEqual(timeout.block.final_observation_value, 0.03, places=6)
        self.assertAlmostEqual(timeout.control_records[-1]["training_reward"], 0.99 * 0.13 - 0.12)

    def test_total_budget_tail_keeps_live_episodes_and_does_not_reset_on_retry(self):
        collector, envs, budget, _ = collector_fixture(budget=FakeBudget(controls=11))
        fragments = collector.collect_batch(10, 100, "v0")
        self.assertEqual([fragment[0].block.num_controls for fragment in fragments], [8, 3])
        self.assertTrue(all(fragment[-1].block.bootstrap_mask == 1 and
                            fragment[-1].block.continuation_mask == 0 for fragment in fragments))
        self.assertFalse(collector.is_episode_boundary)
        before = deepcopy(budget.events)
        self.assertEqual(collector.collect_batch(10, 100, "v0"), [])
        self.assertEqual(budget.events, before)
        self.assertEqual([len(envs[s].reset_seeds) for s in ("T", "L")], [1, 1])
        with self.assertRaisesRegex(RuntimeError, "contact-cache"):
            collector.state_dict()
        audit = collector.state_dict(require_boundary=False)
        self.assertFalse(audit["resumable"])
        self.assertTrue(all("history" not in state for state in audit["streams"].values()))
        restored, new_envs, _, _ = collector_fixture()
        with self.assertRaisesRegex(ValueError, "mid-episode"):
            restored.load_state_dict(audit)
        self.assertTrue(all(not env.reset_seeds and not env.total_steps for env in new_envs.values()))
        forged_boundary = deepcopy(audit)
        forged_boundary.update(resumable=True, episode_boundary=True, boundary_requested=False)
        for stream in forged_boundary["streams"].values():
            stream["active"] = False
        with self.assertRaisesRegex(ValueError, "actual success/timeout"):
            restored.load_state_dict(forged_boundary)

    def test_scene_exhaustion_stops_without_wrapping_or_extra_resets(self):
        scenes = {"T": [700001], "L": [800001]}
        collector, envs, budget, _ = collector_fixture(scenes=scenes, episode_limit=1)
        fragments = collector.collect_batch(10, 100, "v0")
        self.assertEqual(len(fragments), 2)
        self.assertEqual(collector.summary()["last_batch"]["stop_reason"], "scenes_exhausted")
        self.assertEqual(collector.summary()["exhausted_shapes"], ["T", "L"])
        self.assertEqual(budget.completed["train"], {"control": 2, "plan": 2, "reset": 2})
        self.assertEqual(collector.collect_batch(10, 100, "v1"), [])
        self.assertTrue(collector.is_episode_boundary)
        self.assertEqual(envs["T"].reset_seeds, [700001])
        self.assertEqual(envs["L"].reset_seeds, [800001])

    def test_exhausted_stream_is_explicit_while_other_live_stream_continues(self):
        collector, envs, _, _ = collector_fixture(scenes={"T": [700001], "L": [800001]},
                                                  success_after={700001: 1})
        fragments = collector.collect_batch(3, 100, "v0")
        self.assertEqual(collector.summary()["last_batch"]["plan_order"], ["T", "L", "L"])
        self.assertEqual([len(fragment) for fragment in fragments], [1, 2])
        self.assertEqual([envs[s].total_steps for s in ("T", "L")], [1, 16])
        self.assertAlmostEqual(collector.summary()["T_to_L_control_ratio"], 1 / 16)
        self.assertEqual(envs["T"].reset_seeds, [700001])

    def test_time_limit_can_cut_a_prefix_or_consume_a_sample_without_zero_block(self):
        collector, envs, budget, _ = collector_fixture(budget=FakeBudget(stop_after_controls=3))
        fragments = collector.collect_batch(10, 100, "v0")
        self.assertEqual(fragments[0][0].block.num_controls, 3)
        self.assertEqual(fragments[0][0].block.boundary, BlockBoundary.BUFFER_CUTOFF)
        self.assertEqual(envs["T"].total_steps, 3)
        self.assertEqual(budget.completed["train"]["control"], 3)
        self.assertFalse(collector.summary()["resumable"])
        zero, zero_envs, zero_budget, _ = collector_fixture(budget=FakeBudget(stop_after_plans=1))
        self.assertEqual(zero.collect_batch(10, 100, "v0"), [])
        self.assertEqual(zero_budget.completed["train"], {"control": 0, "plan": 1, "reset": 1})
        self.assertEqual(zero.summary()["by_shape"]["T"]["sampled_but_unexecuted_plans"], 1)
        self.assertFalse(zero.is_episode_boundary)
        self.assertEqual(sum(env.total_steps for env in zero_envs.values()), 0)

    def test_boundary_restore_preserves_scene_cursor_local_rng_and_next_plan(self):
        collector, envs, budget, _ = collector_fixture(episode_limit=2)
        collector.collect_batch(2, 4, "v0")
        self.assertTrue(collector.is_episode_boundary)
        state = collector.state_dict()
        saved_budget = deepcopy(budget)
        expected = collector.collect_batch(2, 4, "v1")
        restored, restored_envs, _, _ = collector_fixture(
            episode_limit=2, budget=saved_budget, generator_seed=999)
        restored.load_state_dict(state)
        self.assertTrue(restored.is_episode_boundary)
        self.assertTrue(all(not env.reset_seeds for env in restored_envs.values()))
        actual = restored.collect_batch(2, 4, "v1")
        for expected_fragment, actual_fragment in zip(expected, actual):
            expected_plan, actual_plan = expected_fragment[0], actual_fragment[0]
            self.assertEqual(expected_plan.reset_seed, actual_plan.reset_seed)
            self.assertEqual(expected_plan.block, actual_plan.block)
            torch.testing.assert_close(expected_plan.sample.chain, actual_plan.sample.chain, rtol=0, atol=0)
            torch.testing.assert_close(expected_plan.sample.actions, actual_plan.sample.actions, rtol=0, atol=0)
        self.assertEqual(restored.summary(), collector.summary())
        self.assertEqual(restored_envs["T"].reset_seeds, [700002])
        self.assertEqual(restored_envs["L"].reset_seeds, [800002])
        with self.assertRaisesRegex(RuntimeError, "newly constructed"):
            restored.load_state_dict(state)

    def test_failed_operation_is_not_refunded_or_retried(self):
        collector, envs, budget, _ = collector_fixture(failed_shape="T")
        with self.assertRaisesRegex(RuntimeError, "fake physical"):
            collector.collect_batch(2, 16, "v0")
        self.assertEqual(budget.reserved["train"]["control"], 1)
        self.assertEqual(budget.completed["train"]["control"], 0)
        self.assertEqual(envs["T"].total_steps, 1)
        with self.assertRaisesRegex(RuntimeError, "failed previously"):
            collector.collect_batch(2, 16, "v0")
        self.assertEqual(envs["T"].total_steps, 1)
        with self.assertRaisesRegex(RuntimeError, "not resumable"):
            collector.state_dict()

    def test_returned_chains_are_released_between_batches(self):
        collector, _, _, _ = collector_fixture()

        def batch_references():
            fragments = collector.collect_batch(2, 16, "v0")
            return [weakref.ref(tensor) for fragment in fragments for plan in fragment
                    for tensor in (plan.sample.chain, plan.sample.old_log_prob, plan.features,
                                   plan.control_records[-1]["observation_after"]["image"])]

        references = batch_references()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        second = collector.collect_batch(2, 16, "v1")
        self.assertEqual(sum(plan.block.num_controls for fragment in second for plan in fragment), 16)
        self.assertTrue(all(reference() is None for reference in references))

    def test_requested_boundary_drains_across_batches_without_restarting_done_streams(self):
        collector, envs, budget, _ = collector_fixture(episode_limit=5)
        first = collector.collect_batch(2, 6, "v0")
        self.assertEqual([fragment[0].block.num_controls for fragment in first], [5, 1])
        self.assertEqual(first[0][0].block.boundary, BlockBoundary.TIMEOUT)
        collector.request_boundary()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            collector.release_boundary()
        second = collector.collect_batch(1, 2, "v1")
        self.assertEqual(second[0][0].shape, "L")
        self.assertEqual(second[0][0].block.num_controls, 2)
        self.assertFalse(collector.is_episode_boundary)
        self.assertEqual(envs["T"].reset_seeds, [700001])
        third = collector.collect_batch(1, 8, "v2")
        self.assertEqual(third[0][0].block.boundary, BlockBoundary.TIMEOUT)
        self.assertEqual(third[0][0].block.num_controls, 2)
        self.assertTrue(collector.is_episode_boundary)
        self.assertTrue(collector.summary()["draining"])
        state = collector.state_dict()
        self.assertTrue(state["saved_at_requested_boundary"])
        self.assertFalse(state["boundary_requested"])
        before = deepcopy(budget.events)
        self.assertEqual(collector.collect_batch(8, 100, "v3"), [])
        self.assertEqual(budget.events, before)
        self.assertEqual(budget.completed["train"], {"control": 10, "plan": 4, "reset": 2})
        collector.release_boundary()
        fourth = collector.collect_batch(1, 1, "v3")
        self.assertEqual(fourth[0][0].shape, "T")
        self.assertEqual(envs["T"].reset_seeds, [700001, 700002])
        self.assertEqual(budget.completed["train"], {"control": 11, "plan": 5, "reset": 3})

    def test_drain_stops_at_authorized_tail_without_forced_completion(self):
        collector, envs, budget, _ = collector_fixture(budget=FakeBudget(controls=7), episode_limit=5)
        collector.collect_batch(2, 6, "v0")
        collector.request_boundary()
        final = collector.collect_batch(8, 100, "v1")
        self.assertEqual(final[0][0].block.num_controls, 1)
        self.assertEqual(final[0][0].block.boundary, BlockBoundary.BUFFER_CUTOFF)
        self.assertFalse(collector.summary()["resumable"])
        self.assertTrue(collector.summary()["draining"])
        self.assertEqual(budget.completed["train"]["control"], 7)
        self.assertEqual((envs["T"].episode_steps, envs["L"].episode_steps), (5, 2))
        self.assertEqual(envs["T"].reset_seeds, [700001])
        self.assertEqual(envs["L"].reset_seeds, [800001])


if __name__ == "__main__":
    unittest.main()
