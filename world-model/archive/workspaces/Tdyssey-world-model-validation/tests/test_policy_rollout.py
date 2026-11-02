"""Fake-only collector/budget checks; never instantiate or step a real simulator."""

from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    torch = None

from tdyssey.policy_returns import BlockBoundary, plan_gae

if torch is not None:
    from tdyssey.policy_rl import SampledChain, SamplerConfig
    from tdyssey.policy_rollout import BudgetBusy, BudgetExceeded, HardBudget, collect_fragment

_FakeModule = nn.Module if torch is not None else object


class FakeOfficial:
    def observation_tensors(self, history):
        return {key: torch.stack([torch.as_tensor(obs[key]) for obs in history])[None]
                for key in ("image", "agent_pos")}


class FakeActor(_FakeModule):
    def __init__(self, budget_path, fail_sample=False):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.budget_path = budget_path
        self.fail_sample = fail_sample
        self.samples = 0
        self.condition_keys = []

    def encode(self, observations):
        self.condition_keys.append(set(observations))
        return observations["agent_pos"][:, :, 0].float() + self.bias * 0.0

    def sample(self, observations, *, generator, mode, actor_version):
        ledger = json.loads(self.budget_path.read_text())
        assert ledger["events"][-1]["kind"] == "plan"
        assert not ledger["events"][-1]["completed"]
        assert not torch.is_grad_enabled()
        assert generator is not None and mode == "training"
        self.samples += 1
        if self.fail_sample:
            raise RuntimeError("fake sampler failure")
        targets = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
        chain = torch.zeros(1, 101, 16, 2) + self.bias * 0.0
        return SampledChain(observations, chain, torch.arange(99, -1, -1),
                            torch.zeros(1, 100, 16, 2), torch.ones(100),
                            torch.zeros(1, 100), targets, targets[:, 1:9].clone(),
                            SamplerConfig(), "fake-locked-scheduler", mode, actor_version)


class FakeCritic(_FakeModule):
    def forward(self, features):
        return features[:, -1:] * 0.1


class FakeEnv:
    """No physics backend; marker pixels/positions reveal observation timing."""

    sim_hz, control_hz = 100, 10

    def __init__(self, budget_path, coverage=None, *, shape="T", fail_step=False,
                 external_timeout=None):
        self.budget_path = budget_path
        self.project_shape = shape
        self.coverages = coverage or [0.1] * 65
        self.fail_step = fail_step
        self.external_timeout = external_timeout
        self.steps = self.resets = 0
        self.actions = []

    def seed(self, seed):
        self.reset_seed = seed

    def _observation(self):
        return {"image": np.full((3, 96, 96), self.steps / 100.0, dtype=np.float32),
                "agent_pos": np.array([self.steps, self.steps + 1], dtype=np.float64),
                "privileged_pose": np.array([123.0, 456.0, 789.0])}

    def reset(self):
        ledger = json.loads(self.budget_path.read_text())
        assert ledger["events"][-1]["kind"] == "reset"
        assert not ledger["events"][-1]["completed"]
        self.resets += 1
        self.steps = 0
        return self._observation()

    def step(self, action):
        ledger = json.loads(self.budget_path.read_text())
        assert ledger["events"][-1]["kind"] == "control"
        assert not ledger["events"][-1]["completed"]
        assert not torch.is_grad_enabled()
        self.steps += 1
        self.actions.append(np.array(action).copy())
        if self.fail_step:
            raise RuntimeError("fake step failure")
        coverage = self.coverages[self.steps]
        timeout = self.external_timeout == self.steps
        return (self._observation(), min(coverage / 0.95, 1.0),
                coverage > 0.95 or timeout, {"TimeLimit.truncated": timeout})


def fake_metrics(env):
    return {"coverage": env.coverages[env.steps], "iou": env.coverages[env.steps] / 2.0}


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class HardBudgetChecks(unittest.TestCase):
    def test_durable_reservations_completion_and_reopen_never_refund(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "budget.json"
            with HardBudget(path, max_controls=2, max_plans=1) as budget:
                attempt = budget.reserve("control", "T")
                persisted = json.loads(path.read_text())
                self.assertEqual(persisted["reserved"]["control"], 1)
                self.assertEqual(persisted["completed"]["control"], 0)
                budget.mark_completed(attempt)
                budget.reserve("control", "L")  # Simulates a failed attempt: no completion.
                with self.assertRaises(BudgetExceeded):
                    budget.reserve("control", "T")
                budget.reserve("actor_step")
                budget.reserve("critic_step")
                budget.reserve("plan", "L")
                for kind in ("actor_step", "critic_step", "plan"):
                    with self.subTest(kind=kind), self.assertRaises(BudgetExceeded):
                        budget.reserve(kind)
                snapshot = budget.snapshot()
                snapshot["reserved"]["control"] = 0
                self.assertEqual(budget.snapshot()["reserved"]["control"], 2)
            with HardBudget(path, max_controls=2, max_plans=1) as reopened:
                self.assertEqual(reopened.snapshot()["completed"]["control"], 1)
                self.assertEqual(reopened.snapshot()["by_shape"]["L"]["control"], 1)
                with self.assertRaises(BudgetExceeded):
                    reopened.reserve("control")
            with self.assertRaisesRegex(ValueError, "persisted"):
                HardBudget(path)  # Cannot expand a ledger created with smaller limits.

    def test_cross_process_owner_and_crash_lock_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "budget.json"
            code = ("import sys; from tdyssey.policy_rollout import HardBudget, BudgetBusy\n"
                    "try:\n HardBudget(sys.argv[1])\n"
                    "except BudgetBusy:\n print('locked')\n"
                    "else:\n raise RuntimeError('unexpected concurrent owner')\n")
            with HardBudget(path):
                child = subprocess.run([sys.executable, "-B", "-c", code, str(path)],
                                       capture_output=True, text=True, check=True)
                self.assertEqual(child.stdout.strip(), "locked")
            crash_path = Path(directory) / "crashed.json"
            crash_code = ("import os,sys; from tdyssey.policy_rollout import HardBudget; "
                          "b=HardBudget(sys.argv[1]); b.reserve('control','T'); os._exit(0)")
            subprocess.run([sys.executable, "-B", "-c", crash_code, str(crash_path)], check=True)
            self.assertEqual(json.loads(crash_path.read_text())["reserved"]["control"], 1)
            with self.assertRaises(BudgetBusy):
                HardBudget(crash_path)

    def test_corrupt_ledger_and_excess_authorization_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "budget.json"
            with self.assertRaises(ValueError):
                HardBudget(path, max_controls=65)
            with HardBudget(path) as budget:
                budget.reserve("control", "T")
            corrupt = json.loads(path.read_text())
            corrupt["reserved"]["control"] = 0
            path.write_text(json.dumps(corrupt))
            with self.assertRaisesRegex(ValueError, "disagree"):
                HardBudget(path)


@unittest.skipIf(torch is None, "Optional numpy/torch runtime is unavailable")
class FragmentChecks(unittest.TestCase):
    def collect(self, directory, *, coverage=None, controls=13, plans=2,
                episode_limit=100, shape="T", fail_step=False, fail_sample=False,
                external_timeout=None, reset_seed=200019):
        path = Path(directory) / "budget.json"
        budget = HardBudget(path)
        self.addCleanup(budget.close)
        actor = FakeActor(path, fail_sample=fail_sample)
        env = FakeEnv(path, coverage, shape=shape, fail_step=fail_step,
                      external_timeout=external_timeout)
        kwargs = dict(shape=shape, reset_seed=reset_seed,
                      generator=torch.Generator().manual_seed(7), budget=budget,
                      max_controls=controls, max_plans=plans, episode_limit=episode_limit,
                      actor_version="fake-v0", metrics_fn=fake_metrics)
        records = collect_fragment(FakeOfficial(), actor, FakeCritic(), env, **kwargs)
        return records, budget, actor, env

    def test_eight_plus_five_controls_exact_histories_and_cpu_records(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            records, budget, actor, env = self.collect(directory)
            self.assertEqual([record.block.num_controls for record in records], [8, 5])
            self.assertEqual([record.block.boundary for record in records],
                             [BlockBoundary.CONTINUE, BlockBoundary.BUFFER_CUTOFF])
            self.assertEqual(env.steps, 13)
            self.assertEqual(env.resets, 1)
            self.assertEqual(records[0].sample.observations["agent_pos"][0, :, 0].tolist(), [0, 0])
            self.assertEqual(records[1].sample.observations["agent_pos"][0, :, 0].tolist(), [7, 8])
            self.assertEqual(records[1].final_observation["agent_pos"][0, :, 0].tolist(), [12, 13])
            self.assertAlmostEqual(records[0].block.final_observation_value, 0.8, places=6)
            self.assertAlmostEqual(records[1].block.value, 0.8, places=6)
            self.assertAlmostEqual(records[1].block.final_observation_value, 1.3, places=6)
            self.assertEqual(records[1].features.tolist(), [[7.0, 8.0]])
            self.assertEqual(len(plan_gae([record.block for record in records])), 2)
            self.assertTrue(all(keys == {"image", "agent_pos"} for keys in actor.condition_keys))
            for plan in records:
                for field in fields(plan.sample):
                    value = getattr(plan.sample, field.name)
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertFalse(value.requires_grad)
                self.assertFalse(plan.features.requires_grad)
                for index, step in enumerate(plan.control_records):
                    torch.testing.assert_close(step["action"], plan.sample.actions[0, index])
                    self.assertEqual(step["control_physics_substeps"], 10)
                    self.assertEqual(step["upstream_done"], False)
                    self.assertAlmostEqual(step["raw_reward"], 0.1 / 0.95)
                    self.assertAlmostEqual(step["training_reward"], -0.001)
                    self.assertEqual(step["observation_after"]["agent_pos"][0].item(),
                                     step["control_index"])
            counts = budget.snapshot()
            self.assertEqual(counts["reserved"], {"control": 13, "plan": 2,
                                                   "actor_step": 0, "critic_step": 0})
            self.assertEqual(counts["completed"], counts["reserved"])
            self.assertEqual((counts["reset_attempts"], counts["reset_completed"]), (1, 1))
            budget.close()

    def test_third_step_success_wins_over_timeout_and_never_executes_suffix(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            records, budget, actor, env = self.collect(
                directory, coverage=[0.1, 0.2, 0.95, 0.951], episode_limit=3)
            self.assertEqual((env.steps, actor.samples), (3, 1))
            block = records[0].block
            self.assertEqual(block.boundary, BlockBoundary.SUCCESS)
            self.assertEqual((block.bootstrap_mask, block.continuation_mask), (0, 0))
            self.assertIsNone(block.final_observation_value)
            self.assertEqual([r["first_success"] for r in records[0].control_records],
                             [False, False, True])
            self.assertAlmostEqual(records[0].control_records[-1]["training_reward"], 0.05)
            self.assertEqual(records[0].control_records[-1]["coverage"], 0.951)
            self.assertEqual(budget.snapshot()["reserved"]["control"], 3)
            budget.close()

    def test_episode_timeout_uses_pre_reset_final_observation_and_potential(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            records, budget, _, env = self.collect(directory, shape="L", episode_limit=5,
                                                   coverage=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
            record, = records
            self.assertEqual(env.resets, 1)
            self.assertEqual(record.block.boundary, BlockBoundary.TIMEOUT)
            self.assertEqual((record.block.bootstrap_mask, record.block.continuation_mask), (1, 0))
            self.assertAlmostEqual(record.block.final_observation_value, 0.5)
            self.assertEqual(record.final_observation["agent_pos"][0, :, 0].tolist(), [4, 5])
            self.assertAlmostEqual(record.control_records[-1]["training_reward"], 0.99 * 0.7 - 0.5)
            self.assertEqual(record.shape, "L")
            budget.close()

    def test_plan_cap_finishes_sampled_prefix_and_external_timeout_stops_early(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            records, budget, _, env = self.collect(directory, controls=32, plans=1)
            self.assertEqual(env.steps, 8)
            self.assertEqual(records[-1].block.boundary, BlockBoundary.BUFFER_CUTOFF)
            budget.close()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            records, budget, _, env = self.collect(directory, external_timeout=2)
            self.assertEqual(env.steps, 2)
            self.assertEqual(records[-1].block.boundary, BlockBoundary.TIMEOUT)
            self.assertTrue(records[-1].control_records[-1]["upstream_done"])
            budget.close()

    def test_T_L_fragments_share_counts_and_keep_separate_gae_streams(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "budget.json"
            with HardBudget(path) as budget:
                actor, critic = FakeActor(path), FakeCritic()
                fragments = []
                for shape, steps, seed in (("T", 3, 200019), ("L", 5, 200020)):
                    env = FakeEnv(path, shape=shape)
                    fragment = collect_fragment(
                        FakeOfficial(), actor, critic, env, shape=shape, reset_seed=seed,
                        generator=torch.Generator().manual_seed(7), budget=budget,
                        max_controls=steps, max_plans=1, episode_limit=100,
                        actor_version="fake-v0", metrics_fn=fake_metrics)
                    fragments.append(fragment)
                    self.assertEqual(len(plan_gae([row.block for row in fragment])), 1)
                counts = budget.snapshot()
                self.assertEqual((counts["reserved"]["control"], counts["reserved"]["plan"],
                                  counts["reset_attempts"]), (8, 2, 2))
                self.assertEqual(counts["by_shape"]["T"]["control"], 3)
                self.assertEqual(counts["by_shape"]["L"]["control"], 5)
                self.assertEqual(fragments[1][0].control_records[0]["global_control_attempt"], 4)
                self.assertNotEqual(fragments[0][0].block.episode_id, fragments[1][0].block.episode_id)

    def test_disjoint_seeds_invalid_requests_and_exhaustion_precede_reset(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "budget.json"
            with HardBudget(path, max_controls=1, max_plans=1) as budget:
                env = FakeEnv(path)
                base = dict(shape="T", reset_seed=100000, generator=torch.Generator(),
                            budget=budget, max_controls=1, max_plans=1, episode_limit=10,
                            actor_version="fake", metrics_fn=fake_metrics)
                with self.assertRaisesRegex(ValueError, "development"):
                    collect_fragment(FakeOfficial(), FakeActor(path), FakeCritic(), env, **base)
                base["reset_seed"] = 200019
                budget.reserve("control", "T")
                with self.assertRaises(BudgetExceeded):
                    collect_fragment(FakeOfficial(), FakeActor(path), FakeCritic(), env, **base)
                self.assertEqual((env.steps, env.resets), (0, 0))
                self.assertEqual(budget.snapshot()["reset_attempts"], 0)

    def test_failed_step_and_sampler_keep_attempts_across_reopen(self):
        for failure in ("step", "sample"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
                path = Path(directory) / "budget.json"
                with self.assertRaisesRegex(RuntimeError, f"fake {failure if failure == 'step' else 'sampler'}"):
                    self.collect(directory, fail_step=failure == "step", fail_sample=failure == "sample")
                # Release the still-live owner explicitly; exceptions do not release
                # a caller-owned budget or erase any of its reservations.
                self.doCleanups()
                with HardBudget(path) as budget:
                    counts = budget.snapshot()
                    self.assertEqual(counts["reserved"]["plan"], 1)
                    self.assertEqual(counts["reserved"]["control"], int(failure == "step"))
                    self.assertEqual(counts["completed"]["control"], 0)
                    self.assertEqual(counts["reset_completed"], 1)


if __name__ == "__main__":
    unittest.main()
