"""Synthetic CPU updates only: no simulator calls or research checkpoint loads."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_policy_rl_runtime import actor_fixture, observation_fixture, torch

if torch is not None:
    from tdyssey.policy_returns import ActionBlock, BlockBoundary
    from tdyssey.policy_rl import transition_ppo_loss
    from tdyssey.policy_rollout import BudgetExceeded, HardBudget, RolloutPlan
    from tdyssey.policy_training import (
        UpdateConfig, VisualCritic, make_optimizers, prepare_batch, update_once,
    )


def plan_fixture(actor, critic, index, *, rewards, value=None, next_value=None,
                 boundary=None, episode=None):
    observations = observation_fixture(1)
    observations["agent_pos"].add_(16 * index)
    sample = actor.sample(observations, generator=torch.Generator().manual_seed(100 + index),
                          actor_version="synthetic-before-update")
    features = actor.encode(observations).detach().cpu().clone()
    if value is None:
        with torch.no_grad():
            value = float(critic(features).reshape(-1).item())
    boundary = BlockBoundary.SUCCESS if boundary is None else boundary
    block = ActionBlock(episode_id=f"synthetic-{index}" if episode is None else episode,
                        rewards=tuple(rewards), value=value,
                        final_observation_value=next_value, boundary=boundary)
    return RolloutPlan(sample=sample, block=block, shape="T" if index % 2 == 0 else "L",
                       reset_seed=1000 + index,
                       control_records=[{"training_reward": reward, "synthetic": True}
                                        for reward in rewards],
                       final_observation={key: item.clone() for key, item in observations.items()},
                       features=features)


def model_fixture():
    actor = actor_fixture()
    feature_dim = actor.encode(observation_fixture(1)).shape[1]
    critic = VisualCritic(feature_dim, hidden_dim=4)
    # A known zero old value makes independent return checks simple. The
    # critic's output bias can still receive a meaningful nonzero gradient.
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    return actor, critic


def training_batch_fixture(actor, critic):
    return prepare_batch([[plan_fixture(actor, critic, index, rewards=(reward,))]
                          for index, reward in enumerate((.1, .6, .9))])


def parameters_snapshot(module):
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


@unittest.skipIf(torch is None, "Installed torch and diffusers are required")
class PolicyTrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tdyssey-toy-update-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def assert_parameters_equal(self, module, expected, *, rtol=0, atol=0):
        self.assertEqual(set(dict(module.named_parameters())), set(expected))
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=rtol, atol=atol, msg=name)

    def test_prepare_uses_raw_gae_for_targets_and_normalizes_once_across_fragments(self):
        actor, critic = model_fixture()
        first = plan_fixture(actor, critic, 0, rewards=(1., 2.), value=.2, next_value=.7,
                             boundary=BlockBoundary.CONTINUE, episode="T-one")
        timeout = plan_fixture(actor, critic, 1, rewards=(-.5,), value=.7, next_value=3.,
                               boundary=BlockBoundary.TIMEOUT, episode="T-one")
        success = plan_fixture(actor, critic, 2, rewards=(2., 1., 3.), value=-.3,
                               boundary=BlockBoundary.SUCCESS, episode="L-two")
        # prepare_batch must detach, even if a caller accidentally retained a
        # feature graph. Critic targets always use the fixed rollout values.
        first = replace(first, features=first.features.clone().requires_grad_(True))
        batch = prepare_batch([[first, timeout], [success]], gamma=.9, gae_lambda=.8)
        raw = torch.tensor([4.139, 1.5, 5.63])
        torch.testing.assert_close(batch.raw_advantages, raw, rtol=0, atol=1e-6)
        torch.testing.assert_close(batch.old_values, torch.tensor([.2, .7, -.3]), rtol=0, atol=1e-7)
        torch.testing.assert_close(batch.returns, torch.tensor([4.339, 2.2, 5.33]), rtol=0, atol=1e-6)
        mean = math.fsum((4.139, 1.5, 5.63)) / 3
        sigma = math.sqrt(math.fsum((value - mean) ** 2 for value in (4.139, 1.5, 5.63)) / 3)
        expected = torch.tensor([(value - mean) / sigma for value in (4.139, 1.5, 5.63)])
        torch.testing.assert_close(batch.advantages, expected, rtol=0, atol=1e-6)
        self.assertEqual(len(batch.plans), 3)
        self.assertEqual(batch.features.shape[0], 3)
        for tensor in (batch.features, batch.old_values, batch.raw_advantages,
                       batch.advantages, batch.returns):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        self.assertFalse(torch.equal(batch.returns, batch.advantages + batch.old_values))

    def test_default_optimizers_have_explicit_rates_and_no_weight_decay(self):
        actor, critic = model_fixture()
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic, actor_lr=1e-5, critic_lr=3e-4)
        self.assertIsInstance(actor_optimizer, torch.optim.Adam)
        self.assertIsInstance(critic_optimizer, torch.optim.Adam)
        for optimizer, rate in ((actor_optimizer, 1e-5), (critic_optimizer, 3e-4)):
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], rate)
                self.assertEqual(group["weight_decay"], 0)
        expected = {id(parameter) for parameter in actor.parameters() if parameter.requires_grad}
        actual = {id(parameter) for group in actor_optimizer.param_groups for parameter in group["params"]}
        self.assertEqual(actual, expected)

    def test_toy_update_uses_all_transitions_preserves_old_data_and_moves_both_models(self):
        actor, critic = model_fixture()
        batch = training_batch_fixture(actor, critic)
        actor_before, critic_before = parameters_snapshot(actor), parameters_snapshot(critic)
        fixed_batch = {key: getattr(batch, key).clone() for key in
                       ("features", "old_values", "raw_advantages", "advantages", "returns")}
        fixed_records = [(plan.sample.chain.clone(), plan.sample.old_log_prob.clone(), plan.features.clone())
                         for plan in batch.plans]
        actor_optimizer = torch.optim.SGD((p for p in actor.parameters() if p.requires_grad), lr=1e-3)
        critic_optimizer = torch.optim.SGD(critic.parameters(), lr=1e-3)
        grad_counts = Counter()
        original = actor.recompute_log_probs

        def observed(record, **kwargs):
            if torch.is_grad_enabled():
                indices = kwargs.get("step_indices")
                indices = range(100) if indices is None else indices
                for step in indices:
                    grad_counts[step] += record.chain.shape[0]
            return original(record, **kwargs)

        with torch.no_grad():
            critic_loss_before = (critic(batch.features).reshape(-1) - batch.returns).square().mean().item()
        with HardBudget(self.root / "update.json") as budget:
            with patch.object(actor, "recompute_log_probs", side_effect=observed):
                result = update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget,
                                     config=UpdateConfig(plan_microbatch=2, step_microbatch=3))
            counts = budget.snapshot()
        self.assertEqual(grad_counts, Counter({step: 3 for step in range(100)}))
        self.assertEqual(result["actor_optimizer_steps"], 1)
        self.assertEqual(result["critic_optimizer_steps"], 1)
        self.assertTrue(math.isfinite(result["actor_grad_norm"]) and result["actor_grad_norm"] > 0)
        self.assertTrue(math.isfinite(result["critic_grad_norm"]) and result["critic_grad_norm"] > 0)
        self.assertEqual(counts["completed"]["actor_step"], 1)
        self.assertEqual(counts["completed"]["critic_step"], 1)
        self.assertEqual(counts["reserved"]["control"], 0)  # No physical interaction in this fixture.
        self.assertTrue(any(not torch.equal(parameter, actor_before[name])
                            for name, parameter in actor.named_parameters() if parameter.requires_grad))
        self.assertTrue(any(not torch.equal(parameter, critic_before[name]) for name, parameter in critic.named_parameters()))
        for name, parameter in actor.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, actor_before[name], rtol=0, atol=0)
        for key, expected in fixed_batch.items():
            torch.testing.assert_close(getattr(batch, key), expected, rtol=0, atol=0)
        for plan, (chain, old_log_prob, features) in zip(batch.plans, fixed_records):
            torch.testing.assert_close(plan.sample.chain, chain, rtol=0, atol=0)
            torch.testing.assert_close(plan.sample.old_log_prob, old_log_prob, rtol=0, atol=0)
            torch.testing.assert_close(plan.features, features, rtol=0, atol=0)
        with torch.no_grad():
            after = torch.cat([actor.recompute_log_probs(plan.sample) for plan in batch.plans])
            old = torch.cat([plan.sample.old_log_prob for plan in batch.plans])
            after_loss = transition_ppo_loss(after, old, batch.advantages)
            before_loss = transition_ppo_loss(old, old, batch.advantages)
            self.assertLess(after_loss.item(), before_loss.item())
            self.assertLess((critic(batch.features).reshape(-1) - batch.returns).square().mean().item(), critic_loss_before)

    def test_unequal_microbatches_match_independent_dense_toy_optimizer_update(self):
        actor, critic = model_fixture()
        batch = training_batch_fixture(actor, critic)
        other_actor, other_critic = deepcopy(actor), deepcopy(critic)
        selected = [p for p in actor.parameters() if p.requires_grad]
        actor_optimizer = torch.optim.SGD(selected, lr=1e-3)
        critic_optimizer = torch.optim.SGD(critic.parameters(), lr=1e-3)
        # The independent dense oracle is safe only for this dozen-weight toy.
        # It does not request an unsupported large real-network microbatch.
        new = torch.cat([actor.recompute_log_probs(plan.sample) for plan in batch.plans])
        old = torch.cat([plan.sample.old_log_prob for plan in batch.plans])
        transition_ppo_loss(new, old, batch.advantages).backward()
        (critic(batch.features) - batch.returns).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(selected, 1.)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        with HardBudget(self.root / "dense.json") as budget:
            actor_reservation, critic_reservation = budget.reserve("actor_step"), budget.reserve("critic_step")
            actor_optimizer.step()
            budget.mark_completed(actor_reservation)
            critic_optimizer.step()
            budget.mark_completed(critic_reservation)
        other_actor_optimizer = torch.optim.SGD((p for p in other_actor.parameters() if p.requires_grad), lr=1e-3)
        other_critic_optimizer = torch.optim.SGD(other_critic.parameters(), lr=1e-3)
        with HardBudget(self.root / "split.json") as budget:
            update_once(other_actor, other_critic, batch, other_actor_optimizer, other_critic_optimizer,
                        budget=budget, config=UpdateConfig(plan_microbatch=2, step_microbatch=3))
        self.assert_parameters_equal(other_actor, parameters_snapshot(actor), rtol=2e-5, atol=2e-7)
        self.assert_parameters_equal(other_critic, parameters_snapshot(critic), rtol=2e-5, atol=2e-7)

    def test_zero_actor_advantage_is_noop_and_budget_prevents_second_update(self):
        actor, critic = model_fixture()
        plan = plan_fixture(actor, critic, 0, rewards=(.6,))
        batch = prepare_batch([[plan]])
        torch.testing.assert_close(batch.advantages, torch.zeros(1), rtol=0, atol=0)
        before_actor, before_critic = parameters_snapshot(actor), parameters_snapshot(critic)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic)
        with HardBudget(self.root / "noop.json") as budget:
            result = update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget)
            self.assertEqual(result["actor_grad_norm"], 0)
            self.assert_parameters_equal(actor, before_actor)
            self.assertTrue(any(not torch.equal(parameter, before_critic[name]) for name, parameter in critic.named_parameters()))
            after_critic = parameters_snapshot(critic)
            # Fresh numerical batch, unchanged actor, same consumed ledger:
            # a second optimizer operation must be refused by the hard cap.
            with torch.no_grad():
                current_value = float(critic(plan.features).item())
            fresh_plan = replace(plan, block=replace(plan.block, value=current_value))
            fresh = prepare_batch([[fresh_plan]])
            with self.assertRaises(BudgetExceeded):
                update_once(actor, critic, fresh, actor_optimizer, critic_optimizer, budget=budget)
            self.assert_parameters_equal(actor, before_actor)
            self.assert_parameters_equal(critic, after_critic)
            self.assertEqual(budget.snapshot()["completed"]["actor_step"], 1)
            self.assertEqual(budget.snapshot()["completed"]["critic_step"], 1)

    def test_changed_actor_rejects_stale_samples_without_optimizer_steps(self):
        actor, critic = model_fixture()
        batch = training_batch_fixture(actor, critic)
        with torch.no_grad():
            actor.network.model.final_conv.bias.add_(.5)
        before_actor, before_critic = parameters_snapshot(actor), parameters_snapshot(critic)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic)
        with HardBudget(self.root / "stale.json") as budget:
            with patch.object(actor_optimizer, "step", wraps=actor_optimizer.step) as actor_step:
                with patch.object(critic_optimizer, "step", wraps=critic_optimizer.step) as critic_step:
                    with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)(stale|probab|likelihood|ratio)"):
                        update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget)
            actor_step.assert_not_called()
            critic_step.assert_not_called()
            self.assertEqual(budget.snapshot()["completed"]["actor_step"], 0)
            self.assertEqual(budget.snapshot()["completed"]["critic_step"], 0)
        self.assert_parameters_equal(actor, before_actor)
        self.assert_parameters_equal(critic, before_critic)

    def test_changed_critic_rejects_stale_old_values_without_optimizer_steps(self):
        actor, critic = model_fixture()
        batch = training_batch_fixture(actor, critic)
        with torch.no_grad():
            critic.head[-1].bias.add_(.1)
        before_actor, before_critic = parameters_snapshot(actor), parameters_snapshot(critic)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic)
        with HardBudget(self.root / "stale-critic.json") as budget:
            with patch.object(actor_optimizer, "step", wraps=actor_optimizer.step) as actor_step:
                with patch.object(critic_optimizer, "step", wraps=critic_optimizer.step) as critic_step:
                    with self.assertRaisesRegex(ValueError, "Critic changed"):
                        update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget)
            actor_step.assert_not_called()
            critic_step.assert_not_called()
        self.assert_parameters_equal(actor, before_actor)
        self.assert_parameters_equal(critic, before_critic)

    def test_second_optimizer_budget_preflight_failure_cannot_partially_update_actor(self):
        actor, critic = model_fixture()
        batch = training_batch_fixture(actor, critic)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic)
        before_actor, before_critic = parameters_snapshot(actor), parameters_snapshot(critic)
        with HardBudget(self.root / "critic-exhausted.json") as budget:
            used = budget.reserve("critic_step")
            budget.mark_completed(used)
            with patch.object(actor_optimizer, "step", wraps=actor_optimizer.step) as actor_step:
                with patch.object(critic_optimizer, "step", wraps=critic_optimizer.step) as critic_step:
                    with self.assertRaises(BudgetExceeded):
                        update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget)
            actor_step.assert_not_called()
            critic_step.assert_not_called()
            self.assertEqual(budget.snapshot()["completed"]["actor_step"], 0)
        self.assert_parameters_equal(actor, before_actor)
        self.assert_parameters_equal(critic, before_critic)


if __name__ == "__main__":
    unittest.main()
