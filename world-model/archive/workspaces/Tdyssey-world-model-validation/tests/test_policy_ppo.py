"""C PPO protocol on controlled CPU toy networks; no physical interactions."""

from dataclasses import replace
import gc
import json
import unittest
import weakref
from unittest.mock import patch

from tests.test_policy_rl_runtime import torch, actor_fixture, observation_fixture

if torch is not None:
    from tdyssey.policy_ppo import MultiPassConfig, PPOUpdater, explained_variance
    from tdyssey.policy_training import UpdateConfig, VisualCritic, make_optimizers
    from tests.test_policy_training import plan_fixture


class ToyBudget:
    def __init__(self):
        self.reserved, self.completed = {}, {}
        self.events = []

    def reserve(self, kind, *, phase="train", **kwargs):
        self.reserved[kind] = self.reserved.get(kind, 0) + 1
        self.events.append(kind)
        return len(self.events)

    def mark_completed(self, index):
        kind = self.events[index - 1]
        self.completed[kind] = self.completed.get(kind, 0) + 1

    def check_time(self):
        return 100.0


def setup_fixture(*, cached=True, plans=2, steps=3, target_kl=100):
    actor = actor_fixture()
    critic = VisualCritic(actor.encode(observation_fixture(1)).shape[1], hidden_dim=4)
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()
    optimizers = make_optimizers(actor, critic, actor_lr=1e-4, critic_lr=1e-4)
    budget = ToyBudget()
    config = MultiPassConfig(target_kl=target_kl, cache_features=cached,
                             update=UpdateConfig(plan_microbatch=plans, step_microbatch=steps))
    return PPOUpdater(actor, critic, *optimizers, budget=budget, run_id="toy-C", config=config), budget


def fresh_fragments(updater):
    fragments = []
    for index, reward in enumerate((.1, .6, .9)):
        plan = plan_fixture(updater.actor, updater.critic, index, rewards=(reward,))
        plan = replace(plan, sample=replace(plan.sample, actor_version=updater.actor_version))
        fragments.append([plan])
    return fragments


@unittest.skipIf(torch is None, "Installed torch/diffusers required for toy PPO checks")
class MultiPassChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_three_passes_keep_behavior_targets_and_check_entry_once(self):
        updater, budget = setup_fixture()
        fragments = fresh_fragments(updater)
        original_log = [f[0].sample.old_log_prob.clone() for f in fragments]
        buffer = updater.prepare(fragments)
        old = buffer.prepared.old_values.clone()
        advantages, targets = buffer.prepared.advantages.clone(), buffer.prepared.returns.clone()
        result = updater.update(buffer, passes=3)
        self.assertEqual(result["completed_passes"], 3)
        self.assertEqual(updater.entry_checks, 1)
        self.assertEqual(budget.completed, {"batch": 1, "actor_step": 3, "critic_step": 3})
        self.assertEqual(result["entry"]["max_logprob_error"], 0)
        self.assertGreater(result["passes"][-1]["after"]["max_logprob_error"], 0)
        self.assertEqual(result["data_statistics"]["returns"]["count"], 3)
        self.assertAlmostEqual(result["data_statistics"]["advantages"]["mean"], 0, places=6)
        self.assertAlmostEqual(result["data_statistics"]["advantages"]["std"], 1, places=6)
        self.assertGreater(result["passes"][0]["actor_weight_change"]["l2"], 0)
        self.assertGreater(result["passes"][0]["critic_weight_change"]["l2"], 0)
        for actual, expected in ((buffer.prepared.old_values, old), (buffer.prepared.advantages, advantages),
                                  (buffer.prepared.returns, targets)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for fragment, expected in zip(fragments, original_log):
            torch.testing.assert_close(fragment[0].sample.old_log_prob, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Consumed"):
            updater.update(buffer)
        with self.assertRaisesRegex(ValueError, "actor version"):
            updater.prepare(fragments)
        second = updater.prepare(fresh_fragments(updater))
        updater.update(second, passes=1)
        self.assertEqual(updater.entry_checks, 2)
        self.assertEqual(updater.actor_steps, 4)

    def test_cached_and_observation_paths_have_equivalent_probabilities_and_gradients(self):
        actor = actor_fixture(subset="last_up_and_output")
        observations = observation_fixture(3)
        record = actor.sample(observations, generator=torch.Generator().manual_seed(20))
        steps = [9, 49, 99]
        reference = actor.recompute_log_probs(record, step_indices=steps)
        reference.sum().backward()
        gradients = {n: p.grad.clone() for n, p in actor.network.named_parameters() if p.requires_grad}
        actor.zero_grad(set_to_none=True)
        features = actor.encode(observations)
        cached = torch.stack([actor.transition_log_prob_from_features(features, record.chain[:, i],
                              record.chain[:, i + 1], int(record.timesteps[i])) for i in steps], dim=1)
        torch.testing.assert_close(cached, reference.detach(), rtol=0, atol=0)
        cached.sum().backward()
        for name, parameter in actor.network.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    def test_unequal_microbatches_match_dense_full_objective_and_critic_step(self):
        updater, _ = setup_fixture(plans=2, steps=3)
        fragments = fresh_fragments(updater)
        # Independent tiny dense graph, with no real policy or large actor.
        reference, _ = setup_fixture(plans=3, steps=4)
        reference.actor.load_state_dict(updater.actor.state_dict())
        reference.critic.load_state_dict(updater.critic.state_dict())
        lr = 1e-3
        updater.actor_optimizer = torch.optim.SGD([p for p in updater.actor.parameters() if p.requires_grad], lr=lr)
        updater.critic_optimizer = torch.optim.SGD(updater.critic.parameters(), lr=lr)
        updater.config = replace(updater.config, update=replace(updater.config.update, max_grad_norm=1e6))
        buffer = updater.prepare(fragments)
        batch = buffer.prepared
        from tdyssey.policy_training import _pack_samples
        from tdyssey.policy_rl import transition_ppo_loss
        packed = _pack_samples(batch.plans, "cpu")
        new = reference.actor.recompute_log_probs(packed)
        loss = transition_ppo_loss(new, packed.old_log_prob, batch.advantages)
        loss.backward()
        with torch.no_grad():
            for p in reference.actor.parameters():
                if p.requires_grad:
                    p.add_(p.grad, alpha=-lr)
        prediction = reference.critic(batch.features)
        (prediction - batch.returns).square().mean().backward()
        with torch.no_grad():
            for p in reference.critic.parameters():
                p.add_(p.grad, alpha=-lr)
        updater.update(buffer, passes=1)
        for p, expected in zip(updater.actor.parameters(), reference.actor.parameters()):
            torch.testing.assert_close(p, expected, rtol=1e-5, atol=2e-7)
        for p, expected in zip(updater.critic.parameters(), reference.critic.parameters()):
            torch.testing.assert_close(p, expected, rtol=1e-5, atol=2e-7)

    def test_kl_stop_does_not_reserve_or_step_another_pass(self):
        updater, budget = setup_fixture(target_kl=1e-16)
        buffer = updater.prepare(fresh_fragments(updater))
        result = updater.update(buffer, passes=3)
        self.assertEqual((result["stop_reason"], result["completed_passes"]), ("target_kl", 1))
        self.assertEqual(budget.completed["actor_step"], 1)
        self.assertEqual(budget.completed["critic_step"], 1)

    def test_stale_probabilities_bad_cache_and_mutated_targets_hard_stop(self):
        updater, budget = setup_fixture()
        fragments = fresh_fragments(updater)
        with torch.no_grad():
            updater.actor.network.model.final_conv.bias.add_(.01)
        with self.assertRaisesRegex(ValueError, "probabilities differ"):
            updater.prepare(fragments)
        self.assertNotIn("actor_step", budget.reserved)
        updater, _ = setup_fixture()
        fragments = fresh_fragments(updater)
        fragments[0][0].features.add_(1)
        with self.assertRaisesRegex(ValueError, "Cached features"):
            updater.prepare(fragments)
        updater, budget = setup_fixture()
        buffer = updater.prepare(fresh_fragments(updater))
        buffer.prepared.returns.add_(1)
        with self.assertRaisesRegex(ValueError, "Fixed behavior"):
            updater.update(buffer)
        self.assertTrue(buffer.consumed)
        self.assertNotIn("actor_step", budget.reserved)

    def test_consumed_buffers_and_graphs_are_not_retained_across_batches(self):
        updater, _ = setup_fixture()
        for _ in range(3):
            fragments = fresh_fragments(updater)
            buffer = updater.prepare(fragments)
            chain_ref = weakref.ref(buffer.prepared.plans[0].sample.chain)
            buffer_ref = weakref.ref(buffer)
            updater.update(buffer, passes=1)
            del buffer, fragments
            gc.collect()
            self.assertIsNone(buffer_ref())
            self.assertIsNone(chain_ref())
            self.assertIsNone(updater._active)
        self.assertEqual(updater.batches, 3)

    def test_explained_variance_has_explicit_zero_variance_case(self):
        self.assertIsNone(explained_variance(torch.ones(3), torch.zeros(3)))
        self.assertAlmostEqual(explained_variance(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 3.])), 1)

    def test_nonfinite_records_and_frozen_weight_changes_hard_stop_before_steps(self):
        updater, budget = setup_fixture()
        fragments = fresh_fragments(updater)
        fragments[0][0].sample.old_log_prob[0, 5] = float("nan")
        with self.assertRaises(ValueError):
            updater.prepare(fragments)
        self.assertNotIn("actor_step", budget.reserved)
        updater, budget = setup_fixture()
        fragments = fresh_fragments(updater)
        with torch.no_grad():
            updater.actor.network.model.stem.weight.add_(.1)
        with self.assertRaisesRegex(ValueError, "Frozen actor"):
            updater.prepare(fragments)
        self.assertNotIn("actor_step", budget.reserved)

    def test_cached_behavior_records_cannot_bypass_sampler_or_timestep_identity(self):
        from tdyssey.policy_rl import SamplerConfig
        for cached in (True, False):
            for mismatch in ("floor", "scheduler", "timesteps", "zero_std"):
                with self.subTest(cached=cached, mismatch=mismatch):
                    updater, budget = setup_fixture(cached=cached)
                    fragments = fresh_fragments(updater)
                    for fragment in fragments:
                        plan = fragment[0]
                        sample = plan.sample
                        if mismatch == "floor":
                            sample = replace(sample, sampler_config=SamplerConfig(std_floor=.02))
                        elif mismatch == "scheduler":
                            signature = json.loads(sample.scheduler_signature)
                            signature["beta_end"] = .021
                            sample = replace(sample, scheduler_signature=json.dumps(signature, sort_keys=True))
                        elif mismatch == "timesteps":
                            sample = replace(sample, timesteps=sample.timesteps.flip(0))
                        else:
                            sample = replace(sample, old_stds=torch.zeros_like(sample.old_stds))
                        fragment[0] = replace(plan, sample=sample)
                    if mismatch in ("floor", "scheduler"):
                        with self.assertRaisesRegex(ValueError, "sampler configuration"):
                            updater.prepare(fragments)
                    else:
                        # The shared payload validator may reject these before
                        # the PPO-specific contract check is reached.
                        with self.assertRaises(ValueError):
                            updater.prepare(fragments)
                    self.assertNotIn("actor_step", budget.reserved)
                    self.assertNotIn("critic_step", budget.reserved)
                    self.assertNotIn("batch", budget.completed)
                    self.assertEqual((updater.batches, updater.entry_checks), (0, 0))

    def test_pass_preflight_rejects_late_frozen_whitelist_and_optimizer_changes(self):
        for mutation in ("frozen", "whitelist", "optimizer"):
            with self.subTest(mutation=mutation):
                updater, budget = setup_fixture()
                buffer = updater.prepare(fresh_fragments(updater))
                if mutation == "frozen":
                    with torch.no_grad():
                        updater.actor.network.model.stem.weight.add_(.1)
                elif mutation == "whitelist":
                    updater.actor.network.model.final_conv.bias.requires_grad_(False)
                else:
                    updater.actor_optimizer.param_groups[0]["params"].pop()
                with patch.object(updater.actor_optimizer, "step") as actor_step:
                    with patch.object(updater.critic_optimizer, "step") as critic_step:
                        with self.assertRaisesRegex(ValueError, "Frozen actor|whitelist|Optimizer parameters"):
                            updater.update(buffer, passes=1)
                actor_step.assert_not_called()
                critic_step.assert_not_called()
                self.assertNotIn("actor_step", budget.reserved)
                self.assertTrue(buffer.consumed)
                self.assertIsNone(updater._active)

    def test_every_pass_revalidates_without_requiring_initial_ratios_again(self):
        updater, _ = setup_fixture()
        buffer = updater.prepare(fresh_fragments(updater))
        with patch.object(updater, "_validate_optimizers", wraps=updater._validate_optimizers) as validate:
            result = updater.update(buffer, passes=3)
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(result["completed_passes"], 3)
        self.assertEqual(updater.entry_checks, 1)
        self.assertGreater(result["passes"][-1]["after"]["max_logprob_error"], 0)

    def test_state_restore_validates_complete_counter_relations_atomically(self):
        updater, _ = setup_fixture()
        original = updater.state_dict()
        candidates = [
            {**original, "actor_steps": 19, "critic_steps": "invalid"},
            {**original, "actor_steps": 17, "critic_steps": 2},
            {**original, "actor_steps": 1, "critic_steps": 1},
            {**original, "batches": 1, "entry_checks": 1, "last_consumed": 0},
            {**original, "actor_steps": 4, "critic_steps": 4, "batches": 1,
             "entry_checks": 1, "last_consumed": 0},
            {**original, "entry_checks": 1},
            {**original, "last_consumed": "corrupt"},
            {**original, "last_consumed": False},
            {**original, "actor_steps": True},
            {**original, "schema_version": True},
            {**original, "unexpected": 1},
            {name: value for name, value in original.items() if name != "last_consumed"},
        ]
        for state in candidates:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    updater.load_state_dict(state)
                self.assertEqual(updater.state_dict(), original)
        # Failed validation preserves the ability to restore one valid boundary.
        restored = {**original, "actor_steps": 3, "critic_steps": 3, "batches": 1,
                    "entry_checks": 1, "last_consumed": 0}
        updater.load_state_dict(restored)
        self.assertEqual(updater.state_dict(), restored)
        self.assertEqual(updater.actor_version, "toy-C:actor:3")
        with self.assertRaises(ValueError):
            updater.load_state_dict(restored)

    def test_initial_state_can_restore_once_and_prepare_attempt_ends_freshness(self):
        updater, _ = setup_fixture()
        original = updater.state_dict()
        updater.load_state_dict(original)
        with self.assertRaises(ValueError):
            updater.load_state_dict(original)
        updater, _ = setup_fixture()
        original = updater.state_dict()
        fragments = fresh_fragments(updater)
        bad = fragments[0][0]
        fragments[0][0] = replace(bad, sample=replace(bad.sample, actor_version="foreign:actor:0"))
        with self.assertRaisesRegex(ValueError, "actor version"):
            updater.prepare(fragments)
        with self.assertRaises(ValueError):
            updater.load_state_dict(original)

    def test_wrong_buffer_sequence_is_rejected_before_any_optimizer_reservation(self):
        updater, budget = setup_fixture()
        buffer = updater.prepare(fresh_fragments(updater))
        buffer.sequence += 10
        with self.assertRaisesRegex(ValueError, "Consumed, foreign or wrong-version"):
            updater.update(buffer)
        self.assertNotIn("actor_step", budget.reserved)
        self.assertNotIn("critic_step", budget.reserved)


if __name__ == "__main__":
    unittest.main()
