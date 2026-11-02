"""CPU likelihood/gradient oracles; tiny fixtures never load a research checkpoint."""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except ImportError:
    torch = None

if torch is not None:
    from tdyssey.official import official_spec
    from tdyssey.policy_rl import (
        DenoisingPolicy, SamplerConfig, ddpm_transition, gaussian_log_prob,
        select_trainable_parameters, transition_ppo_loss,
    )


def scheduler_fixture():
    config = dict(official_spec()["policy_config"]["noise_scheduler"])
    config.pop("_target_")
    scheduler = DDPMScheduler(**config)
    scheduler.set_timesteps(100)
    return scheduler


def actor_fixture(*, subset="output", floor=.01, zero_final_mean=False):
    """Expose the official interfaces and block names with only dozens of weights."""
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, observations):
            image = observations["image"]
            # Deliberately stochastic in train mode so an accidental train()
            # propagation would be observable, as it is with upstream cropping.
            start = int(torch.randint(0, 12, ()).item()) if self.training else 6
            crop = image[..., start:start + 84, start:start + 84]
            self.crop_shape = crop.shape[-2:]
            return torch.cat((crop.mean(dim=(-2, -1)), observations["agent_pos"]), dim=1) * self.scale

    class ActionNormalizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(100.))

        def unnormalize(self, actions):
            return actions * self.scale + 256.

    class Normalizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.action = ActionNormalizer()

        def normalize(self, observations):
            return {"image": observations["image"] * 2 - 1,
                    "agent_pos": observations["agent_pos"] / 512}

        def __getitem__(self, key):
            if key != "action":
                raise KeyError(key)
            return self.action

    scheduler = scheduler_fixture()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(2, 2)
            self.up_modules = nn.ModuleList((nn.Linear(2, 2), nn.Linear(2, 2)))
            self.final_conv = nn.Linear(2, 2)
            with torch.no_grad():
                for layer in (self.stem, *self.up_modules, self.final_conv):
                    layer.weight.copy_(torch.eye(2) * .1)
                    layer.bias.zero_()

        def forward(self, x, timestep, *, local_cond=None, global_cond=None):
            if zero_final_mean and int(timestep) == 0:
                # Independently force DDPM's final posterior mean to zero;
                # use this to test the sampler's actual Gaussian residuals.
                return x / (1 - scheduler.alphas_cumprod[0]).sqrt()
            hidden = self.stem(x)
            for layer in self.up_modules:
                hidden = layer(hidden)
            hidden = hidden + global_cond.mean(dim=1)[:, None, None] * .01
            return self.final_conv(hidden)

    class Network(nn.Module):
        obs_as_global_cond = True
        horizon, action_dim = 16, 2
        n_obs_steps, n_action_steps, num_inference_steps = 2, 8, 100
        dtype, device = torch.float32, torch.device("cpu")

        def __init__(self):
            super().__init__()
            self.model, self.obs_encoder = Model(), Encoder()
            self.normalizer, self.noise_scheduler = Normalizer(), scheduler

    return DenoisingPolicy(SimpleNamespace(network=Network()), subset=subset,
                           config=SamplerConfig(std_floor=floor))


def observation_fixture(batch=3):
    # A spatial ramp makes fixed center cropping distinguishable from random
    # cropping while avoiding any simulator or image-codec dependency.
    image = torch.linspace(0, 1, 96).reshape(1, 1, 1, 1, 96)
    return {"image": image.expand(batch, 2, 3, 96, 96).clone(),
            "agent_pos": torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 2, 2)}


@unittest.skipIf(torch is None, "Installed torch and diffusers are required")
class PolicyRLLikelihoodChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_joint_gaussian_density_matches_independent_normal_and_hand_value(self):
        x = torch.linspace(-.3, .3, 64, dtype=torch.float64).reshape(2, 16, 2)
        mean = torch.full_like(x, .1)
        std = torch.tensor([.01, .5], dtype=x.dtype)[:, None, None]
        expected = torch.distributions.Normal(mean, std).log_prob(x).sum(dim=(-2, -1))
        actual = gaussian_log_prob(x, mean, std)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-11)
        single = gaussian_log_prob(torch.zeros(1, 16, 2, dtype=x.dtype),
                                   torch.full((1, 16, 2), .1, dtype=x.dtype),
                                   torch.tensor(.5, dtype=x.dtype))
        manual = 32 * (-.5 * (.1 / .5) ** 2 - math.log(.5) - .5 * math.log(2 * math.pi))
        self.assertAlmostEqual(single.item(), manual, places=12)
        with self.assertRaisesRegex(ValueError, "positive std"):
            gaussian_log_prob(x, mean, torch.tensor(0.))
        for endpoints, centers in ((torch.full_like(x, float("nan")), mean),
                                   (x, torch.full_like(mean, float("inf")))):
            with self.assertRaisesRegex(ValueError, "finite"):
                gaussian_log_prob(endpoints, centers, torch.tensor(.5))

    def test_every_ddpm_mean_and_variance_match_real_locked_scheduler(self):
        scheduler = scheduler_fixture()
        x = torch.linspace(-2.5, 2.5, 64).reshape(2, 16, 2)
        epsilon = torch.cos(torch.arange(64, dtype=x.dtype)).reshape_as(x)
        for t in range(99, -1, -1):
            with self.subTest(t=t):
                # The installed scheduler is the independent oracle, with its
                # random draw zeroed to expose its actual posterior mean.
                with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor",
                           return_value=torch.zeros_like(x)) as draw:
                    expected = scheduler.step(epsilon, t, x).prev_sample
                mean, std = ddpm_transition(scheduler, epsilon, t, x)
                torch.testing.assert_close(mean, expected, rtol=0, atol=0)
                expected_std = scheduler._get_variance(t).sqrt() if t else torch.tensor(0.)
                torch.testing.assert_close(std, expected_std, rtol=0, atol=0)
                training_mean, training_std = ddpm_transition(scheduler, epsilon, t, x, std_floor=.05)
                torch.testing.assert_close(training_mean, expected, rtol=0, atol=0)
                torch.testing.assert_close(training_std,
                                           scheduler._get_variance(t).sqrt().clamp_min(.05), rtol=0, atol=0)
                self.assertEqual(draw.call_count, int(t > 0))
        generator = torch.Generator().manual_seed(21)
        state = generator.get_state().clone()
        scheduler.step(epsilon, 0, x, generator=generator)
        torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)

    def test_baseline_complete_chain_actions_and_next_plan_rng_match_scheduler(self):
        actor, observations = actor_fixture(), observation_fixture(2)
        adapted_rng, direct_rng = (torch.Generator().manual_seed(317) for _ in range(2))

        def direct_sample():
            with torch.no_grad():
                condition = actor.encode(observations)
                n = actor.network
                n.noise_scheduler.set_timesteps(100)
                x = torch.randn((2, 16, 2), generator=direct_rng)
                chain = [x.clone()]
                for t in n.noise_scheduler.timesteps:
                    epsilon = n.model(x, t, local_cond=None, global_cond=condition)
                    x = n.noise_scheduler.step(epsilon, t, x, generator=direct_rng).prev_sample
                    chain.append(x.clone())
                return torch.stack(chain, dim=1), n.normalizer["action"].unnormalize(x)

        for _ in range(2):
            record = actor.sample(observations, generator=adapted_rng, mode="baseline")
            direct_chain, direct_targets = direct_sample()
            torch.testing.assert_close(record.chain, direct_chain, rtol=0, atol=0)
            torch.testing.assert_close(record.targets, direct_targets, rtol=0, atol=0)
            torch.testing.assert_close(record.actions, direct_targets[:, 1:9], rtol=0, atol=0)
            torch.testing.assert_close(adapted_rng.get_state(), direct_rng.get_state(), rtol=0, atol=0)
            self.assertIsNone(record.old_log_prob)
            self.assertEqual(record.old_stds[-1].item(), 0)
        with self.assertRaisesRegex(ValueError, "training sampler"):
            actor.recompute_log_probs(record, step_indices=[99])

    def test_training_record_density_recomputes_before_any_update(self):
        actor, observations = actor_fixture(), observation_fixture(3)
        record = actor.sample(observations, generator=torch.Generator().manual_seed(11), actor_version="fixture-v1")
        self.assertEqual(record.chain.shape, (3, 101, 16, 2))
        self.assertEqual(record.timesteps.tolist(), list(range(99, -1, -1)))
        self.assertEqual(record.actor_version, "fixture-v1")
        independent = torch.distributions.Normal(record.old_means, record.old_stds[None, :, None, None])
        expected = independent.log_prob(record.chain[:, 1:]).sum(dim=(-2, -1))
        torch.testing.assert_close(record.old_log_prob, expected, rtol=0, atol=1e-4)
        recalculated = actor.recompute_log_probs(record)
        # Both paths use the same FP32 arithmetic/device. 1e-4 is a joint
        # 32-coordinate log-density tolerance, not a density-ratio clip.
        torch.testing.assert_close(recalculated, record.old_log_prob, rtol=0, atol=1e-4)
        torch.testing.assert_close((recalculated - record.old_log_prob).exp(),
                                   torch.ones_like(recalculated), rtol=0, atol=1e-4)
        for tensor in (record.chain, record.old_means, record.old_stds, record.old_log_prob,
                       record.targets, record.actions, *record.observations.values()):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        observations["agent_pos"].add_(100)
        self.assertFalse(torch.equal(observations["agent_pos"], record.observations["agent_pos"]))

    def test_legacy_scheduler_bookkeeping_order_preserves_density_but_not_numeric_changes(self):
        actor = actor_fixture()
        record = actor.sample(observation_fixture(2), generator=torch.Generator().manual_seed(412))
        legacy_config = json.loads(record.scheduler_signature)
        defaults = legacy_config["_use_default_values"]
        self.assertGreater(len(defaults), 1)
        legacy_config["_use_default_values"] = list(reversed(defaults))
        legacy_signature = json.dumps(legacy_config, sort_keys=True)
        self.assertNotEqual(legacy_signature, record.scheduler_signature)
        legacy = replace(record, scheduler_signature=legacy_signature)
        new = actor.recompute_log_probs(legacy)
        torch.testing.assert_close(new, record.old_log_prob, rtol=0, atol=1e-4)
        torch.testing.assert_close((new - record.old_log_prob).exp(), torch.ones_like(new), rtol=0, atol=1e-4)

        # Canonicalization must retain every numeric setting. A changed legacy
        # record and a changed live scheduler are independently unacceptable.
        altered = dict(legacy_config, beta_end=.021)
        with self.assertRaisesRegex(ValueError, "training sampler configuration"):
            actor.recompute_log_probs(replace(record, scheduler_signature=json.dumps(altered, sort_keys=True)))
        config = dict(official_spec()["policy_config"]["noise_scheduler"])
        config.pop("_target_")
        actor.network.noise_scheduler = DDPMScheduler(**dict(config, beta_end=.021))
        with self.assertRaisesRegex(ValueError, "beta_end"):
            actor.recompute_log_probs(legacy, step_indices=[0])

    def test_scheduler_signature_is_stable_across_python_hash_seeds(self):
        # Two fresh scheduler-only processes reproduce the actual source of
        # nondeterminism without allocating an actor/checkpoint or simulator.
        script = "\n".join((
            "from diffusers.schedulers.scheduling_ddpm import DDPMScheduler",
            "from tdyssey.official import official_spec",
            "from tdyssey.policy_rl import _scheduler_signature",
            "config = dict(official_spec()['policy_config']['noise_scheduler'])",
            "config.pop('_target_')",
            "print(_scheduler_signature(DDPMScheduler(**config)))",
        ))
        root = Path(__file__).resolve().parents[1]
        signatures = []
        for seed in ("1", "2"):
            environment = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8",
                               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            environment.setdefault("HF_HOME", str(root / "artifacts" / "hf-cache"))
            completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
                                       env=environment, check=True, capture_output=True,
                                       text=True, encoding="utf-8", timeout=45)
            signatures.append(completed.stdout.strip())
        self.assertEqual(signatures[0], signatures[1])
        # Require the complete locked config, rather than accepting a stable
        # fingerprint that silently omitted the values governing DDPM math.
        canonical = json.loads(signatures[0])
        for key, value in official_spec()["policy_config"]["noise_scheduler"].items():
            if key != "_target_":
                self.assertEqual(canonical[key], value)
        self.assertEqual(canonical["_use_default_values"], sorted(canonical["_use_default_values"]))

    def test_final_step_has_unclipped_gaussian_noise_and_fixed_mean_statistics(self):
        floor, batch = 1.2, 256
        actor = actor_fixture(floor=floor, zero_final_mean=True)
        generator = torch.Generator().manual_seed(718)
        record = actor.sample(observation_fixture(batch), generator=generator)
        self.assertAlmostEqual(record.old_stds[-1].item(), floor, places=6)
        torch.testing.assert_close(record.old_means[:, -1], torch.zeros(batch, 16, 2), rtol=0, atol=2e-6)
        # Use known mean zero and configured sigma, rather than subtracting
        # the sampler's own recorded mean. There are 8192 independent values.
        residuals = record.chain[:, -1] / floor
        self.assertLess(abs(residuals.mean().item()), .04)
        self.assertLess(abs(residuals.var(unbiased=False).item() - 1), .05)
        self.assertTrue((record.chain[:, -1].abs() > 1).any())
        # Initial noise plus every one of the 100 transitions consumes one
        # full-shaped draw, including t=0. No hidden final-step exception.
        independent_rng = torch.Generator().manual_seed(718)
        for _ in range(101):
            torch.randn((batch, 16, 2), generator=independent_rng)
        torch.testing.assert_close(generator.get_state(), independent_rng.get_state(), rtol=0, atol=0)

    def test_parameter_whitelists_and_train_mode_preserve_frozen_fixed_encoder(self):
        actor = actor_fixture()
        self.assertEqual(actor.selection.total, 6)
        self.assertEqual(set(actor.selection.names), {"model.final_conv.weight", "model.final_conv.bias"})
        selection = select_trainable_parameters(actor.network, "last_up_and_output")
        self.assertEqual(selection.total, 12)
        self.assertEqual(len(selection.names), 4)
        for name, parameter in actor.network.named_parameters():
            self.assertEqual(parameter.requires_grad, name in selection.names)
        actor.train()
        self.assertTrue(actor.training)
        self.assertFalse(any(module.training for module in actor.network.modules()))
        observations = observation_fixture(2)
        rng = torch.random.get_rng_state().clone()
        first, second = actor.encode(observations), actor.encode(observations)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(actor.network.obs_encoder.crop_shape, (84, 84))
        self.assertFalse(first.requires_grad)

    def test_recompute_treats_endpoints_as_constants_and_gradients_stay_in_whitelist(self):
        actor = actor_fixture(subset="last_up_and_output")
        record = actor.sample(observation_fixture(2), generator=torch.Generator().manual_seed(63))
        endpoint = record.chain.clone().requires_grad_(True)
        images = record.observations["image"].clone().requires_grad_(True)
        record = replace(record, chain=endpoint,
                         observations={**record.observations, "image": images})
        selected = [10, 50, 99]
        recomputed = actor.recompute_log_probs(record, step_indices=selected, batch_indices=slice(0, 1))
        loss = transition_ppo_loss(recomputed, record.old_log_prob[:1, selected], torch.ones(1))
        loss.backward()
        self.assertIsNone(endpoint.grad)
        self.assertIsNone(images.grad)
        selected_gradients = []
        for name, parameter in actor.network.named_parameters():
            if name in actor.selection.names:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                selected_gradients.append(parameter.grad)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any((gradient != 0).any() for gradient in selected_gradients))

    def test_controlled_toy_update_follows_advantage_and_zero_advantage_is_noop(self):
        # This scalar Gaussian toy is a mathematical gradient test, not an
        # official actor update or evidence of physical policy improvement.
        for advantage in (1., -1., 0.):
            with self.subTest(advantage=advantage):
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=0)
                action, std = torch.full((1, 16, 2), .25), torch.tensor(.7)

                def density():
                    mean = model(torch.ones(1, 1)).reshape(1, 1, 1).expand_as(action)
                    return gaussian_log_prob(action, mean, std).reshape(1, 1)

                before = density().detach()
                loss = transition_ppo_loss(density(), before, torch.tensor([advantage]))
                loss.backward()
                self.assertTrue(torch.isfinite(model.weight.grad).all())
                if advantage:
                    self.assertLess(model.weight.grad.item() * advantage, 0)
                else:
                    self.assertEqual(model.weight.grad.item(), 0)
                optimizer.step()
                change = (density().detach() - before).item()
                if advantage:
                    self.assertGreater(change * advantage, 0)
                else:
                    self.assertEqual(change, 0)
                    self.assertEqual(model.weight.item(), 0)

    def test_ppo_positive_and_negative_clip_branches_and_denoising_weights(self):
        ratio = torch.tensor([[1.3, .7, 1.1, .9, .7, 1.3]], dtype=torch.float64)
        log_prob = ratio.log().requires_grad_(True)
        advantage = torch.tensor([[2., -2., 2., -2., 2., -2.]], dtype=ratio.dtype)
        loss = transition_ppo_loss(log_prob, torch.zeros_like(log_prob), advantage, reduction="none")
        expected = torch.tensor([[-2.4, 1.6, -2.2, 1.8, -1.4, 2.6]], dtype=ratio.dtype)
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        torch.testing.assert_close(log_prob.grad,
                                   torch.tensor([[0., 0., -2.2, 1.8, -1.4, 2.6]], dtype=ratio.dtype))
        weighted = transition_ppo_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2),
                                       denoising_discount=.5, timesteps=torch.tensor([2, 1, 0]), reduction="none")
        torch.testing.assert_close(weighted, -torch.tensor([[.25, .5, 1.], [.25, .5, 1.]]))

    def test_ppo_overflow_rejects_update_before_nan_can_enter_gradients(self):
        for advantage in (0., 1., -1.):
            with self.subTest(advantage=advantage):
                new = torch.tensor([[1000.]], requires_grad=True)
                with self.assertRaisesRegex(ValueError, "ratio overflow"):
                    transition_ppo_loss(new, torch.zeros_like(new), torch.tensor([advantage]))
                self.assertIsNone(new.grad)
        # The ratio itself can be representable while ratio * advantage is
        # not. Both signs must fail, including the positively clipped branch.
        for advantage in (1e30, -1e30):
            with self.subTest(surrogate_advantage=advantage):
                new = torch.tensor([[50.]], requires_grad=True)
                self.assertTrue(torch.isfinite(new.exp()).all())
                with self.assertRaisesRegex(ValueError, "surrogate overflow"):
                    transition_ppo_loss(new, torch.zeros_like(new), torch.tensor([advantage]))
                self.assertIsNone(new.grad)

    def test_unequal_batch_and_time_accumulation_matches_full_objective_and_gradient(self):
        features = torch.linspace(-.8, .8, 15, dtype=torch.float64).reshape(5, 3)
        advantages = torch.tensor([-1., .5, 2., -.2, 0.], dtype=features.dtype)
        old = torch.zeros_like(features)
        full_parameter = torch.nn.Parameter(torch.tensor(.1, dtype=features.dtype))
        full_loss = transition_ppo_loss(full_parameter * features, old, advantages)
        full_loss.backward()
        partial_parameter = torch.nn.Parameter(full_parameter.detach().clone())
        accumulated_loss = 0.
        for rows in (slice(0, 2), slice(2, 5)):
            for columns in (slice(0, 1), slice(1, 3)):
                partial_loss = transition_ppo_loss(partial_parameter * features[rows, columns],
                                                   old[rows, columns], advantages[rows], reduction="sum") / features.numel()
                accumulated_loss += partial_loss.item()
                partial_loss.backward()
        self.assertAlmostEqual(accumulated_loss, full_loss.item(), places=14)
        torch.testing.assert_close(partial_parameter.grad, full_parameter.grad, rtol=1e-13, atol=1e-14)

    def test_record_microbatches_match_dense_recomputed_actor_gradients(self):
        actor = actor_fixture(subset="last_up_and_output")
        record = actor.sample(observation_fixture(5), generator=torch.Generator().manual_seed(317))
        steps, advantages = [10, 40, 99], torch.tensor([-1., .5, 2., -.2, 0.])
        full = transition_ppo_loss(actor.recompute_log_probs(record, step_indices=steps),
                                   record.old_log_prob[:, steps], advantages)
        full.backward()
        expected = {name: p.grad.clone() for name, p in actor.network.named_parameters()
                    if p.requires_grad}
        actor.zero_grad(set_to_none=True)
        accumulated = 0.
        for rows in (slice(0, 2), slice(2, 5)):
            for selected in ([10], [40, 99]):
                new = actor.recompute_log_probs(record, step_indices=selected, batch_indices=rows)
                loss = transition_ppo_loss(new, record.old_log_prob[rows][:, selected],
                                           advantages[rows], reduction="sum") / (5 * len(steps))
                accumulated += loss.item()
                loss.backward()
        self.assertAlmostEqual(accumulated, full.item(), places=6)
        for name, parameter in actor.network.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
