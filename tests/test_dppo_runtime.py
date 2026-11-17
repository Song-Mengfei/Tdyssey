"""Real official sampler parity, independent densities and actual CUDA updates."""

import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"), "Enable official CUDA RL checks")
class DPPORuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.set_num_threads(2)
        from tdyssey.official import OfficialDiffusionPolicy
        from tdyssey.rl_protocol import RLConfig
        from tdyssey.dppo import TailPPO
        cls.official = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"],
                                               os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        cls.agent = TailPPO(cls.official, RLConfig(train_timesteps=2, minibatch_size=2))

    def test_lora_zero_delta_dense_kernel_equivalence_and_frozen_base(self):
        import torch
        from tdyssey.lora import LoRAConv1d
        base = torch.nn.Conv1d(5, 7, 3, stride=2, padding=1).cuda().requires_grad_(False)
        layer = LoRAConv1d(base, rank=2, alpha=4.)
        x = torch.randn(2, 5, 16, device="cuda")
        frozen = base.weight.detach().clone()
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        with torch.no_grad():
            layer.delta.b.normal_(0, .1)
        kernel = base.weight + layer.delta.scale*(layer.delta.b @ layer.delta.a).reshape(base.weight.shape)
        expected = torch.nn.functional.conv1d(x, kernel, base.bias, stride=2, padding=1)
        torch.testing.assert_close(layer(x), expected)
        layer(x).square().mean().backward()
        self.assertGreater(float(layer.delta.a.grad.abs().sum()), 0)
        self.assertGreater(float(layer.delta.b.grad.abs().sum()), 0)
        self.assertIsNone(base.weight.grad)
        torch.testing.assert_close(base.weight, frozen, rtol=0, atol=0)
        layer.enabled = False
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)

    def test_distributed_lora_initial_sampler_update_and_compact_reload(self):
        from dataclasses import replace
        import numpy as np
        import torch
        from tdyssey.dppo import TailPPO, ppo_update
        from tdyssey.official import OfficialDiffusionPolicy, load_official_source
        from tdyssey.policy_training import frozen_digest
        official = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"],
                                            os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        env = load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"])(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        history = [env.reset()]*2
        official.reset_seed(456)
        expected = official.predict_actions(history)
        agent = TailPPO(official, replace(self.agent.config, trainable="lora", lora_target="action_network",
                                         train_timesteps=1, minibatch_size=1))
        self.assertGreater(len(agent.lora_layers), 2)
        condition = agent.encode(history)
        agent.generator.manual_seed(456)
        actual, first = agent.sample(condition)
        np.testing.assert_array_equal(actual, expected)
        _, second = agent.sample(condition)
        frozen = frozen_digest(agent.network)
        before = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
        records = [{"condition": condition, "transitions": t} for t in (first, second)]
        metrics = ppo_update(agent, records, [1., -1.], [1., -1.],
            torch.optim.Adam(agent.actor.parameters(), lr=1e-5),
            torch.optim.Adam(agent.critic.parameters(), lr=3e-4))
        self.assertTrue(metrics["minibatches"])
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in agent.actor.state_dict().items()))
        self.assertEqual(frozen, frozen_digest(agent.network))
        with tempfile.TemporaryDirectory(prefix="tdyssey-distributed-lora-") as root:
            path = Path(root)/"lora.pt"
            torch.save(agent.state(), path)
            self.assertLess(path.stat().st_size, 64*1024**2)
            rng = agent.generator.get_state()
            expected, _ = agent.sample(condition)
            agent.load(torch.load(path, map_location="cuda", weights_only=True))
            agent.generator.set_state(rng)
            actual, _ = agent.sample(condition)
            np.testing.assert_array_equal(actual, expected)

    def test_ddpm_conditional_and_joint_density_match_independent_calculation(self):
        import torch
        from tdyssey.dppo import ddpm_mean_std, gaussian_log_prob
        scheduler = self.agent.network.noise_scheduler
        sample, prediction = torch.randn(2, 16, 2, device="cuda"), torch.randn(2, 16, 2, device="cuda")
        for t in (4, 1, 0):
            mean, std = ddpm_mean_std(scheduler, prediction, t, sample)
            generator = torch.Generator(device="cuda").manual_seed(123)
            noise = torch.randn(sample.shape, generator=generator, device="cuda")
            generator.manual_seed(123)
            actual = scheduler.step(prediction, t, sample, generator=generator).prev_sample
            torch.testing.assert_close(mean+(std*noise if t else 0), actual, rtol=0, atol=0)
            if t:
                expected = torch.distributions.Independent(torch.distributions.Normal(mean, std), 2).log_prob(actual)
                torch.testing.assert_close(gaussian_log_prob(actual, mean, std), expected)
            else:
                with self.assertRaises(ValueError):
                    gaussian_log_prob(actual, mean, std)

    def test_sampler_rng_parity_real_update_frozen_parameters_and_checkpoint(self):
        import numpy as np
        import torch
        from tdyssey.dppo import ppo_update
        from tdyssey.official import load_official_source
        from tdyssey.policy_training import frozen_digest
        env = load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"])(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        observation = env.reset()
        history = [observation]*2
        official, agent = self.official, self.agent
        official.reset_seed(123)
        expected = official.predict_actions(history)
        expected_rng = official.generator.get_state()
        agent.generator.manual_seed(123)
        condition = agent.encode(history)
        actual, transitions = agent.sample(condition)
        np.testing.assert_array_equal(actual, expected)
        torch.testing.assert_close(agent.generator.get_state(), expected_rng, rtol=0, atol=0)
        for transition in transitions:
            with torch.no_grad():
                logp = agent.log_prob(condition, transition["before"], transition["after"], transition["t"])
            torch.testing.assert_close(logp, transition["old_logp"], rtol=0, atol=0)
        _, other = agent.sample(condition)
        records = [{"condition": condition, "transitions": chain} for chain in (transitions, other)]
        initial = {k: p.detach().clone() for k, p in agent.actor.named_parameters()}
        frozen = frozen_digest(agent.network)
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=1e-5)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=3e-4)
        metrics = ppo_update(agent, records, [1., -1.], [1., -1.], optimizer, critic_optimizer)
        self.assertTrue(metrics["minibatches"])
        self.assertGreater(metrics["minibatches"][0]["actor_gradient_norm"], 0)
        self.assertTrue(any(not torch.equal(p, initial[k]) for k, p in agent.actor.named_parameters()))
        self.assertEqual(frozen, frozen_digest(agent.network))
        with tempfile.TemporaryDirectory(prefix="tdyssey-dppo-test-") as root:
            path = Path(root)/"delta.pt"
            torch.save(agent.state(), path)
            rng = agent.generator.get_state()
            expected, _ = agent.sample(condition)
            agent.load(torch.load(path, map_location="cuda", weights_only=True))
            agent.generator.set_state(rng)
            actual, _ = agent.sample(condition)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
