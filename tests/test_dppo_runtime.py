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
