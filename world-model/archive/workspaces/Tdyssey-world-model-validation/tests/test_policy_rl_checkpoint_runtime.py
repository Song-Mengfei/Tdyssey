"""Opt-in, bounded real-checkpoint likelihood/gradient checks; no actor updates."""

import os
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and os.environ.get("TDYSSEY_RL_CHECKPOINT_CHECK") == "1",
                     "Set TDYSSEY_RL_CHECKPOINT_CHECK=1 and official asset paths for bounded CUDA checks")
class CheckpointLikelihoodChecks(unittest.TestCase):
    def test_official_chain_parity_probability_and_selected_gradients(self):
        import numpy as np
        from tdyssey.official import OfficialDiffusionPolicy, load_official_source
        from tdyssey.policy_rl import DenoisingPolicy, select_trainable_parameters

        torch.set_num_threads(4)
        official = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"],
                                           os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"],
                                           checkpoint_sha256=os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT_SHA256"))
        env = load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"])(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        observation = env.reset()
        history = [observation] * 2
        observations = official.observation_tensors(history)
        actor = DenoisingPolicy(official)
        self.assertEqual(actor.selection.total, 1_313_282)
        self.assertEqual(sum(p.numel() for p in actor.network.model.parameters()), 251_511_938)
        actor.train()
        self.assertTrue(all(not m.training for m in actor.network.modules()))
        global_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        torch.testing.assert_close(actor.encode(observations), actor.encode(observations), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
        # Actual upstream crop is fixed 84x84 and acts as the center crop in eval.
        for module in actor.network.obs_encoder.modules():
            if hasattr(module, "crop_height"):
                self.assertEqual((module.crop_height, module.crop_width), (84, 84))
                image = actor.network.normalizer.normalize(observations)["image"].reshape(2, 3, 96, 96)
                torch.testing.assert_close(module.forward_in(image), image[:, :, 6:90, 6:90], rtol=0, atol=0)

        official.reset_seed(123)
        expected, trace = official.predict_actions_with_trace(history)
        expected_next = official.predict_actions(history)
        generator = torch.Generator(device="cuda").manual_seed(123)
        baseline = actor.sample(observations, generator=generator, mode="baseline")
        np.testing.assert_array_equal(baseline.actions[0].cpu().numpy(), expected)
        np.testing.assert_array_equal(baseline.chain[0].cpu().numpy(), trace["normalized_samples"])
        np.testing.assert_array_equal(actor.sample(observations, generator=generator, mode="baseline").actions[0].cpu().numpy(),
                                      expected_next)
        self.assertIsNone(baseline.old_log_prob)
        self.assertEqual(float(baseline.old_stds[-1]), 0)
        del baseline

        record = actor.sample(observations, generator=torch.Generator(device="cuda").manual_seed(124),
                              actor_version="pinned-ema-before-update")
        self.assertEqual(record.chain.shape, (1, 101, 16, 2))
        self.assertFalse(record.chain.requires_grad)
        self.assertGreaterEqual(float(record.old_stds.min()), actor.config.std_floor - 1e-9)
        # Same-device FP32, same batch shape, fixed endpoints. A small absolute
        # log-density tolerance covers arithmetic accumulation over 32 terms;
        # it is not a cross-device/cross-precision reproducibility guarantee.
        with torch.no_grad():
            recomputed = actor.recompute_log_probs(record)
        max_error = float((recomputed - record.old_log_prob).abs().max())
        torch.testing.assert_close(recomputed, record.old_log_prob, rtol=0, atol=1e-4)
        torch.testing.assert_close((recomputed - record.old_log_prob).exp(), torch.ones_like(recomputed),
                                   rtol=0, atol=1e-4)
        for subset, count in (("output", 1_313_282), ("last_up_and_output", 13_127_170)):
            actor.selection = select_trainable_parameters(actor.network, subset)
            self.assertEqual(actor.selection.total, count)
            probability = actor.recompute_log_probs(record, step_indices=[49])
            (-probability.mean()).backward()
            selected = dict(actor.network.named_parameters())
            for name, parameter in selected.items():
                if name in actor.selection.names:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            self.assertGreater(sum(float(selected[n].grad.abs().sum()) for n in actor.selection.names), 0)
            print(f"RL bounded gradient subset={subset} count={count} tensors={len(actor.selection.names)}")
            actor.network.zero_grad(set_to_none=True)
        print(f"RL checkpoint max_logprob_error={max_error:.8g}; "
              f"peak_cuda_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}; no optimizer update")


if __name__ == "__main__":
    unittest.main()
