"""Opt-in checks for frozen, aligned batched official-policy candidates."""

import os
import unittest

from tdyssey.official import OfficialDiffusionPolicy, official_spec


@unittest.skipUnless(os.environ.get("TDYSSEY_DATA_TESTS") or os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"),
                     "Set TDYSSEY_DATA_TESTS=1 for installed tensor candidate checks")
class OfficialCandidateRuntimeChecks(unittest.TestCase):
    def fixture(self):
        import numpy as np
        import torch

        class Sampler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.kwargs = {}
                self.observations = []
                self.predictions = []

            def reset(self):
                pass

            def predict_action(self, observation):
                self.grad_enabled = torch.is_grad_enabled()
                self.observations.append({key: value.clone() for key, value in observation.items()})
                batch = observation["image"].shape[0]
                full = 256 + 10 * torch.randn((batch, 16, 2), generator=self.kwargs["generator"])
                self.predictions.append(full.clone())
                return {"action": full[:, 1:9], "action_pred": full}

        policy = OfficialDiffusionPolicy.__new__(OfficialDiffusionPolicy)
        policy.spec, policy.device = official_spec(), torch.device("cpu")
        policy.network = Sampler().eval().requires_grad_(False)
        policy.generator = None
        history = [{"image": np.full((3, 96, 96), value, dtype=np.float32),
                    "agent_pos": np.array([100 + index, 200 + index], dtype=np.float32)}
                   for index, value in enumerate((0.2, 0.4))]
        return policy, history

    def test_batched_samples_share_history_use_local_rng_and_leave_weights_frozen(self):
        import numpy as np
        import torch

        policy, history = self.fixture()
        before_global = torch.random.get_rng_state().clone()
        before_weight = policy.network.weight.clone()
        policy.reset_seed(123)
        candidates = policy.predict_action_candidates(history, 4)
        self.assertEqual(candidates.shape, (4, 8, 2))
        np.testing.assert_array_equal(candidates, policy.network.predictions[0][:, 1:9].numpy())
        self.assertFalse(np.array_equal(candidates[0], candidates[1]))
        for key in ("image", "agent_pos"):
            expected = np.stack([obs[key] for obs in history])
            for batch in policy.network.observations[0][key].numpy():
                np.testing.assert_array_equal(batch, expected)
        self.assertFalse(policy.network.grad_enabled)
        self.assertFalse(policy.network.training)
        self.assertFalse(policy.network.weight.requires_grad)
        torch.testing.assert_close(policy.network.weight, before_weight, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_global))
        later = policy.predict_action_candidates(history, 4)
        self.assertFalse(np.array_equal(candidates, later))
        policy.reset_seed(123)
        np.testing.assert_array_equal(policy.predict_action_candidates(history, 4), candidates)

    def test_one_candidate_preserves_baseline_actions_and_rng_consumption_exactly(self):
        import numpy as np
        import torch

        policy, history = self.fixture()
        policy.reset_seed(456)
        expected = policy.predict_actions(history)
        expected_rng = policy.generator.get_state().clone()
        policy.reset_seed(456)
        actual = policy.predict_action_candidates(history, 1)
        np.testing.assert_array_equal(actual, expected[None, ...])
        self.assertTrue(torch.equal(policy.generator.get_state(), expected_rng))


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT") and os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"),
                     "Set official source/checkpoint paths for real CUDA batched-policy parity")
class OfficialCandidateCheckpointChecks(unittest.TestCase):
    def test_batched_candidates_match_direct_official_sampler_and_executable_slice(self):
        import numpy as np
        import torch

        from tdyssey.official import load_official_source

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        policy = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"],
                                         os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        env_class = load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"])
        env = env_class(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        history = [env.reset()] * 2

        # Keep the snapshot on the CPU so checking frozen inference does not
        # require another full model on the GPU or a serialization fingerprint.
        before_weights = {name: tensor.detach().cpu().clone()
                          for name, tensor in policy.network.state_dict().items()}
        before_cpu_rng = torch.random.get_rng_state().clone()
        before_cuda_rng = torch.cuda.get_rng_state(policy.device).clone()
        policy.reset_seed(123)
        candidates = policy.predict_action_candidates(history, 4)
        candidate_rng = policy.generator.get_state().clone()
        self.assertEqual(candidates.shape, (4, 8, 2))
        self.assertFalse(np.array_equal(candidates[0], candidates[1]))
        policy.reset_seed(123)
        observation = {}
        for key in ("image", "agent_pos"):
            stacked = torch.stack([torch.as_tensor(obs[key], device=policy.device) for obs in history])
            observation[key] = stacked.unsqueeze(0).expand(4, *stacked.shape).contiguous()
        self.assertEqual(observation["image"].shape, (4, 2, 3, 96, 96))
        self.assertEqual(observation["agent_pos"].shape, (4, 2, 2))
        with torch.inference_mode():
            direct = policy.network.predict_action(observation)
        np.testing.assert_array_equal(candidates, direct["action"].cpu().numpy())
        np.testing.assert_array_equal(candidates, direct["action_pred"][:, 1:9].cpu().numpy())
        self.assertTrue(torch.equal(policy.generator.get_state(), candidate_rng))

        policy.reset_seed(456)
        baseline = policy.predict_actions(history)
        baseline_rng = policy.generator.get_state().clone()
        policy.reset_seed(456)
        one_candidate = policy.predict_action_candidates(history, 1)
        np.testing.assert_array_equal(one_candidate, baseline[None, ...])
        self.assertTrue(torch.equal(policy.generator.get_state(), baseline_rng))
        after_weights = policy.network.state_dict()
        self.assertEqual(list(after_weights), list(before_weights))
        for name, before in before_weights.items():
            after = after_weights[name].detach().cpu()
            self.assertEqual(after.dtype, before.dtype, name)
            self.assertEqual(after.shape, before.shape, name)
            self.assertTrue(torch.equal(after, before), name)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_cpu_rng))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(policy.device), before_cuda_rng))
        self.assertFalse(policy.network.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in policy.network.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in policy.network.parameters()))


if __name__ == "__main__":
    unittest.main()
