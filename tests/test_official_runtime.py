"""Opt-in official simulator alignment and checkpoint/inference parity checks."""

from collections import deque
import os
from pathlib import Path
import unittest

from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics, physical_state


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Set TDYSSEY_OFFICIAL_ROOT for original PushT tests")
class OfficialRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        cls.env_class = load_official_source(Path(os.environ["TDYSSEY_OFFICIAL_ROOT"]))

    def test_step_history_and_budget_match_official_multistep_wrapper(self):
        import numpy as np
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

        direct = self.env_class(legacy=True)
        wrapped = MultiStepWrapper(self.env_class(legacy=True), n_obs_steps=2, n_action_steps=8, max_episode_steps=10)
        self.addCleanup(direct.close)
        self.addCleanup(wrapped.close)
        direct.seed(100000)
        wrapped.seed(100000)
        initial, stacked = direct.reset(), wrapped.reset()
        history = deque([initial] * 2, maxlen=2)
        np.testing.assert_array_equal(stacked["image"], np.stack([initial["image"]] * 2))
        for count in (8, 2):
            actions = np.array([[235.0, 124.0]] * count)
            rewards = []
            for action in actions:
                observation, reward, done, _ = direct.step(action)
                self.assertFalse(done)
                rewards.append(reward)
                history.append(observation)
            stacked, maximum_reward, done, _ = wrapped.step(actions)
            for key in ("image", "agent_pos"):
                np.testing.assert_array_equal(stacked[key], np.stack([obs[key] for obs in history]))
            self.assertEqual(maximum_reward, max(wrapped.reward))
        self.assertTrue(done)  # Exactly ten controls, not ten action chunks.
        self.assertEqual(len(wrapped.reward), 10)

    def test_metrics_use_official_collision_geometry_and_fixed_goal(self):
        import numpy as np

        env = self.env_class(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        env.reset()
        np.testing.assert_allclose(env.goal_pose, [256, 256, np.pi / 4])
        self.assertEqual(env.block.mass, 1)
        # Put the actual body at the goal without altering geometry or assuming
        # the GPC T coordinate origin. The official constructor has a different CoG.
        env.block.angle = float(env.goal_pose[2])
        env.block.position = env.goal_pose[:2].tolist()
        env.space.reindex_shapes_for_body(env.block)
        metrics = physical_metrics(env)
        self.assertAlmostEqual(metrics["coverage"], 1.0)
        self.assertAlmostEqual(metrics["iou"], 1.0)
        self.assertEqual(physical_state(env)["object"]["position"], [256.0, 256.0])

    @unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_CHECKPOINT"), "Set TDYSSEY_OFFICIAL_CHECKPOINT for real CUDA policy parity")
    def test_checkpoint_normalizers_and_prediction_match_direct_official_class(self):
        import numpy as np
        import torch

        torch.set_num_threads(4)
        policy = OfficialDiffusionPolicy(os.environ["TDYSSEY_OFFICIAL_ROOT"], os.environ["TDYSSEY_OFFICIAL_CHECKPOINT"])
        self.assertTrue(all(p.device.type == "cuda" for p in policy.network.parameters()))
        self.assertEqual(set(policy.network.normalizer.params_dict), {"action", "agent_pos", "image"})
        env = self.env_class(legacy=True)
        self.addCleanup(env.close)
        env.seed(100000)
        observation = env.reset()
        history = [observation] * 2
        policy.reset_seed(123)
        adapter_actions = policy.predict_actions(history)
        # Independent call through the unchanged official class, with matching RNG.
        policy.reset_seed(123)
        stacked = {key: torch.as_tensor(np.stack([o[key] for o in history]), device="cuda").unsqueeze(0)
                   for key in ("image", "agent_pos")}
        with torch.inference_mode():
            direct = policy.network.predict_action(stacked)
        np.testing.assert_array_equal(adapter_actions, direct["action"][0].cpu().numpy())
        np.testing.assert_array_equal(adapter_actions, direct["action_pred"][0, 1:9].cpu().numpy())
        self.assertEqual(adapter_actions.shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
