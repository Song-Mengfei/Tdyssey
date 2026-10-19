"""Installed tensor and rollout-contract fixtures; no downloads or policy claims."""

from dataclasses import replace
import os
import time
import unittest
from unittest.mock import patch

from tdyssey.baseline import rollout
from tdyssey.geometry import Pose, get_shape
from tdyssey.policy import GPCDiffusionPolicy, load_ema_parameters, policy_spec
from tdyssey.state import BodyState, Snapshot


@unittest.skipUnless(os.environ.get("TDYSSEY_DATA_TESTS") or os.environ.get("TDYSSEY_GPC_ROOT"),
                     "Set TDYSSEY_DATA_TESTS=1 for installed policy-contract checks")
class PolicyRuntimeChecks(unittest.TestCase):
    def test_cuda_request_fails_without_cuda_instead_of_loading_on_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "No CPU fallback"):
                GPCDiffusionPolicy("missing-source", "missing-checkpoint", device="cuda")

    def test_optional_viewer_draws_and_honors_close_events_without_desktop(self):
        import numpy as np
        import pygame
        from tdyssey.viewer import PushTViewer

        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"}):
            viewer = PushTViewer()
            try:
                self.assertTrue(viewer.draw(np.zeros((96, 96, 3), dtype=np.uint8), "fixture"))
                self.assertEqual(viewer.window.get_size(), (768, 768))
                pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, size=(600, 600), w=600, h=600))
                viewer.poll()
                self.assertEqual(viewer.window.get_size(), (600, 600))
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                self.assertFalse(viewer.poll())
                self.assertFalse(viewer.draw(np.zeros((96, 96, 3), dtype=np.uint8), "fixture"))
            finally:
                viewer.close()

    def test_ema_copies_all_parameters_and_rejects_partial_or_invalid_weights(self):
        import torch

        network = torch.nn.Linear(2, 3)
        shadows = [torch.full_like(p, i + 1) for i, p in enumerate(network.parameters())]
        load_ema_parameters(network, {"shadow_params": shadows})
        for parameter, expected in zip(network.parameters(), shadows):
            torch.testing.assert_close(parameter, expected)
        bad_payloads = [shadows[:-1], [shadows[0], shadows[1].double()],
                        [shadows[0], torch.zeros(4)], [shadows[0], torch.full_like(shadows[1], float("nan"))]]
        for bad in bad_payloads:
            with self.assertRaises(ValueError):
                load_ema_parameters(network, {"shadow_params": bad})
            for parameter, expected in zip(network.parameters(), shadows):
                torch.testing.assert_close(parameter, expected)  # Validation is atomic.
        with self.assertRaisesRegex(ValueError, "buffers"):
            load_ema_parameters(torch.nn.BatchNorm1d(2), {"shadow_params": []})

    def fixture(self, stop=None):
        import numpy as np

        state = Snapshot("I", get_shape("I").geometry_hash, "a" * 64,
                         BodyState(Pose((200, 200), 0)), BodyState(Pose((100, 100), 0)),
                         Pose((300, 300), 0), 0)
        class Environment:
            object, block_scale, sim_hz, control_hz = "I", 30, 100, 10
            def __init__(self):
                self.state, self.actions = state, []
            def capture_state(self):
                return self.state
            def observation(self):
                return {"marker": self.state.control_step}
            def step(self, action):
                self.actions.append(action.tolist())
                self.state = replace(self.state, control_step=self.state.control_step + 1)
                done = self.state.control_step == stop
                return self.observation(), 0, done, done, {"n_contacts": int(done)}
        class Policy:
            spec = policy_spec()
            def __init__(self):
                import torch
                self.device = torch.device("cpu")
                self.histories = []
            def reset_seed(self, seed):
                self.seed = seed
            def predict(self, history):
                self.histories.append([obs["marker"] for obs in history])
                return np.array([[i, i] for i in range(16)], dtype=np.float32)
        return Environment(), Policy()

    def counters(self):
        return dict.fromkeys(("planning_calls", "denoising_steps", "control_steps", "control_physics_substeps"), 0)

    def test_control_budget_history_alignment_and_short_final_chunk(self):
        env, policy = self.fixture()
        counters = self.counters()
        result = rollout(env, policy, env.observation(), 10, 123, time.monotonic() + 10, counters)
        self.assertEqual(env.actions, [[i, i] for i in [*range(1, 9), 1, 2]])
        self.assertEqual(policy.histories, [[0, 0], [7, 8]])
        self.assertEqual(counters, {"planning_calls": 2, "denoising_steps": 200,
                                    "control_steps": 10, "control_physics_substeps": 100})
        self.assertEqual(result["stop_reason"], "control_budget")

    def test_upstream_terminal_flags_stop_inside_chunk_and_wall_budget_stops_before_plan(self):
        env, policy = self.fixture(stop=3)
        result = rollout(env, policy, env.observation(), 10, 0, time.monotonic() + 10, self.counters())
        self.assertEqual(result["control_steps"], 3)
        self.assertTrue(result["steps"][-1]["terminated"])
        self.assertTrue(result["steps"][-1]["truncated"])
        self.assertEqual(result["stop_reason"], "upstream_termination")
        env, policy = self.fixture()
        result = rollout(env, policy, env.observation(), 10, 0, 0, self.counters())
        self.assertEqual(result["stop_reason"], "wall_time_budget")
        self.assertEqual(policy.histories, [])


if __name__ == "__main__":
    unittest.main()
