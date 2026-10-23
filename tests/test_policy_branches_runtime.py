"""Warm-contact replay and finite physical-budget admission, no trained policy."""

from dataclasses import asdict, replace
import os
import unittest


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT"), "Enable official branch-physics checks")
class PolicyBranchChecks(unittest.TestCase):
    def test_full_prefix_reconstructs_contacts_images_and_physics(self):
        import numpy as np
        from tdyssey.envs.official import make_official_shape_env_class
        from tdyssey.official import load_official_source
        from tdyssey.policy_branches import replay_prefix, step_record, observation_hash
        from tdyssey.state import BodyState
        from tdyssey.geometry import Pose
        cls = make_official_shape_env_class(load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"]))
        env = cls(shape="L", legacy=True)
        self.addCleanup(env.close)
        env.seed(42)
        env.reset()
        # Explicit privileged contact fixture; never an actor training input.
        state = replace(env.capture_state(), object_body=BodyState(Pose((256, 256), 0)),
                        pusher_body=BodyState(Pose((205, 210), 0)), pusher_target=None)
        episode = {"initial_state": asdict(state), "steps": []}
        counters = {"initial_state_restores": 0, "replay_controls": 0, "training_controls": 0}
        history = replay_prefix(env, episode, counters, lambda: None)
        for _ in range(12):
            observation, row = step_record(env, np.array([280., 210.]))
            history.append(observation)
            episode["steps"].append(row)
        self.assertGreater(sum(r["contacts"] for r in episode["steps"]), 0)
        expected = observation_hash(history)
        from tdyssey.viewer import presentation_frame
        before = env.capture_state()
        cache = env.render_cache.copy()
        self.assertEqual(presentation_frame(env).shape, (512,512,3))
        self.assertEqual(env.render_size, 96)
        self.assertEqual(before, env.capture_state())
        np.testing.assert_array_equal(cache, env.render_cache)
        replayed = replay_prefix(env, episode, counters, lambda: None)
        self.assertEqual(observation_hash(replayed), expected)
        self.assertEqual(counters["replay_controls"], 12)
        self.assertEqual(counters["training_controls"], 12)
        episode["steps"][3]["contacts"] += 1
        with self.assertRaises(ValueError):
            replay_prefix(env, episode, counters, lambda: None)

    def test_group_budget_refuses_before_sampling_or_physics(self):
        from tdyssey.forward_protocol import ForwardConfig
        from tdyssey.policy_branches import BranchCollector
        from tdyssey.envs.official import make_official_shape_env_class
        from tdyssey.official import load_official_source
        cls = make_official_shape_env_class(load_official_source(os.environ["TDYSSEY_OFFICIAL_ROOT"]))
        counts = {"training_controls": 0}
        collector = BranchCollector(None, cls, ForwardConfig(group_size=4, max_train_controls=31),
                                    counts, {}, lambda: None)
        self.addCleanup(collector.close)
        self.assertIsNone(collector.collect("T"))
        self.assertEqual(counts, {"training_controls": 0})


if __name__ == "__main__":
    unittest.main()
