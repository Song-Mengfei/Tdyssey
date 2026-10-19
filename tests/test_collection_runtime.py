"""Opt-in tests of actual play collection and compact image/state persistence."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.collection import CollectionConfig, collect_episode
from tdyssey.dataset import DatasetWriter, load_dataset
from tdyssey.envs.gpc import make_shape_env_class
from tdyssey.trajectories import split_trajectories


@unittest.skipUnless(os.environ.get("TDYSSEY_GPC_ROOT"), "Set TDYSSEY_GPC_ROOT for real collection tests")
class CollectionRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for key, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
                           "MPLBACKEND": "Agg", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
            os.environ.setdefault(key, value)
        from examples.gpc_shape_env import load_upstream
        import numpy as np

        root = Path(os.environ["TDYSSEY_GPC_ROOT"]).resolve()
        upstream, _ = load_upstream(root)
        cls.env_class, cls.np = make_shape_env_class(upstream.PushTImageEnv), np
        previous_cwd = Path.cwd()
        cls.addClassCleanup(os.chdir, previous_cwd)
        os.chdir(root / "gpc_rank_evaluation")

    def environment(self):
        env = self.env_class(shape="L", domain_filename="push_t", resize_scale=96)
        self.addCleanup(env.close)
        return env

    def counters(self):
        return {"reset_attempts": 0, "reset_physics_substeps": 0, "control_steps": 0, "control_physics_substeps": 0}

    def write_episode(self, root, config, episode, counters):
        writer = DatasetWriter(root, config, {"kind": "runtime_test"})
        writer.manifest["counts"] = counters
        writer.manifest["rates"] = {"simulation_hz": 100, "control_hz": 10}
        writer.add(episode)
        splits = {key: [r.trajectory_id for r in records]
                  for key, records in split_trajectories([episode.trajectory]).items()}
        writer.finish(splits)
        return writer

    def test_saved_observations_align_with_states_and_horizon_flags(self):
        config = CollectionConfig(episodes=1, steps=5)
        env, counters = self.environment(), self.counters()
        episode = collect_episode(env, config, 0, counters)
        self.assertEqual(counters["control_steps"], 5)
        self.assertTrue(episode.trajectory.transitions[-1].truncated)
        self.assertFalse(episode.steps[-1]["upstream_truncated"])
        self.assertTrue(episode.steps[-1]["collector_truncated"])
        with tempfile.TemporaryDirectory(prefix="tdyssey-data-test-") as temporary:
            output = Path(temporary) / "dataset"
            self.write_episode(output, config, episode, counters)
            manifest, loaded = load_dataset(output)
            trajectory, record, arrays = loaded[0]
            self.assertEqual(trajectory, episode.trajectory)
            self.assertEqual(arrays["images"].shape, (6, 3, 96, 96))
            self.assertEqual(arrays["images"].dtype, self.np.uint8)
            observation, _ = env.restore_state(trajectory.transitions[0].before)
            error = abs(arrays["images"][0].astype(float) / 255 - observation["image"]).max()
            self.assertLessEqual(float(error), 0.5 / 255 + 1e-7)
            self.assertLess(sum(p.stat().st_size for p in output.iterdir()), config.max_bytes)
            self.assertEqual(record["stop_reason"], "collection_horizon")
            self.assertEqual(manifest["counts"]["control_physics_substeps"], 50)

    def test_rejected_resets_are_counted_and_actions_are_reproducible(self):
        config = CollectionConfig(episodes=1, steps=4, mode="object-directed")
        env, counters = self.environment(), self.counters()
        original_reset = env.reset
        attempts = []

        def first_reset_is_invalid():
            result = original_reset()
            if not attempts:
                env.agent.position = (10, 100)  # Inject an invalid physical reset.
            attempts.append(True)
            return result

        with patch.object(env, "reset", side_effect=first_reset_is_invalid):
            first = collect_episode(env, config, 0, counters)
        self.assertGreaterEqual(counters["reset_attempts"], 2)
        self.assertEqual(counters["reset_attempts"], counters["reset_physics_substeps"])
        self.assertIn("pusher_outside_walls", first.resets[0]["issues"])
        self.assertEqual(first.resets[-1]["issues"], [])
        # The same protocol and same rejected attempt reproduce the accepted run.
        attempts.clear()
        with patch.object(env, "reset", side_effect=first_reset_is_invalid):
            second = collect_episode(env, config, 0, self.counters())
        self.assertEqual(first.trajectory, second.trajectory)
        self.np.testing.assert_array_equal(first.images, second.images)

    def test_upstream_terminal_stops_recording_before_collection_horizon(self):
        env, counters = self.environment(), self.counters()
        original_step = env.step

        def terminate_after_one_action(action):
            observation, reward, _, _, info = original_step(action)
            return observation, reward, True, False, info

        with patch.object(env, "step", side_effect=terminate_after_one_action):
            episode = collect_episode(env, CollectionConfig(episodes=1, steps=8), 0, counters)
        self.assertEqual(len(episode.trajectory.transitions), 1)
        self.assertEqual(counters["control_steps"], 1)
        self.assertEqual(len(episode.images), 2)
        self.assertTrue(episode.trajectory.transitions[0].terminated)
        self.assertFalse(episode.trajectory.transitions[0].truncated)
        self.assertEqual(episode.stop_reason, "upstream_termination")

    def test_writer_refuses_overwrite_and_loader_detects_corrupt_data(self):
        config = CollectionConfig(episodes=1, steps=2)
        counters = self.counters()
        episode = collect_episode(self.environment(), config, 0, counters)
        with tempfile.TemporaryDirectory(prefix="tdyssey-data-test-") as temporary:
            output = Path(temporary) / "dataset"
            self.write_episode(output, config, episode, counters)
            original = (output / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                DatasetWriter(output, config, {})
            self.assertEqual((output / "manifest.json").read_bytes(), original)
            image_path = output / "episode-0000.npz"
            image_path.write_bytes(image_path.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "hash/size"):
                load_dataset(output)

    def test_byte_limit_and_failed_run_are_explicit(self):
        config = CollectionConfig(episodes=1, steps=1, max_bytes=1024 * 1024)
        with tempfile.TemporaryDirectory(prefix="tdyssey-data-test-") as temporary:
            output = Path(temporary) / "too-large"
            with self.assertRaisesRegex(ValueError, "byte limit"):
                DatasetWriter(output, config, {"oversized": "x" * config.max_bytes})
            self.assertEqual(list(output.iterdir()), [])
            failed = Path(temporary) / "failed"
            writer = DatasetWriter(failed, config, {})
            writer.finish(error="Reset attempts exhausted")
            self.assertEqual(json.loads((failed / "manifest.json").read_text())["status"], "failed")
            with self.assertRaisesRegex(ValueError, "complete"):
                load_dataset(failed)


if __name__ == "__main__":
    unittest.main()
