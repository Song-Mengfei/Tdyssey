"""Window bounds, action scaling and split quarantine without a simulator."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tdyssey.gpc_denoiser_training import (anchors, dataset_name, decide, limit_round, load_official_windows,
                                           model_action, phase_two_sigma, window_slices)
from tdyssey.world_model_click import click_to_target, conditioning_window
from tdyssey.official_collection import ENVIRONMENT_FAMILY, OfficialCollectionConfig
from tdyssey.official_collector import OfficialShapeCollector

ROOT = Path(__file__).resolve().parents[1]


def manifest(root):
    entries = []
    for index, identifier in enumerate(("a", "b", "c")):
        entries.append({"trajectory_id": identifier, "control_steps": 4, "observation_count": 5,
                        "contact_steps": 0, "files": [
                            {"name": f"episode-{index:04d}.npz", "bytes": 1, "sha256": "a" * 64},
                            {"name": f"episode-{index:04d}.json", "bytes": 1, "sha256": "b" * 64}]})
    payload = {"schema_version": 1, "status": "complete",
               "config": {"episodes": 3, "shape": "T", "mode": "object-directed"},
               "provenance": {"environment_family": ENVIRONMENT_FAMILY},
               "episodes": entries,
               "splits": {"train": ["a"], "validation": ["b"], "model_test": ["c"]},
               "counts": {"reset_attempts": 3, "reset_physics_substeps": 3,
                          "control_steps": 12, "control_physics_substeps": 120},
               "rates": {"simulation_hz": 100, "control_hz": 10}}
    (root / "manifest.json").write_text(json.dumps(payload))
    return root


class OfficialWorldModelChecks(unittest.TestCase):
    def test_windows_start_at_four_real_frames_and_clip_actions_at_the_boundary(self):
        self.assertEqual(list(anchors(4)), [3])
        self.assertEqual(list(anchors(3)), [])
        self.assertEqual(window_slices(3), (slice(0, 5), slice(0, 4)))
        self.assertEqual(model_action(512), 1)
        self.assertEqual(model_action(0), -1)
        self.assertEqual(model_action(511), 1)

    def test_collector_keeps_the_shape_and_requires_object_directed_splits(self):
        collector = OfficialShapeCollector(OfficialCollectionConfig(shape="T"), ROOT, workers=2)
        self.assertEqual(collector.config.shape, "T")
        self.assertEqual(collector.config.mode, "object-directed")
        contact = OfficialShapeCollector(OfficialCollectionConfig(shape="T", mode="contact-balanced"), ROOT)
        self.assertEqual(contact.config.mode, "contact-balanced")
        self.assertEqual(contact.config.max_target_delta, 200)
        with self.assertRaisesRegex(ValueError, "object-directed"):
            OfficialShapeCollector(OfficialCollectionConfig(mode="random-waypoints"), ROOT)
        with self.assertRaisesRegex(ValueError, "seven"):
            OfficialShapeCollector(OfficialCollectionConfig(episodes=6), ROOT)

    def test_training_windows_do_not_open_model_test_and_refuse_mixed_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = manifest(Path(temporary))
            opened = []
            with self.assertRaisesRegex(ValueError, "model_test"):
                load_official_windows(root, ("model_test",), opened)
            self.assertEqual(opened, [])

            def fake(_root, entry):
                shape = "L" if entry["trajectory_id"] == "b" else "T"
                transition = SimpleNamespace(before=SimpleNamespace(shape=shape), action=(0, 0))
                trajectory = SimpleNamespace(trajectory_id=entry["trajectory_id"], transitions=(transition,) * 4)
                return trajectory, {}, {}

            with patch("tdyssey.gpc_denoiser_training.load_episode", fake):
                with self.assertRaisesRegex(ValueError, "mixed shapes"):
                    load_official_windows(root, ("train", "validation"), opened)
            self.assertEqual(opened, ["episode-0000.npz", "episode-0001.npz"])

    def test_phase_two_sigma_matches_the_pinned_training_file(self):
        source = ROOT / "third_party/gpc"
        if not (source / "world_model_train_phase_two/train.py").exists():
            self.skipTest("GPC phase-two source is not checked out")
        sigma = phase_two_sigma(source)
        self.assertEqual((sigma.loc, sigma.scale, sigma.sigma_min, sigma.sigma_max), (-1.2, 1.2, 0.002, 20))

    def test_logged_error_chooses_collection_training_or_stop(self):
        def record(**changes):
            base = {"contact_steps": 0, "adapted_mse": .02, "persistence_mse": .001, "previous_mse": .03,
                    "collection_runs": 1, "updates": 1, "steps": 4, "episodes": 8, "seed": 200000}
            base.update(changes)
            return base

        no_contact = decide(record())
        self.assertEqual(no_contact["action"], "collect")
        self.assertEqual((no_contact["steps"], no_contact["seed"]), (100, 200001))
        improved = decide(record(contact_steps=5, adapted_mse=.01, persistence_mse=.001, previous_mse=.02))
        self.assertEqual(improved["action"], "train")
        ready = decide(record(contact_steps=5, adapted_mse=.001, persistence_mse=.002))
        self.assertEqual(ready["action"], "stop")
        capped = decide(record(contact_steps=5, collection_runs=2, updates=20, previous_mse=.01, adapted_mse=.02))
        self.assertEqual(capped["action"], "stop")
        limited = limit_round(no_contact, rounds_completed=1, max_rounds=1)
        self.assertEqual(limited["action"], "collect")
        self.assertEqual(limited["executed"], "stop")
        self.assertIn("round cap", limited["executed_reason"])
        held = decide(record(contact_steps=5, adapted_mse=.02, previous_mse=.01, non_improving_rounds=1),
                      {"patience": 2})
        self.assertEqual(held["action"], "train")
        exhausted = decide(record(contact_steps=5, adapted_mse=.02, previous_mse=.01, non_improving_rounds=2),
                           {"patience": 2})
        self.assertEqual(exhausted["action"], "collect")
        self.assertEqual([dataset_name(index) for index in (1, 2, 3)], ["dataset", "dataset-002", "dataset-003"])

    def test_click_maps_into_the_workspace_and_history_is_padded(self):
        self.assertEqual(click_to_target(0, 0, 10, 10), (0, 0))
        self.assertEqual(click_to_target(5, 0, 10, 20), (256, 0))
        self.assertEqual(click_to_target(10, 10, 10, 10), (511, 511))
        self.assertEqual(click_to_target(-5, 30, 10, 10), (0, 511))
        images, actions = conditioning_window(["now"], [], ("click",), ("hold",))
        self.assertEqual(images, ["now", "now", "now", "now"])
        self.assertEqual(actions, [("hold",), ("hold",), ("hold",), ("click",)])

    def test_multiple_datasets_do_not_open_model_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "a", Path(temporary) / "b"
            first.mkdir()
            second.mkdir()
            manifest(first)
            manifest(second)
            opened = []

            def fake(_root, entry):
                transition = SimpleNamespace(before=SimpleNamespace(shape="T"), action=(0, 0))
                trajectory = SimpleNamespace(trajectory_id=entry["trajectory_id"], transitions=(transition,) * 4)
                return trajectory, {}, {}

            with patch("tdyssey.gpc_denoiser_training.load_episode", fake):
                with self.assertRaisesRegex(ValueError, "model_test"):
                    load_official_windows([first, second], ("model_test",), opened)
            self.assertEqual(opened, [])


if __name__ == "__main__":
    unittest.main()
