"""Source checks for global contact sampling, frozen validation, and resume."""

import inspect
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tdyssey.gpc_denoiser_training import GPCDenoiserTrainer
from tdyssey.official_collection import ENVIRONMENT_FAMILY
from tdyssey.official_contact_trainer import (discover_batches, freeze_validation_keys,
                                              list_train_keys, load_selected_windows,
                                              sample_train_keys, stratify_validation_keys,
                                              train_contact_batches)
from tdyssey.policy import hash_file

ROOT = Path(__file__).resolve().parents[1]


def write_batch(root, name, status="complete"):
    path = root / name
    path.mkdir()
    entries = []
    for index, identifier in enumerate(("a", "b", "c")):
        entries.append({"trajectory_id": identifier, "control_steps": 4, "observation_count": 5,
                        "contact_steps": 0, "files": [
                            {"name": f"episode-{index:04d}.npz", "bytes": 1, "sha256": "a" * 64},
                            {"name": f"episode-{index:04d}.json", "bytes": 1, "sha256": "b" * 64}]})
    payload = {"schema_version": 1, "status": status,
               "config": {"episodes": 3, "shape": "T", "mode": "contact-balanced"},
               "provenance": {"environment_family": ENVIRONMENT_FAMILY},
               "episodes": entries,
               "splits": {"train": ["a"], "validation": ["b"], "model_test": ["c"]},
               "counts": {"reset_attempts": 3, "reset_physics_substeps": 3,
                          "control_steps": 12, "control_physics_substeps": 120},
               "rates": {"simulation_hz": 100, "control_hz": 10}}
    (path / "manifest.json").write_text(json.dumps(payload))
    return path


def fake_chunk(context, payload=b"updated"):
    (context["output"] / "denoiser.pt").write_bytes(payload)
    (context["output"] / "denoiser.json").write_text(json.dumps({"updates": context["updates"]}) + "\n")
    return {"updates": context["updates"], "weights_changed": True, "model_test_used": False,
            "validation": {"windows": len(context["validation_keys"]),
                           "before": {"mse": 0.02, "persistence_mse": 0.01},
                           "adapted": {"mse": 0.015, "persistence_mse": 0.01}},
            "opened_images": [], "resumed": True, "elapsed_seconds": 0.1,
            "replay": {"moved_windows": 1, "still_windows": 1, "moved_fraction": 0.6, "batches": []}}


class OfficialContactTrainerChecks(unittest.TestCase):
    def test_discover_batches_uses_numeric_order_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_batch(root, "batch-010")
            write_batch(root, "batch-002")
            write_batch(root, "batch-001")
            write_batch(root, "batch-003", status="running")
            (root / "model").mkdir()
            names = [path.name for path in discover_batches(root)]
            self.assertEqual(names, ["batch-001", "batch-002", "batch-010"])

    def test_train_keys_cover_every_batch_and_never_model_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batches = [write_batch(root, "batch-001"), write_batch(root, "batch-002"),
                       write_batch(root, "batch-010")]
            keys = list_train_keys(batches)
            self.assertEqual({key["batch"] for key in keys}, {"batch-001", "batch-002", "batch-010"})
            self.assertEqual({key["episode_id"] for key in keys}, {"a"})
            sampled = sample_train_keys(keys, 2, random.Random(0))
            self.assertEqual(len(sampled), 2)
            self.assertTrue(all(key["episode_id"] == "a" for key in sampled))
            opened = []
            with self.assertRaisesRegex(ValueError, "model_test"):
                load_selected_windows(root, [{"batch": "batch-001", "episode_id": "c", "anchor": 3}],
                                      opened, allowed_splits=("train",))
            self.assertEqual(opened, [])

    def test_frozen_validation_never_names_model_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batches = [write_batch(root, "batch-001"), write_batch(root, "batch-002")]
            keys = freeze_validation_keys(batches, count=4, seed=0)
            self.assertEqual(len(keys), 2)
            self.assertEqual({key["episode_id"] for key in keys}, {"b"})
            self.assertTrue(all(key["anchor"] == 3 for key in keys))
            opened = []
            with self.assertRaisesRegex(ValueError, "model_test"):
                load_selected_windows(root, [{"batch": "batch-001", "episode_id": "c", "anchor": 3}], opened)
            self.assertEqual(opened, [])

    def test_selected_windows_open_only_named_validation_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_batch(root, "batch-001")
            opened = []

            def fake(_root, entry):
                self.assertEqual(entry["trajectory_id"], "b")
                return object(), {}, {}

            with patch("tdyssey.official_contact_trainer.load_episode", fake):
                with patch("tdyssey.official_contact_trainer._windows",
                           lambda _trajectory, _arrays, _record=None: [{"episode_id": "b", "anchor": 3}]):
                    loaded = load_selected_windows(
                        root, [{"batch": "batch-001", "episode_id": "b", "anchor": 3}], opened)
            self.assertEqual(opened, ["episode-0001.npz"])
            self.assertEqual([window["anchor"] for window in loaded], [3])

    def test_interrupt_keeps_last_weights_and_resume_reuses_frozen_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            output = Path(temporary) / "train"
            init = Path(temporary) / "init.pt"
            data.mkdir()
            write_batch(data, "batch-001")
            write_batch(data, "batch-002")
            init.write_bytes(b"init")
            calls = {"n": 0}
            seen = []

            def first(context):
                calls["n"] += 1
                seen.append(context["sampled_keys"])
                self.assertTrue(all(key["episode_id"] == "a" for key in context["sampled_keys"]))
                self.assertGreaterEqual(len(context["train_keys"]), 2)
                if calls["n"] == 1:
                    return fake_chunk(context, b"after-one")
                raise KeyboardInterrupt

            report = train_contact_batches(
                data, output, init, ROOT / "third_party/gpc", resume=True,
                pool_windows=2, validation_windows=2, updates=8, train_chunk=first)
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["updates"], 8)
            self.assertEqual((output / "denoiser.pt").read_bytes(), b"after-one")
            self.assertEqual({key["episode_id"] for key in report["validation_keys"]}, {"b"})
            self.assertNotIn("c", [key["episode_id"] for key in report["validation_keys"]])

            def second(context):
                self.assertEqual(context["validation_keys"], report["validation_keys"])
                self.assertTrue(all(key["episode_id"] == "a" for key in context["sampled_keys"]))
                self.assertEqual(context["checkpoint"], output / "denoiser.pt")
                return fake_chunk(context, b"after-two")

            continued = train_contact_batches(
                data, output, init, ROOT / "third_party/gpc", resume=True,
                pool_windows=2, validation_windows=2, updates=8, max_updates=16,
                train_chunk=second)
            self.assertEqual(continued["status"], "stopped")
            self.assertEqual(continued["stop_reason"], "update cap")
            self.assertEqual(continued["validation_keys"], report["validation_keys"])
            self.assertEqual(continued["updates"], 16)
            self.assertEqual((output / "denoiser.pt").read_bytes(), b"after-two")
            with self.assertRaises(FileExistsError):
                train_contact_batches(data, output, init, ROOT / "third_party/gpc", resume=False,
                                      train_chunk=second)

    def test_new_run_does_not_inherit_source_sidecar_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            output = Path(temporary) / "train"
            source = Path(temporary) / "old"
            data.mkdir()
            source.mkdir()
            write_batch(data, "batch-001")
            write_batch(data, "batch-002")
            weights = source / "denoiser.pt"
            weights.write_bytes(b"k8-weights")
            (source / "denoiser.json").write_text(
                json.dumps({"updates": 2792, "init_sha256": "old" * 16}) + "\n")

            def once(context):
                sidecar = json.loads((output / "denoiser.json").read_text())
                self.assertEqual(sidecar["updates"], 0)
                self.assertEqual(sidecar["init_sha256"], hash_file(weights))
                self.assertNotEqual(sidecar["init_sha256"], "old" * 16)
                raise KeyboardInterrupt

            report = train_contact_batches(
                data, output, weights, ROOT / "third_party/gpc", resume=True,
                pool_windows=2, validation_windows=2, updates=8, train_chunk=once)
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["updates"], 0)
            self.assertEqual(report["source_checkpoint"], str(weights))
            self.assertEqual((output / "denoiser.pt").read_bytes(), b"k8-weights")
            sidecar = json.loads((output / "denoiser.json").read_text())
            self.assertEqual(sidecar["updates"], 0)
            self.assertEqual(sidecar["init_sha256"], hash_file(weights))

    def test_stratify_validation_takes_half_useful_and_half_miss(self):
        keys = [{"id": index} for index in range(8)]
        flags = [True, True, True, True, False, False, False, False]
        selected = stratify_validation_keys(keys, flags, 4, seed=0)
        self.assertEqual(len(selected), 4)
        useful = sum(1 for key in selected if key["id"] < 4)
        self.assertEqual(useful, 2)
        self.assertEqual(len(selected) - useful, 2)

    def test_trainer_run_accepts_preloaded_windows(self):
        parameters = inspect.signature(GPCDenoiserTrainer.run).parameters
        self.assertIn("validation", parameters)
        self.assertIn("training", parameters)
        self.assertIsNone(parameters["validation"].default)
        self.assertIsNone(parameters["training"].default)


if __name__ == "__main__":
    unittest.main()
