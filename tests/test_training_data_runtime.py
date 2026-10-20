"""Installed NumPy/PyTorch checks using marker fixtures, not simulated results.

Set TDYSSEY_DATA_TESTS=1 to run without a GPC checkout. Full simulator test runs
also enable these tests through TDYSSEY_GPC_ROOT. No new packages are installed.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from tdyssey.collection import CollectedEpisode, CollectionConfig
from tdyssey.dataset import DatasetWriter
from tdyssey.geometry import Pose, get_shape
from tdyssey.state import BodyState, Snapshot
from tdyssey.training_data import WindowDataset, WindowSpec, read_catalog, write_catalog
from tdyssey.trajectories import Trajectory, Transition


@unittest.skipUnless(os.environ.get("TDYSSEY_DATA_TESTS") or os.environ.get("TDYSSEY_GPC_ROOT"),
                     "Set TDYSSEY_DATA_TESTS=1 for NumPy/PyTorch loader checks")
class TrainingDataRuntimeChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.temporary = tempfile.TemporaryDirectory(prefix="tdyssey-windows-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_fixture(self, name="run", lengths=(6,), splits=("train",), id_prefix=None, scene_offset=0):
        np = self.np
        root = self.root / name
        config = CollectionConfig(shape="I", episodes=len(lengths), steps=max(lengths))
        writer = DatasetWriter(root, config, {"kind": "marker_test_fixture_not_physical_data"})
        writer.manifest["rates"] = {"simulation_hz": 100, "control_hz": 10}
        membership = {"train": [], "validation": [], "model_test": []}
        for episode_index, (length, split) in enumerate(zip(lengths, splits)):
            scene = scene_offset + episode_index

            def state(step):
                return Snapshot("I", get_shape("I").geometry_hash, "a" * 64,
                                BodyState(Pose((200 + scene * 10, 200), 0)),
                                BodyState(Pose((100 + scene * 10 + step, 120), 0)),
                                Pose((256, 256), math.pi / 4), step,
                                None if step == 0 else (150 + step - 1, 170 + step - 1))

            transitions = tuple(Transition(state(t), (150 + t, 170 + t), state(t + 1), t / 10,
                                           False, t + 1 == length) for t in range(length))
            trajectory = Trajectory(f"{id_prefix or name}-{episode_index}", "fixture", "v1", scene, 0.1, transitions)
            diagnostics = [{"n_contacts": int(t % 2 == 0), "iou": 0.0, "coverage": 0.0,
                            "upstream_terminated": False, "upstream_truncated": False,
                            "collector_truncated": t + 1 == length} for t in range(length)]
            marker = 10 + episode_index * 20
            images = np.stack([np.full((3, 96, 96), marker + t, dtype=np.uint8) for t in range(length + 1)])
            states = [state(t) for t in range(length + 1)]
            episode = CollectedEpisode(trajectory, images,
                                       np.asarray([s.pusher_body.pose.position for s in states]),
                                       np.asarray([(*s.goal.position, s.goal.angle) for s in states]),
                                       diagnostics, [{"seed": scene, "issues": []}], scene, scene, "collection_horizon")
            writer.add(episode)
            membership[split].append(trajectory.trajectory_id)
        writer.manifest["counts"] = {"control_steps": sum(lengths), "control_physics_substeps": 10 * sum(lengths),
                                     "reset_attempts": len(lengths), "reset_physics_substeps": len(lengths)}
        writer.finish(membership)
        return root

    def catalog(self, roots):
        path = self.root / "catalog.json"
        write_catalog(roots, path)
        return path

    def test_window_alignment_uses_independent_frame_and_action_markers(self):
        root = self.run_fixture()
        dataset = WindowDataset(self.catalog([root]), spec=WindowSpec(2, 2))
        self.assertEqual(len(dataset), 4)
        first, last = dataset[0], dataset[-1]
        self.assertEqual((first["anchor"], last["anchor"]), (1, 4))
        self.np.testing.assert_allclose(first["observations"]["image"][:, 0, 0, 0], [10 / 255, 11 / 255])
        self.np.testing.assert_allclose(first["targets"]["image"][:, 0, 0, 0], [12 / 255, 13 / 255])
        self.np.testing.assert_array_equal(first["actions"], [[151, 171], [152, 172]])
        self.np.testing.assert_array_equal(first["targets"]["agent_pos"][:, 0], [102, 103])
        self.assertEqual(first["observations"]["image"].dtype, self.np.float32)
        self.assertTrue(last["labels"]["truncated"][-1])
        self.assertTrue(last["labels"]["collector_truncated"][-1])
        self.assertFalse(last["labels"]["upstream_truncated"][-1])
        first["observations"]["image"][:] = 0
        self.assertGreater(float(dataset[0]["observations"]["image"].max()), 0)  # Augmentation cannot corrupt cache.
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_short_episodes_are_skipped_and_splits_never_mix(self):
        root = self.run_fixture(lengths=(1, 5, 6), splits=("train", "train", "validation"))
        catalog = self.catalog([root])
        train = WindowDataset(catalog, "train", WindowSpec(2, 2))
        validation = WindowDataset(catalog, "validation", WindowSpec(2, 2))
        self.assertEqual(len(train), 3)
        self.assertEqual({train[i]["episode_id"] for i in range(len(train))}, {"run-1"})
        self.assertEqual({validation[i]["episode_id"] for i in range(len(validation))}, {"run-2"})
        self.assertEqual(len(WindowDataset(catalog, "model_test")), 0)
        self.assertEqual(train.summary()["control_steps"], 6)  # Source steps, not overlapping-window count.

    def test_duplicate_ids_and_repeated_initial_scene_leakage_are_rejected(self):
        first = self.run_fixture("one", id_prefix="same")
        duplicate = self.run_fixture("two", id_prefix="same")
        with self.assertRaisesRegex(ValueError, "Duplicate trajectory"):
            self.catalog([first, duplicate])
        second_scene = self.run_fixture("three", splits=("validation",), id_prefix="different")
        with self.assertRaisesRegex(ValueError, "initial state crosses"):
            self.catalog([first, second_scene])
        self.assertFalse((self.root / "catalog.json").exists())

    def test_frozen_source_and_split_changes_are_detected(self):
        root = self.run_fixture()
        catalog = self.catalog([root])
        source = root / "manifest.json"
        original = source.read_bytes()
        value = json.loads(original)
        value["elapsed_seconds"] = 999
        source.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            WindowDataset(catalog)
        source.write_bytes(original)
        value = json.loads(catalog.read_text())
        value["runs"][0]["episodes"][0]["split"] = "validation"
        catalog.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "split/summary"):
            read_catalog(catalog)

    def test_corrupt_images_are_rejected_on_lazy_load(self):
        root = self.run_fixture()
        dataset = WindowDataset(self.catalog([root]))
        image = root / "episode-0000.npz"
        image.write_bytes(image.read_bytes()[:-8])
        with self.assertRaisesRegex(ValueError, "hash/size"):
            dataset[0]

    def test_mixed_control_rates_and_catalog_overwrite_are_rejected(self):
        first = self.run_fixture("one")
        second = self.run_fixture("two", scene_offset=5)
        source = second / "manifest.json"
        value = json.loads(source.read_text())
        value["rates"]["simulation_hz"] = 200
        value["counts"]["control_physics_substeps"] *= 2
        source.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "same simulation/control"):
            self.catalog([first, second])
        catalog = self.catalog([first])
        original = catalog.read_bytes()
        with self.assertRaises(FileExistsError):
            write_catalog([first], catalog)
        self.assertEqual(catalog.read_bytes(), original)

    def test_pytorch_batches_preserve_shapes_dtypes_and_seeded_order(self):
        import torch
        from torch.utils.data import DataLoader

        dataset = WindowDataset(self.catalog([self.run_fixture()]), spec=WindowSpec(2, 2))
        def first_batch():
            return next(iter(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                                        generator=torch.Generator().manual_seed(7))))
        batch, repeated = first_batch(), first_batch()
        self.assertEqual(tuple(batch["observations"]["image"].shape), (2, 2, 3, 96, 96))
        self.assertEqual(tuple(batch["actions"].shape), (2, 2, 2))
        self.assertEqual(batch["observations"]["image"].dtype, torch.float32)
        self.assertEqual(batch["labels"]["truncated"].dtype, torch.bool)
        torch.testing.assert_close(batch["anchor"], repeated["anchor"])
        torch.testing.assert_close(batch["actions"], repeated["actions"])


if __name__ == "__main__":
    unittest.main()
