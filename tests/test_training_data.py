"""Source-only checks of window indexing and split membership."""

import json
from pathlib import Path
import tempfile
import unittest

from tdyssey.dataset import read_manifest
from tdyssey.training_data import WindowSpec, check_membership


class TrainingProtocolChecks(unittest.TestCase):
    def test_windows_have_exact_history_and_future_without_padding(self):
        spec = WindowSpec(history=2, future=3)
        self.assertEqual(list(spec.anchors(3)), [])
        self.assertEqual(list(spec.anchors(4)), [1])
        self.assertEqual(list(spec.anchors(6)), [1, 2, 3])
        self.assertEqual(list(WindowSpec(1, 1).anchors(3)), [0, 1, 2])

    def test_duplicate_trajectories_and_initial_scene_leakage_are_rejected(self):
        first = {"trajectory_id": "a", "split": "train", "initial_state_sha256": "scene-1"}
        with self.assertRaisesRegex(ValueError, "Duplicate trajectory"):
            check_membership([first, first])
        with self.assertRaisesRegex(ValueError, "initial state crosses"):
            check_membership([first, {**first, "trajectory_id": "b", "split": "validation"}])
        check_membership([first, {**first, "trajectory_id": "b"}])  # Same scene may stay within one split.
        check_membership([first, {**first, "trajectory_id": "c", "split": "model_test", "initial_state_sha256": "scene-2"}])

    def test_manifest_checks_membership_and_counts_without_importing_numpy(self):
        value = {"schema_version": 1, "status": "complete", "config": {"episodes": 1},
                 "episodes": [{"trajectory_id": "a", "control_steps": 2, "observation_count": 3,
                               "files": [{"name": "e.json"}, {"name": "e.npz"}]}],
                 "splits": {"train": ["a"], "validation": [], "model_test": []},
                 "counts": {"control_steps": 2, "control_physics_substeps": 20,
                            "reset_attempts": 1, "reset_physics_substeps": 1},
                 "rates": {"simulation_hz": 100, "control_hz": 10}}
        with tempfile.TemporaryDirectory(prefix="tdyssey-manifest-test-") as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps(value))
            self.assertEqual(read_manifest(root), value)
            value["splits"]["validation"] = ["a"]
            (root / "manifest.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                read_manifest(root)
            value["splits"]["validation"] = []
            value["counts"]["control_physics_substeps"] = 19
            (root / "manifest.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "accounting"):
                read_manifest(root)


if __name__ == "__main__":
    unittest.main()
