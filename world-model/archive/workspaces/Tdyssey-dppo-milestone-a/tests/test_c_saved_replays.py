"""Stdlib checks for honest saved-frame indices and no-overwrite export gate."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_replays", ROOT / "scripts/inspect_c_saved_replays.py")
REPLAYS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAYS)


class SavedReplayChecks(unittest.TestCase):
    def test_frame_mapping_preserves_episode_local_control_indices(self):
        records = [{"control_index": 41}, {"control_index": 42}]
        self.assertEqual(REPLAYS.frame_mapping(records, 3), [
            {"frame_index": 0, "relation": "before", "episode_control_index": 41},
            {"frame_index": 1, "relation": "after", "episode_control_index": 41},
            {"frame_index": 2, "relation": "after", "episode_control_index": 42}])
        for controls, count in ((records, 2), ([], 1), ([{"control_index": 1}, {"control_index": 3}], 3),
                                ([{"control_index": True}], 2), ([{"control_index": 0}], 2)):
            with self.subTest(controls=controls, count=count), self.assertRaises(ValueError):
                REPLAYS.frame_mapping(controls, count)

    def test_existing_output_refused_before_loading_tensor_or_image_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(REPLAYS, "load_replay") as load, \
                    patch.object(sys, "argv", ["inspect", "--output-dir", directory]), \
                    self.assertRaises(FileExistsError):
                REPLAYS.main()
            load.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_help_import_remains_source_only(self):
        result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "scripts/inspect_c_saved_replays.py"), "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no simulation/model", result.stdout)


if __name__ == "__main__":
    unittest.main()
