"""Source-only resource integrity and action-budget checks; no network access."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.fetch_policy_assets import fetch_checkpoint
from tdyssey.baseline import validate_budget
from tdyssey.policy import action_prefix, policy_spec, validate_horizons


class PolicyProtocolChecks(unittest.TestCase):
    def test_action_prefix_alignment_and_last_chunk_budget(self):
        config = policy_spec()["policy"]
        self.assertEqual(action_prefix(list(range(16)), config, 300), list(range(1, 9)))
        self.assertEqual(action_prefix(list(range(16)), config, 3), [1, 2, 3])
        with self.assertRaises(ValueError):
            action_prefix(list(range(15)), config, 8)
        with self.assertRaises(ValueError):
            action_prefix(list(range(16)), config, 0)

    def test_inconsistent_horizons_and_unbounded_runs_are_rejected(self):
        config = policy_spec()["policy"]
        for change in ({"obs_horizon": 0}, {"pred_horizon": True}, {"action_horizon": 16}):
            with self.assertRaises(ValueError):
                validate_horizons({**config, **change})
        validate_budget(["T", "I", "L"], 1, 300, 100000, 360)
        for shapes, episodes, steps, seed, seconds in ((["T", "T"], 1, 10, 100000, 60),
                (["T", "I", "L"], 2, 300, 100000, 60), (["T"], 1, 10, 42, 60),
                (["T"], 1, 301, 100000, 60), (["T"], 1, 10, 100000, 901)):
            with self.assertRaises(ValueError):
                validate_budget(shapes, episodes, steps, seed, seconds)

    def test_verified_download_and_reuse_without_network_or_duplicates(self):
        data = b"small-checkpoint-fixture"
        spec = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "repo_id": "fixture/model", "revision": "a" * 40, "path": "model.pth"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pth"
            with patch("scripts.fetch_policy_assets.urllib.request.urlopen", return_value=io.BytesIO(data)) as network:
                self.assertEqual(fetch_checkpoint(spec, target, 100), "downloaded_verified")
                self.assertEqual(fetch_checkpoint(spec, target, 100), "already_verified")
                self.assertEqual(network.call_count, 1)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_budget_bad_hash_and_oversized_stream_leave_no_published_asset(self):
        spec = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest(),
                "repo_id": "fixture/model", "revision": "a" * 40, "path": "model.pth"}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pth"
            with patch("scripts.fetch_policy_assets.urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "budget"):
                    fetch_checkpoint(spec, target, 2)
                network.assert_not_called()
            for data in (b"bad", b"abc-extra", b"ab"):
                with patch("scripts.fetch_policy_assets.urllib.request.urlopen", return_value=io.BytesIO(data)):
                    with self.assertRaises(ValueError):
                        fetch_checkpoint(spec, target, 100)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_wrong_or_incomplete_files_are_preserved(self):
        spec = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.pth"
            target.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                fetch_checkpoint(spec, target, 100)
            self.assertEqual(target.read_bytes(), b"bad")
            target.unlink()
            part = target.with_name("model.pth.part")
            part.write_bytes(b"previous-part")
            with self.assertRaises(FileExistsError):
                fetch_checkpoint(spec, target, 100)
            self.assertEqual(part.read_bytes(), b"previous-part")


if __name__ == "__main__":
    unittest.main()
