"""Atomic JSON sharing-violation retries only; no Torch, model or optimizer."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.native_rgb_training import atomic_json

ROOT = Path(__file__).resolve().parents[1]


class AtomicJSONRetryChecks(unittest.TestCase):
    def setUp(self):
        sandbox = (ROOT/"artifacts").resolve()
        sandbox.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="native-rgb-io-cpu-test-", dir=sandbox)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(sandbox))

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to((ROOT/"artifacts").resolve()))
        self.temporary.cleanup()

    def test_two_temporary_denials_then_one_successful_atomic_replace(self):
        path = self.directory/"ledger.json"
        original = b'{"old":true}\n'
        path.write_bytes(original)
        replace = os.replace
        attempts = []

        def transient(source, destination):
            attempts.append((source, destination))
            self.assertEqual(path.read_bytes(), original)
            if len(attempts) <= 2:
                raise PermissionError("transient sharing violation")
            return replace(source, destination)

        with patch("tdyssey.native_rgb_training.os.replace", side_effect=transient):
            atomic_json(path, {"completed_updates": 877})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(path.read_text()), {"completed_updates": 877})
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_persistent_denial_keeps_old_bytes_and_cleans_temporary_file(self):
        path = self.directory/"ledger.json"
        original = b'{"completed_updates":876}\n'
        path.write_bytes(original)
        elapsed = [0.]

        def advance(seconds):
            self.assertGreaterEqual(seconds, 0.)
            elapsed[0] += seconds

        with patch("tdyssey.native_rgb_training.time.monotonic", side_effect=lambda: elapsed[0]), \
             patch("tdyssey.native_rgb_training.time.sleep", side_effect=advance), \
             patch("tdyssey.native_rgb_training.os.replace", side_effect=PermissionError("persistent sharing violation")) as replace:
            with self.assertRaisesRegex(PermissionError, "persistent sharing violation"):
                atomic_json(path, {"completed_updates": 877})
        self.assertGreater(replace.call_count, 1)
        self.assertLessEqual(elapsed[0], 2.+1e-9)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
