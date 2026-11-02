"""Stdlib-only tiny child scripts; no models, environment, budget or CUDA."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts/guarded_c_child.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_unit_checks", BOOTSTRAP)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BootstrapChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.handshake = self.directory / "identity.json"
        self.ack = self.directory / "ack.json"
        self.marker = self.directory / "target.json"
        self.target = self.directory / "tiny_target.py"
        self.target.write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'argv': sys.argv, "
            "'torch_imported': 'torch' in sys.modules, "
            "'budget_imported': 'tdyssey.policy_protocol' in sys.modules}))\n",
            encoding="utf-8")

    def command(self):
        return [str(BOOTSTRAP), "--handshake", str(self.handshake), "--ack", str(self.ack),
                "--nonce", "test-fresh-nonce", "--", str(self.target), str(self.marker),
                "argument with spaces", "--literal-target-option"]

    def start(self):
        child = subprocess.Popen([sys.executable, "-S", "-B", *self.command()], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def cleanup():
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=2)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 2
        while not self.handshake.exists():
            if child.poll() is not None:
                stdout, stderr = child.communicate(timeout=2)
                self.fail(f"Tiny bootstrap exited before handshake: {stdout} {stderr}")
            if time.monotonic() >= deadline:
                self.fail("Tiny bootstrap did not publish its bounded handshake")
            time.sleep(.005)
        identity = json.loads(self.handshake.read_text(encoding="utf-8"))
        self.assertEqual(identity["nonce"], "test-fresh-nonce")
        self.assertIs(type(identity["pid"]), int)
        self.assertGreater(identity["pid"], 0)
        self.assertFalse(self.marker.exists(), "Target code must wait until ownership ACK")
        return child, identity

    def acknowledge(self, identity, *, nonce="test-fresh-nonce"):
        temporary = self.directory / "ack.tmp"
        temporary.write_text(json.dumps({"pid": identity["pid"], "nonce": nonce}), encoding="utf-8")
        os.replace(temporary, self.ack)

    def test_good_ack_runs_in_handshake_pid_with_exact_target_argv(self):
        child, identity = self.start()
        self.acknowledge(identity)
        stdout, stderr = child.communicate(timeout=2)
        self.assertEqual(child.returncode, 0, stdout + stderr)
        executed = json.loads(self.marker.read_text(encoding="utf-8"))
        self.assertEqual(executed["pid"], identity["pid"])
        self.assertEqual(executed["argv"], [str(self.target), str(self.marker),
                                         "argument with spaces", "--literal-target-option"])
        self.assertFalse(executed["torch_imported"])
        self.assertFalse(executed["budget_imported"])
        self.assertFalse(self.handshake.with_suffix(".tmp").exists())

    def test_wrong_nonce_rejects_without_running_target(self):
        child, identity = self.start()
        self.acknowledge(identity, nonce="wrong-nonce")
        stdout, stderr = child.communicate(timeout=2)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("Parent ACK identity differs", stderr)
        self.assertFalse(self.marker.exists())

    def test_no_ack_deadline_fails_before_target_without_real_wait(self):
        with (patch.object(MODULE.sys, "argv", ["bootstrap", *self.command()[1:]]),
              patch.object(MODULE.time, "monotonic", side_effect=[100.0, 130.0]),
              patch.object(MODULE.time, "sleep") as sleep):
            with self.assertRaisesRegex(TimeoutError, "no target code started"):
                MODULE.main()
        self.assertFalse(self.marker.exists())
        self.assertTrue(self.handshake.is_file())
        self.assertFalse(self.ack.exists())
        sleep.assert_not_called()

    def test_target_nonzero_exit_propagates_instead_of_false_success(self):
        self.target.write_text(self.target.read_text(encoding="utf-8") + "raise SystemExit(7)\n", encoding="utf-8")
        child, identity = self.start()
        self.acknowledge(identity)
        child.communicate(timeout=2)
        self.assertEqual(child.returncode, 7)
        self.assertEqual(json.loads(self.marker.read_text())["pid"], identity["pid"])


if __name__ == "__main__":
    unittest.main()
