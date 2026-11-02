"""Check supported native platforms and pins without research-package imports."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("doctor", ROOT / "scripts/doctor.py")
DOCTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCTOR)
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
PYTHON_VERSION = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
PINS = dict(requirement.split("==", 1) for requirement in (
    PROJECT["dependencies"] + PROJECT["optional-dependencies"]["cpu"]
))


class DoctorChecks(unittest.TestCase):
    def run_doctor(self, system="win32", machine="AMD64", pointer_bytes=8,
                   python_version=PYTHON_VERSION, versions=None):
        output = io.StringIO()
        with (
            patch.object(DOCTOR.sys, "argv", ["doctor.py"]),
            patch.object(DOCTOR.sys, "platform", system),
            patch.object(DOCTOR.platform, "machine", return_value=machine),
            patch.object(DOCTOR.platform, "python_version", return_value=python_version),
            patch.object(DOCTOR.struct, "calcsize", return_value=pointer_bytes),
            patch.object(DOCTOR.metadata, "version", side_effect=(versions or PINS).__getitem__),
            contextlib.redirect_stdout(output),
        ):
            status = DOCTOR.main()
        return status, json.loads(output.getvalue())

    def test_windows_and_linux_pass_metadata_checks(self):
        for system, machine in (("win32", "AMD64"), ("linux", "x86_64")):
            with self.subTest(system=system, machine=machine):
                status, report = self.run_doctor(system=system, machine=machine)
                self.assertEqual(status, 0)
                self.assertEqual(report["errors"], [])
                self.assertEqual(report["python_bits"], 64)
                self.assertFalse(report["runtime_requested"])

    def test_unsupported_platform_and_32_bit_python_are_rejected(self):
        for system, machine, pointer_bytes in (
            ("win32", "AMD64", 4), ("win32", "ARM64", 8),
            ("linux", "aarch64", 8), ("darwin", "x86_64", 8),
        ):
            with self.subTest(system=system, machine=machine, pointer_bytes=pointer_bytes):
                status, report = self.run_doctor(system=system, machine=machine,
                                                 pointer_bytes=pointer_bytes)
                self.assertEqual(status, 1)
                self.assertTrue(any("Supported runtime platforms" in error
                                    for error in report["errors"]))

    def test_windows_still_enforces_python_and_package_pins(self):
        versions = {**PINS, "numpy": "2.0.0"}
        status, report = self.run_doctor(python_version="3.11.4", versions=versions)
        self.assertEqual(status, 1)
        self.assertIn(f"Expected Python {PYTHON_VERSION}", report["errors"])
        self.assertTrue(any("Package mismatch: numpy" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
