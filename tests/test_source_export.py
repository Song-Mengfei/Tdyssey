"""Portable source exports must retain the same pinned-byte trust boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tdyssey.source_export import EXPORT_MANIFEST, verify_source


class ExportChecks(unittest.TestCase):
    def fixture(self, root):
        (root / "source.py").write_bytes(b"original\n")
        spec = dict(source_commit="a" * 40,
                    source_sha256={"source.py": hashlib.sha256(b"original\n").hexdigest()})
        (root / EXPORT_MANIFEST).write_text(json.dumps(
            dict(kind="tdyssey.pinned-source-export.v1", **spec)), encoding="utf-8")
        return spec

    def test_export_bytes_and_commit_are_both_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = self.fixture(root)
            self.assertEqual(verify_source(root, spec)["verification"], "hash_verified_export")
            (root / "source.py").write_bytes(b"changed\n")
            with self.assertRaises(ValueError): verify_source(root, spec)

    def test_export_cannot_self_authorize_another_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = self.fixture(root)
            provenance = json.loads((root / EXPORT_MANIFEST).read_text())
            provenance["source_commit"] = "b" * 40
            (root / EXPORT_MANIFEST).write_text(json.dumps(provenance))
            with self.assertRaises(ValueError): verify_source(root, spec)

    def test_inventory_cannot_point_outside_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = self.fixture(root)
            spec["source_sha256"] = {"../outside.py": "0" * 64}
            (root / EXPORT_MANIFEST).write_text(json.dumps(
                dict(kind="tdyssey.pinned-source-export.v1", **spec)))
            with self.assertRaises(ValueError): verify_source(root, spec)
