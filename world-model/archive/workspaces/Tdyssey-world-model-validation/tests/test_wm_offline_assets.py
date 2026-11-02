"""Byte-only audits distinguish new diagnostic output from old evidence changes."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_wm_offline_assets import check_after, record_file, take_before


class OfflineAssetAuditChecks(unittest.TestCase):
    def fixture(self, parent):
        root = Path(parent) / "repo"
        artifacts = root / "artifacts"
        asset = artifacts / "frozen-wm-assets"
        asset.mkdir(parents=True)
        checkpoint = asset / "denoiser.pth"
        checkpoint.write_bytes(b"not a pickle: hashing must never deserialize this")
        raw = artifacts / "frozen-wm-validation-v1" / "raw.npz"
        raw.parent.mkdir()
        raw.write_bytes(b"not an NPZ: hashing must not load this either")
        media = root / "docs/media/world-model-2026-10-19"
        media.mkdir(parents=True)
        image = media / "image.png"
        image.write_bytes(b"fake PNG, bytes only")
        record = record_file(image)
        (media / "manifest.json").write_text(json.dumps({"files": [
            {"path": image.name, "bytes": record["bytes"], "sha256": record["sha256"]}]}))
        manifest = root / "manifests/gpc_world_model.json"
        manifest.parent.mkdir()
        pinned = record_file(checkpoint)
        manifest.write_text(json.dumps({"checkpoint": {"filename": checkpoint.name,
            "bytes": pinned["bytes"], "sha256": pinned["sha256"]}}))
        legacy = Path(parent) / "legacy.pth"
        legacy.write_bytes(b"legacy bytes")
        bc = Path(parent) / "other-checkout"
        bc.mkdir()
        ledger = bc / "original.json"
        ledger.write_bytes(b"opaque ledger bytes; do not read semantics")
        expected = record_file(ledger)
        baseline = Path(parent) / "bc-baseline.json"
        baseline.write_text(json.dumps({"protected_checkout": str(bc), "files": [
            {"path": ledger.name, "bytes": expected["bytes"], "sha256": expected["sha256"]}]}))
        audit_dir = artifacts / "wm-interface-v2-audit"
        return root, raw, baseline, legacy, audit_dir

    def test_all_old_opaque_bytes_stay_immutable_and_new_diagnostics_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, raw, baseline, legacy, audit = self.fixture(temporary)
            before, result = take_before(root, audit, baseline, legacy)
            output = root / "artifacts/wm-interface-v2-diagnostic"
            output.mkdir()
            (output / "new.json").write_text("new development result")
            after = check_after(before, audit / "after.json")
            self.assertTrue(result["all_prechecks_pass"])
            self.assertTrue(after["all_old_assets_unchanged"])
            self.assertIn(output.name, after["allowed_new_diagnostic_directories"])
            self.assertEqual(raw.read_bytes(), b"not an NPZ: hashing must not load this either")
            with self.assertRaises(FileExistsError):
                take_before(root, audit, baseline, legacy)

    def test_same_length_rewrite_cannot_hide_behind_unchanged_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, raw, baseline, legacy, audit = self.fixture(temporary)
            before, _ = take_before(root, audit, baseline, legacy)
            original = raw.read_bytes()
            raw.write_bytes(b"X" * len(original))
            after = check_after(before, audit / "after.json")
            self.assertFalse(after["all_old_assets_unchanged"])
            self.assertEqual(len(after["changed"]), 1)
            self.assertEqual(after["changed"][0]["before"]["bytes"], after["changed"][0]["after"]["bytes"])

    def test_deletion_and_addition_inside_old_root_are_not_new_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, raw, baseline, legacy, audit = self.fixture(temporary)
            before, _ = take_before(root, audit, baseline, legacy)
            raw.unlink()
            (raw.parent / "extra.npz").write_bytes(b"unauthorized addition")
            after = check_after(before, audit / "after.json")
            self.assertFalse(after["all_old_assets_unchanged"])
            self.assertEqual(after["missing"], [str(raw.resolve())])
            self.assertEqual(after["added_inside_protected_roots"], [str((raw.parent / "extra.npz").resolve())])

    def test_new_directory_outside_declared_diagnostic_prefix_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, baseline, legacy, audit = self.fixture(temporary)
            before, _ = take_before(root, audit, baseline, legacy)
            (root / "artifacts/unexpected-experiment").mkdir()
            after = check_after(before, audit / "after.json")
            self.assertFalse(after["all_old_assets_unchanged"])
            self.assertEqual(after["unexpected_new_artifact_directories"], ["unexpected-experiment"])

    def test_preexisting_B_C_change_is_rejected_without_replacing_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, baseline, legacy, audit = self.fixture(temporary)
            expected_bytes = baseline.read_bytes()
            (Path(temporary) / "other-checkout/original.json").write_bytes(b"changed before start")
            with self.assertRaisesRegex(ValueError, "B/C assets already differ"):
                take_before(root, audit, baseline, legacy)
            self.assertEqual(baseline.read_bytes(), expected_bytes)
            self.assertFalse((audit / "before.json").exists())

    def test_audit_cannot_write_its_own_output_into_protected_old_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, raw, baseline, legacy, audit = self.fixture(temporary)
            before, _ = take_before(root, audit, baseline, legacy)
            forbidden = raw.parent / "audit-report.json"
            with self.assertRaisesRegex(ValueError, "wm-interface-v2"):
                check_after(before, forbidden)
            self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
