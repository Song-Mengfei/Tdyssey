"""The asset audit distinguishes file serialization from tensor-state changes."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from scripts.audit_frozen_wm_assets import compare_summaries, discover_assets, tensor_summary


class FrozenAssetChecks(unittest.TestCase):
    def test_inventory_does_not_enter_heldout_or_dependency_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["artifacts/best.pt", "data/pilot.catalog.json",
                         "model-test/best.pt", "model_test/best.pt", ".venv/best.pt"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"filenames only")
            result = discover_assets([root])
            self.assertEqual(result["custom_checkpoints"], [str((root / "artifacts/best.pt").resolve())])
            self.assertEqual(result["catalogs"], [str((root / "data/pilot.catalog.json").resolve())])

    def test_key_or_tensor_change_cannot_be_labelled_serialization_only(self):
        left = {"tensors": {"a": {"sha256": "same"}, "b": {"sha256": "left"}}}
        right = {"tensors": {"a": {"sha256": "same"}, "b": {"sha256": "right"}}}
        result = compare_summaries(left, right)
        self.assertFalse(result["identical_tensor_state"])
        self.assertEqual(result["changed_keys"], ["b"])
        result = compare_summaries(left, {"tensors": {"a": {"sha256": "same"}}})
        self.assertFalse(result["same_state_keys"])
        self.assertEqual(result["left_only_keys"], ["b"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "installed torch required")
    def test_same_tensors_have_same_canonical_digest_across_serializations(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            torch.save(state, root / "one.pt")
            torch.save(state, root / "different-filename.pt")
            first, second = tensor_summary(root / "one.pt"), tensor_summary(root / "different-filename.pt")
            self.assertEqual(first["state_fingerprint_sha256"], second["state_fingerprint_sha256"])
            self.assertTrue(compare_summaries(first, second)["identical_tensor_state"])
            state["weight"][0, 0] = 42
            torch.save(state, root / "changed.pt")
            self.assertFalse(compare_summaries(first, tensor_summary(root / "changed.pt"))["identical_tensor_state"])


if __name__ == "__main__":
    unittest.main()
