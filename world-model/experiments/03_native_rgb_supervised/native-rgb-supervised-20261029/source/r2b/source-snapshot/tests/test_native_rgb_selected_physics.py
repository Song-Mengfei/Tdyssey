"""Selected-default pointer tests; no real environment or physics calls."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_selected_physics as module


class SelectedPhysicsTests(unittest.TestCase):
    def test_exact_existing_protocol_and_external_display(self):
        config, metadata = module.load_selected_config(), module.selected_metadata()
        self.assertEqual(config["candidate_arm"], "D-half-life-0.1s")
        self.assertEqual(metadata["display_name"], "0.1秒半衰期（已选）")
        self.assertEqual(metadata["protocol_sha256"], "58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6")
        self.assertIn("candidate", metadata["protocol_version"])
        self.assertTrue(config["default_for_all_new_data_generation"])
        self.assertEqual(metadata["canonical_default"], "configs/native_rgb_default.json")

    def test_factory_delegates_existing_protocol_without_reset(self):
        sentinel = object()
        with patch.object(module, "make_r2a2_env", return_value=sentinel) as factory:
            self.assertIs(module.make_selected_env("mock-source", "T"), sentinel)
            factory.assert_called_once_with("mock-source", "T", "D-half-life-0.1s")

    def test_wrong_actual_identity_rejected_without_relabelling(self):
        identity = dict(arm=module.SELECTED_ARM, protocol_sha256=module.SELECTED_PROTOCOL_SHA256,
            physics=dict(space=dict(damping=module.SELECTED_DAMPING)))
        self.assertTrue(module.verify_selected_identity(identity))
        old = deepcopy(identity)
        old["physics"]["space"]["damping"] = 0.
        before = deepcopy(old)
        with self.assertRaises(ValueError):
            module.verify_selected_identity(old)
        self.assertEqual(old, before)

    def test_original_environment_sources_remain_byte_identical(self):
        root = Path(module.__file__).resolve().parents[1]
        for relative, expected in {
            "tdyssey/native_rgb_env.py": "47ee5c53ed3369da7548ea1e2512e5cf1eed315e68b290dcf2fa1ee7dd5500d5",
            "tdyssey/native_rgb_r2a_env.py": "0b2965801c310334ebfa0270eb33abb6da3e731d8f7b7b97ff24c6709c6642e4",
            "tdyssey/native_rgb_r2a2_env.py": "0ac37b2d9bbbe73937c5e3258e84be7648cb86a7362c54ceb2d532694d6f5088",
        }.items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
