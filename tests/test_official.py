"""Source-only checks of official source integrity and namespace isolation."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey.official import load_official_source
from tdyssey.policy import hash_file


class OfficialSourceChecks(unittest.TestCase):
    def test_modified_source_is_rejected_before_any_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.py").write_text("x = 1\n")
            spec = {"source_commit": "a" * 40, "source_sha256": {"policy.py": hash_file(root / "policy.py")}}
            (root / "policy.py").write_text("x = 2\n")
            with patch("tdyssey.official.subprocess.check_output", return_value="a" * 40), patch("tdyssey.official.importlib.import_module") as imported:
                with self.assertRaisesRegex(ValueError, "Modified/missing"):
                    load_official_source(root, spec)
                imported.assert_not_called()

    def test_namespace_package_can_be_reused_but_other_locations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"source_commit": "a" * 40, "source_sha256": {}}
            namespace = SimpleNamespace(__file__=None, __path__=[str(root / "diffusion_policy")])
            previous = list(sys.path)
            try:
                with patch("tdyssey.official.subprocess.check_output", return_value="a" * 40), patch.dict(sys.modules, {"diffusion_policy": namespace}), patch("tdyssey.official.importlib.import_module", return_value=SimpleNamespace(PushTImageEnv="expected")):
                    self.assertEqual(load_official_source(root, spec), "expected")
                    namespace.__path__ = [str(root / "other")]
                    with self.assertRaisesRegex(ValueError, "Another diffusion_policy"):
                        load_official_source(root, spec)
            finally:
                sys.path[:] = previous


if __name__ == "__main__":
    unittest.main()
