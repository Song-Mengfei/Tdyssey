"""One-command official T collection and a single GPC denoiser update."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def pipeline():
    spec = importlib.util.spec_from_file_location(
        "train_official_world_model", ROOT / "scripts/train_official_world_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("TDYSSEY_OFFICIAL_ROOT") and os.environ.get("TDYSSEY_GPC_ROOT")
                     and os.environ.get("TDYSSEY_GPC_WORLD_CHECKPOINT"),
                     "Set official source, GPC source and the pinned denoiser checkpoint")
class OfficialWorldModelRuntimeChecks(unittest.TestCase):
    def test_one_command_collects_and_updates_without_reading_model_test(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for the denoiser update")
        module = pipeline()
        real = module.train_collected

        def corrupt_then_train(args, datasets, checkpoint=None, updates=None):
            roots = datasets if isinstance(datasets, (list, tuple)) else [datasets]
            for root in roots:
                manifest = json.loads((Path(root) / "manifest.json").read_text())
                held_out = set(manifest["splits"]["model_test"])
                for entry in manifest["episodes"]:
                    if entry["trajectory_id"] in held_out:
                        name = next(item["name"] for item in entry["files"] if item["name"].endswith(".npz"))
                        (Path(root) / name).write_bytes(b"unread model-test payload")
                        self.held_out = name
                        self.dataset_root = Path(root)
            return real(args, datasets, checkpoint, updates)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            module.train_collected = corrupt_then_train
            report = module.main([
                "--output", str(output), "--episodes", "8", "--steps", "4", "--updates", "1",
                "--workers", "2", "--seconds", "180", "--train-seconds", "180", "--max-rounds", "1",
                "--source-root", os.environ["TDYSSEY_OFFICIAL_ROOT"],
                "--gpc-root", os.environ["TDYSSEY_GPC_ROOT"],
                "--checkpoint", os.environ["TDYSSEY_GPC_WORLD_CHECKPOINT"]])
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["splits"], {"train": 6, "validation": 1, "model_test": 1})
            training = report["training"]
            self.assertFalse(training["model_test_used"])
            self.assertTrue(training["weights_changed"])
            self.assertEqual(training["updates"], 1)
            self.assertNotIn(self.held_out, training["opened_images"])
            self.assertEqual(self.dataset_root, output / "dataset")
            self.assertTrue((self.dataset_root / self.held_out).read_bytes().startswith(b"unread"))
            self.assertEqual(training["validation"]["windows"], 1)
            for stage in ("frozen", "adapted"):
                metrics = training["validation"][stage]
                for name in ("mse", "persistence_mse"):
                    self.assertEqual(metrics[name], metrics[name])
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["status"], "complete")


if __name__ == "__main__":
    unittest.main()
