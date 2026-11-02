"""CPU accounting, exact checkpoint state and causal sampling; no optimizer steps."""
from copy import deepcopy
from collections import OrderedDict
import json
from pathlib import Path
import random
import threading
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.native_rgb_training import (BatchRenderer, BudgetExceeded, CostLedger, WindowOrder,
    normalized_config, restore_checkpoint, save_training_checkpoint, sha256, write_source_snapshot)

ROOT = Path(__file__).resolve().parents[1]


def temporary_directory():
    root = ROOT / "artifacts"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="native-rgb-cpu-test-", dir=root)


class DatasetDouble:
    entries = [{"shape": "A"}, {"shape": "B"}]
    windows = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4)]

    def __len__(self):
        return len(self.windows)


class NativeAccountingChecks(unittest.TestCase):
    def test_durable_attempts_and_legacy_cost_cannot_reset(self):
        with temporary_directory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = CostLedger(path, 2, 30)
            with ledger.start("engineering") as stage:
                stage.reserve_update()
                stage.complete_update()
                stage.reserve_update()  # An overflow/interruption still spends this attempt.
            value = CostLedger(path, 2, 30).snapshot()
            self.assertEqual((value["charged_updates"], value["completed_updates"]), (2, 1))
            self.assertFalse(value["historical_cancelled_experiment"]["included_in_native_rgb_budget"])
            with ledger.start("main") as stage:
                with self.assertRaises(BudgetExceeded):
                    stage.reserve_update()
            with self.assertRaises(ValueError):
                CostLedger(path, 3, 30)

    def test_unfinished_stage_needs_explicit_monotonic_reconciliation(self):
        with temporary_directory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = CostLedger(path, 5, 30)
            stage = ledger.start("probe_forward_backward_without_update")
            stage.heartbeat()
            with self.assertRaises(RuntimeError):
                CostLedger(path, 5, 30).start("main")
            with self.assertRaises(ValueError):
                ledger.recover_interrupted(-1)
            ledger.recover_interrupted(2.5)
            value = ledger.snapshot()
            self.assertEqual(value["charged_updates"], 0)
            self.assertEqual(value["charged_seconds"], 2.5)
            self.assertIsNone(value["active_stage"])

    def test_sampling_balances_letters_and_restores_order(self):
        dataset = DatasetDouble()
        sampler = WindowOrder(dataset)
        first = sampler.take(4)
        self.assertEqual([dataset.entries[dataset.windows[i][0]]["shape"] for i in first], ["A", "B", "A", "B"])
        state = sampler.state_dict()
        expected = sampler.take(17)
        restored = WindowOrder(dataset)
        restored.load_state_dict(state)
        self.assertEqual(restored.take(17), expected)
        state["pools"]["A"].pop()
        with self.assertRaises(ValueError):
            restored.load_state_dict(state)

    def test_configuration_keeps_native_contract(self):
        self.assertEqual(normalized_config({})["lr"], 1e-4)
        for value in ({"seed": 0}, {"microbatch": 8}, {"effective_batch": 8}, {"lr": 3e-4}):
            with self.assertRaises(ValueError):
                normalized_config(value)
        server = normalized_config(dict(local_pilot=False, seed=5479, effective_batch=128,
                                         microbatch=64, max_updates=100000, max_train_seconds=172800))
        self.assertEqual(server["microbatch"], 64)

    def test_independent_trajectories_are_balanced_despite_unequal_lengths(self):
        class UnequalDataset:
            entries = [{"shape": "A"}, {"shape": "A"}]
            windows = [(0, i) for i in range(100)] + [(1, 3)]

            def __len__(self):
                return len(self.windows)
        dataset = UnequalDataset()
        order = WindowOrder(dataset)
        selected = order.take(40)
        self.assertEqual(sum(dataset.windows[index][0] == 0 for index in selected), 20)
        self.assertEqual(sum(dataset.windows[index][0] == 1 for index in selected), 20)


class NativeCheckpointChecks(unittest.TestCase):
    def test_complete_cpu_state_roundtrip_without_an_optimizer_update(self):
        import numpy as np
        import torch
        from tdyssey.raw_world_model_training import make_generator
        torch.set_num_threads(1)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # Populate representative Adam state directly: .step() is never called.
        for parameter in model.parameters():
            optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=torch.ones_like(parameter) * .3,
                                              exp_avg_sq=torch.ones_like(parameter) * .7)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
        generator = make_generator(5478, "cpu")
        order = WindowOrder(DatasetDouble())
        order.take(7)
        config, identity = {"seed": 5478}, {"manifest_sha256": "a" * 64}
        expected_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        progress = {"updates": 7, "samples": 112}
        with temporary_directory() as directory:
            path = Path(directory) / "last.pt"
            saved = save_training_checkpoint(path, model, optimizer, scaler, scheduler, generator,
                                              order, progress, config, identity)
            self.assertEqual(saved["updates"], 7)
            expected = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator), order.take(11))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(5)
            restored = restore_checkpoint(path, model, optimizer, scaler, scheduler, generator, order, config, identity)
            actual = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, generator=generator), order.take(11))
            self.assertEqual(restored, progress)
            self.assertEqual(expected[:2], actual[:2])
            self.assertTrue(torch.equal(expected[2], actual[2]))
            self.assertTrue(torch.equal(expected[3], actual[3]))
            self.assertEqual(expected[4], actual[4])
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected_weights[name]))
            for parameter in model.parameters():
                self.assertEqual(float(optimizer.state[parameter]["step"]), 7.)
            self.assertFalse(torch.cuda.is_initialized())
            with self.assertRaises(ValueError):
                restore_checkpoint(path, model, optimizer, scaler, scheduler, generator, order, {"seed": 1}, identity)


class NativeRenderingChecks(unittest.TestCase):
    def test_prefetch_is_bounded_and_worker_caches_environments_are_independent(self):
        import numpy as np
        import torch
        barrier = threading.Barrier(2)

        class RenderingDataset:
            def __init__(self):
                self.entries, self.windows, self.manifest = ({"shape": "A"},), ((0, 3),), {"fixed": True}
                self._images, self._environments = OrderedDict(), {}
                self._image_bytes, self._trajectory_cache, self.cache_bytes = 0, None, 1024

            def __getitem__(self, index):
                if not self._environments:
                    self._environments["thread"] = threading.get_ident()
                    barrier.wait(timeout=5)
                self._images[index] = index
                self._trajectory_cache = index
                return {"image": np.asarray([index], dtype=np.float32),
                        "action": np.asarray([index / 10], dtype=np.float32)}

        dataset = RenderingDataset()
        renderer = BatchRenderer(dataset, workers=2, cache_bytes=128, max_pending=4)
        try:
            batches = list(renderer.batches([0, 1, 2, 3], 2, "cpu"))
            self.assertEqual([batch["image"].tolist() for batch in batches], [[[0.], [1.]], [[2.], [3.]]])
            self.assertEqual(len(renderer.clones), 2)
            first, second = renderer.clones
            self.assertIs(first.entries, dataset.entries)
            self.assertIs(first.windows, dataset.windows)
            self.assertIs(first.manifest, dataset.manifest)
            self.assertIsNot(first._images, second._images)
            self.assertIsNot(first._environments, second._environments)
            self.assertNotEqual(first._environments["thread"], second._environments["thread"])
            self.assertEqual(sum(clone.cache_bytes for clone in renderer.clones), 128)
            self.assertFalse(dataset._images)
            self.assertFalse(dataset._environments)
            self.assertIsNone(dataset._trajectory_cache)
            self.assertFalse(renderer.active)
            with self.assertRaises(ValueError):
                list(renderer.batches([0, 1, 2, 3, 4], 1, "cpu"))
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            renderer.close()

    def test_source_snapshot_preserves_bytes_and_rejects_replacement(self):
        with temporary_directory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("value=1\n", encoding="utf-8")
            identity = {"source": {"files": {"module.py": sha256(source)}}}
            with patch("tdyssey.native_rgb_training.ROOT", root):
                write_source_snapshot(root / "run", identity)
                source.write_text("value=2\n", encoding="utf-8")
                write_source_snapshot(root / "run", identity)
                self.assertEqual((root / "run/source-snapshot/module.py").read_text(), "value=1\n")
                changed = {"source": {"files": {"module.py": sha256(source)}}}
                with self.assertRaises(ValueError):
                    write_source_snapshot(root / "run", changed)


if __name__ == "__main__":
    unittest.main()
