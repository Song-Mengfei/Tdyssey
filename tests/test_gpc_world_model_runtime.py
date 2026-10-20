"""CUDA parity with the actual released checkpoint; no asset download in tests."""

import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("TDYSSEY_GPC_WORLD_CHECKPOINT") and os.environ.get("TDYSSEY_GPC_ROOT"),
                     "Set existing GPC source and world-model checkpoint paths")
class GPCWorldModelRuntimeChecks(unittest.TestCase):
    def test_wrapper_matches_upstream_and_preserves_imports_and_cuda_rng(self):
        import numpy as np
        import torch
        from tdyssey.gpc_world_model import GPCWorldModel

        if not torch.cuda.is_available():
            self.skipTest("CUDA required for pretrained world-model inference")
        torch.set_num_threads(2)
        sentinels = {name: ModuleType(name) for name in ("diffusion", "data", "utils")}
        with patch.dict(sys.modules, sentinels):
            model = GPCWorldModel(Path(os.environ["TDYSSEY_GPC_ROOT"]),
                                  Path(os.environ["TDYSSEY_GPC_WORLD_CHECKPOINT"]))
            for name, sentinel in sentinels.items():
                self.assertIs(sys.modules[name], sentinel)
        images = np.random.default_rng(4).random((4,3,96,96), dtype=np.float32)
        past = np.array([[0,511], [100,200], [200,300]], dtype=np.float32)
        proposed = np.array([[300,400], [450,100]], dtype=np.float32)
        before = torch.cuda.get_rng_state().clone()
        actual = model.predict(images, past, proposed, seed=9)
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        history = torch.as_tensor(images, device="cuda")[None]
        actions = np.concatenate((past, proposed))
        expected = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            torch.cuda.manual_seed(9)
            for index in range(2):
                window = torch.as_tensor(actions[index:index+4], device="cuda")[None] / 511*2-1
                prediction, _ = model.sampler.sample(history, window)
                expected.append(prediction[0].cpu().numpy())
                history = torch.cat((history[:,1:], prediction[:,None]), dim=1)
        np.testing.assert_array_equal(actual, np.stack(expected))
        changed = model.predict(images, past, proposed[:1], seed=9)
        np.testing.assert_array_equal(actual[:1], changed)
        self.assertEqual(model.parameter_count, 8729827)
        self.assertEqual(model.state_entries, 298)


if __name__ == "__main__":
    unittest.main()
