"""Remote-route parity with NumPy/arithmetic spies; no Torch, model or physics."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy is required for arithmetic parity checks")
import numpy as np

from scripts import train_official_contact as entry
from tdyssey import gpc_denoiser_training as training
from tdyssey.motion_replay import planned_replay_take, sample_replay_batch
from tdyssey.official_contact_trainer import preserve_init_checkpoint, train_contact_batches
from tdyssey.region_weight import foreground_weights
from tests.test_official_contact_trainer import write_batch

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT.parent / "Tdyssey-remote-review"
REMOTE_AVAILABLE = (REMOTE/"tdyssey/gpc_denoiser_training.py").is_file()


def remote_node(filename, name):
    tree = ast.parse((REMOTE / "tdyssey" / filename).read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


class Tensor(np.ndarray):
    def __new__(cls, value): return np.asarray(value, dtype=np.float32).view(cls)
    def size(self): return self.shape
    def expand_as(self, other): return Tensor(np.broadcast_to(self, other.shape))
    def sum(self, *args, **kwargs): return Tensor(np.asarray(self).sum(*args, **kwargs))
    def clamp_min(self, value): return Tensor(np.maximum(np.asarray(self), value))


class LossNetwork:
    def __init__(self):
        self.cfg = SimpleNamespace(inner_model=SimpleNamespace(num_steps_conditioning=4),
                                   noise_previous_obs=True, sigma_offset_noise=.1)
        self.calls, self.sigma_calls = [], 0
    def sample_sigma_training(self, batch, device):
        self.sigma_calls += 1
        return Tensor(np.full(batch, .2 if self.sigma_calls == 1 else .4))
    def apply_noise(self, value, sigma, offset):
        self.calls.append(("noise", np.asarray(value).copy(), np.asarray(sigma).copy(), offset))
        return Tensor(value + sigma.reshape((-1, 1, 1, 1)) + offset)
    def compute_conditioners(self, sigma, sigma_cond):
        self.calls.append(("condition", np.asarray(sigma).copy(), np.asarray(sigma_cond).copy()))
        return SimpleNamespace(c_skip=Tensor([[[[.25]]]]), c_out=Tensor([[[[.5]]]]))
    def compute_model_output(self, noisy, history, actions, conditioners):
        self.calls.append(("model", np.asarray(history).copy(), np.asarray(actions).copy()))
        return Tensor(np.full_like(noisy, .3))


class OfficialContactParityTests(unittest.TestCase):
    def test_read_only_and_help_require_no_torch_or_filesystem_output(self):
        original_import = __import__
        def forbid(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("read-only/help must not import Torch")
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.__import__", side_effect=forbid):
            output = Path(temporary)/"not-created"
            with redirect_stdout(io.StringIO()):
                plan = entry.main(["--datasets", str(Path(temporary)/"not-read"), "--output", str(output)])
                with self.assertRaises(SystemExit) as stopped:
                    entry.main(["--help"])
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(plan["mode"], "read_only")
            self.assertFalse(plan["training_executed"])
            self.assertFalse(output.exists())
            with self.assertRaises(PermissionError):
                training.GPCDenoiserTrainer("unused", "unused").run("unused", output)
            with self.assertRaises(PermissionError):
                train_contact_batches("unused", output, "unused", "unused")

    @unittest.skipUnless(REMOTE_AVAILABLE,"read-only remote review checkout unavailable")
    def test_region_and_replay_modules_are_exact_remote_files(self):
        for name in ("region_weight.py", "motion_replay.py"):
            self.assertEqual((ROOT/"tdyssey"/name).read_bytes(), (REMOTE/"tdyssey"/name).read_bytes())

    @unittest.skipUnless(REMOTE_AVAILABLE,"read-only remote review checkout unavailable")
    def test_weighted_loss_matches_remote_body_and_numpy_foreground_reduction(self):
        remote = remote_node("gpc_denoiser_training.py", "weighted_denoiser_loss")
        local_tree = ast.parse((ROOT/"tdyssey/gpc_denoiser_training.py").read_text(encoding="utf-8"))
        local = next(node for node in local_tree.body if isinstance(node,ast.FunctionDef) and node.name==remote.name)
        self.assertEqual(ast.dump(local),ast.dump(remote))
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[remote],type_ignores=[])),
                     "<remote-weighted-loss>","exec"),namespace)
        image = Tensor(np.linspace(0,1,5*3*96*96,dtype=np.float32).reshape(1,5,3,96,96))
        actions = Tensor(np.linspace(-1,1,8,dtype=np.float32).reshape(1,4,2))
        region = np.zeros((96,96),bool)
        region[10,10] = region[30,70] = True
        weights = Tensor(foreground_weights(region)[None])
        fake_torch = SimpleNamespace(isfinite=lambda value: bool(np.isfinite(np.asarray(value)).all()))
        first, second = LossNetwork(), LossNetwork()
        with patch.dict(sys.modules,{"torch":fake_torch}):
            actual = training.weighted_denoiser_loss(first,image,actions,weights,"marker")
            expected = namespace["weighted_denoiser_loss"](second,image,actions,weights,"marker")
        np.testing.assert_array_equal(actual,expected)
        target = (np.asarray(image[:,4])-.25*(np.asarray(image[:,4])+.4+.1))/.5
        manual = np.square(.3-target)[:,:,region].mean()
        self.assertAlmostEqual(float(actual),float(manual),places=6)
        self.assertEqual([row[0] for row in first.calls],["noise","noise","condition","model"])
        for given,wanted in zip(first.calls[-1][1:],second.calls[-1][1:]):
            np.testing.assert_array_equal(given,wanted)

    def test_replay_preserves_moved_fraction_and_short_pool_rules(self):
        moved=[dict(id=n,moved=True) for n in range(20)]
        still=[dict(id=n,moved=False) for n in range(20,40)]
        batch,mix=sample_replay_batch(moved,still,10,.6,random.Random(5478))
        self.assertEqual((mix["moved"],mix["still"]),(6,4))
        self.assertEqual(sum(item["moved"] for item in batch),6)
        self.assertEqual(planned_replay_take(1,0,4,.6),dict(moved=1,still=0,replace_moved=3,replace_still=0,requested_moved=2))

    @unittest.skipUnless(REMOTE_AVAILABLE,"read-only remote review checkout unavailable")
    def test_fake_chunks_match_remote_sampling_validation_and_update_order(self):
        spec=importlib.util.spec_from_file_location("tdyssey._remote_contact_parity",REMOTE/"tdyssey/official_contact_trainer.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            data=root/"data"
            data.mkdir()
            write_batch(data,"batch-001")
            write_batch(data,"batch-002")
            checkpoint=root/"init.pt"
            checkpoint.write_bytes(b"synthetic-init")
            seen=[]
            def chunk(context):
                seen.append((context["seed"],context["sampled_keys"],context["validation_keys"]))
                (context["output"]/"denoiser.pt").write_bytes(b"synthetic-updated")
                return dict(updates=8,weights_changed=True,model_test_used=False,
                    validation=dict(before=dict(mse=.02),adapted=dict(mse=.01,persistence_mse=.015)))
            arguments=dict(pool_windows=2,validation_windows=2,updates=8,max_updates=16,train_chunk=chunk)
            with redirect_stdout(io.StringIO()):
                remote=module.train_contact_batches(data,root/"remote",checkpoint,"unused",**arguments)
                remote_seen=list(seen)
                seen.clear()
                local=train_contact_batches(data,root/"local",checkpoint,"unused",execute=True,**arguments)
            self.assertEqual(seen,remote_seen)
            for key in ("status","updates","rounds","validation_keys","train_windows","batches","model_test_used"):
                self.assertEqual(local[key],remote[key])
            self.assertEqual((root/"local/denoiser.pt").read_bytes(),(root/"remote/denoiser.pt").read_bytes())

    def test_initializer_copy_remains_copy_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/"source"
            source.mkdir()
            (source/"denoiser.pt").write_bytes(b"synthetic-weights")
            (source/"denoiser.json").write_text("{}\n")
            first=preserve_init_checkpoint(source/"denoiser.pt",root/"init")
            self.assertEqual(first["copied"],["denoiser.pt","denoiser.json"])
            (source/"denoiser.pt").write_bytes(b"changed")
            second=preserve_init_checkpoint(source/"denoiser.pt",root/"init")
            self.assertEqual(second["copied"],[])
            self.assertEqual((root/"init/denoiser.pt").read_bytes(),b"synthetic-weights")


if __name__=="__main__":
    unittest.main()
