"""Tiny spies executing pinned native bodies; no real Torch/pixels/weights."""
import ast
from collections import namedtuple
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List, Optional, Tuple
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy unavailable")
import numpy as np
from tdyssey.gpc_conditioning_contract import make_conditioning_sampler, SAMPLER_SPEC

ROOT = Path(__file__).resolve().parents[1]
GPC = Path(os.environ.get("TDYSSEY_GPC_ROOT", ROOT.parent / "Tdyssey/third_party/gpc_code"))
NATIVE_AVAILABLE = all((GPC / ("gpc_rank_evaluation/diffusion/" + name)).is_file()
                       for name in ("denoiser.py", "diffusion_sampler.py"))


class ArrayTensor(np.ndarray):
    def __new__(cls, value, dtype=np.float32): return np.asarray(value, dtype=dtype).view(cls)
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return ArrayTensor(value, getattr(value, "dtype", np.float32))
    def __iter__(self): return (self[index] for index in range(len(self)))
    @property
    def device(self): return "cpu"
    def size(self): return self.shape
    def detach(self): return self
    def cpu(self): return self
    def new_zeros(self, shape): return ArrayTensor(np.zeros(shape, np.float32))
    def log(self): return ArrayTensor(np.log(np.asarray(self)))
    def sqrt(self): return ArrayTensor(np.sqrt(np.asarray(self)))
    def clamp(self, low, high): return ArrayTensor(np.clip(np.asarray(self), low, high))
    def add(self, value): return ArrayTensor(np.asarray(self) + value)
    def sub(self, value): return ArrayTensor(np.asarray(self) - value)
    def div(self, value): return ArrayTensor(np.asarray(self) / value)
    def mul(self, value): return ArrayTensor(np.asarray(self) * value)
    def byte(self): return np.asarray(self, dtype=np.uint8).view(ArrayTensor)


class FakeTorch:
    """Fixed arithmetic target; no random generator or real framework."""
    device = str
    def __init__(self): self.draws, self.full_calls = [], []
    def randn(self, *shape, **kwargs):
        values = np.linspace(-.7, .7, math.prod(shape), dtype=np.float32).reshape(shape)
        self.draws.append(values.copy())
        return ArrayTensor(values)
    def randn_like(self, value): raise AssertionError("unexpected extra/history noise draw")
    def ones(self, count, **kwargs): return ArrayTensor(np.ones(count, np.float32))
    def zeros_like(self, value): return ArrayTensor(np.zeros_like(np.asarray(value)))
    def linspace(self, start, stop, count, **kwargs):
        return ArrayTensor(np.linspace(start, stop, count, dtype=np.float32))
    def cat(self, values): return ArrayTensor(np.concatenate(values))
    def full(self, shape, *, fill_value, device):
        self.full_calls.append(dict(shape=tuple(shape), value=fill_value, device=device))
        return ArrayTensor(np.full(shape, fill_value, np.float32))


def native_source(torch):
    """Hash-check and extract actual bodies, omitting imports/NN/decorators."""
    spec = json.loads((ROOT / "manifests/gpc_world_model.json").read_bytes())
    nodes = []
    for filename in ("denoiser.py", "diffusion_sampler.py"):
        relative = "gpc_rank_evaluation/diffusion/" + filename
        source = (GPC / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != spec["source_sha256"][relative]:
            raise AssertionError("native source differs from fixed manifest")
        tree = ast.parse(source.decode())
        if filename == "denoiser.py":
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Denoiser")
            nodes.append(deepcopy(next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "add_dims")))
            for name in ("compute_conditioners", "compute_model_output", "wrap_model_output", "denoise"):
                node = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
                node.decorator_list = []
                nodes.append(node)
        else:
            sampler = deepcopy(next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "DiffusionSampler"))
            for node in sampler.body:
                if isinstance(node, ast.FunctionDef): node.decorator_list = []
            nodes.extend([sampler, deepcopy(next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "build_sigmas"))])
    namespace = dict(torch=torch, Tensor=ArrayTensor, Optional=Optional, List=List, Tuple=Tuple,
        Denoiser=object, DiffusionSamplerConfig=SimpleNamespace,
        Conditioners=namedtuple("Conditioners", "c_in c_out c_skip c_noise c_noise_cond"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<pinned-native-conditioning-methods>", "exec"), namespace)

    class OrdinaryDenoiser:
        device = "cpu"
        def __init__(self):
            self.cfg = SimpleNamespace(sigma_data=.5, sigma_offset_noise=.1)
            self.conditioner_calls, self.model_calls, self.wrap_calls, self.inner_calls = [], [], [], []
        def inner_model(self, noisy, noise, noise_cond, history, action):
            self.inner_calls.append(dict(noisy=np.array(noisy), noise=np.array(noise),
                noise_cond=np.array(noise_cond), history=np.array(history), action=np.array(action)))
            return ArrayTensor(.1 * np.asarray(noisy) + .1 * np.asarray(history[:, :3])
                + .2 * np.asarray(noise_cond)[:, None, None, None])
        def apply_noise(self, *args, **kwargs): raise AssertionError("history noise forbidden")
    for name in ("compute_conditioners", "compute_model_output", "wrap_model_output", "denoise"):
        setattr(OrdinaryDenoiser, name, namespace[name])

    def model():
        instance = OrdinaryDenoiser()
        for name, target in (("compute_conditioners", "conditioner_calls"),
                             ("compute_model_output", "model_calls"), ("wrap_model_output", "wrap_calls")):
            original = namespace[name]
            def spy(self, *args, _name=name, _target=target, _original=original):
                result = _original(self, *args)
                getattr(self, _target).append(result if _name == "compute_conditioners" else args[-1])
                return result
            setattr(instance, name, MethodType(spy, instance))
        return instance

    def config(**values): return SimpleNamespace(s_tmin=0, s_tmax=float("inf"), s_noise=1, **values)
    return SimpleNamespace(Denoiser=OrdinaryDenoiser, DiffusionSampler=namespace["DiffusionSampler"],
                           DiffusionSamplerConfig=config), model


@unittest.skipUnless(NATIVE_AVAILABLE, "pinned source cache unavailable")
class ConditioningContractTests(unittest.TestCase):
    def inputs(self):
        history = ArrayTensor(np.linspace(.1, .8, 48, dtype=np.float32).reshape(1, 4, 3, 2, 2))
        actions = ArrayTensor(np.linspace(-.8, .8, 8, dtype=np.float32).reshape(1, 4, 2))
        return history, actions

    def execute_arm(self, arm, observer=None):
        torch = FakeTorch()
        source, factory = native_source(torch)
        model = factory()
        sampler = make_conditioning_sampler(model, source, arm, torch_module=torch, observer=observer)
        history, actions = self.inputs()
        before = history.tobytes(), actions.tobytes()
        result, trajectory = sampler.sample(history, actions)
        self.assertEqual((history.tobytes(), actions.tobytes()), before)
        return source, model, sampler, torch, result, trajectory

    def test_A_matches_original_native_sampler_exactly(self):
        source, model, sampler, torch, result, trajectory = self.execute_arm("A")
        original_torch = FakeTorch()
        original_source, factory = native_source(original_torch)
        original_model = factory()
        original = original_source.DiffusionSampler(original_model, original_source.DiffusionSamplerConfig(**SAMPLER_SPEC))
        expected, expected_trajectory = original.sample(*self.inputs())
        self.assertIs(type(sampler), source.DiffusionSampler)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(len(trajectory), 4)
        for actual, expected in zip(trajectory, expected_trajectory):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(torch.draws), 1)
        self.assertEqual(torch.full_calls, [])
        np.testing.assert_array_equal(torch.draws[0], original_torch.draws[0])
        self.assertEqual(len(sampler.denoiser.records), 3)

    def test_B_changes_only_noise_conditioner_preserving_clean_inputs_and_draws(self):
        _, model_a, sampler_a, torch_a, result_a, _ = self.execute_arm("A")
        _, model_b, sampler_b, torch_b, result_b, _ = self.execute_arm("B")
        self.assertEqual((len(model_a.conditioner_calls), len(model_b.conditioner_calls)), (3, 3))
        self.assertEqual((len(model_b.model_calls), len(model_b.wrap_calls)), (3, 3))
        for actual, model_cs, wrap_cs in zip(model_b.conditioner_calls, model_b.model_calls, model_b.wrap_calls):
            self.assertIs(actual, model_cs)
            self.assertIs(actual, wrap_cs)
        for index, (a, b) in enumerate(zip(sampler_a.denoiser.records, sampler_b.denoiser.records)):
            self.assertIsNone(a["sigma_cond"])
            np.testing.assert_allclose(b["sigma_cond"], [.002], rtol=1e-7)
            self.assertEqual(a["sigma"], b["sigma"])
            np.testing.assert_array_equal(a["sigma"], np.asarray(sampler_a.sigmas[index]).reshape(-1))
            for key in ("c_in", "c_out", "c_skip", "c_noise"):
                self.assertEqual(a["conditioners"][key], b["conditioners"][key])
            self.assertEqual(a["conditioners"]["c_noise_cond"], [0.])
            np.testing.assert_allclose(b["conditioners"]["c_noise_cond"], [math.log(.002)/4], rtol=1e-7)
        for a, b in zip(model_a.inner_calls, model_b.inner_calls):
            np.testing.assert_array_equal(a["history"], b["history"])
            np.testing.assert_array_equal(a["action"], b["action"])
        np.testing.assert_array_equal(model_a.inner_calls[0]["noisy"], model_b.inner_calls[0]["noisy"])
        self.assertEqual((len(torch_a.draws), len(torch_b.draws)), (1, 1))
        np.testing.assert_array_equal(torch_a.draws[0], torch_b.draws[0])
        self.assertEqual(len(torch_b.full_calls), 3)
        self.assertFalse(np.array_equal(result_a, result_b))

    def test_observer_receives_scalar_copy_not_model_tensor_or_retained_record(self):
        seen = []
        def observer(value):
            seen.append(json.loads(json.dumps(value, allow_nan=False)))
            value["conditioners"]["c_in"][0] = 12345
        _, _, sampler, _, _, _ = self.execute_arm("B", observer)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen, sampler.denoiser.records)
        self.assertNotEqual(sampler.denoiser.records[0]["conditioners"]["c_in"][0], 12345)

    def test_sampler_drift_and_direct_history_noise_fail_before_model_call(self):
        torch = FakeTorch()
        source, factory = native_source(torch)
        model = factory()
        sampler = make_conditioning_sampler(model, source, "B", torch_module=torch)
        with self.assertRaises(ValueError):
            sampler.denoiser.denoise(ArrayTensor(np.zeros((1, 3, 2, 2))), ArrayTensor(.5),
                ArrayTensor([.1]), ArrayTensor(np.zeros((1, 12, 2, 2))), None)
        with self.assertRaises(RuntimeError): sampler.denoiser.apply_noise(None, None)
        self.assertEqual(model.conditioner_calls, [])
        self.assertEqual(torch.full_calls, [])

    def test_lazy_import_and_invalid_arm_fail_before_torch_or_source_use(self):
        with patch("builtins.__import__", side_effect=AssertionError("no import allowed")):
            with self.assertRaises(PermissionError): make_conditioning_sampler(None, None, "A")
            with self.assertRaises(ValueError): make_conditioning_sampler(None, None, "unknown")


if __name__ == "__main__":
    unittest.main()
