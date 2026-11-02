"""CPU tensors and a lightweight spy; no network/checkpoint/GPU/optimizer."""
import ast
from collections import namedtuple
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest

import torch

from tdyssey.native_rgb_r3a_conditioning import (
    make_conditioning_rngs, make_conditioning_plan, conditioning_residual_loss)
from tdyssey.raw_world_model_training import make_noise_plan, make_generator

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent/'Tdyssey/third_party/gpc_code/gpc_rank_evaluation/diffusion/denoiser.py'


def native_conditioner_body():
    """Execute the real upstream method body without importing/constructing a net."""
    tree = ast.parse(UPSTREAM.read_text(encoding='utf-8'))
    add_dims = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'add_dims')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Denoiser')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_conditioners')
    fields = ('c_in', 'c_out', 'c_skip', 'c_noise', 'c_noise_cond')
    namespace = dict(torch=torch, Tensor=torch.Tensor, Optional=Optional, Conditioners=namedtuple('Conditioners', fields))
    exec(compile(ast.Module(body=[add_dims, method], type_ignores=[]), str(UPSTREAM), 'exec'), namespace)
    return namespace['compute_conditioners']


class SpyDenoiser:
    cfg = SimpleNamespace(sigma_data=.5, sigma_offset_noise=.1, noise_previous_obs=True,
        inner_model=SimpleNamespace(num_steps_conditioning=4))
    is_upsampler = False
    native_conditioners = staticmethod(native_conditioner_body())

    def __init__(self, output_dtype=torch.float32):
        self.anchor = torch.zeros(())
        self.output_dtype = output_dtype
        self.condition_calls = []
        self.output_calls = []

    def parameters(self):
        yield self.anchor

    def compute_conditioners(self, sigma, sigma_cond):
        self.condition_calls.append((sigma.clone(), None if sigma_cond is None else sigma_cond.clone()))
        cs = self.native_conditioners(self, sigma, sigma_cond)
        return cs

    def compute_model_output(self, noisy_target, history, actions, cs):
        self.output_calls.append((noisy_target.clone(), history.clone(), actions.clone(), cs))
        return torch.zeros_like(noisy_target, dtype=self.output_dtype)


class ConditioningContracts(unittest.TestCase):
    def batch(self, b=4):
        rng = torch.Generator().manual_seed(5478)
        return dict(image=torch.rand((b, 5, 3, 96, 96), generator=rng),
            action=torch.linspace(-1, 1, b*8).reshape(b, 4, 2))

    def rngs(self, branch_seed=0):
        return make_conditioning_rngs(branch_seed=branch_seed, target_seed=200, weak_seed=300)

    def assert_same_plan(self, a, b):
        self.assertEqual((a.branch, a.branch_uniform, a.rng_before, a.rng_after),
            (b.branch, b.branch_uniform, b.rng_before, b.rng_after))
        for field in ('sigma_cond', 'history_offset', 'history_epsilon', 'sigma', 'target_offset', 'target_epsilon'):
            left, right = getattr(a, field), getattr(b, field)
            if left is None:
                self.assertIsNone(right)
            else:
                self.assertTrue(torch.equal(left, right), field)

    def test_clean_is_actual_None_and_native_zero_encoding_without_pixel_changes(self):
        batch = self.batch(); rngs = self.rngs(); plan = make_conditioning_plan(batch, rngs)
        self.assertEqual(plan.branch, 'clean'); self.assertIsNone(plan.sigma_cond)
        self.assertEqual(plan.rng_before['weak'], plan.rng_after['weak'])
        self.assertFalse(bool(plan.history_epsilon.any())); self.assertFalse(bool(plan.history_offset.any()))
        model = SpyDenoiser(); loss, details = conditioning_residual_loss(model, batch, noise_plan=plan)
        self.assertIsNone(model.condition_calls[0][1])  # Not a sigma=1 substitute.
        _, history, actions, cs = model.output_calls[0]
        self.assertTrue(torch.equal(history, batch['image'][:, :4].reshape(4, 12, 96, 96)))
        self.assertTrue(torch.equal(actions, batch['action']))
        self.assertTrue(torch.equal(cs.c_noise_cond, torch.zeros(4)))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(details['sigma_cond_is_none']); self.assertFalse(details['actions_renormalized'])

    def test_weak_loguniform_per_sample_uses_real_sigma_encoding_and_zero_offset(self):
        batch = self.batch(); rngs = self.rngs(1); plan = make_conditioning_plan(batch, rngs)
        self.assertEqual(plan.branch, 'weak'); self.assertIsNotNone(plan.sigma_cond)
        self.assertEqual(plan.sigma_cond.shape, (4,)); self.assertFalse(bool(plan.history_offset.any()))
        expected_rng = make_generator(300)
        u = torch.rand((4,), generator=expected_rng)
        # Compare exact scalar math used by the implementation (Python log constants).
        import math
        expected_sigma = (math.log(.002)+u*(math.log(.05)-math.log(.002))).exp().clamp(.002, .05)
        expected_epsilon = torch.randn((4, 12, 96, 96), generator=expected_rng)
        self.assertTrue(torch.equal(plan.sigma_cond, expected_sigma))
        self.assertTrue(torch.equal(plan.history_epsilon, expected_epsilon))
        self.assertGreater(float(plan.sigma_cond.min()), .002)
        self.assertLess(float(plan.sigma_cond.max()), .05)
        self.assertGreater(len(torch.unique(plan.sigma_cond)), 1)
        model = SpyDenoiser(); conditioning_residual_loss(model, batch, noise_plan=plan)
        self.assertTrue(torch.equal(model.condition_calls[0][1], plan.sigma_cond))
        _, history, actions, cs = model.output_calls[0]
        self.assertTrue(torch.equal(cs.c_noise_cond, plan.sigma_cond.log()/4))
        self.assertTrue(torch.equal(history, batch['image'][:, :4].reshape(4, 12, 96, 96)+plan.history_epsilon*plan.sigma_cond[:, None, None, None]))
        self.assertTrue(torch.equal(actions, batch['action']))

    def test_target_stream_is_independent_of_clean_weak_branch(self):
        batch = self.batch(); clean = make_conditioning_plan(batch, self.rngs(0)); weak = make_conditioning_plan(batch, self.rngs(1))
        self.assertEqual(clean.branch, 'clean'); self.assertEqual(weak.branch, 'weak')
        for field in ('sigma', 'target_offset', 'target_epsilon'):
            self.assertTrue(torch.equal(getattr(clean, field), getattr(weak, field)))
        self.assertEqual(clean.rng_after['target'], weak.rng_after['target'])

    def test_target_native_draw_subsequence_matches_old_plan_when_rng_is_aligned(self):
        batch = self.batch(); b = len(batch['image']); old_rng = make_generator(200)
        original = make_noise_plan(batch, old_rng)
        # Explicitly align to the old target substream, after its history draws.
        # New independent streams are not falsely claimed to be old full streams.
        aligned = make_generator(200)
        torch.randn((b,), generator=aligned)
        torch.randn((b, 12, 1, 1), generator=aligned)
        torch.randn((b, 12, 96, 96), generator=aligned)
        rngs = self.rngs(); rngs.target.set_state(aligned.get_state())
        new = make_conditioning_plan(batch, rngs)
        for field in ('sigma', 'target_offset', 'target_epsilon'):
            self.assertTrue(torch.equal(getattr(original, field), getattr(new, field)), field)
        self.assertTrue(torch.equal(old_rng.get_state(), rngs.target.get_state()))

    def test_fp32_residual_MSE_has_native_target_and_no_target_noise_in_history_condition(self):
        batch = self.batch(); plan = make_conditioning_plan(batch, self.rngs(1)); model = SpyDenoiser(torch.bfloat16)
        loss, _ = conditioning_residual_loss(model, batch, noise_plan=plan)
        noisy_target, _, _, cs = model.output_calls[0]
        expected_input = batch['image'][:, 4]+plan.target_offset+plan.target_epsilon*plan.sigma[:, None, None, None]
        self.assertTrue(torch.equal(noisy_target, expected_input))
        residual = (batch['image'][:, 4]-cs.c_skip*expected_input)/cs.c_out
        expected = torch.nn.functional.mse_loss(torch.zeros_like(residual), residual)
        self.assertTrue(torch.equal(loss, expected)); self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.equal(model.condition_calls[0][1], plan.sigma_cond))
        # Changing target epsilon cannot change the history conditioning code.
        modified = replace(plan, target_epsilon=plan.target_epsilon+1)
        other = SpyDenoiser(); conditioning_residual_loss(other, batch, noise_plan=modified)
        self.assertTrue(torch.equal(other.condition_calls[0][1], model.condition_calls[0][1]))
        self.assertTrue(torch.equal(other.output_calls[0][3].c_noise_cond, cs.c_noise_cond))

    def test_all_three_rng_states_restore_every_plan_value_and_do_not_touch_global_rng(self):
        batch = self.batch(); rngs = self.rngs(1); saved = rngs.state_dict()
        global_state = torch.random.get_rng_state().clone()
        a = make_conditioning_plan(batch, rngs); b = make_conditioning_plan(batch, rngs)
        rngs.load_state_dict(saved)
        self.assert_same_plan(a, make_conditioning_plan(batch, rngs))
        self.assert_same_plan(b, make_conditioning_plan(batch, rngs))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_state))
        before = rngs.hashes(); bad = deepcopy(saved); bad['states']['weak'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'checksum'): rngs.load_state_dict(bad)
        self.assertEqual(before, rngs.hashes())

    def test_absolute_actions_and_fake_clean_sigma_are_rejected(self):
        batch = self.batch(); rngs = self.rngs(); before = rngs.hashes()
        bad = dict(batch, action=torch.full_like(batch['action'], 511.))
        with self.assertRaisesRegex(ValueError, 'Expected constant'): make_conditioning_plan(bad, rngs)
        self.assertEqual(before, rngs.hashes())
        plan = make_conditioning_plan(batch, rngs)
        with self.assertRaisesRegex(ValueError, 'actual None'):
            conditioning_residual_loss(SpyDenoiser(), batch, noise_plan=replace(plan, sigma_cond=torch.ones(4)))


if __name__ == '__main__':
    unittest.main()
