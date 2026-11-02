"""Tiny CPU substitutes only; root executes Torch checks in one process.

No real GPC constructor, checkpoint, source loader, sampler call, policy, image
asset or simulator is used. The substitute is two scalars, not a custom CNN.
"""
import builtins
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Tiny objective checks require the installed Torch profile")
import torch
from tdyssey.gpc_supervised_adaptation import (NoisePlan, make_generator, make_noise_plan,
    supervised_denoising_loss, validation_loss, select_trainable_parameters, make_optimizer,
    make_real_denoiser, make_sampler)


class TinyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_scale=torch.nn.Parameter(torch.tensor(.4))
        self.output_scale=torch.nn.Parameter(torch.tensor(.2))
        self.register_buffer("fixed_frequency",torch.tensor(3.))

    def forward(self,noisy,history,action):
        feature=noisy+.05*history.mean(1,keepdim=True)+.1*action.mean((1,2))[:,None,None,None]
        return self.output_scale*self.backbone_scale*feature


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__();self.inner_model=TinyInner();self.is_upsampler=False;self.calls=0
        self.cfg=SimpleNamespace(sigma_data=.5,sigma_offset_noise=.1,noise_previous_obs=True,
                                 inner_model=SimpleNamespace(num_steps_conditioning=4))

    def compute_conditioners(self,sigma,sigma_cond):
        # Independent source-equation transcription for the tiny substitute.
        effective=(sigma.square()+.01).sqrt();skip=.25/(effective.square()+.25)
        return SimpleNamespace(c_in=(effective.square()+.25).rsqrt()[:,None,None,None],
            c_skip=skip[:,None,None,None],c_out=(effective*skip.sqrt())[:,None,None,None],
            c_noise=effective.log()/4,c_noise_cond=sigma_cond.log()/4)

    def compute_model_output(self,noisy,history,action,conditioners):
        self.calls+=1;self.seen_noisy=noisy.detach().clone();self.seen_history=history.detach().clone();self.seen_action=action.detach().clone()
        return self.inner_model(noisy*conditioners.c_in,history/.5,action)

    def sample(self,*args,**kwargs):raise AssertionError("No inference sampler in supervised loss")
    def denoise(self,*args,**kwargs):raise AssertionError("No no-grad inference path in training")
    def wrap_model_output(self,*args,**kwargs):raise AssertionError("One-step quantized tail is unused")


def batch():
    images=torch.full((1,5,3,96,96),.1,dtype=torch.float32);images[:,4]=.2
    return {"image":images,"action":torch.full((1,4,2),.25,dtype=torch.float32)}


class SupervisedObjectiveChecks(unittest.TestCase):
    def test_hand_computed_target_offset_and_unchanged_normalized_action(self):
        model=TinyDenoiser()
        with torch.no_grad():model.inner_model.output_scale.zero_()
        sample=batch();before=sample["image"].clone()
        plan=NoisePlan(torch.tensor([.4]),torch.full((1,12,1,1),.02),torch.full((1,12,96,96),.5),
                       torch.tensor([2.]),torch.full((1,3,1,1),.1),torch.zeros((1,3,96,96)))
        loss,metrics=supervised_denoising_loss(model,sample,noise_plan=plan)
        effective=math.sqrt(4+.01);skip=.25/(4+.01+.25);out=effective*math.sqrt(skip)
        self.assertAlmostEqual(float(loss.detach()),((.2-skip*.3)/out)**2,places=6)
        torch.testing.assert_close(model.seen_noisy,torch.full((1,3,96,96),.3))
        torch.testing.assert_close(model.seen_history,torch.full((1,12,96,96),.32))
        self.assertTrue(torch.equal(model.seen_action,sample["action"]))
        self.assertTrue(torch.equal(sample["image"],before))
        self.assertEqual(model.calls,1);self.assertEqual(metrics["inference_sampler_calls"],0)
        self.assertFalse(metrics["actions_renormalized"])

    def test_native_draw_order_clamps_and_unscaled_channel_offsets(self):
        sample={"image":torch.zeros((2,5,3,96,96)),"action":torch.zeros((2,4,2))};shapes=[]
        def normal(shape,**kwargs):
            shapes.append(tuple(shape))
            return torch.tensor([-100.,100.]) if tuple(shape)==(2,) else torch.full(shape,2.)
        with patch.object(torch,"randn",normal):plan=make_noise_plan(sample,make_generator(4))
        self.assertEqual(shapes,[(2,),(2,12,1,1),(2,12,96,96),(2,),(2,3,1,1),(2,3,96,96)])
        torch.testing.assert_close(plan.sigma_cond,torch.tensor([.002,20.]))
        torch.testing.assert_close(plan.sigma,torch.tensor([.002,20.]))
        torch.testing.assert_close(plan.history_offset,torch.full((2,12,1,1),.2))
        torch.testing.assert_close(plan.target_offset,torch.full((2,3,1,1),.2))

    def test_finite_gradients_update_only_selected_parameters(self):
        model=TinyDenoiser();selection=select_trainable_parameters(model,("inner_model.output_scale",))
        self.assertEqual(selection["trainable_numel"],1)
        before={k:v.clone() for k,v in model.state_dict().items()};optimizer=make_optimizer(model)
        loss,_=supervised_denoising_loss(model,batch(),generator=make_generator(7))
        loss.backward();gradient=model.inner_model.output_scale.grad
        self.assertTrue(bool(torch.isfinite(gradient)));self.assertNotEqual(float(gradient),0.)
        self.assertIsNone(model.inner_model.backbone_scale.grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before["inner_model.output_scale"],model.inner_model.output_scale))
        self.assertTrue(torch.equal(before["inner_model.backbone_scale"],model.inner_model.backbone_scale))
        self.assertTrue(torch.equal(before["inner_model.fixed_frequency"],model.inner_model.fixed_frequency))
        self.assertEqual(optimizer.defaults["lr"],1e-4);self.assertEqual(optimizer.defaults["weight_decay"],.01)

    def test_matching_validation_noise_and_independent_rng_streams(self):
        model=TinyDenoiser();sample=batch();global_before=torch.get_rng_state().clone()
        a,b=make_generator(11),make_generator(11)
        plan,duplicate=make_noise_plan(sample,a),make_noise_plan(sample,b)
        for name in ("sigma_cond","sigma","history_offset","history_epsilon","target_offset","target_epsilon"):
            self.assertTrue(torch.equal(getattr(plan,name),getattr(duplicate,name)))
        validation_state=a.get_state().clone();original_noise=plan.target_epsilon.clone()
        initial,_=validation_loss(model,sample,noise_plan=plan)
        repeat,_=validation_loss(model,sample,noise_plan=plan);self.assertEqual(initial,repeat)
        self.assertTrue(model.training)
        train=make_generator(12);select_trainable_parameters(model);optimizer=make_optimizer(model)
        loss,_=supervised_denoising_loss(model,sample,generator=train);loss.backward();optimizer.step()
        train_state=train.get_state().clone();updated,_=validation_loss(model,sample,noise_plan=plan)
        self.assertNotEqual(initial,updated)
        self.assertTrue(torch.equal(a.get_state(),validation_state));self.assertTrue(torch.equal(train.get_state(),train_state))
        self.assertTrue(torch.equal(plan.target_epsilon,original_noise));self.assertTrue(torch.equal(torch.get_rng_state(),global_before))

    def test_bad_action_range_fails_before_noise_or_forward(self):
        model=TinyDenoiser();sample=batch();sample["action"].fill_(200.)
        generator=make_generator(3);state=generator.get_state().clone()
        with self.assertRaises(ValueError):supervised_denoising_loss(model,sample,generator=generator)
        self.assertEqual(model.calls,0);self.assertTrue(torch.equal(generator.get_state(),state))

    def test_selection_validates_names_before_mutation_and_real_scope_is_all_inner(self):
        model=TinyDenoiser();flags=[p.requires_grad for p in model.parameters()]
        with self.assertRaises(ValueError):select_trainable_parameters(model,("missing.",))
        self.assertEqual([p.requires_grad for p in model.parameters()],flags)
        model.inner_model.backbone_scale.grad=torch.tensor(3.)
        select_trainable_parameters(model,("inner_model.output_scale",))
        self.assertIsNone(model.inner_model.backbone_scale.grad)
        selection=select_trainable_parameters(model)
        self.assertEqual(selection["prefixes"],["inner_model."])
        self.assertEqual(selection["trainable_numel"],2);self.assertEqual(selection["frozen_numel"],0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_factory_default_denies_before_assets_or_runtime_import(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name in ("torch","tdyssey.gpc_world_model"):raise AssertionError("Factory imported runtime before execute")
            return original(name,*args,**kwargs)
        with patch.object(builtins,"__import__",guarded):
            with self.assertRaises(PermissionError):make_real_denoiser("missing-source","missing-weights","cuda",{"execute":False})

    def test_eval_sampler_factory_keeps_native_three_step_config_without_sampling(self):
        model=TinyDenoiser();created=[]
        class FakeSampler:
            def __init__(self,denoiser,cfg):created.append((denoiser,cfg))
            def sample(self,*args,**kwargs):raise AssertionError("No sampler calls in preparation tests")
        sampler=make_sampler(model,source=SimpleNamespace(DiffusionSampler=FakeSampler,DiffusionSamplerConfig=SimpleNamespace))
        self.assertIsInstance(sampler,FakeSampler);self.assertEqual(model.calls,0)
        self.assertIs(created[0][0],model);cfg=created[0][1]
        self.assertEqual((cfg.num_steps_denoising,cfg.sigma_min,cfg.sigma_max,cfg.rho,cfg.order),(3,.002,5,7,1))
        self.assertEqual((cfg.s_churn,cfg.s_cond),(0,0));self.assertFalse(model.training)


if __name__=="__main__":unittest.main()
