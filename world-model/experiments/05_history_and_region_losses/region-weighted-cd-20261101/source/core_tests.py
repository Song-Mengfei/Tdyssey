"""CPU C/D gradient/noise/label-isolation tests; no real model or optimizer step."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(REPO)]
import training_core as core
from tdyssey.gpc_world_model import load_world_model_source
from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,conditioning_residual_loss

torch.set_num_threads(2)
SOURCE=load_world_model_source(REPO.parent/'Tdyssey/third_party/gpc_code')
SEEDS=dict(branch_seed=11,target_seed=22,weak_seed=33)

class Toy(torch.nn.Module):
    compute_conditioners=SOURCE.Denoiser.compute_conditioners
    wrap_model_output=SOURCE.Denoiser.wrap_model_output
    denoise=SOURCE.Denoiser.denoise
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.07))
        self.cfg=SimpleNamespace(sigma_data=.5,sigma_offset_noise=.1,noise_previous_obs=True,inner_model=SimpleNamespace(num_steps_conditioning=4))
        self.is_upsampler=False;self.seen=[]
    @property
    def device(self):return self.weight.device
    def compute_model_output(self,noisy,history,actions,cs):
        self.seen.append((torch.is_grad_enabled(),self.training))
        return self.weight*(noisy+.1*history.reshape(len(noisy),4,3,96,96).mean(1)+actions.mean((1,2))[:,None,None,None])

class IdentityScaler:
    def scale(self,x):return x

class Contracts(unittest.TestCase):
    def arrays(self):
        gen=torch.Generator().manual_seed(9)
        return torch.rand((4,8,3,96,96),generator=gen),torch.linspace(-1,1,56).reshape(4,7,2)
    def regions(self,images):
        R=torch.ones((len(images),4,96,96),dtype=torch.bool);E=torch.zeros_like(R);E[:,:,20:70,20:70]=True
        return dict(R=R,E=E)
    def test_split_loss_is_original_residual_and_same_draw_stream(self):
        images,actions=self.arrays();m=Toy();r1=make_conditioning_rngs(**SEEDS);r2=make_conditioning_rngs(**SEEDS)
        branches=[]
        for h in range(4):
            old,_=conditioning_residual_loss(m,dict(image=images[:,h:h+5],action=actions[:,h:h+4]),rngs=r1)
            noise=core.draw_noise(r2,4,torch.device('cpu'));new,finite,_=core.residual_loss(m,images[:,h:h+4],images[:,h+4],actions[:,h:h+4],noise,{k:v[:,h]for k,v in self.regions(images).items()},coeffs=(1,0,0))
            branches.append(noise.branch)
            self.assertTrue(torch.equal(old,new));self.assertTrue(bool(finite));self.assertEqual(r1.hashes(),r2.hashes())
            self.assertTrue(torch.equal(torch.autograd.grad(old,m.weight,retain_graph=True)[0],torch.autograd.grad(new,m.weight)[0]))
        self.assertIn('weak',branches);self.assertIn('clean',branches)
    def test_generation_is_exact_upstream_sampler_and_preserves_global_rng(self):
        images,actions=self.arrays();m=Toy().train();native=SOURCE.DiffusionSampler(m,SOURCE.DiffusionSamplerConfig(**core.SAMPLER))
        prior=torch.get_rng_state();torch.manual_seed(55);expected,_=native.sample(images[:,:4],actions[:,:4]);torch.set_rng_state(prior)
        g=core.NativeHistoryGenerator(m,SOURCE,torch.Generator().manual_seed(55));actual=g.generate(images[:,:4],actions[:,:4])
        self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(prior,torch.get_rng_state()));self.assertTrue(m.training)
        self.assertFalse(actual.requires_grad);self.assertFalse(actual.is_inference())
    def test_changed_future_labels_do_not_change_generated_P(self):
        images,actions=self.arrays();changed=images.clone();changed[:,4:]=1-changed[:,4:]
        outputs=[]
        for true in (images,changed):
            m=Toy();rng=make_conditioning_rngs(**SEEDS);g=core.NativeHistoryGenerator(m,SOURCE,torch.Generator().manual_seed(71))
            result=core.four_step_backward(m,true,actions,rng,g,IdentityScaler(),'D',regions=self.regions(true),generated_slots=[0,2])
            outputs.append(result['generated'])
            self.assertEqual(g.batch_calls,3);self.assertIsNotNone(m.weight.grad);self.assertTrue(torch.isfinite(m.weight.grad))
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(*outputs)))
        self.assertTrue(torch.equal(images[:,4:],1-changed[:,4:]))
    def test_four_losses_have_gradients_and_generated_history_can_exceed_rgb_range(self):
        images,actions=self.arrays();m=Toy();rng=make_conditioning_rngs(**SEEDS);grads=[]
        for h in range(4):
            m.zero_grad(set_to_none=True);history=images[:,h:h+4].clone();history[:,3]+=2
            loss,finite,_=core.residual_loss(m,history,images[:,h+4],actions[:,h:h+4],core.draw_noise(rng,4,torch.device('cpu')),{k:v[:,h]for k,v in self.regions(images).items()})
            loss.backward();grads.append(float(m.weight.grad));self.assertTrue(bool(finite))
        self.assertTrue(all(x!=0 for x in grads))
    def test_CD_noise_pairing_and_real_history_alignment(self):
        images,actions=self.arrays();states=[]
        for arm in ('C','D'):
            m=Toy();rng=make_conditioning_rngs(**SEEDS);g=core.NativeHistoryGenerator(m,SOURCE,torch.Generator().manual_seed(88))
            out=core.four_step_backward(m,images,actions,rng,g,IdentityScaler(),arm,regions=self.regions(images),generated_slots=[0,2]if arm=='D'else None)
            states.append((rng.hashes(),out['branches']));self.assertEqual(out['backward_calls'],4)
            supervised=[x for x in m.seen if x[0]];self.assertEqual(len(supervised),4);self.assertTrue(all(x[1]for x in supervised))
            self.assertEqual(g.batch_calls,0 if arm=='C'else 3)
            self.assertEqual(g.generated_frames,0 if arm=='C'else 6)
            self.assertEqual(out['fully_real_target_frames'],16 if arm=='C'else 10)
            self.assertEqual(out['generated_context_target_frames'],0 if arm=='C'else 6)
        self.assertEqual(states[0],states[1])

    def test_D_only_two_frozen_slots_use_generated_history(self):
        images,_=self.arrays();slots=torch.tensor([0,2]);generated=images[slots,:4]+7
        actual=core.mixed_history(images,generated,2,slots)
        self.assertTrue(torch.equal(actual[1],images[1,2:6]));self.assertTrue(torch.equal(actual[3],images[3,2:6]))
        self.assertTrue(torch.equal(actual[slots],generated));self.assertFalse(actual.requires_grad)

if __name__=='__main__':unittest.main()
