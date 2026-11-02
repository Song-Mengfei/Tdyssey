"""CPU toy math and new fitting-control contracts; no real model/GPU/update."""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import random
import sys
import tempfile
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];sys.path[:0]=[str(HERE),str(REPO)]
import engine
import training_core as core
from data import FrozenSampler
from tdyssey.gpc_world_model import load_world_model_source
from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs,conditioning_residual_loss
from tdyssey.native_rgb_training import capture_rng
from tdyssey.native_rgb_r2c1r1_training import _tree_digest

torch.set_num_threads(2)
SOURCE=load_world_model_source(REPO.parent/'Tdyssey/third_party/gpc_code')
SEEDS=dict(branch_seed=11,target_seed=22,weak_seed=33)

class Toy(torch.nn.Module):
    compute_conditioners=SOURCE.Denoiser.compute_conditioners
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.07));self.is_upsampler=False;self.calls=0
        self.cfg=SimpleNamespace(sigma_data=.5,sigma_offset_noise=.1,noise_previous_obs=True,inner_model=SimpleNamespace(num_steps_conditioning=4))
    def compute_model_output(self,noisy,history,actions,cs):
        self.calls+=1
        return self.weight*(noisy+.1*history.reshape(len(noisy),4,3,96,96).mean(1)+actions.mean((1,2))[:,None,None,None])

class IdentityScaler:
    def scale(self,value):return value
    def state_dict(self):return {'fixture_scale':1.}

class Contracts(unittest.TestCase):
    def arrays(self):
        g=torch.Generator().manual_seed(9)
        return torch.rand((16,5,3,96,96),generator=g),torch.linspace(-1,1,128).reshape(16,4,2)
    def test_loss_noise_and_accumulated_gradient_exact_original(self):
        images,actions=self.arrays();original=Toy();new=Toy();r1=make_conditioning_rngs(**SEEDS);r2=make_conditioning_rngs(**SEEDS)
        old_losses=[]
        for micro in range(4):
            b=images[micro*4:(micro+1)*4];a=actions[micro*4:(micro+1)*4]
            loss,_=conditioning_residual_loss(original,dict(image=b,action=a),rngs=r1);old_losses.append(loss.detach());(loss*.25).backward()
        before=images.clone();events=[]
        out=core.microbatch_backward(new,images,actions,r2,IdentityScaler(),observer=lambda *x:events.append(x))
        self.assertTrue(torch.equal(out['loss'],torch.stack(old_losses).mean()));self.assertTrue(torch.equal(original.weight.grad,new.weight.grad))
        self.assertEqual(r1.hashes(),r2.hashes());self.assertTrue(torch.equal(images,before));self.assertEqual(out['generated_history_frames'],0)
        self.assertEqual(sum(x[0]=='backward_complete'for x in events),4);self.assertEqual(new.calls,4);self.assertTrue(bool(out['forward_finite']))

    def test_budget_never_replaces_AMPskip_with_extra_draw(self):
        p=engine.initial_progress();p.update(successful_updates=2999,attempted_batches=2999,reserved_exposures=47984)
        self.assertTrue(engine.budget_allows(p,{}))
        p.update(attempted_batches=3000,reserved_exposures=48000,AMP_skips=1)
        self.assertFalse(engine.budget_allows(p,{}));self.assertEqual(p['successful_updates'],2999)
        p.update(attempted_batches=2,reserved_exposures=47990)
        self.assertFalse(engine.budget_allows(p,{}))

    def test_early_stop_requires_full48_and_gap_and_no_unknown(self):
        rule=engine.StopRule()
        self.assertFalse(rule.consider(0,{'full_train_all_passed':True,'full_train_unknown_count':1}))
        self.assertFalse(rule.consider(250,{'panel_all_passed':True,'full_train_all_passed':False}))
        self.assertFalse(rule.consider(500,{'full_train_all_passed':True,'full_train_unknown_count':0}))
        self.assertFalse(rule.consider(749,{'full_train_all_passed':True,'full_train_unknown_count':0}))
        self.assertTrue(rule.consider(750,{'full_train_all_passed':True,'full_train_unknown_count':0}))
        self.assertFalse(engine.StopRule().consider(250,{'full_train_all_passed':'true'}))

    def fake_protected(self):
        model=Toy().train();rngs=make_conditioning_rngs(**SEEDS);sampler=FrozenSampler(HERE)
        receipts=[]
        return SimpleNamespace(generator=torch.Generator().manual_seed(45),rngs=rngs,model=model,sampler=sampler,
            progress=engine.initial_progress(),windows=Counter(),application_count=0,activity='training',rng_proofs=[],
            output=HERE,atomic_json=lambda path,value:receipts.append(value),td=lambda x:_tree_digest(x,torch))

    def test_evaluation_restores_all_RNG_and_mode_and_rejects_data_draw(self):
        fake=self.fake_protected();before=fake.td(capture_rng(fake.generator));noise=fake.rngs.hashes()
        def evaluation():
            random.random();np.random.rand();torch.rand(7);torch.rand(3,generator=fake.rngs.target);fake.model.eval();return 'ok'
        self.assertEqual(engine.Session.protected(fake,'toy_evaluation',evaluation),'ok')
        self.assertEqual(before,fake.td(capture_rng(fake.generator)));self.assertEqual(noise,fake.rngs.hashes());self.assertTrue(fake.model.training)
        with self.assertRaisesRegex(RuntimeError,'altered training'):
            engine.Session.protected(fake,'bad_sampler_draw',lambda:fake.sampler.next_batch())

    def test_full_checkpoint_preserves_frozen_sampler_and_exposure_counts(self):
        # Synthetic completed-boundary fixture, with no optimizer.step ever run.
        from common import atomic_json
        sampler=FrozenSampler(HERE);targets=engine.read(HERE/'selection/index.json')['targets'];windows=Counter();families=Counter();letters=Counter()
        for _ in range(3):
            for i in sampler.next_batch():
                row=targets[i];windows[row['target_id']]+=1;families[row['family_id']]+=1;letters[row['letter']]+=1
        with tempfile.TemporaryDirectory(dir=HERE,prefix='offline-checkpoint-fixture-')as directory:
            folder=Path(directory);model=Toy();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            log=(folder/'toy-loss.jsonl').open('a');reservations=(folder/'toy-reservations.jsonl').open('a')
            p=engine.initial_progress();p.update(attempted_batches=3,reserved_exposures=48,actual_exposures=48)
            fake=SimpleNamespace(clean=True,progress=p,root=folder,c={},check_storage=lambda *args,**kwargs:None,
                log=log,reservations=reservations,output=folder,torch=torch,model=model,optimizer=optimizer,scaler=IdentityScaler(),
                generator=torch.Generator().manual_seed(8),rngs=make_conditioning_rngs(**SEEDS),sampler=sampler,
                exposure_summary=lambda:dict(target_counts=dict(windows),family_counts=dict(families),letter_counts=dict(letters)),
                identity={'offline_fixture':True},forward_counts={},backward_counts=Counter(),parent={'legacy_scheduler_provenance':{},'order':{}},
                parent_path=engine.PARENT,timings=defaultdict(float),atomic_json=atomic_json,last_full=None)
            try:
                receipt=engine.Session.save_full(fake,'offline-fixture-only');saved=torch.load(receipt['path'],map_location='cpu',weights_only=False)
                restored=FrozenSampler(HERE);restored.load_state_dict(saved['sampler'])
                self.assertEqual(restored.state_dict(),sampler.state_dict());self.assertEqual(saved['exposures']['target_counts'],dict(windows))
                self.assertEqual(saved['conditioning_rngs']['state_sha256'],fake.rngs.hashes());self.assertTrue(torch.equal(saved['model']['weight'],model.weight))
                self.assertEqual(optimizer.state_dict()['state'],{})
            finally:log.close();reservations.close()

if __name__=='__main__':unittest.main()
