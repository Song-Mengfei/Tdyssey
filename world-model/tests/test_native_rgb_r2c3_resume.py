"""Focused restore and saved-RGB loss-input checks, without Torch/model work."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
from tdyssey.native_rgb_r2c3_training import restore_payload,fixed_loss_batch_arrays,legacy_prepared_snapshot,scaled_optimizer_step


class Stateful:
    def __init__(self):self.state={'fresh_placeholder':True};self.strict=None
    def load_state_dict(self,value,**kwargs):self.state=deepcopy(value);self.strict=kwargs.get('strict')
    def state_dict(self):return deepcopy(self.state)


class RestoreContracts(unittest.TestCase):
    def test_all_complete_training_components_restore_and_progress_is_not_reset(self):
        components={key:Stateful() for key in ('model','optimizer','scaler','scheduler','prepared')}
        payload=dict(kind='native_gpc_rgb_training.v1',model={'weight':[1,2]},optimizer={'moments':[3,4]},
            scaler={'scale':128.,'growth_tracker':300},scheduler={'last_epoch':300,'_last_lr':[9.779754323328192e-5]},
            order={'cursor':4800,'slot_rng':{'saved':1},'queues':{'saved':2}},rng={'python':1,'numpy':2,'torch':3,'cuda':4,'noise_generator':5},
            progress={'updates':300,'attempts':300,'samples':4800,'complete_optimizer_boundary':True,'stop_reason':'stage_attempt_target'})
        rng_calls=[];generator=object()
        def restore_rng(state,target):rng_calls.append((deepcopy(state),target))
        progress=restore_payload(payload,components['model'],components['optimizer'],components['scaler'],components['scheduler'],generator,components['prepared'],restore_rng_fn=restore_rng)
        self.assertEqual(progress,payload['progress']);self.assertIsNot(progress,payload['progress'])
        for name in ('model','optimizer','scaler','scheduler'):self.assertEqual(components[name].state,payload[name])
        self.assertTrue(components['model'].strict);self.assertEqual(components['prepared'].state,payload['order'])
        self.assertEqual(rng_calls,[(payload['rng'],generator)])
        self.assertEqual(components['scheduler'].state['last_epoch'],300)
        self.assertNotIn('torch',sys.modules)

    def test_noncomplete_or_wrong_lineage_payload_rejected_before_mutation(self):
        for change in ({'updates':0},{'updates':299},{'samples':16},{'complete_optimizer_boundary':False}):
            states=[Stateful() for _ in range(5)]
            progress={'updates':300,'samples':4800,'complete_optimizer_boundary':True};progress.update(change)
            with self.assertRaisesRegex(ValueError,'complete300'):
                restore_payload({'kind':'native_gpc_rgb_training.v1','progress':progress},*states[:4],object(),states[4],restore_rng_fn=lambda *args:None)
            self.assertTrue(all(s.state=={'fresh_placeholder':True} for s in states))

    def test_fixed_loss_uint8_target_uses_exact_once_normalization(self):
        import numpy as np
        target=np.full((4,3,96,96),255,np.uint8);target[1]=128
        source=dict(histories=np.full((4,4,3,96,96),.5,np.float32),truth_h1=target,
            absolute_actions=np.full((4,7,2),255.5,np.float32))
        images,actions=fixed_loss_batch_arrays(source)
        self.assertEqual(images.shape,(4,5,3,96,96));self.assertEqual(images.dtype,np.float32)
        np.testing.assert_array_equal(images[:,:4],source['histories'])
        np.testing.assert_array_equal(images[:,4],target.astype(np.float32)/np.float32(255))
        self.assertEqual(float(images[0,4,0,0,0]),1.);self.assertLessEqual(float(images.max()),1.)
        np.testing.assert_array_equal(actions,np.zeros((4,4,2),np.float32))
        np.testing.assert_array_equal(source['truth_h1'],target);self.assertNotIn('torch',sys.modules)

    def test_versioned_prepared_wrapper_can_be_compared_to_legacy_semantics(self):
        import hashlib
        class Prepared:
            class Order:
                def legacy_state_dict(self):return {'kind':'old_order','cursor':4800}
            order=Order()
            def state_dict(self):return {'kind':'prepared','order':{'kind':'new_scope','cursor':4800},'delivered_draws':4800,'state_sha256':'new'}
        expected={'kind':'prepared','order':{'kind':'old_order','cursor':4800},'delivered_draws':4800}
        expected['state_sha256']=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.assertEqual(legacy_prepared_snapshot(Prepared()),expected)

    def test_real_step_observer_distinguishes_scale_skip_then_success(self):
        count=[0]
        class Optimizer:
            weight=3.;moment=0.
            def step(self):
                self.weight-=.25;self.moment+=1.;count[0]+=1
                return None
        class Scaler:
            scale=128.;calls=0
            def get_scale(self):return self.scale
            def step(self,optimizer):
                self.calls+=1
                if self.calls>1:return optimizer.step()
                return None
            def update(self):
                if self.calls==1:self.scale/=2
        optimizer=Optimizer();scaler=Scaler()
        skipped=scaled_optimizer_step(scaler,optimizer,application_count=lambda:count[0])
        self.assertEqual(skipped,{'applied':0,'scale_before':128.,'scale_after':64.})
        self.assertEqual((optimizer.weight,optimizer.moment,count[0]),(3.,0.,0))
        completed=scaled_optimizer_step(scaler,optimizer,application_count=lambda:count[0])
        self.assertEqual(completed,{'applied':1,'scale_before':64.,'scale_after':64.})
        self.assertEqual((optimizer.weight,optimizer.moment,count[0]),(2.75,1.,1))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
