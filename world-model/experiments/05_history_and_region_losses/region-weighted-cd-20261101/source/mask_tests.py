"""CPU-only exact native loss/gradient and truth-mask boundary checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[2])]
import masks

class MaskTests(unittest.TestCase):
    def images(self):
        x=np.full((1,3,96,96),255,np.uint8)
        x[0,:,20:70,20:60]=np.asarray([119,136,153],np.uint8)[:,None,None]
        x[0,:,27:41,29:49]=255;x[0,:,49:63,29:49]=255
        return x
    def test_hole_white_and_both_motion_sides_are_supervised(self):
        prev=self.images();target=np.roll(prev,4,axis=3);before=prev.copy();after=target.copy()
        q=masks.compute_masks(prev,target)
        self.assertTrue(q['R'][0,34,38]);self.assertTrue(q['R'][0,56,38])
        self.assertTrue(q['E'][0,25,20]);self.assertTrue(q['E'][0,25,63])
        self.assertTrue(q['E'][0,27,38]);self.assertGreater(q['stats']['enclosed_hole_truth_pixels'],0)
        np.testing.assert_array_equal(prev,before);np.testing.assert_array_equal(target,after)
    def test_empty_falls_back_and_one_level_is_not_change(self):
        a=np.full((1,3,96,96),255,np.uint8);b=a.copy();b[:,:,20,20]=254
        q=masks.compute_masks(a,b);self.assertEqual(q['stats']['R_empty'],1);self.assertEqual(q['stats']['E_empty'],1)
        b[:,:,20,20]=253;q=masks.compute_masks(a,b);self.assertTrue(q['E'][0,20,20])
        output=torch.randn((1,3,96,96),requires_grad=True);target=torch.randn_like(output)
        empty=torch.zeros((1,96,96),dtype=torch.bool)
        l,stats,finite=masks.weighted_residual_loss(output,target,empty,empty)
        self.assertTrue(torch.allclose(l,torch.nn.functional.mse_loss(output,target),atol=1e-7,rtol=1e-7));self.assertTrue(bool(finite));self.assertEqual(int(stats['R_empty']),1)
    def test_full_native_loss_and_gradient_exact_and_each_region_has_gradient(self):
        torch.manual_seed(13);p=torch.randn((2,3,96,96),requires_grad=True);t=torch.randn_like(p)
        region=torch.zeros((2,96,96),dtype=torch.bool);region[:,20:40,30:50]=True
        detail=torch.zeros_like(region);detail[:,35:55,35:55]=True
        old=torch.nn.functional.mse_loss(p,t);oldgrad=torch.autograd.grad(old,p)[0]
        restored,_,_=masks.weighted_residual_loss(p,t,region,detail,(1,0,0));newgrad=torch.autograd.grad(restored,p)[0]
        self.assertTrue(torch.equal(old,restored));self.assertTrue(torch.equal(oldgrad,newgrad))
        for coeff in ((0,1,0),(0,0,1),(.5,.25,.25)):
            loss,_,_=masks.weighted_residual_loss(p,t,region,detail,coeff);g=torch.autograd.grad(loss,p)[0]
            self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
    def test_four_target_mask_timing_matches_direct_pairs(self):
        images=np.repeat(self.images()[:,None],8,axis=1)
        for k in range(8):images[:,k]=np.roll(images[:,k],k,axis=3)
        b=masks.batch(images)
        for h in range(4):
            p=masks.compute_masks(images[:,h+3],images[:,h+4]);np.testing.assert_array_equal(b['R'][:,h],p['R']);np.testing.assert_array_equal(b['E'][:,h],p['E'])

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
