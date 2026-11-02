"""Focused CPU interface/causality checks; actual GPU/UI checks are separate."""
import unittest
import numpy as np
from server import Playground,PHYSICS_SHA

class Spy:
    def __init__(self):self.inputs=[]
    def __call__(self,images,actions,seed):
        self.inputs.append((images.copy(),actions.copy(),seed))
        return np.full((3,96,96),len(self.inputs)/10,dtype=np.float32)

class Tests(unittest.TestCase):
    def setUp(self):self.spy=Spy();self.app=Playground(self.spy)
    def tearDown(self):self.app.env.close()
    def test_exact_historical_physics_and_real_history(self):
        i=self.app.env.initial_identity();self.assertEqual(i['protocol_sha256'],PHYSICS_SHA)
        self.assertEqual(i['physics']['agent']['body_type'],0)
        self.assertEqual(i['physics']['space']['damping'],.0009765625)
        self.assertEqual(self.app.env.control_step,3);self.assertEqual(len(self.app.history),4)
    def test_open_loop_never_uses_future_truth(self):
        a=np.array(self.app.target)+[12,0];initial=np.stack(self.app.history);past=np.stack(self.app.past)
        out=self.app.step(self.app.session,self.app.revision,a,4)
        self.assertEqual(out['steps'],4);self.assertEqual(len(self.spy.inputs),4)
        np.testing.assert_array_equal(self.spy.inputs[0][0],initial)
        np.testing.assert_array_equal(self.spy.inputs[0][1][:3],past)
        for k in range(4):
            np.testing.assert_array_equal(self.spy.inputs[k][1][-1],a)
            if k:np.testing.assert_array_equal(self.spy.inputs[k][0][-1],np.full((3,96,96),k/10,np.float32))
        self.assertFalse(np.array_equal(np.stack(self.app.history)[-1],self.spy.inputs[-1][0][-1]))
        # Next ordinary H1 is reconditioned on actual physical images, not stale rollout.
        actual=np.stack(self.app.history).copy();self.app.step(self.app.session,self.app.revision,a,1)
        np.testing.assert_array_equal(self.spy.inputs[4][0],actual)
    def test_stale_and_invalid_actions_do_not_advance_physics(self):
        before=self.app.env.state().copy();step=self.app.env.control_step
        for revision,target in [(self.app.revision-1,[200,200]),(self.app.revision,[512,100]),(self.app.revision,[float('nan'),0])]:
            with self.assertRaises(ValueError):self.app.step(self.app.session,revision,target,1)
        np.testing.assert_array_equal(before,self.app.env.state());self.assertEqual(step,self.app.env.control_step);self.assertFalse(self.spy.inputs)
    def test_sealed_letters_rejected_and_scenes_reset_cleanly(self):
        with self.assertRaises(ValueError):self.app.reset('D','center')
        for letter in ('T','L','O','C','X','B'):
            for scene in ('center','left','right','rotated'):
                s=self.app.reset(letter,scene);self.assertEqual(s['steps'],0);self.assertFalse(s['terminal']);self.assertEqual(len(self.app.history),4)
                self.assertLessEqual(float(self.app.env.penetrations().max()),2)

if __name__=='__main__':unittest.main()
