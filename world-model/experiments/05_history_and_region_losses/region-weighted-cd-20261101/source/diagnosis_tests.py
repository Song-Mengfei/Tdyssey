"""CPU synthetic/saved-only checks. No Torch, model, physics or inference."""
import json
from pathlib import Path
import sys
import subprocess
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import diagnosis
import direction

class ArrayTensor:
    def __init__(self,a):self.a=np.asarray(a,np.float32);self.shape=self.a.shape
    def detach(self):return self
    def float(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def numpy(self):return self.a
    def __getitem__(self,i):return ArrayTensor(self.a[i])

class FakeTorch:
    def __init__(self):self.last=None
    def randn(self,*shape,**kw):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        self.last=ArrayTensor(np.arange(np.prod(shape),dtype=np.float32).reshape(shape));return self.last

class Tests(unittest.TestCase):
    def test_noise_capture_is_observation_restores_on_success_and_error(self):
        fake=FakeTorch();original=fake.randn;records=[]
        with diagnosis.capture_initial_noise(fake,[17],(4,3,2,2),records,replay=False):
            got=fake.randn((4,3,2,2));self.assertIs(got,fake.last)
        self.assertEqual(fake.randn,original);self.assertEqual(len(records),1)
        self.assertEqual(records[0]['per_case_initial_epsilon_sha256'][2],diagnosis.tensor_bytes_sha(got[2]))
        with self.assertRaises(ValueError):
            with diagnosis.capture_initial_noise(fake,[17],(4,3,2,2),[],replay=False):fake.randn((1,3,2,2))
        self.assertEqual(fake.randn,original)
        with self.assertRaises(RuntimeError):
            with diagnosis.capture_initial_noise(fake,[17],(4,3,2,2),[],replay=False):raise RuntimeError('forced')
        self.assertEqual(fake.randn,original)
    def test_true_history_has_no_own_target_and_action_timeline(self):
        row=dict(anchor=3);rgb=np.repeat(np.arange(21,dtype=np.uint8)[:,None,None,None],3*96*96,axis=1).reshape(21,3,96,96)
        data=dict(rgb=rgb,passed_actions=np.column_stack((np.arange(20),100+np.arange(20))).astype(np.float64),control_valid=np.ones(20,bool))
        class Store:
            def arrays(self,row):return data
        value=diagnosis.ev().prepare_inputs(Store(),[row],'real_history')
        np.testing.assert_array_equal(value['real_histories'][0,:,3,0,0,0],np.arange(3,19,dtype=np.float32)/np.float32(255))
        np.testing.assert_array_equal(value['absolute_actions'][0,:,0],np.arange(19))
        before=value['real_histories'][:,0].copy();data['rgb'][4:]=200
        after=diagnosis.ev().prepare_inputs(Store(),[row],'real_history')
        np.testing.assert_array_equal(before,after['real_histories'][:,0])
        self.assertFalse(np.array_equal(value['real_histories'][:,1],after['real_histories'][:,1]))
    def test_RH_does_not_accumulate_independent_outputs(self):
        def m(deg):
            return dict(gray=dict(angle=deg,ratio=2,pixels=100),hole=dict(angle=deg,ratio=3,pixels=20),hole_count=1,
                cropped=False,significant_fragmentation=False,blue_in_hole=False)
        # Truth rotates +6/step. Every independent output is 7deg behind its
        # preceding real frame: a visibly opposite single-step transition.
        truth=list(range(17));pred=list(range(17,33))
        lookup={k:m(6*k) for k in truth};lookup.update({17+k:m(6*k-7) for k in range(16)})
        original=direction.measure;direction.measure=lambda im,*args:lookup[im]
        try:
            result=direction._o_label(truth,np.radians(np.arange(17)*6),pred,'real_history',{},direction.protocol())
            self.assertEqual(result['classification'],'opposite')
            self.assertTrue(all(f['cumulative_direction'] is None for f in result['frames']))
            self.assertTrue(all(f['local_direction']['classification']=='opposite' for f in result['frames']))
        finally:direction.measure=original
    def test_copy_small_rotation_unknown_and_unknown_never_supports(self):
        result=direction._classify(dict(gray=0.,hole=0.),.5,dict(gray=0.,hole=0.),[],direction.protocol())
        self.assertEqual(result['classification'],'unknown');self.assertFalse(result['known'])
        analysis=dict(o_direction_rows=[dict(sample_id='AC-O-01-08:x',labels={'C':dict(classification='unknown',known=False,input_mode='autoregressive')})],b_hole_rows=[])
        self.assertFalse(direction.support_signals(analysis,'C')['O_display_known_correct_clockwise'])
    def test_native_batch_scope_saved_only(self):
        source=diagnosis.read(diagnosis.OLD/'formal-spec.json')
        selected=[(i,b) for i,b in enumerate(source['batches']) if {r['letter'] for r in b['rows']}&{'O','B'}]
        self.assertEqual([i for i,b in selected],[0,1,6,7]);self.assertEqual(sum(len(b['rows']) for i,b in selected),16)
        self.assertTrue(all(len(b['rows'])==4 and len(b['seeds'])==16 and len({r['letter'] for r in b['rows']})==1 for i,b in selected))
        for i,b in selected:
            for row in b['rows']:self.assertTrue(diagnosis.check_saved_timeline(row)['own_target_and_later_RGB_absent_from_own_history'])
    def test_no_torch_imported(self):
        # The combined suite legitimately imports Torch for other CPU tests.
        # Verify import-time isolation in a fresh interpreter instead.
        result=subprocess.run([sys.executable,'-X','utf8','-B','-c',
            'import sys; import diagnosis, direction; assert "torch" not in sys.modules'],
            cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
