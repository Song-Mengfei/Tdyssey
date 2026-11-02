"""Offline saved-array/mocked sampler contracts; no Torch/model/physics calls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np

FILE=Path(__file__).with_name('train_eval.py')
sys.path.insert(0,str(FILE.resolve().parents[2]))
spec=importlib.util.spec_from_file_location('accelerated_train_eval',FILE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class TrainingContracts(unittest.TestCase):
    def test_every_sample_once_including_short_final_batch(self):
        rows=[dict(family_id=f'f{i%9}',event=f'e{i%3}') for i in range(35)]
        b=m.epoch_batches(rows)
        self.assertEqual([len(x) for x in b],[16,16,3])
        self.assertEqual(sorted(i for x in b for i in x),list(range(35)))
        self.assertEqual(b,m.epoch_batches(rows))

    def test_recursive_raw_feedback_and_action_alignment(self):
        history=np.zeros((1,4,3,96,96),np.float32)
        actions=np.arange(38,dtype=np.float32).reshape(1,19,2)
        seen=[]
        def sample(h,a,seed):
            k=len(seen); seen.append((h.copy(),a.copy(),seed))
            return np.full((1,3,96,96),k+.25,np.float32)
        result=m.rollout_raw(sample,history,actions,seeds=list(range(16)))
        self.assertEqual(result.shape,(1,16,3,96,96))
        self.assertEqual(float(seen[2][0][0,-1,0,0,0]),1.25)
        self.assertTrue(np.array_equal(seen[5][1],actions[:,5:9]/np.float32(255.5)-np.float32(1)))
        self.assertGreater(float(result.max()),1)  # no clipping or truth feedback

    def test_action512_rejected_without_clipping(self):
        actions=np.zeros((1,19,2),np.float32); actions[0,4,1]=512
        with self.assertRaises(ValueError):
            m.rollout_raw(lambda *a,**k:None,np.zeros((1,4,3,96,96),np.float32),actions,seeds=list(range(16)))
        self.assertEqual(actions[0,4,1],512)

    def test_family_isolation_and_eval_selection(self):
        train=[dict(sample_id='a',family_id='f',letter='T',anchor=3)]
        with self.assertRaises(ValueError): m.validate_indices(train,[dict(train[0],sample_id='b')])
        val=[dict(sample_id=f'v{i}',family_id=f'v{i//2}',letter='O',task='translate',anchor=3,horizon=16,committed_future=True) for i in range(10)]
        e=m.eval_spec(val,cap=3)
        self.assertEqual(len({r['family_id'] for r in e['rows']}),3)
        self.assertEqual(e,m.eval_spec(val,cap=3))

    def test_saved_timeline_and_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'raw.npz'; rgb=np.zeros((6,3,96,96),np.uint8)
            for k in range(6): rgb[k]=k
            actions=np.arange(10,dtype=np.float64).reshape(5,2)
            np.savez_compressed(p,rgb=rgb,states=np.zeros((6,15)),passed_actions=actions,requested_actions=actions,control_valid=np.ones(5,bool))
            x=m.RawStore().training(dict(raw_path=str(p),anchor=3))
            self.assertEqual(x['image'].shape,(5,3,96,96)); self.assertAlmostEqual(float(x['image'][-1,0,0,0]),4/255)
            np.testing.assert_array_equal(x['action'],actions[:4].astype(np.float32)/np.float32(255.5)-np.float32(1))

    def test_existing_scorer_signature_with_synthetic_rgb(self):
        calibration=m.read(m.REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
        im=np.ones((3,96,96),np.float32)
        color=np.asarray(calibration['object_rgb'],np.float32)
        im[:,25:65,25:65]=color[:,None,None]
        im[:,35:55,35:55]=1
        im[:,10:15,10:15]=np.asarray([.2,.3,.95])[:,None,None]
        score=m._metric(im,im,im,'O',calibration)
        self.assertIn('raw_mask_iou',score)
        self.assertIn('predicted_raw_corner_counts',score)
        self.assertAlmostEqual(score['raw_mask_iou'],1.)

if __name__=='__main__': unittest.main()
