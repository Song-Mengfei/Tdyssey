"""Stage3 raw output/batch contracts, with no real model or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.native_rgb_stage3_prepare import batch_plan
from scripts.native_rgb_stage3_run import compose_comparisons,prediction_hook

ROOT=Path(__file__).resolve().parents[1]


class Stage3EntryTests(unittest.TestCase):
    def test_paired_batch_members_sizes_and_seed_are_order_bound(self):
        groups={'train18':[{'window_id':f't:{i}'} for i in range(18)],'dev36':[{'window_id':f'd:{i}'} for i in range(36)]}
        a=batch_plan(groups);b=batch_plan(groups)
        self.assertEqual(a,b);self.assertEqual([r['batch_size'] for r in a['cohorts']['train18']],[4,4,4,4,2])
        self.assertEqual([r['batch_size'] for r in a['cohorts']['dev36']],[4]*9)
        groups['train18'][0],groups['train18'][1]=groups['train18'][1],groups['train18'][0]
        self.assertNotEqual(a['cohorts']['train18'][0]['batch_seed'],batch_plan(groups)['cohorts']['train18'][0]['batch_seed'])

    def test_missing_or_invalid_predictions_keep_every_row_and_raw_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);(root/'truth').mkdir();(root/'predictions').mkdir()
            for cohort in ['train18','dev36']:
                rgb=np.ones((2,3,96,96),np.float32)
                np.savez(root/'truth'/f'{cohort}.npz',last=rgb,true=rgb,hold=rgb,last_full_mask=np.ones((2,96,96)),true_full_mask=np.ones((2,96,96)),window_ids=np.array(['a','b']))
            invalid=np.full((2,3,96,96),9,np.float32)
            np.savez(root/'predictions/frozen-train18.npz',prediction=invalid,status=np.array(['out_of_range','out_of_range']),window_ids=np.array(['a','b']))
            compose_comparisons(root)
            with np.load(root/'predictions/train18.npz') as result:
                self.assertEqual(len(result['fitted']),2);self.assertTrue(np.isnan(result['fitted']).all())
                self.assertTrue((result['frozen']==9).all());self.assertTrue((result['frozen_status']=='out_of_range').all())
                self.assertTrue((result['hold']==1).all())

    def test_prediction_hook_uses_absolute_actions_and_saves_unmodified_batch_RGB(self):
        class Param:
            requires_grad=True
            def numel(self):return 8729827
        class Model:
            def parameters(self):return [Param()]
        class Budget:
            elapsed=1.;baseline_seconds=None
            def require_operation(self,*args):pass
        groups={'train18':[{'window_id':f't:{i}'} for i in range(18)],'dev36':[{'window_id':f'd:{i}'} for i in range(36)]}
        plans=batch_plan(groups);truth={}
        for cohort,rows in groups.items():truth[cohort]=dict(histories=np.full((len(rows),4,3,96,96),.25,np.float32),actions_absolute=np.full((len(rows),4,2),200,np.float32),window_ids=np.array([r['window_id'] for r in rows]))
        calls=[]
        def fake_predictor(model,source):
            def predict(history,actions,*,batch_seed):
                self.assertTrue((actions==200).all());self.assertEqual(history.shape[1:],(4,3,96,96));calls.append((len(actions),batch_seed))
                return np.full((len(actions),3,96,96),.12345,np.float32)
            return predict
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);(root/'paired-batches.json').write_text(json.dumps(plans))
            with patch('scripts.native_rgb_stage3_run._parameter_digest',return_value=('digest',{'p':'sha'})),\
                patch('tdyssey.native_rgb_evaluation.native_sampler_batch_predictor',side_effect=fake_predictor):
                result=prediction_hook(root,'frozen',truth,plans)(dict(model=Model(),source=object(),check=lambda:None,budget=Budget(),progress={'updates':0}))
            self.assertEqual(len(calls),14);self.assertEqual(result['native_prediction_frames'],54)
            with np.load(root/'predictions/frozen-train18.npz') as arrays:
                self.assertTrue((arrays['prediction']==np.float32(.12345)).all());self.assertTrue((arrays['status']=='ok').all())


if __name__=='__main__':unittest.main()
