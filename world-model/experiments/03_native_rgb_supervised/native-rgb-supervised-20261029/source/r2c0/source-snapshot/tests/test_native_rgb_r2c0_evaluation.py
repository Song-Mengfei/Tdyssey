"""Pure array/callback tests; no native model, environment or truth generation."""
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tdyssey.native_rgb_r2c0_evaluation import normalize_actions, rollout_raw, score_endpoints


class R2C0EvaluationTests(unittest.TestCase):
    def _mock_catalogues(self,single_val_anchor=False):
        from scripts import native_rgb_r2c0_evaluation as prep
        def dataset(split,val=False):
            entries=[];rows=[];h4=[]
            for letter in prep.LETTERS:
                for family in range(2 if val else 1):
                    tid=f'{split}-{letter}-{family}';ei=len(entries)
                    entries.append(dict(trajectory_id=tid,letter=letter,sha256='a'*64,metadata_sha256='b'*64,split=split))
                    anchors=[3] if val and single_val_anchor else [3,4]
                    lowest=min(anchors,key=lambda a:prep.rank('val_natural',tid,a))
                    for a in anchors:
                        moving=not val or single_val_anchor or a!=lowest
                        rows.append(dict(trajectory_id=tid,window_id=f'{tid}:{a}:H1',letter=letter,anchor=a,
                            split=split,dataset_index=len(rows),translation=moving,rotation=moving,moving=moving))
                        h4.append((ei,a))
            return SimpleNamespace(entries=entries,window_rows=rows,h4_windows=h4)
        train,val=dataset('train'),dataset('seen_val',True)
        indices=[i%len(train.window_rows) for i in range(480)]
        draws=dict(seed=5478,indices=indices,window_ids=[train.window_rows[i]['window_id'] for i in indices])
        return prep,train,val,draws

    def test_index_natural_unfiltered_roles_unique_motion_deduplicated(self):
        prep,train,val,draws=self._mock_catalogues()
        cohorts,missing=prep.choose_indices(train,val,draws)
        self.assertFalse(missing)
        self.assertEqual({k:len(v) for k,v in cohorts.items()},dict(train_visual=12,val_natural=12,val_motion=12))
        self.assertTrue(all(not r['moving'] for r in cohorts['val_natural']))
        self.assertTrue(all(r['moving'] for r in cohorts['val_motion']))
        ids=[r['window_id'] for values in cohorts.values() for r in values]
        self.assertEqual(len(ids),len(set(ids)))

    def test_missing_motion_not_borrowed_after_natural_uses_only_anchor(self):
        prep,train,val,draws=self._mock_catalogues(True)
        cohorts,missing=prep.choose_indices(train,val,draws)
        self.assertEqual(len(cohorts['val_natural']),12)
        self.assertEqual(cohorts['val_motion'],[])
        self.assertEqual(len(missing),12)

    def test_four_step_generated_history_and_actions4_only(self):
        history=np.zeros((3,4,3,96,96),np.float32)
        absolute=np.arange(3*7*2,dtype=np.float64).reshape(3,7,2)
        calls=[]
        def callback(images,actions,**context):
            calls.append((images.copy(),actions.copy(),context))
            return np.full((len(images),3,96,96),(context["step"]+1)/10,dtype=np.float32)
        result=rollout_raw(callback,history,absolute,window_ids=["a","b","c"],cohort="mock",batch_size=2)
        self.assertEqual(len(calls),8)
        np.testing.assert_array_equal(calls[1][0][:,-1],np.float32(.1))
        np.testing.assert_array_equal(calls[2][0][:,-2],np.float32(.1))
        np.testing.assert_array_equal(calls[2][0][:,-1],np.float32(.2))
        np.testing.assert_array_equal(calls[1][1],normalize_actions(absolute[:2,1:5]))
        self.assertEqual(calls[1][1].shape,(2,4,2))
        np.testing.assert_array_equal(result["h1"],np.float32(.1))
        np.testing.assert_array_equal(result["h4"],np.float32(.4))
        self.assertFalse(result["future_truth_in_feedback"])

    def test_finite_out_of_range_unmodified_feedback(self):
        seen=[]
        def callback(images,actions,**context):
            seen.append(images.copy())
            return np.full((len(images),3,96,96),-.25 if context["step"]==0 else 1.25,dtype=np.float64)
        result=rollout_raw(callback,np.zeros((1,4,3,96,96),np.float32),np.zeros((1,7,2)),window_ids=["x"],cohort="mock")
        self.assertEqual(len(seen),4)
        self.assertEqual(seen[1][0,-1,0,0,0],-.25)
        self.assertEqual(seen[2][0,-1,0,0,0],1.25)
        self.assertEqual(result["status_h1"],["finite_out_of_range"])
        self.assertFalse(result["hard_stop"])

    def test_nonfinite_and_wrong_shape_stop_keep_denominator_and_raw(self):
        for bad in (np.full((2,3,96,96),np.nan),np.ones((2,1,96,96))):
            calls=[]
            def callback(images,actions,**kwargs):
                calls.append(1)
                return bad
            result=rollout_raw(callback,np.zeros((5,4,3,96,96),np.float32),np.zeros((5,7,2)),
                window_ids=list("abcde"),cohort="mock",batch_size=2)
            self.assertEqual(len(calls),1)
            self.assertEqual(result["planned_windows"],5)
            self.assertTrue(result["hard_stop"])
            np.testing.assert_array_equal(result["raw_failures"][0]["raw"],bad)
            self.assertTrue(np.isnan(result["h4"]).all())

    def test_paired_seeds_and_rng_unchanged(self):
        history=np.zeros((1,4,3,96,96),np.float32);actions=np.zeros((1,7,2))
        before=np.random.get_state()
        first=rollout_raw(lambda x,a,**k:x[:,-1].copy(),history,actions,window_ids=["x"],cohort="same")
        second=rollout_raw(lambda x,a,**k:x[:,-1].copy(),history,actions,window_ids=["x"],cohort="same")
        self.assertEqual([r["seed"] for r in first["step_records"]],[r["seed"] for r in second["step_records"]])
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])

    def test_score_bridge_object_and_blue_separate_and_occlusion_unknown(self):
        root=Path(__file__).resolve().parents[1]
        calibration=json.loads((root/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json').read_text())
        truth=np.ones((1,3,96,96),dtype=np.float32)
        truth[0,:,20:45,20:45]=np.asarray((119,136,153),dtype=np.float32)[:,None,None]/255
        truth[0,:,65:75,65:75]=np.asarray((65,105,225),dtype=np.float32)[:,None,None]/255
        wrong_blue=truth.copy();wrong_blue[0,:,65:75,65:75]=1
        wrong_blue[0,:,65:75,45:55]=np.asarray((65,105,225),dtype=np.float32)[:,None,None]/255
        result=dict(planned_windows=1,window_ids=['synthetic'],h1=wrong_blue,h4=wrong_blue,status_h1=['ok'],status_h4=['ok'])
        scored=score_endpoints(result,truth,truth,truth,[dict(letter='T')],calibration)
        self.assertEqual(scored[0]['raw_object_error'],0.)
        self.assertGreater(scored[0]['pusher_error'],.9)
        occluded=truth.copy();occluded[0,:,38:48,38:48]=np.asarray((65,105,225),dtype=np.float32)[:,None,None]/255
        result.update(h1=occluded,h4=occluded)
        scored=score_endpoints(result,occluded,occluded,occluded,[dict(letter='T')],calibration)
        self.assertEqual(scored[0]['shape_outcome'],'unknown')
        self.assertFalse(scored[0]['legacy_shape_assessable'])
        self.assertIsNone(scored[0]['legacy_shape_deformation_failure'])

    def test_score_finite_out_of_range_raw_preserved_hard_failure(self):
        root=Path(__file__).resolve().parents[1]
        calibration=json.loads((root/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json').read_text())
        raw=np.ones((1,3,96,96),dtype=np.float32);raw[0,0,0,0]=-.25
        truth=np.ones_like(raw)
        result=dict(planned_windows=1,window_ids=['synthetic'],h1=raw,h4=raw,
            status_h1=['finite_out_of_range'],status_h4=['finite_out_of_range'])
        scored=score_endpoints(result,truth,truth,truth,[dict(letter='T')],calibration)
        self.assertTrue(scored[0]['hard_generation_failure'])
        self.assertIn('out_of_range',scored[0]['failure_reasons'])
        self.assertEqual(raw[0,0,0,0],-.25)


if __name__=="__main__":unittest.main()
