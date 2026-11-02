"""CPU callbacks cover timing, no future-GT feedback and exact paired seeds."""
import unittest
import numpy as np
from tdyssey.native_rgb_playground_diagnostic_runner import paired_rollout,make_plan,RawPredictionFailure

class PairedDiagnosticTests(unittest.TestCase):
    def test_true_corrected_A_and_raw_B_are_paired_without_future_truth_leak(self):
        truth=np.zeros((8,3,96,96),np.float32)
        for j in range(4,8):truth[j]=10+j
        past=np.asarray([[0,511],[255.5,255.5],[511,0]],np.float64)
        future=np.asarray([[100+j,200+j]for j in range(4)],np.float64)
        seeds=np.arange(4)+5478;calls=[]
        def sample(history,actions,*,seed):
            calls.append((history.copy(),actions.copy(),seed))
            return np.full((1,3,96,96),np.float32(1.2+(seed-5478)),np.float32)
        out,records=paired_rollout(sample,truth,past,future,seeds)
        self.assertEqual(len(calls),8)
        self.assertTrue(np.array_equal(out['A'][0],out['B'][0]))
        np.testing.assert_array_equal(calls[0][1],np.concatenate([past,future[:1]])[None].astype(np.float32)/511*2-1)
        self.assertEqual([r[2]for r in calls[:4]],[r[2]for r in calls[4:]])
        for j in range(1,4):
            self.assertTrue(np.any(calls[j][0]==np.float32(13+j)))
            self.assertFalse(np.any(calls[4+j][0]>=10))
            np.testing.assert_array_equal(calls[4+j][0][0,-1],out['B'][j-1])
        self.assertEqual(records['B'][0]['history_sha256'],records['A'][0]['history_sha256'])
    def test_nonfinite_raw_is_retained_and_not_repaired_or_retried(self):
        calls=[]
        def sample(history,actions,*,seed):
            calls.append(seed);return np.full((1,3,96,96),np.nan,np.float32)
        with self.assertRaises(RawPredictionFailure)as caught:
            paired_rollout(sample,np.zeros((8,3,96,96),np.float32),np.zeros((3,2)),np.zeros((4,2)),np.arange(4))
        self.assertEqual(calls,[0]);self.assertTrue(np.isnan(caught.exception.partial['A'][0]).all())
    def test_fixed_plan_is_bounded_no_sampler_or_seed_search(self):
        plan=make_plan();self.assertEqual(len(plan['cases']),8)
        self.assertEqual(plan['planned_controls'],152)
        self.assertEqual(plan['planned_sampler_calls'],320)
        self.assertEqual(plan['extra_seeds'],[])
        self.assertEqual({c['letter']for c in plan['cases']},set('BCOX'))
        self.assertTrue(all(len(c['actions_absolute'])==19 and c['anchors']==[3,7,11,15]for c in plan['cases']))
        self.assertTrue(all(c['initial_material_clearance']>=15 for c in plan['cases']))

if __name__=='__main__':unittest.main()
