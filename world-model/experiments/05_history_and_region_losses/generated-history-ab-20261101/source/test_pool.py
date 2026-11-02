"""Meaningful offline timing/gap/isolation checks; no model or physics."""
import unittest
import numpy as np
from pool import group_rows,validate_arrays,normalized

class PoolTests(unittest.TestCase):
    def rows(self,anchors):
        return [dict(split='train',horizon=1,letter='C',family_id='f',raw_path='x',raw_sha256='a',anchor=a,
            task='rotation',sample_id=f'f:a{a}',diagnostic_contact_and_motion=True) for a in anchors]
    def test_gap_is_not_stitched(self):
        s,left=group_rows(self.rows([3,4,5,6,8,9,10,11,12]))
        self.assertEqual([x['anchor'] for x in s],[3,8]);self.assertEqual(left,['f:a12'])
    def fixture(self):
        acts=np.arange(40,dtype=np.float64).reshape(20,2)
        d=dict(rgb=np.zeros((21,3,96,96),np.uint8),passed_actions=acts,requested_actions=acts.copy(),control_valid=np.ones(20,bool))
        m=dict(split='train',family_id='f',clean_control_count=20,segments=[dict(start_control=3,end_control=19,committed_before_control=3,committed_actions=acts[3:19].tolist())])
        return d,m,group_rows(self.rows(range(3,7)))[0]
    def test_timing_shift_and_future_boundary_rejected(self):
        d,m,s=self.fixture();validate_arrays(d,m,s)
        m['segments'][0]['committed_actions']=d['passed_actions'][4:20].tolist()
        with self.assertRaisesRegex(ValueError,'Committed future'):validate_arrays(d,m,s)
        d,m,s=self.fixture();m['segments'][0]['end_control']=6
        with self.assertRaisesRegex(ValueError,'replanning'):validate_arrays(d,m,s)
    def test_validity_and_512_not_clipped(self):
        d,m,s=self.fixture();d['control_valid'][5]=False
        with self.assertRaisesRegex(ValueError,'invalid'):validate_arrays(d,m,s)
        d,m,s=self.fixture();d['passed_actions'][5,0]=512;d['requested_actions'][5,0]=512
        with self.assertRaisesRegex(ValueError,'contract'):validate_arrays(d,m,s)
        self.assertEqual(float(normalized(np.asarray([511],np.float64))[0]),1.)

if __name__=='__main__':unittest.main()
