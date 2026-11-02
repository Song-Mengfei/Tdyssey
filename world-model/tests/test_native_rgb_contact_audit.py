"""CPU fixtures for saved-array contact audit; no simulation/model imports."""
import math
import unittest
import numpy as np
from tdyssey.native_rgb_contact_audit import (actual_signals,boolean_runs,control_arc_summary,
    classify_motion,motion,b014_regression,PHASES)


def fixture(m=10):
    return dict(contact_flags=np.zeros((m,3),np.uint8),contact_tags=np.zeros((0,5),np.int16),
        contact_values=np.zeros((0,11),np.float32),contact_arc_offsets=np.zeros(1,np.int32),
        contact_arc_ids=np.zeros(0,np.int16))


class ContactAuditContracts(unittest.TestCase):
    def test_postsolve_only_impulse_existence_not_point_sum(self):
        a=fixture();a['contact_flags'][0,0]=2;a['contact_flags'][1,0]=1;a['contact_flags'][2,0]=2
        # Same event copied twice plus another arbiter row. Presence remains one
        # substep, and epsilon applies to each individual vector, never their sum.
        a['contact_tags']=np.array([[0,0,1,1,4],[0,0,1,1,4],[0,0,1,1,4],[1,0,0,1,4]],np.int16)
        a['contact_values']=np.zeros((4,11),np.float32);a['contact_values'][:,7]=[7e-7,7e-7,2e-7,1.]
        a['contact_arc_offsets']=np.zeros(5,np.int32)
        s=actual_signals(a)
        np.testing.assert_array_equal(s['ao'],[True,False,True]+[False]*7)
        np.testing.assert_array_equal(s['nonzero'].sum(0),[1,1,0,0])
        self.assertAlmostEqual(s['impulse_max_norm'][0],7e-7,places=12)
        self.assertTrue(np.isnan(s['impulse_max_norm'][2]));self.assertFalse(s['impulse_known'][2])

    def test_arc_votes_distinct_substeps_and_minimum_id_tie(self):
        # Arc9 appears twenty times at one substep, arc3 spans two substeps.
        # Point multiplicity never outweighs actual time coverage.
        s=dict(arcs=[{9,3},{3}]+[set() for _ in range(8)]+[{7},{2}]+[set() for _ in range(8)])
        dominant,offsets,ids=control_arc_summary(s,2)
        np.testing.assert_array_equal(dominant,[3,2]);np.testing.assert_array_equal(offsets,[0,2,4])
        np.testing.assert_array_equal(ids,[3,9,2,7])

    def test_nearstatic_obvious_and_intermediate_exact_thresholds(self):
        classes,intermediate=classify_motion([.099,.1,2.,.49,.5,0.],[.099,0.,0.,1.,0.,5.])
        np.testing.assert_array_equal(classes,[0,1,2,1,1,2]);np.testing.assert_array_equal(intermediate,[0,0,1,1,1,1])

    def test_rotation_uses_area_centroid_not_body_origin(self):
        states=np.zeros((2,15),np.float64);states[1,4]=math.pi/2
        trans,rot,push=motion(states,'T')
        self.assertGreater(trans[0],0.)
        np.testing.assert_allclose(rot,[90.]);np.testing.assert_array_equal(push,[0.])

    def test_runs_do_not_merge_trajectory_boundaries(self):
        self.assertEqual(boolean_runs([1,1,0,1]),[(0,2),(3,4)])
        self.assertEqual(boolean_runs([0,1,1]),[(1,3)])

    def test_B014_interval_start_and_terminal_frame_regression(self):
        a=fixture(480);a['contact_flags'][30:181,0]=2;a['contact_flags'][391:443,0]=2
        s=actual_signals(a);phases=np.zeros(48,np.uint8);phases[26:31]=PHASES.index('recontact')
        r=b014_regression('R3A-seen_val-B-014-a0',s,phases,'release_recontact',['B:translation'])
        self.assertAlmostEqual(r['recontact_latency_seconds'],1.31)
        self.assertAlmostEqual(r['observed_poststate_delay_seconds'],1.32)
        self.assertEqual(r['endpoint_postsolve_contact'],{'0':None,'12':True,'24':False,'36':False,'48':False})


if __name__=='__main__':unittest.main()
