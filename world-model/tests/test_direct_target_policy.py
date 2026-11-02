"""Geometry/fixture checks only: no simulator creation, reset or stepping."""
import unittest
from collections import Counter,defaultdict
from unittest.mock import patch
import numpy as np
from shapely.geometry import Point,LineString

from tdyssey.direct_target_policy import (build_plan,DirectTargetPolicy,
    validate_targets,geometry_at,LETTERS,SCENARIOS)


class DirectTargetPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('tdyssey.native_rgb_env.make_env',side_effect=AssertionError('physics forbidden')):
            cls.plan=build_plan()

    def test_finite_budget_and_independent_families(self):
        self.assertEqual(len(self.plan['families']),96)
        self.assertEqual(len(self.plan['attempts']),120)
        self.assertEqual(Counter(a['letter'] for a in self.plan['attempts']),{l:20 for l in LETTERS})
        self.assertEqual(Counter((f['letter'],f['scenario']) for f in self.plan['families']),
                         {(l,s):2 for l in LETTERS for s in SCENARIOS})
        self.assertEqual(sum(len(a['actions']) for a in self.plan['attempts']),7680)
        self.assertEqual(len({f['initial_state_sha256'] for f in self.plan['families']}),96)
        self.assertTrue(all(f['plan_valid'] for f in self.plan['families']))

    def test_original_domain_endpoints_and_no_limiting(self):
        a=np.asarray([[0.,512.],[512.,0.],[400.,400.]])
        self.assertTrue(np.array_equal(validate_targets(a),a))
        with self.assertRaises(ValueError):validate_targets([[-.001,100.]])
        with self.assertRaises(ValueError):validate_targets([[512.001,100.]])
        self.assertTrue(any(np.max(np.linalg.norm(np.diff(a['actions'],axis=0),axis=1))>24.
                            for a in self.plan['attempts']))
        self.assertTrue(any(np.max(a['actions'])==512. for a in self.plan['attempts']))

    def test_lookup_passes_exact_committed_target_without_state(self):
        for a in self.plan['attempts']:
            policy=DirectTargetPolicy(a)
            for i in (0,3,18,63):
                action,decision=policy.next_action(i)
                self.assertTrue(np.array_equal(action,np.asarray(a['actions'][i])))
                self.assertFalse(decision['target_clipped'])
                self.assertFalse(decision['future_state_read'])
                action[:]=0
                self.assertTrue(np.array_equal(policy.next_action(i)[0],np.asarray(a['actions'][i])))

    def test_branch_family_history_is_identical_and_future_different(self):
        groups=defaultdict(list)
        for a in self.plan['attempts']:groups[a['family_id']].append(a)
        self.assertEqual(sum(len(g)==3 for g in groups.values()),12)
        for group in groups.values():
            for a in group[1:]:
                self.assertEqual(a['initial_state15'],group[0]['initial_state15'])
                self.assertEqual(a['actions'][:3],group[0]['actions'][:3])
                self.assertNotEqual(a['actions'][3:19],group[0]['actions'][3:19])
                self.assertEqual(a['split'],group[0]['split'])

    def test_initial_scene_is_nonpenetrating_and_exterior(self):
        for f in self.plan['families']:
            state=np.asarray(f['initial_state15']);g=geometry_at(f['letter'],state)
            x0,y0,x1,y1=g.bounds
            self.assertGreaterEqual(min(x0,y0)-7.,-.000001)
            self.assertLessEqual(max(x1,y1),504.000001)
            self.assertGreaterEqual(g.distance(Point(*state[:2])),17.99999)
            self.assertTrue((state[:2]>=22.).all() and (state[:2]<=489.).all())
            self.assertEqual(f['selected_contact']['ring'],0)
            self.assertEqual(f['initial_pose_bin'],f['family_index']%8)

    def test_table_is_deterministic_and_modes_present(self):
        self.assertEqual(build_plan()['plan_sha256'],self.plan['plan_sha256'])
        self.assertEqual({f['target_mode'] for f in self.plan['families']},{'fixed','moving'})
        self.assertTrue({0,1,2}.issubset({f['target_distance_design']['requested_bin'] for f in self.plan['families']}))
        self.assertEqual({f['selected_boundary_intent'] for f in self.plan['families'] if f['scenario']=='boundary'},
            {'left','right','top','bottom','left-top','right-top','left-bottom','right-bottom'})

    def test_full_domain_negative_intent_has_static_object_clearance(self):
        full=[a for a in self.plan['attempts'] if a['scenario']=='negative' and a['target_mode']=='moving']
        self.assertEqual(len(full),6)
        for a in full:
            g=geometry_at(a['letter'],np.asarray(a['initial_state15']))
            self.assertGreaterEqual(g.distance(LineString((a['initialization']['pusher'],a['actions'][18]))),15.000001)
        self.assertTrue(any(a['target_distance_design']['actual_main_target_fraction_D']>2. for a in full))


if __name__=='__main__':unittest.main()
