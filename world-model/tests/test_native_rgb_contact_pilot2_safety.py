"""Current-state regressions for the production CP2 wall-risk gate, no physics."""
import json
from pathlib import Path
import unittest

import numpy as np

from tdyssey.native_rgb_contact_pilot2_safety import assess_wall_risk
from tdyssey.native_rgb_r2b_sampling import geometry_at,wall_gaps
from tdyssey.native_rgb_r3a_collection import rot

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/native-rgb-supervised-20261029/contact-pilot1-20261030/trajectories'


def saved(name):
    with np.load(DATA/(name+'.npz'),allow_pickle=False)as archive:
        return {key:archive[key]for key in('states','attempted_actions','control_point_local','control_normal_local')}


def contact(arrays,index):
    return dict(point_local=arrays['control_point_local'][index],outward_normal_local=arrays['control_normal_local'][index])


class ContactPilot2SafetyTests(unittest.TestCase):
    def test_actual_x_guard_trace_warns_before_control5(self):
        arrays=saved('CP1-A-X-03');before=arrays['states'].copy();actions=arrays['attempted_actions'].copy()
        risks=[assess_wall_risk('X',state,actions[i],contact=contact(arrays,i))
               for i,state in enumerate(arrays['states'])]
        self.assertEqual([r['risky']for r in risks],[False,False,False,True,True,True])
        self.assertIn('top',risks[3]['hazardous_walls'])
        self.assertLess(risks[3]['wall_risks']['top']['projected_remaining_room_world_px'],8.)
        self.assertAlmostEqual(risks[3]['wall_risks']['top']['gap_world_px'],32.471652830170356)
        self.assertTrue(np.array_equal(arrays['states'],before))
        self.assertTrue(np.array_equal(arrays['attempted_actions'],actions))
        self.assertTrue(all(r['physics_rollout_steps']==0 and not r['state_modified']for r in risks))

    def test_rotating_corner_is_not_hidden_by_zero_COM_translation(self):
        arrays=saved('CP1-A-X-03');state=arrays['states'][0].copy();target=contact(arrays,0)
        shift=25.-wall_gaps(geometry_at('X',state))['top'];state[3]+=shift;state[1]+=shift
        normal=rot(state[4])@target['outward_normal_local'];normal/=np.linalg.norm(normal)
        point=state[2:4]+rot(state[4])@target['point_local']
        state[:2]=point+normal*15.5;state[7:10]=0
        action=state[:2]-normal*8.
        still=assess_wall_risk('X',state,action,contact=target)
        self.assertFalse(still['risky'])
        # The closest top corner is right of the physical COM, so clockwise
        # screen rotation gives that actual corner a velocity toward the top.
        state[9]=-2.5
        turning=assess_wall_risk('X',state,action,contact=target)
        self.assertTrue(turning['risky']);self.assertIn('top',turning['hazardous_walls'])
        self.assertEqual(turning['object_COM_velocity'],[0.,0.])
        self.assertGreater(turning['wall_risks']['top']['closing_boundary_speed_world_px_s'],100.)
        self.assertGreater(turning['rotation_curvature_allowance_world_px'],0.)

    def test_mode_label_cannot_bypass_inward_action_but_retreat_is_mitigation(self):
        arrays=saved('CP1-A-X-03');state=arrays['states'][3];target=contact(arrays,3)
        for label in('follow','entry','route','withdraw'):
            risk=assess_wall_risk('X',state,arrays['attempted_actions'][3],contact=target,mode=label)
            self.assertTrue(risk['risky'])
        state=arrays['states'][5];target=contact(arrays,5)
        normal=rot(state[4])@target['outward_normal_local'];normal/=np.linalg.norm(normal)
        retreat=assess_wall_risk('X',state,state[:2]+18*normal,contact=target,mode='retreat')
        self.assertFalse(retreat['pushing_command']);self.assertFalse(retreat['risky'])
        self.assertTrue(retreat['state_risky'])
        self.assertTrue(retreat['mitigation_does_not_prove_safety'])

    def test_real_T040_open_commands_are_not_all_suppressed(self):
        arrays=saved('CP1-A-T-01')
        for index in(0,3,30):
            risk=assess_wall_risk('T',arrays['states'][index],arrays['attempted_actions'][index],contact=contact(arrays,index))
            self.assertFalse(risk['risky'])
            self.assertTrue(risk['heuristic_not_mathematical_guarantee'])

    def test_physical_T_COM_and_damping_are_logged_without_state_change(self):
        arrays=saved('CP1-A-T-01');state=arrays['states'][3]
        risk=assess_wall_risk('T',state,arrays['attempted_actions'][3],contact=contact(arrays,3))
        expected=state[2:4]+rot(state[4])@np.array((0.,45.))
        self.assertTrue(np.allclose(risk['physical_COM_world'],expected,rtol=0,atol=1e-12))
        self.assertAlmostEqual(risk['damping_effective_seconds'],.1/(2*np.log(2)))
        self.assertGreater(risk['union_exterior_vertices_checked'],4)

    def test_zero_velocity_TTC_uses_null_and_every_value_is_JSON_safe(self):
        arrays=saved('CP1-A-X-03')
        risk=assess_wall_risk('X',arrays['states'][0],arrays['attempted_actions'][0],contact=contact(arrays,0))
        self.assertTrue(all(row['time_to_wall_seconds']is None for row in risk['wall_risks'].values()))
        encoded=json.dumps(risk,allow_nan=False)
        self.assertNotIn('Infinity',encoded);self.assertNotIn('NaN',encoded)


if __name__=='__main__':unittest.main()
