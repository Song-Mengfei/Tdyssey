"""Focused static-geometry/observed-trace checks; never constructs a simulator."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from tdyssey.native_rgb_contact_pilot_controller import ContactPilotPolicy,trace_observation,RULES
from tdyssey.native_rgb_r3a_collection import scene_candidate,PHASES,rot,segments
from tdyssey.native_rgb_r2b_sampling import contact_frame,geometry_at,_candidate,_free


class StateOnly:
    def __init__(self,state):self.current=np.asarray(state,np.float64)
    def state(self):return self.current.copy()


def fixture():
    candidate,reason=scene_candidate('L','train',0,0,0,20261031,'release_recontact')
    assert candidate is not None,reason
    env=StateOnly(candidate['initial_state15'])
    return env,ContactPilotPolicy(env,dict(letter='L',scenario='release_recontact'),candidate)


def trace(policy,*,contact_count=0,positive=True,wall=False,pre=False,local_point=None):
    point=policy.contact['point_local']if local_point is None else local_point
    normal=(rot(policy.env.state()[4])@np.asarray(policy.contact['outward_normal_local'])).tolist()
    rows=[]
    for i in range(10):
        events=[]
        if i>=10-contact_count:
            events.append(dict(phase='pre_solve'if pre else'post_solve',kind='agent_object',
                normal=normal,shapes_order=['object','agent'],total_impulse=[2.,0.]if positive else[0.,0.],
                points=[dict(object_local=list(point),arc_ids=[policy.contact['arc_id']])for _ in range(3)]))
        if wall:events.append(dict(phase='post_solve',kind='object_wall'))
        rows.append(dict(executed=True,contacts=events))
    return dict(substeps=rows,agent_object_contact_substeps=10)


class ContactPilotControllerTests(unittest.TestCase):
    def test_actual_postsolve_and_impulse_presence_not_aggregate_or_point_sum(self):
        _,p=fixture()
        observed=trace_observation(trace(p,contact_count=10,pre=True),p.contact)
        self.assertEqual(observed['AO'],0);self.assertFalse(observed['established'])
        observed=trace_observation(trace(p,contact_count=5),p.contact)
        self.assertEqual(observed['AO'],5);self.assertEqual(observed['nonzero_impulse'],5)
        self.assertTrue(observed['terminal_AO']);self.assertTrue(observed['established'])
        observed=trace_observation(trace(p,contact_count=10,positive=False),p.contact)
        self.assertEqual(observed['valid_impulse'],10);self.assertEqual(observed['nonzero_impulse'],0)
        self.assertFalse(observed['established'])
        with self.assertRaises(ValueError):trace_observation(dict(agent_object_contact_substeps=10))

    def test_confirmed_normal_entry_does_not_chase_old_staging_waypoint(self):
        env,p=fixture();state=env.state();point,normal,_=contact_frame(state,p.contact)
        env.current[:2]=point+64*normal;p.next_index=26;p.task_phase='withdraw'
        _,decision=p.next_action(26)
        self.assertEqual(decision['phase'],PHASES.index('recontact'))
        self.assertEqual(decision['planned_task_phase'],'recontact')
        self.assertEqual(decision['controller_state'],'recovery_entry')
        self.assertEqual(decision['mode'],2);self.assertTrue(decision['geometry_verified'])
        self.assertFalse(decision['actual_contact_established_before_action'])
        self.assertIsNone(decision['route'])
        self.assertGreater(decision['route_remaining_world_px'],40)

    def test_actual_contact_point_follows_current_body_and_velocity(self):
        env,p=fixture();before=env.state();action,_=p.next_action(0)
        actual=np.asarray(p.contact['point_local'])+np.array((0.,3.))
        after=before.copy();after[:4]+=np.array((10.,0.,10.,0.));after[7:9]=[20.,0.]
        env.current=after.copy();p.observe(before,action,after,trace(p,contact_count=10,local_point=actual))
        self.assertTrue(p.established);np.testing.assert_allclose(p.contact['point_local'],actual)
        _,decision=p.next_action(1)
        point,normal,tangent=contact_frame(after,p.contact)
        expected=point+normal*15.5-normal*48+tangent*48*p.contact['drive_tangent_ratio']+np.array((2.,0.))
        self.assertEqual(decision['controller_state'],'contact_follow')
        np.testing.assert_allclose(decision['raw_target'],expected,atol=1e-10)
        n=np.asarray(p.contact['outward_normal_local']);lever=float(actual[0]*(-n[1])-actual[1]*(-n[0]))
        self.assertAlmostEqual(decision['signed_lever'],lever)
        self.assertAlmostEqual(p.contact['signed_lever_world_px'],lever)

    def test_two_unreliable_contacts_demote_following_despite_touch_flags(self):
        env,p=fixture();p.established=True;p.awaiting_selected_contact=False
        for i in range(2):
            before=env.state();action,_=p.next_action(i)
            p.observe(before,action,before,trace(p,contact_count=10,positive=False))
            self.assertEqual(p.established,i==0)
        self.assertEqual(p.lost_contact,2)
        self.assertEqual(p.last_observation['AO'],10)

    def test_productive_route_not_reset_and_old_contact_cannot_undo_replacement(self):
        env,p=fixture();state=env.state();geometry=geometry_at('L',state)
        p.target_dwell=10;p.recovery_elapsed=100;p.eta_controls=1
        with patch.object(p,'_choose_replacement')as choose:
            self.assertIsNone(p._replace_if_needed(state,geometry,None));choose.assert_not_called()
        old=deepcopy(p.contact);fresh=None
        for segment in segments('L'):
            proposed,_=_candidate('L',state,geometry,segment,'generic','center',_free(geometry))
            if proposed and np.linalg.norm(np.asarray(proposed['point_local'])-old['point_local'])>40:
                fresh=proposed;break
        self.assertIsNotNone(fresh)
        p.wall_stall=3
        with patch.object(p,'_choose_replacement',return_value=fresh)as choose:
            action,decision=p.next_action(0);self.assertEqual(choose.call_count,1)
            self.assertEqual(p.target_revision,1)
            p.wall_stall=3
            self.assertIsNone(p._replace_if_needed(state,geometry,None));self.assertEqual(choose.call_count,1)
        # Still observing the departed old point is real contact, but not contact
        # with the new selected target and must not overwrite its local geometry.
        info=trace(p,contact_count=10,local_point=old['point_local'])
        p.observe(state,action,state,info)
        self.assertFalse(p.last_observation['selected_contact_established'])
        np.testing.assert_array_equal(p.contact['point_local'],fresh['point_local'])
        self.assertFalse(p.established)

    def test_fixed_withdraw_task_survives_stall_and_full_horizon(self):
        env,p=fixture();p.next_index=18;p.task_phase='push';p.target_dwell=10;p.wall_stall=99
        release=None
        for i in range(18,26):
            before=env.state();action,decision=p.next_action(i)
            self.assertEqual(decision['planned_task_phase'],'withdraw')
            self.assertEqual(decision['controller_state'],'withdraw');self.assertEqual(p.target_revision,0)
            if release is None:release=p.release_target.copy()
            else:np.testing.assert_array_equal(p.release_target,release)
            p.observe(before,action,before,trace(p))
        _,decision=p.next_action(26)
        self.assertEqual(decision['planned_task_phase'],'recontact')
        self.assertFalse(decision['actual_contact_established_before_action'])
        self.assertFalse(decision['task_budget_extended'])
        self.assertEqual(RULES['controls'],48)


if __name__=='__main__':unittest.main()
