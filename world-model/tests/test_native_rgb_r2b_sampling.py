import unittest
import numpy as np
from shapely.geometry import box
from tdyssey.native_rgb_r2b_sampling import (FeedbackPolicy,RULES,PROTOCOL_SHA256,_bounded,_candidate,contact_frame,free_route,planned_phase)


class FakeEnv:
    def __init__(self,state):self.value=np.asarray(state,dtype=float)
    def state(self):return self.value.copy()
    def step(self,*args):raise AssertionError("No real/synthetic physics step permitted")


class R2BSamplingTests(unittest.TestCase):
    def request(self):
        return dict(letter="T",dispatchable=True,protocol_sha256=PROTOCOL_SHA256,
            selected_contact=dict(arc_id=0,point_local=[0.,0.],outward_normal_local=[0.,-1.],drive_tangent_ratio=0.))

    def test_phases_exact_budget_and_h4(self):
        self.assertEqual([sum(planned_phase(i)==p for i in range(24)) for p in ('approach','push','withdraw','recontact')],[4,10,6,4])
        self.assertEqual(sum(all(planned_phase(i)=='push' for i in range(a,a+4)) for a in range(3,21)),7)
        with self.assertRaises(ValueError):planned_phase(24)

    def test_free_path_does_not_cross_object_and_routes_outside(self):
        geom=box(200,200,300,300)
        route=free_route(geom,[170,250],[330,250]);self.assertIsNotNone(route);self.assertGreater(len(route),2)
        self.assertIsNone(free_route(geom,[250,250],[330,250]))

    def test_holes_rejected_and_same_source_pose_transform(self):
        state=np.zeros(15);state[2:5]=[256,200,np.pi/2]
        p,n,t=contact_frame(state,dict(point_local=[20,0],outward_normal_local=[1,0]))
        np.testing.assert_allclose(p,[256,220]);np.testing.assert_allclose(n,[0,1],atol=1e-12)
        value,reason=_candidate('T',state,box(0,0,20,20),dict(ring=1),'translation_1','center')
        self.assertIsNone(value);self.assertIn('closed_hole',reason)

    def test_target_limits_and_no_model_or_physics(self):
        result=_bounded([900,900],np.array([250.,250.]),np.array([250.,250.]))
        self.assertLessEqual(np.linalg.norm(result-[250,250]),64.000001)
        self.assertTrue(((result>=22.5)&(result<=488.5)).all())
        env=FakeEnv([256,166,256,200,0]+[0]*10);policy=FeedbackPolicy(env,self.request())
        action,decision=policy.next_action(0)
        self.assertEqual(decision['fixed_arc_id'],0);self.assertEqual(decision['used_future_state'],False)
        self.assertEqual(decision['current_state'],env.state().tolist())
        observed=policy.observe(env.state(),action,env.state(),{'agent_object_contact_substeps':0})
        self.assertEqual(observed['actual_contact'],False);self.assertFalse(observed['phase_budget_extended'])
        with self.assertRaises(ValueError):policy.next_action(0)

    def test_joint_command_balls_wall_projection_and_infeasible_fail_closed(self):
        for current,last,raw in (([200,200],[110,200],[264,240]),([22,200],[40,210],[-50,300])):
            current,last=np.asarray(current,dtype=float),np.asarray(last,dtype=float)
            target=_bounded(raw,current,last)
            self.assertLessEqual(np.linalg.norm(target-current),64+1e-8)
            self.assertLessEqual(np.linalg.norm(target-last),96+1e-8)
            self.assertTrue(((target>=22.5)&(target<=488.5)).all())
        with self.assertRaises(ValueError):_bounded([250,250],np.array([400.,400.]),np.array([40.,40.]))

    def test_world_follow_updates_after_pose_change_without_placement(self):
        env=FakeEnv([256,166,256,200,0]+[0]*10);policy=FeedbackPolicy(env,self.request())
        a,d=policy.next_action(0);policy.observe(env.state(),a,env.state(),{'agent_object_contact_substeps':1})
        env.value[2]+=7;env.value[4]=.1
        a2,d2=policy.next_action(1)
        self.assertNotEqual(d['contact_point_world'],d2['contact_point_world'])
        self.assertEqual(d2['fixed_arc_id'],d['fixed_arc_id']);self.assertEqual(d2['lookahead_physics_steps'],0)

    def test_full24_pure_state_replay_is_exact_and_never_steps(self):
        policies=[FeedbackPolicy(FakeEnv([256,166,256,200,0]+[0]*10),self.request()) for _ in range(2)]
        for t in range(24):
            before=np.array([256,166,256+.1*t,200,.001*t]+[0]*10,dtype=float)
            after=np.array([256,166,256+.1*(t+1),200,.001*(t+1)]+[0]*10,dtype=float)
            results=[]
            for policy in policies:
                policy.env.value=before.copy();action,decision=policy.next_action(t)
                self.assertLessEqual(np.linalg.norm(action-before[:2]),64+1e-8)
                results.append((action.tolist(),decision))
                policy.observe(before,action,after,{'agent_object_contact_substeps':8 if 4<=t<14 else 0})
            self.assertEqual(results[0],results[1])
        self.assertEqual(len(policies[0].history),24)
        with self.assertRaises(ValueError):FeedbackPolicy(FakeEnv(before),self.request(),dict(RULES,normal_advance_world_px=49.))


if __name__=='__main__':unittest.main()
