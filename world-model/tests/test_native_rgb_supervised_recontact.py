"""Stage1B recovery tests use only geometric arrays/mock feedback, never physics."""

import unittest
import numpy as np

from tests.test_native_rgb_supervised_sampling import GeometricEnvironment
from tdyssey.native_rgb_data import contour_segments,_reachable_segments
from tdyssey.native_rgb_supervised_sampling import (plan_scenes,initialize_scene,ContinuousContactSampler,
    _free_route,RECOVERY_LIMITS)


class RecontactStateMachineTests(unittest.TestCase):
    def setup_policy(self,index=4,offset=18):
        env=GeometricEnvironment("T");scene=initialize_scene(env,plan_scenes({})[0]);segments=contour_segments("T")
        for row in _reachable_segments(env,segments):
            if not row["status"].startswith("reachable"):continue
            normal=np.asarray(row["normal"]);tangent=np.asarray((-normal[1],normal[0]));waypoint=np.asarray(row["staging"])
            start=waypoint+tangent*offset
            if _free_route(env.world_geometry(),start,waypoint):break
        else:raise AssertionError("No geometric tangential fixture")
        env.value[:2]=start;scene["staging_segment"]=row["segment_id"]
        counts={segment["id"]:1000 for segment in segments};counts[row["segment_id"]]=0
        policy=ContinuousContactSampler(env,segments,counts,scene)
        policy.index=index;policy.lost_streak=3;env.control_step=index
        return env,policy,segments

    def tick(self,env,policy,segments,*,contact=False,moving=False,path_fraction=.1,path_distance=None):
        before=env.state();action,metadata=policy.next_action(policy.index)
        np.testing.assert_array_equal(env.state(),before)
        if contact and moving:
            normal=policy._normal(before);env.mock_transition(-normal*2.1)
        else:
            if metadata["approach_kind"]=="recovery_path":
                delta=action-before[:2];length=float(np.linalg.norm(delta))
                if path_distance is None:env.value[:2]+=delta*path_fraction
                elif length:env.value[:2]+=delta/length*min(length,path_distance)
            env.control_step+=1
        after=env.state()
        points=np.asarray([segments[policy.active]["point"]]) if contact else np.empty((0,2))
        observed=policy.observe(before,action,after,dict(contact=int(contact),contact_points=points,valid=True))
        np.testing.assert_array_equal(env.state(),after)
        return before,action,metadata,observed

    def test_tangential_staging_path_uses_only_verified_bounded_lookahead_and_arrives(self):
        env,policy,segments=self.setup_policy();arrived=None
        for _ in range(8):
            before,action,metadata,observed=self.tick(env,policy,segments)
            recovery=observed["recovery"]
            if metadata["approach_kind"]=="recovery_path":
                self.assertTrue(_free_route(env.world_geometry(),before[:2],action))
                self.assertLessEqual(metadata["recovery"].get("lookahead_world_px",0),24)
            if recovery["staging_arrived"]:
                arrived=recovery;break
        self.assertIsNotNone(arrived)
        self.assertLessEqual(arrived["controls"],6)
        self.assertIsNone(arrived["first_real_contact_step"])
        self.assertFalse(arrived["contact_confirmed"])
        self.assertNotEqual(arrived["status"],"success")
        self.assertEqual(env.reset_calls,1)

    def test_staging_is_not_contact_and_missing_feedback_fails_boundedly(self):
        env,policy,segments=self.setup_policy();terminal=None;states=[]
        for _ in range(24):
            _,_,_,observed=self.tick(env,policy,segments)
            recovery=observed["recovery"]
            if recovery:
                states.append(recovery["state"])
                if recovery["status"]!="running":terminal=recovery;break
        self.assertIsNotNone(terminal);self.assertEqual(terminal["status"],"failed")
        self.assertEqual(terminal["failure_reason"],"no_real_contact_after_staging")
        self.assertIsNone(terminal["first_real_contact_step"])
        self.assertIn("staging",states);self.assertIn("final_contact",states)
        self.assertLessEqual(terminal["controls"],RECOVERY_LIMITS["episode_controls"])

    def test_success_requires_actual_contact_then_consecutive_contact_and_post_contact_motion(self):
        env,policy,segments=self.setup_policy()
        for _ in range(8):
            _,_,_,observed=self.tick(env,policy,segments)
            if observed["recovery"]["staging_arrived"]:break
        first_step=policy.index
        for i in range(3):
            _,_,_,observed=self.tick(env,policy,segments,contact=True,moving=True)
            recovery=observed["recovery"]
            self.assertEqual(recovery["first_real_contact_step"],first_step)
            if i<2:self.assertNotEqual(recovery["status"],"success")
        self.assertEqual(recovery["status"],"success")
        self.assertEqual(recovery["max_consecutive_contact_controls"],3)
        self.assertEqual(recovery["post_contact_contact_moving_controls"],2)
        self.assertEqual(recovery["end_step"],policy.index-1)
        _,_,metadata,observed=self.tick(env,policy,segments,contact=True,moving=True)
        self.assertIsNone(metadata["recovery"]);self.assertIsNone(observed["recovery"])

    def test_active_path_is_not_cut_off_by_sliding_or_retreat_phase_boundaries(self):
        env,policy,segments=self.setup_policy(index=39,offset=70);episode=None
        for step in range(39,49):
            _,_,metadata,observed=self.tick(env,policy,segments,path_distance=.5)
            recovery=observed["recovery"];episode=episode or recovery["episode_id"]
            self.assertEqual(recovery["episode_id"],episode)
            self.assertEqual(recovery["status"],"running")
            if step in (40,48):self.assertEqual(metadata["phase"],"contact_recovery_route")

    def test_last_control_logs_explicit_budget_exhaustion_without_another_proposal(self):
        env,policy,segments=self.setup_policy(index=63)
        _,_,_,observed=self.tick(env,policy,segments,path_fraction=0)
        recovery=observed["recovery"]
        self.assertEqual(recovery["status"],"failed")
        self.assertEqual(recovery["failure_reason"],"trajectory_budget_exhausted")
        self.assertEqual(recovery["start_step"],63);self.assertEqual(recovery["end_step"],63)
        self.assertIsNone(policy.recovery)

    def test_no_progress_has_bounded_alternatives_episode_attempts_and_failure_logs(self):
        env,policy,segments=self.setup_policy();terminal=[]
        while policy.index<64:
            _,_,_,observed=self.tick(env,policy,segments,path_fraction=0)
            recovery=observed["recovery"]
            if recovery and recovery["end_step"]==policy.index-1:terminal.append(recovery)
        self.assertLessEqual(policy.recovery_serial,RECOVERY_LIMITS["maximum_episodes"])
        self.assertTrue(terminal)
        self.assertTrue(all(r["status"]=="failed" for r in terminal))
        self.assertTrue(all(r["alternative_switches"]<=1 for r in terminal))
        self.assertTrue(all(r["controls"]<=20 for r in terminal))
        self.assertTrue(policy.failed_recovery_arcs)
        self.assertEqual(env.reset_calls,1)

    def test_final_contact_never_commands_outward_from_an_existing_contact_shell(self):
        env,policy,segments=self.setup_policy();policy._begin_recovery(policy.index,"lost_contact")
        record=policy.recovery;record.update(state="final_contact",stage="final_contact",staging_arrived=True)
        waypoint,contact_center,normal=policy._recovery_waypoint()
        env.value[:2]=contact_center-normal # still outside solid, 0.5px overlap of actual pusher radius
        target,kind,evidence=policy._recovery_action(policy.index)
        self.assertEqual(kind,"recovery_final_contact")
        self.assertGreater((target-env.state()[:2]).dot(-normal),0)
        self.assertTrue(record.get("final_approach_verified"))
        self.assertFalse(record["contact_confirmed"])


if __name__=="__main__":unittest.main()
