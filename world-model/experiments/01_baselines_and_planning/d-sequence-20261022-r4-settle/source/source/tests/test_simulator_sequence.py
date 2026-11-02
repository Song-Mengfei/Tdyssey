"""Deterministic protocol tests; fake dynamics are not D performance evidence."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from test_simulator_search import HistoryEnv, LocalRngDouble, initial_state
from tdyssey.contact_geometry import direct_clear, rectangle_exterior_edges
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.simulator_planning import state_error
from tdyssey.simulator_sequence import (ContactSequenceSearch, SequenceConfig, ContactStage,
    ProposalRejected, SearchBudgetExhausted, pose_rms, approach_path, diverse_prefixes,
    stage_actions, settling_reserve, path_key, joint_limits, joint_errors, family_frontier)
from tdyssey.state import BodyState


class Rng(LocalRngDouble):
    def integers(self, count):
        return self.generator.randrange(count)


def builder(shape, state, stage, budget, maximum):
    action = (245., 256.) if stage.edge == 0 else (256., 256.)
    return (action,), ["push"], {"approach_controls": 0, "push_controls": 1}


class SequenceChecks(unittest.TestCase):
    def planner(self, state=None, **kw):
        opts = dict(horizon=6, settle_steps=2, roots=2, beam_width=2,
                    children=2, refinements=0, max_evaluations=8)
        opts.update(kw)
        state = state or initial_state()
        search = ContactSequenceSearch(lambda: HistoryEnv(state), state, SequenceConfig(**opts))
        search.rng = Rng(7)
        search._proposals = lambda current, count: [ContactStage(0 if current.control_step == 0 else 1)]*count
        self.addCleanup(search.close)
        return search

    def test_joint_pose_error_sees_orientation_even_when_center_matches(self):
        shape, state = official_shape("D"), initial_state()
        exact = replace(state, object_body=BodyState(state.goal))
        wrong = replace(exact, object_body=BodyState(Pose(state.goal.position, math.radians(100))))
        self.assertEqual(pose_rms(shape, exact), 0)
        self.assertGreater(pose_rms(shape, wrong), 50)
        displaced = replace(exact, object_body=BodyState(Pose((state.goal.position[0]+20, state.goal.position[1]), 0)))
        self.assertLess(pose_rms(shape, displaced), pose_rms(shape, wrong))

    def test_two_contacts_reconstruct_entire_prefix_and_match_actual_continuation(self):
        search, state = self.planner(), initial_state()
        live = HistoryEnv(state)
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            decision = search.choose(state, 20)
        self.assertEqual(live.capture_state(), state)
        self.assertEqual(decision.report["selected"]["kind"], "second_contact")
        self.assertGreater(search.counts["candidate_steps"], 0)
        for index, action in enumerate(decision.actions):
            live.step(action)
            expected = type(state).from_dict(decision.report["selected_predictions"][index])
            self.assertEqual(state_error(live.capture_state(), expected), 0)
            search.record_action(action)
        self.assertEqual(search.counts["restores"], decision.report["evaluations"])

    def test_second_contact_is_generated_from_first_attained_state(self):
        search = self.planner()
        calls = []
        def proposals(state, count):
            calls.append((state.control_step, state.object_body.pose.position))
            return [ContactStage(0 if state.control_step == 0 else 1)]*count
        search._proposals = proposals
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            search.choose(initial_state(), 20)
        self.assertEqual(calls[0], (0, (220., 256.)))
        self.assertTrue(all(step == 1 and abs(position[0]-245.01) < 1e-8 for step, position in calls[1:]))

    def test_continuation_reuses_predictions_without_new_physics(self):
        search, state = self.planner(), initial_state()
        live = HistoryEnv(state)
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            first = search.choose(state, 20)
            live.step(first.actions[0])
            search.record_action(first.actions[0])
            old = dict(search.counts)
            second = search.choose(live.capture_state(), 19)
        self.assertTrue(second.report["continuation"])
        self.assertEqual(search.counts, old)
        self.assertEqual(second.actions[0], first.actions[1])
        self.assertEqual(search.global_searches, 1)

    def test_precision_executes_the_settling_prefix_used_by_terminal_constraints(self):
        search,state=self.planner(),initial_state()
        live=HistoryEnv(state)
        with patch("tdyssey.simulator_sequence.stage_actions",side_effect=builder):
            first=search.choose(state,20)
            selected=first.report["selected"]
            active=selected["policy_prefix"]
            self.assertEqual(selected["execution_controls"],active+2)
            for action in first.actions[:active]:
                live.step(action);search.record_action(action);search.check_execution(live.capture_state())
            old=dict(search.counts)
            settling=search.choose(live.capture_state(),20-active)
            self.assertTrue(settling.report["continuation"])
            self.assertEqual(settling.report["execute_steps"],2)
            self.assertEqual(search.counts,old)
            for action in settling.actions[:2]:
                live.step(action);search.record_action(action)
                result=search.check_execution(live.capture_state())
            self.assertTrue(result["macro_terminal"])
            self.assertTrue(result["terminal_ok"])
            self.assertEqual(search._active,0)

    def test_rejections_count_and_all_completed_paths_share_remaining_horizon(self):
        # The fake object follows the target: put the pusher at the object so
        # this budget fixture's physical hold does not teleport it off-goal.
        state=replace(initial_state(),pusher_body=initial_state().object_body)
        search = self.planner(state=state,max_evaluations=3)
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=ProposalRejected("unsafe")):
            decision = search.choose(state, 3)
        self.assertEqual(decision.report["evaluations"], 3)
        self.assertEqual(len(decision.actions), 3)
        self.assertEqual(sum(r["status"] == "rejected" for r in decision.report["candidates"]), 2)

    def test_replay_errors_are_never_hidden_as_bad_proposals(self):
        search = self.planner()
        with patch.object(search.branch, "_replay", side_effect=ValueError("bad history")):
            with self.assertRaisesRegex(ValueError, "bad history"):
                search.choose(initial_state(), 20)

    def test_deadline_discards_partial_candidate_and_retains_cost(self):
        state=replace(initial_state(),pusher_body=initial_state().object_body)
        search, clock = self.planner(state=state,horizon=4, settle_steps=1), [0.]
        original = search.branch.env.step
        def slow(action):
            clock[0] += 1
            return original(action)
        with patch.object(search.branch.env, "step", side_effect=slow), \
                patch("tdyssey.simulator_planning.time.monotonic", side_effect=lambda: clock[0]), \
                patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            result = search.choose(state, 20, deadline=4.5)
        self.assertTrue(result.report["timed_out"])
        self.assertEqual(result.report["selected"]["kind"], "hold")
        self.assertEqual(search.counts["candidate_steps"], 5)

    def test_native_success_is_absorbing_without_free_settling(self):
        search = self.planner()
        search.branch.env.success_target = (245., 256.)
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            result = search.choose(initial_state(), 20)
        self.assertTrue(result.report["selected"]["native_success"])
        self.assertEqual(len(result.actions), 1)

    def test_global_budget_and_wrong_actual_prediction_stop_explicitly(self):
        search = self.planner(max_global_searches=1)
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            result = search.choose(initial_state(), 20)
        live = HistoryEnv(initial_state())
        for action in result.actions:
            live.step(action)
            search.record_action(action)
        with self.assertRaises(SearchBudgetExhausted):
            search.choose(live.capture_state(), 10)
        with self.assertRaises(ValueError):
            search.choose(replace(live.capture_state(), pusher_target=(1., 1.)), 10)

    def test_diversity_retains_rotation_prefix_with_worse_overlap(self):
        search = self.planner()
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=builder):
            result = search.choose(initial_state(), 20)
        records = []
        for i, (angle, error) in enumerate(((1.5, 10.), (.4, 60.), (1.4, 12.))):
            records.append(dict(unsafe=False, native_success=False, rows=[{}], pose_rms=error,
                final_coverage=.6, motion=0, approach_controls=0, prefix_state=initial_state(),
                prefix_angle_error=angle, prefix_rms=error, diversity_key=(i, 1, 1)))
        chosen = diverse_prefixes(records, 2)
        self.assertIn(records[1], chosen)

    def test_exterior_and_visibility_preserve_holes_and_collision_clearance(self):
        shape = official_shape("D")
        state = replace(initial_state(), object_body=BodyState(Pose((256., 256.), .3)),
                        pusher_body=BodyState(Pose((110., 256.), 0)))
        edge = max(rectangle_exterior_edges(shape.components), key=lambda e: e.point(.5)[0])
        q = state.object_body.pose.transform(edge.point(.5))
        normal = edge.normal
        c, s = math.cos(.3), math.sin(.3)
        normal = (c*normal[0]-s*normal[1], s*normal[0]+c*normal[1])
        target = tuple(q[k]+18*normal[k] for k in (0, 1))
        path, _ = approach_path(shape, state, target, 1)
        self.assertTrue(all(direct_clear(a, b, shape, state.object_body.pose, 15.9) for a, b in zip(path, path[1:])))

    def test_corner_release_uses_valid_disc_clearance_not_an_arbitrary_face_normal(self):
        shape = official_shape("D")
        corner = min((e.a for e in rectangle_exterior_edges(shape.components)))
        pose = Pose((256., 256.), 0.)
        state = replace(initial_state(), object_body=BodyState(pose),
            pusher_body=BodyState(Pose((256+corner[0]-14.9, 256+corner[1]), 0)))
        target = (340., 256.)
        path, _ = approach_path(shape, state, target, 1)
        self.assertTrue(direct_clear(path[1], path[1], shape, pose, 16.))
        self.assertTrue(direct_clear(path[0], path[1], shape, pose, 14.8))

    def test_single_contact_can_use_short_remaining_budget_without_pair_reservation(self):
        search = self.planner(horizon=6, settle_steps=2)
        def needs_two(shape, state, stage, budget, maximum):
            if budget < 2:
                raise ProposalRejected("needs two actual controls")
            return ((245.,256.),(256.,256.)), ["push"]*2, {"approach_controls":0, "push_controls":2}
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=needs_two):
            result = search.choose(initial_state(), 4)
        self.assertTrue(any(r["kind"]=="first_contact" and r["status"]=="completed" for r in result.report["candidates"]))

    def test_near_goal_strokes_scale_down_without_changing_contact_categories(self):
        search = self.planner()
        # Exercise real proposal generation rather than the fixture override.
        del search._proposals
        state = replace(initial_state(), object_body=BodyState(Pose((255.,256.), .02)))
        specs = search._proposals(state, 8)
        self.assertTrue(all(s.travel < 5 for s in specs))

    def test_geometric_rejections_do_not_replay_the_whole_real_history(self):
        search = self.planner(max_evaluations=3)
        live = HistoryEnv(initial_state())
        live.step((160.,180.))
        search.record_action((160.,180.))
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=ProposalRejected("bad geometry")):
            result = search.choose(live.capture_state(), 10)
        self.assertEqual(result.report["evaluations"], 3)
        self.assertEqual(search.counts["restores"], 1)
        self.assertEqual(search.counts["replay_steps"], 1)

    def test_joint_refinement_rebuilds_second_contact_after_mutating_first(self):
        search = self.planner(refinements=4, max_evaluations=12)
        calls = []
        def parameterized(shape, state, stage, budget, maximum):
            calls.append((stage.edge, stage.u, state.object_body.pose.position))
            x = (230.+30*stage.u) if stage.edge == 0 else (245.+20*stage.u)
            return ((x,256.),), ["push"], {"approach_controls":0, "push_controls":1}
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=parameterized):
            result = search.choose(initial_state(), 20)
        refined = [r for r in result.report["candidates"] if r["kind"]=="joint_refinement" and r["status"]=="completed"]
        self.assertTrue(refined)
        self.assertTrue(any(len(r["stages"])==2 and r["stages"][0]["u"] != .5 for r in refined))
        self.assertTrue(any(edge==1 and abs(position[0]-245.01) > 1e-5 for edge, u, position in calls))

    def test_last_five_controls_are_available_for_a_real_push(self):
        search = self.planner(horizon=6, settle_steps=4)
        def needs_four(shape, state, stage, budget, maximum):
            if budget < 4:
                raise ProposalRejected("requires four controls")
            return ((245.,256.),)*4, ["push"]*4, {"approach_controls":0,"push_controls":4}
        with patch("tdyssey.simulator_sequence.stage_actions", side_effect=needs_four):
            result = search.choose(initial_state(), 5)
        self.assertEqual(result.report["settling_reserve"], 0)
        self.assertEqual(result.report["active_budget"], 5)
        self.assertTrue(any(r["status"]=="completed" and r["policy_prefix"]==4 for r in result.report["candidates"]))
        self.assertEqual(len(result.actions), 5)
        moving = replace(initial_state(), object_body=replace(initial_state().object_body, velocity=(30.,0.)))
        self.assertEqual(settling_reserve(moving, 5, 4), 0)
        self.assertEqual(settling_reserve(moving, 80, 4), 4)

    def precision_fixture(self, distance):
        shape = official_shape("D")
        edges = rectangle_exterior_edges(shape.components)
        edge_id = max(range(len(edges)), key=lambda i: edges[i].point(.5)[0])
        q, normal = edges[edge_id].point(.5), edges[edge_id].normal
        state = replace(initial_state(), object_body=BodyState(Pose((256.,256.),0)),
            pusher_body=BodyState(Pose(tuple(256+q[k]+distance*normal[k] for k in (0,1)),0)),
            pusher_target=None)
        return shape, state, edge_id

    def test_precision_contact_uses_slow_increments_and_physical_press_dwell(self):
        shape,state,edge = self.precision_fixture(15.5)
        spec = ContactStage(edge,.5,.5,1.,speed=12.,approach_speed=220.,sustain=3,precision=True)
        actions, phases, detail = stage_actions(shape,state,spec,8)
        self.assertEqual(detail["route_mode"], "maintain_nearby_contact")
        self.assertEqual(detail["approach_controls"],0)
        self.assertEqual(detail["sustain_controls"],3)
        self.assertEqual(phases,["push"]*5)
        self.assertEqual(actions[-1],actions[-4])
        self.assertGreater(math.dist(actions[-1],state.pusher_body.pose.position),1.)
        self.assertTrue(all(math.dist(a,b)<=1.2+1e-9 for a,b in zip([state.pusher_body.pose.position,*actions],actions)))
        short,_,short_detail=stage_actions(shape,state,spec,2)
        self.assertEqual(len(short),2)
        self.assertEqual(short_detail["sustain_controls"],0)

    def test_precision_approach_speed_is_separate_from_push_speed(self):
        shape,state,edge=self.precision_fixture(90.)
        spec=ContactStage(edge,.5,1.,4.,speed=20.,approach_speed=220.,sustain=3,precision=True)
        actions,_,detail=stage_actions(shape,state,spec,20)
        self.assertLessEqual(detail["approach_controls"],4)
        self.assertGreaterEqual(detail["push_controls"],6)
        self.assertLessEqual(len(actions),20)

    def test_compensating_contact_includes_an_opposite_face_from_attained_pose(self):
        search=self.planner()
        del search._proposals
        search._precision=True
        shape,state,edge=self.precision_fixture(18.)
        state=replace(state,goal=Pose((260.,253.),.15))
        children=search._paired_proposals(state,ContactStage(edge),4)
        normal=rectangle_exterior_edges(shape.components)[edge].normal
        edges=rectangle_exterior_edges(shape.components)
        self.assertEqual(len(children),4)
        self.assertTrue(all(s.precision and s.sustain>0 for s in children))
        self.assertTrue(any(sum(normal[k]*edges[s.edge].normal[k] for k in (0,1))<-.5 for s in children))

    def test_completion_cost_prefers_access_but_cannot_override_native_success(self):
        row=dict(unsafe=False,native_success=False,rows=[{}],pose_rms=5.,completion_score=6.,
                 final_coverage=.9,motion=0.,approach_controls=20)
        nearby=dict(row,completion_score=5.1,approach_controls=1)
        success=dict(row,native_success=True,pose_rms=8.,completion_score=50.)
        self.assertLess(path_key(nearby),path_key(row))
        self.assertLess(path_key(success),path_key(nearby))

    def test_separate_constraints_reject_center_drift_even_if_joint_rms_decreases(self):
        shape,state=official_shape("D"),initial_state()
        near=replace(state,object_body=BodyState(Pose((258.,256.),math.radians(12))))
        limits=joint_limits(near,shape)
        drift=replace(state,object_body=BodyState(Pose((262.,256.),math.radians(5))))
        good=replace(state,object_body=BodyState(Pose((258.,256.),math.radians(10))))
        self.assertLess(pose_rms(shape,drift),pose_rms(shape,good))
        base=dict(unsafe=False,native_success=False,rows=[{}],pose_rms=20.,final_coverage=.8,
                  motion=0,approach_controls=1)
        bad_record=dict(base,**joint_errors([drift]*3,shape,limits))
        good_record=dict(base,**joint_errors([good]*3,shape,limits))
        self.assertFalse(bad_record["joint_feasible"])
        self.assertTrue(good_record["joint_feasible"])
        self.assertLess(path_key(good_record),path_key(bad_record))

    def test_limits_ratchet_between_macros_but_allow_compensated_intermediate_motion(self):
        shape,state=official_shape("D"),initial_state()
        def at(x,angle=10):
            return replace(state,object_body=BodyState(Pose((256+x,256),math.radians(angle))))
        first=joint_limits(at(6),shape)
        second=joint_limits(at(2),shape,first)
        self.assertLess(second["center_limit"],first["center_limit"])
        self.assertEqual(second["center_limit"],3.)
        self.assertTrue(joint_errors([at(5),at(2),at(2),at(2)],shape,second)["joint_feasible"])
        self.assertFalse(joint_errors([at(7),at(2),at(2),at(2)],shape,second)["joint_feasible"])
        search=self.planner()
        search._limits=second
        search._selected={"native_success":False}
        with self.assertRaisesRegex(ValueError,"corridor"):
            search.check_execution(at(7))

    def test_press_duration_variants_consume_existing_attempt_cap(self):
        search=self.planner(refinements=8,max_evaluations=22)
        with patch("tdyssey.simulator_sequence.stage_actions",side_effect=builder):
            result=search.choose(initial_state(),20)
        variants=[r for r in result.report["candidates"] if r["kind"]=="press_duration"]
        self.assertEqual({r["stages"][0]["sustain"] for r in variants},{0,1,2,3})
        self.assertEqual(result.report["evaluations"],22)

    def test_refinement_frontier_preserves_position_angle_and_compensation_families(self):
        base=dict(unsafe=False,native_success=False,rows=[{}],pose_rms=10.,
            final_coverage=.8,motion=0.,approach_controls=1,stage_objects=(ContactStage(0),))
        rows=[dict(base,final_position_error=p,final_angle_error=a,completion_score=c,
                   actual_compensation=comp) for p,a,c,comp in
              ((1.,.4,12.,False),(8.,.1,9.,False),(3.,.2,8.,False),(4.,.3,13.,True))]
        frontier=family_frontier(rows)
        self.assertEqual(len(frontier),4)
        self.assertIs(frontier[0],rows[0])
        self.assertIs(frontier[1],rows[1])
        self.assertIs(frontier[2],rows[2])
        self.assertIs(frontier[3],rows[3])

    def test_third_contact_replays_both_hypothetical_stages_and_real_history(self):
        state=initial_state()
        live=HistoryEnv(state)
        search=self.planner(roots=1,beam_width=1,children=1,refinements=0,max_evaluations=8,horizon=8)
        live.step((220.,256.));search.record_action((220.,256.))
        search._proposals=lambda state,n:[ContactStage(0)]*n
        search._paired_proposals=lambda state,first,n:[ContactStage(first.edge+1)]*n
        def three(shape,state,stage,budget,maximum):
            return ((235.+10*stage.edge,256.),),["push"],{"approach_controls":0,"push_controls":1}
        before=live.capture_state()
        with patch("tdyssey.simulator_sequence.stage_actions",side_effect=three):
            result=search.choose(before,20)
        self.assertEqual(state_error(before,live.capture_state()),0)
        self.assertEqual(result.report["selected"]["kind"],"third_contact")
        self.assertEqual(len(result.report["selected"]["stages"]),3)
        for i,action in enumerate(result.actions):
            live.step(action);search.record_action(action)
            expected=type(state).from_dict(result.report["selected_predictions"][i])
            self.assertEqual(state_error(live.capture_state(),expected),0)
        self.assertLessEqual(result.report["evaluations"],8)


if __name__ == "__main__":
    unittest.main()
