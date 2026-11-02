"""Real saved-state/trace regressions through production CP2 methods; no physics."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import unittest
import numpy as np

from tdyssey.native_rgb_contact_pilot2_controller import ContactPilot2Policy,validate_target_pair,_raw_points,RULES
from tdyssey.native_rgb_r3a_collection import phase_for,PHASES
from tdyssey.native_rgb_r2b_sampling import geometry_at,_free,_line,_departure_route

ROOT=Path(__file__).resolve().parents[1]
CP1=ROOT/'artifacts/native-rgb-supervised-20261029/contact-pilot1-20261030'
CP2=CP1.parent/'contact-pilot2-20261030'
EVIDENCE={}


class StateOnly:
    def __init__(self,state):self.current=np.asarray(state,np.float64).copy()
    def state(self):return self.current.copy()


@lru_cache(maxsize=12)
def case_arrays(case_id):
    case=next(c for c in json.loads((CP1/'plan/cases.json').read_text())['cases']if c['case_id']==case_id)
    with np.load(CP1/f'trajectories/{case_id}.npz',allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
    return case,arrays


def contact_at(case,arrays,index):
    contact=deepcopy(case['candidate']['selected_contact'])
    contact.update(point_local=arrays['control_point_local'][index].astype(float).tolist(),
        outward_normal_local=arrays['control_normal_local'][index].astype(float).tolist(),arc_id=int(arrays['control_arc_id'][index]))
    return contact


def policy_at(case_id,index,*,previous_legal=True):
    case,a=case_arrays(case_id);state=a['states'][index];env=StateOnly(state)
    policy=ContactPilot2Policy(env,case['slot'],case['candidate'])
    chosen=None
    if previous_legal and index>0:
        for j in range(max(0,index-1),-1,-1):
            candidate=contact_at(case,a,j)
            if validate_target_pair(case['letter'],state,candidate)['valid']:chosen=candidate;break
    if chosen is not None:policy.contact=chosen;policy.departure_contact=deepcopy(chosen)
    policy.next_index=index;policy.task_phase=phase_for(index-1,case['scenario'])if index else None
    policy.last_action=None if index==0 else a['actions'][index-1].copy()
    policy.established=index>0;policy.awaiting_selected_contact=not policy.established
    policy.requested_initial_established=index>0;policy.target_dwell=8
    return policy,case,a


def actual_trace(arrays,control):
    events=[[]for _ in range(10)];tags=arrays['contact_tags'];values=arrays['contact_values']
    offsets=arrays['contact_arc_offsets'];arcs=arrays['contact_arc_ids']
    for row in np.flatnonzero((tags[:,0]>=control*10)&(tags[:,0]<(control+1)*10)):
        si,kind,phase,_,mask=map(int,tags[row]);v=values[row]
        normal=v[4:6].astype(float).tolist()
        event=dict(kind=('agent_object','object_wall','agent_wall')[kind],phase=('pre_solve','post_solve')[phase],
            normal=normal,shapes_order=['object','agent'],points=[])
        if mask&4:event['total_impulse']=v[7:9].astype(float).tolist()
        if mask&2:event['points']=[dict(object_local=v[2:4].astype(float).tolist(),arc_ids=arcs[offsets[row]:offsets[row+1]].astype(int).tolist())]
        events[si-control*10].append(event)
    return dict(substeps=[dict(executed=True,contacts=e)for e in events])


class ContactPilot2ControllerTests(unittest.TestCase):
    def test_T03_state10_target_and_departure_route_commit_together(self):
        policy,case,a=policy_at('CP1-A-T-03',10);old_target=deepcopy(policy.contact)
        policy.wall_stall=3;policy.target_dwell=10
        action,d=policy.next_action(10)
        self.assertEqual(d['mode'],1);self.assertEqual(d['controller_state'],'transactional_recovery_route')
        self.assertIn('committed_with_route',d['replan_reason']);tx=d['route_transaction']
        self.assertEqual(tx['kind'],'retained_departure_bridge')
        self.assertEqual(tx['departure_source']['point_local'],old_target['point_local'])
        self.assertNotEqual(tx['target_identity'],tx['route_identity'])
        self.assertGreater(np.linalg.norm(action-policy.env.state()[:2]),.1)
        geom=geometry_at('T',policy.env.state())
        self.assertTrue(_free(geom).covers(_line(policy.env.state()[:2],action))or _departure_route(geom,policy.env.state()[:2],action))
        committed=tx['committed_route_identity']
        # At the identical recorded state, a subsequent production replan still
        # has the old bridge after self.contact has changed. No state is stepped.
        policy.pending=None;policy.next_index=11
        _,again=policy.next_action(11)
        self.assertNotEqual(again['mode'],0);self.assertEqual(again['route_transaction']['committed_route_identity'],committed)
        EVIDENCE['T03_state10']=dict(action=action.tolist(),route_transaction=tx,after_target_commit_mode=again['mode'])

    def test_B04_recorded_invalid_pairs_do_not_destroy_a_valid_target(self):
        results=[]
        for index in(8,15,31,39):
            policy,case,a=policy_at('CP1-A-B-04',index);state=policy.env.state();before=deepcopy(policy.contact)
            self.assertTrue(validate_target_pair('B',state,before)['valid'])
            invalid_cp1_pair=contact_at(case,a,index)
            invalid=validate_target_pair('B',state,invalid_cp1_pair)
            self.assertFalse(invalid['valid'])
            points=_raw_points(actual_trace(a,index-1))
            changed=policy._track_actual_contact(state,dict(raw_points=points))
            current=validate_target_pair('B',state,policy.contact)
            self.assertTrue(current['valid'])
            if not changed:self.assertEqual(policy.contact,before)
            # No componentwise clipped point is ever paired to an unrelated
            # observed normal. Any committed pair is one unchanged raw point.
            if changed:
                self.assertTrue(any(np.allclose(policy.contact['point_local'],p['point_local'],rtol=0,atol=1e-9)for p in points))
            results.append(dict(saved_state=index,CP1_pair_rejection=invalid['reason'],production_track_committed=changed,
                retained_or_committed_target_valid=current['valid'],target=deepcopy(policy.contact),observation_count=len(points)))
        EVIDENCE['B04_real_states']=results

    def test_X03_wall_risk_gates_production_actions_before_control5(self):
        results=[]
        for index in range(6):
            policy,case,a=policy_at('CP1-A-X-03',index);policy.contact=contact_at(case,a,index)
            policy.departure_contact=deepcopy(policy.contact)
            action,d=policy.next_action(index);risk=d['wall_risk']
            results.append(dict(control=index,controller_state=d['controller_state'],risk=risk,action=action.tolist()))
            if risk and risk['risky']:
                self.assertNotEqual(d['mode'],2);self.assertTrue(risk['normal_advance_and_feedforward_cancelled'])
                self.assertIsNotNone(d['wall_risk_fallback'])
        flagged=[r['control']for r in results if r['risk']and r['risk']['risky']]
        self.assertTrue(flagged);self.assertLessEqual(min(flagged),4)
        EVIDENCE['X03_production_risk_timeline']=results

    def test_actual_inward_action_cannot_bypass_gate_by_free_route_label(self):
        policy,_,a=policy_at('CP1-A-X-03',3);state=policy.env.state()
        action,mode,risk,fallback=policy._safety_gate(state,geometry_at('X',state),a['actions'][3].copy(),'free_route')
        self.assertTrue(risk['risky']);self.assertTrue(risk['normal_advance_and_feedforward_cancelled'])
        self.assertIsNotNone(fallback);self.assertFalse(np.array_equal(action,a['actions'][3]))
        EVIDENCE['mode_label_does_not_bypass_actual_inward_action']=dict(original_action=a['actions'][3].tolist(),final_action=action.tolist(),risk=risk,mode=mode,fallback=fallback)

    def test_six_old_weak_arcs_keep_requested_identity_and_incidental_contact_distinct(self):
        results=[]
        for letter in'TLOCXB':
            policy,case,a=policy_at(f'CP1-B-{letter}-04',0);requested=policy.initial_requested_arc
            action,d=policy.next_action(0)
            self.assertEqual(d['initial_requested_arc'],case['candidate']['selected_contact']['arc_id'])
            self.assertEqual(d['beta'],0.)
            self.assertEqual(d['point_local'],case['candidate']['selected_contact']['point_local'])
            contact=np.count_nonzero(a['contact_flags'][:,0].reshape(48,10)&2,axis=1)
            actual_control=int(np.flatnonzero(contact)[0]);points=_raw_points(actual_trace(a,actual_control))
            hit,positive,stable=policy._requested_evidence(points)
            self.assertEqual(hit,set());self.assertEqual(positive,set());self.assertFalse(stable)
            # Processing raw physical evidence cannot mutate initial_requested_arc.
            policy._observed_pair(a['states'][actual_control+1],points)
            self.assertEqual(policy.initial_requested_arc,requested)
            results.append(dict(case_id=case['case_id'],requested_arc=requested,initial_beta=d['beta'],
                first_CP1_contact_control=actual_control,raw_contact_arcs=sorted({arc for p in points for arc in p['arc_ids']}),
                requested_positive_substeps=0,requested_stable=False))
        EVIDENCE['six_old_weak_arcs']=results

    def test_fixed_withdrawal_task_is_not_deleted(self):
        decisions=[]
        for index in range(18,26):
            policy,_,_=policy_at('CP1-A-T-01',index)
            _,d=policy.next_action(index)
            self.assertEqual(d['phase'],PHASES.index('withdraw'));self.assertEqual(d['planned_task_phase'],'withdraw')
            self.assertEqual(d['controller_state'],'withdraw');decisions.append(index)
        EVIDENCE['withdraw_controls_preserved']=decisions

    def test_all_prephysics_case_initial_actions_preserve_target_geometry_and_bounds(self):
        path=CP2/'plan/cases-draft.json'
        cases=json.loads(path.read_text())['cases']if path.exists()else json.loads((CP1/'plan/cases.json').read_text())['cases']
        records=[]
        for case in cases:
            policy=ContactPilot2Policy(StateOnly(case['candidate']['initial_state15']),case['slot'],case['candidate'])
            action,d=policy.next_action(0);start=policy.env.state()[:2]
            self.assertTrue(d['geometry_verified']);self.assertTrue(np.isfinite(action).all())
            self.assertTrue((action>=0).all()and(action<=511).all());self.assertLessEqual(np.linalg.norm(action-start),64+1e-8)
            self.assertEqual(d['initial_requested_arc'],case['candidate']['selected_contact']['arc_id'])
            self.assertEqual(d['point_local'],case['candidate']['selected_contact']['point_local'])
            json.dumps(d,allow_nan=False)
            records.append(dict(case_id=case['case_id'],mode=d['mode'],geometry_verified=True,requested_arc=d['initial_requested_arc']))
        EVIDENCE['initial_actions']=records


if __name__=='__main__':unittest.main()
