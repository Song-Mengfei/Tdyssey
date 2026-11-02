"""Pure budgets and saved-decision IO; never construct a physical environment."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import native_rgb_r2b_run as run
from tdyssey.native_rgb_r2b_sampling import FeedbackPolicy,RULES,canonical_sha


class StateView:
    def __init__(self,state):self.value=np.asarray(state,dtype=np.float64)
    def state(self):return self.value.copy()


class R2BContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=run.read(run.OUT/'proposed-requests.json')
        cls.request=cls.plan['requests'][0]
        cls.identity=dict(kind=RULES['kind'],rules_sha256=canonical_sha(RULES),
            rules_file='frozen-rules.json',rules_file_sha256='fixture',
            source_file='tdyssey/native_rgb_r2b_sampling.py',source_sha256='fixture',policy_sha256='synthetic-fixture')

    def test_budget_charge_is_final_and_cannot_repeat(self):
        ledger=dict(closed=False,pending_control=None,charged_controls=1295)
        row=dict(slot_id='synthetic-not-a-real-slot',charged_controls=23)
        run.guard_charge(ledger,row,23)
        self.assertEqual((ledger['charged_controls'],row['charged_controls']),(1296,24))
        with self.assertRaises(RuntimeError):run.guard_charge(ledger,row,24)
        ledger['pending_control']=None
        with self.assertRaises(RuntimeError):run.guard_charge(ledger,row,24)

    def decision(self):
        state=self.request['initialization']['expected_initial_state15']
        policy=FeedbackPolicy(StateView(state),self.request,RULES)
        action,decision=policy.next_action(0)
        return np.asarray(state,dtype=np.float64),action,decision

    def test_current_state_and_actual_dynamic_action_are_bound(self):
        state,action,decision=self.decision()
        normalized=run.normalize_decision(decision,action,state,0,self.identity)
        self.assertEqual(normalized['policy_sha256'],'synthetic-fixture')
        bad=copy.deepcopy(decision);bad['current_state'][0]+=.01
        with self.assertRaises(RuntimeError):run.normalize_decision(bad,action,state,0,self.identity)
        bad=copy.deepcopy(decision);bad['proposed_action'][0]+=.01
        with self.assertRaises(RuntimeError):run.normalize_decision(bad,action,state,0,self.identity)

    def test_no_future_or_phase_override(self):
        state,action,decision=self.decision()
        for key,value in [('used_future_state',True),('lookahead_physics_steps',1),('planned_phase','push')]:
            bad=copy.deepcopy(decision);bad[key]=value
            with self.assertRaises(RuntimeError):run.normalize_decision(bad,action,state,0,self.identity)

    def test_dynamic_decisionlogs_and_saved_rule_replay_roundtrip(self):
        state,action,decision=self.decision();decision=run.normalize_decision(decision,action,state,0,self.identity)
        row=dict(slot_id='SYNTHETIC-CPU-IO-NOT-PHYSICS',status='completed',charged_controls=1,completed_controls=1,actual_controls=1,partial_substeps=0,
            trajectory_file='trajectories/SYNTHETIC-CPU-IO-NOT-PHYSICS/trajectory.npz',metadata_file='trajectories/SYNTHETIC-CPU-IO-NOT-PHYSICS/metadata.json')
        info=dict(valid=True,contact=0,agent_object_contact_substeps=0,actual_contact_arc_ids=[])
        steps=[dict(index=i,dt=.01,executed=True,pre_state=state.tolist(),post_state=state.tolist(),penetrations=[0.,0.,0.],contacts=[]) for i in range(10)]
        record=run.control_record(info,steps,action,decision,0,True,[0.,45.]);record['feedback_input']=info
        meta=dict(request=self.request,policy_identity=self.identity,control_records=[record],synthetic_CPU_test_only=True)
        testparent=run.OUT/'cpu-io-checks';testparent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='verified-workspace-',dir=testparent) as path:
            out=Path(path).resolve();self.assertTrue(out.is_relative_to(testparent.resolve()))
            with patch.object(run,'OUT',out):
                run.atomic_json(out/'frozen-config.json',{'policy_identity':self.identity})
                run.atomic_json(out/'frozen-rules.json',RULES)
                image=np.full((3,96,96),255,dtype=np.uint8)
                run.save_dynamic_slot(row,meta,[state,state],[action],[image,image],[action],
                    [dict(control_index=0,action=action.tolist(),completed=True,substeps=steps)],[decision])
                self.assertTrue(run.check_saved(row)['passed'])
                with gzip.open(out/meta['decisionlogs_file'],'rt',encoding='utf-8') as stream:
                    self.assertEqual(json.load(stream),[decision])


if __name__=='__main__':unittest.main()
