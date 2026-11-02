"""Focused v2 atomic-I/O and audited selection tests; no real physics."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
from scripts import native_rgb_r3a_collect_v2 as cli
from tests.test_native_rgb_r3a_collection import decision,substeps


class IORepairContracts(unittest.TestCase):
    def test_same_serialized_npz_replaces_after_transient_sharing_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            target=Path(tmp)/'trajectory.npz';clock=[0.];calls=[];real_replace=os.replace
            def replace(source,destination):
                calls.append((Path(source),Path(destination)))
                if len(calls)<3:raise PermissionError('fake Windows sharing lock')
                return real_replace(source,destination)
            real_retry=cli.replace_with_retry
            def retry(source,destination):return real_retry(source,destination,clock=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay),replace=replace)
            values=np.arange(30,dtype=np.float64).reshape(2,15)
            with patch.object(cli,'replace_with_retry',side_effect=retry):cli.save_npz(target,{'states':values})
            self.assertEqual(len(calls),3);self.assertEqual(len({source for source,_ in calls}),1)
            self.assertNotEqual(calls[0][0].name,'trajectory.npz.tmp');self.assertFalse(calls[0][0].exists())
            with np.load(target,allow_pickle=False) as z:np.testing.assert_array_equal(z['states'],values)
            self.assertLess(clock[0],2.)
            self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)

    def test_existing_public_json_writer_retries_same_write_without_model_import(self):
        from tdyssey import native_rgb_training as shared
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            target=Path(tmp)/'ledger.json';calls=[];real_replace=os.replace
            def replace(source,destination):
                calls.append(Path(source))
                if len(calls)<3:raise PermissionError('fake sharing lock')
                return real_replace(source,destination)
            with patch.object(shared.os,'replace',side_effect=replace):cli.atomic_json(target,{'charged':67,'executed':660})
            self.assertEqual(cli.read(target),{'charged':67,'executed':660});self.assertEqual(len(calls),3)
            self.assertEqual(len(set(calls)),1);self.assertFalse(calls[0].exists())
            self.assertNotIn('torch',sys.modules)

    def test_repair_preserves_completed_success_and_uses_only_next_frozen_candidate(self):
        slot=dict(slot_id='L-slot',attempts=[{'attempt_index':i,'seed':100+i} for i in range(3)])
        failed=dict(status='audited_repair_pending',attempts=[{'status':'failed_system','trajectory_id':'L-a0'}])
        receipt=dict(failed_slot_next_attempts={'L-slot':1})
        self.assertEqual(cli.selected_candidates(slot,failed,'repair-first-six',receipt),slot['attempts'][1:])
        with self.assertRaises(PermissionError):cli.selected_candidates(slot,failed,'repair-first-six',{})
        complete=dict(status='completed',successful_trajectory_id='L-a0',attempts=[{'status':'completed','trajectory_id':'L-a0'}])
        self.assertEqual(cli.selected_candidates(slot,complete,'repair-first-six',receipt),[])
        with self.assertRaises(RuntimeError):cli.selected_candidates(slot,complete,'first-six',{})
        fresh=dict(status='not_executed',attempts=[])
        self.assertEqual(cli.selected_candidates(slot,fresh,'repair-first-six',receipt),slot['attempts'])

    def test_first_close_failure_is_consistently_counted_and_never_replaced(self):
        class Env:
            def __init__(self):self.value=np.zeros(15,np.float64);self.control_step=0;self.last_substeps=[]
            def seed(self,value):pass
            def reset_scene(self,**kwargs):return np.full((3,96,96),255,np.uint8)
            def state(self):return self.value.copy()
            def initial_identity(self):return {'physics':{'space':{'damping':.0009765625}},'physics_sha256':'fixture'}
            def penetrations(self):return np.zeros(3)
            def step(self,action):
                self.last_substeps,self.value=substeps(self.value);self.control_step+=1
                return np.full((3,96,96),255,np.uint8),{'valid':True,'physics_substeps':10,'agent_object_contact_substeps':10}
            def close(self):raise OSError('fixture close failure')
        class Policy:
            def __init__(self,*args):pass
            def next_action(self,index):return np.array([250.,250.]),decision()
            def observe(self,*args):pass
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base=Path(tmp);config={'output':str(base),'physics_source':'fixture','storage_roots':[str(base)]}
            slot=dict(slot_id='fixture-L',family_id='fixture-F-L',letter='L',split='train',scenario='translation_nearCOM',attempts=[dict(attempt_index=i,seed=100+i,initial_yaw_bin=0,initial_space='center',initial_state15=np.zeros(15).tolist(),initialization={}) for i in range(3)])
            ledger=dict(closed=False,pending_control=None,charged_controls=0,completed_controls=0,valid_controls=0,attempted_scenes=0,successful_scenes=0,failed_scenes=0,
                engine_step_calls=0,executed_substeps=0,engine_unknown_advance_substeps=0,prior_supervised_controls=41976,prior_all_native_controls=76536,failures=[])
            index=dict(status='prepared',collection_closed=False,gate_passed=False,slots=[dict(slot_id=slot['slot_id'],status='not_executed',attempts=[])])
            for name,value in [('resource-ledger.json',ledger),('index.json',index),('manifest.json',{'entries':[],'collection_closed':False})]:cli.atomic_json(base/name,value)
            with patch.object(cli,'preflight',return_value=(config,{'slots':[slot]},{})),patch.object(cli,'ContactPolicy',Policy),patch('tdyssey.native_rgb_selected_physics.make_selected_env',side_effect=lambda *args:Env()) as factory,patch('tdyssey.native_rgb_selected_physics.verify_selected_identity',return_value=True):
                self.assertEqual(cli.run('fixture','first-six',execute=True),2);self.assertEqual(factory.call_count,1)
            observed=cli.read(base/'resource-ledger.json');self.assertEqual(observed['failed_scenes'],1);self.assertEqual(len(observed['failures']),1)
            self.assertEqual(observed['charged_controls'],48);self.assertEqual(observed['executed_substeps'],480);self.assertEqual(observed['successful_scenes'],0)
            self.assertEqual(cli.read(base/'trajectories/fixture-L-a0.json')['status'],'failed_system')
            self.assertEqual(cli.read(base/'manifest.json')['entries'],[]);self.assertTrue(observed['closed'])
            self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
