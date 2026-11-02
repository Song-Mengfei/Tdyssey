"""CPU geometry and fake-engine collection contracts. No physics is stepped."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
from tdyssey.native_rgb_r3a_collection import (LETTERS,SCENARIOS,SCENARIO_COUNTS,RULES,
    build_plan,phase_for,scene_candidate,ContactPolicy,CompactAttempt,save_npz,verify_saved_attempt)


def decision():return dict(phase=1,mode=2,arc_id=2,point_local=[0.,5.],normal_local=[1.,0.],normal_world=[1.,0.],beta=.0,signed_lever=5.,raw_target=[250.,250.],geometry_verified=True)


def substeps(before,control=0):
    result=[];state=before.copy()
    for j in range(10):
        pre=state.copy();state[2]+=.125
        point=dict(object_world=[float(state[2]),float(state[3]+5)],object_local=[0.,5.],distance=-.1,arc_ids=[2])
        precontact=dict(kind='agent_object',phase='pre_solve',normal=[1.,0.],shapes_order=['object','agent'],points=[point])
        postcontact=dict(precontact,phase='post_solve',total_impulse=[1.,2.],total_ke=3.)
        result.append(dict(index=j,executed=True,pre_state=pre.tolist(),post_state=state.copy().tolist(),penetrations=[.1,.2,0.],
            exact_formula_reconstruction=dict(unclipped_force=[20000.,0.],clipped_force=[10000.,0.]),clipped_body_force=[10000.,0.],velocity_after_cap=[200.,0.],contacts=[precontact,postcontact]))
    return result,state


class CollectionContracts(unittest.TestCase):
    def test_full_independent_slot_seed_split_and_scenario_counts_before_physics(self):
        def cheap_contact(letter,state,scenario,arc_start,**kwargs):
            return dict(arc_id=0,point_local=[0.,0.],outward_normal_local=[1.,0.],staging_world=(state[2:4]+np.array((100.,0.))).tolist(),staging_distance_world_px=33.,signed_lever_world_px=0.,resultant_signed_lever_world_px=0.,physical_local_CoG=[0.,0.],drive_tangent_ratio=0.)
        with patch('tdyssey.native_rgb_r3a_collection.choose_contact',side_effect=cheap_contact):plan=build_plan({'seed':20261030,'train_per_letter':128,'val_per_letter':24})
        self.assertEqual(len(plan['slots']),912);self.assertEqual([s['letter'] for s in plan['slots'][:6]],list(LETTERS))
        self.assertEqual(len({s['family_id'] for s in plan['slots']}),912)
        seeds=[a['seed'] for s in plan['slots'] for a in s['attempts']];self.assertEqual(len(seeds),2736);self.assertEqual(len(set(seeds)),2736)
        for split in ('train','seen_val'):
            for letter in LETTERS:
                rows=[s for s in plan['slots'] if s['letter']==letter and s['split']==split]
                self.assertEqual(Counter(s['scenario'] for s in rows),dict(zip(SCENARIOS,SCENARIO_COUNTS[split])))
                self.assertEqual({r['attempts'][0]['initial_yaw_bin'] for r in rows},set(range(12)))
                self.assertEqual(len({r['attempts'][0]['initial_space'] for r in rows}),9)
        self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)

    def test_current_state_policy_has_real_second_half_push_not_old24_limit(self):
        candidate,_=scene_candidate('T','train',0,0,0,20261030,'translation_nearCOM')
        self.assertIsNotNone(candidate)
        class View:
            def state(self):return np.array(candidate['initial_state15'],np.float64)
        policy=ContactPolicy(View(),{'letter':'T','scenario':'translation_nearCOM'},candidate);last=None;phases=[]
        for i in range(48):
            before=View().state();action,d=policy.next_action(i);phases.append(d['phase'])
            self.assertLessEqual(float(np.linalg.norm(action-before[:2])),64.+1e-8)
            if last is not None:self.assertLessEqual(float(np.linalg.norm(action-last)),96.+1e-8)
            policy.observe(before,action,before,{'agent_object_contact_substeps':10});last=action
        self.assertEqual(phases[47],1);self.assertEqual(sum(phase_for(i,'translation_nearCOM')=='push' for i in range(24,48)),18)
        with self.assertRaises(ValueError):policy.next_action(48)
        self.assertEqual({phase_for(i,'static') for i in range(48)},{'static'})

    def test_compact_actual_telemetry_schema_and_hashes_without_RGB_payload(self):
        before=np.zeros(15,np.float64);before[2:4]=200.;rgb=np.full((3,96,96),255,np.uint8)
        evidence=CompactAttempt('L',before,rgb);steps,after=substeps(before)
        evidence.append(np.array([250.,250.]),decision(),steps,completed=True,after=after,rgb=rgb,wall_seconds=.1)
        arrays=evidence.arrays();self.assertEqual(arrays['states'].dtype,np.float64)
        np.testing.assert_array_equal(arrays['states'],arrays['substep_states'][::10])
        np.testing.assert_array_equal(arrays['substep_forces'][0],np.array([20000.,0.,10000.,0.,10000.,0.],np.float32))
        self.assertTrue((arrays['contact_flags'][:,0]==3).all());self.assertEqual(arrays['contact_values'].shape,(20,11))
        self.assertEqual(arrays['contact_arc_offsets'][-1],20);self.assertTrue((arrays['contact_arc_ids']==2).all())
        self.assertTrue((arrays['contact_tags'][::2,4]==3).all());self.assertTrue((arrays['contact_tags'][1::2,4]==31).all())
        self.assertNotIn('rgb',arrays);self.assertEqual(arrays['rgb_frame_sha256'].shape,(2,32))
        self.assertEqual(arrays['rgb_frame_sha256'][0].tobytes(),hashlib.sha256(rgb.tobytes()).digest())
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path=Path(tmp)/'trajectory.npz';save_npz(path,arrays);check=verify_saved_attempt(path)
            self.assertEqual(check['completed_controls'],1)

    def test_fake_engine_firstsix_actual_work_and_no_recollection(self):
        from scripts import native_rgb_r3a_collect as cli
        from tdyssey.native_rgb_r3a_collection import identity
        class FakeEnv:
            def __init__(self,letter):self.letter=letter;self.control_step=0;self.last_substeps=[];self.value=np.zeros(15,np.float64)
            def seed(self,value):pass
            def reset_scene(self,**kwargs):return np.full((3,96,96),255,np.uint8)
            def state(self):return self.value.copy()
            def initial_identity(self):return dict(physics={'space':{'damping':.0009765625}},physics_sha256='fixture')
            def penetrations(self):return np.zeros(3)
            def step(self,action):
                self.last_substeps,self.value=substeps(self.value);self.control_step+=1
                return np.full((3,96,96),255,np.uint8),dict(valid=True,physics_substeps=10,agent_object_contact_substeps=10)
            def close(self):pass
        class FakePolicy:
            def __init__(self,*args):pass
            def next_action(self,index):return np.array([250.,250.]),decision()
            def observe(self,*args):pass
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base=Path(tmp);config=dict(output=str(base),physics_source='fixture',storage_roots=[str(base)])
            slots=[]
            for letter in LETTERS:
                slots.append(dict(slot_id='fixture-'+letter,family_id='family-'+letter,letter=letter,split='train',scenario='translation_nearCOM',
                    attempts=[dict(attempt_index=0,seed=100+len(slots),initial_yaw_bin=0,initial_space='center',initial_state15=np.zeros(15).tolist(),initialization={})]))
            plan=dict(slots=slots)
            ledger=dict(closed=False,pending_control=None,charged_controls=0,completed_controls=0,valid_controls=0,attempted_scenes=0,successful_scenes=0,failed_scenes=0,
                engine_step_calls=0,executed_substeps=0,engine_unknown_advance_substeps=0,prior_supervised_controls=41976,prior_all_native_controls=76536,failures=[])
            index=dict(status='prepared',collection_closed=False,gate_passed=False,slots=[dict(slot_id=s['slot_id'],status='not_executed',attempts=[]) for s in slots])
            for name,value in [('resource-ledger.json',ledger),('index.json',index),('manifest.json',{'entries':[],'collection_closed':False})]:cli.atomic_json(base/name,value)
            with patch.object(cli,'preflight',return_value=(config,plan,{})),patch.object(cli,'ContactPolicy',FakePolicy),patch('tdyssey.native_rgb_selected_physics.make_selected_env',side_effect=lambda source,letter:FakeEnv(letter)),patch('tdyssey.native_rgb_selected_physics.verify_selected_identity',return_value=True):
                self.assertEqual(cli.run('fixture','first-six',execute=True),0)
                with self.assertRaisesRegex(RuntimeError,'already attempted'):cli.run('fixture','first-six',execute=True)
            observed=cli.read(base/'resource-ledger.json');self.assertEqual(observed['charged_controls'],288);self.assertEqual(observed['executed_substeps'],2880)
            self.assertEqual(observed['successful_scenes'],6);self.assertEqual(len(cli.read(base/'manifest.json')['entries']),6)
            for entry in cli.read(base/'manifest.json')['entries']:verify_saved_attempt(base/entry['file'],complete=True)
        self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
