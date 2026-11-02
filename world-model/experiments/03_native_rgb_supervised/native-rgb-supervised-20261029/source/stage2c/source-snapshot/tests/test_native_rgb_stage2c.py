"""Seen-validation split/cost separation without real physics or model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter
import numpy as np
from scripts import native_rgb_stage2c as stage


class Stage2CTests(unittest.TestCase):
    def test_frozen_plan_has_nine_independent_spaces_and_balanced_boundary_gaps(self):
        parent=stage.ROOT/'artifacts/native-rgb-supervised-20261029'
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as directory:
            with patch.object(stage,'authorize',return_value=(parent,{})):
                stage.prepare(directory,stage.ROOT/'configs/native_rgb_supervised_stage2c.json')
                plan=json.loads((Path(directory)/'frozen-requests.json').read_text());q=plan['requests']
                self.assertEqual(len(q),162);self.assertEqual({x['split'] for x in q},{'seen_val'})
                self.assertEqual(Counter(x['letter'] for x in q),Counter({l:9 for l in stage.LETTERS}))
                for location in stage.LOCATIONS[1:]:
                    self.assertEqual(Counter(x['gap_world_px'] for x in q if x['location']==location),Counter({4:6,18:6,48:6}))
                old=json.loads((Path(directory)/'dedup-reference.json').read_text())
                self.assertFalse(set(x['seed'] for x in q)&set(old['historical_seeds']))
                self.assertEqual(plan['index_spec_sha256'],stage.load_config(stage.ROOT/'configs/native_rgb_supervised_stage2c.json')['index_spec_sha256'])
                with self.assertRaises(FileExistsError):stage.prepare(directory,stage.ROOT/'configs/native_rgb_supervised_stage2c.json')

    def test_mocked_actual_start_keeps_seen_val_and_zero_counter_scope(self):
        class FakeEnv:
            control_step=0
            substep_penetrations=[]
            def state(self):return np.zeros(15,dtype=np.float64)
            def penetrations(self):return np.zeros(3)
            def step(self,action):raise RuntimeError('mock actual method entry; no real physics')
        class FakePolicy:
            def __init__(self,*args):pass
            def next_action(self,t):return [1.,1.],{}
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as directory:
            out=Path(directory);ledger=stage.TrainingLedger(out,cap=10368);attempts={};counts={0:0,1:0}
            request=dict(request_id='mock-val',stage='2C',split='seen_val',letter='M',spatial_class='center',seed=1,gap_world_px=None,gap_band=None)
            with patch('tdyssey.native_rgb_env.make_env',return_value=FakeEnv()),\
                patch('tdyssey.native_rgb_supervised_sampling.initialize_scene',return_value=dict(legal=True,staging_segment=1,selection_version='motion-room-planned-attempts-v2')),\
                patch('tdyssey.native_rgb_supervised_sampling.ContinuousContactSampler',FakePolicy):
                entry,error=stage.run_slot(out,request,'unused',ledger,[],counts,{0:7,1:8},set(),planned_attempts=attempts)
            self.assertIsNotNone(error);self.assertEqual(entry['split'],'seen_val');self.assertEqual(entry['stage'],'2C')
            metadata=json.loads((out/'data'/entry['metadata_file']).read_text());inputs=metadata['selection_inputs']
            self.assertEqual(inputs['seen_val_contact_controls_before'],{'0':0,'1':0})
            self.assertNotIn('formal_train_contact_controls_before',inputs)
            self.assertEqual(inputs['planned_initial_attempts_before'],{});self.assertEqual(inputs['counter_split'],'seen_val')
            self.assertEqual(attempts,{1:1});self.assertEqual(counts,{0:0,1:0})
            self.assertEqual(ledger.value['charged_controls'],1);self.assertEqual(ledger.value['completed_controls'],0)

    def test_unreleased_execution_or_other_split_is_rejected(self):
        import io
        from contextlib import redirect_stdout
        with patch.object(stage,'authorize'),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'--execute'):stage.main(['collect','--physics-root','unused'])
        cfg=json.loads((stage.ROOT/'configs/native_rgb_supervised_stage2c.json').read_text())
        cfg['split']='train'
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as directory:
            path=Path(directory)/'bad.json';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):stage.load_config(path)


if __name__=='__main__':unittest.main()
