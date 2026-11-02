import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from scripts import native_rgb_stage2a as stage


class Stage2ATests(unittest.TestCase):
    def test_ledger_allows_released_slot55_but_rejects163_and_unfinished_retry(self):
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as d:
            ledger=stage.TrainingLedger(d,cap=10368)
            ledger.value['trajectories']=[dict(status='completed') for _ in range(54)]
            index=ledger.begin_trajectory('slot55',64);self.assertEqual(index,54)
            with self.assertRaises(RuntimeError):ledger.begin_trajectory('unfinished',64)
            ledger.finish(index)
            ledger.value['trajectories']=[dict(status='completed') for _ in range(162)]
            with self.assertRaises(RuntimeError):ledger.begin_trajectory('slot163',64)

    def test_frozen162_are_train_only_unique_seeds_and_six_of_each_gap_per_boundary(self):
        parent=stage.ROOT/'artifacts/native-rgb-supervised-20261029'
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as d:
            output=Path(d)
            with patch.object(stage,'authorize',return_value=(parent,{})):
                result=stage.prepare(output,stage.ROOT/'configs/native_rgb_supervised_stage2a.json')
                q=result['requests'];self.assertEqual(len(q),162)
                self.assertEqual(len({x['seed'] for x in q}),162)
                self.assertEqual({x['letter'] for x in q},set(stage.LETTERS))
                self.assertEqual({x['split'] for x in q},{'train'})
                for location in stage.LOCATIONS[1:]:
                    selected=[x for x in q if x['location']==location]
                    self.assertEqual([sum(x['gap_world_px']==g for x in selected) for g in (4,18,48)],[6,6,6])
                old=json.loads((output/'dedup-reference.json').read_text())
                self.assertFalse(set(x['seed'] for x in q)&set(old['historical_seeds']))
                self.assertTrue(any('physics-probe' in x['path'] for x in old['sources']))
                with self.assertRaises(FileExistsError):stage.prepare(output,stage.ROOT/'configs/native_rgb_supervised_stage2a.json')

    def test_nonfinite_failure_is_saved_without_repair_or_hidden_step(self):
        import numpy as np
        class FailedInitialEnv:
            control_step=0
            substep_penetrations=[]
            def state(self):return np.asarray([float('nan')]+[0.]*14)
            def penetrations(self):return np.zeros(3)
            def step(self,*args):raise AssertionError('No hidden initialization step')
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as d:
            output=Path(d);ledger=stage.TrainingLedger(output,cap=10368)
            request=dict(request_id='bad',letter='A',spatial_class='center',seed=1,gap_world_px=None,gap_band=None)
            with patch('tdyssey.native_rgb_env.make_env',return_value=FailedInitialEnv()),patch('tdyssey.native_rgb_supervised_sampling.initialize_scene',return_value=dict(legal=False,staging_segment=0,initial_state=[float('nan')])):
                entry,error=stage.run_slot(output,request,'unused',ledger,[],{}, {},set())
            self.assertIsNotNone(error);self.assertEqual(entry['controls'],0)
            self.assertEqual(ledger.value['charged_controls'],0)
            self.assertEqual(ledger.value['trajectories'][0]['status'],'failed')
            with np.load(output/'data'/entry['file'],allow_pickle=False) as arrays:self.assertTrue(np.isnan(arrays['states'][0,0]))
            meta=json.loads((output/'data'/entry['metadata_file']).read_text());self.assertIsNone(meta['exception_state']['state'][0])

    def test_cli_no_execute_and_audit_failure_stay_fail_closed(self):
        with patch.object(stage,'authorize'),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'--execute'):stage.main(['collect','--physics-root','unused'])
            with patch.object(stage,'audit',return_value={'passed':False,'issues':['mock']}):
                self.assertEqual(stage.main(['audit','--physics-root','unused']),1)


if __name__=='__main__':unittest.main()
