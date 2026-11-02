"""Targeted CPU integration checks; no Torch/model, scoring or live ledger edits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
from tdyssey.native_rgb_r2c2_inference import ExistingInferenceLedger,evaluate_setting,sha


class InferenceContracts(unittest.TestCase):
    def config(self):return json.loads((ROOT/'configs/native_rgb_r2c2.json').read_text(encoding='utf-8'))

    def test_existing_v2_history_never_migrates_or_stops_on_elapsed_time(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp);config=self.config();original=Path(config['ledger_path']).read_bytes()
            config.update(ledger_path=str(base/'ledger.json'),prior_ledger_copy=str(base/'prior.json'),
                prior_ledger_sha256=hashlib.sha256(original).hexdigest())
            for name in ('ledger_path','prior_ledger_copy'):Path(config[name]).write_bytes(original)
            clock=[0.];ledger=ExistingInferenceLedger(config['ledger_path'],config,clock=lambda:clock[0])
            self.assertEqual(Path(config['ledger_path']).read_bytes(),original)
            prior=json.loads(original);stage=ledger.start('CPU-test-inference')
            for elapsed in (900.,1200.,5400.,6001.):
                clock[0]=elapsed;observed=stage.heartbeat()
                self.assertEqual(observed['charged_updates'],550)
                self.assertEqual(observed['max_seconds'],None)
                self.assertEqual(observed['active_stage']['observed_seconds'],elapsed)
            for operation in (stage.reserve_update,stage.complete_update):
                with self.assertRaisesRegex(PermissionError,'inference only'):operation()
            done=stage.finish('complete')
            self.assertIsNone(done['active_stage']);self.assertEqual(done['stages'][:-1],prior['stages'])
            self.assertEqual(done['completed_updates'],550);self.assertEqual(done['charged_updates'],550)
            self.assertEqual(done['policy_migrations'],prior['policy_migrations'])
            self.assertAlmostEqual(done['charged_seconds'],prior['charged_seconds']+6001.)
            self.assertEqual(Path(config['prior_ledger_copy']).read_bytes(),original)
            self.assertNotIn('torch',sys.modules)

    def prepared(self,n=4):
        import numpy as np
        return dict(window_ids=[f'fixture:{i}' for i in range(n)],identity={'CPU_fixture':True},
            histories=np.ones((n,4,3,96,96),dtype=np.float32),
            absolute_actions=np.full((n,7,2),255.5,dtype=np.float32))

    def result(self):
        return dict(settings={'8':dict(sampler_batch_attempts=0,sampler_batch_completed=0,raw_frames_saved=0,
            endpoint_rows_scored=0,CPU_scoring_seconds=0.)})

    def test_real_raw_sink_keeps_finite_OOR_and_unchanged_feedback(self):
        import numpy as np
        prepared=self.prepared();received=[];returned=[];scored=[]
        def sample(history,actions,*,seed):
            received.append((history.copy(),actions.copy(),seed))
            raw=history[:,-1]+np.float32(.1);returned.append(raw.copy());return raw
        def score(a,result,raw_steps,calibration,**kwargs):
            scored.append((result,raw_steps))
            return dict(rows=[{}]*8,raw_step_rows=[{}]*16)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp);folder=base/'run/evaluation/steps8';folder.mkdir(parents=True)
            config=self.config();config['output']=str(base);result=self.result()
            with patch('scripts.native_rgb_r2c1_report.evaluate_saved_predictions',side_effect=score):
                evaluate_setting(config,setting=8,sample=sample,cohorts={'train_visual':prepared},calibration={},
                    check=lambda:None,result=result,events=base/'events.jsonl',call_stats={'elapsed':lambda:0.})
            self.assertEqual(len(received),4);self.assertEqual(len(scored),1)
            for i,(history,actions,seed) in enumerate(received):
                np.testing.assert_array_equal(actions,np.zeros((4,4,2),np.float32))
                np.testing.assert_array_equal(history[:,-1],prepared['histories'][:,-1] if i==0 else returned[i-1])
            self.assertFalse(scored[0][0]['hard_stop'])
            self.assertEqual(scored[0][0]['status_h4'],['finite_out_of_range']*4)
            sidecar=json.loads((folder/'cohort-train_visual.json').read_text())
            self.assertEqual(len(sidecar['raw_step_files']),4)
            for row in sidecar['raw_step_files']:
                path=folder/row['file'];self.assertEqual(sha(path),row['sha256'])
                with np.load(path,allow_pickle=False) as arrays:
                    self.assertEqual(arrays['raw'].dtype,np.float32);self.assertGreater(float(arrays['raw'].min()),1.)
            self.assertEqual(result['settings']['8']['sampler_batch_completed'],4)
            self.assertEqual(result['settings']['8']['raw_frames_saved'],16)
            self.assertNotIn('torch',sys.modules)

    def test_callback_exception_keeps_full_trace_and_missing_denominators(self):
        import numpy as np
        calls=[]
        def broken(history,actions,*,seed):
            calls.append(seed);raise RuntimeError('CPU fixture sampler failure')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base=Path(temp);folder=base/'run/evaluation/steps8';folder.mkdir(parents=True)
            config=self.config();config['output']=str(base)
            with patch('scripts.native_rgb_r2c1_report.evaluate_saved_predictions') as score:
                with self.assertRaisesRegex(RuntimeError,'hard stop'):
                    evaluate_setting(config,setting=8,sample=broken,cohorts={'train_visual':self.prepared()},calibration={},
                        check=lambda:None,result=self.result(),events=base/'events.jsonl',call_stats={'elapsed':lambda:0.})
                score.assert_not_called()
            self.assertEqual(len(calls),1)
            events=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
            error=next(row for row in events if row['kind']=='predictor_exception')
            self.assertIn('Traceback (most recent call last)',error['traceback']);self.assertIn('CPU fixture sampler failure',error['traceback'])
            sidecar=json.loads((folder/'cohort-train_visual.json').read_text())
            self.assertTrue(sidecar['hard_stop']);self.assertEqual(len(sidecar['window_ids']),4)
            with np.load(folder/'cohort-train_visual.npz',allow_pickle=False) as arrays:
                self.assertTrue(np.isnan(arrays['h1']).all());self.assertTrue(np.isnan(arrays['h4']).all())
            self.assertNotIn('torch',sys.modules)

    def test_callback_boundary_keeps_storage_trace_and_wrong_dtype_raw(self):
        import numpy as np
        for mode in ('before_call_storage','wrong_dtype'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(dir=ROOT) as temp:
                base=Path(temp);folder=base/'run/evaluation/steps8';folder.mkdir(parents=True)
                config=self.config();config['output']=str(base);calls=[];checks=[]
                def check():
                    checks.append(True)
                    if mode=='before_call_storage' and len(checks)==2:
                        raise OSError('CPU fixture storage refusal')
                def sample(history,actions,*,seed):
                    calls.append(seed);return history[:,-1].astype(np.float64)
                with patch('scripts.native_rgb_r2c1_report.evaluate_saved_predictions') as score:
                    with self.assertRaisesRegex(RuntimeError,'hard stop'):
                        evaluate_setting(config,setting=8,sample=sample,cohorts={'train_visual':self.prepared()},calibration={},
                            check=check,result=self.result(),events=base/'events.jsonl',call_stats={'elapsed':lambda:0.})
                    score.assert_not_called()
                events=[json.loads(line) for line in (base/'events.jsonl').read_text().splitlines()]
                error=next(row for row in events if row['kind']=='predictor_exception')
                self.assertIn('Traceback (most recent call last)',error['traceback'])
                if mode=='before_call_storage':
                    self.assertEqual(calls,[]);self.assertIn('CPU fixture storage refusal',error['traceback'])
                else:
                    self.assertEqual(len(calls),1);self.assertIn('wrong-dtype raw preserved',error['traceback'])
                    sidecar=json.loads((folder/'cohort-train_visual.json').read_text())
                    self.assertEqual(len(sidecar['raw_step_files']),1)
                    rawfile=folder/sidecar['raw_step_files'][0]['file']
                    with np.load(rawfile,allow_pickle=False) as arrays:self.assertEqual(arrays['raw'].dtype,np.float64)


if __name__=='__main__':unittest.main()
