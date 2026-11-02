"""Focused offline timing/noise contracts. No Torch/model/physical calls."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

STAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(STAGE))
import diagnostics as d

class DiagnosticTests(unittest.TestCase):
    def test_true_history_excludes_own_target_and_raw_recursion_is_not_repaired(self):
        frames=np.zeros((1,20,3,96,96),np.uint8)
        for k in range(20):frames[:,k]=k*10
        actions=np.arange(38,dtype=float).reshape(1,19,2)
        inputs={mode:[] for mode in ('real_history','autoregressive')};results={}
        for mode in inputs:
            def sample(history,normalized,*,seed):
                inputs[mode].append((history.copy(),normalized.copy(),seed))
                return (history[:,-1]+np.float32(.5)).astype(np.float32)
            results[mode]=d.predict_modes(sample,frames,actions,list(range(16)),mode=mode)
        self.assertTrue(np.array_equal(inputs['real_history'][0][0],frames[:,:4].astype(np.float32)/np.float32(255)))
        self.assertTrue(np.array_equal(inputs['real_history'][1][0],frames[:,1:5].astype(np.float32)/np.float32(255)))
        self.assertTrue(np.array_equal(inputs['autoregressive'][1][0][:,-1],results['autoregressive'][:,0]))
        self.assertGreater(float(inputs['autoregressive'][4][0].max()),1.)
        self.assertTrue(np.array_equal(inputs['real_history'][3][1],actions[:,3:7].astype(np.float32)/np.float32(255.5)-np.float32(1)))
        self.assertEqual([x[2] for x in inputs['real_history']],[x[2] for x in inputs['autoregressive']])

    def test_original_batch_noise_slot_and_scoped_restore(self):
        class Fake:
            def __init__(self):self.rng=np.random.default_rng(111)
            def randn(self,*shape,**kw):return self.rng.standard_normal(shape).astype(np.float32)
        expected=Fake().randn(4,3,96,96)[2:3]
        fake=Fake();original=fake.randn;seen=[]
        with d.inject_first_noise(fake,batch_size=4,slot=2,sink=lambda a:seen.append(a.copy())):
            value=fake.randn(1,3,96,96)
        self.assertTrue(np.array_equal(value,expected));self.assertTrue(np.array_equal(seen[0],expected))
        self.assertEqual(fake.randn,original)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with d.inject_first_noise(fake,batch_size=4,slot=1):raise RuntimeError('fixture')
        self.assertEqual(fake.randn,original)
        with self.assertRaisesRegex(RuntimeError,'additional'):
            with d.inject_first_noise(fake,batch_size=4,slot=1):
                fake.randn(1,3,96,96);fake.randn(1,3,96,96)
        self.assertEqual(fake.randn,original)

    def test_selection_is_motion_biased_family_isolated_and_reports_missing(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temporary:
            root=Path(temporary);records=[];train=[]
            for family,split in [('one','train'),('two','train'),('val','seen_val')]:
                raw=root/(family+'.npz');meta=root/(family+'.json')
                states=np.zeros((36,15),float);states[:,4]=np.arange(36)*.02
                actions=np.ones((35,2))*200;positive=np.ones(35,bool)
                np.savez(raw,states=states,passed_actions=actions,requested_actions=actions,control_valid=np.ones(35,bool),contact_positive=positive)
                value=dict(letter='O',split=split,clean_control_count=35,family_id=family,attempt_id=family,task='rotation_cw',segments=[dict(start_control=3,end_control=19),dict(start_control=19,end_control=35)])
                meta.write_text(json.dumps(value))
                records.append(dict(raw_path=str(raw),raw_sha256=d.sha(raw),metadata_path=str(meta),metadata_sha256=d.sha(meta)))
                if split=='train':train += [dict(family_id=family,anchor=a) for a in range(3,35)]
            rows,counts=d.select_moving_rows(records,train,per_group=8)
            self.assertEqual(len(rows),3);self.assertEqual(len({r['family_id'] for r in rows}),3)
            self.assertEqual(counts['train:O']['selected'],2);self.assertEqual(counts['seen_val:O']['missing_full_h16'],7)
            self.assertEqual(counts['train:C']['selected'],0)
            self.assertTrue(all(r['positive_contact'] and abs(r['truth_h1_signed_rotation_deg'])>=.5 for r in rows))

if __name__=='__main__':unittest.main()
