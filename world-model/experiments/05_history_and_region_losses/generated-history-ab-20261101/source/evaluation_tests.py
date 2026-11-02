"""Offline paired-input, raw-feedback and saved-baseline reuse checks."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

STAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(STAGE))
import evaluation as e

class EvaluationTests(unittest.TestCase):
    def test_true_history_excludes_own_target_and_ar_receives_no_future_table(self):
        rgb=np.empty((24,3,96,96),np.uint8)
        for i in range(24):rgb[i]=i*9
        actions=np.arange(46,dtype=np.float64).reshape(23,2)
        data=dict(rgb=rgb,passed_actions=actions,control_valid=np.ones(23,bool))
        store=SimpleNamespace(arrays=lambda row:data);row=dict(anchor=5)
        ar=e.prepare_inputs(store,[row],'autoregressive');real=e.prepare_inputs(store,[row],'real_history')
        self.assertNotIn('real_histories',ar)
        self.assertTrue(np.array_equal(real['real_histories'][0,0],rgb[2:6].astype(np.float32)/np.float32(255)))
        self.assertTrue(np.array_equal(real['real_histories'][0,15],rgb[17:21].astype(np.float32)/np.float32(255)))
        self.assertTrue(np.array_equal(ar['absolute_actions'][0],actions[2:21]))
        self.assertTrue(np.array_equal(ar['initial_histories'][0],real['real_histories'][0,0]))

    def test_paired_seeds_actions_and_raw_feedback_are_unchanged(self):
        initial=np.zeros((2,4,3,96,96),np.float32);actions=np.arange(76,dtype=float).reshape(2,19,2)
        real=np.zeros((2,16,4,3,96,96),np.float32)
        for k in range(16):real[:,k]=k/100
        records={};outputs={}
        for label,histories in [('ar',None),('true',real)]:
            records[label]=[]
            def sample(history,act,*,seed):
                records[label].append((history.copy(),act.copy(),seed))
                return (history[:,-1]+np.float32(.5)).astype(np.float32)
            outputs[label]=e.callback_rollout(sample,initial,actions,list(range(16)),real_histories=histories)
        self.assertTrue(np.array_equal(outputs['ar'][:,0],outputs['true'][:,0]))
        self.assertTrue(np.array_equal(records['ar'][1][0][:,-1],outputs['ar'][:,0]))
        self.assertGreater(float(records['ar'][4][0].max()),1.)
        self.assertEqual([x[2] for x in records['ar']],[x[2] for x in records['true']])
        self.assertTrue(np.array_equal(records['ar'][7][1],actions[:,7:11].astype(np.float32)/np.float32(255.5)-np.float32(1)))
        bad=actions.copy();bad[0,0,0]=512
        with self.assertRaises(ValueError):e.callback_rollout(sample,initial,bad,list(range(16)))

    def test_baseline_reuses_original_four_scores_and_only_scores_missing_steps(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as temporary:
            root=Path(temporary);oc=root/'old';(oc/'evaluation/epoch2').mkdir(parents=True)
            rawpath=root/'oldraw.npz';np.savez_compressed(rawpath,raw=np.zeros((16,3,96,96),np.float32))
            row=dict(sample_id='fixed',family_id='family',split='seen_val',anchor=3,letter='O',task='translation',raw_path='truth.npz',raw_sha256='fixture')
            oldrows=[dict(sample_id='fixed',family_id='family',letter='O',task='translation',method='epoch2',horizon=h,metrics={'raw_mask_iou':h/100}) for h in (1,4,8,16)]
            oldrows += [dict(sample_id='fixed',family_id='family',letter='O',task='translation',method='copy_last',horizon=h,metrics={'raw_mask_iou':.9}) for h in (1,4,8,16)]
            old=dict(rows=oldrows,raw_files=[dict(sample_id='fixed',path=str(rawpath),sha256=e.sha(rawpath),seeds=list(range(16)))])
            (oc/'evaluation/epoch2/metrics.json').write_text(json.dumps(old))
            spec=dict(rows=[row],batches=[dict(rows=[row],seeds=list(range(16)))])
            calls=[]
            def scored(helper,row,store,pred,h,label,mode,calibration):
                calls.append(h);return dict(sample_id='fixed',horizon=h,method=label,metrics={'raw_mask_iou':.123})
            def write(path,value):
                p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            original_read=e.read
            def read(path):
                if str(path).endswith('frozen-renderer-calibration.json'):return {}
                return original_read(path)
            ctx=SimpleNamespace(checkpoint_identity={'sha256':e.START_SHA,'path':'checkpoint.pt'})
            with patch.object(e,'OC',oc),patch.object(e,'_ar1',return_value=SimpleNamespace()),patch.object(e,'_scored_row',side_effect=scored),patch.object(e,'write',side_effect=write),patch.object(e,'read',side_effect=read),patch.object(e,'native_rollout',side_effect=AssertionError('must not run model')):
                result=e._reuse_formal(root,{},ctx,spec,None)
            saved=original_read(result['metrics'])
            self.assertEqual(saved['reused_endpoint_scores'],4);self.assertEqual(saved['new_saved_only_endpoint_scores'],12)
            self.assertEqual(calls,[h for h in range(1,17) if h not in (1,4,8,16)])
            for h in (1,4,8,16):self.assertEqual(next(x for x in saved['rows'] if x['horizon']==h)['metrics']['raw_mask_iou'],h/100)
            self.assertEqual(saved['new_model_calls'],0)
            self.assertEqual(saved['reused_copy_last_endpoint_scores'],4)

    def test_uncertain_angle_is_not_promoted_to_known_rotation(self):
        value=e._annotations({'visible_angle_error_deg':2.,'legacy_shape_assessable':False})
        self.assertEqual(value['visible_angle_error_deg'],2.)
        self.assertFalse(value['visible_angle_assessable']);self.assertFalse(value['physical_rotation_assessable'])
        self.assertIsNone(value['underrotation_deg']);self.assertFalse(value['shape_assessable'])

    def test_frozen_spec_rejects_batch_reordering(self):
        rows=[dict(sample_id=f'{l}{i}',family_id=f'{l}{i}',letter=l,split='seen_val') for l in 'TLOCXB' for i in range(8)]
        spec=dict(rows=rows,batches=[dict(rows=rows[a:a+4],seeds=list(range(16))) for a in range(0,48,4)],sampler=e.SAMPLER)
        e._spec_check(spec,formal=True)
        changed=deepcopy(spec);changed['batches'][0]['rows'].reverse()
        with self.assertRaises(ValueError):e._spec_check(changed,formal=True)

if __name__=='__main__':unittest.main()
