"""Cold report contracts with synthetic arrays/files only; zero model work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import native_rgb_r2c1_report as report


def image():
    rgb=np.ones((3,96,96),np.float32)
    rgb[:,20:45,20:45]=np.asarray((119,136,153),np.float32)[:,None,None]/255
    rgb[:,65:75,65:75]=np.asarray((65,105,225),np.float32)[:,None,None]/255
    return rgb


class R2C1ReportTests(unittest.TestCase):
    def test_display_encoding_keeps_raw_and_marks_invalid(self):
        raw=image();raw[0,0,0]=-.125;before=raw.copy()
        encoded,note=report.display_encoding(raw)
        np.testing.assert_array_equal(raw,before)
        self.assertEqual(encoded.dtype,np.uint8)
        self.assertIn('仅显示编码',note)
        raw[0,0,0]=np.nan
        self.assertIsNone(report.display_encoding(raw)[0])

    def test_midstep_failure_not_hidden_by_clean_endpoint(self):
        calibration=report.read(report.ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
        rgb=image();bad=rgb.copy();bad[0,0,0]=-.2
        metadata=dict(window_id='mock',cohort='train_visual',letter='T',family_id='mock-family',anchor=3,
            translation_image96_px=2.2,signed_rotation_deg=0.,abs_rotation_deg=0.,moving=True,
            h4_translation_image96_px=8.,h4_signed_rotation_deg=0.)
        prepared=dict(window_ids=['mock'],metadata=[metadata],truth_h1=rgb[None],truth_h4=rgb[None],histories=np.stack([[rgb]*4]))
        result=dict(window_ids=['mock'],planned_windows=1,h1=rgb[None],h4=rgb[None],status_h1=['ok'],status_h4=['ok'])
        rows,steps=report.score_method(prepared,result,[[rgb,bad,rgb,rgb]],calibration,'trained')
        h4=next(r for r in rows if r['horizon']==4)
        self.assertEqual(h4['raw_object_error'],0.)
        self.assertFalse(h4['endpoint_hard_generation_failure'])
        self.assertTrue(h4['any_path_step_hard_failure'])
        self.assertEqual(h4['path_failure_steps'],[2])
        self.assertTrue(steps[1]['hard_failure'])
        self.assertEqual(report.summary([h4])['path_hard_failures'],1)
        reference=dict(h4,method='copy_last',raw_object_error=.5)
        paired=report.paired([h4,reference],'trained','copy_last')
        self.assertEqual(paired['endpoint_outcomes'],{'win':1})
        self.assertEqual(paired['all_step_valid_wins'],0)

    def test_failed_penalties_and_shape_unknown_stay_in_mean(self):
        calibration=report.read(report.ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
        rgb=image();metadata=dict(window_id='mock',cohort='train_visual',letter='T',family_id='mock',anchor=3,
            translation_image96_px=.01,signed_rotation_deg=.01,abs_rotation_deg=.01,moving=False,h4_translation_image96_px=.02,h4_signed_rotation_deg=.02)
        prepared=dict(window_ids=['mock'],metadata=[metadata],truth_h1=rgb[None],truth_h4=rgb[None],histories=np.stack([[rgb]*4]))
        result=dict(window_ids=['mock'],planned_windows=1,h1=np.full((1,3,96,96),np.nan),h4=np.full((1,3,96,96),np.nan),status_h1=['not_attempted'],status_h4=['not_attempted'])
        rows,_=report.score_method(prepared,result,[[None]*4],calibration,'trained')
        summary=report.summary(rows)
        self.assertEqual(summary['denominator'],2)
        self.assertEqual(summary['gray_object']['raw_object_error']['mean'],1.)
        self.assertEqual(summary['shape_outcomes'],{'unknown':2})
        self.assertEqual(summary['missing_or_unattempted'],2)

    def test_closed_inventory_rejects_hot_and_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'model-work.json';data=root/'test.npz';data.write_bytes(b'fixture')
            report.write(work,dict(closed=False,evaluation_inventory=[]))
            with self.assertRaises(RuntimeError):report.closed_inventory(work,root)
            report.write(work,dict(closed=True,evaluation_inventory=[dict(file='test.npz',sha256=report.sha(data),bytes=data.stat().st_size)]))
            report.closed_inventory(work,root)
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):report.closed_inventory(work,root)

    def test_raw_step_order_hash_and_endpoint_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage=root/'baseline';(stage/'steps').mkdir(parents=True)
            ids=['a','b'];raw=np.stack([image(),image()]);files=[]
            for step in range(4):
                path=stage/'steps'/f'step{step}.npz';np.savez(path,raw=raw,window_ids=ids)
                files.append(dict(file=f'steps/step{step}.npz',sha256=report.sha(path),step=step,batch_index=0,window_ids=ids))
            npz=stage/'cohort-train_visual.npz';np.savez(npz,window_ids=ids,h1=raw,h4=raw,status_h1=['ok']*2,status_h4=['ok']*2)
            side=stage/'cohort-train_visual.json';report.write(side,dict(cohort_identity={'test':1},window_ids=ids,raw_step_files=files,evaluation_complete=True))
            inv={f'baseline/{p.name}':dict(sha256=report.sha(p)) for p in (npz,side)}
            result,steps,audit=report.load_predictions(root,'baseline','train_visual',dict(window_ids=ids,identity={'test':1}),inv,{})
            self.assertEqual(audit['raw_steps_present'],8)
            np.testing.assert_array_equal(steps[1][2],raw[1])
            np.savez(npz,window_ids=ids,h1=raw+.01,h4=raw,status_h1=['ok']*2,status_h4=['ok']*2)
            with self.assertRaises(ValueError):report.load_predictions(root,'baseline','train_visual',dict(window_ids=ids,identity={'test':1}),inv,{})

    def test_budgeted_metrics_envelope_roundtrip_and_no_cold_recompute(self):
        calibration=report.read(report.ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
        rgb=image();meta=dict(window_id='mock',cohort='train_visual',letter='T',family_id='mock',anchor=3,
            translation_image96_px=2.,signed_rotation_deg=0.,abs_rotation_deg=0.,moving=True,h4_translation_image96_px=8.,h4_signed_rotation_deg=0.)
        prepared=dict(window_ids=['mock'],identity={'frozen':'fixture'},metadata=[meta],truth_h1=rgb[None],truth_h4=rgb[None],histories=np.stack([[rgb]*4]))
        result=dict(window_ids=['mock'],planned_windows=1,h1=rgb[None],h4=rgb[None],status_h1=['ok'],status_h4=['ok'])
        raw=[[rgb]*4]
        value=report.evaluate_saved_predictions(prepared,result,raw,calibration,method='trained',cohort='train_visual')
        self.assertTrue(value['evaluation_complete'])
        self.assertEqual(len(value['rows']),2);self.assertEqual(len(value['raw_step_rows']),4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'final').mkdir();path=root/'final/cohort-train_visual.metrics.json'
            report.write(path,value)
            inventory={'final/cohort-train_visual.metrics.json':dict(sha256=report.sha(path))}
            with patch.object(report,'score_method',side_effect=AssertionError('coldscoringforbidden')):
                rows,steps,binding=report.load_metrics(root,'final','train_visual',prepared,result,raw,inventory,calibration,'trained')
            self.assertFalse(binding['missing']);self.assertEqual(rows[0]['raw_object_error'],0.)
            report.write(path,report.metrics_placeholder(prepared,method='trained',cohort='train_visual',reason='budgetstop'))
            rows,steps,binding=report.load_metrics(root,'final','train_visual',prepared,result,raw,inventory,calibration,'trained')
            self.assertTrue(binding['missing'])
            self.assertIsNone(report.summary(rows)['gray_object']['raw_object_error']['mean'])
            comparison=report.paired([r for r in rows if r['horizon']==1],'trained','trained')
            self.assertEqual(comparison['endpoint_outcomes'],{'not_evaluated':1})


if __name__=='__main__':unittest.main()
