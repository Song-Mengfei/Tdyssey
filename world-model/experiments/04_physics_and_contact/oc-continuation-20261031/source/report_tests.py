"""Saved synthetic fixtures only: pairing, unknowns, split/mode labels and media."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec=importlib.util.spec_from_file_location("oc_saved_report",Path(__file__).with_name("report.py"))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def case(root,name,letter,split="seen_val"):
    raw=root/(name+".npz")
    rgb=np.full((20,3,96,96),255,np.uint8)
    for i in range(20):
        rgb[i,:,20:50,20+i:40+i]=125
        rgb[i,0,65:70,15+i:20+i]=60
        rgb[i,1,65:70,15+i:20+i]=100
    np.savez_compressed(raw,rgb=rgb,control_valid=np.ones(19,bool))
    return dict(sample_id=name,family_id=name,letter=letter,split=split,
                raw_path=str(raw),raw_sha256=report.sha(raw),anchor=3,horizon=16)


def metrics(root,rows,phase,epoch=1):
    folder=root/phase;folder.mkdir(parents=True,exist_ok=True)
    records=[];scored=[]
    for i,row in enumerate(rows):
        truth,last=report._truth(row)
        raw=truth.copy();raw[:,:,20:25,20:25]=.3+epoch*.03
        path=folder/f"raw-{i}.npz";np.savez_compressed(path,raw=raw)
        records.append(dict(sample_id=row['sample_id'],path=str(path),sha256=report.sha(path),seeds=list(range(16))))
        for h in report.HORIZONS:
            scored.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],letter=row['letter'],
                split=row['split'],horizon=h,method=phase,obvious_motion=row['letter'] in 'OC',
                metrics=dict(raw_mask_iou=.8-epoch*.01 if row['letter']=='B' else .7+epoch*.01,
                    object_position_error_image_px=2/epoch,pusher_position_error_image_px=1.,
                    visible_angle_error_deg=None,holes_difference=0,
                    legacy_shape_assessable=False,legacy_shape_deformation_failure=None,
                    failure_reasons=[])))
    result=dict(complete=True,rows=scored,raw_files=records,true_future_feedback=False)
    path=folder/'metrics.json';write(path,result)
    return path,result


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix="offline-report-")
        self.root=Path(self.temp.name);self.ar1=self.root/'ar1';self.ar1.mkdir()
        self.guard=patch.object(report,'_guard',lambda *args:None);self.guard.start()

    def tearDown(self):
        self.guard.stop();self.temp.cleanup()

    def test_saved_end_to_end_preserves_modes_unknowns_and_regression(self):
        rows=[case(self.root,'formal-'+l,l) for l in 'OCB']
        write(self.ar1/'evaluation-spec.json',dict(rows=rows))
        paths={}
        for epoch in (1,2,3):
            path,value=metrics(self.root,rows,'formal'+str(epoch),epoch)
            paths[str(epoch)]=str(path)
        baseline=[]
        for row in report.read(paths['1'])['rows']:
            baseline.append(dict(row,method='copy_last'))
        write(self.ar1/'evaluation/before/metrics.json',dict(rows=baseline))
        diagrows=[case(self.root,split+'-'+letter,letter,split) for split in ('train','seen_val') for letter in 'OC']
        write(self.root/'diagnostic-spec.json',dict(rows=diagrows))
        diag={}
        for epoch in (1,3):
            diag[str(epoch)]={}
            for mode in ('real_history','autoregressive'):
                path,value=metrics(self.root,diagrows,f'diagnostics/epoch{epoch}/{mode}',epoch)
                diag[str(epoch)][mode]=dict(metrics=str(path))
        for epoch in (1,2,3):
            write(self.root/f'diagnostics/epoch{epoch}/fixed-loss.json',dict(
                means={'train':dict(mean=.1/epoch,samples=2),'seen_val':dict(mean=.2/epoch,samples=2)},
                noise_and_sample_identity='fixed-mock-sample-noise'))
        path,d16=metrics(self.root,rows[:2],'diagnostics/epoch1/denoise16')
        diag['1']['denoise16']=dict(metrics=str(path))
        engine=dict(complete=True,formal48=paths,diagnostics=diag,epochs=[],
                    actual_additional_exposures=61870,successful_updates=3868,AMP_skips=0)
        original=report._media
        def boards_only(root,config,name,sample_id,columns,video=False):
            return original(root,config,name,sample_id,columns,video=False)
        with patch.object(report,'_media',boards_only):
            result=report.generate_report(self.root,{'ar1_root':str(self.ar1)},engine)
        summary=report.read(result['summary'])
        self.assertFalse(result['missing_diagnostics'])
        self.assertEqual(len(summary['media']),9) # O/C/B formal,4 split diagnostics,2 inference-only.
        self.assertEqual({x['split'] for x in summary['media'] if x['kind']=='motion_biased_diagnostic'}, {'train','seen_val'})
        self.assertTrue(all(x['means']['visible_angle_error_deg']['unknown']==x['denominator'] for x in summary['formal_groups']))
        b=[x for x in summary['formal_groups'] if x['letter']=='B' and x['horizon']==16]
        self.assertGreater(b[0]['means']['raw_mask_iou']['mean'],b[-1]['means']['raw_mask_iou']['mean'])
        import cv2
        board=cv2.imread(summary['media'][0]['board'])
        self.assertEqual(board.shape,(1000,960,3))
        text=(self.root/'REPORT.md').read_text()
        self.assertIn('never a sixteen-step closed-loop',text)
        self.assertIn('batch-one versus original batch-four',text)
        self.assertEqual(result['report_model_calls'],0)
        self.assertTrue(summary['fixed_loss_sample_noise_identity_matched'])
        self.assertIn('0.10000000',text)

    def test_pairing_rejects_changed_noise_and_unknown_does_not_pass(self):
        row=case(self.root,'same','O')
        _,one=metrics(self.root,[row],'one');_,two=metrics(self.root,[row],'two')
        two['raw_files'][0]['seeds'][0]=99
        with self.assertRaisesRegex(ValueError,'noise'):
            report._check_formal_pairing({'1':one,'2':two})
        summary=report.summarize(one['rows'])
        self.assertEqual(summary['shape_unknown'],4)
        self.assertEqual(summary['means']['visible_angle_error_deg']['available'],0)

    def test_short_native_video_is_real_ten_hz_and_decodes(self):
        row=case(self.root,'video','O');truth,last=report._truth(row)
        result=report._media(self.root,{},'fixture-video',row['sample_id'],
            [('SAVED TRUTH',truth),('COPY LAST',np.repeat(last[None],16,axis=0))],video=True)
        self.assertEqual(result['frames'],16)
        self.assertAlmostEqual(result['fps'],10,places=3)


if __name__=='__main__':
    unittest.main(verbosity=2)
