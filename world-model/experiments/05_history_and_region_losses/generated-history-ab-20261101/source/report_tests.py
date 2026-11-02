"""Offline A/B saved-data/report fixtures; no Torch, model or physics calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec=importlib.util.spec_from_file_location('gh4_saved_report',Path(__file__).with_name('report.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def saved_case(root,name,letter,split='seen_val'):
    path=root/(name+'.npz');rgb=np.full((20,3,96,96),255,np.uint8)
    for i in range(20):
        rgb[i,:,20:55,25+i:45+i]=125
        rgb[i,0,64:69,15+i:20+i]=60;rgb[i,1,64:69,15+i:20+i]=100
    np.savez_compressed(path,rgb=rgb,control_valid=np.ones(19,bool))
    return dict(sample_id=name,family_id=name,letter=letter,split=split,
        raw_path=str(path),raw_sha256=report.sha(path),anchor=3,horizon=16,
        truth_h16_signed_rotation_deg=20.,bitstatic_future=False)


def evaluation(root,cases,arm,mode):
    folder=root/arm/mode;folder.mkdir(parents=True,exist_ok=True);rows=[];files=[]
    for i,case in enumerate(cases):
        raw=report._truth(case);raw[:,:,20:24,20:24]=.2
        path=folder/f'raw-{i}.npz';np.savez_compressed(path,raw=raw)
        files.append(dict(sample_id=case['sample_id'],path=str(path),sha256=report.sha(path),seeds=list(range(16))))
        for h in range(1,17):
            if arm=='baseline':iou,pos,contour=.6,2.,1.
            elif arm=='A':iou,pos,contour=.7,1.,.8
            elif case['letter']=='O':iou,pos,contour=.75,.8,.7
            elif case['letter']=='C':iou,pos,contour=.65,.7,.9
            else:iou,pos,contour=.72,1.2,.8
            rows.append(dict(sample_id=case['sample_id'],family_id=case['family_id'],letter=case['letter'],split=case['split'],
                horizon=h,method=arm,obvious_motion=case['letter'] in 'OC',
                metrics=dict(raw_mask_iou=iou,object_position_error_image_px=pos,raw_contour_error_image_px=contour,
                    motion_vector_error_image_px=pos,pusher_position_error_image_px=.1,pusher_mask_iou=.8,
                    holes_difference=2 if arm=='B' and case['letter']=='B' else 0,
                    predicted_holes=0 if arm=='B' and case['letter']=='B' else 2,truth_holes=2,
                    missing_components=0,extra_fragments=0,aligned_mask_iou=.9,
                    visible_angle_error_deg=1.,angle_assessable=False,angle_unknown_reason='symmetry_or_occlusion',
                    legacy_shape_assessable=False,legacy_shape_unknown_reason='occlusion',failure_reasons=[])))
    evidence=dict(sample_ids=[row['sample_id'] for row in cases],family_ids=[row['family_id'] for row in cases],seeds=list(range(16)),
        initial_history_sha256='fixed-initial-history',absolute_actions_sha256='fixed-absolute-actions',
        normalized_actions_sha256='fixed-normalized-actions',shape_history=[len(cases),4,3,96,96],shape_actions=[len(cases),19,2])
    if mode=='real_history':evidence['per_target_true_histories_sha256']='fixed-per-target-true-history'
    value=dict(rows=rows,raw_files=files,complete=True,input_evidence=[evidence]);path=folder/'metrics.json';dump(path,value)
    return dict(metrics=str(path)),value


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='offline-report-')
        self.root=Path(self.tmp.name);self.ar1=self.root/'data';self.oc=self.root/'oc';self.guard=patch.object(report,'guard',lambda *args:None);self.guard.start()

    def tearDown(self):
        self.guard.stop();self.tmp.cleanup()

    def test_complete_saved_report_compares_B_to_A_and_retains_bad_cases(self):
        formal=[saved_case(self.root,'formal-'+letter,letter) for letter in 'OCB']
        diag=[saved_case(self.root,split+'-'+letter,letter,split) for split in ('train','seen_val') for letter in 'OC']
        dump(self.ar1/'evaluation-spec.json',dict(rows=formal));dump(self.oc/'diagnostic-spec.json',dict(rows=diag))
        results={}
        for arm in report.LABELS:
            f,_=evaluation(self.root,formal,arm,'formal');d={}
            for mode in ('real_history','autoregressive'):d[mode],_=evaluation(self.root,diag,arm,mode)
            results[arm]=dict(formal=f,diagnostics=d)
        engine=dict(complete=True,evaluations=results,arms={},physical_steps=0,supervised_targets_total=60544,optimizer_batches_total=3784,generation_counts={'B':dict(images=22704)})
        output=report.generate_report(self.root,dict(data_root=str(self.ar1),oc_root=str(self.oc)),engine)
        summary=report.read(output['summary']);pairs=report.read(output['cases'])['rows']
        primary=[g for g in summary['comparison_groups'] if g['scope']=='formal48' and g['subset']=='ALL' and g['horizon']==16 and g['comparison']=='B_vs_A'][0]
        self.assertEqual(primary['metrics']['raw_mask_iou']['improved'],2)
        self.assertEqual(primary['metrics']['raw_mask_iou']['regressed'],1)
        self.assertEqual(primary['metrics']['visible_angle_error_deg']['unknown'],3)
        self.assertTrue(any(x['letter']=='B' and x['metrics']['holes_difference']['status']=='regressed' for x in pairs if x['comparison']=='B_vs_A'))
        self.assertEqual(len({x['horizon'] for x in pairs}),16)
        self.assertEqual(len(summary['media']),11)
        import cv2
        first=summary['media'][0]
        self.assertEqual(cv2.imread(first['board']['path']).shape,(1000,768,3))
        self.assertEqual(cv2.imread(first['all16_board']['path']).shape,(4000,768,3))
        self.assertEqual(first['video']['frames'],16)
        page=Path(output['index']).read_text()
        self.assertIn('B versus A',page)
        self.assertIn('not equal compute',Path(output['report']).read_text())
        self.assertFalse(any(g['angle_assessable'] for g in summary['groups']))

    def test_pairing_rejects_missing_target_or_noise_change(self):
        row=saved_case(self.root,'same','O');bundle={}
        for arm in report.LABELS:_,bundle[arm]=evaluation(self.root,[row],arm,'formal')
        bundle['B']['raw_files'][0]['seeds'][0]=99
        with self.assertRaisesRegex(ValueError,'noise'):report.check_pairing(bundle)
        bundle['B']['raw_files'][0]['seeds'][0]=0
        bundle['B']['input_evidence'][0]['absolute_actions_sha256']='different-actions'
        with self.assertRaisesRegex(ValueError,'input evidence differs'):report.check_pairing(bundle)
        bundle['B']['input_evidence'][0]['absolute_actions_sha256']='fixed-absolute-actions'
        bundle['baseline'].pop('input_evidence')
        bundle['baseline']['reused_formal_metrics']='explicit-old-source'
        _,proof=report.check_pairing(bundle)
        self.assertFalse(proof['baseline_runtime_input_evidence_available'])
        bundle['B']['rows'].pop()
        with self.assertRaisesRegex(ValueError,'horizon identities'):report.check_pairing(bundle)

    def test_mixed_and_unknown_are_not_hidden_by_average(self):
        before=dict(metrics=dict(raw_mask_iou=.5,object_position_error_image_px=2.,raw_contour_error_image_px=1.,visible_angle_error_deg=90.,angle_assessable=False))
        after=dict(metrics=dict(raw_mask_iou=.6,object_position_error_image_px=3.,raw_contour_error_image_px=.8,visible_angle_error_deg=0.,angle_assessable=False))
        value=report.compare_pair(before,after)
        self.assertEqual(value['image_metric_direction'],'mixed')
        self.assertEqual(value['metrics']['visible_angle_error_deg']['status'],'unknown')
        del after['metrics']['raw_contour_error_image_px']
        self.assertEqual(report.compare_pair(before,after)['image_metric_direction'],'unknown')


if __name__=='__main__':unittest.main(verbosity=2)
