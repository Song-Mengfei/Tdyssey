"""Offline fixtures for C/D evaluation reuse, epsilon observation and reports."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2  # Load before patch.dict(sys.modules) so native submodules stay intact.


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
report=module('cd_saved_report','report.py');evaluation=module('cd_evaluation','evaluation.py')


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def case(root,name,letter,split='seen_val'):
    path=root/(name+'.npz');rgb=np.full((20,3,96,96),255,np.uint8)
    for i in range(20):rgb[i,:,20:55,20+i:45+i]=125
    np.savez_compressed(path,rgb=rgb,control_valid=np.ones(19,bool))
    return dict(sample_id=name,family_id=name,letter=letter,split=split,raw_path=str(path),raw_sha256=report.sha(path),anchor=3,horizon=16,truth_h16_signed_rotation_deg=20.,bitstatic_future=False)


def metrics(root,rows,arm,mode):
    directory=root/arm/mode;directory.mkdir(parents=True,exist_ok=True);scored=[];files=[]
    for index,row in enumerate(rows):
        path=directory/f'raw-{index}.npz';np.savez_compressed(path,raw=report._truth(row))
        files.append(dict(sample_id=row['sample_id'],path=str(path),sha256=report.sha(path),seeds=list(range(16))))
        for h in range(1,17):
            iou,pos,contour={'baseline':(.6,2.,1.),'A':(.7,1.,.8),'B':(.72,1.2,.8),'C':(.75,.8,.7),'D':(.76,.7,.6)}[arm]
            if arm=='D' and row['letter']=='O':iou=.65;pos=1.1
            scored.append(dict(sample_id=row['sample_id'],family_id=row['family_id'],letter=row['letter'],split=row['split'],horizon=h,method=arm,obvious_motion=row['letter'] in 'OC',
                metrics=dict(raw_mask_iou=iou,object_position_error_image_px=pos,raw_contour_error_image_px=contour,motion_vector_error_image_px=pos,pusher_position_error_image_px=.1,pusher_mask_iou=.8,
                    holes_difference=1 if arm=='D' and row['letter']=='B' else 0,predicted_holes=1 if arm=='D' and row['letter']=='B' else 2,truth_holes=2,missing_components=0,extra_fragments=0,
                    aligned_mask_iou=.9,visible_angle_error_deg=1.,angle_assessable=False,underrotation_assessable=False,absolute_underrotation_deg=None,legacy_shape_assessable=False,
                    legacy_shape_unknown_reason='occlusion',failure_reasons=[])))
    proof=dict(sample_ids=[r['sample_id'] for r in rows],family_ids=[r['family_id'] for r in rows],seeds=list(range(16)),initial_history_sha256='fixed-history',absolute_actions_sha256='fixed-absolute',normalized_actions_sha256='fixed-normalized',shape_history=[len(rows),4,3,96,96],shape_actions=[len(rows),19,2])
    if mode=='real_history':proof['per_target_true_histories_sha256']='fixed-true-target-histories'
    value=dict(rows=scored,raw_files=files,input_evidence=[proof],complete=True)
    if arm in ('C','D'):
        steps=[dict(step=h+1,seed=h,batch_size=len(rows),actual_initial_epsilon_sha256='actual-'+str(h),replayed_seed_batch_epsilon_sha256='actual-'+str(h),per_case_initial_epsilon_sha256=['slot'+str(i) for i in range(len(rows))],actual_matches_independent_replay=True) for h in range(16)]
        value['noise_evidence']=[dict(batch=0,sample_ids=[r['sample_id'] for r in rows],steps=steps)]
    path=directory/'metrics.json';dump(path,value);return dict(metrics=str(path)),value


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='offline-eval-report-');self.root=Path(self.tmp.name)
        self.guard=patch.object(report,'guard',lambda *args:None);self.guard.start()
    def tearDown(self):self.guard.stop();self.tmp.cleanup()

    def test_saved_six_column_report_keeps_unknown_and_new_failures(self):
        formal=[case(self.root,'formal-'+l,l) for l in 'OCB'];diag=[case(self.root,s+'-'+l,l,s) for s in ('train','seen_val') for l in 'OC']
        data=self.root/'data';oc=self.root/'oc';dump(data/'evaluation-spec.json',dict(rows=formal));dump(oc/'diagnostic-spec.json',dict(rows=diag))
        evaluations={}
        for arm in report.LABELS:
            f,_=metrics(self.root,formal,arm,'formal');d={}
            for mode in ('real_history','autoregressive'):d[mode],_=metrics(self.root,diag,arm,mode)
            evaluations[arm]=dict(formal=f,diagnostics=d)
        limits=dict(H1_IoU_min=.7,H1_position_max_imagepx=1.,H16_AR_position_max_imagepx=1.,epoch2_H1_position_reference_imagepx=.9)
        dump(self.root/'support-limits.json',limits)
        def analysis(spec,models,cal):
            return dict(o_direction_rows=[dict(sample_id='formal-O',labels={a:dict(known=False,classification='unknown',signed_delta_deg={}) for a in report.LABELS})],b_hole_rows=[dict(sample_id='formal-B',labels={a:dict(holes_at_H16=1,first_mismatch=14,unknownframes=[15]) for a in report.LABELS})])
        direction=SimpleNamespace(analyze_saved=analysis,support_signals=lambda data,a:dict(O_display_known_correct_clockwise=False,B_visible_hole_support_passed=False))
        engine=dict(complete=True,evaluations=evaluations,arms={},physical_steps=0,supervised_targets_total=60544,optimizer_batches_total=3784)
        with patch.dict('sys.modules',direction=direction):
            result=report.generate_report(self.root,dict(data_root=str(data),oc_root=str(oc)),engine)
        summary=report.read(result['summary']);pairs=report.read(result['cases'])['rows']
        self.assertEqual(summary['primary_comparisons'],['C_vs_A','D_vs_C'])
        self.assertTrue(any(x['comparison']=='D_vs_C' and x['letter']=='O' and x['metrics']['raw_mask_iou']['status']=='regressed' for x in pairs))
        self.assertTrue(any(x['letter']=='B' and x['comparison']=='D_vs_C' and x['metrics']['holes_difference']['status']=='regressed' for x in pairs))
        self.assertTrue(all(g['angle_unknown']==g['denominator'] for g in summary['groups']))
        self.assertFalse(summary['support_signals']['arms']['D']['all_required_support_signals_pass'])
        import cv2
        first=summary['media'][0];self.assertEqual(cv2.imread(first['board']['path']).shape,(1000,1152,3));self.assertEqual(cv2.imread(first['all16_board']['path']).shape,(4000,1152,3))
        self.assertEqual(first['video']['frames'],16);self.assertAlmostEqual(first['video']['fps'],10)
        self.assertIn('not equal compute',Path(result['report']).read_text())
        self.assertIn('negative old-B-versus-old-A',Path(result['report']).read_text())

    def test_strict_full_precision_gate_does_not_relax_or_accept_unknown(self):
        limits=dict(H1_IoU_min=.912996800000001,H1_position_max_imagepx=.270391100000001,H16_AR_position_max_imagepx=1.139164200000001,epoch2_H1_position_reference_imagepx=.252620700000001)
        groups=[]
        for arm in ('C','D'):
            for mode,h in (('real_history',1),('autoregressive',16)):
                groups.append(dict(scope='diagnostic32',split='seen_val',subset='ALL_OC',arm=arm,mode=mode,horizon=h,denominator=16,metrics={'raw_mask_iou':dict(mean=limits['H1_IoU_min']-(1e-15 if arm=='C' else 0),available=16,unknown=0), 'object_position_error_image_px':dict(mean=limits['H1_position_max_imagepx'] if h==1 else limits['H16_AR_position_max_imagepx'],available=16,unknown=0)}))
        support=report.strict_support(groups,limits,{'C':dict(O_display_known_correct_clockwise=True,B_visible_hole_support_passed=True),'D':dict(O_display_known_correct_clockwise=False,B_visible_hole_support_passed=True)})
        self.assertFalse(support['arms']['C']['motion_support']);self.assertTrue(support['arms']['D']['motion_support'])
        self.assertFalse(support['arms']['D']['all_required_support_signals_pass'])
        # Even a favorable mean over15 cases cannot pass any16-case signal.
        for mode,h,key,signal,favorable in (
            ('real_history',1,'raw_mask_iou','H1_IoU_min',1.),
            ('real_history',1,'object_position_error_image_px','H1_position_max_imagepx',0.),
            ('autoregressive',16,'object_position_error_image_px','H16_AR_position_max_imagepx',0.)):
            item=next(g for g in groups if g['arm']=='D' and g['mode']==mode and g['horizon']==h)['metrics'][key]
            original=item.copy();item.update(mean=favorable,available=15,unknown=1)
            partial=report.strict_support(groups,limits,{})['arms']['D']
            self.assertEqual(partial['motion_signals'][signal]['status'],'unknown')
            self.assertFalse(partial['motion_support'])
            item.clear();item.update(original)

    def test_runtime_actual_epsilon_and_inputs_are_required(self):
        row=case(self.root,'same','O');bundle={}
        for arm in report.LABELS:_,bundle[arm]=metrics(self.root,[row],arm,'formal')
        report.check_pairing(bundle)
        bundle['D']['noise_evidence'][0]['steps'][0]['actual_initial_epsilon_sha256']='different'
        with self.assertRaisesRegex(ValueError,'actual initial epsilon'):report.check_pairing(bundle)
        bundle['D']['noise_evidence'][0]['steps'][0]['actual_initial_epsilon_sha256']='actual-0'
        bundle['D']['input_evidence'][0]['absolute_actions_sha256']='different'
        with self.assertRaisesRegex(ValueError,'input evidence differs'):report.check_pairing(bundle)

    def test_noise_observer_returns_exact_tensor_and_reference_reuse_is_model_free(self):
        class Tensor:
            def __init__(self,array):self.array=np.asarray(array,np.float32);self.shape=self.array.shape
            def detach(self):return self
            def float(self):return self
            def cpu(self):return self
            def contiguous(self):return self
            def numpy(self):return self.array
        expected=Tensor(np.arange(24).reshape(2,3,2,2));draw=lambda *a,**k:expected
        fake=SimpleNamespace(randn=draw,equal=lambda a,b:np.array_equal(a.array,b.array));received=[]
        with evaluation.capture_noise(fake,expected,received.append):self.assertIs(fake.randn(2,3,2,2),expected)
        self.assertIs(fake.randn,draw);self.assertTrue(received[0]['actual_matches_independent_replay'])
        source=self.root/'old-result.json';dump(source,dict(complete=True));refs={a:dict(formal=dict(metrics='oldmetrics',wall_seconds=5),diagnostics={m:dict(metrics='oldmetrics',wall_seconds=5) for m in ('real_history','autoregressive')}) for a in ('baseline','A','B')}
        dump(self.root/'reused-evaluations.json',dict(source_result=str(source),source_result_sha256=evaluation.sha(source),evaluations=refs))
        before=evaluation.sha(source)
        with patch.object(evaluation,'old',side_effect=AssertionError('No old model helper allowed')):
            result=evaluation.run_baseline(self.root,{})
        self.assertEqual(before,evaluation.sha(source));self.assertTrue(all(x['reused'] and x['new_model_calls']==0 for x in result.values()))


if __name__=='__main__':unittest.main(verbosity=2)
