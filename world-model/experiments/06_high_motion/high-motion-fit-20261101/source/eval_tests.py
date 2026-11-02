"""Targeted CPU toys for acceptance, timing, and measurement isolation."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
ev=load('_hm_eval_test_target',HERE/'evaluation.py')
report=load('_hm_report_test_target',HERE/'report.py')

def fixture():
    rows=[];copies=[]
    for letter in ('O','C','B'):
        for case in range(16):
            for mode in ('real_history','autoregressive'):
                for h in range(1,17):
                    row=dict(sample_id=f'{letter}{case}',letter=letter,selection_role='train',input_mode=mode,horizon=h,
                        strong_H1=True,dominance='translation' if h%2 else 'rotation',
                        metrics=dict(raw_mask_iou=.84 if letter=='B' and case<4 else .99,
                            object_position_error_image_px=.1,raw_contour_error_image_px=.1,pusher_position_error_image_px=.1,failure_reasons=[]),
                        rotation=dict(required=True,status='pass'),visible_shape_status='pass',visible_topology_status='pass')
                    rows.append(row);copy=deepcopy(row);copy['metrics']['object_position_error_image_px']=1.;copy['metrics']['raw_contour_error_image_px']=1.;copies.append(copy)
    return rows,copies

class Tests(unittest.TestCase):
    def test_total_ninety_percent_not_per_letter_extra_gate(self):
        rows,copies=fixture();a=ev.assess(rows,copies,ev.standard())
        self.assertTrue(a['all_passed'])
        total=next(s for s in a['signals'] if s['letter']=='ALL_OCB' and s['name']=='H16_case_fraction_IoU85')
        self.assertEqual(total['n'],48);self.assertEqual(total['value'],44/48)
        letter=next(s for s in a['signals'] if s['letter']=='B' and s['name']=='H16_case_fraction_IoU85')
        self.assertEqual(letter['value'],.75);self.assertFalse(letter['gate'])
    def test_one_unavailable_measurement_cannot_be_averaged_away(self):
        rows,copies=fixture();rows[0]['metrics']['object_position_error_image_px']=None
        a=ev.assess(rows,copies,ev.standard());self.assertFalse(a['all_passed'])
        signal=next(s for s in a['signals'] if s['letter']=='O' and s['name']=='H1_translation_position')
        self.assertIsNone(signal['passed']);self.assertEqual(signal['stats']['unknown'],1)
        reduction=next(s for s in a['signals'] if s['letter']=='O' and s['name']=='H1_translation_copy_reduction')
        self.assertIsNone(reduction['passed'])
    def test_unknown_and_static_rotation_do_not_pass(self):
        rows,copies=fixture();rows[0]['rotation']['status']='unknown'
        self.assertFalse(ev.assess(rows,copies,ev.standard())['all_passed'])
        def f(angle):return dict(gray=dict(angle=angle,ratio=2),void=dict(angle=angle,ratio=3),cropped=False,blue_in_enclosed_region=False,components=1)
        truth=[f(6*k) for k in range(17)];recorded=list(np.arange(17)*6.)
        # Independent RH outputs follow each preceding real frame but do not move.
        pred=[f(6*k) for k in range(16)]
        x=ev.rotation_assessment(truth,recorded,pred,'real_history',16,ev.standard())
        self.assertEqual(x['classification'],'no_resolved_rotation');self.assertEqual(x['status'],'fail')
        self.assertEqual(x['visible_signed_rotation_deg']['gray'],0.)
    def test_copy_reference_modes_have_different_truth_availability(self):
        # ARcopy holds G0; RHcopy is each own real preceding target.
        anchor=np.arange(17);rh=anchor[:-1];ar=np.repeat(anchor[:1],16)
        self.assertEqual(rh[15],15);self.assertEqual(ar[15],0)
        self.assertEqual(rh[0],ar[0])
    def test_GT_self_response_below_guard_is_measurement_unknown_not_failure(self):
        def f(a):return dict(gray=dict(angle=a,ratio=2),void=dict(angle=a,ratio=3),cropped=False,blue_in_enclosed_region=False,components=1)
        truth=[f(4*k) for k in range(17)];angles=list(np.arange(17)*6.)
        result=ev.rotation_assessment(truth,angles,truth[1:],'real_history',1,ev.standard())
        self.assertEqual(result['status'],'unknown')
        self.assertIn('GT_visible_response_not_resolved',result['unknown_reasons'])
    def test_fresh_import_does_not_import_torch(self):
        result=subprocess.run([sys.executable,'-X','utf8','-B','-c','import evaluation, report; import sys; assert "torch" not in sys.modules'],cwd=HERE,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_report_rejects_actual_epsilon_mismatch_even_with_same_seeds(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='eval-test-') as temp:
            path=Path(temp)/'raw.bin';path.write_bytes(b'fixed saved evidence')
            rf=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_id='s',family_id='f',input_mode='autoregressive',raw_path='truth',raw_sha256='truthsha',anchor=3,batch_id=0,batch_slot=0,seeds=[7]*16,initial_epsilon_sha256=['actualA']*16)
            base=dict(complete=True,raw_files=[rf],input_evidence=[],noise_evidence=[]);final=deepcopy(base)
            self.assertTrue(report.check_pairing({'0':base,'final':final})['passed'])
            final['raw_files'][0]['initial_epsilon_sha256'][0]='actualB'
            with self.assertRaisesRegex(ValueError,'identity differs'):report.check_pairing({'0':base,'final':final})

if __name__=='__main__':unittest.main()
