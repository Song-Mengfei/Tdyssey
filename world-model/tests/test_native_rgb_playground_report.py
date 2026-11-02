"""CPU-only saved-array report checks; no model, renderer or physics."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from tdyssey import native_rgb_playground_report as report


def image(holes=2,blue=True):
    value=np.ones((3,96,96),dtype=np.float32)
    value[:,25:65,28:62]=np.asarray([.46666667,.5333333,.6])[:,None,None]
    value[:,31:40,36:53]=1
    if holes==2:value[:,48:57,36:53]=1
    if blue:value[:,29:34,24:29]=np.asarray([.12,.2,1.])[:,None,None]
    return value


class PlaygroundReportTests(unittest.TestCase):
    def test_manual_deformation_is_visible_despite_palette_unknown(self):
        row=dict(shape_assessment='unknown_occlusion_or_clip',visible_defects=[],
            visual_observation={'reviewed':True,'visible_rounding':True,'visible_deformation':True})
        self.assertEqual(report.observed_defects(row),['manual_rounding','manual_deformation'])
        self.assertEqual(row['shape_assessment'],'unknown_occlusion_or_clip')

    def test_runner_seconds_legacy_and_missing_timing(self):
        timing=report.sampler_timing([{'seconds':.25,'sampler_s':99}]*4,4)
        self.assertEqual(timing['total_seconds'],1.)
        self.assertEqual(timing['source_keys'],{'seconds':4})
        self.assertEqual(report.sampler_timing([{'sampler_s':.5}]*4,4)['total_seconds'],2.)
        incomplete=report.sampler_timing([{'seconds':.25}]*3,4)
        self.assertIsNone(incomplete['total_seconds'])
        self.assertEqual(incomplete['missing_or_invalid_records'],1)
        self.assertIsNone(report.sampler_timing([{'seconds':float('nan')}]*4,4)['total_seconds'])

    def test_paired_sampler_timing_sums_each_branch(self):
        groups=[]
        for steps in(3,8,16):
            groups.append(dict(physics_cohort='fixture',case_id='C-fixed',anchor=7,seed_steps=[1,2,3,4],
                sampler_steps=steps,id=f'fixture-{steps}',grid='fixture.png',
                rows=[dict(branch=b,step=s,object_roi_mae=.1,gray_iou=.8,visible_defects=[])for b in('A','B')for s in range(1,5)],
                source_metadata={'step_records_A':[{'seconds':steps*.1}]*4,'step_records_B':[{'seconds':steps*.2}]*4}))
        pairs=report.paired_sampler_comparison(groups)
        self.assertEqual(len(pairs),6)
        for pair in pairs:
            self.assertAlmostEqual(pair['four_sampler_seconds'],pair['sampler_steps']*(.4 if pair['branch']=='A'else .8))
            self.assertEqual(pair['sampler_timing']['valid_records'],4)

    def test_visible_hole_and_pusher_failure_survive_global_occlusion(self):
        calibration=report.read(report.CALIBRATION)
        truth=image(2,True);prediction=image(1,False)
        metrics=report.measure_frame(prediction,truth,truth,calibration)
        self.assertTrue(metrics['occlusion_ambiguous'])
        self.assertIn('pusher_disappeared',metrics['visible_defects'])
        self.assertIn('visible_hole_count_change',metrics['visible_defects'])
        self.assertEqual(metrics['truth_holes'],2)
        self.assertEqual(metrics['predicted_holes'],1)
        self.assertEqual(metrics['shape_assessment'],'visible_defect')

    def test_finite_out_of_range_raw_is_measured_without_mutation(self):
        calibration=report.read(report.CALIBRATION)
        value=image();value[:,0,0]=1.02;before=value.copy()
        metrics=report.measure_frame(value,image(),image(),calibration)
        self.assertEqual(metrics['status'],'measured')
        self.assertTrue(metrics['raw_finite_out_of_range'])
        np.testing.assert_array_equal(value,before)
        invalid=before.copy();invalid[0,1,1]=np.nan
        measured=report.measure_frame(invalid,image(),image(),calibration)
        self.assertEqual(measured['status'],'nonfinite')
        self.assertEqual(measured['shape_assessment'],'unknown')

    def test_saved_clip_grid_video_and_full_reference_status(self):
        with tempfile.TemporaryDirectory(dir=report.ROOT/'artifacts')as temporary:
            stage=Path(temporary);folder=stage/'clips/B-fixture-a3/3-s5478';folder.mkdir(parents=True)
            truth=np.stack([image()]*5);a=np.stack([image()]*4);b=a.copy();b[2]=image(1,False)
            np.savez_compressed(folder/'clip.npz',truth_rgb=truth,initial_history=truth[:4],actions_absolute=np.zeros((7,2)),predictions_A=a,predictions_B=b,states15=np.zeros((5,15)),seed_steps=np.arange(5478,5482))
            report.write(folder/'metadata.json',dict(case_id='B-fixture',letter='B',anchor=3,
                sampler={'num_steps_denoising':3},physics_protocol_sha256=report.TRAINING_PHYSICS_SHA,
                first_frame_check_mode='independently_checked',first_output_equal=True,
                motion_contact=[dict(step=s,translation_image96_px=0,signed_rotation_deg=0,contact_substeps=1,completed_substeps=10,actual_target=[1,2],mouse_preview_target=None)for s in range(1,5)]))
            raw_before=report.sha(folder/'clip.npz')
            result=report.build_report(stage,videos=True)
            self.assertEqual(result['groups'],1)
            self.assertEqual(result['rows'],8)
            self.assertEqual(report.sha(folder/'clip.npz'),raw_before)
            summary=report.read(stage/'report/summary.json')
            self.assertEqual(summary['groups'][0]['video_info']['decoded_frames'],24)
            status=summary['user_reference_correspondence'][0]
            self.assertEqual(status['status'],'已复现同类')
            self.assertFalse(status['exact_original_screenshot_replay'])
            self.assertEqual(status['components']['pusher_disappeared'][0]['step'],3)
            self.assertEqual(summary['groups'][0]['physics_cohort'],'historical_dynamic_half_life_0.1s')
            self.assertEqual(summary['model_calls_by_report'],0)
            self.assertTrue((stage/'report/assets/clip000-closeup.png').is_file())


if __name__=='__main__':unittest.main()
