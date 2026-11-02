"""Synthetic RGB/failed-array CPU scoring checks; no model or physics calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import native_rgb_stage3_score as score


def image(x=30):
    rgb=np.full((3,96,96),255,dtype=np.uint8)
    rgb[:,30:50,x:x+20]=np.asarray((119,136,153))[:,None,None]
    rgb[:,68:74,68:74]=np.asarray((65,105,225))[:,None,None]
    return rgb


def windows(n):
    return [dict(window_id=f"mock-{i}:3",trajectory_id=f"mock-{i}",anchor=3,letter=score.legacy.TRAIN_LETTERS[i%18],
        h1=dict(motion_class="moving" if i<31 else "near_static" if i<35 else "intermediate",actual_boundary_le32=False,
                actual_boundary_gap_band=">64")) for i in range(n)]


def dynamic_rows(wins=18,failure=None,hold_failure=False):
    rows=[]
    for index in range(18):
        hold=dict(raw_object_error=.5,hard_generation_failure=hold_failure,truth_issue=None)
        fitted=dict(raw_object_error=.1 if index<wins else .5,hard_generation_failure=index==failure,truth_issue=None)
        rows.append(dict(methods=dict(hold=hold,fitted=fitted)))
    return rows


class Stage3ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.protocol,cls.calibration,_=score.load_protocol()

    def measure(self,prediction,truth=None,last=None,**kwargs):
        return score.score_image(prediction,image(34) if truth is None else truth,image() if last is None else last,
                                 self.calibration,letter="T",**kwargs)

    def test_frozen_calibration_and_explicit_registration_scope(self):
        self.assertEqual(self.calibration["sha256"],"cd2ed20812e8e53f138579939c45934b19329ef1afd111ec777b20ec0588e20c")
        self.assertEqual(self.calibration["palette"],dict(minimum_alpha=.4,maximum_alpha=1.25,colour_residual=.12))
        self.assertEqual(self.calibration["angle_grid_degrees"],10)
        self.assertEqual(self.calibration["registration_translation_grid_pixels"],[-1,-.5,0,.5,1])

    def test_perfect_raw_prediction_zero_errors_with_no_image_transform(self):
        value=image(34);original=value.copy();row=self.measure(value)
        self.assertEqual(row["raw_object_error"],0)
        self.assertEqual(row["raw_contour_error_image_px"],0)
        self.assertEqual(row["object_foreground_rgb_mae"],0)
        self.assertEqual(row["pusher_error"],0)
        self.assertFalse(row["hard_generation_failure"])
        np.testing.assert_array_equal(value,original)

    def test_empty_object_retains_error_one_and_contour_penalty(self):
        value=image(34);value[:,30:50,34:54]=255
        row=self.measure(value)
        self.assertEqual(row["raw_mask_iou"],0)
        self.assertEqual(row["raw_object_error"],1)
        self.assertEqual(row["raw_contour_error_image_px"],score.DIAGONAL)
        self.assertIn("empty_object",row["failure_reasons"])
        self.assertTrue(row["denominator_retained"])

    def test_empty_pusher_does_not_erase_valid_object_score(self):
        value=image(34);value[:,68:74,68:74]=255
        row=self.measure(value)
        self.assertEqual(row["raw_object_error"],0)
        self.assertEqual(row["pusher_error"],1)
        self.assertTrue(row["hard_generation_failure"])
        self.assertIn("empty_pusher",row["failure_reasons"])
        self.assertIsNotNone(row["legacy_score_error"])
        self.assertFalse(row["legacy_shape_assessable"])

    def test_nonfinite_range_and_inference_failures_are_not_finite_subset_means(self):
        for kind in ("nonfinite","range","status"):
            value=image(34).astype(float)/255
            kwargs={}
            if kind=="nonfinite":value[0,0,0]=np.nan
            elif kind=="range":value[0,0,0]=1.1
            else:kwargs["status"]="inference_error: budget_exhausted"
            row=self.measure(value,**kwargs)
            self.assertEqual(row["raw_object_error"],1)
            self.assertEqual(row["object_foreground_rgb_mae"],1)
            self.assertTrue(row["hard_generation_failure"])
            json.dumps(score.legacy._json_value(row),allow_nan=False)

    def test_fragment_and_hole_counts_remain_visible(self):
        value=image(34);value[:,10:13,10:13]=np.asarray((119,136,153))[:,None,None]
        value[:,36:41,40:45]=255
        row=self.measure(value)
        self.assertEqual(row["predicted_components"],2)
        self.assertEqual(row["predicted_holes"],1)
        self.assertEqual(row["extra_fragments"],1)
        self.assertTrue(row["legacy_shape_deformation_failure"])

    def test_fixed_gray_roi_cannot_expand_with_predicted_blob(self):
        normal=self.measure(image(34))
        blob=image(34);blob[:,5:90,5:90]=np.asarray((119,136,153))[:,None,None]
        damaged=self.measure(blob)
        self.assertEqual(normal["object_roi_pixels"],damaged["object_roi_pixels"])
        self.assertEqual(normal["combined_roi_pixels"],damaged["combined_roi_pixels"])

    def test_pusher_change_does_not_change_gray_metric(self):
        normal=self.measure(image(34))
        value=image(34);value[:,68:74,68:74]=255;value[:,74:80,74:80]=np.asarray((65,105,225))[:,None,None]
        row=self.measure(value)
        self.assertEqual(row["raw_object_error"],normal["raw_object_error"])
        self.assertEqual(row["object_foreground_rgb_mae"],normal["object_foreground_rgb_mae"])
        self.assertGreater(row["pusher_error"],0)

    def test_legacy_exception_does_not_drop_primary_measurement(self):
        with patch.object(score.legacy,"score_rgb",side_effect=ValueError("mock scoring exception")):
            row=self.measure(image(34))
        self.assertEqual(row["raw_object_error"],0)
        self.assertIn("mock scoring exception",row["legacy_score_error"])
        self.assertTrue(row["denominator_retained"])

    def test_border_contact_retains_unknown_shape_not_a_pass(self):
        row=self.measure(image(0))
        self.assertTrue(row["prediction_object_border_touch"])
        self.assertFalse(row["legacy_shape_assessable"])
        self.assertIsNone(row["legacy_shape_deformation_failure"])
        self.assertFalse(row["object_outside_viewport_assessable"])

    def test_fit_target_requires_twenty_percent_and_twelve_wins(self):
        self.assertTrue(score.train_target(dynamic_rows(12))["passed"])
        self.assertFalse(score.train_target(dynamic_rows(11))["passed"])
        low_improvement=dynamic_rows(18)
        for row in low_improvement:row["methods"]["fitted"]["raw_object_error"]=.49
        self.assertFalse(score.train_target(low_improvement)["passed"])

    def test_generation_failure_invalid_hold_or_protocol_prevents_target_pass(self):
        self.assertFalse(score.train_target(dynamic_rows(18,failure=0))["passed"])
        self.assertFalse(score.train_target(dynamic_rows(18,hold_failure=True))["passed"])
        self.assertFalse(score.train_target(dynamic_rows(18),protocol_valid=False)["passed"])

    def test_missing_prediction_file_keeps_all_eighteen_rows(self):
        with tempfile.TemporaryDirectory(dir=score.ROOT/"artifacts",prefix="mock-stage3-score-") as temporary:
            result,_=score.score_cohort(Path(temporary)/"missing.npz",windows(18),"train",self.calibration)
            self.assertEqual(len(result["rows"]),18)
            self.assertEqual(result["methods"]["fitted"]["denominator"],18)
            self.assertEqual(result["methods"]["fitted"]["means_all_rows"]["raw_object_error"],1)
            self.assertFalse(result["dynamic_fit_target"]["passed"])
            self.assertIsNotNone(result["raw_io_error"])

    def test_failed_full_report_keeps_all_ordered_windows_and_nine_gallery_pages(self):
        with tempfile.TemporaryDirectory(dir=score.ROOT/"artifacts",prefix="mock-stage3-score-") as temporary:
            root=Path(temporary)
            result=score.build_report(root/"missing-train.npz",root/"missing-dev.npz",windows(18),windows(36),root/"report")
            self.assertEqual(result["train"]["denominator"],18)
            self.assertEqual(result["development"]["denominator"],36)
            self.assertFalse(result["raw_array_files_retained"])
            for name,n in (("train",18),("development",36)):
                pages=json.loads((root/"report"/f"gallery-{name}"/"pages.json").read_text())
                self.assertEqual([key for page in pages["pages"] for key in page["window_ids"]],[x["window_id"] for x in windows(n)])
                self.assertEqual(len(pages["pages"]),n//6)


if __name__=="__main__":unittest.main()
