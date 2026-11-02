"""Synthetic summaries and saved-array previews; no physical/model execution."""

from copy import deepcopy
import json
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from tdyssey.frozen_wm_protocol import CANDIDATES, NOISE_SEEDS, anchors
from tdyssey.frozen_wm_summary import summarize, make_previews

ROOT = Path(__file__).resolve().parents[1]


def fake_report():
    report = {"calibration_gate": {shape: {"passed": True} for shape in ("L", "T")}, "branches": [], "calibrations": []}
    for spec in anchors():
        for index, name in enumerate(CANDIDATES):
            coverage = [.1, .6, .3, .05][index]
            latest = {"pose": {"position": [200., 200.], "angle": 0.}, "coverage": .1, "pusher": [100., 100.]}
            truth = {"pose": {"position": [210., 200.], "angle": math.radians(10)}, "coverage": coverage, "pusher": [120., 100.]}
            row = {"anchor": spec["id"], "shape": spec["shape"], "family": spec["family"], "candidate": name,
                   "status": "complete", "latest": latest, "actual": [], "predictions": []}
            for h in range(1, 5):
                row["actual"].append({"horizon": h, "truth": deepcopy(truth), "contacts": 1, "absorbing": False})
            for seed in NOISE_SEEDS:
                prediction = {"seed": seed, "steps": []}
                for h in range(1, 5):
                    prediction["steps"].append({"horizon": h, "decoded": {"valid": True, "reason": "ok", **deepcopy(truth)},
                        "score_coverage": coverage, "predicted_absorbing": False,
                        "motion": {"valid": True, "position_error": 0., "angle_error": 0., "pusher_error": 0.,
                            "persistence_position_error": 10., "persistence_angle_error": 10., "rotation_direction_correct": True},
                        "rgb": {"model": {"image_mse": .01, "object_mask_iou": .8, "centroid_error_pixels": 1.},
                                "persistence": {"image_mse": .02, "object_mask_iou": .6, "centroid_error_pixels": 3.}}})
                row["predictions"].append(prediction)
            report["branches"].append(row)
    return report


class SummaryChecks(unittest.TestCase):
    def test_persistence_state_reference_remains_available_when_model_measurement_fails(self):
        report = fake_report()
        for row in report["branches"]:
            for prediction in row["predictions"]:
                prediction["steps"][-1]["motion"]["valid"] = False
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        contact = h4["motion"]["contact"]
        self.assertEqual(contact["position_error"]["valid_branches"], 0)
        self.assertEqual(contact["persistence_position_error"]["valid_branches"], 48)
        self.assertEqual(contact["persistence_position_error"]["anchor_equal_mean"], 10)

    def test_unknown_physical_row_cannot_mask_known_prediction_failure_as_sample_shortage(self):
        report = fake_report()
        report["branches"] = [r for r in report["branches"] if not (r["anchor"] == "L01" and r["candidate"] == "hold")]
        for prediction in report["branches"][0]["predictions"]:
            prediction["steps"][-1]["motion"]["valid"] = False
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        self.assertGreaterEqual(h4["motion"]["contact"]["anchors"], 6)
        self.assertEqual(h4["gates"]["contact_motion"]["status"], "fail")
        self.assertEqual(h4["gates"]["rotation"]["status"], "fail")
        self.assertEqual(h4["gates"]["coverage"]["status"], "coverage insufficient")

    def test_perfect_scores_and_motion_pass_with_fixed_physical_denominators(self):
        report = fake_report()
        result = summarize(report)
        json.dumps(result, allow_nan=False)
        for shape, count in (("L", 48), ("T", 16)):
            h4 = result["by_shape"][shape]["horizons"]["H4"]
            self.assertEqual(h4["expected_branches"], count)
            self.assertEqual(h4["prediction_expected_noises"], count*3)
            self.assertEqual(h4["coverage"]["observed"]["anchor_equal_mean"], 0)
            self.assertEqual(h4["ranking"]["anchor_equal_pair_accuracy"], 1)
            self.assertEqual(h4["ranking"]["mean_selected_endpoint_regret_conservative"], 0)
            for name in ("coverage", "contact_motion", "rotation", "ranking"):
                self.assertEqual(h4["gates"][name]["status"], "pass")
            self.assertEqual(h4["gates"]["contact_probability"]["status"], "unavailable")

    def test_missing_noise_invalidates_candidate_and_stays_in_pair_and_regret_denominators(self):
        report = fake_report()
        row = next(r for r in report["branches"] if r["anchor"] == "L00" and r["candidate"] == "toward")
        row["predictions"].pop()
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        rank = next(r for r in h4["ranking"]["anchors"] if r["anchor"] == "L00")
        self.assertIsNone(rank["predicted"][1])
        self.assertTrue(rank["selection_abstention"])
        self.assertEqual(rank["regret_conservative"], 1)
        self.assertEqual(rank["eligible_pairs"], 6)
        self.assertEqual(rank["correct_pairs"], 3)
        self.assertEqual(h4["ranking"]["selection_abstentions"], 1)
        self.assertAlmostEqual(h4["ranking"]["mean_selected_endpoint_regret_conservative"], 1/12)
        self.assertEqual(h4["coverage"]["observed"]["missing_branches"], 1)
        self.assertEqual(h4["gates"]["coverage"]["status"], "fail")

    def test_failed_calibration_blocks_pose_coverage_and_rank_even_with_truth_like_decodes(self):
        report = fake_report()
        report["calibration_gate"]["L"]["passed"] = False
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        self.assertIsNone(h4["coverage"]["observed"]["anchor_equal_mean"])
        self.assertEqual(h4["coverage"]["observed"]["missing_branches"], 48)
        self.assertEqual(h4["ranking"]["selection_abstentions"], 12)
        self.assertIsNone(h4["ranking"]["anchor_equal_pair_accuracy"])
        self.assertIsNone(h4["ranking"]["mean_selected_endpoint_regret_conservative"])
        self.assertEqual(h4["gates"]["ranking"]["status"], "unavailable")
        self.assertEqual(h4["rgb"]["model_image_mse"]["anchor_equal_mean"], .01)

    def test_score_absorption_controls_ranking_and_real_absorption_excludes_only_motion(self):
        report = fake_report()
        row = report["branches"][0]
        row["actual"][-1]["absorbing"] = True
        for prediction in row["predictions"]:
            step = prediction["steps"][-1]
            step["score_coverage"], step["predicted_absorbing"] = .99, True
            step["motion"] = None
            step["decoded"]["coverage"] = .01
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        branch = h4["branch_records"][0]
        self.assertAlmostEqual(branch["score"], .99)
        self.assertAlmostEqual(branch["coverage_error"], .89)
        self.assertEqual(h4["physical_absorbing_branches"], 1)
        self.assertEqual(h4["predicted_absorbing_noise_rows"], 3)
        self.assertEqual(h4["motion"]["contact"]["branches"], 47)
        self.assertEqual(h4["motion"]["contact"]["position_error"]["missing_branches"], 0)

    def test_noise_mae_does_not_cancel_and_anchor_weights_do_not_depend_on_branch_count(self):
        report = fake_report()
        row = report["branches"][0]
        for prediction, value in zip(row["predictions"], (.0, .1, .2)):
            prediction["steps"][-1]["score_coverage"] = value
        h4 = summarize(report)["by_shape"]["L"]["horizons"]["H4"]
        self.assertAlmostEqual(h4["branch_records"][0]["coverage_error"], 2*.1/3)
        self.assertAlmostEqual(h4["branch_records"][0]["mean_score_coverage_error"], 0)
        self.assertAlmostEqual(h4["coverage"]["observed"]["anchor_equal_mean"], (2*.1/3)/48)
        self.assertAlmostEqual(h4["branch_records"][0]["score"], .1)

    def test_missing_physics_stays_in_fixed_plan_and_reports_insufficient_coverage(self):
        report = fake_report()
        report["branches"] = [r for r in report["branches"] if r["anchor"] == "L00"]
        result = summarize(report)
        h4 = result["by_shape"]["L"]["horizons"]["H4"]
        self.assertEqual(h4["expected_branches"], 48)
        self.assertEqual(h4["missing_actual_branches"], 44)
        self.assertEqual(h4["ranking"]["unknown_actual_pairs"], 66)
        self.assertEqual(h4["gates"]["ranking"]["status"], "coverage insufficient")
        self.assertEqual(h4["gates"]["contact_motion"]["status"], "coverage insufficient")

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "installed image dependencies required")
    def test_saved_fake_arrays_generate_reviewable_pngs_without_decoder_or_physics(self):
        import numpy as np
        from PIL import Image
        report = fake_report()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            pixels = np.zeros((4, 3, 96, 96), dtype=np.float32)
            np.savez_compressed(directory / "L00-hold-physical.npz", history=pixels, actual=pixels)
            np.savez_compressed(directory / "L00-hold-noise-5478.npz", raw=pixels+.25)
            np.savez_compressed(directory / "L00-away-physical.npz", history=pixels[:1], actual=pixels[:0])
            report["calibrations"] = [{"id": "cal-L-00", "shape_name": "L", "truth": report["branches"][0]["latest"],
                "prediction": {"valid": False, "reason": "synthetic_occlusion", "pose": None, "coverage": None}}]
            np.savez_compressed(directory / "cal-L-00.npz", rgb=pixels[0])
            paths = make_previews(report, directory)
            self.assertEqual(len(paths), 19)
            for path in paths:
                with Image.open(directory / path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreaterEqual(image.width, 800)


if __name__ == "__main__":
    unittest.main()
