"""CPU checks for RGB feedback, sealed splits, metrics and failure denominators."""

import tempfile
import hashlib
import json
from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch

if any(find_spec(name) is None for name in ('numpy', 'scipy', 'PIL', 'shapely')):
    raise unittest.SkipTest('Install the numerical profile for RGB evaluation tests')
import numpy as np

from tdyssey.native_rgb_evaluation import (
    TRAIN_LETTERS, action_response_probe, align_masks_for_score, calibrate_renderer,
    evaluate_windows, palette_masks, rollout_rgb, score_rgb, select_windows,
    evaluate, final_protocol_identity, validate_calibration,
)


def image_at(x=32, y=30, *, fragment=False, blue=(12, 12)):
    image = np.ones((3, 96, 96), dtype=np.float32)
    mask = np.zeros((96, 96), dtype=bool)
    mask[y:y+25, x:x+6] = True
    mask[y+19:y+25, x:x+20] = True
    if fragment:
        mask[68:74, 65:71] = True
    image[:, mask] = np.asarray([119, 136, 153], dtype=np.float32)[:, None]/255
    bx, by = blue
    image[:, by:by+5, bx:bx+5] = np.asarray([65, 105, 225], dtype=np.float32)[:, None, None]/255
    return image


def calibration():
    return dict(object_rgb=[119/255, 136/255, 153/255],
        palette=dict(minimum_alpha=.4, maximum_alpha=1.25, colour_residual=.12),
        minimum_component_pixels=2, angle_grid_degrees=30, sha256="test-fixed",
        contour_tolerance_px=3., missing_extra_fraction_tolerance=.04, aligned_iou_floor=.95,
        per_letter={letter: dict(max_area_fraction_tolerance=.03, components_min=1,
                               components_max=1, holes_min=0, holes_max=0)
                    for letter in TRAIN_LETTERS})


class FakeDataset:
    def __init__(self, *, split="seen_val", invalid=False):
        self.split = split
        self.entries = [dict(shape="D" if split == "unseen_test" else "L", split=split,
                             controls=10, file="trajectory.npz", trajectory_id="episode-0")]
        self.windows = [(0, 3), (0, 4)]
        self.reads = 0
        states = np.zeros((11, 15))
        states[:, 0:2] = (50, 50)
        states[:, 2] = 180+np.arange(11)*5
        states[:, 3] = 200
        self.arrays = dict(states=states, actions=np.arange(20).reshape(10, 2).astype(np.float32),
                           contact=np.ones(10), valid_transition=np.ones(10, dtype=bool))
        if invalid:
            self.arrays["valid_transition"][5] = False

    def trajectory(self, index):
        self.reads += 1
        return self.entries[index], self.arrays

    def render_state(self, entry, state):
        return image_at(x=round(state[2]/5), y=30)


class NativeRGBEvaluationTests(unittest.TestCase):
    def test_generated_rgb_is_fed_back_exactly_and_action_windows_align(self):
        history = np.stack([image_at(x=20+i) for i in range(4)])
        actions = np.arange(14, dtype=np.float32).reshape(7, 2)
        seen = []
        def predictor(images, controls, seed):
            seen.append((images.copy(), controls.copy(), seed))
            result = images[-1].copy()
            result[0, 0, 0] = .123 + len(seen)*.01
            return result
        predictions = list(rollout_rgb(predictor, history, actions, horizon=4, seed=40))
        np.testing.assert_array_equal(seen[0][0], history)
        for step in range(4):
            np.testing.assert_array_equal(seen[step][1], actions[step:step+4])
            if step:
                np.testing.assert_array_equal(seen[step][0][-1], predictions[step-1])
        self.assertAlmostEqual(float(predictions[-1][0, 0, 0]), .163, places=6)
        np.testing.assert_array_equal(history[-1], image_at(x=23))

    def test_unseen_guard_runs_before_any_trajectory_read(self):
        dataset = FakeDataset(split="unseen_test")
        with self.assertRaisesRegex(ValueError, "prohibited"):
            select_windows(dataset)
        self.assertEqual(dataset.reads, 0)

    def test_nan_failure_and_fatal_gpu_failure_keep_every_denominator(self):
        calls = [0]
        def broken(history, actions, seed):
            calls[0] += 1
            raise RuntimeError("CUDA out of memory")
        report = evaluate_windows(FakeDataset(), {"fatal": broken, "nan": lambda *a: np.full((3,96,96), np.nan)},
                                  calibration=calibration(), horizons=(1,4), windows_per_letter=2, gallery_limit=0)
        self.assertEqual(calls[0], 1)
        self.assertEqual(len(report["rows"]), 2*3*2)
        for method in ("fatal", "nan"):
            for horizon in ("1", "4"):
                stats = report["summary"][method][horizon]["overall"]
                self.assertEqual((stats["total"], stats["failed"]), (2, 2))
                self.assertFalse(report["qualification"][method][horizon]["motion_passed"])
        self.assertEqual(len({(r["window"], r["method"], r["horizon"]) for r in report["rows"]}), 12)

    def test_invalid_truth_is_not_a_successful_prediction_or_removed_row(self):
        report = evaluate_windows(FakeDataset(invalid=True), {"copy": lambda h, a, s: h[-1]},
                                  calibration=calibration(), horizons=(1,4), windows_per_letter=2, gallery_limit=0)
        rows = [r for r in report["rows"] if r["horizon"] == 4]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["status"] == "failed" and r["reason"] == "invalid_truth_transition" for r in rows))

    def test_offline_alignment_separates_translation_from_shape_damage(self):
        predicted, truth = image_at(x=39), image_at(x=32)
        first = predicted.copy()
        score = score_rgb(predicted, truth, calibration(), letter="L")
        self.assertGreater(score["object_position_error_px"], 35)
        self.assertEqual(score["aligned_mask_iou"], 1.)
        self.assertFalse(score["shape_deformation_failure"])
        np.testing.assert_array_equal(predicted, first)
        damaged = score_rgb(image_at(fragment=True), truth, calibration(), letter="L")
        self.assertEqual(damaged["extra_fragments"], 1)
        self.assertTrue(damaged["shape_deformation_failure"])

    def test_occlusion_and_clipping_are_ambiguous_not_topology_success(self):
        score = score_rgb(image_at(blue=(37, 45)), image_at(blue=(37,45)), calibration(), letter="L")
        self.assertTrue(score["occlusion_ambiguous"])
        self.assertIsNone(score["shape_deformation_failure"])
        edge = image_at(x=0)
        score = score_rgb(edge, edge, calibration(), letter="L")
        self.assertTrue(score["viewport_clipped"])
        self.assertFalse(score["shape_measurement_available"])

    def test_empty_truth_and_missing_object_are_explicit_failures(self):
        with self.assertRaisesRegex(ValueError, "truth"):
            score_rgb(image_at(), np.ones((3,96,96)), calibration(), letter="L")
        with self.assertRaisesRegex(ValueError, "Prediction"):
            score_rgb(np.ones((3,96,96)), image_at(), calibration(), letter="L")

    def test_topology_preserves_holes_and_single_pixel_fragments(self):
        from tdyssey.native_rgb_evaluation import _topology
        mask = np.zeros((96,96), dtype=bool)
        mask[30:50,30:50] = True
        mask[35:45,35:45] = False
        mask[80,80] = True
        self.assertEqual(_topology(mask), dict(components=1, holes=1, euler=0, tiny_fragment_pixels=1))

    def test_calibration_uses_renderer_only_and_is_frozen_deterministically(self):
        class Renderer:
            entries = [dict(shape="L")]
            calls = 0
            def render_state(self, entry, state):
                self.calls += 1
                return image_at(x=35+round(state[2]-256))
        renderer = Renderer()
        first = calibrate_renderer(renderer, letters=("L",), angles=(0,))
        second = calibrate_renderer(renderer, letters=("L",), angles=(0,))
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertFalse(first["model_outputs_used"])
        self.assertEqual(first["per_letter"]["L"]["components_max"], 1)
        with self.assertRaisesRegex(ValueError, "18 seen"):
            calibrate_renderer(renderer, letters=("D",))

    def test_output_is_bounded_and_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_windows(FakeDataset(), {"copy": lambda h,a,s: h[-1]},
                    calibration=calibration(), horizons=(1,), windows_per_letter=2,
                    output=directory, gallery_limit=1)
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 1)
            self.assertEqual(len(report["gallery"]), 1)
            with self.assertRaises(FileExistsError):
                evaluate_windows(FakeDataset(), {}, calibration=calibration(), horizons=(1,), output=directory)

    def test_static_action_probe_is_not_presented_as_physical_success(self):
        probe = action_response_probe(lambda h,a,s: h[-1], image_at(), (50,50), calibration())
        self.assertFalse(probe["physical_contact_success_test"])
        self.assertFalse(probe["observed_action_response"])

    def test_true_batched_path_uses_one_callback_per_step_and_same_model_noise(self):
        class Batched:
            def __init__(self):
                self.calls = []
            def __call__(self, *args):
                raise AssertionError("Batched run must not call a sequential sampler")
            def batch_predict(self, histories, actions, *, batch_seed):
                self.calls.append((histories.copy(), actions.copy(), batch_seed))
                outputs = histories[:, -1].copy()
                outputs[:, 0, 0, 0] = .321
                return outputs
        first, second = Batched(), Batched()
        report = evaluate_windows(FakeDataset(), {"frozen": first, "tuned": second},
            calibration=calibration(), horizons=(1,4), windows_per_letter=2,
            gallery_limit=0, batch_size=2)
        self.assertEqual(report["inference_batch_size"], 2)
        self.assertEqual(len(first.calls), 4)
        self.assertEqual(len(second.calls), 4)
        self.assertEqual(len(report["rows"]), 12)
        for step, (one, two) in enumerate(zip(first.calls, second.calls)):
            self.assertEqual(one[0].shape, (2,4,3,96,96))
            self.assertEqual(one[2], two[2])
            if step:
                self.assertTrue((one[0][:,-1,0,0,0] == np.float32(.321)).all())

    def test_batched_fatal_is_not_retried_in_later_chunks(self):
        class Broken:
            calls = 0
            def batch_predict(self, *args, **kwargs):
                self.calls += 1
                raise MemoryError("out of memory")
        dataset = FakeDataset()
        dataset.windows = [(0,3),(0,4),(0,5)]
        predictor = Broken()
        report = evaluate_windows(dataset, {"broken": predictor}, calibration=calibration(),
            horizons=(1,4), windows_per_letter=3, gallery_limit=0, batch_size=2)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(len(report["rows"]), 12)
        self.assertEqual(report["summary"]["broken"]["4"]["overall"]["failed"], 3)

    def test_final_identity_binds_checkpoint_data_config_source_and_calibration(self):
        from tdyssey.native_rgb_evaluation import _renderer_source_identity
        from tdyssey.gpc_world_model import world_model_spec
        value = calibration()
        value.update(letters=list(TRAIN_LETTERS), model_outputs_used=False,
                     renderer_source_sha256=_renderer_source_identity())
        value.pop("sha256")
        value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.assertEqual(validate_calibration(value), value)
        with patch("tdyssey.native_rgb_evaluation._renderer_source_identity", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "renderer"):
                validate_calibration(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"manifest.json").write_text("{}", encoding="utf-8")
            (root/"physics-audit.json").write_text("{}", encoding="utf-8")
            (root/"model.pt").write_bytes(b"mock trusted checkpoint")
            manifest_sha = hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()
            payload = dict(identity=dict(initialization=world_model_spec()["checkpoint"],
                                          data=dict(manifest_sha256=manifest_sha)))
            config = dict(server=True, local_pilot=False, evaluation=dict(batch_size=2))
            with patch("tdyssey.native_rgb_training.load_checkpoint_payload", return_value=payload):
                frozen = final_protocol_identity(config, root/"model.pt", root, value)
                self.assertEqual(frozen["data_manifest_sha256"], manifest_sha)
                self.assertIn("evaluation_source_sha256", frozen)
                changed = dict(frozen, config_sha256="modified")
                with self.assertRaisesRegex(ValueError, "differs"):
                    evaluate(config, root/"eval", root, source_root=root, init_checkpoint=root/"original.pt",
                        checkpoint=root/"model.pt", split="seen_test", allow_unseen=True,
                        calibration=value, frozen_identity=changed)

    def test_final_split_requires_frozen_identity_before_loading_any_dataset(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            evaluate(dict(server=True), None, "does-not-exist", source_root="none",
                     init_checkpoint="none", split="unseen_test", allow_unseen=True)


if __name__ == "__main__":
    unittest.main()
