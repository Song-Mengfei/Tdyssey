"""Synthetic only: no saved image, model, SDK, environment, or subprocess."""
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy unavailable")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("offline_pixels", ROOT / "scripts/gpc_checkpoint500_offline_pixels.py")
pixels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pixels)


class SelectedWindowTests(unittest.TestCase):
    def test_only_five_selected_image_rows_are_numerically_decoded(self):
        images = np.stack([np.full((3, 96, 96), index, np.uint8) for index in range(20)])
        buffer = io.BytesIO()
        np.savez_compressed(buffer, images=images)
        original = np.frombuffer
        calls = []

        def selected_only(value, *args, **kwargs):
            self.assertEqual(len(value), 5 * 3 * 96 * 96)
            calls.append(len(value))
            return original(value, *args, **kwargs)

        with patch.object(np, "load", side_effect=AssertionError("whole-array loading forbidden")), \
                patch.object(np, "frombuffer", side_effect=selected_only):
            actual, identity = pixels.read_npy_window(buffer.getvalue(), "images.npy", 14, 5, (3, 96, 96), "u")
        np.testing.assert_array_equal(actual, images[14:19])
        self.assertEqual(calls, [5 * 3 * 96 * 96])
        self.assertEqual(identity["unselected_numeric_rows_decoded"], 0)
        self.assertEqual(identity["prefix_transport_bytes"], 14 * 3 * 96 * 96)
        self.assertEqual(identity["selected_sha256"], pixels._sha(images[14:19].tobytes()))

    def test_invalid_window_and_shape_rejected_before_numeric_decode(self):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, images=np.zeros((6, 3, 96, 96), np.uint8))
        with patch.object(np, "frombuffer", side_effect=AssertionError("must validate header first")):
            for start, tail in ((3, (3, 96, 96)), (0, (96, 96, 3))):
                with self.assertRaises(ValueError):
                    pixels.read_npy_window(buffer.getvalue(), "images.npy", start, 5, tail, "u")

    def test_conditioning_and_action_hash_alignment(self):
        history = np.stack([np.full((3, 96, 96), i, np.float32) for i in range(4)])
        future = np.stack([np.full((3, 96, 96), i+4, np.float32) for i in range(4)])
        past = np.asarray([[0, 511], [10, 20], [30, 40]], dtype=np.float64)
        actions = np.asarray([[50, 60], [70, 80], [90, 100], [110, 120]], dtype=np.float64)
        seen = []

        def marker(frame):
            seen.append(float(frame[0, 0, 0]))
            return frame + np.float32(10), {}

        given, result = pixels.conditioning(history, future, past, actions, "AR", 1, marker)
        np.testing.assert_array_equal(given[:, 0, 0, 0], [10, 11, 12, 13])
        expected = np.concatenate((past, actions[:1])).astype(np.float32) / np.float32(511) * np.float32(2)-np.float32(1)
        self.assertEqual(pixels._sha(result.tobytes()), pixels._sha(expected.tobytes()))
        self.assertEqual(seen, [0, 1, 2, 3])
        seen.clear()
        given, result = pixels.conditioning(history, future, past, actions, "TF", 4, marker)
        np.testing.assert_array_equal(given[:, 0, 0, 0], [13, 14, 15, 16])
        self.assertEqual(seen, [3, 4, 5, 6])
        expected = actions.astype(np.float32) / np.float32(511) * np.float32(2)-np.float32(1)
        np.testing.assert_array_equal(result, expected)
        given, _ = pixels.conditioning(history, future, past, actions, "AR", 4, marker)
        self.assertIsNone(given)
        self.assertEqual(seen, [3, 4, 5, 6])

    def test_display_clipping_preserves_raw_values(self):
        raw = np.full((3, 96, 96), .5, np.float32)
        raw[:, 0, 0] = [-.2, .5, 1.2]
        before = raw.copy()
        shown = pixels.display_rgb(raw)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_array_equal(shown[0, 0], [0, 128, 255])
        self.assertLess(float(raw.min()), 0)
        self.assertGreater(float(raw.max()), 1)

    @unittest.skipIf(importlib.util.find_spec("scipy") is None, "fixed helper requires SciPy")
    def test_weak_outside_alpha_is_not_a_second_seed_component(self):
        from tdyssey import wm_image_measurement_v2 as fixed
        alpha = np.zeros((96, 96), dtype=float)
        alpha[10:18, 10:18] = 1
        alpha[60:80, 60:80] = .1
        raw = np.full((3, 96, 96), .5, np.float32)
        before = raw.copy()
        with patch.object(fixed, "continuous_maps", return_value=(alpha, np.zeros_like(alpha), "red")), \
                patch.object(fixed, "fit_object", side_effect=AssertionError("no fit allowed")), \
                patch.object(fixed, "decode", side_effect=AssertionError("no decode allowed")):
            metrics, _ = pixels.pixel_diagnostics(raw, fixed)
        self.assertEqual(metrics["seed_component_count"], 1)
        self.assertEqual(metrics["largest_seed_pixels"], 64)
        self.assertGreater(metrics["alpha_mass_outside_roi"], 39)
        self.assertLess(metrics["largest_component_mass_fraction"], .95)
        self.assertNotIn("pose", metrics)
        np.testing.assert_array_equal(raw, before)

    def test_disconnected_seeds_and_prediction_path(self):
        seed = np.zeros((96, 96), bool)
        seed[1:4, 1:4] = True
        seed[20:22, 20:22] = True
        self.assertEqual(pixels._component_sizes(seed), [9, 4])
        row = dict(raw_path="predictions\\adapted-T12-away-5478-AR-H1.npy",
                   measurement_path="artifacts/run/measurements/adapted-T12-away-5478-AR-H1.json")
        self.assertEqual(pixels._prediction_path(row),
                         "artifacts/run/predictions/adapted-T12-away-5478-AR-H1.npy")

    @unittest.skipIf(importlib.util.find_spec("scipy") is None, "fixed helper requires SciPy")
    def test_complete_nested_record_serializes_roi_without_changing_metrics(self):
        from tdyssey import wm_image_measurement_v2 as fixed
        baseline = None
        baseline_path = os.environ.get("TDYSSEY_OFFLINE_PIXEL_BASELINE")
        if baseline_path:
            old_spec = importlib.util.spec_from_file_location("offline_pixels_baseline", baseline_path)
            baseline = importlib.util.module_from_spec(old_spec)
            old_spec.loader.exec_module(baseline)

        raw = np.full((3, 96, 96), .5, np.float32)
        raw[:, 0, 0] = [-.2, .5, 1.2]
        original_bytes = raw.tobytes()
        cases = (("interior", (slice(20, 25), slice(30, 36)), [16, 29, 26, 40]),
                 ("empty", None, None),
                 ("edge_clamped", (slice(0, 4), slice(92, 96)), [0, 8, 88, 96]))

        def nested_record(metrics):
            return dict(kind="gpc_checkpoint500_fixed_offline_pixels.v1",
                branches=[dict(shape="L", stratum="independent", anchor="synthetic",
                    candidate="away", contact_by_h4=False,
                    predictions=[dict(role="adapted", mode="AR", horizon=1,
                        original_measurement=dict(state=dict(valid=False, pose=None, reason="synthetic"),
                            shape=dict(valid=False), pusher=dict(valid=False),
                            coverage=dict(valid=False, estimate=None), termination=dict(status="unknown")),
                        metrics=metrics, input_hash_check="matched")])],
                train_windows=[], selected_array_reads=[], input_identities=[],
                decoded_counts=dict(prediction_frames=1, physical_frames=0, training_frames=0),
                forbidden_operations=dict(pixels.ZERO_OPERATIONS), outputs_before_report=[])

        for name, region, expected_roi in cases:
            with self.subTest(roi=name):
                alpha = np.zeros((96, 96), dtype=float)
                if region is not None:
                    alpha[region] = 1
                with patch.object(fixed, "continuous_maps", return_value=(alpha, np.zeros_like(alpha), "red")), \
                        patch.object(fixed, "fit_object", side_effect=AssertionError("no optimizing fit")), \
                        patch.object(fixed, "decode", side_effect=AssertionError("no decoder")):
                    metrics, views = pixels.pixel_diagnostics(raw, fixed)
                    self.assertEqual(metrics["roi"], expected_roi)
                    if expected_roi is not None:
                        self.assertTrue(all(type(value) is int for value in metrics["roi"]))
                    record = nested_record(metrics)
                    self.assertEqual(json.loads(pixels._json(record)), record)
                    if baseline is not None:
                        old_metrics, old_views = baseline.pixel_diagnostics(raw, fixed)
                        self.assertEqual({k: v for k, v in metrics.items() if k != "roi"},
                                         {k: v for k, v in old_metrics.items() if k != "roi"})
                        self.assertEqual(metrics["roi"], old_metrics["roi"])
                        for key in ("rgb", "alpha", "blue", "seed"):
                            np.testing.assert_array_equal(views[key], old_views[key])
                        self.assertEqual(views["bounds"], old_views["bounds"])
                        if expected_roi is None:
                            self.assertEqual(json.loads(baseline._json(nested_record(old_metrics))), record)
                        else:
                            self.assertTrue(any(isinstance(value, np.integer) for value in old_metrics["roi"]))
                            with self.assertRaises(TypeError):
                                baseline._json(nested_record(old_metrics))
                self.assertEqual(raw.tobytes(), original_bytes)
                record["decoded_counts"]["nonfinite_sentinel"] = float("nan")
                with self.assertRaises(ValueError):
                    pixels._json(record)
                if baseline is not None:
                    with self.assertRaises(ValueError):
                        baseline._json(dict(nonfinite_sentinel=float("nan")))


if __name__ == "__main__":
    unittest.main()
