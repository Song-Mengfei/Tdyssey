"""Serialization bounds on fake numeric records; no saved RGB or inference."""

from copy import deepcopy
import importlib.util
import json
import unittest

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy required for the injectable export engine imports")

from tdyssey.wm_compat_budget import DEFAULT_LIMITS, canonical_plan
from tdyssey.wm_compat_diagnostics import forecast_diagnostics
from tdyssey.wm_compat_engine import (
    FORECAST_RECORD_BYTES, REAL_RECORD_BYTES, RNG_MAX_BYTES, compact_measurement,
)
from tdyssey.wm_compat_evaluation import accumulate_transform, save_summary
from tdyssey.wm_compat_reporting import measurement_screen, paired_rankings, t_stage_gate
from tdyssey.wm_measurement_contract import DEVELOPMENT_BOUND, coverage_measurement, measurement_frame
from tests.test_wm_compat_reporting import real_rows


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def rich_measurement(estimate=.41234567890123455, *, valid=True):
    """V2-shaped fields, with three alternatives as a conservative size case.

    Current v2 has three survivors and exports two alternatives. This deliberately
    retains three alternatives and long float representations without running it.
    """
    pose = {"position": [255.12345678901235, 255.12345678901235], "angle": -3.1234567890123452}
    reason = "ok" if valid else "rigid_shape_mismatch"
    frame = measurement_frame(
        state={"valid": valid, "pose": deepcopy(pose) if valid else None,
               "candidate_pose": deepcopy(pose), "reason": reason, "ambiguity_assessed": False, "ambiguity": None},
        shape={"valid": valid, "reason": reason, "area_ratio": .9123456789012346,
               "component_mass_fraction": .9876543210987654, "full_soft_iou": .9123456789012346,
               "roi_soft_iou": .9123456789012346, "squared_residual": 123.12345678901235,
               "outside_roi_template_mass": -1.4210854715202004e-14, "palette": "red",
               "alternative_fits": [dict(deepcopy(pose), loss=123.12345678901235) for _ in range(3)],
               "fitted_outline": 1.2345678901234567, "fitted_blur_sigma": .12345678901234566},
        pusher={"valid": valid, "position": deepcopy(pose["position"]) if valid else None,
                "reason": "ok" if valid else "pusher_size_or_shape_mismatch", "soft_iou": .9123456789012346,
                "area_ratio": .9123456789012346, "component_mass_fraction": .9876543210987654},
        coverage=coverage_measurement(estimate, valid=valid, half_width=.03, uncertainty_kind=DEVELOPMENT_BOUND),
    )
    return compact_measurement(frame)


def rich_truth(coverage=.41234567890123455):
    return {"pose": {"position": [255.12345678901235, 255.12345678901235], "angle": -3.1234567890123452},
            "pusher": [255.12345678901235, 255.12345678901235], "coverage": coverage, "iou": .12345678901234566}


def fake_transform_diagnostics():
    """Bounded aggregate-shaped metadata, not output from a transform call."""
    names = ("W--Q", "W--P0", "P0--P1", "W--P1", "W--G0", "G0--G1", "W--G1", "W--V", "V--G0", "V--G1")
    counts = {"%02d:%s" % (i+1, name): 927 if i == 0 else 921 for i, name in enumerate(names)}
    aggregate = {}
    for arm in ("A", "B"):
        diag = dict(arm=arm, pixel_count=9216, changed_pixels=5526 if arm == "B" else 0,
                    accepted_pixels=9216 if arm == "B" else None, unmatched_pixels=0 if arm == "B" else None,
                    converted_pixels=5526 if arm == "B" else 0, identity_preserved_pixels=3690 if arm == "B" else 9216,
                    unchanged_pixels=3690 if arm == "B" else 9216,
                    input_range=[0., 1.], output_range=[0., 1.], output_finite=True, output_in_range=True,
                    input_mutated=False, winning_segment_counts=counts if arm == "B" else {},
                    accepted_segment_counts=counts if arm == "B" else {},
                    residual_rms_rgb_255=dict(min=1.2345678901234567, max=7.123456789012345,
                        mean=3.1234567890123457, p50=3.1234567890123457, p90=6.123456789012345,
                        p99=7.123456789012345) if arm == "B" else None,
                    residual_by_winning_segment={key: dict(winner_pixels=count, accepted_pixels=count,
                        mean_rms_rgb_255=3.1234567890123457, max_rms_rgb_255=7.123456789012345)
                        for key, count in counts.items()} if arm == "B" else {})
        for _ in range(528):
            accumulate_transform(aggregate, diag)
    return aggregate


def fake_result(*, m_failure):
    """Complete planned table and independent physical labels, all hand-built."""
    real = real_rows()
    for row in real:
        if row["status"] == "completed":
            row["measurement"] = rich_measurement(valid=not m_failure)
            row["truth"] = rich_truth()
    forecasts = []
    candidates = ("hold", "toward", "side", "away")
    for planned in canonical_plan()["calls"]:
        missing = planned["status"] == "missing"
        row = dict(planned, shape=planned["phase"],
                   status="missing" if missing else "not_attempted" if m_failure else "completed")
        actual = .11234567890123457+.1*candidates.index(row["candidate"])+row["horizon"]*.00345678901234567
        prediction = actual+(row["seed"]-5479)*.01234567890123456+(.00345678901234567 if row["arm"] == "B" else 0)
        row.update(measurement=None if missing or m_failure else rich_measurement(prediction),
                   truth=None if missing else rich_truth(actual), contact_by_h4=row["candidate"] == "toward")
        forecasts.append(row)
    ar = [row for row in forecasts if row["mode"] == "AR"]
    physical = []
    for branch in canonical_plan()["branches"]:
        if branch["status"] != "missing":
            truth = [next(r["truth"] for r in ar if r["anchor"] == branch["anchor"] and r["candidate"] == branch["candidate"]
                          and r["horizon"] == h) for h in (1, 2, 3, 4)]
            physical.append(dict(anchor=branch["anchor"], candidate=branch["candidate"], latest=rich_truth(),
                                 truth=truth, contact_by_h4=branch["candidate"] == "toward"))
    counts = {"images": 0 if m_failure else 2268, "euler_steps": 0 if m_failure else 6804,
              "model_loads": 0 if m_failure else 1, "T_images": 0 if m_failure else 576,
              "L_images": 0 if m_failure else 1692}
    return dict(kind="tdyssey.wm_input_compatibility_engineering.v1", development_only=True,
                independent_validation=False, rows=forecasts, real_rows=real, M=measurement_screen(real),
                T=None if m_failure else t_stage_gate(forecasts), ranking=paired_rankings(ar),
                diagnostics=forecast_diagnostics(forecasts, physical),
                transform_diagnostics=fake_transform_diagnostics(),
                budget=dict(status="stopped" if m_failure else "active", counts=counts,
                            stage_status={"M": "failed" if m_failure else "passed",
                                          "T": "not_attempted" if m_failure else "passed", "L": "not_attempted" if m_failure else "active"},
                            stop_reason="M_gate_failed" if m_failure else None,
                            started_monotonic=1234567.123456789, deadline_monotonic=1236367.123456789),
                real_operations=0, policy_sampling=0, optimizer_updates=0)


class MemoryBudget:
    def __init__(self, state):
        self.state = deepcopy(state)
        self.finished = False

    def finish(self, evidence):
        self.finished = True
        self.state.update(status="complete", stop_reason=None)
        self.state["stage_status"]["L"] = "complete"

    def snapshot(self):
        return deepcopy(self.state)


class MemorySink:
    def __init__(self):
        self.records = {}

    def record(self, name, record, *, maximum=FORECAST_RECORD_BYTES, final=False):
        if name in self.records:
            raise AssertionError("An immutable record was written twice")
        payload = encoded(record)
        if len(payload) > maximum:
            raise AssertionError("%s: %d bytes exceeds %d" % (name, len(payload), maximum))
        if maximum > DEFAULT_LIMITS["chunk_bytes"]:
            raise AssertionError("A declared chunk exceeds the fixed chunk cap")
        self.records[name] = {"payload": payload, "maximum": maximum, "final": final}


class CompatibilityExportChecks(unittest.TestCase):
    def test_split_exports_fit_full_valid_and_M_failure_with_full_denominators(self):
        for m_failure in (False, True):
            with self.subTest(m_failure=m_failure):
                result = fake_result(m_failure=m_failure)
                sink, budget = MemorySink(), MemoryBudget(result["budget"])
                save_summary(result, sink, budget)
                limits = {"M.json": 512*1024, "ranking.json": 1024*1024, "summary.json": 64*1024,
                          "forecast-diagnostics.json": 192*1024, "planned-status.json": 256*1024, "completion.json": 8*1024}
                self.assertEqual(set(sink.records), set(limits))
                for name, maximum in limits.items():
                    self.assertEqual(sink.records[name]["maximum"], maximum)
                    self.assertLessEqual(len(sink.records[name]["payload"]), maximum)
                    self.assertTrue(sink.records[name]["final"])
                measurement = json.loads(sink.records["M.json"]["payload"])
                self.assertNotIn("rows", measurement)
                self.assertEqual(measurement["available_total"], 1056)
                self.assertEqual(measurement["passed"], not m_failure)
                status = json.loads(sink.records["planned-status.json"]["payload"])
                self.assertEqual(len(status["rows"]), 2304)
                self.assertEqual(sum(s == "missing" for _, s in status["rows"]), 36)
                summary = json.loads(sink.records["summary.json"]["payload"])
                self.assertEqual(summary["expected_forecast_rows"], 2304)
                self.assertEqual(summary["completed_forecast_rows"], 0 if m_failure else 2268)
                self.assertIn("M.json", str(summary["M"]))
                self.assertIn("ranking.json", str(summary["ranking"]))
                self.assertIn("forecast-diagnostics.json", str(summary["diagnostics"]))
                self.assertNotIn("rows", summary)
                completion = json.loads(sink.records["completion.json"]["payload"])
                self.assertEqual(completion["status"], "stopped" if m_failure else "complete")
                self.assertEqual(budget.finished, not m_failure)

    def test_rich_real_and_forecast_records_fit_predeclared_bounds(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                measurement = rich_measurement(valid=valid)
                truth = rich_truth()
                real = dict(source_key="L00-toward-future-H4", source="physical_future", anchor="L00", candidate="toward",
                            shape="L", horizon=4, truth=truth, key="L00-toward-future-H4/A", arm="A", mode="REAL",
                            status="completed", measurement=measurement,
                            transform=dict(input_sha256="a"*64, output_sha256="b"*64, changed_pixels=9216, unmatched_pixels=9216))
                forecast = dict(key="L00/toward/5478/TF/A/H4", anchor="L00", candidate="toward", shape="L",
                                seed=5478, arm="A", mode="TF", horizon=4, status="completed",
                                source_history_sha256="a"*64, source_context_sha256="b"*64,
                                conditioning_source="saved_real_window", rng_entry="c"*64, rng_exit="d"*64,
                                input_sha256="e"*64, action_sha256="f"*64, raw_sha256="0"*64,
                                raw_path="predictions/L00-toward-5478-TF-A-H4.npy", raw_min=-1.2345678901234567,
                                raw_max=1.2345678901234567, sampler_seconds=.12345678901234566,
                                clipped_fraction=.12345678901234566, measurement=measurement, truth=truth, contact_by_h4=False,
                                appearance=dict(forecast_rgb_mse=.12345678901234566, persistence_rgb_mse=.23456789012345678))
                sink = MemorySink()
                sink.record("fake-real.json", real, maximum=REAL_RECORD_BYTES)
                sink.record("fake-forecast.json", forecast, maximum=FORECAST_RECORD_BYTES)
                self.assertEqual(len(measurement["shape"]["alternative_fits"]), 3)

    def test_total_declared_metadata_and_npy_header_allocations_fit(self):
        # Reservations, not compression or current payload lengths, determine
        # this bound. Missing rows are metadata-only entries in the status table.
        forecast_records = 2268*FORECAST_RECORD_BYTES
        real_records = 1072*REAL_RECORD_BYTES
        rng_records = 63*3*4*RNG_MAX_BYTES
        # A separate 128KiB reservation covers the fixed case plates/manifest.
        final_exports = (512+1024+64+256+8+192+128)*1024
        binding = 128*1024
        total = forecast_records+real_records+rng_records+final_exports+binding
        self.assertLessEqual(total, DEFAULT_LIMITS["metadata_bytes"])
        self.assertLessEqual(2268*256, DEFAULT_LIMITS["header_bytes"])
        # The parent additionally charges its one unknown destination/temp slot.
        self.assertLessEqual(sum(DEFAULT_LIMITS[name] for name in (
            "raw_bytes", "metadata_bytes", "header_bytes", "temporary_bytes"))+2*1024**2,
            DEFAULT_LIMITS["result_bytes"])


if __name__ == "__main__":
    unittest.main()
