"""Record-only fake diagnostics: no NumPy, images, transforms, or inference."""

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from tdyssey.wm_compat_diagnostics import MAX_SERIALIZED_BYTES, REASONS, forecast_diagnostics
from tests.test_wm_compat_reporting import ANCHORS, ARMS, CANDIDATES, SEEDS, ar_rows, fake_measurement, fake_truth


def cell(report, **selectors):
    return next(value for value in report["cells"] if all(value[key] == expected for key, expected in selectors.items()))


def counts(report, value):
    return dict(zip(report["columns"]["counts"], value["counts"]))


def metric(report, value, name):
    row = next(row for row in value["metrics"] if row[0] == name)
    return dict(zip(report["columns"]["metrics"], row))


def full_fixture():
    """All planned AR/TF rows, diverse fixed labels and long numeric fields."""
    branches, metadata = [], {}
    for anchor in ANCHORS:
        for i, candidate in enumerate(CANDIDATES):
            if anchor == "L01" and candidate == "hold":
                continue
            latest = dict(pose=dict(position=[100., 100.], angle=0.), pusher=[50., 50.], coverage=.4)
            truths = []
            for horizon in (1, 2, 3, 4):
                truths.append(dict(pose=dict(position=[100.+horizon, 100.+horizon], angle=.21234567890123456 if i % 2 else .012345678901234567),
                                   pusher=[50.+horizon, 50.+horizon], coverage=.9412345678901234 if i < 2 else .41234567890123455))
            branch = dict(anchor=anchor, candidate=candidate, latest=latest, truth=truths, contact_by_h4=i % 2 == 0)
            branches.append(branch)
            metadata[(anchor, candidate)] = branch
    rows = ar_rows()
    for row in rows:
        branch = metadata.get((row["anchor"], row["candidate"]))
        if branch is None:
            row["contact_by_h4"] = None
            continue
        actual = deepcopy(branch["truth"][row["horizon"]-1])
        row["truth"] = actual
        row["contact_by_h4"] = branch["contact_by_h4"]
        error = 4.123456789012345 if row["arm"] == "A" else 2.1234567890123457
        noise = (row["seed"]-5479)*.01234567890123456
        row["measurement"] = fake_measurement(actual["coverage"]+.01234567890123456)
        row["measurement"]["state"]["pose"] = dict(position=[actual["pose"]["position"][0]+error+noise, actual["pose"]["position"][1]],
            angle=actual["pose"]["angle"]+(.04123456789012346 if row["arm"] == "A" else .02123456789012346))
        row["measurement"]["pusher"]["position"] = [actual["pusher"][0]+error+noise, actual["pusher"][1]]
        row["measurement"]["shape"].update(fitted_outline=1.2345678901234567, fitted_blur_sigma=.12345678901234566,
            full_soft_iou=.9123456789012346, area_ratio=.9123456789012346,
            component_mass_fraction=.9876543210987654, squared_residual=123.12345678901235)
        row.update(raw_min=-.12345678901234566, raw_max=1.1234567890123457, clipped_fraction=.01234567890123456,
                   appearance=dict(forecast_rgb_mse=.01234567890123456, persistence_rgb_mse=.02345678901234567))
    tf = []
    for row in rows:
        if row["horizon"] in (2, 4):
            copy = deepcopy(row)
            copy["mode"] = "TF"
            copy["key"] = row["key"].replace("/AR/", "/TF/")
            tf.append(copy)
    return rows+tf, branches


def mixed_reason_fixture():
    rows, branches = full_fixture()
    reasons = sorted(REASONS)
    for mode in ("AR", "TF"):
        for arm in ARMS:
            for shape in ("L", "T"):
                for horizon in ((1, 2, 4) if mode == "AR" else (2, 4)):
                    selected = [r for r in rows if r["mode"] == mode and r["arm"] == arm and r["shape"] == shape
                                and r["horizon"] == horizon and r["status"] == "completed"]
                    for index, row in enumerate(selected[:40]):
                        part = ("state", "shape", "pusher", "coverage")[index % 4]
                        record = row["measurement"][part]
                        record.update(valid=False, reason=reasons[index//4])
                        if part == "state":
                            record["pose"] = None
                        elif part == "pusher":
                            record["position"] = None
    return rows, branches


class CompatibilityDiagnosticsChecks(unittest.TestCase):
    def test_full_denominators_modes_strata_and_conditional_screens(self):
        rows, branches = full_fixture()
        original = deepcopy(rows)
        # Fail loudly if a record-only diagnostic enters any stopping API.
        with patch("tdyssey.wm_compat_reporting.sequence_endpoint", side_effect=AssertionError("no stopping")):
            report = forecast_diagnostics(rows, branches)
        self.assertEqual(len(report["cells"]), 50)
        self.assertFalse(any(c["mode"] == "TF" and c["horizon"] == 1 for c in report["cells"]))
        l = cell(report, mode="AR", shape="L", horizon=1, stratum="all")
        t = cell(report, mode="TF", shape="T", horizon=4, stratum="all")
        self.assertEqual((counts(report, l)["planned_per_arm"], counts(report, l)["available_per_arm"]), (144, 141))
        self.assertEqual((counts(report, t)["planned_per_arm"], counts(report, t)["available_per_arm"]), (48, 48))
        self.assertEqual(l["paired_validity_B_win_tie_loss"]["object_shape"], [0, 141, 0])
        self.assertAlmostEqual(metric(report, l, "position")["common_fraction_of_available"], 1.)
        self.assertTrue(all(screen["conditional_limited_screen"]["passed"] for screen in report["B_AR_screens"]))
        self.assertEqual(rows, original)
        self.assertEqual(report["definitions"]["strata"]["near_095"]["lower"], .92)
        self.assertFalse(report["independent_validation"])

    def test_unattempted_is_unknown_not_failure_or_validity_tie(self):
        rows, branches = full_fixture()
        for row in rows:
            if row["status"] == "completed":
                row["status"] = "not_attempted"
                row["measurement"] = None
        report = forecast_diagnostics(rows, branches)
        screen = next(s for s in report["B_AR_screens"] if s["shape"] == "L" and s["horizon"] == 4)
        self.assertEqual((screen["planned"], screen["available"], screen["original_missing"]), (144, 141, 3))
        self.assertEqual(screen["shape_screen"], dict(status="not_attempted", passed=None))
        self.assertIsNone(screen["conditional_limited_screen"]["passed"])
        l = cell(report, mode="AR", shape="L", horizon=4, stratum="all")
        self.assertEqual(l["paired_validity_B_win_tie_loss"]["object_shape"], [0, 0, 0])
        self.assertEqual(counts(report, l)["paired_unknown"], 141)
        self.assertEqual(metric(report, l, "position")["common_rows_per_arm"], 0)
        self.assertIsNone(metric(report, l, "position")["A_row_p90"])

    def test_TF_cannot_rescue_AR_and_common_subsets_do_not_erase_B_own_metrics(self):
        rows, branches = full_fixture()
        for row in rows:
            if row["mode"] == "AR" and row["status"] == "completed" and row["arm"] == "A":
                row["measurement"]["state"] = dict(valid=False, pose=None, reason="rigid_shape_mismatch")
                row["measurement"]["shape"]["valid"] = False
        report = forecast_diagnostics(rows, branches)
        ar = cell(report, mode="AR", shape="T", horizon=2, stratum="all")
        tf = cell(report, mode="TF", shape="T", horizon=2, stratum="all")
        self.assertEqual(ar["paired_validity_B_win_tie_loss"]["object_shape"], [48, 0, 0])
        self.assertEqual(metric(report, ar, "position")["common_rows_per_arm"], 0)
        self.assertIsNone(metric(report, ar, "position")["B_equal_anchor_mean"])
        self.assertEqual(metric(report, tf, "position")["common_rows_per_arm"], 48)
        screen = next(s for s in report["B_AR_screens"] if s["shape"] == "T" and s["horizon"] == 2)
        self.assertTrue(screen["conditional_limited_screen"]["passed"])
        self.assertEqual(screen["measurable_row_metrics"][0][1], 48)
        for row in rows:
            if row["mode"] == "AR" and row["arm"] == "B" and row["status"] == "completed":
                row["measurement"]["shape"]["valid"] = False
        report = forecast_diagnostics(rows, branches)
        self.assertTrue(all(s["shape_screen"]["status"] == "failed" for s in report["B_AR_screens"]))
        self.assertTrue(all(not s["conditional_limited_screen"]["passed"] for s in report["B_AR_screens"]))

    def test_unpaired_completed_record_is_unknown_and_row_p90_is_not_anchor_mean(self):
        rows, branches = full_fixture()
        a = next(r for r in rows if r["arm"] == "A" and r["mode"] == "AR" and r["shape"] == "T" and r["horizon"] == 1)
        a["status"] = "not_attempted"
        for row in rows:
            if row["mode"] == "AR" and row["arm"] == "B" and row["horizon"] == 1 and row["anchor"] in ("L00", "L02"):
                position = row["truth"]["pose"]["position"]
                row["measurement"]["state"]["pose"]["position"] = [position[0]+9., position[1]]
        report = forecast_diagnostics(rows, branches)
        t = cell(report, mode="AR", shape="T", horizon=1, stratum="all")
        self.assertEqual(t["paired_validity_B_win_tie_loss"]["object_shape"], [0, 47, 0])
        self.assertEqual(counts(report, t)["paired_unknown"], 1)
        l = cell(report, mode="AR", shape="L", horizon=1, stratum="all")
        self.assertLess(metric(report, l, "position")["B_equal_anchor_mean"], 5.)
        screen = next(s for s in report["B_AR_screens"] if s["shape"] == "L" and s["horizon"] == 1)
        self.assertTrue(screen["shape_screen"]["passed"])
        self.assertGreater(screen["measurable_row_metrics"][0][4], 5.)
        self.assertFalse(screen["conditional_limited_screen"]["passed"])

    def test_noise_branch_anchor_weighting_and_persistence_same_subset(self):
        rows = []
        branches = []
        for anchor, candidate, errors in (("T12", "hold", [0., 0., 0.]), ("T12", "toward", [12.]), ("T13", "hold", [9., 9., 9.])):
            latest = fake_truth() if anchor == "T12" else None
            branches.append(dict(anchor=anchor, candidate=candidate, latest=latest, truth=[fake_truth()]*4, contact_by_h4=False))
            for seed, error in zip(SEEDS, errors):
                for arm in ARMS:
                    value = error if arm == "A" else error/2
                    measurement = fake_measurement()
                    measurement["state"]["pose"]["position"] = [10.+value, 20.]
                    rows.append(dict(key=f"{anchor}/{candidate}/{seed}/AR/{arm}/H1", mode="AR", arm=arm, anchor=anchor,
                        candidate=candidate, seed=seed, shape="T", horizon=1, status="completed", measurement=measurement,
                        truth=fake_truth(), contact_by_h4=False))
        report = forecast_diagnostics(rows, branches)
        value = metric(report, cell(report, mode="AR", shape="T", horizon=1, stratum="all"), "position")
        self.assertEqual((value["common_rows_per_arm"], value["common_branches"], value["common_anchors"]), (7, 3, 2))
        self.assertAlmostEqual(value["A_equal_anchor_mean"], 7.5)
        self.assertAlmostEqual(value["B_equal_anchor_mean"], 3.75)
        self.assertAlmostEqual(value["B_minus_A_equal_anchor_mean"], -3.75)
        self.assertEqual(value["persistence_common_rows_per_arm"], 4)
        self.assertAlmostEqual(value["A_minus_persistence_equal_anchor_mean"], 6.)
        self.assertAlmostEqual(value["B_minus_persistence_equal_anchor_mean"], 3.)
        screen = next(s for s in report["B_AR_screens"] if s["shape"] == "T" and s["horizon"] == 1)
        self.assertFalse(screen["shape_screen"]["passed"])  # Seven measurable rows cannot pass 44/48.

    def test_wrapped_rotation_near_boundary_labels_and_unknowns(self):
        for coverage in (.92, .98, .9199999999999, .9800000000001):
            with self.subTest(coverage=coverage):
                rows = [r for r in ar_rows(("T12",)) if r["candidate"] == "hold" and r["horizon"] == 1]
                actual = fake_truth(coverage)
                actual["pose"]["angle"] = math.radians(-179)
                latest = fake_truth(coverage)
                latest["pose"]["angle"] = math.radians(179)
                for row in rows:
                    row["truth"] = deepcopy(actual)
                    row["contact_by_h4"] = True
                    row["measurement"]["pusher"] = dict(valid=False, position=None, reason="pusher_missing")
                branches = [dict(anchor="T12", candidate="hold", latest=latest, truth=[actual]*4, contact_by_h4=True)]
                report = forecast_diagnostics(rows, branches)
                near = cell(report, mode="AR", shape="T", horizon=1, stratum="near_095")
                rotation = cell(report, mode="AR", shape="T", horizon=1, stratum="rotation_gt_5")
                in_stratum = .92 <= coverage <= .98
                self.assertEqual(counts(report, near)["available_per_arm"], 3 if in_stratum else 0)
                self.assertEqual(counts(report, rotation)["available_per_arm"], 0)
                self.assertEqual(metric(report, near, "position")["common_rows_per_arm"], 3 if in_stratum else 0)
                self.assertEqual(metric(report, near, "pusher")["common_rows_per_arm"], 0)
                labels = next(r for r in report["label_counts"] if r["mode"] == "AR" and r["shape"] == "T" and r["horizon"] == 1)
                self.assertEqual(labels["labels"]["near_095"], [3, 0, 45] if in_stratum else [0, 3, 45])

    def test_bounded_base_reason_nuisance_range_and_appearance_diagnostics(self):
        rows, branches = full_fixture()
        a = next(r for r in rows if r["arm"] == "A" and r["mode"] == "AR" and r["shape"] == "T" and r["horizon"] == 1)
        a["measurement"]["shape"].update(valid=False, reason="an arbitrary long unexpected reason"*100)
        report = forecast_diagnostics(rows, branches)
        t = cell(report, mode="AR", shape="T", horizon=1, stratum="all")
        self.assertEqual(t["base_only"]["A"]["invalid_reasons"]["shape"], {"other": 1})
        self.assertEqual(t["base_only"]["A"]["nuisance"]["fitted_outline"][0], 48)
        self.assertEqual(t["base_only"]["A"]["raw"]["out_of_range_fraction"][0], 48)
        self.assertGreater(t["base_only"]["A"]["raw"]["out_of_range_fraction"][3], 0)
        self.assertTrue(all("base_only" not in c for c in report["cells"] if c["stratum"] != "all"))
        self.assertEqual(len(t["appearance"]), 4)
        payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertLessEqual(len(payload)+1, 128*1024)

    def test_duplicate_identity_conflicting_labels_and_invalid_mode_rejected(self):
        rows, branches = full_fixture()
        for malformed in (rows+[deepcopy(rows[0])], [dict(rows[0], mode="REAL")]):
            with self.assertRaises(ValueError):
                forecast_diagnostics(malformed, branches)
        rows[0]["truth"]["coverage"] = .01
        with self.assertRaisesRegex(ValueError, "Conflicting physical truth"):
            forecast_diagnostics(rows, branches)

    def test_mixed_reason_categories_and_long_statistics_fit_writer_bound(self):
        rows, branches = mixed_reason_fixture()
        report = forecast_diagnostics(rows, branches)
        payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()+b"\n"
        self.assertLessEqual(len(payload), MAX_SERIALIZED_BYTES)
        self.assertLessEqual(MAX_SERIALIZED_BYTES, 1024*1024)
        base = cell(report, mode="AR", shape="T", horizon=1, stratum="all")["base_only"]["A"]
        self.assertEqual(len(base["invalid_reasons"]["state"]), 10)


if __name__ == "__main__":
    unittest.main()
