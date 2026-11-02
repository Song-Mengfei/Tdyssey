"""Fake numeric records only: no images, decoder, model, or environment."""

from copy import deepcopy
import json
import math
import unittest

from tdyssey.wm_measurement_contract import DEVELOPMENT_BOUND, coverage_measurement, measurement_frame
from tdyssey.wm_compat_reporting import (
    ANCHORS, ARMS, CANDIDATES, HORIZONS, SEEDS,
    ar_sequence_scores, measurement_screen, paired_rankings, t_stage_gate,
)


def fake_measurement(estimate=.2, *, state=True, shape=True, pusher=True, coverage=True, half_width=.03):
    return measurement_frame(
        state={"valid": state, "pose": {"position": [10., 20.], "angle": 0.} if state else None},
        shape={"valid": shape, "reason": "ok" if shape else "fake_shape_failure"},
        pusher={"valid": pusher, "position": [3., 4.] if pusher else None},
        coverage=coverage_measurement(estimate, valid=coverage, half_width=half_width, uncertainty_kind=DEVELOPMENT_BOUND),
    )


def fake_truth(coverage=.2):
    return {"pose": {"position": [10., 20.], "angle": 0.}, "pusher": [3., 4.], "coverage": coverage}


def real_rows():
    """Independent fixture builds the documented inventory identity syntax."""
    result = []
    for arm in ARMS:
        for shape in ("L", "T"):
            for i in range(12):
                source_key = "cal-%s-%02d" % (shape, i)
                result.append({"key": arm+"/"+source_key, "source_key": source_key, "mode": "REAL", "arm": arm,
                               "shape": shape, "source": "calibration", "status": "completed",
                               "measurement": fake_measurement(), "truth": fake_truth()})
        for anchor in ANCHORS:
            for candidate in CANDIDATES:
                for source, indices in (("history", range(4)), ("physical_future", HORIZONS)):
                    for index in indices:
                        suffix = "history-%d" % index if source == "history" else "future-H%d" % index
                        source_key = "%s-%s-%s" % (anchor, candidate, suffix)
                        missing = anchor == "L01" and candidate == "hold"
                        result.append({"key": arm+"/"+source_key, "source_key": source_key, "mode": "REAL", "arm": arm,
                                       "shape": anchor[0], "source": source, "status": "missing" if missing else "completed",
                                       "measurement": {} if missing else fake_measurement(), "truth": {} if missing else fake_truth()})
    return result


def ar_rows(anchors=ANCHORS):
    result = []
    for arm in ARMS:
        for anchor in anchors:
            for candidate_index, candidate in enumerate(CANDIDATES):
                value = .1*(candidate_index+1)
                for seed in SEEDS:
                    for horizon in HORIZONS:
                        missing = anchor == "L01" and candidate == "hold"
                        result.append({"key": "%s/%s/%s/%d/AR/H%d" % (arm, anchor, candidate, seed, horizon),
                                       "arm": arm, "anchor": anchor, "candidate": candidate, "seed": seed,
                                       "shape": anchor[0], "mode": "AR", "horizon": horizon,
                                       "status": "missing" if missing else "completed",
                                       "measurement": {} if missing else fake_measurement(value),
                                       "truth": {} if missing else fake_truth(value),
                                       "contact_by_h4": candidate == "toward"})
    return result


def cell(report, **selectors):
    return next(row for row in report if all(row.get(key) == value for key, value in selectors.items()))


class MeasurementScreenTests(unittest.TestCase):
    def test_fixed_full_inventory_known_missing_and_no_input_mutation(self):
        rows = real_rows()
        original = deepcopy(rows)
        report = measurement_screen(rows)
        self.assertTrue(report["passed"])
        self.assertEqual((len(rows), len(report["rows"])), (1072, 1072))
        self.assertEqual((report["available_total"], report["known_missing_total"]), (1056, 16))
        for arm in ARMS:
            l = cell(report["by_arm_shape"], arm=arm, shape="L")
            t = cell(report["by_arm_shape"], arm=arm, shape="T")
            self.assertEqual((l["planned"], l["available"], l["known_missing"]), (396, 388, 8))
            self.assertEqual((t["planned"], t["available"], t["known_missing"]), (140, 140, 0))
            self.assertIsNone(l["interval_confusion"]["precision"])
        self.assertEqual(rows, original)
        json.dumps(report, allow_nan=False)

    def test_cannot_replace_missing_source_with_duplicate_or_wrong_identity(self):
        rows = real_rows()
        removed = rows.pop(0)
        report = measurement_screen(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["rows"]), 1072)
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate REAL"):
            measurement_screen(rows)
        rows = real_rows()
        rows[0]["source_key"] = "arbitrary-replacement"
        report = measurement_screen(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["unexpected_sources"]), 1)
        self.assertEqual(removed["source_key"], "cal-L-00")

    def test_local_pusher_failure_does_not_erase_object_error(self):
        rows = real_rows()
        rows[0]["measurement"]["pusher"] = {"valid": False, "position": None, "reason": "fake_occluded"}
        report = measurement_screen(rows)
        self.assertFalse(report["passed"])
        l = cell(report["by_arm_shape"], arm="A", shape="L")
        self.assertEqual(l["errors"]["position"]["count"], 388)
        self.assertEqual(l["errors"]["pusher"]["count"], 387)
        self.assertEqual(l["losses"][0]["reasons"]["pusher"], "fake_occluded")
        rows[0]["measurement"] = fake_measurement(state=False, shape=False, coverage=False)
        record = cell(measurement_screen(rows)["rows"], arm="A", source_key="cal-L-00")
        self.assertFalse(record["flags"]["object"])
        self.assertFalse(record["flags"]["coverage"])
        self.assertTrue(record["flags"]["pusher"])

    def test_nominal_is_only_diagnostic_and_interval_fp_is_gate(self):
        rows = real_rows()
        rows[0]["measurement"] = fake_measurement(.96)
        rows[0]["truth"] = fake_truth(.95)
        report = measurement_screen(rows)
        l = cell(report["by_arm_shape"], arm="A", shape="L")
        self.assertTrue(report["passed"])
        self.assertEqual(l["nominal_confusion"]["fp"], 1)
        self.assertEqual((l["interval_confusion"]["fp"], l["interval_confusion"]["unknown"]), (0, 1))
        rows[0]["measurement"] = fake_measurement(.99)
        report = measurement_screen(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(cell(report["by_arm_shape"], arm="A", shape="L")["interval_confusion"]["fp"], 1)

    def test_p90_uses_wrapped_angles_and_gates_shape_separately(self):
        rows = real_rows()
        for row in rows:
            if row["status"] == "completed":
                row["measurement"]["state"]["pose"]["angle"] = math.radians(359)
                row["truth"]["pose"]["angle"] = math.radians(1)
                row["measurement"]["state"]["pose"]["position"] = [12., 20.]
        report = measurement_screen(rows)
        self.assertTrue(report["passed"])
        rows[0]["measurement"]["shape"]["valid"] = False
        report = measurement_screen(rows)
        self.assertFalse(report["passed"])
        l = cell(report["by_arm_shape"], arm="A", shape="L")
        self.assertEqual(l["valid_counts"]["object"], 388)
        self.assertEqual(l["valid_counts"]["coverage"], 387)


class TStageTests(unittest.TestCase):
    def test_exact_44_shape_46_pusher_and_tf_cannot_rescue(self):
        rows = ar_rows(tuple(a for a in ANCHORS if a[0] == "T"))
        for horizon in (1, 2, 4):
            for arm in ARMS:
                selected = [r for r in rows if r["horizon"] == horizon and r["arm"] == arm]
                for row in selected[:4]:
                    row["measurement"]["shape"]["valid"] = False
                for row in selected[4:6]:
                    row["measurement"]["pusher"]["valid"] = False
        report = t_stage_gate(rows)
        self.assertTrue(report["passed"])
        for result in report["by_arm_horizon"]:
            self.assertEqual((result["object_shape_valid"], result["pusher_valid"]), (44, 46))
        b_h1 = [r for r in rows if r["arm"] == "B" and r["horizon"] == 1]
        b_h1[6]["measurement"]["shape"]["valid"] = False
        tf = deepcopy(b_h1)
        for row in tf:
            row["mode"] = "TF"
            row["measurement"] = fake_measurement()
        report = t_stage_gate(rows+tf)
        self.assertFalse(report["passed"])
        self.assertEqual(report["ignored_TF_rows"], 48)

    def test_contact_denominator_includes_invalid_and_pusher_regression_not_extra_gate(self):
        rows = ar_rows(tuple(a for a in ANCHORS if a[0] == "T"))
        selected = [r for r in rows if r["arm"] == "B" and r["horizon"] == 1 and r["contact_by_h4"]]
        self.assertEqual(len(selected), 12)
        selected[0]["measurement"]["shape"]["valid"] = False
        report = t_stage_gate(rows)
        self.assertFalse(report["passed"])
        stratum = cell(report["contact_strata"], horizon=1, contact_by_h4=True)
        self.assertEqual(stratum["rows"], 12)
        self.assertAlmostEqual(stratum["B_minus_A_validity"]["object_shape"], -1/12)
        selected[0]["measurement"]["shape"]["valid"] = True
        selected[0]["measurement"]["pusher"]["valid"] = False
        self.assertTrue(t_stage_gate(rows)["passed"])
        rows[0]["contact_by_h4"] = 1
        self.assertFalse(t_stage_gate(rows)["passed"])


class ARScoresAndRankingTests(unittest.TestCase):
    def test_tf_rejected_and_full_denominators_retained(self):
        row = ar_rows(("T12",))[0]
        row["mode"] = "TF"
        for function in (ar_sequence_scores, paired_rankings):
            with self.assertRaisesRegex(ValueError, "AR-only"):
                function([row])
        report = ar_sequence_scores([])
        self.assertEqual(len(report["rows"]), 1536)
        self.assertEqual(report["summary"]["planned_streams"], 384)
        self.assertEqual(sum(r["known_missing_physical_branch"] for r in report["rows"]), 24)
        self.assertTrue(all(not r["raw_endpoint_valid"] and not r["sequence_score_eligible"] for r in report["rows"]))

    def test_raw_survives_unknown_history_pusher_independent_absorption_model_only(self):
        rows = ar_rows(("T12",))
        stream = [r for r in rows if r["arm"] == "A" and r["candidate"] == "hold" and r["seed"] == 5478]
        stream[0]["measurement"] = fake_measurement(.95)  # Interval unknown.
        stream[3]["measurement"]["pusher"]["valid"] = False
        stream[3]["truth"]["coverage"] = 1.  # Truth cannot repair prediction.
        report = ar_sequence_scores(rows)
        final = cell(report["rows"], arm="A", anchor="T12", candidate="hold", seed=5478, horizon=4)
        self.assertTrue(final["raw_endpoint_valid"])
        self.assertEqual(final["raw_endpoint_coverage"], .1)
        self.assertFalse(final["sequence_score_eligible"])
        self.assertIsNone(final["sequence_score"])
        stream[0]["measurement"] = fake_measurement(.99)
        stream[0]["truth"]["coverage"] = 0.  # Model-only declared success absorbs.
        for row in stream[1:]:
            row["status"] = "not_attempted"
            row["measurement"] = {}
        final = cell(ar_sequence_scores(rows)["rows"], arm="A", anchor="T12", candidate="hold", seed=5478, horizon=4)
        self.assertFalse(final["raw_endpoint_valid"])
        self.assertTrue(final["sequence_score_eligible"])
        self.assertEqual((final["sequence_score"], final["stop_horizon"]), (.99, 1))
        self.assertTrue(all(x["after_certain_absorption"] for x in final["unknown_history"]))

    def test_finite_invalid_diagnostics_abstain_with_unknown_observed_accuracy(self):
        rows = ar_rows(("T12",))
        for row in rows:
            row["measurement"]["shape"]["valid"] = False
        report = paired_rankings(rows)
        anchor = cell(report["anchor_rows"], arm="A", anchor="T12", horizon=1, scoring_mode="raw_endpoint")
        self.assertFalse(anchor["selection_eligible"])
        self.assertIsNone(anchor["selected_candidate"])
        self.assertEqual(anchor["pairs"]["eligible_pairs"], 6)
        self.assertEqual((anchor["pairs"]["scored_pairs"], anchor["pairs"]["abstained_pairs"]), (0, 6))
        self.assertIsNone(anchor["pairs"]["observed_pair_accuracy"])
        self.assertEqual(anchor["pairs"]["conservative_correct_over_all_eligible"], 0.)
        self.assertEqual([anchor["candidates"][c]["noises"][0]["value"] for c in CANDIDATES], [.1, .2, .30000000000000004, .4])
        subset = cell(report["paired_complete_subset"], shape="T", horizon=1, scoring_mode="raw_endpoint")
        self.assertEqual(subset["paired_complete_anchors"], 0)
        self.assertIsNone(subset["A_regret"])

    def test_matched_anchor_subset_and_independent_physical_references(self):
        rows = ar_rows(("T12", "T13"))
        # T12: hold is worst, both model arms choose away, regret zero.
        # T13: hold is best, but one B noise invalidates the complete anchor.
        for row in rows:
            if row["anchor"] == "T13":
                row["truth"]["coverage"] = .8 if row["candidate"] == "hold" else .1
                if row["arm"] == "B" and row["candidate"] == "hold" and row["seed"] == 5478:
                    row["measurement"]["coverage"]["valid"] = False
        original = deepcopy(rows)
        report = paired_rankings(rows)
        subset = cell(report["paired_complete_subset"], shape="T", horizon=4, scoring_mode="raw_endpoint")
        baseline = cell(report["independent_physical_baselines"], shape="T", horizon=4)
        self.assertEqual(subset["included_anchors"], ["T12"])
        self.assertEqual((subset["planned_anchors"], subset["paired_complete_anchors"]), (4, 1))
        self.assertAlmostEqual(subset["hold_regret"], .3)
        self.assertAlmostEqual(subset["uniform_expected_regret"], .15)
        self.assertEqual((subset["A_regret"], subset["B_regret"]), (0., 0.))
        self.assertEqual(baseline["complete_physical_anchors"], 2)
        self.assertAlmostEqual(baseline["hold_regret"], .15)
        self.assertEqual(rows, original)
        json.dumps(report, allow_nan=False)

    def test_noise_mean_fixed_ties_and_physical_label_conflicts(self):
        rows = ar_rows(("T12",))
        for row in rows:
            # One optimistic away noise is insufficient to beat the side mean.
            if row["candidate"] == "away":
                row["measurement"] = fake_measurement(.6 if row["seed"] == 5478 else .1)
        report = paired_rankings(rows)
        a = cell(report["anchor_rows"], arm="A", anchor="T12", horizon=1, scoring_mode="raw_endpoint")
        self.assertEqual(a["selected_candidate"], "side")
        # Predicted exact tie breaks by declared candidate order, not truth.
        for row in rows:
            row["measurement"] = fake_measurement(.2)
        a = cell(paired_rankings(rows)["anchor_rows"], arm="A", anchor="T12", horizon=1, scoring_mode="raw_endpoint")
        self.assertEqual(a["selected_candidate"], "hold")
        rows[0]["truth"]["coverage"] = .8
        physical = cell(paired_rankings(rows)["physical_anchor_rows"], anchor="T12", horizon=1)
        self.assertFalse(physical["complete"])
        self.assertEqual(physical["truth_conflicts"], ["hold"])

    def test_unattempted_predictions_keep_independent_physical_denominator(self):
        rows = ar_rows()
        for row in rows:
            if row["status"] == "completed":
                row["status"] = "not_attempted"
                row["measurement"] = {}
        report = paired_rankings(rows)
        physical = cell(report["independent_physical_baselines"], shape="all", horizon=4)
        paired = cell(report["paired_complete_subset"], shape="all", horizon=4, scoring_mode="stopping")
        self.assertEqual((physical["planned_anchors"], physical["complete_physical_anchors"]), (16, 15))
        self.assertEqual(physical["omitted_anchors"], ["L01"])
        self.assertEqual(paired["paired_complete_anchors"], 0)
        self.assertEqual(paired["all_planned_anchor_pair_metrics"]["A"]["scored_pairs"], 0)
        self.assertIsNone(paired["all_planned_anchor_pair_metrics"]["A"]["pooled_observed_correct_over_scored"])


if __name__ == "__main__":
    unittest.main()
