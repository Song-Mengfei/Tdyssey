"""Focused selection fixtures: no images, Torch, models, or optimizer work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_formal_selection as selection


def windows():
    result = []
    for index, letter in enumerate(selection.SEEN_LETTERS):
        for extra in (False, True):
            trajectory = index if not extra or index >= 14 else 18 + index
            anchor = 3 + index + int(extra)
            tid = "seen-val-fixture-" + str(trajectory)
            motion = "moving" if extra or index < 13 else "near_static" if index < 17 else "intermediate"
            result.append(dict(window_id=f"{tid}:{anchor}", trajectory_id=tid, anchor=anchor,
                letter=letter, split="seen_val", development_role="extra_moving" if extra else "natural",
                h1=dict(motion_class=motion), h4=dict(endpoint_moving=extra or index < 13 or index == 17)))
    return result


def catalog():
    return selection.build_selection_catalog(windows(), source_sha256="a" * 64)


def candidate(update=250, error=.4, role="scheduled"):
    records = [dict(window_id=row["window_id"], raw_object_error=error,
                    hard_generation_failure=False) for row in windows()]
    return dict(formal_experiment_id="formal4b", candidate_role=role, update=update,
                evaluation_complete=True, h1=deepcopy(records), h4=deepcopy(records))


class FormalSelectionTests(unittest.TestCase):
    def test_frozen_catalog_identity_and_independent_horizon_groups(self):
        value = catalog()
        self.assertEqual(value["summary"], dict(windows=36, independent_trajectories=32,
            h1_motion={"moving": 31, "near_static": 4, "intermediate": 1}, h4_endpoint_moving=32,
            development_roles={"natural": 18, "extra_moving": 18}))
        self.assertEqual([r["window_id"] for r in value["windows"]], [r["window_id"] for r in windows()])
        raw = json.dumps(dict(development=windows())).encode()
        expected = hashlib.sha256(raw).hexdigest()
        with patch.object(Path, "read_bytes", return_value=raw), patch.object(selection, "FROZEN_INDEX_SHA256", expected):
            self.assertEqual(selection.load_selection_catalog("mock.json", expected_sha256=expected)["summary"], value["summary"])
            with self.assertRaises(ValueError):
                selection.load_selection_catalog("mock.json", expected_sha256="b" * 64)

    def test_empty_groups_and_changed_catalog_fail_before_model_work(self):
        for horizon in ("h1", "h4"):
            values = windows()
            for row in values:
                if horizon == "h1": row["h1"]["motion_class"] = "near_static"
                else: row["h4"]["endpoint_moving"] = False
            with self.assertRaisesRegex(ValueError, "empty"):
                selection.build_selection_catalog(values, source_sha256="a" * 64)
        changed = catalog()
        changed["windows"][0]["h1_motion_class"] = "near_static"
        with self.assertRaisesRegex(ValueError, "modified"):
            selection.assess_candidate(candidate(), changed, formal_experiment_id="formal4b")

    def test_all72_failures_rank_before_better_moving_error(self):
        safe = candidate(250, .8)
        better_error = candidate(500, .1)
        # A static H1 failure still counts: failures are not restricted to moving.
        better_error["h1"][26]["hard_generation_failure"] = True
        result = selection.select_best([better_error, safe], catalog(), formal_experiment_id="formal4b")
        self.assertEqual(result["eligible_candidates"], 2)
        self.assertEqual(result["best"]["update"], 250)
        self.assertEqual(result["candidates"][0]["total_hard_generation_failures"], 1)
        self.assertEqual(result["candidates"][0]["metrics"]["h1"]["denominator"], 36)

    def test_balanced_means_use_frozen31_and32_groups_not_caller_labels(self):
        value = catalog()
        current = candidate(error=1.)
        for row, truth in zip(current["h1"], value["windows"]):
            if truth["h1_motion_class"] == "moving": row["raw_object_error"] = .2
            row["motion_class"] = "near_static"  # Caller labels cannot change selection.
        for row, truth in zip(current["h4"], value["windows"]):
            if truth["h4_endpoint_moving"]: row["raw_object_error"] = .4
            row["endpoint_moving"] = False
        assessed = selection.assess_candidate(current, value, formal_experiment_id="formal4b")
        self.assertTrue(assessed["eligible"])
        self.assertEqual(assessed["metrics"]["h1"]["moving_denominator"], 31)
        self.assertEqual(assessed["metrics"]["h4"]["moving_denominator"], 32)
        self.assertAlmostEqual(assessed["balanced_moving_gray_error"], .3)

    def test_incomplete72_placeholders_or_changed_IDs_cannot_win(self):
        safe = candidate(500, .4)
        cases = []
        incomplete = candidate(error=0.); incomplete["evaluation_complete"] = False; cases.append(incomplete)
        missing = candidate(error=0.); missing["h4"].pop(); cases.append(missing)
        duplicate = candidate(error=0.); duplicate["h1"][1]["window_id"] = duplicate["h1"][0]["window_id"]; cases.append(duplicate)
        reordered = candidate(error=0.); reordered["h4"].reverse(); cases.append(reordered)
        for bad in cases:
            result = selection.select_best([bad, safe], catalog(), formal_experiment_id="formal4b")
            self.assertFalse(result["candidates"][0]["eligible"])
            self.assertEqual(result["best"]["update"], 500)

    def test_strict_flags_and_nonfinite_scores_never_sort_as_valid(self):
        for field, value in (("hard_generation_failure", "False"), ("hard_generation_failure", 0),
                             ("raw_object_error", float("nan")), ("raw_object_error", float("inf")),
                             ("raw_object_error", True), ("raw_object_error", -.1)):
            bad = candidate(); bad["h4"][0][field] = value
            assessed = selection.assess_candidate(bad, catalog(), formal_experiment_id="formal4b")
            self.assertFalse(assessed["eligible"])
            self.assertIsNone(assessed["rank"])
            json.dumps(assessed, allow_nan=False)

    def test_ties_choose_earlier_and_denoising_loss_never_selects(self):
        early = candidate(250, .4); late = candidate(500, .4)
        early["denoising_loss"] = 100.; late["denoising_loss"] = 0.
        result = selection.select_best([late, early], catalog(), formal_experiment_id="formal4b")
        self.assertEqual(result["best"]["update"], 250)
        late = candidate(500, .3); late["denoising_loss"] = 1000.
        self.assertEqual(selection.select_best([early, late], catalog(), formal_experiment_id="formal4b")["best"]["update"], 500)

    def test_formal_scope_roles_and_duplicate_update_guard(self):
        bad = candidate(); bad["formal_experiment_id"] = "microfit3c"
        self.assertFalse(selection.assess_candidate(bad, catalog(), formal_experiment_id="formal4b")["eligible"])
        for update, role, eligible in ((129, "scheduled", False), (0, "early_stop", False),
                                       (341, "early_stop", True), (500, "early_stop", False), (500, "scheduled", True)):
            self.assertEqual(selection.assess_candidate(candidate(update, role=role), catalog(),
                formal_experiment_id="formal4b")["eligible"], eligible)
        with self.assertRaisesRegex(ValueError, "only one"):
            selection.select_best([candidate(), candidate()], catalog(), formal_experiment_id="formal4b")


if __name__ == "__main__":
    unittest.main()
