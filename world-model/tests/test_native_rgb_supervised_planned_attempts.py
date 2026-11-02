"""Stage2B's initial-target attempt rotation; zero real physics controls."""

import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import box

from tdyssey import native_rgb_supervised_sampling as sampling
from tests.test_native_rgb_supervised_sampling import GeometricEnvironment

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/native-rgb-supervised-20261029/stage2b/baseline-source/native_rgb_supervised_sampling.py"
BASELINE_SHA256 = "38fc876f8c182a28f3b1a5e067c6a49e349367cb2485dbf85aa6e91f72858f11"


class PlannedAttemptSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_source = BASELINE.read_text(encoding="utf-8")
        if hashlib.sha256(BASELINE.read_bytes()).hexdigest() != BASELINE_SHA256:
            raise AssertionError("The preserved Stage2A baseline identity changed")
        spec = importlib.util.spec_from_file_location("tdyssey._stage2b_baseline", BASELINE)
        cls.old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.old)

    def test_all_controller_methods_constants_and_other_functions_are_byte_identical(self):
        current = Path(sampling.__file__).read_text(encoding="utf-8")
        old_nodes, new_nodes = ast.parse(self.old_source).body, ast.parse(current).body
        allowed = {"_candidate_priority", "plan_scene", "initialize_scene"}
        old = [ast.get_source_segment(self.old_source, node) for node in old_nodes
               if not isinstance(node, ast.FunctionDef) or node.name not in allowed]
        new = [ast.get_source_segment(current, node) for node in new_nodes
               if not isinstance(node, ast.FunctionDef) or node.name not in allowed]
        self.assertEqual(old, new)
        old_class = next(node for node in old_nodes if isinstance(node, ast.ClassDef))
        new_class = next(node for node in new_nodes if isinstance(node, ast.ClassDef))
        self.assertEqual(ast.dump(old_class, include_attributes=False), ast.dump(new_class, include_attributes=False))

    def test_none_keeps_exact_old_priority_tuple_and_rng_consumption(self):
        geometry = box(100, 100, 300, 300)
        rows = [dict(segment_id=0, normal=[-1, 0]), dict(segment_id=1, normal=[0, 1])]
        new, prior = {0: 2, 1: 0}, {0: 0, 1: 7}
        rng, old_rng = np.random.default_rng(76), np.random.default_rng(76)
        for row in rows:
            self.assertEqual(sampling._candidate_priority(row, geometry, new, prior, rng),
                             self.old._candidate_priority(row, geometry, new, prior, old_rng))
        self.assertEqual(rng.random(), old_rng.random())

    def test_explicit_attempt_count_follows_room_tier_and_precedes_old_factors(self):
        geometry = box(100, 100, 300, 300)
        rows = [dict(segment_id=0, normal=[-1, 0]), dict(segment_id=1, normal=[0, -1]),
                dict(segment_id=2, normal=[0, 1])]
        actual, prior, attempts = {0: 0, 1: 12, 2: 0}, {0: 0, 1: 4}, {0: 9, 1: 0, 2: 0}
        rng = np.random.default_rng(3)
        ordered = sorted(rows, key=lambda row: sampling._candidate_priority(row, geometry, actual, prior, rng, attempts))
        self.assertEqual([row["segment_id"] for row in ordered], [1, 0, 2])
        self.assertEqual(actual, {0: 0, 1: 12, 2: 0})
        self.assertEqual(attempts, {0: 9, 1: 0, 2: 0})

    def test_equal_attempt_counts_preserve_every_old_factor_and_seeded_tie(self):
        geometry = box(100, 100, 300, 300)
        row = dict(segment_id=3, normal=[-1, 0])
        legacy = self.old._candidate_priority(row, geometry, {3: 2}, {3: 4}, np.random.default_rng(15))
        enhanced = sampling._candidate_priority(row, geometry, {3: 2}, {3: 4}, np.random.default_rng(15), {"3": 7})
        self.assertEqual(enhanced, (legacy[0], 7, *legacy[1:]))
        empty = sampling._candidate_priority(row, geometry, {3: 2}, {3: 4}, np.random.default_rng(15), {})
        self.assertEqual(empty, (legacy[0], 0, *legacy[1:]))

    def test_independent_attempts_rotate_same_tier_without_fabricating_hits(self):
        geometry = box(100, 100, 300, 300)
        candidates = [dict(segment_id=index, normal=[-1, 0]) for index in range(4)]
        attempts, real_hits, planned = Counter(), {}, []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            chosen = min(candidates, key=lambda row: sampling._candidate_priority(row, geometry, real_hits, {}, rng, attempts))
            arc = chosen["segment_id"]
            planned.append(arc)
            attempts[arc] += 1  # Mock root event: one genuinely started independent trajectory.
        self.assertEqual(sorted(attempts.values()), [3, 3, 3, 3])
        for start in (0, 4, 8):
            self.assertEqual(set(planned[start:start+4]), set(range(4)))
        self.assertEqual(real_hits, {})

    def test_none_preserves_old_geometric_scene_and_records_legacy_version(self):
        for letter, location, seed in (("A", "center", 742), ("L", "right-top", 743)):
            kwargs = dict(gap_band=[8, 32], gap_world_px=18, coverage_counts={0: 2}, historical_counts={1: 4})
            old = self.old.plan_scene(letter, location, seed, **kwargs)
            new = sampling.plan_scene(letter, location, seed, **kwargs)
            self.assertEqual({key: new[key] for key in old}, old)
            self.assertEqual(new["selection_version"], "motion-room-actual-hit-prior-v1")
            self.assertFalse(new["selection_evidence"]["planned_attempts_enabled"])

    def test_explicit_empty_dict_versions_evidence_and_initializer_is_single_reset_only(self):
        env = GeometricEnvironment("A")
        attempts, hits = {}, {0: 5}
        scene = sampling.initialize_scene(env, dict(letter="A", spatial_class="center", seed=748),
                                          coverage_counts=hits, historical_counts={}, planned_attempts=attempts)
        self.assertTrue(scene["legal"])
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.control_step, 0)
        self.assertEqual(scene["selection_version"], "motion-room-planned-attempts-v2")
        evidence = scene["selection_evidence"]
        self.assertTrue(evidence["planned_attempts_enabled"])
        self.assertEqual(evidence["selected_arc"], scene["staging_segment"])
        self.assertEqual(evidence["sorted_candidates"][0]["segment_id"], scene["staging_segment"])
        self.assertEqual(evidence["priority_order"][1], "independent_planned_attempts")
        reachable = {row["segment_id"] for row in scene["reachability"] if row["status"] == "reachable_exterior_component"}
        self.assertEqual({row["segment_id"] for row in evidence["sorted_candidates"]}, reachable)
        json.dumps(scene, allow_nan=False)
        self.assertEqual(attempts, {})
        self.assertEqual(hits, {0: 5})

    def test_held_letters_stay_rejected_and_invalid_attempt_counts_fail_before_reset(self):
        self.assertEqual(sampling.ALLOWED_TRAINING_LETTERS, tuple("ABCEFHIJLMOPSTUVXY"))
        for letter in "DGKNQRWZ":
            with self.assertRaises(ValueError):
                sampling.plan_scene(letter, "center", 3, planned_attempts={})
        for value in (-1, .5, True):
            env = GeometricEnvironment("A")
            with self.assertRaises(ValueError):
                sampling.initialize_scene(env, dict(letter="A", spatial_class="center", seed=3), planned_attempts={0: value})
            self.assertEqual(env.reset_calls, 0)
            self.assertEqual(env.control_step, 0)


if __name__ == "__main__":
    unittest.main()
