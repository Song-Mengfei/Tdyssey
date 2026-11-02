"""Pure deterministic fake probes; no simulator, model, ledger or clock."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from tdyssey.policy_preflight import run_reset_preflight
from tdyssey.policy_protocol import CConfig


ROOT = Path(__file__).resolve().parents[1]


def original_config():
    data = CConfig.from_json(ROOT / "configs/policy_c.example.json").to_dict()
    # The checked-in example may be repaired after a real audit. These tests
    # explicitly retain the original declared seed ranges as a fake fixture.
    data["scenes"]["training"] = {"T": list(range(400000, 400032)),
                                   "L": list(range(500000, 500032))}
    data["evaluation"]["scenes"] = list(range(100000, 100005))
    data["evaluation"]["new_development_scenes"] = [100003, 100004]
    data["scenes"]["excluded_evaluation"] = list(range(100000, 100005))
    return CConfig.from_dict(data)


def result(valid=True, *, issue="reset_geometry_invalid"):
    return {"valid": valid, "issues": [] if valid else [issue], "error": None,
            "blocked": False, "classification": "valid" if valid else "geometry_invalid",
            "reset_attempts": 1, "reset_physics_substeps": 1, "control_steps": 0,
            "snapshot": {"position": [1.0, 2.0]}, "test_only": True}


class FakeProbe:
    def __init__(self, behavior=None):
        self.calls = []
        self.behavior = behavior

    def __call__(self, shape, seed, phase):
        self.calls.append((shape, seed, phase))
        return result() if self.behavior is None else self.behavior(shape, seed, phase)


class ResetPreflightChecks(unittest.TestCase):
    def setUp(self):
        self.config = original_config()

    def test_all_valid_rechecks_all_74_with_exact_order_counts_and_hash(self):
        probe = FakeProbe()
        before = self.config.to_dict()
        report = run_reset_preflight(self.config, probe)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"], {"probe_calls": 148, "reset_attempts": 148,
            "reset_physics_substeps": 148, "control_steps": 0,
            "counts_complete": True, "unknown_counter_records": 0})
        expected_train = [(shape, seed, "original_train") for shape, start in (("T", 400000), ("L", 500000))
                          for seed in range(start, start + 32)]
        expected_eval = [(shape, seed, "original_evaluation") for seed in range(100000, 100005)
                         for shape in ("T", "L")]
        self.assertEqual(probe.calls[:74], expected_train + expected_eval)
        self.assertEqual(probe.calls[74:], [(shape, seed, phase.replace("original_", "final_"))
                                          for shape, seed, phase in expected_train + expected_eval])
        self.assertEqual(self.config.to_dict(), before)
        self.assertEqual(report["proposed_config"], before)
        self.assertEqual(report["proposed_config_hash"], self.config.config_hash)
        self.assertEqual(report["final_verification"], {"status": "complete",
            "config_hash": self.config.config_hash, "expected_records": 74, "records": 74,
            "invalid_records": []})
        self.assertEqual(report["replacements"], [])
        self.assertEqual(report["candidate_decisions"], [])
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_invalid_slots_replace_in_place_in_declared_order_not_by_coverage(self):
        invalid = {("T", 400001), ("T", 400003), ("L", 500000)}

        def behavior(shape, seed, phase):
            value = result(not ((phase == "original_train" and (shape, seed) in invalid)
                                or (phase == "candidate_train" and seed == 400032)))
            value["coverage"] = -999.0 if value["valid"] else 999.0
            return value

        probe = FakeProbe(behavior)
        report = run_reset_preflight(self.config, probe)
        self.assertEqual(report["status"], "complete")
        expected = self.config.to_dict()
        expected["scenes"]["training"]["T"][1] = 400033
        expected["scenes"]["training"]["T"][3] = 400034
        expected["scenes"]["training"]["L"][0] = 500032
        self.assertEqual(report["proposed_config"], expected)
        self.assertEqual([(x["shape"], x["slot_index"], x["replacement_seed"]) for x in report["replacements"]],
                         [("T", 1, 400033), ("T", 3, 400034), ("L", 0, 500032)])
        self.assertEqual(probe.calls[74:78], [("T", 400032, "candidate_train"),
            ("T", 400033, "candidate_train"), ("T", 400034, "candidate_train"),
            ("L", 500032, "candidate_train")])
        proposed_hash = CConfig.from_dict(expected).config_hash
        self.assertNotEqual(proposed_hash, self.config.config_hash)
        self.assertEqual(report["proposed_config_hash"], proposed_hash)
        self.assertEqual(report["final_verification"]["config_hash"], proposed_hash)
        for entry in report["records"]:
            self.assertEqual(entry["config_hash"], proposed_hash if entry["phase"].startswith("final_")
                             else self.config.config_hash)
        self.assertEqual(report, run_reset_preflight(self.config, FakeProbe(behavior)))

    def test_skips_complete_exclusion_original_and_cross_shape_used_without_reset(self):
        data = self.config.to_dict()
        data["scenes"]["excluded_evaluation"].append(123456)
        config = CConfig.from_dict(data)
        invalid = {400000, 400001, 500000}

        def behavior(shape, seed, phase):
            return result(not ((phase == "original_train" and seed in invalid)
                               or (phase == "candidate_train" and seed == 700000)))

        candidates = {"T": [123456, 100000, 400000, 500001, 700000, 700000, 700001, 700001, 700002],
                      "L": [700000, 700001, 700002, 700003]}
        probe = FakeProbe(behavior)
        report = run_reset_preflight(config, probe, candidates=candidates)
        self.assertEqual(report["status"], "complete")
        self.assertEqual([seed for _, seed, phase in probe.calls if phase == "candidate_train"],
                         [700000, 700001, 700002, 700003])
        skipped = [x["reason"] for x in report["candidate_decisions"] if x["action"] == "skipped"]
        self.assertEqual(skipped, ["evaluation_excluded", "evaluation_excluded", "original_training",
            "original_training", "already_probed", "already_selected", "already_probed",
            "already_selected", "already_selected"])
        self.assertEqual([x["replacement_seed"] for x in report["replacements"]], [700001, 700002, 700003])
        self.assertEqual(candidates["L"], [700000, 700001, 700002, 700003])

    def test_maximum_exact_212_attempts_without_retry(self):
        probe = FakeProbe(lambda shape, seed, phase: result(phase != "original_train"))
        report = run_reset_preflight(self.config, probe)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["counts"]["probe_calls"], 212)
        self.assertEqual(report["counts"]["reset_attempts"], 212)
        self.assertEqual(report["counts"]["reset_physics_substeps"], 212)
        self.assertEqual(len(report["replacements"]), 64)
        self.assertEqual(len([x for x in probe.calls if x[2] == "candidate_train"]), 64)
        self.assertEqual(len(set(probe.calls)), 212)

    def test_candidate_exhaustion_retains_attempts_and_never_starts_final(self):
        probe = FakeProbe(lambda shape, seed, phase: result(
            not ((phase == "original_train" and seed == 400000) or phase == "candidate_train")))
        report = run_reset_preflight(self.config, probe)
        self.assertEqual((report["status"], report["reason"]), ("blocked", "candidate_sequence_exhausted"))
        self.assertEqual(report["counts"]["probe_calls"], 106)
        self.assertEqual(report["counts"]["reset_attempts"], 106)
        self.assertEqual(len(report["candidate_decisions"]), 32)
        self.assertIsNone(report["proposed_config"])
        self.assertEqual(report["final_verification"]["status"], "not_started")
        self.assertFalse(any(phase.startswith("final_") for _, _, phase in probe.calls))

    def test_invalid_fixed_evaluation_is_not_replaced_and_blocks_before_candidates(self):
        probe = FakeProbe(lambda shape, seed, phase: result(not (
            (phase == "original_train" and seed == 400000)
            or (phase == "original_evaluation" and shape == "L" and seed == 100001))))
        report = run_reset_preflight(self.config, probe)
        self.assertEqual((report["status"], report["reason"]), ("blocked", "fixed_evaluation_geometry_invalid"))
        self.assertEqual(report["counts"]["probe_calls"], 74)
        self.assertEqual(report["original_invalid_evaluation"], [68])
        self.assertEqual(report["candidate_decisions"], [])
        self.assertEqual(report["replacements"], [])
        self.assertIsNone(report["proposed_config_hash"])

    def test_final_geometry_failure_finishes_full_verify_without_second_repair(self):
        probe = FakeProbe(lambda shape, seed, phase: result(not (
            (phase == "original_train" and seed == 400000)
            or (phase == "final_train" and seed == 400032)
            or (phase == "final_evaluation" and shape == "L" and seed == 100004))))
        report = run_reset_preflight(self.config, probe)
        self.assertEqual((report["status"], report["reason"]), ("blocked", "final_verification_geometry_invalid"))
        self.assertEqual(report["counts"]["probe_calls"], 149)
        self.assertEqual(len(report["replacements"]), 1)
        self.assertEqual(report["final_verification"]["records"], 74)
        self.assertEqual(report["final_verification"]["status"], "blocked")
        self.assertEqual(len(report["final_verification"]["invalid_records"]), 2)
        self.assertEqual(CConfig.from_dict(report["proposed_config"]).config_hash,
                         report["final_verification"]["config_hash"])

    def test_runtime_error_blocks_immediately_and_preserves_actual_counters(self):
        for phase_to_fail, expected_calls in (("original_train", 1), ("candidate_train", 75),
                                              ("final_train", 75)):
            with self.subTest(phase=phase_to_fail):
                def behavior(shape, seed, phase):
                    if phase == phase_to_fail:
                        return {**result(False), "classification": "runtime_error", "blocked": True,
                                "error": {"type": "InjectedFailure", "message": "fake only"},
                                "reset_attempts": 1, "reset_physics_substeps": 0}
                    return result(not (phase_to_fail == "candidate_train" and phase == "original_train"
                                       and seed == 400000))
                probe = FakeProbe(behavior)
                report = run_reset_preflight(self.config, probe)
                self.assertEqual((report["status"], report["reason"]), ("blocked", "probe_runtime_error"))
                self.assertEqual(report["counts"]["probe_calls"], expected_calls)
                self.assertEqual(report["counts"]["reset_attempts"], expected_calls)
                self.assertEqual(report["counts"]["reset_physics_substeps"], expected_calls - 1)
                self.assertTrue(report["counts"]["counts_complete"])
                self.assertEqual(report["records"][-1]["result"]["error"]["type"], "InjectedFailure")
                if phase_to_fail == "final_train":
                    self.assertEqual(report["final_verification"]["status"], "blocked")

    def test_callback_exception_retains_unknown_attempt_without_fabricated_counts(self):
        def behavior(shape, seed, phase):
            if seed == 400001:
                raise RuntimeError("fake interruption after reset may have started")
            return result()
        probe = FakeProbe(behavior)
        report = run_reset_preflight(self.config, probe)
        self.assertEqual((report["status"], report["reason"]), ("blocked", "probe_exception"))
        self.assertEqual(report["counts"], {"probe_calls": 2, "reset_attempts": 1,
            "reset_physics_substeps": 1, "control_steps": 0,
            "counts_complete": False, "unknown_counter_records": 1})
        last = report["records"][-1]
        self.assertTrue(last["counts_unknown"])
        self.assertIsNone(last["result"])
        self.assertEqual(last["error"]["type"], "RuntimeError")
        self.assertEqual(len(probe.calls), 2)

    def test_ambiguous_malformed_or_guard_results_never_trigger_replacement(self):
        cases = [({**result(False), "classification": "invalid_geometry"}, "unrecognized_or_inconsistent"),
                 ({**result(False), "classification": "guard_violation"}, "unrecognized_or_inconsistent"),
                 ({**result(), "reset_attempts": 2}, "probe_counter_guard"),
                 ({**result(), "reset_physics_substeps": 2}, "probe_counter_guard"),
                 ({**result(), "control_steps": 1}, "probe_counter_guard"),
                 ({**result(), "reset_attempts": True}, "malformed_probe"),
                 ({**result(), "snapshot": {"coverage": float("nan")}}, "malformed_probe"),
                 ({"valid": False}, "malformed_probe"), (None, "malformed_probe")]
        for value, reason in cases:
            with self.subTest(value=value):
                probe = FakeProbe(lambda *args: deepcopy(value))
                report = run_reset_preflight(self.config, probe)
                self.assertEqual(report["status"], "blocked")
                self.assertIn(reason, report["reason"])
                self.assertEqual(report["counts"]["probe_calls"], 1)
                self.assertEqual(report["candidate_decisions"], [])
                json.dumps(report, allow_nan=False)
        counted = run_reset_preflight(self.config, FakeProbe(lambda *args: {**result(), "control_steps": 1}))
        self.assertEqual(counted["counts"]["control_steps"], 1)

    def test_invalid_inputs_fail_before_probe(self):
        probe = FakeProbe()
        bad_candidates = ({"T": [], "L": list(range(33))}, {"T": [], "L": [True]},
                          {"T": [], "L": [-1]}, {"T": []})
        for candidates in bad_candidates:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                run_reset_preflight(self.config, probe, candidates=candidates)
        data = self.config.to_dict()
        data["scenes"]["training"]["T"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            run_reset_preflight(CConfig.from_dict(data), probe)
        self.assertEqual(probe.calls, [])


if __name__ == "__main__":
    unittest.main()
