"""Opt-in acceptance of recorded real resets; this test runs no physics.

Set TDYSSEY_RESET_PREFLIGHT_REPORT to a completed reset-only report. Ordinary
discovery skips the file. Only the explicitly gated preflight CLI executes resets.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import unittest

from tdyssey.policy_protocol import CConfig

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("TDYSSEY_RESET_PREFLIGHT_REPORT"),
                     "Set TDYSSEY_RESET_PREFLIGHT_REPORT to accept recorded reset-only evidence")
class RecordedResetAcceptance(unittest.TestCase):
    def test_current_config_and_code_bind_all_real_attempts_and_final_scenes(self):
        path = Path(os.environ["TDYSSEY_RESET_PREFLIGHT_REPORT"])
        report = json.loads(path.read_text(encoding="utf-8"))
        original = CConfig.from_json(path.parent / "original-config.json")
        current = CConfig.from_json(ROOT / "configs/policy_c.example.json")
        self.assertEqual(report["kind"], "tdyssey.C.reset_only_preflight")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(original.config_hash, report["original_config_hash"])
        self.assertEqual(current.to_dict(), report["proposed_config"])
        self.assertEqual(current.config_hash, report["proposed_config_hash"])
        self.assertEqual(current.to_dict()["evaluation"], original.to_dict()["evaluation"])
        self.assertEqual(current.to_dict()["scenes"]["excluded_evaluation"],
                         original.to_dict()["scenes"]["excluded_evaluation"])
        for relative, digest in report["provenance"]["code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)
        self.assertTrue(all(report["preserved_evidence_unchanged"].values()))
        self.assertFalse(report["formal_C_experiment_initialized"])
        self.assertFalse(report["formal_C_clock_started"])
        self.assertFalse(report["formal_evaluation"])
        self.assertFalse(report["policy_loaded"])
        self.assertFalse(report["tensor_runtime_imported"])
        self.assertEqual((report["policy_samples"], report["optimizer_steps"]), (0, 0))
        records = report["records"]
        self.assertEqual(len(records), report["counts"]["probe_calls"])
        self.assertTrue(report["counts"]["counts_complete"])
        self.assertEqual(report["counts"]["unknown_counter_records"], 0)
        self.assertLessEqual(len(records), 212)
        for entry in records:
            result = entry["result"]
            self.assertIsNone(result["error"])
            self.assertEqual((result["reset_attempts"], result["reset_physics_substeps"],
                              result["space_step_attempts"], result["control_steps"]), (1, 1, 1, 0))
            self.assertEqual((result["blocked_space_step_attempts"], result["blocked_control_attempts"]), (0, 0))
            self.assertEqual(result["guard_violations"], [])
            self.assertTrue(result["provenance"]["source_verified"])
            self.assertFalse(result["provenance"]["injected_test_dependencies"])
            self.assertEqual(result["snapshot"]["control_step"], 0)
        for key in ("reset_attempts", "reset_physics_substeps", "control_steps"):
            self.assertEqual(report["counts"][key], sum(entry["result"][key] for entry in records))
            self.assertEqual(report["audit_counts"][key], report["counts"][key])
        self.assertEqual(report["audit_counts"]["unresolved_probe_claims"], [])
        final = [entry for entry in records if entry["phase"].startswith("final_")]
        self.assertEqual(len(final), 74)
        self.assertTrue(all(entry["result"]["valid"] and entry["config_hash"] == current.config_hash for entry in final))
        self.assertEqual(report["final_verification"]["config_hash"], current.config_hash)
        self.assertEqual(report["final_verification"]["invalid_records"], [])
        self.assertEqual(Counter(entry["phase"] for entry in final), {"final_train": 64, "final_evaluation": 10})
        for shape in ("T", "L"):
            actual = [entry["seed"] for entry in final if entry["phase"] == "final_train" and entry["shape"] == shape]
            self.assertEqual(actual, current.to_dict()["scenes"]["training"][shape])
        for entry in final:
            if entry["phase"] == "final_evaluation" and entry["shape"] == "L":
                matched = entry["result"]["matched_com"]
                self.assertTrue(matched["valid"])
                self.assertFalse(matched["state_restored"])
                self.assertEqual((matched["physics_substeps"], matched["control_steps"]), (0, 0))
        journal = [json.loads(line) for line in (path.parent / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(Counter(row["event"] for row in journal),
                         {"attempt_reserved": len(records), "attempt_result": len(records)})
        self.assertEqual([row["id"] for row in journal[::2]], list(range(1, len(records) + 1)))
        self.assertTrue(all(a["id"] == b["id"] and a["shape"] == b["shape"] and a["seed"] == b["seed"]
                            for a, b in zip(journal[::2], journal[1::2])))


if __name__ == "__main__":
    unittest.main()
