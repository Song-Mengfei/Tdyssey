"""Report-only H4 shape gate with otherwise qualified offline evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.native_rgb_world_model import evaluation_rows_sha256, report

ROOT = Path(__file__).resolve().parents[1]


class NativeReportGateChecks(unittest.TestCase):
    def test_h4_shape_failure_or_unassessable_output_blocks_an_otherwise_passing_run(self):
        sandbox = (ROOT/"artifacts").resolve()
        sandbox.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="native-rgb-report-gate-cpu-test-", dir=sandbox)
        root = Path(temporary.name).resolve()
        self.assertTrue(root.is_relative_to(sandbox))

        def put(name, value):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value)+"\n", encoding="utf-8")
            return path

        try:
            data = put("data/manifest.json", {"fixture": "independent seen validation"})
            model = root/"training/best.pt"
            model.parent.mkdir()
            model.write_bytes(b"opaque checkpoint fixture: never loaded")
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            put("data/physics-audit.json", {"physics_collection_gate_passed": True})
            put("engineering/report.json", {"passed": True})
            put("microfit/result.json", {"progress": {"fixed_train_loss_before": 1., "fixed_train_loss_after": .1}})
            physical = put("evaluation/seen_val/physical-report.json", dict(
                physical_dynamics_verified=True, physical_measurement_calibrated=True,
                checkpoint_sha256=model_sha, data_manifest_sha256=data_sha,
                core_rows_sha256=evaluation_rows_sha256([])))
            physical_bytes = physical.read_bytes()
            rgb = dict(checkpoint_sha256=model_sha, data_manifest_sha256=data_sha, rows=[],
                physical_evaluation_identity={"integration_source_sha256": "a"*64},
                qualification={"fine_tuned": {
                    "1": dict(motion_passed=True, prediction_failures=0, shape_assessable=20, shape_failures=0),
                    "4": dict(motion_passed=True, prediction_failures=0, shape_assessable=20, shape_failures=1)}})
            config = dict(max_artifact_bytes=3*1024**3, unseen_letters=list("DGKNQRWZ"))
            with patch("scripts.native_rgb_world_model.evaluate_with_physical", side_effect=AssertionError("Report must not run inference")):
                path = put("evaluation/seen_val/report.json", rgb)
                rgb_bytes = path.read_bytes()
                failed = report(config, root)
                self.assertFalse(failed["qualified_for_server_scaleup"])
                self.assertFalse(failed["shape_gate_passed"])
                self.assertEqual(failed["required_shape_horizons"], [1, 4])
                self.assertTrue(failed["shape_gates"]["1"]["passed"])
                self.assertFalse(failed["shape_gates"]["4"]["passed"])
                self.assertEqual(path.read_bytes(), rgb_bytes)
                self.assertEqual(physical.read_bytes(), physical_bytes)
                rgb["qualification"]["fine_tuned"]["4"].update(shape_assessable=0, shape_failures=0)
                put("evaluation/seen_val/report.json", rgb)
                self.assertFalse(report(config, root)["qualified_for_server_scaleup"])
                rgb["qualification"]["fine_tuned"]["4"].update(shape_assessable=20)
                put("evaluation/seen_val/report.json", rgb)
                passing = report(config, root)
                self.assertTrue(passing["qualified_for_server_scaleup"])
                self.assertTrue(passing["shape_gate_passed"])
                self.assertEqual(passing["evaluation_integration_source_sha256"], "a"*64)
        finally:
            self.assertTrue(root.resolve().is_relative_to(sandbox))
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
