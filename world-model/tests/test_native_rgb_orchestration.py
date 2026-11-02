"""Metadata/CLI orchestration checks; no Torch, GPU or physical controls."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.native_rgb_world_model import (
    current_validation_directory, freeze_model, main, report, evaluate_with_physical,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_sha(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def config():
    return dict(max_artifact_bytes=3*1024**3, server=False, seed=5478,
        train_letters=list("ABCEFHIJLMOPSTUVXY"), unseen_letters=list("DGKNQRWZ"))


def temporary_directory():
    # Atomic replace is sandbox-restricted in the system Temp on this host.
    parent = Path(__file__).resolve().parents[1]/"artifacts/native-rgb-orchestration-test-tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def make_metadata(root):
    output, data = root/"seed-5478", root/"shared-data"
    write_json(data/"manifest.json", dict(kind="mock_metadata_only"))
    write_json(data/"physics-audit.json", dict(physics_collection_gate_passed=True))
    write_json(data/"coverage/summary.json", dict(train={"L": dict(contacted_fraction=.8)}))
    write_json(output/"engineering/report.json", dict(passed=True))
    write_json(output/"microfit/result.json", dict(progress=dict(fixed_train_loss_before=1.,fixed_train_loss_after=.5)))
    checkpoint = output/"training/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"metadata-only-native-checkpoint")
    horizon = dict(motion_passed=True,prediction_failures=0,shape_assessable=16,shape_failures=0,
                   stationary=dict(passed=True))
    evaluation = dict(checkpoint_sha256=sha(checkpoint), data_manifest_sha256=sha(data/"manifest.json"),rows=[],
        qualification=dict(fine_tuned={"1":horizon,"4":horizon}))
    write_json(output/"evaluation/seen_val/report.json", evaluation)
    write_json(output/"evaluation/seen_val/physical-report.json",
               dict(physical_dynamics_verified=True,physical_measurement_calibrated=True,
                    checkpoint_sha256=evaluation["checkpoint_sha256"],data_manifest_sha256=evaluation["data_manifest_sha256"],
                    core_rows_sha256=rows_sha(evaluation["rows"])))
    return output, data, evaluation


class NativeRGBOrchestrationTests(unittest.TestCase):
    def test_per_seed_report_uses_actual_shared_data_audit_and_current_hashes(self):
        with temporary_directory() as directory:
            output, data, _ = make_metadata(Path(directory))
            result = report(config(), output, data_root=data)
            self.assertTrue(result["physics_passed"])
            self.assertTrue(result["validation_matches_current_best"])
            self.assertTrue(result["validation_matches_current_data"])
            self.assertTrue(result["qualified_for_server_scaleup"])
            self.assertFalse((output/"data/physics-audit.json").exists())
            self.assertEqual(result["evidence_paths"]["data/physics-audit.json"], "../shared-data/physics-audit.json")

    def test_stale_best_or_data_identity_cannot_qualify_then_matching_best_can(self):
        with temporary_directory() as directory:
            output, data, evaluation = make_metadata(Path(directory))
            checkpoint = output/"training/best.pt"
            checkpoint.write_bytes(b"new checkpoint after resume")
            stale = report(config(), output, data)
            self.assertFalse(stale["validation_matches_current_best"])
            self.assertFalse(stale["qualified_for_server_scaleup"])
            evaluation["checkpoint_sha256"] = sha(checkpoint)
            write_json(output/"evaluation/seen_val/report.json", evaluation)
            self.assertTrue(report(config(), output, data)["validation_matches_current_best"])
            write_json(data/"manifest.json", dict(kind="different-data"))
            changed_data = report(config(), output, data)
            self.assertFalse(changed_data["validation_matches_current_data"])
            self.assertFalse(changed_data["qualified_for_server_scaleup"])

    def test_failed_h4_motion_or_prediction_cannot_be_hidden_by_passing_h1(self):
        for change in (dict(motion_passed=False), dict(prediction_failures=1)):
            with self.subTest(change=change), temporary_directory() as directory:
                output, data, evaluation = make_metadata(Path(directory))
                # Copy the shared fixture horizon so H1 remains independently good.
                evaluation["qualification"]["fine_tuned"]["4"] = dict(
                    evaluation["qualification"]["fine_tuned"]["4"], **change)
                write_json(output/"evaluation/seen_val/report.json", evaluation)
                result = report(config(), output, data)
                self.assertTrue(result["validation_matches_current_best"])
                self.assertTrue(result["validation_matches_current_data"])
                self.assertFalse(result["qualified_for_server_scaleup"])

    def test_missing_or_failed_physical_calibration_cannot_qualify(self):
        for physical in (None,dict(physical_dynamics_verified=False,physical_measurement_calibrated=True),
                         dict(physical_dynamics_verified=True,physical_measurement_calibrated=False)):
            with self.subTest(physical=physical),temporary_directory() as directory:
                output,data,_=make_metadata(Path(directory))
                path=output/"evaluation/seen_val/physical-report.json"
                if physical is None:path.unlink()
                else:write_json(path,physical)
                result=report(config(),output,data)
                self.assertTrue(result["apparent_motion_gate_passed"])
                self.assertFalse(result["qualified_for_server_scaleup"])

    def test_stale_physical_model_data_or_raw_rows_cannot_qualify(self):
        for field in ("checkpoint_sha256","data_manifest_sha256","core_rows_sha256"):
            with self.subTest(stale=field),temporary_directory() as directory:
                output,data,_=make_metadata(Path(directory))
                path=output/"evaluation/seen_val/physical-report.json"
                physical=json.loads(path.read_text(encoding="utf-8"));physical[field]="old-result"
                write_json(path,physical)
                result=report(config(),output,data)
                self.assertTrue(result["validation_matches_current_best"])
                self.assertTrue(result["validation_matches_current_data"])
                self.assertFalse(result["physical_report_matches_current_rgb_rows"])
                self.assertFalse(result["physical_dynamics_verified"])
                self.assertFalse(result["qualified_for_server_scaleup"])
        with temporary_directory() as directory:
            output,data,evaluation=make_metadata(Path(directory))
            evaluation["rows"].append(dict(window="changed-raw-window",method="fine_tuned",horizon=4))
            write_json(output/"evaluation/seen_val/report.json",evaluation)
            result=report(config(),output,data)
            self.assertFalse(result["physical_report_matches_current_rgb_rows"])
            self.assertFalse(result["qualified_for_server_scaleup"])

    def test_validation_pointer_rejects_parent_traversal_and_absolute_escape(self):
        with temporary_directory() as directory:
            root = Path(directory)
            output = root/"experiment"
            for value in ("evaluation/../../shared-data", str((root/"outside").resolve())):
                write_json(output/"evaluation/current-seen-val.json", dict(directory=value))
                with self.assertRaisesRegex(ValueError, "escapes"):
                    current_validation_directory(output)
            write_json(output/"evaluation/current-seen-val.json", dict(directory="evaluation/seen_val/checkpoint-example"))
            self.assertEqual(current_validation_directory(output),
                             (output/"evaluation/seen_val/checkpoint-example").resolve())

    def test_freeze_requires_seen_calibration_before_accessing_final_protocol_data(self):
        with temporary_directory() as directory:
            root = Path(directory)
            checkpoint = root/"best.pt"
            checkpoint.write_bytes(b"mock complete checkpoint")
            with patch("tdyssey.native_rgb_training.load_checkpoint_payload",
                       return_value=dict(identity=dict(phase="train"),progress={})), \
                 patch("tdyssey.native_rgb_evaluation.final_protocol_identity",
                       side_effect=AssertionError("Final protocol data must not be inspected yet")) as identity:
                with self.assertRaises(FileNotFoundError):
                    freeze_model(config(), root, checkpoint, root/"nonexistent-final-data")
                identity.assert_not_called()
            self.assertFalse((root/"frozen-model.json").exists())

    def test_freeze_uses_pointed_seen_calibration_and_preserves_seal(self):
        with temporary_directory() as directory:
            root = Path(directory)
            checkpoint = root/"best.pt"
            checkpoint.write_bytes(b"mock complete checkpoint")
            write_json(root/"evaluation/current-seen-val.json", dict(directory="evaluation/seen_val/checkpoint-current"))
            write_json(root/"evaluation/seen_val/checkpoint-current/calibration.json", dict(frozen_calibration="seen-only"))
            write_json(root/"physical-measurement-calibration-final.json",dict(physical_calibration="fixed-seen"))
            with patch("tdyssey.native_rgb_training.load_checkpoint_payload", return_value=dict(identity=dict(phase="train"),progress={})), \
                 patch("tdyssey.native_rgb_evaluation.final_protocol_identity", return_value=dict(frozen="all-identities")) as identity, \
                 patch("scripts.native_rgb_world_model.physical_evaluation_identity",return_value=dict(physical_frozen=True)):
                result = freeze_model(config(), root, checkpoint, root/"external-data")
                self.assertEqual(identity.call_args.args[-1], dict(frozen_calibration="seen-only"))
                self.assertEqual(result["evaluation_identity"], dict(frozen="all-identities"))
                self.assertEqual(result["physical_evaluation_identity"],dict(physical_frozen=True))
                saved = (root/"frozen-model.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    freeze_model(config(), root, checkpoint, root/"external-data")
                self.assertEqual((root/"frozen-model.json").read_bytes(), saved)

    def test_cli_new_seen_checkpoint_keeps_original_report_and_reuses_calibration(self):
        with temporary_directory() as directory:
            root = Path(directory)
            output, data, prior = make_metadata(root)
            original_report = (output/"evaluation/seen_val/report.json").read_bytes()
            write_json(output/"evaluation/seen_val/calibration.json", dict(renderer_only="frozen"))
            checkpoint = output/"training/best.pt"
            checkpoint.write_bytes(b"new checkpoint to evaluate")
            target = output/"evaluation/seen_val"/("checkpoint-"+sha(checkpoint)[:12])
            def mocked_evaluate(cfg, path, selected_data, **kwargs):
                self.assertEqual(path,target)
                self.assertEqual(selected_data,data.resolve())
                self.assertEqual(kwargs["calibration"],dict(renderer_only="frozen"))
                self.assertFalse(kwargs["allow_unseen"])
                write_json(path/"report.json",dict(checkpoint_sha256=sha(checkpoint)))
                return dict(status="mocked_without_inference")
            with patch("scripts.native_rgb_world_model.load_config",return_value=config()), \
                 patch("scripts.native_rgb_world_model.evaluate_with_physical",side_effect=mocked_evaluate):
                result = main(["evaluate","--output",str(output),"--data",str(data),
                    "--config","unused-config","--physics-root","unused-physics",
                    "--gpc-root","unused-gpc","--init-checkpoint","unused-original"])
            self.assertEqual(result["status"],"mocked_without_inference")
            self.assertEqual((output/"evaluation/seen_val/report.json").read_bytes(),original_report)
            self.assertEqual(current_validation_directory(output),target.resolve())

    def test_microfit_checkpoint_cannot_be_frozen_for_final_test(self):
        with temporary_directory() as directory:
            root=Path(directory)
            with patch("tdyssey.native_rgb_training.load_checkpoint_payload",
                       return_value=dict(identity=dict(phase="microfit"),progress={})), \
                 patch("tdyssey.native_rgb_evaluation.final_protocol_identity") as identity:
                with self.assertRaisesRegex(ValueError,"mixed-shape"):
                    freeze_model(config(),root,root/"unused.pt",root/"unused-data")
                identity.assert_not_called()

    def test_wrapper_reuses_same_raw_after_models_close_then_clears_trace(self):
        import numpy as np
        from tdyssey.native_rgb_physical_metrics import PredictionTrace
        with temporary_directory() as directory:
            root=Path(directory);models=[];traces=[];order=[]
            write_json(root/"physical-measurement-calibration-final.json",
                       dict(palette_calibration=dict(sha256="palette"),physical_measurement_calibrated=True))
            class Model:
                def __init__(self,*args,**kwargs):
                    self.closed=False;self.calls=0;self.number=len(models);models.append(self)
                def __call__(self,history,actions,seed):
                    self.calls+=1;self.raw=np.full((3,96,96),.2+self.number*.1,dtype=np.float32)
                    order.append("predict")
                    return self.raw
                def close(self):self.closed=True;order.append("close")
            def instrument(model,**kwargs):
                trace=PredictionTrace(model,**kwargs);traces.append(trace);return trace
            def core_evaluate(cfg,path,data,**kwargs):
                history=np.ones((4,3,96,96),dtype=np.float32);actions=np.zeros((4,2),dtype=np.float32)
                for trace in kwargs["predictors"].values():
                    result=trace(history,actions,10)
                    self.assertIs(result,trace.predictor.raw)
                return dict(status="mock_rgb_completed",checkpoint_sha256="mock-checkpoint",
                            data_manifest_sha256="mock-data",rows=[dict(window="same-raw",method="fine_tuned")])
            def physical_score(rgb,data,passed_traces,calibration,**kwargs):
                order.append("score")
                self.assertTrue(all(model.closed for model in models))
                for trace in passed_traces.values():
                    np.testing.assert_array_equal(trace.raw(0),trace.predictor.raw)
                return dict(physical_dynamics_verified=False)
            with patch("tdyssey.native_rgb_evaluation.NativePredictor",Model), \
                 patch("tdyssey.native_rgb_evaluation.validate_calibration",side_effect=lambda value:value), \
                 patch("tdyssey.native_rgb_evaluation.evaluate",side_effect=core_evaluate), \
                 patch("tdyssey.native_rgb_physical_metrics.PredictionTrace",side_effect=instrument), \
                 patch("tdyssey.native_rgb_physical_metrics.physical_metric_identity",return_value=dict(physical_measurement_calibrated=True)), \
                 patch("tdyssey.native_rgb_physical_metrics.score_from_trace",side_effect=physical_score), \
                 patch("tdyssey.native_rgb_data.NativeRGBDataset",return_value=object()):
                result=evaluate_with_physical(config(),root/"evaluation/seen_val",root/"unused-data",output=root,
                    source_root="unused",physics_source_root="unused",init_checkpoint="unused",checkpoint="unused",
                    split="seen_val",calibration=dict(sha256="palette"))
            self.assertTrue(result["raw_predictions_scored_without_rerun_or_repair"])
            self.assertTrue(all(model.calls==1 and model.closed for model in models))
            self.assertTrue(all(trace.retained_bytes==0 and not trace.records for trace in traces))
            self.assertEqual(order,["predict","predict","close","close","score"])
            physical=json.loads((root/"evaluation/seen_val/physical-report.json").read_text(encoding="utf-8"))
            self.assertEqual(physical["checkpoint_sha256"],result["checkpoint_sha256"])
            self.assertEqual(physical["data_manifest_sha256"],result["data_manifest_sha256"])
            self.assertEqual(physical["core_rows_sha256"],rows_sha(result["rows"]))

    def test_wrapper_missing_physical_seal_rejects_before_final_dataset_or_model(self):
        with temporary_directory() as directory:
            root=Path(directory);cfg=dict(config(),server=True)
            write_json(root/"physical-measurement-calibration-final.json",
                       dict(palette_calibration=dict(sha256="palette"),physical_measurement_calibrated=True))
            with patch("tdyssey.native_rgb_evaluation.final_protocol_identity",return_value=dict(core="frozen")), \
                 patch("tdyssey.native_rgb_evaluation.validate_calibration",side_effect=lambda value:value), \
                 patch("tdyssey.native_rgb_physical_metrics.physical_metric_identity",return_value=dict(physical_measurement_calibrated=True)), \
                 patch("tdyssey.native_rgb_data.NativeRGBDataset",side_effect=AssertionError("Final data must stay sealed")) as data, \
                 patch("tdyssey.native_rgb_evaluation.NativePredictor",side_effect=AssertionError("No model may load")) as model:
                with self.assertRaisesRegex(ValueError,"physical instrument"):
                    evaluate_with_physical(cfg,root/"evaluation/unseen_test",root/"unused-data",output=root,
                        source_root="unused",physics_source_root="unused",init_checkpoint="unused",checkpoint="unused",
                        split="unseen_test",allow_unseen=True,calibration=dict(sha256="palette"),
                        frozen_identity=dict(core="frozen"),frozen_physical_identity=None)
                data.assert_not_called();model.assert_not_called()

    def test_wrapper_cleans_all_created_models_and_traces_on_core_error(self):
        from tdyssey.native_rgb_physical_metrics import PredictionTrace
        with temporary_directory() as directory:
            root=Path(directory);models=[];traces=[]
            write_json(root/"physical-measurement-calibration-final.json",
                       dict(palette_calibration=dict(sha256="palette"),physical_measurement_calibrated=True))
            class Model:
                def __init__(self,*args,**kwargs):self.closed=0;models.append(self)
                def close(self):self.closed+=1
            def instrument(model,**kwargs):
                trace=PredictionTrace(model,**kwargs);traces.append(trace);return trace
            with patch("tdyssey.native_rgb_evaluation.NativePredictor",Model), \
                 patch("tdyssey.native_rgb_evaluation.validate_calibration",side_effect=lambda value:value), \
                 patch("tdyssey.native_rgb_evaluation.evaluate",side_effect=RuntimeError("mock inference error")), \
                 patch("tdyssey.native_rgb_physical_metrics.PredictionTrace",side_effect=instrument), \
                 patch("tdyssey.native_rgb_physical_metrics.physical_metric_identity",return_value=dict(physical_measurement_calibrated=True)), \
                 patch("tdyssey.native_rgb_physical_metrics.score_from_trace",side_effect=AssertionError("Cannot score absent predictions")) as scorer:
                with self.assertRaisesRegex(RuntimeError,"mock inference error"):
                    evaluate_with_physical(config(),root/"evaluation/seen_val",root/"unused-data",output=root,
                        source_root="unused",physics_source_root="unused",init_checkpoint="unused",checkpoint="unused",
                        split="seen_val",calibration=dict(sha256="palette"))
                scorer.assert_not_called()
            self.assertEqual(len(models),2)
            self.assertTrue(all(model.closed==1 for model in models))
            self.assertTrue(all(trace.retained_bytes==0 and not trace.records for trace in traces))


if __name__ == "__main__":
    unittest.main()
