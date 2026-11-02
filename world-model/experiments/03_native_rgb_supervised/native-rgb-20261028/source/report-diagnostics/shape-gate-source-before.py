"""Explicit stages for the original GPC RGB experiment; imports/help never train."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path, output):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    from tdyssey.native_rgb_data import TRAIN_LETTERS, UNSEEN_LETTERS
    if tuple(config["train_letters"]) != TRAIN_LETTERS or tuple(config["unseen_letters"]) != UNSEEN_LETTERS:
        raise ValueError("The approved whole-letter split is fixed before training")
    config["artifact_root"] = str(Path(output).resolve())
    config["ledger_path"] = str(Path(output).resolve() / "native-rgb-cost-ledger.json")
    return config


def preflight(config, output, physics_root, gpc_root, init_checkpoint, gpu=False):
    from tdyssey.gpc_world_model import world_model_spec
    from tdyssey.native_rgb_env import rotational_split_audit, load_physics_source, CONTROL_PROTOCOL, PROTOCOL_SHA256
    from tdyssey.native_rgb_training import atomic_json, sha256, disk_bytes
    from tdyssey.source_export import verify_source
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec = world_model_spec()
    if (Path(init_checkpoint).stat().st_size != spec["checkpoint"]["bytes"]
            or sha256(init_checkpoint) != spec["checkpoint"]["sha256"]):
        raise ValueError("Only the original released GPC initialization is allowed")
    source = verify_source(gpc_root, spec, "GPC")
    load_physics_source(physics_root)
    split = rotational_split_audit(config["train_letters"], config["unseen_letters"])
    if not split["passed"]: raise ValueError("Rotationally identical geometry crosses held-letter split")
    disk = shutil.disk_usage(output)
    if disk.free < config["disk_reserve_bytes"] + 256 * 1024**2:
        raise RuntimeError("Insufficient free disk for bounded native RGB work")
    result = dict(status="passed", architecture="original_full_GPC_96px_RGB", source=source,
        initialization=spec["checkpoint"], split_audit=split, controller=CONTROL_PROTOCOL,
        controller_sha256=PROTOCOL_SHA256, free_disk_bytes=disk.free,
        current_artifact_bytes=disk_bytes(output), platform=platform.platform(),
        python=sys.version, gpu_checked=False, final_unseen_predictions_inspected=False,
        shape_supervision=False, training_objective="original_phase_two_H1_residual_mse")
    if gpu:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        capability = torch.cuda.get_device_capability(device)
        # A real kernel catches an incompatible wheel even when is_available() is true.
        x = torch.ones((8, 8), device="cuda", requires_grad=True)
        (x @ x).sum().backward(); torch.cuda.synchronize()
        if config.get("server") and (platform.system() != "Linux" or total < 60 * 1024**3):
            raise RuntimeError("Server configuration requires Linux and the intended large GPU")
        result.update(gpu_checked=True, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device), capability=list(capability),
            wheel_architectures=torch.cuda.get_arch_list(), free_cuda_bytes=free, total_cuda_bytes=total,
            preflight_optimizer_updates=0)
        del x; torch.cuda.empty_cache()
    atomic_json(output / "preflight.json", result)
    atomic_json(output / "run-config.json", config)
    return result


def current_validation_directory(output):
    output = Path(output).resolve()
    pointer = output / "evaluation/current-seen-val.json"
    if not pointer.exists(): return output / "evaluation/seen_val"
    row = json.loads(pointer.read_text(encoding="utf-8"))
    directory = (output / row["directory"]).resolve()
    if not directory.is_relative_to(output / "evaluation"):
        raise ValueError("Validation pointer escapes this experiment")
    return directory


def physical_evaluation_identity(calibration):
    from tdyssey.native_rgb_physical_metrics import physical_metric_identity
    from tdyssey.native_rgb_training import sha256
    return dict(**physical_metric_identity(calibration), integration_source_sha256=sha256(__file__))


def evaluation_rows_sha256(rows):
    from tdyssey.native_rgb_evaluation import _json_value
    payload = json.dumps(_json_value(rows), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def evaluate_with_physical(config, evaluation_output, data, *, output, source_root,
        physics_source_root, init_checkpoint, checkpoint, split, allow_unseen=False,
        calibration=None, frozen_identity=None, frozen_physical_identity=None):
    """Score the same native RGB outputs twice; neither instrument changes them."""
    from tdyssey.native_rgb_data import NativeRGBDataset
    from tdyssey.native_rgb_evaluation import (NativePredictor, evaluate, calibrate_renderer,
                                               final_protocol_identity, validate_calibration)
    from tdyssey.native_rgb_physical_metrics import (PredictionTrace, calibrate_physical_measurement,
                                                    score_from_trace, physical_metric_identity)
    from tdyssey.native_rgb_training import atomic_json
    output = Path(output); evaluation_output = Path(evaluation_output)
    final = split != "seen_val"
    if final:
        if not allow_unseen or not config.get("server") or calibration is None:
            raise ValueError("Final evaluation requires the frozen server protocol")
        if frozen_identity != final_protocol_identity(config, checkpoint, data, calibration):
            raise ValueError("Final core identity changed before reading final images")
    if calibration is None:
        seen = NativeRGBDataset(data, "seen_val", source_root=physics_source_root)
        calibration = calibrate_renderer(seen)
        atomic_json(output / "frozen-renderer-calibration.json", calibration)
    calibration = validate_calibration(calibration)
    physical_path = output / "physical-measurement-calibration-final.json"
    if physical_path.exists():
        physical_calibration = json.loads(physical_path.read_text(encoding="utf-8"))
    else:
        if final: raise ValueError("Final tests cannot fit a new measurement calibration")
        seen = NativeRGBDataset(data, "seen_val", source_root=physics_source_root)
        physical_calibration = calibrate_physical_measurement(seen, calibration, output=physical_path)
    physical_metric_identity(physical_calibration)
    if physical_calibration["palette_calibration"]["sha256"] != calibration["sha256"]:
        raise ValueError("Physical and RGB measurement palettes identify different frozen calibrations")
    if final and frozen_physical_identity != physical_evaluation_identity(physical_calibration):
        raise ValueError("Final physical instrument/integration identity changed before final images")
    models, traces = [], {}
    try:
        for name, weights in (("frozen_gpc", None), ("fine_tuned", checkpoint)):
            model = NativePredictor(source_root, init_checkpoint, weights, device=config.get("device", "cuda"))
            models.append(model)
            traces[name] = PredictionTrace(model, max_bytes=(768 if config.get("server") else 128)*1024**2)
        rgb = evaluate(config, evaluation_output, data, source_root=source_root,
            physics_source_root=physics_source_root, init_checkpoint=init_checkpoint,
            checkpoint=checkpoint, split=split, allow_unseen=allow_unseen,
            calibration=calibration, frozen_identity=frozen_identity, predictors=traces)
        # Release CUDA before the CPU-only geometric measurement; raw copies remain bounded.
        for model in models: model.close()
        models.clear()
        scored_data = NativeRGBDataset(data, split, source_root=physics_source_root)
        physical = score_from_trace(rgb, scored_data, traces, physical_calibration,
            base_seed=config["seed"], output=evaluation_output, allow_final=final)
        physical.update(checkpoint_sha256=rgb["checkpoint_sha256"],
            data_manifest_sha256=rgb["data_manifest_sha256"], core_rows_sha256=evaluation_rows_sha256(rgb["rows"]))
        atomic_json(evaluation_output / "physical-report.json", physical)
        rgb["physical_dynamics_verified"] = physical["physical_dynamics_verified"]
        rgb["physical_report"] = "physical-report.json"
        rgb["physical_evaluation_identity"] = physical_evaluation_identity(physical_calibration)
        rgb["raw_predictions_scored_without_rerun_or_repair"] = True
        atomic_json(evaluation_output / "report.json", rgb)
        return rgb
    finally:
        for model in models: model.close()
        for trace in traces.values(): trace.clear()


def report(config, output, data_root=None):
    from tdyssey.native_rgb_training import atomic_json, disk_bytes, sha256
    output = Path(output)
    data_root = Path(data_root or output / "data").resolve()
    files = ["preflight.json", "budget.json", "physics-probe/report.json", "data/physics-audit.json",
             "data/coverage/summary.json", "data-qualification.json", "engineering/report.json", "microfit/result.json",
             "training/result.json", "native-rgb-cost-ledger.json", "evaluation/seen_val/report.json",
             "evaluation/seen_val/physical-report.json", "training-recovery-plan.json",
             "recovery-io-retry/migrated-step-750.json", "recovery-io-retry/resume-overlap.json",
             "frozen-renderer-calibration.json", "physical-measurement-calibration-final.json"]
    evidence, evidence_paths = {}, {}
    for relative in files:
        path = output / relative
        if relative.startswith("data/"): path = data_root / relative.removeprefix("data/")
        if relative.startswith("evaluation/seen_val/"): path = current_validation_directory(output) / Path(relative).name
        if path.exists():
            evidence[relative] = json.loads(path.read_text(encoding="utf-8"))
            evidence_paths[relative] = os.path.relpath(path, output).replace("\\", "/")
    audit = evidence.get("data/physics-audit.json", {})
    evaluation = evidence.get("evaluation/seen_val/report.json", {})
    qualification = evaluation.get("qualification", {}).get("fine_tuned", {})
    h1 = qualification.get("1", {})
    h4 = qualification.get("4", {})
    tiny = evidence.get("microfit/result.json", {}).get("progress", {})
    fit_passed = (tiny.get("fixed_train_loss_after") is not None and tiny.get("fixed_train_loss_before") is not None
                  and tiny["fixed_train_loss_after"] < tiny["fixed_train_loss_before"])
    selected_checkpoint = output / "training/best.pt"
    current_model = (selected_checkpoint.exists() and evaluation.get("checkpoint_sha256") == sha256(selected_checkpoint))
    current_data = ((data_root / "manifest.json").exists()
                    and evaluation.get("data_manifest_sha256") == sha256(data_root / "manifest.json"))
    apparent_gate = bool(h1.get("motion_passed") and h4.get("motion_passed")
        and h1.get("prediction_failures") == 0 and h4.get("prediction_failures") == 0)
    physical_evidence = evidence.get("evaluation/seen_val/physical-report.json", {})
    physical_identity_matches = (bool(evaluation.get("checkpoint_sha256"))
        and physical_evidence.get("checkpoint_sha256") == evaluation.get("checkpoint_sha256")
        and physical_evidence.get("data_manifest_sha256") == evaluation.get("data_manifest_sha256")
        and physical_evidence.get("core_rows_sha256") == evaluation_rows_sha256(evaluation.get("rows", [])))
    physical_verified = (physical_evidence.get("physical_dynamics_verified") is True
        and physical_evidence.get("physical_measurement_calibrated") is True and physical_identity_matches)
    result = dict(kind="native-gpc-rgb-local-report.v1", evidence=evidence, evidence_paths=evidence_paths,
        artifact_bytes=disk_bytes(output), max_artifact_bytes=config["max_artifact_bytes"],
        physics_passed=audit.get("physics_collection_gate_passed") is True,
        qualified_for_server_scaleup=bool(audit.get("physics_collection_gate_passed") is True
            and evidence.get("engineering/report.json", {}).get("passed") is True and fit_passed and current_model and current_data
            and apparent_gate and physical_verified
            and h1.get("shape_assessable", 0) > 0 and h1.get("shape_failures") == 0),
        no_local_unseen_evaluation=True, final_unseen_letters=config["unseen_letters"],
        validation_matches_current_best=bool(current_model), validation_matches_current_data=bool(current_data),
        required_motion_horizons=[1,4], required_relative_motion_improvement=0.20,
        apparent_motion_gate_passed=apparent_gate, physical_dynamics_verified=physical_verified,
        physical_report_matches_current_rgb_rows=bool(physical_identity_matches),
        note="A failed data, implementation, tiny-fit or validation gate stops local advancement.")
    atomic_json(output / "local-report.json", result)
    text = ["# Original GPC RGB local evidence", "", "This report contains measured outputs only.", "",
        f"Physical collection gate: **{result['physics_passed']}**.",
        f"Qualified for server scale-up: **{result['qualified_for_server_scaleup']}**.",
        "No local prediction results from the eight final unseen letters were inspected.", "",
        "## Evidence", ""]
    text += [f"- [{relative}]({evidence_paths[relative]})" for relative in evidence]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def freeze_model(config, output, checkpoint, data):
    from tdyssey.native_rgb_training import atomic_json, sha256, load_checkpoint_payload
    output = Path(output); seal = output / "frozen-model.json"
    if seal.exists(): raise FileExistsError("Model is already frozen; do not replace the final-test identity")
    payload = load_checkpoint_payload(checkpoint)
    if payload.get("identity", {}).get("phase") not in ("train", "main"):
        raise ValueError("Final model must be from mixed-shape training, not engineering or microfit")
    from tdyssey.native_rgb_evaluation import final_protocol_identity
    calibration_path = current_validation_directory(output) / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    result = dict(kind="native-gpc-rgb-frozen.v1", checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint), identity=payload["identity"], progress=payload["progress"],
        unseen_letters=config["unseen_letters"], final_test_used=False,
        evaluation_identity=final_protocol_identity(config, checkpoint, data, calibration),
        physical_evaluation_identity=physical_evaluation_identity(
            json.loads((output / "physical-measurement-calibration-final.json").read_text(encoding="utf-8"))))
    atomic_json(seal, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "probe", "collect", "audit", "engineering",
        "microfit", "train", "evaluate", "report", "freeze"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native_rgb_local.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native-rgb-20261028")
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--gpc-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=["seen_val", "seen_test", "unseen_test"], default="seen_val")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--updates", type=int)
    args = parser.parse_args(argv)
    output = args.output.resolve(); data = (args.data or output / "data").resolve()
    config = load_config(args.config, output)
    from tdyssey.native_rgb_training import atomic_json
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "preflight":
            return preflight(config, output, args.physics_root, args.gpc_root, args.init_checkpoint, args.gpu)
        if args.stage == "probe":
            from tdyssey.native_rgb_data import physics_probe
            return physics_probe(config, output, args.physics_root)
        if args.stage == "collect":
            from tdyssey.native_rgb_data import collect_dataset
            splits = ("train", "seen_val", "seen_test", "unseen_test") if config.get("server") else ("train", "seen_val")
            if config.get("server") and config.get("collection", {}).get("workers", 1) > 1:
                from tdyssey.native_rgb_parallel import collect_dataset_parallel
                return collect_dataset_parallel(config, data, args.physics_root, splits=splits)
            return collect_dataset(config, data, args.physics_root, splits=splits)
        if args.stage == "audit":
            from tdyssey.native_rgb_data import audit_dataset
            result = audit_dataset(data)
            if not result.get("physics_collection_gate_passed"): raise RuntimeError("Physical collection gate failed")
            return result
        if args.stage == "engineering":
            from scripts.native_rgb_engineering import run_engineering
            return run_engineering(config, output, data, args.physics_root, args.gpc_root, args.init_checkpoint)
        if args.stage in ("microfit", "train"):
            from tdyssey.native_rgb_training import train
            from tdyssey.native_rgb_data import NativeRGBDataset
            if (output / "frozen-model.json").exists(): raise RuntimeError("Frozen final-test model cannot continue training")
            if not (output / "engineering/report.json").exists(): raise RuntimeError("Complete native engineering gate first")
            engineering = json.loads((output / "engineering/report.json").read_text())
            if engineering.get("passed") is not True: raise RuntimeError("Native engineering gate failed")
            indices = None
            if args.stage == "microfit":
                dataset = NativeRGBDataset(data, "train", source_root=args.physics_root)
                first = {}
                for index, (trajectory, _) in enumerate(dataset.windows):
                    first.setdefault(dataset.entries[trajectory]["shape"], index)
                indices = [first[letter] for letter in sorted(first)[:16]]
                target = args.updates or config["microfit_updates"]
            else:
                tiny = json.loads((output / "microfit/result.json").read_text())
                progress = tiny["progress"]
                if not progress["fixed_train_loss_after"] < progress["fixed_train_loss_before"]:
                    raise RuntimeError("Fixed-sample microfit did not improve; inspect locally")
                target = args.updates
            run_output = output / ("microfit" if args.stage == "microfit" else "training")
            from tdyssey.native_rgb_run_control import monitor_training
            control_config = config if args.stage == "train" else dict(config, server=False)
            with monitor_training(control_config, run_output, artifact_root=output):
                return train(config, run_output, data, args.gpc_root, args.init_checkpoint,
                    physics_source_root=args.physics_root, execute=True, phase=args.stage,
                    updates=target, indices=indices, resume=args.resume)
        if args.stage == "evaluate":
            checkpoint = args.checkpoint or output / "training/best.pt"
            final = args.split != "seen_val"
            calibration = frozen_identity = frozen_physical_identity = None
            if not final and (output / "frozen-renderer-calibration.json").exists():
                calibration = json.loads((output / "frozen-renderer-calibration.json").read_text(encoding="utf-8"))
            evaluation_output = output / "evaluation" / args.split
            if not final and (evaluation_output / "report.json").exists():
                from tdyssey.native_rgb_training import sha256
                prior = json.loads((evaluation_output / "report.json").read_text(encoding="utf-8"))
                if prior.get("checkpoint_sha256") != sha256(checkpoint):
                    calibration = json.loads((evaluation_output / "calibration.json").read_text(encoding="utf-8"))
                    evaluation_output = evaluation_output / ("checkpoint-" + sha256(checkpoint)[:12])
            if final:
                if not config.get("server"): raise RuntimeError("Final test predictions are prohibited in the local pilot")
                seal = json.loads((output / "frozen-model.json").read_text())
                from tdyssey.native_rgb_training import sha256
                if sha256(checkpoint) != seal["checkpoint_sha256"]: raise RuntimeError("Final-test checkpoint differs from frozen identity")
                calibration = json.loads((current_validation_directory(output) / "calibration.json").read_text(encoding="utf-8"))
                frozen_identity = seal["evaluation_identity"]
                frozen_physical_identity = seal["physical_evaluation_identity"]
            result = evaluate_with_physical(config, evaluation_output, data, output=output,
                source_root=args.gpc_root, physics_source_root=args.physics_root,
                init_checkpoint=args.init_checkpoint, checkpoint=checkpoint, split=args.split,
                allow_unseen=final, calibration=calibration, frozen_identity=frozen_identity,
                frozen_physical_identity=frozen_physical_identity)
            if not final:
                from tdyssey.native_rgb_training import sha256
                atomic_json(output / "evaluation/current-seen-val.json", dict(
                    directory=str(evaluation_output.relative_to(output)).replace("\\", "/"),
                    checkpoint_sha256=sha256(checkpoint), report_sha256=sha256(evaluation_output / "report.json")))
            return result
        if args.stage == "freeze":
            return freeze_model(config, output, args.checkpoint or output / "training/best.pt", data)
        if args.stage == "report": return report(config, output, data)
    except BaseException as error:
        atomic_json(output / (args.stage + "-error.json"), dict(error=type(error).__name__ + ": " + str(error)))
        try: report(config, output, data)
        except Exception: pass
        raise


if __name__ == "__main__":
    result = main()
    display = dict(stage=sys.argv[1], status=result.get("status", "completed"))
    for key in ("passed", "physics_collection_gate_passed", "physics_probe_gate_passed", "qualified_for_server_scaleup"):
        if key in result: display[key] = result[key]
    if "progress" in result: display["updates"] = result["progress"]["updates"]
    if "manifest" in result: display["actual_controls"] = result["manifest"].get("actual_controls")
    print(json.dumps(display, indent=2, default=str))
