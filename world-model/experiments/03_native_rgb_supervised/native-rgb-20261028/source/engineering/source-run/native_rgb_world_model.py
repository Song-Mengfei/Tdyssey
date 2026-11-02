"""Explicit stages for the original GPC RGB experiment; imports/help never train."""
import argparse
from copy import deepcopy
import hashlib
import json
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


def report(config, output):
    from tdyssey.native_rgb_training import atomic_json, disk_bytes
    output = Path(output)
    files = ["preflight.json", "budget.json", "physics-probe/report.json", "data/physics-audit.json",
             "data/coverage/coverage.json", "engineering/report.json", "microfit/result.json",
             "training/result.json", "native-rgb-cost-ledger.json", "evaluation/seen_val/summary.json",
             "evaluation/seen_val/report.json"]
    evidence = {}
    for relative in files:
        path = output / relative
        if path.exists(): evidence[relative] = json.loads(path.read_text(encoding="utf-8"))
    audit = evidence.get("data/physics-audit.json", {})
    qualification = evidence.get("evaluation/seen_val/report.json", {}).get("qualification", {}).get("fine_tuned", {})
    h1 = qualification.get("1", {})
    result = dict(kind="native-gpc-rgb-local-report.v1", evidence=evidence,
        artifact_bytes=disk_bytes(output), max_artifact_bytes=config["max_artifact_bytes"],
        physics_passed=audit.get("physics_collection_gate_passed") is True,
        qualified_for_server_scaleup=bool(h1.get("motion_passed") and h1.get("prediction_failures") == 0
            and h1.get("shape_assessable", 0) > 0 and h1.get("shape_failures") == 0),
        no_local_unseen_evaluation=True, final_unseen_letters=config["unseen_letters"],
        note="A failed data, implementation, tiny-fit or validation gate stops local advancement.")
    atomic_json(output / "local-report.json", result)
    text = ["# Original GPC RGB local evidence", "", "This report contains measured outputs only.", "",
        f"Physical collection gate: **{result['physics_passed']}**.",
        f"Qualified for server scale-up: **{result['qualified_for_server_scaleup']}**.",
        "No local prediction results from the eight final unseen letters were inspected.", "",
        "## Evidence", ""]
    text += [f"- [{relative}]({relative})" for relative in evidence]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def freeze_model(config, output, checkpoint):
    from tdyssey.native_rgb_training import atomic_json, sha256, load_checkpoint_payload
    output = Path(output); seal = output / "frozen-model.json"
    if seal.exists(): raise FileExistsError("Model is already frozen; do not replace the final-test identity")
    payload = load_checkpoint_payload(checkpoint)
    result = dict(kind="native-gpc-rgb-frozen.v1", checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint), identity=payload["identity"], progress=payload["progress"],
        unseen_letters=config["unseen_letters"], final_test_used=False)
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
            return train(config, output / ("microfit" if args.stage == "microfit" else "training"),
                data, args.gpc_root, args.init_checkpoint, physics_source_root=args.physics_root,
                execute=True, phase=args.stage, updates=target, indices=indices, resume=args.resume)
        if args.stage == "evaluate":
            from tdyssey.native_rgb_evaluation import evaluate
            checkpoint = args.checkpoint or output / "training/best.pt"
            final = args.split != "seen_val"
            if final:
                if not config.get("server"): raise RuntimeError("Final test predictions are prohibited in the local pilot")
                seal = json.loads((output / "frozen-model.json").read_text())
                from tdyssey.native_rgb_training import sha256
                if sha256(checkpoint) != seal["checkpoint_sha256"]: raise RuntimeError("Final-test checkpoint differs from frozen identity")
            return evaluate(config, output / "evaluation" / args.split, data,
                source_root=args.gpc_root, physics_source_root=args.physics_root,
                init_checkpoint=args.init_checkpoint, checkpoint=checkpoint, split=args.split,
                allow_unseen=final)
        if args.stage == "freeze":
            return freeze_model(config, output, args.checkpoint or output / "training/best.pt")
        if args.stage == "report": return report(config, output)
    except BaseException as error:
        atomic_json(output / (args.stage + "-error.json"), dict(error=type(error).__name__ + ": " + str(error)))
        try: report(config, output)
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
