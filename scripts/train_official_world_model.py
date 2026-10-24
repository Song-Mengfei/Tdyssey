"""Collect official T play, then fine-tune the GPC image denoiser.

Collection finishes and is validated before any weight update. After each round
the logged validation error decides whether to collect another disjoint batch,
continue from the saved weights, or stop. --max-rounds 1 records that choice
without acting on it. This does not train a policy.
"""

import argparse
from importlib import metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.collection import integer
from tdyssey.dataset import digest
from tdyssey.envs.official import PROTOCOL, official_shape
from tdyssey.gpc_denoiser_training import GPCDenoiserTrainer, dataset_name, decide, limit_round
from tdyssey.official import official_spec
from tdyssey.official_collection import DEVELOPMENT_SEEDS, ENVIRONMENT_FAMILY, VERSION, OfficialCollectionConfig, validate_official_dataset
from tdyssey.official_collector import OfficialShapeCollector, _quiet_sdl
from tdyssey.official_contact_sampler import preserve_init_checkpoint


def provenance(config):
    paths = ["tdyssey/official_collector.py", "tdyssey/official_collection.py", "tdyssey/gpc_denoiser_training.py",
             "tdyssey/gpc_world_model.py", "scripts/train_official_world_model.py", "manifests/gpc_world_model.json",
             "manifests/official_dp.json"]
    spec = official_spec()
    return {"environment_family": ENVIRONMENT_FAMILY, "collector_version": VERSION,
            "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "project_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
            "source_sha256": {name: digest((ROOT / name).read_bytes()) for name in paths},
            "upstream_commit": spec["source_commit"], "geometry_hash": official_shape(config.shape).geometry_hash,
            "physics_protocol": PROTOCOL, "python": platform.python_version(),
            "platform": f"{sys.platform}-{platform.machine()}",
            "packages": {item.metadata["Name"]: item.version for item in metadata.distributions()},
            "action_prior": "object-directed uses privileged object position; no reward optimization",
            "excluded_evaluation_reset_seeds": list(config.excluded_reset_seeds)}


def train_collected(args, datasets, checkpoint=None, updates=None, model_dir=None):
    """Score the current weights, apply one bounded update chunk, then score again."""
    trainer = GPCDenoiserTrainer(args.gpc_root, checkpoint or args.checkpoint,
                                 updates=args.updates if updates is None else updates,
                                 batch_size=args.batch_size, seed=args.seed, seconds=args.train_seconds)
    return trainer.run(datasets, model_dir or (args.output / "model"))


def _collect(args, config, root):
    OfficialShapeCollector(config, args.source_root, args.workers).collect(
        root, provenance(config), time.monotonic() + config.seconds)
    manifest, _episodes = validate_official_dataset(root)
    return manifest


def _write(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def execute(args, config):
    _quiet_sdl()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    integer(args.max_rounds, "max rounds", 1, 200)
    args.output.mkdir()
    datasets, rounds = [], []
    checkpoint, steps, seed = None, config.steps, config.seed
    contact_steps = total_updates = non_improving = 0
    latest_manifest = latest_training = None
    last_round_seconds = None
    caps = {"max_collections": args.max_collections, "max_updates": args.max_updates, "patience": args.patience}
    report = {"schema_version": 1, "kind": "official_t_gpc_denoiser_pipeline", "status": "running",
              "shape": config.shape, "model_test_used": False,
              "limits": {"episodes": config.episodes, "steps": config.steps, "updates": args.updates,
                         "workers": args.workers, "max_rounds": args.max_rounds,
                         "budget_seconds": args.budget_seconds, **caps}}
    started = time.monotonic()
    try:
        while True:
            if rounds and args.budget_seconds:
                spent = time.monotonic() - started
                remaining_time = args.budget_seconds - spent
                if remaining_time <= 0 or (last_round_seconds is not None and remaining_time < last_round_seconds):
                    last = rounds[-1]
                    if last["executed"] != "stop":
                        last["executed"] = "stop"
                        last["executed_reason"] = f"time budget; logged choice was {last['action']}: {last['reason']}"
                    report.update(_summary(latest_manifest, latest_training, datasets, rounds, contact_steps, started))
                    _write(args.output / "report.json", report)
                    break
            round_started = time.monotonic()
            used_steps = steps
            if len(datasets) == 0 or (rounds and rounds[-1]["executed"] == "collect"):
                root = args.output / dataset_name(len(datasets) + 1)
                manifest = _collect(args, OfficialCollectionConfig(
                    shape=config.shape, episodes=config.episodes, steps=steps, seed=seed,
                    mode="object-directed", seconds=config.seconds, excluded_reset_seeds=DEVELOPMENT_SEEDS), root)
                datasets.append(root)
                contact_steps += sum(entry["contact_steps"] for entry in manifest["episodes"])
                latest_manifest = manifest
            remaining = min(args.updates, args.max_updates - total_updates)
            if remaining < 1:
                last = rounds[-1]
                if last["executed"] != "stop":
                    last["executed"] = "stop"
                    last["executed_reason"] = f"update cap; logged choice was {last['action']}: {last['reason']}"
                report.update(_summary(latest_manifest, latest_training, datasets, rounds, contact_steps, started))
                _write(args.output / "report.json", report)
                break
            trained = train_collected(args, datasets, checkpoint, remaining)
            total_updates += trained["updates"]
            checkpoint = args.output / "model" / "denoiser.pt"
            latest_training = trained
            validation = trained["validation"]["adapted"]
            previous_mse = trained["validation"]["before"]["mse"]
            non_improving = 0 if validation["mse"] < previous_mse else non_improving + 1
            choice = decide({"contact_steps": contact_steps, "adapted_mse": validation["mse"],
                             "persistence_mse": validation["persistence_mse"],
                             "previous_mse": previous_mse, "non_improving_rounds": non_improving,
                             "collection_runs": len(datasets), "updates": total_updates,
                             "steps": used_steps, "episodes": config.episodes, "seed": seed}, caps)
            limited = limit_round(choice, len(rounds) + 1, args.max_rounds)
            rounds.append({"contact_steps": contact_steps, "steps": used_steps,
                           "collection_runs": len(datasets), "updates": total_updates,
                           "non_improving_rounds": non_improving,
                           "validation_mse": validation["mse"], "persistence_mse": validation["persistence_mse"],
                           "previous_mse": previous_mse, **limited})
            if limited["executed"] == "collect":
                steps, seed = choice["steps"], choice["seed"]
            last_round_seconds = time.monotonic() - round_started
            report.update(_summary(latest_manifest, latest_training, datasets, rounds, contact_steps, started))
            _write(args.output / "report.json", report)
            if limited["executed"] == "stop":
                break
        report["status"] = "complete"
        _write(args.output / "report.json", report)
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": time.monotonic() - started})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    return report


def _summary(manifest, trained, datasets, rounds, contact_steps, started):
    return {"status": "running", "model_test_used": trained["model_test_used"],
            "counts": manifest["counts"], "contact_steps": contact_steps,
            "splits": {name: len(ids) for name, ids in manifest["splits"].items()},
            "datasets": [path.name for path in datasets], "training": trained, "rounds": rounds,
            "elapsed_seconds": time.monotonic() - started}


def continue_training(args):
    """Keep updating saved weights on the existing datasets until the clock ends.

    The still-frame stop is not used. New weights go to model-30m and the earlier
    checkpoint stays in model/.
    """
    if not args.output.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {args.output}")
    names = ["dataset", "dataset-002", "dataset-003"]
    datasets = [args.output / name for name in names]
    missing = [str(path) for path in datasets if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing collected datasets: {missing}")
    checkpoint = args.output / "model" / "denoiser.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing saved denoiser: {checkpoint}")
    model_dir = args.output / "model-30m"
    integer(args.budget_seconds, "budget seconds", 1, 7200)
    prior_elapsed = 0
    rounds, total_updates = [], 0
    if model_dir.exists():
        saved_report = model_dir / "report.json"
        saved_weights = model_dir / "denoiser.pt"
        if not saved_report.is_file() or not saved_weights.is_file():
            raise FileExistsError(f"Continued weights already exist: {model_dir}")
        previous = json.loads(saved_report.read_text())
        if previous.get("status") == "complete":
            raise FileExistsError(f"Continued training is already complete: {model_dir}")
        checkpoint = saved_weights
        rounds = list(previous.get("rounds", []))
        total_updates = int(previous.get("updates", 0))
        prior_elapsed = float(previous.get("elapsed_seconds", 0))
    else:
        model_dir.mkdir()
    from tdyssey.dataset import read_manifest
    contact_steps = 0
    for root in datasets:
        manifest = read_manifest(root)
        contact_steps += sum(entry["contact_steps"] for entry in manifest["episodes"])
    latest = None
    last_round_seconds = None
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "official_t_gpc_denoiser_continuation", "status": "running",
              "datasets": names, "contact_steps": contact_steps, "model_test_used": False,
              "budget_seconds": args.budget_seconds + prior_elapsed, "prior_elapsed_seconds": prior_elapsed,
              "source_checkpoint": "artifacts/official-wm-T-hour-2026-10-23/model/denoiser.pt",
              "updates": total_updates, "rounds": rounds}
    try:
        while True:
            if rounds:
                remaining_time = args.budget_seconds - (time.monotonic() - started)
                if remaining_time <= 0 or (last_round_seconds is not None and remaining_time < last_round_seconds):
                    break
            round_started = time.monotonic()
            latest = train_collected(args, datasets, checkpoint, args.updates, model_dir)
            checkpoint = model_dir / "denoiser.pt"
            total_updates += latest["updates"]
            validation = latest["validation"]["adapted"]
            rounds.append({"updates": total_updates, "validation_mse": validation["mse"],
                           "persistence_mse": validation["persistence_mse"],
                           "previous_mse": latest["validation"]["before"]["mse"],
                           "weights_changed": latest["weights_changed"],
                           "model_test_used": latest["model_test_used"]})
            last_round_seconds = time.monotonic() - round_started
            report.update({"status": "running", "updates": total_updates, "rounds": rounds,
                           "training": latest, "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
            _write(model_dir / "report.json", report)
        report["status"] = "complete"
        report["elapsed_seconds"] = prior_elapsed + (time.monotonic() - started)
        _write(model_dir / "report.json", report)
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
        (model_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    return report


def train_named_datasets(args):
    """Update one checkpoint on an explicit dataset list until the clock ends."""
    datasets = [Path(path) for path in args.train_datasets]
    missing = [str(path) for path in datasets if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {missing}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    integer(args.budget_seconds, "budget seconds", 1, 7200)
    prior_elapsed = 0
    checkpoint = args.checkpoint
    rounds, total_updates, latest, last_round_seconds = [], 0, None, None
    if args.model_dir.exists():
        saved_report = args.model_dir / "report.json"
        saved_weights = args.model_dir / "denoiser.pt"
        if not saved_report.is_file() or not saved_weights.is_file():
            raise FileExistsError(args.model_dir)
        previous = json.loads(saved_report.read_text())
        if previous.get("status") == "complete":
            raise FileExistsError(f"Training is already complete: {args.model_dir}")
        checkpoint = saved_weights
        rounds = list(previous.get("rounds", []))
        total_updates = int(previous.get("updates", 0))
        prior_elapsed = float(previous.get("elapsed_seconds", 0))
    else:
        args.model_dir.mkdir(parents=True)
    from tdyssey.dataset import read_manifest
    contact_steps = 0
    for root in datasets:
        manifest = read_manifest(root)
        contact_steps += sum(entry["contact_steps"] for entry in manifest["episodes"])
    checkpoint = checkpoint
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "official_t_angle_coverage_retrain", "status": "running",
              "datasets": [path.name for path in datasets], "contact_steps": contact_steps,
              "model_test_used": False, "budget_seconds": args.budget_seconds + prior_elapsed,
              "prior_elapsed_seconds": prior_elapsed, "source_checkpoint": str(args.checkpoint),
              "updates": total_updates, "rounds": rounds}
    try:
        while True:
            if rounds:
                remaining_time = args.budget_seconds - (time.monotonic() - started)
                if remaining_time <= 0 or (last_round_seconds is not None and remaining_time < last_round_seconds):
                    break
            round_started = time.monotonic()
            latest = train_collected(args, datasets, checkpoint, args.updates, args.model_dir)
            checkpoint = args.model_dir / "denoiser.pt"
            total_updates += latest["updates"]
            validation = latest["validation"]["adapted"]
            rounds.append({"updates": total_updates, "validation_mse": validation["mse"],
                           "persistence_mse": validation["persistence_mse"],
                           "previous_mse": latest["validation"]["before"]["mse"],
                           "weights_changed": latest["weights_changed"],
                           "model_test_used": latest["model_test_used"]})
            last_round_seconds = time.monotonic() - round_started
            report.update({"status": "running", "updates": total_updates, "rounds": rounds,
                           "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
            _write(args.model_dir / "report.json", report)
            print(json.dumps(rounds[-1]), flush=True)
        report["status"] = "complete"
        report["elapsed_seconds"] = prior_elapsed + (time.monotonic() - started)
        _write(args.model_dir / "report.json", report)
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                       "elapsed_seconds": prior_elapsed + (time.monotonic() - started)})
        (args.model_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=("T",), default="T")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--train-seconds", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-collections", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--budget-seconds", type=int, default=0)
    parser.add_argument("--source-root", type=Path, default=ROOT / "third_party/diffusion_policy")
    parser.add_argument("--gpc-root", type=Path, default=ROOT / "third_party/gpc")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/gpc-world-model/denoiser.pth")
    parser.add_argument("--continue-training", action="store_true",
                        help="Update the saved denoiser on existing datasets until --budget-seconds")
    parser.add_argument("--train-datasets", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--init-from-best", action="store_true",
                        help="Copy the angles denoiser to artifacts/official-wm-T-init if needed and start from it")
    args = parser.parse_args(argv)
    if args.init_from_best:
        try:
            preserved = preserve_init_checkpoint(root=ROOT)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        args.checkpoint = preserved["checkpoint"]
    if args.train_datasets:
        if args.model_dir is None:
            parser.error("--model-dir is required with --train-datasets")
        try:
            integer(args.budget_seconds, "budget seconds", 1, 7200)
            integer(args.updates, "updates", 1, 20)
            integer(args.train_seconds, "train seconds", 1, 900)
        except ValueError as exc:
            parser.error(str(exc))
        report = train_named_datasets(args)
        last = report["rounds"][-1]
        print(json.dumps({"status": report["status"], "updates": report["updates"],
                          "elapsed_seconds": report["elapsed_seconds"],
                          "validation_mse": last["validation_mse"],
                          "persistence_mse": last["persistence_mse"],
                          "contact_steps": report["contact_steps"]}, indent=2))
        return report
    if args.continue_training:
        try:
            integer(args.budget_seconds, "budget seconds", 1, 7200)
            integer(args.updates, "updates", 1, 20)
            integer(args.train_seconds, "train seconds", 1, 900)
        except ValueError as exc:
            parser.error(str(exc))
        report = continue_training(args)
        last = report["rounds"][-1]
        print(json.dumps({"status": report["status"], "updates": report["updates"],
                          "elapsed_seconds": report["elapsed_seconds"],
                          "validation_mse": last["validation_mse"],
                          "persistence_mse": last["persistence_mse"],
                          "contact_steps": report["contact_steps"]}, indent=2))
        return report
    try:
        if args.steps < 4:
            raise ValueError("Use at least four control steps so each episode has one four-frame window")
        integer(args.max_rounds, "max rounds", 1, 200)
        integer(args.max_collections, "max collections", 1, 20)
        integer(args.max_updates, "max updates", 1, 4000)
        integer(args.patience, "patience", 1, 10)
        integer(args.budget_seconds, "budget seconds", 0, 7200)
        config = OfficialCollectionConfig(shape=args.shape, episodes=args.episodes, steps=args.steps,
                                          seed=args.seed, mode="object-directed", seconds=args.seconds,
                                          excluded_reset_seeds=DEVELOPMENT_SEEDS)
    except ValueError as exc:
        parser.error(str(exc))
    report = execute(args, config)
    last = report["rounds"][-1]
    print(json.dumps({"status": report["status"], "splits": report["splits"], "contact_steps": report["contact_steps"],
                      "weights_changed": report["training"]["weights_changed"],
                      "validation": report["training"]["validation"],
                      "decision": last["action"], "executed": last["executed"],
                      "reason": last["executed_reason"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
