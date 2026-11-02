"""Fixed, evaluation-only C base/final-delta comparison. Default is read-only."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAINING_ID = "c-tl-2048-20261024"
CONFIG_SHA = "0c682afbaf03360dc7af77c3c0917c3cfd120d5e7a8da10532c1d3d5251a8db3"
DELTA_SHA = "d3128c7fcfb1a23842b4863d96ae88b7f124e93d751b371b23696eb24d6703dd"
DELTA_BYTES = 16003916
METHODS = ("base_training", "delta_training")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "init", "run"), nargs="?", default="plan")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--account-id")
    p.add_argument("--registry", type=Path, default=ROOT / "artifacts/policy-c-evaluation-registry")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "configs/policy_c.example.json")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--base-checkpoint", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--training-ledger", type=Path)
    return p


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_cases():
    return [{"shape": shape, "scene_seed": scene, "policy_seed": seed}
            for shape in ("T", "L") for scene in range(100000, 100005) for seed in (5478, 5479)]


def prepare(args):
    from tdyssey.policy_protocol import CConfig, _identifier
    config = CConfig.from_json(args.config)
    if config.config_hash != CONFIG_SHA:
        raise ValueError("Evaluation requires the exact reviewed training configuration")
    binding = {"training_experiment_id": TRAINING_ID, "config_hash": config.config_hash,
        "base_identity": config.to_dict()["base"], "delta_identity": {"bytes": DELTA_BYTES, "sha256": DELTA_SHA},
        "case_digest": hashlib.sha256(canonical(ordered_cases())).hexdigest(),
        "methods": list(METHODS), "max_controls": 300}
    if args.mode == "plan":
        return config, binding, None
    if not args.execute or any(getattr(args, name) is None for name in
            ("account_id", "run_dir", "source_root", "base_checkpoint", "checkpoint", "training_ledger")):
        raise ValueError("Runtime requires --execute and explicit account, output, source, base, delta and training ledger")
    _identifier(args.account_id, "evaluation_id")
    args.run_dir = args.run_dir.resolve()
    args.registry = args.registry.resolve()
    ledger_path = args.training_ledger.resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (ledger.get("experiment_id") != TRAINING_ID or ledger.get("config_hash") != config.config_hash
            or ledger.get("config") != config.to_dict() or ledger.get("pending")
            or ledger["completed"]["train"]["control"] != 2048
            or ledger["completed"]["global"] != {"actor_step": 52, "critic_step": 52, "batch": 18}
            or ledger_path.with_name("owner.lock").exists()):
        raise ValueError("Original training ledger is not the completed, reviewed immutable run")
    protected = (ledger_path.parent.parent, args.checkpoint.resolve().parent,
                 ROOT / "artifacts/milestone-b-20261023")
    for target in (args.registry, args.run_dir):
        if (not target.is_relative_to(ROOT / "artifacts")
                or any(target.is_relative_to(path) or path.is_relative_to(target) for path in protected)):
            raise ValueError("Evaluation account/outputs must be disjoint from original training/B artifacts")
    if args.checkpoint.stat().st_size != DELTA_BYTES or sha(args.checkpoint) != DELTA_SHA:
        raise ValueError("Final delta differs from its fixed size/SHA; older boundaries are forbidden")
    base = binding["base_identity"]
    if args.base_checkpoint.stat().st_size != base["bytes"] or sha(args.base_checkpoint) != base["sha256"]:
        raise ValueError("Base differs from the explicit local pin")
    paths = {ledger_path, ledger_path.parent.parent / "index.json", args.base_checkpoint.resolve()}
    training_dir = args.checkpoint.resolve().parent
    paths.update(path.resolve() for path in training_dir.rglob("*") if path.is_file())
    paths.update(path.resolve() for path in ledger_path.parent.rglob("*") if path.is_file())
    paths.update(path.resolve() for path in (ROOT / "artifacts/milestone-b-20261023").rglob("*") if path.is_file())
    inventory = {"files": {str(path): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(paths)},
        "trees": {str(path.resolve()): sorted(str(file.resolve()) for file in path.rglob("*") if file.is_file())
                  for path in (training_dir, ledger_path.parent, ROOT / "artifacts/milestone-b-20261023")}}
    if args.mode == "run":
        declared = json.loads((args.run_dir / "preserved-declared.json").read_text(encoding="utf-8"))
        if declared != inventory:
            raise ValueError("Preserved original evidence changed between evaluation account init and worker start")
    return config, binding, inventory


def unchanged(inventory):
    changed = [path for path, identity in inventory["files"].items()
               if not Path(path).is_file() or Path(path).stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]]
    changed += [path for path, files in inventory["trees"].items()
                if sorted(str(file.resolve()) for file in Path(path).rglob("*") if file.is_file()) != files]
    if changed:
        raise ValueError("Preserved training/base/B evidence changed: " + repr(changed))
    return True


def write_json(path, value, budget):
    payload = canonical(value) + b"\n"
    reservation = budget.reserve_storage(len(payload), path=path)
    with Path(path).open("xb") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    budget.mark_storage_written(reservation, len(payload))
    return hashlib.sha256(payload).hexdigest()


def append_case(path, value, budget):
    payload = canonical(value) + b"\n"
    reservation = budget.reserve_storage(len(payload), path=path)
    with Path(path).open("ab") as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    budget.mark_storage_written(reservation, len(payload))


def isolation_fingerprint(actor):
    import random
    import numpy as np
    import torch
    scheduler = actor.network.noise_scheduler
    h = hashlib.sha256(repr(random.getstate()).encode())
    state = np.random.get_state()
    h.update(state[0].encode()); h.update(state[1].tobytes()); h.update(repr(state[2:]).encode())
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for tensor in torch.cuda.get_rng_state_all():
            h.update(tensor.cpu().numpy().tobytes())
    modes = [(name, module.training) for name, module in actor.named_modules()]
    scheduler_state = {}
    for name in ("timesteps", "num_inference_steps", "custom_timesteps"):
        if hasattr(scheduler, name):
            value = getattr(scheduler, name)
            scheduler_state[name] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
    return {"rng_sha256": h.hexdigest(), "modes_sha256": hashlib.sha256(canonical(modes)).hexdigest(),
            "scheduler_sha256": hashlib.sha256(canonical(scheduler_state)).hexdigest()}


def cuda_resources():
    import torch
    return {name: int(getattr(torch.cuda, fn)()) for name, fn in (
        ("allocated_bytes", "memory_allocated"), ("reserved_bytes", "memory_reserved"),
        ("peak_allocated_bytes", "max_memory_allocated"), ("peak_reserved_bytes", "max_memory_reserved"))}


def evaluate_one(args, config, budget, method, inventory):
    from tdyssey.policy_eval_budget import PROTOCOL
    budget.check_time()
    unchanged(inventory)
    group_id = method
    budget.claim_evaluation_group(group_id, method=method, protocol=PROTOCOL,
        case_digest=hashlib.sha256(canonical(ordered_cases())).hexdigest())
    import torch
    from scripts.run_c_policy import _new_actor, _runtime
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    from tdyssey.policy_checkpoint import load_update_checkpoint
    from tdyssey.policy_evaluation import EvaluationCase, MATCHED_TL_PROTOCOL, bind_evaluation_identity, evaluate_cases
    device = _runtime(args, config)
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    loaded = official = actor = None
    try:
        if method == "base_training":
            official, actor = _new_actor(args, config, device)
        else:
            loaded = load_update_checkpoint(args.checkpoint, source_root=args.source_root,
                base_checkpoint=args.base_checkpoint, device=device, restore_rng=False)
            official, actor = loaded.official, loaded.actor
            if loaded.payload["metadata"]["experiment_id"] != TRAINING_ID:
                raise ValueError("Delta belongs to a different original training experiment")
        actor.requires_grad_(False).eval()
        network_reference = weakref.ref(actor.network)
        identity = {"base_sha256": config.to_dict()["base"]["sha256"],
                    "delta_sha256": DELTA_SHA if method == "delta_training" else None}
        bind_evaluation_identity(actor, **identity)
        env_class = make_official_shape_env_class(load_official_source(args.source_root))
        from tdyssey.policy_training import state_fingerprint
        before = isolation_fingerprint(actor)
        frozen_before = state_fingerprint(actor.network, exclude=actor.selection.names)
        report = evaluate_cases(official, actor, lambda shape: env_class(shape=shape, legacy=True),
            [EvaluationCase(**case) for case in ordered_cases()], method=method, protocol=MATCHED_TL_PROTOCOL,
            budget=budget, max_controls=300, delta_applied=method == "delta_training", checkpoint_identity=identity,
            on_case_start=lambda case: append_case(args.run_dir / "case-events.jsonl", {"event": "started", **case}, budget),
            on_case=lambda case: append_case(args.run_dir / "case-events.jsonl", {"event": "finished", **case}, budget))
        after = isolation_fingerprint(actor)
        if before != after or any(parameter.grad is not None for parameter in actor.parameters()):
            raise ValueError("Evaluation changed RNG/mode/scheduler state or acquired gradients")
        frozen_after = state_fingerprint(actor.network, exclude=actor.selection.names)
        if frozen_before != frozen_after:
            raise ValueError("Evaluation changed the frozen actor/normalizer subset")
        report.update(group_id=group_id, comparison_label=method, evaluation_id=args.account_id,
            training_experiment_id=TRAINING_ID, config_hash=config.config_hash, sampler_config=asdict(actor.config),
            isolation_before=before, isolation_after=after, isolation_verified=True, resources=cuda_resources(),
            frozen_fingerprint_before=frozen_before, frozen_fingerprint_after=frozen_after,
            evaluation_split="development", untouched_test=False, new_training_steps=0)
        unchanged(inventory)
        digest = write_json(args.run_dir / (method + ".json"), report, budget)
        if report["status"] == "completed":
            budget.complete_evaluation_group(group_id, digest)
        return report, network_reference
    finally:
        loaded = actor = official = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def run_pair(args, config, budget, inventory):
    import importlib.metadata
    import platform
    from tdyssey.policy_eval_comparison import compare_evaluation_reports
    dependencies = {"python": sys.version, "platform": platform.platform()}
    for name in ("torch", "torchvision", "numpy", "diffusers", "robomimic", "pygame", "pymunk",
                 "gym", "shapely", "scikit-image", "opencv-python-headless"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not_installed"
    write_json(args.run_dir / "preserved-before.json", inventory, budget)
    reports, releases = [], []
    for method in METHODS:
        report, reference = evaluate_one(args, config, budget, method, inventory)
        if reference() is not None:
            raise RuntimeError("Prior complete model remains resident; refusing another model load")
        releases.append({"method": method, "network_released": True, "resources": cuda_resources()})
        write_json(args.run_dir / (method + "-released.json"), releases[-1], budget)
        reports.append(report)
        if report["status"] != "completed":
            break
    available = len(reports) == 2 and all(report["status"] == "completed" for report in reports)
    comparison = compare_evaluation_reports(*reports) if available else None
    if available and (reports[0]["frozen_fingerprint_before"] != reports[1]["frozen_fingerprint_before"]
                      or reports[0]["state_fingerprint_before"] == reports[1]["state_fingerprint_before"]):
        raise ValueError("Base/delta frozen state differs or selected learned weights did not change")
    if comparison is not None:
        write_json(args.run_dir / "paired-comparison.json", comparison, budget)
    unchanged(inventory)
    result = {"schema_version": 1, "kind": "tdyssey.C.evaluation_only_pair", "status": "completed" if available else "incomplete",
        "evaluation_id": args.account_id, "training_experiment_id": TRAINING_ID, "training_steps": 0,
        "config_hash": config.config_hash, "groups": reports, "comparison": {"available": available, "report": comparison},
        "dependencies": dependencies, "config": config.to_dict(), "command": list(sys.argv),
        "preserved_before_after_equal": True, "model_releases": releases, "budget": budget.snapshot(),
        "limitations": ["Fixed development scenes, five scene blocks per shape with two correlated policy seeds.",
            "Same positive-floor sampler; comparison concerns learned weights, not original sampler performance.",
            "Local base pin does not independently prove published EMA tensor identity."]}
    write_json(args.run_dir / "evaluation-pair-report.json", result, budget)
    return result


def main():
    args = parser().parse_args()
    try:
        config, binding, inventory = prepare(args)
        if args.mode == "plan":
            print(json.dumps({"binding": binding, "cases": ordered_cases(), "training_steps": 0, "runtime_started": False}, indent=2))
            return 0
        from tdyssey.policy_eval_budget import EvaluationBudget
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/hf-cache"))
        with EvaluationBudget(args.registry, args.account_id, binding, args.run_dir, create=args.mode == "init") as budget:
            if args.mode == "init":
                write_json(args.run_dir / "preserved-declared.json", inventory, budget)
                result = {"status": "initialized_evaluation_only", "training_steps": 0, "budget": budget.snapshot()}
            else:
                try:
                    result = run_pair(args, config, budget, inventory)
                except Exception as exc:
                    failure = {"status": "failed", "training_steps": 0, "error_type": type(exc).__name__,
                        "error": str(exc), "budget": budget.snapshot(), "safe_retry": False,
                        "at_utc": datetime.now(timezone.utc).isoformat()}
                    try:
                        write_json(args.run_dir / "evaluation-failure.json", failure, budget)
                    except Exception as reporting_error:
                        print("Failure telemetry unavailable: " + str(reporting_error), file=sys.stderr)
                    raise
        print(json.dumps({"status": result["status"], "training_steps": 0,
            "report": str(args.run_dir / "evaluation-pair-report.json") if args.mode == "run" else None}, indent=2))
        return 0 if result["status"] in ("completed", "initialized_evaluation_only") else 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
