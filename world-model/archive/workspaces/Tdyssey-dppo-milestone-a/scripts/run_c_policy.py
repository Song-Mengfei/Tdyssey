"""Milestone C proposal, bounded training, and independent development evaluation.

The default plan is read-only and uses only stdlib. Runtime modes require an
explicit --execute and an initialized permanent experiment registry. This entry
point never imports or consumes milestone B's smoke-check budget.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", nargs="?", default="plan", choices=("plan", "init", "train", "evaluate", "compare"))
    result.add_argument("--config", type=Path, default=ROOT / "configs/policy_c.example.json")
    result.add_argument("--execute", action="store_true", help="Explicit runtime opt-in; not needed for plan/compare")
    result.add_argument("--experiment-id")
    result.add_argument("--registry", type=Path, default=ROOT / "artifacts/policy-c-registry")
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--source-root", type=Path)
    result.add_argument("--base-checkpoint", type=Path)
    result.add_argument("--controls", type=int, help="Absolute cumulative training target, including restored controls")
    result.add_argument("--batches", type=int, help="Absolute cumulative batch cap, including restored batches")
    result.add_argument("--resume-checkpoint", type=Path)
    result.add_argument("--checkpoint-every", type=int, default=4, help="Request a genuine joint episode boundary after this many batches")
    result.add_argument("--stop-after-empty-batches", type=int, default=0, help="Optional consecutive zero-contact/zero-raw-reward stop; 0 disables")
    result.add_argument("--method", choices=("base_original", "base_training", "delta_training"))
    result.add_argument("--protocol", choices=("original_t", "matched_tl"))
    result.add_argument("--group", help="Unique, permanently consumed development evaluation group ID")
    result.add_argument("--checkpoint", type=Path, help="A C delta for delta_training evaluation")
    result.add_argument("--reports", nargs="+", type=Path, help="Offline complete matched reports to compare")
    return result


def validate_args(args, config):
    from tdyssey.policy_protocol import _identifier
    data = config.to_dict()
    if args.mode in ("init", "train", "evaluate"):
        if not args.execute:
            raise ValueError("Runtime modes require explicit --execute; plan does not start a job")
        _identifier(args.experiment_id, "experiment_id")
        if args.run_dir is None:
            raise ValueError("Runtime modes require --run-dir")
    if args.mode in ("train", "evaluate") and (args.source_root is None or args.base_checkpoint is None):
        raise ValueError("Provide the existing --source-root and --base-checkpoint; assets are never downloaded")
    if args.mode == "train":
        if args.controls is None or not 1 <= args.controls <= data["budgets"]["train"]["control"]:
            raise ValueError("Explicit --controls must fit the permanent training budget (at most 2048)")
        if args.batches is None or not 1 <= args.batches <= data["budgets"]["global"]["batch"]:
            raise ValueError("Explicit --batches must fit the permanent batch budget (at most 24)")
        if not 1 <= args.checkpoint_every <= 24 or not 0 <= args.stop_after_empty_batches <= 24:
            raise ValueError("Invalid checkpoint cadence or optional empty-batch stop")
        if args.controls > args.batches * data["training"]["batch_controls"]:
            raise ValueError("Requested controls cannot fit the explicit cumulative batch cap")
    if args.mode == "evaluate":
        if not args.method or not args.protocol or not args.group:
            raise ValueError("Evaluation requires one explicit --method, --protocol and unique --group")
        _identifier(args.group, "evaluation group")
        if (args.method == "delta_training") != (args.checkpoint is not None):
            raise ValueError("Only delta_training requires --checkpoint; base methods load the verified base alone")
    if args.mode != "train" and args.resume_checkpoint is not None:
        raise ValueError("--resume-checkpoint applies only to training")
    if args.mode != "train" and (args.controls is not None or args.batches is not None):
        raise ValueError("--controls and --batches apply only to training")
    if args.mode == "compare" and (not args.reports or len(args.reports) < 2):
        raise ValueError("Offline comparison requires at least two --reports")
    if args.mode != "evaluate" and args.checkpoint is not None:
        raise ValueError("--checkpoint applies only to independent delta evaluation")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path, value, budget):
    payload = _json_bytes(value)
    reservation = budget.reserve_storage(len(payload), path=path)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    budget.mark_storage_written(reservation, len(payload))
    return hashlib.sha256(payload).hexdigest()


def _append_log(path, value, budget):
    payload = _json_bytes(value)
    reservation = budget.reserve_storage(len(payload), path=path)
    with path.open("ab") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    budget.mark_storage_written(reservation, len(payload))


def _preview_delta(path, config, experiment_id=None, *, resumable=False):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else None
    if (not isinstance(metadata, dict) or metadata.get("kind") != "tdyssey-C" or metadata.get("schema_version") != 1
            or metadata.get("config_hash") != config.config_hash
            or metadata.get("config") != config.to_dict()
            or payload.get("base_identity") != config.to_dict()["base"]):
        raise ValueError("Checkpoint must be a C delta with this exact configuration and pinned base")
    if experiment_id is not None and metadata.get("experiment_id") != experiment_id:
        raise ValueError("Training resume requires the original permanent experiment identity")
    if resumable and metadata.get("resumable") is not True:
        raise ValueError("Active-episode audit/final checkpoints cannot resume training")
    # Drop the preview tensor payload before reconstructing the single base actor.
    return metadata


def _runtime(args, config):
    import numpy as np
    import torch
    data = config.to_dict()
    device = data["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Configured CUDA is unavailable; no automatic CPU fallback")
    torch.set_num_threads(data["runtime"]["threads"])
    torch.manual_seed(data["train_seed"])
    random.seed(data["train_seed"])
    np.random.seed(data["train_seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(data["train_seed"])
    if args.base_checkpoint.stat().st_size != data["base"]["bytes"]:
        raise ValueError("Existing base byte count differs from the proposal")
    return device


def _optimizers(actor, critic, data):
    from tdyssey.policy_training import make_optimizers
    return make_optimizers(actor, critic, actor_lr=data["optimization"]["actor_lr"],
                           critic_lr=data["optimization"]["critic_lr"])


def _new_actor(args, config, device):
    from tdyssey.official import OfficialDiffusionPolicy
    from tdyssey.policy_rl import DenoisingPolicy, SamplerConfig
    data = config.to_dict()
    official = OfficialDiffusionPolicy(args.source_root, args.base_checkpoint, device=device,
                                       checkpoint_sha256=data["base"]["sha256"])
    actor = DenoisingPolicy(official, subset=data["optimization"]["subset"],
                            config=SamplerConfig(std_floor=data["optimization"]["std_floor"]))
    return official, actor


def _selected_replay(fragments, run_dir, budget):
    """At most one first-batch short trace per shape; no full buffer/checkpoint."""
    import torch
    saved = []
    for shape in ("T", "L"):
        plan = next((p for fragment in fragments for p in fragment if p.shape == shape), None)
        if plan is None:
            continue
        records = plan.control_records
        if not records:
            raise ValueError("Selected replay requires at least one actually executed control")
        # One initial RGB and each real next frame, quantized for compact replay.
        observations = [records[0]["observation_before"], *[r["observation_after"] for r in records]]
        frames = torch.stack([obs["image"] for obs in observations]).clamp(0, 1).mul(255).round().to(torch.uint8)
        details = [{k: record[k] for k in ("coverage", "coverage_delta", "raw_reward", "training_reward",
                                           "shaping", "n_contacts", "control_index")} for record in records]
        payload = {"schema_version": 1, "kind": "tdyssey.C.selected_control_trace", "shape": shape,
                   "reset_seed": plan.reset_seed, "actor_version": plan.sample.actor_version,
                   "frames_uint8": frames, "agent_pos": torch.stack([o["agent_pos"] for o in observations]),
                   "actions": torch.stack([r["action"] for r in records]), "controls": details}
        path = run_dir / f"replay-first-{shape}.pt"
        reservation = budget.reserve_storage(2 * 1024**2, path=path)
        with path.open("xb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        budget.mark_storage_written(reservation, path.stat().st_size)
        saved.append({"shape": shape, "path": str(path), "bytes": path.stat().st_size,
                      "controls": len(records)})
    return saved


def run_train(args, config, budget):
    budget.check_time()
    import torch
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    from tdyssey.policy_checkpoint import load_update_checkpoint, restore_optimizers
    from tdyssey.policy_experiment import (restore_c_checkpoint, save_c_checkpoint, train_batches,
                                           validate_first_batch)
    from tdyssey.policy_ppo import MultiPassConfig, PPOUpdater
    from tdyssey.policy_stream import ContinuousCollector, SceneCursor
    from tdyssey.policy_training import UpdateConfig, VisualCritic
    data = config.to_dict()
    device = _runtime(args, config)
    if args.resume_checkpoint:
        _preview_delta(args.resume_checkpoint, config, args.experiment_id, resumable=True)
        budget.validate_boundary(args.resume_checkpoint)
        loaded = load_update_checkpoint(args.resume_checkpoint, source_root=args.source_root,
                    base_checkpoint=args.base_checkpoint, device=device, restore_rng=True)
        official, actor, critic = loaded.official, loaded.actor, loaded.critic
    else:
        if any(budget.reserved_count(kind, phase="train") for kind in ("control", "plan", "reset", "batch", "actor_step", "critic_step")):
            raise ValueError("This experiment already has training attempts; an exact C boundary resume is required")
        official, actor = _new_actor(args, config, device)
        critic = VisualCritic(actor.network.obs_feature_dim * actor.network.n_obs_steps).to(device)
    actor_opt, critic_opt = _optimizers(actor, critic, data)
    if args.resume_checkpoint:
        restore_optimizers(loaded, actor_opt, critic_opt)
    optimization = data["optimization"]
    updater = PPOUpdater(actor, critic, actor_opt, critic_opt, budget=budget, run_id=args.experiment_id,
        config=MultiPassConfig(max_passes=optimization["max_passes"], target_kl=optimization["target_kl"],
            gamma=optimization["gamma"], gae_lambda=optimization["gae_lambda"],
            update=UpdateConfig(**{key: optimization[key] for key in
                ("clip", "max_grad_norm", "plan_microbatch", "step_microbatch", "denoising_discount")})))
    env_class = make_official_shape_env_class(load_official_source(args.source_root))
    environments = {}
    try:
        for shape in ("T", "L"):
            environments[shape] = env_class(shape=shape, legacy=True)
        generators = {shape: torch.Generator(device=device).manual_seed(data["train_seed"] + i)
                      for i, shape in enumerate(("T", "L"))}
        collector = ContinuousCollector(official, actor, critic, environments, generators,
            SceneCursor(data["scenes"]["training"], set(data["scenes"]["excluded_evaluation"])), budget,
            episode_limit=data["training"]["episode_controls"], gamma=optimization["gamma"], beta=optimization["beta"])
        resume_state = None
        loop_options = {"checkpoint_every": args.checkpoint_every, "stop_after_empty_batches": args.stop_after_empty_batches}
        if args.resume_checkpoint:
            metadata = restore_c_checkpoint(loaded, collector, updater, budget, config, args.resume_checkpoint)
            if metadata.get("loop_options") != loop_options:
                raise ValueError("Resume must preserve the saved checkpoint/diagnostic stop options")
            resume_state = metadata.get("loop_state")
        invocation = len(list(args.run_dir.glob("train-report-*.json")))
        report_path = args.run_dir / f"train-report-{invocation:04d}.json"
        if report_path.exists():
            raise FileExistsError(report_path)

        def checkpoint(*, resumable, terminal, loop_state):
            suffix = "boundary" if resumable else "audit"
            path = args.run_dir / f"checkpoint-{updater.batches:04d}-{updater.actor_steps:04d}-{suffix}.pt"
            return save_c_checkpoint(path, official, actor, critic, actor_opt, critic_opt, collector,
                updater, budget, config, terminal=terminal,
                metadata={"loop_state": loop_state, "loop_options": loop_options})

        result = train_batches(collector, updater, budget, batch_plans=data["training"]["batch_plans"],
            batch_controls=data["training"]["batch_controls"], first_batch_passes=optimization["first_batch_passes"],
            passes=optimization["max_passes"], max_batches=args.batches, control_target=args.controls,
            **loop_options, on_checkpoint=checkpoint,
            on_replay=lambda **kw: _selected_replay(kw["fragments"], args.run_dir, budget),
            log=lambda row: _append_log(args.run_dir / "training.jsonl", row, budget), resume_state=resume_state,
            first_batch_gate=validate_first_batch)
        report = {"kind": "tdyssey.C.training", "config_hash": config.config_hash,
                  "experiment_id": args.experiment_id, "invocation_targets": {"controls": args.controls, "batches": args.batches},
                  "parameter_selection": asdict(actor.selection), "critic_config": critic.config,
                  "device": device, "result": result, "budget": budget.snapshot()}
        _write_json(report_path, report, budget)
        return {"report": str(report_path), "stop_reason": result["stop_reason"],
                "controls": result["controls"], "batches": result["batches"], "last_checkpoint": result["last_checkpoint"]}
    finally:
        _close_environments(environments.values(), original_error=sys.exc_info()[1])


def run_evaluation(args, config, budget):
    budget.check_time()
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    from tdyssey.policy import hash_file
    from tdyssey.policy_checkpoint import load_update_checkpoint
    from tdyssey.policy_evaluation import (EvaluationCase, ORIGINAL_T_PROTOCOL, MATCHED_TL_PROTOCOL,
        bind_evaluation_identity, evaluate_cases)
    data = config.to_dict()
    device = _runtime(args, config)
    protocol = ORIGINAL_T_PROTOCOL if args.protocol == "original_t" else MATCHED_TL_PROTOCOL
    identity = {"base_sha256": data["base"]["sha256"], "delta_sha256": None}
    if args.checkpoint:
        _preview_delta(args.checkpoint, config)
        identity["delta_sha256"] = hash_file(args.checkpoint)
        loaded = load_update_checkpoint(args.checkpoint, source_root=args.source_root,
            base_checkpoint=args.base_checkpoint, device=device, restore_rng=False)
        official, actor = loaded.official, loaded.actor
    else:
        official, actor = _new_actor(args, config, device)
    bind_evaluation_identity(actor, **identity)
    shapes = ("T",) if args.protocol == "original_t" else ("T", "L")
    cases = [EvaluationCase(shape, scene, seed) for shape in shapes
             for scene in data["evaluation"]["scenes"] for seed in data["evaluation"]["policy_seeds"]]
    digest = hashlib.sha256(_json_bytes([asdict(case) for case in cases])).hexdigest()
    budget.claim_evaluation_group(args.group, method=args.method, protocol=protocol, case_digest=digest)
    env_class = make_official_shape_env_class(load_official_source(args.source_root))
    report = evaluate_cases(official, actor, lambda shape: env_class(shape=shape, legacy=True), cases,
        method=args.method, protocol=protocol, budget=budget, max_controls=data["evaluation"]["episode_controls"],
        delta_applied=args.method == "delta_training", checkpoint_identity=identity)
    report.update(group_id=args.group, comparison_label=args.group, experiment_id=args.experiment_id,
                  config_hash=config.config_hash, evaluation_split="development",
                  untouched_test=False, new_development_scenes=data["evaluation"]["new_development_scenes"],
                  budget=budget.snapshot())
    path = args.run_dir / f"evaluation-{args.group}.json"
    report_hash = _write_json(path, report, budget)
    budget.complete_evaluation_group(args.group, report_hash)
    return {"report": str(path), "status": report["status"], "success_fraction": report["success_fraction"]}


def _close_environments(environments, *, original_error=None):
    """Close every constructed environment, preserving an existing failure."""
    errors = []
    for environment in environments:
        try:
            environment.close()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if errors:
        message = "Environment cleanup failed: " + "; ".join(errors)
        if original_error is None:
            raise RuntimeError(message)
        print(message, file=sys.stderr)


def _record_failure(args, config, budget, error):
    """Best-effort bounded failure telemetry; never checkpoint unsafe weights."""
    try:
        snapshot = budget.snapshot()
        counts = {key: snapshot[key] for key in ("reserved", "completed") if key in snapshot}
        path = args.run_dir / f"failure-{uuid.uuid4().hex}.json"
        report = {"schema_version": 1, "kind": "tdyssey.C.runtime_failure", "mode": args.mode,
                  "experiment_id": args.experiment_id, "config_hash": config.config_hash,
                  "error_type": type(error).__name__, "error": str(error)[:2000], "counts": counts,
                  "failure_handler_created_checkpoint": False, "safe_resume_claim": False}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            report["cuda"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                              "reserved_bytes": torch.cuda.memory_reserved(),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        for key in ("elapsed_wall_seconds", "last_train_operation_id"):
            if type(snapshot.get(key)) in (int, float):
                report[key] = snapshot[key]
        _write_json(path, report, budget)
        print(f"Failure report: {path}", file=sys.stderr)
        return path
    except Exception as report_error:
        print(f"Failure report unavailable: {type(report_error).__name__}: {report_error}; "
              "the authoritative ledger remains the recovery reference", file=sys.stderr)
        return None


def main():
    args = parser().parse_args()
    try:
        from tdyssey.policy_protocol import CConfig
        config = CConfig.from_json(args.config)
        validate_args(args, config)
        data = config.to_dict()
        if args.mode == "plan":
            value = {"status": "proposed_not_executed", "config_hash": config.config_hash, "config": data,
                     "runtime_started": False, "physical_controls": 0, "optimizer_steps": 0}
        elif args.mode == "compare":
            from tdyssey.policy_evaluation import paired_comparison
            value = paired_comparison([json.loads(path.read_text(encoding="utf-8")) for path in args.reports])
        else:
            from tdyssey.policy_protocol import ExperimentBudget
            args.run_dir = args.run_dir.resolve()
            # No optional library or asset access occurs before explicit gating.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_HOME", str(args.run_dir / "hf-cache"))
            with ExperimentBudget(args.registry, args.experiment_id, config, args.run_dir, create=args.mode == "init") as budget:
                try:
                    # Reject an expired/full permanent authority before any
                    # torch import, asset hash/read or model construction.
                    budget.check_time()
                    budget.check_storage()
                    if args.mode == "init":
                        value = {"status": "initialized_no_interaction", "budget": budget.snapshot()}
                    elif args.mode == "train":
                        value = run_train(args, config, budget)
                    else:
                        value = run_evaluation(args, config, budget)
                except Exception as exc:
                    _record_failure(args, config, budget, exc)
                    raise
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
        return 1 if value.get("stop_reason") == "first_batch_gate_failed" else 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
