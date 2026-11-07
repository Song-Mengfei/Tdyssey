"""Bounded simulator-scored forward RL, with recoverable update boundaries."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .forward_policy import ForwardPolicy, forward_update
from .forward_protocol import ForwardConfig
from .official import OfficialDiffusionPolicy, load_official_source
from .envs.official import make_official_shape_env_class
from .policy import hash_file
from .policy_branches import BranchCollector
from .policy_training import evaluate, frozen_digest
from .denoising_probe import DenoisingProbe
from .tracking import RunLogger, evaluation_scalars, evaluation_score

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("tdyssey/forward_policy.py", "tdyssey/forward_protocol.py", "tdyssey/forward_training.py",
           "tdyssey/policy_branches.py", "tdyssey/policy_training.py", "tdyssey/lora.py",
           "tdyssey/official.py", "tdyssey/envs/official.py", "tdyssey/rl_protocol.py",
           "tdyssey/collection.py", "tdyssey/official_collection.py", "tdyssey/geometry.py",
           "tdyssey/letters.py", "tdyssey/state.py", "manifests/official_dp.json",
           "scripts/train_forward_policy.py", "tdyssey/denoising_probe.py", "tdyssey/tracking.py")


def check_resume(payload, config, source_hashes):
    previous = ForwardConfig(**payload["config"])
    allowed = {"updates", "max_train_controls", "seconds", "max_bytes"}
    for name, value in asdict(previous).items():
        if name not in allowed and value != getattr(config, name):
            raise ValueError(f"Resume changes training protocol: {name}")
    state = payload.get("training_state")
    if not state or state["report"]["source_sha256"] != source_hashes:
        raise ValueError("Resume requires matching training source hashes and runner state")
    if config.updates < state["completed_updates"]:
        raise ValueError("Resume update cap precedes the saved boundary")
    if config.max_train_controls < state["report"]["counts"]["training_controls"]:
        raise ValueError("Resume control cap precedes saved interaction")
    return state


def audit_resume_costs(checkpoint, report):
    """Charge recorded work after a recovery boundary even when updates roll back.

    A hard kill can leave no final report; that uncertainty is explicitly retained.
    Logical context/RNG counters stay at the checkpoint's optimizer boundary.
    """
    checkpoint = Path(checkpoint)
    sidecar = checkpoint.parent/"report.json"
    audit = {"checkpoint_sha256": hash_file(checkpoint), "post_boundary_costs_known": False}
    if sidecar.is_file():
        latest = json.loads(sidecar.read_text())
        if (latest.get("kind") == "official_forward_rl_pilot"
                and latest.get("checkpoint", {}).get("sha256") == audit["checkpoint_sha256"]
                and latest.get("source_sha256") == report["source_sha256"]):
            keys = ("training_controls", "branch_controls", "replay_controls", "context_controls",
                    "training_plans", "reference_plans", "initial_state_restores", "evaluation_controls",
                    "evaluation_plans", "reset_physics_substeps", "optimizer_minibatches",
                    "checkpoint_verification_plans")
            delta = {key: latest["counts"][key]-report["counts"][key] for key in keys}
            if any(value < 0 for value in delta.values()):
                raise ValueError("Parent cost report precedes its checkpoint")
            for key, value in delta.items():
                report["counts"][key] += value
            old_time = sum(s.get("elapsed_seconds", 0.) for s in report.get("sessions", []))
            new_time = sum(s.get("elapsed_seconds", 0.) for s in latest.get("sessions", []))
            audit.update(post_boundary_costs_known=(latest.get("status") in ("complete", "failed")
                         and not latest.get("counts_are_lower_bounds", False)),
                         additional_counts=delta, additional_elapsed_seconds=max(0., new_time-old_time),
                         parent_status=latest.get("status"))
    report.setdefault("resume_cost_audits", []).append(audit)


def _size(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value)
    return 0


def storage_room(output, config, extra):
    used = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if used+extra > config.max_bytes or shutil.disk_usage(output).free < extra+8*1024**2:
        raise ValueError("Output budget or free disk space exhausted")


def write_report(output, config, report):
    data = json.dumps(report, indent=2, allow_nan=False)+"\n"
    storage_room(output, config, len(data.encode()))
    temporary = output/"report.tmp"
    temporary.write_text(data)
    temporary.replace(output/"report.json")


def save_recovery(output, config, agent, optimizer, report, completed, active, probe_state=None):
    # Large base weights are excluded in LoRA/head modes; one rolling checkpoint.
    snapshot = json.loads(json.dumps(report, allow_nan=False))
    snapshot.pop("checkpoint", None)
    payload = agent.state()
    payload.update(optimizer=optimizer.state_dict(), training_state={
        "completed_updates": completed, "active_contexts": dict(active), "report": snapshot,
        "denoising_probe": probe_state or []})
    estimate = _size(payload)+len(json.dumps(snapshot).encode())+2*1024**2
    storage_room(output, config, estimate)
    temporary = output/"last.tmp"
    try:
        torch.save(payload, temporary)
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > config.max_bytes:
            raise ValueError("Serialized checkpoint exceeds output budget")
        temporary.replace(output/"last.pt")
    finally:
        temporary.unlink(missing_ok=True)
    report["checkpoint"] = {"file": "last.pt", "bytes": (output/"last.pt").stat().st_size,
                            "sha256": hash_file(output/"last.pt"), "completed_updates": completed}


def compact_updates(report, keep):
    """Bound detailed action traces; keep all scalar update summaries and counters.

    Active episode prefixes remain intact for exact contact replay/recovery.
    """
    rows = report["updates"][:-keep] if keep else report["updates"]
    for row in rows:
        if "metrics" not in row:
            continue
        row.pop("metrics")
        row.pop("groups")
        row["details_pruned"] = True


def train_forward(source_root, checkpoint, output, config, resume=None, tracking=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output directory; resume also writes to a new directory")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required; no CPU training fallback")
    source_hashes = {path: hash_file(ROOT/path) for path in SOURCES}
    parent = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    previous = check_resume(parent, config, source_hashes) if parent else None
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    output.mkdir(parents=True)
    counts = {key: 0 for key in ("training_controls", "branch_controls", "replay_controls", "context_controls",
              "training_plans", "reference_plans", "candidate_groups", "initial_state_restores",
              "evaluation_controls", "evaluation_plans", "reset_physics_substeps",
              "optimizer_minibatches", "checkpoint_verification_plans", "context_attempts")}
    report = {"schema_version": 1, "kind": "official_forward_rl_pilot", "status": "running",
        "config": asdict(config), "counts": counts, "updates": [], "training_episodes": [],
        "source_sha256": source_hashes, "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "protocol": {"objective": config.objective, "score_source": "physical_simulator",
            "world_model": False, "new_human_demonstrations": False,
            "privileged_reward": "simulator goal coverage; RGB/pusher history actor",
            "score": "discounted shaped return for first chunk + frozen-rollout continuation; no value bootstrap",
            "cutoff": "finite scoring horizon; only genuine success zeros potential",
            "parameter_use": "updated denoiser at every original DDPM step, including t=0",
            "prediction_space": "epsilon, uniform discrete training timestep, coordinate mean squared error",
            "reference": "initial T checkpoint on its own generated plans at automated T contexts",
            "context_advance": "last IID candidate, independent of scores",
            "resume": "complete optimizer-update boundaries; full-prefix contact replay"}}
    collector, probe, logger = None, None, None
    diagnostic = DenoisingProbe(previous.get("denoising_probe") if previous else None)
    completed = 0
    session = None

    def deadline():
        if time.monotonic()-start > config.seconds:
            raise TimeoutError("Forward training/evaluation wall-time budget exhausted")

    def update_session():
        if session is not None:
            session["elapsed_seconds"] = time.monotonic()-start
            session["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
            session["status"] = report["status"]
            session["counts_delta"] = {key: counts[key]-value
                                       for key, value in session["counts_at_start"].items()}

    def recovery():
        update_session()
        save_recovery(output, config, agent, optimizer, report, completed, collector.active, diagnostic.rows)

    def log_values(values):
        values.update({"progress/update": completed,
            "progress/training_controls": counts["training_controls"],
            "progress/evaluation_controls": counts["evaluation_controls"],
            "runtime/session_seconds": time.monotonic()-start,
            "runtime/peak_cuda_bytes": torch.cuda.max_memory_allocated()})
        logger.log(values, counts["optimizer_minibatches"])

    def development_evaluation(episodes):
        score = evaluation_score(episodes)
        entry = {"update": completed, "training_controls": counts["training_controls"],
                 "episodes": episodes, "selection_score": list(score)}
        report.setdefault("evaluations", []).append(entry)
        log_values(evaluation_scalars(episodes))
        current = report.get("best_evaluation")
        if current is None or score > tuple(current["selection_score"]):
            report["best_evaluation"] = {k: v for k, v in entry.items() if k != "episodes"}
            recovery()
            storage_room(output, config, (output/"last.pt").stat().st_size)
            temporary = output/"best.tmp"
            try:
                shutil.copyfile(output/"last.pt", temporary)
                temporary.replace(output/"best.pt")
                report["best_checkpoint"] = {"file": "best.pt", "sha256": hash_file(output/"best.pt"),
                                              "update": completed}
            finally:
                temporary.unlink(missing_ok=True)
        recovery()

    try:
        official = OfficialDiffusionPolicy(source_root, checkpoint)
        agent = ForwardPolicy(official, config)
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.actor_lr)
        initial = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        frozen = frozen_digest(agent.network)
        active = {}
        if previous:
            agent.load(parent)
            optimizer.load_state_dict(parent["optimizer"])
            report = previous["report"]
            counts = report["counts"]
            active = previous["active_contexts"]
            completed = previous["completed_updates"]
            report.update(status="running", config=asdict(config))
            report.pop("error", None)
            report.pop("final", None)
            audit_resume_costs(resume, report)
            if counts["training_controls"] > config.max_train_controls:
                raise ValueError("Resume cap does not cover recorded post-checkpoint interaction")
            report.setdefault("resume_chain", []).append({"checkpoint_sha256": hash_file(resume),
                "completed_updates": completed, "counts_at_resume": dict(counts)})
            if report.get("best_checkpoint"):
                parent_best = Path(resume).parent/"best.pt"
                if not parent_best.is_file() or hash_file(parent_best) != report["best_checkpoint"]["sha256"]:
                    raise ValueError("Copy the matching best.pt beside the recovery checkpoint")
                storage_room(output, config, parent_best.stat().st_size)
                shutil.copyfile(parent_best, output/"best.pt")
        session = {"counts_at_start": {k: v for k, v in counts.items()
                                      if k not in ("control_physics_substeps", "denoising_steps")},
                   "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        report.setdefault("sessions", []).append(session)
        report["runtime"] = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                             "cuda": torch.version.cuda, "device": str(agent.device),
                             "dependency_lock_sha256": hash_file(ROOT/"uv.lock"),
                             "compute_capability": list(torch.cuda.get_device_capability())}
        report["base_sha256"] = official.spec["compact"]["sha256"]
        report["trainable_names"] = agent.names
        report["trainable_parameters"] = sum(p.numel() for p in agent.actor.parameters())
        logger = RunLogger(output, tracking, {"trainer": "forward", "config": asdict(config),
            "base_sha256": report["base_sha256"], "project_commit": report["project_commit"],
            "source_sha256": source_hashes,
            "resume_checkpoint_sha256": hash_file(resume) if resume else None})
        report["tracking"] = logger.state
        report["probe_protocol"] = "fixed baseline-generated plans/noise at t=0,25,50,99; diagnostic only, never optimized"
        report["checkpoint_selection"] = "development macro shape success, then final coverage; not final test scenes"
        estimate = sum(p.numel()*p.element_size() for p in agent.actor.parameters())*8+8*1024**2
        storage_room(output, config, estimate)
        cls = make_official_shape_env_class(load_official_source(source_root))
        if not previous:
            report["baseline"] = evaluate(agent, cls, config, counts, deadline, diagnostic.capture)
            log_values(evaluation_scalars(report["baseline"]))
        log_values(diagnostic.evaluate(agent))
        collector = BranchCollector(agent, cls, config, counts, report, deadline, active)
        for update in range(completed, config.updates):
            records, groups = [], []
            for _ in range(config.contexts_per_update):
                group = None
                for _ in config.shapes:
                    shape = config.shapes[counts["context_attempts"] % len(config.shapes)]
                    counts["context_attempts"] += 1
                    group = collector.collect(shape)
                    if group is not None:
                        break
                if group is None:
                    break
                records.extend(group.pop("records"))
                groups.append(group)
            if not records:
                report["stop_reason"] = "no_complete_group_within_control_budget_or_warmup_success"
                break
            probe = records[0]["condition"]
            def on_minibatch(metric):
                counts["optimizer_minibatches"] += 1
                if counts["optimizer_minibatches"] % logger.config.log_every == 0:
                    log_values({"train/"+k: v for k, v in metric.items()
                                if k not in ("noise_timesteps", "batch_size")})
            metrics = forward_update(agent, records, optimizer, deadline, on_minibatch)
            completed = update+1
            weight = sum(m["batch_size"] for m in metrics)
            summary = {k: sum(m[k]*m["batch_size"] for m in metrics)/weight
                       for k in ("loss", "T_retention_loss", "gradient_norm")}
            values = {"train/update_mean_"+k: v for k, v in summary.items()}
            for shape in config.shapes:
                selected = [g for g in groups if g["shape"] == shape and "candidates" in g]
                candidates = [c for g in selected for c in g["candidates"]]
                if candidates:
                    values.update({f"rollout/{shape}/score_mean": float(np.mean([c["score"] for c in candidates])),
                        f"rollout/{shape}/score_spread": float(np.mean([np.ptp([c["score"] for c in g["candidates"]]) for g in selected])),
                        f"rollout/{shape}/tied_group_fraction": float(np.mean([len(set(c["score"] for c in g["candidates"])) == 1 for g in selected])),
                        f"rollout/{shape}/contact_fraction": float(np.mean([r["contacts"] > 0 for c in candidates for r in c["steps"]])),
                        f"rollout/{shape}/candidate_success_rate": float(np.mean([c["success"] for c in candidates]))})
            values.update(diagnostic.evaluate(agent))
            log_values(values)
            report["updates"].append({"update": completed, "groups": groups, "metrics": metrics,
                                      "summary": summary, "scalars": values})
            compact_updates(report, config.report_detail_updates)
            recovery()
            if config.eval_every and completed % config.eval_every == 0:
                development_evaluation(evaluate(agent, cls, config, counts, deadline))
            write_report(output, config, report)
            print(json.dumps({"update": completed, "training_controls": counts["training_controls"],
                "groups": counts["candidate_groups"], "mean_gradient_norm": float(np.mean(
                    [m["gradient_norm"] for m in metrics]))}), flush=True)
            del records
            deadline()
        if not counts["candidate_groups"]:
            raise ValueError("No complete candidate group was collected; increase control budget")
        last_eval = report.get("evaluations", [])
        if last_eval and last_eval[-1]["update"] == completed:
            report["final"] = last_eval[-1]["episodes"]
        else:
            report["final"] = evaluate(agent, cls, config, counts, deadline)
            development_evaluation(report["final"])
        report["actor_l2_change"] = sum(float((v.detach().cpu()-initial[k]).double().square().sum())
            for k, v in agent.actor.state_dict().items())**.5
        report["actor_changed"] = report["actor_l2_change"] > 0
        report["frozen_parameters_unchanged"] = frozen == frozen_digest(agent.network)
        if not report["frozen_parameters_unchanged"]:
            raise AssertionError("Frozen checkpoint parameters changed")
        if probe is None:
            # Resume can reach its cap without a fresh group; use a real eval history.
            env = cls(shape="T", legacy=True)
            try:
                env.seed(config.eval_seeds[0])
                observation = env.reset()
                counts["reset_physics_substeps"] += 1
                probe = agent.encode([observation]*2)
            finally:
                env.close()
        recovery()
        rng = agent.generator.get_state()
        expected, _ = agent.sample(probe)
        counts["checkpoint_verification_plans"] += 1
        payload = torch.load(output/"last.pt", map_location=agent.device, weights_only=True)
        agent.load(payload)
        optimizer.load_state_dict(payload["optimizer"])
        agent.generator.set_state(rng)
        actual, _ = agent.sample(probe)
        counts["checkpoint_verification_plans"] += 1
        agent.generator.set_state(rng)
        np.testing.assert_array_equal(actual, expected)
        report["checkpoint_reload_action_parity"] = True
        report["status"] = "complete"
        # Include evaluation/verification accounting in the resumable final boundary.
        recovery()
        agent.load(torch.load(output/"last.pt", map_location=agent.device, weights_only=True))
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if collector:
            collector.close()
        counts["control_physics_substeps"] = 10*(counts["training_controls"]+counts["evaluation_controls"])
        counts["denoising_steps"] = 100*sum(counts[k] for k in ("training_plans", "reference_plans",
                                      "evaluation_plans", "checkpoint_verification_plans"))
        report["session_elapsed_seconds"] = time.monotonic()-start
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        report["counts_are_lower_bounds"] = report["status"] != "complete"
        if report["counts_are_lower_bounds"]:
            report["count_limit"] = "completed plans/controls; an interrupted model/physics call may have partial work"
        update_session()
        if logger:
            logger.finish(report["status"])
        write_report(output, config, report)
    return report
