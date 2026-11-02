"""Bounded T/L on-policy update and fresh-process reload verification.

Hard limits include failed reservations: 64 physical controls, eight sampled
plans INCLUDING two inference-only replay plans, and one step per optimizer.
This command is a smoke check, not a learning-curve training entry point.
"""

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("run", "resume-update", "verify"))
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--base-checkpoint", type=Path, required=True)
    result.add_argument("--base-sha256")
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--controls", type=int, default=48, help="T+L physical rollout controls; never more than 64")
    result.add_argument("--max-controls", type=int, default=64)
    result.add_argument("--max-plans", type=int, default=8, help="Includes two nonphysical replay plans")
    result.add_argument("--episode-limit", type=int, default=300)
    result.add_argument("--seed", type=int, default=5478)
    result.add_argument("--t-scene", type=int, default=3005478)
    result.add_argument("--l-scene", type=int, default=3005479)
    result.add_argument("--threads", type=int, default=4)
    return result


def validate_args(args):
    if args.base_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.base_sha256):
        raise ValueError("base-sha256 must be a recorded lowercase SHA-256 digest")
    if not 2 <= args.controls <= args.max_controls <= 64 or not 4 <= args.max_plans <= 8:
        raise ValueError("Require 2 <= controls <= max-controls <= 64 and 4 <= max-plans <= 8")
    available = args.max_plans - 2
    plan_caps = (available // 2, available - available // 2)
    control_caps = (args.controls // 2, args.controls - args.controls // 2)
    if any(controls > plans * 8 for controls, plans in zip(control_caps, plan_caps)):
        raise ValueError("Rollout controls exceed plan capacity after reserving two replay plans")
    if not 1 <= args.threads <= 4 or not 1 <= args.episode_limit <= 100000:
        raise ValueError("Use 1..4 CPU threads and a positive bounded episode limit")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in (args.seed, args.t_scene, args.l_scene)):
        raise ValueError("All seeds must be uint32")
    from tdyssey.official_collection import DEVELOPMENT_SEEDS
    if args.t_scene in DEVELOPMENT_SEEDS or args.l_scene in DEVELOPMENT_SEEDS:
        raise ValueError("Training scenes must exclude known development evaluation scenes")
    return control_caps, plan_caps


def write_json(path, value):
    from tdyssey.policy_rollout import _atomic_json
    _atomic_json(path, value)


def save_tensors(path, value):
    import torch
    with path.open("xb") as output:
        torch.save(value, output)
        output.flush()
        os.fsync(output.fileno())


def rollout_payload(fragments, critic):
    from tdyssey.policy_checkpoint import sample_to_payload
    return {"schema_version": 1, "initial_critic": {n: v.detach().cpu().clone() for n, v in critic.state_dict().items()},
            "fragments": [[{"sample": sample_to_payload(plan.sample),
                             "block": {**asdict(plan.block), "boundary": plan.block.boundary.value},
                             "shape": plan.shape, "reset_seed": plan.reset_seed,
                             "control_records": plan.control_records,
                             "final_observation": plan.final_observation, "features": plan.features}
                            for plan in fragment] for fragment in fragments]}


def decode_rollout(payload):
    from tdyssey.policy_checkpoint import sample_from_payload
    from tdyssey.policy_returns import ActionBlock
    from tdyssey.policy_rollout import RolloutPlan
    if payload.get("schema_version") != 1:
        raise ValueError("Unknown rollout buffer schema")
    return [[RolloutPlan(sample=sample_from_payload(plan["sample"]), block=ActionBlock(**plan["block"]),
                         **{k: plan[k] for k in ("shape", "reset_seed", "control_records", "final_observation", "features")})
             for plan in fragment] for fragment in payload["fragments"]]


def run_update(args, budget, report):
    import numpy as np
    import torch
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import OfficialDiffusionPolicy, load_official_source
    from tdyssey.policy_checkpoint import save_update_checkpoint, sample_to_payload
    from tdyssey.policy_rollout import collect_fragment
    from tdyssey.policy_rl import DenoisingPolicy
    from tdyssey.policy_training import (VisualCritic, UpdateConfig, make_optimizers, measured_phase,
                                         prepare_batch, update_once)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    control_caps, plan_caps = validate_args(args)
    run_config = {name: getattr(args, name) for name in
                  ("seed", "t_scene", "l_scene", "controls", "max_controls", "max_plans", "episode_limit")}
    with measured_phase(report["timings"], "load", args.device):
        official = OfficialDiffusionPolicy(args.source_root, args.base_checkpoint, device=args.device,
                                           checkpoint_sha256=args.base_sha256)
        actor = DenoisingPolicy(official, subset="output")
        critic = VisualCritic(actor.network.obs_feature_dim * actor.network.n_obs_steps).to(args.device)
        actor_optimizer, critic_optimizer = make_optimizers(actor, critic)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    base_identity = {"sha256": official.checkpoint_sha256, "bytes": args.base_checkpoint.stat().st_size}
    report.update(base_identity=base_identity, parameter_selection=asdict(actor.selection),
                  critic_config=critic.config, critic_parameters=sum(p.numel() for p in critic.parameters()))
    if args.mode == "resume-update":
        payload = torch.load(args.run_dir / "rollout.pt", map_location="cpu", weights_only=True)
        fragments = decode_rollout(payload)
        critic.load_state_dict(payload["initial_critic"], strict=True)
        generator.set_state(payload["policy_generator_state"])
        if payload["base_identity"] != base_identity:
            raise ValueError("Buffer belongs to a different base checkpoint")
        if payload["run_config"] != run_config:
            raise ValueError("Resume arguments differ from the saved rollout provenance")
    else:
        if any(budget.snapshot()["reserved"].values()) or budget.snapshot()["reset_attempts"]:
            raise ValueError("Existing budget has attempts; use saved-buffer resume, never restart collection")
        fragments = []
        env_class = make_official_shape_env_class(load_official_source(args.source_root))
        with measured_phase(report["timings"], "rollout", args.device):
            for shape, seed, controls, plans in zip(("T", "L"), (args.t_scene, args.l_scene), control_caps, plan_caps):
                env = env_class(shape=shape, legacy=True)
                try:
                    fragment = collect_fragment(official, actor, critic, env, shape=shape, reset_seed=seed,
                                                generator=generator, budget=budget, max_controls=controls,
                                                max_plans=plans, episode_limit=args.episode_limit,
                                                actor_version="milestone-b-before-update")
                    fragments.append(fragment)
                finally:
                    env.close()
        payload = rollout_payload(fragments, critic)
        payload.update(base_identity=base_identity, policy_generator_state=generator.get_state().cpu(), run_config=run_config)
        save_tensors(args.run_dir / "rollout.pt", payload)
    batch = prepare_batch(fragments)
    targets = {name: getattr(batch, name) for name in
               ("features", "old_values", "raw_advantages", "advantages", "returns")}
    target_path = args.run_dir / "targets.pt"
    if target_path.exists():
        if args.mode != "resume-update":
            raise FileExistsError(target_path)
        saved_targets = torch.load(target_path, map_location="cpu", weights_only=True)
        if set(saved_targets) != set(targets) or any(not torch.equal(saved_targets[k], v) for k, v in targets.items()):
            raise ValueError("Saved targets differ from this fixed on-policy buffer")
    else:
        save_tensors(target_path, targets)
    report["rollout"] = {"controls": sum(p.block.num_controls for p in batch.plans), "plans": len(batch.plans),
                         "by_shape": {shape: {"controls": sum(p.block.num_controls for p in batch.plans if p.shape == shape),
                                               "plans": sum(p.shape == shape for p in batch.plans),
                                               "final_coverage": next(p.control_records[-1]["coverage"] for p in reversed(batch.plans)
                                                                      if p.shape == shape)} for shape in ("T", "L")},
                         "boundaries": [p.block.boundary.value for p in batch.plans],
                         "physics_substeps": sum(r["control_physics_substeps"] for p in batch.plans for r in p.control_records),
                         "reset_settling_substeps": budget.snapshot()["reset_completed"],
                         "sampling_denoising_steps": len(batch.plans) * 100}
    report["targets"] = {name: getattr(batch, name).tolist() for name in ("old_values", "raw_advantages", "advantages", "returns")}
    write_json(args.run_dir / "report.json", {**report, "budget": budget.snapshot(), "status": "collected"})
    config = UpdateConfig()
    update = update_once(actor, critic, batch, actor_optimizer, critic_optimizer, budget=budget, config=config)
    report["update"] = update
    if (update["actor_grad_norm"] <= 0 or update["critic_grad_norm"] <= 0
            or update["actor_change"]["l2"] <= 0 or update["critic_change"]["l2"] <= 0):
        raise ValueError("Bounded real check did not produce both required nonzero weight changes")
    report["update_config"] = asdict(config)
    # Save RNG immediately before the inference-only replay. Reload reconstructs
    # that exact state in a different process; there is no physical replay.
    metadata = {"purpose": "one-update-smoke-check", "update_config": asdict(config),
                "seed": args.seed, "scenes": {"T": args.t_scene, "L": args.l_scene},
                "budgets": budget.snapshot(), "critic_lr": 3e-4, "actor_lr": 1e-5}
    with measured_phase(report["timings"], "save", args.device):
        report["checkpoint"] = save_update_checkpoint(args.run_dir / "update.pt", actor, critic,
            actor_optimizer, critic_optimizer, base_identity=base_identity, generator=generator, metadata=metadata)
    observations = {k: v.to(args.device) for k, v in batch.plans[0].sample.observations.items()}
    reservation = budget.reserve("plan")
    with measured_phase(report["timings"], "reference_inference_replay", args.device):
        reference = actor.sample(observations, generator=generator, actor_version="milestone-b-after-update")
    budget.mark_completed(reservation)
    record = batch.plans[0].sample
    from tdyssey.policy_checkpoint import sample_from_payload
    record = sample_from_payload(sample_to_payload(record), device=args.device)
    with torch.no_grad():
        probabilities = actor.recompute_log_probs(record).cpu()
        values = critic(actor.encode(observations)).cpu()
    save_tensors(args.run_dir / "probe.pt", {"sample": sample_to_payload(batch.plans[0].sample),
        "reference_replay": sample_to_payload(reference), "updated_log_prob": probabilities,
        "critic_value": values, "local_generator_after": generator.get_state().cpu()})
    report["reference_replay"] = {"physical_controls": 0, "sampled_plans": 1, "denoising_steps": 100}
    return report


def verify(args, budget, report):
    import torch
    from tdyssey.policy_checkpoint import load_update_checkpoint, restore_optimizers, sample_from_payload
    from tdyssey.policy_training import make_optimizers, measured_phase, state_fingerprint
    if os.getpid() == report["pid"]:
        raise ValueError("Reload evidence requires a genuinely different process")
    if any(budget.snapshot()["completed"][name] != 1 for name in ("actor_step", "critic_step")):
        raise ValueError("Verify requires exactly one completed actor and critic update")
    with measured_phase(report["timings"], "reload", args.device):
        loaded = load_update_checkpoint(args.run_dir / "update.pt", source_root=args.source_root,
            base_checkpoint=args.base_checkpoint, device=args.device, restore_rng=True)
        actor_opt, critic_opt = make_optimizers(loaded.actor, loaded.critic)
        restore_optimizers(loaded, actor_opt, critic_opt)
        probe = torch.load(args.run_dir / "probe.pt", map_location="cpu", weights_only=True)
    record = sample_from_payload(probe["sample"], device=args.device)
    with measured_phase(report["timings"], "reload_probability_check", args.device):
        with torch.no_grad():
            probability = loaded.actor.recompute_log_probs(record).cpu()
            value = loaded.critic(loaded.actor.encode(record.observations)).cpu()
        torch.testing.assert_close(probability, probe["updated_log_prob"], rtol=0, atol=0)
        torch.testing.assert_close(value, probe["critic_value"], rtol=0, atol=0)
        expected = report["update"]["frozen_digest_after"]
        if state_fingerprint(loaded.actor.network, exclude=loaded.actor.selection.names) != expected:
            raise ValueError("Reconstructed frozen state differs from the updated actor")
        if loaded.generator is None:
            raise ValueError("Cannot replay the saved local RNG on this device")
        # Check optimizer restoration without making a second optimizer step.
        optimizer_steps = {}
        for name, optimizer in (("actor", actor_opt), ("critic", critic_opt)):
            states = [float(state["step"].item()) for state in optimizer.state.values()]
            if not states or set(states) != {1.0}:
                raise ValueError("Reloaded Adam states do not represent exactly one update")
            optimizer_steps[name] = {"state_tensors": len(states), "step": 1}
    reservation = budget.reserve("plan")
    with measured_phase(report["timings"], "fresh_process_inference_replay", args.device):
        replay = loaded.actor.sample(record.observations, generator=loaded.generator,
                                      actor_version="milestone-b-after-update")
    budget.mark_completed(reservation)
    reference = sample_from_payload(probe["reference_replay"])
    for name in ("chain", "actions", "targets", "old_log_prob"):
        torch.testing.assert_close(getattr(replay, name).cpu(), getattr(reference, name), rtol=0, atol=0)
    torch.testing.assert_close(loaded.generator.get_state().cpu(), probe["local_generator_after"], rtol=0, atol=0)
    report["reload"] = {"new_process": True, "pid": os.getpid(), "physical_controls": 0,
                        "sampled_plans": 1, "denoising_steps": 100, "optimizer_states": optimizer_steps,
                        "probability_max_error": 0.0, "action_chain_targets_logprob_bitwise_equal": True,
                        "local_rng_equal": True, "critic_value_equal": True, "frozen_state_equal": True}
    return report


def main():
    args = parser().parse_args()
    try:
        validate_args(args)
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(args.run_dir / "hf-cache"))
        import torch
        from tdyssey.policy_rollout import HardBudget
        torch.set_num_threads(args.threads)
        args.run_dir = args.run_dir.resolve()
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with HardBudget(args.run_dir / "budget.json", max_controls=args.max_controls, max_plans=args.max_plans) as budget:
            if args.mode == "run" and (args.run_dir / "report.json").exists():
                raise ValueError("Run already exists; use resume-update or verify without new collection")
            if args.mode == "resume-update" and any(budget.snapshot()["reserved"][kind] for kind in ("actor_step", "critic_step")):
                raise ValueError("An optimizer step was already reserved; another update is forbidden")
            if args.mode == "verify":
                report = json.loads((args.run_dir / "report.json").read_text(encoding="utf-8"))
                if report.get("reload"):
                    raise ValueError("Reload verification already consumed its plan; refusing another replay")
                if report.get("error") and not report.get("verification_failures"):
                    report["verification_failures"] = [{"error": report["error"],
                        "reserved_at_failure": report["budget"]["reserved"],
                        "stage": "reload validation before inference replay"}]
            else:
                report = {"purpose": "bounded-one-update", "pid": os.getpid(), "timings": {},
                          "device": args.device, "torch": torch.__version__, "threads": args.threads,
                          "limits": {"controls": args.max_controls, "plans_including_replays": args.max_plans},
                          "base_checkpoint": str(args.base_checkpoint.resolve()), "source_root": str(args.source_root.resolve())}
            try:
                report = verify(args, budget, report) if args.mode == "verify" else run_update(args, budget, report)
                report["status"] = "verified" if args.mode == "verify" else "updated-awaiting-fresh-process-verify"
                report.pop("error", None)
                report["budget"] = budget.snapshot()
                # Exclude the live owner lock and the self-sized JSON report.
                report["storage"] = {path.name: path.stat().st_size for path in args.run_dir.iterdir()
                                     if path.is_file() and path.name not in {"budget.json.lock", "report.json"}}
                write_json(args.run_dir / "report.json", report)
                print(json.dumps({"status": report["status"], "budget": report["budget"]["reserved"],
                                  "report": str(args.run_dir / "report.json")}))
                return 0
            except BaseException as exc:
                if args.mode == "verify":
                    report.setdefault("verification_failures", []).append({"error": f"{type(exc).__name__}: {exc}",
                        "reserved_at_failure": budget.snapshot()["reserved"], "pid": os.getpid()})
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}", budget=budget.snapshot())
                write_json(args.run_dir / "report.json", report)
                raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
