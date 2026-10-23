"""Bounded direct-simulator visual-policy RL; no learned dynamics or installs."""

from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .collection import derived_seed, reset_issues
from .dppo import TailPPO, ppo_update
from .envs.official import make_official_shape_env_class, matched_reset, official_shape
from .official import OfficialDiffusionPolicy, load_official_source, physical_metrics
from .official_collection import DEVELOPMENT_SEEDS
from .policy import hash_file
from .rl_protocol import advantages, block_return, shaped_reward
from .tracking import RunLogger, evaluation_scalars

ROOT = Path(__file__).resolve().parents[1]


def frozen_digest(network):
    digest = hashlib.sha256()
    for name, tensor in network.named_parameters():
        if not tensor.requires_grad:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(agent, cls, config, counters, deadline, on_plan=None):
    """Paired development scenes, original strict success, no policy updates."""
    saved_rng = agent.generator.get_state()
    envs = {shape: cls(shape=shape, legacy=True) for shape in getattr(config, "eval_shapes", ("T", "L"))}
    episodes = []
    try:
        for seed in config.eval_seeds:
            deadline()
            observations, resets = matched_reset(envs, seed, counters)
            for shape, env in envs.items():
                agent.generator.manual_seed(derived_seed(config.seed, "rl-eval", seed))
                history = deque([observations[shape]]*2, maxlen=2)
                rows, done = [], False
                for start in range(0, config.eval_steps, 8):
                    deadline()
                    condition = agent.encode(history)
                    actions, sample = agent.sample(condition)
                    counters["evaluation_plans"] += 1
                    if on_plan is not None:
                        on_plan(shape, seed, condition, sample)
                    for action in actions[:min(8, config.eval_steps-start)]:
                        deadline()
                        obs, raw, done, info = env.step(action)
                        counters["evaluation_controls"] += 1
                        history.append(obs)
                        rows.append({"action": action.tolist(), "raw_reward": float(raw),
                                     "done": bool(done), "contacts": int(info["n_contacts"]),
                                     **physical_metrics(env)})
                        if done:
                            break
                    if done:
                        break
                episodes.append({"shape": shape, "requested_seed": seed, "reset_attempts": resets,
                                 "initial_scene_seed": resets[-1]["seed"], "success": bool(done),
                                 "controls": len(rows), "peak_coverage": max(r["coverage"] for r in rows),
                                 "final_coverage": rows[-1]["coverage"], "steps": rows})
    finally:
        agent.generator.set_state(saved_rng)
        for env in envs.values():
            env.close()
    return episodes


def train(source_root, checkpoint, output, config, tracking=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required; no CPU training fallback")
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    output.mkdir(parents=True)
    counters = {key: 0 for key in ("training_controls", "training_plans", "evaluation_controls",
                                   "evaluation_plans", "checkpoint_verification_plans",
                                   "reset_physics_substeps", "optimizer_minibatches")}
    paths = ("tdyssey/dppo.py", "tdyssey/lora.py", "tdyssey/rl_protocol.py", "tdyssey/policy_training.py",
             "scripts/train_official_policy.py", "tdyssey/official.py", "tdyssey/envs/official.py")
    report = {"schema_version": 1, "kind": "official_visual_tail_ppo_pilot", "status": "running",
              "config": asdict(config), "counts": counters, "updates": [], "resets": [],
              "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {p: hash_file(ROOT/p) for p in paths},
              "protocol": {"world_model": False, "new_human_demonstrations": False,
                           "privileged_reward": "simulator goal coverage; no object state in actor/critic",
                           "variant": "tail-transition joint-density PPO; not an upstream DPPO reproduction",
                           "trainable_transitions": list(range(config.train_timesteps, 0, -1)),
                           "frozen_transitions": "early denoising and deterministic t=0",
                           "denoising_discount": 1., "reward": "completion plus coverage potential",
                           "timeout": "bootstrap final observation; cut GAE across reset",
                           "evaluation": "paired development scenes, not untouched final tests",
                           "resume": "not implemented; checkpoints contain final weights/optimizer state"}}
    envs = {}
    logger = RunLogger(output, tracking, {"trainer": "tail_ppo", "config": asdict(config),
                                        "source_sha256": report["source_sha256"]})
    report["tracking"] = logger.state

    def deadline():
        if time.monotonic()-start > config.seconds:
            raise TimeoutError("Training/evaluation wall-time budget exhausted")

    try:
        deadline()
        official = OfficialDiffusionPolicy(source_root, checkpoint)
        cls = make_official_shape_env_class(load_official_source(source_root))
        agent = TailPPO(official, config)
        initial = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
        frozen_before = frozen_digest(agent.network)
        report["runtime"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "device": str(agent.device)}
        report["base_sha256"] = official.spec["compact"]["sha256"]
        report["trainable_names"] = agent.names
        report["trainable_parameters"] = sum(p.numel() for p in agent.actor.parameters())
        estimated = sum(p.numel()*p.element_size() for p in agent.actor.parameters())*4 + 4*1024**2
        if estimated > config.max_bytes or shutil.disk_usage(output).free < estimated+8*1024**2:
            raise ValueError("Checkpoint estimate exceeds configured output budget or free storage")
        report["baseline"] = evaluate(agent, cls, config, counters, deadline)
        logger.log(evaluation_scalars(report["baseline"]), counters["optimizer_minibatches"])
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.actor_lr)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=config.critic_lr)
        episodes = {shape: 0 for shape in ("T", "L")}
        histories, elapsed = {}, {}
        excluded = set(DEVELOPMENT_SEEDS) | {s+i for s in config.eval_seeds for i in range(20)}
        for shape in ("T", "L"):
            envs[shape] = cls(shape=shape, legacy=True)

        def reset(shape):
            env = envs[shape]
            for attempt in range(20):
                deadline()
                seed = derived_seed(config.seed, "rl-training", shape, episodes[shape], attempt)
                if seed in excluded:
                    continue
                env.seed(seed)
                env.reset()
                counters["reset_physics_substeps"] += 1
                state = env.capture_state()
                issues = reset_issues(official_shape(shape), state)
                report["resets"].append({"shape": shape, "seed": seed, "issues": issues,
                                         "initial_state": asdict(state) if not issues else None})
                if not issues:
                    # Reset the rendering/controller target left by an earlier
                    # episode; do not restore a warm contact state.
                    env.latest_action, env.render_cache = None, None
                    histories[shape] = deque([env._get_obs()]*2, maxlen=2)
                    elapsed[shape] = 0
                    episodes[shape] += 1
                    return
            raise ValueError("No safe training reset within candidate budget")

        for update in range(config.updates):
            records = []
            for plan in range(config.plans_per_update):
                deadline()
                remaining = config.max_train_controls-counters["training_controls"]
                if remaining <= 0:
                    break
                shape = ("T", "L")[counters["training_plans"] % 2]
                if shape not in histories:
                    reset(shape)
                history, env = histories[shape], envs[shape]
                condition = agent.encode(history)
                with torch.no_grad():
                    value = float(agent.value(condition))
                actions, transitions = agent.sample(condition)
                counters["training_plans"] += 1
                rewards, rows = [], []
                before = physical_metrics(env)["coverage"]
                size = min(8, remaining, config.episode_steps-elapsed[shape])
                for action in actions[:size]:
                    deadline()
                    obs, raw, done, info = env.step(action)
                    elapsed[shape] += 1
                    counters["training_controls"] += 1
                    history.append(obs)
                    metrics = physical_metrics(env)
                    reward = shaped_reward(before, metrics["coverage"], bool(done), config.gamma, config.beta)
                    rewards.append(reward)
                    rows.append({"action": action.tolist(), "training_reward": reward,
                                 "raw_reward": float(raw), "done": bool(done),
                                 "contacts": int(info["n_contacts"]), **metrics})
                    before = metrics["coverage"]
                    if done:
                        break
                with torch.no_grad():
                    next_value = 0. if done else float(agent.value(agent.encode(history)))
                end = bool(done) or elapsed[shape] == config.episode_steps
                records.append({"shape": shape, "condition": condition, "transitions": transitions,
                                "value": value, "next_value": next_value,
                                "controls": len(rewards), "reward": block_return(rewards, config.gamma),
                                "terminated": bool(done), "episode_end": end, "steps": rows})
                if end:
                    del histories[shape]
            if not records:
                break
            adv, targets = [0.]*len(records), [0.]*len(records)
            for shape in ("T", "L"):
                indices = [i for i, r in enumerate(records) if r["shape"] == shape]
                a, v = advantages([records[i] for i in indices], config.gamma, config.gae_lambda)
                for index, aa, vv in zip(indices, a, v):
                    adv[index], targets[index] = aa, vv
            metrics = ppo_update(agent, records, adv, targets, optimizer, critic_optimizer)
            counters["optimizer_minibatches"] += len(metrics["minibatches"])
            if metrics["minibatches"]:
                logger.log({"train/"+k: float(np.mean([m[k] for m in metrics["minibatches"]]))
                            for k in metrics["minibatches"][0]} | {
                    "progress/update": update+1, "progress/training_controls": counters["training_controls"],
                    "rollout/mean_reward": float(np.mean([r["reward"] for r in records]))},
                    counters["optimizer_minibatches"])
            report["updates"].append({"update": update+1, "metrics": metrics,
                "plans": [{k: v for k, v in r.items() if k not in ("condition", "transitions")}
                          for r in records]})
            print(json.dumps({"update": update+1, "training_controls": counters["training_controls"],
                              "optimizer_minibatches": counters["optimizer_minibatches"],
                              "mean_training_reward": float(np.mean([r["reward"] for r in records]))}), flush=True)
            (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            del records
            deadline()
        report["final"] = evaluate(agent, cls, config, counters, deadline)
        logger.log(evaluation_scalars(report["final"]), counters["optimizer_minibatches"])
        report["actor_l2_change"] = sum(float((v.detach().cpu()-initial[k]).double().square().sum())
                                         for k, v in agent.actor.state_dict().items())**.5
        report["frozen_parameters_unchanged"] = frozen_before == frozen_digest(agent.network)
        if not report["frozen_parameters_unchanged"]:
            raise AssertionError("Frozen upstream parameters changed")
        if not counters["optimizer_minibatches"] or not report["actor_l2_change"] > 0:
            raise AssertionError("No actual actor update occurred")
        # Compact delta: original pinned checkpoint is required for reload.
        payload = agent.state()
        payload.update(actor_optimizer=optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict())
        torch.save(payload, output/"last.pt")
        # Reload the saved actor AND optimizer state into existing objects and
        # verify identical sampled actions under the same evaluation input/noise.
        probe = agent.encode(next(iter(histories.values()))) if histories else condition
        saved_rng = agent.generator.get_state()
        expected, _ = agent.sample(probe)
        counters["checkpoint_verification_plans"] += 1
        loaded = torch.load(output/"last.pt", map_location=agent.device, weights_only=True)
        agent.load(loaded)
        optimizer.load_state_dict(loaded["actor_optimizer"])
        critic_optimizer.load_state_dict(loaded["critic_optimizer"])
        agent.generator.set_state(saved_rng)
        actual, _ = agent.sample(probe)
        counters["checkpoint_verification_plans"] += 1
        np.testing.assert_array_equal(actual, expected)
        report["checkpoint_reload_action_parity"] = True
        report["checkpoint"] = {"file": "last.pt", "bytes": (output/"last.pt").stat().st_size,
                                "sha256": hash_file(output/"last.pt")}
        report["status"] = "complete"
    except BaseException as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for env in envs.values():
            env.close()
        counters["control_physics_substeps"] = 10*(counters["training_controls"]+counters["evaluation_controls"])
        counters["denoising_steps"] = 100*(counters["training_plans"]+counters["evaluation_plans"]
                                          +counters["checkpoint_verification_plans"])
        report["elapsed_seconds"] = time.monotonic()-start
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        logger.finish(report["status"])
        text = json.dumps(report, indent=2, allow_nan=False)+"\n"
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())+len(text.encode()) > config.max_bytes:
            report["status"], report["error"] = "failed", "Output budget exceeded"
            text = json.dumps(report, indent=2, allow_nan=False)+"\n"
        (output/"report.json").write_text(text)
    return report
