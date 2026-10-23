"""Physically matched candidate trials by replaying complete machine histories."""

from collections import deque
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np

from .collection import derived_seed, reset_issues
from .envs.official import official_shape
from .forward_protocol import group_feedback
from .official import physical_metrics, physical_state
from .official_collection import DEVELOPMENT_SEEDS
from .rl_protocol import block_return, shaped_reward
from .state import Snapshot


def observation_hash(history):
    digest = hashlib.sha256()
    for observation in history:
        for name in ("image", "agent_pos"):
            value = np.ascontiguousarray(observation[name])
            digest.update(str((name, value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def step_record(env, action):
    observation, raw, done, info = env.step(np.asarray(action, dtype=np.float32))
    row = {"action": np.asarray(action, dtype=np.float32).tolist(), "raw_reward": float(raw),
           "done": bool(done), "contacts": int(info["n_contacts"]),
           "state": physical_state(env), **physical_metrics(env)}
    return observation, row


def verify_row(actual, expected):
    """Solver/contact checks in addition to images or body positions alone."""
    if actual["done"] != expected["done"] or actual["contacts"] != expected["contacts"]:
        raise ValueError("Physical branch replay contact/termination mismatch")
    for name in ("raw_reward", "coverage", "iou"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=1e-9)
    for body in ("object", "pusher"):
        for name in actual["state"][body]:
            np.testing.assert_allclose(actual["state"][body][name], expected["state"][body][name],
                                       rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual["state"]["goal"], expected["state"]["goal"], rtol=0, atol=1e-9)


def replay_prefix(env, episode, counters, deadline):
    """Fresh initial space + every action, rebuilding warm contacts naturally."""
    observation = env.restore_state(Snapshot.from_dict(episode["initial_state"]))
    counters["initial_state_restores"] += 1
    history = deque([observation]*2, maxlen=2)
    for expected in episode["steps"]:
        deadline()
        observation, actual = step_record(env, expected["action"])
        counters["replay_controls"] += 1
        counters["training_controls"] += 1
        verify_row(actual, expected)
        if actual["done"]:
            raise ValueError("Cannot branch after terminal success")
        history.append(observation)
    return history


class BranchCollector:
    def __init__(self, agent, cls, config, counters, report, deadline, active=None):
        self.agent, self.config = agent, config
        self.counters, self.report, self.deadline = counters, report, deadline
        self.envs = {shape: cls(shape=shape, legacy=True) for shape in config.shapes}
        self.active = dict(active or {})
        self.report.setdefault("training_episodes", [])
        self.report.setdefault("reset_attempts", [])
        self.excluded = set(DEVELOPMENT_SEEDS) | {s+i for s in config.eval_seeds for i in range(20)}

    def close(self):
        for env in self.envs.values():
            env.close()

    def new_episode(self, shape):
        env = self.envs[shape]
        index = len(self.report["training_episodes"])
        for attempt in range(20):
            self.deadline()
            seed = derived_seed(self.config.seed, "forward-training", shape, index, attempt)
            if seed in self.excluded:
                continue
            env.seed(seed)
            env.reset()
            self.counters["reset_physics_substeps"] += 1
            state = replace(env.capture_state(), pusher_target=None)
            issues = reset_issues(official_shape(shape), state)
            self.report["reset_attempts"].append({"shape": shape, "seed": seed, "issues": issues})
            if not issues:
                episode = {"shape": shape, "seed": seed, "initial_state": asdict(state), "steps": []}
                self.report["training_episodes"].append(episode)
                self.active[shape] = index
                return episode
        raise ValueError("No safe training scene within reset budget")

    def run_actions(self, env, history, actions, category):
        rows = []
        for action in actions:
            self.deadline()
            observation, row = step_record(env, action)
            self.counters[category] += 1
            self.counters["training_controls"] += 1
            history.append(observation)
            rows.append(row)
            if row["done"]:
                break
        return rows

    def collect(self, shape):
        """One complete group only; budget admission includes every prefix replay."""
        config, counters = self.config, self.counters
        episode = (self.report["training_episodes"][self.active[shape]]
                   if shape in self.active else None)
        prefix = len(episode["steps"]) if episode else 0
        warmup = config.warmup_controls if episode is None else 0
        remaining = config.max_train_controls-counters["training_controls"]
        horizon = min(8*config.score_chunks, config.episode_steps-prefix-warmup)
        worst = warmup + config.group_size*(prefix+warmup+horizon)
        if horizon <= 0 or remaining < worst:
            return None
        if episode is None:
            episode = self.new_episode(shape)
        episode_index = self.active[shape]
        env = self.envs[shape]
        history = replay_prefix(env, episode, counters, self.deadline)
        # Optional policy-only context warmup, never a geometric reset curriculum.
        while len(episode["steps"]) < warmup:
            actions, _ = self.agent.sample(self.agent.encode(history), "old")
            counters["training_plans"] += 1
            rows = self.run_actions(env, history, actions[:warmup-len(episode["steps"])], "context_controls")
            episode["steps"].extend(rows)
            if rows[-1]["done"]:
                del self.active[shape]
                return {"records": [], "warmup_success": True, "shape": shape}
        prefix = len(episode["steps"])
        prefix_hash = hashlib.sha256(json.dumps(episode["steps"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        condition, observed_hash, reference_plan = None, None, None
        records, candidates, scores, last_rows = [], [], [], None
        for candidate in range(config.group_size):
            if candidate:
                history = replay_prefix(env, episode, counters, self.deadline)
            digest = observation_hash(history)
            if condition is None:
                condition, observed_hash = self.agent.encode(history), digest
                if shape == "T" and config.reference_weight:
                    # Separate sampler RNG: the retention query cannot change candidate noise.
                    rng = self.agent.generator.get_state()
                    _, reference_plan = self.agent.sample(condition, "reference")
                    self.agent.generator.set_state(rng)
                    counters["reference_plans"] += 1
            elif digest != observed_hash:
                raise ValueError("Branch conditioning image/history mismatch")
            actions, clean = self.agent.sample(condition, "old")
            counters["training_plans"] += 1
            rows, rewards = [], []
            before = physical_metrics(env)["coverage"]
            for chunk in range(config.score_chunks):
                if chunk:
                    actions, _ = self.agent.sample(self.agent.encode(history), "old")
                    counters["training_plans"] += 1
                size = min(8, config.episode_steps-prefix-len(rows))
                if size <= 0:
                    break
                part = self.run_actions(env, history, actions[:size], "branch_controls")
                for row in part:
                    reward = shaped_reward(before, row["coverage"], row["done"], config.gamma, config.shaping_beta)
                    row["training_reward"] = reward
                    rewards.append(reward)
                    before = row["coverage"]
                rows.extend(part)
                if rows[-1]["done"]:
                    break
            score = block_return(rewards, config.gamma)
            scores.append(score)
            records.append({"condition": condition.detach(), "plan": clean.detach(),
                            "reference_plan": reference_plan, "shape": shape})
            candidates.append({"normalized_plan": clean[0].cpu().tolist(), "score": score,
                               "success": rows[-1]["done"], "steps": rows})
            last_rows = rows
        feedback = group_feedback(scores, config.score_scale, config.anchor_margin)
        for index, record in enumerate(records):
            record.update(optimality=feedback["optimality"][index],
                          advantage=feedback["advantages"][index], anchor=feedback["anchor"])
        # Advance along the last IID candidate, independently of its reward.
        # Its continuation is also actual rollout-actor interaction.
        episode["steps"].extend(last_rows)
        if last_rows[-1]["done"] or len(episode["steps"]) >= config.episode_steps:
            del self.active[shape]
        counters["candidate_groups"] += 1
        return {"records": records, "shape": shape, "episode": episode_index, "prefix_controls": prefix,
                "prefix_sha256": prefix_hash, "observation_sha256": observed_hash,
                "score_source": "physical_simulator", "score_horizon_controls": horizon,
                "continuation": "fixed rollout policy", "feedback": feedback, "candidates": candidates}
