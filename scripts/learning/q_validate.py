"""Does a learned value rank candidate plans better than the trainer's current score? (no training)

At training-like contexts (same warmup and policy-only advances as the trainer), G candidate
first plans are each followed by the frozen base policy to the END of the episode, K times with
independent noise. Per candidate:
  truth       mean over K draws of the full-episode shaped return (what we actually want to rank by)
  current     the trainer's signal: first draw, shaped return over its first 32 controls
  value       own 8 controls of reward + gamma^8 * V(state after them), V from value_fn.py
Reports, per shape, the across-candidate correlation of current and value with truth.
Usage (from the repository root):
  .venv/bin/python -B scripts/learning/q_validate.py --value $RUNS/value-fn/v1.pt --shapes Y --contexts 6
"""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.forward_policy import ForwardPolicy
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics, physical_state
from tdyssey.policy_branches import BranchCollector, replay_prefix, step_record
from tdyssey.rl_protocol import block_return, shaped_reward
import value_fn


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def state_vector(env):
    s = physical_state(env)
    o, p = s["object"], s["pusher"]
    return np.array([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], s["goal"][2]],
                    np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["T", "T_narrow", "Y", "D"])
    parser.add_argument("--contexts", type=int, default=6)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = ForwardConfig(**json.loads((ROOT/"configs/nft_nearmiss_batched.json").read_text()))
    config = replace(config, seed=args.seed)   # training-like contexts on fresh scenes
    G, K, gamma, beta, T_END = config.group_size, args.k, config.gamma, config.shaping_beta, config.episode_steps
    source = ROOT/"third_party/diffusion_policy"
    agent = ForwardPolicy(OfficialDiffusionPolicy(source, ROOT/"checkpoints/official-dp/ema_policy.pt"), config)
    agent.generator.manual_seed(args.seed)
    blob = torch.load(args.value, map_location=agent.device, weights_only=True)
    V = value_fn.Value().to(agent.device)
    V.load_state_dict(blob["state_dict"]); V.eval()
    cls = make_official_shape_env_class(load_official_source(source))
    counts, report = defaultdict(int), {}
    collector = BranchCollector(agent, cls, config, counts, report, lambda: None)
    out, start = {}, time.time()
    for shape in args.shapes:
        lanes = [cls(shape=shape, legacy=True) for _ in range(G*K)]
        rows = []
        while len(rows) < args.contexts:
            episode = collector.new_episode(shape)
            env = collector.envs[shape]
            history = replay_prefix(env, episode, counts, lambda: None)
            alive = True
            for target in (config.warmup_controls, config.warmup_controls+32, config.warmup_controls+64):
                while alive and len(episode["steps"]) < target:
                    actions, _ = agent.sample(agent.encode(history), "old")
                    part = collector.run_actions(env, history, actions[:target-len(episode["steps"])], "context_controls")
                    episode["steps"].extend(part)
                    alive = not part[-1]["done"]
                if not alive or len(rows) >= args.contexts:
                    break
                prefix = len(episode["steps"])
                histories = [replay_prefix(lane, episode, counts, lambda: None) for lane in lanes]
                condition = agent.encode(histories[0])
                first, _ = agent.sample_batch(condition.expand(G, -1), agent.child_generators(G), "old")
                gens = agent.child_generators(G*K)
                plans = [first[j // K] for j in range(G*K)]
                cov0 = physical_metrics(lanes[0])["coverage"]
                before = [cov0]*len(lanes)
                rewards, done = [[] for _ in lanes], [False]*len(lanes)
                after_own = [None]*len(lanes)
                chunk = 0
                while True:
                    active = [j for j in range(len(lanes)) if not done[j] and prefix+chunk*8 < T_END]
                    if not active:
                        break
                    if chunk:
                        follow, _ = agent.sample_batch(agent.encode_batch([histories[j] for j in active]),
                                                       [gens[j] for j in active], "old")
                        for j, a in zip(active, follow):
                            plans[j] = a
                    for j in active:
                        for action in plans[j][:min(8, T_END-prefix-chunk*8)]:
                            obs, row = step_record(lanes[j], action)
                            histories[j].append(obs)
                            rewards[j].append(shaped_reward(before[j], row["coverage"], row["done"], gamma, beta))
                            before[j] = row["coverage"]
                            if row["done"]:
                                done[j] = True
                                break
                        if chunk == 0:   # state after the candidate's own controls (identical across draws)
                            after_own[j] = (state_vector(lanes[j]), before[j], prefix+len(rewards[j]), done[j])
                    chunk += 1
                full = np.array([block_return(r, gamma) for r in rewards]).reshape(G, K)
                current = np.array([block_return(rewards[i*K][:32], gamma) for i in range(G)])
                own = [after_own[i*K] for i in range(G)]
                feats = np.stack([value_fn.features(s, c, t, value_fn.SHAPES.index(shape)) for s, c, t, _ in own])
                with torch.no_grad():
                    W = V(torch.tensor(feats, device=agent.device)).cpu().numpy()
                value = []
                for i in range(G):
                    state, cov_after, t_after, terminal = own[i]
                    n_own = t_after-prefix                       # controls the candidate itself executed
                    bootstrap = 0.0 if terminal else gamma**n_own*(W[i]-cov_after)   # V = W - coverage
                    value.append(block_return(rewards[i*K][:n_own], gamma)+bootstrap)
                value = np.array(value)
                truth = full.mean(1)
                rows.append({"prefix": prefix, "corr_current": corr(current, truth), "corr_value": corr(value, truth),
                             "corr_single_full_draw": corr(full[:, 0], truth), "truth_spread": float(truth.std()),
                             "within_draw_std": float(full.std(1).mean())})
                r = rows[-1]
                print(f"[q {time.time()-start:6.0f}s] {shape} prefix {prefix}: corr(current, truth) {r['corr_current']:+.2f}  "
                      f"corr(value, truth) {r['corr_value']:+.2f}  truth spread {r['truth_spread']:.3f} "
                      f"draw std {r['within_draw_std']:.3f}", flush=True)
        for lane in lanes:
            lane.close()
        valid = lambda k: [x[k] for x in rows if x[k] == x[k]]
        out[shape] = {"contexts": rows, **{f"mean_{k}": (st.mean(valid(k)) if valid(k) else None)
                                           for k in ("corr_current", "corr_value", "corr_single_full_draw")}}
        print(f"== {shape}: current {out[shape]['mean_corr_current']}  value {out[shape]['mean_corr_value']}", flush=True)
    collector.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"G": G, "K": K, "value": str(args.value), "results": out}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
