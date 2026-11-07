"""How much of a candidate's training score comes from its own plan? (diagnostic, no training)

Reproduces the forward trainer's scoring at training-like contexts with the frozen initial
policy: policy warmup from a fresh training reset, G candidate first plans, each executed for
8 controls and continued for score_chunks-1 follow-up plans. Each candidate is scored K times
with independent follow-up noise. Reports per shape:
  plan share   between-candidate variance / total score variance (1 = score set by the plan)
  split-half   correlation of candidate scores between two independent follow-up draws (the
               reliability of the single-draw scores the trainer actually ranks)
Usage (from the repository root): .venv/bin/python -B scripts/learning/score_snr.py [--contexts 6] [--k 8]
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.forward_policy import ForwardPolicy
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics
from tdyssey.policy_branches import BranchCollector, replay_prefix, step_record
from tdyssey.rl_protocol import block_return, shaped_reward


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, default=6, help="Contexts per shape (3 per episode)")
    parser.add_argument("--k", type=int, default=8, help="Independent follow-up draws per candidate")
    parser.add_argument("--shapes", nargs="+", default=["T", "T_narrow", "Y", "D"])
    parser.add_argument("--out", type=Path, default=Path(os.environ["W"])/"logs/score_snr.json")
    args = parser.parse_args()
    config = ForwardConfig(**json.loads((ROOT/"configs/nft_nearmiss_batched.json").read_text()))
    G, K, chunks, gamma, beta = config.group_size, args.k, config.score_chunks, config.gamma, config.shaping_beta
    source = ROOT/"third_party/diffusion_policy"
    agent = ForwardPolicy(OfficialDiffusionPolicy(source, ROOT/"checkpoints/official-dp/ema_policy.pt"), config)
    agent.generator.manual_seed(20261025)
    cls = make_official_shape_env_class(load_official_source(source))
    counts, report = defaultdict(int), {}
    collector = BranchCollector(agent, cls, config, counts, report, lambda: None)
    results = {}
    start = time.time()
    for shape in args.shapes:
        lanes = [cls(shape=shape, legacy=True) for _ in range(G*K)]
        stats = []
        while len(stats) < args.contexts:
            episode = collector.new_episode(shape)
            env = collector.envs[shape]
            history = replay_prefix(env, episode, counts, lambda: None)
            alive = True
            for target in (config.warmup_controls, config.warmup_controls+32, config.warmup_controls+64):
                while alive and len(episode["steps"]) < target:   # policy-only advance, as in training
                    actions, _ = agent.sample(agent.encode(history), "old")
                    rows = collector.run_actions(env, history, actions[:target-len(episode["steps"])], "context_controls")
                    episode["steps"].extend(rows)
                    alive = not rows[-1]["done"]
                if not alive or len(stats) >= args.contexts:
                    break
                prefix = len(episode["steps"])
                histories = [replay_prefix(lane, episode, counts, lambda: None) for lane in lanes]
                condition = agent.encode(histories[0])
                first, _ = agent.sample_batch(condition.expand(G, -1), agent.child_generators(G), "old")
                gens = agent.child_generators(G*K)
                plans = [first[i // K] for i in range(G*K)]
                before = [physical_metrics(lane)["coverage"] for lane in lanes]
                rewards, done = [[] for _ in lanes], [False]*len(lanes)
                for chunk in range(chunks):
                    active = [j for j in range(len(lanes)) if not done[j] and prefix+chunk*8 < config.episode_steps]
                    if not active:
                        break
                    if chunk:
                        follow, _ = agent.sample_batch(agent.encode_batch([histories[j] for j in active]),
                                                       [gens[j] for j in active], "old")
                        for j, a in zip(active, follow):
                            plans[j] = a
                    for j in active:
                        for action in plans[j][:min(8, config.episode_steps-prefix-chunk*8)]:
                            obs, row = step_record(lanes[j], action)
                            histories[j].append(obs)
                            r = shaped_reward(before[j], row["coverage"], row["done"], gamma, beta)
                            rewards[j].append(r)
                            before[j] = row["coverage"]
                            if row["done"]:
                                done[j] = True
                                break
                S = np.array([block_return(r, gamma) for r in rewards]).reshape(G, K)
                between = float(S.mean(1).var(ddof=1))
                within = float(S.var(1, ddof=1).mean())
                pairs = [corr(S[:, a], S[:, b]) for a in range(K) for b in range(a+1, K)]
                pairs = [p for p in pairs if p == p]
                stats.append({"prefix": prefix, "between": between, "within": within,
                              "plan_share": between/(between+within) if between+within > 0 else float("nan"),
                              "split_half": st.mean(pairs) if pairs else float("nan"),
                              "score_mean": float(S.mean()), "score_std": float(S.std())})
                print(f"[snr {time.time()-start:6.0f}s] {shape} prefix {prefix}: plan share {stats[-1]['plan_share']:.2f}  "
                      f"split-half r {stats[-1]['split_half']:.2f}  score {S.mean():+.3f}+-{S.std():.3f}", flush=True)
        for lane in lanes:
            lane.close()
        valid = lambda key: [s[key] for s in stats if s[key] == s[key]]
        results[shape] = {"contexts": stats, "plan_share_mean": st.mean(valid("plan_share")) if valid("plan_share") else None,
                          "split_half_mean": st.mean(valid("split_half")) if valid("split_half") else None}
        print(f"== {shape}: plan share {results[shape]['plan_share_mean']}  split-half r {results[shape]['split_half_mean']}", flush=True)
    collector.close()
    args.out.write_text(json.dumps({"G": G, "K": K, "score_chunks": chunks, "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
