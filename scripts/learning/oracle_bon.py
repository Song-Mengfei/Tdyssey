"""Upper bound for reweighting/steering methods: simulator-oracle best-of-N with the frozen base.

At every decision (8 controls), N plans are sampled from the frozen base policy at the actual
observation. Each candidate is tried in its own environment restored by exact full-prefix replay:
its 8 controls, then LOOKAHEAD more controls of the base policy (one draw), scored by the
coverage at the end (success counts as 1). The best candidate's 8 controls are committed to the
real episode. This uses privileged simulation at decision time, so it is NOT a deployable policy:
it measures whether the base policy's own proposals contain solutions (pass for Y/D/L?).
Usage (from the repository root):
  .venv/bin/python -B scripts/learning/oracle_bon.py --shape Y --start 0 --scenes 6 --n 16 --lookahead 24 --out $RUNS/oracle/Y-0.json
Scenes use seeds 700000 + 20*i (separate from development, test and value-data scenes); the
same scenes are also run with the plain base policy (first candidate of every decision = base).
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

import base_rollouts
from tdyssey.collection import derived_seed
from tdyssey.envs.official import matched_reset
from tdyssey.official import physical_metrics
from tdyssey.policy_branches import replay_prefix, step_record
from fast_replay import fast_replay_prefix
from dataclasses import asdict, replace


@torch.no_grad()
def run_episode(agent, cls, shape, seed, n, lookahead, oracle, lanes, data_dir=None, fast=False):
    partner = "L" if shape == "T" else shape   # matched_reset needs T plus another shape; T's scene is unchanged
    envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
    _, resets = matched_reset(envs, seed, defaultdict(int))
    env = envs[shape]
    for other in envs.values():
        if other is not env:
            other.close()
    episode = {"initial_state": asdict(replace(env.capture_state(), pusher_target=None)), "steps": []}
    counts = defaultdict(int)
    history = replay_prefix(env, episode, counts, lambda: None)
    gen = torch.Generator(device=agent.device).manual_seed(derived_seed(42, "oracle", shape, seed))
    done, peak = False, physical_metrics(env)["coverage"]
    record = defaultdict(list)   # oracle decisions: training data for imitating the oracle's choices
    while not done and len(episode["steps"]) < 300:
        cond = agent.encode(history)
        seeds = torch.randint(0, 2**62, (n,), device=agent.device, generator=gen).tolist()
        gens = [torch.Generator(device=agent.device).manual_seed(int(s)) for s in seeds]
        plans, xs = agent.sample_batch(cond.expand(n, -1), gens)
        choice = 0
        if oracle and n > 1:
            budget = 300-len(episode["steps"])
            replay = fast_replay_prefix if fast else replay_prefix   # identical result; fast skips unused renders
            hist = [replay(lane, episode, counts, lambda: None) for lane in lanes[:n]]
            score, alive = np.zeros(n), [True]*n
            for i in range(n):
                for a in plans[i][:min(8, budget)]:
                    obs, row = step_record(lanes[i], a)
                    hist[i].append(obs)
                    if row["done"]:
                        score[i], alive[i] = 2.0, False   # success within its own controls
                        break
            steps = 8
            while steps < 8+lookahead and steps < budget and any(alive):
                idx = [i for i in range(n) if alive[i]]
                follow, _ = agent.sample_batch(agent.encode_batch([hist[i] for i in idx]), [gens[i] for i in idx])
                for j, i in enumerate(idx):
                    for a in follow[j][:min(8, budget-steps)]:
                        obs, row = step_record(lanes[i], a)
                        hist[i].append(obs)
                        if row["done"]:
                            score[i], alive[i] = 1.0+1.0/(1+steps), False
                            break
                steps += 8
            for i in range(n):
                if alive[i]:
                    score[i] = physical_metrics(lanes[i])["coverage"]
            choice = int(np.argmax(score))
            record["image"].append(np.stack([np.round(np.asarray(o["image"])*255) for o in history]).astype(np.uint8))
            record["agent_pos"].append(np.stack([np.asarray(o["agent_pos"]) for o in history]).astype(np.float32))
            record["candidate_x"].append(xs.cpu().numpy().astype(np.float16))   # normalized 16x2 plans
            record["score"].append(score.astype(np.float32))
            record["choice"].append(choice)
            record["control"].append(len(episode["steps"]))
        for a in plans[choice][:min(8, 300-len(episode["steps"]))]:
            obs, row = step_record(env, a)
            history.append(obs)
            episode["steps"].append(row)
            peak = max(peak, row["coverage"])
            if row["done"]:
                done = True
                break
    result = {"shape": shape, "seed": resets[-1]["seed"], "oracle": oracle, "n": n, "lookahead": lookahead,
              "success": done, "controls": len(episode["steps"]), "peak_coverage": peak,
              "final_coverage": episode["steps"][-1]["coverage"], "replay_controls": counts["replay_controls"]}
    env.close()
    if oracle and data_dir is not None and record["choice"]:
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_dir/f"{shape}-{result['seed']}.npz", shape=shape, success=done,
                            final_coverage=result["final_coverage"], **{k: np.stack(v) if k not in ("choice", "control")
                                                                       else np.array(v) for k, v in record.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fast-replay", action="store_true", help="Render only the 2 frames the policy needs")
    parser.add_argument("--oracle-only", action="store_true", help="Skip the paired plain-base episode (data collection)")
    parser.add_argument("--student", type=Path, help="Distilled actor checkpoint: sample candidates from the student (DAgger)")
    args = parser.parse_args()
    agent, cls = base_rollouts.build(args.student)
    lanes = [cls(shape=args.shape, legacy=True) for _ in range(args.n)]
    results, start = [], time.time()
    for i in range(args.scenes):
        seed = 700000+20*(args.start+i)
        for oracle in ((True,) if args.oracle_only else (False, True)):
            r = run_episode(agent, cls, args.shape, seed, args.n, args.lookahead, oracle, lanes, args.out.parent/"data",
                            args.fast_replay)
            results.append(r)
            print(f"[oracle {time.time()-start:6.0f}s] {args.shape} seed {seed} {'oracle' if oracle else 'base  '}: "
                  f"success {r['success']} controls {r['controls']} peak {r['peak_coverage']:.3f} final {r['final_coverage']:.3f}",
                  flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"shape": args.shape, "n": args.n, "lookahead": args.lookahead, "episodes": results}, indent=1))
    for lane in lanes:
        lane.close()
    for oracle in (False, True):
        r = [x for x in results if x["oracle"] == oracle]
        if not r:
            continue
        print(f"== {args.shape} {'oracle' if oracle else 'base'}: success {sum(x['success'] for x in r)}/{len(r)}  "
              f"mean final {np.mean([x['final_coverage'] for x in r]):.3f}  mean peak {np.mean([x['peak_coverage'] for x in r]):.3f}")


if __name__ == "__main__":
    main()
