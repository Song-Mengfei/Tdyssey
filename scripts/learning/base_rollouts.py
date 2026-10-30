"""Frozen-base-policy rollouts on fresh scenes (no training).

  eval:    held-out evaluation, per-episode outcomes and Wilson 95% intervals per shape. Pools:
           --seed-base 300000 monitor test (100 scenes; repeated checks), 900000 final test (1000 scenes;
           reserved for the pre-registered acceptance test). --student evaluates a distilled checkpoint.
  collect: full-episode rollouts (seeds 600000 + 20*i) recording privileged simulator state, actions,
           pusher velocity, contacts and coverage at every control -> value-function data, and with
           --action-noise (pixels) diverse non-expert dynamics data for the world models.

Usage (from the repository root):
  .venv/bin/python -B scripts/learning/base_rollouts.py eval --scenes 100 --out $RUNS/test-eval/base.json
  .venv/bin/python -B scripts/learning/base_rollouts.py collect --start 0 --scenes 50 --out $RUNS/value-data/part0.npz
Scenes are spaced 20 seeds apart because matched_reset may advance up to 19 seeds per scene.
Evaluation and collection use derived per-scene policy noise exactly like the trainer's batched evaluation.
"""

import argparse
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

from tdyssey.collection import derived_seed
from tdyssey.envs.official import make_official_shape_env_class, matched_reset
from tdyssey.forward_policy import ForwardPolicy
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.official import OfficialDiffusionPolicy, load_official_source, physical_metrics, physical_state
from tdyssey.policy_training import evaluate_batched

SHAPES = ("T", "T_narrow", "Y", "D", "L")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k/n
    centre, half = (p+z*z/(2*n))/(1+z*z/n), z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (centre-half, centre+half)


def build(student=None):
    """Frozen base policy, or a distilled student (distill.py actor checkpoint) when `student` is given."""
    from dataclasses import replace
    config = ForwardConfig(**json.loads((ROOT/"configs/nft_nearmiss_batched.json").read_text()))
    payload = torch.load(student, map_location="cpu", weights_only=True) if student else None
    if payload:
        config = replace(config, trainable=payload["trainable"])
    source = ROOT/"third_party/diffusion_policy"
    agent = ForwardPolicy(OfficialDiffusionPolicy(source, ROOT/"checkpoints/official-dp/ema_policy.pt"), config)
    if payload:
        agent.actor.load_state_dict({k: v.to(agent.device) for k, v in payload["actor"].items()})
        if "encoder" in payload:                      # students trained with distill.py --train-encoder
            agent.network.obs_encoder.load_state_dict({k: v.to(agent.device) for k, v in payload["encoder"].items()})
    return agent, make_official_shape_env_class(load_official_source(source))


def run_eval(args):
    agent, cls = build(args.student)
    seeds = [args.seed_base+20*(args.start+i) for i in range(args.scenes)]
    shapes = tuple(args.shapes) if args.shapes else SHAPES
    cfg = SimpleNamespace(eval_shapes=shapes, eval_seeds=seeds, eval_steps=300, seed=args.noise_seed, execute=args.execute)
    counts, start = defaultdict(int), time.time()
    episodes = evaluate_batched(agent, cls, cfg, counts, lambda: None)
    rows = [{k: e[k] for k in ("shape", "requested_seed", "initial_scene_seed", "success", "controls",
                                "peak_coverage", "final_coverage")} for e in episodes]
    summary = {}
    for shape in shapes:
        r = [x for x in rows if x["shape"] == shape]
        k = sum(x["success"] for x in r)
        summary[shape] = {"success": k, "n": len(r), "rate": k/len(r), "wilson95": wilson(k, len(r)),
                          "mean_final_coverage": float(np.mean([x["final_coverage"] for x in r])),
                          "mean_peak_coverage": float(np.mean([x["peak_coverage"] for x in r]))}
        print(f"{shape:9s} {k:3d}/{len(r)}  rate {k/len(r):.2f}  95% CI [{summary[shape]['wilson95'][0]:.2f}, "
              f"{summary[shape]['wilson95'][1]:.2f}]  final cov {summary[shape]['mean_final_coverage']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"policy": str(args.student) if args.student else "frozen base (official EMA)", "seeds": seeds, "policy_noise_seed": args.noise_seed,
                                    "counts": dict(counts), "seconds": time.time()-start,
                                    "summary": summary, "episodes": rows}, indent=1))
    print(f"wrote {args.out} in {time.time()-start:.0f}s")


@torch.no_grad()
def run_collect(args):
    agent, cls = build()
    seeds = [600000+20*(args.start+i) for i in range(args.scenes)]
    counts, lanes, start = defaultdict(int), [], time.time()
    noise_rng = np.random.default_rng(args.start*7919+int(args.action_noise*10))
    for seed in seeds:
        envs = {shape: cls(shape=shape, legacy=True) for shape in SHAPES}
        observations, resets = matched_reset(envs, seed, counts)
        for shape, env in envs.items():
            lanes.append({"shape": shape, "seed": resets[-1]["seed"], "env": env, "done": False,
                          "history": deque([observations[shape]]*2, maxlen=2),
                          "rows": [(physical_state(env), physical_metrics(env)["coverage"], 0)], "actions": [],
                          "generator": torch.Generator(device=agent.device).manual_seed(derived_seed(42, "value-data", seed))})
    for chunk_start in range(0, 300, 8):
        active = [lane for lane in lanes if not lane["done"]]
        if not active:
            break
        actions, _ = agent.sample_batch(agent.encode_batch([l["history"] for l in active]), [l["generator"] for l in active])
        for index, lane in enumerate(active):
            for action in actions[index][:min(8, 300-chunk_start)]:
                if args.action_noise:   # exploration for world-model data; the executed (noisy) action is recorded
                    action = np.clip(action + noise_rng.normal(0, args.action_noise, 2), 0, 512).astype(np.float32)
                lane["actions"].append(np.asarray(action, np.float32))
                obs, raw, done, info = lane["env"].step(action)
                lane["history"].append(obs)
                lane["rows"].append((physical_state(lane["env"]), physical_metrics(lane["env"])["coverage"], int(info["n_contacts"])))
                if done:
                    lane["done"] = True
                    break
        print(f"[collect {time.time()-start:5.0f}s] controls {chunk_start+8}/300 active {len(active)}", flush=True)
    n, T = len(lanes), 301
    state = np.zeros((n, T, 9), np.float32)   # object x,y,angle,vx,vy,omega; pusher x,y; goal angle
    pusher_velocity = np.zeros((n, T, 2), np.float32)
    action = np.zeros((n, T-1, 2), np.float32)  # action[t] moves state[t] -> state[t+1]
    coverage, contacts, mask = np.zeros((n, T), np.float32), np.zeros((n, T), np.int16), np.zeros((n, T), bool)
    for i, lane in enumerate(lanes):
        for t, (s, cov, c) in enumerate(lane["rows"]):
            o, p = s["object"], s["pusher"]
            state[i, t] = (*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], s["goal"][2])
            coverage[i, t], contacts[i, t], mask[i, t] = cov, c, True
            pusher_velocity[i, t] = p["velocity"]
        for t, a in enumerate(lane["actions"]):
            action[i, t] = a
        lane["env"].close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, state=state, coverage=coverage, contacts=contacts, mask=mask,
                        pusher_velocity=pusher_velocity, action=action, action_noise=args.action_noise,
                        success=np.array([l["done"] for l in lanes]), shape=np.array([l["shape"] for l in lanes]),
                        seed=np.array([l["seed"] for l in lanes]), goal=np.array([lanes[0]["rows"][0][0]["goal"]]))
    print(f"wrote {args.out}: {n} episodes in {time.time()-start:.0f}s; success "
          + ", ".join(f"{s} {sum(l['done'] for l in lanes if l['shape'] == s)}/{sum(l['shape'] == s for l in lanes)}" for s in SHAPES))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--scenes", type=int, default=100)
    e.add_argument("--noise-seed", type=int, default=42, help="Policy-noise stream; same scenes, different sampling")
    e.add_argument("--student", type=Path, help="Evaluate a distilled actor checkpoint instead of the base")
    e.add_argument("--seed-base", type=int, default=300000, help="300000 monitor-test pool; 900000 final-test pool")
    e.add_argument("--start", type=int, default=0, help="First scene index (to split a large evaluation)")
    e.add_argument("--execute", type=int, default=8, help="Controls executed per plan before replanning (official: 8)")
    e.add_argument("--shapes", nargs="*", default=None, help="Shapes to evaluate (default T T_narrow Y D L; any of the 26 capitals)")
    e.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--scenes", type=int, default=50)
    c.add_argument("--action-noise", type=float, default=0.0, help="Gaussian action noise (pixels) for diverse dynamics data")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_eval(args) if args.mode == "eval" else run_collect(args)


if __name__ == "__main__":
    main()
