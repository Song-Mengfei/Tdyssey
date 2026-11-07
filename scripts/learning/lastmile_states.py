"""Collect near-goal start states for last-mile learning: the trained camera policy plays training scenes on all 27
shapes, and whenever the block covers at least --threshold of its goal (and the episode is not yet solved) the exact
simulator state is saved, together with the previous control's state (the policy looks at two camera frames) and how
the episode ended. Restoring a saved pair reproduces the policy's input exactly.

    .venv/bin/python -B scripts/learning/lastmile_states.py --student CKPT --start 20000 --scenes 40 --out DIR

Training scenes: seeds 800000 + 20*i with i in [20000, 30000) (reserved for last-mile work, D058).
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import base_rollouts  # noqa: E402
from tdyssey.collection import derived_seed  # noqa: E402
from tdyssey.envs.official import matched_reset  # noqa: E402
from tdyssey.official import physical_metrics  # noqa: E402

TRAINED27 = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"] + list("ABCGHIKMNOQRSVWXZ")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth near-goal control (neighbours are near copies)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not 20000 <= a.start and a.start + a.scenes <= 30000:
        raise SystemExit("last-mile scenes must use i in [20000, 30000)")
    agent, cls = base_rollouts.build(a.student)
    size = getattr(agent, "render_size", 96)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(a.start, a.start + a.scenes):
        seed = 800000 + 20 * i
        envs = {s: cls(shape=s, legacy=True, render_size=size) for s in TRAINED27}
        obs, _ = matched_reset(envs, seed, {"reset_physics_substeps": 0})
        lanes = [{"shape": s, "env": envs[s], "done": False, "history": deque([obs[s]] * 2, maxlen=2), "prev": envs[s].capture_state(),
                  "gen": torch.Generator(device=agent.device).manual_seed(derived_seed(42, "lastmile-harvest", seed)),
                  "states": [], "coverage": [physical_metrics(envs[s])["coverage"]], "near": 0} for s in TRAINED27]
        for start in range(0, 300, 8):
            active = [l for l in lanes if not l["done"]]
            if not active:
                break
            cond = agent.encode_batch([l["history"] for l in active])
            actions, _ = agent.sample_batch(cond, [l["gen"] for l in active])
            for j, lane in enumerate(active):
                for action in actions[j][:min(8, 300 - start)]:
                    before = lane["env"].capture_state()
                    o, _, done, _ = lane["env"].step(action)
                    lane["history"].append(o)
                    cov = physical_metrics(lane["env"])["coverage"]
                    lane["coverage"].append(cov)
                    if cov >= a.threshold and not done:
                        lane["near"] += 1
                        if lane["near"] % a.every == 1 or a.every == 1:
                            lane["states"].append({"control": len(lane["coverage"]) - 1, "coverage": round(cov, 5),
                                                   "previous": json.loads(before.to_json()),
                                                   "state": json.loads(lane["env"].capture_state().to_json())})
                    if done:
                        lane["done"] = True
                        break
        with gzip.open(a.out / f"scene{i}.jsonl.gz", "wt") as f:
            for l in lanes:
                cov = l["coverage"]
                for s in l["states"]:
                    f.write(json.dumps({"shape": l["shape"], "seed": seed, "scene_index": i, **s,
                                        "episode_solved": l["done"], "episode_controls": len(cov) - 1,
                                        "episode_final_coverage": round(cov[-1], 5), "episode_peak_coverage": round(max(cov), 5),
                                        "solved_within": (len(cov) - 1 - s["control"]) if l["done"] else None}) + "\n")
                l["env"].close()
        n = sum(len(l["states"]) for l in lanes)
        print(f"scene {i}: {n} near-goal states, solved {sum(l['done'] for l in lanes)}/27, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
