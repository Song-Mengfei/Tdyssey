"""Does the frozen base policy keep its performance under deterministic DDIM sampling? (no training)

DSRL-style latent steering needs a deterministic map from initial noise to action plan. This
re-runs the held-out TEST evaluation (same scenes and per-scene initial noise as
base_rollouts.py eval) with diffusers DDIMScheduler built from the policy's own scheduler config,
eta = 0, for the given step counts, and compares with the DDPM-100 reference file.
Usage (from the repository root):
  .venv/bin/python -B scripts/learning/ddim_eval.py --steps 10 --reference $RUNS/test-eval/base.json --out $RUNS/test-eval/ddim10.json
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch
from diffusers import DDIMScheduler

import base_rollouts
from tdyssey.policy_training import evaluate_batched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent, cls = base_rollouts.build()
    ddim = DDIMScheduler.from_config(agent.network.noise_scheduler.config)
    ddim.set_timesteps(args.steps)

    @torch.no_grad()
    def sample_batch(self, condition, generators, mode="actor"):
        x = torch.cat([torch.randn((1, 16, 2), device=self.device, generator=g) for g in generators])
        for t in ddim.timesteps:
            x = ddim.step(self.predict(x, t, condition, mode), t, x, eta=0.0).prev_sample
        actions = self.network.normalizer["action"].unnormalize(x)[:, 1:9].cpu().numpy()
        return actions, x.detach()

    agent.sample_batch = MethodType(sample_batch, agent)
    seeds = [300000+20*i for i in range(args.scenes)]
    cfg = SimpleNamespace(eval_shapes=base_rollouts.SHAPES, eval_seeds=seeds, eval_steps=300, seed=42)
    start = time.time()
    episodes = evaluate_batched(agent, cls, cfg, defaultdict(int), lambda: None)
    reference = {}
    if args.reference and args.reference.is_file():
        for e in json.loads(args.reference.read_text())["episodes"]:
            reference[(e["shape"], e["requested_seed"])] = e["success"]
    summary = {}
    for shape in base_rollouts.SHAPES:
        rows = [e for e in episodes if e["shape"] == shape]
        k = sum(e["success"] for e in rows)
        paired = [(reference[(shape, e["requested_seed"])], e["success"]) for e in rows if (shape, e["requested_seed"]) in reference]
        only_ref = sum(a and not b for a, b in paired)
        only_new = sum(b and not a for a, b in paired)
        summary[shape] = {"success": k, "n": len(rows), "wilson95": base_rollouts.wilson(k, len(rows)),
                          "mean_final_coverage": float(np.mean([e["final_coverage"] for e in rows])),
                          "ddpm_only": only_ref, "ddim_only": only_new}
        print(f"{shape:9s} DDIM-{args.steps}: {k:3d}/{len(rows)}  final cov {summary[shape]['mean_final_coverage']:.3f}  "
              f"vs DDPM-100 discordant pairs: DDPM-only {only_ref}, DDIM-only {only_new}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"sampler": f"DDIM eta=0, {args.steps} steps", "seeds": seeds, "summary": summary,
                                    "seconds": time.time()-start,
                                    "episodes": [{k: e[k] for k in ("shape", "requested_seed", "success", "controls",
                                                                    "peak_coverage", "final_coverage")} for e in episodes]}, indent=1))
    print(f"wrote {args.out} in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
