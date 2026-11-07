"""Diagnosis: can a policy that reads EXACT state copy the improved world-model teacher?

Same action interface as the camera Diffusion Policy (16-step pusher-target plans, the first `--execute` executed;
the official ConditionalUnet1D, DDPM training), but conditioned on the exact object/pusher state of the last two
steps plus the letter, instead of camera images. If it reaches the teacher's success, copying works and the camera
policy's gap is perception; if not, the copying itself (open-loop chunks, data, model) is the problem.
Privileged state input: a diagnostic only, not a deployable policy.

  train: .venv/bin/python -B state_student.py train --data $RUNS/teacher-noisy-demos --out $RUNS/state-student/v1
  eval:  .venv/bin/python -B state_student.py eval --ckpt $RUNS/state-student/v1/model.pt --shape D --scenes 50 \
             --execute 8 --out $RUNS/state-student/v1/eval-D-x8.json
"""
import argparse
import copy
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"third_party"/"diffusion_policy"))
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from diffusers import DDIMScheduler, DDPMScheduler  # noqa: E402

SHAPES = ("T", "T_narrow", "Y", "D", "L")
H = 16


def features(state, prev, shape_k):
    """state/prev: (N, 9) [ox, oy, angle, vx, vy, omega, px, py, goal angle] -> (N, F)."""
    def one(s):
        return [(s[:, 0]-256)/256, (s[:, 1]-256)/256, np.sin(s[:, 2]), np.cos(s[:, 2]), (s[:, 6]-256)/256, (s[:, 7]-256)/256]
    oh = np.eye(len(SHAPES))[shape_k]
    return np.concatenate([np.stack(one(state) + one(prev), 1), oh], 1).astype(np.float32)


class Student(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fdim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.unet = ConditionalUnet1D(input_dim=2, global_cond_dim=256, diffusion_step_embed_dim=128,
                                      down_dims=[128, 256, 512], kernel_size=5, n_groups=8, cond_predict_scale=True)

    def forward(self, x, t, f):
        return self.unet(x, t, global_cond=self.enc(f))


def load_data(data_dir):
    F, A = [], []
    for f in sorted(f for d in (data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]) for f in Path(d).glob("*.npz")):
        z = {k: v for k, v in np.load(f).items()}
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum())
            if n < 2:
                continue
            k = SHAPES.index(str(z["shape"][i]))
            st = z["state"][i, :n]; act = z["action"][i, :n-1]
            prev = np.concatenate([st[:1], st[:-1]])
            F.append(features(st[:n-1], prev[:n-1], np.full(n-1, k)))
            idx = np.minimum(np.arange(n-1)[:, None] + np.arange(H)[None], n-2)       # pad with the last action
            A.append(act[idx])
    return np.concatenate(F), (np.concatenate(A).astype(np.float32) - 256) / 256


def train(args):
    torch.manual_seed(0); dev = "cuda"
    F, A = load_data(args.data)
    print(f"[state-student] {len(F)} samples from {args.data}", flush=True)
    F, A = torch.tensor(F, device=dev), torch.tensor(A, device=dev)
    model = Student(F.shape[1]).to(dev); ema = copy.deepcopy(model).eval()
    sched = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    start = time.time()
    for step in range(args.steps):
        b = torch.randint(0, len(F), (args.batch,), device=dev)
        x0, f = A[b], F[b]
        noise = torch.randn_like(x0); t = torch.randint(0, 100, (len(b),), device=dev)
        loss = nn.functional.mse_loss(model(sched.add_noise(x0, noise, t), t, f), noise)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            d = min(0.999, (1+step)/(10+step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1-d)
        if step % 1000 == 0 or step == args.steps-1:
            print(f"[state-student {time.time()-start:5.0f}s] step {step} loss {loss.item():.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ema.state_dict(), "fdim": F.shape[1], "samples": len(F), "data": str(args.data)}, args.out/"model.pt")
    print(f"wrote {args.out/'model.pt'}")


@torch.no_grad()
def sample(model, f, steps=16, gen=None):
    sched = DDIMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    sched.set_timesteps(steps)
    x = torch.randn((len(f), H, 2), device=f.device, generator=gen)
    for t in sched.timesteps:
        x = sched.step(model(x, t.expand(len(f)).to(f.device), f), t, x).prev_sample
    return (x.clamp(-1, 1)*256 + 256).cpu().numpy()


def evaluate(args):
    from tdyssey.envs.official import make_official_shape_env_class, matched_reset
    from tdyssey.official import load_official_source, physical_metrics, physical_state
    dev = "cuda"; ck = torch.load(args.ckpt, map_location=dev)
    model = Student(ck["fdim"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party"/"diffusion_policy"))
    seeds = [args.seed_base+20*(args.start+i) for i in range(args.scenes)]
    envs, keep = [], []
    for seed in seeds:                       # same scene setup as mpc_teacher.py (paired with T; L for T itself)
        partner = "L" if args.shape == "T" else args.shape
        pair = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
        matched_reset(pair, seed, defaultdict(int)); env = pair[args.shape]; env._get_obs = lambda: None
        envs.append(env); keep.append(pair)
    E, k = len(envs), SHAPES.index(args.shape)
    gen = torch.Generator(device=dev).manual_seed(0)
    def read(env):
        s = physical_state(env); o, p = s["object"], s["pusher"]
        return np.array([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], s["goal"][2]])
    prev = [read(e) for e in envs]; plan = [None]*E; age = np.zeros(E, int)
    done = np.zeros(E, bool); cov = np.zeros(E); start = time.time()
    for step in range(300):
        idx = [i for i in range(E) if not done[i]]
        if not idx:
            break
        cur = [read(envs[i]) for i in idx]
        need = [j for j, i in enumerate(idx) if plan[i] is None or age[i] >= args.execute]
        if need:
            f = torch.tensor(features(np.array([cur[j] for j in need]), np.array([prev[idx[j]] for j in need]),
                                      np.full(len(need), k)), device=dev)
            chunks = sample(model, f, gen=gen)
            for c, j in zip(chunks, need):
                plan[idx[j]], age[idx[j]] = c, 0
        for j, i in enumerate(idx):
            _, _, d, _ = envs[i].step(plan[i][age[i]]); age[i] += 1; prev[i] = cur[j]
            cov[i] = physical_metrics(envs[i])["coverage"]
            if d:
                done[i] = True
        if step % 50 == 0:
            print(f"[eval {args.shape} {time.time()-start:4.0f}s] step {step} success {int(done.sum())}/{E} coverage {cov.mean():.3f}", flush=True)
    out = {"shape": args.shape, "execute": args.execute, "seeds": seeds, "success": done.tolist(), "final_coverage": cov.tolist()}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out))
    print(f"{args.shape} execute {args.execute}: success {int(done.sum())}/{E} final coverage {cov.mean():.3f} -> {args.out}")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", type=Path, nargs="+", required=True); t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000); t.add_argument("--batch", type=int, default=1024); t.add_argument("--lr", type=float, default=3e-4)
    e = sub.add_parser("eval"); e.add_argument("--ckpt", type=Path, required=True); e.add_argument("--shape", choices=SHAPES, required=True)
    e.add_argument("--scenes", type=int, default=50); e.add_argument("--start", type=int, default=0)
    e.add_argument("--seed-base", type=int, default=400000); e.add_argument("--execute", type=int, default=8)
    e.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    train(a) if a.mode == "train" else evaluate(a)


if __name__ == "__main__":
    main()
