"""Structured privileged PushT world model (v2): exact pusher + contact geometry, learned object motion.

Physics facts used (audited in docs/research/notes/supermimic_adaptation.md): the pusher is a kinematic
disk (radius 15) driven per control by 10 substeps of v <- v + 0.01*(100(a-p) - 20v), p <- p + 0.01 v,
unaffected by contact; space.damping = 0 so the object is quasi-static (velocity 0 after any no-contact
control); letters are unions of body-frame axis-aligned rectangles.
Model: state in centre-of-mass (COG) coordinates [cx, cy, theta, vx, vy, omega, px, py, pvx, pvy].
  pusher   exact 10-substep PD integration
  contact  signed distance from the pusher disk to the letter at every substep (exact geometry)
  object   MLP in the OBJECT frame on substep pusher positions/velocities, signed distances, carried
           velocity and shape; outputs (dx, dy, dtheta) in the object frame and end velocities, gated by a
           contact gate: hard rule (no motion if the swept disk stays > 3 px away and the object is at
           rest) at evaluation, a steep sigmoid of the same rule while training/imagining.
Fit on k-step autoregressive windows of logged transitions (SuperMimic eq. 1, state term), contact x3.
  train: .venv/bin/python -B wm2.py train --data $RUNS/dyn-data --out $RUNS/wm2/v1
"""

import argparse
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
from torch import nn

import wm_state
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import load_official_source

SHAPES = wm_state.SHAPES
RADIUS, MARGIN = 15.0, 3.0


def check_shapes(blob):
    """A model trained on one letter set must run with the same TDYSSEY_SHAPES (old checkpoints: the original five)."""
    trained = tuple(blob.get("shapes", ("T", "T_narrow", "Y", "D", "L")))
    if trained != SHAPES:
        raise SystemExit(f"world model trained on {trained}, but TDYSSEY_SHAPES gives {SHAPES}")


def geometry():
    """Per shape: rectangles in the COG frame (xmin, ymin, xmax, ymax), COG offset, goal pose (COG frame)."""
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
    rects, cogs, goals = [], [], []
    for name in SHAPES:
        env = cls(shape=name, legacy=True); env.seed(0); env.reset()
        cog = np.array(env.block.center_of_gravity, np.float32)
        r = []
        for poly in env.block.shapes:
            v = np.array(poly.get_vertices(), np.float32) - cog
            r.append([v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()])
        gx, gy, gt = env.goal_pose
        c, s = np.cos(gt), np.sin(gt)
        goals.append([gx + c*cog[0] - s*cog[1], gy + s*cog[0] + c*cog[1], gt])
        rects.append(r); cogs.append(cog)
        env.close()
    R = max(len(r) for r in rects)
    pad = [r + [r[-1]]*(R-len(r)) for r in rects]
    return (torch.tensor(np.array(pad, np.float32)), torch.tensor(np.array(cogs, np.float32)),
            torch.tensor(np.array(goals, np.float32)))


def rot(x, y, th):
    c, s = torch.cos(th), torch.sin(th)
    return c*x - s*y, s*x + c*y


def box_sdf(q, rects):
    """q (B, P, 2) local points, rects (B, R, 4) -> (B, P) signed distance to the union of boxes."""
    cx, cy = (rects[..., 0]+rects[..., 2])/2, (rects[..., 1]+rects[..., 3])/2
    hx, hy = (rects[..., 2]-rects[..., 0])/2, (rects[..., 3]-rects[..., 1])/2
    dx = (q[..., 0:1]-cx[:, None]).abs()-hx[:, None]
    dy = (q[..., 1:2]-cy[:, None]).abs()-hy[:, None]
    outside = torch.sqrt(dx.clamp(min=0)**2 + dy.clamp(min=0)**2 + 1e-9)
    inside = torch.minimum(torch.maximum(dx, dy), torch.zeros_like(dx))
    return (outside + inside).min(-1).values


def to_cog(s, shape, cogs):
    """Logged body-origin state -> COG state (velocities are COG velocities only up to omega x r; kept as logged)."""
    cog = cogs[shape]
    ox, oy = rot(cog[..., 0], cog[..., 1], s[..., 2])
    out = s.clone()
    out[..., 0], out[..., 1] = s[..., 0] + ox, s[..., 1] + oy
    return out


class WM2(nn.Module):
    def __init__(self, rects, width=512, depth=4):
        super().__init__()
        self.register_buffer("rects", rects)
        d, layers = 10*2 + 10*2 + 10 + 3 + len(SHAPES), []
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 6))
        self.register_buffer("out_scale", torch.tensor([2.0, 2.0, 0.03, 20.0, 20.0, 0.5]))

    def pusher(self, p, v, a):
        ps, vs = [], []
        for _ in range(10):
            v = v + 0.01*(100*(a-p) - 20*v)
            p = p + 0.01*v
            ps.append(p); vs.append(v)
        return torch.stack(ps, 1), torch.stack(vs, 1)                     # (B, 10, 2)

    def forward(self, s, a, shape, hard=False, margin=MARGIN):
        ps, vs = self.pusher(s[:, 6:8], s[:, 8:10], a)
        th = s[:, 2]
        rx, ry = rot(ps[..., 0]-s[:, 0:1], ps[..., 1]-s[:, 1:2], -th[:, None])
        vx, vy = rot(vs[..., 0], vs[..., 1], -th[:, None])
        local = torch.stack([rx, ry], -1)
        sdf = box_sdf(local, self.rects[shape]) - RADIUS                    # (B, 10)
        cvx, cvy = rot(s[:, 3], s[:, 4], -th)
        carried = torch.stack([cvx, cvy, s[:, 5]], -1)
        feats = torch.cat([local.flatten(1)/50, torch.stack([vx, vy], -1).flatten(1)/100, sdf/20, carried/torch.tensor([20., 20., .5], device=s.device),
                           nn.functional.one_hot(shape, len(SHAPES)).float()], -1)
        out = self.net(feats)*self.out_scale
        near = margin - sdf.min(1).values
        moving = carried.abs().sum(1)
        if hard:
            gate = ((near >= 0) | (moving > 1e-6)).float()
        else:
            gate = torch.maximum(torch.sigmoid(near/0.5), (moving > 1e-6).float())
        out = out*gate[:, None]
        dx, dy = rot(out[:, 0], out[:, 1], th)
        nth = th + out[:, 2]
        nvx, nvy = rot(out[:, 3], out[:, 4], nth)
        return torch.stack([s[:, 0]+dx, s[:, 1]+dy, nth, nvx, nvy, out[:, 5], ps[:, -1, 0], ps[:, -1, 1],
                            vs[:, -1, 0], vs[:, -1, 1]], -1)

    def rollout(self, s0, actions, shape, hard=False):
        out, s = [], s0
        for t in range(actions.shape[1]):
            s = self(s, actions[:, t], shape, hard)
            out.append(s)
        return torch.stack(out, 1)


def prepare(d, cogs, device):
    s = torch.tensor(d["s"], device=device)
    shape = torch.tensor(d["shape"], device=device)
    s = to_cog(s, shape[:, None].expand(-1, s.shape[1]), cogs.to(device))
    return s, torch.tensor(d["a"], device=device), shape


def evaluate(model, S, A, SH, d, episodes, horizons=(1, 8, 32), samples=6000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for H in horizons:
        eps, t = wm_state.windows(d, episodes, H, rng, samples)
        e_t, t_t = torch.tensor(eps, device=S.device), torch.tensor(t, device=S.device)
        idx = t_t[:, None] + torch.arange(H+1, device=S.device)[None]
        st = S[e_t[:, None], idx]; ac = A[e_t[:, None], idx[:, :-1]]; sh = SH[e_t]
        contact = torch.tensor(d["contacts"][eps[:, None], t[:, None]+np.arange(1, H+1)[None]].max(1) > 0, device=S.device)
        with torch.no_grad():
            pred = model.rollout(st[:, 0], ac, sh, hard=True)[:, -1]
        true = st[:, -1]
        pos = (pred[:, :2]-true[:, :2]).norm(dim=1)
        ang = torch.rad2deg(torch.atan2(torch.sin(pred[:, 2]-true[:, 2]), torch.cos(pred[:, 2]-true[:, 2])).abs())
        pus = (pred[:, 6:8]-true[:, 6:8]).norm(dim=1)
        moved = (true[:, :2]-st[:, 0, :2]).norm(dim=1)
        for shp in range(len(SHAPES)):
            for name, sel in (("contact", contact), ("no_contact", ~contact)):
                m = sel & (sh == shp)
                if m.sum() < 20:
                    continue
                out[f"H{H}_{SHAPES[shp]}_{name}"] = {"n": int(m.sum()), "pos_med": float(pos[m].median()), "pos_p90": float(pos[m].quantile(.9)),
                    "ang_med": float(ang[m].median()), "ang_p90": float(ang[m].quantile(.9)), "pusher_max": float(pus[m].max()),
                    "moved_med": float(moved[m].median())}
    return out


def train(args):
    device = "cuda"
    d = wm_state.load(args.data)
    rects, cogs, goals = geometry()
    S, A, SH = prepare(d, cogs, device)
    n = len(d["seed"]); seeds = np.unique(d["seed"]); rng = np.random.default_rng(0)
    val_seeds = set(rng.choice(seeds, size=max(1, len(seeds)//7), replace=False).tolist())
    val = np.array([i for i in range(n) if d["seed"][i] in val_seeds]); tr = np.array([i for i in range(n) if d["seed"][i] not in val_seeds])
    model = WM2(rects.to(device), args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    contacts = torch.tensor(d["contacts"], device=device)
    print(json.dumps({"episodes": n, "train": len(tr), "val": len(val), "transitions": int(d["mask"][:, 1:].sum())}), flush=True)
    log, start = {"args": {k: str(v) for k, v in vars(args).items()}, "evals": []}, time.time()
    for step in range(1, args.steps+1):
        eps, t = wm_state.windows(d, tr, args.k, rng, args.batch)
        e_t, t_t = torch.tensor(eps, device=device), torch.tensor(t, device=device)
        idx = t_t[:, None] + torch.arange(args.k+1, device=device)[None]
        st, ac, sh = S[e_t[:, None], idx], A[e_t[:, None], idx[:, :-1]], SH[e_t]
        pred = model.rollout(st[:, 0], ac, sh)
        tgt = st[:, 1:]
        dpos = (pred[..., :2]-tgt[..., :2])/2.0
        dang = torch.atan2(torch.sin(pred[..., 2]-tgt[..., 2]), torch.cos(pred[..., 2]-tgt[..., 2]))/0.02
        dvel = (pred[..., 3:6]-tgt[..., 3:6])/torch.tensor([20., 20., .5], device=device)
        w = 1 + 2*(contacts[e_t[:, None], idx[:, 1:]] > 0).float()
        loss = ((dpos.square().sum(-1) + dang.square() + 0.1*dvel.square().sum(-1))*w).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, S, A, SH, d, val)
            log["evals"].append({"step": step, "loss": float(loss), **ev})
            brief = "  ".join(f"{k.replace('_contact','')}: {v['pos_med']:.2f}/{v['pos_p90']:.2f}px {v['ang_med']:.2f}/{v['ang_p90']:.2f}deg"
                              for k, v in ev.items() if k.startswith("H8_") and k.endswith("_contact") and "no_contact" not in k)
            nc = [v["pos_med"] for k, v in ev.items() if k.startswith("H8_") and "no_contact" in k]
            print(f"[wm2 {time.time()-start:5.0f}s] step {step} loss {float(loss):.3f}  H8 contact med/p90: {brief}  | H8 no-contact pos med max {max(nc) if nc else float('nan'):.3f}px", flush=True)
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "width": args.width, "depth": args.depth, "shapes": list(SHAPES)}, args.out/"model.pt")
            (args.out/"log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--k", type=int, default=8)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=3000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
