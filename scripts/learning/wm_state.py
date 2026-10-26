"""Privileged state-space PushT world model (SuperMimic-style), trained on logged simulator transitions.

State s = [object x, y, angle, vx, vy, omega, pusher x, y, vx, vy]; action a = pusher target (x, y);
context = shape one-hot. f(s, a, shape) = s + delta(s, a, shape), an MLP on normalized features
(angle enters as sin/cos). Trained on k-step autoregressive windows rolled forward on logged actions
(SuperMimic: k=8), loss = normalized state error summed over the window. Validation on held-out
SCENES (by seed): open-loop errors at horizons 1/8/32, split by contact, against persistence.

  train: .venv/bin/python -B wm_state.py train --data $RUNS/dyn-data --out $RUNS/wm-state/v1
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

SHAPES = ("T", "T_narrow", "Y", "D", "L")
STATE = ("ox", "oy", "angle", "vx", "vy", "omega", "px", "py", "pvx", "pvy")


def load(data_dir):
    parts = [np.load(p) for p in sorted(Path(data_dir).glob("part*.npz")) if "action" in np.load(p).files]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    state, pv = cat("state"), cat("pusher_velocity")
    s = np.concatenate([state[..., :8], pv], -1).astype(np.float32)              # (n, 301, 10)
    return {"s": s, "a": cat("action").astype(np.float32), "mask": cat("mask"), "contacts": cat("contacts"),
            "coverage": cat("coverage"),
            "shape": np.array([SHAPES.index(x) for x in cat("shape")]), "seed": cat("seed"),
            "noise": np.concatenate([np.full(len(p["seed"]), float(p["action_noise"])) for p in parts]),
            "files": len(parts)}


class StateWM(nn.Module):
    def __init__(self, stats, width=1024, depth=3):
        super().__init__()
        for k, v in stats.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        layers, d = [], 11+2+len(SHAPES)
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 10))

    def features(self, s, a, shape):
        x = (s - self.s_mean)/self.s_std
        feats = torch.cat([x[..., :2], torch.sin(s[..., 2:3]), torch.cos(s[..., 2:3]), x[..., 3:],
                           (a - self.a_mean)/self.a_std, nn.functional.one_hot(shape, len(SHAPES)).float()], -1)
        return feats

    def forward(self, s, a, shape):
        return s + self.net(self.features(s, a, shape))*self.d_std

    def rollout(self, s0, actions, shape):
        """s0 (B, 10), actions (B, H, 2) -> predicted states (B, H, 10)."""
        out, s = [], s0
        for t in range(actions.shape[1]):
            s = self(s, actions[:, t], shape)
            out.append(s)
        return torch.stack(out, 1)


def windows(d, episodes, k, rng, count):
    """Random (episode, t) starts with k valid future steps."""
    valid_len = d["mask"].sum(1)-1                               # number of transitions per episode
    eps = rng.choice(episodes, size=count)
    t = (rng.random(count)*np.maximum(valid_len[eps]-k, 1)).astype(int)
    return eps, t


def evaluate(model, d, episodes, device, horizons=(1, 8, 32), samples=4000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for H in horizons:
        eps, t = windows(d, episodes, H, rng, samples)
        idx = t[:, None]+np.arange(H+1)[None]
        S = torch.tensor(d["s"][eps[:, None], idx], device=device)
        A = torch.tensor(d["a"][eps[:, None], idx[:, :-1]], device=device)
        sh = torch.tensor(d["shape"][eps], device=device)
        contact = d["contacts"][eps[:, None], idx[:, 1:]].max(1) > 0
        with torch.no_grad():
            pred = model.rollout(S[:, 0], A, sh)[:, -1]
        true, start = S[:, -1], S[:, 0]
        def errs(p):
            pos = (p[:, :2]-true[:, :2]).norm(dim=1).cpu().numpy()
            ang = np.degrees(np.abs(np.angle(np.exp(1j*(p[:, 2]-true[:, 2]).cpu().numpy()))))
            pus = (p[:, 6:8]-true[:, 6:8]).norm(dim=1).cpu().numpy()
            return pos, ang, pus
        mpos, mang, mpus = errs(pred)
        ppos, pang, _ = errs(start)                               # persistence: object does not move
        moved = (true[:, :2]-start[:, :2]).norm(dim=1).cpu().numpy()
        for name, sel in (("contact", contact), ("no_contact", ~contact)):
            if sel.sum() == 0:
                continue
            out[f"H{H}_{name}"] = {"n": int(sel.sum()), "obj_pos_px": float(np.median(mpos[sel])),
                                   "obj_angle_deg": float(np.median(mang[sel])), "pusher_px": float(np.median(mpus[sel])),
                                   "persist_pos_px": float(np.median(ppos[sel])), "persist_angle_deg": float(np.median(pang[sel])),
                                   "true_motion_px": float(np.median(moved[sel])),
                                   "obj_pos_p90_px": float(np.percentile(mpos[sel], 90))}
    return out


def train(args):
    d = load(args.data)
    n = len(d["seed"])
    seeds = np.unique(d["seed"])
    rng = np.random.default_rng(0)
    val_seeds = set(rng.choice(seeds, size=max(1, len(seeds)//7), replace=False).tolist())
    val = np.array([i for i in range(n) if d["seed"][i] in val_seeds])
    tr = np.array([i for i in range(n) if d["seed"][i] not in val_seeds])
    m = d["mask"][tr]
    s_tr, a_tr = d["s"][tr][m], d["a"][tr][m[:, 1:]]
    delta = (d["s"][tr][:, 1:]-d["s"][tr][:, :-1])[m[:, 1:]]
    stats = {"s_mean": s_tr.mean(0), "s_std": s_tr.std(0)+1e-3, "a_mean": a_tr.mean(0), "a_std": a_tr.std(0)+1e-3,
             "d_std": delta.std(0)+1e-4}
    device = "cuda"
    model = StateWM(stats, args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    s_std = model.s_std
    print(json.dumps({"files": d["files"], "episodes": n, "train_eps": len(tr), "val_eps": len(val),
                      "transitions": int(d["mask"][:, 1:].sum())}), flush=True)
    start, log = time.time(), {"args": vars(args) | {"data": str(args.data), "out": str(args.out)}, "evals": []}
    for step in range(1, args.steps+1):
        eps, t = windows(d, tr, args.k, rng, args.batch)
        idx = t[:, None]+np.arange(args.k+1)[None]
        S = torch.tensor(d["s"][eps[:, None], idx], device=device)
        A = torch.tensor(d["a"][eps[:, None], idx[:, :-1]], device=device)
        sh = torch.tensor(d["shape"][eps], device=device)
        pred = model.rollout(S[:, 0], A, sh)
        err = (pred-S[:, 1:])/s_std
        err[..., 2] = torch.atan2(torch.sin(pred[..., 2]-S[:, 1:, 2]), torch.cos(pred[..., 2]-S[:, 1:, 2]))/s_std[2]
        loss = err.square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, d, val, device)
            log["evals"].append({"step": step, "loss": float(loss), **ev})
            brief = "  ".join(f"{k}: pos {v['obj_pos_px']:.2f}px (persist {v['persist_pos_px']:.2f}, moved {v['true_motion_px']:.2f}) "
                              f"ang {v['obj_angle_deg']:.2f}deg" for k, v in ev.items() if k.startswith(("H8", "H32")))
            print(f"[wm {time.time()-start:5.0f}s] step {step} loss {float(loss):.4f}  {brief}", flush=True)
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "width": args.width, "depth": args.depth}, args.out/"model.pt")
            (args.out/"log.json").write_text(json.dumps(log, indent=1, default=str))
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--k", type=int, default=8)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--width", type=int, default=1024)
    t.add_argument("--depth", type=int, default=3)
    t.add_argument("--eval-every", type=int, default=2000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
