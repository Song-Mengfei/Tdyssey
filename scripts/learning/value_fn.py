"""Privileged state-value function for the frozen base policy (training-signal component, not an actor).

Target: the trainer's own shaped return from control t to episode end,
  G_t = sum_k gamma^(k-t) r_k,  r = 1[success] + gamma*cov' - cov   (rl_protocol.shaped_reward, beta 1)
which telescopes to G_t = -cov_t + W_t with W_t = gamma^(tau-t) on success at tau, else gamma^(300-t)*cov_300.
The network predicts W (in [0, 1]); V(s) = W(s) - cov(s). Inputs are privileged simulator state
(object pose relative to the goal, velocities, pusher position in the object frame, coverage,
time left) plus shape identity, so this is an asymmetric critic; the actor stays image-based.

  train:    .venv/bin/python -B value_fn.py train --data $RUNS/value-data --out $RUNS/value-fn/v1.pt
Episodes are split into train/validation by scene seed, never by step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SHAPES = ("T", "T_narrow", "Y", "D", "L")
GAMMA = 0.99


def features(state, coverage, t, shape_index):
    """state[..., 9] = object x,y,angle,vx,vy,omega, pusher x,y, goal angle; goal xy is (256, 256)."""
    ox, oy, angle, vx, vy, w, px, py, goal_angle = np.moveaxis(state, -1, 0)
    d = angle-goal_angle
    c, s = np.cos(angle), np.sin(angle)
    rx, ry = px-ox, py-oy
    local_x, local_y = c*rx+s*ry, -s*rx+c*ry                # pusher in the object frame
    shape = np.eye(len(SHAPES), dtype=np.float32)[shape_index]
    base = np.stack([(ox-256)/100, (oy-256)/100, np.sin(d), np.cos(d), vx/100, vy/100, w,
                     local_x/100, local_y/100, (px-256)/100, (py-256)/100, coverage, 1-t/300], -1)
    return np.concatenate([base.astype(np.float32), np.broadcast_to(shape, base.shape[:-1]+(len(SHAPES),))], -1)


class Value(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13+len(SHAPES), width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)       # W in (0, 1)


def targets(coverage, mask, success):
    """W_t for every valid control of every episode."""
    n, T = coverage.shape
    W = np.zeros((n, T), np.float32)
    for i in range(n):
        last = int(mask[i].sum())-1
        end_value = 1.0 if success[i] else float(coverage[i, last])
        for t in range(last+1):
            W[i, t] = GAMMA**(last-t)*end_value
    return W


def load(data_dir):
    parts = [np.load(p) for p in sorted(Path(data_dir).glob("part*.npz"))]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return {k: cat(k) for k in ("state", "coverage", "mask", "success", "shape", "seed")}


def train(args):
    d = load(args.data)
    shape_index = np.array([SHAPES.index(s) for s in d["shape"]])
    W = targets(d["coverage"], d["mask"], d["success"])
    t = np.broadcast_to(np.arange(d["mask"].shape[1]), d["mask"].shape)
    X = features(d["state"], d["coverage"], t, shape_index[:, None].repeat(d["mask"].shape[1], 1))
    seeds = np.unique(d["seed"])
    rng = np.random.default_rng(0)
    val_seeds = set(rng.choice(seeds, size=max(1, len(seeds)//5), replace=False).tolist())
    val_ep = np.array([s in val_seeds for s in d["seed"]])
    pick = lambda eps: (X[eps][d["mask"][eps]], W[eps][d["mask"][eps]])
    Xtr, Ytr = pick(~val_ep)
    Xva, Yva = pick(val_ep)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Value().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t, Ytr_t = torch.tensor(Xtr, device=device), torch.tensor(Ytr, device=device)
    Xva_t, Yva_t = torch.tensor(Xva, device=device), torch.tensor(Yva, device=device)
    best, best_state = float("inf"), None
    g = torch.Generator(device=device).manual_seed(0)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xtr_t), device=device, generator=g)
        for start in range(0, len(order), 4096):
            idx = order[start:start+4096]
            loss = (model(Xtr_t[idx])-Ytr_t[idx]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = float((model(Xva_t)-Yva_t).square().mean())
        if val < best:
            best, best_state = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs-1:
            print(f"epoch {epoch:3d} train {float(loss):.4f} val {val:.4f} (target var {float(Yva_t.var()):.4f})", flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xva_t).cpu().numpy()
    report = {"val_mse": best, "val_target_var": float(Yva.var()), "val_r2": 1-best/float(Yva.var()),
              "train_samples": len(Xtr), "val_samples": len(Xva), "val_scenes": len(val_seeds), "per_shape": {}}
    va_shape = shape_index[val_ep][:, None].repeat(d["mask"].shape[1], 1)[d["mask"][val_ep]]
    for i, s in enumerate(SHAPES):
        m = va_shape == i
        if m.any():
            report["per_shape"][s] = {"r2": float(1-((pred[m]-Yva[m])**2).mean()/max(Yva[m].var(), 1e-9)),
                                      "corr": float(np.corrcoef(pred[m], Yva[m])[0, 1])}
    print(json.dumps(report, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "shapes": SHAPES, "gamma": GAMMA, "report": report}, args.out)
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
