"""SuperMimic-style privileged teacher for PushT: BPTT through a frozen state world model + GAE critic.

Policy  a_t = pi(s_t, shape)   (pusher target; a = pusher + scale*tanh(net)); deterministic, privileged state.
Reward  r_{t+1} = exp(-(d(s_{t+1})/sigma)^2), d = mean distance between the object's polygon vertices at
        its pose and at the goal pose (a smooth stand-in for goal coverage; disclosed privileged signal).
Steps   (0) behaviour-clone the frozen base policy's logged actions (noise-free data) so training starts
        at the base behaviour; (1) per iteration, roll B imagined trajectories of H steps from logged
        states through the frozen world model, maximise sum gamma^t r + gamma^H V(s_H) by backprop
        (SuperMimic eq. policy-obj), then fit V to GAE targets of the imagined rollout (eq. critic);
        (2) every few iterations, run the teacher closed-loop in the REAL simulator on validation scenes.
Usage: .venv/bin/python -B teacher_bptt.py --wm $RUNS/wm-state/v1/model.pt --data $RUNS/dyn-data --out DIR
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
from torch import nn

import wm_state
from tdyssey.envs.official import make_official_shape_env_class, matched_reset
from tdyssey.official import load_official_source, physical_metrics, physical_state

SHAPES = wm_state.SHAPES


def shape_geometry(cls):
    """Body-frame polygon vertices and goal pose per shape, read from the real environments."""
    verts, goals = [], []
    for name in SHAPES:
        env = cls(shape=name, legacy=True)
        env.seed(0); env.reset()
        v = np.concatenate([np.array(p.get_vertices()) for p in env.block.shapes]).astype(np.float32)
        verts.append(v); goals.append(np.array(env.goal_pose, np.float32))
        env.close()
    k = max(len(v) for v in verts)
    padded = np.stack([np.concatenate([v, np.repeat(v[-1:], k-len(v), 0)]) for v in verts])   # repeat last vertex
    return torch.tensor(padded), torch.tensor(np.stack(goals))


def keypoint_distance(s, shape, verts, goals):
    """Mean vertex distance (px) between the object at pose s[...,0:3] and at its shape's goal pose."""
    v, g = verts[shape], goals[shape]                                    # (B, K, 2), (B, 3)
    def place(x, y, th):
        c, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]
        return torch.stack([c*v[..., 0]-s_*v[..., 1]+x[:, None], s_*v[..., 0]+c*v[..., 1]+y[:, None]], -1)
    return (place(s[:, 0], s[:, 1], s[:, 2])-place(g[:, 0], g[:, 1], g[:, 2])).norm(dim=-1).mean(-1)


class Policy(nn.Module):
    def __init__(self, wm, scale):
        super().__init__()
        self.wm, self.scale = wm, scale
        self.net = nn.Sequential(nn.Linear(11+len(SHAPES)+2, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, s, shape, goal_feat):
        x = self.wm.features(s, torch.zeros_like(s[..., :2])+self.wm.a_mean, shape)[..., :11]
        h = torch.cat([x, nn.functional.one_hot(shape, len(SHAPES)).float(), goal_feat], -1)
        return s[..., 6:8] + self.scale*torch.tanh(self.net(h))          # target near the pusher


class Critic(nn.Module):
    def __init__(self, wm):
        super().__init__()
        self.wm = wm
        self.net = nn.Sequential(nn.Linear(11+len(SHAPES)+1, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, s, shape, dist):
        x = self.wm.features(s, torch.zeros_like(s[..., :2])+self.wm.a_mean, shape)[..., :11]
        return self.net(torch.cat([x, nn.functional.one_hot(shape, len(SHAPES)).float(), dist[:, None]/100], -1)).squeeze(-1)


def goal_features(s, shape, goals):
    g = goals[shape]
    return torch.stack([(s[:, 0]-g[:, 0])/100, (s[:, 1]-g[:, 1])/100], -1)


@torch.no_grad()
def simulate(policy, cls, shapes, seeds, verts, goals, device, steps=300):
    """Closed-loop privileged teacher in the real simulator (rendering disabled)."""
    out = {}
    for shape_name in shapes:
        rows = []
        for seed in seeds:
            partner = "L" if shape_name == "T" else shape_name
            envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
            matched_reset(envs, seed, defaultdict(int))
            env = envs[shape_name]
            env._get_obs = lambda: None
            peak, done = 0.0, False
            for _ in range(steps):
                st = physical_state(env)
                o, p = st["object"], st["pusher"]
                s = torch.tensor([[*o["position"], o["angle"], *o["velocity"], o["angular_velocity"],
                                   *p["position"], *p["velocity"]]], device=device)
                sh = torch.tensor([SHAPES.index(shape_name)], device=device)
                a = policy(s, sh, goal_features(s, sh, goals)).cpu().numpy()[0]
                _, _, done, _ = env.step(np.clip(a, 0, 512))
                cov = physical_metrics(env)["coverage"]
                peak = max(peak, cov)
                if done:
                    break
            rows.append((done, cov, peak))
            for e in envs.values():
                e.close()
        out[shape_name] = {"success": sum(r[0] for r in rows), "n": len(rows),
                           "final_coverage": float(np.mean([r[1] for r in rows])), "peak_coverage": float(np.mean([r[2] for r in rows]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--sigma", type=float, default=30.0, help="Reward width (px) of the keypoint distance")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-scenes", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    torch.manual_seed(0)
    blob = torch.load(args.wm, map_location=device, weights_only=True)
    d = wm_state.load(args.data)
    stats = {k: blob["state_dict"][k].cpu().numpy() for k in ("s_mean", "s_std", "a_mean", "a_std", "d_std")}
    wm = wm_state.StateWM(stats, blob["width"], blob["depth"]).to(device)
    wm.load_state_dict(blob["state_dict"]); wm.eval().requires_grad_(False)
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
    verts, goals = (x.to(device) for x in shape_geometry(cls))
    m = d["mask"][:, :-1] & d["mask"][:, 1:]
    S = torch.tensor(d["s"][:, :-1][m], device=device)
    A = torch.tensor(d["a"][m], device=device)
    SH = torch.tensor(np.repeat(d["shape"][:, None], m.shape[1], 1)[m], device=device)
    CLEAN = torch.tensor(np.repeat((d["noise"] == 0)[:, None], m.shape[1], 1)[m], device=device)   # BC: base actions only
    offset = (A-S[:, 6:8]).abs()
    scale = float(torch.quantile(offset.flatten(), 0.99))
    policy, critic = Policy(wm, scale).to(device), Critic(wm).to(device)
    print(json.dumps({"transitions": len(S), "action_offset_p99_px": scale, "shapes": SHAPES}), flush=True)
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "bc": [], "iterations": [], "evals": []}
    seeds = [400000+20*i for i in range(args.eval_scenes)]
    start = time.time()

    # (0) behaviour cloning of the base policy's executed actions (noise-free parts only)
    opt_bc = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    clean_idx = torch.nonzero(CLEAN).squeeze(1)
    for step in range(1, args.bc_steps+1):
        i = clean_idx[torch.randint(0, len(clean_idx), (args.batch,), device=device)]
        loss = (policy(S[i], SH[i], goal_features(S[i], SH[i], goals))-A[i]).square().mean()
        opt_bc.zero_grad(); loss.backward(); opt_bc.step()
        if step % 1000 == 0:
            log["bc"].append({"step": step, "mse_px2": float(loss)})
            print(f"[teacher {time.time()-start:5.0f}s] bc step {step} action mse {float(loss):.1f} px^2", flush=True)
    ev = simulate(policy, cls, SHAPES, seeds, verts, goals, device)
    log["evals"].append({"iteration": 0, **ev})
    print(f"[teacher {time.time()-start:5.0f}s] after BC, real sim: " +
          "  ".join(f"{k} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for k, v in ev.items()), flush=True)
    torch.save({"policy": policy.state_dict(), "scale": scale, "iteration": 0}, args.out/"teacher-it0.pt")

    # (1) SuperMimic-style BPTT through the frozen world model with a GAE critic
    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    H, g = args.horizon, args.gamma
    for it in range(1, args.iterations+1):
        i = torch.randint(0, len(S), (args.batch,), device=device)
        s, sh = S[i], SH[i]
        states, rewards = [s], []
        for t in range(H):
            a = policy(s, sh, goal_features(s, sh, goals))
            s = wm(s, a, sh)
            states.append(s)
            rewards.append(torch.exp(-(keypoint_distance(s, sh, verts, goals)/args.sigma)**2))
        R = torch.stack(rewards, 1)                                          # (B, H)
        vH = critic(states[-1], sh, keypoint_distance(states[-1], sh, verts, goals))
        J = (R*(g**torch.arange(H, device=device))).sum(1) + g**H*vH
        loss_pi = -J.mean()/H
        opt_pi.zero_grad(); loss_pi.backward()
        gn = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_pi.step()
        with torch.no_grad():                                                 # GAE targets on imagined transitions only
            Sd = torch.stack([x.detach() for x in states], 1)                  # (B, H+1, 10)
            V = torch.stack([critic(Sd[:, t], sh, keypoint_distance(Sd[:, t], sh, verts, goals)) for t in range(H+1)], 1)
            targ = torch.zeros_like(R)
            nxt = V[:, H]
            for t in reversed(range(H)):
                nxt = R[:, t] + g*((1-args.lam)*V[:, t+1] + args.lam*nxt)
                targ[:, t] = nxt
        for _ in range(5):
            perm = torch.randperm(len(i), device=device)
            for chunk in perm.chunk(4):
                t_idx = torch.randint(0, H, (len(chunk),), device=device)
                st = Sd[chunk, t_idx]
                lv = (critic(st, sh[chunk], keypoint_distance(st, sh[chunk], verts, goals))-targ[chunk, t_idx]).square().mean()
                opt_v.zero_grad(); lv.backward(); opt_v.step()
        if it % 50 == 0:
            entry = {"iteration": it, "imagined_return": float(J.mean()), "reward_last": float(R[:, -1].mean()),
                     "critic_loss": float(lv), "grad_norm": float(gn)}
            log["iterations"].append(entry)
            print(f"[teacher {time.time()-start:5.0f}s] it {it} J {entry['imagined_return']:.3f} r_H {entry['reward_last']:.3f} "
                  f"Vloss {entry['critic_loss']:.4f} |g| {entry['grad_norm']:.3f}", flush=True)
        if it % args.eval_every == 0 or it == args.iterations:
            ev = simulate(policy, cls, SHAPES, seeds, verts, goals, device)
            log["evals"].append({"iteration": it, **ev})
            print(f"[teacher {time.time()-start:5.0f}s] it {it} real sim: " +
                  "  ".join(f"{k} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for k, v in ev.items()), flush=True)
            torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(), "scale": scale, "iteration": it},
                       args.out/f"teacher-it{it}.pt")
        (args.out/"log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
