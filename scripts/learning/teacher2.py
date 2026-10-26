"""SuperMimic-style privileged teacher on the structured world model (wm2): BPTT + GAE critic.

Result (2026-10-26, negative): trained from scratch it learns to approach the object but not to push it to
the goal; real-simulator coverage stayed ~0.04-0.12 after 500+ iterations for every reward variant tried
(including --dist). Gradients through contact appear to trap it in local optima. The sampling planner in
mpc_teacher.py replaced it; its reward/geometry helpers (Geometry) are reused there.

Policy  a = pusher + scale*tanh(net(features))  deterministic; features: object COG pose relative to the
        goal, pusher position/velocity in the object frame, object velocity, pusher position, shape.
Reward  r = 0.5*coverage + 0.5*phi_kp + B*success_soft  (privileged geometry, disclosed):
        coverage  = fraction of fixed points sampled inside the goal letter that lie inside the object
                    (soft membership sigmoid(-sdf/0.5px)) -> differentiable exact-coverage estimate
        phi_kp    = 0.5*[exp(-d2/30^2) + exp(-d2/4^2)], d2 = mean squared rectangle-corner distance to goal
        success   = sigmoid((coverage-0.95)/0.01): bonus B and survival weight c_{t+1} = c_t*(1-success)
Objective sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H) through the frozen wm2 (SuperMimic eq. 2);
critic on GAE targets of the imagined rollout (eq. 3). Gradients clipped (global 1.0, then element 0.1).
BC warm start on the base policy's noise-free logged actions; periodic closed-loop real-simulator eval.
Usage: .venv/bin/python -B teacher2.py --wm $RUNS/wm2/v1/model.pt --data $RUNS/dyn-data --out DIR
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

import wm2
import wm_state
from tdyssey.envs.official import make_official_shape_env_class, matched_reset
from tdyssey.official import load_official_source, physical_metrics, physical_state

SHAPES = wm2.SHAPES


class Geometry:
    def __init__(self, device, points=128):
        rects, cogs, goals = wm2.geometry()
        self.rects, self.cogs, self.goals = rects.to(device), cogs.to(device), goals.to(device)
        rng = np.random.default_rng(0)
        pts, corners = [], []
        for k in range(len(SHAPES)):
            r = rects[k].numpy()
            r = np.unique(r, axis=0)
            area = (r[:, 2]-r[:, 0])*(r[:, 3]-r[:, 1])
            which = rng.choice(len(r), size=points, p=area/area.sum())
            u = rng.random((points, 2))
            local = np.stack([r[which, 0]+u[:, 0]*(r[which, 2]-r[which, 0]), r[which, 1]+u[:, 1]*(r[which, 3]-r[which, 1])], -1)
            g = goals[k].numpy()
            c, s = np.cos(g[2]), np.sin(g[2])
            pts.append(np.stack([g[0]+c*local[:, 0]-s*local[:, 1], g[1]+s*local[:, 0]+c*local[:, 1]], -1))
            cr = np.concatenate([r[:, [0, 1]], r[:, [2, 1]], r[:, [2, 3]], r[:, [0, 3]]])
            corners.append(cr)
        K = max(len(c) for c in corners)
        self.goal_points = torch.tensor(np.array(pts, np.float32), device=device)                   # (5, P, 2) world
        self.corners = torch.tensor(np.array([np.concatenate([c, np.repeat(c[-1:], K-len(c), 0)]) for c in corners], np.float32), device=device)

    def place(self, local, x, y, th):
        lx, ly = local[..., 0], local[..., 1]
        c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
        return torch.stack([c*lx-s*ly+x[:, None], s*lx+c*ly+y[:, None]], -1)

    def coverage(self, s, shape, tau=0.5):
        p = self.goal_points[shape]                                                    # (B, P, 2)
        rx, ry = wm2.rot(p[..., 0]-s[:, 0:1], p[..., 1]-s[:, 1:2], -s[:, 2:3])
        sdf = wm2.box_sdf(torch.stack([rx, ry], -1), self.rects[shape])
        return torch.sigmoid(-sdf/tau).mean(1)

    def area_d2(self, s, shape):
        """Mean squared displacement over points spread across the letter's area (tracks coverage near the goal)."""
        g = self.goals[shape]
        gp = self.goal_points[shape]                                               # (B, P, 2) world, inside the goal letter
        lx, ly = wm2.rot(gp[..., 0]-g[:, 0:1], gp[..., 1]-g[:, 1:2], -g[:, 2:3])    # same material points, object frame
        w = self.place(torch.stack([lx, ly], -1), s[:, 0], s[:, 1], s[:, 2])
        return (w-gp).square().sum(-1).mean(-1)

    def keypoint_d2(self, s, shape):
        if getattr(self, "kp_mode", "corners") == "area":
            return self.area_d2(s, shape)
        c = self.corners[shape]
        g = self.goals[shape]
        return (self.place(c, s[:, 0], s[:, 1], s[:, 2])-self.place(c, g[:, 0], g[:, 1], g[:, 2])).square().sum(-1).mean(-1)

    def pusher_gap(self, s, shape):
        """Distance (px) from the pusher disk to the object surface; 0 when touching (disclosed geometric prior)."""
        rx, ry = wm2.rot(s[:, 6:7]-s[:, 0:1], s[:, 7:8]-s[:, 1:2], -s[:, 2:3])
        return (wm2.box_sdf(torch.stack([rx, ry], -1), self.rects[shape])[:, 0] - wm2.RADIUS).clamp(min=0)

    def reward(self, s, shape, bonus, approach=0.0, dist=0.0, tight=0.0, tight_scale=2.0):
        cov = self.coverage(s, shape)
        d2 = self.keypoint_d2(s, shape)
        phi = 0.5*(torch.exp(-d2/30**2) + torch.exp(-d2/4**2))
        succ = torch.sigmoid((cov-0.95)/0.01)
        # dist: dense progress term, mean corner distance to the goal pose (px/100); constant-size gradient far
        # from the goal, where coverage and the narrow keypoint kernels are flat (privileged geometry, disclosed)
        return (0.5*cov + 0.5*phi + bonus*succ - approach*self.pusher_gap(s, shape)/100
                - dist*torch.sqrt(d2+1.0)/100 + tight*torch.exp(-d2/tight_scale**2), succ, cov)   # tight: final 1-2 px


def features(s, shape, geo):
    g = geo.goals[shape]
    th = s[:, 2]
    px, py = wm2.rot(s[:, 6]-s[:, 0], s[:, 7]-s[:, 1], -th)
    pvx, pvy = wm2.rot(s[:, 8], s[:, 9], -th)
    vx, vy = wm2.rot(s[:, 3], s[:, 4], -th)
    dth = th-g[:, 2]
    return torch.cat([torch.stack([(s[:, 0]-g[:, 0])/100, (s[:, 1]-g[:, 1])/100, torch.sin(dth), torch.cos(dth),
                                   torch.cos(th), torch.sin(th), px/100, py/100, pvx/200, pvy/200, vx/20, vy/20, s[:, 5]/0.5,
                                   s[:, 6]/512, s[:, 7]/512], -1), nn.functional.one_hot(shape, len(SHAPES)).float()], -1)


class Policy(nn.Module):
    def __init__(self, geo, scale):
        super().__init__()
        self.geo, self.scale = geo, scale
        self.net = nn.Sequential(nn.Linear(15+len(SHAPES), 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                 nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, s, shape):
        return s[:, 6:8] + self.scale*torch.tanh(self.net(features(s, shape, self.geo)))


class Critic(nn.Module):
    def __init__(self, geo):
        super().__init__()
        self.geo = geo
        self.net = nn.Sequential(nn.Linear(15+len(SHAPES), 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, s, shape):
        return self.net(features(s, shape, self.geo)).squeeze(-1)


@torch.no_grad()
def simulate(policy, cls, geo, seeds, device, steps=300):
    out = {}
    for name in SHAPES:
        k = SHAPES.index(name)
        rows = []
        for seed in seeds:
            partner = "L" if name == "T" else name
            envs = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
            matched_reset(envs, seed, defaultdict(int))
            env = envs[name]; env._get_obs = lambda: None
            peak, done, cov = 0.0, False, 0.0
            for _ in range(steps):
                st = physical_state(env); o, p = st["object"], st["pusher"]
                raw = torch.tensor([[*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], *p["velocity"]]], device=device)
                sh = torch.tensor([k], device=device)
                s = wm2.to_cog(raw, sh, geo.cogs)
                a = policy(s, sh).cpu().numpy()[0]
                _, _, done, _ = env.step(np.clip(a, 0, 512))
                cov = physical_metrics(env)["coverage"]; peak = max(peak, cov)
                if done:
                    break
            rows.append((done, cov, peak))
            for e in envs.values():
                e.close()
        out[name] = {"success": int(sum(r[0] for r in rows)), "n": len(rows), "final_coverage": float(np.mean([r[1] for r in rows])),
                     "peak_coverage": float(np.mean([r[2] for r in rows]))}
    return out


def clip_grads(params):
    params = list(params)                       # a generator would be exhausted by the first call
    nn.utils.clip_grad_norm_(params, 1.0)
    nn.utils.clip_grad_value_(params, 0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--bonus", type=float, default=10.0)
    parser.add_argument("--approach", type=float, default=0.5, help="Weight of the pusher-to-object gap penalty")
    parser.add_argument("--dist", type=float, default=0.0, help="Weight of the dense goal-distance term")
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-scenes", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    device = "cuda"; torch.manual_seed(0)
    geo = Geometry(device)
    blob = torch.load(args.wm, map_location=device, weights_only=True)
    model = wm2.WM2(geo.rects, blob["width"], blob["depth"]).to(device)
    model.load_state_dict(blob["state_dict"]); model.eval().requires_grad_(False)
    d = wm_state.load(args.data)
    S, A, SH = wm2.prepare(d, geo.cogs.cpu(), device)
    m = torch.tensor(d["mask"][:, :-1] & d["mask"][:, 1:], device=device)
    S0, A0 = S[:, :-1][m], A[m]
    SH0 = SH[:, None].expand(-1, m.shape[1])[m]
    clean = torch.tensor(np.repeat((d["noise"] == 0)[:, None], m.shape[1], 1), device=device)[m]
    scale = float(torch.quantile((A0-S0[:, 6:8]).abs().flatten().float()[:1_000_000], 0.99))
    policy, critic = Policy(geo, scale).to(device), Critic(geo).to(device)
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
    seeds = [400000+20*i for i in range(args.eval_scenes)]
    log, start = {"args": {k: str(v) for k, v in vars(args).items()}, "bc": [], "iterations": [], "evals": []}, time.time()
    print(json.dumps({"start_states": int(len(S0)), "bc_states": int(clean.sum()), "action_scale_px": scale}), flush=True)
    with torch.no_grad():   # sanity: the differentiable coverage estimate should match the simulator's
        cov_hat = geo.coverage(S0[:20000], SH0[:20000])
    logged = torch.tensor(d["coverage"][:, :-1], device=device)[m][:20000]
    log["coverage_check"] = {"corr": float(torch.corrcoef(torch.stack([cov_hat, logged]))[0, 1]),
                             "mean_abs_err": float((cov_hat-logged).abs().mean())}
    print(json.dumps({"coverage_check": log["coverage_check"]}), flush=True)
    idx_clean = torch.nonzero(clean).squeeze(1)
    opt_bc = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for step in range(1, args.bc_steps+1):
        i = idx_clean[torch.randint(0, len(idx_clean), (args.batch,), device=device)]
        loss = (policy(S0[i], SH0[i])-A0[i]).square().mean()
        opt_bc.zero_grad(); loss.backward(); opt_bc.step()
        if step % 1000 == 0:
            log["bc"].append({"step": step, "mse_px2": float(loss)})
            print(f"[teacher2 {time.time()-start:5.0f}s] bc {step} action mse {float(loss):.1f} px^2", flush=True)
    ev = simulate(policy, cls, geo, seeds, device)
    log["evals"].append({"iteration": 0, **ev})
    print(f"[teacher2 {time.time()-start:5.0f}s] it 0 (BC) real sim: " + "  ".join(f"{k} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for k, v in ev.items()), flush=True)
    torch.save({"policy": policy.state_dict(), "scale": scale, "iteration": 0}, args.out/"teacher-it0.pt")
    opt_pi = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    H, g = args.horizon, args.gamma
    disc = g**torch.arange(H, device=device)
    for it in range(1, args.iterations+1):
        i = torch.randint(0, len(S0), (args.batch,), device=device)
        s, sh = S0[i], SH0[i]
        states, rewards, conts = [s], [], []
        c = torch.ones(len(i), device=device)
        for t in range(H):
            s = model(s, policy(s, sh), sh)
            r, succ, _ = geo.reward(s, sh, args.bonus, args.approach, args.dist)
            rewards.append(r); conts.append(c); c = c*(1-succ)
            states.append(s)
        R, C = torch.stack(rewards, 1), torch.stack(conts, 1)
        boot = 0.0 if args.no_critic else g**H*c*critic(states[-1], sh)
        J = (disc*C*R).sum(1) + boot
        loss_pi = -J.mean()/H
        opt_pi.zero_grad(); loss_pi.backward(); clip_grads(policy.parameters()); opt_pi.step()
        if not args.no_critic:
            with torch.no_grad():
                Sd = torch.stack([x.detach() for x in states], 1)
                V = torch.stack([critic(Sd[:, t], sh) for t in range(H+1)], 1)
                Cd = torch.cat([C, c[:, None]], 1).detach()
                targ = torch.zeros_like(R); nxt = V[:, H]
                for t in reversed(range(H)):
                    alive = Cd[:, t+1]/Cd[:, t].clamp(min=1e-6)
                    nxt = R[:, t].detach() + g*alive*((1-args.lam)*V[:, t+1] + args.lam*nxt)
                    targ[:, t] = nxt
            for _ in range(5):
                for chunk in torch.randperm(len(i), device=device).chunk(4):
                    t_idx = torch.randint(0, H, (len(chunk),), device=device)
                    lv = (critic(Sd[chunk, t_idx], sh[chunk])-targ[chunk, t_idx]).square().mean()
                    opt_v.zero_grad(); lv.backward(); opt_v.step()
        if it % 50 == 0:
            e = {"iteration": it, "J": float(J.mean()), "r_last": float(R[:, -1].mean()), "cont_last": float(C[:, -1].mean())}
            log["iterations"].append(e)
            print(f"[teacher2 {time.time()-start:5.0f}s] it {it} J {e['J']:.3f} r_H {e['r_last']:.3f} alive_H {e['cont_last']:.3f}", flush=True)
        if it % args.eval_every == 0 or it == args.iterations:
            ev = simulate(policy, cls, geo, seeds, device)
            log["evals"].append({"iteration": it, **ev})
            print(f"[teacher2 {time.time()-start:5.0f}s] it {it} real sim: " + "  ".join(f"{k} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for k, v in ev.items()), flush=True)
            torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(), "scale": scale, "iteration": it}, args.out/f"teacher-it{it}.pt")
        (args.out/"log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
