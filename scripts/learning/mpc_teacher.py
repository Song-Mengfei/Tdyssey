"""Sampling-based model-predictive teacher on the structured world model (wm2), evaluated in the real simulator.

At every control step the real simulator's privileged state (object pose/velocity, pusher) is read, a
cross-entropy-method planner samples smooth pusher-target sequences plus structured push candidates (--pushes:
leave the object radially to a safe circle, travel around it, then push one boundary point along a direction
near its inward normal for 1-40 px), rolls them out INSIDE wm2 (hard contact gate), scores them with the
teacher2 reward (privileged geometry, disclosed) and executes the first target of the best plan. No simulator
state is used to score imagined transitions; the simulator only supplies the current state and the outcome.
Batched over all scenes of one shape.

Recommended flags (each fixed a measured stall 2-4 px short of the goal; see docs/LEARNING_FINDINGS.md):
  --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2
  20 validation scenes: D 20/20, Y 16/20, L 7/20, T_narrow 3/20, T about 10% (base: 0.3%, 9.4%, 0.1%, 42%, 44%).

  .venv/bin/python -B mpc_teacher.py --wm $RUNS/wm2/v1/model.pt --shape T --scenes 20 --out $RUNS/mpc/T.json
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

import teacher2
import wm2
from tdyssey.envs.official import make_official_shape_env_class, matched_reset
from tdyssey.official import load_official_source, physical_metrics, physical_state

SHAPES = wm2.SHAPES


def smooth_noise(E, K, H, knots, device, gen):
    """(E, K, H, 2) standard-normal noise, linearly interpolated from `knots` control points."""
    z = torch.randn(E, K, knots, 2, device=device, generator=gen)
    pos = torch.linspace(0, knots-1, H, device=device)
    lo = pos.floor().long().clamp(max=knots-2); w = (pos-lo)[None, None, :, None]
    return z[:, :, lo]*(1-w) + z[:, :, lo+1]*w


class Pushes:
    """Structured push candidates (classic planar-pushing primitive; geometric prior, disclosed): leave the
    object radially to a safe circle, travel around it, come in to a pre-contact point behind a boundary point,
    then push through it along a direction near the inward normal for L px at a slow speed."""
    def __init__(self, geo, device, pool=512, seed=0):
        rng = np.random.default_rng(seed)
        rects = geo.rects.cpu()
        pts, nrm, radius = [], [], []
        for k in range(len(SHAPES)):
            r = np.unique(rects[k].numpy(), axis=0)
            cand, normals = [], []
            for x0, y0, x1, y1 in r:
                u = rng.random(64)
                for q, n in (((x0+u*(x1-x0), np.full(64, y0)), (0, -1)), ((x0+u*(x1-x0), np.full(64, y1)), (0, 1)),
                             ((np.full(64, x0), y0+u*(y1-y0)), (-1, 0)), ((np.full(64, x1), y0+u*(y1-y0)), (1, 0))):
                    cand.append(np.stack(q, -1)); normals.append(np.repeat([n], 64, 0))
            cand, normals = np.concatenate(cand).astype(np.float32), np.concatenate(normals).astype(np.float32)
            rk = torch.tensor(r)[None]
            out = wm2.box_sdf(torch.tensor(cand + 0.5*normals)[None], rk)[0].numpy()
            inn = wm2.box_sdf(torch.tensor(cand - 0.5*normals)[None], rk)[0].numpy()
            keep = (out > 0.2) & (inn < 0)
            idx = rng.choice(np.nonzero(keep)[0], size=pool)
            pts.append(cand[idx]); nrm.append(normals[idx])
            corners = np.concatenate([r[:, [0, 1]], r[:, [2, 3]], r[:, [0, 3]], r[:, [2, 1]]])
            radius.append(np.linalg.norm(corners, axis=1).max())
        self.pts = torch.tensor(np.array(pts), device=device)                 # (5, pool, 2) object (COG) frame
        self.nrm = torch.tensor(np.array(nrm), device=device)
        self.radius = torch.tensor(radius, device=device, dtype=torch.float32)
        self.pool = pool

    def sample(self, s, sh, M, H, gen, transit=20.0, lmin=1.0):
        E, dev = len(s), s.device
        i = torch.randint(0, self.pool, (E, M), device=dev, generator=gen)
        c = self.pts[sh[:, None], i]; n = self.nrm[sh[:, None], i]          # (E, M, 2)
        delta = (torch.rand(E, M, device=dev, generator=gen)*2-1)*np.radians(35)
        L = lmin*torch.exp(torch.rand(E, M, device=dev, generator=gen)*np.log(40.0/lmin))  # lmin..40 px, log-uniform
        vp = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dev)[torch.randint(0, 4, (E, M), device=dev, generator=gen)]
        th = s[:, None, 2]
        def world(v):
            x, y = wm2.rot(v[..., 0], v[..., 1], th)
            return torch.stack([x, y], -1)
        dx, dy = wm2.rot(-n[..., 0], -n[..., 1], delta)                      # push direction (object frame)
        d = world(torch.stack([dx, dy], -1))
        o = s[:, None, :2]
        cw = o + world(c)
        pre = cw - d*(wm2.RADIUS+4)
        end = pre + d*(4+L[..., None])
        p0 = s[:, None, 6:8].expand(-1, M, -1)
        Rs = (self.radius[sh] + wm2.RADIUS + 8)[:, None]
        rel0 = p0-o; r0 = rel0.norm(dim=-1).clamp(min=1e-3)
        w1 = o + rel0/r0[..., None]*Rs[..., None]                          # move radially onto the safe circle (in or out)
        relp = pre-o; rp = relp.norm(dim=-1).clamp(min=1e-3)
        w2 = o + relp/rp[..., None]*Rs[..., None]
        direct = rp >= Rs                                                     # pre-contact point outside the circle: go straight
        a1, a2 = torch.atan2(rel0[..., 1], rel0[..., 0]), torch.atan2(relp[..., 1], relp[..., 0])
        dphi = torch.atan2(torch.sin(a2-a1), torch.cos(a2-a1))
        lA = (w1-p0).norm(dim=-1); lB = Rs*dphi.abs(); lC = (pre-w2).norm(dim=-1)
        lA, lB, lC = [torch.where(direct, torch.zeros_like(x), x) for x in (lA, lB, lC)]
        lS = torch.where(direct, (pre-p0).norm(dim=-1), torch.zeros_like(lA))
        lT = lA+lB+lC+lS; lD = 4+L
        tt = torch.arange(1, H+1, device=dev, dtype=torch.float32)[None, None]
        Ttr = lT/transit
        sarc = torch.where(tt <= Ttr[..., None], tt*transit, lT[..., None] + (tt-Ttr[..., None])*vp[..., None])
        sarc = torch.minimum(sarc, (lT+lD)[..., None])                        # (E, M, H)
        def lerp(a, b, f):
            return a[..., None, :] + (b-a)[..., None, :]*f[..., None].clamp(0, 1)
        fA = sarc/lA.clamp(min=1e-3)[..., None]
        phi = a1[..., None] + torch.sign(dphi)[..., None]*((sarc-lA[..., None])/Rs[..., None]).clamp(min=0)
        phi = torch.where(sarc-lA[..., None] > lB[..., None], a2[..., None], phi)
        arc = o[..., None, :] + torch.stack([torch.cos(phi), torch.sin(phi)], -1)*Rs[..., None, None]
        fC = (sarc-lA[..., None]-lB[..., None])/lC.clamp(min=1e-3)[..., None]
        fS = sarc/lS.clamp(min=1e-3)[..., None]
        fD = (sarc-lT[..., None])/lD[..., None]
        path = torch.where((sarc <= lA[..., None])[..., None], lerp(p0, w1, fA),
               torch.where((sarc <= (lA+lB)[..., None])[..., None], arc, lerp(w2, pre, fC)))
        path = torch.where(direct[..., None, None], lerp(p0, pre, fS), path)
        path = torch.where((sarc > lT[..., None])[..., None], lerp(pre, end, fD), path)
        return path.clamp(0, 512)                                             # (E, M, H, 2)


@torch.no_grad()
def plan(model, geo, s, sh, mean, args, gen, pushes=None):
    """s (E, 10) COG states, mean (E, H, 2) warm-start targets -> refined mean, best return per env."""
    E, H, K = len(s), args.horizon, args.samples
    std = torch.full_like(mean, args.sigma)
    if args.near > 0:                                                        # distance-adaptive noise (fine pushes near the goal)
        corner = torch.sqrt(geo.keypoint_d2(s, sh))
        std = std*(corner/args.near).clamp(args.min_scale, 1.0)[:, None, None]
    disc = args.gamma**torch.arange(H, device=s.device)
    best_a, best_R = None, None
    for it in range(args.iters):
        eps = smooth_noise(E, K, H, args.knots, s.device, gen)
        if args.scales:                                                      # multi-scale samples: fine pushes near the goal
            sc = torch.tensor(args.scales, device=s.device)
            eps = eps*sc[torch.arange(K, device=s.device) % len(sc)][None, :, None, None]
        eps[:, 0] = 0                                                        # keep the current mean as a candidate
        acts = (mean[:, None] + std[:, None]*eps).clamp(0, 512)             # (E, K, H, 2)
        if pushes is not None and it == 0 and args.pushes > 0:               # structured push candidates
            acts[:, -args.pushes:] = pushes.sample(s, sh, args.pushes, H, gen, transit=args.transit, lmin=args.push_min)
        if best_a is not None:
            acts[:, 1] = best_a                                               # keep the best sample found so far
        x = s[:, None].expand(-1, K, -1).reshape(E*K, 10)
        shk = sh[:, None].expand(-1, K).reshape(E*K)
        a = acts.reshape(E*K, H, 2)
        R = torch.zeros(E*K, device=s.device); c = torch.ones(E*K, device=s.device)
        for t in range(H):
            x = model(x, a[:, t], shk, hard=True, margin=args.gate_margin)
            r, succ, _ = geo.reward(x, shk, args.bonus, args.approach, args.dist, args.tight, args.tight_scale)
            R = R + disc[t]*c*r; c = c*(1-succ)
        R = R.view(E, K)
        if it == 0:
            R_warm, a_warm = R[:, 0].clone(), acts[:, 0].clone()             # the previous plan, shifted
        b = R.argmax(1)
        best_R, best_a = R[torch.arange(E), b], acts[torch.arange(E), b]
        top = R.topk(args.elites, dim=1).indices                             # (E, elites)
        el = torch.gather(acts, 1, top[..., None, None].expand(-1, -1, H, 2))
        mean, std = el.mean(1), el.std(1).clamp(min=2.0)
    if args.select == "best":
        if args.commit > 0:                                                   # keep following the previous plan unless clearly beaten
            switch = best_R > R_warm + args.commit*R_warm.abs().clamp(min=1.0)
            best_a = torch.where(switch[:, None, None], best_a, a_warm); best_R = torch.where(switch, best_R, R_warm)
        return best_a, best_R
    return mean, R.max(1).values


def cog_after(envs, idx, sh, geo, device):
    raw = []
    for i in idx:
        st = physical_state(envs[i]); o, p = st["object"], st["pusher"]
        raw.append([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], *p["velocity"]])
    return wm2.to_cog(torch.tensor(raw, device=device, dtype=torch.float32), sh, geo.cogs), sh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=400000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--knots", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=40.0, help="Initial target noise (px)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--near", type=float, default=0.0, help="Shrink the noise when the mean corner distance is below this (px)")
    parser.add_argument("--min-scale", type=float, default=0.1)
    parser.add_argument("--scales", type=float, nargs="*", default=[], help="Per-sample noise scales, e.g. 1 0.3 0.1")
    parser.add_argument("--bonus", type=float, default=10.0)
    parser.add_argument("--approach", type=float, default=0.5)
    parser.add_argument("--dist", type=float, default=1.0)
    parser.add_argument("--tight", type=float, default=0.0, help="Weight of exp(-d2/scale^2), the last 1-2 px of corner error")
    parser.add_argument("--tight-scale", type=float, default=2.0)
    parser.add_argument("--push-min", type=float, default=1.0, help="Shortest push candidate (px)")
    parser.add_argument("--transit", type=float, default=20.0, help="Push candidates: travel speed of the target (px/control)")
    parser.add_argument("--kp", choices=("corners", "area"), default="corners", help="Pose error: rectangle corners or area points")
    parser.add_argument("--execute", type=int, default=1, help="Controls executed open-loop per plan (8 = like the image policy)")
    parser.add_argument("--gate-margin", type=float, default=3.0, help="Planning: object moves only if the pusher disk comes this close (px)")
    parser.add_argument("--pushes", type=int, default=0, help="Structured push candidates per plan (first iteration)")
    parser.add_argument("--select", choices=("mean", "best"), default="mean", help="Execute the elite mean or the best sample")
    parser.add_argument("--commit", type=float, default=0.0, help="Relative return margin needed to abandon the previous plan")
    parser.add_argument("--trace", action="store_true", help="Print the plan near the goal every control")
    parser.add_argument("--log-pred", action="store_true", help="Compare the model's 1-step prediction with reality")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda"; gen = torch.Generator(device=device); gen.manual_seed(0)
    geo = teacher2.Geometry(device); geo.kp_mode = args.kp
    blob = torch.load(args.wm, map_location=device, weights_only=True)
    model = wm2.WM2(geo.rects, blob["width"], blob["depth"]).to(device)
    model.load_state_dict(blob["state_dict"]); model.eval()
    cls = make_official_shape_env_class(load_official_source(ROOT/"third_party/diffusion_policy"))
    pushes = Pushes(geo, device) if args.pushes > 0 else None
    k = SHAPES.index(args.shape)
    seeds = [args.seed_base+20*(args.start+i) for i in range(args.scenes)]
    envs, keep = [], []
    for seed in seeds:
        partner = "L" if args.shape == "T" else args.shape
        pair = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
        matched_reset(pair, seed, defaultdict(int))
        env = pair[args.shape]; env._get_obs = lambda: None
        envs.append(env); keep.append(pair)
    E = len(envs)
    sh_all = torch.full((E,), k, device=device)
    active = np.ones(E, bool); done = np.zeros(E, bool); steps = np.zeros(E, int)
    cov, peak = np.zeros(E), np.zeros(E)
    mean = None; start = time.time(); pred_log = []
    for step in range(args.steps):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        raw = []
        for i in idx:
            st = physical_state(envs[i]); o, p = st["object"], st["pusher"]
            raw.append([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], *p["velocity"]])
        sh = sh_all[idx]
        s = wm2.to_cog(torch.tensor(raw, device=device, dtype=torch.float32), sh, geo.cogs)
        if mean is None:
            mean = s[:, None, 6:8].expand(-1, args.horizon, -1).clone()
        m = mean[torch.as_tensor(np.searchsorted(np.nonzero(prev_active)[0], idx), device=device)] if step else mean
        if step % args.execute == 0:
            m, best = plan(model, geo, s, sh, m, args, gen, pushes)
        acts = m[:, 0].cpu().numpy()
        if args.trace and step % args.execute == 0:
            with torch.no_grad():
                xr = s.clone()
                for u in range(args.horizon):
                    xr = model(xr, m[:, u], sh, hard=True, margin=args.gate_margin)
                cnow = torch.sqrt(geo.keypoint_d2(s, sh)); cend = torch.sqrt(geo.keypoint_d2(xr, sh))
                gap = geo.pusher_gap(s, sh)
            for j in range(len(idx)):
                if float(cnow[j]) < 10:
                    print(f"    trace scene {idx[j]} step {step}: corner now {float(cnow[j]):.2f}px, plan's predicted end {float(cend[j]):.2f}px, "
                          f"pusher gap {float(gap[j]):.1f}px, planned return {float(best[j]):.2f}, first target moves pusher "
                          f"{float((m[j, 0]-s[j, 6:8]).norm()):.1f}px", flush=True)
        if args.log_pred:
            with torch.no_grad():
                pred = model(s, m[:, 0].clamp(0, 512), sh, hard=True)
                corner_before = torch.sqrt(geo.keypoint_d2(s, sh)).cpu().numpy()
                corner_pred = torch.sqrt(geo.keypoint_d2(pred, sh)).cpu().numpy()
        for j, i in enumerate(idx):
            _, _, d, _ = envs[i].step(np.clip(acts[j], 0, 512))
            cov[i] = physical_metrics(envs[i])["coverage"]; peak[i] = max(peak[i], cov[i]); steps[i] += 1
            if d:
                done[i] = True; active[i] = False
        if args.log_pred:
            s_after, _ = cog_after(envs, idx, sh, geo, device)
            corner_after = torch.sqrt(geo.keypoint_d2(s_after, sh)).cpu().numpy()
            err = (pred[:, :2]-s_after[:, :2]).norm(dim=1).cpu().numpy()
            moved_pred = (pred[:, :2]-s[:, :2]).norm(dim=1).cpu().numpy()
            moved_real = (s_after[:, :2]-s[:, :2]).norm(dim=1).cpu().numpy()
            for j in range(len(idx)):
                pred_log.append((corner_before[j], corner_pred[j], corner_after[j], err[j], moved_pred[j], moved_real[j]))
        prev_active = np.zeros(E, bool); prev_active[idx] = True
        mean = torch.cat([m[:, 1:], m[:, -1:]], 1)                          # shift the plan by one control
        if step % 25 == 0 or not active.any():
            print(f"[mpc {args.shape} {time.time()-start:5.0f}s] step {step} success {int(done.sum())}/{E} "
                  f"mean coverage {cov.mean():.3f} peak {peak.mean():.3f} best planned return {float(best.mean()):.2f}", flush=True)
    if args.log_pred:
        L = np.array(pred_log)
        for name, sel in (("corner dist < 8px", L[:, 0] < 8), ("8-20px", (L[:, 0] >= 8) & (L[:, 0] < 20)), (">= 20px", L[:, 0] >= 20)):
            r = L[sel]
            if len(r):
                print(f"  1-step prediction, {name:18s} n={len(r):5d}: predicted corner change {np.median(r[:,1]-r[:,0]):+.3f}px, "
                      f"real {np.median(r[:,2]-r[:,0]):+.3f}px; position error med {np.median(r[:,3]):.3f} p90 {np.percentile(r[:,3],90):.3f}px; "
                      f"moved pred {np.median(r[:,4]):.3f} real {np.median(r[:,5]):.3f}px; frac predicted-better-but-real-worse "
                      f"{np.mean((r[:,1]<r[:,0]) & (r[:,2]>=r[:,0])):.2f}")
    raw = []
    for env in envs:
        st = physical_state(env); o, p = st["object"], st["pusher"]
        raw.append([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], *p["velocity"]])
    s = wm2.to_cog(torch.tensor(raw, device=device, dtype=torch.float32), sh_all, geo.cogs)
    g = geo.goals[k]
    pos_err = (s[:, :2]-g[:2]).norm(dim=1).cpu().numpy()
    ang_err = np.degrees(np.abs(np.angle(np.exp(1j*(s[:, 2]-g[2]).cpu().numpy()))))
    cov_hat = geo.coverage(s, sh_all).cpu().numpy()
    for pair in keep:
        for e in pair.values():
            e.close()
    out = {"shape": args.shape, "seeds": seeds, "success": done.tolist(), "final_coverage": cov.tolist(), "peak_coverage": peak.tolist(),
           "steps": steps.tolist(), "final_pos_err_px": pos_err.tolist(), "final_angle_err_deg": ang_err.tolist(),
           "final_coverage_estimate": cov_hat.tolist(), "args": {k_: str(v) for k_, v in vars(args).items()}, "seconds": time.time()-start}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{args.shape}: success {int(done.sum())}/{E}  final coverage {cov.mean():.3f}  peak {peak.mean():.3f}  "
          f"median pose error {np.median(pos_err):.1f}px {np.median(ang_err):.1f}deg  wrote {args.out}")
    for c_, ch, pe, ae in sorted(zip(cov, cov_hat, pos_err, ang_err)):
        print(f"  coverage {c_:.3f} (model estimate {ch:.3f})  position error {pe:5.1f}px  angle error {ae:5.1f}deg")


if __name__ == "__main__":
    main()
