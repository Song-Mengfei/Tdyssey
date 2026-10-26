"""Training records for distilling the world-model planning teacher (mpc_teacher.py) into the image policy.

All scenes of one shape run in lock-step with rendering. The planning teacher replans EVERY control from the
simulator's privileged state inside wm2 (no simulator look-ahead) and executes its first target; its plans only
work closed-loop (open-loop 8-control execution loses most of the gain), so the label at control t is the
teacher's ACTUALLY EXECUTED targets [a_{t-1}, a_t, ..., a_{t+14}] (padded with the last one), normalized like
the policy's 16-step plans. Every EVERY controls the 2-frame observation is saved together with 8 plans sampled
from the image policy (frozen base, or --student) at that observation, which distill.py uses for its anchor.
Records use the oracle_bon.py format: candidate_x = 8 policy samples + teacher label (index 8, choice = 8).
Pass the same planner flags as mpc_teacher.py (its recommended set); the defaults are the older CEM-only planner.
--execute 8 --every 8 makes the teacher plan once per 8 controls and labels each record with the plan it then
executes, so labels are consistent 8-control chunks like the image policy's (round 2; round 1 used --every 4).
With --beta < 1 the policy's own plan is executed for 8 controls at a decision with probability 1-beta, and the
teacher label at those states is still the teacher's closed-loop continuation, computed in a branch environment
restored from the captured state (DAgger). Scenes: seeds 800000 + 20*i (training only).
  .venv/bin/python -B mpc_dagger.py --wm $RUNS/wm2/v1/model.pt --shape Y --scenes 20 --out $RUNS/mpc-data/r1
"""

import argparse
from collections import defaultdict, deque
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

import base_rollouts
import mpc_teacher
import teacher2
import wm2
from tdyssey.collection import derived_seed
from tdyssey.envs.official import matched_reset
from tdyssey.official import physical_state
from tdyssey.policy_branches import step_record

SHAPES = wm2.SHAPES


def cog_states(envs, shape_k, geo, device):
    raw = []
    for env in envs:
        st = physical_state(env); o, p = st["object"], st["pusher"]
        raw.append([*o["position"], o["angle"], *o["velocity"], o["angular_velocity"], *p["position"], *p["velocity"]])
    sh = torch.full((len(envs),), shape_k, device=device)
    return wm2.to_cog(torch.tensor(raw, device=device, dtype=torch.float32), sh, geo.cogs), sh


class Teacher:
    """Closed-loop planner over a batch of environments with per-environment warm starts."""
    def __init__(self, model, geo, shape_k, args, gen):
        self.model, self.geo, self.k, self.args, self.gen = model, geo, shape_k, args, gen
        self.pushes = mpc_teacher.Pushes(geo, geo.goals.device) if args.pushes > 0 else None

    def act(self, envs, warm):
        s, sh = cog_states(envs, self.k, self.geo, self.geo.goals.device)
        if warm is None:
            warm = s[:, None, 6:8].expand(-1, self.args.horizon, -1).clone()
        m, _ = mpc_teacher.plan(self.model, self.geo, s, sh, warm, self.args, self.gen, self.pushes)
        return m[:, 0].cpu().numpy(), torch.cat([m[:, 1:], m[:, -1:]], 1)

    def plan_full(self, envs, warm_all, idx, fresh):
        """Full plans (n, H, 2) for chunked execution; fresh rows start from the current pusher position."""
        s, sh = cog_states(envs, self.k, self.geo, self.geo.goals.device)
        warm = warm_all[torch.as_tensor(idx, device=s.device)].clone()
        f = torch.as_tensor(fresh[idx], device=s.device)
        warm[f] = s[f, None, 6:8].expand(-1, self.args.horizon, -1)
        m, _ = mpc_teacher.plan(self.model, self.geo, s, sh, warm, self.args, self.gen, self.pushes)
        return m


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--every", type=int, default=4, help="Save a record every N controls")
    parser.add_argument("--beta", type=float, default=1.0, help="Probability that the teacher acts at a decision")
    parser.add_argument("--execute", type=int, default=1,
                        help="Teacher replans every N controls; N > 1 labels each record with the executed plan itself "
                             "(chunk-consistent labels; use --every N, beta 1)")
    parser.add_argument("--student", type=Path, help="Distilled actor checkpoint for the policy samples / DAgger")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--knots", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=40.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--near", type=float, default=0.0, help="Shrink the noise when the mean corner distance is below this (px)")
    parser.add_argument("--min-scale", type=float, default=0.1)
    parser.add_argument("--gate-margin", type=float, default=3.0)
    parser.add_argument("--scales", type=float, nargs="*", default=[])
    parser.add_argument("--bonus", type=float, default=0.0)
    parser.add_argument("--approach", type=float, default=0.5)
    parser.add_argument("--dist", type=float, default=1.0)
    parser.add_argument("--tight", type=float, default=0.0)
    parser.add_argument("--tight-scale", type=float, default=2.0)
    parser.add_argument("--push-min", type=float, default=1.0)
    parser.add_argument("--transit", type=float, default=20.0)
    parser.add_argument("--kp", choices=("corners", "area"), default="corners")
    parser.add_argument("--pushes", type=int, default=0)
    parser.add_argument("--select", choices=("mean", "best"), default="mean")
    parser.add_argument("--commit", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    agent, cls = base_rollouts.build(args.student)
    device = agent.device
    geo = teacher2.Geometry(device); geo.kp_mode = args.kp
    blob = torch.load(args.wm, map_location=device, weights_only=True)
    model = wm2.WM2(geo.rects, blob["width"], blob["depth"]).to(device)
    model.load_state_dict(blob["state_dict"]); model.eval()
    norm = agent.network.normalizer["action"]
    k = SHAPES.index(args.shape)
    teacher = Teacher(model, geo, k, args, torch.Generator(device=device).manual_seed(derived_seed(42, "mpc-plan", args.shape, args.start)))
    pgen = torch.Generator(device=device).manual_seed(derived_seed(42, "mpc-policy", args.shape, args.start))
    rng = np.random.default_rng(derived_seed(42, "mpc-beta", args.shape, args.start))
    seeds = [800000+20*(args.start+i) for i in range(args.scenes)]
    envs, hists = [], []
    for seed in seeds:
        partner = "L" if args.shape == "T" else args.shape
        pair = {"T": cls(shape="T", legacy=True), partner: cls(shape=partner, legacy=True)}
        matched_reset(pair, seed, defaultdict(int))
        env = pair[args.shape]
        for other in pair.values():
            if other is not env:
                other.close()
        obs = env._get_obs()
        envs.append(env); hists.append(deque([obs, obs], maxlen=2))
    E = len(envs)
    branch = cls(shape=args.shape, legacy=True) if args.beta < 1 else None
    actions = [[] for _ in range(E)]                  # executed targets per env
    saved = [[] for _ in range(E)]                    # (control, image, agent_pos, policy x (8,16,2), label or None)
    active, done = np.ones(E, bool), np.zeros(E, bool)
    cov, peak = np.zeros(E), np.zeros(E)
    student_left = np.zeros(E, int); student_plan = [None]*E
    warm = None; start = time.time(); t = 0
    if args.execute > 1 and (args.beta < 1 or args.every != args.execute):
        raise SystemExit("--execute N needs --every N and --beta 1")
    warm_all = torch.zeros(E, args.horizon, 2, device=device); fresh = np.ones(E, bool); chunk = [None]*E
    while active.any() and t < 300:
        idx = np.nonzero(active)[0]
        if t % args.every == 0 or (args.beta < 1 and t % 8 == 0):
            with torch.no_grad():
                cond = agent.encode_batch([list(hists[i]) for i in idx])
                gens = [torch.Generator(device=device).manual_seed(int(x)) for x in
                        torch.randint(0, 2**62, (len(idx)*8,), device=device, generator=pgen).tolist()]
                plans, xs = agent.sample_batch(cond.repeat_interleave(8, 0), gens)
            xs = xs.view(len(idx), 8, 16, 2).cpu().numpy().astype(np.float16)
            for j, i in enumerate(idx):
                if t % args.every == 0:
                    saved[i].append([t, np.stack([np.round(np.asarray(o["image"])*255) for o in hists[i]]).astype(np.uint8),
                                     np.stack([np.asarray(o["agent_pos"]) for o in hists[i]]).astype(np.float32), xs[j], None])
                if args.beta < 1 and t % 8 == 0 and student_left[i] == 0 and rng.random() >= args.beta:
                    student_left[i], student_plan[i] = 8, plans[8*j]
        if args.beta < 1:                              # teacher labels for states the policy is about to drive
            for i in idx:
                if student_left[i] == 8 and saved[i] and saved[i][-1][0] == t:
                    saved[i][-1][4] = branch_label(envs[i], branch, teacher, actions[i])
        if args.execute > 1:
            if t % args.execute == 0:
                m = teacher.plan_full([envs[i] for i in idx], warm_all, idx, fresh)
                ii = torch.as_tensor(idx, device=device)
                warm_all[ii] = torch.cat([m[:, args.execute:], m[:, -1:].expand(-1, args.execute, -1)], 1)
                fresh[idx] = False
                for j, i in enumerate(idx):
                    chunk[i] = m[j].cpu().numpy()
                    if saved[i] and saved[i][-1][0] == t:          # label = the plan the teacher now executes
                        prev = actions[i][-1] if actions[i] else chunk[i][0]
                        saved[i][-1][4] = np.concatenate([np.asarray(prev, np.float32)[None], chunk[i][:15]]).astype(np.float32)
            a_teacher = np.stack([chunk[i][t % args.execute] for i in idx])
        else:
            a_teacher, warm_next = teacher.act([envs[i] for i in idx], warm)
        for j, i in enumerate(idx):
            a = student_plan[i][8-student_left[i]] if student_left[i] > 0 else a_teacher[j]
            student_left[i] = max(0, student_left[i]-1)
            obs, row = step_record(envs[i], np.clip(a, 0, 512))
            hists[i].append(obs); actions[i].append(np.asarray(row["action"], np.float32))
            cov[i] = row["coverage"]; peak[i] = max(peak[i], cov[i])
            if row["done"]:
                done[i] = True; active[i] = False
        if args.execute == 1:
            warm = warm_next[torch.as_tensor(active[idx], device=device)]
        t += 1
        if t % 25 == 0 or not active.any():
            print(f"[mpc-data {args.shape} {time.time()-start:5.0f}s] control {t} success {int(done.sum())}/{E} "
                  f"coverage {cov.mean():.3f} peak {peak.mean():.3f}", flush=True)
    data = args.out/"data"; data.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, seed in enumerate(seeds):
        acts = np.array(actions[i], np.float32)
        rec = defaultdict(list)
        for c, image, agent_pos, x, label in saved[i]:
            if label is None:                          # the teacher acted from here: its executed continuation
                seq = [acts[max(c-1, 0)]] + [acts[min(c+u, len(acts)-1)] for u in range(15)]
                label = np.array(seq, np.float32)
            lab = norm.normalize(torch.as_tensor(label, device=device)).cpu().numpy().astype(np.float16)
            rec["image"].append(image); rec["agent_pos"].append(agent_pos)
            rec["candidate_x"].append(np.concatenate([x, lab[None]])); rec["score"].append(np.array([0.0]*8+[1.0], np.float32))
            rec["choice"].append(8); rec["control"].append(c)
        if rec["choice"]:
            np.savez_compressed(data/f"{args.shape}-{seed}.npz", shape=args.shape, success=done[i], final_coverage=cov[i],
                                **{key: np.stack(v) if key not in ("choice", "control") else np.array(v) for key, v in rec.items()})
            total += len(rec["choice"])
        envs[i].close()
    summary = {"shape": args.shape, "beta": args.beta, "student": str(args.student), "seeds": seeds, "success": done.tolist(),
               "final_coverage": cov.tolist(), "peak_coverage": peak.tolist(), "controls": [len(a) for a in actions],
               "records": total, "seconds": time.time()-start, "args": {k_: str(v) for k_, v in vars(args).items()}}
    (args.out/f"{args.shape}-{args.start}.json").write_text(json.dumps(summary, indent=1))
    print(f"{args.shape}: success {int(done.sum())}/{E} coverage {cov.mean():.3f}; {total} records -> {data}")


def branch_label(env, branch, teacher, past):
    """Teacher's closed-loop continuation (16 targets incl. the previous one) from env's current state."""
    branch.restore_state(env.capture_state())
    seq, warm = [past[-1] if past else np.asarray(physical_state(env)["pusher"]["position"], np.float32)], None
    for _ in range(15):
        a, warm = teacher.act([branch], warm)
        branch.step(np.clip(a[0], 0, 512))
        seq.append(np.asarray(a[0], np.float32))
    return np.array(seq, np.float32)


if __name__ == "__main__":
    main()
