"""Last-mile RL on top of the frozen camera policy (D058): short episodes restarted from recorded near-goal states.

Two methods, same states, reward and optimizer (SAC, off-policy; the base policy never changes):
  dsrl      the actor chooses the 32-number starting noise of the frozen policy's deterministic 10-step DDIM sampler
            (noise steering, as DSRL); every plan stays one the base policy can produce.
  residual  the actor sees the base plan (DDIM-10 from random noise, as deployed) and adds a bounded offset (at most
            --residual-px) to the 8 executed targets.
Actors read only the frozen policy's camera features (and, for residual, the base plan). Critics also read exact
simulator state (block and pusher relative to the goal, coverage, shape): privileged, training only, disclosed.

Reward per decision (8 controls): --reward-gain * coverage gain, minus --reward-loss * coverage lost within the
decision, a one-time --reward-bonus when coverage first exceeds 0.95 (success, which ends the episode), and minus
--reward-step per decision. Episodes last at most --episode-decisions decisions; time-limit cut-offs bootstrap.
v1 (D058): 10 / 10 / 5 / 0.05, 6 decisions. v2 (D058 amendment): 10 / 2 / 20 / 0.2, 12 decisions, gamma 0.95.

    .venv/bin/python -B scripts/learning/lastmile_rl.py train --method dsrl --student CKPT --states DIR --out DIR
    .venv/bin/python -B scripts/learning/lastmile_rl.py eval --student CKPT --states DIR --scenes 20030-20039 \
        [--actor runs/.../actor.pt] --out FILE
"""
import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
TRAINED27 = ["T", "T_narrow", "Y", "D", "L", "E", "F", "J", "P", "U"] + list("ABCGHIKMNOQRSVWXZ")
EXECUTE, SUCCESS = 8, 0.95


# ---------------------------------------------------------------------------------------------- simulator workers
def privileged(env, cov):
    """Block and pusher relative to the goal (goal frame), coverage, shape one-hot: critic-only input."""
    g = env.goal_pose
    c, s = math.cos(-g[2]), math.sin(-g[2])
    bx, by = env.block.position[0] - g[0], env.block.position[1] - g[1]
    px, py = env.agent.position[0] - g[0], env.agent.position[1] - g[1]
    da = (env.block.angle - g[2] + math.pi) % (2 * math.pi) - math.pi
    v = [(c * bx - s * by) / 50, (s * bx + c * by) / 50, math.sin(da), math.cos(da),
         (c * px - s * py) / 100, (s * px + c * py) / 100, cov]
    onehot = [0.0] * len(TRAINED27)
    onehot[TRAINED27.index(env.project_shape)] = 1.0
    return v + onehot


def worker(conn, render_size):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(1)
    import base_rollouts  # noqa: F401  (sets up the official source path)
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source, physical_metrics
    from tdyssey.state import Snapshot
    root = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
    cls = make_official_shape_env_class(load_official_source(root / "third_party/diffusion_policy"))
    envs, slots = {}, {}
    u8 = lambda o: (np.moveaxis(o["image"], 0, -1) * 255).round().astype(np.uint8)
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            break
        out = []
        if cmd == "reset":
            for slot, rec in payload:
                env = envs.get((slot, rec["shape"]))
                if env is None:
                    env = envs[(slot, rec["shape"])] = cls(shape=rec["shape"], legacy=True, render_size=render_size)
                o1 = env.restore_state(Snapshot.from_dict(rec["previous"]))
                o2 = env.restore_state(Snapshot.from_dict(rec["state"]))
                slots[slot] = env
                cov = physical_metrics(env)["coverage"]
                out.append((slot, [u8(o1), u8(o2)], [o1["agent_pos"], o2["agent_pos"]], privileged(env, cov), cov))
        elif cmd == "step":
            for slot, plan in payload:
                env = slots[slot]
                c0 = physical_metrics(env)["coverage"]
                hist, pos, done, lost, crossed, cov = [], [], False, 0.0, False, c0
                prev = c0
                for a in plan[:EXECUTE]:
                    o, _, done, _ = env.step(np.clip(a, 0, 512))
                    cov = physical_metrics(env)["coverage"]
                    lost += max(0.0, prev - cov)
                    prev = cov
                    hist.append(u8(o)); pos.append(o["agent_pos"])
                    if done:
                        crossed = True
                        break
                out.append((slot, hist[-2:] if len(hist) >= 2 else hist, pos[-2:] if len(pos) >= 2 else pos,
                            privileged(env, cov), c0, cov, lost, crossed))
        conn.send(out)


class Pool:
    def __init__(self, n_workers, slots, render_size):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, render_size), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)
        self.owner = {s: s % n_workers for s in range(slots)}

    def call(self, cmd, items):
        per = [[] for _ in self.conns]
        for slot, x in items:
            per[self.owner[slot]].append((slot, x))
        for c, p in zip(self.conns, per):
            c.send((cmd, p))
        out = {}
        for c, p in zip(self.conns, per):
            for r in c.recv():
                out[r[0]] = r[1:]
        return out

    def close(self):
        for c in self.conns:
            c.send(("close", None))


# ---------------------------------------------------------------------------------------------- base policy
class Base:
    """The frozen camera policy: features (its own observation encoder) and a deterministic DDIM-10 sampler."""

    def __init__(self, student, steps=10):
        import base_rollouts
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.agent, _ = base_rollouts.build(student)
        self.device = self.agent.device
        self.size = getattr(self.agent, "render_size", 96)
        self.ddim = DDIMScheduler.from_config(self.agent.network.noise_scheduler.config)
        self.ddim.set_timesteps(steps)
        self.norm = self.agent.network.normalizer["action"]

    @torch.no_grad()
    def features(self, images, positions):
        hist = [[{"image": im.astype(np.float32).transpose(2, 0, 1) / 255, "agent_pos": np.asarray(p, np.float32)}
                 for im, p in zip(ims, ps)] for ims, ps in zip(images, positions)]
        return self.agent.encode_batch(hist).float()

    @torch.no_grad()
    def plan_x(self, cond, noise):
        """Normalized 16-step plan (B, 16, 2) from the given starting noise (deterministic)."""
        x = noise.view(-1, 16, 2).to(self.device, torch.float32)
        for t in self.ddim.timesteps:
            x = self.ddim.step(self.agent.predict(x, t, cond, "actor"), t, x, eta=0.0).prev_sample
        return x

    @torch.no_grad()
    def plan(self, cond, noise):
        return self.norm.unnormalize(self.plan_x(cond, noise))                 # (B, 16, 2) pixels


# ---------------------------------------------------------------------------------------------- SAC
def mlp(i, o, h=512, n=3):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, scale):
        super().__init__()
        self.net, self.scale = mlp(obs_dim, 2 * act_dim), scale

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(-5, 1)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1) \
            - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a * self.scale, logp


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q1, self.q2 = mlp(in_dim, 1), mlp(in_dim, 1)

    def forward(self, x):
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, cap, dims, device):
        self.cap, self.n, self.i, self.device = cap, 0, 0, device
        self.buf = {k: torch.zeros((cap, d), device=device) for k, d in dims.items()}

    def add(self, **kw):
        b = next(iter(kw.values())).shape[0]
        idx = (torch.arange(b, device=self.device) + self.i) % self.cap
        for k, v in kw.items():
            self.buf[k][idx] = v.reshape(b, -1).to(self.device, torch.float32)
        self.i = (self.i + b) % self.cap
        self.n = min(self.n + b, self.cap)

    def sample(self, b):
        idx = torch.randint(0, self.n, (b,), device=self.device)
        return {k: v[idx] for k, v in self.buf.items()}


def load_states(directory, scenes):
    lo, hi = (int(x) for x in scenes.split("-"))
    recs = []
    for i in range(lo, hi + 1):
        p = Path(directory) / f"scene{i}.jsonl.gz"
        if p.exists():
            with gzip.open(p, "rt") as f:
                recs += [json.loads(line) for line in f]
    return recs


REWARD = {"gain": 10.0, "loss": 10.0, "bonus": 5.0, "step": 0.05}           # set from the command line in main()


def reward(c0, c1, lost, crossed):
    """Coverage gain over the decision, an extra penalty on coverage lost within it, a one-time success bonus and a
    cost per decision. v1 (D058) weights: 10, 10, 5, 0.05."""
    return REWARD["gain"] * (c1 - c0) - REWARD["loss"] * lost + (REWARD["bonus"] if crossed else 0.0) - REWARD["step"]


# ---------------------------------------------------------------------------------------------- train / eval
def run(args, train=True):
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    base = Base(args.student)
    dev = base.device
    recs = load_states(args.states, args.scenes)
    if not recs:
        raise SystemExit("no states")
    by_shape = {}
    for r in recs:
        by_shape.setdefault(r["shape"], []).append(r)
    shapes = sorted(by_shape)
    pool = Pool(args.workers, args.slots, base.size)

    def draw():                                                     # equal share per shape
        return random.choice(by_shape[random.choice(shapes)])

    method = args.method
    obs_res = base.features([np.zeros((2, base.size, base.size, 3), np.uint8)], [[np.zeros(2), np.zeros(2)]])
    feat_dim = obs_res.shape[1]
    priv_dim = 7 + len(TRAINED27)
    act_dim = 32 if method == "dsrl" else 16
    actor_in = feat_dim + (32 if method == "residual" else 0)
    scale = args.noise_bound if method == "dsrl" else args.residual_px
    actor = Actor(actor_in, act_dim, scale).to(dev)
    if args.actor:
        actor.load_state_dict(torch.load(args.actor, map_location=dev, weights_only=True)["actor"])

    def act_obs(feat, base_x):
        """Actor input: camera features, plus the base plan (normalized) for the residual."""
        return feat if method == "dsrl" else torch.cat([feat, base_x.view(len(feat), -1)], -1)

    def base_plan_x(feat, noise):
        return base.plan_x(feat, noise).reshape(len(feat), 32) if method == "residual" else torch.zeros((len(feat), 1), device=dev)

    def make_plans(feat, actions, base_x):
        if method == "dsrl":
            return base.plan(feat, actions)
        plans = base.norm.unnormalize(base_x.view(-1, 16, 2)).clone()
        plans[:, 1:1 + EXECUTE] += actions.view(-1, EXECUTE, 2)
        return plans

    # ---- environment state per slot
    state = {}

    def reset(slots):
        res = pool.call("reset", [(s, draw()) for s in slots])
        for s, (ims, pos, priv, cov) in res.items():
            state[s] = {"ims": ims, "pos": pos, "priv": priv, "cov": cov, "t": 0}

    if not train:
        return evaluate(args, base, pool, actor, method, recs, act_obs, make_plans)
    # (dev is defined above; base_plan_x needs it)

    critic_in = feat_dim + priv_dim + act_dim + (32 if method == "residual" else 0)
    critic, target = Critic(critic_in).to(dev), Critic(critic_in).to(dev)
    target.load_state_dict(critic.state_dict())
    log_alpha = torch.tensor(math.log(0.1), device=dev, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_al = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -float(act_dim)
    base_dim = 32 if method == "residual" else 1
    rb = Replay(args.buffer, {"f": feat_dim, "p": priv_dim, "b": base_dim, "a": act_dim, "r": 1, "f2": feat_dim,
                              "p2": priv_dim, "b2": base_dim, "d": 1}, dev)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    slots = list(range(args.slots))
    reset(slots)
    t0, decisions, episodes, successes, ret_sum = time.time(), 0, 0, 0, 0.0
    gamma = args.gamma
    while decisions < args.decisions:
        feat = base.features([state[s]["ims"] for s in slots], [state[s]["pos"] for s in slots])
        priv = torch.tensor([state[s]["priv"] for s in slots], device=dev, dtype=torch.float32)
        base_x = base_plan_x(feat, torch.randn((len(slots), 32), device=dev))   # residual: the deployed-style base plan
        with torch.no_grad():
            if decisions < args.warmup:
                if method == "dsrl":                          # warm-up = the base policy: N(0, I) noise, clipped
                    a = torch.randn((len(slots), act_dim), device=dev).clamp(-scale, scale)
                else:
                    a = torch.zeros((len(slots), act_dim), device=dev)
            else:
                a, _ = actor(act_obs(feat, base_x))
            plans = make_plans(feat, a, base_x).cpu().numpy()
        res = pool.call("step", [(s, plans[k, 1:1 + EXECUTE]) for k, s in enumerate(slots)])
        r = torch.zeros(len(slots), device=dev); d = torch.zeros(len(slots), device=dev)
        done_slots = []
        nxt_ims, nxt_pos = [], []
        for k, s in enumerate(slots):
            hist, pos, priv2, c0, c1, lost, crossed = res[s]
            ims = (state[s]["ims"] + hist)[-2:]
            ps = (list(state[s]["pos"]) + list(pos))[-2:]
            r[k] = reward(c0, c1, lost, crossed); ret_sum += float(r[k])
            state[s].update(ims=ims, pos=ps, priv=priv2, cov=c1, t=state[s]["t"] + 1)
            d[k] = 1.0 if crossed else 0.0                    # success is terminal; time limit bootstraps
            nxt_ims.append(ims); nxt_pos.append(ps)
            if crossed or state[s]["t"] >= args.episode_decisions:
                done_slots.append(s); episodes += 1; successes += int(crossed)
        feat2 = base.features(nxt_ims, nxt_pos)
        priv2 = torch.tensor([state[s]["priv"] for s in slots], device=dev, dtype=torch.float32)
        base2 = base_plan_x(feat2, torch.randn((len(slots), 32), device=dev))
        rb.add(f=feat, p=priv, b=base_x, a=a, r=r[:, None], f2=feat2, p2=priv2, b2=base2, d=d[:, None])
        decisions += len(slots)
        if done_slots:
            reset(done_slots)
        if rb.n >= args.batch and decisions >= args.warmup:
            for _ in range(args.updates):
                bt = rb.sample(args.batch)
                with torch.no_grad():
                    a2, logp2 = actor(act_obs(bt["f2"], bt["b2"]))
                    q1t, q2t = target(torch.cat([bt["f2"], bt["p2"], a2] + ([bt["b2"]] if method == "residual" else []), -1))
                    y = bt["r"][:, 0] + gamma * (1 - bt["d"][:, 0]) * (torch.min(q1t, q2t) - log_alpha.exp() * logp2)
                q1, q2 = critic(torch.cat([bt["f"], bt["p"], bt["a"]] + ([bt["b"]] if method == "residual" else []), -1))
                lc = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
                opt_c.zero_grad(); lc.backward(); opt_c.step()
                an, logp = actor(act_obs(bt["f"], bt["b"]))
                q1n, q2n = critic(torch.cat([bt["f"], bt["p"], an] + ([bt["b"]] if method == "residual" else []), -1))
                la = (log_alpha.exp().detach() * logp - torch.min(q1n, q2n)).mean()
                opt_a.zero_grad(); la.backward(); opt_a.step()
                lal = -(log_alpha * (logp.detach() + target_entropy)).mean()
                opt_al.zero_grad(); lal.backward(); opt_al.step()
                with torch.no_grad():
                    for p_, tp in zip(critic.parameters(), target.parameters()):
                        tp.mul_(1 - args.tau).add_(p_, alpha=args.tau)
        if decisions // len(slots) % 50 == 0:
            rate = successes / max(1, episodes)
            line = {"decisions": decisions, "episodes": episodes, "success_rate": round(rate, 4),
                    "mean_reward": round(ret_sum / max(1, decisions), 4), "alpha": round(float(log_alpha.exp()), 4),
                    "seconds": round(time.time() - t0)}
            print(json.dumps(line), flush=True); log.write(json.dumps(line) + "\n"); log.flush()
            episodes = successes = 0; ret_sum = 0.0
        if decisions // len(slots) % 500 == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "method": method, "args": vars(args),
                        "decisions": decisions}, out / "actor.pt")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "method": method, "args": vars(args),
                "decisions": decisions}, out / "actor.pt")
    pool.close()


@torch.no_grad()
def evaluate(args, base, pool, actor, method, recs, act_obs, make_plans):
    """Finishing rate from held-out near-goal states: each state once, up to 6 decisions. Policies: the base with the
    deployed-style random-noise DDIM-10 sampler ('base'), or the trained actor (deterministic mean)."""
    rng = random.Random(args.seed)
    by_shape = {}
    for r in recs:
        by_shape.setdefault(r["shape"], []).append(r)
    chosen = []
    for s in sorted(by_shape):
        lst = by_shape[s][:]
        rng.shuffle(lst)
        chosen += lst[:args.per_shape]
    rows = []
    for start in range(0, len(chosen), args.slots):
        batch = chosen[start:start + args.slots]
        slots = list(range(len(batch)))
        res = pool.call("reset", list(zip(slots, batch)))
        st = {s: {"ims": res[s][0], "pos": res[s][1], "cov0": res[s][3], "cov": res[s][3], "done": False, "t": 0} for s in slots}
        gen = torch.Generator(device=base.device).manual_seed(args.seed + start)
        for _ in range(args.episode_decisions):
            live = [s for s in slots if not st[s]["done"]]
            if not live:
                break
            feat = base.features([st[s]["ims"] for s in live], [st[s]["pos"] for s in live])
            noise = torch.randn((len(live), 32), device=base.device, generator=gen)
            if args.actor:
                bx = base.plan_x(feat, noise).reshape(len(live), 32) if method == "residual" else None
                a, _ = actor(act_obs(feat, bx), deterministic=not args.stochastic)
                plans = make_plans(feat, a, bx)
            else:
                plans = base.plan(feat, noise)
            out = pool.call("step", [(s, plans[k, 1:1 + EXECUTE].cpu().numpy()) for k, s in enumerate(live)])
            for s in live:
                hist, pos, priv2, c0, c1, lost, crossed = out[s]
                st[s]["ims"] = (st[s]["ims"] + hist)[-2:]; st[s]["pos"] = (list(st[s]["pos"]) + list(pos))[-2:]
                st[s]["cov"] = c1; st[s]["t"] += 1
                if crossed:
                    st[s]["done"] = True
        for s, rec in zip(slots, batch):
            rows.append({"shape": rec["shape"], "scene_index": rec["scene_index"], "control": rec["control"],
                         "start_coverage": round(st[s]["cov0"], 5), "solved": st[s]["done"], "decisions": st[s]["t"],
                         "final_coverage": round(st[s]["cov"], 5)})
    pool.close()
    k = sum(r["solved"] for r in rows)
    summary = {"policy": args.actor or "base (DDIM-10, random noise)", "states": len(rows), "solved": k,
               "rate": round(k / max(1, len(rows)), 4),
               "per_shape": {s: round(sum(r["solved"] for r in rows if r["shape"] == s) / max(1, sum(r["shape"] == s for r in rows)), 3)
                             for s in sorted({r["shape"] for r in rows})}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "episodes": rows}, open(args.out, "w"), indent=1)
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "per_shape"}), flush=True)


@torch.no_grad()
def episodes(args):
    """Full episodes (300 controls) from normal starting scenes (matched across shapes, as every project test), with the
    RL actor deciding every plan (or the base policy with the DDIM-10 sampler when --actor is not given). Seeds:
    --seed-base + 20 * (start + k). Writes base_rollouts-style episode rows for paired comparison."""
    import base_rollouts
    from tdyssey.envs.official import matched_reset
    torch.manual_seed(args.seed)
    base = Base(args.student)
    dev = base.device
    shapes = args.shapes.split(",") if args.shapes else TRAINED27
    method = args.method
    actor = None
    if args.actor:
        blob = torch.load(args.actor, map_location=dev, weights_only=True)
        method = blob["method"]
        feat_dim = base.features([np.zeros((2, base.size, base.size, 3), np.uint8)], [[np.zeros(2), np.zeros(2)]]).shape[1]
        actor = Actor(feat_dim + (32 if method == "residual" else 0), 32 if method == "dsrl" else 16,
                      args.noise_bound if method == "dsrl" else args.residual_px).to(dev)
        actor.load_state_dict(blob["actor"])
    from tdyssey.envs.official import make_official_shape_env_class
    from tdyssey.official import load_official_source
    root = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
    cls = make_official_shape_env_class(load_official_source(root / "third_party/diffusion_policy"))
    seeds = [args.seed_base + 20 * (args.start + k) for k in range(args.count)]
    jobs = []                                                       # (seed, shape, snapshot json)
    for seed in seeds:
        envs = {s: cls(shape=s, legacy=True, render_size=base.size) for s in (shapes if "T" in shapes else ["T"] + shapes)}
        matched_reset(envs, seed, {"reset_physics_substeps": 0})
        for s in shapes:
            snap = json.loads(envs[s].capture_state().to_json())
            jobs.append((seed, s, {"shape": s, "previous": snap, "state": snap}))
        for e in envs.values():
            e.close()
    pool = Pool(args.workers, args.slots, base.size)
    rows = []
    for start in range(0, len(jobs), args.slots):
        batch = jobs[start:start + args.slots]
        slots = list(range(len(batch)))
        res = pool.call("reset", [(s, j[2]) for s, j in zip(slots, batch)])
        st = {s: {"ims": res[s][0], "pos": res[s][1], "cov": res[s][3], "peak": res[s][3], "done": False, "controls": 0} for s in slots}
        gens = {s: torch.Generator(device=dev).manual_seed(args.seed * 1000003 + batch[s][0] * 31 + TRAINED27.index(batch[s][1]) if batch[s][1] in TRAINED27 else batch[s][0]) for s in slots}
        while True:
            live = [s for s in slots if not st[s]["done"] and st[s]["controls"] < 300]
            if not live:
                break
            feat = base.features([st[s]["ims"] for s in live], [st[s]["pos"] for s in live])
            noise = torch.cat([torch.randn((1, 32), device=dev, generator=gens[s]) for s in live])
            if actor is not None:
                bx = base.plan_x(feat, noise).reshape(len(live), 32) if method == "residual" else None
                a, _ = actor(feat if method == "dsrl" else torch.cat([feat, bx], -1), deterministic=True)
                if method == "dsrl":
                    plans = base.plan(feat, a)
                else:
                    plans = base.norm.unnormalize(bx.view(-1, 16, 2)).clone()
                    plans[:, 1:1 + EXECUTE] += a.view(-1, EXECUTE, 2)
            else:
                plans = base.plan(feat, noise)
            items = []
            for k, s in enumerate(live):
                n = min(EXECUTE, 300 - st[s]["controls"])
                items.append((s, plans[k, 1:1 + n].cpu().numpy()))
            out = pool.call("step", items)
            for k, s in enumerate(live):
                hist, pos, priv2, c0, c1, lost, crossed = out[s]
                st[s]["ims"] = (st[s]["ims"] + hist)[-2:]; st[s]["pos"] = (list(st[s]["pos"]) + list(pos))[-2:]
                st[s]["controls"] += len(hist); st[s]["cov"] = c1; st[s]["peak"] = max(st[s]["peak"], c1)
                if crossed:
                    st[s]["done"] = True
        for s, (seed, shape, _) in zip(slots, batch):
            rows.append({"shape": shape, "requested_seed": seed, "success": st[s]["done"], "controls": st[s]["controls"],
                         "final_coverage": st[s]["cov"], "peak_coverage": st[s]["peak"]})
        print(f"episodes {len(rows)}/{len(jobs)}", flush=True)
    pool.close()
    summary = {}
    for s in shapes:
        r = [x for x in rows if x["shape"] == s]
        summary[s] = {"success": sum(x["success"] for x in r), "n": len(r),
                      "mean_final_coverage": float(np.mean([x["final_coverage"] for x in r])),
                      "mean_peak_coverage": float(np.mean([x["peak_coverage"] for x in r]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": args.actor or f"{args.student} (DDIM-10)", "seeds": seeds, "summary": summary, "episodes": rows},
              open(args.out, "w"), indent=1)
    k = sum(x["success"] for x in rows)
    print(json.dumps({"policy": args.actor or "base DDIM-10", "episodes": len(rows), "success": k, "rate": round(k / max(1, len(rows)), 4)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "eval", "episodes"))
    ap.add_argument("--method", choices=("dsrl", "residual"), default="dsrl")
    ap.add_argument("--student", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--scenes", default="20000-20029", help="scene index range of the start states (inclusive)")
    ap.add_argument("--actor", default=None, help="trained actor (eval) or warm start (train)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--slots", type=int, default=128, help="parallel episodes")
    ap.add_argument("--decisions", type=int, default=150000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--updates", type=int, default=1, help="gradient updates per batch of decisions")
    ap.add_argument("--buffer", type=int, default=300000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--noise-bound", type=float, default=1.5)
    ap.add_argument("--residual-px", type=float, default=10.0)
    ap.add_argument("--per-shape", type=int, default=40, help="eval: states per shape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-decisions", type=int, default=6, help="decisions (of 8 controls) per episode")
    ap.add_argument("--reward-gain", type=float, default=10.0)
    ap.add_argument("--reward-loss", type=float, default=10.0, help="extra penalty per unit of coverage lost")
    ap.add_argument("--reward-bonus", type=float, default=5.0)
    ap.add_argument("--reward-step", type=float, default=0.05)
    ap.add_argument("--stochastic", action="store_true", help="eval: sample from the actor instead of its mean")
    ap.add_argument("--seed-base", type=int, default=400000, help="episodes: scene pool")
    ap.add_argument("--start", type=int, default=0, help="episodes: first scene index")
    ap.add_argument("--count", type=int, default=50, help="episodes: number of scenes")
    ap.add_argument("--shapes", default="", help="episodes: comma-separated shapes (default: the 27 trained)")
    args = ap.parse_args()
    REWARD.update(gain=args.reward_gain, loss=args.reward_loss, bonus=args.reward_bonus, step=args.reward_step)
    if args.mode == "episodes":
        episodes(args)
    else:
        run(args, train=args.mode == "train")


if __name__ == "__main__":
    main()
