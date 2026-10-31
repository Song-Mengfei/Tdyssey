"""Standard Diffusion Policy training on demonstrations (the official recipe), for "can the student match the teacher".

The teacher's episodes are treated like human demonstrations: every step is a training sample (official windowing:
2 observation frames, 16-step action window starting one step back, padding 1 before / 7 after), the official
loss (DiffusionUnetHybridImagePolicy.compute_loss, which includes the official random-crop augmentation in train
mode), AdamW (lr 1e-4, betas 0.95/0.999, wd 1e-6) with 500 warm-up steps and cosine decay, and the official EMA
schedule (power 0.75, max 0.9999). Starts from the official checkpoint and keeps its normalizer. Data:
  --demos DIR...   teacher trajectories (mpc_teacher.py --save-traj *.npz) with replayed frames in DIR/frames/
  --human-t ZARR   the official human T demonstrations (pusht_cchi_v7_replay.zarr), so T is learned from the
                   data the original policy was trained on (not new demonstrations)
Letters are sampled in equal shares (T split evenly between human and teacher episodes).
Saves EMA checkpoints readable by base_rollouts.py eval --student.

  .venv/bin/python -B bc_train.py --demos $RUNS/teacher-noisy-demos --human-t $DATA/pusht-official/pusht/pusht_cchi_v7_replay.zarr \
      --steps 15000 --save-every 5000 --out $RUNS/bc/v1
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from tdyssey.official import OfficialDiffusionPolicy  # noqa: E402

LETTERS = ("T", "T_narrow", "Y", "D", "L")


def load_episodes(args):
    eps = []   # (frames uint8 (m,96,96,3), agent_pos (m,2), actions (m,2), group)
    for f in sorted(f for d in args.demos for f in Path(d).glob("*.npz")):
        z = {k: v for k, v in np.load(f).items()}
        fr = np.load(f.parent/"frames"/f"{f.stem}_frames.npz")["frames"]
        for i in range(len(z["seed"])):
            n = int(z["mask"][i].sum()); m = n - 1                         # m (frame, action) pairs
            if m < 2:
                continue
            eps.append((fr[i, :m], z["state"][i, :m, 6:8], z["action"][i, :m], f"{z['shape'][i]}-teacher", z["coverage"][i, :m]))
    if args.human_t:
        import zarr
        g = zarr.open(str(args.human_t), "r")
        img, st, act = g["data"]["img"][:], g["data"]["state"][:], g["data"]["action"][:]
        ends = g["meta"]["episode_ends"][:]
        for s, e in zip(np.concatenate([[0], ends[:-1]]), ends):
            eps.append((img[s:e].round().clip(0, 255).astype(np.uint8), st[s:e, :2], act[s:e], "T-human", None))
    return eps


def load_records(dirs, unnormalize):
    """DAgger records (mpc_dagger.py): 2 frames + the teacher's 16-step label chunk (normalized) at visited states."""
    out = []   # (frames (2,96,96,3) uint8, agent_pos (2,2), actions (16,2) pixels, group)
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            with np.load(f) as z:
                if "candidate_x" not in z.files or len(z["choice"]) == 0:
                    continue
                n = len(z["choice"])
                lab = z["candidate_x"][np.arange(n), z["choice"]].astype(np.float32)          # (n, 16, 2) normalized
                acts = unnormalize(lab)
                img = np.moveaxis(z["image"], 2, -1)                                         # (n, 2, 96, 96, 3)
                for r in range(n):
                    out.append((img[r], z["agent_pos"][r].astype(np.float32), acts[r], f"{z['shape']}-dagger"))
    return out


def build_index(eps, recs=(), near_cov=0.8):
    frames, pos, acts, obs_idx, act_idx, group, near = [], [], [], [], [], [], []
    off = 0
    for fr, ap, ac, g, cov in eps:
        m = len(fr)
        frames.append(fr); pos.append(ap); acts.append(ac)
        for t in range(0, max(1, m-7)):                                     # official padding: 1 before, 7 after
            obs_idx.append([off + max(t-1, 0), off + t])
            act_idx.append([off + min(max(k, 0), m-1) for k in range(t-1, t+15)])
            group.append(g); near.append(bool(cov is not None and cov[t] >= near_cov))
        off += m
    aoff = sum(len(a) for a in acts)
    for fr2, ap2, ac16, g in recs:                                          # records: obs = their own 2 frames
        frames.append(fr2); pos.append(ap2); acts.append(ac16)
        obs_idx.append([off, off+1]); act_idx.append(list(range(aoff, aoff+16))); group.append(g); near.append(False)
        off += 2; aoff += 16
    return (np.concatenate(frames), np.concatenate(pos).astype(np.float32), np.concatenate(acts).astype(np.float32),
            np.array(obs_idx), np.array(act_idx), np.array(group), np.array(near))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", type=Path, nargs="+", required=True, help="One or more demonstration folders")
    p.add_argument("--human-t", type=Path, default=None)
    p.add_argument("--records", type=Path, nargs="*", default=[], help="DAgger record folders (mpc_dagger.py .../data)")
    p.add_argument("--init", type=Path, default=None, help="Start from a bc_train/distill checkpoint instead of the original")
    p.add_argument("--near-goal-weight", type=float, default=1.0, help="Oversample teacher steps near the goal by this factor")
    p.add_argument("--near-coverage", type=float, default=0.8, help="Coverage at which a teacher step counts as near the goal")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); dev = "cuda"
    policy = OfficialDiffusionPolicy(ROOT/"third_party/diffusion_policy", ROOT/"checkpoints/official-dp/ema_policy.pt")
    net = policy.network
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        net.model.load_state_dict(ck["actor"]); net.obs_encoder.load_state_dict(ck["encoder"])
    unnorm = lambda x: net.normalizer["action"].unnormalize(torch.tensor(x, device=dev)).cpu().numpy()
    eps = load_episodes(args)
    recs = load_records(args.records, unnorm)
    frames, pos, acts, obs_idx, act_idx, group, near = build_index(eps, recs, args.near_coverage)
    groups = sorted(set(group))
    letters = sorted({g.rsplit("-", 1)[0] for g in groups})                 # each letter an equal share,
    share = {g: (1/len(letters)) / sum(1 for h in groups if h.rsplit("-", 1)[0] == g.rsplit("-", 1)[0]) for g in groups}
    # within each group, steps whose coverage is already >= --near-coverage get --near-goal-weight x the weight
    raw = np.where(near, args.near_goal_weight, 1.0)
    gsum = {g: raw[group == g].sum() for g in groups}
    w = np.array([share[g]/gsum[g] for g in group]) * raw
    info = {"samples": int(len(group)), "frames": int(len(frames)), "per_group": {g: int(np.sum(group == g)) for g in groups},
            "near_goal_samples": int(near.sum()),
            "share": share, "args": {k: str(v) for k, v in vars(args).items()}}
    print(json.dumps(info), flush=True)
    (args.out/"info.json").write_text(json.dumps(info, indent=1))
    frames = torch.tensor(frames, device=dev)                                # uint8 (F, 96, 96, 3)
    pos, acts = torch.tensor(pos, device=dev), torch.tensor(acts, device=dev)
    obs_idx, act_idx = torch.tensor(obs_idx, device=dev), torch.tensor(act_idx, device=dev)
    weights = torch.tensor(w, device=dev, dtype=torch.float64)

    net.train()
    net.model.requires_grad_(True); net.obs_encoder.requires_grad_(True)  # normalizer stays fixed, as in the official recipe
    trained = list(net.model.parameters()) + list(net.obs_encoder.parameters())
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(trained, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_at = lambda s: args.lr * (s/500 if s < 500 else 0.5*(1 + math.cos(math.pi*(s-500)/max(1, args.steps-500))))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    start = time.time(); running = 0.0; raw_losses = []
    for step in range(1, args.steps+1):
        b = torch.multinomial(weights, args.batch, replacement=True, generator=gen)
        oi = obs_idx[b]                                                       # (B, 2)
        img = frames[oi].permute(0, 1, 4, 2, 3).float()/255                  # (B, 2, 3, 96, 96)
        batch = {"obs": {"image": img, "agent_pos": pos[oi]}, "action": acts[act_idx[b]]}
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        loss = net.compute_loss(batch)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():                                                 # official EMA schedule
            decay = min(0.9999, max(0.0, 1 - (1 + step)**-0.75))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(decay).add_(pn.detach(), alpha=1-decay)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        raw_losses.append(loss.item())                                         # every step, saved with each checkpoint
        running = 0.98*running + 0.02*loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f"[bc {time.time()-start:5.0f}s] step {step} loss {running:.4f} lr {lr_at(step):.2e}", flush=True)
            np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))   # live view (W&B sync)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"trainable": "action_network", "actor": ema.model.state_dict(), "encoder": ema.obs_encoder.state_dict(),
                        "step": step, "recipe": "official DP recipe on teacher demos (+ human T)"}, args.out/f"ema-step{step}.pt")
            np.save(args.out/"loss_per_step.npy", np.array(raw_losses, dtype=np.float32))
            print(f"saved {args.out}/ema-step{step}.pt", flush=True)


if __name__ == "__main__":
    main()
