"""Imitate simulator-oracle choices or planning-teacher labels: denoising loss toward chosen plans + anchor to the base.

Data: oracle_bon.py decision records (image history, pusher history, N base-sampled normalized plans,
the oracle's choice). Loss per minibatch:
  imitation  standard DDPM epsilon loss on the CHOSEN plan (the diffusion policy's own training loss)
  anchor     MSE between the student's and the frozen base's epsilon predictions on noised NON-chosen
             base samples at the same observations (keeps the policy near the base on every shape)
Records from mpc_dagger.py (teacher label at index 8) use the same loss. --anchor-shapes T T_narrow
applies the anchor only on those shapes' records, so Y/D/L can follow the teacher while T stays near the base.
The image encoder stays frozen (conditions are precomputed). Validation runs the batched evaluator on
VALIDATION scenes (400000 + 20*i), never development (100000), test (300000), value (600000) or
oracle (700000) scenes, with the base policy evaluated once on the same scenes for comparison.
Usage (from the repository root):
  .venv/bin/python -B scripts/learning/distill.py --data $RUNS/oracle/data --trainable lora --lr 3e-4 \
      --anchor 1.0 --epochs 20 --val-scenes 50 --out $RUNS/distill/lora-a1
"""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import torch

import base_rollouts
from tdyssey.forward_policy import ForwardPolicy, forward_noise
from tdyssey.forward_protocol import ForwardConfig
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.official import OfficialDiffusionPolicy, load_official_source
from tdyssey.policy_training import evaluate_batched


def load_records(dirs, shapes):
    rec = defaultdict(list)
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            if shapes and str(z["shape"]) not in shapes:
                continue
            n = len(z["choice"])
            rec["image"].append(z["image"]); rec["agent_pos"].append(z["agent_pos"])
            cand = z["candidate_x"].astype(np.float32)               # (n, N, 16, 2); N differs between runs
            rec["chosen"].append(cand[np.arange(n), z["choice"]])     # the oracle's plan
            rec["x"].append(cand[:, :8])                               # 8 base samples per decision, for the anchor
            score = z["score"].astype(np.float32)
            reps = -(-16 // cand.shape[1])                             # pad decisions to 16 candidates (repeats weigh 0 extra)
            rec["cand16"].append(np.concatenate([cand]*reps, 1)[:, :16])
            rec["score16"].append(np.concatenate([score]+[np.full_like(score, -np.inf)]*(reps-1), 1)[:, :16])
            rec["choice"].append(z["choice"])
            rec["shape"] += [str(z["shape"])]*n
    if not rec["choice"]:
        raise SystemExit("no oracle decision records found")
    return {k: (np.concatenate(v) if k != "shape" else np.array(v)) for k, v in rec.items()}


def evaluate(agent, cls, seeds, mode):
    cfg = SimpleNamespace(eval_shapes=base_rollouts.SHAPES, eval_seeds=seeds, eval_steps=300, seed=42)
    original = agent.sample_batch
    agent.sample_batch = lambda c, g, m="actor": original(c, g, mode)
    try:
        episodes = evaluate_batched(agent, cls, cfg, defaultdict(int), lambda: None)
    finally:
        agent.sample_batch = original
    out = {}
    for s in base_rollouts.SHAPES:
        rows = [e for e in episodes if e["shape"] == s]
        out[s] = {"success": sum(e["success"] for e in rows), "n": len(rows),
                  "final_coverage": float(np.mean([e["final_coverage"] for e in rows]))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--shapes", nargs="*", default=None, help="Train only on these shapes' decisions")
    parser.add_argument("--trainable", choices=("lora", "action_network"), default="lora")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anchor", type=float, default=1.0)
    parser.add_argument("--anchor-shapes", nargs="*", default=None,
                        help="Apply --anchor only on these shapes' records (others get --anchor-other)")
    parser.add_argument("--anchor-other", type=float, default=0.0)
    parser.add_argument("--weighting", choices=("chosen", "softmax"), default="chosen",
                        help="chosen: imitate the oracle's pick; softmax: all candidates weighted by softmax(score/tau)")
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--val-scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    config = ForwardConfig(**json.loads((ROOT/"configs/nft_nearmiss_batched.json").read_text()))
    config = replace(config, trainable=args.trainable, actor_lr=args.lr, seed=args.seed)
    source = ROOT/"third_party/diffusion_policy"
    agent = ForwardPolicy(OfficialDiffusionPolicy(source, ROOT/"checkpoints/official-dp/ema_policy.pt"), config)
    cls = make_official_shape_env_class(load_official_source(source))
    data = load_records(args.data, args.shapes)
    n = len(data["choice"])
    with torch.no_grad():   # frozen encoder: precompute conditions once
        conds = []
        for i in range(0, n, 256):
            hist = [[{"image": data["image"][j, k].astype(np.float32)/255, "agent_pos": data["agent_pos"][j, k]}
                     for k in range(2)] for j in range(i, min(n, i+256))]
            conds.append(agent.encode_batch(hist))
        cond = torch.cat(conds)
    x = torch.tensor(data["x"], device=agent.device)                          # (n, N, 16, 2)
    chosen = torch.tensor(data["chosen"], device=agent.device)
    cand16 = torch.tensor(data["cand16"], device=agent.device)                  # (n, 16, 16, 2)
    weights16 = torch.softmax(torch.tensor(data["score16"], device=agent.device)/args.tau, dim=1)
    shapes_arr = np.asarray(data["shape"])
    anchor_w = torch.tensor(np.where(np.isin(shapes_arr, args.anchor_shapes), args.anchor, args.anchor_other)
                            if args.anchor_shapes is not None else np.full(n, args.anchor), device=agent.device, dtype=torch.float32)
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "records": n,
           "per_shape_records": {s: int((data["shape"] == s).sum()) for s in np.unique(data["shape"])}, "evals": []}
    print(json.dumps({k: log[k] for k in ("records", "per_shape_records")}), flush=True)
    seeds = [400000+20*i for i in range(args.val_scenes)]
    start = time.time()
    base_eval = evaluate(agent, cls, seeds, "reference")   # the frozen original policy, for LoRA and dense
    log["base_eval"] = base_eval
    print(f"[distill {time.time()-start:5.0f}s] base on validation: " +
          "  ".join(f"{s} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for s, v in base_eval.items()), flush=True)
    optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = agent.network.noise_scheduler
    g = torch.Generator(device=agent.device).manual_seed(args.seed)
    for epoch in range(1, args.epochs+1):
        order = torch.randperm(n, device=agent.device, generator=g)
        totals = defaultdict(float)
        for b in range(0, n, args.batch):
            idx = order[b:b+args.batch]
            c = cond[idx]
            t = torch.randint(0, 100, (len(idx),), device=agent.device, generator=g)
            noise = torch.randn(chosen[idx].shape, device=agent.device, generator=g)
            if args.weighting == "chosen":
                imitation = (agent.predict(forward_noise(scheduler, chosen[idx], noise, t), t, c) - noise).square().mean()
            else:   # every candidate, weighted by its oracle score; same (t, noise) across a decision's candidates
                B = len(idx)
                xs = cand16[idx].reshape(B*16, 16, 2)
                tt, nn_ = t.repeat_interleave(16), noise.repeat_interleave(16, 0)
                err = (agent.predict(forward_noise(scheduler, xs, nn_, tt), tt, c.repeat_interleave(16, 0)) - nn_)
                imitation = (err.square().mean((1, 2)).reshape(B, 16)*weights16[idx]).sum(1).mean()
            other = torch.randint(0, x.shape[1], (len(idx),), device=agent.device, generator=g)
            xb = forward_noise(scheduler, x[idx, other], torch.randn(chosen[idx].shape, device=agent.device, generator=g), t)
            with torch.no_grad():
                reference = agent.predict(xb, t, c, "reference")
            anchor = ((agent.predict(xb, t, c) - reference).square().mean((1, 2))*anchor_w[idx]).mean()
            loss = imitation + anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            optimizer.step()
            totals["imitation"] += float(imitation)*len(idx); totals["anchor"] += float(anchor)*len(idx)
        msg = f"[distill {time.time()-start:5.0f}s] epoch {epoch}: imitation {totals['imitation']/n:.4f} anchor {totals['anchor']/n:.2e}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev = evaluate(agent, cls, seeds, "actor")
            log["evals"].append({"epoch": epoch, **ev})
            msg += "  val: " + "  ".join(f"{s} {v['success']}/{v['n']} {v['final_coverage']:.3f}" for s, v in ev.items())
            torch.save({"actor": {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
                        "trainable": args.trainable, "epoch": epoch}, args.out/f"actor-epoch{epoch}.pt")
        print(msg, flush=True)
        (args.out/"log.json").write_text(json.dumps(log, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
