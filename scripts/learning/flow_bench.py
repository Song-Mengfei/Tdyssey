"""Speed and memory of the flow policy (flow_policy.py) at the training configuration, on a synthetic batch.

train: the flow_train.py step (bf16 autocast, compiled loss unless --no-compile, AdamW fused with its parameter groups,
       gradient clipping, EMA update) on one fixed random batch of the real shapes (2 frames 96 px, 2 pusher positions,
       16 x 2 plan); reports seconds per step after --warmup steps and the peak GPU memory (allocated, as train.log's
       gpu-peak, and reserved). Data loading is not included (flow_train.py assembles batches on a background thread).
infer: one decision as the evaluation runs it (FlowAgent: encode under bf16 autocast, then integrate: condition once,
       10 Euler steps) for each --infer-batch; milliseconds per decision (median of --reps).
One JSON line per measurement is appended to --out. Run one configuration per process (clean memory and compile):

  REPO=$PWD .venv/bin/python -B scripts/learning/flow_bench.py train --head prefix --pool 16 --batch 128 --out bench.jsonl
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flow_policy as fp  # noqa: E402


def build(args):
    keys = fp.prefix_arch(args.cond_width, args.cond_mlp_ratio) if args.head == "prefix" else {}
    torch.manual_seed(0)
    net = fp.FlowPolicy(fp.default_arch(pool=args.pool, **keys), fp.official_normalizer(), pretrained=False)
    net.vit.grad_checkpoint = args.grad_checkpoint
    net.head_checkpoint = args.head_checkpoint
    return net


def synthetic(batch, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    img = torch.rand(batch, 2, 3, 96, 96, device=dev, generator=g)
    pos = 60 + 400 * torch.rand(batch, 2, 2, device=dev, generator=g)
    act = 60 + 400 * torch.rand(batch, 16, 2, device=dev, generator=g)
    return img, pos, act


def train(args, net, dev, record):
    import flow_train
    net = net.to(dev).train()
    groups = flow_train.param_groups(net, SimpleNamespace(lr=1e-4, encoder_lr_scale=0.1, weight_decay=1e-3,
                                                         encoder_weight_decay=1e-6))
    trainable = [q for g in groups for q in g["params"]]
    opt = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), fused=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    named_ema = dict(ema.named_parameters())
    net_params = [q for n, q in net.named_parameters() if q.requires_grad]
    ema_params = [named_ema[n] for n, q in net.named_parameters() if q.requires_grad]
    loss_fn = net.compute_loss if args.no_compile else torch.compile(net.compute_loss)
    img, pos, act = synthetic(args.batch, dev)
    torch.cuda.reset_peak_memory_stats()
    t_start, blocks, losses, first = time.time(), [], [], None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for step in range(1, args.steps + 1):                     # synchronised only every 50 steps, as flow_train logs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(img, pos, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            decay = min(0.9999, max(0.0, 1 - (1 + step) ** -0.75))
            torch._foreach_lerp_(ema_params, net_params, 1 - decay)
        if step == 1:
            torch.cuda.synchronize(); first = time.perf_counter() - t0; t0 = time.perf_counter()
        elif step % 50 == 0:
            losses.append(float(loss)); t1 = time.perf_counter()
            blocks.append((step, (t1 - t0) / (50 - (step == 50))))
            t0 = t1
            print(f"step {step} loss {losses[-1]:.4f} {blocks[-1][1]:.3f}s/step "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    steady = [b for s_, b in blocks if s_ > args.warmup]
    record(mode="train", s_per_step=round(sum(steady) / len(steady), 4), s_per_step_blocks=[round(b, 4) for _, b in blocks],
           first_step_s=round(first, 1), peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
           peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2), loss_every_50=losses,
           wall_s=round(time.time() - t_start, 1))


@torch.no_grad()
def infer(args, net, dev, record):
    net = net.to(dev).eval().requires_grad_(False)
    for b in args.infer_batch:
        img, pos, _ = synthetic(b, dev)
        x0 = torch.randn(b, 16, 2, device=dev)
        enc, integ = [], []
        for r in range(args.reps + 5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cond = net.encode(img, pos).float()                    # FlowAgent.encode_batch returns float
                torch.cuda.synchronize(); t1 = time.perf_counter()
                net.integrate(cond, x0)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if r >= 5:
                enc.append(t1 - t0); integ.append(t2 - t1)
        med = lambda v: sorted(v)[len(v) // 2] * 1000
        tot = [e + i for e, i in zip(enc, integ)]
        record(mode="infer", infer_batch=b, encode_ms=round(med(enc), 2), integrate_ms=round(med(integ), 2),
               decision_ms=round(med(tot), 2), decision_ms_min=round(min(tot) * 1000, 2),
               peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("train", "infer"))
    p.add_argument("--head", choices=("cross", "prefix"), default="cross")
    p.add_argument("--cond-width", type=int, default=512)
    p.add_argument("--cond-mlp-ratio", type=float, default=4.0)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50, help="Steps excluded from the timing (compile, allocator warm-up)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--head-checkpoint", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--infer-batch", type=int, nargs="+", default=[1, 128])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--tag", default="", help="Label stored with each JSON line")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    net = build(args)
    counts = net.parameter_counts()
    config = {"tag": args.tag, "head": args.head, "pool": args.pool, "head_parameters": counts["head"],
              "gpu": torch.cuda.get_device_name()}
    if args.mode == "train":
        config.update(grad_checkpoint=args.grad_checkpoint, head_checkpoint=args.head_checkpoint,
                      compile=not args.no_compile, batch=args.batch)

    def record(**vals):
        row = {**config, **vals}
        print(json.dumps(row), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    try:
        (train if args.mode == "train" else infer)(args, net, dev, record)
    except torch.OutOfMemoryError:
        record(mode=args.mode, error="out of memory", peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
